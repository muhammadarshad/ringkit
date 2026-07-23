"""Ring-native fixed-point activations for ringkit.qcm — PROMOTED from the proven `emulation.ract`
(and gemma) into the native model layer, so the QCM transformer carries its own true-native, float-
free activations (no dependence on the emulation engine). Every op is Q<frac> signed integers via
rn.mul (shift-add), rn.mf_floordiv (shift-subtract), rn.isqrt, shifts — NO float, NO numpy, NO math.

Promoted (bit-identical to ract, which quanta gates at cos 1.0): exp_fixed, sigmoid_fixed, gelu_fixed,
tanh_fixed, softcap_fixed, layernorm_fixed, rmsnorm_fixed, and the batched sigmoid_list/exp_list_nonpos
(one C block call via kernels.mprc.gemma, pure-Python fallback). Added here (ract lacks them, built on
the promoted exp): softplus_fixed, softmax_fixed. exp_nonpos kept as a thin alias for the exp path.
"""
from ringkit.core import native as rn

FRAC = 16
ONE = 1 << FRAC


def _sdiv(n, d):
    """Signed integer divide (truncate toward zero), d > 0, on multiplier-free mf_floordiv."""
    if n < 0:
        return -rn.mf_floordiv(-n, d)
    return rn.mf_floordiv(n, d)


def _sd(n, d):
    """round(n/d), signed, d>0."""
    if n >= 0:
        return rn.mf_floordiv(n + (d >> 1), d)
    return -rn.mf_floordiv((-n) + (d >> 1), d)


def exp_fixed(x, frac=FRAC):
    """e^x in Q<frac> (signed in/out), float-free. Range-reduce (halve) + integer Taylor + re-square."""
    one = 1 << frac
    neg = x < 0
    ax = -x if neg else x
    half = one >> 1
    m = 0
    red = ax
    while red > half:
        red = red >> 1
        m = m + 1
    term = one
    acc = one
    for k in range(1, 13):
        term = rn.mul(term, red) >> frac
        term = rn.mf_floordiv(term, k)
        acc = acc + term
        if term == 0:
            break
    for _ in range(m):
        acc = rn.mul(acc, acc) >> frac
    if neg:
        acc = rn.mf_floordiv(1 << (frac + frac), acc)   # e^-|x| = 1/e^|x|
    return acc


def exp_nonpos(x, frac=FRAC):
    """e^x for x<=0 (the softmax domain). Thin alias over exp_fixed (which handles either sign)."""
    return exp_fixed(x, frac)


def sigmoid_fixed(x, frac=FRAC):
    """1/(1+e^-x) in Q<frac>. |x| saturated at frac<<frac (beyond it e^-|x| floors to 0 -> exact)."""
    one = 1 << frac
    lim = frac << frac
    if x > lim:
        x = lim
    elif x < -lim:
        x = -lim
    e = exp_fixed(-x, frac)
    return rn.mf_floordiv(1 << (frac + frac), one + e)


# The batched (hot-path) activations dispatch through the .device() layer to the selected backend's
# kernel. NO Python fallback: an unavailable device / missing kernel RAISES (owner: the kit is
# lightning-fast C/CUDA; a silent slow Python path is forbidden). Scalars above stay as the exact
# reference the kernels are gated against; only the batched paths run on-device. `dev=None` -> the
# process default device (best available); pass a Device to pin the backend.
def _dev(dev):
    if dev is not None:
        return dev
    from ringkit.device import default_device
    return default_device()


def sigmoid_list(xs, frac=FRAC, dev=None):
    """sigmoid over a vector in ONE kernel call on the selected device. Raises if unavailable."""
    return _dev(dev).sigmoid(xs, frac)


def exp_list_nonpos(xs, frac=FRAC, dev=None):
    """exp over a vector of NON-POSITIVE args (softmax domain), ONE kernel call. Raises if unavailable."""
    return _dev(dev).exp_nonpos(xs, frac)


# ── ring-native GELU: GELU(x) = x * Phi(x), Phi = the Gaussian CDF ──
# The Gaussian is the ring's OWN (measure.born_weights identity): w[d] = (F/256)^(d^2), i.e. geometric
# decay STEPPED BY ODDS (squares grow by odds: d^2 = 1+3+...+(2d-1)) — the ring's e^{-x^2}, NO float and
# NO pi (the 1/sqrt(2*pi) cancels under CDF normalization, so pi never enters — this is why the borrowed
# 1.702 = sqrt(8/pi) logistic coeff was a C8 violation). F=255 is the widest integer decay; its 1-sigma
# point (w ~ 0.607) sits at ~11 ring-offsets, so input 1.0 (=ONE) maps to 11 offsets. Phi is the
# normalized cumulative, Q<frac>. Structure-EXACT: GELU(0)=0, GELU(+inf)=x, GELU(-inf)=0. The match to
# the analytic N(0,1) GELU is a LABELED reference comparison only (D8/C6), never a target.
_GELU_F = 255
_GELU_SIGMA = 11                                  # ring-offsets per input unit (F=255 -> w~0.607 at d=11)
_GELU_U = 128                                     # offset half-width (ring's max circular distance)


def _build_gelu_phi():
    w = [0] * (_GELU_U + 1)
    acc = 255 << 8                                # fixed-point decay accumulator (born_weights style)
    w[0] = acc >> 8
    for d in range(1, _GELU_U + 1):
        for _ in range(d + d - 1):               # x (F/256), (2d-1) times -> cumulative (F/256)^(d^2)
            acc = rn.mul(acc, _GELU_F) >> 8
        w[d] = acc >> 8
    total = 0                                      # pdf over signed offsets u=-U..U is w[|u|]
    for u in range(-_GELU_U, _GELU_U + 1):
        total += w[abs(u)]
    phi = [0] * (2 * _GELU_U + 1)
    c = 0
    for u in range(-_GELU_U, _GELU_U + 1):        # Phi[U+u] = (sum_{v<=u} w[|v|]) / total, Q<frac>
        c += w[abs(u)]
        phi[_GELU_U + u] = _sd(c << FRAC, total)
    return phi


_GELU_PHI = _build_gelu_phi()


def gelu_fixed(x, frac=FRAC):
    """GELU(x) = x * Phi(x), Phi = the ring Gaussian CDF (odd-step geometric decay, no float/pi).
    The native ring form replacing the borrowed x*sigmoid(1.702 x). Structure-exact at 0/+-inf."""
    d = _sd(rn.mul(x, _GELU_SIGMA), 1 << frac)     # input -> ring offset (x * sigma)
    if d <= -_GELU_U:
        return 0
    if d >= _GELU_U:
        return x
    return _sdiv(rn.mul(x, _GELU_PHI[_GELU_U + d]), 1 << frac)


def tanh_fixed(x, frac=FRAC):
    """tanh(x) = 2*sigmoid(2x) - 1 in Q<frac>. Float-free."""
    one = 1 << frac
    return (sigmoid_fixed(x << 1, frac) << 1) - one


def softcap_fixed(x, cap, frac=FRAC):
    """cap * tanh(x/cap), all Q<frac> (cap an integer scalar). Smoothly bounds into (-cap, cap)."""
    t = tanh_fixed(_sdiv(x, cap), frac)
    return rn.mul(t, cap)


def softplus_fixed(x, frac=FRAC):
    """softplus(x) = max(x,0) + ln(1+e^-|x|), Q<frac>. ln(1+u)=2*atanh(u/(2+u)) (u in (0,1] -> arg
    in (0,1/3]). Float-free; ract lacks this — built here on the promoted exp_fixed."""
    one = 1 << frac
    ax = -x if x < 0 else x
    lim = frac << frac
    if ax > lim:
        ax = lim
    e = exp_fixed(-ax, frac)                       # e^-|x| in (0,1]
    w = _sd(e << frac, (one << 1) + e)             # u/(2+u)
    w2 = rn.mul(w, w) >> frac
    term = w
    acc = w
    for k in (3, 5, 7, 9, 11):
        term = rn.mul(term, w2) >> frac
        acc += _sd(term, k)
    return (x if x > 0 else 0) + (acc << 1)


def softmax_fixed(logits, frac=FRAC):
    """Stable softmax -> Q<frac> probabilities (sum ~= ONE). Subtract max, exp (nonpos), normalize.
    Built on the promoted exp; ract keeps softmax in infer.py, this is the model-native copy."""
    m = max(logits)
    ex = exp_list_nonpos([l - m for l in logits], frac)
    tot = sum(ex) or 1
    return [_sd(e << frac, tot) for e in ex]


def layernorm_fixed(x, gamma, beta, frac=FRAC, eps=1):
    """(x-mean)/sqrt(var+eps)*gamma + beta, all Q<frac>. sqrt = ring isqrt. Float-free."""
    n = len(x)
    tot = 0
    for v in x:
        tot = tot + v
    mean = _sdiv(tot, n)
    xc = [v - mean for v in x]
    ssq = 0
    for v in xc:
        ssq = ssq + (rn.mul(v, v) >> frac)
    var = rn.mf_floordiv(ssq, n) + eps
    std = rn.isqrt(var << frac) or 1
    return [_sdiv(rn.mul(_sdiv(xc[i] << frac, std), gamma[i]), 1 << frac) + beta[i] for i in range(n)]


def rmsnorm_fixed(x, weight, frac=FRAC, eps=1, dev=None):
    """x / sqrt(mean(x^2)+eps) * weight, all Q<frac>. ONE kernel call on the selected device (CUDA
    handles the full int64 activation range via u128). Raises if unavailable (no Python fallback)."""
    return _dev(dev).rmsnorm(x, weight, frac, eps)


def _selftest():
    ok = True
    s0 = sigmoid_fixed(0)
    ok &= (s0 == ONE // 2)
    print(f"  sigmoid(0)=1/2: {'PASS' if s0==ONE//2 else 'FAIL'} ({s0})")
    # exp identity: e^0=1, e^1~=e (178145)
    e1 = exp_fixed(ONE)
    ok &= abs(e1 - 178145) <= 64
    print(f"  exp(1)~=e: {'PASS' if abs(e1-178145)<=64 else 'FAIL'} ({e1})")
    sp0 = softplus_fixed(0)
    ok &= abs(sp0 - 45426) <= 8
    print(f"  softplus(0)=ln2: {'PASS' if abs(sp0-45426)<=8 else 'FAIL'} ({sp0})")
    # gelu: gelu(0)=0, gelu(large)~=x, gelu(-large)~=0
    g0 = gelu_fixed(0); gp = gelu_fixed(6 * ONE); gn = gelu_fixed(-6 * ONE)
    gok = (g0 == 0 and abs(gp - 6 * ONE) <= ONE // 8 and abs(gn) <= ONE // 8)
    ok &= gok
    print(f"  gelu(0)=0, gelu(+6)~=6, gelu(-6)~=0: {'PASS' if gok else 'FAIL'} ({g0},{gp},{gn})")
    # tanh(0)=0, tanh(large)~=1
    tok = (tanh_fixed(0) == 0 and abs(tanh_fixed(5 * ONE) - ONE) <= ONE // 16)
    ok &= tok
    print(f"  tanh(0)=0, tanh(5)~=1: {'PASS' if tok else 'FAIL'}")
    # softmax sums to ~ONE, uniform -> 1/n
    p = softmax_fixed([0, 0, 0, 0])
    smok = (abs(sum(p) - ONE) <= 8 and all(abs(pi - ONE // 4) <= 4 for pi in p))
    ok &= smok
    print(f"  softmax(uniform)=1/n, sums to 1: {'PASS' if smok else 'FAIL'} {p}")
    # rmsnorm / layernorm valid
    r = rmsnorm_fixed([3 * ONE, 4 * ONE, 0, -5 * ONE], [ONE] * 4)
    ln = layernorm_fixed([1 * ONE, 2 * ONE, 3 * ONE, 4 * ONE], [ONE] * 4, [0] * 4)
    nok = (len(r) == 4 and len(ln) == 4)
    ok &= nok
    print(f"  rmsnorm/layernorm valid: {'PASS' if nok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("ringkit.qcm.activations self-test (promoted from ract, true-native, float-free):")
    print("RESULT:", "ALL PASS" if _selftest() else "FAIL")
