"""
ringkit.ract — ring fixed-point ACTIVATIONS for inference. NO float, NO FPU, multiplier-free.

Pretrained models use exp/sigmoid/GELU/RMSNorm — the places a float would otherwise appear on the
inference path. Here they are computed in Q<frac> signed integers only:

  * exp: INTEGER Taylor series with range-reduction. e = Σ 1/k! emerges from the series itself —
    no float constant, no math.exp. Reduce x by halving (>>) until small, Taylor ~12 terms
    (rn.mul + mf_floordiv), then square back (rn.mul). Float-free by construction.
  * sigmoid = 1/(1+e^-x); GELU ≈ x·sigmoid(1.702·x) (1.702 = 1702/1000, a rational int constant).
  * rmsnorm: x / sqrt(mean(x²)) · w — the sqrt is the ring's integer isqrt (no float).

Multiplier-free: rn.mul (shift-add), rn.mf_floordiv (shift-subtract), rn.isqrt, shifts. No '*'/'/'.
"""
from ringkit.core import native as rn

_C_GELU = 1702      # 1.702 * 1000  (the tanh/sigmoid-GELU coefficient, as a rational integer)
_MILLE = 1000


def _sdiv(n, d):
    """Signed integer divide (truncate toward zero), d > 0. Built on the multiplier-free mf_floordiv."""
    if n < 0:
        return -rn.mf_floordiv(-n, d)
    return rn.mf_floordiv(n, d)


def exp_fixed(x, frac):
    """e^x in Q<frac> (signed int in, signed int out), float-free. Range-reduce + integer Taylor + square."""
    one = 1 << frac
    neg = x < 0
    ax = -x if neg else x                       # work on the magnitude (>= 0)
    half = one >> 1
    m = 0
    red = ax
    while red > half:                           # halve until the argument is small (fast Taylor)
        red = red >> 1
        m = m + 1
    term = one                                  # Taylor: 1 + red + red²/2! + ...
    acc = one
    for k in range(1, 13):
        term = rn.mul(term, red) >> frac        # term *= red   (Q<frac>)
        term = rn.mf_floordiv(term, k)          # term /= k
        acc = acc + term
        if term == 0:
            break
    for _ in range(m):                          # undo the halving: (e^(x/2^m))^(2^m)
        acc = rn.mul(acc, acc) >> frac
    if neg:
        acc = rn.mf_floordiv(1 << (frac + frac), acc)   # e^-|x| = 1 / e^|x|
    return acc


def sigmoid_fixed(x, frac):
    """1/(1+e^-x) in Q<frac>. A (0,1) volume-knob; float-free."""
    one = 1 << frac
    e = exp_fixed(-x, frac)                      # e^-x
    return rn.mf_floordiv(1 << (frac + frac), one + e)   # one / (1 + e^-x)


def gelu_fixed(x, frac):
    """GELU(x) ≈ x · sigmoid(1.702 x) in Q<frac>. Float-free (1.702 = 1702/1000)."""
    arg = _sdiv(rn.mul(_C_GELU, x), _MILLE)      # 1.702 * x
    s = sigmoid_fixed(arg, frac)
    return rn.mul(x, s) >> frac if x >= 0 else -(rn.mul(-x, s) >> frac)


def tanh_fixed(x, frac):
    """tanh(x) in Q<frac> via the identity tanh(x) = 2·sigmoid(2x) − 1. Float-free."""
    one = 1 << frac
    return (sigmoid_fixed(x << 1, frac) << 1) - one


def softcap_fixed(x, cap, frac):
    """Gemma logit/attention soft-cap: cap · tanh(x / cap), all Q<frac> (cap a plain int scalar).
    Bounds a score into (−cap, cap) smoothly. Float-free (tanh via sigmoid, /cap via mf_floordiv)."""
    t = tanh_fixed(_sdiv(x, cap), frac)      # tanh(x / cap)  in Q<frac>
    return rn.mul(t, cap)                    # · cap  (still Q<frac>; cap is an integer)


def layernorm_fixed(x, gamma, beta, frac, eps=1):
    """LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta, all Q<frac>. Mean-centered (unlike
    RMSNorm). sqrt = ring isqrt. Float-free, multiplier-free."""
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
    std = rn.isqrt(var << frac)
    if std == 0:
        std = 1
    out = []
    for i in range(n):
        norm = _sdiv(xc[i] << frac, std)
        out.append(_sdiv(rn.mul(norm, gamma[i]), 1 << frac) + beta[i])
    return out


def rmsnorm_fixed(x, weight, frac, eps=1):
    """RMSNorm: x / sqrt(mean(x²) + eps) · weight, all Q<frac>. sqrt = ring isqrt. Float-free.

    x, weight are Q<frac> integer lists. Normalizes MAGNITUDE (energy); a ring statistic, no Euclidean
    sqrt from libm — the ring's own integer isqrt."""
    if eps == 1:
        from ringkit.kernels.mprc.gemma import host as _kh
        fused = _kh.rmsnorm(x, weight, frac)     # ONE C block call, bit-for-bit gated at load (D9)
        if fused is not None:
            return fused
    n = len(x)
    ssq = 0
    for v in x:
        ssq = ssq + (rn.mul(v, v) >> frac)       # Σ x_i²  (Q<frac>)
    ms = rn.mf_floordiv(ssq, n) + eps            # mean square
    rms = rn.isqrt(ms << frac)                   # sqrt in Q<frac>: isqrt(ms·2^frac)
    if rms == 0:
        rms = 1
    out = []
    for i in range(n):
        norm = _sdiv(x[i] << frac, rms)          # (x_i / rms)  (Q<frac>)
        out.append(_sdiv(rn.mul(norm, weight[i]), 1 << frac))   # · weight_i, back to Q<frac>
    return out
