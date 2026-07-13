"""
ringkit.infer — ring-native inference on loaded models. NO float on the inference path.

Pretrained weights arrive as IEEE floats; `checkpoint.load_fixed` converts them to signed
fixed-point integers (value * 2**frac) by INTEGER mantissa-shift alone (no FPU). Inference then
runs entirely in ring integer arithmetic — `rn.mul` (shift-add), integer accumulate, and a `>>frac`
rescale (fold-late). So wherever a float would appear, it is already a ring integer: no float stands.

A fixed-point integer IS the ring (energy, phase) pair: ARC = value & 0xFF, ENERGY = value >> 8.

Multiplier-free. No numpy, no math, no floats.
"""
from ringkit.core import native as rn


def linear(x, W, b, out_features, in_features, frac):
    """y = W·x + b in ring fixed-point. x/W/b are signed Q<frac> integers; W is row-major flat
    (out_features * in_features). Product Q<2·frac> accumulated in ENERGY, rescaled >>frac to Q<frac>.
    Shift-add only — no float, no '*'."""
    y = []
    base = 0
    for _j in range(out_features):
        acc = 0
        for i in range(in_features):
            acc += rn.mul(x[i], W[base + i])       # shift-add product, exact integer
        y.append((acc >> frac) + b[_j])            # Q2f -> Qf, add bias (already Qf)
        base += in_features
    return y


def dot(a, b, frac):
    """Ring fixed-point dot product of two Q<frac> integer vectors -> Q<frac> scalar. Shift-add."""
    acc = 0
    for i in range(len(a)):
        acc += rn.mul(a[i], b[i])
    return acc >> frac


def _sdiv(n, d):
    """Signed integer divide (truncate toward zero), d > 0. Multiplier-free (mf_floordiv)."""
    if n < 0:
        return -rn.mf_floordiv(-n, d)
    return rn.mf_floordiv(n, d)


def inv_sqrt(k, frac):
    """1/sqrt(k) in Q<frac>, float-free (ring isqrt). Used as the attention score scale."""
    sq = rn.isqrt(k << (frac + frac))              # sqrt(k) in Q<frac>
    if sq == 0:
        sq = 1
    return rn.mf_floordiv(1 << (frac + frac), sq)   # 1 / sqrt(k)


def softmax(scores, frac):
    """Numerically-stable fixed-point softmax: subtract max, exp (ring Taylor), normalize by the
    integer sum. Returns Q<frac> weights that sum to ~1. Float-free."""
    from ringkit.emulation import ract
    m = scores[0]
    for s in scores:
        if s > m:
            m = s
    exps = [ract.exp_fixed(s - m, frac) for s in scores]      # in (0, 2^frac]
    z = 0
    for e in exps:
        z += e
    if z == 0:
        z = 1
    return [rn.mf_floordiv(e << frac, z) for e in exps]


def attention(Q, K, V, frac, scale=None):
    """Scaled-dot-product attention in ring fixed-point. Q:(Lq,d) K:(Lk,d) V:(Lk,dv), all Q<frac>.
    scores = (q·k)·scale, softmax, out = Σ w·v. Float-free (dot/softmax/blend all integer)."""
    dv = len(V[0])
    out = []
    for qi in Q:
        scores = [dot(qi, kj, frac) for kj in K]
        if scale is not None:
            scores = [_sdiv(rn.mul(s, scale), 1 << frac) for s in scores]
        w = softmax(scores, frac)
        acc = [0 for _ in range(dv)]
        for j in range(len(V)):
            wj = w[j]
            vj = V[j]
            for d in range(dv):
                acc[d] += rn.mul(wj, vj[d])
        out.append([a >> frac for a in acc])
    return out
