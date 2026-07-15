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

# C fast-path routing (D9): tensors at/above this size go through ONE gated C GEMV block call
# per weight digit-plane instead of the per-element Python shift-add loop. Bit-identical.
_C_MIN_WORK = 1 << 12
_plane_cache = {}     # id(W) -> (W, [bytearray plane, ...]): balanced base-256 weight digits
_sz_cache = {}        # M -> ([0]*M, [1]*M) pinned so the host's s/z identity cache holds


def _weight_planes(W):
    """Balanced base-256 digit planes of a flat signed integer weight tensor: W[i] = Σ_p d_p·256^p
    with d_p in [-128,127], stored as bytes d_p+128 (the C kernel's xbar form). Exact, and built
    with shifts/masks only (multiplier-free). Cached per tensor identity — encoded once."""
    key = id(W)
    got = _plane_cache.get(key)
    if got is not None and got[0] is W:
        return got[1]
    rem = [int(v) for v in W]
    planes = []
    while any(rem):
        pb = bytearray(len(rem))
        for i in range(len(rem)):
            w = rem[i]
            d = ((w + 128) & 255) - 128            # balanced digit in [-128, 127]
            pb[i] = d + 128
            rem[i] = (w - d) >> 8                  # exact: (w - d) is a multiple of 256
        planes.append(pb)
    _plane_cache[key] = (W, planes)
    return planes


def _sz_rows(M):
    got = _sz_cache.get(M)
    if got is None:
        got = ([0 for _ in range(M)], [1 for _ in range(M)])   # raw dot: shift 0, divisor 1
        _sz_cache[M] = got
    return got


def _linear_c(x, W, b, M, K, frac):
    """The same y = (W·x >> frac) + b through the gated C GEMV (kernels/mprc/gemma/host), one
    block call per weight digit-plane, plane dots recombined by shifts. Exact integer dot per
    plane == the kernel's proven identity, so the sum is bit-identical to the Python loop.
    Returns None when the kernel (or the int64 activation range) is unavailable."""
    from ringkit.kernels.mprc.gemma import host as _kh
    if not _kh.available():
        return None
    planes = _weight_planes(W)
    s_row, z_row = _sz_rows(M)
    acc = [0 for _ in range(M)]
    sh = 0
    try:
        for pb in planes:
            d = _kh.gemv_exact(pb, x, M, K, s_row, z_row, frac)
            if d is None:
                return None
            for j in range(M):
                acc[j] += d[j] << sh
            sh += 8
    except OverflowError:                          # activations beyond int64: Python reference
        return None
    return [(acc[j] >> frac) + b[j] for j in range(M)]


def linear(x, W, b, out_features, in_features, frac):
    """y = W·x + b in ring fixed-point. x/W/b are signed Q<frac> integers; W is row-major flat
    (out_features * in_features). Product Q<2·frac> accumulated in ENERGY, rescaled >>frac to Q<frac>.
    Shift-add only — no float, no '*'. Large tensors run as gated C block calls (bit-identical)."""
    if rn.mul(out_features, in_features) >= _C_MIN_WORK:
        y = _linear_c(x, W, b, out_features, in_features, frac)
        if y is not None:
            return y
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
    lim = frac << frac                             # exact saturation: for s-m <= -frac·2^frac the
    exps = [ract.exp_fixed(s - m if s - m > -lim else -lim, frac)   # reciprocal exp floors to 0
            for s in scores]                       # anyway (e^frac > 2^frac), so bit-identical —
                                                   # and the huge-arg bigint blowup never happens
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
