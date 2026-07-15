"""Ring-native encoder layers shared by the MPRC architectures.

`rope_encoder_layer` — one QK-normed QuantumRoPE transformer encoder block (pre-norm, residual):
    norm1 -> per-head Q/K/V proj -> per-head Q-norm,K-norm -> QuantumRoPE(Q,K) -> softmax attn
    -> out_proj (+residual) -> norm2 -> Linear-GELU-Linear FFN (+residual).
Gluon stacks `depth` of these with per-layer weights; Rotor shares ONE across recursive steps.

Float-free: ring layernorm/gelu (ract), shift-add linear + attention (infer), ring mul (native).
"""
from ringkit.core import native as rn
from ringkit.emulation import infer, ract
from ringkit.quanta._ringtrig import FRAC, ONE, _sd


def _rot_half(v, hd):
    """NeoX-style half rotation used by QuantumRoPE, split into two hd/2 halves each rotated
    by hd/4 (matches the deployed rope: [-x[8:], x[:8]] per 16-dim half for hd=32)."""
    q = hd >> 1; h = q >> 1
    xz, xx = v[:q], v[q:]
    return ([-a for a in xz[h:]] + xz[:h] +
            [-a for a in xx[h:]] + xx[:h])


def apply_rope(seq, cosq, sinq, hd, N):
    out = []
    for t in range(N):
        v = seq[t]; rh = _rot_half(v, hd)
        out.append([_sd(rn.mul(v[d], cosq[t][d]), ONE) + _sd(rn.mul(rh[d], sinq[t][d]), ONE)
                    for d in range(hd)])
    return out


def _rot_half_4d(v, hd):
    """QuantumRoPE4D rotate-half: FOUR hd/4 ADI-axis chunks, each rotated by its half
    (the deployed 4D rope convention; at hd=8 each 2-dim chunk becomes (-x1, x0))."""
    c = hd >> 2; h = c >> 1
    out = []
    for i in range(4):
        xc = v[rn.mul(i, c):rn.mul(i + 1, c)]
        out.extend(-a for a in xc[h:])
        out.extend(xc[:h])
    return out


def apply_rope_4d(seq, cosq, sinq, hd, N):
    out = []
    for t in range(N):
        v = seq[t]; rh = _rot_half_4d(v, hd)
        out.append([_sd(rn.mul(v[d], cosq[t][d]), ONE) + _sd(rn.mul(rh[d], sinq[t][d]), ONE)
                    for d in range(hd)])
    return out


def _gelu_tanh_list(xs):
    """gelu_pytorch_tanh over a vector — ONE C block call (host.gelu_mul with unit multiplier,
    bit-for-bit == gemma4.gelu_tanh_fixed), python reference fallback. The tanh form tracks the
    deployed exact-erf nn.GELU to ~3e-4 (the sigmoid form is ~30x coarser)."""
    from ringkit.kernels.mprc.gemma import host as _kh
    fused = _kh.gelu_mul(xs, _ones(len(xs)), FRAC)
    if fused is not None:
        return fused
    from ringkit.emulation.gemma4 import gelu_tanh_fixed
    return [gelu_tanh_fixed(v, FRAC) for v in xs]


_ones_cache = {}


def _ones(n):
    got = _ones_cache.get(n)
    if got is None:
        got = [ONE for _ in range(n)]
        _ones_cache[n] = got
    return got


def lattice_encoder_layer(x, W, name, cosRq, sinRq, N, C, NH, HD, gh, gw, radius=1):
    """One encoder block whose attention is the LOCAL lattice interaction: each token attends
    to its (2r+1)² torus neighbours on the (gh,gw) grid (the deployed LatticeAttention — the
    metropolis staple; physics locality IS the attention). QK-normed, QuantumRoPE4D, then the
    same pre-LN out_proj/FFN wrapper as rope_encoder_layer, with the tanh GELU."""
    hq = [ract.layernorm_fixed(r, W(name + "norm1.weight"), W(name + "norm1.bias"), FRAC) for r in x]
    qq = [infer.linear(r, W(name + "self_attn.q_proj.weight"), W(name + "self_attn.q_proj.bias"), C, C, FRAC) for r in hq]
    kq = [infer.linear(r, W(name + "self_attn.k_proj.weight"), W(name + "self_attn.k_proj.bias"), C, C, FRAC) for r in hq]
    vq = [infer.linear(r, W(name + "self_attn.v_proj.weight"), W(name + "self_attn.v_proj.bias"), C, C, FRAC) for r in hq]
    qnw = W(name + "self_attn.q_norm.weight"); qnb = W(name + "self_attn.q_norm.bias")
    knw = W(name + "self_attn.k_norm.weight"); knb = W(name + "self_attn.k_norm.bias")
    offs = [(di, dj) for di in range(-radius, radius + 1) for dj in range(-radius, radius + 1)]
    nb = [[(rn.mul((r + di) % gh, gw) + (c + dj) % gw) for di, dj in offs]
          for r in range(gh) for c in range(gw)]                      # torus neighbour indices
    ctx = [[0] * C for _ in range(N)]
    sc = infer.inv_sqrt(HD, FRAC)
    for hh in range(NH):
        base = rn.mul(hh, HD)
        qH = [ract.layernorm_fixed(qq[t][base:base + HD], qnw, qnb, FRAC) for t in range(N)]
        kH = [ract.layernorm_fixed(kq[t][base:base + HD], knw, knb, FRAC) for t in range(N)]
        vH = [vq[t][base:base + HD] for t in range(N)]
        qHr = apply_rope_4d(qH, cosRq, sinRq, HD, N)
        kHr = apply_rope_4d(kH, cosRq, sinRq, HD, N)
        for t in range(N):
            scores = [_sd(rn.mul(infer.dot(qHr[t], kHr[j], FRAC), sc), ONE) for j in nb[t]]
            w = infer.softmax(scores, FRAC)
            acc = [0] * HD
            for wi, j in zip(w, nb[t]):
                vj = vH[j]
                for d in range(HD):
                    acc[d] += rn.mul(wi, vj[d])
            row = ctx[t]
            for d in range(HD):
                row[base + d] = acc[d] >> FRAC
    aq = [infer.linear(r, W(name + "self_attn.out_proj.weight"), W(name + "self_attn.out_proj.bias"), C, C, FRAC) for r in ctx]
    x1 = [[x[t][j] + aq[t][j] for j in range(C)] for t in range(N)]
    ff_in = [ract.layernorm_fixed(r, W(name + "norm2.weight"), W(name + "norm2.bias"), FRAC) for r in x1]
    dff = len(W(name + "linear1.bias"))
    h1 = [_gelu_tanh_list(infer.linear(r, W(name + "linear1.weight"), W(name + "linear1.bias"), dff, C, FRAC)) for r in ff_in]
    l2 = [infer.linear(r, W(name + "linear2.weight"), W(name + "linear2.bias"), C, dff, FRAC) for r in h1]
    return [[x1[t][j] + l2[t][j] for j in range(C)] for t in range(N)]


def rope_encoder_layer(x, W, name, cosRq, sinRq, N, C, NH, HD):
    """One encoder block. `W(n)->fixed weight`; `name` is the layer weight prefix."""
    hq = [ract.layernorm_fixed(r, W(name + "norm1.weight"), W(name + "norm1.bias"), FRAC) for r in x]
    qq = [infer.linear(r, W(name + "self_attn.q_proj.weight"), W(name + "self_attn.q_proj.bias"), C, C, FRAC) for r in hq]
    kq = [infer.linear(r, W(name + "self_attn.k_proj.weight"), W(name + "self_attn.k_proj.bias"), C, C, FRAC) for r in hq]
    vq = [infer.linear(r, W(name + "self_attn.v_proj.weight"), W(name + "self_attn.v_proj.bias"), C, C, FRAC) for r in hq]
    qnw = W(name + "self_attn.q_norm.weight"); qnb = W(name + "self_attn.q_norm.bias")
    knw = W(name + "self_attn.k_norm.weight"); knb = W(name + "self_attn.k_norm.bias")
    ctx = [[0] * C for _ in range(N)]
    sc = infer.inv_sqrt(HD, FRAC)
    for hh in range(NH):
        qH = [ract.layernorm_fixed(qq[t][hh*HD:(hh+1)*HD], qnw, qnb, FRAC) for t in range(N)]
        kH = [ract.layernorm_fixed(kq[t][hh*HD:(hh+1)*HD], knw, knb, FRAC) for t in range(N)]
        vH = [vq[t][hh*HD:(hh+1)*HD] for t in range(N)]
        qHr = apply_rope(qH, cosRq, sinRq, HD, N)
        kHr = apply_rope(kH, cosRq, sinRq, HD, N)
        a = infer.attention(qHr, kHr, vH, FRAC, scale=sc)
        for t in range(N):
            for d in range(HD):
                ctx[t][hh*HD + d] = a[t][d]
    aq = [infer.linear(r, W(name + "self_attn.out_proj.weight"), W(name + "self_attn.out_proj.bias"), C, C, FRAC) for r in ctx]
    x1 = [[x[t][j] + aq[t][j] for j in range(C)] for t in range(N)]
    ff_in = [ract.layernorm_fixed(r, W(name + "norm2.weight"), W(name + "norm2.bias"), FRAC) for r in x1]
    dff = len(W(name + "linear1.bias"))
    h1 = [[ract.gelu_fixed(v, FRAC) for v in infer.linear(r, W(name + "linear1.weight"), W(name + "linear1.bias"), dff, C, FRAC)] for r in ff_in]
    l2 = [infer.linear(r, W(name + "linear2.weight"), W(name + "linear2.bias"), C, dff, FRAC) for r in h1]
    return [[x1[t][j] + l2[t][j] for j in range(C)] for t in range(N)]
