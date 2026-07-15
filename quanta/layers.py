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
