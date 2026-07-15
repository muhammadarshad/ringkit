"""The MPRC architectures as ring-native forwards (float-free), sharing the QCM front-end.

  Rotor  (MPRCRDT)   — ONE shared rope encoder layer, `depth` recursive steps, inject + GRU gates.
  Gluon  (MPRCViT)   — `depth` INDEPENDENT rope encoder layers (per-layer weights).      [awaiting .pth]
  Soliton(MPRCMamba2)— num_layers × SSD selective scan.                                  [next build]

Each returns the L2-normalized image embedding. `W(name)->Q<frac> weight` is the fixed-point
accessor (see quanta.weights); the caller supplies the quantized RoPE cache + config so these
stay pure ring forwards. Gated end-to-end vs a numpy float oracle in tests/test_quanta.py.
"""
from ringkit.core import native as rn
from ringkit.emulation import infer, ract
from ringkit.quanta._ringtrig import FRAC, ONE, _sd
from ringkit.quanta.frontend import frontend
from ringkit.quanta.layers import lattice_encoder_layer, rope_encoder_layer
from ringkit.quanta.ssd import mamba2_ssd_layer


def _proj_head(pooled, W, C):
    """image_proj: Linear -> GELU -> LayerNorm -> Linear -> L2 normalize."""
    p = infer.linear(pooled, W("image_proj.net.0.weight"), W("image_proj.net.0.bias"), C, C, FRAC)
    p = [ract.gelu_fixed(v, FRAC) for v in p]
    p = ract.layernorm_fixed(p, W("image_proj.net.2.weight"), W("image_proj.net.2.bias"), FRAC)
    p = infer.linear(p, W("image_proj.net.3.weight"), W("image_proj.net.3.bias"), C, C, FRAC)
    ss = 0
    for v in p:
        ss += rn.mul(v, v) >> FRAC
    nrm = rn.isqrt(ss << FRAC) or 1
    return [_sd(v << FRAC, nrm) for v in p]


def rotor_forward(grid_q, W, cosRq, sinRq, D, Hh, C, NH, HD, depth,
                  prefix="vision_encoder."):
    """Rotor (RDT): shared recursive rope layer with inject + GRU gates. Returns Q<frac> embedding."""
    N = D * Hh
    E = prefix + "encoder."
    LY = E + "layer."
    zt = frontend(grid_q, W, D, Hh, C, prefix)
    x0 = [r[:] for r in zt]; h = [r[:] for r in zt]
    de = W(E + "depth_embed.weight")
    for step in range(depth):
        al = [[ract.sigmoid_fixed(v, FRAC) for v in
               infer.linear(r, W(E + "inject_gate.0.weight"), W(E + "inject_gate.0.bias"), C, C, FRAC)]
              for r in x0]
        h_in = [[h[t][j] + _sd(rn.mul(al[t][j], x0[t][j]), ONE) + de[step * C + j] for j in range(C)]
                for t in range(N)]
        hn = rope_encoder_layer(h_in, W, LY, cosRq, sinRq, N, C, NH, HD)
        g = [[ract.sigmoid_fixed(v, FRAC) for v in
              infer.linear(h[t] + hn[t], W(E + "gate.0.weight"), W(E + "gate.0.bias"), C, 2 * C, FRAC)]
             for t in range(N)]
        h = [[_sd(rn.mul(g[t][j], hn[t][j]), ONE) + _sd(rn.mul(ONE - g[t][j], h[t][j]), ONE)
              for j in range(C)] for t in range(N)]
    pooled = [_sd(sum(h[t][j] for t in range(N)), N) for j in range(C)]
    return _proj_head(pooled, W, C)


def gluon_forward(grid_q, W, cosRq, sinRq, D, Hh, C, NH, HD, depth,
                  prefix="vision_encoder."):
    """Gluon (ViT): `depth` INDEPENDENT rope encoder layers with per-layer weights, then mean+head.
    (Register tokens omitted — the deployed VLM path uses none; add when the .pth carries them.)"""
    N = D * Hh
    LY = prefix + "encoder.layers."
    z = frontend(grid_q, W, D, Hh, C, prefix)
    for li in range(depth):
        z = rope_encoder_layer(z, W, f"{LY}{li}.", cosRq, sinRq, N, C, NH, HD)
    pooled = [_sd(sum(z[t][j] for t in range(N)), N) for j in range(C)]
    return _proj_head(pooled, W, C)


def rotor_lattice_forward(grid_q, W, cosRq, sinRq, D, Hh, C, NH, HD, depth, n_classes,
                          prefix="enc.", head="cls.", radius=1):
    """Rotor as DEPLOYED in the webapp (RotorHeads): the same shared-recursive RDT loop as
    rotor_forward, but the layer's attention is the LOCAL lattice interaction
    (lattice_encoder_layer, radius-1 torus window, QuantumRoPE4D) and the readout is a bare
    classifier Linear on the mean-pooled tokens. Returns Q<frac> logits [n_classes]."""
    N = D * Hh
    E = prefix + "encoder."
    LY = E + "layer."
    zt = frontend(grid_q, W, D, Hh, C, prefix)
    x0 = [r[:] for r in zt]; h = [r[:] for r in zt]
    de = W(E + "depth_embed.weight")
    ig_w = W(E + "inject_gate.0.weight"); ig_b = W(E + "inject_gate.0.bias")
    g_w = W(E + "gate.0.weight"); g_b = W(E + "gate.0.bias")
    al = [ract.sigmoid_list(infer.linear(r, ig_w, ig_b, C, C, FRAC), FRAC) for r in x0]
    for step in range(depth):
        h_in = [[h[t][j] + _sd(rn.mul(al[t][j], x0[t][j]), ONE) + de[rn.mul(step, C) + j]
                 for j in range(C)] for t in range(N)]
        hn = lattice_encoder_layer(h_in, W, LY, cosRq, sinRq, N, C, NH, HD, D, Hh, radius)
        g = [ract.sigmoid_list(infer.linear(h[t] + hn[t], g_w, g_b, C, 2 * C, FRAC), FRAC)
             for t in range(N)]
        h = [[_sd(rn.mul(g[t][j], hn[t][j]), ONE) + _sd(rn.mul(ONE - g[t][j], h[t][j]), ONE)
              for j in range(C)] for t in range(N)]
    pooled = [_sd(sum(h[t][j] for t in range(N)), N) for j in range(C)]
    return infer.linear(pooled, W(head + "weight"), W(head + "bias"), n_classes, C, FRAC)


def soliton_forward(grid_q, W, cosRq, sinRq, D, Hh, C, NH, HD, num_layers, n_classes,
                    gh, gw, prefix="model."):
    """Soliton (Mamba2 gate_lat2d): num_layers × SSD, mean-pool, classifier head. Returns Q<frac>
    logits [n_classes]. (No residual between SSD layers — the deployed model replaces z each layer.)"""
    N = D * Hh
    z = frontend(grid_q, W, D, Hh, C, prefix)
    for li in range(num_layers):
        z = mamba2_ssd_layer(z, W, f"{prefix}layers.{li}.", cosRq, sinRq, N, C, NH, HD, gh, gw)
    pooled = [_sd(sum(z[t][j] for t in range(N)), N) for j in range(C)]
    return infer.linear(pooled, W(prefix + "head.weight"), W(prefix + "head.bias"), n_classes, C, FRAC)
