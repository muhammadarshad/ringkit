"""Shared QCM physics front-end for all three MPRC architectures (Gluon/Rotor/Soliton), ring-native.

    grid (quantized arc, [N][C] Q<frac>)
      -> QuadrantRingProjector : per channel, the 4 ring quadrants (cos, sin, -cos, -sin)+, ×modulation,
                                 then a linear projection  ->  [N][C]
      -> VacuumDepthEmbedding  : + per-depth-row embedding (the 4 vacuum positions carry depth)

Float-free (integer arc → ring cos/sin table, shift-add MAC via infer.linear). This is the exact
front-end the deployed encoders share; the three models differ only in the body that follows.
"""
from ringkit.core import native as rn
from ringkit.emulation import infer
from ringkit.quanta._ringtrig import COSQ, SINQ, FRAC, ONE, _sd


def quadrant_project(grid_q, modulation, proj_w, proj_b, D, Hh, C):
    """grid_q: [N=D*Hh][C] arc values in Q<frac>. Returns [N][C] projected tokens (Q<frac>)."""
    N = D * Hh
    proj_in = C * 4
    out = []
    for r in grid_q:
        ch = []
        for chan in range(4):                       # cos, sin, -cos, -sin  (ReLU: keep positive)
            for w in range(C):
                a = (r[w] >> 8) & 0xFF               # arc byte
                cc = COSQ[a]; sc = SINQ[a]
                val = (cc, sc, -cc, -sc)[chan]
                ch.append(val if val > 0 else 0)
        modded = [_sd(rn.mul(ch[j], modulation[j]), ONE) for j in range(proj_in)]
        out.append(infer.linear(modded, proj_w, proj_b, C, proj_in, FRAC))
    return out


def add_vacuum_depth(tokens, depth_w, D, Hh, C):
    """+ per-depth-row embedding: token at (depth d, spatial hh) gets depth_w[d*C : d*C+C]."""
    return [[tokens[d * Hh + hh][j] + depth_w[d * C + j] for j in range(C)]
            for d in range(D) for hh in range(Hh)]


def frontend(grid_q, W, D, Hh, C, prefix="vision_encoder."):
    """Full shared front-end: quadrant project + vacuum depth. `W(name)->fixed weight` accessor."""
    tok = quadrant_project(grid_q,
                           W(prefix + "quadrant_proj.modulation"),
                           W(prefix + "quadrant_proj.proj.weight"),
                           W(prefix + "quadrant_proj.proj.bias"),
                           D, Hh, C)
    return add_vacuum_depth(tok, W(prefix + "vacuum_emb.depth_emb.weight"), D, Hh, C)
