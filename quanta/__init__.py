"""
ringkit.quanta — the MPRC quantum architectures on the ring (float-free forwards).

The family (codenames are the physics quanta they're named for):
    Rotor   = MPRCRDT     recursive shared-layer rotor            (rotor_forward)   [SHIPPED, gated]
    Gluon   = MPRCViT     independent rope encoder stack          (gluon_forward)   [awaiting .pth]
    Soliton = MPRCMamba2  SSD selective scan                      (soliton_forward) [next]

All three share the QCM front-end (quadrant projection + vacuum depth embedding) and the
L2-normalized projection head; they differ only in the body. No float on the compute path —
ring cos/sin table, shift-add MAC (infer), ring layernorm/gelu/sigmoid (ract), ring isqrt.
Naming: original per D10 (NOT `transformers` — that names the library it parallels).
"""
from ringkit.quanta.models import (rotor_forward, rotor_lattice_forward, gluon_forward,
                                   soliton_forward)
from ringkit.quanta.frontend import frontend, quadrant_project, add_vacuum_depth
from ringkit.quanta.layers import lattice_encoder_layer, rope_encoder_layer
from ringkit.quanta.ssd import mamba2_ssd_layer, softplus_fixed

__all__ = ["rotor_forward", "rotor_lattice_forward", "gluon_forward", "soliton_forward",
           "frontend", "quadrant_project", "add_vacuum_depth", "lattice_encoder_layer",
           "rope_encoder_layer", "mamba2_ssd_layer", "softplus_fixed"]
