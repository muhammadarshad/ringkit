"""ringkit.qcm — the QCM Quantum Transformer, native to ringkit (Z256, float-free, kernel-backed).

A full native port of the `qcm-med-vision/transformer/` package (Soliton=MPRCMamba2,
Rotor=MPRCRDT, Gluon=MPRCViT + the shared QCM front-end, RoPE, and SSD body). The prior
`ringkit.quanta` reproduced the deployed torch model through the *emulation* engine (external
checkpoints, Q16); THIS package is the model's own native home: every op runs on the ring
substrate — QSM lossless int8 multiply (`core.native.qsm`), ring SIN/COS (SCALE=21), ring
isqrt/exp/log — with no IEEE float on the compute path and no torch/numpy anywhere.

Numeric substrate (owner: "no float — RingTensor already solved this"):
  - a value is a signed ring integer; a "float" is TWO rings (num/den, RingRational) — exact.
  - linear maps use QSM: offset-binary int8 (xbar = x+128) with power-of-2 (shift) scales;
    the quarter-square product a*b = (sq[|a+b|]-sq[|a-b|])>>2 is exact over all int8 pairs.
  - position/arc is mod-256 (ARC, fold-late); magnitude/energy is never folded.

Python is the interface only — the heavy ops route to the C/CUDA kernels (backend.gemm qsm,
nvidia.cuda). Verified by the ring's own laws (QSM losslessness, ring identities), never by a
float/torch oracle.
"""
from ringkit.qcm import constants
from ringkit.qcm.tensor import QSMLinear, qsm_matmul, quantize, rmsnorm
from ringkit.qcm.frontend import QuadrantRingProjector, VacuumDepthEmbedding
from ringkit.qcm.rope import QuantumRoPE, QuantumRoPE4D, mprc_grid_size
from ringkit.qcm.activations import sigmoid_fixed, softplus_fixed, gelu_fixed, exp_nonpos
from ringkit.qcm.ssd import Mamba2SSD
from ringkit.qcm.models import MPRCMamba2, MPRCRDT, MPRCViT

__all__ = [
    "constants",
    "QSMLinear", "qsm_matmul", "quantize", "rmsnorm",
    "QuadrantRingProjector", "VacuumDepthEmbedding",
    "QuantumRoPE", "QuantumRoPE4D", "mprc_grid_size",
    "sigmoid_fixed", "softplus_fixed", "gelu_fixed", "exp_nonpos",
    "Mamba2SSD", "MPRCMamba2", "MPRCRDT", "MPRCViT",
]
