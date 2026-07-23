# kernels/cuda — GEVHV + TKV CUDA backend (D9 silicon)

CUDA C++ ports of the gevhv research kernels (the proven WHAT lives in
G:\quantum\research\gevhv\kernels\ and verify\; this backend is the HOW inside ringkit).

Planned contents (RINGKIT_INTEGRATION_PLAN.md P2/P4 — no code before plan approval):
- arc-side IMMA GEMM (licensed exact by Theorems A/B: s8 IMMA + `& 0xFF` is the ring product)
- fused GEVHV operator (bind-absorbed LUT + 5-site staple react + cdist measure,
  one int32 energy per manifold leaving the device)
- ring-L1 attention byte lanes (Theorem D SIMD identity)
- TKV probe/scoring paths as they earn a GPU tier

Discipline:
- D9: every kernel reproduces its pure-Python semantic reference (ml/gevhv.py,
  ml/tkvcache.py) BIT-FOR-BIT, self-tested at host load, before any timing.
- Build: nvcc with pinned -arch (bench discipline), MSVC host toolchain via vcvars64.
- Artifacts to kernels/build/ (arch-keyed), never committed.
- Performance reported last, caveated, oracle-gated (CORRECTION_REFOCUS).
