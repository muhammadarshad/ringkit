# CLAUDE.md — ringkit

Ring-topology (Z₂₅₆ / QH4) compute ecosystem. Hard ring physics/math lives in a verified
substrate; three engineer-facing facades hide it so ordinary engineers use it like numpy/torch
without touching mod-256, energy/phase, vacuums, or zero-divisor collapse.

## Run the verification loop (do this after every change)

```bash
# Run from the PARENT of the repo (the repo root IS the `ringkit` package):
cd .. && python3 -m ringkit.tests.run_all        # must print "ECOSYSTEM: ALL GREEN"
cd .. && python3 -m ringkit.tests.test_<module>  # a single suite
```

Kernels (C) build on first import via ctypes into `kernels/build/` (gitignored); if a `.so` is
stale, delete `kernels/build/` and re-import.
Builds target the **running interpreter's** architecture and are arch-keyed (`<stem>-<machine>.so`),
so the Rosetta x86_64 dev Python and a native arm64 one coexist. Native run (4x faster C kernels):
`arch -arm64 /usr/bin/python3 -m ringkit.tests.run_all` — also ALL GREEN.

## Non-negotiable disciplines (docs/project-governance/CHARTER.md — D1–D11) — break these and the work is wrong

- **D1 Verify by execution.** Never conclude without running it. Every non-trivial claim is
  backed by a test. Prefer exhaustive checks over the 256 ring where feasible.
- **Multiplier-free semantic layers.** No `*`, `//`, `**`, `/` and **no standard-math imports**
  — architectural, not aesthetic: multipliers ARE the silicon bottleneck; QSM + the ring
  dot-product (tables/adds/shifts) are the bypass that makes multiplier-free silicon buildable.
  (numpy/math/scipy) anywhere under `core/ stats/ linalg/ rnp/ rmath.py rcollections/ physics/ ml/ nn/ data.py`.
  (`ringkit.rnp` / `ringkit.rmath` are our REPLACEMENTS for numpy/math — original names per D10.)
  Use `rn.mul` (shift-add), `rn.ipow`, `rn.mf_floordiv`, `rn.ring_pow`, etc. AST-audit new files:
  `python3 -c "import ast,sys;[print(n.lineno,type(n.op).__name__) for n in ast.walk(ast.parse(open(sys.argv[1]).read())) if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Mult,ast.FloorDiv,ast.Pow,ast.Div))]" <file>`
- **Two-layer rule (D9).** Only `kernels/` (C/SIMD) may use hardware `*`/`<<`; it must reproduce
  the Python semantic reference **bit-for-bit**. Tests may use numpy/math **as a labeled oracle only**.
- **Fold late.** A ring quantity is `(energy, phase)` = `(quotient, remainder mod 256)`. Stay in
  unfolded ENERGY to keep operations reversible; fold to phase only when you need the ring position.
  Gradients (ENERGY) must NOT fold mod 256; values (ARC) do. Odd = unit (reversible); even =
  zero-divisor (collapses). Ring-native e = **3** (`rn.RING_E`, `ring_exp`/`ring_log`).
- **Solve, don't descend.** Linear layers = exact `linalg.solve` (mod 256). Nonlinear invertible =
  `linalg.fit` (invert-then-solve). Descent (`ml/autograd`) is the fallback, and it's flaky on the ring.
- **Exact vs approximate is labeled.** `SIN/COS` use the `_arch` semicircle (exact only at the 4
  cardinals; ~28–31% shape gap elsewhere). Never present an approximation as exact. Rotation by a
  quarter turn (`rotate`, iota) IS exact; general angle-addition is NOT.
- **Names are handles.** Wilson/Metropolis/Euclidean etc. are borrowed labels, not standard-math
  imports. Don't force ring behavior onto standard math or vice-versa.
- **Naming obeys 5W (D10).** Minted names answer Who/What/When/Where/Why and stay ORIGINAL:
  our namespaces are never named after the libraries they replace (`rnp` not numpy, `rmath` not math).
- **Math and physics first (D11).** Ask the mathematical/physical form BEFORE building. The physics
  here is QUANTUM, realized through MPRC (spins, vacuums, quantum walk, gauge, criticality) — never
  a re-implementation of standard-model/continuum methods; those are labeled handles only.

## The honesty bar for ANY learning/ML claim

Self-retrieval / "fits training" proves nothing (memorizing is trivial). Every ML capability must
show **held-out generalization** on data never seen, PLUS a **control that fails**: a random-label
run must collapse to chance, and/or a position-only/content-only baseline must fail. See
`tests/test_ml.py`, `tests/test_attention.py`, `tests/test_nn_facade.py`.

## Layout

```text
ringkit/
  core/constants.py  CORE ring constants (TAU/HALF/Q/Q2/SCALE/VACUUMS/RING_E/IOTA), single-sourced
                     and FROZEN (assignment raises). Never re-declare these in a subgroup — import
                     them. Subgroup-domain constants (qcm.HV_*, measure rulers) stay local.
  core/native.py     substrate ISA: Z256 consts, mul/ipow/mf_floordiv, qsm, isqrt, ring_neg,
                     SIN/COS/TAN family, ARC*, iota/IOTA rotor, ring_exp/ring_log/ring_pow (e=3),
                     ring_cis (Euler), rotate/cis_rotate, ADI, codec  — all multiplier-free
  core/calculus.py   d_rot / integral / differential / FTRC
  linalg/            solve.py (exact mod-256 solve, modinv, is_invertible), fit.py (invert-then-solve)
  stats/stats.py     ring_dist, ARCTAN2, circular mean/median, geometric_mean
  rnp/               numpy REPLACEMENT (original name, D10): __init__.py (the surface: rk.rnp /
                     import ringkit.rnp), tensor.py (RingTensor: the package's ndarray, bytearray-backed)
  rcollections/      ring-native data structures (placeholder — containers, not math objects;
                     original name per D10, and never shadow a stdlib module name: a package dir
                     named `collections` broke stdlib imports for any python run inside the repo)
  physics/           measure.py, qcm.py, gauge.py (SU(256) plaquette + Metropolis + criticality),
                     sim.py (Gauge facade class)
  ml/                autograd.py, tensor_autograd.py (TVar), optim.py, nn.py (low-level), attention.py
  kernels/           [D9 silicon] backend/ (ctypes loader __init__.py + ring_ops.c, zero-copy,
                     Python fallback; gemm.py + ring_gemm.c: ring GEMM in 3 gated variants —
                     hardware-`*` bridge 215 GMAC/s, multiplier-free shiftadd 55 [beats BLAS],
                     multiplier-free QSM table 13 — serves rnp matmul), mprc/qcm/ (qcm_kernel.c, cache_manifold.c),
                     mprc/lattice/ (gauge.c [threaded *_mt slab bins + C observable reductions] +
                     host.py: ctypes host, py reference, observables, GPU session_for),
                     apple/metal/ (ring_ops+gauge+gemm shaders, shim.m, host.py — all bit-for-bit
                     verified: elementwise OPT-IN [C wins]; gauge sweep AUTO >=32^3 + derived-RNG
                     thermalize [rk_mix32, py==C==metal] 0.12 ns/node/sweep; GPU GEMM mul 105 /
                     qsm-LUT 59 GMAC/s [zero-multiply, beats torch-mps] — CPU bridge keeps route), mprc/hpq/ + nvidia/cuda/ +
                     apple/ml/ (placeholders; CoreML descoped — unified-GPU focus),
                     build/ (arch-keyed .so, gitignored)
  tests/             one test_<module>.py each; run_all.py aggregates (20 suites)
  bench/             apples-to-apples vs numpy/torch (C6 scaffolding — the ONLY place standard
                     engines may be imported; baselines bit-for-bit gated before timing).
                     Results: docs/BENCHMARKS.md (native fight: GPU thermalize ~85x vs torch-mps
                     ON THE SAME unified GPU; ~65x vs multithreaded torch-cpu)
  docs/              project-governance/ (SDLC docs: CHARTER.md, SRD.md, ECOSYSTEM_SRD.md,
                     ECOSYSTEM.md, MANIFEST.md)

ringkit/nn/          FACADE (top-level pkg): layers.py (Layer, Linear, Dense, Sequential),
                     transformer.py (RoPE, Attention, TransformerBlock, Transformer: induction +
                     in-context recall; HopBlock + Stacked: multi-block solve-trained deep recall,
                     held-out 1.0 with depth + random controls at chance). All re-exported at
                     rk.nn. Ring hidden; .raw hatch.
ringkit/rmath.py     stdlib-math REPLACEMENT (original name, D10): math-shaped handles (sin/cos/exp/
                     log/isqrt, tau/pi/e with e = RING_E = 3) re-exported from core — no behavior of its own
ringkit/data.py      FACADE: encode/encode_range, one_hot, split, batches
```

Engineer entrypoint: `import ringkit as rk` → `rk.nn`, `rk.data`, `rk.physics.Gauge`, `rk.rnp`
(also `import ringkit.rnp as rnp`, `import ringkit.rmath as rmath`).
Every facade object hides ring internals and exposes `.raw` for power users.

## Status

All 20 suites green. Substrate (core/stats/linalg/rnp/physics/ml/kernels) is production-grade
and AST-clean (ops AND float literals — gauge/sim/data brought into compliance 2026-07-12). Facades (`rk.nn`, `rk.data`, `rk.physics`) built and verified with held-out + controls.
Next candidates: rnp-surface polish, top-level quickstart. Apple backends: Phases 0-1c
DONE (docs/project-governance/APPLE_BACKENDS_SRD.md); CoreML descoped.
