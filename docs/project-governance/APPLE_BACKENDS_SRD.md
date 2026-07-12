# Apple Backends SRD — Metal + CoreML/ANE (kernels/apple/)

**Status 2026-07-12:** Phase 0 DONE. Phase 1 elementwise DONE (bit-for-bit; C wins -> opt-in).
Phase 1b DONE: gauge sweep on Metal AUTO-ROUTES at >=32^3 (measured 2.9x at 48^3 -> 8.4x at
160^3 vs C), plus fused GPU-resident thermalize (batch of sweeps, grid crosses the bus once
per batch — unified-memory win). Plaquette measured ~10x slower on GPU (bandwidth-trivial),
stays on C. Phase 2 (CoreML/ANE) DESCOPED per user direction: the target is the unified-memory
GPU at CUDA-class throughput, access API irrelevant; ANE/CoreML parked. See Phase 1c roadmap.

Plan to wire the two Apple silicon paths as D9 silicon backends. Charter applies in full:
hardware ops are legal in these layers, but every kernel must reproduce the Python semantic
reference **bit-for-bit** before it is trusted (D9), every claim is verified by execution (D1),
and anything not bit-exact is labeled with its measured bound or rejected (D3/D4).

## Phase 0 — prerequisites

- **Native arm64 Python.** DONE (evidence): `/usr/bin/python3` is universal; the full suite
  runs ALL GREEN natively via `arch -arm64 /usr/bin/python3 -m ringkit.tests.run_all`.
  Native C-kernel baseline: 22.9 GMUPS vs 5.8 GMUPS under Rosetta (~4x). The dev interpreter
  is still Rosetta x86_64; switching it remains recommended for daily work.
  Found + fixed en route: `ringkit/collections` shadowed stdlib `collections` for any python
  run with CWD inside the repo -> renamed `rcollections` (D10: never mint a stdlib name).
- **Backend registry.** DONE: `kernels/backend` probes `[metal, cpu-c, python]`; a backend
  is eligible only after a load-time bit-for-bit self-test (all three ops, fixed vectors);
  `backend.backends()` / `backend.active(n)` report status and routing. Build artifacts are
  arch-keyed (`<stem>-<machine>.so`) so Rosetta and native interpreters share kernels/build/.

## Phase 1 — Metal compute (kernels/apple/metal/)

Bulk elementwise ring ops + the gauge stencil as Metal compute shaders (uint8 buffers,
zero-copy via `MTLBuffer` no-copy wrapping of the same bytearray memory).

- Files: `ring_ops.metal`, `gauge.metal`, `host.py` (loader), `shim.m` (small ObjC shim
  compiled to a `.dylib` by the same `_arch_flags()` build path, exposing
  create-device/compile-library/dispatch as C symbols for ctypes — mirrors `backend/`).
- Ops, in order: `ring_mul/add/sub` -> `plaquette` -> `metropolis_sweep` (checkerboard
  parity maps cleanly to threadgroups).
- Acceptance (each op):
  1. bit-for-bit vs the Python semantic reference over the full 256x256 operand table
     AND >=1M random-buffer elements (same bar test_kernels uses for C);
  2. beats the C SIMD path at >=1M elements or documents why it stays optional;
  3. graceful absence: no Metal device (CI, Linux) -> registry falls through to C.
- **Elementwise trio result (M1 Pro, measured):** (1) PASSED — exhaustive table + 1M random,
  from both Rosetta and native interpreters. (2) NOT met and never will be for single
  elementwise passes: they are bandwidth-trivial and the GPU path pays 3 buffer copies
  (C 6.5 GMUPS Rosetta / 22.9 native; Metal ~0.7-1.3 GMUPS). Documented optional:
  auto-routing off (`backend.METAL_MIN = None`). (3) PASSED — test_metal skip-as-pass.
  **Consequence:** Metal's justification is Phase 1b (gauge stencil: 6-neighbor + LUT work
  per byte) and fused GPU-resident chains — not elementwise.

## Phase 1c — unified-memory GPU roadmap (the CUDA-class push)

Measured on M1 Pro: sweep kernel 1.9-2.9 ns/node on GPU vs ~15.8 C; end-to-end
Gauge(128^3).thermalize(16) = 2.5x (17.65 -> 7.13 ns/node/sweep) because the CPU-side
Mersenne RNG now dominates (2 x 33 MB of randoms per 8-sweep batch). Next levers, in order:

1. **GPU-side counter RNG — DONE 2026-07-12.** rk_mix32 (lowbias32 mixer over
   (seed, sweep, node)) implemented THREE times — Python reference (bit-truth, lattice
   host), C (gauge.c metropolis_sweep_rng), Metal (gauge.metal) — and proven identical
   before serving (self-test gate + test_metal 3-way check). Only grid + 256-byte LUT
   cross the bus for a whole thermalize. Measured end-to-end (warmed, M1 Pro, 16 sweeps):
       64^3: 11.31 -> 1.14 ns/node/sweep ( 9.9x)     128^3: 11.71 -> 0.51 (23.0x)
      160^3: 11.76 -> 0.21 (56.0x)                   192^3: 11.90 -> 0.21 (57.4x)
   ~4.8 G node-updates/s at 160^3+ — CUDA-class for an 8-bit 6-neighbor Metropolis.
   Facade determinism verified: same seed -> identical grids on C and GPU routes.
   En route fix: dyld caches images BY PATH, so all three loaders now rebuild stale
   artifacts BEFORE the first CDLL (mtime vs source) — in-process reload never works.
2. **True zero-copy buffers**: page-aligned grid allocation (mmap) wrapped with
   newBufferWithBytesNoCopy — UMA means the copy in rk_metal_thermalize is pure waste.
3. **Persistent GPU session — DONE 2026-07-12** (shim ABI 6: session create/thermalize/
   read/write/free; GaugeSession in metal host; sim.Gauge holds one when eligible and syncs
   self.grid lazily on observable reads; bit-for-bit == per-call path, gated in test_metal).
   Measured on mixed workflows (thermalize+action cycles): ~1.9x at 128^3, wash at 64^3 —
   honest verdict: useful, not dramatic, because the derived RNG had already removed most
   bus traffic. THE REAL FIND of this item: the observables (mean_action/correlation) were
   pure-python loops eating ~99% of cycle time (~500 ms at 128^3); they now reduce in C
   (integer sums, host does the final divide) -> whole cycles dropped ~250x to ~2 ms.
4. ~~Multi-command-buffer overlap (fill randoms for batch k+1 while batch k runs)~~ —
   OBSOLETE: the derived counter RNG (item 1) removed random uploads entirely; there is
   nothing left to overlap. Kept for the record.

**Benchmark campaign (bench/apples_to_apples + docs/BENCHMARKS.md): COMPLETE 2026-07-12.**
Native fair fight incl. torch-mps on the same GPU; all engines bit-for-bit gated; physics
kernels swept, ring GEMM campaign measured (bridge 215 GMAC/s, multiplier-free shift-add
beats BLAS). Remaining open GPU items: ~~Metal GEMM~~ DONE 2026-07-12 (gemm.metal mul + qsm-LUT, ABI 5,
bit-for-bit gated: metal-mul 105 GMAC/s = 2x torch-mps; metal-qsm ZERO-multiply 59 GMAC/s
BEATS torch-mps's hardware-mul matmul — the LUT thesis on GPU fabric; CPU bridge 217 keeps
the tensor route). Persistent GPU session (item 3) still open.

## Phase 2 — CoreML/ANE (kernels/apple/ml/) [DESCOPED 2026-07-12]

Parked per user direction (unified-GPU focus). The plan below is kept for the record.

Serve `rk.nn` inference (solve-trained Linear/Dense, attention routing) through CoreML so
batched predict can ride the ANE.

- Route: export a fitted facade layer's ring weights to an int-arithmetic CoreML model
  (mod-256 realized as subtract-and-mask graph ops). NO float path is accepted silently:
  - if the compiled model reproduces the semantic reference bit-for-bit over exhaustive
    (small dims) or >=10^6 random cases, it may serve values (ARC);
  - otherwise it is REJECTED for values and may only be offered as a labeled approximate
    scorer (D3), never the default.
- The honesty bar applies to any speed/accuracy claim: held-out inputs, plus a control
  (e.g. a perturbed-weight model must NOT match — proves the check can fail).
- Non-goal: training on ANE. The ring trains by exact solve (D: solve, don't descend);
  only inference is offloaded.

## Phase 3 — integration + status

- `test_metal.py`, `test_coreml.py` suites in run_all (skip-as-pass with a printed reason
  when the hardware/toolchain is absent, so CI stays green off-Mac).
- CLAUDE.md layout + README status updated; benchmarks recorded with the environment
  (native vs Rosetta) stated — never a bare number.

## Risks / open questions

- CoreML int8 ops may not compose to an exact mod-256 matmul on ANE (it may promote to
  float internally). If exactness is unachievable, Phase 2 ships as labeled-approximate
  only, or not at all — the charter decides, not the benchmark.
- Metal shim adds an ObjC build dependency; keep it optional (absence = fall through).
- Zero-copy across Python/Metal needs `storageModeShared`; verify no hidden copies on
  Apple silicon unified memory.
