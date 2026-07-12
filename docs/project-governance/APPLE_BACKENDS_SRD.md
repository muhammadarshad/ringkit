# Apple Backends SRD — Metal + CoreML/ANE (kernels/apple/)

Plan to wire the two Apple silicon paths as D9 silicon backends. Charter applies in full:
hardware ops are legal in these layers, but every kernel must reproduce the Python semantic
reference **bit-for-bit** before it is trusted (D9), every claim is verified by execution (D1),
and anything not bit-exact is labeled with its measured bound or rejected (D3/D4).

## Phase 0 — prerequisites

- **Native arm64 Python.** The current interpreter is x86_64 under Rosetta 2
  (`sysctl.proc_translated=1`). Metal works under Rosetta but ANE dispatch and honest
  benchmarks need a native toolchain. Acceptance: `platform.machine() == 'arm64'`,
  full suite green, C-kernel baseline re-measured native (expect >210x figure to move).
- **Backend registry.** Generalize `kernels/backend` into a probe/dispatch registry:
  ordered candidates `[metal, cpu-c, python]` per op family; each backend must pass a
  load-time bit-for-bit self-test on a fixed vector set before it is eligible. Fallback is
  automatic and silent-but-logged (`backend.active()` reports what's serving).

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

## Phase 2 — CoreML/ANE (kernels/apple/ml/)

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
