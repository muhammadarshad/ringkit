# ringkit — Ecosystem SRD (Specs · Requirements · Tasks)

Governing plan to bring the ring-topology ecosystem to **production grade, layer by layer, in
dependency order**. North star: `ECOSYSTEM.md`. Rules: `CHARTER.md` (D1–D8). No weak code.

---

## 1. Specification

**What we are building.** A complete ring-topology (Z₂₅₆ / QH4) compute ecosystem — silicon
kernels → foundation → libraries → frameworks → engines → domains — so all MPRC work (Quantum,
Physics, Math, Geometry, AI) runs on our own foundation, never on borrowed math.

**Definition of "production-grade" (the bar every layer must clear):**
1. **Complete API** — the operations a real user of that layer expects, not just a ring port.
2. **Robust errors** — `ValueError`/`IndexError`/`TypeError` with messages; no user-facing `assert`.
3. **Full tests** — a `tests/test_<layer>.py`: ring-internal invariants + external oracle cross-checks
   (numpy/math used ONLY as a labeled oracle, never imported by the library).
4. **Documented** — module docstring + per-form docstrings; entry in README/MANIFEST.
5. **Charter-clean** — AST-audited: no `*`/`//`/`**`/`/`, no standard-math imports.
6. **Two-layer discipline** — Python is the correct *semantic* reference; speed lives in the
   `kernels/` silicon layer (C/SIMD), cross-checked against the Python reference.

---

## 2. Requirements

### 2.1 Non-functional (every layer)
NFR1 multiplier-free (mul/ipow/mf_floordiv) · NFR2 no standard-math import in library ·
NFR3 no floats in ring semantics · NFR4 exact at structure, approximations labeled (D3/D4) ·
NFR5 proper exceptions · NFR6 per-module tests with oracle cross-check · NFR7 docstrings + manifest
entry · NFR8 names-are-handles (D8) · NFR9 whole-package AST audit stays clean.

### 2.2 Functional (per layer) + production acceptance

| Layer | Module | Production functional scope | Accept when |
|-------|--------|------------------------------|-------------|
| Foundation | `core/native` | ring consts, mul/ipow/mf_floordiv/mf_mod, qsm, isqrt, ring_neg, SIN/COS/TAN/KS4, SEC/CSC/COT, ARC*, iota/polar, ADI, codec — all with bounds/errors + exhaustive 256 tests | all identities exact over 256; errors raised; tests green |
| Foundation | `core/calculus` | d_rot/integral/differential + FTRC, exposed cleanly | FTRC exact on random seqs; d(SIN)=COS over 256 |
| Library | `stats/stats` | ring_dist, ARCTAN2, circular_mean/median, geometric_mean, resultant_length + errors | verified vs brute/float oracle; edge cases |
| Library | `linalg/solve` | modinv, solve (singular detection), det parity, batched | exact recovery; singular raises; random systems |
| Library | `linalg/fit` | invert-then-solve, preimage search, unsatisfiable→None | exact on solvable; honest None otherwise |
| Array | `array/tensor` + `array/numpy` | **DONE — production** | ✓ 40+ tests vs numpy mod 256 |
| Physics | `physics/measure` | rulers, measure_sin (overspill), errors | measure_sin exact vs bridge; N-range guarded |
| Physics | `physics/qcm` | state ops, 4 rings + anti-strides, walk, hypervector, manifold | matches QCM docs; verified constants |
| Framework | `ml/autograd` | dual-ring Var, ops, backward, broadcast over tensors (T4.5) | grads exact vs finite-diff; tensor autograd |
| Framework | `ml/optim` | sgd/coordinate, schedules | converges on covered problems |
| Framework | `ml/nn` | RingModule, layers (Linear=solve, Neuron), params | linear solve exact; training verified |
| Silicon | `kernels/` | C 8-bit SIMD kernels + **ctypes binding**, cross-checked vs Python | C == Python reference; measured speedup |
| Engines/Domains | (later) | QCM gauge engine, MPRC transformer | per their own specs |

---

## 3. Design

- **Package** (`ringkit/`) is set: `core / linalg / array / stats / physics / ml / kernels / tests / docs`.
- **Dependency order for hardening** (bottom-up so each rests on a finished base):
  `core/native` → `stats` → `core/calculus` → `linalg/{solve,fit}` → `array` (done) →
  `physics/{measure,qcm}` → `ml/{autograd,optim,nn}` → `kernels` (binding) → engines/domains.
- **Testing**: one `test_<module>.py` per module; a top-level `run_tests` aggregates; oracle
  cross-checks are labeled and confined to tests.
- **Semantic vs silicon**: Python modules are the reference of record; `kernels/` provides the fast
  path and must reproduce the Python results bit-for-bit (mod 256) before being trusted.

Known open research (not blockers, flagged in charter/manifest): global optimization for multi-point
*nonlinear* fitting (T5.6 handled by invert-then-solve for invertible activations; genuinely
non-invertible compositions remain descent-only).

---

## 4. Tasks (prioritized backlog, dependency-ordered)

**Done:** `array/tensor` + `array/numpy` — production (this pass).

- [x] **P1 — `core/native` to production** ✓ input validation + errors on public forms
      (qsm/isqrt ranges, mf_floordiv/mf_mod/ipow/scale/encode domains), docstrings, stale header fixed;
      `tests/test_native.py` exhaustive (primitives vs oracle, every trig identity over 256, ADI,
      codec, error paths) — ALL PASS; AST-clean; no regression in tensor/numpy.
- [x] P2 — `stats/stats` to production ✓ fixed `resultant_length` bug (any arc count, via new general
      `core.native.isqrt`); empty/negative validation on all aggregates; `tests/test_stats.py`
      (ring_dist, ARCTAN2, circular mean/median, geometric_mean vs oracle + edge/error) — ALL PASS;
      no regression; AST-clean.
- [x] P3 — `core/calculus` to production ✓ empty/single edge fixed in `ftrc_holds`; added
      `nth_differential`/`d_rot_power` with validation; `tests/test_calculus.py` (period-4 rotor cycle,
      FTRC on 3000 seqs, differential==odd-increments, edge/error) — ALL PASS; no regression; AST-clean.
      (Caught + fixed a test-oracle off-by-one; code was correct.)
- [x] P4 — `linalg/solve` + `linalg/fit` to production ✓ shape validation, `is_invertible` query,
      singular-mod-2 detection; `tests/test_linalg.py` (modinv 128 odd, 1011 random systems exact,
      fit 106/120 exact + 14 honest-None, all error paths) — ALL PASS; no regression; AST-clean.
- [x] P5 — `physics/measure` + `physics/qcm` to production ✓ measure_sin N even/range guards;
      `tests/test_physics.py` (measure + qcm state/topology + QCM constants vs source: anti-strides,
      128 singularity, N=3 tree, stride-7=36) — ALL PASS; AST-clean.
- [x] P6 — `ml/autograd` + `ml/optim` + `ml/nn` to production ✓ `tests/test_ml.py` (dual-ring grads
      over 256, chain/reuse, SIN scalar descent 256/256, Neuron train) — ALL PASS; AST-clean.
      (T4.5 tensor-autograd deferred as an ADDITION, not a gap.)
- [x] P7 — `kernels/`: `ring_ops.c` + `backend.py` (ctypes + Python fallback) ✓ `tests/test_kernels.py`
      — C == Python bit-for-bit, mul==qsm mod256, ~371x speedup — ALL PASS. (Silicon layer, D9.)
- [x] Cross-cutting: `tests/run_all.py` aggregator — **ECOSYSTEM ALL GREEN (9/9 suites)**; whole-package
      AST audit clean across all semantic layers (kernels excluded by D9).

**Rule of order:** finish and verify each Px before starting P(x+1). Do not jump around.

## ENGINES tier (beyond P1–P7 — the physics the ecosystem exists to run)

- [x] E1 — **SU(256) gauge engine** (`physics/gauge.py` + `kernels/gauge.c`): Wilson plaquette action
      (cache-blocked stencil, 0.09 ns/node) + checkerboard **Metropolis sweep** (ring-native, integer
      Boltzmann LUT, branchless). Verified: C == Python bit-for-bit; **thermalizes** (cold β=60 orders
      64→20, hot β=0 stays 64). `tests/test_gauge.py` ALL PASS.
- [x] E2 — **criticality scan** (`gauge.criticality_scan` + `gauge.correlation`): β-sweep with a
      ring-native order parameter (neighbor alignment) + mean action. Locates the confinement/
      deconfinement transition: β=0 → corr 0.52 / action 64 (disordered); β≥8 → corr ~0.83 /
      action ~24 (ordered). Sharp transition, monotone. `tests/test_gauge.py` ALL PASS.
- [~] E3 — [x] **tensor-autograd (T4.5)** (`ml/tensor_autograd.py`): dual-ring `TVar` over
      RingTensors — elementwise add/sub/mul, sin/cos, sum, 2-D matmul, reverse-mode. Grads are a
      signed non-wrapping buffer (values fold, gradients don't). Verified: grads match the scalar
      autograd cell-by-cell + matmul backward vs manual signed reference; AST-clean.
      `tests/test_tensor_autograd.py` ALL PASS. — [ ] MPRC transformer; [ ] GPU/CUDA sweep path.

Memory model: `RingTensor` now backs onto a C `bytearray` buffer; elementwise ops run zero-copy
through `kernels/backend.py` (data maintained in C, not Python lists). `+ 64-lane unroll kernels`.
