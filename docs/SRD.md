# Ring-Native Library — Specs · Requirements · Design · Tasks

Target: a ring-native (Z256 / QH4) numerical + tensor + autodiff library — a Torch/TF-kind of
stack whose math is derived from the ring, never borrowed from standard math.
Governed by `RING_NATIVE_CHARTER.md`. Current code: `ring_native.py`, `ring_stats.py`.

---

# 1. Specification

## 1.1 Purpose
Provide a complete numerical stack — scalars, trig, complex/rotor, calculus, statistics,
tensors, and automatic differentiation — where every operation is defined on the integer ring
Z256 (QH4). Standard math (floats, `math.*`, Euler, `sin/cos`) is used only to *verify*, never
to *compute*.

## 1.2 Vision
Reach feature-parity-in-spirit with a small Torch: `RingTensor` with vectorized ops, autograd,
`nn`-style modules, and optimizers — with a ring-native autodiff (`d = iota`, `∫ = accumulation`)
as the genuine differentiator rather than a re-skin of standard autodiff.

## 1.3 Scope
**In scope:** ring arithmetic; full trig family; rotor/complex; ADI engine; time codec; circular
& geometric statistics; tensor container; vectorized elementwise ops; broadcasting; reductions;
matmul; autograd; modules; optimizers; verification harness.
**Out of scope (for now):** GPU kernels; distributed training; a spatial codec for displacement
`d` (open in kinematics); interop that would require importing standard math into the core.

## 1.4 Definitions
- **Ring / τ = 256** — the state space; all base values are `0..255`.
- **ARC** — angular/positional unit (circular). **ENERGY** — magnitude unit (multiplicative).
- **Vacuums {0,64,128,192}** — structural nodes; identities are exact here.
- **QSM** — quarter-square product `x·y = (s²−d²)>>2`, `s`=accumulation, `d`=differential.
- **ADI** — accumulation/differential recovery; the `(∫, d)` pair.
- **Product ring C** — the larger ring a product lives in (`x·y` up to 255²).
- **(state, r) codec** — time ↔ ring, `SCALE_n = 675·1024ⁿ`.

---

# 2. Requirements

## 2.1 Functional (FR)

| ID | Requirement | Status |
|----|-------------|--------|
| FR1 | Ring arithmetic: neg, product (QSM), floor-div, isqrt, squares | DONE |
| FR2 | Direct trig: SIN, COS, TAN, KS4 | DONE |
| FR3 | Reciprocal trig: SEC, CSC, COT (VACUUM at poles) | DONE |
| FR4 | Inverse trig: ARCSIN, ARCCOS, ARCTAN (principal branch) | DONE |
| FR5 | Complex/rotor: qh_iota, iota_mul, polar_axis (4-value quadrant fold) | DONE |
| FR6 | ADI engine: derived_delta, recover, compress, evolve, mprc_axis_arcs | DONE |
| FR7 | Time codec: scale, encode, decode (lossless within one ring) | DONE |
| FR8 | Statistics: ring_dist, ARCTAN2, circular_mean, circular_median, geometric_mean | DONE |
| FR9 | Calculus operators as first-class API: `d` (=iota rotation), `integral` (=accumulation) | PARTIAL (verified as property; not yet exposed as ops) |
| FR10 | `RingTensor`: shape, ring-valued buffer, indexing, views | TODO |
| FR11 | Vectorized elementwise ring ops over `RingTensor` | TODO |
| FR12 | Broadcasting rules | TODO |
| FR13 | Reductions along axes: ring sum, circular_mean, median, geometric_mean | TODO |
| FR14 | Linear algebra: ring matmul (QSM-based), dot | TODO |
| FR15 | Autograd: computation graph, `backward()`, ring-native gradients | TODO |
| FR16 | Modules & params: `RingModule`, parameter registration | TODO |
| FR17 | Optimizers: ring-native update rule (SGD analogue) | TODO |

## 2.2 Non-functional (NFR) — from the Charter

| ID | Requirement |
|----|-------------|
| NFR1 | **No floats** in system code; integer only (C1). |
| NFR2 | **No standard-math imports** in the system; no `math`, no `numpy` in the core (C2). |
| NFR3 | **No Euler / π / sin-cos**; rotation = arc shift (C3). |
| NFR4 | **Multiplier-free** except the declared exceptions register (C4). |
| NFR5 | **Single source of truth** per form (C5). |
| NFR6 | Standard math **only in verification**, never imported by system (C6). |
| NFR7 | **Meaning preserved**: refactors reproduce prior values exactly (C7). |
| NFR8 | **Constants derived**, none arbitrary (C8). |
| NFR9 | **Verify by execution** before any claim (D1). |
| NFR10 | **Exact at structure**; approximations labeled with bounds (D3/D4). |

## 2.3 Acceptance criteria (per new form)
Pass the Charter's 8-point checklist: no standard imports, no floats, no undeclared `*`/`//`,
values reproduced where required, exact-at-structure with any approximation bounded, constants
derived, filed as system vs scaffolding, and every claim run.

---

# 3. Design

## 3.1 Layered architecture
```
L0  primitives      _SQ, ring_neg, qsm, isqrt_lut, scale21, mf_floordiv        [DONE]
L1  trig            _arch, SIN/COS/TAN/KS4, SEC/CSC/COT, ARC{SIN,COS,TAN}      [DONE]
L2  complex/rotor   qh_iota, iota_mul, polar_axis                              [DONE]
L3  ADI + calculus  derived_delta, recover, compress, evolve, axis_arcs;       [DONE / FR9 partial]
                    d = iota, integral = accumulation (FTRC)
L4  codec           scale, encode, decode                                      [DONE]
L5  statistics      ring_dist, ARCTAN2, circular_mean/median, geometric_mean   [DONE]
------------------------------------------------------------------------------------
L6  RingTensor      container + vectorized elementwise ops + broadcasting      [TODO]
L7  linalg/reduce   matmul (QSM), axis reductions                              [TODO]
L8  autograd        graph, backward, ring-native gradients                     [TODO]
L9  nn / optim      RingModule, parameters, optimizer update                   [TODO]
```
Rule: each layer may only call layers below it; none may import standard math (NFR2/NFR6).

**Core vs Measurement (immutable rule).** The **256 core** (ARC/identity: `[(a+b)²(c+d)²]² = 16² = 2⁸`)
is fixed — trig, identities, calculus, ADI all live here at amplitude SCALE=21. **Resolution is a
separate MEASUREMENT LAYER** (ENERGY side) applied *on* the core; higher resolution is never pushed
into the 256 core. The measurement rings are structural ring-counts, not arbitrary multipliers:

| ring | derivation | meaning |
|------|------------|---------|
| 256  | one axis / the core identity | ARC, immutable |
| 1024 | 4 × 256 | XYZU — each axis its own 256 ring |
| 512  | 256 + 256 | U accumulator + Energy overspill |
| 1536 | 1024 + 512 | full working ring (4 axes + accumulator + overspill) |
| 1808 | 16 × 113 (D × H) | **SIMD/L1-aligned hypervector**: 128 lanes × 113 = 14464 bits = 1808 bytes = 113 × 16-byte registers |

The `1808` = hardware layout: a 128-bit-SIMD-wide, L1-cache-aligned hypervector. This is a direct
input to **T2.1 (backing store)** — the natural `RingTensor` vectorized layout is these 113×128-lane
hypervectors, not an arbitrary buffer. Ties the measurement layer to concrete hardware.

## 3.2 Data model
- **RingScalar** — an `int` in `0..255` (implicit today). ARC or ENERGY by context.
- **RingTensor** — `{shape, data}` where `data` is a flat sequence of ring values plus strides.
  Backing store decision is OPEN (see 3.5).
- **Dual unit** — ARC values use circular ops (L2/L5 circular_*), ENERGY values use
  multiplicative ops (QSM, geometric_mean). The tensor should carry a unit tag so reductions pick
  the right central tendency (circular vs geometric vs plain ring-sum).
- **(state, r)** — the codec's sub-state residual `r` is the natural place a *finer-than-ring*
  quantity lives; candidate carrier for gradients (see 3.4).

## 3.3 Key interfaces (sketch)
- `RingTensor(shape, data, unit='arc'|'energy')`; `.reshape`, `.__getitem__`, `.T`.
- Elementwise: `radd, rsub, rmul(=qsm), rneg`; trig lifted: `SIN(t)`, etc. (table gather).
- Reductions: `rsum(t, axis)`, `cmean(t, axis)`, `cmedian(t, axis)`, `gmean(t, axis)`.
- `rmatmul(a, b)` — QSM products accumulated (ring sum) along the contract axis.
- Autograd: `t.requires_grad`, `y.backward()`, `t.grad`.

## 3.4 Autograd design (the differentiator, and the open fork)
- **Forward** builds a graph of ring ops.
- **Local derivatives are ring-native**: `d(SIN)=COS` (a +64 shift), `d(product)` via the QSM
  structure, chain rule = composition of arc shifts / accumulations.
- **DECIDED (T1.3, by spike + user): DUAL-RING.** Carry `(value, dvalue)` per node; `dvalue` is a
  second ring tracking the differential (ADI-style); backward composes differentials; local grads
  are ring closed-forms (`d SIN = COS`, etc.). Spike measured: corr 0.983 with analytic cos, ~0.8%
  dead gradients, full ±21 range — vs straight-through (72% dead) and residual (untested fairly).
  Autograd = ADI applied to the compute graph. Residual-carried is parked as a future refinement.

## 3.5 Backing-store decision (OPEN)
Vectorization needs a buffer. Options, weighed against NFR2:
- **Own list/bytearray + LUT gather** — fully charter-clean, slower.
- **numpy uint8 as dumb storage, arithmetic only via ring LUTs/ops** — fast, but risks NFR2;
  permissible ONLY if no numpy *arithmetic* touches ring values (numpy used as memory, not math),
  and it stays out of the core import graph. Must be explicitly ruled in or out.

## 3.6 Verification strategy
- Every form diffed against a reference oracle (`mprc_trig.py`) or brute force, across all 256 /
  exhaustive where feasible (D1).
- AST audit for `*`/`//`/`**`/`/` and imports on every change (Charter checklist item 3).
- Autograd: finite-difference check *in ring units* against the analytic derivative, reported as
  exact-at-structure + bounded elsewhere (D3/D4).

---

# 4. Tasks

Ordered, with dependencies. `[x]` done this session, `[ ]` remaining.

## Phase 0 — Foundation (COMPLETE)
- [x] T0.1 Primitives: `_SQ`, `qsm`, `isqrt_lut`, `scale21`, `mf_floordiv`, `ring_neg`
- [x] T0.2 Trig direct + reciprocal + inverse (FR2–FR4), verified 256/256
- [x] T0.3 Rotor/complex: `qh_iota`, `iota_mul`, `polar_axis`
- [x] T0.4 ADI engine + time codec
- [x] T0.5 Statistics: circular mean/median, geometric mean
- [x] T0.6 Charter, Manifest, this SRD

## Phase 1 — Calculus as API (FR9)  [COMPLETE]
- [x] T1.1 `d_rot` (iota rotation): d(SIN)=COS, d²=−SIN verified over 256  (`ring_calculus.py`)
- [x] T1.2 `differential`/`integral`; FTRC verified exact on 2000 random seqs
- [x] T1.3 Gradient representation DECIDED = dual-ring (spike measured; see §3.4)

## Phase 2 — RingTensor + vectorization (FR10–FR12)
- [x] T2.1 Backing store RESOLVED = pure-Python list for the core (charter-clean, NFR2). numpy-as-buffer
      deferred as an optional perf swap that would require a Charter exception.
- [ ] T2.2 `RingTensor` container: shape, strides, views, indexing
- [ ] T2.3 Lift L0/L1 ops to elementwise over `RingTensor` (table gather)
- [ ] T2.4 Broadcasting rules + tests
- [ ] T2.5 Unit tag (arc/energy) plumbed through ops

## Phase 3 — linalg + reductions (FR13–FR14)  [COMPLETE]
- [x] T3.1 Axis reductions: `rsum`/`mean`/`median` along a 2D axis, unit-aware; verified
- [x] T3.2 `matmul` via QSM + ring accumulation; verified == brute integer matmul mod 256 (300 shapes)
- [x] T3.3 Overspill square table in `ring_measure` (`_SQ_EXT`, size 1024); `measure_sin` to N=1808

## Phase 4 — Autograd (FR15)  [core COMPLETE]
- [x] T4.1 Computation graph (tape) over ring ops — `ring_autograd.Var`
- [x] T4.2 Backward, dual-ring local grads (add/sub/mul/neg/sin/cos) — reverse-mode, verified
- [x] T4.3 Verified: product rule EXACT vs finite-diff (step 1); SIN grad = closed-form COS (spike 0.983)
- [x] T4.4 Grads: trig (d SIN=COS, d COS=−SIN over 256), product (QSM product rule); chain rule verified
- [ ] T4.5 (next) lift autograd over `RingTensor` (elementwise + matmul backward)

## Phase 5 — nn + optim (FR16–FR17)  [module works; nonlinear dynamics open]
- [x] T5.1 `RingModule` + parameter registration (`ring_nn.py`): params register, grads flow to all.
- [x] T5.7 **Linear = SOLVE, not descend (100%).** `ring_solve.solve` (mod-256 Gaussian elimination,
      Newton modinv) recovers linear layers EXACTLY — 652/652 random solvable systems, loss 0.
      Gradient descent's 98%/stall was the wrong tool: wrapping loss is non-convex. Architecture:
      solve linear layers, descend nonlinear ones.
- [x] T5.4 DIAGNOSED (math-first). Stall is NOT activation coarseness (the earlier measure_sin
      hypothesis was WRONG). Real cause: **arg-space step granularity** — one ±1 on W_i moves arg by
      qsm(1,x_i)=x_i, so an epoch moves arg by Σx_i (a coarse lattice, e.g. 15); the target angle sits
      between lattice points -> exact limit cycle (arg 0<->15, loss 36<->49).
- [x] T5.5 FIX built + verified (`ring_optim.coordinate_step`): coarse-to-fine, loss-gated, with
      plateau crossing (accept non-worsening moves in the descent direction to cross SIN level-sets).
      **Single-point / simple nonlinear now converges to loss 0** — the math-derived fix works.
- [x] T5.6 RESOLVED by math (invert-then-solve, `ring_fit`). SIN is invertible (ARCSIN preimages),
      so nonlinear fit = invert activation + exact linear solve — NOT descent. The 5-point case that
      stalled at 70 now solves EXACTLY (loss 0). Random SIN-fits: 185/200 exact; the 15 misses have no
      invertible-mod-256 point triple (genuinely under-determined, real Z256 property). Descent was the
      wrong tool; the ring solves it.
- [x] T5.2 Optimizer DECIDED by spike: update rule is NOT the bottleneck — sign-SGD suffices ONCE
      gradients are ENERGY (non-wrapping). Real fix was ARC-value / ENERGY-gradient split in autograd.
      `ring_optim.sgd_step`. Measured 36% (wrapping) -> 100% (energy) convergence.
- [x] T5.3 End-to-end verified: system autograd + ring_optim converges 256/256 from all starts (avg 36 steps).

## Milestones
- **M1** (Phase 1–2): "feels like a library" — `RingTensor` with vectorized ring ops.
- **M2** (Phase 3): ring linear algebra usable.
- **M3** (Phase 4): ring-native autograd verified.
- **M4** (Phase 5): a model trains end-to-end, no standard math in the core.

## Cross-cutting (every task)
- [ ] Charter 8-point checklist run and recorded
- [ ] Reference diff / brute verification by execution
- [ ] Filed as system vs scaffolding in the Manifest
