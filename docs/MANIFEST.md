# System vs Scaffolding — MPRC ring-native

Line drawn after this session. "System" = production forms that ship.
"Scaffolding" = tools used to understand/verify, kept for reference but not part of the engine.

## IN THE SYSTEM — `ring_native.py`

The single refined module. All values verified unchanged from the reference.

| Form | Role |
|------|------|
| constants: TAU, HALF, Q, Q2, SCALE, VACUUMS | ring geometry |
| `ring_neg` | ring negation |
| `_SQ`, `qsm`, `isqrt_lut`, `scale21`, `mf_floordiv` | multiplier-free primitives |
| `_arch`, `SIN`, `COS`, `TAN`, `KS4` | ring trig (direct) |
| `SEC`, `CSC`, `COT` | reciprocal trig (sign-correct, scaled 441; VACUUM at poles) |
| `ARCSIN`, `ARCCOS`, `ARCTAN` | inverse trig (principal-branch reverse LUT) |
| `qh_iota`, `iota_mul`, `polar_axis` | complex unit, rotor, the 4-value quadrant axis |
| `derived_delta`, `recover`, `compress`, `evolve`, `mprc_axis_arcs` | ADI engine |
| `scale`, `encode`, `decode` | time codec (kinematics), SCALE_n = 675*1024^n |

### `ring_stats.py` (statistics layer, imports ring_native)

| Form | Role |
|------|------|
| `ring_dist` | circular L1 distance min(\|a-b\|, 256-\|a-b\|) |
| `ARCTAN2` | ring atan2 -> arc (direction of a vector) |
| `circular_mean`, `resultant_length` | resultant-vector mean direction + concentration |
| `circular_median` | L1 circular center (exact minimizer; verified vs brute) |
| `geometric_mean` | multiplicative center; n=2 via isqrt(qsm), n>2 integer n-th root |

Note: `circular_median` and `geometric_mean` are exact; `circular_mean`/`ARCTAN2` inherit the
arch-vs-sine shape error (exact for concentrated data, ~5 units / ~7deg drift on spread data).

### `ring_calculus.py` (Phase 1, imports ring_native)

| Form | Role |
|------|------|
| `d_rot`, `integral_rot` | rotational derivative/integral (iota): d(SIN)=COS, verified |
| `differential`, `integral`, `ftrc_holds` | ADI (accumulation, differential) pair; FTRC exact |

### `ring_tensor.py` (Phase 2, imports ring_native + ring_stats)

| Form | Role |
|------|------|
| `RingTensor` | pure-Python container: shape, flat ring data, `unit` tag (arc/energy) |
| `radd/rsub/rmul(=qsm)/rneg/apply` | elementwise ops (same-shape + scalar broadcast) |
| `SIN/COS` (lifted), `rsum/mean/median(axis=)` | vectorized trig + unit-aware reductions (whole or 2D axis) |
| `matmul(A,B)`, `transpose` | 2D ring matmul (QSM); transpose |
| `__add__/__sub__/__mul__/__neg__/__matmul__/.T` | numpy-style operators (`+ - * @ .T`) |

### `ring_numpy.py` — OUR OWN numpy (ring-native), imports ring_native + ring_tensor

NOT the standard numpy. A numpy-style ndarray namespace on RingTensor.

| Form | Role |
|------|------|
| `array/zeros/ones/full/arange/eye` | creation |
| `dot/matmul`, `sum/mean/median(axis=)`, `reshape` | numpy-style free functions |

Every element is a ring value; every op ring-native + multiplier-free. Matmul cross-checked ==
real numpy mod 256. Does NOT import numpy.

### `ring_qcm.py` — QCM: holds the topologies (imports ring_native)

From SILIQ (ALGORITHM.md, verified) + QCM paper (Zenodo 18883754, abstract only — PDF body binary).

| Form | Role |
|------|------|
| `spin/polarity/state/conjugate/quadrant/is_vacuum` | bit-encoded QCM node state (UP+/UP-/DN+/DN-) |
| `seven_prime_walk()` | LATTICE traversal: 7-prime steps (2,3,5,7,11,13,17), 252/252 coverage |
| `HV_W/HV_H/HV_CELLS/HV_BYTES`, `hypervector()` | HYPERVECTOR: 128x113 = 14464 uint8 = 14.5KB (L1) |
| `midpoint/arms/manifold_coord` | MANIFOLD: biopod (radius k, height arctan(k/N)) |

Topologies held: LATTICE (Z256 ternary-state ring), TORUS (conjugate/periodic), HYPERVECTOR
(128x113 uint8), MANIFOLD (biopod). Verified: conjugate==ring_neg, product rule 252/252, 7-prime
100% coverage, HV=14464 bytes. NOTE: the full QCM SU(N_c) gauge MCMC engine (Weyl-hash PRNG,
register-forced 8-bit SIMD) is NOT reproduced — only its abstract was accessible; this module holds
the verified state/topology primitives.

Decisions locked: gradient representation = **dual-ring** (Phase 4); RingTensor backing =
**pure-Python list** (numpy-as-buffer deferred, would need a Charter exception).

### `ring_measure.py` (MEASUREMENT layer / ENERGY side, imports ring_native)

The selectable ruler applied *on* the immutable 256 core — never inside it.

| Form | Role |
|------|------|
| `CORE/AXES/ACC_OVR/WORKING` | measurement rings: 256, 1024 (XYZU), 512 (U-accum+overspill), 1536 |
| `measure_sin(phi, N)` | high-res amplitude (undivided arch); N≤1808 (overspill table) |
| `SIN512` | spinor double-cover wave (= measure_sin at 512); ENERGY side |
| `layout()` | measurement-ring summary (hypervector moved to ring_qcm) |

Note: measurement rings > 1024 need an extended "overspill" square table (measure_sin refuses,
does not fake it). Whole layer is multiplier-free and does not touch the 256 core.

### `ring_autograd.py` (Phase 4, imports ring_native)

Dual-ring reverse-mode autodiff — ADI applied to the compute graph.

| Form | Role |
|------|------|
| `Var` | node: value + accumulated differential (grad), both ring mod 256 |
| `add/sub/mul(=qsm)/neg/sin/cos` | ops with ring-native local grads |
| `backward(seed)` | reverse-mode; topological, gradient accumulation |

Values are ARC (fold mod 256, qsm); **gradients are ENERGY (signed, non-wrapping, mul)** — the split
that makes optimization work. Verified: `d SIN=signed(COS)`, `d(a²)=2·signed(a)`, product/chain rules.

### `ring_optim.py` (Phase 5, imports ring_native)

| Form | Role |
|------|------|
| `sign`, `sgd_step` | sign-SGD on an ARC parameter using an ENERGY gradient |
| `coordinate_step` | coarse-to-fine, loss-gated, plateau-crossing coordinate descent (T5.5) |

Spike finding: update rule isn't the bottleneck; ENERGY gradients are. Full stack (autograd+optim)
converges 256/256 from all starts. Multiplier-free.

### `ring_nn.py` (Phase 5 T5.1, imports ring_autograd + ring_optim)

| Form | Role |
|------|------|
| `RingModule` | parameter registration, `zero_grad`, `step(lr)` via ring_optim |
| `Neuron` | `SIN(sum W_i x_i + b)`, learnable ARC params |

Status (honest): grads flow to all params. **Linear layers SOLVE exactly (100%) — see ring_solve**,
not gradient descent. Nonlinear (SIN) simple cases converge to 0 (coordinate_step); multi-point
non-convex remains open (T5.6). Multiplier-free.

### `ring_solve.py` (linear = exact solve, imports ring_native)

| Form | Role |
|------|------|
| `modinv(a)` | inverse of odd a mod 256 (Newton/Hensel, multiplier-free) |
| `solve(A,b)` | exact mod-256 linear solve (elimination, odd pivots) |

A linear layer is a linear system mod 256 -> closed-form solution (det odd). 652/652 random
solvable systems fit exactly. Charter rule: solve linear, descend nonlinear.

### `ring_fit.py` (exact nonlinear fit, imports ring_native + ring_solve)

| Form | Role |
|------|------|
| `sin_preimages(t)` | ARCSIN preimage set (all arcs with SIN=t) |
| `fit(data, targets, ...)` | invert-then-solve: invert activation + exact linear solve |

Nonlinear fit = invert (invertible activation) + solve — NOT descent. 5-point case that stalled
at loss 70 now solves EXACTLY; 185/200 random SIN-fits exact (15 under-determined mod 256).

Properties held (verified): trig identical to reference across all 256; multiplier-free
except `evolve` scalar stepping; ADI odd-increment consistent; `iota` 4-cycle and
`i^2=-1` exact; `polar_axis` = {+UP, -UP, -DOWN, +DOWN}; time codec lossless within one
ring traversal [0, 86400*1024^n) with vacuum nodes at r=0 (float-free; uses integer * and //).

## NOT IN THE SYSTEM — scaffolding / reference

Kept on disk for provenance, not imported by the engine.

| Item | Why it's out |
|------|--------------|
| `mprc_trig.py` | reference oracle used to diff the multiplier-free forms; superseded by `ring_native.py` |
| `adi_wave.py` | working port + QSM bounds probes; forms folded into `ring_native.py` |
| `mprc_trig_mf.py` | intermediate multiplier-free draft; superseded by `ring_native.py` |
| `analytic_21sin` column / `mprc_sin_divergence.csv` / `mprc_sin_x1e4.csv` | comparison to standard sine — analysis, not engine |
| all PNG charts (arch_vs_sine, render_proof, which_column, polar_family, limacon_family, polar_quadrants, 21sin_shape) | visualization / understanding |
| `mprc_closing_map.md` | conceptual map to standard math (vectors/matrices/tensors/Lie) — documentation |

## Findings that inform the forms (not code)

- Ring `_arch` is a semicircle projection: exact = analytic sine only at the 4 vacuums; the
  shape gap (max ~28% relative) is geometric, not precision or degree-conversion.
- `Polar_SIN` = `255 + SIN` is a limaçon; the heart/cardioid is the `b=a` case; the family
  `r = b + SIN` is the microphone/antenna directivity family (omni -> cardioid -> figure-8).
- The multiply SCALE=21 dissolves as the XYZ scalar-axis sum 16+4+1 (`scale21`), no `*`.
- Squares are accumulated odd numbers; that is both `_SQ` and the ADI differential (one mechanism).

## Open (deferred, not in system)

- true-sine `_arch` variant (would trade exact-Pythagorean-at-cardinals for tighter angle-addition)
- CORDIC-style reconstruction of SIN/COS from one lobe + `polar_axis` sign bits
- `qsm_mul_quad_pure` lossy fold (fix if the product ring C must stay full width)
