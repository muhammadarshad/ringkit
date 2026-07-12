# Ring-Native Charter — principles, disciplines, constraints

The rules this work is held to. Check every new form against this before it enters the system.
Status column reflects the current system (`core/native.py`, `stats/stats.py`, and layers above) as audited by execution.

## Prime Directive

**Find the ring-native form first, then code it. Never pollute the ring with standard math.**
Standard math may be used only as an external *reference* to check against — never mixed into ring values.

---

## Constraints (hard rules — a violation is a defect)

| # | Constraint | Status |
|---|------------|--------|
| C1 | **No floats in system code.** Integer only, mod 256 (or the product/codec rings). | HELD — core is int-only |
| C2 | **No standard-math imports in the system** (no `math`, no `numpy`, no `import` of standard trig). | HELD — `core/native` imports nothing standard; layers import only ring modules |
| C3 | **No Euler, no `math.pi`, no `sin/cos` calls.** Rotation = arc shift (ring addition), trig = tables/`_arch`. | HELD |
| C4 | **Multiplier-free** everywhere (`+ - << >> \| &` and table reads). `*`/`//`/`**` are realized as `mul`/`mf_floordiv`/`ipow` (shift-add). | HELD — NO exceptions |
| C5 | **Single source of truth per form.** No two live definitions of the same function. | HELD (the old dual-SIN was resolved) |
| C6 | **Standard math lives only in scaffolding/verification**, in separate files, never imported by the system. | HELD (`mprc_trig.py`, charts, CSVs are reference-only) |
| C7 | **Meaning is preserved even when values or representation change.** Refactors must reproduce prior values exactly. | HELD (all refactors diffed 256/256) |
| C8 | **Constants carry meaning; none are arbitrary.** Every constant has a documented ring derivation. | HELD (SCALE=21=1+4+16; 7=quantum walk; 16=QH4 const; 1024=4·τ; 675=86400/128) |

## Exceptions register

**EMPTY.** No system form uses `*`, `//`, `**`, or `/` — audited by AST across all files.
The multiply/divide/power family is realized ring-natively:
- `mul(a,b)` — shift-and-add (repeated doubling), any size, signed.
- `mf_floordiv(n,d)` — shift-subtract long division.
- `ipow(base,n)` / `mf_mod(n,d)` — built from `mul`/`mf_floordiv`.
- `qsm` — quarter-square product; `scale21` — shift-sum for ×21.

Rule: if a new form seems to "need" `*`/`//`/`**`, use `mul`/`mf_floordiv`/`ipow` (or QSM). There is no exception path — "no `*`" means no `*`.

---

## Disciplines (the working method)

| # | Discipline |
|---|------------|
| D1 | **Verify by execution before concluding.** No claim ships without a run. "Verify geometry first." |
| D2 | **Forms before code.** Establish the ring-native form (its meaning, its exactness) before implementing. |
| D3 | **Exact at structure; label the rest.** Vacuums, cardinals, iota powers, nodes must be exact. Any in-between approximation is stated with its magnitude, never hidden. |
| D4 | **Distinguish exact vs definitive-not-exact.** A form is either bit-exact or an approximation with a measured bound — say which. |
| D5 | **Separate system from scaffolding.** Every artifact is either shipped system or reference/verification. Keep the line (see SYSTEM_MANIFEST). |
| D6 | **Honesty over agreement.** Flag internal inconsistencies, don't overclaim, own mistakes. Reference math is a comparison, never a source. |
| D7 | **Earn the place, or leave the main.** A form belongs in the 256 core only if it carries the core identity (returns ring positions, participates in the structure, is not duplicated by a layer). We do not force placement — if it doesn't earn the core, it goes to the layer where its meaning sits (measurement/ENERGY, stats, etc.). Test each form: does it return a ring value? does it carry an identity? is it unique here? |
| D8 | **Names are handles, not constraints.** Standard-math/physics terms in the sources (Wilson, Metropolis, Euclidean, sine, matmul, gauge, path integral) are *labels* attached because they're the nearest familiar concept — AI/human naming, not definitions. The real object is the MPRC/Z₂₅₆/QH4 construct that emerges from the geometry. Take the ring construct as primary (e.g. `circ_dist` IS the action). Overlap with a standard identity is incidental; divergence is **not a bug** — do not "correct" the ring toward the textbook. Standard math is a labeled external comparison only, never the arbiter. Verify ring **internal** consistency, not conformance to standard math. |
| D9 | **Two layers: semantic vs silicon.** The multiplier-free rule (C4, no `*`/`//`/`**`) governs the ring **semantic** layers (core, stats, calculus, linalg, rnp, physics, ml) — they are the correct reference. The `kernels/` **silicon** layer is the designated exception: it uses hardware ops (`*`, SIMD) on purpose, to trade ALU cycles for speed, and must reproduce the semantic layer **bit-for-bit** (cross-checked in tests) before it is trusted. Hardware ops in `kernels/` are not a violation; they are the point of that layer. The AST audit excludes `kernels/`. |
| D10 | **Naming obeys 5W and stays original.** Every name we mint — package, module, public form — must answer the five Ws: **Who** it serves, **What** it is, **When** it applies, **Where** it sits (which layer/ring context), **Why** it exists. Our namespaces are identity and stay ORIGINAL to the ring ecosystem: we do not name them after the standard libraries they replace (`rnp` not `numpy`, `rmath` not `math`). This composes with D8: borrowed standard terms remain handles *on forms* (Wilson, sine, matmul); they are never the names *of our namespaces*. |
| D11 | **Math and physics first — and the physics is QUANTUM, ring-realized.** Before any build step, ask what the mathematical/physical form IS (this sharpens D2 into a working order: the question precedes the code, always). The subject of ringkit's physics layers is quantum physics realized through MPRC (Z256/QH4: spin/polarity node states, vacuums as structural nodes, the 7-prime quantum walk, gauge fields, criticality) — NOT a fast re-implementation of standard-model/continuum methods. Standard physics and math enter only as labeled handles (D8) and external comparisons (C6). The forms (walk, measurement/collapse, gauge, composites) are ONE interlocking physics, not a menu: the JOB at hand — the problem actually being faced — selects which form must be stated first. When a proposed feature has no stated ring-native quantum/math form behind it, it is not ready to build. |

---

## Meaning rules (ring semantics)

- **256 is the CORE IDENTITY — immutable.** It is derived, not chosen: `[(a+b)²·(c+d)²]²` at unit inputs = `16² = 2⁸ = 256` (16 = QH4 constant; squaring ladder 2→4→16→256). The core carries the algebraic identities and is never modified for resolution or any other reason.
- **512 / 1024 / 1536 / 1808 / … are MEASUREMENT TOOLS, not the core.** They are rulers (ENERGY-side resolution) applied *on* the 256 core: 2×, 4×, 6×, ~7.06×. Higher resolution lives in a separate measurement layer; it must never be pushed into the 256 core trig/identity. (So: no "high-res SIN" inside `ring_native` core — resolution is selected by the measurement layer.)
- **ARC vs ENERGY are different units.** ARC = angle/position (circular), the 256 core. ENERGY = magnitude (multiplicative), where measurement resolution lives. Circular ops on ARC, geometric ops on ENERGY.
- **The ring SOLVES; it rarely descends.** Linear layer = linear system mod 256 -> exact closed-form
  solve when the determinant is odd (ring_solve; ADI-family recovery). Nonlinear-of-linear with an
  INVERTIBLE activation = invert the activation (ARCSIN preimage sets) then solve the linear system
  exactly (ring_fit, invert-then-solve). Gradient descent is the FALLBACK for genuinely
  non-invertible compositions only. Descent stalls on solvable problems because the wrapping loss is
  non-convex — so recognize solvable structure and solve it, don't descend it.
- **Values are ARC (fold mod 256); gradients and losses are ENERGY (do NOT fold).** Discovered in the Phase 5 spike: folding a gradient/loss into the 256 core destroys the descent signal for large errors (measured 36% vs 100% convergence). So autodiff carries ARC values forward (qsm, wraps) and ENERGY differentials backward (mul, signed, unbounded). The primal is ARC; the differential is ENERGY.
- **Rotation is arc shift** (ring addition); `iota` = +64; the quantum walk = +7 (ergodic, reversible: 7⁻¹=183).
- **Vacuums {0,64,128,192} are structural.** φ passes through zero there and does not stop; they are the nodes where identities are exact.
- **Multiplication is accumulation.** `×` via QSM = (accumulation² − differential²)/4; squares = accumulated odd numbers. `∫` and `d` are the same (accumulation, differential) pair (FTRC).
- **The no-`*`/`//` constraint is architectural, not aesthetic: multipliers ARE the silicon bottleneck.** Multiplier/divider ALUs dominate area, power, and latency in hardware; the ring exists to compute without them. QSM and the ring dot-product (accumulate QSM terms — tables + adds + shifts only) are not workarounds for a missing operator, they are the architecture's answer: the bypass that makes multiplier-free silicon (MPRC/HPQ) buildable. Consequences: (1) performance work on linear maps strengthens the ring forms — QSM/LUT/accumulation kernels — and never reintroduces semantic `*`; (2) hardware `*` stays quarantined in the D9 silicon layer as a speed bridge on today's commodity chips, validated bit-for-bit, and is NOT the long-term thesis; (3) benchmarks are weighted by ring-native workloads (walks, sweeps, accumulation, solve) — numpy-idiom elementwise ties are facade service-bar checks, not fronts; (4) a linear-map kernel campaign ships in two variants: a D9 hardware-`*` bridge for commodity silicon AND a multiplier-free QSM/table form that measures the thesis itself.
- **Known approximation:** `_arch` is a semicircle projection, exact = analytic sine only at the 4 vacuums (≤~28% relative between). Anything built on ring SIN/COS (e.g. `circular_mean`) inherits this; the L1 median and geometric mean do not.

---

## Standard-math boundary (the anti-pollution rule)

- Standard sine/`math.*`/floats may appear **only** in files that are not imported by the system, and only to **compare** against ring output (e.g. the `analytic_21sin` column, the divergence charts).
- The mapping to standard objects (vectors, matrices, tensors, Lie algebra, limaçon/cardioid, directivity patterns) is **interpretation/documentation**, not a dependency. Ring code never calls it.
- If a ring form "needs" a standard-math function to work, that is a signal the ring-native form has not been found yet — stop and find it (D2).

---

## Adherence checklist (run on any change)

1. Does the new form import anything standard? (must be no — C2)
2. Any float literal or float op in system code? (no — C1)
3. Any `*`/`//`/`**`/`/` not on the exceptions register? (no — C4)
4. Does it reproduce prior values exactly where it should? (diff — C7, D1)
5. Is it exact at the structural points; is any approximation labeled with a bound? (D3/D4)
6. Is every new constant derived, not arbitrary? (C8)
7. System or scaffolding — is it filed correctly? (D5)
8. Was every claim run before stated? (D1)
