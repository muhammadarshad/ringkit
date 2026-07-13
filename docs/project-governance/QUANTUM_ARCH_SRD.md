# QUANTUM_ARCH_SRD — the three MPRC architectures, native on ringkit

Specs · Requirements · Design · Tasks · Verification · Bugfix · Report.
Goal: **RDT, Mamba-2/SSD, and QuantumRoPE run — and eventually train — on ringkit alone** (no torch,
no numpy, no safetensors). They are MPRC quantum constructs (not transformer variants); this SRD
governs building their ring-native forms.

## 0. Method — WHAT vs WHY/HOW (D11 + D8)

- **The frontier models give the WHAT.** Gemma/Gemini, Z.ai GLM, OpenMythos, Mamba-2 converge on a
  known component set: pre-norm (RMSNorm), a gated MLP (GeGLU/SwiGLU), grouped-query attention with
  RoPE, an SSD/SSM scan, recurrent/looped depth, tied embeddings, cross-entropy. We read these as the
  *inventory of parts a modern stack has* — the WHAT — and nothing more. (Established architecture
  knowledge; not a claim about any one model's internals.)
- **MPRC quantum physics gives the WHY and HOW.** Each part has a ring-native form with a physical
  reason: normalization touches only ENERGY (the ARC never leaves 256, C9); gating is routing across
  the four MPRC axes; attention is thermal (Boltzmann) weighting; position is phase on four co-prime
  rings, ADI-locked; the SSM state is accumulation, its decay is geometric, its mixing is diffusion on
  the torus; depth is the rotor applied repeatedly (i⁴=1). **The ring construct is primary (D8); the
  mainstream part is the handle.** Where a mainstream trick is a float band-aid (log-domain cumsum,
  Euclidean norm), the ring form does not need it.

## 1. Specification — the component map (WHAT → WHY/HOW → ringkit status)

| Component | WHAT (frontier handle) | WHY/HOW (MPRC ring form) | ringkit |
|---|---|---|---|
| RoPE / position | rotary position | **4 co-prime rings (X3/Y5/U7/W9) by anti-stride walk; QuantumRoPE4D = 4 axes ADI-locked (`δ_k=δ₁+k²−1`); vacuums = 4th roots of unity** | HAVE (`rotate`, `native.mprc_axis_arcs`, `qcm` anti-strides) |
| Attention | GQA + softmax | **Boltzmann-soft weighting** (geometric decay in the score gap), circular value blend; hard argmax = cold limit | HAVE fwd (`ml.kvcache`); learned QKV + backward: gap |
| SSM / SSD scan | Mamba-2 duality | **state = accumulation (`integral`), decay = `boltzmann_lut`, mixing = torus heat-kernel diffusion**; log-domain float fix NOT needed (integer-exact) | forms HAVE; assembly: task |
| Gate | GeGLU / SwiGLU | **5-arm ADI router over center + R/M/P/C axes** (routing = choosing a physical direction) | ADI axes HAVE; router (learned) + sigmoid: task |
| Recurrent depth | looped / universal depth | **recurrent rotor**: one shared quantum rotor applied `depth` times, gated (depth = repeated rotation) | rotor HAVE (`rotate`/`cis_rotate`); gate: task |
| Normalization | RMSNorm / LayerNorm | **ENERGY-only normalization** — the ARC/phase is already bounded (never leaves 256), so norm rescales magnitude/energy, never the arc (C9) | MISSING → design |
| Activation | GELU / SiLU | ring activation forms; SIN present; sigmoid/softplus are LUTs; GELU needs a stated ring form | PARTIAL |
| Embedding | tied token embeddings | token → ring phase; learned lookup (or fixed ring code) | MISSING (learned) |
| Loss | cross-entropy / InfoNCE | **ENERGY-domain** loss (gradients don't fold, D4) | MISSING |
| Diffusion / probability cloud | — (MPRC-specific) | **Born-rule cloud + Metropolis** (`measure.born_weights`, `gauge.boltzmann_lut`); QCM's `np.exp(-βΔE)` is the Euclidean anti-pattern | HAVE |
| Serialization | safetensors / state_dict | ring-tensor `save`/`load` (bytearray-backed → trivial) | MISSING (easy) |
| Training | AdamW + backward | **solve where exact (linear/invertible); descent where needed** — the keystone | keystone |

## 2. Requirements

**Non-functional (binding — CHARTER):** NFR1 multiplier-free semantic layers (C4); NFR2 no
standard-math imports (C2); NFR3 no floats in ring semantics (C1); NFR4 exact-at-structure, label
approximations (D3/D4); NFR5 **no Euclidean form imported to define the ring identity (C9)**; NFR6
**never quantize the ARC** (C9); NFR7 two-layer D9 (kernels bit-for-bit vs semantic reference); NFR8
physics/form first (D11/D2); NFR9 honesty bar on every ML claim.

**Functional (per component):** each ring form must (a) be exact where the physics is exact, (b)
reproduce its intended behavior, (c) carry a test. For learned components: held-out generalization +
a control that fails. For kernels: bit-for-bit vs the Python reference. For quantum forms: a physics
check (e.g. QuantumRoPE4D == `native.mprc_axis_arcs` bit-for-bit).

## 3. Design — the ring-native forms (WHY/HOW, form before code)

**QuantumRoPE (ring).** Position on four co-prime rings via anti-stride (modular-inverse) walks;
`QuantumRoPE4D` = the four axes locked by ADI (`native.mprc_axis_arcs`, odd-increment `δ_k=δ₁+k²−1`).
Rotation is exact at the vacuums (fourth roots of unity). Applied by `rotate` (additive, exact for
any position). No sinusoid, no float. — mostly built; needs the 4-ring wiring + a physics-equality test.

**Mamba-2/SSD (ring).** The scan is: state `h` = **accumulation** of `δ·(k⊙v)` (`calculus.integral`,
exact), transition decay = **`boltzmann_lut`** (geometric, = `exp(δA)`), spatial mixing =
**torus heat-kernel diffusion** (mass-conserving 4-neighbour evolve; global head = t→∞ Σ, local =
geometric multi-scale). QuantumRoPE modulates q,k first. Gate = **5-arm ADI router**. No log-domain
hack (integer accumulation is exact). — assemble from existing forms; the router + `softplus`(δ) are
new ring forms.

**RDT/Rotor (ring).** One shared quantum rotor (a RoPE attention block) applied recursively for
`depth` steps; each step blends the input back (injection gate) and a GRU-style gate (biased toward
identity to prevent collapse) decides how much new state to keep. The rotor is `rotate`/`cis_rotate`
(i⁴=1). — needs the gate (sigmoid) + depth embedding + the shared-layer recursion.

**Shell forms (new, form-first):**
- **RMSNorm/LayerNorm (ring):** separate ARC (bounded, untouched) from ENERGY (magnitude); normalize
  the energy channel only. State the exact ring statistic (accumulation-based) before coding.
- **Activations:** `sigmoid`/`softplus` as monotone LUTs (geometric-decay family); **GELU** needs a
  stated ring form (open — do not fake it). `softmax` forward = Boltzmann (built); backward: open.
- **Embedding:** learned lookup table (needs descent) or a fixed ring code (token→phase, data-free).
- **Cross-entropy/InfoNCE:** ENERGY-domain loss; gradients unfolded (D4).

**Keystone — training.** Linear/invertible parts by exact `solve`/invert-then-solve (no descent).
The genuinely-learned parts (gates, routers, A_log, embeddings) need a **ring descent that
generalizes** — proven on a small nonlinear net with held-out + a random-label control *before* it
is trusted on an architecture. This is the gating research item.

## 4. Tasks (dependency-ordered backlog)

**Phase F — foundations (unblock everything, mostly mechanical):**
- F1 ring-tensor **serialization** (`save`/`load`, safetensors-shaped) — bytearray-backed.
- F2 **rnp op breadth**: flip, roll, split, where, clip, argsort, cumsum(=integral), exp(=geom decay),
  sqrt(=isqrt), pad, random(=`rk_mix32`).
- F3 **conv/patchify** (ViT `unfold`/`pad`; wave encoder ports as fixed no-param code).

**Phase S — shell forms (form-first, tested):**
- S1 ring **RMSNorm/LayerNorm** (ENERGY-only).  S2 `sigmoid`/`softplus` LUTs.  S3 **GELU** ring form
  (open — design first).  S4 softmax **backward**.  S5 **cross-entropy/InfoNCE** (ENERGY).
- S6 **Embedding** (fixed ring code first; learned later behind the keystone).

**Phase A — architectures (assemble on the physics):**
- A1 **QuantumRoPE / 4D** wired to `mprc_axis_arcs` + physics-equality test.
- A2 **RDT/Rotor** (recurrent rotor + gate + depth-embed).
- A3 **Mamba-2/SSD** (integral scan + boltzmann decay + torus diffusion + 5-arm ADI gate).
- A4 **ViT** (wave encoder + quadrant projector + RMSNorm + head).
- A5 **Diffusion/probability cloud** (Born + Metropolis, already ring-native — wrap + test).

**Phase K — keystone (the research gate):**
- K1 ring descent that generalizes on a small nonlinear net (held-out + random-label control) — or
  the honest finding that it doesn't, routing training through solve/invert-then-solve.
- K2 train one architecture end-to-end on the ring; matched-pair vs the torch reference.

**Rule of order:** F → S → A(inference, weights converted) → K(training). Do not skip to A-training
before K.

## 5. Verification (acceptance per task)

- Kernels/silicon: **bit-for-bit** vs the Python semantic reference (D9), gated at load.
- Quantum forms: a **physics-equality** check (QuantumRoPE4D == `mprc_axis_arcs`; torus diffusion
  conserves mass; rotor⁴ = identity; vacuums are fixed points).
- Learned components: **held-out generalization + a control that fails** (random-label → chance;
  position/content baseline fails). No self-retrieval, no "fits training".
- Whole-suite: `python3 -m ringkit.tests.run_all` → ALL GREEN after every change; AST audit clean.
- No number ships without a run (D1). No "beats X" without a correctness gate + apples-to-apples (C6).

## 6. Bugfix discipline

**No bug is claimed without an analytic/mathematical proof first** (the QCM standard). File `BUG-NNN`
with: the invariant violated, the minimal reproduction, the proof it is wrong (not just a failing
number), the fix, and the regression test that now guards it. A failing test that is *flaky* is a
bug in the test (fix the assertion), not a defect — distinguish the two.

## 7. Report discipline

Each completed task closes with a `REPORT-NNN`: what was built, the ring form + physics reason, the
verification result (with real numbers from a run), honest caveats, and what stays open. **Regime
hygiene:** matched-pair comparisons only; never cross-cite regimes; never present an approximation as
exact or a Python number as silicon. Reports are the source of truth over any status table.

---
Traceability: every change traces Component (§1) → Requirement (§2) → Design form (§3) → Task (§4) →
Verification (§5) → (Bug §6) → Report (§7). Paperwork before code (D11): the ring form is stated
before it is built.
