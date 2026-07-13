# Stance — emulation engine vs operating system (owner, 2026-07-13)

Two relationships ringkit has to models, and one rule that binds both.

## Traditional models we cannot control → ringkit is an EMULATION ENGINE

For models whose architecture we do not own (Gemma, and any external checkpoint), ringkit is an
**emulation engine** — the same *category* of thing as TurboQuant / PolarQuant (ingest the trained
weights, run the compute on our substrate). We do not redesign the model; we emulate its forward
pass on the ring. `ringkit.checkpoint` (.pth → ring), `ringkit.onix` (Gemma .onix → ring), and
`ringkit.infer`/`ract` (the ring compute) are the emulation path.

## Our MPRC architectures → ringkit is the OPERATING SYSTEM

For RDT, Mamba-2/SSD, QuantumRoPE and the rest of MPRC, ringkit is **native** — the OS. These are
built *from* the ring physics (4-ring ADI phase lattice, toroidal diffusion, recurrent rotor,
Boltzmann attention), not emulated onto it.

## The rule that binds both: NO FLOAT — the FPU is replaced by ringkit's QCM kernels

Whether emulating or running native, **there is no float and no FPU**. Every place an FPU op would
appear is replaced by a ringkit **QCM-enabled** primitive:

- multiply / MAC → `rn.qsm` (quarter-square) or `rn.mul` (shift-add) — ringkit's own, exact
- exp / softmax / decay → integer Taylor `ract.exp_fixed` / geometric-decay `boltzmann_lut`
- sqrt / norm → `rn.isqrt` (`ract.rmsnorm_fixed` / `layernorm_fixed`)
- scale → power-of-two **shift**; divide → `mf_floordiv`
- float weights → ring by **integer mantissa-shift** (`checkpoint`/`onix`); ONIX weights are already
  integer (xbar/s/z), so no decode at all

**Do NOT copy the reference kernel.** hpq-kernel (and TurboQuant/PolarQuant) tell us *what* the ops
are — they are references for the WHAT. The arithmetic must be **ringkit QCM code**, not a
transliteration of hpq's `dot_qph`, and never a float fallback. `ringkit.onix.project_row` uses
`rn.qsm`, ringkit's signature quarter-square, precisely for this reason — verified bit-exact against
a direct integer reference on the real 2B Gemma.
