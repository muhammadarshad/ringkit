# Sources Map — the QCM/MPRC corpus vs ringkit (surveyed 2026-07-12)

The full sources live in `/Users/marshad/Projects/` (qcm-source with paper.tex + the published
QCM.tex + the Julia engines; vlm-transformers = the 3 vision-transformer architectures /
MPRC ML stack; hpq-* = loss-free int quantization, the QSM test-bed; HpqKernelSwift = the
on-device inference sibling). ringkit's physics layer predates access to the paper body —
this map records what the full sources state, what ringkit has, and the derived job queue.

## The forms as the sources state them

- **Z256 is one ring, NOT a CRT system** of independent moduli. 4 vacuums {0,64,128,192} are
  phase singularities (128 = strict singularity where all 4 rings converge); 252 active bins
  = 4x63. Strides {X:3, Y:5, U:7, W:9}, anti-strides {171,205,183,57}.
- **The stride-7 quantum walk** (QCM form): single orbit x -> x+7, exactly 36 bins (9 per
  quadrant), the unique stride satisfying gcd(7,256)=1, 252 mod 7 = 0, AND vacuum avoidance.
  NOTE: ringkit's `seven_prime_walk` (cycles 2,3,5,7,11,13,17 covering all 252) is the SILIQ
  construct — a different, valid lineage; both forms should exist, labeled.
- **Measurement/Born rule**: probability cloud P(b|x) ~ exp(-d_circ(x,b)^2 / 2 sigma^2) — the
  Green's function of imaginary-time Schrodinger on the discrete circle. Collapse reading:
  frozen checkerboard sublattice = measured; updating sublattice = superposition.
- **Gauge**: FOUR action forms coexist in the sources — (1) true Wilson SU(N_c) with 3-link
  staples (the papers' ideal; papers disagree on the sign of dS in the acceptance), (2) the
  cos/sin U(1) rotor with trig LUT (qcm.jl — the actual criticality engine, 512^3, beta scan
  0.5..1.2), (3) a DEGENERATE uint8-sum proxy where the staple algebraically cancels (the
  CUDA/QVK throughput kernels — sources label it a hardware-flow test), (4) the per-site
  plaquette stencil as an OBSERVABLE only. ringkit's engine is the corrected circ_dist U(1)
  sampler (same universality family as (2), different microscopic potential) — NOT (3).
- **Weyl-hash PRNG exact spec**: h0 = (i*10048) ^ (j*14464) ^ (k*0x04F6CDD1) ^ seed, then two
  rounds of (h ^= h>>16; h *= 0x45D9F3B), h ^= h>>16; uniform = h * 2^-32. Constants
  10048 = 157*64, 14464 = 113*128. Per-sweep seed evolution: Knuth 0x9E3779B1 (ringkit's
  rk_mix32 uses the same idea — derived, never stored — with lowbias32 constants and
  0x9E3779B9; a paper-fidelity RNG mode is a small add).
- **Constants derive from N = 3**: D=16=(N+1)^2, d_model=128=16x8, H=113=7D+1 (prime),
  L=1808=DxH, Q=64, 36=(2N)^2, 15=(N+1)^2-1. beta is the only free (thermodynamic) knob.

## Acceptance targets (with the transfer warning)

- SU(3) engine (if ever ported): staple trace 2.92 -> ~-0.7 over 200 sweeps at 256^3
  beta=5.7; acceptance settling at 68-69%; ~511 MUPS on RTX 4060.
- U(1) transition: QUALITATIVE shape transfers (order parameter monotone in beta; C(R)
  decay length grows through the transition; mass-gap C(R) at beta=0.4: 0.235, 0.065,
  0.020, ... 3.5e-5 at R=10). ABSOLUTE beta values do NOT transfer to ringkit: the sources
  use a cos action + float exponential acceptance; ringkit uses circ_dist + integer LUT —
  a different chain that must locate its own critical beta.
- Perf: prime-stride manifold (157x64x256) +44.6%; CPU->CUDA 137x (0.027 ns/node at 256^3
  on the degenerate kernel — not comparable to ringkit's full-action 0.12 ns/node).

## Warnings that BIND ringkit

- The derived constants are physics, not hyperparameters (ringkit's frozen core/constants.py
  is the right shape). Vacuums are singularities — never normalized, never remapped.
  Z256 is not CRT. The stride-7 orbit's vacuum avoidance is deliberate.
- The Julia SU(3) staple is a 6-neighbor MATRIX SUM, not the 3-link Wilson path — the
  sources' own roadmap flags it (P5.1) while the published paper claims exactness; both
  recorded, not adjudicated.

## Derived job queue (D11: each job's form stated before build)

1. **Boltzmann acceptance form (THE live problem).** ringkit's linear LUT (255 - beta*dS)
   is not e^{-beta dS}: the chain's stationary measure is not Boltzmann — detailed balance
   broken vs the stated physics. Ring-native fix: EXPONENTIAL decay is GEOMETRIC decay —
   lut[dS+1] = (lut[dS] * f) >> s built by repeated fixed-point rn.mul (multiplier-free,
   integer-only, no float exp anywhere). State the form, verify monotonicity + ratio
   property exhaustively, re-locate ringkit's integer-beta critical point, update tests.
2. **Multi-R correlation + mass-gap labeling** in criticality_scan (C(R), R=1..10;
   confined/deconfined phase tagging) — the sources' main physics observable.
3. **Stride-7 orbit walk** (QCM form) alongside the SILIQ 7-prime walk in qcm.py, labeled.
4. **Weyl-hash PRNG paper-fidelity mode** (exact spec above) beside rk_mix32, gated.
5. **Born-rule measurement form** (probability cloud on the ring; integer-native sigma
   handling to be stated first).
6. **cos-rotor action option** for XY-universality matching against qcm.jl.
7. **SU(3)/non-abelian path** — biggest and needs a form decision first: float ComplexF32
   matrices are silicon-layer legal (D9) but a ring-native SU analog is the thesis-pure
   route; requires the user's direction.
8. **HPQ mission** (kernels/mprc/hpq): loss-free int quantization on QSM — connect with
   hpq-kernel-rust/HpqKernelSwift lineage; our GEMM qsm/shiftadd variants are the seed.
9. **Prime-stride manifold experiment** (157x64x256 geometry) in the bench harness.
