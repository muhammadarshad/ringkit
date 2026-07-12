# ringkit — ring-topology ecosystem (Z₂₅₆ / QH4)

Core framework for MPRC: kernels, engines, frameworks, libraries — all obeying the ring
topology, none polluted by standard math. See `docs/project-governance/ECOSYSTEM.md` (north star) and
`docs/project-governance/CHARTER.md` (the rules: multiplier-free, ARC/ENERGY, solve-don't-descend, names-are-handles).

## Layout

```
ringkit/
  core/      native.py     — the ISA/identity: Z256, QH4, trig family, iota/rotor, ADI,
                              calculus, codec, mul/ipow/mf_floordiv  (multiplier-free)
             calculus.py   — d (=iota rotation) / integral (=accumulation), FTRC
  linalg/    solve.py      — exact mod-256 linear solve (Newton modinv, elimination)
             fit.py        — invert-then-solve exact nonlinear fit
  rnp/       our numpy replacement: __init__.py (rnp surface), tensor.py (RingTensor ndarray)  ★
  rmath.py   our math replacement: sin/cos/exp/log/isqrt + tau/pi/e (ring e = 3)
  collections/  ring-native data structures (placeholder)
  stats/     stats.py      — circular mean/median, geometric mean, ring_dist, ARCTAN2
  physics/   measure.py    — ENERGY rulers, measure_sin, overspill table
             qcm.py        — QCM topologies: 4 rings, quadrants, walk, hypervector, manifold
  ml/        autograd.py   — dual-ring reverse-mode autodiff (ARC value / ENERGY grad)
             optim.py      — sign-SGD + coarse-to-fine coordinate descent
             nn.py         — RingModule, Neuron
  kernels/   qcm_kernel.c, cache_manifold.c   — silicon: 8-bit SIMD, prime-stride
  tests/     test_tensor.py                    — cross-checked vs numpy mod 256 (oracle only)
  docs/      project-governance/ — SDLC docs (CHARTER, SRD, ECOSYSTEM_SRD, ECOSYSTEM, MANIFEST)
```

## Status

**All layers are now production-grade** (see `docs/project-governance/ECOSYSTEM_SRD.md` for the plan and acceptance).
Every module has input validation + real errors, docstrings, and a test suite that cross-checks
numpy-equivalent / oracle ops and verifies ring-internal identities. Run the full suite:

```
python3 -m ringkit.tests.run_all      # -> ECOSYSTEM: ALL GREEN (17/17 suites)
```

- `core/native` — exhaustive over 256 (primitives vs oracle, all trig identities, ADI, codec, errors)
- `core/constants` (frozen ring identity), `rmath` (exhaustive vs core), `stats`, `core/calculus`,
  `linalg` (solve/fit), `rnp` (tensor + surface), `physics` (measure/qcm/gauge), `ml`, facades
  (`nn`/`data`/`physics.Gauge`) with held-out generalization + failing controls
- `kernels` — C SIMD backend == Python reference bit-for-bit (~210x measured under
  Rosetta emulation; gauge stencil 0.107 ns/node)

**Charter-clean:** whole-package AST audit shows no `*`/`//`/`**` and no standard-math imports in any
**semantic** layer; the `kernels/` **silicon** layer intentionally uses hardware ops (charter D9),
validated bit-for-bit against the semantic reference.

## Use

```python
import ringkit.rnp as rnp
a = rnp.arange(12).reshape(3, 4)
b = rnp.eye(4)
c = a @ b                    # ring matmul (mod 256)
s = rnp.sum(a, axis=0)       # ring reduction
```

Run tests: `python3 -m ringkit.tests.test_tensor`
