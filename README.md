# ringkit — quantum physics on the ring (Z₂₅₆ / QH4), by MPRC

ringkit is not a faster numpy. It is the compute ecosystem for **MPRC quantum physics**: the
ring Z₂₅₆/QH4 is the physical object — spin/polarity node states, vacuums {0,64,128,192} as
exact structural nodes, the 7-prime quantum walk (ergodic, reversible), gauge fields and their
phase transition — and every layer, from Metal shaders to the `rk.nn` facade, obeys that ring.
Standard math never enters the system: no floats in ring values, **no hardware multiply in any
semantic layer** (multipliers are the silicon bottleneck; QSM and shift-add are the ring's own
answer), and names like "Wilson" or "sine" are borrowed handles, not definitions.

The forms — walk, measurement/collapse, gauge, composites — are one interlocking physics.
The rules are law: [docs/project-governance/CHARTER.md](docs/project-governance/CHARTER.md)
(D1–D11); the north star is
[docs/project-governance/ECOSYSTEM.md](docs/project-governance/ECOSYSTEM.md).

## Quickstart (every snippet below is executed before it is documented — D1)

```python
import ringkit as rk

# The physics: a gauge field ordering out of heat (cold coupling), on your GPU when present.
g = rk.physics.Gauge(size=(32, 32, 32), beta=60, seed=0)
g.action()                    # 64.1  (disordered start)
g.thermalize(sweeps=40)       # Metropolis sweeps; randoms DERIVED on-device (rk_mix32)
g.action(), g.order()         # 18.9, 0.861  (ordered)  — hot beta=0 stays at 63.9, 0.501

# The ring's own math: e is 3 (the unit-subgroup generator), tau is 256.
import ringkit.rmath as rmath
rmath.e, rmath.tau            # 3, 256
rmath.exp(5)                  # 243 = 3^5 mod 256
rmath.sin(64)                 # 21  (quarter turn; exact at the 4 cardinals)

# Learning is EXACT SOLVE, not descent: recover the true ring map from 4 examples.
layer = rk.nn.Linear(in_features=3, out_features=2)
layer.fit(X, Y)               # solves the ring system; layer.raw["W_ring"] == W_true
layer.predict([10, 20, 30])   # [110, 244] — generalizes because the map is exact

# In-context: induction on tokens never seen in any training.
rk.nn.Transformer().induction([5, 9, 200, 7, 5])   # (9, 0): the follower of the last 5

# Deep recall: stacked solve-trained blocks (held-out 1.0; depth + random controls at chance).
model = rk.nn.Stacked(blocks=2, dim=2)             # see tests/test_stacked.py

# Plumbing and tensors (the numpy REPLACEMENT — original names per D10).
rk.data.encode([300, -5, 42])                      # [44, 251, 42]  (wraps onto the ring)
import ringkit.rnp as rnp
a = rnp.arange(12).reshape(3, 4)
a @ rnp.eye(4)                                     # ring matmul, silicon-served
```

## Performance (measured, gated, reproducible — docs/BENCHMARKS.md)

Every number below survived a bit-for-bit gate against the multiplier-free Python reference
before timing; externals (numpy, torch incl. torch-mps on the same GPU) ran the identical
gated algorithms at their best (native arm64, resident tensors).

- Gauge thermalize (derived RNG, GPU-resident): **0.12 ns/node/sweep** — ~65-85x faster than
  torch on the same unified GPU; C(mt) fallback beats torch-cpu ~4x.
- Ring GEMM: hardware bridge **215 GMAC/s** (2.2x torch-mps, 16x Accelerate BLAS);
  **multiplier-free shift-add 55 GMAC/s beats every external CPU engine**; on GPU the
  zero-multiply QSM-LUT beats torch-mps's hardware matmul — the bottleneck thesis, measured.
- Elementwise ops tie everyone at the bandwidth wall (they carry no ring structure — that
  parity is the physically honest result).

## Layout

```text
ringkit/
  core/      constants.py (FROZEN ring identity) · native.py (the ISA: trig, iota, ADI,
             mul/ipow/mf_floordiv, ring_exp e=3) · calculus.py  — all multiplier-free
  linalg/    solve.py (exact mod-256 solve) · fit.py (invert-then-solve)
  rnp/       our numpy: __init__.py (surface) + tensor.py (RingTensor)  — silicon-served
  rmath.py   our math: sin/cos/exp/log/isqrt + tau/pi/e (ring e = 3)
  rcollections/  ring-native data structures (reserved)
  stats/     circular mean/median, ring_dist, ARCTAN2
  physics/   qcm.py (spins, walk, hypervector, manifold) · gauge.py · sim.py (Gauge facade)
  ml/, nn/   autograd (ARC/ENERGY) · solve-trained facade (Linear/Dense/Attention/
             Transformer/Stacked)
  kernels/   D9 silicon: backend/ (registry + ring_ops + ring GEMM x3 variants),
             mprc/lattice/ (gauge.c threaded + observables), apple/metal/ (shaders + sessions),
             build/ (arch-keyed .so, gitignored)
  bench/     apples-to-apples vs numpy/torch (C6 scaffolding; gates first, then timing)
  tests/     20 suites; run_all prints "ECOSYSTEM: ALL GREEN"
```

## Verify (after every change)

```bash
cd .. && python3 -m ringkit.tests.run_all      # must print: ECOSYSTEM: ALL GREEN (20 suites)
```

Kernels build on first import (arch-keyed into `kernels/build/`); a native arm64 interpreter
runs the same suite green and ~4x faster on CPU paths: `arch -arm64 /usr/bin/python3 -m ...`.
