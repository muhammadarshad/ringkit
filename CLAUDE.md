# CLAUDE.md — ringkit

Ring-topology (Z₂₅₆ / QH4) compute ecosystem. Hard ring physics/math lives in a verified
substrate; three engineer-facing facades hide it so ordinary engineers use it like numpy/torch
without touching mod-256, energy/phase, vacuums, or zero-divisor collapse.

## Run the verification loop (do this after every change)

```bash
python3 -m ringkit.tests.run_all        # must print "ECOSYSTEM: ALL GREEN"
python3 -m ringkit.tests.test_<module>  # a single suite
```

Kernels (C) build on first import via ctypes; if a `.so` is stale, delete it and re-import.

## Non-negotiable disciplines (docs/CHARTER.md — D1–D9). Break these and the work is wrong.

- **D1 Verify by execution.** Never conclude without running it. Every non-trivial claim is
  backed by a test. Prefer exhaustive checks over the 256 ring where feasible.
- **Multiplier-free semantic layers.** No `*`, `//`, `**`, `/` and **no standard-math imports**
  (numpy/math/scipy) anywhere under `core/ stats/ linalg/ array/ physics/ ml/ nn.py data.py`.
  Use `rn.mul` (shift-add), `rn.ipow`, `rn.mf_floordiv`, `rn.ring_pow`, etc. AST-audit new files:
  `python3 -c "import ast,sys;[print(n.lineno,type(n.op).__name__) for n in ast.walk(ast.parse(open(sys.argv[1]).read())) if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Mult,ast.FloorDiv,ast.Pow,ast.Div))]" <file>`
- **Two-layer rule (D9).** Only `kernels/` (C/SIMD) may use hardware `*`/`<<`; it must reproduce
  the Python semantic reference **bit-for-bit**. Tests may use numpy/math **as a labeled oracle only**.
- **Fold late.** A ring quantity is `(energy, phase)` = `(quotient, remainder mod 256)`. Stay in
  unfolded ENERGY to keep operations reversible; fold to phase only when you need the ring position.
  Gradients (ENERGY) must NOT fold mod 256; values (ARC) do. Odd = unit (reversible); even =
  zero-divisor (collapses). Ring-native e = **3** (`rn.RING_E`, `ring_exp`/`ring_log`).
- **Solve, don't descend.** Linear layers = exact `linalg.solve` (mod 256). Nonlinear invertible =
  `linalg.fit` (invert-then-solve). Descent (`ml/autograd`) is the fallback, and it's flaky on the ring.
- **Exact vs approximate is labeled.** `SIN/COS` use the `_arch` semicircle (exact only at the 4
  cardinals; ~28–31% shape gap elsewhere). Never present an approximation as exact. Rotation by a
  quarter turn (`rotate`, iota) IS exact; general angle-addition is NOT.
- **Names are handles.** Wilson/Metropolis/Euclidean etc. are borrowed labels, not standard-math
  imports. Don't force ring behavior onto standard math or vice-versa.

## The honesty bar for ANY learning/ML claim

Self-retrieval / "fits training" proves nothing (memorizing is trivial). Every ML capability must
show **held-out generalization** on data never seen, PLUS a **control that fails**: a random-label
run must collapse to chance, and/or a position-only/content-only baseline must fail. See
`tests/test_ml.py`, `tests/test_attention.py`, `tests/test_nn_facade.py`.

## Layout

```
ringkit/
  core/native.py     substrate ISA: Z256 consts, mul/ipow/mf_floordiv, qsm, isqrt, ring_neg,
                     SIN/COS/TAN family, ARC*, iota/IOTA rotor, ring_exp/ring_log/ring_pow (e=3),
                     ring_cis (Euler), rotate/cis_rotate, ADI, codec  — all multiplier-free
  core/calculus.py   d_rot / integral / differential / FTRC
  linalg/            solve.py (exact mod-256 solve, modinv, is_invertible), fit.py (invert-then-solve)
  stats/stats.py     ring_dist, ARCTAN2, circular mean/median, geometric_mean
  array/             tensor.py (RingTensor: nD ndarray, bytearray-backed), numpy.py (rk.rnp)
  physics/           measure.py, qcm.py, gauge.py (SU(256) plaquette + Metropolis + criticality),
                     sim.py (Gauge facade class)
  ml/                autograd.py, tensor_autograd.py (TVar), optim.py, nn.py (low-level), attention.py
  kernels/           ring_ops.c, gauge.c + backend.py (ctypes, zero-copy, Python fallback)  [D9 silicon]
  tests/             one test_<module>.py each; run_all.py aggregates (15 suites)
  docs/              CHARTER.md, ECOSYSTEM_SRD.md, ECOSYSTEM.md, MANIFEST.md

ringkit/nn.py        FACADE (top-level): Linear, Dense, Attention+RoPE, TransformerBlock,
                     Transformer (induction + in-context recall), Sequential. Ring hidden; .raw hatch.
ringkit/data.py      FACADE: encode/encode_range, one_hot, split, batches
```

Engineer entrypoint: `import ringkit as rk` → `rk.nn`, `rk.data`, `rk.physics.Gauge`, `rk.rnp`.
Every facade object hides ring internals and exposes `.raw` for power users.

## Status

All 15 suites green. Substrate (core/stats/linalg/array/physics/ml/kernels) is production-grade
and AST-clean. Facades (`rk.nn`, `rk.data`, `rk.physics`) built and verified with held-out + controls.
Next candidates: stacked multi-block trained model, numpy-surface polish, top-level quickstart.
Note: `README.md` predates the facade layer / e=3 / attention — refresh it when convenient.
