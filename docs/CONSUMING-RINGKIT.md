# Consuming ringkit — briefing for an external coding agent

You are consuming a **ring-topology (Z₂₅₆) compute ecosystem**. It replaces the FPU with exact
integer math: every value on a compute path is a signed fixed-point integer (`Q<frac>`, default
`FRAC=16`, so `1.0 == 1<<16`), every multiply is shift-add or a gated C kernel, and every claim
in the repo is backed by an executable gate. You do not need to understand the ring physics to
use it — the facades hide it — but you MUST respect the boundaries below, because the kit's
guarantees (bit-exactness, float-freedom, verified kernels) are only as good as its call sites.

## 1. Entry points (use these, don't reach inside)

```python
import ringkit as rk
import ringkit.rnp as rnp        # numpy REPLACEMENT (RingTensor, matmul via gated C GEMM)
import ringkit.rmath as rmath    # stdlib-math replacement (sin/cos/exp/log/isqrt; e = 3 on the ring)

rk.nn        # Layer/Linear/Dense/Sequential, Transformer blocks, KVCache (decode-time attention)
rk.data      # encode/encode_range, one_hot, split, batches
rk.physics   # Gauge facade (thermalize/action/order/profile/phase) — GPU-accelerated where it wins
ringkit.quanta      # the MPRC architectures as ring forwards: rotor_forward, rotor_lattice_forward,
                    #   soliton_forward, gluon_forward (+ frontend, lattice/rope encoder layers)
ringkit.emulation   # external-checkpoint engine (Gemma2/Gemma4 .onix, .pth): checkpoint/infer/ract
```

Every facade object exposes `.raw` if you truly need the ring internals. Prefer not to.

## 2. The contract at the boundary (what you feed in, what you get out)

- **Fixed point in, fixed point out.** Quantize floats ONCE at the boundary:
  `q = int(round(v * (1 << 16)))`; dequantize once at the end: `v = q / (1 << 16)`. Never mix
  floats into the middle of a ring computation — the kit's paths are integer-exact and
  AST-audited; a float you smuggle in breaks the discipline and usually the tests.
- **Weights for the quanta/emulation forwards** come from
  `ringkit.emulation.checkpoint.load_fixed(path, frac=16)` → `{name: (int_list, shape)}`.
  The forwards take an accessor `W(name) -> flat Q16 int list`. RoPE caches are read from the
  checkpoint (already ring-valued) and sliced to the grid, never recomputed.
- **Precision expectations.** Ring vs float reference: cosine ≥ 0.999 and matching argmax is
  the shipped bar for model forwards; exact kernels (GEMV, rmsnorm, sigmoid, attention blocks)
  are BIT-IDENTICAL to their Python semantic references, enforced by load-time selftests that
  refuse to serve on any disagreement.

## 3. Speed model — the rule that bites hardest

**Python never owns memory or processing at real sizes.** The kit is fast because hot loops are
single C block calls over contiguous slabs (specialised MPP row-splits, merge-free), not because
Python got optimized. As a consumer:

- `emulation.infer.linear` auto-routes any tensor ≥ 2^12 MACs through the gated C GEMV
  (balanced base-256 weight digit-planes, cached per tensor identity — keep your weight lists
  ALIVE and REUSED; a fresh list per call defeats the cache). ~385× over the Python loop.
- Batch activations: `ract.sigmoid_list(xs)`, `ract.exp_list_nonpos(xs)` (softmax domain),
  `host.gelu_mul` — one C call per vector/batch, never a Python loop of scalar calls.
- Environment: `RINGKIT_GEMV=bridge` (default, CPU dev) | `qsm` (multiplier-free silicon form) |
  `metal` (unified GPU over mmapped onix weights — only worth it for GB-scale weight files;
  cache-resident weights are FASTER on the CPU bridge). Run natively for 4×+ kernel speed:
  `arch -arm64 /usr/bin/python3 ...` (Rosetta lacks the vector paths).
- C kernels build on first import into `kernels/build/` (arch-keyed `.so`). If one looks stale,
  delete `kernels/build/` and re-import.

If your forward crawls or "hangs", you are on a Python path at real sizes — route it, don't
optimize the Python. And know the failure signature we hit: a process pinned at 99% CPU with
small flat RSS is an infinite loop (was: an int64 overflow feeding a C isqrt), not OOM;
`faulthandler.dump_traceback_later` localizes it in one step.

## 4. Verification — how you prove anything here

```bash
cd <parent-of-ringkit>
python3 -m ringkit.tests.run_all           # MUST print "ECOSYSTEM: ALL GREEN"
python3 -m ringkit.tests.test_<module>     # one suite
```

- **D1: verify by execution.** No conclusion without a run. Any non-trivial change → run_all,
  on BOTH pythons if you touched kernels (Rosetta x86_64 dev + native arm64).
- **The oracle chain for model work** (learned the hard way, twice): gate against the REAL
  deployed system first (torch app / reference impl on real inputs), match a numpy oracle to it
  at fp precision, THEN match the ring to the oracle. A hand-written oracle alone is circular —
  ours was self-consistently wrong about a rope convention until the deployed app exposed it.
  Synthetic random inputs also skip real-input branches (vacuum/padding tokens, saturated
  pixels): include a deterministic real-regime input. Recipe: `bench/webapp_e2e/`.
- **ML claims need held-out generalization PLUS a control that fails** (random labels at
  chance, position-only baseline failing). "Fits training" proves nothing.
- numpy/torch/math are allowed ONLY in `tests/` (as labeled oracles) and `bench/`. Never in
  package code you contribute.

## 5. Semantics you must not violate

- **Never quantize position.** The ARC/phase position (RoPE positions, Δticks, quadrants) is
  carried at full precision, always. Compression may crush redundant magnitude/energy, never
  the arc. The KV element is ADI `(x+y, x−y)` — exact and reversible — not a polar form.
- **Exact vs approximate is labeled.** Ring SIN/COS is the `_arch` semicircle: exact at the 4
  cardinals only. Quarter-turn rotation is exact; general angle addition is not. tanh-GELU
  tracks erf-GELU to ~3e-4; sigmoid-GELU is ~30× coarser — pick per the reference you match.
- **KVCache bar:** cached decode == uncached recompute BIT-FOR-BIT. If you build on
  `rk.nn.KVCache`, keep that property testable.
- **Solve, don't descend** where possible: linear layers have an exact mod-256 solve
  (`linalg.solve/fit`); gradient descent on the ring is the flaky fallback.

## 6. Known model-level facts you can rely on (all gated)

- **Gemma2-2B and Gemma4-12B run float-free on the ring**, bit-exact to the exact integer dot;
  0.36 s/token on the unified GPU. Ground truth for the 12B onix is the f64 mirror with exact
  activations; MLX is the independent architecture check; hpq's f16 is NOT ground truth
  (retracted — it contradicts the f64 eval of its own weights).
- **quanta (MPRC family):** `rotor_forward` (RDT, cos 1.000000), `soliton_forward` (Mamba2
  gate_lat2d), `rotor_lattice_forward` (the deployed webapp's RotorHeads path) — all gated at
  the deployed vlm-transformers webapp's full scale (N=1808, real checkpoints, real photo):
  cos 1.000000 + argmax vs the actual torch app on every path, fused prediction included.
  `gluon_forward` is written but has no checkpoint yet — do not claim it works.
- The 4D QuantumRoPE rotate-half convention is FOUR hd/4 chunks (not two halves); vacuum
  tokens (mean phase < 1e-3) take the vacuum embedding, not the depth embedding; phase 1.0
  clamps to arc 255. These are encoded in `quanta` — if you re-implement a frontend, you will
  reintroduce those three bugs.

## 7. Naming and contribution rules (if you add code)

- D10: minted names are ORIGINAL — never name a namespace after the library it replaces
  (`rnp` not "numpy"); never shadow a stdlib module name (a dir named `collections` once broke
  every python run inside the repo).
- Multiplier-free semantic layers: no `*`, `//`, `**`, `/`, no numpy/math imports under
  `core/ stats/ linalg/ rnp/ rmath.py rcollections/ physics/ ml/ nn/ data.py` (AST-audited).
  Index arithmetic in `quanta`/`emulation` may use `*`; compute may not.
- Only `kernels/` may use hardware `*`/`<<` — and any kernel you add MUST ship a load-time
  selftest proving bit-identity to its Python reference, INCLUDING the magnitude regimes real
  models produce (an int64 accumulator that was "gated" only on small values overflowed and
  hung in production paths).
- Losing variants: measure, record the number in the commit, then DELETE the losing code path
  (the measurement is the receipt, not the dead code).

Read next, in order: `CLAUDE.md` (the disciplines, D1–D11), `docs/REPORT-GEMMA4.md` (what
"exact emulation" means here), `tests/test_quanta.py` (the gate pattern to copy).
