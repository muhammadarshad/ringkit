# Course correction — you are a CODING agent, not a BENCHMARK agent

Read this before continuing. Recent history drifted: the last several commits are a
GPU/GEMM/Metal speed campaign ("beats torch-mps ~85x", "215 GMAC/s", "beats BLAS",
"~250x cycles"). **That is not the goal.** Performance is the *last* priority here and is
*never* the objective. Re-anchor now.

## The goal (unchanged, from CLAUDE.md + CHARTER D1–D11)

Build a **correct, honest, ring-native (Z₂₅₆ / QH4)** ecosystem whose hard physics/math is
**abstracted behind clean engineer-facing facades** (`rk.nn`, `rk.data`, `rk.physics`) so an
ordinary engineer uses it like numpy/torch **without touching mod-256, energy/phase, vacuums,
or zero-divisor collapse**. Math-and-physics-first (D11). Ring identity preserved; no
standard-math pollution in the semantic layers.

## Priority order — when these conflict, the higher one wins

1. **Correctness + ring identity** — the forms are right; substrate exact; state the
   physical/mathematical form FIRST (D11); multiplier-free semantic layers (D-constraints).
2. **Honesty** — D1 verify-by-execution; every claim backed by a test; the held-out + failing-control
   bar for any capability claim.
3. **Engineer-facing abstraction** — the facades hide the ring; ergonomic, documented, `.raw` hatch.
4. **Performance** — subordinate, informational, gated. Never the objective, never the headline.

## Benchmarks go back in their box

- Benchmarks live **only** in `bench/`. Standard engines (numpy/torch/BLAS) may be imported
  **only there**, **only as a labeled oracle** (D9 two-layer rule).
- **No speed claim without a preceding bit-for-bit correctness gate.** A ring kernel must reproduce
  the semantic Python reference **exactly** (verified in `tests/`) before any timing is reported.
  A fast wrong kernel is worthless.
- **"Beats X by N×" is the performance twin of self-retrieval** — hollow unless it is
  (a) correctness-gated, (b) apples-to-apples (same hardware, same problem size, same precision),
  and (c) reported *with* its caveats (what is excluded, warmup, variance). **Never lead with the number.**
- **Never** trade away ring identity, exactness, or multiplier-freedom to win a benchmark, and
  **never** present an approximation as exact to hit a target (exact-vs-approximate discipline).

## Do now

1. **Stop the GPU/GEMM/Metal performance campaign** unless a specific engineer-facing goal
   requires it. Kernels already in the tree stay (if bit-for-bit gated); write no new
   "beats-torch" work.
2. **Re-anchor on the ecosystem roadmap.** Ask: what does an engineer still need from
   `rk.nn` / `rk.data` / `rk.physics`? Pick the next capability, **state its physical/mathematical
   form first (D11)**, build it, and prove it with the **honesty bar** (held-out generalization +
   a control that fails), *not* a benchmark.
3. **Sober the existing perf reporting.** Ensure every kept kernel has a bit-for-bit test vs the
   semantic reference; rewrite loud "N× faster" language in `reports/benchmarks/BENCHMARKS.md` and
   commit messages into caveated, oracle-gated statements.
4. After every change, run `cd .. && python3 -m ringkit.tests.run_all` — must print
   **ECOSYSTEM: ALL GREEN**.

## Keep the on-goal work you already did

The stacked solve-trained model (held-out 1.0 with both controls at chance), the D11 charter
refinement, and the QCM sources map are exactly right — capability proven honestly, ring-first.
Continue in *that* spirit: **capability + honesty + abstraction, not speed theater.**
