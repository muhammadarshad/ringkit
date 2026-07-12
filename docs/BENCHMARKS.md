# Benchmarks — ringkit vs external engines (apples to apples)

Harness: `python3 -m ringkit.bench.apples_to_apples` (charter C6 scaffolding — the only place
numpy/torch may be imported, as labeled external comparisons). Every baseline is verified
**bit-for-bit against the ringkit Python reference before it is timed** — a wrong baseline is
not a baseline. Identical semantics, identical inputs, best-of-3, GPU warmed.

Measured 2026-07-12. Environment: python 3.14.6 **x86_64 under Rosetta 2** on an Apple M1 Pro;
numpy 2.4.6 (same interpreter — equally Rosetta-handicapped, so CPU rows are fair); torch not
installed (no macOS x86_64/cp314 wheels exist) — the GPU entry is ringkit's Metal backend,
which drives the M1 Pro natively regardless of the host interpreter's architecture.

## Elementwise ring mul (uint8, mod-256 wrap) — GMUPS, higher is better

| size | ringkit C | numpy | ringkit Metal |
|------|-----------|-------|---------------|
| 2^20 | 6.58 | 6.40 | 0.82 |
| 2^24 | 6.55 | 6.68 | 0.90 |

Verdict: **parity with numpy** — both are memory-bandwidth-bound, as they should be.
The GPU loses here by design (3 buffer copies around a trivial op) and is opt-in
(`backend.METAL_MIN = None` documents this).

## Wilson plaquette stencil, 128³ — ns/node, lower is better

| ringkit C | numpy | ringkit Metal |
|-----------|-------|---------------|
| 0.106 | 0.506 | 0.802 |

Verdict: **ringkit C is 4.8x faster than numpy** — the cache-blocked single-pass C stencil
beats numpy's temporary-array pipeline. GPU stays out (bandwidth-trivial op).

## Metropolis sweep (arrays supplied), 128³ — ns/node/sweep, lower is better

| ringkit C | numpy | ringkit Metal |
|-----------|-------|---------------|
| 15.64 | 19.53 | 1.69 |

Verdict: ringkit C edges numpy 1.25x; **the Metal path is 11.6x faster than numpy**.

## Thermalize with derived RNG (rk_mix32), 8 sweeps — ns/node/sweep, lower is better

| lattice | ringkit C | numpy (vectorized mix32) | ringkit Metal GPU |
|---------|-----------|--------------------------|-------------------|
| 128³ | 16.70 | 19.87 | **0.21** |
| 160³ | 16.77 | 22.45 | **0.15** |

Verdict: **the unified-memory GPU path is 95x (128³) to 150x (160³) faster than numpy**
running the identical algorithm — ~6.7 G node-updates/s at 160³. This is the headline:
same physics, same spec (numpy's mix32/sweep passed the bit-for-bit gate), and the gap is
the difference between temporaries-through-DRAM and derived randoms on a resident grid.

## Honesty notes

- numpy's vectorized sweep has its parity masks hoisted out of the timed loop (favors numpy).
- CPU rows (ringkit C, numpy) are equally Rosetta-emulated; native arm64 would lift both
  (~4x for our C kernels, measured earlier) without changing the GPU's dominance.
- torch could not be included in this environment; on Apple silicon its MPS backend would be
  the interesting comparison — rerun when a native-arm64 dev interpreter lands.
- All numbers reproduce via the harness; gates print first and abort the run on any mismatch.
