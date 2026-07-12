# Benchmarks — ringkit vs numpy vs torch (apples to apples)

Harness: `python3 -m ringkit.bench.apples_to_apples` (charter C6 scaffolding — the only place
numpy/torch may be imported, as labeled external comparisons). Every engine is verified
**bit-for-bit against the ringkit Python reference before it is timed** — a wrong baseline is
not a baseline; a gated-out engine is reported, not hidden. Identical semantics, identical
inputs, best-of-3, GPU warmed, MPS synchronized. External engines run device-resident tensors
with hoisted masks (their best case); ringkit-metal timings INCLUDE host-buffer copies (our
real API cost).

## Native environment (the fair fight) — 2026-07-12

python 3.11.15 **arm64 native** (`~/.venvs/ringkit-bench`), numpy 2.4.6, torch 2.13.0 with
**MPS on the SAME Apple M1 Pro unified GPU** ringkit's Metal backend uses. Gates: numpy OK,
torch-cpu OK, torch-mps OK (mix32 int32-wrap semantics hold on MPS).

### Elementwise ring mul (uint8 wrap) — GMUPS, higher is better

| size | ringkit C | numpy | torch cpu | torch mps | ringkit metal |
|------|-----------|-------|-----------|-----------|---------------|
| 2^20 | 23.2      | 31.1  | 15.7      | 4.1       | 1.7           |
| 2^24 | 26.1      | 24.2  | 29.0      | 23.8      | 1.8           |

**Verdict: a tie, correctly.** Elementwise uint8 is pure memory bandwidth; every competent
engine converges to the same wall (~24-30 GMUPS native). GPUs don't help (even torch's
resident-tensor MPS only matches CPU at 16M), which is why ringkit's metal elementwise is
opt-in and its copies-included number is worst — as documented in backend.METAL_MIN.

### Wilson plaquette stencil, 128³ — ns/node, lower is better

| ringkit C | numpy | torch cpu | torch mps | ringkit metal |
|-----------|-------|-----------|-----------|---------------|
| **0.077** | 0.285 | 0.307     | 1.185     | 0.463         |

**Verdict: ringkit C wins by 3.7x** over the best external (numpy). Single fused cache-blocked
pass beats temporary-array pipelines; the GPU round trip can't pay for a bandwidth-trivial op.

### Metropolis sweep (arrays supplied), 128³ — ns/node/sweep, lower is better

| ringkit C | numpy | torch cpu | torch mps | ringkit metal |
|-----------|-------|-----------|-----------|---------------|
| 10.91     | 18.45 | 7.48      | 10.16     | **0.78**      |

**Verdicts:** honest one first — **multithreaded torch-cpu beats our single-threaded C**
(7.5 vs 10.9): threading the C sweep is a real future lever. But **ringkit metal is 9.6x
faster than torch-cpu and 13x faster than torch-mps**, copies included.

### Thermalize, derived RNG (rk_mix32), 8 sweeps — ns/node/sweep, lower is better

| lattice | ringkit C | numpy | torch cpu | torch mps | ringkit metal GPU |
|---------|-----------|-------|-----------|-----------|-------------------|
| 128³    | 11.69     | 20.79 | 9.10      | 10.49     | **0.13**          |
| 160³    | 11.94     | 21.70 | 8.13      | 10.89     | **0.12**          |

**The headline verdict: on the SAME unified GPU, running the SAME bit-for-bit-gated
algorithm, ringkit's Metal path is ~85x faster than torch-mps** (and ~65x faster than
torch-cpu): ~8.5 G node-updates/s. The gap is architectural, not incidental: torch launches
~30 kernels per sweep and streams every intermediate (neighbor distances, dS, masks) through
memory, while ringkit runs ONE fused kernel per parity — randoms derived in registers, grid
resident, 2 x sweeps dispatches deep-queued in a single command buffer, and only grid + a
256-byte LUT ever cross the bus.

## Rosetta environment (appendix) — same date

python 3.14.6 x86_64 under Rosetta 2, numpy 2.4.6, no torch wheels. Both CPU columns equally
emulated; ringkit metal drives the GPU natively regardless.

| workload                            | ringkit C | numpy | ringkit metal |
|-------------------------------------|-----------|-------|---------------|
| elementwise 2^24 (GMUPS)            | 6.55      | 6.68  | 0.90          |
| plaquette 128³ (ns/node)            | 0.106     | 0.506 | 0.802         |
| sweep arrays 128³ (ns/node/sweep)   | 15.64     | 19.53 | 1.69          |
| thermalize rng 160³ (ns/node/sweep) | 16.77     | 22.45 | 0.15          |

Native lifted our C kernels ~1.4-4x (as predicted in the SRD); the GPU numbers barely moved
because the GPU never was emulated.

## Takeaways

1. Where physics is trivial (elementwise), ringkit ties the best engines — the substrate
   wastes nothing, and nobody beats bandwidth.
2. Where structure exists (stencil), the fused C kernel beats numpy/torch by ~4x.
3. Where the ring compute is dense (Metropolis + derived RNG), the unified-memory design
   wins by ~65-85x against engines using the same hardware — including torch on the same GPU.
4. Fair-play debt recorded: torch-cpu's threading beats our single-threaded C sweep;
   a threaded C path is a known lever if CPU-only hosts ever matter.

Reproduce: `~/.venvs/ringkit-bench/bin/python -m ringkit.bench.apples_to_apples` (native)
or any interpreter with numpy (torch rows appear when importable). Gates print first and
the run aborts on any mismatch.
