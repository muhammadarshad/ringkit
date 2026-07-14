# REPORT — Gemma4-12B on the ring (emulation engine): the ring GEMV is EXACT; fox argmax reproduced

**The stance (do not lose this again): the ring does NOT quantize the model.** ringkit replaces the
FPU with exact math kernels native to silicon (QSM tables, shifts, integer energy). The fidelity
bar is the fp16/fp32 semantics the weights actually carry — never "close to a quantized reference."
A lossy step inside a ring primitive is a philosophy violation, not a tuning knob.

**Result (verified by execution, D1):**

Gemma4-12B (`gemma4_unified`, the dense 12B) runs its **full 48-layer forward on the ring, float-free**
— every FPU op replaced by a ringkit QCM primitive. After this session's fix, the ring linear
(`gemma.proj`) is **BIT-EXACT to the exact integer dot** of the Q16 activations with the weights —
**zero activation loss**, proven as an integer identity in `tests/test_gemma2.py` §4 and
`tests/test_gemma4.py` §5b+§7, including under 60× activation outliers (with a failing truncating
control). End-to-end: on "The quick brown fox jumps over the lazy" (ids
`[2,818,3823,8864,37423,38167,1024,506,31770]`) the **ring produces `4799 ' dog'`, MATCH=True**
(9-token prefill, ≈105 s/token; the exact decomposition costs ~nothing — 4 QSM passes worst-case,
0.03 s per 512×3840 proj) — the same token as hpq (f16 arithmetic over the same weights) and MLX
`gemma4_text`, used as end-to-end anchors. The high-margin chat "capital of France?" also holds on
the ring: 50429 ' Paris'; the Gemma2 "→ Paris" regression holds (proj is shared).

## The bug this report previously mis-framed (and what it actually was)

An earlier revision framed the fox miss (`ring → 2268 ' little'`) as "Q16 non-linear precision to
close." That framing was **wrong**, and the error was found by refusing to compare the ring against
its own float64 mirror (circular) and comparing against independent references instead:

- **Symptom:** mirror-vs-MLX per-layer cosine cliffed L40→L47 (0.95 → 0.76) at fox pos 8, but showed
  NO cliff at pos 1 — position-dependent, late-layer, shared by ring and mirror.
- **Localization:** teacher-forced per-layer isolation (run layer *li* on MLX's own layer-*li* input,
  all 9 positions) showed every layer at ~0.99 except **L41 (global): 0.99 at pos 0 but 0.93 at
  pos ≥ 4**. Sublayer bisection of L41 pinned it to the **Q/K projections** (input norm cos 1.000,
  K after norm cos **0.55** at pos 2) — before RoPE, before attention.
- **Root cause:** `proj`'s **single-pass power-of-2 int8 activation quantization**. Late-layer
  activations carry channel outliers (L41: max|x|/rms ≈ **60** for content tokens vs 13 for BOS).
  One int8 grid pinned to the spike gives quantum ≈ spike/127 — larger than the entire bulk of the
  vector, which collapses onto ~2 levels; the projection **direction** is destroyed (cos 0.55 vs
  0.996 achievable with exact activations, same int8 weights). BOS at pos 0 has a different
  activation profile, which is why every pos-0/pos-1 check passed while multi-position text failed.
  hpq never saw this because it dequantizes to f16 **activations**; the weights and formulas were
  never the problem.
- **Fix — activation quantization ELIMINATED, not refined:** `emulation/gemma.py::proj` now
  decomposes the Q16 activation vector EXACTLY into power-of-2-scaled int8 digit passes: encode,
  form the integer residual `r = x − xs·2^(frac+a)` (pure shifts, no clipping by construction),
  re-encode the residual, **repeat until the residual is ZERO** (each pass peels ≥7 bits; Q16
  terminates in ≤4 passes). One energy-QSM dot per pass, summed at a common scale. The result is
  **bit-exact to the exact integer dot** — the 8-bit QSM table consumes exact digits; nothing is
  quantized. Float-free, multiplier-free, no kernel change.
- **After the fix:** the ring GEMV is exact (integer identity in tests, not a cosine); isolated
  per-layer cosine ≥ **0.99 at every layer and position** vs MLX (L41 worst-case 0.9263 → 0.9986;
  the residue is MLX's own 4-bit weights); L41 K-projection 0.547 → 0.996+ ; fox argmax =
  **4799 ' dog'** on the ring, MATCH=True.
- **Regression pinned:** `tests/test_gemma4.py` §5b — under synthetic 60× outlier activations,
  `proj == exact integer dot` BIT-FOR-BIT, AND the single-pass truncating control must fail
  (cos < 0.9). Tiny-activation negative-exponent digit passes pinned in `tests/test_gemma2.py` §4.

**Methodology rule this bought (charter-level):** a self-built oracle that reimplements the same
algorithm measures precision-consistency, not correctness. Localize against an INDEPENDENT reference
(MLX / hpq), and prefer *isolated* per-layer comparison (teacher-forcing) over end-to-end cosines,
which conflate compounding with cause.

## Multi-token correctness — and the reference hierarchy CORRECTION (2026-07-15)

Token-for-token greedy on fox (12 tokens) + the raw France prompt, ring vs hpq vs MLX, with the
**float64 mirror (EXACT activations, float64 arithmetic, the same int8 onix weights — the highest-
precision evaluation of the model the ring runs)** as tiebreaker at every divergence:

| decision | f64 mirror | ring (Q16) | hpq (f16) | MLX (4-bit) |
|---|---|---|---|---|
| fox tok 1 (' dog') | — | 4799 ✓ | 4799 | 4799 |
| fox tok 2 | **107** (16.89 vs 14.34) | **107** (16.63 vs 13.73) | 236761 ✗ | 107 |
| fox tok 3-4 | — | 100, 45518 | (diverged) | 100, 45518 |
| France (raw, contested) | **7001** (19.20/17.71/16.47) | **7001** (19.03/16.91/16.24) | 1144 ✗ | 496 |

- **The ring tracks the f64 mirror**: same argmax, same top-3/4 ordering, logits within ~0.6
  (Q16 non-linear noise), decision gaps ≥ 2.5 preserved. The ring ALSO matches independent MLX
  for 4 straight fox tokens (divergence afterwards is int8-vs-4bit weight difference on
  shrinking margins — MLX runs different weights).
- **hpq is NOT "exact/lossless" — that claim is retracted.** Its f16 forward disagrees with the
  float64 evaluation of its OWN int8 weights on both tested decisions (self-consistently: hpq
  prefill and decode agree with each other). hpq remains a useful fast cross-check and the
  architecture WHAT, but it is NOT ground truth.
- **The correctness bar, restated:** for the int8-onix model the ring runs, ground truth = the
  float64 mirror (exact activations); independent MLX validates architecture/assembly. The raw
  France prompt stays a bad oracle for anything (3-way implementation split, low margin) — but
  note the ring == mirror there too, exactly.
- The original checkpoint's own bf16/fp32 forward (the weights' native semantics) is still the
  ultimate bar; it needs the full-precision model fetched (~24 GB) — open decision.

## Gemma4 ≠ Gemma2 (the departures, each verified — this is the point of the exercise)

| aspect | Gemma2-2B | Gemma4-12B (dense) |
|---|---|---|
| layers / hidden / inter | 26 / 2304 / 9216 | **48 / 3840 / 15360** |
| vocab / Q heads | 256000 / 8 | **262144 / 16** |
| RMSNorm | `(1+γ)` offset | **γ straight, no offset** |
| attention | all-global, soft-cap 50 | **sliding/global alternation (global iff layer%6==5)**, **no soft-cap**, scale 1.0 |
| global heads | — | **head_dim 512, 1 KV head, θ=1e6, partial rotation (128/512)** |
| local heads | head_dim 256, θ=1e4 | head_dim 256, 8 KV, θ=1e4, full rotation |
| Q/K/V norm | none | **per-head Q-norm, K-norm (before RoPE), V-norm (no scale)** |
| global V | v_proj | **attention_k_eq_v: V = v_norm(raw K projection), no v_proj tensor** |
| FFN gate | GeGLU (sigmoid GELU) | **gelu_pytorch_tanh** |
| per-layer residual | — | **learned scalar (`layer_scalars.bin`), multiplies the whole hidden at layer end** |

The 12B is DENSE: `num_experts=None`, `hidden_size_per_layer_input=0`, `num_kv_shared_layers=0` — the
MoE / per-layer-input / KV-sharing paths in the reference are config-gated OFF, and are correctly
omitted here. Confirmed from the onix tensor inventory (global layers carry no `v_proj`) and the
`norms.bin` size (1,539,984 bytes = exactly header + 40 local·(4·3840 + 256·2 q/k) + 8 global·(… +
512·2 q/k) + final).

## Faithfulness — proofs against canonical MLX `gemma4_text` (independent reference)

1. **Per-layer hidden magnitude, BOS @ pos 0.** The ring's `mean|h|` after each of the 48 layers
   tracks MLX to **~1-2%** at every layer — including the sharp **layer-11 collapse to 0.15** and every
   sliding↔global transition.
2. **RoPE, numerically exact — at position.** Ring `rope_tables`+`apply_rope` vs MLX `nn.RoPE`
   (local) and `ProportionalRoPE` (global, partial rotation) agree at pos 3 to max |Δ| ≈ 7e-4
   (CORDIC precision), and the formula was re-verified against MLX at **pos 8/20/50** (max |Δ| ≤
   1.3e-5 in float, cos 1.00000, passthrough dims untouched). The proportional-RoPE freq divisor
   (full head_dim) and NeoX pair offset (head_dim/2, first 64 pairs only) match the reference exactly.
3. **Leaf bit-exactness / correctness.** Real `q_proj` over the int8 onix is **bit-exact** vs a
   the exact integer dot; `gelu_pytorch_tanh`, no-offset RMSNorm, and the f16/f32 field decoders
   match float oracles to < 2e-3.
4. **Isolated per-layer assembly (multi-position).** Teacher-forced layer-by-layer on the fox prompt:
   every layer ≥ 0.99 vs MLX at every position (the residual is int8-vs-4bit weight quantization).

## Memory & performance

10.9 GB onix + 2.0 GB embed streamed as reclaimable page cache (shared read-only mmap, one tensor
sliced at a time — never materialized). The LM head `lm_argmax_file` mmaps `embed.bin` read-only in
C (Python holds no embedding memory).

RingKit's speed model is C + QCM memory-BLOCK processing: Python never owns object memory or
processing — it only orchestrates (`kernels/mprc/kv/` C-owned slabs read in place;
`kernels/mprc/lattice/` specialised MPP — disjoint slabs, split/merge by construction, lock-free;
`kernels/mprc/qcm/cache_manifold.c`). The Gemma forward has been moved onto this model:

- **115 s/token → 12.3 s/token (9.3×)**, every step bit-for-bit gated (D9):
  - `qsm_gemv_exact` — the whole proj (exact digit decomposition, all QSM passes in one slab
    sweep, scaling, symmetric divide) in ONE C call, reading the onix tensor IN PLACE
    (MAP_PRIVATE mmap + zero-copy memoryviews; no Python weight copies). 115 → ~115 (exactness
    first), then with the blocks below → 52 s.
  - `gelu_mul_block` + `rmsnorm_block` — the FFN activation and every RMSNorm as C blocks
    (Python bigint exp saturation mirrored exactly by a divisor-equivalence clamp). → 52 s.
  - `qsm_gemv_exact_mt` — specialised-MPP split: decompose once, rows split into disjoint
    blocks (merge-free by construction, bit-identical, gated in the load selftest). → 12.3 s.
  - `attn_block` + `rope_block` over **C-owned KV slabs** (`host.KVSlab`, [kv_head, cap, hd]
    slabs grown by doubling, read in place; all query heads in one call, thread-split over
    disjoint ctx rows; softmax == infer.softmax bit-for-bit incl. the exp-clamp regime).
    → **9.5 s/token**, and attention now scales at C speed with sequence length. → 12.3 → 9.5 s.
  - `qsm_gemv_bridge_mt` — hardware-* BRIDGE variant of the GEMV (the kit's ring_gemm
    precedent: gated variants bridge/shiftadd/QSM-table). With hardware multiply the exact dot
    needs NO digit passes: one auto-vectorizable sweep of Σ(xbar−128)·x, bit-identical to the
    QSM digit path by construction and gated against it + the exact-dot reference at load.
    DEFAULT on CPU dev; `RINGKIT_GEMV=qsm` forces the multiplier-free silicon/reference form.
    → **3.1 s/token** (9.5 → 3.1; 115 → 3.1 = 37×).
  - **C-resident activations** (`gemma4.layer_forward_c`) — the hidden vector lives in C
    buffers across the whole 48-layer pass: embed decode, rmsnorm_rows (full-hidden AND
    per-head), GEMV, RoPE, KV-slab insert (pure memmove), attention, gelu_mul, residual adds
    and the layer scalar are ALL C blocks; the only Python crossing is the final hidden for the
    LM head. Plus an int32 fast path in the bridge (products ≤ 2^38, int64 accumulator exact,
    auto-vectorizable). → **1.9 s/token** (115 → 1.9 = 60×, ≈ hpq's CPU rate).
- **Composition gate pinned in `tests/test_gemma4.py`:** the C-resident forward must equal the
  Python-reference forward BIT-FOR-BIT across positions (multi-position, real weights).
- Anchors re-verified on this path: fox → 4799 ' dog' MATCH=True (**27 s total e2e** — was ~17
  min at session start); Gemma2 → ' Paris'; run_all ALL GREEN on the bridge default AND the qsm
  reference path.
- Cost anatomy that got us here (measured, not guessed): of the 3.1 s/token before this step,
  1.53 s was the C sweep (already at reference rate) and ~1.6 s was Python still owning the data
  path (list↔ctypes marshalling around every block + residual adds + layer scalar in Python).
  Moving residency into C recovered almost all of it.
  - **Vectorized bridge (int32 narrowing)** — activations narrowed once to int32 (exact:
    products ≤ 2^38, int64 accumulators; range-checked __int128 scalar fallback, BOTH paths
    gated) so the u8→i32 widening multiply-accumulate vectorizes (NEON smlal; Rosetta lacks
    AVX2). → **1.04 s/token Rosetta, 0.81 s/token native arm64** (115 → 0.81 = **142×**).
- **GPU GEMV (`RINGKIT_GEMV=metal`)** — `emu_gemv` (kernels/apple/metal/emulation.metal): the
  same exact integer dot on the unified GPU, with the 10.9 GB onix mmap wrapped as a NO-COPY
  shared MTLBuffer (`rk_metal_onix_map`) so the GPU reads the file's own page-cache pages —
  nothing is materialized or copied. One threadgroup (256 lanes) per output row, long
  accumulators, tree reduction; range/region guards route to the CPU bridge so bit-identity is
  never compromised. Gated bit-for-bit vs the CPU bridge on real tensors, and the dual-path
  forward + f64-mirror greedy pins PASS with the GPU in the loop.
  → **0.36 s/token** (115 → 0.36 = **320×**; 4.4× faster than hpq).
- **The kit's sentence is now literally true: the ring reproduces the model's output FASTER
  than the reference (0.36 vs 1.6 s/token) and MORE faithfully (f64-mirror-verified where hpq's
  f16 diverges).**
- GPU dispatches are BATCHED (q/k/v and gate/up each run as one command buffer; per-tensor s/z
  rows uploaded once and cached GPU-side; several weight files map concurrently — Gemma2's onix
  runs the same GPU path, ' Paris' verified). Batching holds bit-identity; the remaining limiter
  is the GEMV kernel itself (~55 GB/s effective of ~200 available) — next lever is uchar4/
  simdgroup reads inside emu_gemv.
- The bf16-checkpoint semantics bar is DEFERRED (download vetoed — bandwidth); all verification
  anchors to locally-present weights: the 12B onix (f64 mirror = ground truth), the 2B onix,
  MLX 4-bit as the independent architecture check.

## Charter compliance

- **D1** verified by execution (`tests/test_gemma4.py`: bit-exact leaf + AST + outlier regression
  with failing control + gated BOS per-layer proof).
- **No float, no FPU** on the compute path — AST-clean (`gemma4.py`/`gemma4_weights.py`: no `float()`,
  no float literals, no numpy/torch/math). Every value product is `rn.mul`/`rn.qsm`; `*` only in
  integer index/offset math.
- **Honesty bar:** the argmax claim is anchored to an independent two-reference consensus (hpq + MLX
  agreeing on ' dog' out of 262k tokens), with the failing single-pass control preserved in the test.

## Reproduce

```bash
export PYTHONPATH=$PWD
python3 -m ringkit.tests.test_gemma4                 # portable + opportunistic (in run_all)
RINGKIT_GEMMA4_GEN=1 python3 -m ringkit.tests.test_gemma4   # gated BOS per-layer proof (~2 min)
```
