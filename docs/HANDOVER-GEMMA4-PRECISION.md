# HANDOVER — Gemma4-12B on the ring: RESOLVED (2026-07-14, follow-up session)

**Status: the bug below was LOCALIZED and FIXED.** See `docs/REPORT-GEMMA4.md` for the full corrected
story. Summary of the resolution:

- **Root cause:** NOT a formula bug and NOT generic Q16 drift — the old **single-pass truncating
  int8 activation grid** in `emulation/gemma.py::proj` (a quantizer, which the ring must never
  contain: the ring replaces the FPU, it does not quantize the model). Late-layer activations carry ~60× channel
  outliers (L41 pre-attn, content tokens; BOS ≈ 13×, which is why every pos-0/pos-1 check passed);
  a spike-pinned int8 grid crushes the bulk of the vector onto ~2 levels and destroys the Q/K
  projection DIRECTION (L41 K: cos 0.55 vs MLX; 0.996 achievable with exact activations on the SAME
  int8 weights). hpq avoids it by using f16 activations.
- **Localization method (reusable):** RoPE-at-pos-8/20/50 vs MLX first (CLEAN — ruled out), then
  **teacher-forced per-layer isolation** (`scratchpad/mirror_isolate.py`: run mirror layer li on
  MLX's own layer-li input at all positions — compounding removed) → only L41 was bad (0.93 @ pos≥4)
  → sublayer stage dump (`mlx_l41_stages.py` + `mirror_l41_stages.py`) → collapse begins AT the
  projections (input-norm cos 1.000, k_norm 0.55), before RoPE/attention → activation-quant test
  (`actquant_test.py`) confirmed cos_full 0.996 vs cos_actq 0.547.
- **Fix (final form — the ring does NOT quantize the model):** activation quantization was
  ELIMINATED from `proj`, not refined. The Q16 activation vector is decomposed EXACTLY into
  power-of-2-scaled int8 digit passes (integer residual by shifts, re-encoded until the residual is
  ZERO — ≤4 passes for Q16), one energy-QSM dot per pass. `proj` is now **bit-exact to the exact
  integer dot** (integer identity, tested). Float-free, multiplier-free, no kernel change.
- **Verified:** `proj` == exact integer dot BIT-FOR-BIT (integer identity, incl. under 60×
  outliers); isolated per-layer ≥0.99 all layers/positions vs MLX (L41 0.9263→0.9986); mirror fox
  argmax = **4799 ' dog'** = hpq = MLX (gap 3.8 logits); **ring e2e fox = 4799 MATCH=True**
  (~105 s/token — the digit passes cost ~nothing); Gemma2 "→ Paris" regression holds (proj is
  shared); regression tests: `tests/test_gemma4.py` §5b (bit-exact + failing truncating control),
  tiny-activation negative-exponent digit pass in `tests/test_gemma2.py` §4; run_all ALL GREEN.

Everything below is the ORIGINAL handover, kept as the historical record of the debugging log.

---

# ORIGINAL HANDOVER — Gemma4-12B on the ring: an OPEN systematic bug (not just precision)

**Date:** 2026-07-14 · **Read first:** `CLAUDE.md`, `docs/REPORT-GEMMA4.md` (NOTE: that report is now
partly WRONG — see §"Status of the report" below), then this file.

## TL;DR

Gemma4-12B (`gemma4_unified`, dense 12B) runs its full 48-layer forward on the ring float-free. The
**architecture is verified faithful** (RoPE, all departures, leaf ops — §2). BUT the greedy argmax is
wrong on the consensus prompt, and — this is the key correction from late in the session — **it is NOT
just Q16 precision. There is a SYSTEMATIC formula bug in my emulation algorithm that bites at
multi-position in the LATE layers (L40-L47).** It has not been localized yet. Do not ship "faithful."

The trap that hid it for most of the session: I built a float64 "oracle" (`scratchpad/g4_f64.py`) to
isolate the ring's error — but that mirror **reimplements gemma4.py's own algorithm**, so
`ring-vs-mirror ≈ 0.99` proves precision-consistency, NOT correctness. A shared *formula* error is
invisible to it. Only comparing against an **independent** reference (MLX `gemma4_text`) exposed the
bug. **Rule for next session: localize/verify against MLX or hpq, never against the mirror.**

## The evidence the bug is real (independent references agree; my algorithm is the outlier)

Prompt "The quick brown fox jumps over the lazy" (a semantically forced continuation → "dog"),
ids `[2,818,3823,8864,37423,38167,1024,506,31770]`:

| implementation | weights | arithmetic | first token |
|---|---|---|---|
| **hpq** (`~/Projects/hpq-kernel-rust`, `examples/generate`) | int8 onix | f16 | **4799 ' dog'** |
| **MLX** `gemma4_text` | 4-bit | f16 | **4799 ' dog'** |
| ring (`emulation/gemma4.py`) | int8 onix | Q16 | 2268 ' little' |
| float64 mirror (`scratchpad/g4_f64.py`) | int8 onix | float64 | 8321 |

Two **independent** implementations on **different** quantizations land on `dog` (out of 262k tokens)
→ `dog` is robust. My algorithm (ring **and** its float64 mirror — same code) is the outlier, and the
two runs of my code don't even agree with each other. Earlier I misread "3 precisions → 3 tokens" as
"inherent instability"; that was wrong — 8321 and 2268 are both *my* algorithm.

**Localization so far — mirror(float64,int8) vs MLX(4-bit), per-layer, fox last position (pos 8):**
faithful and weight-quant-sized through L0-L28 (cos ~0.98-0.99), then a **CLIFF L40→L47**:
`L40 0.950, L41 0.903, L45 0.841, L46 0.755, L47 0.804`. On the 2-token `[2,818]` (pos 1) the same
comparison had NO cliff (L47 = 0.977). So the bug is **position-dependent (grows with #keys/pos) and
concentrated in the late layers.** Dumps saved: `scratchpad/mirror_fox_layers.json`,
`scratchpad/mlx_fox_layers.json`.

## Prime suspects (position-dependent / multi-position-only)

The mirror's RoPE uses EXACT `theta^(-2i/hd)` (numpy pow) yet still cliffs vs MLX — so it is likely a
**formula/assembly** difference, not the ring's `r^i` accumulation. Ranked:

1. **RoPE formula at HIGH position.** I only numerically verified ring/mirror RoPE vs MLX at **pos 3**
   (max|Δ|≈7e-4). The last action (INTERRUPTED) was re-testing at pos 8/20/50, local + global. Finish
   this first. If Δ grows with position → RoPE pairing/freq is subtly wrong in a way pos-3 didn't
   expose. (Local uses `nn.RoPE(256,base=1e4)`; global uses `ProportionalRoPE(dims=512,rotated_dims=128,
   base=1e6)` — see `.venv-mlx31/.../mlx_lm/models/rope_utils.py`.)
2. **Attention over multiple keys** (error scales with #keys → worse at pos 8): KV-cache RoPE offset
   (how cached keys are rotated vs MLX prefill), GQA grouping (`i//group`), causal/sliding-window
   masking (window 1024; 9 tokens shouldn't differ, VERIFY), or the `attention_k_eq_v` V handling
   (global V = `v_norm(raw k_proj)`) in a multi-key context.
3. **Late-layer-specific** compounding — but nothing structurally special about L40-47 is known;
   more likely (1)/(2) accumulating.

## How to localize (cheap, both oracles are fast — NO 50-min ring runs needed)

- **Independent per-layer:** `mlx_fox_layers.py` (MLX) and `g4_f64.py::forward_perlayer(ids)` (mirror)
  already dump per-layer at the last position. Re-run for any prompt; diff cosine per layer.
- **Sublayer bisection at a cliff layer** (e.g. L45): adapt `~/Projects/mprc-scratchpad/diag_layer0.py`
  (it dumps input_layernorm/attn_out/post_attn/ff_in/mlp/down/post_ff for layer 0) to layer 45 at the
  fox last position, and dump the SAME intermediates from the mirror. Whichever sublayer (attention_out
  vs ffn_out) diverges is the bug's home. This is the definitive next step if RoPE-at-pos is clean.
- Once localized: fix the formula in `emulation/gemma4.py`, re-run mirror-vs-MLX on fox (cliff must
  vanish, cos → ~0.97 flat like L0-28), THEN confirm the ring reproduces hpq/MLX `4799 ' dog'`.

## What IS verified faithful (do not re-litigate — §2 checks all pass)

- **RoPE θ:** local `ln(1e4)=603609` Q16, global `ln(1e6)=905414` Q16 — confirmed by the model's own
  `config.json` (`rope_parameters`), the Gemma 4 tech report (arXiv 2607.02770 — 1M global/10k local,
  p-RoPE 128/512), AND numerical match to MLX **at pos 3**. (NOT 1e5 — a user asked; it's 1e4/1e6.)
- Partial global RoPE: 128 of 512 dims rotate, pairs `(i, i+256)`, rest pass through bit-exactly.
- `attention_k_eq_v`: global layers carry NO `v_proj` in the onix (verified from tensor inventory);
  global V = `v_norm(raw k_proj)`. Per-head Q/K norm BEFORE RoPE, V-norm (no scale). No attn soft-cap,
  scale 1.0. gelu_pytorch_tanh. Per-layer residual scalar (`layer_scalars.bin`, f32, multiplies whole
  hidden at layer end — matches MLX `layer_scalar` and hpq `transformer.rs:1663`). RMSNorm NO `(1+γ)`.
- Dense: `num_experts=None`, `hidden_size_per_layer_input=0` (no PLE), `num_kv_shared_layers=0` — the
  MoE/per-layer-input/KV-share paths are config-gated off and correctly omitted.
- `norms.bin` size = 1,539,984 bytes = exact (header + 40 local·(4·3840+256·2) + 8 global·(…+512·2) +
  final). Leaf ops accurate: proj **bit-exact**, exp 8.7e-5, softmax-weights 5e-5, gelu 7e-5,
  rmsnorm 8e-6. Per-layer magnitudes match MLX at pos 0 (~2%) and pos 1 (~17%, low-mag amplified).
- These prove the ops and the pos-0/pos-1 assembly. They do NOT catch the pos-8 late-layer bug.

## Deliverables this session (durable, in the working tree)

- **`tests/test_gemma4.py`** — portable primitive checks + AST float-free audit + opportunistic
  real-weight checks + gated (`RINGKIT_GEMMA4_GEN=1`) BOS per-layer magnitude proof. Wired into
  `tests/run_all.py` → **ECOSYSTEM: ALL GREEN** (31 suites). It PASSES but does not test the
  multi-position path, so it does NOT catch the open bug. Add a mirror/MLX multi-position check once
  fixed. It correctly does NOT pin an argmax token.
- **`docs/REPORT-GEMMA4.md`** — see next section; needs revision.

## Status of the report (`docs/REPORT-GEMMA4.md`) — REVISE

It currently frames the argmax miss as "a Q16 precision gap to close" and "faithful architecture."
The precision-gap framing is **wrong**: the mirror-vs-MLX cliff shows a systematic algorithm bug, not
mere Q16 drift (the ring's arithmetic vs its own algorithm is 0.99; the algorithm vs MLX cliffs to
0.76). Do NOT rewrite it to "faithful" until the cliff is localized and fixed. Architecture-level
faithfulness (§2) stands; end-to-end faithfulness does not.

## Environment / commands

- Ring suite: `cd .. && python3 -m ringkit.tests.run_all` (ALL GREEN). Single: `... test_gemma4`.
- Weights (reachable): onix `~/Projects/hpq-kernel-rust/gemma4_12b.onix` (10.9 GB, int8),
  `~/Projects/mprc-scratchpad/hpq_kernel/weights_12b/{embed.bin,norms.bin,layer_scalars.bin}`.
  `gemma4_weights.default_paths()` finds them.
- Independent oracles (D9 labeled — numpy/mlx allowed, all in `scratchpad/`, outside the package):
  - MLX venv: `~/Projects/mprc-scratchpad/.venv-mlx31/bin/python`; model dir
    `~/Projects/mprc-scratchpad/models/gemma-4-12B-it-4bit` (4-bit); loader = MLX `gemma4_text`.
  - hpq int8 ref: `~/Projects/hpq-kernel-rust/target/release/examples/generate --onix <onix>
    --embed <…>/embed.bin --norms <…>/norms.bin --tok <model>/tokenizer.json --prompt "…" --max-tokens N`
    (~1 tok/s). Its `Gemma4Tokenizer` prepends bos=2.
- Ring is ~110 s/token (48 layers, 3840/15360). A 26-token chat prompt ≈ 50 min. Prefer the fast
  mirror + MLX for localization; use the ring only for the final argmax confirmation.
- Scratchpad scripts (key ones): `g4_f64.py` (float64 mirror — NOT independent; `forward_prompt`,
  `forward_perlayer`, argv=op-names forces Q16 for bisection), `mlx_fox_layers.py`/`mlx_pos_vecs.py`
  (independent MLX per-layer), `ring_paris.py` (ring on high-margin chat "capital of France" ids
  `[2,105,2364,107,3689,563,506,5279,529,7001,236881,25685,607,1186,886,3658,236761,106,107,105,4368,107,100,45518,107,101]`
  → MLX 50429 ' Paris' gap 15.17; mirror already returns 50429 ✓; ring run was in background, may be
  incomplete — NOTE a gap-15 prediction survives a moderate bug, so ring→Paris is necessary but NOT
  sufficient to declare faithful).

## Git

Untouched: the pre-existing messy index (spurious staged deletions predating this session) and the
`kvpolar`/`kvadi` staged changes — a separate cleanup decision, deliberately not touched. Do not `git
reset`/commit the emulation work into that mess; stage `tests/test_gemma4.py`, `tests/run_all.py`,
`docs/REPORT-GEMMA4.md`, `docs/HANDOVER-GEMMA4-PRECISION.md` deliberately when ready.

## FULL DEBUGGING LOG — every failure path tried (so nothing is re-chased)

Chronological, with the exact numbers and verdicts. Dead-ends are marked ✗ RULED OUT.

1. **First e2e miss.** Ring on "The capital of France is" (`[2,818,5279,529,7001,563]`) → 496 ' a'.
   hpq int8 → 1144 ' what'. Looked like a bug. ✗ RULED OUT as a *test-target* problem: that prompt is
   CONTESTED — hpq→1144, MLX(bos-prefixed)→496; the two references disagree, so it's a bad oracle. (My
   first MLX run wrongly dropped bos and gave 236772; with bos MLX gives 496.) Do not assert 496.

2. **Fox consensus prompt.** hpq int8 = MLX 4-bit = **4799 ' dog'** (independent agreement). Ring →
   2268 ' little'. This is the real, well-posed failure. ' little' is a coherent near-miss
   ("lazy little dog"), NOT garbage.

3. **CORDIC angle reduction hypothesis** (fox is 9 tokens, pos up to 8, RoPE angle >2π at low-freq
   dims; Gemma2's "Paris" was only 6 tokens so never exceeded 2π). ✗ RULED OUT: `gemma._cordic`
   reduces mod 2π (`while theta > _PI`) and is accurate at 8/10 rad (err <1e-4). Not it.

4. **RoPE formula (local + global partial rotation).** Ring `rope_tables`+`apply_rope` vs MLX
   `nn.RoPE`/`ProportionalRoPE` **at pos 3**: max|Δ|≈7e-4 (CORDIC precision), both local and global;
   passthrough dims bit-exact; global freq divisor = full head_dim, pairs (i,i+256). ✗ RULED OUT AT
   POS 3 — **but NOT at high position** (see suspect #1 above; the re-test at pos 8/20/50 was the
   interrupted final action — DO THIS FIRST next session).

5. **Softmax / uncapped scores.** Gemma4 dropped Gemma2's attn soft-cap (50), so scores are uncapped.
   Bisection (force one op to Q16 in the float64 mirror) showed forcing softmax→Q16 craters L11 to
   cos 0.65 (gelu→0.72). Looked like softmax was the culprit. ✗ RULED OUT: direct op test —
   `infer.softmax` vs float softmax on realistic score vectors (std 16/22, 2-9 keys) has worst
   |Δweight| = 5e-5; `exp_fixed` rel err 8.7e-5. The L11 crater is **low-magnitude AMPLIFICATION**
   (L11 residual collapses to |h|~0.07 due to tiny early `layer_scalars`), not an inaccurate op. Every
   op is accurate: proj bit-exact, exp 8.7e-5, softmax 5e-5, gelu 7e-5, rmsnorm 8e-6, RoPE(pos3) 7e-4.

6. **"No failing math" (WRONG conclusion, then corrected).** Built float64 mirror (`g4_f64.py`), found
   ring-vs-mirror cosine 0.9917 final, mostly 0.99+; per-layer decomposition (pos 1): `ring~mir`
   (ring Q16 arithmetic) 0.99+, `mir~mlx` (int8-vs-4bit weights) comparable/larger. Concluded "ring
   arithmetic is faithful; argmax instability is inherent + weight-quant." ✗ THIS CONCLUSION WAS
   WRONG. The mirror **reimplements gemma4.py's algorithm**, so ring~mir is CIRCULAR (measures
   precision-consistency, not correctness). A shared formula bug is invisible to it.

7. **The correcting test (advisor-driven): mirror vs INDEPENDENT MLX on fox, per-layer, pos 8.**
   Cliff at L40-47 (0.95→0.90→0.84→**0.76**→0.80), vs no cliff on 2-token `[2,818]` pos-1 (L47 0.977).
   → the bug is SYSTEMATIC (shared by ring+mirror), position-dependent, late-layer. **This is the open
   lead.** The float64→8321 argmax (my algorithm's highest precision) diverging from the two
   independent 4799s is the tell, not "instability."

8. **Mirror faithful on HIGH margin.** Chat "capital of France?" (gap 15.17) → mirror 50429 ' Paris' =
   MLX. So the bug is small enough to be swamped by a 15-logit margin but flips the 2.44-gap fox. ⇒
   Paris→Paris on the ring (background run) is necessary-but-NOT-sufficient to declare faithful.

**Net:** ops are accurate; architecture (pos-0/pos-1) is faithful; there is a real multi-position
late-layer FORMULA bug, not yet localized. Next: finish the RoPE-at-high-pos test (#4→suspect #1),
then sublayer-bisect a cliff layer (e.g. L45) with `diag_layer0.py` adapted, mirror vs MLX.
