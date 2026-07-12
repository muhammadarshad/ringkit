# Gap analysis — what ringkit must build to replace numpy + torch/tensorflow

Goal (owner): the ecosystem eventually ships a version that **removes standard math (numpy) and
AI/ML libraries (torch/tensorflow)** and runs entirely on our ring-native (Z₂₅₆/QH4), multiplier-free
stack. This document confirms the gap across three axes, grounded in a code survey of ringkit plus
the two attached reference projects.

## What the two attached references actually are (they map the gap, they don't fill it)

- **`hpq-kernel-c`** — a pure-C, single-model inference runtime for Gemma4-12B. Real whole-transformer
  forward pass (GQA attention + KV cache, RoPE, RMSNorm, GELU, softmax, embedding lookup, tied LM head,
  INT8 weights). **Not ring-native:** standard int8 + fp32 (`libm`: sqrtf/expf/tanhf/cosf/sinf), ordinary
  two's-complement (overflow *prevented*, not folded). The **one** overlap with our thesis is its
  quarter-square multiplier-free dot (`a·b = (sq[|a+b|]−sq[|a−b|])>>2`) — which ringkit already has.
  Value: an existence proof that the multiplier-free dot survives in a real 12B forward pass, and a
  complete inventory of the inference ops we must provide ring-native. No training, no autograd.
- **`vlm-transformers` (QCM)** — a torch-hosted image classifier / vision-language model. **Measured,
  real results:** 90.58% RP2K-2384 classification, 98.7% OCR char, 96.9% EAN-13 barcode. It has both an
  **image embedding** pipeline (wave encoder → 16-ch manifold → quadrant projector → RoPE transformer →
  pooled 128-d) and a **text embedding** pipeline (Z256 byte tokenizer → `nn.Embedding` → sinusoidal PE →
  RoPE transformer → pooled 128-d), aligned by CLIP InfoNCE. **But the heavy lifting is 100% torch fp32:**
  every `transformer/*.py` imports torch; QKV/MLP are `nn.Linear`, attention is
  `F.scaled_dot_product_attention`, norm is `nn.LayerNorm`, training is `AdamW + backward()`. Ring content
  is confined to *fixed* RoPE cos/sin tables and *post-training* int8. Value: it defines exactly the
  learned-layer + training + embedding surface ringkit must replace.
- **`silly-noether` (MPRC 4D RoPE retrieval)** — the one reference that IS ring-native and framework-free.
  Its **core embedding + retrieval math is pure-Python Z₂₅₆ ring arithmetic** (hash→phase, odd co-prime
  strides, additive RoPE, 4D conjugate-quadrant tensor, ring-L1 matching). **No learned component** —
  embeddings are *fixed*, hash-derived; "weights" are IDF / `1/(1+dist)` heuristics; no backward/optimizer/
  autograd. Value: an existence proof that a **useful ring-native text embedding + retrieval system already
  runs with zero AI/ML libraries** — it closes part of the text-embedding axis today, and narrows the
  remaining gap to *learned* (vs fixed) embeddings.
  - **Dependency footprint — VERIFIED 2026-07-13** (`~/Documents/antigravity/silly-noether`): **zero `torch`,
    zero `tensorflow`** anywhere in the repo. `numpy` appears in **two** peripheral files (`mprc_faq_engine.py`,
    `paper_eval/gen_data.py`), *neither on the core path* — `mprc_engine/`, `retriever.py` and `cli.py` import
    neither. pandas/pyarrow/nltk are IO/tokenizer plumbing only. So **no ML library touches the math**.
  - **Accuracy — UNVERIFIED, RELAYED.** The **93.6% held-out** FAQ-retrieval figure (with OOV/gibberish
    rejection) comes from a **prior session's report and was NOT re-run here**: the repo contains no results
    artifact, no metric in any README, and no occurrence of the number. `paper_eval/run_eval.py` + `queries.json`
    exist but need `pandas` (absent) and a built index. **Treat as a claim to re-establish, not a datum** —
    re-run the eval before any decision leans on it.

## Axis 1 — Math library (numpy replacement): CLOSEST TO DONE

| Capability | ringkit | Note |
|---|---|---|
| nd-array, creation, reshape, slicing, broadcasting | ✅ `rnp` (RingTensor) | bytearray-backed, mod-256 |
| elementwise + reductions + matmul | ✅ | multiplier-free; C fast-path bit-for-bit gated |
| exact linear solve / modinv / invert-then-solve | ✅ `linalg` | ring superpower |
| trig / exp / log / rotation | ✅ `core/native` | SIN/COS (`_arch`, exact only at cardinals), `ring_exp`/`log` (e=3), `rotate`/`ring_cis` |
| circular stats, ring distance | ✅ `stats` | |
| **general float-range numerics** | ⚠️ energy/phase | real-valued data lives as (energy, phase); ergonomics not numpy-parity |
| **SVD / eig / QR / general inverse, FFT, random distributions, conv/correlate** | ❌ | not needed for ring ML yet, but numpy-parity requires them |

**Verdict:** for ring-native purposes, numpy is largely replaced. Remaining gaps are breadth
(decompositions, FFT, RNG distributions, convolution) — mostly additive, not blocking.

## Axis 2 — Deep-learning framework (torch/tf replacement): THE BIG GAP

| Capability | torch/tf | ringkit today | Gap |
|---|---|---|---|
| Tensor + autograd | ✅ | ✅ dual-ring `Var`/`TVar` (elementwise, matmul, sin/cos) | autograd coverage narrow |
| **Gradient-descent training that generalizes** | ✅ (SGD/Adam) | ⚠️ **descent is "flaky on ring"**; real learning is exact **solve** / invert-then-solve | **CRITICAL** — solve covers linear+invertible only, not representation learning |
| `nn.Linear` (learned) | ✅ | ✅ but fit by exact solve, not trained | ok for linear |
| **`nn.Embedding` (learned lookup table)** | ✅ | ❌ | needed for text/image tokens |
| **Softmax attention (learned QKV, soft weights)** | ✅ | ⚠️ hard argmax content-routing only, exact-match tested | soft weighting is now constructible (see below); **learned** QKV still gated on Axis-2 |
| **LayerNorm / RMSNorm (ring-native)** | ✅ | ❌ | normalization is float in both refs |
| **GELU / softmax / activations (ring-native)** | ✅ | ⚠️ SIN activation only; **forward softmax has a form** (Boltzmann LUT + energy-domain normalize) but is unbuilt, and its derivative does not exist | forward: build it; **backward: open** |
| Conv2d / pooling | ✅ | ❌ | needed for image encoders |
| **Loss functions (cross-entropy, InfoNCE) ring-native** | ✅ | ❌ | gates classifier + contrastive training |
| Optimizers (Adam), LR schedules | ✅ | ⚠️ sign-SGD + coordinate descent (toy) | needs to scale |
| Dataloaders / batching | ✅ | ⚠️ `rk.data` split/batch/encode | basic |
| GPU execution | ✅ mature CUDA/MPS | ⚠️ Metal/C kernels, GPU numbers unverified off-Mac | maturity + portability |

**Verdict:** this is the real gap. ringkit's "solve, don't descend" is genuinely powerful for
linear and invertible-nonlinear maps, but **representation learning** (embeddings, deep classifiers,
contrastive VLMs) needs *descent that generalizes* plus learned primitives (Embedding, softmax
attention, LayerNorm, GELU, cross-entropy/InfoNCE) with autograd through all of them. Today those
are either missing or toy-scale. **The gating item is a ring-native training path that trains a
multi-layer nonlinear net end-to-end and generalizes on held-out data with a control that fails.**

### Correction: softmax is NOT an open ring problem in the forward direction (D11 — the form already exists)

An earlier draft of this doc said softmax was blocked because `ring_exp` is periodic and so no
monotone exponential base exists on the ring. **That reasoning was wrong, and it looked for the form
in the wrong place.** The ring-native exponential is not `ring_exp` (the e = 3 discrete-log map, which
is indeed periodic); it is **geometric decay**, and it is already built and shipped twice:

- `physics/gauge.py::boltzmann_lut(beta)` — `lut[d] = floor(255 * f^d)`, `f = (256-beta)/256`, by
  fixed-point accumulator (`rn.mul` + shift, multiplier-free). This IS `e^{-beta d}` on the ring.
- `physics/measure.py::born_weights(f)` — the same decay stepped by odds, giving the Gaussian.

Both are **monotone**, not periodic. Softmax never needed `ring_exp`.

The obstacle the earlier draft failed to name is **normalization**: softmax is `e^{z_i} / Σ e^{z_j}`,
and a modular divide collapses on even divisors (zero-divisors). This dissolves under **fold-late (D4)**:
keep the weighted sum in **ENERGY** (unfolded) and normalize once with `rn.mf_floordiv` — that is integer
division, *not* a modular inverse, so zero-divisors never arise. `lut[0] = 255` for every beta, so the
denominator is never zero. The whole construction, run and reproducible (D1 — not yet a suite):

```python
from ringkit.core import native as rn
from ringkit.physics.gauge import boltzmann_lut

def attend(logits, values, beta):                  # ring softmax-weighted attention
    lut = boltzmann_lut(beta)
    m = max(logits)
    w = [lut[min(m - z, 255)] for z in logits]     # Boltzmann in the logit gap; w[argmax] = 255
    num = 0
    for wi, vi in zip(w, values):
        num += rn.mul(wi, vi)                      # shift-add, stays in ENERGY (never folds)
    return rn.mf_floordiv(num, sum(w)), w          # ONE integer divide — not a modular inverse
```

Over logits `[10,200,190,3,255,128]` and values `[0,40,80,120,200,250]`:

| beta | weights | out |
|---|---|---|
| 0 | `[255,255,255,255,255,255]` | 115 (uniform mean) |
| 4 | `[5,107,91,4,255,34]` | 144 (soft) |
| 16 | `[0,7,3,0,255,0]` | 194 (peaked) |
| 64+ | `[0,0,0,0,255,0]` | 200 (exact argmax) |

A real temperature, uniform → soft → argmax. An even weight-sum (510) divided exactly (mean of 60 and
100 returned 80), confirming the zero-divisor wall is an artifact of folding early, not of the ring.

Two honest limits carry forward:

1. **Coarse temperature.** The band saturates to argmax by roughly beta=16 — expect ~5–6 usable
   temperature settings, not a continuum.
2. **The backward pass is still open.** This is a forward construction. The derivative of the LUT
   (`d softmax`) does not exist, so this does **not** unblock training — softmax is critical-path item 2,
   and item 1 (descent that generalizes) is untouched by it.

## Axis 3 — Text / Image embedding: ENTIRELY GATED BY AXIS 2

| Piece | QCM (torch) | ringkit / ring-native today | Gap |
|---|---|---|---|
| Byte/Z256 tokenizer | ✅ `Z256Tokenizer` | ✅ **silly-noether** (hash→phase, trigram/word) | none for bytes |
| **Fixed (hash) ring embedding + retrieval, torch-free** | — | ✅ **silly-noether** 4D quadrant tensor (93.6% held-out — *relayed, unverified*) | **already works, zero ML libs** (dep-footprint verified) |
| Learned token embedding | ✅ `nn.Embedding(256,128)` | ❌ | needs learned lookup + descent |
| Positional encoding | ✅ (float sinusoid) | ✅ **exact additive RoPE** (ours is better here) | — |
| Image wave/patch encoder | ✅ `QCMWaveEncoder` (fixed, no-param) | ❌ | portable (no-param) — good first target |
| Transformer encoder → pooled vector | ✅ (torch) | ⚠️ blocks exist, not trained at scale | needs Axis-2 training |
| Contrastive alignment (CLIP/InfoNCE) | ✅ | ❌ | needs InfoNCE loss + descent |
| **Learned embeddings that generalize semantically** | ✅ (90.58% cls proven) | ❌ | **blocked on Axis 2** |

**Verdict (revised):** the text-embedding axis is further along than a ringkit-only view suggests.
`silly-noether` already delivers a **useful, torch-free, ring-native text embedding + retrieval**
system (fixed 4D-quadrant embeddings; 93.6% held-out with correct OOV/gibberish rejection — a
**relayed number, not re-verified this session**, see the caveat above). So "embed text on the ring with
zero AI/ML libraries" is **done for the fixed-basis case** — the *torch-free* half of that sentence is
verified, the *how-well* half rests on an unreproduced figure. The remaining gap
is narrower and precise: **learned** embeddings — a trained semantic space (needed for image
classification-grade recognition and CLIP-style alignment) — which is still blocked on the Axis-2
training keystone. Image embedding remains open (the no-param wave encoder is portable, but a trained
pooled representation waits on descent).

## The critical path (what to build, in order)

1. **Ring-native losses + a training path that generalizes.** cross-entropy and InfoNCE over the ring;
   a descent (or hybrid solve+descent) loop proven to train a *nonlinear multi-layer* net to held-out
   generalization with a failing random-label control. This is the keystone — everything else waits on it.
2. **Learned layer primitives, ring-native, autograd-covered:** `Embedding` (trainable lookup),
   softmax-weighted attention with learned QKV, `LayerNorm`/`RMSNorm`, and a ring-native normalization/
   activation story. The softmax **forward** form is settled (Boltzmann geometric decay + energy-domain
   `mf_floordiv` normalize — see the correction under Axis 2); what is left is to build it into the
   substrate with a suite, and to derive its **backward** (the LUT has no derivative yet). GELU remains
   unformed. Note the forward build does not depend on (1) — it can land first.
3. **Image + text embedding pipelines** on top of (1)+(2): port the no-param wave encoder and the Z256
   tokenizer, train a small encoder, prove a held-out retrieval/classification number with a control.
4. **Scale + hardware:** conv, larger models, and portable GPU execution to approach torch/tf parity.

## Honest bottom line

- numpy-replacement: **mostly there** for ring purposes.
- torch/tf-replacement: **real gap**, and it is specifically *learning by descent that generalizes* +
  the learned/normalization/activation/loss primitives — not matmul (we have that), and no longer the
  softmax *forward* (its form is the Boltzmann geometric decay we already ship; only its derivative is open).
- embeddings: **fixed-basis text embedding + retrieval already works torch-free** (`silly-noether` —
  dependency footprint verified; its 93.6% held-out figure is *relayed and still to be re-established*);
  the gap narrows to *learned* embeddings + image embedding, which are gated by the training keystone.
- **provenance rule for this doc:** claims verified by execution this session are marked VERIFIED (the
  softmax construction; the silly-noether dependency footprint). Everything relayed from a prior session
  or another report is marked as such. Do not let the two blur — D1.
- `hpq-kernel-c` and `vlm-transformers` are **torch/float** — the map of the target, not a shortcut;
  ready-to-transfer pieces from them are the quarter-square dot (already ours), the exact ring trig, and
  the no-param wave encoder. `silly-noether` is the exception: it is **already ring-native and
  framework-free**, and is the template for the retrieval/embedding surface (fixed-basis) plus the
  "ring way" idioms (fold-late energy/phase, odd-stride channels, 4D quadrant orthogonalization).
