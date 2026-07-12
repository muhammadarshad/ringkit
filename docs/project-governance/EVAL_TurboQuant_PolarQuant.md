# Deep evaluation — TurboQuant & PolarQuant, and what transfers to a ring-native quantizer

Read from the primary PDFs (arXiv), not summaries. Both papers share DNA: authors **Amir Zandieh +
Vahab Mirrokni (Google Research)** appear on both, and both use the same two tricks — **random
rotation preconditioning** and the **1-bit Quantized-JL (QJL)** transform. They are *inference-time
vector/KV quantizers*, data-free, not training methods.

## Sources

- **PolarQuant: Quantizing KV Caches with Polar Transformation** — Insu Han (KAIST), Praneeth Kacham
  (Google), Amin Karbasi (Yale), Vahab Mirrokni (Google), Amir Zandieh (Google). arXiv **2502.02617v1**,
  4 Feb 2025. Code: github.com/ericshwu/PolarQuant.
- **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate** — Amir Zandieh (Google),
  Majid Daliri (NYU), Majid Hadian (Google DeepMind), Vahab Mirrokni (Google). arXiv **2504.19874v1**,
  28 Apr 2025. Blog: research.google/blog/turboquant-...; community code: github.com/0xsero/turboquant.

---

## PolarQuant

1. **Algorithm.** (a) **Random preconditioning:** multiply each KV embedding by a shared random
   rotation/sketch `S` (JL lemma → inner products preserved; rotated vector ≈ isotropic Gaussian).
   (b) **Recursive polar transform:** for `d` a power of 2, pair up coordinates → (radius, angle),
   recurse `log₂d` levels → one final radius `‖x‖₂` and `d−1` angles organized in `log₂d` levels
   (level 1: `d/2` angles in `[0,2π)`; deeper levels in `[0,π/2]`). (c) **Quantize the ANGLES** with a
   fixed codebook; keep the radius in fp. Reconstruction is a product of cos/sin of the dequantized
   angles times the radius.
2. **What/how much:** KV cache only. Bit allocation `b=4` bits for level-1 angles, `b=2` for the rest;
   radius fp16. Llama-3.1-8B example ≈ **3.875 bits/coordinate**, **×4.2** KV-cache compression vs fp16.
3. **Calibration:** **data-free / data-oblivious.** Key insight: after preconditioning the angle
   distribution is *analytically known* (`fΨ ∝ sin^(2^{ℓ-1}−1)(2θ)`), so **no per-block normalization**
   (no stored zero-point/scale — which otherwise costs ~1 extra bit/number). Codebook built once by
   1-D k-means++ on the analytic density; an **offline** shared codebook works across prompts/layers/heads.
4. **Results (Llama-3.1-8B-Instruct):** LongBench avg **48.37** (PolarQuant-R online) vs **48.63** exact
   fp16, beating KIVI 46.70, SnapKV 44.57, PyramidKV 44.03. Needle-in-Haystack score **0.991** vs 0.995
   exact, 0.984 KIVI. Essentially quality-neutral at ×4.2.
5. **Cost / kernels:** angle indices packed into `torch.uint8`; needs a **custom dequant** (cos/sin
   product reconstruction) + the preconditioning matmul + recursive transform. **Honest caveat: it is
   slow at prefill** — Table 2 shows prefill **11.6s vs 2.9s** exact (≈4× slower); token generation
   43.7s, only ~14% faster than KIVI. It's a **memory** win, not a speed win.
6. **Vs prior art:** channel/token-block quantizers (KIVI, KVQuant) pay the normalization-overhead bit;
   token-eviction (SnapKV, PyramidKV, StreamingLLM) is faster but lower quality on long-context recall;
   QJL is 1-bit data-oblivious sketching (PolarQuant reuses its spirit); Lexico is dictionary/sparse but
   slow. PolarQuant's edge = no normalization overhead + best quality at its bit-rate.
7. **One-line:** *quantize KV vectors as angles in polar coordinates after a random rotation, so a
   fixed data-free codebook suffices and normalization constants vanish.*

## TurboQuant

1. **Algorithm.** Two-stage, data-oblivious. **Stage 1 (MSE-optimal):** random-rotate the vector →
   each coordinate then follows a **Beta distribution** → apply a per-coordinate **optimal Lloyd-Max
   scalar quantizer** (continuous k-means on the known Beta). This is MSE-optimal and minimizes residual
   L2. **Stage 2 (unbiased inner product):** MSE-optimal quantizers are *biased* for inner products, so
   quantize the **residual to 1 bit with QJL**, yielding an **unbiased, low-distortion inner-product**
   estimator.
2. **What/how much:** general vectors — **model weights, activations, KV cache, and DB vectors for NN
   search**. Any bit-width `b`.
3. **Calibration:** **data-free / online / worst-case** (no assumptions on data; accelerator-friendly).
4. **Results:** proves an information-theoretic distortion lower bound and matches it within **≈2.7×**
   across all bit-widths/dims (exponential improvement over prior VQ in bit-width dependence). KV cache:
   **quality-neutral at 3.5 bits/channel**, marginal degradation at **2.5 bits/channel**. NN search:
   **beats product quantization (PQ) in recall** with **near-zero indexing time**.
5. **Cost / kernels:** random rotation (they note random **Hadamard** works — see below), per-coordinate
   scalar dequant (LUT-friendly), and a 1-bit QJL side channel. Designed to be **vectorization-friendly**
   (the explicit critique of slow PQ/OPQ codebook lookups).
6. **Vs prior art:** replaces **product quantization / OPQ** (which lack accelerator-friendliness or hit
   suboptimal distortion) and improves KV quantizers; composes with QJL for inner products.
7. **One-line:** *random-rotate so coordinates become i.i.d.-ish Beta, then use the provably optimal
   per-coordinate scalar quantizer, plus a 1-bit residual to keep inner products unbiased.*

## TurboQuant vs PolarQuant

| | PolarQuant | TurboQuant |
|---|---|---|
| Scope | KV cache | general VQ: weights/activations/KV/NN-search |
| Coordinate system | **polar (radius+angles)**, quantize angles | **Cartesian per-coordinate** after rotation (Beta) |
| Optimality | asymptotically optimal for worst-case KV (Thm 1) | **near-optimal MSE + inner-product**, within 2.7× of the info-theoretic bound |
| Inner products | preserved via preconditioning | **explicitly unbiased** via 1-bit QJL residual |
| Bit-rate shown | ~3.875 b/coord, ×4.2 | neutral @3.5 b/ch, usable @2.5 b/ch |
| Speed | memory win, **slow prefill** | built for accelerators / online |
| Shared trick | random rotation, data-free fixed codebook | random rotation, data-free scalar codebook |

**Where each wins:** PolarQuant when the geometry is naturally angular (RoPE'd KV) and you want the
normalization overhead to vanish; TurboQuant when you want provable optimality, unbiased inner products,
generality (weights + NN search), and accelerator speed.

## Relevance to a multiplier-free integer/modular (Z₂₅₆) ring quantizer

**The single most transferable idea — and it is directly ring-compatible:** *random rotation
preconditioning makes the coordinate/angle distribution analytic, which removes per-block normalization
and licenses a **fixed, data-free codebook**.* This is exactly the justification our **fixed-basis ring**
wants (no learned scale/zero-point; cf. silly-noether's fixed embeddings). And crucially the rotation can
be a **random Hadamard transform — entries ±1, so it is pure add/subtract, zero multiplies** — fully
inside the ring's multiplier-free discipline (D9). Both papers already cite random Hadamard as the
practical preconditioner.

**PolarQuant maps almost one-to-one onto our representation.** Our ring value *is* `(energy, phase) =
(magnitude, angle)`; PolarQuant quantizes **angles** and keeps a magnitude — the same decomposition.
Adoptable pieces:
- The **recursive pairwise polar transform** (`log₂d` levels) is a concrete, multiply-light encoder for
  turning a real vector into ring **phases** — a principled alternative to hash→phase.
- "**No normalization**" ↔ our fold-late / fixed-scale ethos; the analytic angle density gives a
  *derived* ring codebook instead of a heuristic one.
- The per-level bit budget (more bits where the angular range is larger) is a ready quantization schedule
  for phase channels.

**TurboQuant contributes the accuracy machinery:**
- Its finding that **MSE-optimal quantization is biased for inner products** is a direct warning for our
  **quarter-square/shift-add dot** — if we quantize ring values for speed, dot-products can drift; the
  **1-bit QJL residual** is a cheap, ring-friendly (±1) correction to keep inner products unbiased.
- The **per-coordinate scalar Lloyd-Max on a known distribution** → a **precomputed fixed LUT** (built
  offline, so its float k-means never touches our hot path); the LUT is pure table lookup — ring-native.

**What conflicts / must be adapted:**
- Both quantize **real fp vectors down to low bits**; our ring is *already* u8/mod-256. So their value to
  us is as **encoders** (how to map fp data into ring phase/magnitude with minimal, provable distortion),
  not as runtime engines to import.
- They store the **radius/norm in fp**; a ring port needs a ring-native **energy** channel for magnitude.
- Codebook construction uses float optimization — fine **offline** (one-time, data-free), but it must be
  frozen into an integer LUT for the ring; don't call k-means in the hot path.
- PolarQuant's **prefill slowness** is a caution: the recursive transform + preconditioning has real
  overhead; adopt the *representation*, not necessarily their reference implementation.

**Worth adopting into ringkit (concrete):**
1. **Hadamard preconditioning (±1, multiplier-free)** as an optional front-end to ring quantization — the
   theory that earns a fixed, normalization-free ring codebook.
2. **Polar/angle encoding** of real vectors into ring phase (PolarQuant's recursive transform), matching
   our `(energy, phase)` — a rigorous fp→ring encoder with an analytic codebook.
3. **1-bit QJL residual** to keep our quarter-square dot's inner products unbiased.
4. Both reinforce the **data-free / fixed-basis** stance we already hold — with published theory and
   quality-neutral numbers behind it.

**Where this sits in the gap analysis:** these are **quantization/encoding-axis** wins (fp→low-bit,
data-free), i.e. the *codec* side that our ring is natively about — they strengthen the "encode real data
into the ring accurately" story and the multiplier-free-dot fidelity. They do **not** touch the keystone
gap (ring-native **training by descent that generalizes**); they are inference-time compressors, not
learners.
