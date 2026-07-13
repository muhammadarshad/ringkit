# REPORT — Gemma2-2B autoregressive generation on the ring (emulation engine)

**Result (verified by execution, D1):**

```
PROMPT:  "The capital of France is"
prompt ids: [2, 714, 6037, 576, 6081, 603]
OUTPUT:  "The capital of France is Paris."
```

Gemma2-2B, run **entirely on the ring** — no float on the compute path, the FPU replaced by
ringkit's own QCM primitives. Coherent output is the honest end-to-end proof (a wrong forward
yields token soup, not "Paris").

## Stance

Gemma is a model we do not control → ringkit is an **emulation engine** (like TurboQuant/PolarQuant:
ingest weights, run the forward), but float-free. All of this lives in `ringkit/emulation/` +
`kernels/mprc/gemma/`, SEPARATE from the pure ring-native `nn`. hpq is the reference for WHAT the
ops are; the arithmetic is ringkit QCM-enabled code, never a copy of hpq's kernel.

## Config (Gemma2-2B, from hpq.h G2_*)

26 layers · hidden 2304 · intermediate 9216 · vocab 256000 · 8 Q / 4 KV heads (GQA) · head_dim 256 ·
RoPE θ=1e4 · RMSNorm ε=1e-6 with the `(1+γ)` convention · embed scale 48 (=√2304) · attention
logit soft-cap 50 · output logit soft-cap 30 · tied f16 LM head.

## Pipeline (float-free)

- **Weights.** ONIX int8 linears streamed from a shared read-only `mmap` (reclaimable page cache,
  not RAM-resident — one tensor sliced at a time). `embed.bin` (f16, tied) and `norms.bin` (f16
  gammas) decoded to Q16 fixed-point by **integer mantissa-shift** (`_f16_to_fixed`, no FPU).
- **Linear.** `out = dot(xbar−128, x_s8) · act_scale · 2^s_row / z_row`. Activation quant uses a
  power-of-2 `act_scale = 2^a`, so every scale is a **shift**; the dot runs on the **energy-QSM
  kernel** (int64 accumulate, no fold, quarter-square — multiplier-free); `/z` is one ring divide.
- **RoPE.** `inv_freq_i = θ^(−i/128)` is **geometric** — `r = θ^(−1/128)` via `ract.exp_fixed`, then
  `r^i` by repeated `rn.mul`; cos/sin from a **ring CORDIC** (shifts + adds + a baked atan table,
  max err 7.9e-5). No runtime trig.
- **Norms / activations.** RMSNorm via ring `isqrt`; GeGLU via `ract.gelu_fixed`; attention softmax
  via integer Taylor `exp`; soft-cap via `tanh = 2σ(2x)−1`.
- **LM head.** The soft-cap is monotone, so greedy decoding = **argmax of the raw dot**. The kernel
  `lm_argmax_file` mmaps `embed.bin` READ-ONLY itself (zero-copy streaming, Python holds no
  embedding memory) and decodes f16→fixed with integer bit ops.

## Memory & performance

Peak resident stays low — the 2 GB onix and 1.18 GB embed are streamed as reclaimable page cache
(`free`: used ≈230 MB, available 3.4 GB during a run). Answers the on-device reality (hpq runs 8B on
an iPhone 13): stream, never materialize. ~20 s / token on this sandbox CPU (kernel ≈10 s + Python
orchestration ≈10 s); LM-head argmax over the full 256k vocab ≈1.4 s (warm).

## Charter compliance

- **D1** verified by execution (real generation + `tests/test_gemma2.py` bit-exact checks).
- **No float, no FPU** on the compute path — AST-clean (no `float()`, no float literals, no
  numpy/torch/math imports); runtime pulls none. Every value product is `rn.mul`/`rn.qsm`; `*`
  appears only in integer index/byte-offset math.
- **D9** the kernels reproduce a Python semantic reference bit-for-bit (self-tested at load).
- Emulation-only: the pure ring `nn` is untouched.

## Reproduce

```bash
export PYTHONPATH=$PWD
# fast machinery checks (in run_all):
python3 -m ringkit.tests.test_gemma2
# full generation proof (slow, ~min/token):
RINGKIT_GEMMA_GEN=1 python3 -m ringkit.tests.test_gemma2
```
