# HANDOVER — Gemma emulation on the ring (4 GB VM edition)

**Date:** 2026-07-13 · **Repo:** `ringkit` · **Read first:** `CLAUDE.md`, then `docs/REPORT-GEMMA2.md`.

## TL;DR

Gemma2-2B runs **autoregressively on the ring, float-free**: `"The capital of France is" → "Paris."`
(REPORT-GEMMA2.md). Everything lives in `ringkit/emulation/` + `kernels/mprc/gemma/`, SEPARATE from
the pure ring `nn`. All 30 suites green (`python3 -m ringkit.tests.run_all` → `ECOSYSTEM: ALL GREEN`).
The two hard-won lessons this session: **(1) never materialize weights — stream via mmap**, and
**(2) the shell caps at ~40 s per call, so long generation must checkpoint state to disk and resume.**

## The 4 GB constraint — the rules that keep us under it

The VM has **~3.4 GB usable**. The onix is 2 GB and `embed.bin` is 1.18 GB; loaded naively they OOM.
The discipline (hpq runs 8B on an iPhone 13 — same idea):

- **Stream, never load.** Weights are read through a **shared read-only `mmap`** (`MAP_SHARED`,
  `PROT_READ`). Mapped pages count as *reclaimable page cache*, NOT resident RAM. During a full run:
  `used ≈ 230 MB`, `available ≈ 3.4 GB`.
- **Onix:** `Gemma2Weights` mmaps the file once and **slices one tensor at a time** (`self._onix[base:base+of*inf]`),
  a transient ≤21 MB bytes copy that's freed after the `proj`. It caches only the tiny `s_row`/`z_row`.
  Do **NOT** re-introduce a full-tensor cache (that was the 2 GB OOM).
- **Embed / LM head:** the kernel `lm_argmax_file` **mmaps `embed.bin` read-only in C itself** and
  munmaps after — Python holds zero embedding memory. Do **NOT** build a Python `c_uint16` array over
  the 1.18 GB table, and do **NOT** use a writable/`MAP_PRIVATE` COW map for it (COW faults the whole
  1.18 GB into private RAM → OOM). `embed_row(token)` reads just one 2304-value row (fine).
- **Never** `np.load` the `.npz` float weights during inference (that's ~5 GB expanded). They exist in
  `weights_2b/` only as a float oracle for tests.
- Watch it: `free -h | head -2` — `available` must stay > ~1 GB.

## Shell/runtime constraints (this environment)

- **Each shell call ≈ 40 s max, and background jobs are killed shortly after the launching call ends.**
  A full forward token is ~20 s, so you cannot generate a sentence in one call.
- **Solution = the resumable stepper:** `outputs/run_step.py` persists `{ids, gen, pos, cache, hn}` to
  `outputs/gstate.pkl` and does ~34 s of work per invocation, then exits. Call it repeatedly:
  ```bash
  export PYTHONPATH=/sessions/<session>/mnt        # or the repo root
  python3 outputs/run_step.py init "The capital of France is" 6   # tokenize + start
  python3 outputs/run_step.py resume                              # repeat until done=True
  cat outputs/gemma_gen.log
  ```
- **PYTHONPATH must point at the repo root** (the dir containing the `ringkit/` package) or imports fail.
- **Kernel builds on first import** into `kernels/build/qsm_energy-<arch>.so` (gitignored). You may not be
  able to `rm` it (permissions) — that's fine, it rebuilds when the `.c` mtime is newer. It self-tests
  bit-for-bit at load (`qsm_dot` + `lm_argmax`); if a test fails it silently falls back to slow Python.

## What exists (file map)

```
ringkit/emulation/
  gemma.py          Gemma2 forward + greedy generate on ring primitives (float-free). Config G2.
                    _f16_to_fixed, _cordic, rope_tables (geometric inv_freq), proj (act-quant+dequant),
                    rmsnorm_g2 (1+gamma), apply_rope (NeoX half-split), attention (GQA+softcap50),
                    layer_forward, forward_token, generate.
  gemma_weights.py  Gemma2Weights: onix mmap (slice per tensor), embed.bin row decode, norms.bin parse,
                    lm_argmax (-> streaming kernel). default_paths() locates the mounted files.
  tokenizer.py      GemmaTokenizer: BPE encode/decode from tokenizer.json (string only, no math).
  onix.py infer.py ract.py checkpoint.py   (existing; ract gained tanh_fixed + softcap_fixed)
kernels/mprc/gemma/
  qsm_energy.c      qsm_dot (energy-QSM GEMV, multiplier-free, int64 no-fold) + lm_argmax / lm_argmax_file
                    (f16 LM-head argmax, mmaps embed.bin read-only; D9 hardware-bridge, self-tested).
  host.py           ctypes host + Python references + load-time self-tests.
tests/test_gemma2.py    fast portable checks + opportunistic real-weight checks; full gen gated by
                        RINGKIT_GEMMA_GEN=1 (slow).
docs/REPORT-GEMMA2.md   the proof + config + method.
```

## Where the real weights are (mounted via symlink; verify each session)

- onix: `~/Projects/hpq-kernel-rust/gemma2_2b.onix` (182 tensors = the 7 linears × 26 layers only)
- embed + norms: `~/Projects/mprc-scratchpad/hpq_kernel/weights_2b/{embed.bin,norms.bin}`
- tokenizer: `~/Projects/GemmaApp/GemmaApp/tokenizer.json`
- `default_paths()` / `tokenizer.default_path()` probe both `~/Projects/...` and `/sessions/.../mnt/...`.
  If unmounted, tests skip the opportunistic parts and stay green on the portable proofs.

## Config facts you must not get wrong (Gemma2-2B, from hpq.h `G2_*`)

26 layers · hidden 2304 · inter 9216 · vocab 256000 · 8 Q / 4 KV heads · head_dim 256 · RoPE θ=1e4 ·
RMS ε=1e-6 with **(1+γ)** · embed scale **48** (=√2304) · **attention** logit soft-cap **50** ·
**output** logit soft-cap **30** · tied f16 LM head. Residual is Gemma-style post-norm:
`h = x + PostNorm(sublayer(PreNorm(x)))`. `norms.bin` layout: 16-byte header, then per layer
`[pre_attn, post_attn, pre_mlp, post_ff]` each 2304 f16 + 8-byte meta (q/k-norm hd = 0 for Gemma2),
then final norm 2304 f16.

## Charter compliance (keep it this way)

No float / no FPU on the compute path — AST-clean (no `float()`, no float literals, no numpy/torch/math
in `gemma.py`/`gemma_weights.py`; runtime pulls none). Every value product is `rn.mul`/`rn.qsm`; `*`
appears only in integer index/byte-offset math. Kernels reproduce a Python reference bit-for-bit (D9).
Emulation stays out of the pure ring `nn`.

## Performance (this CPU sandbox)

~20 s / token (kernel ≈10 s + Python orchestration ≈10 s); LM-head argmax over 256k vocab ≈1.4 s warm.
Biggest Python cost is `gelu_fixed` over the 9216-wide intermediate (26×). See "Next".

## Next steps (priority order)

1. **Gemma4-12B autoregressive** — same emulation path, `G4_` config: 48 layers, sliding/global
   attention alternation (period 6), partial rotation (0.25 of the 512-dim global heads), per-layer
   residual scalars (`layer_scalars.bin`), per-head Q/K norm (norms.bin carries q/k gammas when hd≠0),
   two RoPE thetas (1e4 local / 1e6 global). Weights: `gemma4_12b.onix` + `weights_12b/`. At 12B the
   onix is 10.9 GB — **must** stream (mmap slice); never cache. Watch the 4 GB ceiling hard.
2. **Speed:** precompute a `gelu` LUT over a fixed Q16 input grid to kill the per-element Taylor `exp`
   in the FFN — the main per-token cost. Then reconsider batching the proj\s.
3. **Verification upgrade:** add a numpy-float oracle forward from the `.npz` weights (test-only,
   labeled) and assert the ring first-token argmax matches it (stronger than coherence alone).
4. Optional: a ring-native RoPE (the CORDIC atan table is already baked; inv_freq is geometric) so
   even table-gen is float-free end to end (already is — noted for audit).

## Gotchas that cost time this session

- `#include <stddef.h>` / `<sys/mman.h>` needed in the C kernel (size_t + mmap).
- ctypes `from_buffer` needs a **writable** buffer → don't feed it a read-only mmap; use the C-side
  file mmap instead.
- `resource.ru_maxrss` is **KB on Linux** — don't misread it as bytes.
- The tokenizer prepends `<bos>` (id 2) and a metaspace `▁`; decode strips a leading space.
