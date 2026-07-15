# HANDOVER — shipping the MPRC architectures (Gluon/Rotor/Soliton) in `ringkit/quanta/`

**Date:** 2026-07-15 · **Read first:** `CLAUDE.md`, `README.md`, then this file.

> **RESOLVED 2026-07-15 (same day, follow-up session): both OPEN problems are closed; Soliton is
> GATED (logit cosine 0.999965, argmax 1246 == oracle, e2e ring forward 0.72 s).**
>
> 1. **Compute path** — `emulation/infer.linear` now routes any tensor ≥ 2^12 MACs through the
>    gated C GEMV (`kernels/mprc/gemma/host.gemv_exact`) via balanced base-256 **weight digit
>    planes** (cached per tensor; shifts/masks only), bit-identical to the Python shift-add
>    reference by construction and gated in `test_quanta` (2384×128 head: 761 ms → 2 ms, ~385×).
>    All quanta linears (frontend proj, in_proj, gates, out_proj, head) inherit this with zero
>    quanta-code changes; Rotor also sped up. Metal GPU was NOT used deliberately: `emu_gemv`
>    serves the 10.9 GB mmapped onix; quanta's ~7 MB .pth weights are cache-resident, where the
>    multithreaded CPU C bridge is the right tier.
> 2. **"Something really wrong"** — NOT in `soliton_forward` (teacher-forced per-stage diff vs the
>    numpy oracle: EVERY stage cos 1.000000). The bug was in the C kernel `rmsnorm_block`
>    (`qsm_energy.c`): Soliton's huge-but-legit Q16 activations (y_prenorm ~4e8 float = ~2^45 Q16)
>    overflowed the **int64 Σx² accumulator** → wrapped negative → cast to unsigned __int128 →
>    `isqrt_c`'s `while ((c<<2) <= m)` wrapped to 0 at m ≥ 2^126 and **spun forever**. THAT was the
>    original "exit 137" (killed hang, not OOM) — reachable from the old pure-Python path too, via
>    `ract.rmsnorm_fixed`'s C route. Fixed: __int128 accumulator + wrap-proof isqrt_c
>    (`c <= (m>>2)`) + host-side |x| < 2^58 guard (Python bigint fallback beyond) + load-time
>    selftest extended with the 2^45..2^55 regimes. Also hardened (exact, bit-identical saturations):
>    `ract.sigmoid_fixed`, `infer.softmax`, `quanta.ssd.softplus_fixed` clamp at frac·2^frac, where
>    the reciprocal exp already floors to 0 — kills the huge-arg bigint blowup class entirely.
>
> Both run_alls (Rosetta dev + native arm64) ALL GREEN incl. test_quanta, test_gemma2, test_gemma4.
> Gluon remains blocked on a checkpoint (unchanged, see §Gluon). Historical handoff text follows.
>
> **WEBAPP E2E (owner follow-up, same day): "the vlm-transformer webapp is the complete test" — DONE.**
> The deployed app (`vlm-transformers/app/serve.py`: image → 16-ch wave cube (113×128) → Rotor
> [RDT + LatticeAttention r=1 + cls head, `heads_rdt_L2_k2384_best.pth`] ∥ Soliton [Mamba2 HF
> export == the gated .pth, verified tensor-equal] → fused softmax) is now reproduced by the ring
> at full scale (D=16, H=113, N=1808): **cos 1.000000 + argmax match on all four (model × image)
> combinations** — a deterministic real-regime frame AND a real shelf photo — against logits
> anchored to the ACTUAL TORCH APP (`bench/webapp_e2e/` — anchor/oracle/ring recipe, numpy oracle
> matches torch to ~2e-5). Committed as the APP-E2E section of test_quanta (rotor + soliton + fused).
>
> The anchor exposed THREE bugs the synthetic 4×4 gates could not see (all fixed):
> 1. **`_rot_half8` convention** — QuantumRoPE4D rotates FOUR 2-dim ADI chunks, not two 4-dim
>    halves. The old form was self-consistently wrong in BOTH the ring and the test's numpy
>    oracle (the circular-mirror trap; small-grid Soliton argmax was 1246, truth is 1946).
> 2. **Vacuum branch missing** — `VacuumDepthEmbedding` gives vacuum tokens (mean phase < 1e-3,
>    e.g. a real image's zero-padded border rows) the learned vacuum embedding INSTEAD of the
>    depth embedding; random grids never have such sites, real frames do (291 in the e2e frame).
> 3. **Arc wrap at phase 1.0** — `(v>>8)&0xFF` sent a saturated 255-pixel (grid 1.0) to arc 0;
>    the deployed clip sends it to 255. Now clamped.
> New ring pieces: `quanta.layers.lattice_encoder_layer` (radius-1 torus window attention,
> QuantumRoPE4D, tanh-GELU ≈ the app's exact-erf nn.GELU to ~3e-4), `quanta.models.
> rotor_lattice_forward` (the RotorHeads path), vacuum-aware `frontend`, C `sigmoid_block`/
> `exp_block` (bit-for-bit == ract, selftested) so the 1808-token forwards run in ~1-2 min.
> CAVEAT (small-grid vlm Rotor gate only): its oracle+ring share the sigmoid-GELU form while the
> real vlm layer uses exact-erf GELU — self-consistent, unanchored on that axis; the APP-scale
> rotor gate IS erf-anchored. Anchor the vlm gate against its torch model when it next matters.
>
> ## WHAT'S LEFT (open ledger as of 2026-07-15 end of session)
>
> 1. **Gluon (MPRCViT)** — forward written (`gluon_forward`), NO checkpoint exists anywhere
>    (all HF repos + local searched). Gate it the moment the owner exports a .pth; use the
>    torch-anchor chain in `bench/webapp_e2e/`, not a hand-written oracle alone.
> 2. **Webapp parity is CLASSIFIER-only.** The ring reproduces the app's cls paths (rotor ∥
>    soliton ∥ fused). The app's OCR read-text and barcode heads (`QCMOCR` decoder — a
>    cross-attention decoder over the shared encoder memory) and the SPEC-017 gates
>    (reject/gallery-kNN/intent) are NOT emulated. Next natural quanta target: `QCMOCR`.
> 3. **vlm Rotor small-grid gate** — anchor against the real torch `vlm_rdt_best` forward
>    (erf-GELU axis, see CAVEAT above). Half a day with the existing recipe.
> 4. **quanta app-scale test cost** — the APP-E2E section adds ~2.5 min to test_quanta (ring
>    rotor ~100 s, soliton ~57 s). If run_all time starts to hurt: C-block the radius-1 window
>    attention and the toroidal diffusion (both are trivial slab kernels; the D9 selftest
>    pattern is established). Not blocking today.
> 5. **Gemma path (pre-existing, unchanged):** chat-template multi-token runs on local weights;
>    bf16-checkpoint semantics bar DEFERRED (local weights only); GPU GEMV kernel-level speed
>    (uchar4/simdgroup reads in `emu_gemv`).

## The task

Ship the three MPRC quantum architectures as ring-native (float-free) forwards under a package.
Family (codenames = the physics quanta), from `~/Projects/vlm-transformers/HANDOVER_2026_06_27_v2.md`:

| codename | class | body | ringkit status |
|---|---|---|---|
| **Rotor**   | MPRCRDT    | shared RoPE encoder, recursive depth, inject+GRU gates | **SHIPPED, gated cosine 1.000000** |
| **Soliton** | MPRCMamba2 | SSD selective scan (gate_lat2d config)                 | **BUILT, NOT yet gated — test dies (exit 137)** |
| **Gluon**   | MPRCViT    | independent RoPE encoder stack                         | forward written; **no checkpoint exists** (see below) |

Package name decided with owner: **`quanta`** (NOT `transformers` — D10 forbids naming a namespace
after the library it parallels; NOT `mprc`; `arc` collides with the ring's phase/position term).

## Package layout — `ringkit/quanta/`  (ALL float-free; AST-guarded in test_quanta)

- `_ringtrig.py` — the `_arch` semicircle ring cos/sin tables (SCALE=21), integer only (`rn.isqrt`,
  no `math`). Exposes `COSQ/SINQ` (Q<frac>) and `COS_U/SIN_U/SCALE` (integers for a test oracle).
  **Float mirror tables were REMOVED from here** (owner caught a float smuggle — oracle floats now
  live in the test, never the package).
- `frontend.py` — shared QCM front-end: `quadrant_project` (4 rectified ring quadrants ×modulation
  → proj) + `add_vacuum_depth` (+depth_emb; the vacuum_emb term is inert for a random grid, which
  has no <1e-3 sites). `frontend(grid_q, W, D, Hh, C, prefix)`.
- `layers.py` — `rope_encoder_layer` (QK-normed QuantumRoPE transformer block; head_dim=32 for RDT).
  Used by Rotor (shared, recursive) and Gluon (stacked).
- `ssd.py` — Soliton body: `softplus_fixed` (max(x,0)+2·atanh(u/(2+u)), u=e^-|x| — ln only sees
  (1,2], no libm), `_rope`/`_rot_half8` (head_dim=8 QuantumRoPE), `_diffuse_step` (toroidal
  (up+dn+lf+rt+4c)>>3), `lattice_state` (global heads=Σ, local=N·diffuse(t)), `mamba2_ssd_layer`.
- `models.py` — `rotor_forward`, `gluon_forward`, `soliton_forward`. Rotor/Gluon use `_proj_head`
  (image_proj → L2 norm). Soliton uses a bare `head` Linear(128→2384) → logits.
- `__init__.py` — surface. Import-clean (no numpy/math).

## Gates — `tests/test_quanta.py` (registered in `tests/run_all.py`)

- **Rotor:** loads `~/Projects/vlm-transformers/vlm_rdt_best.pth`, runs `quanta.rotor_forward`
  (ring) vs an inline numpy float oracle → **max err 1.01e-04, cosine 1.000000. PASSES.**
  Config: D=4,Hh=4,C=128,NH=4,HD=32,DEPTH=2, prefix `vision_encoder.`.
- **Soliton:** loads `full_mamba2_L2_k2384_h16_gate_lat2d_best.pth` (see paths), inline numpy
  oracle + `quanta.soliton_forward` (ring), gates logit cosine>0.999 + argmax. **This is what dies.**
  Config: C=128, NH=16, HD=8, NL=2, NCLS=2384, D=4,Hh=4, prefix `model.`, gate_lat2d
  (use_gate=True, use_causal=False, use_lattice2d=True, use_trapezoidal=True → dA/A UNUSED).
- **AST guard:** every `quanta/*.py` asserted float-free / no numpy·math. PASSES.

The numpy oracle is verified independently: `scratchpad/soliton_oracle.py` runs on the real
checkpoint → finite logits, argmax class 1246 (saved `scratchpad/soliton_ref.npy`). So the
ARCHITECTURE + checkpoint reading are correct; the failure is in the RING forward / its cost.

## ⚠ OPEN PROBLEMS (owner-flagged at handoff) — fix these first

1. **Everything runs on CPU in pure Python; we have a GPU.** `quanta` reuses `emulation.infer`
   (`infer.linear` = pure-Python shift-add MAC) and `emulation.ract`. Rotor's grid is tiny (16
   tokens, C=128, depth 2) so pure Python finishes. **Soliton adds the 2384-class head
   (2384×128 ≈ 305k MACs), 16 heads, and up-to-16-step lattice diffusion, all pure-Python → it
   crawls and the test was KILLED (exit 137, likely OOM/time).** THE FIX: route `quanta`'s
   linear/GEMV through the kernels the Gemma path already uses — `kernels/mprc/gemma/host.gemv_*`
   (C block, bit-for-bit gated) and the metal GPU `emu_gemv` (`kernels/apple/metal`, no-copy onix;
   `RINGKIT_GEMV=metal`). Mirror `emulation/gemma4.py::layer_forward_c` (C-resident activations,
   0.36 s/tok GPU). The `quanta` forwards must NOT call the pure-Python `infer.linear` for real
   sizes. This is the primary reason Soliton "doesn't work" — it's not (only) a logic bug, it's the
   wrong compute path. Verify GPU availability and use it.
2. **"Something really wrong in the code" (owner).** Beyond the CPU issue, LOCALIZE whether
   `soliton_forward` has a genuine bug before trusting any later green: run it on the REDUCED grid
   with per-stage timing/prints (frontend → each SSD layer → head), and diff each ring stage
   against the numpy oracle in `scratchpad/soliton_oracle.py` (teacher-forced per-stage, the
   lesson from the Gemma bug: localize against the INDEPENDENT oracle, not end-to-end). Suspects:
   `lattice_state` local-head indexing / the `N·diffuse` scale, the 5-arm gate (`route`→`gate_proj`
   + `w·arm_bias`→sigmoid), `softplus_fixed` precision, or the head cost. Do NOT declare cosine
   until it actually runs and matches.

## Weight paths (all LOCAL, reachable)

- Rotor: `~/Projects/vlm-transformers/vlm_rdt_best.pth`
- Soliton: `~/Projects/vlm-transformers/huggingface/exported/qcm-rp2k-cloud-matched-results/full_mamba2_L2_k2384_h16_gate_lat2d_best.pth`
- Loader: `ringkit.emulation.checkpoint` — `ck.load_fixed(path, frac=16)` → `fx[name][0]` = Q<16>
  weight (the `W`/`Wf` accessor); `ck._RingUnpickler`/`ck._flatten` for names+shapes+f32 (oracle).

## Gluon (ViT) — NO checkpoint exists

Searched ALL 5 HF repos under `marshadbits` (via HfApi, sandbox off) + local: **no ViT/Gluon `.pth`
anywhere** (only a script `scratch/unroll_rotor_vit.py`). The "7 MB" file the owner recalled is the
Mamba2 (Soliton), not ViT. Owner said (this session): "we have trained model we can resolve that;
most important are Rotor + Soliton." So: build/gate Gluon only when its checkpoint is exported.
`gluon_forward` is already written (independent `rope_encoder_layer` stack, prefix + `encoder.layers.N.`).

## Everything else shipped this session (durable, in the working tree — UNCOMMITTED except the
first Gemma commit `2069fba`)

- **Gemma4-12B** exact ring forward, float-free, **0.36 s/tok GPU / 0.81 CPU (native arm64)**:
  `gemma.proj` = EXACT int8 digit decomposition (no activation quantization; bug fix), C-resident
  `layer_forward_c`, GEMV in 3 gated variants (qsm / hardware-bridge / metal `emu_gemv` over the
  no-copy mmapped onix). Multi-token chat generation matches MLX 24/24 tokens. Ground truth = the
  **f64 mirror** (exact activations); **hpq's f16 is NOT ground truth** (retracted — it contradicts
  the f64 eval of its own weights). See `docs/REPORT-GEMMA4.md`, `docs/HANDOVER-GEMMA4-PRECISION.md`.
- **ADI lossless codec** `ml/adicodec.py` (+ `tests/test_adicodec.py`): bijective byte codec (cascaded
  ADI + constant-column elision + zigzag bit-pack), 2.13× on the raw Laplacian cube, **1× on
  calibrated `.qcm` vectors** (calibration already whitens them — ADI belongs pre-calibration).
  Fixed a dim>255 header-overflow bug caught by real 256-wide `.qcm` data.
- **Benchmarks corrected** (`README.md`, `reports/benchmarks/BENCHMARKS.md`): elementwise is a tie
  with numpy at cache-resident sizes but **beats it ~1.8× at DRAM scale via the MPP block-split**
  (persistent pool `ring_ew_pool` in `kernels/backend/ring_ops.c`); multi-pass L1-resident 14KB-
  canvas tiling + MPP compounds to **~7× (211 GMUPS)** — ablated, bit-identical (`scratchpad/
  ablation.{c,py}`). The "nobody beats bandwidth" claim was per-core, corrected.
- **CXR encoder** (`~/Projects/bitlogix/cxr-model`, Rust on `siliq`): the real system behind the
  Laplacian cube. `.qcm` header stores the ring identity (generator 7, gen_inv 183, vacuum 128,
  quad 64, scales 0/1/4/16/64/128, 128×113). The earlier encoder-symptom analysis (coarse scales
  0, band-independence) traced to the vision front-end's `+128` operator recenter + the photon
  generator, NOT siliq (siliq's real `curvature_field` has NO +128; real defect densities cluster
  ~128, confirmed from `output_overlay/image_1.json`).

## Reproduce / commands

```bash
cd .. && python3 -m ringkit.tests.run_all               # ALL GREEN before the Soliton addition
cd .. && python3 -m ringkit.tests.test_quanta           # Rotor PASSES; Soliton HANGS/KILLED — the bug
python3 scratchpad/soliton_oracle.py                    # numpy oracle runs (reference), argmax 1246
# GPU Gemma path (the pattern quanta must follow): RINGKIT_GEMV=metal arch -arm64 /usr/bin/python3 ...
```

Native arm64 for full kernel/GPU speed: `arch -arm64 /usr/bin/python3`. Bench venv (numpy+torch+MPS):
`~/.venvs/ringkit-bench/bin/python`. MLX oracle venv: `~/Projects/mprc-scratchpad/.venv-mlx31/bin/python`.

## Next two moves (in order)

1. Route `quanta` linear/GEMV onto the C-block + metal GPU path (mirror `gemma4.layer_forward_c`);
   confirm Soliton actually RUNS in reasonable time on the GPU.
2. Teacher-forced per-stage diff of `soliton_forward` vs `scratchpad/soliton_oracle.py` to find the
   real bug; only then gate cosine>0.999 + argmax and register as green. Then commit the whole
   `quanta` package + tests deliberately (the working index still has pre-existing spurious staged
   deletions — stage the new files explicitly, do NOT `git add -A` blindly into that mess... actually
   `git add -A` was used safely once, commit `2069fba`; re-verify `git status` first).
```
