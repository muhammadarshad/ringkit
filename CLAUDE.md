# CLAUDE.md — ringkit

Ring-topology (Z₂₅₆ / QH4) compute ecosystem. Hard ring physics/math lives in a verified
substrate; three engineer-facing facades hide it so ordinary engineers use it like numpy/torch
without touching mod-256, energy/phase, vacuums, or zero-divisor collapse.

## Run the verification loop (do this after every change)

```bash
# Run from the PARENT of the repo (the repo root IS the `ringkit` package):
cd .. && python3 -m ringkit.tests.run_all        # must print "ECOSYSTEM: ALL GREEN"
cd .. && python3 -m ringkit.tests.test_<module>  # a single suite
```

Kernels (C) build on first import via ctypes into `kernels/build/` (gitignored); if a `.so` is
stale, delete `kernels/build/` and re-import.
Builds target the **running interpreter's** architecture and are arch-keyed (`<stem>-<machine>.so`),
so the Rosetta x86_64 dev Python and a native arm64 one coexist. Native run (4x faster C kernels):
`arch -arm64 /usr/bin/python3 -m ringkit.tests.run_all` — also ALL GREEN.

## Non-negotiable disciplines (docs/project-governance/CHARTER.md — D1–D11) — break these and the work is wrong

- **D1 Verify by execution.** Never conclude without running it. Every non-trivial claim is
  backed by a test. Prefer exhaustive checks over the 256 ring where feasible.
- **Multiplier-free semantic layers.** No `*`, `//`, `**`, `/` and **no standard-math imports**
  — architectural, not aesthetic: multipliers ARE the silicon bottleneck; QSM + the ring
  dot-product (tables/adds/shifts) are the bypass that makes multiplier-free silicon buildable.
  (numpy/math/scipy) anywhere under `core/ stats/ linalg/ rnp/ rmath.py rcollections/ physics/ ml/ nn/ data.py`.
  (`ringkit.rnp` / `ringkit.rmath` are our REPLACEMENTS for numpy/math — original names per D10.)
  Use `rn.mul` (shift-add), `rn.ipow`, `rn.mf_floordiv`, `rn.ring_pow`, etc. AST-audit new files:
  `python3 -c "import ast,sys;[print(n.lineno,type(n.op).__name__) for n in ast.walk(ast.parse(open(sys.argv[1]).read())) if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Mult,ast.FloorDiv,ast.Pow,ast.Div))]" <file>`
- **Two-layer rule (D9).** Only `kernels/` (C/SIMD) may use hardware `*`/`<<`; it must reproduce
  the Python semantic reference **bit-for-bit**. Tests may use numpy/math **as a labeled oracle only**.
- **Fold late.** A ring quantity is `(energy, phase)` = `(quotient, remainder mod 256)`. Stay in
  unfolded ENERGY to keep operations reversible; fold to phase only when you need the ring position.
  Gradients (ENERGY) must NOT fold mod 256; values (ARC) do. Odd = unit (reversible); even =
  zero-divisor (collapses). Ring-native e = **3** (`rn.RING_E`, `ring_exp`/`ring_log`).
- **Solve, don't descend.** Linear layers = exact `linalg.solve` (mod 256). Nonlinear invertible =
  `linalg.fit` (invert-then-solve). Descent (`ml/autograd`) is the fallback, and it's flaky on the ring.
- **Exact vs approximate is labeled.** `SIN/COS` use the `_arch` semicircle (exact only at the 4
  cardinals; ~28–31% shape gap elsewhere). Never present an approximation as exact. Rotation by a
  quarter turn (`rotate`, iota) IS exact; general angle-addition is NOT.
- **Never quantize the exact identity to save bits; never import a Euclidean/standard form to define it (CHARTER C9).**
  The ARC/phase position (exact additive RoPE, exact Δtick, exact quadrant) is carried at FULL
  precision, always. The ring-native KV element is **ADI (accumulation, differential)** —
  `(x,y)→(x+y, x−y)`, exact/reversible/multiplier-free (`ml/kvadi.py`) — NOT the Euclidean polar
  form (`atan2`+`sqrt(x²+y²)`), which is a lossy anti-pattern and was removed. Compression may crush
  only redundant magnitude/energy, never the arc, and NEVER to beat a paper's bit-count.
- **Names are handles.** Wilson/Metropolis/Euclidean etc. are borrowed labels, not standard-math
  imports. Don't force ring behavior onto standard math or vice-versa.
- **Naming obeys 5W (D10).** Minted names answer Who/What/When/Where/Why and stay ORIGINAL:
  our namespaces are never named after the libraries they replace (`rnp` not numpy, `rmath` not math).
- **Math and physics first (D11).** Ask the mathematical/physical form BEFORE building. The physics
  here is QUANTUM, realized through MPRC (spins, vacuums, quantum walk, gauge, criticality) — never
  a re-implementation of standard-model/continuum methods; those are labeled handles only. The forms
  are ONE interlocking physics, not a menu: the problem at hand selects which form to state first.

## Stance: emulation engine vs OS (docs/project-governance/STANCE_emulation_vs_os.md)

Traditional models we can't control (Gemma, external checkpoints) → ringkit is an **emulation
engine** (like TurboQuant/PolarQuant: ingest weights, emulate the forward on the ring). The
emulation code lives in **`ringkit/emulation/`** (`checkpoint`/`onix`/`infer`/`ract`/`gemma`/
`gemma_weights`/`tokenizer`) + the `kernels/mprc/gemma/` kernel (energy-QSM GEMV + streaming f16
LM-head argmax) — SEPARATE from the pure ring `nn` so it never disturbs the native stack.
**PROVEN (docs/REPORT-GEMMA2.md): full Gemma2-2B autoregressive generation on the ring,
float-free — "The capital of France is" → "Paris."** Weights streamed via mmap (2GB onix + 1.18GB
embed as reclaimable page cache, ~230MB resident — stream, never materialize; hpq runs 8B on an
iPhone 13). RoPE inv_freq is geometric (r^i) with cos/sin from a ring CORDIC; LM head = argmax of
the raw dot (soft-cap is monotone). **Gemma4-12B (docs/REPORT-GEMMA4.md): same path at G4_ config —
faithful 48-layer forward, fox argmax ' dog' reproduced after making `gemma.proj` EXACT — the ring
does NOT quantize the model: activations are decomposed exactly into int8 digit passes for the QSM
table (residual re-encoded until ZERO, ≤4 passes), bit-exact to the exact integer dot. The old
single-pass truncating grid broke under ~60× late-layer activation outliers; localize such bugs
against INDEPENDENT refs [MLX, f64-mirror-with-exact-activations] with teacher-forced per-layer
isolation — a self-built mirror running the SAME algorithm is circular, and hpq's f16 is NOT
ground truth (retracted 2026-07-15: it contradicts the f64 evaluation of its own weights).** Our MPRC architectures (RDT/Mamba2/QuantumRoPE) → ringkit is the **operating system**
(native: `nn`/`core`/`physics`/`ml`). BOTH: no float, no FPU — every FPU op is replaced by a ringkit
QCM-enabled primitive (`qsm`/`mul`, `ract.exp_fixed`, `boltzmann_lut`, `isqrt`, shifts). Do NOT copy
the reference kernel (hpq is the WHAT; ringkit QCM code is the HOW); never a float fallback.

## The honesty bar for ANY learning/ML claim

Self-retrieval / "fits training" proves nothing (memorizing is trivial). Every ML capability must
show **held-out generalization** on data never seen, PLUS a **control that fails**: a random-label
run must collapse to chance, and/or a position-only/content-only baseline must fail. See
`tests/test_ml.py`, `tests/test_attention.py`, `tests/test_nn_facade.py`.

## Layout

```text
ringkit/
  core/constants.py  CORE ring constants (TAU/HALF/Q/Q2/SCALE/VACUUMS/RING_E/IOTA), single-sourced
                     and FROZEN (assignment raises). Never re-declare these in a subgroup — import
                     them. Subgroup-domain constants (qcm.HV_*, measure rulers) stay local.
  core/native.py     substrate ISA: Z256 consts, mul/ipow/mf_floordiv, qsm, isqrt, ring_neg,
                     SIN/COS/TAN family, ARC*, iota/IOTA rotor, ring_exp/ring_log/ring_pow (e=3),
                     ring_cis (Euler), rotate/cis_rotate, ADI, codec  — all multiplier-free
  core/calculus.py   d_rot / integral / differential / FTRC
  linalg/            solve.py (exact mod-256 solve, modinv, is_invertible), fit.py (invert-then-solve)
  stats/stats.py     ring_dist, ARCTAN2, circular mean/median, geometric_mean
  rnp/               numpy REPLACEMENT (original name, D10): __init__.py (the surface: rk.rnp /
                     import ringkit.rnp), tensor.py (RingTensor: the package's ndarray, bytearray-backed)
  rcollections/      ring-native data structures (placeholder — containers, not math objects;
                     original name per D10, and never shadow a stdlib module name: a package dir
                     named `collections` broke stdlib imports for any python run inside the repo)
  physics/           measure.py (+ Born-rule cloud/collapse: ring Gaussian by odd-step geometric
                     decay), qcm.py (+ stride7_orbit: 36-bin vacuum-avoiding QCM walk), gauge.py
                     (EXPONENTIAL integer Boltzmann [geometric decay — the physics form], criticality
                     + mass_gap_scan), sim.py (Gauge facade: thermalize/action/order/profile/phase)
  ml/                autograd.py, tensor_autograd.py (TVar), optim.py, nn.py (low-level), attention.py,
                     kvcache.py (RingKVCache + Boltzmann-soft attention: score in ENERGY, weights by
                     geometric decay [gauge.boltzmann_lut — the ring exponential, NOT the periodic
                     ring_exp], values blended CIRCULARLY around the winner [angles: a linear mean
                     wraps], normalized once in ENERGY by mf_floordiv so no zero-divisor arises.
                     RoPE applied at insert. Bar: cached == uncached BIT-FOR-BIT. Data-free: the
                     ring IS the codebook — no scales/zero-points stored)
  kernels/           [D9 silicon] backend/ (ctypes loader __init__.py + ring_ops.c, zero-copy,
                     Python fallback; gemm.py + ring_gemm.c: ring GEMM in 3 gated variants —
                     hardware-`*` bridge 215 GMAC/s, multiplier-free shiftadd 55 [beats BLAS],
                     multiplier-free QSM table 13 — serves rnp matmul), mprc/qcm/ (qcm_kernel.c, cache_manifold.c),
                     mprc/lattice/ (gauge.c [threaded *_mt slab bins + C observable reductions] +
                     host.py: ctypes host, py reference, observables, GPU session_for),
                     apple/metal/ (ring_ops+gauge+gemm shaders, shim.m, host.py — all bit-for-bit
                     verified: elementwise OPT-IN [C wins]; gauge sweep AUTO >=32^3 + derived-RNG
                     thermalize [rk_mix32, py==C==metal] 0.12 ns/node/sweep; GPU GEMM mul 105 /
                     qsm-LUT 59 GMAC/s [zero-multiply, beats torch-mps] — CPU bridge keeps route), mprc/hpq/ + nvidia/cuda/ +
                     apple/ml/ (placeholders; CoreML descoped — unified-GPU focus),
                     build/ (arch-keyed .so, gitignored)
  tests/             one test_<module>.py each; run_all.py aggregates (21 suites)
  bench/             apples-to-apples vs numpy/torch (C6 scaffolding — the ONLY place standard
                     engines may be imported; baselines bit-for-bit gated before timing).
                     Results: docs/BENCHMARKS.md (native fight: GPU thermalize ~85x vs torch-mps
                     ON THE SAME unified GPU; ~65x vs multithreaded torch-cpu)
  docs/              project-governance/ (SDLC docs: CHARTER.md, SRD.md, ECOSYSTEM_SRD.md,
                     ECOSYSTEM.md, MANIFEST.md)

ringkit/nn/          FACADE (top-level pkg): layers.py (Layer, Linear, Dense, Sequential),
                     KVCache (= ml.kvcache.RingKVCache) re-exported for decode-time attention,
                     transformer.py (RoPE, Attention, TransformerBlock, Transformer: induction +
                     in-context recall; HopBlock + Stacked: multi-block solve-trained deep recall,
                     held-out 1.0 with depth + random controls at chance). All re-exported at
                     rk.nn. Ring hidden; .raw hatch.
ringkit/rmath.py     stdlib-math REPLACEMENT (original name, D10): math-shaped handles (sin/cos/exp/
                     log/isqrt, tau/pi/e with e = RING_E = 3) re-exported from core — no behavior of its own
ringkit/data.py      FACADE: encode/encode_range, one_hot, split, batches
ringkit/quanta/      the MPRC architectures as ring-native float-free forwards (D10 name — NOT
                     "transformers"): _ringtrig (integer _arch tables), frontend (shared QCM
                     quadrant+vacuum front-end: vacuum tokens get the vacuum embedding, arc
                     clamped at phase 1.0 — both load-bearing on REAL images only), layers
                     (rope_encoder_layer + lattice_encoder_layer: radius-1 torus window attention,
                     QuantumRoPE4D [rotate-half = FOUR hd/4 ADI chunks — anchored vs torch],
                     tanh-GELU), ssd (Mamba2 SSD gate_lat2d), models (rotor/rotor_lattice/gluon/
                     soliton_forward). GATED at the deployed webapp's full scale (test_quanta
                     APP-E2E: N=1808, real checkpoints, numpy oracle matched to the ACTUAL torch
                     app at ~2e-5): Rotor cos 1.000000, Soliton cos 1.000000, fused argmax ==
                     the app, on a real-regime frame; a real shelf photo verified in scratch.
                     Gluon (MPRCViT) written, awaits a .pth. Linears auto-route through the gated
                     C GEMV (infer.linear digit-plane path). AST-guarded float-free.
```

Engineer entrypoint: `import ringkit as rk` → `rk.nn`, `rk.data`, `rk.physics.Gauge`, `rk.rnp`
(also `import ringkit.rnp as rnp`, `import ringkit.rmath as rmath`).
Every facade object hides ring internals and exposes `.raw` for power users.

## Status

All suites green (run_all, incl. test_gemma2 + test_gemma4 + test_quanta), Rosetta dev AND native
arm64. **quanta (2026-07-15): Rotor + Soliton SHIPPED and gated AT THE DEPLOYED WEBAPP's full
scale** — the vlm-transformers webapp e2e (image → wave cube → RotorHeads ∥ Mamba2 → fused) is
reproduced by the ring at N=1808 on real checkpoints, cos 1.000000 + argmax on every path,
against logits anchored to the ACTUAL torch app (never a self-built oracle alone — the anchor
caught a self-consistently-wrong 4D-rope rotate-half, a missing vacuum branch, and an arc wrap
at phase 1.0 that random-grid gates could never see). `infer.linear` routes ≥2^12-MAC
tensors through the gated C GEMV via balanced base-256 weight digit-planes (bit-identical, ~385×),
+ C sigmoid/exp blocks; the Soliton "exit 137" was a C `rmsnorm_block` int64 Σx² overflow →
`isqrt_c` infinite loop on legit ~2^45 Q16 activations (fixed: __int128 + wrap-proof isqrt +
host range-guard + selftest magnitude regimes; sigmoid/softmax/softplus got EXACT frac·2^frac
saturations — provably bit-identical, no more huge-arg bigint blowups). Substrate (core/stats/linalg/rnp/
physics/ml/kernels) is production-grade and AST-clean. Facades (`rk.nn`, `rk.data`, `rk.physics`)
built and verified with held-out + controls. Emulation engine: loads real .pth (RDT/Mamba2) and
Gemma .onix; ring inference verified bit-exact / cosine 1.0 vs float; **Gemma2-2B generates real
text on the ring, float-free (REPORT-GEMMA2.md); Gemma4-12B faithful forward, `gemma.proj` GEMV now
BIT-EXACT (exact digit decomposition — no activation quantization), fox argmax ' dog'
(REPORT-GEMMA4.md).** Quickstart DONE. Apple backends Phases 0-1c DONE; CoreML descoped.
Speed: the Gemma4 forward is FULLY on the kit's speed model — 115 → 0.81 s/token native-arm64
CPU → **0.36 s/token on the unified GPU (RINGKIT_GEMV=metal; 320×; 4.4× faster than hpq)**, all
bit-for-bit gated incl. through the GPU (dual-path + mirror pins pass under metal). GPU GEMV =
`emu_gemv` in kernels/apple/metal/emulation.metal + `rk_metal_onix_map` (the 10.9 GB onix mmap
wrapped as a NO-COPY shared MTLBuffer — the GPU reads the file's own page-cache pages; one
threadgroup per output row, long accumulators; CPU fallback for out-of-region tensors /
out-of-range activations keeps bit-identity by routing). CPU rate came from narrowing bridge
activations to int32 (exact: products ≤ 2^38; range-checked __int128 scalar fallback) so the
u8→i32 widening MLA vectorizes (NEON smlal; Rosetta lacks AVX2 — run natively). All
bit-for-bit gated: C-RESIDENT activations (`layer_forward_c`: the hidden vector
lives in C buffers across all 48 layers — embed decode, rmsnorm_rows, GEMV, RoPE, KV-slab
insert (memmove), attention, gelu_mul, residual add and layer scalar are ALL C blocks; ONE list
crossing per token, for the LM head), fused exact GEMV reading onix slabs IN PLACE (MAP_PRIVATE
zero-copy) in TWO GATED VARIANTS per the ring_gemm precedent — `qsm_gemv_exact_mt`
(multiplier-free QSM digit path, silicon/reference; RINGKIT_GEMV=qsm) and `qsm_gemv_bridge_mt`
(hardware-* exact dot, one sweep + int32 vectorizable fast path, DEFAULT for CPU dev) —
specialised-MPP row/head splits (merge-free), C-owned KV slabs (`host.KVSlab`). The composition
gate is in test_gemma4: C-resident forward == Python reference forward BIT-FOR-BIT across
positions. Anchors re-verified each step (fox ' dog' 27 s e2e, Paris).
Multi-token correctness (2026-07-15, REPORT-GEMMA4 §multi-token): ring == f64-mirror argmax on
both tested divergence points (fox pos-9 → 107 gap 2.6; raw France → 7001 — same top-3 order,
logits within ~0.6) and == independent MLX for 4 straight fox tokens. **hpq's "exact/lossless"
claim is RETRACTED** — its f16 forward contradicts the f64 evaluation of its own int8 weights on
both points; hpq = fast cross-check + architecture WHAT, never ground truth. Reference hierarchy:
f64 mirror (exact acts, same weights) = ground truth for the onix model; MLX = independent
architecture check; raw France stays a bad oracle (3-way split).
Next: chat-template multi-token runs on local weights; bf16-checkpoint semantics bar DEFERRED
(no downloads — verify against local weights only: 12B onix f64-mirror = ground truth, 2B onix,
MLX 4-bit independent check); GPU GEMV kernel-level speed (uchar4/simdgroup reads in emu_gemv).
