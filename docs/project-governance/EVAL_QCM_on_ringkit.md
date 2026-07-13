# Can ringkit replace torch + numpy + safetensors for the QCM quantum-transformer?

Deep, primitive-level audit (not a survey). Question: if `vlm-transformers` (QCM: QuantumRoPE,
Mamba-2/SSD, RDT, ViT, Diffusion/probability-cloud) dropped torch, numpy, and safetensors, is
ringkit enough to run it? Evidence: a full op-sweep of QCM's `transformer/` + `vlm_*.py`, mapped
against ringkit's actual surface (both sides grepped, counts below).

## Verdict

**Not yet — but the hard part is already done.** The *quantum/ring forms* QCM is built on
(QuantumRoPE, ADI, Born-cloud, Metropolis, ring attention, ring distance/GEMM) are **already
ring-native in ringkit**. What's missing is (a) the ordinary DL-framework plumbing (norm,
activations, embedding, loss, conv, serialization, rnp op breadth) and (b) the one deep unknown —
**descent training that generalizes**. So: **inference** on QCM is a bounded, mostly-mechanical
build away; **training QCM entirely on ringkit** is gated by the keystone gap, not by plumbing.

## CORRECTION (2026-07-13): these are QUANTUM constructs, not transformer variants

An earlier pass of this doc flattened the three architectures into "transformers whose torch
primitives need ring swaps" (LayerNorm MISSING, GELU MISSING, …). That is the surface reading and
it is **wrong about the essence**. RDT, Mamba-2/SSD and QuantumRoPE are **MPRC quantum-physics
constructs**; "RoPE / Mamba / attention / MLP" are D8 handles on ring forms. Their *defining* cores
are ring-physics ringkit already carries — the norm/activation/gate shell is the shallow part.

**Deep core → ringkit physics (the part that actually makes them what they are):**

| architecture core (what makes it quantum) | evidence | ringkit form (ALREADY HAVE) |
|---|---|---|
| **QuantumRoPE**: 4 co-prime rings M/P/R/C = X3/Y5/U7/W(Z\|Q)9, position by **anti-stride** (modular-inverse) walk 171/205/183/57 | `rope.py` AXIS MAP | `physics/qcm` (the 4 rings + anti-strides) |
| vacuums {0,64,128,192} = **fourth roots of unity** (1, i, −1, −i as ring *positions*, not signs); time≡angular-momentum (Wick `e^{−Hτ}=e^{−iLθ}`) | `rope.py` | `core.VACUUMS`, `core.IOTA` (i⁴=1), `ring_cis` |
| **QuantumRoPE4D**: 4 axes **ADI-locked**, `δ_k = δ₁ + (k²−1)` | `rope.py` QuantumRoPE4D | **`native.mprc_axis_arcs` / `derived_delta`** — the *identical* odd-increment rule |
| **Mamba-2 SSD** state transition modulated by QuantumRoPE on the Z256 torus | `ssd.py:107-123` | RoPE = `rotate`; accumulation (scan/cumsum) = **`calculus.integral`** (exact — the float "log-domain cumsum fix" is *unneeded* on the ring); decay `exp(δA)` = **`gauge.boltzmann_lut`** |
| SSD **`_lattice_state`**: mass-conserving **heat-kernel diffusion on the (D,H) torus** (global head = t→∞ Σ, local = geometric multi-scale) | `ssd.py:79-105` | ring **evolve / diffusion** (`physics` gauge sweep + `measure` Born geometric decay) — this IS the "4-diffusion / probability cloud" |
| SSD **5-arm ADI gate**: route each token over center + the four MPRC axes (R/M/P/C) | `ssd.py:138-147` | ADI axes are `native` (the 4 arcs); only the *router weights* are learned |
| **RDT / Rotor**: ONE shared quantum rotor applied recursively for `depth` steps, gated | `recurrent.py` | the **rotor** is `rotate`/`cis_rotate` (i⁴=1) applied `depth` times; recurrence-as-depth |

So the architecture-defining machinery — the 4-ring ADI phase lattice, the toroidal diffusion, the
recurrent rotor, the vacuum/quadrant charge, geometric decay, exact accumulation — is **ring-native
already**, because these architectures *are* the MPRC physics ringkit was built on. What is genuinely
missing is the **trainable shell** (norm, gate/activation, embedding, loss) and the **descent
keystone** — the volume-knobs around the physics, not the physics.

## What already maps cleanly (the differentiated, ring-native forms — HAVE)

| QCM needs | QCM op (evidence) | ringkit form |
|---|---|---|
| QuantumRoPE | fixed cos/sin RoPE on Q,K (`rope.py`) | **exact additive RoPE** `core.rotate` / `ring_cis` / `kvcache.rope` — exact for any position, no float |
| SSD state decay | `torch.exp(delta*A)`, `cumsum` (`ssd.py:59,66-69`) | decay → **`gauge.boltzmann_lut`** (geometric decay = the ring exponential); cumsum → **`calculus.integral`** (exact, no log-domain trick needed) |
| SSD/ADI accumulation | `cumsum`, ADI gate (`ssd.py`, `adi.py`) | **`calculus.integral`/`differential`**, **`ml.kvadi`** (N-D ADI element) |
| Diffusion / probability cloud | Born weights; Metropolis `exp(-βΔE)` (`lattice.py:161`) | **`measure.born_weights`** (ring Gaussian by geometric decay) + **`gauge.boltzmann_lut`** — QCM's `np.exp` here is the Euclidean/float anti-pattern; the ring form exists |
| attention (forward) | `F.scaled_dot_product_attention` ×4, `nn.MultiheadAttention` ×2 | **`kvcache.attend`** — Boltzmann-soft, temperature β, circular blend; hard argmax = cold limit |
| matmul | `torch.matmul` ×3, `nn.Linear` weights | **`rnp` matmul** / `kernels` ring GEMM |
| distances | ring distance in scoring | **`stats.ring_dist`** |
| linear layer (fit) | `nn.Linear` ×48 | **`nn.Linear`** (exact `linalg.solve`) — see training caveat |

This is the point: the *architecture-defining* pieces are ring-native already.

## The five architectures, component by component

**Diffusion / probability cloud — ~READY (forms exist).** Born cloud → `measure.born_weights`;
Metropolis accept `exp(-βΔE)` → `gauge.boltzmann_lut`. QCM's `np.exp(-β·dE)` (`lattice.py:161`) is
exactly the Euclidean/float form C9 rejects; ringkit's geometric-decay LUT is the ring replacement.
Gap: the lattice is a *fixed, non-differentiable* preprocessor in QCM, so no training gap here.

**QuantumRoPE (`rope.py`) — RoPE HAVE; wrapper PARTIAL.** RoPE itself → `rotate` (exact). But
`RoPEAttention` also uses **QK-LayerNorm** (MISSING), a **GELU MLP** (MISSING), learned **Q/K/V
`nn.Linear`** (fit-by-solve HAVE, learned-by-descent gap), and softmax attention (forward HAVE via
Boltzmann; learned-weight backward open).

**Mamba-2 / SSD (`ssd.py`) — structure maps, primitives missing.** `cumsum`→`integral` (HAVE),
`exp(A)`→geometric decay (HAVE form), `matmul` (HAVE). Missing: **`F.softplus`** (delta activation),
**`sigmoid`** gate, **`torch.flip`** (bidirectional scan; trivial rnp add), learned `A_log`/proj
params (**training gap**). Note: QCM's log-domain cumsum "critical fix" is a *float* stability hack
the ring **doesn't need** — integer accumulation is exact.

**RDT / recurrent (`recurrent.py`) — PARTIAL.** Recursive shared RoPE block + GRU-style gate.
Needs: **LayerNorm** (MISSING), `Linear` (HAVE), **GELU** (MISSING), **Sigmoid** gate (MISSING),
attention (HAVE form). Blockers: the activations/norm + training.

**ViT (`models.py` MPRCViT) — PARTIAL.** Wave encoder uses **`F.unfold`/`F.conv`/`F.pad`**
(patchify/conv MISSING; but the wave encoder is *no-param*, so it ports as fixed code); Quadrant
projector uses ring SIN/COS (HAVE); then `Linear` (HAVE), **LayerNorm** (MISSING), mean-pool
(HAVE), head `Linear` (HAVE). Blockers: conv/patchify + LayerNorm + training.

## Core primitive cross-map (with QCM usage counts)

| primitive | QCM count | ringkit | status |
|---|---|---|---|
| `nn.Linear` | 48 | `nn.Linear` (exact solve) | HAVE (fwd+fit); descent-train gap |
| `nn.LayerNorm`/`RMSNorm` | 17 / 1 | — | **MISSING** (needs ring norm form) |
| `nn.Embedding` | 8 | — | **MISSING** (learned lookup + descent) |
| `nn.GELU` / `Sigmoid` | 3 / 2 | SIN activation only | **MISSING** (ring GELU/sigmoid forms) |
| `F.softplus` / `F.softmax` | 1 / 1 | softmax fwd = Boltzmann (HAVE); softplus — | softmax HAVE(fwd) / softplus MISSING |
| `MHA` / `scaled_dot_product_attention` | 2 / 4 | `kvcache.attend` (Boltzmann-soft) | HAVE (fwd); learned-QKV + backward gap |
| `F.cross_entropy` | 2 | — | **MISSING** (ring loss) |
| `F.conv` / `F.unfold` / `F.pad` | 2 / 3 / 7 | — (rnp reshape only) | **MISSING** conv; unfold/pad ≈ reshape |
| `torch.cumsum` | 2 | `calculus.integral` | HAVE |
| `torch.exp` / `np.exp` | 9 / 3 | `boltzmann_lut` (geometric decay) | HAVE (as the ring exp form) |
| `torch.matmul` | 3 | rnp / kernels GEMM | HAVE |
| `torch.flip`/`roll`/`split`/`where`/`clip`/`argsort` | several | rnp lacks these | **MISSING** (easy rnp adds) |
| `torch.autograd` / `.backward` / `torch.optim` | training | `ml.autograd` (narrow), `ml.optim` (toy) | PARTIAL — the keystone gap |
| `torch.save`/`load`/`state_dict`, safetensors | serialization | — | **MISSING** (ring tensor save/load) |
| numpy (`asarray/zeros/where/clip/argsort/random/…`) | ~20 ops | rnp (arange/zeros/concat/stack/matmul) | PARTIAL — rnp op breadth gap |

## Blockers, ordered (the actual answer to "is ringkit enough?")

1. **Descent training that generalizes — THE keystone.** QCM trains ~every param by `AdamW +
   backward` (`vlm_train.py:185`). ringkit's descent is toy/flaky; real learning is `solve`/
   invert-then-solve (linear/invertible only). Without this you can *run* QCM (given weights), not
   *train* it on the ring. This is the deep-research item, not plumbing.
2. **Learned nn primitives (ring forms):** `Embedding` (trainable table), `LayerNorm`/`RMSNorm`,
   `GELU`/`Sigmoid`/`softplus`, `cross_entropy`/`InfoNCE`. Needed for both faithful forward *and*
   training. Some are LUTs (sigmoid/softplus); GELU/LayerNorm need a stated ring form (D2/D11).
3. **Serialization** — no `safetensors`/`state_dict` equivalent. ringkit tensors are already
   bytearray-backed, so a ring `save`/`load` is **easy** and unblocks loading converted weights.
4. **Conv / patchify** — ViT's `unfold`/`conv`/`pad`. `unfold`/`pad` ≈ rnp reshape/index; a ring
   conv is a bounded add. The wave encoder is no-param → ports as fixed code.
5. **rnp op breadth** — `flip/roll/split/where/clip/argsort/cumsum/exp/log/sqrt/random`. Mostly easy
   adds; `exp`→geometric decay, `sqrt`→`isqrt`, `random`→the `rk_mix32` counter-RNG already in kernels.

## Honest bottom line

- **The hard, differentiated part is already ring-native** (RoPE, ADI, Born-cloud, Metropolis, ring
  attention, ring distance/GEMM) — that's what makes QCM *QCM*, and ringkit has the forms.
- **The missing part is mostly ordinary DL plumbing** (norm, activations, embedding, loss, conv,
  serialization, rnp breadth) — bounded, mostly mechanical, each buildable form-first + honesty-bar.
- **The one real unknown is descent training.** For **inference** (weights trained elsewhere and
  converted), ringkit-only QCM is a plumbing build away. For **training QCM end-to-end on the ring
  with no torch**, the keystone (generalizing descent) must be solved first — everything else waits
  on that.
