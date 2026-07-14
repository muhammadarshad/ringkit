"""Test for ringkit.emulation.gemma4 — Gemma4-12B emulation on the ring, float-free.

Gemma4 is NOT a Gemma2 config reskin (verified against the real 12B onix + the canonical MLX
`gemma4_text` reference): no (1+gamma) RMSNorm, sliding/global attention alternation (period 6),
partial RoPE rotation on the 512-dim global heads, per-head Q/K norm BEFORE RoPE, attention_k_eq_v
(global V = raw K projection, no v_proj tensor), gelu_pytorch_tanh gate, per-layer learned residual
scalar, two RoPE thetas. Each departure is exercised below.

Portable (always): gelu-tanh vs float oracle, no-offset RMSNorm vs float oracle, f16/f32 field
decode, layer geometry (_is_global / head-dim / kv-heads / rot-dim per layer), partial-rotation RoPE
(only the rotated span changes; the rest pass through bit-exactly). Compute path asserted
float-free / import-clean by AST.

Opportunistic (real 12B mounted): norms parse (per-head Q/K gammas of length head_dim, 256 local /
512 global), layer_scalars parse, the onix carries NO v_proj on global layers (attention_k_eq_v),
and a real q_proj is bit-exact vs an integer reference.

Assembly proof (GATED behind RINGKIT_GEMMA4_GEN=1, slow ~2 min/token): the full 48-layer forward on
the BOS token reproduces the canonical MLX (`gemma4_text`) per-layer hidden magnitudes within 12%.
This is the check that discharges "different architecture" — a mis-wired sublayer passes every leaf
test above and still throws the per-layer trajectory off (the layer-11 magnitude collapse is the
tell). It deliberately does NOT pin a greedy token: the forward is faithful but Q16 arithmetic drifts
directionally at multi-position (final-hidden cosine vs MLX 0.986@pos0 -> 0.907@pos1), which flips
the argmax only on this instruct model's low-margin raw completions. See docs/REPORT-GEMMA4.md.
Run: python3 -m ringkit.tests.test_gemma4"""
import ast
import math
import os
import random
from ringkit.emulation import gemma4 as g4
from ringkit.emulation.gemma4_weights import _f32_bits_to_fixed
from ringkit.emulation import gemma

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)

FRAC = g4.FRAC
ONE = 1 << FRAC

print("== 1. f16/f32 field decode (integer bit ops) ==")
check("f16 1.0 -> ONE", gemma._f16_to_fixed(0x3C00) == ONE)
# f32 layer-scalar decode: 1.0 exact, and a representable fraction
import struct as _struct
def _f32bits(v):
    return _struct.unpack("<I", _struct.pack("<f", v))[0]
check("f32 1.0 -> ONE", _f32_bits_to_fixed(_f32bits(1.0), FRAC) == ONE)
check("f32 -0.5 -> -ONE/2", _f32_bits_to_fixed(_f32bits(-0.5), FRAC) == -(ONE >> 1))
check("f32 0.052979 decode err < 1e-4",
      abs(_f32_bits_to_fixed(_f32bits(0.052979), FRAC) / ONE - 0.052979) < 1e-4)

print("== 2. gelu_pytorch_tanh vs float oracle ==")
def gelu_tanh(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
emax = max(abs(g4.gelu_tanh_fixed(round(xr * ONE), FRAC) / ONE - gelu_tanh(xr))
           for xr in (-3, -1.5, -0.5, 0, 0.5, 1, 2, 3))
check(f"gelu_tanh max err {emax:.2e} < 2e-3", emax < 2e-3)

print("== 3. Gemma4 RMSNorm applies gamma straight (NO 1+gamma offset) ==")
rng = random.Random(1)
n = 16
x = [rng.uniform(-2, 2) for _ in range(n)]
gm = [rng.uniform(0.5, 1.5) for _ in range(n)]
rms = math.sqrt(sum(v * v for v in x) / n + 1e-6)
ref_nooff = [(v / rms) * g for v, g in zip(x, gm)]          # gamma straight
ref_1plus = [(v / rms) * (1 + g) for v, g in zip(x, gm)]    # the Gemma2 form (must NOT match)
ri = g4.rmsnorm_g4([round(v * ONE) for v in x], [round(g * ONE) for g in gm], FRAC)
err = max(abs(ri[i] / ONE - ref_nooff[i]) for i in range(n))
check(f"rmsnorm_g4 == gamma-straight (err {err:.2e} < 2e-3)", err < 2e-3)
check("rmsnorm_g4 != (1+gamma) form",
      max(abs(ri[i] / ONE - ref_1plus[i]) for i in range(n)) > 0.1)

print("== 4. layer geometry: sliding/global alternation, per-layer dims ==")
check("global layers are {5,11,...,47}",
      [li for li in range(48) if g4._is_global(li)] == list(range(5, 48, 6)))
# local: head_dim 256, 8 KV heads, full rotation 256, group 2 ; global: 512, 1 KV, rot 128, group 16
check("layer_geom(0) local = (False,256,8,256,·,2)",
      g4.layer_geom(0)[:4] == (False, 256, 8, 256) and g4.layer_geom(0)[5] == 2)
check("layer_geom(5) global = (True,512,1,128,·,16)",
      g4.layer_geom(5)[:4] == (True, 512, 1, 128) and g4.layer_geom(5)[5] == 16)
check("local theta_ln < global theta_ln (1e4 vs 1e6)",
      g4.layer_geom(0)[4] < g4.layer_geom(5)[4])

print("== 5. partial-rotation RoPE (global: only 128 of 512 dims rotate) ==")
cos0, sin0 = g4.rope_tables(0, 512, 128, g4.G4.theta_ln_global, FRAC)
check("global rope_tables: 64 pairs, pos0 cos~1 sin~0",
      len(cos0) == 64 and all(abs(c - ONE) < 4 for c in cos0) and all(abs(s) < 4 for s in sin0))
cosp, sinp = g4.rope_tables(3, 512, 128, g4.G4.theta_ln_global, FRAC)
vec = [(i + 1) * ONE for i in range(512)]
out = g4.apply_rope(vec, cosp, sinp, 256, FRAC)
rotated = set(range(0, 64)) | set(range(256, 320))         # the two halves of the 64 rotated pairs
check("dims outside rotated span pass through bit-exactly",
      all(out[i] == vec[i] for i in range(512) if i not in rotated))
check("exactly the rotated span (128 dims) changed",
      all(i in rotated for i in range(512) if out[i] != vec[i]))
# local layer rotates the whole head (rot == head_dim), so pair_off = 128 and all 256 dims move
cl, sl = g4.rope_tables(3, 256, 256, g4.G4.theta_ln_local, FRAC)
check("local RoPE produces head_dim/2 = 128 pairs", len(cl) == 128)

print("== 5b. proj is EXACT under activation OUTLIERS (digit decomposition, zero act loss) ==")
# Late-layer Gemma4 activations reach max|x|/rms ≈ 60 (L41). A single-pass power-of-2 int8 grid
# pins its quantum to the spike and crushes the bulk of the vector onto ~2 levels, destroying the
# projection DIRECTION (K collapsed to cos ≈ 0.55 -> wrong argmax on the fox prompt). The ring does
# NOT quantize activations: the exact digit decomposition must equal the exact integer dot
# BIT-FOR-BIT even here; the single-pass truncating control must fail.
import random as _rnd
_r = _rnd.Random(41)
OF_, INF_ = 8, 256
SPIKES_ = (7, 100, 200)
_xb = bytearray(_r.randrange(256) for _ in range(OF_ * INF_))
for r_ in range(OF_):                     # outlier channels get ~zero weight (as in the real model:
    for i_ in SPIKES_:                    # the OUTPUT direction is carried by the bulk, which the
        _xb[r_ * INF_ + i_] = 128         # spike-pinned quant grid crushes)
xbar_ = bytes(_xb)
s0_ = [0] * OF_; z1_ = [1] * OF_
xo = [_r.randint(-ONE >> 2, ONE >> 2) for _ in range(INF_)]    # bulk ±0.25
for i_ in SPIKES_:
    xo[i_] = 40 * ONE                                          # spikes: quantum > bulk max
ring_o = gemma.proj((xbar_, s0_, z1_, OF_, INF_), xo, FRAC)
exact_o = [sum((xbar_[r_ * INF_ + i] - 128) * xo[i] for i in range(INF_))
           for r_ in range(OF_)]                               # labeled integer oracle (exact dot)
check("proj == exact integer dot BIT-FOR-BIT under 60x outliers (s=0, z=1)",
      ring_o == exact_o)
def _cos(u, v):
    du = math.sqrt(sum(a * a for a in u)); dv = math.sqrt(sum(b * b for b in v))
    return sum(a * b for a, b in zip(u, v)) / (du * dv or 1.0)
# single-pass control: quantize once at the spike-pinned scale -> direction must break
_mx = max(abs(v) for v in xo); _a = 0
while (127 << (_a + FRAC)) < _mx: _a += 1
_sh = FRAC + _a
_xs = [max(-127, min(127, ((v + (1 << (_sh - 1))) >> _sh) if v >= 0
           else -((-v + (1 << (_sh - 1))) >> _sh))) for v in xo]
_single = [sum((xbar_[r_ * INF_ + i] - 128) * _xs[i] for i in range(INF_)) for r_ in range(OF_)]
check("CONTROL: single-pass act-quant breaks direction (cos < 0.9)",
      _cos(_single, exact_o) < 0.9)

print("== 6. compute path is float-free / import-clean (AST) ==")
for mod in ("gemma4", "gemma4_weights"):
    src = open(os.path.join(os.path.dirname(__file__), "..", "emulation", mod + ".py")).read()
    t = ast.parse(src)
    floats = [n for n in ast.walk(t) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == "float"]
    flit = [n for n in ast.walk(t) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    imps = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for al in n.names:
                imps.add(al.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module:
            imps.add(n.module.split(".")[0])
    check(f"{mod}.py: no float()/float-literal, no numpy/torch/math",
          not floats and not flit and not (imps & {"numpy", "torch", "math", "scipy", "safetensors"}))

print("== 7. opportunistic: REAL Gemma4-12B weights ==")
from ringkit.emulation.gemma4_weights import default_paths, Gemma4Weights
from ringkit.emulation import onix as onix_mod
paths = default_paths()
if paths:
    onix_p = paths[0]
    W = Gemma4Weights(*paths)
    check("norms.bin: 48 layers x 3840, final 3840",
          len(W._norms) == 48 and len(W.norm(0, "pre_attn")) == 3840 and len(W.final_norm()) == 3840)
    check("per-head Q/K norm length = head_dim (256 local, 512 global)",
          len(W.norm(0, "q_norm")) == 256 and len(W.norm(0, "k_norm")) == 256 and
          len(W.norm(5, "q_norm")) == 512 and len(W.norm(5, "k_norm")) == 512)
    check("layer_scalars: 48 values decoded", len(W._scalars) == 48)
    # onix carries NO v_proj on global layers (attention_k_eq_v); DOES on local
    _, ents = onix_mod.index(onix_p)
    has_v = lambda li: f"model.layers.{li}.self_attn.v_proj" in ents
    check("onix: local layer HAS v_proj, global layers do NOT (attention_k_eq_v)",
          has_v(0) and not has_v(5) and not has_v(11) and not has_v(47))
    # real q_proj bit-exact vs integer reference (proj over a real onix tensor)
    xb, s_row, z_row, of, inf = W.lin(0, "q_proj")
    xr = [random.Random(0).randint(-2 * ONE, 2 * ONE) for _ in range(inf)]
    ringq = gemma.proj((xb, s_row, z_row, of, inf), xr, FRAC)[:6]
    # THE reference: the EXACT integer dot (proj's digit decomposition has zero activation loss)
    refq = []
    for r in range(6):
        D = sum((xb[r * inf + i] - 128) * xr[i] for i in range(inf))
        t = D << s_row[r] if s_row[r] >= 0 else D >> (-s_row[r])
        z = z_row[r] or 1
        refq.append(-((-t) // z) if t < 0 else t // z)
    check("real q_proj bit-exact vs the EXACT integer dot", ringq == refq)

    # C-resident forward (activations in C buffers, KV slabs) must equal the Python-list
    # reference forward BIT-FOR-BIT across positions — the composition gate for every block
    # kernel (gemv/rmsnorm/gelu/rope/attention/residual/scalar/embed).
    from ringkit.kernels.mprc.gemma import host as kh
    if kh.available():
        slab_cache = g4.new_cache()
        list_cache = [{"k": [[] for _ in range(g4.layer_geom(li)[2])],
                       "v": [[] for _ in range(g4.layer_geom(li)[2])]}
                      for li in range(g4.G4.layers)]
        ok = True
        for pos, tok in enumerate([2, 818]):
            hc = g4.forward_token(W, tok, pos, slab_cache, FRAC)
            hp = g4.forward_token(W, tok, pos, list_cache, FRAC)
            ok = ok and hc == hp
        check("C-resident forward == Python reference forward BIT-FOR-BIT (2 positions)", ok)

    if os.environ.get("RINGKIT_GEMMA4_GEN") == "1":
        # --- assembly proof: per-layer hidden magnitude vs canonical MLX (gemma4_text, f16 4-bit).
        # Oracle = |h|.mean() for BOS (token 2) at position 0: [embed, layer0, layer1, ..., layer47].
        # Ring is the int8-onix path (vs MLX 4-bit) so allow 12% relative + a 0.02 absolute floor for
        # the tiny post-global magnitudes. A mis-wired sublayer breaks this trajectory (esp. the
        # layer-11 collapse to 0.15 and every FULL-attention layer 5/11/.../47).
        from ringkit.core import native as rn
        MLX = [0.3044, 0.358, 0.393, 0.427, 0.705, 1.009, 1.118, 1.093, 1.195, 1.350, 0.864,
               1.059, 0.151, 0.244, 0.307, 0.334, 0.303, 0.307, 0.410, 0.473, 0.645, 0.634,
               0.657, 0.730, 0.854, 0.916, 1.027, 0.993, 1.149, 1.118, 1.224, 1.335, 1.263,
               1.262, 1.227, 1.270, 1.441, 1.556, 1.352, 1.161, 1.075, 1.084, 1.105, 1.297,
               1.382, 1.356, 1.553, 1.615, 0.129]   # 1 embed + 48 layer outputs = 49 entries
        def mabs(v):
            return (sum(-a if a < 0 else a for a in v) / len(v)) / ONE
        cache = g4.new_cache()
        h = [rn.mul(e, W.embed_scale()) >> FRAC for e in W.embed_row(2)]
        worst = ("embed", abs(mabs(h) - MLX[0]) / MLX[0])
        for li in range(48):
            h = g4.layer_forward(W, li, h, cache[li], 0, FRAC)
            rel = abs(mabs(h) - MLX[li + 1]) / (MLX[li + 1] + 0.02)
            if rel > worst[1]:
                worst = (f"layer{li}", rel)
        check(f"BOS per-layer |h| tracks MLX within 12% (worst {worst[0]}={worst[1]:.1%})",
              worst[1] < 0.12)
        # --- mirror-verified greedy pins (docs/REPORT-GEMMA4.md §multi-token): the f64 mirror
        # (EXACT activations, float64, the SAME int8 onix weights — ground truth for this model)
        # fixes these decisions with gaps >= 2.4 logits, and the ring must reproduce them:
        # fox pos-8 -> 4799 ' dog' (= MLX), pos-9 -> 107 (= MLX). hpq's f16 forward disagrees
        # with the f64 evaluation of its own weights on pos-9 and is NOT ground truth.
        cache = g4.new_cache()
        hn = None
        for pos, tok in enumerate([2, 818, 3823, 8864, 37423, 38167, 1024, 506, 31770]):
            hn = g4.forward_token(W, tok, pos, cache, FRAC)
        nt1, _ = W.lm_argmax(hn)
        hn = g4.forward_token(W, nt1, 9, cache, FRAC)
        nt2, _ = W.lm_argmax(hn)
        check(f"fox greedy pos8/pos9 == f64-mirror (4799, 107): got {nt1}, {nt2}",
              nt1 == 4799 and nt2 == 107)
    else:
        print("  (full proofs gated behind RINGKIT_GEMMA4_GEN=1: BOS per-layer magnitude vs MLX +")
        print("   the f64-mirror-verified greedy pins [fox -> ' dog', 107]. Ring == f64 mirror on")
        print("   every tested decision; hpq f16 is NOT ground truth — see docs/REPORT-GEMMA4.md)")
    W.close()
else:
    print("  (real gemma4_12b weights not mounted — portable proofs above are permanent)")

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
