"""Test for ringkit.emulation.gemma — the Gemma2-2B emulation on the ring, float-free.

Portable (always): f16 decode, ring CORDIC accuracy, geometric RoPE inv_freq, the linear dequant
(power-of-2 activation quant + energy-QSM kernel + 2^(a+s)/z) bit-exact vs an integer reference,
and Gemma BPE tokenizer round-trip. Opportunistic (if the real 2B files are mounted): norms parse,
embed row decode vs a direct f16 reference, and a real q_proj bit-exact — plus, GATED behind
RINGKIT_GEMMA_GEN=1 (slow, ~min/token), the actual autoregressive proof "The capital of France is"
-> " Paris". The compute path is asserted float-free / import-clean by AST.
Run: python3 -m ringkit.tests.test_gemma2"""
import ast
import math
import os
from ringkit.emulation import gemma, tokenizer as tk
from ringkit.core import native as rn

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)

FRAC = gemma.FRAC
ONE = 1 << FRAC

print("== 1. f16 -> Q<frac> decode (integer bit ops) ==")
# 1.0=0x3C00, 2.0=0x4000, 0.5=0x3800, -1.0=0xBC00
check("f16 1.0 -> ONE", gemma._f16_to_fixed(0x3C00) == ONE)
check("f16 2.0 -> 2*ONE", gemma._f16_to_fixed(0x4000) == ONE << 1)
check("f16 0.5 -> ONE/2", gemma._f16_to_fixed(0x3800) == ONE >> 1)
check("f16 -1.0 -> -ONE", gemma._f16_to_fixed(0xBC00) == -ONE)

print("== 2. ring CORDIC cos/sin vs oracle ==")
emax = 0.0
for deg in range(0, 360, 5):
    th = round(math.radians(deg) * ONE)
    c, s = gemma._cordic(th)
    emax = max(emax, abs(c / ONE - math.cos(math.radians(deg))), abs(s / ONE - math.sin(math.radians(deg))))
check(f"CORDIC max err {emax:.2e} < 2e-3", emax < 2e-3)

print("== 3. RoPE tables: geometric inv_freq, cos/sin sane ==")
cos0, sin0 = gemma.rope_tables(0)
check("pos 0 -> cos=1, sin=0 everywhere", all(abs(c - ONE) < 4 for c in cos0) and all(abs(s) < 4 for s in sin0))
cos1, sin1 = gemma.rope_tables(1)
# dim 0 has inv_freq=1 -> angle 1 rad: cos≈0.5403, sin≈0.8415
check("pos 1 dim0 = (cos 1rad, sin 1rad)", abs(cos1[0] / ONE - math.cos(1)) < 2e-3 and abs(sin1[0] / ONE - math.sin(1)) < 2e-3)
# high dims -> inv_freq -> 0 -> angle -> 0 -> cos->1
check("pos 1 high dim -> cos≈1", abs(cos1[-1] - ONE) < 200)

print("== 4. linear dequant (act-quant + kernel + 2^(a+s)/z) bit-exact vs integer reference ==")
import random
rng = random.Random(7)
OF, INF = 5, 40
xbar = bytes(rng.randrange(256) for _ in range(OF * INF))
s_row = [rng.randrange(-6, 4) for _ in range(OF)]
z_row = [rng.randrange(1, 7) for _ in range(OF)]
x = [rng.randrange(-3 * ONE, 3 * ONE) for _ in range(INF)]
ring = gemma.proj((xbar, s_row, z_row, OF, INF), x, FRAC)
# independent integer reference of the SAME math
mx = max(abs(v) for v in x)
a = 0
if (127 << FRAC) >= mx:
    while (127 << (a - 1 + FRAC)) >= mx:
        a -= 1
else:
    while (127 << (a + FRAC)) < mx:
        a += 1
sh = FRAC + a
def q8(v):
    if sh > 0:
        r = 1 << (sh - 1); q = (v + r) >> sh if v >= 0 else -((-v + r) >> sh)
    elif sh == 0:
        q = v
    else:
        q = v << (-sh)
    return max(-127, min(127, q))
xs = [q8(v) for v in x]
ref = []
for r in range(OF):
    dot = sum((xbar[r * INF + i] - 128) * xs[i] for i in range(INF))
    shift = a + s_row[r] + FRAC
    acc = dot << shift if shift >= 0 else dot >> (-shift)
    z = z_row[r] or 1
    ref.append(-((-acc) // z) if acc < 0 else acc // z)
check("proj bit-exact vs integer reference", ring == ref)

print("== 5. Gemma BPE tokenizer round-trip ==")
tp = tk.default_path()
if tp:
    T = tk.GemmaTokenizer(tp)
    for s in ["The capital of France is", "Hello, world!", "2 + 2 = 4"]:
        ids = T.encode(s)
        dec = T.decode(ids).lstrip()
        check(f"roundtrip {s!r} -> {ids[:4]}... -> {dec!r}", dec == s and ids[0] == T.bos)
else:
    print("  (tokenizer.json not mounted — skipped)")

print("== 6. compute path is float-free / import-clean (AST) ==")
for mod in ("gemma", "gemma_weights"):
    src = open(os.path.join(os.path.dirname(__file__), "..", "emulation", mod + ".py")).read()
    t = ast.parse(src)
    floats = [n for n in ast.walk(t) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "float"]
    flit = [n for n in ast.walk(t) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    imps = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for al in n.names: imps.add(al.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module:
            imps.add(n.module.split(".")[0])
    check(f"{mod}.py: no float()/float-literal, no numpy/torch/math",
          not floats and not flit and not (imps & {"numpy", "torch", "math", "scipy", "safetensors"}))

print("== 7. opportunistic: REAL Gemma2-2B weights on the ring ==")
from ringkit.emulation.gemma_weights import default_paths
paths = default_paths()
if paths:
    from ringkit.emulation.gemma_weights import Gemma2Weights
    W = Gemma2Weights(*paths)
    check("norms.bin parsed: 26 layers x 2304, final 2304",
          len(W._norms) == 26 and len(W.norm(0, "pre_attn")) == 2304 and len(W.final_norm()) == 2304)
    # embed row decode matches a direct f16 reference read
    row = W.embed_row(2)
    base = 2 * gemma.G2.hidden * 2
    raw = W._emb[base:base + 8]
    ref0 = [gemma._f16_to_fixed(raw[2 * j] | (raw[2 * j + 1] << 8)) for j in range(4)]
    check("embed_row(bos) decodes vs direct f16 ref", row[:4] == ref0 and len(row) == 2304)
    # real q_proj bit-exact vs integer reference (proj over a real onix tensor)
    xb, s_row, z_row, of, inf = W.lin(0, "q_proj")
    xr = [random.Random(0).randint(-2 * ONE, 2 * ONE) for _ in range(inf)]
    ringq = gemma.proj((xb, s_row, z_row, of, inf), xr, FRAC)[:6]
    mx = max(abs(v) for v in xr); a = 0
    if (127 << FRAC) >= mx:
        while (127 << (a - 1 + FRAC)) >= mx: a -= 1
    else:
        while (127 << (a + FRAC)) < mx: a += 1
    sh = FRAC + a
    xs = [max(-127, min(127, ((v + (1 << (sh - 1))) >> sh) if sh > 0 and v >= 0 else
              (-((-v + (1 << (sh - 1))) >> sh)) if sh > 0 else (v if sh == 0 else v << (-sh)))) for v in xr]
    refq = []
    for r in range(6):
        dot = sum((xb[r * inf + i] - 128) * xs[i] for i in range(inf))
        shift = a + s_row[r] + FRAC
        acc = dot << shift if shift >= 0 else dot >> (-shift)
        z = z_row[r] or 1
        refq.append(-((-acc) // z) if acc < 0 else acc // z)
    check("real q_proj bit-exact vs integer reference", ringq == refq)

    if os.environ.get("RINGKIT_GEMMA_GEN") == "1":
        T = tk.GemmaTokenizer(tk.default_path())
        ids = T.encode("The capital of France is")
        cache = gemma.new_cache(); pos = 0; hn = None
        for tok in ids:
            hn = gemma.forward_token(W, tok, pos, cache); pos += 1
        nt, _ = W.lm_argmax(hn)
        check(f"greedy next token is ' Paris' (id 7127), got {nt} {T.id2piece.get(nt)!r}", nt == 7127)
    else:
        print("  (full generation gated behind RINGKIT_GEMMA_GEN=1 — slow; proven in docs/REPORT-GEMMA2.md)")
    W.close()
else:
    print("  (real gemma2_2b weights not mounted — portable proofs above are permanent)")

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
