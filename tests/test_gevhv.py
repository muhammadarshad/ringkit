"""Test for ringkit.ml.gevhv — the GEVHV operator (bind -> react -> measure).

Ports the gevhv research bed's verification protocol T1-T10 (GEVHV_MATH.md §V) into the
kit's suite: exhaustive wherever the domain allows. Standard math ('*', '%') appears here
ONLY as the labeled external oracle (D9 two-layer rule) — never in the module under test,
which is AST-audited below.
Run: python -m ringkit.tests.test_gevhv"""
import ast
import os
import random
from ringkit.ml import gevhv as gv
from ringkit.ml.attention import ring_distance

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)
def _raises(f):
    try: f(); return False
    except ValueError: return True
    except Exception: return False

random.seed(2026)
H, W = gv.H, gv.W
NSITES = H * W                               # oracle arithmetic (test layer only)
INTERIOR = (H - 2) * (W - 2)

print("== T1 — Theorem A/A1: signedness is invisible mod 256 (exhaustive 65,536) ==")
bad = 0
for a in range(256):
    sa = a - 256 if a > 127 else a
    for b in range(256):
        sb = b - 256 if b > 127 else b
        if ((a * b) & 0xFF) != ((sa * sb) & 0xFF):      # oracle '*'
            bad += 1
check(f"u8-read vs s8-read products agree mod 256 (mismatches = {bad})", bad == 0)

print("== T2 — Theorem B/B1: wrap-through-u32-accumulator GEMM fold, every K ==")
ok2 = True
for K in (1, 255, 256, 66051, 66052, 1000000):
    for mode in ("all255", "random"):
        if mode == "all255":
            wv = [255] * K; xv = [255] * K
        else:
            wv = [random.randint(0, 255) for _ in range(K)]
            xv = [random.randint(0, 255) for _ in range(K)]
        acc32 = 0
        big = 0
        for j in range(K):
            p = wv[j] * xv[j]                            # oracle '*'
            acc32 = (acc32 + p) & 0xFFFFFFFF
            big += p
        if (acc32 & 0xFF) != (big & 0xFF):
            ok2 = False
check("u32-wrapping fold & 0xFF == bigint & 0xFF for every K incl. past overflow", ok2)

print("== T3 — Theorem C: radix-transposed shift-add dot == multiply reference ==")
bad = 0
for wv in range(256):
    for xv in range(256):
        if gv.dot_radix([wv], [xv]) != wv * xv:          # oracle '*'
            bad += 1
check(f"exhaustive u8 x u8 element identity (mismatches = {bad})", bad == 0)
ok3 = True
for k in (3, 64, 255):
    wv = [random.randint(-500, 500) for _ in range(k)]
    xv = [random.randint(0, 255) for _ in range(k)]
    want = sum(wv[j] * xv[j] for j in range(k))          # oracle
    if gv.dot_radix(wv, xv) != want:
        ok3 = False
    if gv.gemv_radix([wv, wv], xv) != [want, want]:
        ok3 = False
check("random rows (signed weights) + gemv_radix rows", ok3)

print("== T4 — Theorem C-bound: u32 exactness sharp at k = 66052 ==")
def u32dot(k):
    acc = 0
    for _ in range(k):
        acc = (acc + 255 * 255) & 0xFFFFFFFF             # oracle
    return acc
true51 = 66051 * 255 * 255
true52 = 66052 * 255 * 255
check("k = 66051 all-255 exact in u32", u32dot(66051) == true51)
check("k = 66052 all-255 overflows u32 (differs from bigint)", u32dot(66052) != true52)

print("== T5 — Theorem D: cdist = min of the two ring subtractions (exhaustive) ==")
bad = 0
for a in range(256):
    for b in range(256):
        ref = min((a - b) % 256, (b - a) % 256)          # oracle '%'
        if ring_distance(a, b) != ref:
            bad += 1
check(f"ring_distance == reference over all 65,536 pairs (mismatches = {bad})", bad == 0)

print("== T6 — Proposition D2: cdist metric axioms ==")
ok6 = True
triples = [(0, 128, 192), (0, 192, 128), (128, 0, 192)]
triples += [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(20000)]
for a, b, c in triples:
    if ring_distance(a, b) != ring_distance(b, a):
        ok6 = False
    if ring_distance(a, c) > ring_distance(a, b) + ring_distance(b, c):
        ok6 = False
    if ring_distance(a, a) != 0:
        ok6 = False
check("symmetry + identity + triangle over adversarial and 20k random triples", ok6)

print("== T7 — Theorem E: bind is a bijection for ALL odd s, all t (exhaustive) ==")
domain = list(range(256))
ok7 = True
for s in range(1, 256, 2):
    base = gv.bind_scalar(domain, s, 0)                  # the module's own map, t = 0
    si_ok = (gv.anti_stride(s) * s) & 0xFF == 1          # oracle '*' checks the inverse
    if not si_ok:
        ok7 = False
    for t in range(256):
        seen = 0
        acc = [False] * 256
        for x in domain:
            y = (base[x] + t) & 0xFF                     # phi_{s,t}(x) = phi_{s,0}(x)+t
            if not acc[y]:
                acc[y] = True
                seen += 1
        if seen != 256:
            ok7 = False
check("all 128 odd s x 256 t: phi is a permutation of Z256 (8.4M checks)", ok7)
ok7b = True
for _ in range(200):
    s = random.randrange(1, 256, 2); t = random.randint(0, 255)
    xs = [random.randint(0, 255) for _ in range(32)]
    if gv.unbind_scalar(gv.bind_scalar(xs, s, t), s, t) != [x & 0xFF for x in xs]:
        ok7b = False
check("unbind(bind(x)) == x on 200 random (s, t) rows", ok7b)
check("even s is refused (zero-divisor cannot bind)",
      all(_raises(lambda s=s: gv.bind_scalar([1], s, 0)) for s in (0, 2, 128)))

print("== T8 — Theorem F: bind absorption, full-manifold bit-exact ==")
def rand_grid():
    return [random.randint(0, 255) for _ in range(NSITES)]
ok8 = True
grids = [rand_grid(), rand_grid(), [0] * NSITES, [128] * NSITES, [255] * NSITES]
for g in grids:
    lut = [random.randint(0, 255) for _ in range(256)]
    s = random.randrange(1, 256, 2); t = random.randint(0, 255)
    unfused = gv.react(gv.bind_scalar(g, s, t), lut)
    fused = gv.react_bound_scalar(g, lut, s, t)
    if unfused != fused:
        ok8 = False
check("rho(phi(g)) == rho'(g) via L' interior + phi pointwise boundary (5 manifolds)", ok8)

print("== T9 — Theorem F2: vector bind via the shared offset field ==")
ok9 = True
v = rand_grid()
c = gv.offset_field(v)
lut = [random.randint(0, 255) for _ in range(256)]
for g in (rand_grid(), rand_grid(), rand_grid()):        # a batch sharing ONE c
    unfused = gv.react(gv.bind_vector(g, v), lut)
    fused = gv.react_bound_vector(g, lut, v, c=c)
    if unfused != fused:
        ok9 = False
check("rho(g+v) == offset-field form, batch of 3 sharing one precomputed c", ok9)

print("== T10 — Theorem G: measure exactness, bounds, batch independence ==")
g0 = [0] * NSITES
q128 = [128] * NSITES
e_int = gv.measure(g0, q128, interior=True)
e_full = gv.measure(g0, q128)
check(f"adversarial interior energy == 13,986*128 = 1,790,208 (got {e_int})",
      e_int == INTERIOR * 128 == 1790208)
check(f"full-grid energy == 14,464*128 = 1,851,392 (got {e_full})",
      e_full == NSITES * 128 == 1851392)
check("both bounds int32-exact (< 2^31)", e_full < (1 << 31) and e_int < (1 << 31))
batch = [rand_grid() for _ in range(3)]
q = rand_grid()
singles = [gv.measure(m, q) for m in batch]
lut = [random.randint(0, 255) for _ in range(256)]
fused_batch = [gv.gevhv_vector(m, lut, q, v, c=c) for m in batch]
ok10 = all(fused_batch[i] == gv.measure(gv.react(gv.bind_vector(batch[i], v), lut), q)
           for i in range(3))
check("batch entries independent; fused operator == composed reference per manifold", ok10)
check("singles are per-manifold (no cross term): re-measure equals first pass",
      [gv.measure(m, q) for m in batch] == singles)

print("== charter audit: the MODULE is multiplier-free and import-clean ==")
src = open(os.path.join(os.path.dirname(__file__), "..", "ml", "gevhv.py")).read()
tree = ast.parse(src)
bad_ops = [type(n.op).__name__ for n in ast.walk(tree)
           if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Div,
                                                             ast.FloorDiv, ast.Pow, ast.Mod))]
check(f"no * / // ** %% in ml/gevhv.py (found {bad_ops})", not bad_ops)
imports = [n.names[0].name for n in ast.walk(tree) if isinstance(n, (ast.Import,))]
imports += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
std = [m for m in imports if m and not m.startswith("ringkit")]
check(f"no standard-math imports (found {std})", not std)

print("RESULT:", "ALL PASS" if not fails else f"FAIL ({len(fails)}): {fails}")
