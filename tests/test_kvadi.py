"""Test for ringkit.ml.kvadi — the ring-native ADI KV element, N-DIMENSIONAL (NO Euclidean).

The Euclidean polar magnitude (a^2 + b^2 = c^2) is an MPRC anti-pattern (Prime Directive) and is
gone. The ADI (integral, differential) element is EXACT and REVERSIBLE for ANY dimension N — not
pinned to a 2-D (x, y) pair — and the codec imports no standard math.
Run: python3 -m ringkit.tests.test_kvadi"""
import ast
import os
import random
from ringkit.ml import kvadi as ka

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)
def raises(exc, f):
    try: f(); return False
    except exc: return True
    except Exception: return False

random.seed(1)

print("== 1. THE BAR: exact & reversible for ANY dimension N (not fixed to 2-D) ==")
bad = 0; tried = 0
for dim in (1, 2, 3, 4, 5, 8, 16, 113, 128):
    for _ in range(150):
        row = [random.randint(0, 255) for _ in range(dim)]
        tried += 1
        if ka.decode(*ka.encode(row)) != row:
            bad += 1
check(f"encode->decode bit-exact over dims 1..128 ({tried} rows, mismatches = {bad})", bad == 0)

print("== 2. the general N-D vector the SPECIALIZED (2-D) primitive could not do ==")
v = [10, 200, 3, 99]                      # not ADI-consistent -> old compress() RAISED on this
check("arbitrary 4-vector round-trips exactly", ka.decode(*ka.encode(v)) == v)

# the worked 4-D vector (36, 25, 22, 19)
w = [36, 25, 22, 19]
lw, dw = ka.encode(w)
check("4-D (36,25,22,19): lead=36, delta=[245,253,253]", lw == 36 and dw == [245, 253, 253])
check("4-D (36,25,22,19): accumulation Λ = [36,61,83,102]", ka.accumulation(w) == [36, 61, 83, 102])
check("4-D (36,25,22,19): decode is EXACT", ka.decode(lw, dw) == w)

print("== 3. the worked 2-D example is just N=2 ==")
lead, delta = ka.encode([30, 30])
check("(30,30) round-trips", ka.decode_pair(lead, delta) == (30, 30))
check("accumulation total (Lambda) of (30,30) is 60", ka.accumulation([30, 30])[-1] == 60)
check("differential (delta) of (30,30) is [0]", ka.differential([30, 30]) == [0])

print("== 4. the Euclidean anti-pattern is GONE (Prime Directive) ==")
src = open(os.path.join(os.path.dirname(__file__), "..", "ml", "kvadi.py")).read()
tree = ast.parse(src)
used = set()
for n in ast.walk(tree):
    if isinstance(n, ast.Attribute): used.add(n.attr)
    elif isinstance(n, ast.Name): used.add(n.id)
euclid = {"isqrt", "qsm", "sqrt", "ARCTAN2", "arctan2", "atan2", "SIN", "COS", "_arch"}
hits = sorted(used & euclid)
check(f"no Euclidean/approx calls in the CODE (found {hits})", not hits)
ops = [type(n.op).__name__ for n in ast.walk(tree)
       if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.FloorDiv, ast.Pow, ast.Div))]
check(f"multiplier-free (no * // ** /): {ops}", ops == [])
check("no standard-math imports", not any(m in src for m in ("import numpy", "import math", "from math")))

print("== 5. C KERNEL (D9): batch encode/decode == Python reference, BIT-FOR-BIT ==")
from ringkit.kernels.mprc.kv import host as kvh
rows = [[random.randint(0, 255) for _ in range(d)] for d in (2, 4, 4, 8, 16)]  # equal-length groups
# group by dim (the batch API needs equal length); test dim=4 batch of 3
batch = [[random.randint(0, 255) for _ in range(4)] for _ in range(64)]
leads, deltas = kvh.adi_encode_rows(batch)
ref = [ka.encode(r) for r in batch]
check("C/host batch encode == per-row ka.encode (bit-for-bit)",
      leads == [lead for lead, _ in ref] and deltas == [list(d) for _, d in ref])
check("C/host batch decode recovers the rows exactly", kvh.adi_decode_rows(leads, deltas) == batch)
print(f"    kernel path: {'C' if kvh.adi_available() else 'python fallback'} (both bit-for-bit by the D9 gate)")

print("== 6. guards + honesty ==")
check("empty row -> ValueError", raises(ValueError, lambda: ka.encode([])))
check("ships the EXACT element only, no fabricated quantizer", not hasattr(ka, "quantize_element"))

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
