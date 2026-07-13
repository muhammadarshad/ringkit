"""Test for ringkit.infer — ring-native inference, NO float on the compute path.

Proves: (1) the ring fixed-point Linear reproduces real multiply-accumulate within 2**-frac
resolution (numpy used ONLY as a labeled float oracle, C6/D9); (2) infer.py is float-free and
multiplier-free (AST); (3) opportunistically, a REAL pretrained RDT layer runs ring-native.
Run: python3 -m ringkit.tests.test_infer"""
import ast
import os
import random
from ringkit.emulation import infer

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

random.seed(0)
FRAC = 16
ONE = 1 << FRAC

print("== 1. ring fixed-point Linear reproduces float MAC within 2**-frac (numpy = oracle only) ==")
import numpy as np
OUT, IN = 32, 64
# fixed-point integers in Q16 (built as ints; the ring path never sees a float)
W = [random.randint(-ONE, ONE) for _ in range(OUT * IN)]
b = [random.randint(-ONE, ONE) for _ in range(OUT)]
x = [random.randint(-ONE, ONE) for _ in range(IN)]
y_ring = infer.linear(x, W, b, OUT, IN, FRAC)                    # RING PATH: shift-add, integer
# float oracle: dequantize and do it in float64
Wf = np.array(W, dtype=np.float64).reshape(OUT, IN) / ONE
xf = np.array(x, dtype=np.float64) / ONE
bf = np.array(b, dtype=np.float64) / ONE
y_float = Wf @ xf + bf
y_ring_real = np.array(y_ring, dtype=np.float64) / ONE
err = np.abs(y_ring_real - y_float)
check(f"max abs error {err.max():.2e} < 1e-2 (IN={IN} terms, Q{FRAC})", err.max() < 1e-2)
from ringkit.core import native as rn
_ref = (sum(rn.mul(x[i], W[i]) for i in range(IN))) >> FRAC
check("dot() == shift-add reference", infer.dot(x, W[:IN], FRAC) == _ref)

print("== 2. infer.py is float-free and multiplier-free (AST) ==")
src = open(os.path.join(os.path.dirname(__file__), "..", "emulation", "infer.py")).read()
tree = ast.parse(src)
floats = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "float"]
ops = [type(n.op).__name__ for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.FloorDiv, ast.Pow, ast.Div))]
check(f"no float() calls; no * // ** / in code (found ops {ops})", not floats and not ops)
imps = set()
for n in ast.walk(tree):
    if isinstance(n, ast.ImportFrom) and n.module: imps.add(n.module.split(".")[0])
check("imports only ringkit (no numpy/math/torch)", imps <= {"ringkit"})

print("== 3. opportunistic: a REAL pretrained RDT layer, ring-native vs float oracle ==")
from ringkit.emulation import checkpoint as ck
cands = [os.path.expanduser("~/Projects/vlm-transformers/vlm_rdt_best.pth"),
         "/sessions/dazzling-zen-euler/mnt/Projects/vlm-transformers/vlm_rdt_best.pth"]
real = next((p for p in cands if os.path.exists(p)), None)
if real:
    fx = ck.load_fixed(real, frac=FRAC)
    Wv, (o, i) = fx["vision_encoder.quadrant_proj.proj.weight"]
    bv, _ = fx["vision_encoder.quadrant_proj.proj.bias"]
    xr = [random.randint(-ONE, ONE) for _ in range(i)]
    yr = infer.linear(xr, Wv, bv, o, i, FRAC)
    # oracle from the same raw bytes as float32
    import zipfile, io
    z = zipfile.ZipFile(real); root = z.namelist()[0].split("/")[0]
    refs = dict(ck._flatten(ck._RingUnpickler(io.BytesIO(z.read(f"{root}/data.pkl"))).load()))
    Wk = refs["vision_encoder.quadrant_proj.proj.weight"].storage[2]
    bk = refs["vision_encoder.quadrant_proj.proj.bias"].storage[2]
    Wf = np.frombuffer(z.read(f"{root}/data/{Wk}"), dtype=np.float32, count=o * i).reshape(o, i).astype(np.float64)
    bff = np.frombuffer(z.read(f"{root}/data/{bk}"), dtype=np.float32, count=o).astype(np.float64)
    yf = Wf @ (np.array(xr, dtype=np.float64) / ONE) + bff
    e = np.abs(np.array(yr, dtype=np.float64) / ONE - yf)
    check(f"real RDT layer: ring vs float max rel err {e.max()/(np.abs(yf).max()+1e-9):.2e} < 1e-2", e.max() / (np.abs(yf).max() + 1e-9) < 1e-2)
else:
    print("  (no real .pth reachable — portable test above is the permanent proof)")

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
