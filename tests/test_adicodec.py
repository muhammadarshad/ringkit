"""Test for ringkit.ml.adicodec — the LOSSLESS ADI cube compressor.

THE BAR: decode_cube(encode_cube(x)) == x BIT-FOR-BIT for any ring cube (it is a bijective codec,
NOT a quantizer — C9). Plus: it actually compresses (real bytes < raw), multiplier-free, no
standard math. Run: python3 -m ringkit.tests.test_adicodec"""
import ast
import os
import random
from ringkit.ml import adicodec as ac

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

random.seed(3)

print("== 1. THE BAR: lossless (bit-for-bit) over cubes of every shape ==")
bad = 0; tried = 0
for (nz, R, C) in [(1, 1, 1), (1, 2, 3), (9, 16, 6), (4, 7, 5), (2, 1, 8), (16, 32, 6),
                   (3, 16, 1), (5, 5, 5), (1, 128, 6)]:
    for _ in range(30):
        cube = [[[random.randint(0, 255) for _ in range(C)] for _ in range(R)] for _ in range(nz)]
        tried += 1
        if ac.decode_cube(ac.encode_cube(cube)) != cube:
            bad += 1
check(f"decode∘encode == identity ({tried} cubes, mismatches = {bad})", bad == 0)

print("== 2. it compresses — and HARDER when columns are redundant (still exact) ==")
# a Laplacian-cube-shaped case: two coarse scale-columns identically zero, bands near-identical
cube = []
for z in range(9):
    base = [159, 124, 137, 3, 0, 0]
    rows = []
    for b in range(16):
        rows.append([(v + random.randint(-2, 2)) & 0xFF if v else 0 for v in base])
    cube.append(rows)
raw, coded = ac.ratio(cube)
check(f"redundant cube: {raw}B -> {coded}B lossless = {raw/coded:.2f}x (>1)", coded < raw
      and ac.decode_cube(ac.encode_cube(cube)) == cube)
# incompressible control: random cube should NOT expand much (codec overhead bounded)
rnd = [[[random.randint(0, 255) for _ in range(6)] for _ in range(16)] for _ in range(9)]
rr, rc = ac.ratio(rnd)
check(f"random cube stays ~1x, no blow-up ({rr}B -> {rc}B) and lossless",
      rc <= rr + rr // 4 and ac.decode_cube(ac.encode_cube(rnd)) == rnd)

print("== 3. constant-column elision: an all-constant cube is near-free ==")
flat = [[[7 for _ in range(6)] for _ in range(16)] for _ in range(9)]
raw, coded = ac.ratio(flat)
check(f"constant cube {raw}B -> {coded}B (columns elided) and exact",
      coded < 20 and ac.decode_cube(ac.encode_cube(flat)) == flat)

print("== 4. multiplier-free / no standard math (AST) ==")
src = open(os.path.join(os.path.dirname(__file__), "..", "ml", "adicodec.py")).read()
tree = ast.parse(src)
ops = [type(n.op).__name__ for n in ast.walk(tree)
       if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.FloorDiv, ast.Pow, ast.Div))]
check(f"no * // ** / : {ops}", ops == [])
check("no numpy/math imports", not any(m in src for m in ("import numpy", "import math", "from math")))

print("== 5. honesty: exact codec, no lossy knob ==")
check("no quantize/tolerance parameter on the codec",
      not any(hasattr(ac, n) for n in ("quantize", "quantize_cube", "tolerance", "lossy")))
# self-inverse is the guarantee, restated on a fixed vector
v = [[[10, 200, 3, 99, 0, 255]]]
check("adversarial values (wrap-heavy) round-trip exactly", ac.decode_cube(ac.encode_cube(v)) == v)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
