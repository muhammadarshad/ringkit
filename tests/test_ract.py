"""Test for ringkit.ract — ring fixed-point activations, NO float on the compute path.

Each activation is checked against a numpy float ORACLE (labeled, verification only, C6/D9) across a
range of inputs; the ring path itself is integer/shift-add only. AST-verified float-free + multiplier-free.
Run: python3 -m ringkit.tests.test_ract"""
import ast
import os
import math
import numpy as np
from ringkit.emulation import ract

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

FRAC = 16
ONE = 1 << FRAC
def q(v): return int(round(v * ONE))          # oracle-side helper (test may use float)
def dq(v): return v / ONE

print("== 1. exp: ring Taylor vs math.exp oracle ==")
errs = []
for xf in [-6, -3, -1, -0.5, 0, 0.5, 1, 2, 3, 5]:
    ring = dq(ract.exp_fixed(q(xf), FRAC))
    errs.append(abs(ring - math.exp(xf)) / (math.exp(xf) + 1e-9))
check(f"exp max rel err {max(errs):.2e} < 5e-3", max(errs) < 5e-3)

print("== 2. sigmoid vs oracle ==")
es = []
for xf in [-8, -4, -2, -1, 0, 1, 2, 4, 8]:
    ring = dq(ract.sigmoid_fixed(q(xf), FRAC))
    es.append(abs(ring - 1.0 / (1.0 + math.exp(-xf))))
check(f"sigmoid max abs err {max(es):.2e} < 5e-3", max(es) < 5e-3)
check("sigmoid saturates: (-infish)->~0, (+infish)->~1",
      dq(ract.sigmoid_fixed(q(-12), FRAC)) < 0.01 and dq(ract.sigmoid_fixed(q(12), FRAC)) > 0.99)

print("== 3. GELU vs oracle ==")
eg = []
for xf in [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]:
    ring = dq(ract.gelu_fixed(q(xf), FRAC))
    exact = xf * 0.5 * (1.0 + math.erf(xf / math.sqrt(2)))     # true GELU
    eg.append(abs(ring - exact))
check(f"GELU (sigmoid-approx) max abs err {max(eg):.2e} < 3e-2", max(eg) < 3e-2)

print("== 4. RMSNorm vs oracle ==")
np.random.seed(0)
xf = np.random.randn(128).astype(np.float64)
wf = np.random.randn(128).astype(np.float64) * 0.5 + 1.0
xq = [q(v) for v in xf]; wq = [q(v) for v in wf]
ring = np.array([dq(v) for v in ract.rmsnorm_fixed(xq, wq, FRAC)])
rms = math.sqrt((xf * xf).mean())
oracle = xf / rms * wf
check(f"RMSNorm max abs err {np.abs(ring - oracle).max():.2e} < 2e-2", np.abs(ring - oracle).max() < 2e-2)

print("== 4b. LayerNorm (mean-centered) vs oracle ==")
xf2 = np.random.randn(128).astype(np.float64)
g = np.abs(np.random.randn(128)) * 0.3 + 1.0
be = np.random.randn(128) * 0.1
ring_ln = np.array([dq(v) for v in ract.layernorm_fixed([q(v) for v in xf2], [q(v) for v in g], [q(v) for v in be], FRAC)])
m = xf2.mean(); var = ((xf2 - m) ** 2).mean()
oracle_ln = (xf2 - m) / math.sqrt(var + dq(1)) * g + be
check(f"LayerNorm max abs err {np.abs(ring_ln - oracle_ln).max():.2e} < 2e-2", np.abs(ring_ln - oracle_ln).max() < 2e-2)

print("== 5. ract.py is float-free and multiplier-free (AST) ==")
src = open(os.path.join(os.path.dirname(__file__), "..", "emulation", "ract.py")).read()
tree = ast.parse(src)
floats = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "float"]
ops = [type(n.op).__name__ for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.FloorDiv, ast.Pow, ast.Div))]
imps = set()
for n in ast.walk(tree):
    if isinstance(n, ast.ImportFrom) and n.module: imps.add(n.module.split(".")[0])
check(f"no float(); no * // ** / in code (ops {ops})", not floats and not ops)
check(f"imports only ringkit (got {sorted(imps)})", imps <= {"ringkit"})

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
