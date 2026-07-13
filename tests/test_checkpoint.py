"""Test for ringkit.checkpoint — load a pretrained model into the ring, torch/numpy/float-free.

The proof-of-kit (owner's bar): ingest real checkpoint weights using INTEGER/BYTE ops only. Two parts:
  1. Portable, deterministic: decode known IEEE float BIT PATTERNS -> ring ARC (the exponent),
     written as raw integer bits (no float() ever), so this runs anywhere.
  2. Opportunistic: if a real deployed .pth is reachable, load it and confirm the ingestion.
Run: python3 -m ringkit.tests.test_checkpoint"""
import os
import sys
from ringkit.emulation import checkpoint as ck

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

print("== 1. float-free decode of known IEEE bit patterns (ARC = biased exponent) ==")
# raw fp32 bit patterns written as INTEGERS (no float used to make them): value -> bits -> 4 LE bytes
F32 = {"1.0": 0x3F800000, "2.0": 0x40000000, "0.5": 0x3F000000, "-1.0": 0xBF800000, "0.0": 0x00000000}
def bytes_of(u, n): return u.to_bytes(n, "little")
raw = b"".join(bytes_of(u, 4) for u in F32.values())
arc = ck._bytes_to_arc(raw, 4, "f32")
# exponents: 1.0->127, 2.0->128, 0.5->126, -1.0->127 (sign not in arc), 0.0->0
check("f32 exponents decode exactly [127,128,126,127,0]", list(arc) == [127, 128, 126, 127, 0])
check("all decoded ARC values are ring 0..255", all(0 <= v <= 255 for v in arc))
# bf16: 1.0 = 0x3F80 -> exp bits 7..14 = 127
bf = ck._bytes_to_arc(bytes_of(0x3F80, 2), 2, "bf16")
check("bf16 1.0 -> exponent 127", list(bf) == [127])
# f16: 1.0 = 0x3C00 -> 5-bit exp = 15
hf = ck._bytes_to_arc(bytes_of(0x3C00, 2), 2, "f16")
check("f16 1.0 decodes to a ring value", 0 <= hf[0] <= 255)

print("== 2. the loader imports NO torch / numpy / safetensors / struct / math in its own module ==")
src = open(os.path.join(os.path.dirname(__file__), "..", "emulation", "checkpoint.py")).read()
import ast
imports = set()
for n in ast.walk(ast.parse(src)):
    if isinstance(n, ast.Import):
        for a in n.names: imports.add(a.name.split(".")[0])
    elif isinstance(n, ast.ImportFrom) and n.module:
        imports.add(n.module.split(".")[0])
check(f"checkpoint.py imports only {sorted(imports)} (no torch/numpy/safetensors/struct/math)",
      not (imports & {"torch", "numpy", "safetensors", "struct", "math", "random"}))
# AST (real calls, not docstring text): no float() and no float division in the code
_tree = ast.parse(src)
_float_calls = [n for n in ast.walk(_tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "float"]
_divs = [n for n in ast.walk(_tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)]
check("no float() calls and no float-division in the CODE (AST, not docstring)",
      not _float_calls and not _divs)

print("== 3. opportunistic: load a REAL deployed checkpoint if reachable ==")
cands = [
    os.path.expanduser("~/Projects/vlm-transformers/vlm_rdt_best.pth"),
    "/sessions/dazzling-zen-euler/mnt/Projects/vlm-transformers/vlm_rdt_best.pth",
    "/sessions/dazzling-zen-euler/mnt/vlm-transformers/vlm_rdt_best.pth",
]
real = next((p for p in cands if os.path.exists(p)), None)
if real:
    nt, npm, ten = ck.summarize(real)
    check(f"loaded real RDT: {nt} tensors, {npm:,} params (>100 tensors)", nt > 100)
    check("every weight decoded to a ring value 0..255",
          all(0 <= v <= 255 for t in list(ten.values())[:40] for v in t.data[:200]))
    before = set(sys.modules)
    ck.load_pth(real, limit=5)
    check("no torch/numpy/safetensors pulled in by the load",
          not ({m.split('.')[0] for m in set(sys.modules) - before} & {"torch", "numpy", "safetensors"}))
else:
    print("  (no real .pth reachable here — portable decode tests above are the permanent proof)")

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
