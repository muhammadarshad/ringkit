"""Test for ringkit.onix — load Gemma .onix (hpq) weights into the ring, torch/numpy/float-free.

Part 1 (portable): synthesize a minimal ONIX file (bytes only) and check the parser + ring
projection are bit-exact vs a direct integer reference. Part 2 (opportunistic): parse a REAL 2B
Gemma .onix and compute a real q_proj row-set on the ring, bit-exact.
Run: python3 -m ringkit.tests.test_onix"""
import os
import random
import tempfile
from ringkit.emulation import onix

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

def u32(v): return int(v).to_bytes(4, "little")
def u64(v): return int(v).to_bytes(8, "little")

print("== 1. portable: synthetic ONIX round-trips + ring projection bit-exact ==")
OF, INF = 3, 8
xbar = bytes([random.Random(1).randint(0, 255) for _ in range(OF * INF)])
s_row = [-4, 0, 3]                              # int8 exponents (shifts)
z_row = [1, 5, 3]                              # uint8 divisors
data = xbar + bytes([v & 0xFF for v in s_row]) + bytes(z_row)
# header (256) + one 192-byte entry + data
hdr = bytearray(256)
hdr[0:4] = b"ONIX"; hdr[8:12] = u32(1); hdr[68:76] = u64(256); hdr[76:84] = u64(256 + 192)
ent = bytearray(192)
ent[0:1] = b"w"; ent[128:136] = u64(0); ent[136:140] = u32(OF); ent[140:144] = u32(1)
ent[144:148] = u32(INF); ent[148:156] = u64(OF * INF); ent[156:164] = u64(OF)
blob = bytes(hdr) + bytes(ent) + data
tf = tempfile.NamedTemporaryFile(suffix=".onix", delete=False); tf.write(blob); tf.close()

data_off, ents = onix.index(tf.name)
check("parses magic/index; 1 tensor 'w' shape 3x8", list(ents) == ["w"] and ents["w"]["out_feat"] == 3 and ents["w"]["in_feat"] == 8)
xb, s, z, of, inf = onix.tensor(tf.name, "w")
check("tensor bytes recovered", bytes(xb) == xbar and s == s_row and z == z_row and (of, inf) == (3, 8))
x = [random.Random(2).randint(-127, 127) for _ in range(INF)]
def ref(r):
    dot = sum((xb[r * inf + i] - 128) * x[i] for i in range(inf))
    acc = dot << s[r] if s[r] >= 0 else dot >> (-s[r])
    zz = z[r] or 1
    return -((-acc) // zz) if acc < 0 else acc // zz
ring = [onix.project_row(xb, r, s[r], z[r], x, inf) for r in range(OF)]
check("ring projection bit-exact vs integer reference", ring == [ref(r) for r in range(OF)])
os.unlink(tf.name)

print("== 2. opportunistic: REAL 2B Gemma .onix, ring q_proj bit-exact ==")
cands = [os.path.expanduser("~/Projects/hpq-kernel-rust/gemma2_2b.onix"),
         "/sessions/dazzling-zen-euler/mnt/hpq-kernel-rust/gemma2_2b.onix"]
real = next((p for p in cands if os.path.exists(p)), None)
if real:
    import sys
    before = set(sys.modules)
    _, E = onix.index(real)
    check("parsed real Gemma .onix (>100 tensors)", len(E) > 100)
    nm = "model.layers.0.self_attn.q_proj"
    xb, s, z, of, inf = onix.tensor(real, nm, rows=6)
    xr = [random.Random(0).randint(-127, 127) for _ in range(inf)]
    def refr(r):
        dot = sum((xb[r * inf + i] - 128) * xr[i] for i in range(inf))
        acc = dot << s[r] if s[r] >= 0 else dot >> (-s[r])
        zz = z[r] or 1
        return -((-acc) // zz) if acc < 0 else acc // zz
    rg = [onix.project_row(xb, r, s[r], z[r], xr, inf) for r in range(6)]
    check("real Gemma q_proj: ring bit-exact vs integer reference", rg == [refr(r) for r in range(6)])
    bad = [m for m in set(sys.modules) - before if m.split(".")[0] in ("torch", "numpy", "math", "safetensors")]
    check("no torch/numpy/math used", not bad)
else:
    print("  (no real gemma2_2b.onix reachable — portable synthetic test above is the permanent proof)")

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
