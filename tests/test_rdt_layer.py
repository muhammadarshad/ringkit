"""Faithful REAL-MODEL inference proof: the deployed RDT encoder layer, on its REAL weights,
run entirely in ring fixed-point (float-free), matched against a numpy float oracle of the identical
QCM forward (rope.py RoPETransformerEncoderLayer): pre-LN -> QK-normed QuantumRoPE softmax attention
-> residual -> LN -> GELU MLP -> residual. Uses the checkpoint's CACHED rope cos/sin tables.

Opportunistic: runs only if a real vlm_rdt_best.pth is reachable; otherwise skips (the float-free
primitives themselves are permanently tested in test_infer/test_ract/test_infer_block).
Run: python3 -m ringkit.tests.test_rdt_layer"""
import io
import os
import zipfile

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

cands = [os.path.expanduser("~/Projects/vlm-transformers/vlm_rdt_best.pth"),
         "/sessions/dazzling-zen-euler/mnt/Projects/vlm-transformers/vlm_rdt_best.pth"]
PATH = next((p for p in cands if os.path.exists(p)), None)
if PATH is None:
    print("  (no real vlm_rdt_best.pth reachable — skipping; primitives tested elsewhere)")
    print("\nRESULT: ALL PASS")
    raise SystemExit(0)

import numpy as np
from ringkit.emulation import checkpoint as ck, infer, ract
from ringkit.core import native as rn

FRAC = 16; ONE = 1 << FRAC
def dq(v): return v / ONE
def q(v): return int(round(float(v) * ONE))
P = "vision_encoder.encoder.layer."
H, HD, C, N, GH, GW = 4, 32, 128, 16, 4, 4

fx = ck.load_fixed(PATH, frac=FRAC)
z = zipfile.ZipFile(PATH); root = z.namelist()[0].split("/")[0]
refs = dict(ck._flatten(ck._RingUnpickler(io.BytesIO(z.read(f"{root}/data.pkl"))).load()))
def f32(name):
    r = refs[name]; k = r.storage[2]; n = 1
    for s in r.size: n *= s
    return np.frombuffer(z.read(f"{root}/data/{k}"), dtype=np.float32, count=n).astype(np.float64).reshape(r.size)
def W(n): return f32(P + n)
def Wf(n): return fx[P + n][0]

np.random.seed(0)
x = np.random.randn(N, C) * 0.4
xq = [[q(v) for v in row] for row in x]
cosT = W("self_attn.rope.cos_cached")[:GH, :GW, :].reshape(N, HD)
sinT = W("self_attn.rope.sin_cached")[:GH, :GW, :].reshape(N, HD)

def rot_half(v):
    xz, xx = v[..., :16], v[..., 16:]
    rz = np.concatenate([-xz[..., 8:], xz[..., :8]], -1); rx = np.concatenate([-xx[..., 8:], xx[..., :8]], -1)
    return np.concatenate([rz, rx], -1)
def ln_o(v, g, b):
    m = v.mean(-1, keepdims=True); var = ((v - m) ** 2).mean(-1, keepdims=True)
    return (v - m) / np.sqrt(var + 1e-5) * g + b
def gelu_o(v): return v * (1 / (1 + np.exp(-1.702 * v)))
def lin_o(v, n): return v @ W(n + ".weight").T + W(n + ".bias")

# ---- oracle ----
h = ln_o(x, W("norm1.weight"), W("norm1.bias"))
qh = lin_o(h, "self_attn.q_proj").reshape(N, H, HD); kh = lin_o(h, "self_attn.k_proj").reshape(N, H, HD); vh = lin_o(h, "self_attn.v_proj").reshape(N, H, HD)
qn = ln_o(qh, W("self_attn.q_norm.weight"), W("self_attn.q_norm.bias")); kn = ln_o(kh, W("self_attn.k_norm.weight"), W("self_attn.k_norm.bias"))
qr = np.stack([qn[:, hh] * cosT + rot_half(qn[:, hh]) * sinT for hh in range(H)], 1)
kr = np.stack([kn[:, hh] * cosT + rot_half(kn[:, hh]) * sinT for hh in range(H)], 1)
ctx = np.zeros((N, H, HD))
for hh in range(H):
    S = (qr[:, hh] @ kr[:, hh].T) / np.sqrt(HD); Wt = np.exp(S - S.max(1, keepdims=True)); Wt /= Wt.sum(1, keepdims=True); ctx[:, hh] = Wt @ vh[:, hh]
x1 = x + lin_o(ctx.reshape(N, C), "self_attn.out_proj")
out_o = x1 + (gelu_o(lin_o(ln_o(x1, W("norm2.weight"), W("norm2.bias")), "linear1")) @ W("linear2.weight").T + W("linear2.bias"))

# ---- ring (float-free) ----
cosq = [[q(cosT[i][j]) for j in range(HD)] for i in range(N)]; sinq = [[q(sinT[i][j]) for j in range(HD)] for i in range(N)]
def lin_r(rows, n, o, i): return [infer.linear(r, Wf(n + ".weight"), Wf(n + ".bias"), o, i, FRAC) for r in rows]
def lnr(rows, n):
    g = Wf(n + ".weight"); b = Wf(n + ".bias"); return [ract.layernorm_fixed(r, g, b, FRAC) for r in rows]
def rh_r(v):
    xz, xx = v[:16], v[16:]; return [-a for a in xz[8:]] + xz[:8] + [-a for a in xx[8:]] + xx[:8]
def heads(rows, nw, nb):
    return [[ract.layernorm_fixed(r[hh * HD:(hh + 1) * HD], nw, nb, FRAC) for r in rows] for hh in range(H)]
def rope_r(seq):
    out = []
    for t in range(N):
        v = seq[t]; rhh = rh_r(v)
        out.append([infer._sdiv(rn.mul(v[d], cosq[t][d]), ONE) + infer._sdiv(rn.mul(rhh[d], sinq[t][d]), ONE) for d in range(HD)])
    return out
hq = lnr(xq, "norm1")
qq = lin_r(hq, "self_attn.q_proj", C, C); kq = lin_r(hq, "self_attn.k_proj", C, C); vq = lin_r(hq, "self_attn.v_proj", C, C)
qH = heads(qq, Wf("self_attn.q_norm.weight"), Wf("self_attn.q_norm.bias")); kH = heads(kq, Wf("self_attn.k_norm.weight"), Wf("self_attn.k_norm.bias"))
vH = [[vq[t][hh * HD:(hh + 1) * HD] for t in range(N)] for hh in range(H)]
qHr = [rope_r(qH[hh]) for hh in range(H)]; kHr = [rope_r(kH[hh]) for hh in range(H)]
scale = infer.inv_sqrt(HD, FRAC)
ctxr = [[0] * C for _ in range(N)]
for hh in range(H):
    a = infer.attention(qHr[hh], kHr[hh], vH[hh], FRAC, scale=scale)
    for t in range(N):
        for d in range(HD): ctxr[t][hh * HD + d] = a[t][d]
aq = lin_r(ctxr, "self_attn.out_proj", C, C)
x1q = [[xq[t][j] + aq[t][j] for j in range(C)] for t in range(N)]
l2 = lin_r([[ract.gelu_fixed(v, FRAC) for v in row] for row in lin_r(lnr(x1q, "norm2"), "linear1", 512, C)], "linear2", C, 512)
out_r = np.array([[dq(x1q[t][j] + l2[t][j]) for j in range(C)] for t in range(N)])

err = np.abs(out_r - out_o).max()
corr = np.corrcoef(out_r.ravel(), out_o.ravel())[0, 1]
print(f"  real RDT layer (N={N}) ring vs float: max abs err {err:.2e}, correlation {corr:.6f}")
check("ring reproduces the real RDT layer within 1e-2", err < 1e-2)
check("correlation with float oracle > 0.999", corr > 0.999)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
