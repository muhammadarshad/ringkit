"""END-TO-END faithful inference proof: the FULL deployed RDT image encoder, on its REAL weights,
run entirely in ring fixed-point (float-free), vs a numpy float oracle of the identical QCM forward:

  grid -> QuadrantRingProjector -> VacuumDepthEmbedding -> RoPERDTEncoder (gated recurrent depth,
  shared QuantumRoPE layer) -> mean pool -> QCMProjectionHead (Linear->GELU->LayerNorm->Linear)
  -> L2 normalize -> image embedding.

No float on the ring compute path (integer mantissa-shift dequant, shift-add MAC, integer Taylor
exp, ring isqrt). Opportunistic (needs the real .pth); reduced grid for speed. The whole model,
not a layer.  Run: python3 -m ringkit.tests.test_rdt_e2e"""
import io
import math
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

FRAC = 16; ONE = 1 << FRAC; SCALE = 21; HALF = 128; Q = 64
def qz(v): return int(round(float(v) * ONE))
def dq(v): return v / ONE
def _sd(n, d): return -rn.mf_floordiv(-n, d) if n < 0 else rn.mf_floordiv(n, d)

# QCM ring trig (replicated, pure integer)
def _arch(p, hp): return 0 if (p <= 0 or p >= hp) else SCALE * 2 * math.isqrt(p * (hp - p)) // hp
def SIN(p):
    p &= 0xFF; return _arch(p, HALF) if p < HALF else (-_arch(p - HALF, HALF)) % 256
def COS(p): return SIN((p + Q) & 0xFF)
def sg(v): return v - 256 if v > HALF else v
COS_U = [sg(COS(p)) for p in range(256)]; SIN_U = [sg(SIN(p)) for p in range(256)]
cosq = [_sd(COS_U[a] << FRAC, SCALE) for a in range(256)]; sinq = [_sd(SIN_U[a] << FRAC, SCALE) for a in range(256)]
cosf = [COS_U[a] / SCALE for a in range(256)]; sinf = [SIN_U[a] / SCALE for a in range(256)]

fx = ck.load_fixed(PATH, frac=FRAC); z = zipfile.ZipFile(PATH); root = z.namelist()[0].split("/")[0]
refs = dict(ck._flatten(ck._RingUnpickler(io.BytesIO(z.read(f"{root}/data.pkl"))).load()))
def f32(n):
    r = refs[n]; k = r.storage[2]; nn = 1
    for s in r.size: nn *= s
    return np.frombuffer(z.read(f"{root}/data/{k}"), dtype=np.float32, count=nn).astype(np.float64).reshape(r.size)
def Wf(n): return fx[n][0]
V = "vision_encoder."; E = V + "encoder."; LY = E + "layer."
D, Hh, C, NH, HD, DEPTH = 4, 4, 128, 4, 32, 2
N = D * Hh
np.random.seed(0)
grid = np.random.rand(D, Hh, C)

# ---------- ORACLE ----------
def ln_o(v, g, b):
    m = v.mean(-1, keepdims=True); return (v - m) / np.sqrt(((v - m) ** 2).mean(-1, keepdims=True) + 1e-5) * g + b
def gelu_o(v): return v * (1 / (1 + np.exp(-1.702 * v)))
def lin_o(v, n): return v @ f32(n + ".weight").T + f32(n + ".bias")
def sig_o(v): return 1 / (1 + np.exp(-v))
def rot_half(v):
    xz, xx = v[..., :16], v[..., 16:]
    return np.concatenate([np.concatenate([-xz[..., 8:], xz[..., :8]], -1), np.concatenate([-xx[..., 8:], xx[..., :8]], -1)], -1)
arc = np.clip((grid * 256).astype(int), 0, 255); c = np.array(cosf)[arc]; s = np.array(sinf)[arc]
stacked = np.concatenate([np.clip(c, 0, None), np.clip(s, 0, None), np.clip(-c, 0, None), np.clip(-s, 0, None)], -1).reshape(N, C * 4) * f32(V + "quadrant_proj.modulation")
zt = stacked @ f32(V + "quadrant_proj.proj.weight").T + f32(V + "quadrant_proj.proj.bias") + np.repeat(f32(V + "vacuum_emb.depth_emb.weight")[:D], Hh, axis=0)
cosR = f32(LY + "self_attn.rope.cos_cached")[:D, :Hh, :].reshape(N, HD); sinR = f32(LY + "self_attn.rope.sin_cached")[:D, :Hh, :].reshape(N, HD)
def layer_o(x):
    h = ln_o(x, f32(LY + "norm1.weight"), f32(LY + "norm1.bias"))
    qh = lin_o(h, LY + "self_attn.q_proj").reshape(N, NH, HD); kh = lin_o(h, LY + "self_attn.k_proj").reshape(N, NH, HD); vh = lin_o(h, LY + "self_attn.v_proj").reshape(N, NH, HD)
    qn = ln_o(qh, f32(LY + "self_attn.q_norm.weight"), f32(LY + "self_attn.q_norm.bias")); kn = ln_o(kh, f32(LY + "self_attn.k_norm.weight"), f32(LY + "self_attn.k_norm.bias"))
    qr = np.stack([qn[:, hh] * cosR + rot_half(qn[:, hh]) * sinR for hh in range(NH)], 1); kr = np.stack([kn[:, hh] * cosR + rot_half(kn[:, hh]) * sinR for hh in range(NH)], 1)
    ctx = np.zeros((N, NH, HD))
    for hh in range(NH):
        S = (qr[:, hh] @ kr[:, hh].T) / np.sqrt(HD); W = np.exp(S - S.max(1, keepdims=True)); W /= W.sum(1, keepdims=True); ctx[:, hh] = W @ vh[:, hh]
    x = x + lin_o(ctx.reshape(N, C), LY + "self_attn.out_proj")
    return x + (gelu_o(lin_o(ln_o(x, f32(LY + "norm2.weight"), f32(LY + "norm2.bias")), LY + "linear1")) @ f32(LY + "linear2.weight").T + f32(LY + "linear2.bias"))
x0 = zt.copy(); h = zt.copy(); de = f32(E + "depth_embed.weight")
for step in range(DEPTH):
    h_in = h + sig_o(lin_o(x0, E + "inject_gate.0")) * x0 + de[step]
    hn = layer_o(h_in)
    g = sig_o(np.concatenate([h, hn], -1) @ f32(E + "gate.0.weight").T + f32(E + "gate.0.bias"))
    h = g * hn + (1 - g) * h
p = lin_o(h.mean(0)[None], "image_proj.net.0")[0]; p = gelu_o(p); p = ln_o(p[None], f32("image_proj.net.2.weight"), f32("image_proj.net.2.bias"))[0]; p = lin_o(p[None], "image_proj.net.3")[0]
emb_o = p / np.sqrt((p * p).sum())

# ---------- RING (float-free) ----------
def lin_r(rows, n, o, i): return [infer.linear(r, Wf(n + ".weight"), Wf(n + ".bias"), o, i, FRAC) for r in rows]
def lnr(rows, n):
    g = Wf(n + ".weight"); b = Wf(n + ".bias"); return [ract.layernorm_fixed(r, g, b, FRAC) for r in rows]
def rh_r(v):
    xz, xx = v[:16], v[16:]; return [-a for a in xz[8:]] + xz[:8] + [-a for a in xx[8:]] + xx[:8]
gq = [[qz(grid[d][hh][w]) for w in range(C)] for d in range(D) for hh in range(Hh)]
mod = Wf(V + "quadrant_proj.modulation"); tok = []
for r in gq:
    ch = []
    for chan in range(4):
        for w in range(C):
            a = (r[w] >> 8) & 0xFF; cc = cosq[a]; sc = sinq[a]; val = [cc, sc, -cc, -sc][chan]; ch.append(val if val > 0 else 0)
    tok.append([_sd(rn.mul(ch[j], mod[j]), ONE) for j in range(C * 4)])
zt_r = lin_r(tok, V + "quadrant_proj.proj", C, C * 4); dr = Wf(V + "vacuum_emb.depth_emb.weight")
zt_r = [[zt_r[d * Hh + hh][j] + dr[d * C + j] for j in range(C)] for d in range(D) for hh in range(Hh)]
cosRq = [[qz(cosR[i][j]) for j in range(HD)] for i in range(N)]; sinRq = [[qz(sinR[i][j]) for j in range(HD)] for i in range(N)]
def rope_r(seq):
    o = []
    for t in range(N):
        v = seq[t]; rh = rh_r(v); o.append([_sd(rn.mul(v[d], cosRq[t][d]), ONE) + _sd(rn.mul(rh[d], sinRq[t][d]), ONE) for d in range(HD)])
    return o
def layer_r(x):
    hq = lnr(x, LY + "norm1")
    qq = lin_r(hq, LY + "self_attn.q_proj", C, C); kq = lin_r(hq, LY + "self_attn.k_proj", C, C); vq = lin_r(hq, LY + "self_attn.v_proj", C, C)
    qnw = Wf(LY + "self_attn.q_norm.weight"); qnb = Wf(LY + "self_attn.q_norm.bias"); knw = Wf(LY + "self_attn.k_norm.weight"); knb = Wf(LY + "self_attn.k_norm.bias")
    qH = [[ract.layernorm_fixed(r[hh * HD:(hh + 1) * HD], qnw, qnb, FRAC) for r in qq] for hh in range(NH)]
    kH = [[ract.layernorm_fixed(r[hh * HD:(hh + 1) * HD], knw, knb, FRAC) for r in kq] for hh in range(NH)]
    vH = [[vq[t][hh * HD:(hh + 1) * HD] for t in range(N)] for hh in range(NH)]
    qHr = [rope_r(qH[hh]) for hh in range(NH)]; kHr = [rope_r(kH[hh]) for hh in range(NH)]
    sc = infer.inv_sqrt(HD, FRAC); ctx = [[0] * C for _ in range(N)]
    for hh in range(NH):
        a = infer.attention(qHr[hh], kHr[hh], vH[hh], FRAC, scale=sc)
        for t in range(N):
            for d in range(HD): ctx[t][hh * HD + d] = a[t][d]
    aq = lin_r(ctx, LY + "self_attn.out_proj", C, C); x1 = [[x[t][j] + aq[t][j] for j in range(C)] for t in range(N)]
    l2 = lin_r([[ract.gelu_fixed(v, FRAC) for v in row] for row in lin_r(lnr(x1, LY + "norm2"), LY + "linear1", 512, C)], LY + "linear2", C, 512)
    return [[x1[t][j] + l2[t][j] for j in range(C)] for t in range(N)]
x0r = [r[:] for r in zt_r]; hr = [r[:] for r in zt_r]; drb = Wf(E + "depth_embed.weight")
for step in range(DEPTH):
    al = [[ract.sigmoid_fixed(v, FRAC) for v in row] for row in lin_r(x0r, E + "inject_gate.0", C, C)]
    h_in = [[hr[t][j] + _sd(rn.mul(al[t][j], x0r[t][j]), ONE) + drb[step * C + j] for j in range(C)] for t in range(N)]
    hn = layer_r(h_in)
    g = [[ract.sigmoid_fixed(v, FRAC) for v in row] for row in lin_r([hr[t] + hn[t] for t in range(N)], E + "gate.0", C, 2 * C)]
    hr = [[_sd(rn.mul(g[t][j], hn[t][j]), ONE) + _sd(rn.mul(ONE - g[t][j], hr[t][j]), ONE) for j in range(C)] for t in range(N)]
pooled = [_sd(sum(hr[t][j] for t in range(N)), N) for j in range(C)]
p = infer.linear(pooled, Wf("image_proj.net.0.weight"), Wf("image_proj.net.0.bias"), C, C, FRAC)
p = [ract.gelu_fixed(v, FRAC) for v in p]
p = ract.layernorm_fixed(p, Wf("image_proj.net.2.weight"), Wf("image_proj.net.2.bias"), FRAC)
p = infer.linear(p, Wf("image_proj.net.3.weight"), Wf("image_proj.net.3.bias"), C, C, FRAC)
ss = 0
for v in p: ss += rn.mul(v, v) >> FRAC
nrm = rn.isqrt(ss << FRAC) or 1
emb_r = np.array([dq(_sd(v << FRAC, nrm)) for v in p])

cos = float(emb_r @ emb_o / (np.linalg.norm(emb_r) * np.linalg.norm(emb_o) + 1e-12))
print(f"  end-to-end RDT image encoder (grid {D}x{Hh}, depth {DEPTH}): max abs err {np.abs(emb_r-emb_o).max():.2e}, cosine {cos:.6f}")
check("full model ring embedding matches float oracle (max err < 1e-2)", np.abs(emb_r - emb_o).max() < 1e-2)
check("cosine similarity > 0.999", cos > 0.999)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
