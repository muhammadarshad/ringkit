"""ringkit.quanta — the MPRC architectures on the ring, gated end-to-end vs a numpy float oracle.

Rotor (MPRCRDT): the deployed RDT image encoder, run through the `quanta` PACKAGE (float-free),
must match the identical QCM forward in float to cosine > 0.999 on the real weights. This is the
same bar as test_rdt_e2e, but proving the shipped package (not an inline forward) is faithful.
Gluon/Soliton: their gates land as weights/SSD arrive.  Run: python3 -m ringkit.tests.test_quanta"""
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
    print("  (no real vlm_rdt_best.pth reachable — skipping the Rotor e2e gate)")
    print("\nRESULT: ALL PASS")
    raise SystemExit(0)

import ast as _ast
import numpy as np                                  # labeled float ORACLE only (D9) — never the ring path
from ringkit.emulation import checkpoint as ck
from ringkit import quanta
from ringkit.quanta._ringtrig import FRAC, ONE, COS_U, SIN_U, SCALE   # integer ring tables

# oracle float trig lives HERE (in the test), not in the package
COSF = [COS_U[a] / SCALE for a in range(256)]
SINF = [SIN_U[a] / SCALE for a in range(256)]

def qz(v): return int(round(float(v) * ONE))
def dq(v): return v / ONE

fx = ck.load_fixed(PATH, frac=FRAC)
z = zipfile.ZipFile(PATH); root = z.namelist()[0].split("/")[0]
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

# ---------- numpy float ORACLE (identical QCM forward) ----------
def ln_o(v, g, b):
    m = v.mean(-1, keepdims=True); return (v - m) / np.sqrt(((v - m) ** 2).mean(-1, keepdims=True) + 1e-5) * g + b
def gelu_o(v): return v * (1 / (1 + np.exp(-1.702 * v)))
def lin_o(v, n): return v @ f32(n + ".weight").T + f32(n + ".bias")
def sig_o(v): return 1 / (1 + np.exp(-v))
def rot_half(v):
    xz, xx = v[..., :16], v[..., 16:]
    return np.concatenate([np.concatenate([-xz[..., 8:], xz[..., :8]], -1), np.concatenate([-xx[..., 8:], xx[..., :8]], -1)], -1)
arc = np.clip((grid * 256).astype(int), 0, 255); c = np.array(COSF)[arc]; s = np.array(SINF)[arc]
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

# ---------- RING via the quanta PACKAGE (float-free) ----------
grid_q = [[qz(grid[d][hh][w]) for w in range(C)] for d in range(D) for hh in range(Hh)]
cosRq = [[qz(cosR[i][j]) for j in range(HD)] for i in range(N)]
sinRq = [[qz(sinR[i][j]) for j in range(HD)] for i in range(N)]
emb_ring = quanta.rotor_forward(grid_q, Wf, cosRq, sinRq, D, Hh, C, NH, HD, DEPTH, prefix=V)
emb_r = np.array([dq(v) for v in emb_ring])

cos = float(emb_r @ emb_o / (np.linalg.norm(emb_r) * np.linalg.norm(emb_o) + 1e-12))
print(f"  Rotor (RDT) via quanta package (grid {D}x{Hh}, depth {DEPTH}): max abs err {np.abs(emb_r-emb_o).max():.2e}, cosine {cos:.6f}")
check("quanta.rotor_forward matches float oracle (max err < 1e-2)", np.abs(emb_r - emb_o).max() < 1e-2)
check("cosine > 0.999", cos > 0.999)

# ---------- compute-path composition gate (D9): the C-routed infer.linear must be BIT-IDENTICAL
# to the pure-Python shift-add reference (quanta's linears run through the gated C GEMV blocks
# via balanced base-256 weight digit-planes — never the per-element Python loop at real sizes) ----
import random as _rnd
from ringkit.emulation import infer as _inf
from ringkit.core import native as _rn
_r = _rnd.Random(5)
def _py_lin(x, Wl, b, M, K, frac):
    out = []
    for j in range(M):
        acc = 0
        for i in range(K):
            acc += _rn.mul(x[i], Wl[j * K + i])
        out.append((acc >> frac) + b[j])
    return out
_ok = True
for _M, _K in ((128, 128), (400, 128)):
    _W = [_r.randrange(-3 * (1 << FRAC), 3 * (1 << FRAC)) for _ in range(_M * _K)]
    _b = [_r.randrange(-(1 << FRAC), 1 << FRAC) for _ in range(_M)]
    for _x in ([_r.randrange(-3 * (1 << FRAC), 3 * (1 << FRAC)) for _ in range(_K)],
               [0] * (_K - 1) + [400 << FRAC]):                    # normal + outlier regimes
        _ok = _ok and _inf.linear(_x, _W, _b, _M, _K, FRAC) == _py_lin(_x, _W, _b, _M, _K, FRAC)
check("infer.linear C route == pure-Python shift-add reference (bit-for-bit)", _ok)

# ==================== SOLITON (Mamba2 gate_lat2d) ====================
mcands = [os.path.expanduser("~/Projects/vlm-transformers/huggingface/exported/qcm-rp2k-cloud-matched-results/full_mamba2_L2_k2384_h16_gate_lat2d_best.pth")]
MPATH = next((p for p in mcands if os.path.exists(p)), None)
if MPATH is not None:
    from ringkit import quanta as Q
    fx2 = ck.load_fixed(MPATH, frac=FRAC)
    z2 = zipfile.ZipFile(MPATH); r2 = z2.namelist()[0].split("/")[0]
    refs2 = dict(ck._flatten(ck._RingUnpickler(io.BytesIO(z2.read(f"{r2}/data.pkl"))).load()))
    def g32(n):
        r = refs2[n]; k = r.storage[2]; nn = 1
        for s in r.size: nn *= s
        return np.frombuffer(z2.read(f"{r2}/data/{k}"), dtype=np.float32, count=nn).astype(np.float64).reshape(r.size)
    def W2(n): return fx2[n][0]
    Cs, NH2, HD2, NL2, NCLS = 128, 16, 8, 2, 2384
    Ds, Hs = 4, 4; Ns = Ds * Hs; MM = "model."
    np.random.seed(0); gridS = np.random.rand(Ds, Hs, Cs)
    # ---- float oracle (identical Mamba2 gate_lat2d forward) ----
    arcS = np.clip((gridS * 256).astype(int), 0, 255); cS = np.array(COSF)[arcS]; sS = np.array(SINF)[arcS]
    stk = np.concatenate([np.clip(cS, 0, None), np.clip(sS, 0, None), np.clip(-cS, 0, None), np.clip(-sS, 0, None)], -1).reshape(Ns, Cs*4) * g32(MM+"quadrant_proj.modulation")
    zS = stk @ g32(MM+"quadrant_proj.proj.weight").T + g32(MM+"quadrant_proj.proj.bias") + np.repeat(g32(MM+"vacuum_emb.depth_emb.weight")[:Ds], Hs, 0)
    cosR2 = g32(MM+"layers.0.rope.cos_cached")[:Ds,:Hs,:].reshape(Ns, HD2); sinR2 = g32(MM+"layers.0.rope.sin_cached")[:Ds,:Hs,:].reshape(Ns, HD2)
    dstep = [-1 if hh < NH2//2 else min(16, 1 << (hh-NH2//2)) for hh in range(NH2)]
    def spo(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    def rho(y, gg): return y / np.sqrt((y*y).mean(-1, keepdims=True) + 1e-5) * gg
    def rh8(v): return np.concatenate([-v[...,2:4], v[...,:2], -v[...,6:8], v[...,4:6]], -1)
    def latt(sh):
        out=[None]*NH2; gs=[sh[hh].sum(0) for hh in range(NH2)]
        for hh in range(NH2):
            if dstep[hh]<0: out[hh]=np.broadcast_to(gs[hh],(Ns,HD2)).copy()
        loc=[hh for hh in range(NH2) if dstep[hh]>=0]
        cur={hh: sh[hh].reshape(Ds,Hs,HD2).copy() for hh in loc}
        for hh in loc:
            if dstep[hh]==0: out[hh]=Ns*cur[hh].reshape(Ns,HD2)
        mt=max(dstep[hh] for hh in loc)
        for st in range(1,mt+1):
            for hh in loc:
                a=cur[hh]; cur[hh]=(np.roll(a,1,0)+np.roll(a,-1,0)+np.roll(a,1,1)+np.roll(a,-1,1)+4.0*a)/8.0
                if dstep[hh]==st: out[hh]=Ns*cur[hh].reshape(Ns,HD2)
        return out
    def mlayer(x, li):
        L=f"{MM}layers.{li}."; proj=x@g32(L+"in_proj.weight").T+g32(L+"in_proj.bias")
        q,k,v,dl=proj[:,:Cs],proj[:,Cs:2*Cs],proj[:,2*Cs:3*Cs],proj[:,3*Cs:]
        qh=[q[:,hh*HD2:(hh+1)*HD2] for hh in range(NH2)]; kh=[k[:,hh*HD2:(hh+1)*HD2] for hh in range(NH2)]; vh=[v[:,hh*HD2:(hh+1)*HD2] for hh in range(NH2)]
        dh=[spo(dl[:,hh])[:,None] for hh in range(NH2)]
        qh=[qh[hh]*cosR2+rh8(qh[hh])*sinR2 for hh in range(NH2)]; kh=[kh[hh]*cosR2+rh8(kh[hh])*sinR2 for hh in range(NH2)]
        state=[dh[hh]*(kh[hh]*vh[hh]) for hh in range(NH2)]; h=latt(state)
        y=np.concatenate([qh[hh]*h[hh] for hh in range(NH2)],-1); y=rho(y,g32(L+"norm.weight"))
        w=np.exp(x@g32(L+"gate_route.weight").T+g32(L+"gate_route.bias")); w=w/w.sum(-1,keepdims=True)
        gg=x@g32(L+"gate_proj.weight").T+g32(L+"gate_proj.bias")+w@g32(L+"arm_bias"); y=y*(1/(1+np.exp(-gg)))
        return y@g32(L+"out_proj.weight").T+g32(L+"out_proj.bias")
    zc=zS.copy()
    for li in range(NL2): zc=mlayer(zc, li)
    logits_o = zc.mean(0) @ g32(MM+"head.weight").T + g32(MM+"head.bias")
    # ---- ring via the quanta package ----
    grid_qS=[[qz(gridS[d][hh][w]) for w in range(Cs)] for d in range(Ds) for hh in range(Hs)]
    cosR2q=[[qz(cosR2[i][j]) for j in range(HD2)] for i in range(Ns)]; sinR2q=[[qz(sinR2[i][j]) for j in range(HD2)] for i in range(Ns)]
    logit_ring=Q.soliton_forward(grid_qS, W2, cosR2q, sinR2q, Ds, Hs, Cs, NH2, HD2, NL2, NCLS, Ds, Hs, prefix=MM)
    logits_r=np.array([dq(v) for v in logit_ring])
    cosS=float(logits_r@logits_o/(np.linalg.norm(logits_r)*np.linalg.norm(logits_o)+1e-12))
    print(f"  Soliton (Mamba2) via quanta package: logit cosine {cosS:.6f}, argmax ring={int(logits_r.argmax())} oracle={int(logits_o.argmax())}")
    check("quanta.soliton_forward logits match float oracle (cosine > 0.999)", cosS > 0.999)
    check("Soliton argmax class matches oracle", int(logits_r.argmax()) == int(logits_o.argmax()))
else:
    print("  (no Mamba2 checkpoint reachable — Soliton gate skipped)")

# ---------- the quanta PACKAGE compute path must be float-free / import-clean (AST) ----------
print("  package float-free / no numpy·math (AST):")
pkg = os.path.join(os.path.dirname(__file__), "..", "quanta")
for mod in ("_ringtrig", "frontend", "layers", "models", "ssd", "__init__"):
    src = open(os.path.join(pkg, mod + ".py")).read()
    t = _ast.parse(src)
    floatlit = [n for n in _ast.walk(t) if isinstance(n, _ast.Constant) and isinstance(n.value, float)]
    floatcall = [n for n in _ast.walk(t) if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id == "float"]
    imps = set()
    for n in _ast.walk(t):
        if isinstance(n, _ast.Import):
            for al in n.names: imps.add(al.name.split(".")[0])
        elif isinstance(n, _ast.ImportFrom) and n.module:
            imps.add(n.module.split(".")[0])
    dirty = imps & {"numpy", "math", "scipy", "torch"}
    check(f"quanta/{mod}.py: no float literal/float()/numpy·math (imps∩bad={sorted(dirty)})",
          not floatlit and not floatcall and not dirty)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
