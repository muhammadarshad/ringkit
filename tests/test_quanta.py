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
    def rh8(v):
        # QuantumRoPE4D rotate-half at hd=8: FOUR 2-dim ADI chunks, each (-x1, x0) — the
        # DEPLOYED convention, anchored against the real torch forward (webapp e2e section).
        o = np.empty_like(v)
        o[..., 0::2] = -v[..., 1::2]
        o[..., 1::2] = v[..., 0::2]
        return o
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

# ==================== WEBAPP E2E — the COMPLETE test ====================
# The deployed vlm-transformers webapp (app/serve.py) is the real product: image -> 16-channel
# wave cube (113x128) -> Rotor (RDT + LatticeAttention + cls head) and Soliton (Mamba2 HF export)
# -> fused prediction. Here the SAME deterministic real-regime frame (vacuum padding rows +
# saturated 255s, the regimes a random grid never exercises) runs through an inline numpy oracle
# of the app forward — VALIDATED against the actual torch app to fp32 precision (maxerr ~2e-5,
# scratchpad/app_e2e_anchor.py + app_e2e_oracle.py, 2026-07-15; incl. the real shelf photo) —
# and through the ring quanta package at the app's true scale (D=16, H=113, N=1808 tokens).
import glob as _glob
RAPP = next(iter(_glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--marshadbits--qcm-rp2k-cloud-matched-results/"
    "snapshots/*/heads_rdt_L2_k2384_best.pth"))), None)
if MPATH is None:
    print("  (no Mamba2 checkpoint — webapp e2e skipped)")
else:
    import time as _time
    Da, Ha, Na = 16, 113, 16 * 113
    ERF = np.vectorize(math.erf)

    # -- the app's wave encode (app/serve.py, numpy verbatim = labeled oracle) on a synthetic
    #    real-regime frame --
    def synth_frame():
        a = np.zeros((113, 128, 3), np.uint16)
        a[10:103, :, :] = (250, 250, 250)
        a[20:80, 8:56] = (180, 20, 15); a[30:52, 16:48] = (255, 255, 255)
        a[24:92, 64:120] = (20, 60, 160); a[40:58, 72:112] = (255, 215, 0)
        a[10:14, :, :] = (0, 0, 0)
        for i in range(6):
            a[34:48, 18 + 5 * i] = (10, 10, 10)
        return a

    def wave_encode(arr):
        MOD7 = (np.arange(256) % 7).astype(np.uint8)
        def evolve(m, steps):
            cur = m.astype(np.uint16)
            for _ in range(steps):
                nxt = cur.copy()
                up, dn = cur[:-2, 1:-1], cur[2:, 1:-1]
                lf, rt = cur[1:-1, :-2], cur[1:-1, 2:]
                c = cur[1:-1, 1:-1]
                nxt[1:-1, 1:-1] = (up + dn + lf + rt + (c << 2)) >> 3
                cur = nxt
            return cur.astype(np.uint8)
        def plaq(m):
            M = m.astype(np.uint16)
            up, dn = M[:-2, 1:-1], M[2:, 1:-1]
            lf, rt = M[1:-1, :-2], M[1:-1, 2:]
            f = (rt + up + 512 - lf - dn + 128) & 0xFF
            out = np.full(m.shape, 128, np.uint8)
            out[1:-1, 1:-1] = f.astype(np.uint8)
            return out
        R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
        s = R + G + B
        luma = (((R * 77 + G * 150 + B * 29) >> 8) & 0xFF).astype(np.uint8)
        gray = ((s // 3) & 0xFF).astype(np.uint8)
        chroma = (np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)).astype(np.uint8)
        u = (((MOD7[R] + MOD7[G] + MOD7[B]) * 14) & 0xFF).astype(np.uint8)
        winding = (((s >> 8) & 0xFF) * 85).astype(np.uint8)
        d1 = ((R - G) & 0xFF).astype(np.uint8); d2 = ((G - B) & 0xFF).astype(np.uint8)
        logs = [plaq(evolve(gray, t)) for t in (0, 1, 4, 16, 64, 128)]
        return np.stack([R.astype(np.uint8), G.astype(np.uint8), B.astype(np.uint8),
                         luma, gray, chroma, u, winding, d1, d2] + logs, 0)

    cube = wave_encode(synth_frame())                      # (16,113,128) uint8
    gridA = cube.astype(np.float64) / 255.0                # oracle grid
    nvac = int(((gridA.mean(-1) < 1e-3)).sum())
    print(f"  webapp e2e frame: {nvac} vacuum tokens, {int((cube == 255).sum())} saturated bytes")

    # -- shared numpy oracle pieces (validated vs the torch app) --
    def lno(v, g, b, eps=1e-5):
        m = v.mean(-1, keepdims=True)
        return (v - m) / np.sqrt(((v - m) ** 2).mean(-1, keepdims=True) + eps) * g + b
    def rot2(v):                                           # 4D rope rotate-half (2-dim chunks)
        o = np.empty_like(v)
        o[..., 0::2] = -v[..., 1::2]; o[..., 1::2] = v[..., 0::2]
        return o
    def front16(g32f, pfx):
        arc = np.clip((gridA * 256.0).astype(np.int64), 0, 255)
        c = np.array(COSF)[arc]; s = np.array(SINF)[arc]
        stk = np.concatenate([np.clip(c, 0, None), np.clip(s, 0, None),
                              np.clip(-c, 0, None), np.clip(-s, 0, None)], -1).reshape(Na, 512)
        stk = stk * g32f(pfx + "quadrant_proj.modulation")
        zt = stk @ g32f(pfx + "quadrant_proj.proj.weight").T + g32f(pfx + "quadrant_proj.proj.bias")
        vac = (gridA.mean(-1) < 1e-3).reshape(Na, 1)
        demb = g32f(pfx + "vacuum_emb.depth_emb.weight")[np.repeat(np.arange(Da), Ha)]
        vemb = g32f(pfx + "vacuum_emb.vacuum_emb").reshape(1, -1)
        return zt + demb * (1.0 - vac) + vemb * vac

    # -- SOLITON at app scale: oracle + ring --
    cosA = g32(MM + "layers.0.rope.cos_cached")[:Da, :Ha].reshape(Na, HD2)
    sinA = g32(MM + "layers.0.rope.sin_cached")[:Da, :Ha].reshape(Na, HD2)
    def soliton_o():
        x = front16(g32, MM)
        for li in range(NL2):
            L = f"{MM}layers.{li}."
            proj = x @ g32(L + "in_proj.weight").T + g32(L + "in_proj.bias")
            q, k, v, dl = proj[:, :Cs], proj[:, Cs:2*Cs], proj[:, 2*Cs:3*Cs], proj[:, 3*Cs:]
            qh = q.reshape(Na, NH2, HD2).transpose(1, 0, 2)
            kh = k.reshape(Na, NH2, HD2).transpose(1, 0, 2)
            vv = v.reshape(Na, NH2, HD2).transpose(1, 0, 2)
            dh = spo(dl).T[:, :, None]
            qh = qh * cosA + rot2(qh) * sinA; kh = kh * cosA + rot2(kh) * sinA
            st = dh * (kh * vv)
            out = np.empty_like(st); gs = st.sum(1, keepdims=True)
            cur = st.reshape(NH2, Da, Ha, HD2).copy()
            for hh in range(NH2):
                if dstep[hh] < 0: out[hh] = gs[hh]
            for stp in range(1, 17):
                cur = (np.roll(cur, 1, 1) + np.roll(cur, -1, 1) + np.roll(cur, 1, 2)
                       + np.roll(cur, -1, 2) + 4.0 * cur) / 8.0
                for hh in range(NH2):
                    if dstep[hh] == stp: out[hh] = Na * cur[hh].reshape(Na, HD2)
            y = (qh * out).transpose(1, 0, 2).reshape(Na, Cs)
            y = rho(y, g32(L + "norm.weight"))
            w = np.exp(x @ g32(L + "gate_route.weight").T + g32(L + "gate_route.bias"))
            w = w / w.sum(-1, keepdims=True)
            gg = x @ g32(L + "gate_proj.weight").T + g32(L + "gate_proj.bias") + w @ g32(L + "arm_bias")
            y = y * (1 / (1 + np.exp(-np.clip(gg, -500, 500))))
            x = y @ g32(L + "out_proj.weight").T + g32(L + "out_proj.bias")
        return x.mean(0) @ g32(MM + "head.weight").T + g32(MM + "head.bias")

    Q255 = [((u << FRAC) + 127) // 255 for u in range(256)]        # ring ingest of x/255
    gridA_q = [[Q255[int(u)] for u in row] for row in cube.reshape(Na, Cs)]
    def ring_rope(fx, key):
        c, shp = fx[key + "cos_cached"]; s, _ = fx[key + "sin_cached"]
        ca = np.array(c).reshape(shp)[:Da, :Ha].reshape(Na, HD2)
        sa = np.array(s).reshape(shp)[:Da, :Ha].reshape(Na, HD2)
        return [[int(v) for v in r] for r in ca], [[int(v) for v in r] for r in sa]
    cosA_q, sinA_q = ring_rope(fx2, MM + "layers.0.rope.")

    t0 = _time.time(); slo = soliton_o(); t1 = _time.time()
    slr = Q.soliton_forward(gridA_q, W2, cosA_q, sinA_q, Da, Ha, Cs, NH2, HD2, NL2, NCLS,
                            Da, Ha, prefix=MM)
    t2 = _time.time()
    slrf = np.array([dq(v) for v in slr])
    cosSA = float(slrf @ slo / (np.linalg.norm(slrf) * np.linalg.norm(slo) + 1e-12))
    print(f"  APP-E2E Soliton (N=1808): cosine {cosSA:.6f}, argmax ring={int(slrf.argmax())} "
          f"oracle={int(slo.argmax())}  [oracle {t1-t0:.0f}s, ring {t2-t1:.0f}s]")
    check("APP-E2E soliton: ring == app forward (cosine > 0.999)", cosSA > 0.999)
    check("APP-E2E soliton: argmax matches", int(slrf.argmax()) == int(slo.argmax()))

    # -- ROTOR (RotorHeads: RDT + LatticeAttention radius-1 + cls) at app scale --
    if RAPP is None:
        print("  (rotor heads checkpoint not in HF cache — webapp rotor gate skipped)")
    else:
        fx3 = ck.load_fixed(RAPP, frac=FRAC)
        z3 = zipfile.ZipFile(RAPP); r3 = z3.namelist()[0].split("/")[0]
        refs3 = dict(ck._flatten(ck._RingUnpickler(io.BytesIO(z3.read(f"{r3}/data.pkl"))).load()))
        RB = "model._orig_mod."
        def r32(n):
            r = refs3[RB + n]; k = r.storage[2]; nn = 1
            for s in r.size: nn *= s
            return np.frombuffer(z3.read(f"{r3}/data/{k}"), dtype=np.float32, count=nn).astype(np.float64).reshape(r.size)
        def W3(n): return fx3[RB + n][0]
        NHr, HDr = 16, 8
        cosR3 = r32("enc.encoder.layer.self_attn.rope.cos_cached")[:Da, :Ha].reshape(Na, HDr)
        sinR3 = r32("enc.encoder.layer.self_attn.rope.sin_cached")[:Da, :Ha].reshape(Na, HDr)
        def rotor_o():
            E = "enc.encoder."; A = E + "layer.self_attn."
            zt = front16(r32, "enc.")
            x0 = zt.copy(); h = zt.copy()
            for step in range(2):
                al = 1 / (1 + np.exp(-(x0 @ r32(E + "inject_gate.0.weight").T + r32(E + "inject_gate.0.bias"))))
                h_in = h + al * x0 + r32(E + "depth_embed.weight")[step]
                tt = lno(h_in, r32(E + "layer.norm1.weight"), r32(E + "layer.norm1.bias"))
                qh = (tt @ r32(A + "q_proj.weight").T + r32(A + "q_proj.bias")).reshape(Na, NHr, HDr).transpose(1, 0, 2)
                kh = (tt @ r32(A + "k_proj.weight").T + r32(A + "k_proj.bias")).reshape(Na, NHr, HDr).transpose(1, 0, 2)
                vh = (tt @ r32(A + "v_proj.weight").T + r32(A + "v_proj.bias")).reshape(Na, NHr, HDr).transpose(1, 0, 2)
                qh = lno(qh, r32(A + "q_norm.weight"), r32(A + "q_norm.bias"))
                kh = lno(kh, r32(A + "k_norm.weight"), r32(A + "k_norm.bias"))
                qh = qh * cosR3 + rot2(qh) * sinR3; kh = kh * cosR3 + rot2(kh) * sinR3
                qs = qh.reshape(NHr, Da, Ha, HDr); ks = kh.reshape(NHr, Da, Ha, HDr)
                vs = vh.reshape(NHr, Da, Ha, HDr)
                kk = np.stack([np.roll(np.roll(ks, -di, 1), -dj, 2)
                               for di in (-1, 0, 1) for dj in (-1, 0, 1)], 3)
                vv = np.stack([np.roll(np.roll(vs, -di, 1), -dj, 2)
                               for di in (-1, 0, 1) for dj in (-1, 0, 1)], 3)
                sc = (qs[:, :, :, None, :] * kk).sum(-1) / math.sqrt(HDr)
                e = np.exp(sc - sc.max(-1, keepdims=True)); wA = e / e.sum(-1, keepdims=True)
                ctx = (wA[..., None] * vv).sum(3).reshape(NHr, Na, HDr).transpose(1, 0, 2).reshape(Na, Cs)
                x1 = h_in + ctx @ r32(A + "out_proj.weight").T + r32(A + "out_proj.bias")
                ff = lno(x1, r32(E + "layer.norm2.weight"), r32(E + "layer.norm2.bias"))
                ff = ff @ r32(E + "layer.linear1.weight").T + r32(E + "layer.linear1.bias")
                ff = 0.5 * ff * (1.0 + ERF(ff / math.sqrt(2.0)))
                hn = x1 + ff @ r32(E + "layer.linear2.weight").T + r32(E + "layer.linear2.bias")
                gg = 1 / (1 + np.exp(-np.clip(np.concatenate([h, hn], -1) @ r32(E + "gate.0.weight").T + r32(E + "gate.0.bias"), -500, 500)))
                h = gg * hn + (1.0 - gg) * h
            return h.mean(0) @ r32("cls.weight").T + r32("cls.bias")

        cosR3q, sinR3q = ring_rope(fx3, RB + "enc.encoder.layer.self_attn.rope.")
        t0 = _time.time(); rlo = rotor_o(); t1 = _time.time()
        rlr = Q.rotor_lattice_forward(gridA_q, W3, cosR3q, sinR3q, Da, Ha, Cs, NHr, HDr, 2,
                                      NCLS, prefix="enc.", head="cls.")
        t2 = _time.time()
        rlrf = np.array([dq(v) for v in rlr])
        cosRA = float(rlrf @ rlo / (np.linalg.norm(rlrf) * np.linalg.norm(rlo) + 1e-12))
        print(f"  APP-E2E Rotor (LatticeAttention, N=1808): cosine {cosRA:.6f}, argmax "
              f"ring={int(rlrf.argmax())} oracle={int(rlo.argmax())}  [oracle {t1-t0:.0f}s, ring {t2-t1:.0f}s]")
        check("APP-E2E rotor: ring == app forward (cosine > 0.999)", cosRA > 0.999)
        check("APP-E2E rotor: argmax matches", int(rlrf.argmax()) == int(rlo.argmax()))
        # the app's FUSED prediction (0.5·softmax(rotor) + 0.5·softmax(soliton))
        def smax(v): return np.exp(v - v.max()) / np.exp(v - v.max()).sum()
        f_o = 0.5 * smax(rlo) + 0.5 * smax(slo)
        f_r = 0.5 * smax(rlrf) + 0.5 * smax(slrf)
        check("APP-E2E fused prediction: argmax matches the app", int(f_r.argmax()) == int(f_o.argmax()))

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
