"""NUMPY ORACLE of the webapp forward (rotor-lattice + soliton), validated vs the torch anchor.
This is the math the committed test will inline (torch here is only a weight LOADER).
Run: ~/.venvs/ringkit-bench/bin/python app_e2e_oracle.py"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path(__file__).parent
REF = np.load(SP / "app_e2e_ref.npz")

CKPT = Path.home() / (".cache/huggingface/hub/models--marshadbits--qcm-rp2k-cloud-matched-results/"
                      "snapshots/73d6c59a6b4299fb7c4684b054cbddccc55b3179/heads_rdt_L2_k2384_best.pth")
SOL = Path.home() / ("Projects/vlm-transformers/huggingface/exported/qcm-rp2k-cloud-matched-results/"
                     "full_mamba2_L2_k2384_h16_gate_lat2d_best.pth")

D, Hh, C, NH, HD = 16, 113, 128, 16, 8
N = D * Hh
ERF = np.vectorize(math.erf)

# ── rotor weights (torch as loader only) ──
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
RW = {k2: v.numpy().astype(np.float64) for k, v in sd.items()
      if (k2 := k.removeprefix("_orig_mod.")).startswith(("enc.", "cls."))}
sd2 = torch.load(SOL, map_location="cpu", weights_only=False)
sd2 = sd2["model"] if isinstance(sd2, dict) and "model" in sd2 else sd2
SW = {"model." + k.removeprefix("_orig_mod."): v.numpy().astype(np.float64) for k, v in sd2.items()}


def sig(v): return 1.0 / (1.0 + np.exp(-v))
def ln(v, g, b, eps=1e-5):
    m = v.mean(-1, keepdims=True)
    return (v - m) / np.sqrt(((v - m) ** 2).mean(-1, keepdims=True) + eps) * g + b
def gelu_erf(v): return 0.5 * v * (1.0 + ERF(v / math.sqrt(2.0)))
def rmsn(y, g): return y / np.sqrt((y * y).mean(-1, keepdims=True) + 1e-5) * g
def softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def rot_half_4d(v):
    """QuantumRoPE4D rotate-half at hd=8: FOUR 2-dim chunks, each (-x1, x0)."""
    o = np.empty_like(v)
    for i in range(4):
        o[..., 2 * i] = -v[..., 2 * i + 1]
        o[..., 2 * i + 1] = v[..., 2 * i]
    return o


def frontend(grid, W, pfx):
    arc = np.clip((grid * 256.0).astype(np.int64), 0, 255)      # (D,Hh,C)
    # ring trig from the ckpt modulation is already baked; recompute ring cos/sin table:
    SCALE = 21
    def _arch(p, hp): return 0 if (p <= 0 or p >= hp) else SCALE * 2 * math.isqrt(p * (hp - p)) // hp
    def _SIN(p):
        p &= 0xFF
        return _arch(p, 128) if p < 128 else (-_arch(p - 128, 128)) % 256
    def _sg(x): return x - 256 if x > 128 else x
    COSF = np.array([_sg(_SIN((p + 64) & 0xFF)) / SCALE for p in range(256)])
    SINF = np.array([_sg(_SIN(p)) / SCALE for p in range(256)])
    c, s = COSF[arc], SINF[arc]
    stk = np.concatenate([np.clip(c, 0, None), np.clip(s, 0, None),
                          np.clip(-c, 0, None), np.clip(-s, 0, None)], -1).reshape(N, C * 4)
    stk = stk * W[pfx + "quadrant_proj.modulation"]
    z = stk @ W[pfx + "quadrant_proj.proj.weight"].T + W[pfx + "quadrant_proj.proj.bias"]
    vac = (grid.mean(-1) < 1e-3).reshape(N, 1)                  # (N,1) vacuum mask
    depths = np.repeat(np.arange(D), Hh)
    demb = W[pfx + "vacuum_emb.depth_emb.weight"][depths]
    vemb = W[pfx + "vacuum_emb.vacuum_emb"].reshape(1, -1)
    return z + demb * (1.0 - vac) + vemb * vac


def window_attention(q, k, v):
    """q,k,v: (NH,N,HD) on the (D,Hh) torus, radius-1 (9-neighbour) window."""
    qs = q.reshape(NH, D, Hh, HD); ks = k.reshape(NH, D, Hh, HD); vs = v.reshape(NH, D, Hh, HD)
    kk = np.stack([np.roll(np.roll(ks, -di, 1), -dj, 2)
                   for di in (-1, 0, 1) for dj in (-1, 0, 1)], 3)   # (NH,D,Hh,9,HD)
    vv = np.stack([np.roll(np.roll(vs, -di, 1), -dj, 2)
                   for di in (-1, 0, 1) for dj in (-1, 0, 1)], 3)
    sc = (qs[:, :, :, None, :] * kk).sum(-1) / math.sqrt(HD)        # (NH,D,Hh,9)
    e = np.exp(sc - sc.max(-1, keepdims=True)); w = e / e.sum(-1, keepdims=True)
    return (w[..., None] * vv).sum(3).reshape(NH, N, HD)


def rotor_forward(grid):
    E = "enc.encoder."; A = E + "layer.self_attn."
    cos = RW[A + "rope.cos_cached"][:D, :Hh].reshape(N, HD)
    sin = RW[A + "rope.sin_cached"][:D, :Hh].reshape(N, HD)
    z = frontend(grid, RW, "enc.")
    x0 = z.copy(); h = z.copy()
    for step in range(2):
        al = sig(x0 @ RW[E + "inject_gate.0.weight"].T + RW[E + "inject_gate.0.bias"])
        h_in = h + al * x0 + RW[E + "depth_embed.weight"][step]
        t = ln(h_in, RW[E + "layer.norm1.weight"], RW[E + "layer.norm1.bias"])
        qh = (t @ RW[A + "q_proj.weight"].T + RW[A + "q_proj.bias"]).reshape(N, NH, HD).transpose(1, 0, 2)
        kh = (t @ RW[A + "k_proj.weight"].T + RW[A + "k_proj.bias"]).reshape(N, NH, HD).transpose(1, 0, 2)
        vh = (t @ RW[A + "v_proj.weight"].T + RW[A + "v_proj.bias"]).reshape(N, NH, HD).transpose(1, 0, 2)
        qh = ln(qh, RW[A + "q_norm.weight"], RW[A + "q_norm.bias"])
        kh = ln(kh, RW[A + "k_norm.weight"], RW[A + "k_norm.bias"])
        qh = qh * cos + rot_half_4d(qh) * sin
        kh = kh * cos + rot_half_4d(kh) * sin
        ctx = window_attention(qh, kh, vh).transpose(1, 0, 2).reshape(N, C)
        x1 = h_in + ctx @ RW[A + "out_proj.weight"].T + RW[A + "out_proj.bias"]
        f = ln(x1, RW[E + "layer.norm2.weight"], RW[E + "layer.norm2.bias"])
        f = gelu_erf(f @ RW[E + "layer.linear1.weight"].T + RW[E + "layer.linear1.bias"])
        hn = x1 + f @ RW[E + "layer.linear2.weight"].T + RW[E + "layer.linear2.bias"]
        g = sig(np.concatenate([h, hn], -1) @ RW[E + "gate.0.weight"].T + RW[E + "gate.0.bias"])
        h = g * hn + (1.0 - g) * h
    feat = h.mean(0)
    return feat @ RW["cls.weight"].T + RW["cls.bias"]


def soliton_forward(grid):
    M = "model."
    cos = SW[M + "layers.0.rope.cos_cached"][:D, :Hh].reshape(N, HD)
    sin = SW[M + "layers.0.rope.sin_cached"][:D, :Hh].reshape(N, HD)
    dstep = [-1 if hh < NH // 2 else min(16, 1 << (hh - NH // 2)) for hh in range(NH)]
    x = frontend(grid, SW, M)
    for li in range(2):
        L = f"{M}layers.{li}."
        proj = x @ SW[L + "in_proj.weight"].T + SW[L + "in_proj.bias"]
        q, k, v, dl = proj[:, :C], proj[:, C:2 * C], proj[:, 2 * C:3 * C], proj[:, 3 * C:]
        qh = q.reshape(N, NH, HD).transpose(1, 0, 2); kh = k.reshape(N, NH, HD).transpose(1, 0, 2)
        vh = v.reshape(N, NH, HD).transpose(1, 0, 2)
        dh = softplus(dl).T[:, :, None]                                  # (NH,N,1)
        qh = qh * cos + rot_half_4d(qh) * sin
        kh = kh * cos + rot_half_4d(kh) * sin
        s = dh * (kh * vh)                                               # (NH,N,HD)
        out = np.empty_like(s)
        gs = s.sum(1, keepdims=True)
        cur = s.reshape(NH, D, Hh, HD).copy()
        for hh in range(NH):
            if dstep[hh] < 0:
                out[hh] = gs[hh]
        mt = max(t for t in dstep if t >= 0)
        for st in range(1, mt + 1):
            cur = (np.roll(cur, 1, 1) + np.roll(cur, -1, 1) + np.roll(cur, 1, 2)
                   + np.roll(cur, -1, 2) + 4.0 * cur) / 8.0
            for hh in range(NH):
                if dstep[hh] == st:
                    out[hh] = N * cur[hh].reshape(N, HD)
        y = out.transpose(1, 0, 2).reshape(N, C) * 0
        y = (qh * out).transpose(1, 0, 2).reshape(N, C)
        y = rmsn(y, SW[L + "norm.weight"])
        w = np.exp(x @ SW[L + "gate_route.weight"].T + SW[L + "gate_route.bias"])
        w = w / w.sum(-1, keepdims=True)
        g = x @ SW[L + "gate_proj.weight"].T + SW[L + "gate_proj.bias"] + w @ SW[L + "arm_bias"]
        y = y * sig(g)
        x = y @ SW[L + "out_proj.weight"].T + SW[L + "out_proj.bias"]
    feat = x.mean(0)
    return feat @ SW[M + "head.weight"].T + SW[M + "head.bias"]


for tag in ("synth", "shelf"):
    grid = REF[f"{tag}_cube"].astype(np.float64) / 255.0
    for name, fn, ref in (("rotor", rotor_forward, REF[f"{tag}_rotor"]),
                          ("soliton", soliton_forward, REF[f"{tag}_soliton"])):
        got = fn(grid)
        err = np.abs(got - ref).max()
        cos_ = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref)))
        am = "OK" if int(got.argmax()) == int(ref.argmax()) else f"MISMATCH {int(got.argmax())}!={int(ref.argmax())}"
        print(f"[{tag}] {name:8s} maxerr {err:.3e}  cos {cos_:.8f}  argmax {am}")
