"""TORCH ANCHOR for the webapp e2e verification (ground truth, non-circular).

Runs the REAL app model code (vlm-transformers repo modules + real weights) on:
  A) a deterministic synthetic real-regime image (aspect-padded -> vacuum tokens; 255s -> arc clamp)
  B) the real shelf photo ~/Pictures/ShelfImages/Picture 1.jpg
and dumps cubes + rotor/soliton logits + fused top-5 to app_e2e_ref.npz.

Also checks whether the mamba2-fixed safetensors == full_mamba2_L2_k2384_h16_gate_lat2d_best.pth.
Run with: ~/.venvs/ringkit-bench/bin/python app_e2e_anchor.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path.home() / "Projects/vlm-transformers"
for p in [ROOT, ROOT / "scratch", ROOT / "huggingface"]:
    sys.path.insert(0, str(p))

from mprc_transformer import MPRCRDT, MPRCMamba2               # noqa: E402
from transformer.lattice_attention import LatticeAttention     # noqa: E402

OUT = Path(__file__).parent / "app_e2e_ref.npz"

# ── app/serve.py wave encode, copied VERBATIM (pure numpy; the app's own preprocessing) ──
H, W, ACT = 113, 128, 112
MOD7 = (np.arange(256) % 7).astype(np.uint8)
SCALES = [0, 1, 4, 16, 64, 128]


def _frame(im):
    w0, h0 = im.size
    s = min(ACT / w0, ACT / h0)
    nw, nh = max(int(round(w0 * s)), 1), max(int(round(h0 * s)), 1)
    r = np.asarray(im.resize((nw, nh), Image.BILINEAR), np.uint16)
    out = np.zeros((H, W, 3), np.uint16)
    ay, ax = (H - nh) // 2, (W - nw) // 2
    out[ay:ay + nh, ax:ax + nw] = r
    return out


def _evolve(m, steps):
    cur = m.astype(np.uint16)
    for _ in range(steps):
        nxt = cur.copy()
        up, dn = cur[:-2, 1:-1], cur[2:, 1:-1]
        lf, rt = cur[1:-1, :-2], cur[1:-1, 2:]
        c = cur[1:-1, 1:-1]
        nxt[1:-1, 1:-1] = (up + dn + lf + rt + (c << 2)) >> 3
        cur = nxt
    return cur.astype(np.uint8)


def _plaq(m):
    M = m.astype(np.uint16)
    up, dn = M[:-2, 1:-1], M[2:, 1:-1]
    lf, rt = M[1:-1, :-2], M[1:-1, 2:]
    f = (rt + up + 512 - lf - dn + 128) & 0xFF
    out = np.full(m.shape, 128, np.uint8)
    out[1:-1, 1:-1] = f.astype(np.uint8)
    return out


def encode_from_frame(arr):
    """encode_pil AFTER _frame: arr is the framed (113,128,3) uint16 array."""
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    s = R + G + B
    luma = (((R * 77 + G * 150 + B * 29) >> 8) & 0xFF).astype(np.uint8)
    gray = ((s // 3) & 0xFF).astype(np.uint8)
    chroma = (np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)).astype(np.uint8)
    u = (((MOD7[R] + MOD7[G] + MOD7[B]) * 14) & 0xFF).astype(np.uint8)
    winding = (((s >> 8) & 0xFF) * 85).astype(np.uint8)
    d1 = ((R - G) & 0xFF).astype(np.uint8)
    d2 = ((G - B) & 0xFF).astype(np.uint8)
    logs = [_plaq(_evolve(gray, t)) for t in SCALES]
    chans = [R.astype(np.uint8), G.astype(np.uint8), B.astype(np.uint8),
             luma, gray, chroma, u, winding, d1, d2] + logs
    return np.stack(chans, 0)                          # (16,113,128) uint8


def encode_pil(im):
    return encode_from_frame(_frame(im.convert("RGB")))


# ── deterministic synthetic real-regime FRAME, pure numpy (identical in the committed test:
#    no PIL): aspect-padded rows (vacuum tokens), white/black saturation, text-like strokes ──
def synth_frame():
    a = np.zeros((H, W, 3), np.uint16)                     # zero padding = vacuum rows
    a[10:103, :, :] = (250, 250, 250)                      # near-white studio bg (93 rows "content")
    a[20:80, 8:56] = (180, 20, 15)                         # red product block
    a[30:52, 16:48] = (255, 255, 255)                      # saturated white label
    a[24:92, 64:120] = (20, 60, 160)                       # blue product block
    a[40:58, 72:112] = (255, 215, 0)                       # gold label
    a[10:14, :, :] = (0, 0, 0)                             # black strip
    for i in range(6):                                     # text-like strokes
        a[34:48, 18 + 5 * i] = (10, 10, 10)
    return a


# ── models (the app's exact construction) ──
CKPT = Path.home() / (".cache/huggingface/hub/models--marshadbits--qcm-rp2k-cloud-matched-results/"
                      "snapshots/73d6c59a6b4299fb7c4684b054cbddccc55b3179/heads_rdt_L2_k2384_best.pth")
EXPORT = ROOT / "huggingface/exported/qcm-rp2k-cloud-matched-mamba2-fixed/model.safetensors"
GATED_PTH = ROOT / ("huggingface/exported/qcm-rp2k-cloud-matched-results/"
                    "full_mamba2_L2_k2384_h16_gate_lat2d_best.pth")


class RotorCls(torch.nn.Module):                      # enc + cls only (ocr/bc heads not needed)
    def __init__(self, n):
        super().__init__()
        self.enc = MPRCRDT(embed_dim=128, depth=2, num_heads=16, use_lattice=False, rope_4d=True)
        self.cls = torch.nn.Linear(128, n)


def load_rotor():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    n = sd["cls.weight"].shape[0]
    m = RotorCls(n)
    m.enc.encoder.layer.self_attn = LatticeAttention(128, 16, radius=1, rope_4d=True)
    keep = {k: v for k, v in sd.items() if k.startswith(("enc.", "cls."))}
    missing, unexpected = m.load_state_dict(keep, strict=False)
    assert not [k for k in missing if not k.startswith(("enc.lattice",))], f"missing: {missing[:8]}"
    print(f"[rotor] loaded ({n} classes); dropped {len(sd) - len(keep)} ocr/bc tensors")
    return m.eval()


def load_soliton():
    from safetensors.torch import load_file
    sd = load_file(str(EXPORT))
    m = MPRCMamba2(embed_dim=128, num_layers=2, num_heads=16, use_lattice=False,
                   rope_4d=True, use_gate=True, use_causal=False, use_lattice2d=True)
    n = sd["backbone.head.weight"].shape[0]
    m.head = torch.nn.Linear(128, n)
    body = {k.removeprefix("backbone."): v for k, v in sd.items() if k.startswith("backbone.")}
    missing, unexpected = m.load_state_dict(body, strict=False)
    assert not [k for k in missing if "lattice" not in k], f"missing: {missing[:8]}"
    print(f"[soliton] loaded export ({n} classes); unexpected={unexpected[:4]}")
    # is the export the same weights as the gated .pth?
    import io, zipfile
    z = zipfile.ZipFile(GATED_PTH)
    same = None
    try:
        gs = torch.load(GATED_PTH, map_location="cpu", weights_only=False)
        gs = gs["model"] if isinstance(gs, dict) and "model" in gs else gs
        gs = {k.removeprefix("_orig_mod.").removeprefix("model."): v for k, v in gs.items()}
        probe = [k for k in ("layers.0.in_proj.weight", "head.weight") if k in gs and k in body | {"head.weight": m.head.weight}]
        same = all(torch.equal(gs[k], body.get(k, m.head.weight if k == "head.weight" else None))
                   for k in ("layers.0.in_proj.weight",) if k in gs)
        hw = torch.equal(gs["head.weight"], m.head.weight) if "head.weight" in gs else None
        print(f"[soliton] export == gated .pth? in_proj: {same}  head: {hw}")
    except Exception as e:
        print(f"[soliton] .pth compare skipped: {e}")
    return m.eval()


def run(m_rotor, m_soliton, src, tag):
    cube = encode_from_frame(src) if isinstance(src, np.ndarray) else encode_pil(src)
    x = torch.from_numpy(cube.astype(np.float32))[None]
    with torch.no_grad():
        mem = m_rotor.enc.encode_sequence(x)
        feat = mem.mean(1)
        rlog = m_rotor.cls(feat)[0]
        slog = m_soliton.head(m_soliton.encode_sequence(x).mean(1))[0]
    rp, sp = rlog.softmax(-1), slog.softmax(-1)
    fused = 0.5 * rp + 0.5 * sp
    t5 = fused.topk(5)
    print(f"[{tag}] rotor argmax {int(rlog.argmax())}  soliton argmax {int(slog.argmax())}  "
          f"fused top5 {t5.indices.tolist()} probs {[round(float(v)*100,2) for v in t5.values]}")
    vac = (torch.from_numpy(cube.astype(np.float32) / 255.0).mean(-1) < 1e-3).sum()
    sat = int((cube == 255).sum())
    print(f"[{tag}] vacuum tokens: {int(vac)}/1808   saturated bytes: {sat}")
    return cube, rlog.numpy(), slog.numpy()


if __name__ == "__main__":
    torch.manual_seed(0)
    rot = load_rotor()
    sol = load_soliton()
    out = {}
    for tag, im in (("synth", synth_frame()),
                    ("shelf", Image.open(Path.home() / "Pictures/ShelfImages/Picture 1.jpg"))):
        cube, rl, sl = run(rot, sol, im, tag)
        out[f"{tag}_cube"] = cube
        out[f"{tag}_rotor"] = rl
        out[f"{tag}_soliton"] = sl
    np.savez(OUT, **out)
    print(f"wrote {OUT}")
