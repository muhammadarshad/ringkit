"""RING side of the webapp e2e: quanta.rotor_lattice_forward + quanta.soliton_forward on the
app's real (16,113,128) cubes, vs the TORCH-ANCHORED reference logits in app_e2e_ref.npz.
Run: python3 app_e2e_ring.py  (system python3, ringkit importable from ~/Projects)"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path.home() / "Projects"))
import numpy as np

from ringkit.emulation import checkpoint as ck
from ringkit import quanta
from ringkit.quanta._ringtrig import FRAC, ONE

SP = Path(__file__).parent
REF = np.load(SP / "app_e2e_ref.npz")

CKPT = Path.home() / (".cache/huggingface/hub/models--marshadbits--qcm-rp2k-cloud-matched-results/"
                      "snapshots/73d6c59a6b4299fb7c4684b054cbddccc55b3179/heads_rdt_L2_k2384_best.pth")
SOL = Path.home() / ("Projects/vlm-transformers/huggingface/exported/qcm-rp2k-cloud-matched-results/"
                     "full_mamba2_L2_k2384_h16_gate_lat2d_best.pth")

D, Hh, C, NH, HD = 16, 113, 128, 16, 8
N = D * Hh
NCLS = 2384

print("loading fixed-point weights ...")
t0 = time.time()
fxr = ck.load_fixed(CKPT, frac=FRAC)
fxs = ck.load_fixed(SOL, frac=FRAC)
print(f"  loaded in {time.time()-t0:.1f}s ({len(fxr)} + {len(fxs)} tensors)")

BASE = next(k[:k.index("enc.")] for k in fxr if "enc.quadrant_proj.modulation" in k)
def WR(n): return fxr[n][0]
def WS(n): return fxs[n][0]                      # names already carry the "model." prefix
print(f"  rotor key base: {BASE!r}")

_Q255 = [((u << FRAC) + 127) // 255 for u in range(256)]   # round(u*65536/255): ring ingest of x/255


def grid_from_cube(cube):
    """uint8 cube -> Q16 grid rows (the ring ingest of x/255)."""
    flat = cube.reshape(D * Hh, C)
    return [[_Q255[int(u)] for u in row] for row in flat]


def rope_cache(fx, key):
    cos = np.array(fx[key + "cos_cached"][0]).reshape(fx[key + "cos_cached"][1])[:D, :Hh]
    sin = np.array(fx[key + "sin_cached"][0]).reshape(fx[key + "sin_cached"][1])[:D, :Hh]
    return ([[int(v) for v in r] for r in cos.reshape(N, HD)],
            [[int(v) for v in r] for r in sin.reshape(N, HD)])


cosR, sinR = rope_cache(fxr, BASE + "enc.encoder.layer.self_attn.rope.")
cosS, sinS = rope_cache(fxs, "model.layers.0.rope.")


def report(tag, name, ring_logits, ref):
    r = np.array(ring_logits, dtype=np.float64) / ONE
    o = ref.astype(np.float64)
    cos_ = float(r @ o / (np.linalg.norm(r) * np.linalg.norm(o) + 1e-12))
    am = "OK" if int(r.argmax()) == int(o.argmax()) else f"MISMATCH ring={int(r.argmax())} ref={int(o.argmax())}"
    print(f"[{tag}] {name:8s} cos {cos_:.6f}  maxerr {np.abs(r-o).max():.3e}  argmax {am}")
    return cos_, int(r.argmax()) == int(o.argmax())


ok = True
for tag in ("synth", "shelf"):
    grid_q = grid_from_cube(REF[f"{tag}_cube"])
    t0 = time.time()
    rl = quanta.rotor_lattice_forward(grid_q, WR, cosR, sinR, D, Hh, C, NH, HD, 2, NCLS,
                                      prefix=BASE + "enc.", head=BASE + "cls.")
    tr = time.time() - t0
    c1, a1 = report(tag, f"rotor({tr:.0f}s)", rl, REF[f"{tag}_rotor"])
    t0 = time.time()
    sl = quanta.soliton_forward(grid_q, WS, cosS, sinS, D, Hh, C, NH, HD, 2, NCLS, D, Hh,
                                prefix="model.")
    ts = time.time() - t0
    c2, a2 = report(tag, f"soliton({ts:.0f}s)", sl, REF[f"{tag}_soliton"])
    ok = ok and a1 and a2 and c1 > 0.999 and c2 > 0.999

print("E2E RING vs APP:", "PASS" if ok else "FAIL")
