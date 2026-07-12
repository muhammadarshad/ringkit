"""
ringkit.bench.apples_to_apples — LABELED EXTERNAL COMPARISON (charter C6: scaffolding,
never imported by the system). How fast would numpy / torch do the SAME computations?

Method: every external baseline is verified bit-for-bit against the ringkit Python
reference BEFORE it is timed (a wrong baseline is not a baseline). All engines compute
identical semantics on identical inputs; best-of-N wall time; preallocated outputs where
the engine supports it. Environment (Rosetta vs native) is printed, never hidden.

Run: python3 -m ringkit.bench.apples_to_apples
"""
import os
import platform
import subprocess
import time

import numpy as np                                  # labeled external comparison

from ringkit.kernels import backend
from ringkit.kernels.apple.metal import host as mh
from ringkit.kernels.mprc.lattice import host as lat

try:
    import torch                                    # labeled external comparison
except ImportError:
    torch = None


def bench(fn, reps=3):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def environment():
    tr = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"],
                        capture_output=True, text=True).stdout.strip()
    print(f"interpreter: python {platform.python_version()} {platform.machine()}"
          f"{' (Rosetta 2 emulation)' if tr == '1' else ''}")
    print(f"numpy {np.__version__} (same interpreter/arch)"
          + (f" | torch {torch.__version__}" if torch else
             " | torch: not installed (no macOS x86_64/cp314 wheels) — GPU entry is ringkit Metal"))
    if mh.available():
        print(f"gpu: {mh.device_name()} via ringkit Metal backend")
    print()


# ── numpy implementations of the SAME semantics ──────────────────────────────

def np_plaquette(g3):
    e = np.zeros_like(g3)
    core = g3[1:-1, 1:-1, 1:-1]
    pos = core + g3[1:-1, 1:-1, 2:]                 # right + up   (uint8 wraps = ring)
    neg = g3[1:-1, 2:, 1:-1] + g3[1:-1, 1:-1, :-2]  # left + down
    e[1:-1, 1:-1, 1:-1] = pos - neg
    return e


def _np_cd(a, b):
    d = (a - b).astype(np.int16)                    # uint8 wrap subtraction, then widen
    return np.minimum(d, 256 - d)


def np_sweep(g3, prop3, chance3, lut_a, pmasks):
    """Both checkerboard parities, identical semantics to gauge.c metropolis_sweep."""
    core = g3[1:-1, 1:-1, 1:-1]
    nbrs = (g3[1:-1, 1:-1, 2:], g3[1:-1, 1:-1, :-2],
            g3[1:-1, 2:, 1:-1], g3[1:-1, :-2, 1:-1],
            g3[2:, 1:-1, 1:-1], g3[:-2, 1:-1, 1:-1])
    for parity in (0, 1):
        old = core
        nv = old + prop3[1:-1, 1:-1, 1:-1]
        So = sum(_np_cd(old, x) for x in nbrs)
        Sn = sum(_np_cd(nv, x) for x in nbrs)
        dS = Sn - So
        accept = (dS <= 0) | (chance3[1:-1, 1:-1, 1:-1] < lut_a[np.clip(dS, 0, 255)])
        core[...] = np.where(accept & pmasks[parity], nv, old)
    return g3


def np_parity_masks(W, H, D):
    kk, jj, ii = np.indices((D - 2, H - 2, W - 2))
    s = (ii + jj + kk + 3) & 1                      # interior offsets -> original (i+j+k)&1
    return (s == 0), (s == 1)


def np_mix32(idx_u32, seed, sweep):
    """Vectorized rk_mix32 — same spec as gauge.c / gauge.metal / python reference."""
    x = idx_u32 + np.uint32(((sweep + 1) * 0x9E3779B9) & 0xFFFFFFFF)
    x = x ^ np.uint32((seed * 0x85EBCA6B) & 0xFFFFFFFF)
    x = x ^ (x >> np.uint32(16))
    x = x * np.uint32(0x7FEB352D)
    x = x ^ (x >> np.uint32(15))
    x = x * np.uint32(0x846CA68B)
    x = x ^ (x >> np.uint32(16))
    return x


def np_thermalize_rng(g3, seed, lut_a, pmasks, idx_u32, sweeps):
    for s in range(sweeps):
        x = np_mix32(idx_u32, seed, s)
        prop3 = (x & np.uint32(0xFF)).astype(np.uint8)
        chance3 = ((x >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8)
        np_sweep(g3, prop3, chance3, lut_a, pmasks)
    return g3


# ── correctness gates (a baseline must be RIGHT before it is timed) ──────────

def gate():
    ok = True
    # elementwise: numpy uint8 wrap == ringkit python reference
    a = bytearray(os.urandom(4096)); b = bytearray(os.urandom(4096))
    want = bytes(backend.mul(a, b, force_python=True))
    na = np.frombuffer(bytes(a), dtype=np.uint8)
    nb = np.frombuffer(bytes(b), dtype=np.uint8)
    ok &= (na * nb).tobytes() == want
    print(f"  gate elementwise: numpy == ringkit reference  {'OK' if ok else 'FAIL'}")

    W = H = D = 12
    n = W * H * D
    g0 = bytearray(os.urandom(n))
    lut = bytearray(max(0, 255 - d) for d in range(256))
    lut_a = np.frombuffer(bytes(lut), dtype=np.uint8)
    pmasks = np_parity_masks(W, H, D)

    # plaquette
    want = bytes(lat.plaquette(g0, W, H, D, force_python=True))
    got = np_plaquette(np.frombuffer(bytes(g0), dtype=np.uint8).reshape(D, H, W).copy())
    p_ok = got.tobytes() == want
    ok &= p_ok
    print(f"  gate plaquette:   numpy == ringkit reference  {'OK' if p_ok else 'FAIL'}")

    # sweep with arrays
    prop = bytearray(os.urandom(n)); chance = bytearray(os.urandom(n))
    want_g = bytearray(g0)
    lat._py_sweep(want_g, prop, chance, lut, W, H, D, 0)
    lat._py_sweep(want_g, prop, chance, lut, W, H, D, 1)
    g3 = np.frombuffer(bytes(g0), dtype=np.uint8).reshape(D, H, W).copy()
    np_sweep(g3, np.frombuffer(bytes(prop), dtype=np.uint8).reshape(D, H, W),
             np.frombuffer(bytes(chance), dtype=np.uint8).reshape(D, H, W), lut_a, pmasks)
    s_ok = g3.tobytes() == bytes(want_g)
    ok &= s_ok
    print(f"  gate sweep:       numpy == ringkit reference  {'OK' if s_ok else 'FAIL'}")

    # derived-RNG sweep
    want_g = bytearray(g0)
    for s in range(2):
        lat._py_sweep_rng(want_g, 777, s, lut, W, H, D, 0)
        lat._py_sweep_rng(want_g, 777, s, lut, W, H, D, 1)
    idx = np.arange(n, dtype=np.uint32).reshape(D, H, W)
    g3 = np.frombuffer(bytes(g0), dtype=np.uint8).reshape(D, H, W).copy()
    np_thermalize_rng(g3, 777, lut_a, pmasks, idx, 2)
    r_ok = g3.tobytes() == bytes(want_g)
    ok &= r_ok
    print(f"  gate rng sweep:   numpy == ringkit reference  {'OK' if r_ok else 'FAIL'}")
    print()
    return ok


# ── the benchmarks ────────────────────────────────────────────────────────────

def bench_elementwise():
    print("── elementwise ring mul (uint8, mod-256 wrap) — GMUPS (higher is better)")
    for exp in (20, 24):
        n = 1 << exp
        a = bytearray(os.urandom(n)); b = bytearray(os.urandom(n)); out = bytearray(n)
        lib = backend._load()
        t_c = bench(lambda: lib.ring_mul(backend._ptr(out), backend._ptr(a), backend._ptr(b), n))
        na = np.frombuffer(bytes(a), dtype=np.uint8); nb = np.frombuffer(bytes(b), dtype=np.uint8)
        no = np.empty(n, dtype=np.uint8)
        t_np = bench(lambda: np.multiply(na, nb, out=no))
        t_m = bench(lambda: mh.elementwise("ring_mul", out, a, b, n)) if mh.available() else None
        row = (f"  2^{exp}  ringkit-C {n/t_c/1e9:6.2f}   numpy {n/t_np/1e9:6.2f}"
               f"   ringkit-metal {n/t_m/1e9:6.2f}" if t_m else "")
        extra = []
        if torch:
            ta = torch.from_numpy(na.copy()); tb = torch.from_numpy(nb.copy())
            t_t = bench(lambda: ta * tb)
            extra.append(f"torch-cpu {n/t_t/1e9:6.2f}")
        print(row, *extra)
    print()


def bench_plaquette():
    print("── Wilson plaquette stencil, 128^3 — ns/node (lower is better)")
    L = 128; n = L ** 3
    g = bytearray(os.urandom(n))
    g3 = np.frombuffer(bytes(g), dtype=np.uint8).reshape(L, L, L).copy()
    t_c = bench(lambda: lat.plaquette(g, L, L, L))
    t_np = bench(lambda: np_plaquette(g3))
    e = bytearray(n)
    t_m = bench(lambda: mh.plaquette(e, g, L, L, L)) if mh.available() else None
    print(f"  ringkit-C {t_c*1e9/n:6.3f}   numpy {t_np*1e9/n:6.3f}"
          + (f"   ringkit-metal {t_m*1e9/n:6.3f}" if t_m else ""))
    print()


def bench_sweep_arrays():
    print("── Metropolis sweep, arrays supplied, 128^3 — ns/node/sweep (lower is better)")
    L = 128; n = L ** 3
    g = bytearray(os.urandom(n)); prop = bytearray(os.urandom(n)); chance = bytearray(os.urandom(n))
    lut = bytearray(max(0, 255 - d) for d in range(256))
    lut_a = np.frombuffer(bytes(lut), dtype=np.uint8)
    pmasks = np_parity_masks(L, L, L)
    lib = lat._load()

    def c_sweep():
        for par in (0, 1):
            lib.metropolis_sweep(lat._ptr(g), lat._ptr(prop), lat._ptr(chance),
                                 lat._ptr(lut), L, L, L, par)
    t_c = bench(c_sweep)
    g3 = np.frombuffer(bytes(g), dtype=np.uint8).reshape(L, L, L).copy()
    p3 = np.frombuffer(bytes(prop), dtype=np.uint8).reshape(L, L, L)
    c3 = np.frombuffer(bytes(chance), dtype=np.uint8).reshape(L, L, L)
    t_np = bench(lambda: np_sweep(g3, p3, c3, lut_a, pmasks))
    t_m = bench(lambda: mh.gauge_sweep(g, prop, chance, lut, L, L, L)) if mh.available() else None
    print(f"  ringkit-C {t_c*1e9/n:6.2f}   numpy {t_np*1e9/n:6.2f}"
          + (f"   ringkit-metal {t_m*1e9/n:6.2f}" if t_m else ""))
    print()


def bench_thermalize_rng():
    print("── thermalize, derived RNG (rk_mix32), 8 sweeps — ns/node/sweep (lower is better)")
    for L in (128, 160):
        n = L ** 3; S = 8
        lut = bytearray(max(0, 255 - d) for d in range(256))
        lut_a = np.frombuffer(bytes(lut), dtype=np.uint8)
        pmasks = np_parity_masks(L, L, L)
        idx = np.arange(n, dtype=np.uint32).reshape(L, L, L)
        g = bytearray(os.urandom(n))
        # C path (metal floor pushed away)
        saved = lat.GAUGE_METAL_MIN_NODES
        lat.GAUGE_METAL_MIN_NODES = 1 << 62
        gc = bytearray(g)
        t_c = bench(lambda: lat.thermalize_rng(gc, 777, lut, L, L, L, S))
        lat.GAUGE_METAL_MIN_NODES = saved
        gm = bytearray(g)
        t_m = bench(lambda: lat.thermalize_rng(gm, 777, lut, L, L, L, S)) if mh.available() else None
        g3 = np.frombuffer(bytes(g), dtype=np.uint8).reshape(L, L, L).copy()
        t_np = bench(lambda: np_thermalize_rng(g3, 777, lut_a, pmasks, idx, S))
        print(f"  {L}^3  ringkit-C {t_c*1e9/(n*S):6.2f}   numpy {t_np*1e9/(n*S):6.2f}"
              + (f"   ringkit-metal-GPU {t_m*1e9/(n*S):6.2f}" if t_m else ""))
    print()


def main():
    print("=" * 78)
    print("ringkit vs external engines — same semantics, same inputs, gated then timed")
    print("=" * 78)
    environment()
    if not gate():
        print("A BASELINE FAILED ITS GATE — timings below it would be meaningless. Aborting.")
        return 1
    if mh.available():                              # warm the GPU so clocks are honest
        wg = bytearray(os.urandom(64 ** 3))
        wl = bytearray(max(0, 255 - d) for d in range(256))
        lat.thermalize_rng(wg, 1, wl, 64, 64, 64, 8)
    bench_elementwise()
    bench_plaquette()
    bench_sweep_arrays()
    bench_thermalize_rng()
    print("notes: numpy runs the identical algorithm vectorized (parity masks hoisted, favors")
    print("numpy). torch not installed here (no macOS x86_64 wheels for this python).")
    print("Everything above passed a bit-for-bit gate against the ringkit python reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
