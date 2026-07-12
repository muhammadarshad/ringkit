"""
ringkit.bench.apples_to_apples — LABELED EXTERNAL COMPARISON (charter C6: scaffolding,
never imported by the system). How fast do numpy / torch (cpu + MPS on the same unified
GPU) run the SAME computations?

Method: every external baseline is verified bit-for-bit against the ringkit Python
reference BEFORE it is timed (a wrong baseline is not a baseline; a gated-out engine is
reported, not hidden). Identical semantics on identical inputs; best-of-N wall time; MPS
timings synchronize; external engines run device-resident tensors (their best case) while
ringkit-metal timings INCLUDE its host-buffer copies (our real API cost).

Run: python3 -m ringkit.bench.apples_to_apples          (any interpreter with numpy)
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


def _sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def environment():
    tr = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"],
                        capture_output=True, text=True).stdout.strip()
    print(f"interpreter: python {platform.python_version()} {platform.machine()}"
          f"{' (Rosetta 2 emulation)' if tr == '1' else ' (native)'}")
    print(f"numpy {np.__version__}"
          + (f" | torch {torch.__version__} (mps available: {torch.backends.mps.is_available()})"
             if torch else " | torch: not installed in this interpreter"))
    if mh.available():
        print(f"gpu: {mh.device_name()} via ringkit Metal backend"
              + (" AND torch-mps (same unified GPU)" if torch and torch.backends.mps.is_available() else ""))
    print()


def torch_devices():
    if not torch:
        return []
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


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


# ── torch implementations of the SAME semantics (cpu and mps) ────────────────

def _i32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _shr(x, k):
    """Logical right shift on int32 (torch >> is arithmetic)."""
    return (x >> k) & ((1 << (32 - k)) - 1)


def t_cd(a, b):
    d = (a - b).to(torch.int16)
    return torch.minimum(d, 256 - d)


def t_sweep(g3, prop3, chance3, lut_t, pmasks):
    core = g3[1:-1, 1:-1, 1:-1]
    nbrs = (g3[1:-1, 1:-1, 2:], g3[1:-1, 1:-1, :-2],
            g3[1:-1, 2:, 1:-1], g3[1:-1, :-2, 1:-1],
            g3[2:, 1:-1, 1:-1], g3[:-2, 1:-1, 1:-1])
    for parity in (0, 1):
        old = core.clone()
        nv = old + prop3[1:-1, 1:-1, 1:-1]
        So = sum(t_cd(old, x) for x in nbrs)
        Sn = sum(t_cd(nv, x) for x in nbrs)
        dS = Sn - So
        accept = (dS <= 0) | (chance3[1:-1, 1:-1, 1:-1] < lut_t[torch.clamp(dS, 0, 255).long()])
        core.copy_(torch.where(accept & pmasks[parity], nv, old))
    return g3


def t_plaquette(g3):
    e = torch.zeros_like(g3)
    core = g3[1:-1, 1:-1, 1:-1]
    pos = core + g3[1:-1, 1:-1, 2:]
    neg = g3[1:-1, 2:, 1:-1] + g3[1:-1, 1:-1, :-2]
    e[1:-1, 1:-1, 1:-1] = pos - neg
    return e


def t_mix32(idx_i32, seed, sweep):
    x = idx_i32 + _i32((sweep + 1) * 0x9E3779B9)
    x = x ^ _i32(seed * 0x85EBCA6B)
    x = x ^ _shr(x, 16)
    x = x * _i32(0x7FEB352D)
    x = x ^ _shr(x, 15)
    x = x * _i32(0x846CA68B)
    x = x ^ _shr(x, 16)
    return x


def t_thermalize_rng(g3, seed, lut_t, pmasks, idx_i32, sweeps):
    for s in range(sweeps):
        x = t_mix32(idx_i32, seed, s)
        prop3 = (x & 0xFF).to(torch.uint8)
        chance3 = (_shr(x, 8) & 0xFF).to(torch.uint8)
        t_sweep(g3, prop3, chance3, lut_t, pmasks)
    return g3


def t_parity_masks(W, H, D, dev):
    m0, m1 = np_parity_masks(W, H, D)
    return (torch.from_numpy(m0).to(dev), torch.from_numpy(m1).to(dev))


# ── correctness gates (a baseline must be RIGHT before it is timed) ──────────

def gate_numpy():
    ok = True
    a = bytearray(os.urandom(4096)); b = bytearray(os.urandom(4096))
    want = bytes(backend.mul(a, b, force_python=True))
    na = np.frombuffer(bytes(a), dtype=np.uint8)
    nb = np.frombuffer(bytes(b), dtype=np.uint8)
    ok &= (na * nb).tobytes() == want

    W = H = D = 12
    n = W * H * D
    g0, prop, chance, lut, want_sweep, want_rng = _gate_fixture(W, H, D)
    lut_a = np.frombuffer(bytes(lut), dtype=np.uint8)
    pmasks = np_parity_masks(W, H, D)

    got = np_plaquette(np.frombuffer(bytes(g0), dtype=np.uint8).reshape(D, H, W).copy())
    ok &= got.tobytes() == bytes(lat.plaquette(g0, W, H, D, force_python=True))

    g3 = np.frombuffer(bytes(g0), dtype=np.uint8).reshape(D, H, W).copy()
    np_sweep(g3, np.frombuffer(bytes(prop), dtype=np.uint8).reshape(D, H, W),
             np.frombuffer(bytes(chance), dtype=np.uint8).reshape(D, H, W), lut_a, pmasks)
    ok &= g3.tobytes() == bytes(want_sweep)

    idx = np.arange(n, dtype=np.uint32).reshape(D, H, W)
    g3 = np.frombuffer(bytes(g0), dtype=np.uint8).reshape(D, H, W).copy()
    np_thermalize_rng(g3, 777, lut_a, pmasks, idx, 2)
    ok &= g3.tobytes() == bytes(want_rng)
    print(f"  gate numpy (elementwise+plaquette+sweep+rng): {'OK' if ok else 'FAIL'}")
    return ok


def gate_torch(dev):
    if not torch:
        return False
    try:
        W = H = D = 12
        n = W * H * D
        g0, prop, chance, lut, want_sweep, want_rng = _gate_fixture(W, H, D)
        lut_t = torch.from_numpy(np.frombuffer(bytes(lut), dtype=np.uint8).copy()).to(dev)
        pmasks = t_parity_masks(W, H, D, dev)
        to3 = lambda ba: torch.from_numpy(
            np.frombuffer(bytes(ba), dtype=np.uint8).reshape(D, H, W).copy()).to(dev)

        e = t_plaquette(to3(g0))
        _sync(dev)
        ok = bytes(e.cpu().numpy()) == bytes(lat.plaquette(g0, W, H, D, force_python=True))

        g3 = to3(g0)
        t_sweep(g3, to3(prop), to3(chance), lut_t, pmasks)
        _sync(dev)
        ok &= bytes(g3.cpu().numpy()) == bytes(want_sweep)

        idx = torch.arange(n, dtype=torch.int32).reshape(D, H, W).to(dev)
        g3 = to3(g0)
        t_thermalize_rng(g3, 777, lut_t, pmasks, idx, 2)
        _sync(dev)
        ok &= bytes(g3.cpu().numpy()) == bytes(want_rng)
        print(f"  gate torch-{dev} (plaquette+sweep+rng): {'OK' if ok else 'FAIL'}")
        return bool(ok)
    except Exception as e:
        print(f"  gate torch-{dev}: EXCLUDED ({type(e).__name__}: {e})")
        return False


_FIXTURE = {}


def _gate_fixture(W, H, D):
    if not _FIXTURE:
        n = W * H * D
        g0 = bytearray(os.urandom(n))
        prop = bytearray(os.urandom(n)); chance = bytearray(os.urandom(n))
        lut = bytearray(max(0, 255 - d) for d in range(256))
        want_sweep = bytearray(g0)
        lat._py_sweep(want_sweep, prop, chance, lut, W, H, D, 0)
        lat._py_sweep(want_sweep, prop, chance, lut, W, H, D, 1)
        want_rng = bytearray(g0)
        for s in range(2):
            lat._py_sweep_rng(want_rng, 777, s, lut, W, H, D, 0)
            lat._py_sweep_rng(want_rng, 777, s, lut, W, H, D, 1)
        _FIXTURE.update(g0=g0, prop=prop, chance=chance, lut=lut,
                        want_sweep=want_sweep, want_rng=want_rng)
    f = _FIXTURE
    return f["g0"], f["prop"], f["chance"], f["lut"], f["want_sweep"], f["want_rng"]


# ── the benchmarks ────────────────────────────────────────────────────────────

def bench_elementwise(tdevs):
    print("── elementwise ring mul (uint8, mod-256 wrap) — GMUPS (higher is better)")
    for exp in (20, 24):
        n = 1 << exp
        a = bytearray(os.urandom(n)); b = bytearray(os.urandom(n)); out = bytearray(n)
        lib = backend._load()
        cols = [f"ringkit-C {n/bench(lambda: lib.ring_mul(backend._ptr(out), backend._ptr(a), backend._ptr(b), n))/1e9:6.2f}"]
        na = np.frombuffer(bytes(a), dtype=np.uint8); nb = np.frombuffer(bytes(b), dtype=np.uint8)
        no = np.empty(n, dtype=np.uint8)
        cols.append(f"numpy {n/bench(lambda: np.multiply(na, nb, out=no))/1e9:6.2f}")
        for dev in tdevs:
            ta = torch.from_numpy(na.copy()).to(dev); tb = torch.from_numpy(nb.copy()).to(dev)
            to = torch.empty(n, dtype=torch.uint8, device=dev)
            t = bench(lambda: (torch.mul(ta, tb, out=to), _sync(dev)))
            cols.append(f"torch-{dev} {n/t/1e9:6.2f}")
        if mh.available():
            cols.append(f"ringkit-metal {n/bench(lambda: mh.elementwise('ring_mul', out, a, b, n))/1e9:6.2f}")
        print(f"  2^{exp}  " + "   ".join(cols))
    print()


def bench_plaquette(tdevs):
    print("── Wilson plaquette stencil, 128^3 — ns/node (lower is better)")
    L = 128; n = L ** 3
    g = bytearray(os.urandom(n))
    g3 = np.frombuffer(bytes(g), dtype=np.uint8).reshape(L, L, L).copy()
    cols = [f"ringkit-C {bench(lambda: lat.plaquette(g, L, L, L))*1e9/n:6.3f}",
            f"numpy {bench(lambda: np_plaquette(g3))*1e9/n:6.3f}"]
    for dev in tdevs:
        tg = torch.from_numpy(g3.copy()).to(dev)
        t = bench(lambda: (t_plaquette(tg), _sync(dev)))
        cols.append(f"torch-{dev} {t*1e9/n:6.3f}")
    if mh.available():
        e = bytearray(n)
        cols.append(f"ringkit-metal {bench(lambda: mh.plaquette(e, g, L, L, L))*1e9/n:6.3f}")
    print("  " + "   ".join(cols))
    print()


def bench_sweep_arrays(tdevs):
    print("── Metropolis sweep, arrays supplied, 128^3 — ns/node/sweep (lower is better)")
    L = 128; n = L ** 3
    g = bytearray(os.urandom(n)); prop = bytearray(os.urandom(n)); chance = bytearray(os.urandom(n))
    lut = bytearray(max(0, 255 - d) for d in range(256))
    lut_a = np.frombuffer(bytes(lut), dtype=np.uint8)
    pmasks = np_parity_masks(L, L, L)
    lib = lat._load()

    def c_sweep():                                  # ringkit's real CPU offering: threaded slabs
        for par in (0, 1):
            lib.metropolis_sweep_mt(lat._ptr(g), lat._ptr(prop), lat._ptr(chance),
                                    lat._ptr(lut), L, L, L, par, lat.NTHREADS)
    cols = [f"ringkit-C(mt) {bench(c_sweep)*1e9/n:6.2f}"]
    g3 = np.frombuffer(bytes(g), dtype=np.uint8).reshape(L, L, L).copy()
    p3 = np.frombuffer(bytes(prop), dtype=np.uint8).reshape(L, L, L)
    c3 = np.frombuffer(bytes(chance), dtype=np.uint8).reshape(L, L, L)
    cols.append(f"numpy {bench(lambda: np_sweep(g3, p3, c3, lut_a, pmasks))*1e9/n:6.2f}")
    for dev in tdevs:
        tg = torch.from_numpy(g3.copy()).to(dev)
        tp = torch.from_numpy(p3.copy()).to(dev)
        tc = torch.from_numpy(c3.copy()).to(dev)
        lt = torch.from_numpy(lut_a.copy()).to(dev)
        pm = t_parity_masks(L, L, L, dev)
        t = bench(lambda: (t_sweep(tg, tp, tc, lt, pm), _sync(dev)))
        cols.append(f"torch-{dev} {t*1e9/n:6.2f}")
    if mh.available():
        cols.append(f"ringkit-metal {bench(lambda: mh.gauge_sweep(g, prop, chance, lut, L, L, L))*1e9/n:6.2f}")
    print("  " + "   ".join(cols))
    print()


def bench_thermalize_rng(tdevs):
    print("── thermalize, derived RNG (rk_mix32), 8 sweeps — ns/node/sweep (lower is better)")
    for L in (128, 160):
        n = L ** 3; S = 8
        lut = bytearray(max(0, 255 - d) for d in range(256))
        lut_a = np.frombuffer(bytes(lut), dtype=np.uint8)
        pmasks = np_parity_masks(L, L, L)
        idx = np.arange(n, dtype=np.uint32).reshape(L, L, L)
        g = bytearray(os.urandom(n))
        saved = lat.GAUGE_METAL_MIN_NODES
        lat.GAUGE_METAL_MIN_NODES = 1 << 62
        gc = bytearray(g)
        cols = [f"ringkit-C {bench(lambda: lat.thermalize_rng(gc, 777, lut, L, L, L, S))*1e9/(n*S):6.2f}"]
        lat.GAUGE_METAL_MIN_NODES = saved
        g3 = np.frombuffer(bytes(g), dtype=np.uint8).reshape(L, L, L).copy()
        cols.append(f"numpy {bench(lambda: np_thermalize_rng(g3, 777, lut_a, pmasks, idx, S))*1e9/(n*S):6.2f}")
        for dev in tdevs:
            tg = torch.from_numpy(g3.copy()).to(dev)
            lt = torch.from_numpy(lut_a.copy()).to(dev)
            pm = t_parity_masks(L, L, L, dev)
            ti = torch.arange(n, dtype=torch.int32).reshape(L, L, L).to(dev)
            t = bench(lambda: (t_thermalize_rng(tg, 777, lt, pm, ti, S), _sync(dev)))
            cols.append(f"torch-{dev} {t*1e9/(n*S):6.2f}")
        if mh.available():
            gm = bytearray(g)
            cols.append(f"ringkit-metal-GPU {bench(lambda: lat.thermalize_rng(gm, 777, lut, L, L, L, S))*1e9/(n*S):6.2f}")
        print(f"  {L}^3  " + "   ".join(cols))
    print()


def main():
    print("=" * 78)
    print("ringkit vs external engines — same semantics, same inputs, gated then timed")
    print("=" * 78)
    environment()
    if not gate_numpy():
        print("numpy FAILED its gate — aborting.")
        return 1
    tdevs = [d for d in torch_devices() if gate_torch(d)]
    print()
    if mh.available():                              # warm the GPU so clocks are honest
        wg = bytearray(os.urandom(64 ** 3))
        wl = bytearray(max(0, 255 - d) for d in range(256))
        lat.thermalize_rng(wg, 1, wl, 64, 64, 64, 8)
    bench_elementwise(tdevs)
    bench_plaquette(tdevs)
    bench_sweep_arrays(tdevs)
    bench_thermalize_rng(tdevs)
    print("notes: external engines run device-resident tensors with hoisted masks (their best")
    print("case); ringkit-metal timings INCLUDE host-buffer copies (our real API cost). All")
    print("engines passed a bit-for-bit gate against the ringkit python reference first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
