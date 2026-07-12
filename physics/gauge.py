"""
ringkit.physics.gauge — SU(256) lattice gauge engine (the QCM physics core).

Computes the Wilson plaquette action on a Z256 lattice grid (uint8, C-owned memory). The C
kernel (kernels/gauge.c) runs the stencil with cache blocking (64-depth tiles = the Julia
"256 = 4 x 64" unroll that locks the working set in L2). Pure-Python fallback preserves semantics
if the .so can't build. Charter D9: silicon layer, validated bit-for-bit vs the ring reference.

    g = bytearray(...)                      # W*H*D uint8 lattice (row-major, i fastest)
    e = plaquette(g, W, H, D)               # bytearray energy field
"""
import ctypes
import os
import subprocess
from ringkit.core import native as rn
from ringkit.kernels.backend import _arch_flags, _BUILD

_DIR = os.path.join(os.path.dirname(__file__), "..", "kernels", "mprc", "lattice")
_SO = os.path.join(_BUILD, "gauge.so")
_C = os.path.join(_DIR, "gauge.c")
_U8 = ctypes.POINTER(ctypes.c_uint8)
_lib = None
_tried = False


def build():
    os.makedirs(_BUILD, exist_ok=True)
    subprocess.run(["cc", "-O3", "-funroll-loops", "-shared", "-fPIC",
                    *_arch_flags(), "-o", _SO, _C], check=True)


def _load():
    global _lib, _tried
    if _lib is not None or _tried:
        return _lib
    _tried = True
    if not os.path.exists(_SO):
        try:
            build()
        except Exception:
            return None
    try:
        lib = ctypes.CDLL(_SO)
        for name in ("plaquette", "plaquette_blocked"):
            fn = getattr(lib, name)
            fn.argtypes = [_U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long]
            fn.restype = None
        lib.metropolis_sweep.argtypes = [_U8, _U8, _U8, _U8,
                                         ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_int]
        lib.metropolis_sweep.restype = None
        _lib = lib
    except Exception:
        _lib = None
    return _lib


def available():
    return _load() is not None


def _ptr(ba):
    return (ctypes.c_uint8 * len(ba)).from_buffer(ba)


def _py_plaquette(g, W, H, D):
    e = bytearray(len(g))
    for k in range(1, D - 1):
        for j in range(1, H - 1):
            base = (k * H + j) * W
            for i in range(1, W - 1):
                c = base + i
                pos = (g[c] + g[c + 1]) & 0xFF
                neg = (g[c + W] + g[c - 1]) & 0xFF
                e[c] = (pos - neg) & 0xFF
    return e


def plaquette(g, W, H, D, blocked=True, force_python=False):
    """Wilson plaquette action over the W x H x D lattice. Returns an energy bytearray (C mem).
    blocked=True uses 64-depth cache tiling. Falls back to pure Python if no C kernel."""
    if len(g) != W * H * D:
        raise ValueError(f"plaquette: len(g)={len(g)} != W*H*D={W*H*D}")
    lib = None if force_python else _load()
    if lib is None:
        return _py_plaquette(g, W, H, D)
    gb = g if isinstance(g, bytearray) else bytearray(g)
    e = bytearray(len(g))
    fn = lib.plaquette_blocked if blocked else lib.plaquette
    fn(_ptr(e), _ptr(gb), W, H, D)
    return e


def boltzmann_lut(beta):
    """Integer Boltzmann acceptance table (ring-native, no float): higher dS -> lower accept
    threshold; larger integer `beta` = colder (rejects uphill moves harder). 256 bytes, L1-resident."""
    beta = int(beta)
    lut = bytearray(256)
    for ds in range(256):
        v = 255 - (rn.mul(ds, beta) if beta else 0)     # linear decay (ring-native accept policy)
        lut[ds] = v if v > 0 else 0
    return lut


def _py_sweep(grid, prop, chance, lut, W, H, D, parity):
    sk = W * H
    def cd(a, b):
        d = (a - b) & 0xFF
        return d if d < 256 - d else 256 - d
    for k in range(1, D - 1):
        for j in range(1, H - 1):
            base = k * sk + j * W
            for i in range(1, W - 1):
                if ((i + j + k) & 1) != parity:
                    continue
                c = base + i
                old = grid[c]; nv = (old + prop[c]) & 0xFF
                nbs = (grid[c+1], grid[c-1], grid[c+W], grid[c-W], grid[c+sk], grid[c-sk])
                So = sum(cd(old, x) for x in nbs)
                Sn = sum(cd(nv, x) for x in nbs)
                dS = Sn - So
                if dS <= 0 or chance[c] < lut[min(dS, 255)]:
                    grid[c] = nv


def sweep(grid, prop, chance, lut, W, H, D, force_python=False):
    """One full Metropolis sweep (both checkerboard parities), in place on `grid` (bytearray)."""
    lib = None if force_python else _load()
    if lib is None:
        _py_sweep(grid, prop, chance, lut, W, H, D, 0)
        _py_sweep(grid, prop, chance, lut, W, H, D, 1)
        return grid
    for parity in (0, 1):
        lib.metropolis_sweep(_ptr(grid), _ptr(bytearray(prop)), _ptr(bytearray(chance)),
                             _ptr(bytearray(lut)), W, H, D, parity)
    return grid


def correlation(grid, R, W, H, D):
    """Order parameter: mean alignment of links R apart along the i-axis, normalized 0..1.
    1 = perfectly aligned (ordered), ~0.5 = random, 0 = anti-aligned. Ring-native (ring-distance)."""
    sk = W * H
    def cd(a, b):
        d = (a - b) & 0xFF
        return d if d < 256 - d else 256 - d
    tot = 0; n = 0
    for k in range(1, D - 1):
        for j in range(1, H - 1):
            base = k * sk + j * W
            for i in range(1, W - 1 - R):
                c = base + i
                tot += 128 - cd(grid[c], grid[c + R])
                n += 1
    return tot / (n * 128) if n else 0.0


def criticality_scan(betas, W, H, D, therm=30, seed=0):
    """Sweep beta (coupling); for each, thermalize a fresh random lattice and measure the order
    parameters. Returns [(beta, mean_action, correlation(R=1))]. Locates the ordered<->disordered
    transition: high beta (cold) -> low action / high correlation; low beta (hot) -> the reverse."""
    import random as _rng
    n = W * H * D
    out = []
    for b in betas:
        _rng.seed(seed)
        g = bytearray(_rng.randint(0, 255) for _ in range(n))
        L = boltzmann_lut(b)
        for _ in range(therm):
            p = bytearray(_rng.randint(0, 255) for _ in range(n))
            ch = bytearray(_rng.randint(0, 255) for _ in range(n))
            sweep(g, p, ch, L, W, H, D)
        out.append((b, mean_action(g, W, H, D), correlation(g, 1, W, H, D)))
    return out


def mean_action(grid, W, H, D):
    """Average local ring-action (sum of neighbor ring-distances) — order parameter.
    Low = ordered/aligned (cold), high = disordered (hot)."""
    sk = W * H
    def cd(a, b):
        d = (a - b) & 0xFF
        return d if d < 256 - d else 256 - d
    tot = 0; n = 0
    for k in range(1, D - 1):
        for j in range(1, H - 1):
            base = k * sk + j * W
            for i in range(1, W - 1):
                c = base + i
                tot += cd(grid[c], grid[c+1]) + cd(grid[c], grid[c+W]) + cd(grid[c], grid[c+sk])
                n += 3
    return tot / n if n else 0
