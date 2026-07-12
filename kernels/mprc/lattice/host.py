"""
ringkit.kernels.mprc.lattice.host — SILICON host for the SU(256) lattice gauge engine (D9).

Holds everything about the gauge engine that is legally hardware-flavored: the ctypes
build/load of gauge.c, the bit-for-bit pure-Python fallback (the reference of record lives
beside the kernel it validates, exactly like kernels.backend), and the measurement
observables (correlation / mean_action) whose normalized outputs are floats — labeled
MEASUREMENT IO, not ring quantities. Charter D9: this layer uses hardware ops on purpose
and is excluded from the semantic AST audit.

The physics-facing semantic surface is ringkit.physics.gauge, which re-exports these forms.
"""
import ctypes
import os
import subprocess
from ringkit.kernels.backend import _arch_flags, _BUILD, so_path

_DIR = os.path.dirname(__file__)
_SO = so_path("gauge")                    # arch-keyed: x86_64/arm64 interpreters coexist
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
        raise ValueError(f"plaquette: len(g)={len(g)} != W*H*D={W * H * D}")
    lib = None if force_python else _load()
    if lib is None:
        return _py_plaquette(g, W, H, D)
    gb = g if isinstance(g, bytearray) else bytearray(g)
    e = bytearray(len(g))
    fn = lib.plaquette_blocked if blocked else lib.plaquette
    fn(_ptr(e), _ptr(gb), W, H, D)
    return e


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
    1 = perfectly aligned (ordered), ~0.5 = random, 0 = anti-aligned. Ring-distance based;
    the normalized value is a float MEASUREMENT output (IO), not a ring quantity."""
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


def mean_action(grid, W, H, D):
    """Average local ring-action (sum of neighbor ring-distances) — order parameter.
    Low = ordered/aligned (cold), high = disordered (hot). Float MEASUREMENT output (IO)."""
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
