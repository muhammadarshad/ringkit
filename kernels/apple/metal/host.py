"""
ringkit.kernels.apple.metal.host — loader for the Metal backend (D9 silicon).

Builds shim.m (ObjC, C ABI) into an arch-keyed .so on first use, compiles the .metal sources
at runtime through the shim (ring_ops + gauge in one library), and exposes elementwise +
gauge dispatch. Absence of Metal (Linux, CI, no GPU) is normal: available() returns False and
callers fall through to the C path. Backends are self-tested bit-for-bit before serving.

The shim ABI is versioned: a stale compiled shim (older symbol set) is rebuilt automatically.
"""
import ctypes
import os
import platform
import subprocess
from ringkit.kernels.backend import _arch_flags, _BUILD, so_path

_ABI = 4
_DIR = os.path.dirname(__file__)
_SHIM_C = os.path.join(_DIR, "shim.m")
_METAL_SRCS = (os.path.join(_DIR, "ring_ops.metal"), os.path.join(_DIR, "gauge.metal"))
_SO = so_path("metal_shim")
_U8 = ctypes.POINTER(ctypes.c_uint8)
_OPS = {"ring_mul": 0, "ring_add": 1, "ring_sub": 2}
_lib = None
_tried = False


def build():
    """Compile shim.m -> metal_shim-<arch>.so (ObjC + Metal/Foundation). Raises on failure.
    Compiles to a temp path then os.replace, so a fresh inode is always loaded."""
    os.makedirs(_BUILD, exist_ok=True)
    tmp = _SO + ".tmp"
    subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-fobjc-arc",
                    "-framework", "Metal", "-framework", "Foundation",
                    *_arch_flags(), "-o", tmp, _SHIM_C], check=True)
    os.replace(tmp, _SO)


def _open_shim():
    """CDLL the shim, rebuilding FIRST if the artifact is missing or predates shim.m —
    dyld caches images by path, so a rebuild after loading cannot take effect in-process."""
    if not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(_SHIM_C):
        build()
    lib = ctypes.CDLL(_SO)
    lib.rk_metal_abi_version.restype = ctypes.c_int
    if lib.rk_metal_abi_version() != _ABI:
        raise RuntimeError("metal shim ABI stale in this process — restart to pick up rebuild")
    return lib


def _load():
    global _lib, _tried
    if _lib is not None or _tried:
        return _lib
    _tried = True
    if platform.system() != "Darwin":
        return None
    try:
        lib = _open_shim()
        lib.rk_metal_init.argtypes = [ctypes.c_char_p]
        lib.rk_metal_init.restype = ctypes.c_int
        lib.rk_metal_elementwise.argtypes = [ctypes.c_int, _U8, _U8, _U8, ctypes.c_long]
        lib.rk_metal_elementwise.restype = ctypes.c_int
        lib.rk_metal_plaquette.argtypes = [_U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long]
        lib.rk_metal_plaquette.restype = ctypes.c_int
        lib.rk_metal_gauge_sweep.argtypes = [_U8, _U8, _U8, _U8,
                                             ctypes.c_long, ctypes.c_long, ctypes.c_long]
        lib.rk_metal_gauge_sweep.restype = ctypes.c_int
        lib.rk_metal_thermalize.argtypes = [_U8, _U8, _U8, _U8, ctypes.c_long,
                                            ctypes.c_long, ctypes.c_long, ctypes.c_long]
        lib.rk_metal_thermalize.restype = ctypes.c_int
        lib.rk_metal_thermalize_rng.argtypes = [_U8, ctypes.c_uint, ctypes.c_uint, _U8,
                                                ctypes.c_long, ctypes.c_long, ctypes.c_long,
                                                ctypes.c_long]
        lib.rk_metal_thermalize_rng.restype = ctypes.c_int
        lib.rk_metal_device_name.restype = ctypes.c_char_p
        src = b"\n".join(open(p, "rb").read() for p in _METAL_SRCS)
        if lib.rk_metal_init(src) != 0:
            return None
        _lib = lib
    except Exception:
        _lib = None
    return _lib


def available():
    """True iff a Metal device exists and the shim built, loaded, and compiled the shaders."""
    return _load() is not None


def device_name():
    lib = _load()
    return lib.rk_metal_device_name().decode() if lib else ""


def _ptr(ba):
    return (ctypes.c_uint8 * len(ba)).from_buffer(ba)


def elementwise(op, out, a, b, n):
    """Dispatch one elementwise ring op on the GPU. op in ring_mul/add/sub; buffers are
    bytearrays of length n. Returns 0 on success (callers fall through on nonzero)."""
    lib = _load()
    if lib is None:
        return -1
    return lib.rk_metal_elementwise(_OPS[op], _ptr(out), _ptr(a), _ptr(b), n)


def plaquette(e, g, W, H, D):
    """GPU Wilson plaquette into bytearray e (same shape as g). Returns 0 on success."""
    lib = _load()
    if lib is None:
        return -1
    return lib.rk_metal_plaquette(_ptr(e), _ptr(g), W, H, D)


def thermalize(grid, props, chances, lut, W, H, D, sweeps):
    """A batch of full sweeps, GPU-resident (unified memory): grid crosses the bus once per
    batch; props/chances are concatenated sweeps*n arrays. Returns 0 on success."""
    lib = _load()
    if lib is None:
        return -1
    pb = props if isinstance(props, bytearray) else bytearray(props)
    cb = chances if isinstance(chances, bytearray) else bytearray(chances)
    lb = lut if isinstance(lut, bytearray) else bytearray(lut)
    return lib.rk_metal_thermalize(_ptr(grid), _ptr(pb), _ptr(cb), _ptr(lb), W, H, D, sweeps)


def thermalize_rng(grid, seed, sweep0, lut, W, H, D, sweeps):
    """Batch of sweeps with on-GPU derived randoms (rk_mix32 spec): only grid + lut cross
    the bus. sweep0 = starting sweep index (counter continues across batches). Returns 0."""
    lib = _load()
    if lib is None:
        return -1
    lb = lut if isinstance(lut, bytearray) else bytearray(lut)
    return lib.rk_metal_thermalize_rng(_ptr(grid), seed & 0xFFFFFFFF, sweep0 & 0xFFFFFFFF,
                                       _ptr(lb), W, H, D, sweeps)


def gauge_sweep(grid, prop, chance, lut, W, H, D):
    """One FULL Metropolis sweep (both parities) GPU-resident, in place on `grid`.
    Same parity-sequential semantics as the C kernel. Returns 0 on success."""
    lib = _load()
    if lib is None:
        return -1
    return lib.rk_metal_gauge_sweep(_ptr(grid), _ptr(bytearray(prop)),
                                    _ptr(bytearray(chance)), _ptr(bytearray(lut)), W, H, D)
