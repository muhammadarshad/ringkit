"""
ringkit.kernels.apple.metal.host — loader for the Metal ring-ops backend (D9 silicon).

Builds shim.m (ObjC, C ABI) into an arch-keyed .so on first use, compiles ring_ops.metal
at runtime through the shim, and exposes `elementwise` for the backend registry. Absence of
Metal (Linux, CI, no GPU) is normal: available() returns False and the registry falls through
to the C SIMD path. The registry self-tests this backend bit-for-bit before it may serve.
"""
import ctypes
import os
import platform
import subprocess
from ringkit.kernels.backend import _arch_flags, _BUILD, so_path

_DIR = os.path.dirname(__file__)
_SHIM_C = os.path.join(_DIR, "shim.m")
_METAL_SRC = os.path.join(_DIR, "ring_ops.metal")
_SO = so_path("metal_shim")
_U8 = ctypes.POINTER(ctypes.c_uint8)
_OPS = {"ring_mul": 0, "ring_add": 1, "ring_sub": 2}
_lib = None
_tried = False


def build():
    """Compile shim.m -> metal_shim-<arch>.so (ObjC + Metal/Foundation). Raises on failure."""
    os.makedirs(_BUILD, exist_ok=True)
    subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-fobjc-arc",
                    "-framework", "Metal", "-framework", "Foundation",
                    *_arch_flags(), "-o", _SO, _SHIM_C], check=True)


def _load():
    global _lib, _tried
    if _lib is not None or _tried:
        return _lib
    _tried = True
    if platform.system() != "Darwin":
        return None
    if not os.path.exists(_SO):
        try:
            build()
        except Exception:
            return None
    try:
        lib = ctypes.CDLL(_SO)
        lib.rk_metal_init.argtypes = [ctypes.c_char_p]
        lib.rk_metal_init.restype = ctypes.c_int
        lib.rk_metal_elementwise.argtypes = [ctypes.c_int, _U8, _U8, _U8, ctypes.c_long]
        lib.rk_metal_elementwise.restype = ctypes.c_int
        lib.rk_metal_device_name.restype = ctypes.c_char_p
        with open(_METAL_SRC, "rb") as f:
            if lib.rk_metal_init(f.read()) != 0:
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
    bytearrays of length n. Returns 0 on success (registry falls through on nonzero)."""
    lib = _load()
    if lib is None:
        return -1
    return lib.rk_metal_elementwise(_OPS[op], _ptr(out), _ptr(a), _ptr(b), n)
