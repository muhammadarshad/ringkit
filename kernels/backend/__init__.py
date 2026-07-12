"""
ringkit.kernels.backend — the SILICON layer: C/SIMD elementwise ring ops that operate
ZERO-COPY on caller-owned C memory (a Python bytearray's buffer). Data lives in the buffer,
the kernel maintains it — nothing is marshalled through Python lists.

Charter D9: this layer uses hardware ops on purpose (speed); it is validated bit-for-bit against
the multiplier-free ring semantics (ringkit.core.native.qsm etc.). If the .so can't build/load
(no C compiler / wrong arch) it falls back to a pure-Python loop over the same buffer.

Kernels: ring_mul/add/sub (auto-vectorized) and ring_*_u64 (explicit 64-lane unroll; 256 = 4x64).
"""
import ctypes
import os
import platform
import subprocess

_DIR = os.path.dirname(__file__)
_BUILD = os.path.join(_DIR, "..", "build")          # all compiled kernels land in kernels/build/
_SO = os.path.join(_BUILD, "ring_ops.so")
_C = os.path.join(_DIR, "ring_ops.c")
_U8 = ctypes.POINTER(ctypes.c_uint8)
_NAMES = ("ring_mul", "ring_add", "ring_sub", "ring_mul_u64", "ring_add_u64", "ring_sub_u64")
_lib = None
_tried = False


def build():
    """Compile ring_ops.c -> ring_ops.so (-O3, SIMD) for THIS interpreter's architecture.
    On macOS the running Python may be x86_64 (Rosetta) while cc targets arm64 by default,
    so we pass -arch explicitly; -march=native only when host and target arch match. Raises on failure."""
    os.makedirs(_BUILD, exist_ok=True)
    subprocess.run(["cc", "-O3", "-funroll-loops", "-shared", "-fPIC",
                    *_arch_flags(), "-o", _SO, _C], check=True)


def _arch_flags():
    if platform.system() == "Darwin":
        # -march=native is unusable here: it probes the HOST cpu (e.g. apple-m1) even when
        # -arch cross-targets the interpreter's arch (e.g. x86_64 Python under Rosetta).
        return ["-arch", platform.machine()]
    return ["-march=native"]


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
        for name in _NAMES:
            fn = getattr(lib, name)
            fn.argtypes = [_U8, _U8, _U8, ctypes.c_long]
            fn.restype = None
        _lib = lib
    except Exception:
        _lib = None
    return _lib


def available():
    """True iff the compiled C SIMD path is loaded and usable."""
    return _load() is not None


def _ptr(ba):
    """Zero-copy c_uint8* into a bytearray's buffer (no copy)."""
    return (ctypes.c_uint8 * len(ba)).from_buffer(ba)


_PY = {
    "ring_mul": lambda x, y: (x * y) & 0xFF,
    "ring_add": lambda x, y: (x + y) & 0xFF,
    "ring_sub": lambda x, y: (x - y) & 0xFF,
}


def elementwise(name, a, b, unroll=False):
    """Elementwise ring op over two equal-length uint8 buffers -> a NEW bytearray (C-owned mem).
    `a`, `b` may be bytes/bytearray; inputs are used zero-copy when bytearray. name in ring_mul/add/sub."""
    n = len(a)
    if len(b) != n:
        raise ValueError(f"backend.{name}: length mismatch {n} vs {len(b)}")
    out = bytearray(n)
    lib = _load()
    if lib is None:                                   # pure-Python fallback over the same buffer
        f = _PY[name.replace("_u64", "")]
        for i in range(n):
            out[i] = f(a[i], b[i])
        return out
    ab = a if isinstance(a, bytearray) else bytearray(a)
    bb = b if isinstance(b, bytearray) else bytearray(b)
    fn = getattr(lib, name + ("_u64" if unroll else ""))
    fn(_ptr(out), _ptr(ab), _ptr(bb), n)
    return out


# thin bytes-returning API (kept for direct use / tests)
def mul(a, b, force_python=False, unroll=False):
    return bytes(_py_or_c("ring_mul", a, b, force_python, unroll))


def add(a, b, force_python=False, unroll=False):
    return bytes(_py_or_c("ring_add", a, b, force_python, unroll))


def sub(a, b, force_python=False, unroll=False):
    return bytes(_py_or_c("ring_sub", a, b, force_python, unroll))


def _py_or_c(name, a, b, force_python, unroll):
    if force_python:
        n = len(a)
        if len(b) != n:
            raise ValueError(f"backend.{name}: length mismatch {n} vs {len(b)}")
        f = _PY[name]
        return bytearray(f(a[i], b[i]) for i in range(n))
    return elementwise(name, a, b, unroll)
