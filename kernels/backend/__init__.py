"""
ringkit.kernels.backend — the SILICON layer: a registry of hardware backends for elementwise
ring ops, each operating ZERO-COPY (or documented-copy) on caller-owned uint8 buffers.

Charter D9: this layer uses hardware ops on purpose; every backend must reproduce the
multiplier-free ring semantics (ringkit.core.native.qsm etc.) BIT-FOR-BIT before it may serve.
A backend becomes eligible only after passing a load-time self-test against the pure-Python
reference; ineligible or absent backends fall through silently (available via `backends()`).

Registry (probe order): metal (kernels/apple/metal, large buffers only) -> cpu-c (ring_ops.c,
SIMD) -> python (reference loop). `active(n)` reports which backend serves a buffer of size n.

Build artifacts are ARCH-KEYED (ring_ops-<machine>.so) so an x86_64 (Rosetta) and a native
arm64 interpreter can share kernels/build/ without clobbering each other.
"""
import ctypes
import os
import platform
import subprocess

_DIR = os.path.dirname(__file__)
_BUILD = os.path.join(_DIR, "..", "build")          # all compiled kernels land in kernels/build/
_C = os.path.join(_DIR, "ring_ops.c")
_U8 = ctypes.POINTER(ctypes.c_uint8)
_NAMES = ("ring_mul", "ring_add", "ring_sub", "ring_mul_u64", "ring_add_u64", "ring_sub_u64")
_OP_IDX = {"ring_mul": 0, "ring_add": 1, "ring_sub": 2}
NTHREADS = min(os.cpu_count() or 1, 16)      # specialised-MPP block count for ring_ew_mt

# Metal auto-routing is OFF by default (METAL_MIN = None): measured 2026-07-12 on M1 Pro,
# C SIMD wins the elementwise trio at EVERY size (6.5 GMUPS vs ~0.95 GMUPS at 2^24) because
# these ops are bandwidth-trivial and the GPU path pays 3 buffer copies per call. The backend
# stays registered + bit-for-bit verified for experiments (set METAL_MIN to an int to route
# buffers >= that size); Metal's win condition is compute-dense kernels (gauge stencil, fused
# chains that stay GPU-resident), not single elementwise passes.
METAL_MIN = None

_lib = None
_tried = False
_metal = None
_metal_tried = False
_cuda = None
_cuda_tried = False


def so_path(stem):
    """Arch-keyed artifact path: kernels/build/<stem>-<machine>.so (x86_64 and arm64 coexist)."""
    return os.path.join(_BUILD, f"{stem}-{platform.machine()}.so")


_SO = so_path("ring_ops")


def build():
    """Compile ring_ops.c for THIS interpreter's architecture. Raises on failure."""
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
    try:
        if not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(_C):
            build()                       # rebuild BEFORE first CDLL (dyld caches by path)
    except Exception:
        return None
    try:
        lib = ctypes.CDLL(_SO)
        for name in _NAMES:
            fn = getattr(lib, name)
            fn.argtypes = [_U8, _U8, _U8, ctypes.c_long]
            fn.restype = None
        lib.ring_ew_mt.argtypes = [_U8, _U8, _U8, ctypes.c_long, ctypes.c_int, ctypes.c_long]
        lib.ring_ew_mt.restype = None
        lib.ring_ew_pool.argtypes = [_U8, _U8, _U8, ctypes.c_long, ctypes.c_int, ctypes.c_long]
        lib.ring_ew_pool.restype = None
        if not _selftest(lambda op, out, a, b, n: getattr(lib, op)(_ptr(out), _ptr(a), _ptr(b), n)):
            _lib = None
            return None
        if not _selftest_mt(lib):     # MPP block-split == scalar, bit-for-bit, at real (splitting) size
            _lib = None
            return None
        _lib = lib
    except Exception:
        _lib = None
    return _lib


def _load_metal():
    global _metal, _metal_tried
    if _metal is not None or _metal_tried:
        return _metal
    _metal_tried = True
    try:
        from ringkit.kernels.apple.metal import host as mh
        if not mh.available():
            return None
        if not _selftest(lambda op, out, a, b, n: mh.elementwise(op, out, a, b, n)):
            return None
        _metal = mh
    except Exception:
        _metal = None
    return _metal


def _selftest(call):
    """Load-time eligibility gate: the candidate must reproduce the Python reference
    bit-for-bit on a fixed 256-vector for all three ops. D9's bar, enforced at the door."""
    a = bytearray(range(256))
    b = bytearray((i + 89) & 0xFF for i in range(256))          # 89 odd -> hits units + zero-divisors
    for op in ("ring_mul", "ring_add", "ring_sub"):
        want = bytearray(_PY[op](a[i], b[i]) for i in range(256))
        got = bytearray(256)
        try:
            call(op, got, a, b, 256)
        except Exception:
            return False
        if got != want:
            return False
    return True


def _selftest_mt(lib):
    """Gate the specialised-MPP block kernel at a size that ACTUALLY splits (> EW_MT_MIN),
    over an odd block count so the last block is a ragged remainder: every op, threaded ==
    the scalar reference, bit-for-bit. A wrong block boundary would corrupt exactly here."""
    import os as _os
    n = (1 << 18) + 777                                    # > EW_MT_MIN, not a multiple of nthreads
    a = bytearray(_os.urandom(n)); b = bytearray(_os.urandom(n))
    for op, idx in _OP_IDX.items():
        want = bytearray(_PY[op](a[i], b[i]) for i in range(n))
        for fn in (lib.ring_ew_mt, lib.ring_ew_pool):       # both MPP forms, bit-for-bit
            got = bytearray(n)
            try:
                fn(_ptr(got), _ptr(a), _ptr(b), n, idx, 7)   # 7 blocks -> ragged tail
            except Exception:
                return False
            if got != want:
                return False
    return True


def ew_mt(name, a, b, out=None, nthreads=NTHREADS):
    """Elementwise ring op via the specialised-MPP block split (disjoint blocks over C-owned
    memory, lock-free, merge-free). name in ring_mul/add/sub. Falls back to the Python
    reference if the C path is unavailable. Bit-identical to `mul`/`add`/`sub`."""
    n = len(a)
    if out is None:
        out = bytearray(n)
    lib = _load()
    if lib is None:
        pref = _PY[name]
        for i in range(n):
            out[i] = pref(a[i], b[i])
        return out
    lib.ring_ew_mt(_ptr(out), _ptr(a), _ptr(b), n, _OP_IDX[name], nthreads)
    return out


def available():
    """True iff the compiled C SIMD path is loaded, self-tested, and usable."""
    return _load() is not None


def _load_cuda():
    """CUDA elementwise backend (kernels/nvidia/cuda), self-tested before serving. None if absent."""
    global _cuda, _cuda_tried
    if _cuda is not None or _cuda_tried:
        return _cuda
    _cuda_tried = True
    try:
        from ringkit.kernels.nvidia.cuda import host as ch
        if not ch.available():
            return None
        if not _selftest(lambda op, out, a, b, n: ch.elementwise(op, out, a, b, n)):
            return None
        _cuda = ch
    except Exception:
        _cuda = None
    return _cuda


def backends():
    """Status of every registered backend: name -> 'serving' | 'unavailable'."""
    return {"metal": "serving" if _load_metal() is not None else "unavailable",
            "cuda": "serving" if _load_cuda() is not None else "unavailable",
            "cpu-c": "serving" if _load() is not None else "unavailable",
            "python": "serving"}


def active(n=0):
    """Which backend serves a buffer of `n` elements under the current policy."""
    if METAL_MIN is not None and n >= METAL_MIN and _load_metal() is not None:
        return "metal"
    if _load() is not None:
        return "cpu-c"
    if _load_cuda() is not None:                 # GPU serves elementwise when the C path is absent
        return "cuda"
    return "python"


def _ptr(ba):
    """Zero-copy c_uint8* into a bytearray's buffer (no copy)."""
    return (ctypes.c_uint8 * len(ba)).from_buffer(ba)


_PY = {
    "ring_mul": lambda x, y: (x * y) & 0xFF,
    "ring_add": lambda x, y: (x + y) & 0xFF,
    "ring_sub": lambda x, y: (x - y) & 0xFF,
}


def elementwise(name, a, b, unroll=False):
    """Elementwise ring op over two equal-length uint8 buffers -> a NEW bytearray.
    `a`, `b` may be bytes/bytearray. name in ring_mul/add/sub. Dispatches by registry policy:
    metal for >=METAL_MIN elements when eligible, else C SIMD, else the Python reference."""
    n = len(a)
    if len(b) != n:
        raise ValueError(f"backend.{name}: length mismatch {n} vs {len(b)}")
    out = bytearray(n)
    base = name.replace("_u64", "")
    ab = a if isinstance(a, bytearray) else bytearray(a)
    bb = b if isinstance(b, bytearray) else bytearray(b)
    which = active(n)
    if which == "metal":
        if name.endswith("_u64"):                               # metal has no unroll variants
            which = "cpu-c" if _load() is not None else "python"
        elif _metal.elementwise(base, out, ab, bb, n) == 0:
            return out
        else:
            which = "cpu-c" if _load() is not None else "python"    # dispatch failed -> fall through
    if which == "cpu-c":
        fn = getattr(_lib, name if name.endswith("_u64") else base)
        fn(_ptr(out), _ptr(ab), _ptr(bb), n)
        return out
    if which == "cuda":
        if _cuda.elementwise(base, out, ab, bb, n) == 0:        # GPU (when the C path is absent)
            return out
        # dispatch failed -> Python reference
    f = _PY[base]                                               # pure-Python reference
    for i in range(n):
        out[i] = f(a[i], b[i])
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
