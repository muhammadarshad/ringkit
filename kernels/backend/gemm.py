"""
ringkit.kernels.backend.gemm — silicon for the ring linear map (D9).

Loads ring_gemm.c (three variants: hardware-`*` bridge, multiplier-free QSM table,
multiplier-free shift-add) and serves them behind a load-time bit-for-bit self-test
against the multiplier-free Python reference. Rows are predictable bins -> threaded
lock-free; threaded results are bit-identical.

The charter's meaning rule governs here: the hardware-`*` variant is a quarantined
bridge for commodity ALUs; qsm/shiftadd ARE the architecture (no multiplier at runtime).
"""
import ctypes
import os
import subprocess
from ringkit.core import native as rn
from ringkit.kernels.backend import _arch_flags, _BUILD, so_path

NTHREADS = min(os.cpu_count() or 1, 16)
MT_MIN_MACS = 1 << 18                     # thread when M*K*N is at least ~256k MACs

# Which gated variant serves the tensor surface by default. All three are bit-identical;
# this is purely a measured-speed choice (see docs/BENCHMARKS.md) and can be changed freely.
DEFAULT_VARIANT = "mul"

_DIR = os.path.dirname(__file__)
_SO = so_path("ring_gemm")
_C = os.path.join(_DIR, "ring_gemm.c")
_U8 = ctypes.POINTER(ctypes.c_uint8)
VARIANTS = ("mul", "qsm", "shiftadd")
_lib = None
_tried = False


def build():
    os.makedirs(_BUILD, exist_ok=True)
    tmp = _SO + ".tmp"
    subprocess.run(["cc", "-O3", "-funroll-loops", "-shared", "-fPIC",
                    *_arch_flags(), "-o", tmp, _C], check=True)
    os.replace(tmp, _SO)


def _bind(lib):
    lib.ring_gemm_init.restype = None
    for v in VARIANTS:
        st = getattr(lib, f"ring_gemm_{v}")
        st.argtypes = [_U8, _U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long]
        st.restype = None
        mt = getattr(lib, f"ring_gemm_{v}_mt")
        mt.argtypes = [_U8, _U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_int]
        mt.restype = None
    lib.ring_gemm_init()
    return lib


def _selftest(lib):
    """Gate: every variant (st and mt) must reproduce the multiplier-free Python reference
    bit-for-bit on a fixed case that covers all 256 values on both sides."""
    M, K, N = 8, 32, 8
    A = bytearray(((i << 3) + 7) & 0xFF for i in range(M * K))
    B = bytearray(((i << 1) + 89) & 0xFF for i in range(K * N))
    want = bytearray(M * N)
    for i in range(M):
        for j in range(N):
            acc = 0
            for k in range(K):
                acc = (acc + rn.mul(A[i * K + k], B[k * N + j])) & 0xFF
            want[i * N + j] = acc
    for v in VARIANTS:
        got = bytearray(M * N)
        getattr(lib, f"ring_gemm_{v}")(_ptr(got), _ptr(A), _ptr(B), M, K, N)
        if got != want:
            return False
        got_mt = bytearray(M * N)
        getattr(lib, f"ring_gemm_{v}_mt")(_ptr(got_mt), _ptr(A), _ptr(B), M, K, N, 4)
        if got_mt != want:
            return False
    return True


def _load():
    global _lib, _tried
    if _lib is not None or _tried:
        return _lib
    _tried = True
    try:
        if not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(_C):
            build()                       # rebuild BEFORE first CDLL (dyld caches by path)
        lib = _bind(ctypes.CDLL(_SO))
        _lib = lib if _selftest(lib) else None
    except Exception:
        _lib = None
    return _lib


def available():
    return _load() is not None


def _ptr(ba):
    return (ctypes.c_uint8 * len(ba)).from_buffer(ba)


def gemm(A, B, M, K, N, variant="mul", out=None):
    """C = A(MxK) @ B(KxN) mod 256 over flat uint8 buffers. Returns a bytearray, or None
    if the silicon is unavailable (caller falls back to the Python reference)."""
    lib = _load()
    if lib is None or variant not in VARIANTS:
        return None
    Ab = A if isinstance(A, bytearray) else bytearray(A)
    Bb = B if isinstance(B, bytearray) else bytearray(B)
    C = out if out is not None else bytearray(M * N)
    macs = M * K * N
    if macs >= MT_MIN_MACS and NTHREADS > 1:
        getattr(lib, f"ring_gemm_{variant}_mt")(_ptr(C), _ptr(Ab), _ptr(Bb), M, K, N, NTHREADS)
    else:
        getattr(lib, f"ring_gemm_{variant}")(_ptr(C), _ptr(Ab), _ptr(Bb), M, K, N)
    return C
