"""
ringkit.kernels.nvidia.cuda.host — loader for the CUDA backend (D9 silicon).

Mirrors the Apple/Metal backend (kernels/apple/metal/host.py): compile ringkit's OWN ring kernels
to an arch-keyed shared library on first use, load via ctypes, expose elementwise + gemm dispatch,
and SELF-TEST bit-for-bit before serving. Absence of CUDA (no GPU / no toolkit) is normal:
available() returns False and callers fall through to the C / Python path.

Compile lesson (from vlm-1-exp/kernels/cuda): on Windows nvcc needs the MSVC toolchain, which is
present but not on PATH — inject the vcvarsall x64 environment (PATH/INCLUDE/LIB), then nvcc -shared.
No cl.exe-on-PATH, no cupy, no host-compiler headache. Pure integer; D9 hardware ops are allowed here.
"""
import ctypes
import glob
import os
import platform
import subprocess
from ringkit.kernels.backend import _BUILD, so_path

_DIR = os.path.dirname(__file__)
_CU = os.path.join(_DIR, "ring_cuda.cu")
_LIB = so_path("ring_cuda")                 # kernels/build/ring_cuda-<machine>.so (PE DLL on Windows)
_U8 = ctypes.POINTER(ctypes.c_uint8)
_OPS = {"ring_mul": "ring_mul", "ring_add": "ring_add", "ring_sub": "ring_sub"}
_lib = None
_tried = False

_CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
_VCVARS_CANDIDATES = [
    r"C:\Program Files\Microsoft Visual Studio\2022\{ed}\VC\Auxiliary\Build\vcvarsall.bat".format(ed=e)
    for e in ("Enterprise", "Professional", "Community", "BuildTools")
]


def _msvc_env():
    """The vlm-1 lesson: cl.exe/INCLUDE/LIB exist only after vcvarsall x64. Return an env with them."""
    env = dict(os.environ)
    if platform.system() != "Windows":
        return env
    vc = next((p for p in _VCVARS_CANDIDATES if os.path.exists(p)), None)
    if vc is None:
        return env
    out = subprocess.run(f'"{vc}" x64 && set', shell=True, capture_output=True, text=True)
    if out.returncode == 0:
        for line in out.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                if k.upper() in ("PATH", "INCLUDE", "LIB", "LIBPATH"):
                    env[k.upper()] = v
    return env


def build():
    """nvcc -> arch-keyed shared lib. `-arch=native` targets THIS GPU. Raises on failure."""
    os.makedirs(_BUILD, exist_ok=True)
    tmp = _LIB + ".tmp"
    xcompiler = ["-Xcompiler", "/MD"] if platform.system() == "Windows" else ["-Xcompiler", "-fPIC"]
    cmd = ["nvcc", "-O3", "-shared", "-arch=native", *xcompiler, "-o", tmp, _CU]
    subprocess.run(cmd, env=_msvc_env(), check=True, capture_output=True, text=True)
    os.replace(tmp, _LIB)


def _load():
    global _lib, _tried
    if _lib is not None or _tried:
        return _lib
    _tried = True
    try:
        if not os.path.exists(_LIB) or os.path.getmtime(_LIB) < os.path.getmtime(_CU):
            build()
        if platform.system() == "Windows" and os.path.isdir(_CUDA_BIN):
            os.add_dll_directory(_CUDA_BIN)         # cudart lives here
        lib = ctypes.CDLL(_LIB)
        lib.rk_cuda_available.restype = ctypes.c_int
        for nm in _OPS.values():
            fn = getattr(lib, nm)
            fn.argtypes = [_U8, _U8, _U8, ctypes.c_long]
            fn.restype = ctypes.c_int
        lib.ring_gemm.argtypes = [_U8, _U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long]
        lib.ring_gemm.restype = ctypes.c_int
        lib.ring_l1dist.argtypes = [ctypes.POINTER(ctypes.c_int), _U8, _U8,
                                    ctypes.c_long, ctypes.c_long, ctypes.c_long]
        lib.ring_l1dist.restype = ctypes.c_int
        if lib.rk_cuda_available() != 1 or not _selftest(lib):
            _lib = None
            return None
        _lib = lib
    except Exception:
        _lib = None
    return _lib


def _ptr(ba):
    return (ctypes.c_uint8 * len(ba)).from_buffer(ba)


def _selftest(lib):
    """D9 gate: reproduce the pure-Python ring reference bit-for-bit before serving (full 256 ring
    on the elementwise trio + the gemm.py reference case)."""
    a = bytearray(range(256)); b = bytearray((i + 89) & 0xFF for i in range(256))
    for nm, f in (("ring_mul", lambda x, y: (x * y) & 0xFF),
                  ("ring_add", lambda x, y: (x + y) & 0xFF),
                  ("ring_sub", lambda x, y: (x - y) & 0xFF)):
        got = bytearray(256)
        if getattr(lib, nm)(_ptr(got), _ptr(a), _ptr(b), 256) != 0:
            return False
        if got != bytearray(f(a[i], b[i]) for i in range(256)):
            return False
    M, K, N = 8, 32, 8
    A = bytearray(((i << 3) + 7) & 0xFF for i in range(M * K))
    B = bytearray(((i << 1) + 89) & 0xFF for i in range(K * N))
    want = bytearray(M * N)
    for i in range(M):
        for j in range(N):
            acc = 0
            for k in range(K):
                acc = (acc + A[i * K + k] * B[k * N + j]) & 0xFF
            want[i * N + j] = acc
    got = bytearray(M * N)
    if lib.ring_gemm(_ptr(got), _ptr(A), _ptr(B), M, K, N) != 0 or got != want:
        return False
    # ring_l1dist: ENERGY-side ring-L1 distance — reproduce the ml.attention / kv_cache reference
    # (rdist summed, no mod fold) bit-for-bit. This is the distance ring_gemm CANNOT serve.
    def _rdist(x, y):
        d = (x - y) & 0xFF; ee = 256 - d
        return d if d < ee else ee
    m2, n2, dim2 = 4, 5, 7
    Q = bytearray((i * 13 + 2) & 0xFF for i in range(m2 * dim2))
    Kk = bytearray((i * 7 + 50) & 0xFF for i in range(n2 * dim2))
    want_d = [sum(_rdist(Q[i * dim2 + d], Kk[j * dim2 + d]) for d in range(dim2))
              for i in range(m2) for j in range(n2)]
    Dout = (ctypes.c_int * (m2 * n2))()
    if lib.ring_l1dist(Dout, _ptr(Q), _ptr(Kk), m2, n2, dim2) != 0 or list(Dout) != want_d:
        return False
    # REAL-MAGNITUDE regime (the int64-overflow-that-hung lesson): gate the accumulator at the
    # scale real descriptors produce. dimB=2048 all at the MAX ring distance (128) -> 262144 per
    # row; must be exact (no wrap). Plus an asymmetric case exercising the min() branch.
    dimB = 2048
    Qb = bytearray([0] * dimB + [10] * dimB)          # 2 rows
    Kb = bytearray([128] * dimB + [200] * dimB)       # 2 rows
    wantB = [sum(_rdist(Qb[i * dimB + d], Kb[j * dimB + d]) for d in range(dimB))
             for i in range(2) for j in range(2)]
    DB = (ctypes.c_int * 4)()
    if lib.ring_l1dist(DB, _ptr(Qb), _ptr(Kb), 2, 2, dimB) != 0 or list(DB) != wantB:
        return False
    if wantB[0] != 128 * dimB:                        # 0 vs 128 = max arc distance, exact sum
        return False
    return True


def available():
    """True iff the CUDA backend built, loaded, self-tested, and a GPU is present."""
    return _load() is not None


def elementwise(op, out, a, b, n):
    """Ring elementwise into `out` (bytearray). op in ring_mul/add/sub. Returns 0 on success."""
    lib = _load()
    if lib is None or op not in _OPS:
        return -1
    ab = a if isinstance(a, bytearray) else bytearray(a)
    bb = b if isinstance(b, bytearray) else bytearray(b)
    return getattr(lib, _OPS[op])(_ptr(out), _ptr(ab), _ptr(bb), n)


def gemm(A, B, M, K, N, out=None):
    """C = A(MxK) @ B(KxN) mod 256 over flat uint8 buffers. Returns a bytearray, or None if
    the CUDA silicon is unavailable (caller falls back)."""
    lib = _load()
    if lib is None:
        return None
    Ab = A if isinstance(A, bytearray) else bytearray(A)
    Bb = B if isinstance(B, bytearray) else bytearray(B)
    C = out if out is not None else bytearray(M * N)
    if lib.ring_gemm(_ptr(C), _ptr(Ab), _ptr(Bb), M, K, N) != 0:
        return None
    return C


def l1dist(Q, K, m, n, dim):
    """Ring-L1 distance matrix D[i,j] = SUM_d min(|Q[i,d]-K[j,d]|, 256-|.|), the ENERGY side
    (int32, NEVER folded mod 256 — unlike gemm). Q,K are flat uint8 buffers of length m*dim / n*dim
    (row-major). Returns a ctypes c_int array of length m*n (row-major D), or None if CUDA is
    unavailable. Bit-for-bit == ml.attention.ring_distance summed / kv_cache.c::kv_scores."""
    lib = _load()
    if lib is None:
        return None
    Qb = Q if isinstance(Q, bytearray) else bytearray(Q)
    Kb = K if isinstance(K, bytearray) else bytearray(K)
    D = (ctypes.c_int * (m * n))()
    if lib.ring_l1dist(D, _ptr(Qb), _ptr(Kb), m, n, dim) != 0:
        return None
    return D
