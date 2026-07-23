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
        lib.ring_gemv_i64.argtypes = [ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_int),
                                      ctypes.POINTER(ctypes.c_int), ctypes.c_long, ctypes.c_long]
        lib.ring_gemv_i64.restype = ctypes.c_int
        lib.ring_gemm_i64.argtypes = [ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_int),
                                      ctypes.POINTER(ctypes.c_int), ctypes.c_long, ctypes.c_long, ctypes.c_long]
        lib.ring_gemm_i64.restype = ctypes.c_int
        # u8-ONIX fused exact GEMV (the emulation proj, whole) — Windows GPU route
        lib.rk_gemv_u8_exact.argtypes = [ctypes.POINTER(ctypes.c_longlong), _U8,
                                         ctypes.POINTER(ctypes.c_longlong),
                                         ctypes.POINTER(ctypes.c_int8),
                                         ctypes.POINTER(ctypes.c_longlong),
                                         ctypes.c_long, ctypes.c_long]
        lib.rk_gemv_u8_exact.restype = ctypes.c_int
        lib.rk_gemv_u8_exact_res.argtypes = [ctypes.POINTER(ctypes.c_longlong), ctypes.c_void_p,
                                             ctypes.POINTER(ctypes.c_longlong),
                                             ctypes.POINTER(ctypes.c_int8),
                                             ctypes.POINTER(ctypes.c_longlong),
                                             ctypes.c_long, ctypes.c_long]
        lib.rk_gemv_u8_exact_res.restype = ctypes.c_int
        _LL = ctypes.POINTER(ctypes.c_longlong)
        lib.ring_sigmoid.argtypes = [_LL, _LL, ctypes.c_long, ctypes.c_int]
        lib.ring_sigmoid.restype = ctypes.c_int
        lib.ring_exp.argtypes = [_LL, _LL, ctypes.c_long, ctypes.c_int]
        lib.ring_exp.restype = ctypes.c_int
        lib.ring_rmsnorm.argtypes = [_LL, _LL, _LL, ctypes.c_long, ctypes.c_int, ctypes.c_longlong]
        lib.ring_rmsnorm.restype = ctypes.c_int
        lib.ring_ew_q16.argtypes = [ctypes.c_int, _LL, _LL, _LL, ctypes.c_long, ctypes.c_int]
        lib.ring_ew_q16.restype = ctypes.c_int
        lib.ring_escale.argtypes = [_LL, _LL, ctypes.c_longlong, ctypes.c_long, ctypes.c_int]
        lib.ring_escale.restype = ctypes.c_int
        lib.ring_colsum.argtypes = [_LL, _LL, ctypes.c_long, ctypes.c_long]
        lib.ring_colsum.restype = ctypes.c_int
        lib.ring_diffuse.argtypes = [_LL, _LL, ctypes.c_long, ctypes.c_long, ctypes.c_long]
        lib.ring_diffuse.restype = ctypes.c_int
        lib.ring_relu.argtypes = [_LL, _LL, ctypes.c_long]
        lib.ring_relu.restype = ctypes.c_int
        lib.ring_gather.argtypes = [_LL, _LL, ctypes.c_long, _U8, ctypes.c_long]
        lib.ring_gather.restype = ctypes.c_int
        # residency API (SPEC-014 Part B): upload once -> device handle, launch on resident handles.
        lib.rk_dev_upload_i32.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_long]
        lib.rk_dev_upload_i32.restype = ctypes.c_void_p
        lib.rk_dev_alloc_i64.argtypes = [ctypes.c_long]
        lib.rk_dev_alloc_i64.restype = ctypes.c_void_p
        lib.rk_dev_gemm_i64_resident.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_long, ctypes.c_long, ctypes.c_long]
        lib.rk_dev_gemm_i64_resident.restype = ctypes.c_int
        lib.rk_dev_download_i64.argtypes = [ctypes.POINTER(ctypes.c_longlong), ctypes.c_void_p, ctypes.c_long]
        lib.rk_dev_download_i64.restype = ctypes.c_int
        lib.rk_dev_free.argtypes = [ctypes.c_void_p]
        lib.rk_dev_free.restype = None
        # u8 manifold broadcast residency
        lib.rk_dev_upload_u8.argtypes = [_U8, ctypes.c_long]
        lib.rk_dev_upload_u8.restype = ctypes.c_void_p
        lib.rk_dev_alloc_u8.argtypes = [ctypes.c_long]
        lib.rk_dev_alloc_u8.restype = ctypes.c_void_p
        lib.rk_dev_manifold_staple_u8_resident.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                           ctypes.c_void_p, ctypes.c_long,
                                                           ctypes.c_long, ctypes.c_long]
        lib.rk_dev_manifold_staple_u8_resident.restype = ctypes.c_int
        lib.rk_dev_download_u8.argtypes = [_U8, ctypes.c_void_p, ctypes.c_long]
        lib.rk_dev_download_u8.restype = ctypes.c_int
        # resident op chain (owner: no streaming — chain on device handles)
        _vp, _cl, _ci, _cll = ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_longlong
        lib.rk_dev_upload_i64.argtypes = [_LL, _cl]
        lib.rk_dev_upload_i64.restype = _vp
        lib.rk_dev_alloc_i32.argtypes = [_cl]
        lib.rk_dev_alloc_i32.restype = _vp
        for nm, at in (("rk_dev_ew_res", [_ci, _vp, _vp, _vp, _cl, _ci]),
                       ("rk_dev_escale_res", [_vp, _vp, _cll, _cl, _ci]),
                       ("rk_dev_softplus", [_vp, _vp, _cl, _ci]),
                       ("rk_dev_sigmoid_res", [_vp, _vp, _cl, _ci]),
                       ("rk_dev_colsum_seg", [_vp, _vp, _cl, _cl, _cl]),
                       ("rk_dev_tile_rows", [_vp, _vp, _cl, _cl, _cl]),
                       ("rk_dev_rowsum", [_vp, _vp, _cl, _cl]),
                       ("rk_dev_slice_cols", [_vp, _vp, _cl, _cl, _cl, _cl]),
                       ("rk_dev_scatter_cols", [_vp, _vp, _cl, _cl, _cl, _cl]),
                       ("rk_dev_permute_scale", [_vp, _vp, _vp, _vp, _cl, _cl]),
                       ("rk_dev_rmsnorm_rows", [_vp, _vp, _vp, _cl, _cl, _ci, _cll]),
                       ("rk_dev_rmsnorm_rows_bwd", [_vp, _vp, _vp, _vp, _cl, _cl, _ci, _cll]),
                       ("rk_dev_shr_bias", [_vp, _vp, _vp, _cl, _cl, _ci]),
                       ("rk_dev_transpose_i32", [_vp, _vp, _cl, _cl]),
                       ("rk_dev_i64_to_i32", [_vp, _vp, _cl])):
            f = getattr(lib, nm)
            f.argtypes = at
            f.restype = ctypes.c_int
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
    # ring_gemv_i64: exact fixed-point GEMV, int64 ENERGY accumulation. Reproduce the
    # emulation.infer.linear inner sum (SUM W[j*K+i]*x[i]) bit-for-bit, incl. a large-magnitude row.
    Mg, Kg = 5, 300
    Wg = [((i * 37 - 5000)) for i in range(Mg * Kg)]           # signed, ~[-5000, 6000]
    xg = [((i * 131 - 20000)) for i in range(Kg)]              # signed, up to ~2e4
    wantg = [sum(Wg[j * Kg + i] * xg[i] for i in range(Kg)) for j in range(Mg)]
    Warr = (ctypes.c_int * (Mg * Kg))(*Wg); xarr = (ctypes.c_int * Kg)(*xg)
    og = (ctypes.c_longlong * Mg)()
    if lib.ring_gemv_i64(og, Warr, xarr, Mg, Kg) != 0 or list(og) != wantg:
        return False
    # ring_gemm_i64: batched exact GEMM — out[t,m] = Σ_k X[t,k]*W[m,k], vs the integer reference.
    Tg = 3
    Xg = [(i * 53 - 900) for i in range(Tg * Kg)]
    wantG = [sum(Xg[t * Kg + i] * Wg[m * Kg + i] for i in range(Kg)) for t in range(Tg) for m in range(Mg)]
    XA = (ctypes.c_int * (Tg * Kg))(*Xg); oG = (ctypes.c_longlong * (Tg * Mg))()
    if lib.ring_gemm_i64(oG, XA, Warr, Tg, Mg, Kg) != 0 or list(oG) != wantG:
        return False
    # activations: bit-for-bit == the ract fixed-point reference (Q16), the D9 gate for the GPU path.
    from ringkit.emulation import ract
    fr = 16
    xs = [-(3 << 16), -(1 << 15), 0, (1 << 15), (3 << 16), (7 << 16)]      # moderate: no clamp edge
    want_s = [ract.sigmoid_fixed(v, fr) for v in xs]
    xa = (ctypes.c_longlong * len(xs))(*xs); os_ = (ctypes.c_longlong * len(xs))()
    if lib.ring_sigmoid(os_, xa, len(xs), fr) != 0 or list(os_) != want_s:
        return False
    xn = [-(3 << 16), -(1 << 15), 0, -(7 << 16)]                            # exp domain: x <= 0
    want_e = [ract.exp_fixed(v, fr) for v in xn]
    xea = (ctypes.c_longlong * len(xn))(*xn); oe = (ctypes.c_longlong * len(xn))()
    if lib.ring_exp(oe, xea, len(xn), fr) != 0 or list(oe) != want_e:
        return False
    xr = [3 << 16, 4 << 16, 0, -(5 << 16), 2 << 16]; wr = [1 << 16] * 5
    want_r = ract.rmsnorm_fixed(xr, wr, fr, 1)
    xra = (ctypes.c_longlong * 5)(*xr); wra = (ctypes.c_longlong * 5)(*wr); orr = (ctypes.c_longlong * 5)()
    if lib.ring_rmsnorm(orr, xra, wra, 5, fr, 1) != 0 or list(orr) != want_r:
        return False
    # Q16 elementwise + reductions (compose the forward/backward). emul/escale = FLOOR ((u*v)>>f,
    # signed), matching _mul_q16 / infer.linear. Values include a negative product WITH a remainder so
    # the test distinguishes floor from truncate-toward-zero.
    def _ms(u, v, f):
        return (u * v) >> f                                   # Python >> = arithmetic floor
    ea = [(1 << 16) + 1, -((2 << 16) + 1), (3 << 16)]
    eb = [(2 << 16), (3 << 16) + 1, -((1 << 16) + 1)]
    A = (ctypes.c_longlong * 3)(*ea); B = (ctypes.c_longlong * 3)(*eb); O = (ctypes.c_longlong * 3)()
    if lib.ring_ew_q16(0, O, A, B, 3, fr) != 0 or list(O) != [_ms(ea[i], eb[i], fr) for i in range(3)]:
        return False
    if lib.ring_ew_q16(1, O, A, B, 3, 0) != 0 or list(O) != [ea[i] + eb[i] for i in range(3)]:
        return False
    if lib.ring_ew_q16(2, O, A, B, 3, 0) != 0 or list(O) != [ea[i] - eb[i] for i in range(3)]:
        return False
    sc = (3 << 15) + 7
    if lib.ring_escale(O, A, sc, 3, fr) != 0 or list(O) != [_ms(ea[i], sc, fr) for i in range(3)]:
        return False
    cin = [1, 2, 3, 10, 20, 30, 100, 200, 300]                # 3 rows x 3
    CI = (ctypes.c_longlong * 9)(*cin); CO = (ctypes.c_longlong * 3)()
    if lib.ring_colsum(CO, CI, 3, 3) != 0 or list(CO) != [111, 222, 333]:
        return False
    # toroidal diffuse: one 4-neighbour heat step == the Python reference (arithmetic >>3, wrap).
    Dg, Hg, hg = 2, 3, 2
    grid = [((r * 7 + c * 5 + j * 3) << 8) - 400 for r in range(Dg) for c in range(Hg) for j in range(hg)]
    def _ref(g):
        o = [0] * len(g)
        for r in range(Dg):
            for c in range(Hg):
                for j in range(hg):
                    idx = (r * Hg + c) * hg + j
                    up = (((r - 1) % Dg) * Hg + c) * hg + j
                    dn = (((r + 1) % Dg) * Hg + c) * hg + j
                    lf = (r * Hg + (c - 1) % Hg) * hg + j
                    rt = (r * Hg + (c + 1) % Hg) * hg + j
                    o[idx] = (g[up] + g[dn] + g[lf] + g[rt] + (g[idx] << 2)) >> 3
        return o
    GI = (ctypes.c_longlong * len(grid))(*grid); GO = (ctypes.c_longlong * len(grid))()
    if lib.ring_diffuse(GO, GI, Dg, Hg, hg) != 0 or list(GO) != _ref(grid):
        return False
    # relu (quadrant rectifier): max(x,0)
    rv = [-5, 0, 3, -1, 100]; RA = (ctypes.c_longlong * 5)(*rv); RO = (ctypes.c_longlong * 5)()
    if lib.ring_relu(RO, RA, 5) != 0 or list(RO) != [max(v, 0) for v in rv]:
        return False
    # gather (trig-LUT lookup by arc byte): o[i] = lut[idx[i]]
    lut = [i * 7 - 300 for i in range(256)]; gi = [0, 64, 128, 192, 255, 21]
    LT = (ctypes.c_longlong * 256)(*lut); GX = (ctypes.c_uint8 * 6)(*gi); GG = (ctypes.c_longlong * 6)()
    if lib.ring_gather(GG, LT, 256, GX, 6) != 0 or list(GG) != [lut[i] for i in gi]:
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


def gemv_u8_exact(xbar, x, M, K, s_row, z_row):
    """The emulation-proj fused exact GEMV on the GPU over the raw u8 ONIX slab:
    out[r] = sdiv((Σ (xbar-128)·x) shifted by s_row[r], z_row[r]). Semantics =
    kernels/mprc/gemma/host._py_gemv_exact (the caller D9-gates on first use).
    Returns a Q16 int list, or None when CUDA is unavailable / launch fails."""
    lib = _load()
    if lib is None:
        return None
    try:
        if isinstance(xbar, memoryview) and not xbar.readonly:
            xb = (ctypes.c_uint8 * (M * K)).from_buffer(xbar)        # zero-copy slab read
        elif isinstance(xbar, bytearray):
            xb = (ctypes.c_uint8 * (M * K)).from_buffer(xbar)
        else:
            xb = (ctypes.c_uint8 * (M * K)).from_buffer_copy(bytes(xbar))
    except (TypeError, ValueError):
        xb = (ctypes.c_uint8 * (M * K)).from_buffer_copy(bytes(xbar))
    xa = (ctypes.c_longlong * K)(*[int(v) for v in x])
    sa = (ctypes.c_int8 * M)(*[int(s) for s in s_row])
    za = (ctypes.c_longlong * M)(*[int(z) for z in z_row])
    out = (ctypes.c_longlong * M)()
    if lib.rk_gemv_u8_exact(out, ctypes.cast(xb, _U8), xa, sa, za, M, K) != 0:
        return None
    return list(out)


def slab_upload(xbar_ctypes_array, nbytes):
    """Upload a weight slab once; returns a device handle (int) or None."""
    lib = _load()
    if lib is None:
        return None
    h = lib.rk_dev_upload_u8(ctypes.cast(xbar_ctypes_array, _U8), nbytes)
    return h or None


def slab_free(handle):
    lib = _load()
    if lib is not None and handle:
        lib.rk_dev_free(ctypes.c_void_p(handle))


def gemv_u8_exact_res(handle, x, M, K, s_row, z_row):
    """gemv_u8_exact against a DEVICE-RESIDENT slab handle: only the activation
    and s/z rows cross PCIe. Same exact semantics; same caller-side D9 gate."""
    lib = _load()
    if lib is None:
        return None
    xa = (ctypes.c_longlong * K)(*[int(v) for v in x])
    sa = (ctypes.c_int8 * M)(*[int(s) for s in s_row])
    za = (ctypes.c_longlong * M)(*[int(z) for z in z_row])
    out = (ctypes.c_longlong * M)()
    if lib.rk_gemv_u8_exact_res(out, ctypes.c_void_p(handle), xa, sa, za, M, K) != 0:
        return None
    return list(out)


def sigmoid_vec(xs, frac):
    """out[i] = ract.sigmoid_fixed(xs[i]) on the GPU (bit-for-bit). None only if CUDA unavailable."""
    lib = _load()
    if lib is None:
        return None
    n = len(xs)
    xa = (ctypes.c_longlong * n)(*xs); out = (ctypes.c_longlong * n)()
    if lib.ring_sigmoid(out, xa, n, frac) != 0:
        return None
    return list(out)


def exp_vec(xs, frac):
    """out[i] = exp_fixed(xs[i]) on the GPU — VALID for xs[i] <= 0 (softmax domain). None if CUDA
    unavailable or any input positive (that regime needs the arbitrary-precision Python path)."""
    lib = _load()
    if lib is None:
        return None
    for v in xs:
        if v > 0:
            return None
    n = len(xs)
    xa = (ctypes.c_longlong * n)(*xs); out = (ctypes.c_longlong * n)()
    if lib.ring_exp(out, xa, n, frac) != 0:
        return None
    return list(out)


def rmsnorm(x, weight, frac, eps=1):
    """RMSNorm block on the GPU (== ract.rmsnorm_fixed bit-for-bit). None only if CUDA unavailable."""
    lib = _load()
    if lib is None:
        return None
    n = len(x)
    xa = (ctypes.c_longlong * n)(*x); wa = (ctypes.c_longlong * n)(*weight); out = (ctypes.c_longlong * n)()
    if lib.ring_rmsnorm(out, xa, wa, n, frac, eps) != 0:
        return None
    return list(out)


def gemm_i64(X, W, T, M, K):
    """Batched exact energy GEMM: out[t*M+m] = Sum_k X[t*K+k]*W[m*K+k] (all T tokens in ONE call).
    X, W flat int lists; returns T*M int64. None if CUDA unavailable. The batched linear fast path."""
    lib = _load()
    if lib is None:
        return None
    XA = (ctypes.c_int * (T * K))(*X); WA = (ctypes.c_int * (M * K))(*W); O = (ctypes.c_longlong * (T * M))()
    if lib.ring_gemm_i64(O, XA, WA, T, M, K) != 0:
        return None
    return list(O)


def _ew(op, a, b, frac):
    lib = _load()
    if lib is None:
        return None
    n = len(a)
    A = (ctypes.c_longlong * n)(*a); B = (ctypes.c_longlong * n)(*b); O = (ctypes.c_longlong * n)()
    if lib.ring_ew_q16(op, O, A, B, n, frac) != 0:
        return None
    return list(O)


def emul_q16(a, b, frac=16):
    """Q16 energy elementwise product (a*b)>>frac, arithmetic FLOOR (== _mul_q16). None if unavailable."""
    return _ew(0, a, b, frac)


def eadd(a, b):
    """Energy elementwise a+b (int64, no fold). None if CUDA unavailable."""
    return _ew(1, a, b, 0)


def esub(a, b):
    """Energy elementwise a-b (int64, no fold). None if CUDA unavailable."""
    return _ew(2, a, b, 0)


def escale_q16(a, sc, frac=16):
    """Q16 scale (a*sc)>>frac by a scalar sc (Q16). None if CUDA unavailable."""
    lib = _load()
    if lib is None:
        return None
    n = len(a)
    A = (ctypes.c_longlong * n)(*a); O = (ctypes.c_longlong * n)()
    if lib.ring_escale(O, A, sc, n, frac) != 0:
        return None
    return list(O)


def colsum(rows_flat, R, C):
    """Column-sum over R rows of length C: out[c] = Sum_r rows_flat[r*C+c]. None if CUDA unavailable."""
    lib = _load()
    if lib is None:
        return None
    IN = (ctypes.c_longlong * (R * C))(*rows_flat); O = (ctypes.c_longlong * C)()
    if lib.ring_colsum(O, IN, R, C) != 0:
        return None
    return list(O)


def diffuse(grid_flat, D, H, hd):
    """One toroidal 4-neighbour heat step over a (D,H) grid of hd-vectors (SSD lattice2d). None if
    CUDA unavailable. in/out flat [D*H*hd] Q16; == (up+dn+lf+rt+4*center)>>3, wrap-around."""
    lib = _load()
    if lib is None:
        return None
    n = D * H * hd
    IN = (ctypes.c_longlong * n)(*grid_flat); O = (ctypes.c_longlong * n)()
    if lib.ring_diffuse(O, IN, D, H, hd) != 0:
        return None
    return list(O)


def relu(a):
    """max(a[i], 0) elementwise (frontend quadrant rectifier). None if CUDA unavailable."""
    lib = _load()
    if lib is None:
        return None
    n = len(a)
    A = (ctypes.c_longlong * n)(*a); O = (ctypes.c_longlong * n)()
    if lib.ring_relu(O, A, n) != 0:
        return None
    return list(O)


def gather(lut, idx):
    """out[i] = lut[idx[i]] — LUT lookup by u8 arc byte (trig table). None if CUDA unavailable."""
    lib = _load()
    if lib is None:
        return None
    LT = (ctypes.c_longlong * len(lut))(*lut)
    IX = (ctypes.c_uint8 * len(idx))(*[int(v) & 0xFF for v in idx])
    O = (ctypes.c_longlong * len(idx))()
    if lib.ring_gather(O, LT, len(lut), IX, len(idx)) != 0:
        return None
    return list(O)


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


_I32MIN, _I32MAX = -2147483648, 2147483647
_I64GUARD = 1 << 62          # keep worst-case |acc| = max|W|·max|x|·K safely below int64 (2^63)
_gemv_wcache = {}            # id(W) -> (W, c_int[M*K] | None, max|W|) — weights reused across tokens


def gemv_i64(W, x, M, K):
    """Exact fixed-point GEMV on the GPU: returns [out[0..M-1]] with out[j] = SUM_i W[j*K+i]*x[i]
    as Python ints (int64 accumulation, ENERGY side, no fold). The caller does (out>>frac)+bias to
    match emulation.infer.linear BIT-FOR-BIT. Returns None (caller -> Python) if CUDA is absent, any
    input exceeds int32, OR the worst-case accumulation could exceed int64 (the overflow-hang
    lesson). W is cached per identity (converted + max-scanned once)."""
    lib = _load()
    if lib is None:
        return None
    got = _gemv_wcache.get(id(W))
    if got is None or got[0] is not W:
        arr = (ctypes.c_int * (M * K))(); mx = 0
        for i in range(M * K):
            v = W[i]
            if v < _I32MIN or v > _I32MAX:
                _gemv_wcache[id(W)] = (W, None, 0); return None
            arr[i] = v
            a = -v if v < 0 else v
            if a > mx:
                mx = a
        _gemv_wcache[id(W)] = (W, arr, mx); got = (W, arr, mx)
    if got[1] is None:
        return None
    maxw = got[2]
    xa = (ctypes.c_int * K)(); mxx = 0
    for i in range(K):
        v = x[i]
        if v < _I32MIN or v > _I32MAX:
            return None
        xa[i] = v
        a = -v if v < 0 else v
        if a > mxx:
            mxx = a
    if maxw * mxx * K >= _I64GUARD:                 # int64-overflow guard -> Python (exact) fallback
        return None
    out = (ctypes.c_longlong * M)()
    if lib.ring_gemv_i64(out, got[1], xa, M, K) != 0:
        return None
    return list(out)


# ── RESIDENCY API (SPEC-014 Part B): keep weights/activations on-device across calls. This is the fix
# for the 2%-GPU-util problem — no per-call cudaMalloc/H2D/D2H/free, no per-call Python-list marshaling.
# A training loop uploads the WEIGHT once (`dev_upload_i32`), reuses the handle across tokens/epochs,
# and only the changing activation + the output cross the bus. Same bit-exact k_gemm_i64.
def dev_upload_i32(data):
    """Upload an int32 buffer to the device ONCE; returns an opaque device handle (or None). `data`
    may be a Python list or a prebuilt ctypes c_int array (the latter avoids the list-splat marshal)."""
    lib = _load()
    if lib is None:
        return None
    if isinstance(data, (list, tuple)):
        n = len(data)
        arr = (ctypes.c_int * n)(*data)
    else:                                            # already a ctypes c_int array (no marshaling)
        arr = data
        n = len(arr)
    return lib.rk_dev_upload_i32(arr, n)


def dev_alloc_i64(n):
    """Allocate an int64 device buffer (e.g. the resident output). Returns a handle or None."""
    lib = _load()
    return None if lib is None else lib.rk_dev_alloc_i64(n)


def dev_gemm_i64_resident(d_out, d_x, d_w, T, M, K):
    """Batched energy GEMM on RESIDENT device handles — no transfers, no malloc. Returns 0 on success."""
    lib = _load()
    return -1 if lib is None else lib.rk_dev_gemm_i64_resident(d_out, d_x, d_w, T, M, K)


def dev_download_i64(d_out, n):
    """Download an int64 device buffer to a Python list."""
    lib = _load()
    if lib is None:
        return None
    out = (ctypes.c_longlong * n)()
    return list(out) if lib.rk_dev_download_i64(out, d_out, n) == 0 else None


def dev_free(handle):
    """Free a device handle from dev_upload_i32 / dev_alloc_i64 / dev_upload_u8 / dev_alloc_u8."""
    lib = _load()
    if lib is not None and handle:
        lib.rk_dev_free(handle)


# ── u8 MANIFOLD BROADCAST residency: a batch of N HyperVectors [N][H][W] u8 resident, ONE LUT operator
# broadcast across every site of every manifold. 14 KB/manifold -> a whole dataset fits in VRAM
# (unlike the i64 GEMM output). This is the GPU-utilization pattern for the u8 value/forward path.
def dev_upload_u8(data):
    """Upload a u8 buffer (batch of manifolds) ONCE -> device handle. `data`: bytes/bytearray or list."""
    lib = _load()
    if lib is None:
        return None
    buf = data if isinstance(data, (bytes, bytearray)) else bytes(int(v) & 0xFF for v in data)
    arr = (ctypes.c_uint8 * len(buf)).from_buffer_copy(buf)
    return lib.rk_dev_upload_u8(arr, len(buf))


def dev_alloc_u8(n):
    lib = _load()
    return None if lib is None else lib.rk_dev_alloc_u8(n)


def dev_manifold_staple_u8_resident(d_out, d_grid, d_lut, W, H, N):
    """One manifold-staple sweep broadcast across N resident manifolds. Returns 0 on success."""
    lib = _load()
    return -1 if lib is None else lib.rk_dev_manifold_staple_u8_resident(d_out, d_grid, d_lut, W, H, N)


def dev_download_u8(d, n):
    lib = _load()
    if lib is None:
        return None
    out = (ctypes.c_uint8 * n)()
    return bytes(out) if lib.rk_dev_download_u8(out, d, n) == 0 else None


# ── RESIDENT CHAIN (owner: "do not stream data; process on lanes"): all ops take/return device
# HANDLES; Python sequences launches, only logits + weight-grads cross the boundary. Same ring math.
def raw():
    """The loaded lib (resident calls go straight through: lib.rk_dev_* on ctypes handles)."""
    return _load()


def dev_upload_i64(data):
    lib = _load()
    if lib is None:
        return None
    arr = data if not isinstance(data, (list, tuple)) else (ctypes.c_longlong * len(data))(*data)
    return lib.rk_dev_upload_i64(arr, len(arr))


def dev_alloc_i32(n):
    lib = _load()
    return None if lib is None else lib.rk_dev_alloc_i32(n)
