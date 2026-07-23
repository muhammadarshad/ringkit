"""ringkit.kernels.cuda.host — loader for ringkit's OWN GEVHV CUDA backend (D9 silicon).

Separate from ringkit.kernels.nvidia.cuda (that backend serves the elementwise/gemm/activation
ops; this one serves the GEVHV manifold operator: react / measure / the fused bind-absorbed
composition). Pattern mirrored from both siblings: kernels/cpu_rust/host.py (build-on-import,
argtypes, load-time bit-for-bit self-test, Python-fallback contract) and
kernels/nvidia/cuda/host.py (nvcc + vcvars64 MSVC-toolchain injection on Windows).

Build: nvcc -O3 -shared -arch=sm_89 -Xcompiler /MD -> kernels/build/gevhv_cuda-<machine>.dll
(arch pinned per D9 bench discipline — not "-arch=native": this backend's timing, when it is
ever reported, must be reproducible across identical hardware, and CORRECTION_REFOCUS keeps
performance out of the correctness gate entirely; pinning is simply the honest default).

D9 contract: the absorbed LUT (Theorem F: L'[u] = L[(s*u+5t) mod 256]) and the vector-bind
offset field (Theorem F2: c = Sigma5(v)) are computed HOST-SIDE by calling ringkit.ml.gevhv's
own multiplier-free functions directly (absorb_lut / offset_field) — this module never
re-derives that arithmetic; it only hands the already-exact result to the device. Every
exported op is gated bit-for-bit against ringkit.ml.gevhv at load time (_selftest); the
backend is retired to `None` (Python fallback) on ANY self-test failure or absent CUDA.
"""
import ctypes
import os
import platform
import subprocess

from ringkit.ml import gevhv as gv

_DIR = os.path.dirname(__file__)
_CU = os.path.join(_DIR, "gevhv_cuda.cu")
_BUILD = os.path.normpath(os.path.join(_DIR, "..", "build"))
_LIB = os.path.join(_BUILD, f"gevhv_cuda-{platform.machine()}.dll")
_ARCH = "sm_89"

_U8 = ctypes.POINTER(ctypes.c_uint8)
_I32 = ctypes.POINTER(ctypes.c_int)

_CUDA_BIN_CANDIDATES = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin",
]
_VCVARS_CANDIDATES = [
    r"C:\Program Files\Microsoft Visual Studio\2022\{ed}\VC\Auxiliary\Build\vcvarsall.bat".format(ed=e)
    for e in ("Enterprise", "Professional", "Community", "BuildTools")
]

_lib = None
_tried = False


def _find_cuda_bin():
    return next((p for p in _CUDA_BIN_CANDIDATES if os.path.isdir(p)), None)


def _msvc_env():
    """The vlm-1 lesson (shared with kernels/nvidia/cuda/host.py): cl.exe/INCLUDE/LIB exist
    only after vcvarsall x64. Return an env with them injected."""
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
    """nvcc -> the arch-keyed shared lib (pinned -arch=sm_89, D9 bench discipline). Raises on
    failure so callers see the real nvcc/link diagnostics rather than a silent fallback."""
    os.makedirs(_BUILD, exist_ok=True)
    tmp = _LIB + ".tmp"
    cmd = ["nvcc", "-O3", "-shared", f"-arch={_ARCH}", "-Xcompiler", "/MD", "-o", tmp, _CU]
    subprocess.run(cmd, env=_msvc_env(), check=True, capture_output=True, text=True, cwd=_DIR)
    os.replace(tmp, _LIB)


def _load():
    global _lib, _tried
    if _lib is not None or _tried:
        return _lib
    _tried = True
    try:
        if not os.path.exists(_LIB) or os.path.getmtime(_LIB) < os.path.getmtime(_CU):
            build()
        cuda_bin = _find_cuda_bin()
        if platform.system() == "Windows" and cuda_bin:
            os.add_dll_directory(cuda_bin)          # cudart lives here
        lib = ctypes.CDLL(_LIB)
        lib.rk_cuda_available.restype = ctypes.c_int
        lib.rk_gevhv_react.argtypes = [_U8, _U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long]
        lib.rk_gevhv_react.restype = ctypes.c_int
        lib.rk_gevhv_react_bound_scalar.argtypes = [
            _U8, _U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_int]
        lib.rk_gevhv_react_bound_scalar.restype = ctypes.c_int
        lib.rk_gevhv_react_bound_vector.argtypes = [
            _U8, _U8, _U8, _U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long]
        lib.rk_gevhv_react_bound_vector.restype = ctypes.c_int
        lib.rk_gevhv_measure.argtypes = [
            _I32, _U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_int]
        lib.rk_gevhv_measure.restype = ctypes.c_int
        lib.rk_gevhv_fused_scalar.argtypes = [
            _I32, _U8, _U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long,
            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.rk_gevhv_fused_scalar.restype = ctypes.c_int
        lib.rk_gevhv_fused_vector.argtypes = [
            _I32, _U8, _U8, _U8, _U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_int]
        lib.rk_gevhv_fused_vector.restype = ctypes.c_int
        if lib.rk_cuda_available() != 1 or not _selftest(lib):
            _lib = None
            return None
        _lib = lib
    except Exception:
        _lib = None
    return _lib


# ── ctypes helpers ──────────────────────────────────────────────────────────────────────
def _u8arr(data):
    return (ctypes.c_uint8 * len(data))(*[int(x) & 0xFF for x in data])


def _outu8(n):
    return (ctypes.c_uint8 * n)()


def _outi32(n):
    return (ctypes.c_int * n)()


# ── D9 self-test: reproduce ringkit.ml.gevhv bit-for-bit before serving ────────────────
def _rand_grid(rng, h, w):
    sites = h * w
    return [rng.randrange(256) for _ in range(sites)]


def _selftest(lib):
    """Load-time gate: random + adversarial manifolds, including a batch of N=3 independent
    manifolds, reproduced bit-for-bit against ml.gevhv on every op this backend exports."""
    import random
    rng = random.Random(20260717)
    h, w = gv.H, gv.W
    sites = h * w

    # a mixed batch: two random grids + three adversarial constants (all-0 / all-255 / half)
    grids = [_rand_grid(rng, h, w), _rand_grid(rng, h, w),
             [0] * sites, [255] * sites, [128] * sites]
    lut = [rng.randrange(256) for _ in range(256)]
    q = _rand_grid(rng, h, w)
    v = _rand_grid(rng, h, w)
    s_vals = [1, 255, 3, 183]          # 1 and 255 are their own anti-strides (edge units)
    t_vals = [0, 42, 255]

    # -- plain react (identity boundary) --
    N = len(grids)
    flatG = [x for g in grids for x in g]
    got = (ctypes.c_uint8 * (N * sites))()
    if lib.rk_gevhv_react(got, _u8arr(flatG), _u8arr(lut), h, w, N) != 0:
        return False
    want = [x for g in grids for x in gv.react(g, lut)]
    if list(got) != want:
        return False

    # -- react_bound_scalar, several (s, t), batch of 3 --
    batch3 = grids[:3]
    flat3 = [x for g in batch3 for x in g]
    for s in s_vals:
        for t in t_vals:
            lp = gv.absorb_lut(lut, s, t)
            got = (ctypes.c_uint8 * (3 * sites))()
            if lib.rk_gevhv_react_bound_scalar(got, _u8arr(flat3), _u8arr(lp), h, w, 3, s, t) != 0:
                return False
            want = [x for g in batch3 for x in gv.react_bound_scalar(g, lut, s, t)]
            if list(got) != want:
                return False

    # -- react_bound_vector, shared offset field c, batch of 3 --
    c = gv.offset_field(v, h, w)
    got = (ctypes.c_uint8 * (3 * sites))()
    if lib.rk_gevhv_react_bound_vector(got, _u8arr(flat3), _u8arr(lut), _u8arr(v), _u8arr(c),
                                       h, w, 3) != 0:
        return False
    want = [x for g in batch3 for x in gv.react_bound_vector(g, lut, v, c=c)]
    if list(got) != want:
        return False

    # -- measure, full grid and interior-only, batch of 5 (incl. adversarial) --
    for interior in (0, 1):
        got = (ctypes.c_int * N)()
        if lib.rk_gevhv_measure(got, _u8arr(flatG), _u8arr(q), h, w, N, interior) != 0:
            return False
        want = [gv.measure(g, q, interior=bool(interior)) for g in grids]
        if list(got) != want:
            return False

    # -- fused scalar-bind (== ml.gevhv.gevhv_scalar composed), several (s, t), both interior modes --
    for s in s_vals:
        for t in t_vals:
            for interior in (0, 1):
                lp = gv.absorb_lut(lut, s, t)
                got = (ctypes.c_int * N)()
                if lib.rk_gevhv_fused_scalar(got, _u8arr(flatG), _u8arr(lp), _u8arr(q),
                                             h, w, N, s, t, interior) != 0:
                    return False
                want = [gv.gevhv_scalar(g, lut, q, s, t, interior=bool(interior)) for g in grids]
                if list(got) != want:
                    return False

    # -- fused vector-bind (== ml.gevhv.gevhv_vector composed), batch of 5, both interior modes --
    for interior in (0, 1):
        got = (ctypes.c_int * N)()
        if lib.rk_gevhv_fused_vector(got, _u8arr(flatG), _u8arr(lut), _u8arr(v), _u8arr(c),
                                     _u8arr(q), h, w, N, interior) != 0:
            return False
        want = [gv.gevhv_vector(g, lut, q, v, c=c, interior=bool(interior)) for g in grids]
        if list(got) != want:
            return False

    # -- batch independence: re-run manifold 0 alone -> identical energy to its slot in the batch --
    lp0 = gv.absorb_lut(lut, s_vals[0], t_vals[0])
    got_batch = (ctypes.c_int * N)()
    lib.rk_gevhv_fused_scalar(got_batch, _u8arr(flatG), _u8arr(lp0), _u8arr(q),
                              h, w, N, s_vals[0], t_vals[0], 0)
    got_single = (ctypes.c_int * 1)()
    lib.rk_gevhv_fused_scalar(got_single, _u8arr(grids[0]), _u8arr(lp0), _u8arr(q),
                              h, w, 1, s_vals[0], t_vals[0], 0)
    if got_single[0] != got_batch[0]:
        return False

    return True


def available():
    """True iff the CUDA backend built, loaded, self-tested bit-for-bit, and a GPU is present."""
    return _load() is not None


# ── op surface (mirrors ringkit.ml.gevhv's signatures; None if CUDA unavailable) ──────────
def react(G, lut, N, h=gv.H, w=gv.W):
    """Batched identity-boundary react (== ml.gevhv.react per manifold). G is a flat list/bytes
    of length N*h*w. Returns a flat list of length N*h*w, or None if unavailable."""
    lib = _load()
    if lib is None:
        return None
    out = _outu8(N * h * w)
    if lib.rk_gevhv_react(out, _u8arr(G), _u8arr(lut), h, w, N) != 0:
        return None
    return list(out)


def react_bound_scalar(G, lut, s, t, N, h=gv.H, w=gv.W):
    """Batched FUSED scalar-bind react (== ml.gevhv.react_bound_scalar per manifold). The
    absorbed table L' is computed host-side via ml.gevhv.absorb_lut (multiplier-free, exact)."""
    lib = _load()
    if lib is None:
        return None
    lp = gv.absorb_lut(lut, s, t)
    out = _outu8(N * h * w)
    if lib.rk_gevhv_react_bound_scalar(out, _u8arr(G), _u8arr(lp), h, w, N, int(s), int(t)) != 0:
        return None
    return list(out)


def react_bound_vector(G, lut, v, N, h=gv.H, w=gv.W, c=None):
    """Batched FUSED vector-bind react (== ml.gevhv.react_bound_vector per manifold). Pass a
    precomputed `c` (ml.gevhv.offset_field) to share it across a batch; else it is derived here."""
    lib = _load()
    if lib is None:
        return None
    if c is None:
        c = gv.offset_field(v, h, w)
    out = _outu8(N * h * w)
    if lib.rk_gevhv_react_bound_vector(out, _u8arr(G), _u8arr(lut), _u8arr(v), _u8arr(c),
                                       h, w, N) != 0:
        return None
    return list(out)


def measure(G, q, N, h=gv.H, w=gv.W, interior=False):
    """Batched ring-L1 energy readout (== ml.gevhv.measure per manifold) against the shared
    query field q. Returns a list of N ints, or None if unavailable."""
    lib = _load()
    if lib is None:
        return None
    out = _outi32(N)
    if lib.rk_gevhv_measure(out, _u8arr(G), _u8arr(q), h, w, N, 1 if interior else 0) != 0:
        return None
    return list(out)


def gevhv_scalar(G, lut, q, s, t, N, h=gv.H, w=gv.W, interior=False):
    """FUSED scalar-bind GEVHV: measure(react_bound_scalar(G, lut, s, t), q), one pass, no
    intermediate manifold materialized (== ml.gevhv.gevhv_scalar per manifold)."""
    lib = _load()
    if lib is None:
        return None
    lp = gv.absorb_lut(lut, s, t)
    out = _outi32(N)
    if lib.rk_gevhv_fused_scalar(out, _u8arr(G), _u8arr(lp), _u8arr(q), h, w, N,
                                 int(s), int(t), 1 if interior else 0) != 0:
        return None
    return list(out)


def gevhv_vector(G, lut, q, v, N, h=gv.H, w=gv.W, c=None, interior=False):
    """FUSED vector-bind GEVHV: measure(react_bound_vector(G, lut, v), q), one pass
    (== ml.gevhv.gevhv_vector per manifold). Pass `c` to share the offset field across a batch."""
    lib = _load()
    if lib is None:
        return None
    if c is None:
        c = gv.offset_field(v, h, w)
    out = _outi32(N)
    if lib.rk_gevhv_fused_vector(out, _u8arr(G), _u8arr(lut), _u8arr(v), _u8arr(c), _u8arr(q),
                                 h, w, N, 1 if interior else 0) != 0:
        return None
    return list(out)
