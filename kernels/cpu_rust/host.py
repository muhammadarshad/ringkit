"""ringkit.kernels.cpu_rust.host — loader for the native-Rust CPU backend (SPEC-014: CPU = Rust).

Pure PyO3 (no C in Rust, no ctypes): the crate `ring_rust` (kernels/rust) is a Python extension
module built by `maturin develop --release`. This host `import`s it directly (Python-native
args/returns — lists of int, tuples — no ctypes arrays/pointers), builds it on first use if the
import fails, exposes the same op surface as the CUDA host (so ringkit.device dispatches
identically), and SELF-TESTS bit-for-bit against the ring references (qcm.activations /
emulation.ract / pure-Python / the GEVHV judge ml.gevhv) before serving (D9). No float.
"""
import os
import subprocess

_DIR = os.path.dirname(__file__)
_RUST_DIR = os.path.normpath(os.path.join(_DIR, "..", "rust"))
_SRC = os.path.join(_RUST_DIR, "src", "lib.rs")

_lib = None
_tried = False


def build():
    """`maturin develop --release` -> installs ring_rust into the active Python env. Raises on fail.

    maturin refuses to install without a virtualenv/conda env; on a stock (venv-less) interpreter
    we point VIRTUAL_ENV at the running interpreter's own prefix, so it installs into the SAME
    site-packages these tests import from — making build-on-first-use work everywhere (D9)."""
    import sys
    env = dict(os.environ)
    env.setdefault("VIRTUAL_ENV", sys.prefix)
    subprocess.run(["maturin", "develop", "--release"], cwd=_RUST_DIR, check=True,
                   capture_output=True, text=True, env=env)


def _load():
    global _lib, _tried
    if _lib is not None or _tried:
        return _lib
    _tried = True
    try:
        import ring_rust
    except ImportError:
        try:
            build()
            import ring_rust  # noqa: F811 - retry the import after building
        except Exception:
            _lib = None
            return None
    except Exception:
        _lib = None
        return None
    try:
        if ring_rust.ring_rust_probe() != 42 or not _selftest(ring_rust):
            _lib = None
            return None
    except Exception:
        _lib = None
        return None
    _lib = ring_rust
    return _lib


def _selftest(rr):
    """D9 gate: reproduce the ring references bit-for-bit before serving."""
    from ringkit.qcm import activations as A
    fr = 16
    # sigmoid / exp == qcm.activations (the reference device.py checks)
    xs = [-(3 << 16), -(1 << 15), 0, (1 << 15), (3 << 16), (7 << 16)]
    if list(rr.ring_sigmoid(xs, fr)) != [A.sigmoid_fixed(v, fr) for v in xs]:
        return False
    xn = [-(3 << 16), -(1 << 15), 0, -(7 << 16)]
    if list(rr.ring_exp(xn, fr)) != [A.exp_fixed(v, fr) for v in xn]:
        return False
    # rmsnorm == the pure-Python ring reference (qcm.tensor.layernorm-style rms via isqrt)
    xr = [3 << 16, 4 << 16, 0, -(5 << 16), 2 << 16]
    wr = [1 << 16] * 5
    if list(rr.ring_rmsnorm(xr, wr, fr, 1)) != _rmsnorm_ref(xr, wr, fr, 1):
        return False
    # gemm_i64 / gemv_i64 exact integer
    Mg, Kg, Tg = 5, 300, 3
    Wg = [(i * 37 - 5000) for i in range(Mg * Kg)]
    xg = [(i * 131 - 20000) for i in range(Kg)]
    og = rr.ring_gemv_i64(Wg, xg, Mg, Kg)
    if list(og) != [sum(Wg[j * Kg + i] * xg[i] for i in range(Kg)) for j in range(Mg)]:
        return False
    Xg = [(i * 53 - 900) for i in range(Tg * Kg)]
    oG = rr.ring_gemm_i64(Xg, Wg, Tg, Mg, Kg)
    want = [sum(Xg[t * Kg + i] * Wg[m * Kg + i] for i in range(Kg)) for t in range(Tg) for m in range(Mg)]
    if list(oG) != want:
        return False
    # Q16 elementwise (floor), escale, colsum, diffuse, relu, gather — vs pure-Python refs
    ea = [(1 << 16) + 1, -((2 << 16) + 1), (3 << 16)]
    eb = [(2 << 16), (3 << 16) + 1, -((1 << 16) + 1)]
    if list(rr.ring_ew_q16(0, ea, eb, fr)) != [(ea[i] * eb[i]) >> fr for i in range(3)]:
        return False
    if list(rr.ring_ew_q16(1, ea, eb, 0)) != [ea[i] + eb[i] for i in range(3)]:
        return False
    if list(rr.ring_ew_q16(2, ea, eb, 0)) != [ea[i] - eb[i] for i in range(3)]:
        return False
    sc = (3 << 15) + 7
    if list(rr.ring_escale(ea, sc, fr)) != [(ea[i] * sc) >> fr for i in range(3)]:
        return False
    cin = [1, 2, 3, 10, 20, 30, 100, 200, 300]
    if list(rr.ring_colsum(cin, 3, 3)) != [111, 222, 333]:
        return False
    Dg, Hg, hg = 2, 3, 2
    grid = [((r * 7 + c * 5 + j * 3) << 8) - 400 for r in range(Dg) for c in range(Hg) for j in range(hg)]
    if list(rr.ring_diffuse(grid, Dg, Hg, hg)) != _diffuse_ref(grid, Dg, Hg, hg):
        return False
    rv = [-5, 0, 3, -1, 100]
    if list(rr.ring_relu(rv)) != [max(v, 0) for v in rv]:
        return False
    lut = [i * 7 - 300 for i in range(256)]
    gi = [0, 64, 128, 192, 255, 21]
    if list(rr.ring_gather(lut, bytes(gi))) != [lut[i] for i in gi]:
        return False
    # byte ring elementwise (mul/add/sub mod 256) — the op surface kernels.backend needs (D9:
    # bit-for-bit vs the same `(a*b)&0xFF` etc. reference kernels/backend/__init__.py's _PY uses)
    ea8 = bytes([7, 200, 0, 255, 128, 1])
    eb8 = bytes([9, 5, 255, 255, 3, 254])
    _EW = {"ring_mul": (rr.ring_mul, lambda x, y: (x * y) & 0xFF),
           "ring_add": (rr.ring_add, lambda x, y: (x + y) & 0xFF),
           "ring_sub": (rr.ring_sub, lambda x, y: (x - y) & 0xFF)}
    for fn, ref in _EW.values():
        got = fn(ea8, eb8)
        if list(got) != [ref(ea8[i], eb8[i]) for i in range(len(ea8))]:
            return False
    # GEVHV ops — bit-for-bit vs the pure-Python judge ml.gevhv (D9). Lazy import so a
    # kernel load never hard-depends on the ml layer at import time.
    if not _gevhv_selftest(rr):
        return False
    return True


def _gevhv_selftest(rr):
    """Gate the Rust GEVHV ops against ringkit.ml.gevhv (the judge) on adversarial + random
    manifolds at the operator's own dims (H=128, W=113), before this backend may serve them."""
    import random
    from ringkit.ml import gevhv as gv
    H, W = gv.H, gv.W
    N = H * W
    rng = random.Random(20260717)

    def rand_grid():
        return [rng.randint(0, 255) for _ in range(N)]

    grids = [rand_grid(), rand_grid(), [0] * N, [128] * N, [255] * N]
    luts = [[rng.randint(0, 255) for _ in range(256)] for _ in grids]

    # (a) react
    for g, lut in zip(grids, luts):
        if list(rr.gevhv_react(bytes(g), bytes(lut), H, W)) != gv.react(g, lut):
            return False
    # (b) measure (u32; full + interior), incl. adversarial Theorem-G bound
    q = rand_grid()
    for g in grids:
        for interior in (False, True):
            if rr.gevhv_measure(bytes(g), bytes(q), H, W, interior) != gv.measure(g, q, interior=interior):
                return False
    # (c) react_bound_scalar + even-s rejection
    for g, lut in zip(grids, luts):
        for s, t in ((1, 0), (3, 7), (255, 200), (89, 13)):
            if list(rr.gevhv_react_bound_scalar(bytes(g), bytes(lut), s, t, H, W)) != gv.react_bound_scalar(g, lut, s, t):
                return False
    try:
        rr.gevhv_react_bound_scalar(bytes(grids[0]), bytes(luts[0]), 2, 5, H, W)
        return False  # even s MUST be refused
    except Exception:
        pass
    # (d) offset_field + react_bound_vector (batch sharing one c)
    v = rand_grid()
    c_out = list(rr.gevhv_offset_field(bytes(v), H, W))
    if c_out != gv.offset_field(v):
        return False
    for g, lut in zip(grids, luts):
        if list(rr.gevhv_react_bound_vector(bytes(g), bytes(lut), bytes(v), bytes(c_out), H, W)) != gv.react_bound_vector(g, lut, v, c=c_out):
            return False
    # (e) gevhv_scores — the attention GEMM-role op == -Σ ring_distance (ml.attention.scores)
    from ringkit.ml.attention import scores as _ref_scores
    dim, nq, nk = 24, 5, 7
    Q = [[rng.randint(0, 255) for _ in range(dim)] for _ in range(nq)]
    K = [[rng.randint(0, 255) for _ in range(dim)] for _ in range(nk)]
    qf = bytes([v for row in Q for v in row]); kf = bytes([v for row in K for v in row])
    got = list(rr.gevhv_scores(qf, kf, nq, nk, dim))
    want = [s for row in _ref_scores(Q, K) for s in row]
    if got != want:
        return False
    # (f) gevhv_gemv_radix (Theorem C, multiplier-free) == exact dot Σ w·x
    Mg, Kg = 6, 40
    wv = [rng.randint(-500, 500) for _ in range(Mg * Kg)]
    xv = [rng.randint(0, 255) for _ in range(Kg)]
    gr = list(rr.gevhv_gemv_radix(wv, bytes(xv), Mg, Kg))
    if gr != [sum(wv[r * Kg + i] * xv[i] for i in range(Kg)) for r in range(Mg)]:
        return False
    # (g) gevhv_gemm_arc (multiplier-free, mod-256) == the rn.mul reference
    from ringkit.core import native as _rn
    Ma, Ka, Na = 4, 12, 5
    A = [rng.randint(0, 255) for _ in range(Ma * Ka)]
    B = [rng.randint(0, 255) for _ in range(Ka * Na)]
    ca = list(rr.gevhv_gemm_arc(bytes(A), bytes(B), Ma, Ka, Na))
    want_c = [sum(_rn.mul(A[i * Ka + kk], B[kk * Na + j]) for kk in range(Ka)) & 0xFF
              for i in range(Ma) for j in range(Na)]
    if ca != want_c:
        return False
    return True


def _rmsnorm_ref(x, w, frac, eps):
    from ringkit.core import native as rn
    n = len(x)
    ssq = 0
    for v in x:
        a = -v if v < 0 else v
        ssq += (a * a) >> frac
    ms = ssq // n + eps
    rms = rn.isqrt(ms << frac) or 1
    out = []
    for i in range(n):
        axf = ((-x[i] if x[i] < 0 else x[i]) << frac) // rms
        norm = -axf if x[i] < 0 else axf
        neg = (norm < 0)
        r = (abs(norm) * abs(w[i])) >> frac
        out.append(-r if neg != (w[i] < 0) else r)
    return out


def _diffuse_ref(g, D, H, hd):
    o = [0] * len(g)
    for r in range(D):
        for c in range(H):
            for j in range(hd):
                idx = (r * H + c) * hd + j
                up = (((r - 1) % D) * H + c) * hd + j
                dn = (((r + 1) % D) * H + c) * hd + j
                lf = (r * H + (c - 1) % H) * hd + j
                rt = (r * H + (c + 1) % H) * hd + j
                o[idx] = (g[up] + g[dn] + g[lf] + g[rt] + (g[idx] << 2)) >> 3
    return o


def available():
    return _load() is not None


_EW_OPS = ("ring_mul", "ring_add", "ring_sub")


def elementwise(op, out, a, b, n):
    """Byte ring elementwise into `out` (bytearray). op in ring_mul/add/sub. Returns 0 on
    success, -1 if the Rust backend is unavailable or op is unknown (mirrors the CUDA/Metal
    host API so kernels.backend's registry can dispatch to this backend uniformly)."""
    lib = _load()
    if lib is None or op not in _EW_OPS:
        return -1
    # pyo3's `&[u8]` extractor only accepts immutable `bytes` (a `bytearray` could be mutated
    # from Python while Rust holds the borrow — pyo3 refuses it for soundness), so this MUST be
    # `bytes(...)`, not a bytearray slice. Still a single bulk copy, not per-element boxing like
    # `list(...)` would be (D1: real measured cost on the D9 throughput gates).
    ab = bytes(a[:n])
    bb = bytes(b[:n])
    try:
        result = getattr(lib, op)(ab, bb)
    except Exception:
        return -1
    out[:n] = result
    return 0


# ── op surface (same names as the CUDA host so ringkit.device dispatches identically) ──
def sigmoid_vec(xs, frac):
    lib = _load()
    if lib is None:
        return None
    return list(lib.ring_sigmoid(list(xs), frac))


def exp_vec(xs, frac):
    lib = _load()
    if lib is None:
        return None
    for v in xs:
        if v > 0:
            return None
    return list(lib.ring_exp(list(xs), frac))


def rmsnorm(x, weight, frac, eps=1):
    lib = _load()
    if lib is None:
        return None
    try:
        return list(lib.ring_rmsnorm(list(x), list(weight), frac, eps))
    except Exception:
        return None


def gemm_i64(X, W, T, M, K):
    lib = _load()
    if lib is None:
        return None
    try:
        return list(lib.ring_gemm_i64(list(X), list(W), T, M, K))
    except Exception:
        return None


def gemv(W, x, M, K):
    lib = _load()
    if lib is None:
        return None
    try:
        return list(lib.ring_gemv_i64(list(W), list(x), M, K))
    except Exception:
        return None


def _ew(op, a, b, frac):
    lib = _load()
    if lib is None:
        return None
    try:
        return list(lib.ring_ew_q16(op, list(a), list(b), frac))
    except Exception:
        return None


def emul_q16(a, b, frac=16):
    return _ew(0, a, b, frac)


def eadd(a, b):
    return _ew(1, a, b, 0)


def esub(a, b):
    return _ew(2, a, b, 0)


def escale_q16(a, sc, frac=16):
    lib = _load()
    if lib is None:
        return None
    return list(lib.ring_escale(list(a), sc, frac))


def colsum(rows_flat, R, C):
    lib = _load()
    if lib is None:
        return None
    try:
        return list(lib.ring_colsum(list(rows_flat), R, C))
    except Exception:
        return None


def diffuse(grid_flat, D, H, hd):
    lib = _load()
    if lib is None:
        return None
    try:
        return list(lib.ring_diffuse(list(grid_flat), D, H, hd))
    except Exception:
        return None


def relu(a):
    lib = _load()
    if lib is None:
        return None
    return list(lib.ring_relu(list(a)))


def gather(lut, idx):
    lib = _load()
    if lib is None:
        return None
    idx8 = bytes(int(v) & 0xFF for v in idx)
    try:
        return list(lib.ring_gather(list(lut), idx8))
    except Exception:
        return None


# ── GEVHV op surface (mirrors ml.gevhv names; None -> caller falls back to Python) ────────
# `g`/`lut`/`q`/`v`/`c` cross into Rust as `&[u8]` (zero-copy from bytes/bytearray) — `bytes(...)`
# here, not `list(...)`, so a 14,464-site manifold doesn't box every site as a Python int.
def gevhv_react(g, lut, h, w):
    lib = _load()
    if lib is None:
        return None
    try:
        return list(lib.gevhv_react(bytes(g), bytes(lut), h, w))
    except Exception:
        return None


def gevhv_measure(g, q, h, w, interior=False):
    """The ONE measure op — u32 (ENERGY), exact by Theorem G over the whole domain."""
    lib = _load()
    if lib is None:
        return None
    try:
        return lib.gevhv_measure(bytes(g), bytes(q), h, w, bool(interior))
    except Exception:
        return None


def gevhv_scores(q_flat, k_flat, nq, nk, dim):
    """Attention GEMM-role op: the nq×nk score matrix (flat), score[i,j] = -Σ_d cdist(q_i,k_j)
    — GEVHV's measure batched over the Q×K grid, the ring-native stand-in for Q·Kᵀ. None -> caller
    falls back to the pure-Python ml.attention.scores."""
    lib = _load()
    if lib is None:
        return None
    try:
        return list(lib.gevhv_scores(bytes(q_flat), bytes(k_flat), nq, nk, dim))
    except Exception:
        return None


def gevhv_gemv_radix(w, x, m, k):
    """Theorem C multiplier-free gemv: out[r] = Σ_j w[r,j]·x[j] via 8 bit-plane accumulators
    (shifts+adds, no '*'). w signed ints, x u8. None -> caller falls back."""
    lib = _load()
    if lib is None:
        return None
    try:
        return list(lib.gevhv_gemv_radix(list(w), bytes(x), m, k))
    except Exception:
        return None


def gevhv_gemm_arc(a, b, m, k, n):
    """Multiplier-free arc-side ring GEMM C=(A@B)&0xFF (Theorem C shift-add + B1 fold). Returns a
    bytes-like, or None -> caller falls back to the Python/other backend."""
    lib = _load()
    if lib is None:
        return None
    try:
        return bytes(lib.gevhv_gemm_arc(bytes(a), bytes(b), m, k, n))
    except Exception:
        return None


def gevhv_react_bound_scalar(g, lut, s, t, h, w):
    lib = _load()
    if lib is None:
        return None
    try:
        return list(lib.gevhv_react_bound_scalar(bytes(g), bytes(lut), int(s), int(t), h, w))
    except Exception:
        return None


def gevhv_offset_field(v, h, w):
    lib = _load()
    if lib is None:
        return None
    try:
        return list(lib.gevhv_offset_field(bytes(v), h, w))
    except Exception:
        return None


def gevhv_react_bound_vector(g, lut, v, c, h, w):
    lib = _load()
    if lib is None:
        return None
    try:
        return list(lib.gevhv_react_bound_vector(bytes(g), bytes(lut), bytes(v), bytes(c), h, w))
    except Exception:
        return None
