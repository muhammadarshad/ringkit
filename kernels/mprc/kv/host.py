"""
ringkit.kernels.mprc.kv.host — ctypes host for the KV-cache silicon (charter D9).

The cache's key/value memory is ONE contiguous uint8 buffer (a Python bytearray whose C memory is
handed to the kernel ZERO-COPY), row-major: token j is K[j*dim : (j+1)*dim]. Not a list of rows —
a flat slab, which is what lets the scan vectorize and what lets C own the data.

D9 contract: the C kernel must reproduce the multiplier-free semantic reference
(ringkit.ml.kvcache.score_row) BIT-FOR-BIT. It is self-tested at load; on any disagreement — or if
the toolchain is absent — the Python reference serves instead and `available()` reports False.
"""
import ctypes
import os
import platform
import subprocess

_DIR = os.path.dirname(__file__)
_BUILD = os.path.join(_DIR, "..", "..", "build")
_C = os.path.join(_DIR, "kv_cache.c")
_U8 = ctypes.POINTER(ctypes.c_uint8)
_LONG = ctypes.POINTER(ctypes.c_long)

_lib = None
_tried = False


def next_prime(m):
    """The QCM cache manifold: row pitch must be PRIME, never a power of two, or successive token
    rows alias into the same cache sets (kernels/mprc/qcm/cache_manifold.c proves this)."""
    m = int(m)
    if m < 2:
        return 2
    c = m
    while True:
        p = True
        d = 2
        while d + d <= c:                      # d*d <= c without a multiply
            if c % d == 0:
                p = False
                break
            d += 1
        if p:
            return c
        c += 1


def so_path(stem):
    return os.path.join(_BUILD, f"{stem}-{platform.machine()}.so")


def _arch_flags():
    """Target the RUNNING interpreter's arch. On Darwin -march=native is unusable: it probes the
    HOST cpu (apple-m1) even when -arch cross-targets an x86_64 Python under Rosetta."""
    import sys
    if sys.platform == "darwin":
        return ["-arch", platform.machine()]
    return ["-march=native"]


def _build():
    os.makedirs(_BUILD, exist_ok=True)
    so = so_path("kv_cache")
    if not os.path.exists(so) or os.path.getmtime(_C) > os.path.getmtime(so):
        subprocess.run(["cc", "-O3", "-funroll-loops", "-shared", "-fPIC",
                        *_arch_flags(), "-o", so, _C], check=True, capture_output=True)
    return so


def _load():
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    try:
        lib = ctypes.CDLL(_build())
        lib.kv_scores.argtypes = [_LONG, _U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long]
        lib.kv_scores.restype = None
        lib.kv_argmax.argtypes = [_U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long, _LONG]
        lib.kv_argmax.restype = ctypes.c_long
        lib.kv_blend.argtypes = [_U8, _U8, _LONG, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_long]
        lib.kv_blend.restype = None
        if _selftest(lib):
            _lib = lib
    except Exception:
        _lib = None
    return _lib


def available():
    return _load() is not None


def _u8(ba):
    """Zero-copy view of a bytearray's C memory. The cache OWNS this slab; C reads it in place."""
    return (ctypes.c_uint8 * len(ba)).from_buffer(ba)


def py_scores(K, q, n, dim, pitch):
    """Semantic reference: the multiplier-free score row, straight off the prime-pitched slab.

    The row offset walks by += pitch (no '*' — this is the semantic side of the D9 boundary)."""
    from ringkit.ml.attention import ring_distance
    out = []
    off = 0
    for _ in range(n):
        s = 0
        for d in range(dim):
            s -= ring_distance(q[d], K[off + d])
        out.append(s)
        off += pitch
    return out


def scores(K, q, n, dim, pitch):
    """score[j] = -sum_d ring_distance(q[d], K[j][d]) over the prime-pitched slab. C when available.

    Sequential traversal only: the stride-7 walk variant measured ~2% slower (identical scores)
    and was removed — commit 5f755df holds the measurement."""
    lib = _load()
    if lib is None:
        return py_scores(K, q, n, dim, pitch)
    out = (ctypes.c_long * n)()
    qb = bytearray(q)
    lib.kv_scores(out, _u8(K), _u8(qb), n, dim, pitch)
    return list(out)


def argmax(K, q, n, dim, pitch):
    """Fused score+argmax in ONE pass over the cache. Returns (best_index, best_score)."""
    lib = _load()
    if lib is None:
        s = py_scores(K, q, n, dim, pitch)
        best = 0
        for j in range(1, n):
            if s[j] > s[best]:
                best = j
        return best, s[best]
    best = ctypes.c_long(0)
    qb = bytearray(q)
    j = lib.kv_argmax(_u8(K), _u8(qb), n, dim, pitch, ctypes.byref(best))
    return int(j), int(best.value)


def blend(V_slab, w, n, dim, pitch, best):
    """Circular value blend around V[best] over the prime-pitched value slab. C when available, else
    the ringkit.ml.kvcache.circular_blend Python reference. Returns a dim-length list (ARC values)."""
    lib = _load()
    if lib is None:
        from ringkit.ml import kvcache as _kv
        V = [[V_slab[j * pitch + d] for d in range(dim)] for j in range(n)]
        return _kv.circular_blend(V, list(w), best)
    out = (ctypes.c_uint8 * dim)()
    wl = (ctypes.c_long * n)(*[int(x) for x in w])
    lib.kv_blend(out, _u8(V_slab), wl, n, dim, pitch, best)
    return list(out)


def _selftest(lib):
    """D9 gate: C must equal the multiplier-free reference BIT-FOR-BIT, or it does not serve."""
    import random
    rnd = random.Random(0)
    for dim in (1, 4, 16):
        pitch = next_prime(dim)
        for n in (1, 3, 9, 15):
            K = bytearray(rnd.randrange(256) for _ in range(n * pitch))
            q = bytearray(rnd.randrange(256) for _ in range(dim))
            want = py_scores(K, q, n, dim, pitch)
            out = (ctypes.c_long * n)()
            lib.kv_scores(out, _u8(K), _u8(q), n, dim, pitch)
            if list(out) != want:
                return False
            b = ctypes.c_long(0)
            j = lib.kv_argmax(_u8(K), _u8(q), n, dim, pitch, ctypes.byref(b))
            wj = max(range(n), key=lambda i: want[i])
            if want[int(j)] != want[wj]:
                return False
            # kv_blend must equal the circular_blend reference bit-for-bit
            from ringkit.ml import kvcache as _kv
            V_slab = bytearray(rnd.randrange(256) for _ in range(n * pitch))
            w = [rnd.randrange(256) for _ in range(n)]
            w[rnd.randrange(n)] = 255                       # ensure positive mass
            Vrows = [[V_slab[jj * pitch + d] for d in range(dim)] for jj in range(n)]
            best = max(range(n), key=lambda i: w[i])
            out = (ctypes.c_uint8 * dim)()
            wl = (ctypes.c_long * n)(*w)
            lib.kv_blend(out, _u8(V_slab), wl, n, dim, pitch, best)
            if list(out) != _kv.circular_blend(Vrows, w, best):
                return False
    return True


# ── ADI element (ARC-only: encode/decode = differential/accumulation, uint8 mod 256) ──────────
_C_ADI = os.path.join(_DIR, "kvadi.c")
_adi = None
_adi_tried = False


def _build_adi():
    os.makedirs(_BUILD, exist_ok=True)
    so = so_path("kvadi")
    if not os.path.exists(so) or os.path.getmtime(_C_ADI) > os.path.getmtime(so):
        subprocess.run(["cc", "-O3", "-funroll-loops", "-shared", "-fPIC",
                        *_arch_flags(), "-o", so, _C_ADI], check=True, capture_output=True)
    return so


def _load_adi():
    global _adi, _adi_tried
    if _adi_tried:
        return _adi
    _adi_tried = True
    try:
        lib = ctypes.CDLL(_build_adi())
        for fn in (lib.adi_encode_batch, lib.adi_decode_batch):
            fn.restype = None
        lib.adi_encode_batch.argtypes = [_U8, _U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long]
        lib.adi_decode_batch.argtypes = [_U8, _U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long]
        if _selftest_adi(lib):
            _adi = lib
    except Exception:
        _adi = None
    return _adi


def adi_available():
    return _load_adi() is not None


def adi_encode_batch(rows, R, dim, pitch):
    """R rows (contiguous uint8 slab, row r at r*pitch) -> (leads bytearray, deltas slab). C fast
    path when available, else the ringkit.ml.kvadi Python reference. ARC only (mod 256)."""
    lib = _load_adi()
    if lib is None:
        return _py_adi_encode(rows, R, dim, pitch)
    leads = bytearray(R)
    deltas = bytearray(R * pitch)
    lib.adi_encode_batch(_u8(leads), _u8(deltas), _u8(bytearray(rows)), R, dim, pitch)
    return leads, deltas


def adi_decode_batch(leads, deltas, R, dim, pitch):
    """(leads, deltas slab) -> rows slab, exactly (accumulation). C when available, else Python."""
    lib = _load_adi()
    if lib is None:
        return _py_adi_decode(leads, deltas, R, dim, pitch)
    rows = bytearray(R * pitch)
    lib.adi_decode_batch(_u8(rows), _u8(bytearray(leads)), _u8(bytearray(deltas)), R, dim, pitch)
    return rows


def _py_adi_encode(rows, R, dim, pitch):
    """Semantic reference (multiplier-free): per-row ringkit.ml.kvadi.encode over the slab."""
    from ringkit.ml import kvadi as ka
    leads = bytearray(R)
    deltas = bytearray(R * pitch)
    for r in range(R):
        base = 0
        off = r * pitch
        row = [rows[off + i] for i in range(dim)]
        lead, delta = ka.encode(row)
        leads[r] = lead
        for i in range(len(delta)):
            deltas[off + i] = delta[i]
    return leads, deltas


def _py_adi_decode(leads, deltas, R, dim, pitch):
    from ringkit.ml import kvadi as ka
    rows = bytearray(R * pitch)
    for r in range(R):
        off = r * pitch
        delta = [deltas[off + i] for i in range(dim - 1)]
        row = ka.decode(leads[r], delta)
        for i in range(dim):
            rows[off + i] = row[i]
    return rows


def adi_encode_rows(rows):
    """Ergonomic list API: [row, ...] (equal length) -> (leads list, deltas list-of-lists). Uses the
    C batch kernel when available (bit-for-bit == ringkit.ml.kvadi.encode), else the Python fallback."""
    if not rows:
        return [], []
    dim = len(rows[0])
    if any(len(r) != dim for r in rows):
        raise ValueError("adi_encode_rows: all rows must have equal length")
    R = len(rows)
    pitch = next_prime(dim)
    slab = bytearray(R * pitch)
    for r in range(R):
        off = r * pitch
        for i in range(dim):
            slab[off + i] = int(rows[r][i]) & 0xFF
    leads, deltas = adi_encode_batch(slab, R, dim, pitch)
    return list(leads), [[deltas[r * pitch + i] for i in range(dim - 1)] for r in range(R)]


def adi_decode_rows(leads, deltas):
    """Inverse of adi_encode_rows: (leads, deltas list-of-lists) -> [row, ...], exactly."""
    R = len(leads)
    if R == 0:
        return []
    dim = len(deltas[0]) + 1
    pitch = next_prime(dim)
    dslab = bytearray(R * pitch)
    for r in range(R):
        off = r * pitch
        for i in range(dim - 1):
            dslab[off + i] = int(deltas[r][i]) & 0xFF
    rows = adi_decode_batch(bytearray(leads), dslab, R, dim, pitch)
    return [[rows[r * pitch + i] for i in range(dim)] for r in range(R)]


def _selftest_adi(lib):
    """D9 gate: C batch encode/decode must equal the ringkit.ml.kvadi reference BIT-FOR-BIT."""
    import random
    from ringkit.ml import kvadi as ka
    rnd = random.Random(1)
    for dim in (1, 2, 4, 16):
        pitch = next_prime(dim)
        R = 7
        rows = bytearray(rnd.randrange(256) for _ in range(R * pitch))
        # C encode
        leads = bytearray(R)
        deltas = bytearray(R * pitch)
        lib.adi_encode_batch(_u8(leads), _u8(deltas), _u8(bytearray(rows)), R, dim, pitch)
        # Python reference per row + compare, then C decode must recover the rows exactly
        for r in range(R):
            off = r * pitch
            lead, delta = ka.encode([rows[off + i] for i in range(dim)])
            if leads[r] != lead or [deltas[off + i] for i in range(dim - 1)] != list(delta):
                return False
        out = bytearray(R * pitch)
        lib.adi_decode_batch(_u8(out), _u8(leads), _u8(deltas), R, dim, pitch)
        for r in range(R):
            off = r * pitch
            if [out[off + i] for i in range(dim)] != [rows[off + i] for i in range(dim)]:
                return False
    return True
