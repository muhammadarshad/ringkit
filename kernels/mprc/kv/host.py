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
        for fn in (lib.kv_scores, lib.kv_scores_walk):
            fn.argtypes = [_LONG, _U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long]
            fn.restype = None
        lib.kv_argmax.argtypes = [_U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long, _LONG]
        lib.kv_argmax.restype = ctypes.c_long
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


def scores(K, q, n, dim, pitch, walk=False):
    """score[j] = -sum_d ring_distance(q[d], K[j][d]) over the prime-pitched slab. C when available.

    walk=True traverses tokens by the stride-7 QCM quantum walk instead of sequentially."""
    lib = _load()
    if lib is None:
        return py_scores(K, q, n, dim, pitch)
    out = (ctypes.c_long * n)()
    qb = bytearray(q)
    fn = lib.kv_scores_walk if walk else lib.kv_scores
    fn(out, _u8(K), _u8(qb), n, dim, pitch)
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
            for walk in (False, True):
                out = (ctypes.c_long * n)()
                fn = lib.kv_scores_walk if walk else lib.kv_scores
                fn(out, _u8(K), _u8(q), n, dim, pitch)
                if list(out) != want:          # the WALK must produce the SAME scores (bijection)
                    return False
            b = ctypes.c_long(0)
            j = lib.kv_argmax(_u8(K), _u8(q), n, dim, pitch, ctypes.byref(b))
            wj = max(range(n), key=lambda i: want[i])
            if want[int(j)] != want[wj]:
                return False
    return True
