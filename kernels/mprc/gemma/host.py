"""
ringkit.kernels.mprc.gemma.host — ctypes host for the ENERGY-domain QSM GEMV (charter D9).

The emulation fast path: out[row] = sum_k (xbar[row*K+k]-128)*x[k], int64 accumulate, no fold. The
C kernel uses ringkit's quarter-square QSM (multiplier-free), and must reproduce the semantic
reference `sum(rn.qsm(w-128, x))` BIT-FOR-BIT (self-tested at load; Python fallback if absent).
"""
import ctypes
import os
import platform
import subprocess

_DIR = os.path.dirname(__file__)
_BUILD = os.path.join(_DIR, "..", "..", "build")
_C = os.path.join(_DIR, "qsm_energy.c")
_U8 = ctypes.POINTER(ctypes.c_uint8)
_U16 = ctypes.POINTER(ctypes.c_uint16)
_I8 = ctypes.POINTER(ctypes.c_int8)
_I64 = ctypes.POINTER(ctypes.c_int64)
_lib = None
_tried = False


def _arch_flags():
    import sys
    return ["-arch", platform.machine()] if sys.platform == "darwin" else ["-march=native"]


def _build():
    os.makedirs(_BUILD, exist_ok=True)
    so = os.path.join(_BUILD, f"qsm_energy-{platform.machine()}.so")
    if not os.path.exists(so) or os.path.getmtime(_C) > os.path.getmtime(so):
        subprocess.run(["cc", "-O3", "-funroll-loops", "-shared", "-fPIC", *_arch_flags(), "-o", so, _C],
                       check=True, capture_output=True)
    return so


def _load():
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    try:
        lib = ctypes.CDLL(_build())
        lib.qsm_dot.argtypes = [_I64, _U8, _I8, ctypes.c_long, ctypes.c_long]
        lib.qsm_dot.restype = None
        lib.lm_argmax.argtypes = [ctypes.POINTER(ctypes.c_int32), _U16,
                                  ctypes.c_long, ctypes.c_long, ctypes.c_int, _I64]
        lib.lm_argmax.restype = ctypes.c_long
        lib.lm_argmax_file.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_char_p, ctypes.c_long,
                                       ctypes.c_long, ctypes.c_long, ctypes.c_int, _I64]
        lib.lm_argmax_file.restype = ctypes.c_long
        if _selftest(lib) and _selftest_lm(lib):
            _lib = lib
    except Exception:
        _lib = None
    return _lib


def available():
    return _load() is not None


def _py_dot(xbar, x, M, K):
    """Semantic reference: energy QSM dot via ringkit rn.qsm (multiplier-free), int accumulate."""
    from ringkit.core import native as rn
    out = []
    for r in range(M):
        base = r * K
        acc = 0
        for k in range(K):
            acc += rn.qsm(xbar[base + k] - 128, x[k])
        out.append(acc)
    return out


def qsm_dot(xbar, x, M, K):
    """out[row] = sum_k (xbar[row*K+k]-128)*x[k], int64, no fold. C when available, else Python."""
    lib = _load()
    if lib is None:
        return _py_dot(xbar, x, M, K)
    xb = xbar if isinstance(xbar, (bytes, bytearray)) else bytearray(xbar)
    xb = xb if isinstance(xb, bytearray) else bytearray(xb)
    xa = (ctypes.c_int8 * K)(*[int(v) for v in x])
    out = (ctypes.c_int64 * M)()
    lib.qsm_dot(out, (ctypes.c_uint8 * len(xb)).from_buffer(xb), xa, M, K)
    return list(out)


def _f16_fixed_py(h, shift):
    """Semantic reference for the C f16->fixed decode (integer bit ops, no float)."""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    man = h & 0x3FF
    if exp == 0:
        sh = shift - 24
        val = (man << sh) if sh >= 0 else (man >> (-sh))
    elif exp == 0x1F:
        val = 0
    else:
        mant = (1 << 10) | man
        sh = shift + (exp - 15) - 10
        val = (mant << sh) if sh >= 0 else (mant >> (-sh))
    return -val if sign else val


def _py_lm_argmax(hidden, emb, V, H, shift):
    """Reference: argmax_v sum_i hidden[i]*f16_fixed(emb[v*H+i]). Returns (best_id, best_dot)."""
    bi, bs = 0, None
    for v in range(V):
        base = v * H
        acc = 0
        for i in range(H):
            acc += hidden[i] * _f16_fixed_py(emb[base + i], shift)
        if bs is None or acc > bs:
            bs, bi = acc, v
    return bi, bs


def lm_argmax(hidden, emb, V, H, shift=13):
    """Tied LM-head greedy argmax over the f16 embedding table. hidden: Q<frac> int32 list;
    emb: a writable buffer (mmap/bytearray) of V*H little-endian uint16, OR a list of uint16.
    C fast path (zero-copy over the mmap), Python fallback."""
    lib = _load()
    if lib is None:
        seq = emb if isinstance(emb, list) else memoryview(emb).cast("H")
        return _py_lm_argmax(hidden, seq, V, H, shift)
    ha = (ctypes.c_int32 * H)(*[int(v) for v in hidden])
    best = (ctypes.c_int64 * 1)()
    if isinstance(emb, list):
        ep = (ctypes.c_uint16 * len(emb))(*emb)
    else:                                    # mmap / bytearray: cast the raw address, no copy
        char = (ctypes.c_char * (V * H * 2)).from_buffer(emb)
        ep = ctypes.cast(ctypes.addressof(char), _U16)
    bi = lib.lm_argmax(ha, ep, V, H, shift, best)
    return bi, best[0]


def lm_argmax_file(hidden, path, off, V, H, shift=13):
    """Greedy LM-head argmax; the C kernel mmaps `path` READ-ONLY at byte `off` (zero-copy streaming,
    no Python-held embedding memory — the on-device path). Falls back to reading the file in Python."""
    lib = _load()
    if lib is not None:
        ha = (ctypes.c_int32 * H)(*[int(v) for v in hidden])
        best = (ctypes.c_int64 * 1)()
        bi = lib.lm_argmax_file(ha, path.encode(), off, V, H, shift, best)
        if bi >= 0:
            return bi, best[0]
    import mmap as _m
    with open(path, "rb") as f:
        mm = _m.mmap(f.fileno(), 0, prot=_m.PROT_READ)
        seq = memoryview(mm)[off:off + V * H * 2].cast("H")
        r = _py_lm_argmax(hidden, seq, V, H, shift)
        mm.close()
    return r


def _selftest_lm(lib):
    import random
    rnd = random.Random(3)
    V, H, shift = 40, 33, 13
    hidden = [rnd.randrange(-70000, 70000) for _ in range(H)]
    emb = [rnd.randrange(0, 65536) for _ in range(V * H)]
    ha = (ctypes.c_int32 * H)(*hidden)
    ea = (ctypes.c_uint16 * (V * H))(*emb)
    best = (ctypes.c_int64 * 1)()
    bi = lib.lm_argmax(ha, ea, V, H, shift, best)
    wi, wbest = _py_lm_argmax(hidden, emb, V, H, shift)
    return bi == wi and best[0] == wbest


def _selftest(lib):
    import random
    rnd = random.Random(0)
    for M, K in ((3, 8), (5, 33)):
        xbar = bytearray(rnd.randrange(256) for _ in range(M * K))
        x = [rnd.randrange(-127, 128) for _ in range(K)]
        want = _py_dot(xbar, x, M, K)
        xa = (ctypes.c_int8 * K)(*x)
        out = (ctypes.c_int64 * M)()
        lib.qsm_dot(out, (ctypes.c_uint8 * len(xbar)).from_buffer(xbar), xa, M, K)
        if list(out) != want:
            return False
    return True
