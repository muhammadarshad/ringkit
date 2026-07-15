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
NTHREADS = min(os.cpu_count() or 1, 16)
MT_MIN_WORK = 1 << 21                 # M*K below this: thread spawn costs more than it saves
# GEMV variant (kit precedent: ring_gemm's gated variants). "bridge" = hardware-* exact dot in one
# sweep (CPU dev speed); "qsm" = the multiplier-free QSM digit path (silicon/reference form);
# "metal" = the same exact dot on the unified GPU over the mmapped onix (no-copy), CPU fallback
# for tensors outside the mapped file / out-of-range activations. All BIT-IDENTICAL (gated).
GEMV_VARIANT = os.environ.get("RINGKIT_GEMV", "bridge")
_metal_maps = []                      # [(base_addr, size, slot)] of metal-mapped weight mmaps
_metal_host = None


def metal_register_onix(mm):
    """Map a weights mmap for the GPU GEMV (called by the weights host when RINGKIT_GEMV=metal).
    Several models may register (Gemma2 + Gemma4). Returns True on success."""
    global _metal_host
    from ringkit.kernels.apple.metal import host as mh
    if not mh.available():
        return False
    size = len(mm)
    base = ctypes.addressof((ctypes.c_char * size).from_buffer(mm))
    slot = mh.onix_map(base, size)
    if slot < 0:
        return False
    _metal_maps.append((base, size, slot))
    _metal_host = mh
    return True


def _metal_offset(xb_arr, nbytes):
    """(slot, byte offset) of a zero-copy xbar view inside a metal-mapped file, or None."""
    addr = ctypes.addressof(xb_arr)
    for base, size, slot in _metal_maps:
        off = addr - base
        if 0 <= off and off + nbytes <= size:
            return slot, off
    return None


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
        lib.qsm_gemv_exact.argtypes = [_I64, _U8, _I64, ctypes.c_long, ctypes.c_long,
                                       ctypes.POINTER(ctypes.c_int32), _I64, ctypes.c_int,
                                       _I8, _I64]
        lib.qsm_gemv_exact.restype = ctypes.c_long
        lib.qsm_gemv_exact_mt.argtypes = [_I64, _U8, _I64, ctypes.c_long, ctypes.c_long,
                                          ctypes.POINTER(ctypes.c_int32), _I64, ctypes.c_int,
                                          _I8, _I64, ctypes.c_long]
        lib.qsm_gemv_exact_mt.restype = ctypes.c_long
        lib.qsm_gemv_bridge_mt.argtypes = [_I64, _U8, _I64, ctypes.c_long, ctypes.c_long,
                                           ctypes.POINTER(ctypes.c_int32), _I64, ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_int32), ctypes.c_long]
        lib.qsm_gemv_bridge_mt.restype = ctypes.c_long
        lib.gelu_mul_block.argtypes = [_I64, _I64, _I64, ctypes.c_long, ctypes.c_int]
        lib.gelu_mul_block.restype = None
        lib.rmsnorm_block.argtypes = [_I64, _I64, _I64, ctypes.c_long, ctypes.c_int,
                                      ctypes.c_int64]
        lib.rmsnorm_block.restype = None
        lib.attn_block.argtypes = [_I64, _I64, _I64, _I64, ctypes.c_long, ctypes.c_long,
                                   ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_int,
                                   _I64, ctypes.c_long]
        lib.attn_block.restype = None
        lib.rope_block.argtypes = [_I64, _I64, _I64, ctypes.c_long, ctypes.c_long,
                                   ctypes.c_long, ctypes.c_long, ctypes.c_int]
        lib.rope_block.restype = None
        lib.add_into.argtypes = [_I64, _I64, _I64, ctypes.c_long]
        lib.add_into.restype = None
        lib.scale_q16.argtypes = [_I64, ctypes.c_int64, ctypes.c_long, ctypes.c_int]
        lib.scale_q16.restype = None
        lib.rmsnorm_rows.argtypes = [_I64, _I64, _I64, ctypes.c_long, ctypes.c_long,
                                     ctypes.c_int, ctypes.c_int64]
        lib.rmsnorm_rows.restype = None
        lib.embed_row_block.argtypes = [_I64, _U16, ctypes.c_long, ctypes.c_int64, ctypes.c_int]
        lib.embed_row_block.restype = None
        lib.rk_narrow32.argtypes = [_I64, ctypes.POINTER(ctypes.c_int32), ctypes.c_long]
        lib.rk_narrow32.restype = ctypes.c_long
        lib.lm_argmax.argtypes = [ctypes.POINTER(ctypes.c_int32), _U16,
                                  ctypes.c_long, ctypes.c_long, ctypes.c_int, _I64]
        lib.lm_argmax.restype = ctypes.c_long
        lib.lm_argmax_file.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_char_p, ctypes.c_long,
                                       ctypes.c_long, ctypes.c_long, ctypes.c_int, _I64]
        lib.lm_argmax_file.restype = ctypes.c_long
        if _selftest(lib) and _selftest_lm(lib) and _selftest_gemv(lib) and _selftest_act(lib) \
                and _selftest_attn(lib):
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


def _sd_py(n, d):
    return -((-n) // d) if n < 0 else n // d


def _py_gemv_exact(xbar, x, M, K, s_row, z_row, frac):
    """Semantic reference for the fused GEMV: the EXACT integer dot then 2^s shift (floor) and
    symmetric divide — the identity `emulation/gemma.py::proj` is proven bit-exact to. (`//` and
    `*` here are the labeled reference; the kernel and proj never use them.)"""
    out = []
    for r in range(M):
        base = r * K
        D = 0
        for k in range(K):
            D += (xbar[base + k] - 128) * x[k]
        s = s_row[r]
        t = D << s if s >= 0 else D >> (-s)
        out.append(_sd_py(t, z_row[r] or 1))
    return out


def gemv_exact(xbar, x, M, K, s_row, z_row, frac):
    """Fused exact digit-decomposition GEMV (the whole proj) in ONE C block call over the weight
    slab. `xbar` may be a writable memoryview into the onix mmap (ZERO-COPY — C reads the slab in
    place, the kit's C-owned-memory model) or bytes/bytearray. Returns Q<frac> int list, or None
    when the C kernel is unavailable (caller falls back to the Python semantic reference)."""
    lib = _load()
    if lib is None:
        return None
    try:
        if isinstance(xbar, memoryview) and not xbar.readonly:
            xb_arr = (ctypes.c_uint8 * (M * K)).from_buffer(xbar)      # zero-copy, in-place read
        elif isinstance(xbar, bytearray):
            xb_arr = (ctypes.c_uint8 * (M * K)).from_buffer(xbar)
        else:
            xb_arr = (ctypes.c_uint8 * (M * K)).from_buffer_copy(bytes(xbar))
    except (TypeError, ValueError):
        xb_arr = (ctypes.c_uint8 * (M * K)).from_buffer_copy(bytes(xbar))
    xa = (ctypes.c_int64 * K)(*[int(v) for v in x])
    sa, za = _sz_arrays(s_row, z_row, M)          # pinned per-tensor cache (s/z never change)
    xs_scratch, r_scratch, x32_s, out = _scratch(M, K)   # reused per-shape scratch
    nt = NTHREADS if M * K >= MT_MIN_WORK else 1  # disjoint row blocks, merge-free (bit-identical)
    if GEMV_VARIANT == "qsm":                     # silicon/reference form (multiplier-free)
        np_ = lib.qsm_gemv_exact_mt(out, xb_arr, xa, M, K, sa, za, frac, xs_scratch, r_scratch, nt)
    else:                                         # hardware-* bridge: same exact dot, one sweep
        np_ = lib.qsm_gemv_bridge_mt(out, xb_arr, xa, M, K, sa, za, frac, x32_s, nt)
    if np_ < 0:
        return None
    return list(out)


_sz_arr_cache = {}    # id(s_row) -> (pin, sa, za): pinned ctypes arrays for cached s/z lists
_scratch_cache = {}   # K -> (xs_scratch, r_scratch); M -> out array


def _scratch(M, K):
    s = _scratch_cache.get(("k", K))
    if s is None:
        s = ((ctypes.c_int8 * (16 * K))(), (ctypes.c_int64 * K)(), (ctypes.c_int32 * K)())
        _scratch_cache[("k", K)] = s
    o = _scratch_cache.get(("m", M))
    if o is None:
        o = (ctypes.c_int64 * M)()
        _scratch_cache[("m", M)] = o
    return s[0], s[1], s[2], o


def _sz_arrays(s_row, z_row, M):
    key = (id(s_row), M)
    got = _sz_arr_cache.get(key)
    if got is not None and got[0] is s_row:
        return got[1], got[2]
    sa = (ctypes.c_int32 * M)(*[int(v) for v in s_row])
    za = (ctypes.c_int64 * M)(*[int(v) for v in z_row])
    _sz_arr_cache[key] = (s_row, sa, za)          # pin s_row so the id stays valid
    return sa, za


def gelu_mul(g, u, frac):
    """out[i] = (gelu_pytorch_tanh(g[i]) * u[i]) >> frac in ONE C block call, bit-for-bit vs the
    Python reference (gemma4.gelu_tanh_fixed + rn.mul). None when the kernel is unavailable."""
    lib = _load()
    if lib is None:
        return None
    n = len(g)
    ga = (ctypes.c_int64 * n)(*g)
    ua = (ctypes.c_int64 * n)(*u)
    out = (ctypes.c_int64 * n)()
    lib.gelu_mul_block(out, ga, ua, n, frac)
    return list(out)


def rmsnorm(x, weight, frac, eps=1):
    """RMSNorm block (== ract.rmsnorm_fixed bit-for-bit). None when the kernel is unavailable
    or an activation exceeds the kernel's exact __int128 Σx² range (caller falls back to the
    Python bigint reference — same result, just slower)."""
    lib = _load()
    if lib is None:
        return None
    n = len(x)
    for v in x:
        if (v >> 58) if v >= 0 else ((-v) >> 58):
            return None
    xa = (ctypes.c_int64 * n)(*x)
    wa = (ctypes.c_int64 * n)(*weight)
    out = (ctypes.c_int64 * n)()
    lib.rmsnorm_block(out, xa, wa, n, frac, eps)
    return list(out)


# ── Resident-activation layer: the hidden vector lives in C buffers between blocks ──────────
_act_bufs = {}


def actbuf(name, n):
    """Persistent named C activation buffer (allocated once per (name, size))."""
    key = (name, n)
    b = _act_bufs.get(key)
    if b is None:
        b = (ctypes.c_int64 * n)()
        _act_bufs[key] = b
    return b


_pin_cache = {}


def pinned_i64(vals):
    """ctypes int64 array for a CACHED Python list (norm gammas etc.) — pinned by identity so
    the conversion happens once per tensor, not once per call."""
    key = id(vals)
    got = _pin_cache.get(key)
    if got is not None and got[0] is vals:
        return got[1]
    arr = (ctypes.c_int64 * len(vals))(*vals)
    _pin_cache[key] = (vals, arr)
    return arr


_ones_cache = {}


def ones_buf(n, frac):
    key = (n, frac)
    b = _ones_cache.get(key)
    if b is None:
        b = (ctypes.c_int64 * n)(*([1 << frac] * n))
        _ones_cache[key] = b
    return b


def _xbar_ptr(xbar, nbytes):
    try:
        if isinstance(xbar, memoryview) and not xbar.readonly:
            return (ctypes.c_uint8 * nbytes).from_buffer(xbar)      # zero-copy, in-place read
        if isinstance(xbar, bytearray):
            return (ctypes.c_uint8 * nbytes).from_buffer(xbar)
    except (TypeError, ValueError):
        pass
    return (ctypes.c_uint8 * nbytes).from_buffer_copy(bytes(xbar))


def gemv_into(out_buf, tensor, x_buf, frac):
    """proj into a resident buffer: x and out are ctypes int64 buffers — nothing crosses Python."""
    xbar, s_row, z_row, of, inf = tensor
    lib = _lib
    xb = _xbar_ptr(xbar, of * inf)
    sa, za = _sz_arrays(s_row, z_row, of)
    nt = NTHREADS if of * inf >= MT_MIN_WORK else 1
    xs_s, r_s, x32_s, _ = _scratch(of, inf)
    if GEMV_VARIANT == "metal" and _metal_host is not None:
        loc = _metal_offset(xb, of * inf)
        if loc is not None and lib.rk_narrow32(x_buf, x32_s, inf):
            if _metal_host.emu_gemv(loc[0], loc[1], of, inf, x32_s, sa, za, frac, out_buf) == 0:
                return
    if GEMV_VARIANT == "qsm":
        lib.qsm_gemv_exact_mt(out_buf, xb, x_buf, of, inf, sa, za, frac, xs_s, r_s, nt)
    else:
        lib.qsm_gemv_bridge_mt(out_buf, xb, x_buf, of, inf, sa, za, frac, x32_s, nt)


def gemv_multi(pairs, x_buf, frac):
    """Several projections of the SAME activation vector (q/k/v, gate/up). On the metal variant
    they run as ONE batched command buffer (one x upload, per-tensor s/z cached GPU-side);
    otherwise sequential gemv_into calls. pairs = [(out_buf, tensor), ...]."""
    if GEMV_VARIANT == "metal" and _metal_host is not None and len(pairs) > 1:
        K = pairs[0][1][4]
        _, _, x32_s, _ = _scratch(pairs[0][1][3], K)
        if _lib.rk_narrow32(x_buf, x32_s, K):
            slot = None
            offs, Ms, s_arrs, z_arrs, outs = [], [], [], [], []
            for out_buf, tensor in pairs:
                xbar, s_row, z_row, of, inf = tensor
                xb = _xbar_ptr(xbar, of * inf)
                loc = _metal_offset(xb, of * inf)
                if loc is None or (slot is not None and loc[0] != slot):
                    break
                slot = loc[0]
                sa, za = _sz_arrays(s_row, z_row, of)
                offs.append(loc[1]); Ms.append(of); s_arrs.append(sa); z_arrs.append(za)
                outs.append(out_buf)
            else:
                if _metal_host.emu_gemv_batch(slot, offs, Ms, K, x32_s, s_arrs, z_arrs,
                                              frac, outs) == 0:
                    return
    for out_buf, tensor in pairs:
        gemv_into(out_buf, tensor, x_buf, frac)


def rope_buf(buf, cos_row, sin_row, pair_off, nh, hd, frac):
    ca = (ctypes.c_int64 * len(cos_row))(*cos_row)
    sa = (ctypes.c_int64 * len(sin_row))(*sin_row)
    _lib.rope_block(buf, ca, sa, len(cos_row), pair_off, nh, hd, frac)


def attn_into(ctx_buf, q_buf, slab, nq, frac):
    sw = actbuf("sw", nq * slab.cap)
    nt = NTHREADS if slab.n >= 8 else 1
    _lib.attn_block(ctx_buf, q_buf, slab.k, slab.v, nq, slab.n_kv, slab.hd,
                    slab.n, slab.cap, frac, sw, nt)


def embed_into(h_buf, row_bytes, n, esc, frac):
    row = (ctypes.c_uint16 * n).from_buffer_copy(row_bytes)
    _lib.embed_row_block(h_buf, row, n, esc, frac)


class KVSlab:
    """C-owned K/V slabs for one layer: [n_kv, cap, hd] int64 each, grown by doubling. Python
    only appends one row per kv-head per token; attention reads the slabs in place (the kit's
    C-owned-memory model — kernels/mprc/kv pattern)."""

    def __init__(self, n_kv, hd, cap=1024):
        self.n_kv, self.hd, self.cap, self.n = n_kv, hd, cap, 0
        self.k = (ctypes.c_int64 * (n_kv * cap * hd))()
        self.v = (ctypes.c_int64 * (n_kv * cap * hd))()

    def _grow(self):
        new_cap = self.cap << 1
        for name in ("k", "v"):
            old = getattr(self, name)
            new = (ctypes.c_int64 * (self.n_kv * new_cap * self.hd))()
            for j in range(self.n_kv):
                src = j * self.cap * self.hd
                dst = j * new_cap * self.hd
                ctypes.memmove(ctypes.byref(new, dst * 8), ctypes.byref(old, src * 8),
                               self.n * self.hd * 8)
            setattr(self, name, new)
        self.cap = new_cap

    def insert(self, kh_rows, vh_rows):
        """Append one roped-K row and one V row per kv-head (lists of len hd)."""
        if self.n == self.cap:
            self._grow()
        for j in range(self.n_kv):
            off = (j * self.cap + self.n) * self.hd
            self.k[off:off + self.hd] = kh_rows[j]
            self.v[off:off + self.hd] = vh_rows[j]
        self.n += 1

    def insert_bufs(self, k_buf, v_buf):
        """Append from resident C buffers ([n_kv*hd] each) — pure memmove, no Python lists."""
        if self.n == self.cap:
            self._grow()
        for j in range(self.n_kv):
            dst = (j * self.cap + self.n) * self.hd * 8
            src = j * self.hd * 8
            ctypes.memmove(ctypes.byref(self.k, dst), ctypes.byref(k_buf, src), self.hd * 8)
            ctypes.memmove(ctypes.byref(self.v, dst), ctypes.byref(v_buf, src), self.hd * 8)
        self.n += 1


def attention(q_flat, slab, nq, frac):
    """All nq query heads over the layer's KV slabs in ONE C block call (thread-split over heads,
    disjoint ctx rows). Returns ctx as a flat list [nq*hd], or None if the kernel is unavailable."""
    lib = _load()
    if lib is None:
        return None
    hd, nkeys, cap, nkv = slab.hd, slab.n, slab.cap, slab.n_kv
    qa = (ctypes.c_int64 * (nq * hd))(*q_flat)
    ctx = (ctypes.c_int64 * (nq * hd))()
    sw = (ctypes.c_int64 * (nq * nkeys))()
    nt = NTHREADS if nkeys >= 8 else 1
    lib.attn_block(ctx, qa, slab.k, slab.v, nq, nkv, hd, nkeys, cap, frac, sw, nt)
    return list(ctx)


def rope(vec_flat, cos_row, sin_row, pair_off, nh, hd, frac):
    """NeoX RoPE over nh heads in ONE C block call. Returns the rotated flat list, or None."""
    lib = _load()
    if lib is None:
        return None
    va = (ctypes.c_int64 * (nh * hd))(*vec_flat)
    ca = (ctypes.c_int64 * len(cos_row))(*cos_row)
    sa = (ctypes.c_int64 * len(sin_row))(*sin_row)
    lib.rope_block(va, ca, sa, len(cos_row), pair_off, nh, hd, frac)
    return list(va)


def _selftest_attn(lib):
    """Gate attn_block and rope_block bit-for-bit vs the Python semantic references
    (gemma4.attention_g4 / apply_rope), including the softmax exp-saturation regime."""
    import random
    from ringkit.emulation import gemma4 as g4
    rnd = random.Random(31)
    frac = 16
    ONE = 1 << frac
    nq, nkv, hd, nkeys, cap = 4, 2, 16, 5, 8
    group = nq // nkv
    # K/V slabs with one saturating-score setup: one huge key drives score gaps past the clamp
    kh = [[[rnd.randrange(-2 * ONE, 2 * ONE) for _ in range(hd)] for _ in range(nkeys)]
          for _ in range(nkv)]
    vh = [[[rnd.randrange(-2 * ONE, 2 * ONE) for _ in range(hd)] for _ in range(nkeys)]
          for _ in range(nkv)]
    for d in range(hd):
        kh[0][0][d] = 10 * ONE                        # score outlier -> exp clamp regime (e^512+)
    q = [[rnd.randrange(-2 * ONE, 2 * ONE) for _ in range(hd)] for _ in range(nq)]
    want = []
    for i in range(nq):
        want.extend(g4.attention_g4(q[i], kh[i // group], vh[i // group], frac))
    ks = (ctypes.c_int64 * (nkv * cap * hd))()
    vs = (ctypes.c_int64 * (nkv * cap * hd))()
    for j in range(nkv):
        for t in range(nkeys):
            off = (j * cap + t) * hd
            ks[off:off + hd] = kh[j][t]
            vs[off:off + hd] = vh[j][t]
    qa = (ctypes.c_int64 * (nq * hd))(*[x for head in q for x in head])
    ctx = (ctypes.c_int64 * (nq * hd))()
    sw = (ctypes.c_int64 * (nq * nkeys))()
    for nt in (1, 3):                                 # single-thread and split must both match
        lib.attn_block(ctx, qa, ks, vs, nq, nkv, hd, nkeys, cap, frac, sw, nt)
        if list(ctx) != want:
            return False
    # rope_block vs apply_rope (partial rotation: n_rot < hd/2 span exercises passthrough)
    cos_row = [rnd.randrange(-ONE, ONE) for _ in range(4)]
    sin_row = [rnd.randrange(-ONE, ONE) for _ in range(4)]
    heads = [[rnd.randrange(-2 * ONE, 2 * ONE) for _ in range(hd)] for _ in range(3)]
    want_r = []
    for head in heads:
        want_r.extend(g4.apply_rope(head, cos_row, sin_row, hd >> 1, frac))
    va = (ctypes.c_int64 * (3 * hd))(*[x for head in heads for x in head])
    ca = (ctypes.c_int64 * 4)(*cos_row)
    sa = (ctypes.c_int64 * 4)(*sin_row)
    lib.rope_block(va, ca, sa, 4, hd >> 1, 3, hd, frac)
    return list(va) == want_r


def _selftest_act(lib):
    """Gate gelu_mul_block and rmsnorm_block bit-for-bit vs the Python semantic references
    (emulation.ract / emulation.gemma4), across normal, outlier and saturating-tanh regimes."""
    import random
    from ringkit.emulation import ract
    rnd = random.Random(23)
    frac = 16
    ONE = 1 << frac
    # gelu_mul: include the tanh-saturation / exp-clamp regime and tiny values. NOTE: keep the
    # saturating cases MODERATE (±8..12·ONE fully exercises the clamp path — e^64 >> 2^32): the
    # PYTHON reference on much larger args squares astronomically wide bigints (e^16384+) and
    # takes minutes-to-hours, while the kernel's divisor-saturation is exact for ANY size.
    g = [rnd.randrange(-3 * ONE, 3 * ONE) for _ in range(64)]
    g += [8 * ONE, -8 * ONE, 12 * ONE, -12 * ONE, 0, 1, -1, ONE >> 3]
    u = [rnd.randrange(-3 * ONE, 3 * ONE) for _ in range(len(g))]
    from ringkit.emulation.gemma4 import gelu_tanh_fixed
    from ringkit.core import native as rn
    want = [rn.mul(gelu_tanh_fixed(gi, frac), ui) >> frac for gi, ui in zip(g, u)]
    ga = (ctypes.c_int64 * len(g))(*g)
    ua = (ctypes.c_int64 * len(g))(*u)
    out = (ctypes.c_int64 * len(g))()
    lib.gelu_mul_block(out, ga, ua, len(g), frac)
    if list(out) != want:
        return False
    # rmsnorm: normal + outlier vector, weighted and no-scale — plus the huge-activation regime
    # (|x| ~ 2^45..2^55, Soliton y_prenorm) whose Σx² overflowed the old int64 accumulator and
    # whose wrapped-negative mean-square hung the old isqrt_c loop
    for xs in ([rnd.randrange(-2 * ONE, 2 * ONE) for _ in range(33)],
               [rnd.randrange(-ONE >> 2, ONE >> 2) for _ in range(40)] + [60 * ONE],
               [rnd.randrange(-(1 << 55), 1 << 55) for _ in range(128)],
               [rnd.randrange(-(1 << 46), 1 << 46) for _ in range(128)],
               [0] * 17):
        for w in ([rnd.randrange(-ONE, 2 * ONE) for _ in range(len(xs))], [ONE] * len(xs)):
            want = ract.rmsnorm_fixed(xs, w, frac)
            xa = (ctypes.c_int64 * len(xs))(*xs)
            wa = (ctypes.c_int64 * len(xs))(*w)
            o = (ctypes.c_int64 * len(xs))()
            lib.rmsnorm_block(o, xa, wa, len(xs), frac, 1)
            if list(o) != want:
                return False
    return True


def _selftest_gemv(lib):
    """Gate the fused GEMV bit-for-bit vs the exact-dot reference — including the 60x-outlier and
    tiny-activation (negative scale exponent) regimes that broke truncating designs."""
    import random
    rnd = random.Random(11)
    frac = 16
    ONE = 1 << frac
    cases = []
    for M, K in ((3, 8), (5, 33), (4, 256)):
        xbar = bytes(rnd.randrange(256) for _ in range(M * K))
        s_row = [rnd.randrange(-6, 4) for _ in range(M)]
        z_row = [rnd.randrange(1, 7) for _ in range(M)]
        x = [rnd.randrange(-3 * ONE, 3 * ONE) for _ in range(K)]
        cases.append((xbar, x, M, K, s_row, z_row))
        xo = [rnd.randrange(-ONE >> 2, ONE >> 2) for _ in range(K)]
        xo[K >> 1] = 40 * ONE                     # outlier regime
        cases.append((xbar, xo, M, K, s_row, z_row))
        xt = [0] * K
        xt[0] = 1; xt[-1] = -1                    # tiny regime (negative scale exponent)
        cases.append((xbar, xt, M, K, s_row, z_row))
        cases.append((xbar, [0] * K, M, K, s_row, z_row))
    for xbar, x, M, K, s_row, z_row in cases:
        xa = (ctypes.c_int64 * K)(*x)
        sa = (ctypes.c_int32 * M)(*s_row)
        za = (ctypes.c_int64 * M)(*z_row)
        out = (ctypes.c_int64 * M)()
        xs_s = (ctypes.c_int8 * (16 * K))()
        r_s = (ctypes.c_int64 * K)()
        xb = (ctypes.c_uint8 * (M * K)).from_buffer_copy(xbar)
        if lib.qsm_gemv_exact(out, xb, xa, M, K, sa, za, frac, xs_s, r_s) < 0:
            return False
        want = _py_gemv_exact(xbar, x, M, K, s_row, z_row, frac)
        if list(out) != want:
            return False
        out_mt = (ctypes.c_int64 * M)()           # MPP row-block split must be bit-identical
        if lib.qsm_gemv_exact_mt(out_mt, xb, xa, M, K, sa, za, frac, xs_s, r_s, 3) < 0:
            return False
        if list(out_mt) != want:
            return False
        out_br = (ctypes.c_int64 * M)()           # hardware-* bridge must be bit-identical too
        x32s = (ctypes.c_int32 * K)()             # (both the int32 fast path and, via None,
        if lib.qsm_gemv_bridge_mt(out_br, xb, xa, M, K, sa, za, frac, x32s, 3) < 0:
            return False
        if list(out_br) != want:
            return False
        out_bs = (ctypes.c_int64 * M)()           # the scalar __int128 path)
        if lib.qsm_gemv_bridge_mt(out_bs, xb, xa, M, K, sa, za, frac, None, 3) < 0:
            return False
        if list(out_bs) != want:
            return False
    return True


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
