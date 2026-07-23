"""
ringkit.ml.gevhv — the GEVHV operator: bind -> react -> measure on Z256 manifolds.

THE FORMS (stated before code, D2/D11; proved in gevhv GEVHV_MATH.md, Theorems A-G,
machine-verified T1-T10 in the gevhv research bed; re-verified here in tests/test_gevhv.py):

  Manifold  g in R^(H x W), R = Z256, H = 128 rows, W = 113 columns (prime width — the
            QCM cache-manifold geometry). Row-major flat list, site (row, col) at
            offset row*W + col (offsets are walked by addition; no '*' in this layer).

  BIND      phi_{s,t}(x) = s*x + t (mod 256), s odd — a LOSSLESS change of gauge
            (Theorem E: the odd residues are exactly the units of Z256; inverse by the
            anti-stride s^{-1} = s^63, since the unit group has exponent 64).
            Vector form phi_v(g) = g + v (site-dependent translation, Theorem F2).

  REACT     rho(g)_ij = L[(g_ij + g_{i-1,j} + g_{i+1,j} + g_{i,j-1} + g_{i,j+1}) mod 256]
            at interior sites; IDENTITY at boundary. The 5-site staple + 256-byte LUT —
            the Hyperreactor. Deliberately dissipative (5 -> 1): the dynamics stage.

  BIND ABSORPTION (Theorem F — the fusion): the staple is Z256-linear, so
            rho(phi_{s,t}(g)) = rho'(g)  with the re-indexed table
            L'[u] = L[(s*u + 5t) mod 256]  — the scalar bind costs NOTHING per site.
            Boundary clause: rho is identity at the 2(H+W)-4 boundary sites, so the
            fused form applies phi POINTWISE there (Theorem F's boundary term).
            Vector bind absorbs as the OFFSET FIELD c = Sigma5(v) (Theorem F2),
            computed once per query and shared across a whole batch.

  MEASURE   E = Sigma_sites cdist(g_ij, q_ij) — ring-L1 energy readout against the
            query field q, ENERGY-accumulated in ZZ, never folded (Theorem G: interior
            E <= 13,986*128 < 2^31; full grid 14,464*128 < 2^31 — int32-exact).
            cdist is the single-source ml.attention.ring_distance (Theorem D form).

  RADIX DOT (Theorem C — the shift-add GEMV identity, arc-side exact dot):
            w . x = Sigma_{i=0}^{7} 2^i * b_i,  b_i = Sigma_{j: bit_i(x_j)=1} w_j
            — eight independent accumulator chains, additions and shifts ONLY
            (multiplier-free by construction, not by substitution).

Information ledger: bind exact+invertible (gauge), react exact+non-invertible
(dynamics), measure exact+non-invertible (observable). Exactness is never surrendered
to numerics — only to the physics (dissipation, projection).

Role (D7): returns ENERGY observables and serves the attention/retrieval layer — ml/,
beside kvcache and attention; it mints no ring positions, so it does not enter core.
5W (D10): GEVHV — the gevhv research programme's minted name for this operator; What:
transform-and-measure over ring manifolds; When: score/readout paths where gemm/gemv
served before; Where: ml semantic layer (silicon in kernels/cpu-win + kernels/cuda);
Why: the observable is the product the caller wants — one integer per manifold.

Multiplier-free. No numpy, no math, no floats. Python here is the INTERFACE + JUDGE;
the production math layer is C++ (kernels/cpu-win, kernels/cuda), gated bit-for-bit
against these forms at host load.
"""
from ringkit.core import native as rn
from ringkit.ml.attention import ring_distance

H = 128                      # manifold rows
W = 113                      # manifold columns (prime — QCM cache-manifold geometry)


def _dims(g, h, w):
    n = 0
    for _ in g:
        n += 1
    want = rn.mul(h, w)
    if n != want:
        raise ValueError(f"gevhv: manifold length {n} != H*W = {want} (H={h}, W={w})")


# ── BIND (Theorem E) ─────────────────────────────────────────────────────────
def anti_stride(s):
    """s^{-1} mod 256 for odd s: s^63 (the unit group (Z256)^x has exponent 64).
    Raises on even s — a zero-divisor has no inverse (it collapses)."""
    if (int(s) & 1) == 0:
        raise ValueError(f"anti_stride: s={s} is even (zero-divisor) — bind requires a unit")
    return rn.ring_pow(int(s), 63)


def bind_scalar(g, s, t):
    """phi_{s,t}: site-wise s*x + t (mod 256), s odd. Lossless gauge (Theorem E)."""
    if (int(s) & 1) == 0:
        raise ValueError(f"bind_scalar: s={s} is even — not a unit of Z256")
    return [(rn.mul(int(s), int(x)) + int(t)) & 0xFF for x in g]


def unbind_scalar(y, s, t):
    """phi^{-1}: s^{-1} * (y - t) (mod 256). Exact inverse of bind_scalar."""
    si = anti_stride(s)
    return [rn.mul(si, (int(x) - int(t)) & 0xFF) & 0xFF for x in y]


def bind_vector(g, v):
    """phi_v: site-wise g + v (mod 256) — translation by a hypervector field (Thm F2)."""
    out = []
    for a, b in zip(g, v):
        out.append((int(a) + int(b)) & 0xFF)
    return out


# ── REACT (the Hyperreactor: 5-site staple + LUT, identity boundary) ─────────
def _staple5(g, off, w):
    """The interior 5-site sum at flat offset `off` (row stride w). Callers guarantee
    the site is interior, so all four neighbours exist."""
    return (int(g[off]) + int(g[off - 1]) + int(g[off + 1])
            + int(g[off - w]) + int(g[off + w]))


def react(g, lut, h=H, w=W):
    """rho: L[Sigma5 mod 256] at interior sites, identity at boundary."""
    _dims(g, h, w)
    if len(lut) != 256:
        raise ValueError(f"react: LUT must have 256 entries, got {len(lut)}")
    out = [int(x) & 0xFF for x in g]
    row_off = w                              # start of row 1
    for _row in range(1, h - 1):
        for col in range(1, w - 1):
            off = row_off + col
            out[off] = int(lut[_staple5(g, off, w) & 0xFF]) & 0xFF
        row_off += w
    return out


# ── BIND ABSORPTION (Theorems F, F1, F2) ─────────────────────────────────────
def absorb_lut(lut, s, t):
    """L'[u] = L[(s*u + 5t) mod 256] — the one-time 256-step table rewrite that makes
    the scalar bind free per site (Theorem F). 5t = (t<<2) + t (shifts and adds)."""
    if (int(s) & 1) == 0:
        raise ValueError(f"absorb_lut: s={s} is even — the reindex must be a bijection")
    t5 = (int(t) << 2) + int(t)
    return [int(lut[(rn.mul(int(s), u) + t5) & 0xFF]) & 0xFF for u in range(256)]


def offset_field(v, h=H, w=W):
    """c = Sigma5(v) at interior sites (Theorem F2) — computed once per query v,
    independent of g, shared across the whole batch. Boundary entries are 0 and are
    never read (the fused vector form uses pointwise g+v at the boundary)."""
    _dims(v, h, w)
    c = [0 for _ in v]
    row_off = w
    for _row in range(1, h - 1):
        for col in range(1, w - 1):
            off = row_off + col
            c[off] = _staple5(v, off, w) & 0xFF
        row_off += w
    return c


def react_bound_scalar(g, lut, s, t, h=H, w=W):
    """FUSED scalar-affine GEVHV dynamics: rho(phi_{s,t}(g)) computed WITHOUT
    materializing the bound manifold — interior via the absorbed table L'
    (Theorem F), boundary via pointwise phi (the boundary clause)."""
    _dims(g, h, w)
    lut2 = absorb_lut(lut, s, t)
    out = bind_scalar(g, s, t)               # boundary value everywhere, then fix interior
    row_off = w
    for _row in range(1, h - 1):
        for col in range(1, w - 1):
            off = row_off + col
            out[off] = lut2[_staple5(g, off, w) & 0xFF]
        row_off += w
    return out


def react_bound_vector(g, lut, v, h=H, w=W, c=None):
    """FUSED vector-bind GEVHV dynamics: rho(g + v) via the offset field
    c = Sigma5(v) (Theorem F2) — one u8 add per interior site; pointwise g+v at the
    boundary. Pass a precomputed c to share it across a batch."""
    _dims(g, h, w)
    if c is None:
        c = offset_field(v, h, w)
    out = bind_vector(g, v)                   # boundary value everywhere, then fix interior
    row_off = w
    for _row in range(1, h - 1):
        for col in range(1, w - 1):
            off = row_off + col
            out[off] = int(lut[(_staple5(g, off, w) + c[off]) & 0xFF]) & 0xFF
        row_off += w
    return out


# ── MEASURE (Theorem G) ──────────────────────────────────────────────────────
def measure(g, q, h=H, w=W, interior=False):
    """E = Sigma_sites cdist(g_ij, q_ij) against the query FIELD q — ring-L1 energy,
    ENERGY-accumulated (never folded). interior=True restricts to the (H-2)(W-2)
    reacted sites (E <= 13,986*128 < 2^31); full grid likewise int32-exact (Thm G)."""
    _dims(g, h, w)
    _dims(q, h, w)
    e = 0
    if not interior:
        for a, b in zip(g, q):
            e += ring_distance(int(a) & 0xFF, int(b) & 0xFF)
        return e
    row_off = w
    for _row in range(1, h - 1):
        for col in range(1, w - 1):
            off = row_off + col
            e += ring_distance(int(g[off]) & 0xFF, int(q[off]) & 0xFF)
        row_off += w
    return e


# ── The operator, composed (one integer per manifold) ────────────────────────
def gevhv_scalar(g, lut, q, s, t, h=H, w=W, interior=False):
    """Full fused GEVHV with scalar-affine bind: measure(rho(phi_{s,t}(g)), q)."""
    return measure(react_bound_scalar(g, lut, s, t, h, w), q, h, w, interior)


def gevhv_vector(g, lut, q, v, h=H, w=W, c=None, interior=False):
    """Full fused GEVHV with vector bind: measure(rho(g + v), q). Pass c to share the
    offset field across a batch (Theorem F2's batch independence, Theorem G)."""
    return measure(react_bound_vector(g, lut, v, h, w, c), q, h, w, interior)


# ── RADIX DOT (Theorem C — the arc-side exact shift-add GEMV form) ───────────
def dot_radix(wrow, x):
    """w . x for u8 activations x by bit-plane transposition: 8 independent
    accumulator chains b_i = Sigma_{bit_i(x_j)=1} w_j, then Sigma 2^i b_i.
    Additions and shifts only — multiplier-free BY CONSTRUCTION (Theorem C).
    Exact over ZZ (Python integers); the u32 exactness bound k <= 66051 is the
    SILICON layer's concern (Theorem C-bound), tested as such."""
    if len(wrow) != len(x):
        raise ValueError(f"dot_radix: length mismatch {len(wrow)} vs {len(x)}")
    b = [0, 0, 0, 0, 0, 0, 0, 0]
    for j, xv in enumerate(x):
        xv = int(xv)
        if xv < 0 or xv > 255:
            raise ValueError(f"dot_radix: x[{j}]={xv} outside u8")
        wj = int(wrow[j])
        i = 0
        while xv:
            if xv & 1:
                b[i] += wj
            xv >>= 1
            i += 1
    acc = 0
    for i in range(8):
        acc += b[i] << i
    return acc


def gemv_radix(wrows, x):
    """Row-wise dot_radix — the Theorem C GEMV, one output per weight row (out = W . x).

    Routes through the GEVHV silicon kernel (kernels/rust gevhv_gemv_radix, multiplier-free
    shift-add, gated bit-for-bit at host load) and falls back to the pure-Python dot_radix loop
    below when no backend serves — the loop is the semantic reference. This is the ring-native
    linear-map (gemv) replacement: no hardware multiply anywhere on the served path."""
    if wrows and all(len(r) == len(x) for r in wrows) and all(0 <= int(v) <= 255 for v in x):
        m, k = len(wrows), len(x)
        from ringkit.kernels.cpu_rust import host as _rh
        got = _rh.gevhv_gemv_radix([int(v) for r in wrows for v in r], [int(v) for v in x], m, k)
        if got is not None:
            return got
    return [dot_radix(wr, x) for wr in wrows]
