"""
ringkit.ml.tkvcache — TKV: content-addressed ring-native KV cache. RAM-tier semantic layer.

D11 — the form BEFORE the build. `ml.kvcache.RingKVCache.attend` (== `attend_full`, the D1 bar) scores
a query against EVERY cached key — exact, O(T). TKV takes over the retrieval ROLE: address the past
by CONTENT so a query can probe an O(1) bucket instead of scanning, with the recall law stated and
tested, never assumed (TKV_MATH.md Theorems H-O). This module is the interface + judge (P3 of the
integration plan) — RAM slabs only; the NVMe store tier is a later, separately-approved phase.

THE FORMS (proved in ../../gevhv/TKV_MATH.md, re-verified here in tests/test_tkvcache.py):

  Theorem H (temporal gauge)   p(u) = u mod 4 (branchless: u & 3); t(u) = 64*p(u) = (u&3)<<6. Exact,
                                but an information boundary: only mod-4 survives this projection.

  Theorem I / Cor I1, I2 (the load-bearing exact LSH law)
                                q(n) = floor(n/64) — the 2-bit quadrant. Under a uniformly random
                                per-coordinate rotation, Pr[quadrant collision] = max(0,(64-delta)/64)
                                EXACTLY (a rational identity, not an asymptotic). Sampling m coords
                                from the stride-7 orbit (gcd(7,dim)=1: no repeat before dim samples)
                                and concatenating their post-rotation quadrants gives the bucket
                                address A(h); Pr[A(h)=A(h')] = prod_i max(0,(64-delta_i)/64) — content
                                similarity governs recall as a QUANTIFIED, not hoped-for, approximation.

  Theorem J (addressing arithmetic — LBA/tiling; kept here as PURE ARITHMETIC, no I/O in P3)
                                LBA(h,u) = ((A(h)<<2)|p(u))<<12 — branchless (shifts/OR only),
                                sector-aligned (low 12 bits zero), phase-complete (4 phases are 4
                                contiguous sectors). Capacity = 4^m * 4 sectors = 2^(2m+14) bytes.

  Theorem M (the ADI anchor — repairing H's mod-4 boundary at 3.1% storage cost, single-sourced with
             core.native's ADI engine, Theorems K/L; NOT re-derived here)
                                lead = 7*u mod 256 ("rides the U-observer"); write-time recovery is
                                EXACT via the anti-stride 7^-1 = 183 (7*183 = 1281 = 1 mod 256):
                                u mod 256 = 183*lead. Relative time Delta-u = 183*(lead_q - lead_w).
                                Carried per entry alongside the mod-4 phase as routing metadata.

  Theorem N / Cor N1 (bucket-argmax recall + CONDITIONAL EXACTNESS — the D1 gate for this module)
                                Let j* = argmax_j score(q,k_j) (what full attend_full would pick).
                                Pr[j* in the probed bucket] is Cor I1 applied to (q, k*). Whenever
                                j* IS retrieved, hard TKV-attend equals hard attend_full BIT-FOR-BIT
                                (same argmax formula over a set that contains it). L independent
                                rotation tables raise recall to 1-(1-p)^L at L x storage/probes.

  Theorem O (address on CONTENT, not position — where RoPE lives)
                                Keys are stored PRE-RoPE (their raw content) with their position;
                                the bucket address is computed from that same pre-RoPE content, so
                                A(q) vs A(k) is AGE-INVARIANT (Thm O(ii)) — the prerequisite for
                                infinite-context retrieval. Position rides separately (the phase +
                                anchor above) and RoPE is applied to *retrieved* rows at SCORING
                                time via the single-sourced `ml.kvcache.rope` — bit-identical to
                                insert-time roping because the ring RoPE is additive and addition
                                commutes with storage. (Post-RoPE addressing is the deliberate
                                recency-biased alternative; not implemented here — Theorem O says
                                the choice is semantic, and this module takes the age-invariant one.)

Honest boundaries (stated, never hidden — TKV_MATH.md "Honest boundaries" + Cor N1):
  * This is associative-memory attention, not exact attention: O(1) retrieval, recall governed by
    the proved product law, not equivalence.
  * An empty probe (no candidate in ANY of the L buckets) falls back to a full scan over every
    entry — and that fallback is COUNTED, never silently absorbed (see `.raw`).
  * Bucket capacity / eviction policy is NOT this module's concern (RAM tier keeps every entry).

Scoring, weighting and blending are SINGLE-SOURCED with `ringkit.ml.kvcache` (score_row,
boltzmann_weights, circular_blend, rope) — never re-derived here, per D1 (this module's job is
addressing, not re-inventing the attention math it indexes).

Multiplier-free. No numpy, no math, no floats. Python here is the INTERFACE + JUDGE; the C++/store
tiers (kernels/, NVMe host) are later, separately-approved phases (P4+).
"""
from ringkit.core import native as rn
from ringkit.ml.kvcache import rope, score_row, boltzmann_weights, circular_blend


SECTOR = 4096                   # NVMe sector bytes (Theorem J)
ENTRY = 128                     # KV entry bytes (Theorem J.iii: 4096/128 = 32 entries/sector)
STRIDE = 7                      # the stride-7 orbit (Cor I2): gcd(7, dim) = 1 for typical dims
ANTI_STRIDE = 183               # 7^-1 mod 256 (7*183 = 1281 = 1 mod 256) — Theorem M(i)


# ── Theorem H: temporal gauge ────────────────────────────────────────────────
def phase_index(u):
    """p(u) = u mod 4, branchless (Theorem H). Two bits; the retrieval-blind mod-4 projection."""
    return int(u) & 3


def temporal_offset(u):
    """t(u) = 64*p(u) = (u&3)<<6 (Theorem H) — the additive temporal gauge offset."""
    return phase_index(u) << 6


# ── Theorem M: the ADI anchor (position metadata carried per entry) ─────────
def anchor_lead(u):
    """lead = 7*u mod 256 — the U-observer ride (Theorem M). rn.mul, no '*'."""
    return rn.mul(STRIDE, int(u)) & 0xFF


def anchor_recover(lead):
    """Write-time recovery: u mod 256 = 183*lead, exact (Theorem M(i): 7*183 = 1 mod 256)."""
    return rn.mul(ANTI_STRIDE, int(lead)) & 0xFF


def anchor_delta(lead_q, lead_w):
    """Relative time Delta-u = u_q - u_w (mod 256), recovered from anchors alone (Theorem M(ii))."""
    gap = (int(lead_q) - int(lead_w)) & 0xFF
    return rn.mul(ANTI_STRIDE, gap) & 0xFF


# ── Theorem I / Cor I1, I2: quadrant addressing ──────────────────────────────
def sample_coords(m, dim):
    """m coordinates from the stride-7 orbit over [0, dim): c_i = 7*(i+1) mod dim (Cor I2).
    gcd(7, dim) = 1 (dim not a multiple of 7) gives m <= dim distinct coordinates."""
    m = int(m)
    dim = int(dim)
    if dim <= 0:
        raise ValueError(f"sample_coords: dim must be > 0, got {dim}")
    if m <= 0:
        raise ValueError(f"sample_coords: m must be > 0, got {m}")
    if dim != 1 and rn.mf_mod(dim, STRIDE) == 0:
        raise ValueError(f"sample_coords: dim={dim} is a multiple of the stride {STRIDE} — "
                         "the orbit degenerates (gcd must be 1); choose a dim not divisible by 7")
    coords = []
    for i in range(m):
        coords.append(rn.mf_mod(rn.mul(STRIDE, i + 1), dim))
    return coords


def make_rotation(seed, m):
    """Deterministic length-m rotation vector, entries in [0,256) (Cor I1's 'one rotation vector,
    generated once'). A multiplier-free LCG (rn.mul only) seeded by `seed` — NO `import random`,
    so the address family is reproducible from the constructor argument alone."""
    state = int(seed) & 0xFFFFFFFF
    out = []
    for _ in range(int(m)):
        state = (rn.mul(1664525, state) + 1013904223) & 0xFFFFFFFF
        out.append((state >> 24) & 0xFF)
    return out


def quadrant(v):
    """q(n) = floor(n/64) — the top 2 bits of a ring value (Theorem I)."""
    return (int(v) & 0xFF) >> 6


def bucket_address(vec, r, coords):
    """A(h) = concat of the m post-rotation quadrants: A = (A<<2) | q(h[c_i] + r_i) (Cor I1).
    r has length len(coords); r_i rotates coordinate c_i only (independent per-coordinate rotation).
    Address on the RAW (pre-RoPE) content is what makes this age-invariant (Theorem O(ii))."""
    if len(r) != len(coords):
        raise ValueError(f"bucket_address: len(r)={len(r)} != len(coords)={len(coords)}")
    a = 0
    for i, c in enumerate(coords):
        v = (int(vec[c]) + int(r[i])) & 0xFF
        a = (a << 2) | quadrant(v)
    return a


# ── Theorem J: addressing arithmetic (pure, no I/O in this phase) ───────────
def lba(addr, u):
    """LBA(h,u) = ((A(h)<<2)|p(u))<<12 — branchless, sector-aligned, phase-complete (Theorem J).
    Bytes; kept as arithmetic only — the NVMe host that reads/writes at this address is P4."""
    return ((int(addr) << 2) | phase_index(u)) << 12


def capacity_bytes(m):
    """Address-space capacity: 4^m buckets * 4 phases * 4096 B/sector = 2^(2m+14) (Theorem J.v)."""
    m = int(m)
    if m < 0:
        raise ValueError(f"capacity_bytes: m must be >= 0, got {m}")
    return 1 << ((m << 1) + 14)


class TKVCache:
    """Content-addressed KV cache: the RingKVCache-shaped API (append/attend/len/raw), backed by
    L independent quadrant-LSH tables instead of a full scan (Theorem N). RAM tier: plain Python
    lists (the store tier that puts this on NVMe sectors per Theorem J is a separate, later phase).

        c = TKVCache(dim=64, m=8, L=2, seed=2026)
        c.append(k0, v0); c.append(k1, v1)
        out = c.attend(q, beta=16, hard=True)

    D1 bar (Theorem N): whenever the true full-scan argmax key is among the probed candidates,
    `.attend(hard=True)` returns EXACTLY what `ml.kvcache.RingKVCache.attend`/`attend_full` would —
    asserted bit-for-bit in tests/test_tkvcache.py (T20a), never merely approximated.
    """

    def __init__(self, dim, m=8, L=1, seed=2026, rope=True):
        self.dim = int(dim)
        self.m = int(m)
        self.L = int(L)
        self.rope = bool(rope)
        if self.dim <= 0:
            raise ValueError(f"TKVCache: dim must be > 0, got {dim}")
        if self.L <= 0:
            raise ValueError(f"TKVCache: L must be > 0, got {L}")
        self.coords = sample_coords(self.m, self.dim)
        # Cor N1: L INDEPENDENT rotation tables. seed+t is a deterministic per-table derivation
        # (no `import random`): each table gets its own reproducible rotation vector.
        self.tables = [make_rotation(seed + t, self.m) for t in range(self.L)]
        self.buckets = [{} for _ in range(self.L)]     # per-table: address -> [entry index, ...]
        self.K = []            # UN-roped content keys (Theorem O.ii) — position rides separately
        self.V = []
        self.leads = []        # Theorem M anchor per entry
        self.phases = []       # Theorem H phase per entry
        self.n = 0
        self._probes = 0
        self._hits = 0         # >= 1 candidate found in the union of L buckets
        self._fallbacks = 0    # empty union -> full scan (honesty: always counted)

    def __len__(self):
        return self.n

    def _addr(self, contentvec, table):
        return bucket_address(contentvec, self.tables[table], self.coords)

    def append(self, k, v):
        """Lay down one binding. The key is stored AS-IS (pre-RoPE content, Theorem O.ii) and
        addressed into every one of the L tables by that same content; RoPE is applied only at
        scoring time, from the position recorded here."""
        if len(k) != self.dim or len(v) != self.dim:
            raise ValueError(f"append: expected dim {self.dim}, got k={len(k)} v={len(v)}")
        pos = self.n
        kk = [int(x) & 0xFF for x in k]
        vv = [int(x) & 0xFF for x in v]
        for t in range(self.L):
            a = self._addr(kk, t)
            self.buckets[t].setdefault(a, []).append(pos)
        self.K.append(kk)
        self.V.append(vv)
        self.leads.append(anchor_lead(pos))
        self.phases.append(phase_index(pos))
        self.n += 1
        return self

    def probe(self, q):
        """Diagnostic/read-only: the UNION of the L bucket candidates for content q, WITHOUT
        scoring and WITHOUT touching the hit/fallback counters. Sorted entry indices."""
        qc = [int(x) & 0xFF for x in q]
        cand = set()
        for t in range(self.L):
            a = self._addr(qc, t)
            cand.update(self.buckets[t].get(a, ()))
        return sorted(cand)

    def attend(self, q, beta=16, hard=False, pos=None):
        """Probe the L tables on q's CONTENT (pre-RoPE, Theorem O.ii); score the union of retrieved
        candidates (roping both q and the candidate rows at SCORING time, single-sourced with
        ml.kvcache); empty union falls back to a full scan, counted (never hidden)."""
        if not self.n:
            raise ValueError("attend: cache is empty")
        if pos is None:
            pos = self.n - 1
        qc = [int(x) & 0xFF for x in q]
        idxs = self.probe(q)
        self._probes += 1
        if idxs:
            self._hits += 1
        else:
            idxs = list(range(self.n))
            self._fallbacks += 1
        qp = rope(qc, pos) if self.rope else qc
        Kp = [rope(self.K[j], j) if self.rope else self.K[j] for j in idxs]
        row = score_row(qp, Kp)
        w, best_local = boltzmann_weights(row, beta)
        best = idxs[best_local]
        if hard:
            return list(self.V[best])
        Vs = [self.V[j] for j in idxs]
        return circular_blend(Vs, w, best_local)

    @property
    def raw(self):
        return {"dim": self.dim, "m": self.m, "L": self.L, "rope": self.rope, "len": self.n,
                "coords": list(self.coords),
                "probes": self._probes, "hits": self._hits, "fallbacks": self._fallbacks,
                "storage": "python lists (RAM tier, P3 — no NVMe host in this phase)",
                "kernel": "python"}
