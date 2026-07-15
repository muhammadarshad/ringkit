"""
ringkit.ml.adicodec — LOSSLESS ADI cube compressor (the separate, measured step kvadi defers).

kvadi ships the EXACT ADI element (integral/differential, FTRC-reversible). This module is the
compressor built ON it: a bijective byte codec, NOT a quantizer (CHARTER C9 — the arc/identity is
never crushed to chase a bit-count; every bit is recoverable). `decode_cube(encode_cube(x)) == x`
bit-for-bit for ANY ring cube, asserted in tests.

How it compresses without losing anything:
  1. GLOBAL constant-column elision: a scale-column that is one repeated value across every
     zone×band (e.g. the t=64 / t=128 Laplacian columns, identically 0) is stored as one flag
     bit + one value byte, not R×Z bytes.
  2. Cascaded ADI down the smooth axis: each column's band-sequence a_1..a_N -> (lead1, lead2, Δ²)
     via TWO exact differentials (kvadi). Adjacent bands are near-identical, so Δ² is tiny.
  3. Zigzag + bit-pack: Δ² (mod-256) folds to a small unsigned magnitude and packs at the minimal
     per-column bit width. Zero residuals cost zero bits.

Multiplier-free: shifts / masks / add / sub only — no *, //, /, ** (AST-audited).
"""
from ringkit.ml import kvadi as ka


# ── bit I/O (MSB-first), multiplier-free ────────────────────────────────────
class _BitWriter:
    def __init__(self):
        self.acc = 0
        self.nbits = 0
        self.out = bytearray()

    def write(self, val, w):
        if w <= 0:
            return
        self.acc = (self.acc << w) | (val & ((1 << w) - 1))
        self.nbits = self.nbits + w
        while self.nbits >= 8:
            self.nbits = self.nbits - 8
            self.out.append((self.acc >> self.nbits) & 0xFF)

    def finish(self):
        if self.nbits > 0:
            self.out.append((self.acc << (8 - self.nbits)) & 0xFF)
            self.nbits = 0
        return bytes(self.out)


class _BitReader:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.acc = 0
        self.nbits = 0

    def read(self, w):
        if w <= 0:
            return 0
        while self.nbits < w:
            self.acc = (self.acc << 8) | self.data[self.pos]
            self.pos = self.pos + 1
            self.nbits = self.nbits + 8
        self.nbits = self.nbits - w
        return (self.acc >> self.nbits) & ((1 << w) - 1)


# ── zigzag on Z256 (signed-8 interleave): bijection concentrating small |Δ| near 0 ──
def _zigzag(v):
    v = v & 0xFF
    if v < 128:                       # non-negative: 0,1,2,.. -> 0,2,4,..
        return v << 1
    return ((256 - v) << 1) - 1       # -1,-2,.. (v=255,254) -> 1,3,..


def _unzigzag(z):
    if (z & 1) == 0:
        return (z >> 1) & 0xFF
    return (256 - ((z + 1) >> 1)) & 0xFF


def _adi2(seq):
    """Column -> (lead1, lead2, Δ²), the two-level exact ADI (kvadi). Reversible."""
    lead1, d1 = ka.encode(seq)
    if d1:
        return lead1, d1[0], ka.differential(d1)
    return lead1, 0, []


def _iadi2(lead1, lead2, d2, n):
    """(lead1, lead2, Δ²) -> the length-n column, exactly."""
    if n <= 1:
        return [lead1 & 0xFF]
    d1 = ka.decode(lead2, d2)         # integral(Δ², lead2) -> length n-1
    return ka.decode(lead1, d1)       # integral(d1, lead1) -> length n


# ── cube codec ──────────────────────────────────────────────────────────────
def encode_cube(zones):
    """zones: list of Z equal-shape matrices, each R rows × C cols of ring bytes (0..255).
    Returns a lossless byte string. `decode_cube` inverts it bit-for-bit."""
    nz = len(zones)
    if nz == 0:
        raise ValueError("encode_cube: empty")
    R = len(zones[0])
    C = len(zones[0][0])
    bw = _BitWriter()
    bw.write(nz, 32)          # 32-bit dims: R/C routinely exceed 255 (e.g. 256 alive features,
    bw.write(R, 32)           # 1000+ reference vectors). 8-bit fields silently overflowed -> data
    bw.write(C, 32)           # loss; caught on real .qcm matrices (C=256), fixed 2026-07-15.

    const = []                        # per column: (is_const, value)
    for c in range(C):
        first = zones[0][0][c] & 0xFF
        is_const = all((zones[z][r][c] & 0xFF) == first for z in range(nz) for r in range(R))
        const.append((is_const, first))
        bw.write(1 if is_const else 0, 1)
        if is_const:
            bw.write(first, 8)

    for c in range(C):
        if const[c][0]:
            continue
        for z in range(nz):
            seq = [zones[z][r][c] for r in range(R)]
            lead1, lead2, d2 = _adi2(seq)
            bw.write(lead1 & 0xFF, 8)
            bw.write(lead2 & 0xFF, 8)
            zz = [_zigzag(v) for v in d2]
            w = max((v.bit_length() for v in zz), default=0)
            bw.write(w, 4)
            for v in zz:
                bw.write(v, w)
    return bw.finish()


def decode_cube(data):
    """Inverse of encode_cube — the exact cube (list of R×C matrices)."""
    br = _BitReader(data)
    nz = br.read(32)
    R = br.read(32)
    C = br.read(32)

    const = []
    for _ in range(C):
        is_const = br.read(1) == 1
        val = br.read(8) if is_const else 0
        const.append((is_const, val))

    cols = [[None for _ in range(C)] for _ in range(nz)]   # cols[z][c] = length-R band column
    for c in range(C):
        if const[c][0]:
            for z in range(nz):
                cols[z][c] = [const[c][1] for _ in range(R)]
            continue
        for z in range(nz):
            lead1 = br.read(8)
            lead2 = br.read(8)
            w = br.read(4)
            n_res = R - 2 if R >= 2 else 0
            d2 = [_unzigzag(br.read(w)) for _ in range(n_res)]
            cols[z][c] = _iadi2(lead1, lead2, d2, R)

    zones = []
    for z in range(nz):
        mat = [[cols[z][c][r] for c in range(C)] for r in range(R)]
        zones.append(mat)
    return zones


def ratio(zones):
    """Measured lossless ratio: raw fixed-8 bytes / coded bytes."""
    raw = 0
    for z in zones:
        for row in z:
            raw = raw + len(row)
    coded = len(encode_cube(zones))
    return raw, coded
