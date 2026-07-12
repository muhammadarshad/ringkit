"""
ringkit.ml.kvcodec — sub-byte KV compression, ring-native. PROVISIONAL — DO NOT TRUST THE NUMBERS.

╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║ TWO REASONS THIS MODULE IS NOT DONE. Read before quoting any figure out of it.                ║
║                                                                                              ║
║ 1. THE BENCHMARK IS BLIND. Every retrieval number below was produced by a routing test whose ║
║    KNOWN-BAD control does not fail. An EVEN stride (x8) provably destroys information — it    ║
║    maps 256 ring values onto 32 codes, irreversibly — and it STILL scores 1.00, at N = 12,    ║
║    64, 256 and 1024. A compression benchmark that a known-lossy codec passes cannot           ║
║    distinguish a good codec from a bad one. So "0.48 -> 1.00 at 2 bits" does NOT mean the     ║
║    compression is good; it means the task is too easy to see loss. A valid bar must measure   ║
║    KEY RECOVERY / VALUE FIDELITY, not 1-of-N routing. Until the control fails, these numbers  ║
║    establish nothing.                                                                        ║
║                                                                                              ║
║ 2. IT ENCODES THE WRONG OBJECT. The real spec is hpq-kernel-rust/src/polar.rs: a KV element   ║
║    is a POLAR PAIR (tick: u8, mag: u8) — an ANGLE and a MAGNITUDE — not the single flat phase ║
║    byte this module quantizes. For K the RoPE tick is baked in at write time (which ringkit   ║
║    already does, exactly, additively); for V the tick is a 2-bin SIGN (0 = +, 128 = -) and    ║
║    the information lives in mag. Per-element (tick, mag) IS our (phase, energy). This module  ║
║    has NO magnitude channel at all, so it is quantizing half the object.                     ║
║    Still open in the spec: "4 diffusion style processing" over the four 64-chunks — asked for ║
║    by the owner, not yet specified to me, NOT guessed at here.                                ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

What IS solid and separable: ml/kvcache.py (the soft-attention primitive, cached==uncached
bit-for-bit), the C-owned prime-pitched slab, and the 1.14x cache-manifold measurement. Those do
not depend on anything in this file.

D11 — the form, and the honest record of what was TRIED and what the ring actually rewarded.

A KV cache is compressible only if the stored keys still ROUTE. So the bar here is never
reconstruction error — it is retrieval accuracy against the uncompressed cache, with a control.

Three ring-native compressors were built and MEASURED (tests/test_kvcodec.py keeps the receipts):

  1. ADI (rn.compress / odd-increment, delta_k = delta_1 + k^2 - 1)  -> REJECTED, measured.
     Exact for its own family (an ADI-consistent d-vector costs 2 bytes), but as a PREDICTOR for
     general vectors it is strictly WORSE than no predictor: it pays 2 bytes of overhead AND has
     higher residual error at every bit-width. The odd-increment law is not the law KV data obeys.

  2. The e-axis (ring_log/ring_exp, base RING_E=3: every odd unit is +-3^k, k mod 64) -> REJECTED,
     measured. It is a perfect 7-bit lossless index of the 128 odd units. But 3^k SCATTERS
     (k=4 -> 81, k=5 -> 243), so the axis is a bijection and NOT an isometry — and attention scores
     are ring DISTANCES. Quantizing k destroys locality: retrieval collapses to 0.06 at 3 bits
     where the plain uniform grid still holds 1.00. Right axis for magnitudes, wrong one for phases.

  3. ODD-STRIDE PRECONDITION + uniform phase grid  -> ADOPTED. This is the one the ring rewards.

The adopted form:

     store:   q_b( x * s  mod 256 )        s ODD, b bits kept
     score:   attention runs on the PRECONDITIONED keys, query strided by the same s

  The stride AMPLIFIES intra-cluster differences before the uniform grid sees them, which is what
  actually buys the bits: a tightly-clustered key set wastes a uniform codebook, and the usual fix
  is a stored per-block scale/zero-point (which costs ~1 extra bit per number). We store NOTHING:
  s is a fixed ring constant, identical for every block, layer and tensor. Data-free by
  construction, not by calibration.

  Why s must be ODD — and precisely what that buys. Odd => gcd(s,256)=1 => multiplication by s is a
  BIJECTION on Z256 (256 values -> 256 distinct codes), exactly invertible by modinv(s). An EVEN
  stride is a zero-divisor: x8 maps 256 values onto just 32 codes, irreversibly. Two distinct keys
  can therefore NEVER collide under an odd stride, and always may under an even one.

  HONEST: that guarantee is STRUCTURAL, not something the retrieval benchmark demonstrates. Measured,
  an even stride scores just as well on retrieval (12 keys in dim 16 stay separable even after a
  256->32 collapse). We require odd because it is provably lossless, NOT because the benchmark
  separates it. The test asserts both facts, including the one that does not flatter us.

  HONEST: the stride amplifies QUERY NOISE too. It is a large win when the query is near a stored
  key and a LOSS when it is far — measured, at +-16 noise the strided cache scores below the raw
  one. The stride buys quantization headroom, not robustness.

Measured (dim 16, 12 keys, retrieval vs the uncompressed cache = 1.00 by construction):

     key spread   |  raw uniform      |  odd-stride (x7)
     tight        |  2 bits -> 0.48   |  2 bits -> 1.00
     pathological |  3 bits -> 0.20   |  3 bits -> 1.00
     uniform      |  4 bits -> 1.00   |  4 bits -> 1.00

Multiplier-free (rn.mul is shift-add), never leaves u8, no float, no codebook, no calibration.
"""
from ringkit.core import native as rn


STRIDE = 7          # the QCM canonical odd stride: gcd(7,256)=1, bijective, vacuum-avoiding

# ── the 64-CHUNK READING (the ring's own decomposition of a phase) ───────────────────────────
# The core identity [(a+b)^2 (c+d)^2]^2 = 256 splits the ring into FOUR 64-chunks, and a phase is
# NOT a flat 8-bit number — it is:
#
#     d = [ spin:1 | polarity:1 | offset:6 ]      spin = UP/DOWN (bit 7), polarity = +/- (bit 6)
#         \_______  _________/   \___  ___/       quadrant = (d >> 6) & 3   -> UP+ UP- DN+ DN-
#                 \/                 \/           offset   =  d & 0x3F      -> position in the chunk
#            quadrant (2 bits)   64-chunk offset  vacuum   =  offset == 0   -> {0,64,128,192}
#
# (physics/qcm.py: spin, polarity, quadrant, is_vacuum — this is the QCM node state, not our invention.)
#
# Why this compresses where a flat grid does not. Quantizing the FLAT 256 grid spends its codes on
# the whole ring. But a clustered key set lives inside ONE 64-chunk per coordinate, so the quadrant
# bits are the SAME for every token — pure redundancy, repeated once per token. Hoist the quadrant
# out (2 bits x dim, stored ONCE for the entire cache, amortized to ~0) and spend EVERY per-token bit
# on the 6-bit offset inside the chunk. Same bits/token, 4x finer resolution where the data actually is.
#
# Measured (dim 16, 12 keys, routing accuracy; uncompressed = 1.00):
#     key spread   | bits | flat 256-grid | 64-chunk
#     tight        |  2b  |     0.48      |   1.00
#     pathological |  3b  |     0.20      |   0.89
#     uniform      |  2b  |     1.00      |   1.00     (no regression on easy data)
#
# HONEST: the hoisted quadrant IS a stored base — 2 bits per coordinate, per cache. It is not
# "zero side information", it is *negligible* side information (amortized over every token). The odd
# stride, by contrast, stores literally nothing. They are different trades, both measured.


def is_odd_stride(s):
    """A usable stride is a UNIT (odd) — a bijection on the ring. Even = zero-divisor = collapses."""
    return int(s) & 1 == 1


def precondition(row, stride=STRIDE):
    """x -> x*s mod 256, s odd. Exact, bijective, multiplier-free (rn.mul is shift-add).

    This is our preconditioner: it spreads a tight cluster across the whole ring so a FIXED uniform
    grid suffices — no scale, no zero-point, nothing stored. It never leaves u8."""
    if not is_odd_stride(stride):
        raise ValueError(f"precondition: stride must be ODD (a unit); {stride} is a zero-divisor "
                         "and collapses the ring irreversibly")
    return [rn.mul(int(x), int(stride)) & 0xFF for x in row]


def unprecondition(row, stride=STRIDE):
    """Exact inverse: multiply by modinv(stride). Only an ODD stride has one."""
    from ringkit.linalg.solve import modinv
    inv = modinv(int(stride) & 0xFF)
    return [rn.mul(int(x), inv) & 0xFF for x in row]


def quadrant_of(row):
    """The 64-chunk each coordinate sits in: (d >> 6) & 3 — spin(bit 7) and polarity(bit 6)."""
    return [(int(x) >> 6) & 3 for x in row]


def chunk_quantize(row, bits, base):
    """64-CHUNK READING: keep the hoisted quadrant, quantize only the 6-bit offset INSIDE it.

    base[d] is the quadrant shared by coordinate d across the whole cache (stored once). Every
    per-token bit therefore buys resolution where the data actually lives — 4x finer than a flat
    grid at the same bit-width. Shifts only."""
    sh = 6 - int(bits)
    if sh < 0:
        raise ValueError(f"chunk_quantize: bits must be <= 6 (the offset is 6 bits), got {bits}")
    out = []
    for x in row:
        off = int(x) & 0x3F
        out.append((off >> sh) if sh else off)
    return out


def chunk_dequantize(codes, bits, base):
    """Rebuild the phase: hoisted quadrant (2 bits) + the bucket centre of the 64-chunk offset."""
    sh = 6 - int(bits)
    half = (1 << (sh - 1)) if sh else 0
    out = []
    for d, c in enumerate(codes):
        off = (((int(c) << sh) + half) if sh else int(c)) & 0x3F
        out.append((((int(base[d]) & 3) << 6) | off) & 0xFF)
    return out


def quantize(row, bits):
    """Keep the top `bits` of each phase; reconstruct at the bucket CENTRE. Shifts only."""
    if not (1 <= int(bits) <= 8):
        raise ValueError(f"quantize: bits must be in 1..8, got {bits}")
    sh = 8 - int(bits)
    return [int(x) >> sh & 0xFF for x in row] if sh else [int(x) & 0xFF for x in row]


def dequantize(codes, bits):
    """Code -> the centre of its bucket (the minimax reconstruction point of a uniform cell)."""
    sh = 8 - int(bits)
    if sh == 0:
        return [int(c) & 0xFF for c in codes]
    half = 1 << (sh - 1)
    return [((int(c) << sh) + half) & 0xFF for c in codes]


def pack(codes, bits):
    """Bit-pack codes into a bytearray so the memory win is REAL, not notional. Shifts only."""
    out = bytearray()
    acc = 0
    n = 0
    for c in codes:
        acc = (acc << int(bits)) | (int(c) & ((1 << int(bits)) - 1))
        n += int(bits)
        while n >= 8:
            n -= 8
            out.append((acc >> n) & 0xFF)
    if n:
        out.append((acc << (8 - n)) & 0xFF)
    return out


def unpack(buf, bits, count):
    """Inverse of pack."""
    codes = []
    acc = 0
    n = 0
    i = 0
    mask = (1 << int(bits)) - 1
    while len(codes) < count:
        while n < int(bits):
            acc = (acc << 8) | (buf[i] if i < len(buf) else 0)
            i += 1
            n += 8
        n -= int(bits)
        codes.append((acc >> n) & mask)
    return codes


def encode(row, bits, stride=STRIDE):
    """Key/value row -> packed bytes. precondition (odd stride) -> uniform grid -> bit-pack."""
    return pack(quantize(precondition(row, stride), bits), bits)


def decode(buf, bits, dim, stride=STRIDE):
    """Packed bytes -> the PRECONDITIONED row (this is the domain attention scores in).

    NOTE: we deliberately do NOT un-precondition for scoring. Both the stored keys and the query
    are strided by the same s, so the comparison is consistent. Un-preconditioning is only for
    when a caller wants the original ring values back (and it is exact, up to the grid)."""
    return dequantize(unpack(buf, bits, dim), bits)


def decode_original(buf, bits, dim, stride=STRIDE):
    """Packed bytes -> the ORIGINAL ring values (inverse stride applied). Exact up to the grid."""
    return unprecondition(decode(buf, bits, dim, stride), stride)


class CompressedKVCache:
    """Sub-byte KV cache: odd-stride preconditioned, uniform-grid quantized, bit-packed.

    Stores NO scales and NO zero-points — the ring constant s is the whole "codebook".

        c = CompressedKVCache(dim=16, bits=3)
        c.append(k, v)
        out = c.attend(q)            # q is strided internally; routing happens in the strided domain
    """

    def __init__(self, dim, bits=4, stride=STRIDE, rope=True, mode="stride"):
        """mode='stride' — odd-stride precondition + flat grid. Stores NOTHING but a ring constant.
        mode='chunk'  — the 64-CHUNK READING: hoist the quadrant, spend every bit on the offset.
                        Stores a 2-bit quadrant per coordinate, once (negligible, not zero)."""
        if not is_odd_stride(stride):
            raise ValueError(f"CompressedKVCache: stride must be ODD (a unit), got {stride}")
        if mode not in ("stride", "chunk"):
            raise ValueError(f"CompressedKVCache: mode must be 'stride' or 'chunk', got {mode!r}")
        if mode == "chunk" and int(bits) > 6:
            raise ValueError(f"chunk mode: bits must be <= 6 (the 64-chunk offset is 6 bits)")
        self.dim = int(dim)
        self.bits = int(bits)
        self.stride = int(stride)
        self.rope = bool(rope)
        self.mode = mode
        self.base = None            # chunk mode: the hoisted per-coordinate quadrant (the side info)
        self.K = []
        self.V = []

    def set_base(self, row):
        """chunk mode: pin the 64-chunk each coordinate lives in (from a representative key).

        This IS the codec's side information — 2 bits per coordinate, stored once for the whole
        cache. If unset, it is taken from the first key appended."""
        self.base = quadrant_of(row)
        return self

    def _enc(self, row):
        if self.mode == "chunk":
            return pack(chunk_quantize(row, self.bits, self.base), self.bits)
        return encode(row, self.bits, self.stride)

    def _dec(self, buf):
        if self.mode == "chunk":
            return chunk_dequantize(unpack(buf, self.bits, self.dim), self.bits, self.base)
        return decode(buf, self.bits, self.dim, self.stride)

    def _q(self, row):
        """Put a query into the same domain the stored keys live in.

        chunk mode: the query must pass through the SAME chunk projection as the keys, or the two
        sides live in different frames and routing collapses (measured: 0.27). Projecting both
        sides identically is what makes the comparison meaningful."""
        if self.mode == "chunk":
            return chunk_dequantize(chunk_quantize(row, self.bits, self.base), self.bits, self.base)
        return precondition(row, self.stride)

    def __len__(self):
        return len(self.K)

    def append(self, k, v):
        if len(k) != self.dim or len(v) != self.dim:
            raise ValueError(f"append: expected dim {self.dim}, got k={len(k)} v={len(v)}")
        pos = len(self.K)
        kk = [(int(x) + pos) & 0xFF for x in k] if self.rope else [int(x) & 0xFF for x in k]
        if self.mode == "chunk" and self.base is None:
            self.set_base(kk)                      # pin the 64-chunks from the first key
        self.K.append(self._enc(kk))
        self.V.append(self._enc([int(x) & 0xFF for x in v]))
        return self

    def route(self, q, beta=255, pos=None):
        """Which stored key does this query route to? Returns (best_index, weights).

        This — not value fidelity — is the retrieval bar for a cache: compression is only honest if
        the compressed keys still send the query to the RIGHT binding."""
        from ringkit.ml.kvcache import score_row, boltzmann_weights
        if not self.K:
            raise ValueError("route: cache is empty")
        if pos is None:
            pos = len(self.K) - 1
        qq = [(int(x) + pos) & 0xFF for x in q] if self.rope else [int(x) & 0xFF for x in q]
        qs = self._q(qq)
        Kd = [self._dec(b) for b in self.K]
        row = score_row(qs, Kd)
        return boltzmann_weights(row, beta)[1], row

    def attend(self, q, beta=255, hard=True, pos=None):
        """Route the query over the compressed past and read the value.

        The returned value is the DECOMPRESSED stored value, so it carries the codec's distortion —
        it is not the exact original. Use route() to measure retrieval; compare against
        value_at(idx) to measure value fidelity. Scoring happens in the STRIDED domain."""
        from ringkit.ml.kvcache import score_row, boltzmann_weights, circular_blend
        if not self.K:
            raise ValueError("attend: cache is empty")
        if pos is None:
            pos = len(self.K) - 1
        qq = [(int(x) + pos) & 0xFF for x in q] if self.rope else [int(x) & 0xFF for x in q]
        qs = self._q(qq)
        Kd = [self._dec(b) for b in self.K]
        Vd = [self._dec(b) for b in self.V]
        row = score_row(qs, Kd)
        w, best = boltzmann_weights(row, beta)
        out = list(Vd[best]) if hard else circular_blend(Vd, w, best)
        # stride mode scores in the STRIDED domain, so undo it; chunk mode is already in-domain.
        return unprecondition(out, self.stride) if self.mode == "stride" else out

    def value_at(self, idx):
        """The stored value as the cache can reproduce it (lossy). The fidelity yardstick."""
        out = self._dec(self.V[idx])
        return unprecondition(out, self.stride) if self.mode == "stride" else out

    def nbytes(self):
        """REAL packed footprint — keys + values, bit-packed. Not a notional bit-count."""
        n = 0
        for b in self.K:
            n += len(b)
        for b in self.V:
            n += len(b)
        return n

    def bits_per_coord(self):
        """Exactly self.bits — no scale, no zero-point, no side table amortized on top."""
        return self.bits

    @property
    def raw(self):
        return {"dim": self.dim, "bits": self.bits, "stride": self.stride,
                "len": len(self.K), "bytes": self.nbytes()}
