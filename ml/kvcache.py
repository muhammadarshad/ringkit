"""
ringkit.ml.kvcache — ring-native KV cache + Boltzmann-soft attention. OUR construction.

D11 — the form BEFORE the build. A KV cache is the MEMORY OF THE PAST: at decode step t the query
must attend over every (key, value) binding laid down before it, without recomputing them. On the
ring that memory is exact and native, because all three pieces of attention are already ring forms
and not one of them needs a multiplier or a float:

  score(q, k_j) = - sum_d ring_distance(q_d, k_jd)     ENERGY: signed, NOT folded mod 256. Folding
                                                       would wrap the ordering away and destroy the
                                                       ranking the whole mechanism rests on.

  w_j           = lut[best - score_j]                  the ring EXPONENTIAL is GEOMETRIC DECAY, not
                                                       ring_exp (which is periodic). This is exactly
                                                       physics.gauge.boltzmann_lut — e^{-beta*dS} —
                                                       the SAME form the lattice engine thermalizes
                                                       with. beta is the inverse temperature: 0 =
                                                       uniform (hot), large = argmax (cold). D11:
                                                       one interlocking physics, not a menu.

  out           = V[best] rotated by the w-weighted mean of the signed ring offsets of every V[j]
                  from V[best].                        Values are ANGLES. A linear weighted mean is
                                                       WRONG on a circle — it wraps (mean of 255 and
                                                       1 is 128, the antipode, when the true blend is
                                                       0). Blending CIRCULARLY around the winner is
                                                       exact, and it degenerates to hard argmax as
                                                       beta grows. No SIN/COS is used, so this
                                                       inherits NO _arch approximation.

Normalization is the ONE division, taken once in ENERGY via mf_floordiv — integer division, NOT a
modular inverse — so the zero-divisor collapse never arises. lut[0] = 255 for every beta, so the
denominator is never zero. This is fold-late (D4) doing the load-bearing work.

Position rides in by exact additive RoPE at INSERT time (rotation-by-addition is exact on the ring
for ANY position, unlike an analytic sinusoid), so a cached key already carries its position and
nothing is recomputed on read. That is why the ring RoPE is the *right* cache primitive.

Storage: 1 byte per coordinate, natively. No scale, no zero-point, no calibration data — the ring IS
the codebook, so the per-block normalization constants that ordinary quantizers must store simply do
not exist here (data-free by construction, not by choice).

Correctness bar (D1): incremental decode through the cache must equal the full uncached recompute
BIT-FOR-BIT at every step. attend_full() below is the uncached semantic reference; the cache is
asserted equal to it in tests/test_kvcache.py.

Multiplier-free. No numpy, no math, no floats.
"""
from ringkit.core import native as rn
from ringkit.ml.attention import ring_distance
from ringkit.physics.gauge import boltzmann_lut


HALF = 128                      # ring half-turn: the wrap boundary for a signed offset


def signed_offset(a, b):
    """Shortest SIGNED path from b to a around the ring, in [-127, 128] (ENERGY, unfolded).

    This is what makes a circular blend possible: offsets live on a line even though values live on
    a circle, so they may be averaged linearly and rotated back."""
    d = (int(a) - int(b)) & 0xFF
    return d - 256 if d > HALF else d


def score_row(q, K):
    """score(q, k_j) = -sum_d ring_distance(q_d, k_jd) for every key. Signed ENERGY, never folded."""
    row = []
    for k in K:
        s = 0
        for d in range(len(q)):
            s -= ring_distance(q[d], k[d])
        row.append(s)
    return row


def boltzmann_weights(row, beta):
    """Ring softmax weights: geometric decay in the score GAP below the best key.

    Returns (weights, best_index). w[best] = lut[0] = 255 always, so sum(w) >= 255 > 0 — the
    denominator can never collapse."""
    if not row:
        raise ValueError("boltzmann_weights: empty score row (attend to an empty cache?)")
    lut = boltzmann_lut(beta)
    best = 0
    for j in range(1, len(row)):
        if row[j] > row[best]:
            best = j
    top = row[best]
    w = []
    for s in row:
        gap = top - s                      # >= 0 by construction, ENERGY
        w.append(lut[gap] if gap < 256 else 0)
    return w, best


def circular_blend(V, w, best):
    """w-weighted CIRCULAR mean of the value rows, taken around V[best].

    out_d = V[best]_d + mean_j( w_j * signed_offset(V[j]_d, V[best]_d) )   then folded to phase.

    Exact: qsm is an exact product (verified over all ring pairs) and mf_floordiv is exact integer
    division. The mean is taken in ENERGY and folded only at the very end (D4)."""
    ref = V[best]
    den = 0
    for wj in w:
        den += wj
    if den <= 0:
        raise ValueError(f"circular_blend: non-positive weight mass {den}")
    out = []
    for d in range(len(ref)):
        num = 0
        for j in range(len(V)):
            if w[j]:
                num += rn.qsm(w[j], signed_offset(V[j][d], ref[d]))     # |w+off| <= 383 < 512, in domain
        if num < 0:
            m = -rn.mf_floordiv(-num, den)          # truncate toward zero, symmetric in sign
        else:
            m = rn.mf_floordiv(num, den)
        out.append((int(ref[d]) + m) & 0xFF)        # fold LAST
    return out


def rope(row, pos):
    """Exact additive ring RoPE: rotate every coordinate by the position. No analytic sine, so this
    is exact for ANY position (unlike a float sinusoid, and unlike our _arch SIN/COS)."""
    return [(int(v) + int(pos)) & 0xFF for v in row]


def attend_full(Q, K, V, beta=16, hard=False, use_rope=True):
    """UNCACHED semantic reference: recompute everything from scratch, every step.

    This is the ground truth the cache must reproduce bit-for-bit (D1). Q, K, V are row lists;
    row i of Q is the query at position i, row j of K/V the binding laid down at position j."""
    if len(K) != len(V):
        raise ValueError(f"attend_full: keys/values length mismatch {len(K)} vs {len(V)}")
    Kp = [rope(k, j) if use_rope else list(k) for j, k in enumerate(K)]
    out = []
    for i, q in enumerate(Q):
        qp = rope(q, i) if use_rope else list(q)
        row = score_row(qp, Kp)
        w, best = boltzmann_weights(row, beta)
        out.append(list(V[best]) if hard else circular_blend(V, w, best))
    return out


class RingKVCache:
    """The cache: past (key, value) bindings, keys already carrying their position.

    Append as you decode; attend reads the whole past. 1 byte per coordinate, no side tables.

        c = RingKVCache(dim=4)
        c.append(k0, v0); c.append(k1, v1)
        out = c.attend(q, beta=16)
    """

    def __init__(self, dim, rope=True):
        self.dim = int(dim)
        self.rope = bool(rope)
        self.K = []
        self.V = []

    def __len__(self):
        return len(self.K)

    def append(self, k, v):
        """Lay down one binding at the next position. RoPE is applied HERE (exactly, additively), so
        the stored key is read-ready forever."""
        if len(k) != self.dim or len(v) != self.dim:
            raise ValueError(f"append: expected dim {self.dim}, got k={len(k)} v={len(v)}")
        pos = len(self.K)
        kk = rope(k, pos) if self.rope else [int(x) & 0xFF for x in k]
        self.K.append(bytearray(kk))
        self.V.append(bytearray(int(x) & 0xFF for x in v))
        return self

    def attend(self, q, beta=16, hard=False, pos=None):
        """Attend the query over the whole cached past. pos defaults to the newest position, which
        is what a decoder wants at step t."""
        if not self.K:
            raise ValueError("attend: cache is empty")
        if pos is None:
            pos = len(self.K) - 1
        qp = rope(q, pos) if self.rope else [int(x) & 0xFF for x in q]
        row = score_row(qp, self.K)
        w, best = boltzmann_weights(row, beta)
        if hard:
            return list(self.V[best])
        return circular_blend(self.V, w, best)

    def nbytes(self):
        """Exact cache footprint: keys + values, 1 byte per coordinate. No scales, no zero-points."""
        return rn.mul(rn.mul(2, self.dim), len(self.K))

    @property
    def raw(self):
        return {"dim": self.dim, "rope": self.rope, "len": len(self.K), "bytes": self.nbytes()}
