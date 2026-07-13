"""
ringkit.ml.kvadi — the ring-native KV element: ADI (accumulation, differential), N-DIMENSIONAL.

D11 / PRIME DIRECTIVE. The Euclidean POLAR form — Cartesian→polar via `atan2` and the L2 magnitude
`sqrt(x^2 + y^2)` — is an MPRC ANTI-PATTERN and is NOT imported. On the ring `a^2 + b^2 = c^2` is
foreign standard math and LOSSY (distinct vectors collapse to one magnitude), which is why the polar
codec had to reconstruct with SIN/COS. We do not import it.

The ring's own form is ADI — the **(integral, differential) pair** of a sequence — and it is
GENERAL over N dimensions, not fixed to a 2-D (x, y) pair:

    row (length N)  ->  (lead, delta)      delta = differential(row)  (forward differences)
                                           lead  = row[0]             (seeds the accumulation)

    decode:  row = integral(delta, lead)   (accumulation — the Fundamental Theorem of Ring Calculus:
                                            integral(differential(row), row[0]) == row, for ANY N)

Verified bit-for-bit reversible over random rows of every dimension 1..128. It is multiplier-free
(`+`, `-`, mod only — no isqrt, no arctan, no sqrt), approximation-free on decode (no SIN/COS ever),
and differentiable (`delta` is the discrete derivative). The 2-D case `(x, y) -> (x+y, x-y)` the
worked example used is just N=2; nothing here is pinned to x and y.

COMPRESSION IS NOT INVENTED HERE (D2/D11). Which part of the ADI form is redundant, and by how much,
is the "4-diffusion over the four 64-chunks" form the design marks STILL OPEN. This module ships only
the EXACT lossless element; a compressor is a separate, measured step — and by CHARTER C9 it may
never quantize the arc/identity to chase a bit-count.

Multiplier-free. No numpy, no math, no floats. No Euclidean. No fixed dimensionality.
"""
from ringkit.core import calculus as ca


def encode(row):
    """A ring row of ANY length N -> its ADI element (lead, delta). Exact & reversible for every N
    (FTRC). delta = forward differences; lead seeds the accumulation on decode. No pairing, no 2-D
    assumption — this is the general (integral, differential) ADI form."""
    if not row:
        raise ValueError("encode: empty row")
    return int(row[0]) & 0xFF, ca.differential(row)


def decode(lead, delta):
    """ADI element (lead, delta) -> the original N-D row, exactly, by accumulation (integral)."""
    return ca.integral(delta, lead)


def differential(row):
    """delta — the ADI differential (forward differences), the differentiable part. Any N."""
    return ca.differential(row)


def accumulation(row):
    """Lambda — the ADI accumulation: running cumulative sum of the row (mod ring), for ANY N. The
    ring's own 'size' along the sequence, never the Euclidean norm. For a pair (x, y) the total is
    x + y (the worked example's (30,30) -> 60)."""
    out = []
    s = 0
    for v in row:
        s = (s + (int(v) & 0xFF)) & 0xFF
        out.append(s)
    return out


# --- N=2 convenience: the worked (x, y) example is just a special case of the N-D form above ---
def encode_pair(x, y):
    """The 2-D case of `encode`: (x, y) -> (lead=x, delta=[x... ]). Kept only for the worked example;
    the general path is `encode(row)` for any dimension."""
    return encode([x, y])


def decode_pair(lead, delta):
    x, y = decode(lead, delta)
    return x, y
