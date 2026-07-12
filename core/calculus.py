"""
ring_calculus.py — ring-native calculus (Phase 1 of the SRD), on ring_native only.

Two exact notions, both verified:
  ROTATIONAL (closed-form, trig family):  d = iota (+Q arc shift).
      d(SIN)=COS, d(COS)=-SIN, period 4.
  ACCUMULATION / DIFFERENTIAL (general sequences): the ADI (integral, d) pair.
      differential = forward difference (mod ring); integral = cumulative sum.
      FTRC: integral(differential(seq), seq[0]) == seq   (exact, telescoping).
"""
from ringkit.core import native as rn

TAU = 256
Q = 64


# ── rotational derivative (iota) — exact for the trig family ─────────────────
def d_rot(fn):
    """Rotational derivative operator: (d fn)(phi) = fn(phi + Q). d(SIN)=COS, period 4."""
    return lambda phi: fn((int(phi) + Q) % TAU)


def integral_rot(fn):
    """Rotational integral = inverse rotation: (∫ fn)(phi) = fn(phi - Q)."""
    return lambda phi: fn((int(phi) - Q) % TAU)


# ── accumulation / differential — general ring sequences (ADI) ───────────────
def differential(seq):
    """Forward difference mod ring: d[i] = (seq[i+1] - seq[i]) % 256. Length n -> n-1."""
    seq = [int(v) % TAU for v in seq]
    return [(seq[i + 1] - seq[i]) % TAU for i in range(len(seq) - 1)]


def integral(diffs, c0=0):
    """Accumulation mod ring from a start value c0. Inverse of differential."""
    out = [int(c0) % TAU]
    for d in diffs:
        out.append((out[-1] + int(d)) % TAU)
    return out


def ftrc_holds(seq):
    """Fundamental Theorem of Ring Calculus: integral(differential(seq), seq[0]) == seq.
    Vacuously True for length 0 or 1."""
    seq = [int(v) % TAU for v in seq]
    if len(seq) <= 1:
        return True
    return integral(differential(seq), seq[0]) == seq


def nth_differential(seq, order=1):
    """Apply the forward difference `order` times (order >= 0). order=0 returns seq unchanged."""
    if int(order) < 0:
        raise ValueError(f"nth_differential: order must be >= 0, got {order}")
    out = [int(v) % TAU for v in seq]
    for _ in range(int(order)):
        out = differential(out)
    return out


def d_rot_power(fn, order=1):
    """Rotational derivative applied `order` times: period 4 (SIN->COS->-SIN->-COS->SIN)."""
    if int(order) < 0:
        raise ValueError(f"d_rot_power: order must be >= 0, got {order}")
    g = fn
    for _ in range(int(order)):
        g = d_rot(g)
    return g
