"""
ring_measure.py — MEASUREMENT LAYER (ENERGY side), built on ring_native.

The 256 core (ARC / identity) is immutable and lives in ring_native. This layer is the
selectable "ruler": finer amplitude resolution and the hardware (hypervector) layout,
applied ON the core, never inside it.

Measurement rings (structural ring-counts, not arbitrary multipliers):
    CORE    = 256    one axis / the core identity  [(a+b)^2 (c+d)^2]^2 = 16^2 = 2^8
    AXES    = 1024   XYZU: 4 axes, each its own 256 ring  (4 * 256)
    ACC_OVR = 512    U accumulator (256) + Energy overspill (256)
    WORKING = 1536   AXES + ACC_OVR  (full working ring)

NOTE: the HyperVector topology moved to ring_qcm (it's a QCM topology, not measurement) AND was
corrected: per SILIQ ALGORITHM.md it is 128x113 = 14464 uint8 CELLS = 14464 BYTES (L1-resident),
not 1808 (my earlier version wrongly treated 14464 as bits).

Charter: multiplier-free (mul/mf_floordiv/mf_mod/qsm), imports only ring_native.
"""
from ringkit.core import native as rn

# ── measurement rings ────────────────────────────────────────────────────────
CORE = 256
AXES = rn.mul(4, CORE)                 # 1024  (XYZU)
ACC_OVR = CORE + CORE                  # 512   (U accumulator + energy overspill)
WORKING = AXES + ACC_OVR               # 1536


# ── energy-overspill square table ────────────────────────────────────────────
# The core _SQ (ring_native, size 512) covers measurement rings up to N=1024. Larger
# rulers (1536, 1808) overspill it, so the measurement layer keeps its OWN extended square
# table — built the same multiplier-free way (odd-number accumulation). Sized to the
# largest ruler WORKING/... : accumulation s = N/2 reaches 904 for N=1808, so cover 1024.
_EXT_NSQ = 1024
_SQ_EXT = [0 for _ in range(_EXT_NSQ + 1)]
for _n in range(1, _EXT_NSQ + 1):
    _SQ_EXT[_n] = _SQ_EXT[_n - 1] + ((_n - 1) << 1) + 1


def _qsm_ext(x, y):
    s = abs(x + y)
    d = abs(x - y)
    return (_SQ_EXT[s] - _SQ_EXT[d]) >> 2


def _isqrt_ext(m):
    lo, hi = 0, _EXT_NSQ
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if _SQ_EXT[mid] <= m:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ── high-resolution measurement sine (undivided arch, chord-parametrized) ────
# Same shape as the core SIN, but the isqrt magnitude is NOT folded down to SCALE=21 —
# it is kept, giving finer amplitude. Two lobes over a measurement ring of size N (even).
# N=512 reproduces core SIN512 exactly. Uses the overspill table so N up to 2*_EXT_NSQ works.
def measure_sin(phi, N=ACC_OVR):
    """Measurement sine at resolution N (even, >=2). N=512 -> +-2688; larger N -> finer amplitude."""
    if N < 2 or (N & 1):
        raise ValueError(f"measure_sin: N must be an even integer >= 2, got {N}")
    P = N >> 1
    if P > _EXT_NSQ:
        raise ValueError(f"N={N} exceeds overspill table (accumulation {P} > {_EXT_NSQ})")
    p = rn.mf_mod(int(phi), N)
    if p < P:
        return rn.scale21(_isqrt_ext(_qsm_ext(p, P - p)))       # + lobe
    q = p - P
    return -rn.scale21(_isqrt_ext(_qsm_ext(q, P - q)))          # - lobe


def SIN512(phi):
    """Spinor double-cover wave (ACC_OVR = 512 = U-accumulator + energy overspill).
    Full oscillation, peak +-2688. This is measure_sin at N=512 — it lives here, not in
    the core, because 512 is ENERGY-side and it returns amplitude, not a ring position."""
    return measure_sin(phi, ACC_OVR)


def layout():
    """Human-readable summary of the measurement rings (hypervector is in ring_qcm)."""
    return {"CORE": CORE, "AXES(XYZU)": AXES, "ACC_OVR": ACC_OVR, "WORKING": WORKING}
