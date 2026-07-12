"""
ring_qcm.py — QCM: the layer that holds the topologies (on ring_native).

Sources (verified this session):
  - SILIQ (Silicon Lattice Intelligence Quantum), docs/ALGORITHM.md — concrete Z256 ring ops,
    bit-encoded state, 7-prime walk, 128x113 hypervector, biopod (manifold) geometry.
  - QCM paper (Zenodo 18883754, abstract) — SU(N_c) lattice gauge, register-forced 8-bit SIMD,
    Weyl-hash PRNG ("dance of quantum & silicon"). NOTE: only the abstract was accessible (the PDF
    body is binary); the SU(N) gauge MCMC engine is NOT reproduced here — this module holds the
    verified state/topology primitives that the numpy layer needs.

Topologies held:
  LATTICE     — Z256 ring of ternary-state nodes (the discrete lattice)
  TORUS       — periodic ring (conjugate = additive inverse mod 256)
  HYPERVECTOR — 128 x 113 = 14464 uint8 batch, L1-cache resident
  MANIFOLD    — the biopod: circles at radius k, height arctan(k/N)

All integer, multiplier-free (bit ops + ring_native).
"""
from ringkit.core import native as rn

from ringkit.core.constants import TAU, VACUUMS
QUADRANTS = ("UP+", "UP-", "DN+", "DN-")

# ── QCM node state (SILIQ ALGORITHM.md sec 3.2) ──────────────────────────────
def spin(d):
    """bit 7 of the phase: 0 = UP, 1 = DOWN."""
    return (d >> 7) & 1


def polarity(d):
    """bit 6 of the phase: 0 = POS, 1 = NEG."""
    return (d >> 6) & 1


def state(d):
    """ternary bipolar state +1 / -1 from polarity (multiplier-free)."""
    return 1 - (polarity(d) << 1)


def conjugate(d):
    """additive inverse mod 256 (== ring_neg). Torus antipode."""
    return (-d) & 0xFF


def quadrant(d):
    """which of the 4 bit-quadrants: UP+ / UP- / DN+ / DN-."""
    return QUADRANTS[(d >> 6) & 3]


def is_vacuum(d):
    """ring boundary d mod 64 == 0 -> {0,64,128,192}. No prime / no state."""
    return (d & 0x3F) == 0


# ── LATTICE traversal: the 7-prime phase walk (SILIQ sec 4.2) ────────────────
_STEPS = (2, 3, 5, 7, 11, 13, 17)


def seven_prime_walk(count=None):
    """Yield ring phases via the 7-prime cyclic step, skipping vacuums.
    Covers all 252 non-vacuum positions (verified). count=None -> one full cover."""
    d = 1
    si = 0
    seen = set()
    while True:
        d = (d + _STEPS[si % 7]) & 0xFF
        si += 1
        while is_vacuum(d):
            d = (d + _STEPS[si % 7]) & 0xFF
            si += 1
        yield d
        seen.add(d)
        if count is None and len(seen) == 252:
            return
        if count is not None:
            count -= 1
            if count <= 0:
                return


# ── HYPERVECTOR batch (SILIQ sec 4.1): 128 x 113 uint8, L1-resident ──────────
HV_W = 128                       # tau/2, one spin-half of Z256
HV_H = 113                       # prime (no resonance with ring periods)
HV_CELLS = rn.mul(HV_W, HV_H)    # 14464 cells
HV_BYTES = HV_CELLS              # uint8 sieve -> 14464 bytes (~14.5 KB, fits 32 KB L1)


def hypervector(values=()):
    """A 128x113 uint8 batch (14464 cells), the L1-resident compute tile."""
    hv = bytearray(HV_BYTES)
    for i, v in enumerate(values):
        if i >= HV_BYTES:
            break
        hv[i] = int(v) & 0xFF
    return hv


# ── MANIFOLD: the biopod (SILIQ sec 2) ───────────────────────────────────────
def midpoint(x):
    """M node at N = x >> 1 (integer, no division)."""
    return int(x) >> 1


def arms(N, k):
    """bipolar arms of a candidate pair: lo = N-k (u' axis), hi = N+k (u axis)."""
    return N - k, N + k


def manifold_coord(k, N):
    """biopod placement of a pair at displacement k about midpoint N:
    radius = k, height = ring-arc of arctan(k/N)  (uses ring ARCTAN, multiplier-free)."""
    if N == 0:
        return k, 0
    tan_scaled = rn.mf_floordiv(rn.scale21(k), N)   # SCALE * (k/N)
    height_arc = rn.ARCTAN(tan_scaled)
    return k, height_arc
