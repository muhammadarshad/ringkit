"""
ringkit.physics.gauge — SU(256) lattice gauge engine (the QCM physics core), semantic surface.

Computes the Wilson plaquette action on a Z256 lattice grid (uint8, C-owned memory). The C
kernel (kernels/mprc/lattice/gauge.c) runs the stencil with cache blocking (64-depth tiles =
the "256 = 4 x 64" unroll that locks the working set in L2); its ctypes host, bit-for-bit
Python fallback, and float-normalized observables live in kernels/mprc/lattice/host.py
(charter D9 silicon) and are re-exported here. This module itself is semantic-layer clean:
multiplier-free, no floats.

    g = bytearray(...)                      # W*H*D uint8 lattice (row-major, i fastest)
    e = plaquette(g, W, H, D)               # bytearray energy field
"""
from ringkit.core import native as rn
from ringkit.kernels.mprc.lattice.host import (   # D9 silicon host (hardware ops live there)
    build, available, plaquette, sweep, thermalize, thermalize_rng, session_for,
    correlation, correlation_profile, phase_of, mean_action, _load,
)


def boltzmann_lut(beta):
    """Integer Boltzmann acceptance table — the ring-native EXPONENTIAL form.

    The physics (QCM paper) is e^{-beta dS}; an exponential IS a geometric decay, and
    geometric decay is repeated multiplication: with per-step factor f = (256 - beta)/256,
        lut[dS] = floor(255 * f^dS)
    built by a fixed-point accumulator (rn.mul + shift only — no float exp anywhere).
    This makes the Metropolis acceptance Boltzmann in dS with effective rate
    -ln(1 - beta/256): beta = 0 accepts everything (hot); larger integer beta = colder.
    Replaces the earlier LINEAR ramp (255 - beta*dS), whose chain was NOT Boltzmann
    (docs/project-governance/SOURCES_MAP.md, job 1). 256 bytes, L1-resident."""
    beta = int(beta)
    if not (0 <= beta <= 256):
        raise ValueError(f"boltzmann_lut: beta must be in [0, 256], got {beta}")
    f = 256 - beta                         # per-step decay factor, fixed-point /256
    lut = bytearray(256)
    acc = 255 << 8                         # 16-bit fixed-point accumulator (255.0)
    for ds in range(256):
        lut[ds] = acc >> 8                 # top byte = floor(255 * f^ds)
        acc = rn.mul(acc, f) >> 8          # geometric step: acc *= f/256
    return lut


def criticality_scan(betas, W, H, D, therm=30, seed=0):
    """Sweep beta (coupling); for each, thermalize a fresh random lattice and measure the order
    parameters. Returns [(beta, mean_action, correlation(R=1))]. Locates the ordered<->disordered
    transition: high beta (cold) -> low action / high correlation; low beta (hot) -> the reverse.
    The reported observables are float measurement outputs from the silicon host (labeled IO)."""
    import random as _random
    n = rn.mul(rn.mul(W, H), D)
    out = []
    for b in betas:
        rng = _random.Random(seed)
        g = bytearray(rng.randbytes(n))
        L = boltzmann_lut(b)
        thermalize_rng(g, rng.getrandbits(32), L, W, H, D, therm)
        out.append((b, mean_action(g, W, H, D), correlation(g, 1, W, H, D)))
    return out


def mass_gap_scan(betas, W, H, D, therm=30, seed=0, rmax=10):
    """The QCM paper's main physics readout: for each beta, thermalize and measure the FULL
    C(R) profile (R=1..rmax) plus the phase reading (confined = mass gap: alignment dead by
    R=5; deconfined = long-range order). Returns [(beta, profile, phase)]. Observables and
    phase labels are measurement outputs from the silicon host (labeled IO)."""
    import random as _random
    n = rn.mul(rn.mul(W, H), D)
    out = []
    for b in betas:
        rng = _random.Random(seed)
        g = bytearray(rng.randbytes(n))
        L = boltzmann_lut(b)
        thermalize_rng(g, rng.getrandbits(32), L, W, H, D, therm)
        prof = correlation_profile(g, W, H, D, rmax)
        out.append((b, prof, phase_of(prof)))
    return out
