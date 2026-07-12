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
    correlation, mean_action, _load,
)


def boltzmann_lut(beta):
    """Integer Boltzmann acceptance table (ring-native, no float): higher dS -> lower accept
    threshold; larger integer `beta` = colder (rejects uphill moves harder). 256 bytes, L1-resident."""
    beta = int(beta)
    lut = bytearray(256)
    for ds in range(256):
        v = 255 - (rn.mul(ds, beta) if beta else 0)     # linear decay (ring-native accept policy)
        lut[ds] = v if v > 0 else 0
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
