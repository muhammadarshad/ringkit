"""Ring trig tables for the QCM front-end (the `_arch` semicircle, SCALE=21) — PURE INTEGER.

Borrowed handle: SIN/COS are the ring's `_arch` semicircle (exact at the 4 cardinals) — NOT libm
sine. Only the ring (Q<frac> integer) tables live here; any float mirror belongs in a test oracle,
never in the package. No floats, no `math`/`numpy` import — the ring's own isqrt (rn.isqrt).
"""
from ringkit.core import native as rn

FRAC = 16
ONE = 1 << FRAC
SCALE = 21
HALF = 128
Q = 64


def _sd(n, d):
    return -rn.mf_floordiv(-n, d) if n < 0 else rn.mf_floordiv(n, d)


def _arch(p, hp):
    if p <= 0 or p >= hp:
        return 0
    return _sd(SCALE * 2 * rn.isqrt(p * (hp - p)), hp)     # integer arc; rn.isqrt, not math


def _SIN(p):
    p &= 0xFF
    return _arch(p, HALF) if p < HALF else (-_arch(p - HALF, HALF)) % 256


def _COS(p):
    return _SIN((p + Q) & 0xFF)


def _sg(v):
    return v - 256 if v > HALF else v


COS_U = [_sg(_COS(p)) for p in range(256)]                 # signed unit (integer)
SIN_U = [_sg(_SIN(p)) for p in range(256)]
COSQ = [_sd(COS_U[a] << FRAC, SCALE) for a in range(256)]  # cos in Q<frac>, ring (integer)
SINQ = [_sd(SIN_U[a] << FRAC, SCALE) for a in range(256)]
