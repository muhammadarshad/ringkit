"""
ringkit.rmath — the stdlib-math replacement, ring-native (Z256). NOT standard math.

Scalar math with the same shape as `import math`, but every function is the ring's own:
multiplier-free, exact where the ring is exact, and honest about approximation (SIN/COS use
the _arch semicircle — exact only at the 4 cardinals).

    import ringkit.rmath as rmath
    rmath.sin(64)            # ring SIN (quarter turn)
    rmath.exp(5)             # 3^5 mod 256 — ring-native e is 3, not 2.718...
    rmath.isqrt(81)          # exact integer sqrt
    rmath.tau, rmath.e       # 256, 3

Names are handles (charter): `e` here is RING_E = 3, the ring's own exponential base — the
generator of the unit subgroup — not a rounding of Euler's e. Everything re-exported from
core.native / core.constants; this module adds no behavior of its own.
"""

from ringkit.core.constants import TAU, HALF, Q, Q2, RING_E
from ringkit.core.native import (
    SIN, COS, TAN, SEC, CSC, COT, ARCSIN, ARCCOS, ARCTAN, KS4,
    ring_exp, ring_log, ring_pow, ring_cis, rotate, cis_rotate,
    mul, ipow, mf_floordiv, mf_mod, ring_neg, isqrt, qsm,
)

# math-module-shaped lowercase handles
tau = TAU                    # full ring (2*pi)
pi = HALF                    # half ring (pi)
e = RING_E                   # ring-native e = 3

sin = SIN
cos = COS
tan = TAN
sec = SEC
cot = COT
csc = CSC
asin = ARCSIN
acos = ARCCOS
atan = ARCTAN
exp = ring_exp
log = ring_log
pow = ring_pow
floordiv = mf_floordiv
fmod = mf_mod
