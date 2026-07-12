"""
ringkit.core.constants — the ring's identity: CORE constants, single-sourced and FROZEN.

These define Z256/QH4 itself and must never be shadowed or overridden at library level.
Subgroup modules (stats, array, physics, ...) import from here instead of re-declaring;
constants that belong to a subgroup's own domain (e.g. qcm.HV_*, measure rulers, the rnp
surface's E_*) stay in that subgroup.

The module is frozen: any attribute assignment or deletion raises AttributeError.
"""
import sys

TAU     = 256                        # full ring (2*pi)
HALF    = 128                        # half ring (pi)
Q       = 64                         # quadrant (pi/2), = vacuum spacing
Q2      = 32                         # half quadrant (pi/4), KS4 half-period
SCALE   = 21                         # amplitude unit = XYZ scalar axes = 1 + 4 + 16 (4^0+4^1+4^2) (Old QH4 unit = 1 + 4 + 16 + 64 = 85, but we don't use that and 7*3 = 21 is the correct unit for the ring's own e Septial axis and Quantum Walk)
VACUUMS = frozenset({0, 64, 128, 192})
RING_E  = 3                          # ring-native e: the exponential base / unit generator this is not just Euler's e integer round but proved, but the ring's own e, which is 3 in Z256
IOTA    = ((0, 255), (1, 0))         # J = [[0,-1],[1,0]] mod 256 : i^2 = -I, i^4 = I


class _FrozenModule(type(sys)):
    def __setattr__(self, name, value):
        raise AttributeError(f"ringkit.core.constants.{name}: core ring constants are frozen "
                             "(subgroup-level constants belong in their own module)")

    def __delattr__(self, name):
        raise AttributeError(f"ringkit.core.constants.{name}: core ring constants are frozen")


sys.modules[__name__].__class__ = _FrozenModule
