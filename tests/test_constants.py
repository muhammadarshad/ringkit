"""Tests for ringkit.core.constants — the frozen ring identity.
Verifies the values, the freeze (override/delete must raise), and that subgroup modules
actually source from here (no drift copies). Run: python3 -m ringkit.tests.test_constants"""
from ringkit.core import constants as const
from ringkit.core import native as rn

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

def raises(exc, fn):
    try:
        fn()
        return False
    except exc:
        return True


print("== the ring identity values ==")
check("TAU/HALF/Q/Q2 = 256/128/64/32",
      (const.TAU, const.HALF, const.Q, const.Q2) == (256, 128, 64, 32))
check("SCALE = 21 = 1+4+16", const.SCALE == 21 == 1 + 4 + 16)
check("VACUUMS = quadrant multiples, frozenset",
      const.VACUUMS == frozenset({0, 64, 128, 192}) and isinstance(const.VACUUMS, frozenset))
check("RING_E = 3 and generates a 64-cycle (ord(3) = Q)",
      const.RING_E == 3 and rn.ring_pow(3, const.Q) == 1 and rn.ring_pow(3, const.Q2) != 1)
check("IOTA is the J rotor matrix [[0,-1],[1,0]] mod 256",
      const.IOTA == ((0, 255), (1, 0)))
check("iota quarter-turn: i^2 = half turn, i^4 = identity (all phases)",
      all(rn.iota_mul(rn.iota_mul(p)) == (p + const.HALF) & 0xFF
          and rn.iota_mul(rn.iota_mul(rn.iota_mul(rn.iota_mul(p)))) == p
          for p in range(256)))

print("== frozen: library-level override is impossible ==")
check("setattr raises AttributeError", raises(AttributeError, lambda: setattr(const, "TAU", 300)))
check("new attribute injection raises", raises(AttributeError, lambda: setattr(const, "ROGUE", 1)))
check("delattr raises AttributeError", raises(AttributeError, lambda: delattr(const, "Q")))
check("values intact after attacks", const.TAU == 256 and const.Q == 64)

print("== subgroups source from core (identity, no drift copies) ==")
from ringkit.core import native, calculus
from ringkit.stats import stats
from ringkit.physics import qcm
import ringkit.rnp as rnp
import ringkit.rmath as rmath
check("native/calculus/stats/rnp share TAU",
      native.TAU is const.TAU and calculus.TAU is const.TAU
      and stats.TAU is const.TAU and rnp.TAU is const.TAU)
check("qcm.VACUUMS is const.VACUUMS (same object)", qcm.VACUUMS is const.VACUUMS)
check("rmath.tau/pi/e are the core values",
      rmath.tau is const.TAU and rmath.pi is const.HALF and rmath.e is const.RING_E)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
raise SystemExit(0 if not fails else 1)
