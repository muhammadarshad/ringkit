"""Tests for ringkit.rmath — the stdlib-math replacement surface.
The module must add NO behavior: every handle is verified exhaustively (all 256 ring
positions) against the core form it re-exports. Run: python3 -m ringkit.tests.test_rmath"""
import ringkit.rmath as rmath
from ringkit.core import native as rn
from ringkit.core import constants as const

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)


print("== constants are the core constants (identity, not copies of behavior) ==")
check("tau == TAU == 256", rmath.tau == const.TAU == 256)
check("pi == HALF == 128", rmath.pi == const.HALF == 128)
check("e == RING_E == 3 (ring-native e, not Euler rounding)", rmath.e == const.RING_E == 3)

print("== every scalar handle == its core form, exhaustive over the 256 ring ==")
PAIRS = [("sin", rmath.sin, rn.SIN), ("cos", rmath.cos, rn.COS),
         ("asin", rmath.asin, rn.ARCSIN), ("acos", rmath.acos, rn.ARCCOS)]
for name, ours, core in PAIRS:
    ok = True
    for x in range(256):
        try:
            a = ours(x)
        except Exception as ea:
            try:
                core(x)
                ok = False
                break
            except Exception:
                continue                      # both reject x the same way
        if a != core(x):
            ok = False
            break
    check(f"{name} == core over all 256", ok)

print("== exp/log: the <3> subgroup roundtrip (64 powers) ==")
ok = all(rmath.log(rmath.exp(k)) == (k, 0) for k in range(64))   # log -> (exponent, branch)
check("log(exp(k)) == (k, 0) for k in 0..63", ok)
check("exp(0) == 1, exp(1) == e", rmath.exp(0) == 1 and rmath.exp(1) == 3)

print("== integer helpers ==")
check("isqrt exact on squares", all(rmath.isqrt(rn.qsm(v, v)) == v for v in range(128)))
check("floordiv == mf_floordiv", all(rmath.floordiv(a, 7) == rn.mf_floordiv(a, 7) for a in range(256)))
check("pow == ring_pow (base 5, exp 0..20)", all(rmath.pow(5, k) == rn.ring_pow(5, k) for k in range(21)))

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
raise SystemExit(0 if not fails else 1)
