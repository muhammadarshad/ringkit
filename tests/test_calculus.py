"""Production tests for ringkit.core.calculus. Run: python3 -m ringkit.tests.test_calculus"""
import random
from ringkit.core import native as rn
from ringkit.core import calculus as rc

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)
def raises(exc, f):
    try: f(); return False
    except exc: return True
    except Exception: return False

print("== rotational derivative (iota), period-4 cycle ==")
dSIN = rc.d_rot(rn.SIN)
check("d(SIN) == COS over 256", all(dSIN(p) == rn.COS(p) for p in range(256)))
check("d(COS) == -SIN over 256", all(rc.d_rot(rn.COS)(p) == rn.ring_neg(rn.SIN(p)) for p in range(256)))
check("d^2(SIN) == -SIN", all(rc.d_rot_power(rn.SIN, 2)(p) == rn.ring_neg(rn.SIN(p)) for p in range(256)))
check("d^3(SIN) == -COS", all(rc.d_rot_power(rn.SIN, 3)(p) == rn.ring_neg(rn.COS(p)) for p in range(256)))
check("d^4(SIN) == SIN (period 4)", all(rc.d_rot_power(rn.SIN, 4)(p) == rn.SIN(p) for p in range(256)))
check("integral_rot inverts d_rot", all(rc.integral_rot(rc.d_rot(rn.SIN))(p) == rn.SIN(p) for p in range(256)))

print("== accumulation / differential (FTRC) ==")
ok = True
for _ in range(3000):
    seq = [random.randint(0, 255) for _ in range(random.randint(2, 40))]
    if not rc.ftrc_holds(seq): ok = False; break
check("FTRC on 3000 random seqs", ok)
d = [random.randint(0, 255) for _ in range(20)]
check("differential(integral(d,c0)) == d", rc.differential(rc.integral(d, 7)) == d)
check("differential of squares == odd increments (start 3)", rc.differential([rn._SQ[n] % 256 for n in range(1, 30)]) == [(2*n+1) % 256 for n in range(1, 29)])
check("nth_differential order0 = seq", rc.nth_differential([1,2,3], 0) == [1,2,3])
check("nth_differential order2 length", len(rc.nth_differential(list(range(10)), 2)) == 8)

print("== edge / error ==")
check("differential([]) == []", rc.differential([]) == [])
check("differential([5]) == []", rc.differential([5]) == [])
check("ftrc_holds([]) vacuously True", rc.ftrc_holds([]) is True)
check("ftrc_holds([7]) vacuously True", rc.ftrc_holds([7]) is True)
check("nth_differential neg order -> ValueError", raises(ValueError, lambda: rc.nth_differential([1,2], -1)))
check("d_rot_power neg -> ValueError", raises(ValueError, lambda: rc.d_rot_power(rn.SIN, -1)))

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
