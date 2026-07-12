"""Production tests for ringkit.stats.stats. math = external oracle only.
Run: python3 -m ringkit.tests.test_stats"""
import math
import random
from ringkit.core import native as rn
from ringkit.stats import stats as st

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)
def raises(exc, f):
    try: f(); return False
    except exc: return True
    except Exception: return False

print("== ring_dist ==")
check("== brute min(|a-b|,256-|a-b|)", all(st.ring_dist(a, b) == min((a-b) % 256, (b-a) % 256) for a in range(0,256,5) for b in range(0,256,7)))
check("symmetric", all(st.ring_dist(a, b) == st.ring_dist(b, a) for a in range(0,256,11) for b in range(0,256,13)))
check("range 0..128", all(0 <= st.ring_dist(a, b) <= 128 for a in range(256) for b in range(0,256,17)))
check("straddle {250,6} dist=12", st.ring_dist(250, 6) == 12)

print("== ARCTAN2 (direction) ==")
def sg(x): return rn._signed(x)
err = max(st.ring_dist(st.ARCTAN2(sg(rn.SIN(p)), sg(rn.COS(p))), p) for p in range(256))
check("ARCTAN2(SIN,COS) recovers arc (<=3)", err <= 3)
check("zero vector -> None", st.ARCTAN2(0, 0) is None)
check("axis cases", st.ARCTAN2(5, 0) == 64 and st.ARCTAN2(-5, 0) == 192)

print("== circular_mean ==")
def true_cmean(arcs):
    sx = sum(math.cos(2*math.pi*a/256) for a in arcs); sy = sum(math.sin(2*math.pi*a/256) for a in arcs)
    if abs(sx) < 1e-9 and abs(sy) < 1e-9: return None
    return round(math.atan2(sy, sx)*256/(2*math.pi)) % 256
for c in ([10,20,30], [250,6,2], [60,64,68], [100,110,90,105]):
    r, t = st.circular_mean(c), true_cmean(c)
    check(f"circular_mean {c} within 5 of oracle", r is not None and st.ring_dist(r, t) <= 5)
check("straddle-zero -> ~0 not 128", st.ring_dist(st.circular_mean([250, 6]), 0) <= 3)
check("antipodal {0,128} -> None", st.circular_mean([0, 128]) is None)
check("empty -> ValueError", raises(ValueError, lambda: st.circular_mean([])))

print("== resultant_length (bug fix: any arc count) ==")
check("100 arcs no crash", isinstance(st.resultant_length([random.randint(0,255) for _ in range(100)]), int))
check("concentrated > spread", st.resultant_length([64]*20) > st.resultant_length(list(range(0,256,13))))
check("matches float sqrt(sx^2+sy^2)", (lambda arcs: abs(st.resultant_length(arcs) - round(math.hypot(
        sum(sg(rn.COS(a)) for a in arcs), sum(sg(rn.SIN(a)) for a in arcs)))) <= 1)([10,20,30,300%256,7]))
check("empty -> ValueError", raises(ValueError, lambda: st.resultant_length([])))

print("== circular_median (L1) ==")
for c in ([250,6,2], [10,20,200], [64,64,64,192]):
    m = st.circular_median(c)
    brute = min(range(256), key=lambda k: sum(st.ring_dist(k, a) for a in c))
    check(f"median {c} == brute L1 min", m == brute)
check("empty -> ValueError", raises(ValueError, lambda: st.circular_median([])))

print("== geometric_mean ==")
for c in ([4,16], [9,25], [2,8,32], [10,20,40,80], [3,3,3]):
    g = st.geometric_mean(c)
    # exact integer n-th root of the product (more correct than float floor)
    prod = 1
    for v in c: prod *= v
    exact = int(round(prod ** (1.0/len(c))))
    check(f"geomean {c}", abs(g - exact) <= 1 and g ** len(c) <= prod < (g+1) ** len(c))
check("single value", st.geometric_mean([42]) == 42)
check("empty -> ValueError", raises(ValueError, lambda: st.geometric_mean([])))
check("negative -> ValueError", raises(ValueError, lambda: st.geometric_mean([4, -1])))

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
