"""Production tests for ringkit.physics.gauge (SU(256) plaquette engine).
Run: python3 -m ringkit.tests.test_gauge"""
import random
import time
from ringkit.physics import gauge

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)
def raises(exc, f):
    try: f(); return False
    except exc: return True
    except Exception: return False

print("== correctness (C == Python reference, bit-for-bit) ==")
random.seed(2)
W, H, D = 8, 7, 10
g = bytearray(random.randint(0, 255) for _ in range(W * H * D))
c_blocked = gauge.plaquette(g, W, H, D, blocked=True)
c_plain = gauge.plaquette(g, W, H, D, blocked=False)
py = gauge.plaquette(g, W, H, D, force_python=True)
check("C blocked == Python reference", c_blocked == py)
check("C plain == C blocked (blocking is transparent)", c_plain == c_blocked)
check("available (built)", gauge.available())
check("wrong length -> ValueError", raises(ValueError, lambda: gauge.plaquette(bytearray(10), 8, 7, 10)))

print("== cache-blocking benchmark (128 x 113 x 256, the QCM dims) ==")
W, H, D = 128, 113, 256
G = bytearray(random.randint(0, 255) for _ in range(W * H * D))
nodes = (W - 2) * (H - 2) * (D - 2)

def bench(fn, reps=3):
    best = 1e18
    for _ in range(reps):
        t = time.perf_counter(); fn(); dt = time.perf_counter() - t
        best = min(best, dt)
    return best / nodes * 1e9   # ns/node

ns_blocked = bench(lambda: gauge.plaquette(G, W, H, D, blocked=True))
ns_plain = bench(lambda: gauge.plaquette(G, W, H, D, blocked=False))
print(f"  blocked (64-tile): {ns_blocked:.3f} ns/node")
print(f"  plain            : {ns_plain:.3f} ns/node")
print(f"  (report-only) cache-blocking factor: {ns_plain/ns_blocked:.2f}x  "
      f"-- honest: streaming stencil, prefetcher-bound; blocking's win needs L2-spill reuse")
check("throughput sane (<5 ns/node)", ns_blocked < 5.0)   # robust, non-timing-flaky

print("== Metropolis sweep (SU(256) gauge dynamics) ==")
Ws, Hs, Ds = 12, 12, 12
n = Ws * Hs * Ds
random.seed(7)
g0 = bytearray(random.randint(0, 255) for _ in range(n))
prop = bytearray(random.randint(0, 255) for _ in range(n))
chance = bytearray(random.randint(0, 255) for _ in range(n))
lut = gauge.boltzmann_lut(beta=40)                       # cold
# C == Python (identical rng inputs), one sweep bit-for-bit
gc, gp = bytearray(g0), bytearray(g0)
gauge.sweep(gc, prop, chance, lut, Ws, Hs, Ds, force_python=False)
gauge.sweep(gp, prop, chance, lut, Ws, Hs, Ds, force_python=True)
check("C sweep == Python sweep (bit-for-bit)", gc == gp)

# threaded kernels: predictable bins (checkerboard slabs), no locks, no merge —
# results must be BIT-IDENTICAL to single-threaded
from ringkit.kernels.mprc.lattice import host as _lat
_lib = _lat._load()
gmt = bytearray(g0)
for _par in (0, 1):
    _lib.metropolis_sweep_mt(_lat._ptr(gmt), _lat._ptr(bytearray(prop)),
                             _lat._ptr(bytearray(chance)), _lat._ptr(bytearray(lut)),
                             Ws, Hs, Ds, _par, 8)
check("mt(8) sweep == single-thread sweep (bit-for-bit)", gmt == gc)
rst = bytearray(g0); rmt = bytearray(g0)
for _s in range(2):
    for _par in (0, 1):
        _lib.metropolis_sweep_rng(_lat._ptr(rst), 42, _s, _lat._ptr(bytearray(lut)), Ws, Hs, Ds, _par)
        _lib.metropolis_sweep_rng_mt(_lat._ptr(rmt), 42, _s, _lat._ptr(bytearray(lut)), Ws, Hs, Ds, _par, 8)
check("mt(8) rng sweep == single-thread rng sweep (bit-for-bit)", rmt == rst)
emt = bytearray(len(g0)); est = bytearray(len(g0))
_lib.plaquette_mt(_lat._ptr(emt), _lat._ptr(bytearray(g0)), Ws, Hs, Ds, 8)
_lib.plaquette(_lat._ptr(est), _lat._ptr(bytearray(g0)), Ws, Hs, Ds)
check("mt(8) plaquette == single-thread plaquette (bit-for-bit)", emt == est)

# thermalization: cold orders (action drops), hot stays disordered
def run(beta, sweeps):
    g = bytearray(g0); L = gauge.boltzmann_lut(beta)
    before = gauge.mean_action(g, Ws, Hs, Ds)
    for _ in range(sweeps):
        p = bytearray(random.randint(0, 255) for _ in range(n))
        ch = bytearray(random.randint(0, 255) for _ in range(n))
        gauge.sweep(g, p, ch, L, Ws, Hs, Ds)
    return before, gauge.mean_action(g, Ws, Hs, Ds)
cold_b, cold_a = run(beta=60, sweeps=40)   # cold: reject uphill -> orders
hot_b, hot_a = run(beta=0, sweeps=40)      # hot: accept all -> stays disordered
print(f"  cold beta=60: mean action {cold_b:.1f} -> {cold_a:.1f}")
print(f"  hot  beta=0 : mean action {hot_b:.1f} -> {hot_a:.1f}")
check("cold thermalizes to lower action (orders)", cold_a < cold_b * 0.7)
check("cold ends far below hot (phase separation)", cold_a < hot_a * 0.6)

print("== E2: criticality scan (ordered <-> disordered transition) ==")
scan = gauge.criticality_scan([0, 8, 16, 32, 64], 10, 10, 10, therm=25, seed=3)
print("   beta | mean_action | corr(R=1) | phase")
for b, act, cor in scan:
    phase = "ordered" if cor > 0.75 else ("critical" if cor > 0.55 else "disordered")
    print(f"   {b:4d} |   {act:6.1f}    |  {cor:.3f}   | {phase}")
acts = [a for _, a, _ in scan]
cors = [c for _, _, c in scan]
check("action decreases as beta rises (cold orders)", acts[-1] < acts[0] * 0.6)
check("correlation increases as beta rises", cors[-1] > cors[0] + 0.2)
check("hot end disordered, cold end ordered", cors[0] < 0.6 and cors[-1] > 0.7)
check("monotone-ish transition in correlation", all(cors[i] <= cors[i+1] + 0.05 for i in range(len(cors)-1)))

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
