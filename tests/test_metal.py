"""Tests for the Metal ring-ops backend (kernels/apple/metal) — D9 silicon.
Bit-for-bit vs the Python reference over the FULL 256x256 operand table, and vs the
(independently proven) C path over 1M random elements. Perf is RECORDED, not asserted:
measured on M1 Pro, C SIMD wins elementwise at every size (copies dominate), so the
registry keeps Metal opt-in — see backend.METAL_MIN.
Skip-as-pass with a printed reason when no Metal device/toolchain exists (CI, Linux).
Run: python3 -m ringkit.tests.test_metal"""
import os
import time
from ringkit.kernels import backend
from ringkit.kernels.apple.metal import host

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)


if not host.available():
    print("  SKIP  no Metal device/toolchain here — backend correctly reports unavailable")
    check("registry falls through cleanly", backend.backends()["metal"] == "unavailable"
          and backend.active(1 << 24) in ("cpu-c", "python"))
    print()
    print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
    raise SystemExit(0 if not fails else 1)

print(f"== device: {host.device_name()} ==")

print("== bit-for-bit vs Python reference: FULL 256x256 operand table ==")
N = 65536
a = bytearray(N); b = bytearray(N)
for x in range(256):
    for y in range(256):
        a[(x << 8) | y] = x
        b[(x << 8) | y] = y
PY = {"ring_mul": lambda x, y: (x * y) & 0xFF,          # labeled oracle (test layer)
      "ring_add": lambda x, y: (x + y) & 0xFF,
      "ring_sub": lambda x, y: (x - y) & 0xFF}
for op, f in PY.items():
    out = bytearray(N)
    rc = host.elementwise(op, out, a, b, N)
    want = bytearray(f(a[i], b[i]) for i in range(N))
    check(f"{op}: rc==0 and exhaustive table exact", rc == 0 and out == want)

print("== bit-for-bit vs C SIMD path: 1M random ==")
n = 1 << 20
ra = bytearray(os.urandom(n)); rb = bytearray(os.urandom(n))
lib = backend._load()
for op in PY:
    mo = bytearray(n); co = bytearray(n)
    host.elementwise(op, mo, ra, rb, n)
    getattr(lib, op)(backend._ptr(co), backend._ptr(ra), backend._ptr(rb), n)
    check(f"{op}: metal == C over 1M random", mo == co)

print("== registry policy ==")
check("metal registered and self-tested", backend.backends()["metal"] == "serving")
check("auto-routing OFF (measured slower): big buffers stay on cpu-c",
      backend.METAL_MIN is None and backend.active(1 << 24) == "cpu-c")

print("== perf (recorded, not asserted — see backend.METAL_MIN rationale) ==")
out = bytearray(n)
t0 = time.perf_counter(); host.elementwise("ring_mul", out, ra, rb, n); tm = time.perf_counter() - t0
t0 = time.perf_counter(); lib.ring_mul(backend._ptr(out), backend._ptr(ra), backend._ptr(rb), n); tc = time.perf_counter() - t0
print(f"    1M mul: metal {n/tm/1e6:.0f} MUPS | cpu-c {n/tc/1e6:.0f} MUPS")

print("== gauge on GPU: bit-for-bit vs C and Python reference (24^3) ==")
import random
from ringkit.kernels.mprc.lattice import host as lat
random.seed(11)
W = H = D = 24
gn = W * H * D
g = bytearray(random.randrange(256) for _ in range(gn))
e_m = bytearray(gn)
check("plaquette: metal == C == python",
      host.plaquette(e_m, g, W, H, D) == 0
      and e_m == lat.plaquette(g, W, H, D)
      and bytes(e_m) == bytes(lat.plaquette(g, W, H, D, force_python=True)))
prop = bytearray(random.randrange(256) for _ in range(gn))
chance = bytearray(random.randrange(256) for _ in range(gn))
lut = bytearray(max(0, 255 - d) for d in range(256))
gm, gc, gp = bytearray(g), bytearray(g), bytearray(g)
host.gauge_sweep(gm, prop, chance, lut, W, H, D)
lat.sweep(gc, prop, chance, lut, W, H, D)
lat.sweep(gp, prop, chance, lut, W, H, D, force_python=True)
check("full sweep (both parities): metal == C == python", gm == gc == gp)

print("== gauge routing: measured crossover policy ==")
check("metal passes the lattice-host self-test gate", lat._metal_gauge_ready())
check("floor is 32^3 (re-measured vs THREADED C: metal still wins every size >= 24^3)",
      lat.GAUGE_METAL_MIN_NODES == 1 << 15)
W2 = H2 = D2 = 32
n2 = W2 * H2 * D2
g2 = bytearray(random.randrange(256) for _ in range(n2))
p2 = bytearray(random.randrange(256) for _ in range(n2))
c2 = bytearray(random.randrange(256) for _ in range(n2))
auto, forced = bytearray(g2), bytearray(g2)
lat.sweep(auto, p2, c2, lut, W2, H2, D2)              # auto-routes to metal at 32^3
lib2 = lat._load()
for par in (0, 1):
    lib2.metropolis_sweep(lat._ptr(forced), lat._ptr(bytearray(p2)), lat._ptr(bytearray(c2)),
                          lat._ptr(bytearray(lut)), W2, H2, D2, par)
check("auto-routed sweep result == forced-C result (routing is invisible)", auto == forced)

print("== fused GPU-resident thermalize (unified memory): batch == sequential ==")
S = 3
props = bytearray(random.randbytes(n2 * S))
chances = bytearray(random.randbytes(n2 * S))
fused, seq = bytearray(g2), bytearray(g2)
check("fused rc == 0", host.thermalize(fused, props, chances, lut, W2, H2, D2, S) == 0)
for s in range(S):
    for par in (0, 1):
        lib2.metropolis_sweep(lat._ptr(seq), lat._ptr(bytearray(props[s*n2:(s+1)*n2])),
                              lat._ptr(bytearray(chances[s*n2:(s+1)*n2])),
                              lat._ptr(bytearray(lut)), W2, H2, D2, par)
check("fused batch (3 sweeps, 1 bus round-trip) == sequential C", fused == seq)

print("== derived counter RNG (rk_mix32): python reference == C == metal ==")
Wr = Hr = Dr = 12
nr = Wr * Hr * Dr
g0 = bytearray(random.randbytes(nr))
gp = bytearray(g0)
for s in range(3):
    lat._py_sweep_rng(gp, 777, s, lut, Wr, Hr, Dr, 0)
    lat._py_sweep_rng(gp, 777, s, lut, Wr, Hr, Dr, 1)
gc3 = bytearray(g0)
for s in range(3):
    for par in (0, 1):
        lib2.metropolis_sweep_rng(lat._ptr(gc3), 777, s, lat._ptr(bytearray(lut)),
                                  Wr, Hr, Dr, par)
gm3 = bytearray(g0)
check("metal thermalize_rng rc == 0", host.thermalize_rng(gm3, 777, 0, lut, Wr, Hr, Dr, 3) == 0)
check("python reference == C (rng path)", gp == gc3)
check("C == metal (rng path)", gc3 == gm3)
check("rng gate passes", lat._metal_rng_ready())

print("== facade determinism: same seed -> same physics across routing paths ==")
import ringkit as rk
gA = rk.physics.Gauge(size=(32, 32, 32), beta=40, seed=9)
saved = lat.GAUGE_METAL_MIN_NODES
lat.GAUGE_METAL_MIN_NODES = 1 << 62               # force the C path
gA.thermalize(sweeps=4)
lat.GAUGE_METAL_MIN_NODES = saved
gB = rk.physics.Gauge(size=(32, 32, 32), beta=40, seed=9)
gB.thermalize(sweeps=4)                            # auto-routes to metal
check("C-routed and GPU-routed runs produce identical grids", gA.grid == gB.grid)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
raise SystemExit(0 if not fails else 1)
