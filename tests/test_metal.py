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

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
raise SystemExit(0 if not fails else 1)
