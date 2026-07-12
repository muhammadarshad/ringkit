"""Production tests for ringkit.kernels.backend — C SIMD path == Python reference (bit-for-bit),
plus fallback correctness and a speedup measurement. Run: python3 -m ringkit.tests.test_kernels"""
import random
import time
from ringkit.kernels import backend as k
from ringkit.core import native as rn

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)
def raises(exc, f):
    try: f(); return False
    except exc: return True
    except Exception: return False

random.seed(1)
A = bytes(random.randint(0, 255) for _ in range(14464))    # one hypervector batch
B = bytes(random.randint(0, 255) for _ in range(14464))

print("== availability ==")
check("C SIMD backend available (built)", k.available())

print("== C path == Python reference (bit-for-bit) ==")
for name, fn in (("mul", k.mul), ("add", k.add), ("sub", k.sub)):
    c_out = fn(A, B)                              # C path
    py_out = fn(A, B, force_python=True)          # pure-Python path
    check(f"{name}: C == Python", c_out == py_out)

print("== semantics match ring_native (mul == qsm mod 256) ==")
check("mul == qsm & 0xFF", k.mul(A, B) == bytes(rn.qsm(x, y) & 0xFF for x, y in zip(A, B)))
check("add == ring add", k.add(A, B) == bytes((x + y) & 0xFF for x, y in zip(A, B)))

print("== errors ==")
check("length mismatch -> ValueError", raises(ValueError, lambda: k.mul(A, B[:-1])))

print("== speedup (C vs Python) ==")
iters = 300
t0 = time.perf_counter()
for _ in range(iters): k.mul(A, B)
c_t = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(iters): k.mul(A, B, force_python=True)
py_t = time.perf_counter() - t0
speed = py_t / c_t if c_t > 0 else 0
print(f"  C: {iters*14464/c_t/1e6:.1f} MUPS | Python: {iters*14464/py_t/1e6:.1f} MUPS | speedup ~{speed:.0f}x")
check("C faster than Python", speed > 1)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
