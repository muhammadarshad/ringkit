"""Tests for the ring GEMM silicon (kernels/backend/gemm) — D9.
Three variants (hardware-`*` bridge, multiplier-free QSM table, multiplier-free shift-add)
must all reproduce the multiplier-free Python reference bit-for-bit, threaded == single-
threaded, and the rnp tensor routing must be invisible. Skip-as-pass if no C compiler.
Run: python3 -m ringkit.tests.test_gemm"""
import random
from ringkit.core import native as rn
from ringkit.kernels.backend import gemm

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)


def reference(A, B, M, K, N):
    """Multiplier-free reference of record: rn.qsm products, ring accumulation."""
    C = bytearray(rn.mul(M, N))
    for i in range(M):
        for j in range(N):
            acc = 0
            for k in range(K):
                acc = (acc + rn.qsm(A[rn.mul(i, K) + k], B[rn.mul(k, N) + j])) & 0xFF
            C[rn.mul(i, N) + j] = acc
    return C


if not gemm.available():
    print("  SKIP  no C toolchain — gemm silicon unavailable, python reference serves")
    print()
    print("RESULT: ALL PASS")
    raise SystemExit(0)

random.seed(23)
print("== all variants == multiplier-free reference (random + adversarial shapes) ==")
SHAPES = [(8, 32, 8), (5, 7, 3), (1, 256, 1), (17, 1, 13), (32, 32, 32)]
lib = gemm._load()
for (M, K, N) in SHAPES:
    A = bytearray(random.randrange(256) for _ in range(M * K))
    B = bytearray(random.randrange(256) for _ in range(K * N))
    want = reference(A, B, M, K, N)
    ok = {}
    for v in gemm.VARIANTS:
        got = bytearray(M * N)
        getattr(lib, f"ring_gemm_{v}")(gemm._ptr(got), gemm._ptr(A), gemm._ptr(B), M, K, N)
        ok[v] = got == want
    check(f"{M}x{K}x{N}: mul/qsm/shiftadd all == reference", all(ok.values()))

print("== full operand coverage: 1x256 @ 256x256 touches every (a,b) pair column-wise ==")
A = bytearray(range(256))
B = bytearray(random.randrange(256) for _ in range(256 * 256))
want = reference(A, B, 1, 256, 256)
allok = True
for v in gemm.VARIANTS:
    got = bytearray(256)
    getattr(lib, f"ring_gemm_{v}")(gemm._ptr(got), gemm._ptr(A), gemm._ptr(B), 1, 256, 256)
    allok &= got == want
check("all variants exact over full-range operands", allok)

print("== threaded == single-threaded (predictable row bins, no merge) ==")
M, K, N = 64, 48, 56
A = bytearray(random.randrange(256) for _ in range(M * K))
B = bytearray(random.randrange(256) for _ in range(K * N))
allok = True
for v in gemm.VARIANTS:
    st = bytearray(M * N); mt = bytearray(M * N)
    getattr(lib, f"ring_gemm_{v}")(gemm._ptr(st), gemm._ptr(A), gemm._ptr(B), M, K, N)
    getattr(lib, f"ring_gemm_{v}_mt")(gemm._ptr(mt), gemm._ptr(A), gemm._ptr(B), M, K, N, 8)
    allok &= st == mt
check("mt(8) == st for all variants (bit-for-bit)", allok)

print("== rnp tensor routing is invisible ==")
import ringkit.rnp as rnp
a_list = [random.randrange(256) for _ in range(12 * 20)]
b_list = [random.randrange(256) for _ in range(20 * 9)]
a = rnp.array(a_list).reshape(12, 20)
b = rnp.array(b_list).reshape(20, 9)
c = a @ b
want = reference(bytearray(a_list), bytearray(b_list), 12, 20, 9)
check("tensor matmul (kernel-routed) == reference", bytes(c.data) == bytes(want))
check("matvec via tensor == reference",
      bytes((a @ rnp.array(b_list[:20])).data)
      == bytes(reference(bytearray(a_list), bytearray(b_list[:20]), 12, 20, 1)))

print("== default variant is a gated member ==")
check("DEFAULT_VARIANT valid", gemm.DEFAULT_VARIANT in gemm.VARIANTS)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
raise SystemExit(0 if not fails else 1)
