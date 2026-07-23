"""Production test for ringkit.ml.grad — kernel-backed linear forward/backward (Q16 ENERGY).
D9: the device energy GEMM path reproduces the pure-Python ring reference bit-for-bit (dX,dW,db),
the finite-difference gradient check holds in ring units, and sigmoid_backward is exact at the peak.
Also exercises qcm.tensor.QSMLinear.forward_batch parity. Run: python3 -m ringkit.tests.test_grad"""
from ringkit.ml import grad as G
from ringkit.qcm import tensor as T

fails = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)


print("== ml.grad kernel linear forward/backward (D9 bit-exact + FD) ==")
check("grad._selftest (fwd==ref, bwd==ref, finite-diff, sigmoid')", G._selftest())

print("== qcm.tensor forward_batch parity (kernel == per-token qsm_matmul) ==")
check("tensor._selftest (incl. forward_batch bit-exact)", T._selftest())

print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
