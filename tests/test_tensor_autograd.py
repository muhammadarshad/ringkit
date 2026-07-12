"""Production tests for ringkit.ml.tensor_autograd (dual-ring autodiff over RingTensors, T4.5).
Cross-checks: (a) scalar autograd applied elementwise, (b) manual signed references. No float.
Run: python3 -m ringkit.tests.test_tensor_autograd"""
from ringkit.core import native as rn
from ringkit.ml.tensor_autograd import TVar
from ringkit.ml.autograd import Var          # scalar reference of record
from ringkit.array.tensor import RingTensor

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)
def raises(exc, f):
    try: f(); return False
    except exc: return True
    except Exception: return False

sgn = rn._signed

print("== elementwise mul: grad == signed(other) (loss = sum(a*b)) ==")
A = [[3, 100], [200, 7]]
B = [[9, 5], [128, 250]]
a = TVar(A); b = TVar(B)
loss = a.mul(b).sum()
loss.backward()
adata = a.val.data; bdata = b.val.data
check("d/da == signed(b) elementwise", all(a.grad[i] == sgn(bdata[i]) for i in range(4)))
check("d/db == signed(a) elementwise", all(b.grad[i] == sgn(adata[i]) for i in range(4)))
check("forward value == ring product", list(a.mul(b).val.data) == [rn.qsm(adata[i], bdata[i]) & 0xFF for i in range(4)])

print("== sum: grad broadcasts 1 ==")
a = TVar([[1, 2], [3, 4]]); s = a.sum(); s.backward()
check("all input grads == 1", all(g == 1 for g in a.grad))
check("scalar sum value correct", a.sum().val.data[0] == (1 + 2 + 3 + 4) & 0xFF)

print("== sin: grad == signed(COS(x)) (loss = sum(sin(a))) ==")
a = TVar([10, 64, 128, 200]); loss = a.sin().sum(); loss.backward()
check("d sin == signed(COS)", all(a.grad[i] == sgn(rn.COS(a.val.data[i])) for i in range(4)))

print("== elementwise graph matches SCALAR autograd, cell by cell ==")
# loss = sum( sin(a*b + a) )  -> compare each cell's grad to independent scalar Vars
Av = [5, 40, 130, 220]; Bv = [3, 9, 17, 240]
at = TVar(Av); bt = TVar(Bv)
lt = at.mul(bt).add(at).sin().sum(); lt.backward()
ga_ref, gb_ref = [], []
for i in range(4):
    sa = Var(Av[i]); sb = Var(Bv[i])
    sl = sa.mul(sb).add(sa).sin()
    sl.backward()
    ga_ref.append(sa.grad); gb_ref.append(sb.grad)
check("tensor grad_a == scalar grad_a per cell", list(at.grad) == ga_ref)
check("tensor grad_b == scalar grad_b per cell", list(bt.grad) == gb_ref)

print("== matmul backward vs manual signed reference (loss = sum(A@B)) ==")
# A: 2x3, B: 3x2
Am = [[1, 2, 3], [4, 5, 6]]; Bm = [[7, 8], [9, 10], [11, 12]]
at = TVar(Am); bt = TVar(Bm)
loss = at.matmul(bt).sum(); loss.backward()
ad = at.val.data; bd = bt.val.data
M, K, N = 2, 3, 2
# dA[m,k] = sum_n 1 * signed(B[k,n]) ; dB[k,n] = sum_m signed(A[m,k]) * 1
dA_ref = [sum(sgn(bd[k * N + n]) for n in range(N)) for m in range(M) for k in range(K)]
dB_ref = [sum(sgn(ad[m * K + k]) for m in range(M)) for k in range(K) for n in range(N)]
check("matmul dA == manual signed reference", list(at.grad) == dA_ref)
check("matmul dB == manual signed reference", list(bt.grad) == dB_ref)
# forward value equals RingTensor matmul
from ringkit.array.tensor import matmul as _mm
check("matmul forward == RingTensor matmul", list(at.matmul(bt).val.data) == list(_mm(RingTensor(Am), RingTensor(Bm)).data))

print("== errors ==")
check("shape mismatch add -> ValueError", raises(ValueError, lambda: TVar([1, 2]).add(TVar([1, 2, 3]))))
check("non-2D matmul -> ValueError", raises(ValueError, lambda: TVar([1, 2, 3]).matmul(TVar([1, 2, 3]))))
check("matmul inner-dim mismatch -> ValueError", raises(ValueError, lambda: TVar([[1, 2]]).matmul(TVar([[1, 2, 3]]))))

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
