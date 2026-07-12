"""Production tests for ringkit.linalg (solve + fit). Run: python3 -m ringkit.tests.test_linalg"""
import random
from ringkit.core import native as rn
from ringkit.linalg import solve as sv
from ringkit.linalg import fit as ft
from ringkit.physics import measure as _m  # (unused; ensures package import health)

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)
def raises(exc, f):
    try: f(); return False
    except exc: return True
    except Exception: return False

def matvec(A, x):
    return [sum(rn.mul(A[i][k], x[k]) for k in range(len(x))) & 0xFF for i in range(len(A))]

print("== modinv ==")
check("all 128 odd invertible", all(rn.mul(a, sv.modinv(a)) & 0xFF == 1 for a in range(1, 256, 2)))
check("modinv even -> ValueError", raises(ValueError, lambda: sv.modinv(4)))

print("== solve ==")
check("known [3,5,7]", sv.solve([[5,9,1],[30,12,1],[7,20,1]], [67,157,128]) == [3, 5, 7])
random.seed(11); exact = skipped = 0
for _ in range(3000):
    n = random.choice([2, 3, 4])
    A = [[random.randint(0,255) for _ in range(n)] for _ in range(n)]
    xt = [random.randint(0,255) for _ in range(n)]
    b = matvec(A, xt)
    if not sv.is_invertible(A):
        skipped += 1; continue
    sol = sv.solve(A, b)
    if matvec(A, sol) == b: exact += 1
check(f"random invertible systems: A@sol==b exact ({exact} solved, {skipped} singular skipped)", exact > 0 and skipped >= 0 and exact + skipped == 3000)
check("singular (even col) -> ValueError", raises(ValueError, lambda: sv.solve([[2,2],[4,4]], [0,0])))
check("is_invertible True/False", sv.is_invertible([[1,0],[0,1]]) and not sv.is_invertible([[2,2],[4,4]]))
check("non-square -> ValueError", raises(ValueError, lambda: sv.solve([[1,2,3],[4,5,6]], [1,2])))
check("b mismatch -> ValueError", raises(ValueError, lambda: sv.solve([[1,0],[0,1]], [1,2,3])))
check("empty -> ValueError", raises(ValueError, lambda: sv.solve([], [])))

print("== fit (invert-then-solve, exact nonlinear) ==")
# teacher-generated SIN fit (solution exists)
def neuron(W, b, x): return rn.SIN((rn.mul(W[0],x[0]) + rn.mul(W[1],x[1]) + b) & 0xFF)
data = [[5,9],[30,12],[100,60],[7,200],[64,64]]
tgt = [neuron([40,90], 17, x) for x in data]
sol = ft.fit(data, tgt)
check("5-point SIN fit -> exact (loss 0)", sol is not None and all(neuron(sol[:2], sol[2], x) == t for x, t in zip(data, tgt)))

random.seed(5); okc = none_ = 0
for _ in range(120):
    W = [random.randint(0,255), random.randint(0,255)]; bb = random.randint(0,255)
    D = [[random.randint(1,255), random.randint(1,255)] for _ in range(6)]
    T = [neuron(W, bb, x) for x in D]
    s = ft.fit(D, T)
    if s is None: none_ += 1
    elif all(neuron(s[:2], s[2], x) == t for x, t in zip(D, T)): okc += 1
check(f"random SIN-fits exact-or-honestly-None ({okc} exact, {none_} under-determined, sum={okc+none_}/120)", okc + none_ == 120 and okc > 100)

print("== fit errors / edge ==")
check("empty data -> ValueError", raises(ValueError, lambda: ft.fit([], [])))
check("ragged rows -> ValueError", raises(ValueError, lambda: ft.fit([[1,2],[3]], [0,0])))
check("targets mismatch -> ValueError", raises(ValueError, lambda: ft.fit([[1,2],[3,4],[5,6]], [0,0])))
check("too few points -> ValueError", raises(ValueError, lambda: ft.fit([[1,2],[3,4]], [0,0])))
check("unsatisfiable target -> None", ft.fit([[1,2],[3,4],[5,6]], [200,200,200], preimages=lambda t: []) is None)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
