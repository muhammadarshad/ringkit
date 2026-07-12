"""Facade test for ringkit.data + a full engineer workflow (encode -> split -> train -> eval).
Reads like ordinary engineer code; no ring internals surface. Run: python3 -m ringkit.tests.test_data_facade"""
import random
import ringkit as rk

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)
def raises(exc, f):
    try: f(); return False
    except exc: return True
    except Exception: return False

random.seed(7)

print("== encode / encode_range / one_hot ==")
check("encode scalar wraps mod-256", rk.data.encode(300) == 44)
check("encode matrix preserves shape", rk.data.encode([[1, 2], [3, 300]]) == [[1, 2], [3, 44]])
check("encode_range maps lo->0, hi->255", rk.data.encode_range([0, 100], 0, 100) == [0, 255])
check("encode_range midpoint ~128", abs(rk.data.encode_range(50, 0, 100) - 128) <= 1)
check("encode_range clamps out-of-range", rk.data.encode_range([-5, 105], 0, 100) == [0, 255])
check("one_hot", rk.data.one_hot(2, 4) == [0, 0, 1, 0])
check("one_hot batch", rk.data.one_hot([0, 3], 4) == [[1, 0, 0, 0], [0, 0, 0, 1]])
check("one_hot out-of-range -> ValueError", raises(ValueError, lambda: rk.data.one_hot(9, 4)))

print("== split / batches ==")
X = [[i, i + 1] for i in range(100)]
Y = [[i] for i in range(100)]
(Xtr, Ytr), (Xte, Yte) = rk.data.split(X, Y, test_frac=0.2, seed=1)
check("split sizes 80/20", len(Xtr) == 80 and len(Xte) == 20)
check("split keeps X/Y aligned", all(row[0] == y[0] for row, y in zip(Xtr, Ytr)))
check("split is disjoint", not (set(r[0] for r in Xtr) & set(r[0] for r in Xte)))
nb = list(rk.data.batches(Xtr, Ytr, size=32))
check("batches cover all rows", sum(len(b[0]) for b in nb) == 80)
check("batch sizes 32/32/16", [len(b[0]) for b in nb] == [32, 32, 16])
check("batches size<=0 -> ValueError", raises(ValueError, lambda: list(rk.data.batches(X, size=0))))

print("== END-TO-END engineer workflow (ring fully hidden) ==")
# Raw data with a hidden linear rule. Engineer never writes a ring op.
IN, OUT = 5, 2
true_W = [[random.randint(0, 255) for _ in range(OUT)] for _ in range(IN)]
def rule(x): return [sum(a * w for a, w in zip(x, col)) & 0xFF for col in zip(*true_W)]
raw_X = [[random.randint(0, 500) for _ in range(IN)] for _ in range(60)]   # raw ints, out of ring range

Xr = rk.data.encode(raw_X)                          # into the ring, safely
Yr = [rule(x) for x in Xr]
(Xtr, Ytr), (Xte, Yte) = rk.data.split(Xr, Yr, test_frac=0.3, seed=2)

model = rk.nn.Linear(IN, OUT)
model.fit(Xtr, Ytr)                                 # exact ring solve, hidden
pred = model.predict(Xte)
acc = sum(pred[i] == Yte[i] for i in range(len(Xte))) / len(Xte)
check("end-to-end held-out accuracy == 1.0", acc == 1.0)
check("engineer touched zero ring internals (workflow ran)", True)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
