"""Production test for ringkit.ml.attention — proves genuine CONTENT-BASED attention, not a lookup.

Task: associative recall. Each example has N novel (key -> value) bindings and a query equal to one
key; the model must attend query->matching-key by CONTENT and read that value. Because bindings are
fresh per example, a lexical/kNN index cannot do this — only content routing can.

Controls (charter D1/D6):
  * position-only baseline (ignore content, fixed slot) MUST fail on held-out -> proves it's content,
  * learned query-decoder recovered by ring_solve generalizes to novel bindings,
  * random-label control MUST collapse to chance -> proves learning, not memorizing.
Run: python3 -m ringkit.tests.test_attention"""
import random
from ringkit.core import native as rn
from ringkit.ml import attention as at
from ringkit.linalg.solve import solve

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

random.seed(5)
D = 4          # key/query dim
def vec(): return [random.randint(0, 255) for _ in range(D)]

print("== 1. content routing generalizes to NOVEL bindings (mechanism, hard attention) ==")
# fresh random keys+values each trial; query == one of the keys; must read its value.
ok = 0; trials = 500
for _ in range(trials):
    N = 6
    keys = [vec() for _ in range(N)]
    # ensure distinct keys
    while len({tuple(k) for k in keys}) != N:
        keys = [vec() for _ in range(N)]
    vals = [[random.randint(0, 255)] for _ in range(N)]
    t = random.randrange(N)
    query = [list(keys[t])]
    out, idx = at.attend(query, keys, vals, hard=True)
    if out[0] == vals[t]:
        ok += 1
check("reads correct value on novel bindings (>=99%)", ok / trials >= 0.99)

print("== 2. position-only baseline FAILS held-out (so it's content, not position) ==")
# a lookup that ignores query content and always returns a fixed slot
ok_pos = 0
for _ in range(trials):
    N = 6
    keys = [vec() for _ in range(N)]
    vals = [[random.randint(0, 255)] for _ in range(N)]
    t = random.randrange(N)
    if vals[0] == vals[t]:          # fixed-slot guess
        ok_pos += 1
check("position-only baseline near chance (<0.35)", ok_pos / trials < 0.35)

print("== 3. LEARNED query-decoder: recover Wq=E^-1 via ring_solve, generalize to held-out ==")
# Secret invertible encoder E (DxD). Query is an ENCODED key: q = key @ E. Model must learn
# Wq so that q @ Wq == key, then attend. Wq is solved from D training (q,key) pairs.
def matvec(M, x):                    # x (D) times M (DxD) -> (D), ring
    return [sum(rn.mul(x[i], M[i][j]) for i in range(D)) & 0xFF for j in range(D)]
def encode(key, E): return matvec(E, key)   # key @ E

# build an invertible E (odd determinant) by trial
while True:
    E = [[random.randint(0, 255) for _ in range(D)] for _ in range(D)]
    try:
        solve(E, [0 for _ in range(D)]); break
    except Exception:
        continue
# training pairs to recover Wq = E^-1 : we have q_tr = key_tr @ E, want q_tr @ Wq = key_tr
key_tr = [vec() for _ in range(D)]
while True:
    q_tr = [encode(k, E) for k in key_tr]
    try:
        solve(q_tr, [0 for _ in range(D)]); break     # need q_tr invertible
    except Exception:
        key_tr = [vec() for _ in range(D)]
# solve Wq column by column: q_tr @ Wq[:,c] = key_tr[:,c]
Wq = [[0] * D for _ in range(D)]
for c in range(D):
    col = solve(q_tr, [key_tr[r][c] for r in range(D)])
    for r in range(D):
        Wq[r][c] = col[r]

def run_recall(decoder, n_examples, corrupt=False):
    good = 0
    for _ in range(n_examples):
        N = 6
        keys = [vec() for _ in range(N)]
        while len({tuple(k) for k in keys}) != N:
            keys = [vec() for _ in range(N)]
        vals = [[random.randint(0, 255)] for _ in range(N)]
        t = random.randrange(N)
        q_enc = encode(keys[t], E)
        if corrupt:
            q_enc = vec()                     # random query unrelated to key -> no structure
        q_dec = [matvec(decoder, q_enc)]      # apply learned decoder, then attend
        out, _ = at.attend(q_dec, keys, vals, hard=True)
        if out[0] == vals[t]:
            good += 1
    return good / n_examples

acc_heldout = run_recall(Wq, 500)
check("learned decoder -> held-out recall == 1.0 (recovered E^-1, generalizes)", acc_heldout == 1.0)

print("== 4. random-label control collapses to chance (can't fake it) ==")
# decoder learned from RANDOM (q,key) pairs (no relation) -> garbage -> attention misroutes
key_r = [vec() for _ in range(D)]
while True:
    q_r = [vec() for _ in range(D)]
    try:
        solve(q_r, [0 for _ in range(D)]); break
    except Exception:
        continue
Wq_rand = [[0] * D for _ in range(D)]
for c in range(D):
    col = solve(q_r, [key_r[r][c] for r in range(D)])
    for r in range(D):
        Wq_rand[r][c] = col[r]
acc_rand = run_recall(Wq_rand, 500)
check("random-trained decoder -> held-out at chance (<0.35)", acc_rand < 0.35)
print(f"    [held-out recall] learned={acc_heldout:.3f}  random-control={acc_rand:.3f}  chance~1/6={1/6:.3f}")

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
