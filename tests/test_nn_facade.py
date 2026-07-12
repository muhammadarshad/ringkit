"""Facade test for ringkit.nn — the engineer-facing model framework.
Written the way an ordinary engineer would use it: no mod-256, no energy/phase, no vacuums.
Keeps the honesty bar: structured task generalizes held-out; random labels collapse to chance.
Run: python3 -m ringkit.tests.test_nn_facade"""
import random
import ringkit as rk
from ringkit.linalg.solve import is_invertible

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

random.seed(21)

print("== engineer trains a Linear layer (ring hidden) ==")
IN, OUT = 6, 3
# a hidden 'true' rule the engineer doesn't know — they only have examples
true_W = [[random.randint(0, 255) for _ in range(OUT)] for _ in range(IN)]
def apply_true(x):
    return [sum(a * w for a, w in zip(x, col)) & 0xFF for col in zip(*true_W)]

X_train = [[random.randint(0, 255) for _ in range(IN)] for _ in range(40)]
Y_train = [apply_true(x) for x in X_train]

layer = rk.nn.Linear(in_features=IN, out_features=OUT)
layer.fit(X_train, Y_train)
check("fit reports exact training recovery", layer.train_exact is True)

# held-out: inputs never seen in training
X_test = [[random.randint(0, 255) for _ in range(IN)] for _ in range(1000)]
Y_test = [apply_true(x) for x in X_test]
pred = layer.predict(X_test)
acc = sum(pred[i] == Y_test[i] for i in range(len(X_test))) / len(X_test)
check("held-out generalization == 1.0 (learned the real rule)", acc == 1.0)

print("== random-label control MUST fail held-out (no faking) ==")
Y_rand = [[random.randint(0, 255) for _ in range(OUT)] for _ in range(40)]
lr = rk.nn.Linear(IN, OUT)
lr.fit(X_train, Y_rand)               # fit random labels (fits train, learns nothing real)
pr = lr.predict(X_test)
acc_r = sum(pr[i] == Y_test[i] for i in range(len(X_test))) / len(X_test)
check("random-label held-out at chance (< 0.05)", acc_r < 0.05)
print(f"    [held-out] structured={acc:.3f}  random-control={acc_r:.3f}")

print("== single-vector predict + Sequential compose ==")
one = layer.predict([1, 2, 3, 4, 5, 6])
check("single vector -> single output row", isinstance(one, list) and len(one) == OUT)
net = rk.nn.Sequential(rk.nn.Linear(IN, OUT))
net.layers[0].fit(X_train, Y_train)
check("Sequential predict matches inner layer", net.predict(X_test[:5]) == layer.predict(X_test[:5]))

print("== escape hatch exposes ring internals only when asked ==")
raw = layer.raw
check("raw has ring weight matrix", "W_ring" in raw and len(raw["W_ring"]) == IN)
check("raw weights are ring values 0..255", all(0 <= v <= 255 for rowv in raw["W_ring"] for v in rowv))

print("== content attention re-exported at framework level ==")
keys = [[10, 20], [30, 40], [200, 5]]
vals = [[111], [222], [77]]
out, who = rk.nn.attention([[30, 40]], keys, vals, hard=True)
check("attention routes to matching key by content", out[0] == [222] and who[0] == 1)

print("== under-determined fit gives a clear, honest error (not a silent wrong answer) ==")
def _raises():
    try:
        rk.nn.Linear(6, 1).fit([[1, 1, 1, 1, 1, 1]], [[5]]); return False
    except ValueError:
        return True
check("too few examples -> ValueError", _raises())

print("== Dense (nonlinear SIN, invert-then-solve) generalizes; random control fails ==")
DIN = 3
tW = [random.randint(0, 255) for _ in range(DIN)]
tb = random.randint(0, 255)
def teach(x): return rk.core.native.SIN((sum(a * w for a, w in zip(x, tW)) + tb) & 0xFF)
Xd = [[random.randint(0, 255) for _ in range(DIN)] for _ in range(12)]
Yd = [[teach(x)] for x in Xd]
dense = rk.nn.Dense(DIN, 1)
dense.fit(Xd, Yd)
check("Dense fits training exactly", dense.train_exact is True)
Xdt = [[random.randint(0, 255) for _ in range(DIN)] for _ in range(500)]
Ydt = [[teach(x)] for x in Xdt]
pd = dense.predict(Xdt)
accd = sum(pd[i] == Ydt[i] for i in range(len(Xdt))) / len(Xdt)
check("Dense held-out generalization == 1.0", accd == 1.0)

print("== RoPE: position-aware routing when CONTENT is ambiguous ==")
# all keys identical content -> only position can disambiguate; query must pick position p
same = [[7, 7] for _ in range(5)]
valsp = [[i * 10] for i in range(5)]
target_pos = 3
# query = the (identical) content, encoded at target position via RoPE-consistent shift
q_rope = [[(7 + target_pos) & 0xFF, (7 + target_pos) & 0xFF]]
keys_enc = rk.nn.positional_encode(same)          # keys carry their positions
outp, whop = rk.nn.attention(q_rope, keys_enc, valsp, hard=True)
check("RoPE routes to correct position under identical content", whop[0] == target_pos)
# without position info, identical content is ambiguous (routes to first, not target)
outn, whon = rk.nn.attention([[7, 7]], same, valsp, hard=True)
check("no-position baseline cannot select the position (whon != target)", whon[0] != target_pos)

print("== TransformerBlock composes end-to-end (self-attn + residual + FFN) ==")
blk = rk.nn.TransformerBlock(dim=2, rope=True)
seq = [[10, 20], [30, 40], [200, 5]]
res = blk(seq)
check("block runs, preserves shape (self-attn + residual)", len(res) == 3 and len(res[0]) == 2)
check("block.raw exposes attn+ffn internals", "attn" in blk.raw and "ffn" in blk.raw)

print("== Transformer.induction: in-context, generalizes to NOVEL tokens (content+position) ==")
tf = rk.nn.Transformer(key_dim=1, rope=True)
ok = 0; trials = 500
for _ in range(trials):
    # random vocab never 'seen' in any training — pure in-context
    a, b, c = random.sample(range(1, 200), 3)   # distinct tokens
    # sequence: ... a b ... c ... a  -> after previous 'a' came 'b' -> predict b
    seq = [a, b, c, a]
    pred, _ = tf.induction(seq)
    if pred == b:
        ok += 1
check("induction predicts the follower on unseen tokens (>=0.99)", ok / trials >= 0.99)

print("== induction needs POSITION: RoPE picks the MOST-RECENT occurrence, content-only doesn't ==")
# 'a' occurs twice, followed by different tokens; correct induction = most-recent follower
a, b1, b2, c = 50, 111, 222, 77
seq = [a, b1, c, a, b2, c, a]      # last a; previous a (pos3) was followed by b2; older a(pos0) by b1
pred_rope, pos_rope = tf.induction(seq, rope=True)
pred_none, pos_none = tf.induction(seq, rope=False)
check("RoPE picks most-recent follower (b2)", pred_rope == b2)
check("content-only picks the OLDER occurrence (b1, wrong for induction)", pred_none == b1)

print("== Transformer.recall: LEARNED decoder generalizes held-out; random control fails ==")
KD = 4
# secret encoder E; train decoder to invert it, then attention reads values in-context
while True:
    E = [[random.randint(0, 255) for _ in range(KD)] for _ in range(KD)]
    if is_invertible(E): break
def enc(k): return [sum(k[i] * E[i][j] for i in range(KD)) & 0xFF for j in range(KD)]
# training pairs (encoded_key -> true_key) to learn decoder = E^-1
ktr = [[random.randint(0, 255) for _ in range(KD)] for _ in range(KD)]
while not is_invertible([enc(k) for k in ktr]):
    ktr = [[random.randint(0, 255) for _ in range(KD)] for _ in range(KD)]
tf2 = rk.nn.Transformer(key_dim=KD)
tf2.fit([enc(k) for k in ktr], ktr)

def recall_acc(model, corrupt=False, n=400):
    good = 0
    for _ in range(n):
        N = 6
        keys = [[random.randint(0, 255) for _ in range(KD)] for _ in range(N)]
        while len({tuple(k) for k in keys}) != N:
            keys = [[random.randint(0, 255) for _ in range(KD)] for _ in range(N)]
        vals = [[random.randint(0, 255)] for _ in range(N)]
        t = random.randrange(N)
        qenc = [random.randint(0, 255) for _ in range(KD)] if corrupt else enc(keys[t])
        out, _ = model.recall(keys, vals, qenc)
        if out == vals[t]:
            good += 1
    return good / n
acc_ok = recall_acc(tf2)
# random-trained decoder control
tf_r = rk.nn.Transformer(key_dim=KD)
kr = [[random.randint(0, 255) for _ in range(KD)] for _ in range(KD)]
while not is_invertible(kr):
    kr = [[random.randint(0, 255) for _ in range(KD)] for _ in range(KD)]
tf_r.fit(kr, [[random.randint(0, 255) for _ in range(KD)] for _ in range(KD)])
acc_rand = recall_acc(tf_r)
check("learned decoder -> held-out recall == 1.0", acc_ok == 1.0)
check("random-trained decoder -> chance (<0.35)", acc_rand < 0.35)
print(f"    [recall held-out] learned={acc_ok:.3f}  random={acc_rand:.3f}")

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
