"""Stacked multi-block trained model — the honesty bar in full.
Task: TWO-HOP recall. Memory 1 binds k->v; memory 2 binds v->w (keys stored in a second,
different encoded space). Answering w from an enc1-encoded query REQUIRES two trained hops:
hop 1 decodes enc1(k) into key space and retrieves v; hop 2 maps v into memory 2's encoded
key space and retrieves w. Every hop is trained by EXACT SOLVE on pairs never used at test.

Held-out: test bindings and queries are novel. Controls that must FAIL:
  depth control  — a 1-block model (no trained second hop) collapses to chance;
  random control — a second hop trained on shuffled targets collapses to chance.
Run: python3 -m ringkit.tests.test_stacked"""
import random
import ringkit as rk

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

random.seed(41)
DIM = 2

def rand_invertible(dim):
    """Random matrix invertible mod 256 (odd determinant), as row-major lists."""
    from ringkit.linalg.solve import is_invertible
    while True:
        M = [[random.randrange(256) for _ in range(dim)] for _ in range(dim)]
        if is_invertible(M):
            return M

def apply(M, x):
    return [sum(M[i][j] * x[j] for j in range(len(x))) & 0xFF for i in range(len(M))]  # oracle math (test layer)

ENC1 = rand_invertible(DIM)          # the query encoding the first hop must learn to undo
ENC2 = rand_invertible(DIM)          # the key-space transform the second hop must learn to apply

def fresh_vec(seen):
    while True:
        v = tuple(random.randrange(256) for _ in range(DIM))
        if v not in seen:
            seen.add(v)
            return list(v)

print("== train both hops by exact solve (training pairs only) ==")
seen = set()
train_ks = [fresh_vec(seen) for _ in range(10)]
train_vs = [fresh_vec(seen) for _ in range(10)]
model = rk.nn.Stacked(blocks=2, dim=DIM)
model.fit([
    ([apply(ENC1, k) for k in train_ks], train_ks),        # hop1: enc1(k) -> k
    (train_vs, [apply(ENC2, v) for v in train_vs]),        # hop2: v -> enc2(v)
])
check("both hops fit exactly on training pairs (solve, not descent)", model.train_exact)

print("== held-out generalization: NOVEL bindings, NOVEL queries ==")
B = 16
ks = [fresh_vec(seen) for _ in range(B)]
vs = [fresh_vec(seen) for _ in range(B)]
ws = [fresh_vec(seen) for _ in range(B)]
mem1 = (ks, vs)                                            # k -> v
mem2 = ([apply(ENC2, v) for v in vs], ws)                  # enc2(v) -> w

hits = 0
for t in range(B):
    got, path = model.recall([mem1, mem2], apply(ENC1, ks[t]))
    hits += got == ws[t] and path == [t, t]
acc = hits / B
print(f"    two-hop held-out accuracy: {acc:.2f}")
check("held-out two-hop recall == 1.0 (exact, through both trained hops)", acc == 1.0)

print("== depth control: a 1-block model MUST fail the two-hop task ==")
shallow_hits = 0
for t in range(B):
    v_hat, _ = model.blocks[0](apply(ENC1, ks[t]), *mem1)  # hop 1 only
    got, _ = rk.nn.attention([v_hat], *[list(x) for x in mem2], hard=True)  # raw attend, no trained hop 2
    shallow_hits += got[0] == ws[t]
sacc = shallow_hits / B
print(f"    1-block accuracy on 2-hop task: {sacc:.2f}")
check("depth control fails (< 0.35)", sacc < 0.35)

print("== random control: hop 2 trained on SHUFFLED targets MUST fail ==")
shuffled = [apply(ENC2, v) for v in train_vs]
random.shuffle(shuffled)
bad = rk.nn.Stacked(blocks=2, dim=DIM)
try:
    bad.fit([([apply(ENC1, k) for k in train_ks], train_ks), (train_vs, shuffled)])
    bad_hits = 0
    for t in range(B):
        got, _ = bad.recall([mem1, mem2], apply(ENC1, ks[t]))
        bad_hits += got == ws[t]
    bacc = bad_hits / B
    print(f"    random-trained hop-2 accuracy: {bacc:.2f}")
    check("random control fails (< 0.35)", bacc < 0.35)
except ValueError:
    print("    (shuffled targets not even solvable -> fit refused: also a failing control)")
    check("random control fails (fit refused)", True)

print("== stacked TransformerBlocks compose (shape-preserving smoke) ==")
seq = [[random.randrange(256) for _ in range(4)] for _ in range(6)]
stack = rk.nn.Sequential(rk.nn.TransformerBlock(4), rk.nn.TransformerBlock(4))
out = stack(seq)
check("2-block Sequential preserves (rows, dim)", len(out) == 6 and len(out[0]) == 4)
check("Stacked.raw exposes per-block internals", len(rk.nn.Stacked(3, DIM).raw["blocks"]) == 3)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
raise SystemExit(0 if not fails else 1)
