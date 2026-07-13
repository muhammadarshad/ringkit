"""Test: a COMPLETE transformer block runs in ring fixed-point, float-free, matching a float oracle.

Block: RMSNorm -> attention -> residual -> RMSNorm -> GELU-MLP -> residual. Every op is ring integer
(infer.linear/attention/softmax, ract.rmsnorm/gelu) — no float on the compute path. numpy is the
labeled oracle only (C6/D9), computing the IDENTICAL functional forms so the comparison isolates the
fixed-point resolution, not modelling differences. This is the 'loaded model is workable' milestone
at block granularity, and the Gemma-ready attention/softmax path.
Run: python3 -m ringkit.tests.test_infer_block"""
import math
import numpy as np
from ringkit.emulation import infer, ract

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

FRAC = 16; ONE = 1 << FRAC
def q(v): return int(round(float(v) * ONE))
def dq(v): return v / ONE
def qvec(a): return [q(v) for v in a]
def qmat_flat(M): return [q(v) for row in M for v in row]   # row-major flat

np.random.seed(0)

print("== 1. softmax / inv_sqrt / attention vs float oracle ==")
sc = [0.3, -1.2, 2.1, 0.0, 0.7]
w_ring = [dq(v) for v in infer.softmax(qvec(sc), FRAC)]
e = np.exp(np.array(sc) - max(sc)); w_or = e / e.sum()
check(f"softmax max err {np.abs(np.array(w_ring)-w_or).max():.2e} < 5e-3", np.abs(np.array(w_ring)-w_or).max() < 5e-3)
check("inv_sqrt(64) ~ 0.125", abs(dq(infer.inv_sqrt(64, FRAC)) - 0.125) < 1e-3)

d = 8
Q = np.random.randn(3, d) * 0.5; K = np.random.randn(4, d) * 0.5; V = np.random.randn(4, d) * 0.5
scale = 1.0 / math.sqrt(d)
A_ring = np.array([[dq(v) for v in row] for row in
                   infer.attention([qvec(r) for r in Q], [qvec(r) for r in K], [qvec(r) for r in V],
                                   FRAC, scale=infer.inv_sqrt(d, FRAC))])
# oracle
S = (Q @ K.T) * scale
W = np.exp(S - S.max(1, keepdims=True)); W = W / W.sum(1, keepdims=True)
A_or = W @ V
check(f"attention max err {np.abs(A_ring-A_or).max():.2e} < 2e-2", np.abs(A_ring - A_or).max() < 2e-2)

print("== 2. FULL block: ring fixed-point vs float oracle (identical forms) ==")
L, ff = 4, 32
X = np.random.randn(L, d) * 0.5
Wq, Wk, Wv, Wo = [np.random.randn(d, d) * 0.2 for _ in range(4)]
W1 = np.random.randn(ff, d) * 0.2; W2 = np.random.randn(d, ff) * 0.2
ln1 = np.abs(np.random.randn(d)) * 0.3 + 1.0; ln2 = np.abs(np.random.randn(d)) * 0.3 + 1.0
Z = [0 for _ in range(max(d, ff))]

def sig(z): return 1.0 / (1.0 + np.exp(-z))
def gelu_o(z): return z * sig(1.702 * z)
def rms_o(x, w): return x / math.sqrt((x * x).mean() + dq(1)) * w

# --- oracle block (float) ---
Xn = np.stack([rms_o(X[i], ln1) for i in range(L)])
Qo, Ko, Vo = Xn @ Wq.T, Xn @ Wk.T, Xn @ Wv.T
S = (Qo @ Ko.T) * (1.0 / math.sqrt(d)); Wt = np.exp(S - S.max(1, keepdims=True)); Wt /= Wt.sum(1, keepdims=True)
X2 = X + (Wt @ Vo) @ Wo.T
Xn2 = np.stack([rms_o(X2[i], ln2) for i in range(L)])
Bo = X2 + gelu_o(Xn2 @ W1.T) @ W2.T

# --- ring block (fixed-point, float-free) ---
def rl(rows, Wf, out, inn): return [infer.linear(r, Wf, Z[:out], out, inn, FRAC) for r in rows]
Xq = [qvec(X[i]) for i in range(L)]
Xnq = [ract.rmsnorm_fixed(Xq[i], qvec(ln1), FRAC) for i in range(L)]
Qq, Kq, Vq = rl(Xnq, qmat_flat(Wq), d, d), rl(Xnq, qmat_flat(Wk), d, d), rl(Xnq, qmat_flat(Wv), d, d)
Aq = infer.attention(Qq, Kq, Vq, FRAC, scale=infer.inv_sqrt(d, FRAC))
AOq = rl(Aq, qmat_flat(Wo), d, d)
X2q = [[Xq[i][j] + AOq[i][j] for j in range(d)] for i in range(L)]
Xn2q = [ract.rmsnorm_fixed(X2q[i], qvec(ln2), FRAC) for i in range(L)]
H1q = rl(Xn2q, qmat_flat(W1), ff, d)
Gq = [[ract.gelu_fixed(v, FRAC) for v in row] for row in H1q]
H2q = rl(Gq, qmat_flat(W2), d, ff)
Bq = np.array([[dq(X2q[i][j] + H2q[i][j]) for j in range(d)] for i in range(L)])

err = np.abs(Bq - Bo).max()
check(f"FULL block max abs err {err:.2e} < 6e-2 (accumulated fixed-point over the whole block)", err < 6e-2)
print(f"    block output range [{Bo.min():.3f}, {Bo.max():.3f}]; ring reproduced float within {err:.2e}")

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
