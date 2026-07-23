"""ringkit.ml.train_smoke — end-to-end proof that the RING-NATIVE training stack composes:
forward -> ml.loss.cross_entropy (our F.cross_entropy) -> backward (our loss.backward) ->
ml.optim.AdamW (our optim.AdamW) -> step (our opt.step). No torch, no numpy, no float.

Trains a linear softmax classifier on a deterministic linearly-separable toy set. If the loss falls
and train accuracy -> 100%, the four primitives work together. The transformer then reuses the SAME
loss+optim+step; only more per-op backward nodes are added (softplus/rmsnorm/ssd-scan/...).

Numeric domain: Q16 (value*2^16), ENERGY (no mod-256 fold) — the same domain ml.loss/ml.optim use.
The linear layer's backward is the analytic ring gradient (dW = g (x) x, db = g); this IS `backward`
for a Linear node — the general tape adds one such closure per op.
"""
from ringkit.core import native as rn
from ringkit.ml.loss import cross_entropy_batch, FRAC, ONE
from ringkit.ml.optim import AdamW, clip_grad_norm


def _lin_forward(W, b, x, O, K):
    """logits[o] = sum_i W[o*K+i]*x[i] + b[o], all Q16 (mul then >>FRAC)."""
    return [sum(rn.mul(W[o * K + i], x[i]) >> FRAC for i in range(K)) + b[o] for o in range(O)]


def _toy_data():
    """Deterministic separable 2-class set, 4-D. Class 0 ~ (+1,+1,-1,-1), class 1 ~ mirror. Q16 ints,
    a reproducible pseudo-jitter from a linear congruential walk (no float, no random import)."""
    proto = [[1, 1, -1, -1], [-1, -1, 1, 1]]
    X, Y = [], []
    seed = 12345
    for n in range(60):
        c = n & 1
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        row = []
        for k in range(4):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            jit = (seed % 13) - 6                      # -6..6 in 1/16 units
            row.append((proto[c][k] << FRAC) + (jit << (FRAC - 4)))   # Q16 feature + small jitter
        X.append(row); Y.append(c)
    return X, Y, 2, 4


def train(epochs=120, lr=3277):
    X, Y, O, K = _toy_data()
    N = len(X)
    W = [0] * (O * K)                                  # Q16 weights
    b = [0] * O
    opt = AdamW([O * K, O], lr=lr, wd=7)
    first_loss = last_loss = None
    last_acc = 0
    for ep in range(epochs):
        logits = [_lin_forward(W, b, X[n], O, K) for n in range(N)]
        loss, grads = cross_entropy_batch(logits, Y)   # grads[n][o] = dL/dlogit  (Q16, mean-scaled)
        # backward through the Linear: dW[o*K+i] = sum_n g[n][o]*x[n][i] ; db[o] = sum_n g[n][o]
        dW = [0] * (O * K)
        db = [0] * O
        for n in range(N):
            g = grads[n]
            xn = X[n]
            for o in range(O):
                db[o] += g[o]
                go = g[o]
                for i in range(K):
                    dW[o * K + i] += rn.mul(go, xn[i]) >> FRAC
        clip_grad_norm([dW, db], ONE)                  # match torch's clip_grad_norm_(..,1.0)
        opt.step([W, b], [dW, db])
        acc = sum(1 for n in range(N)
                  if (0 if logits[n][0] >= logits[n][1] else 1) == Y[n]) / N
        if ep == 0:
            first_loss = loss
        last_loss, last_acc = loss, acc
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d}  loss={loss/ONE:.4f}  train_acc={acc*100:.1f}%")
    return first_loss, last_loss, last_acc


def _selftest():
    fl, ll, acc = train()
    dropped = ll < fl
    learned = acc >= 0.95
    print(f"  loss fell ({fl/ONE:.4f} -> {ll/ONE:.4f}): {'PASS' if dropped else 'FAIL'}")
    print(f"  train accuracy >= 95% ({acc*100:.1f}%): {'PASS' if learned else 'FAIL'}")
    return dropped and learned


if __name__ == "__main__":
    print("ringkit.ml.train_smoke — end-to-end ring training (backward+CE+AdamW+step):")
    print("RESULT:", "ALL PASS" if _selftest() else "FAIL")
