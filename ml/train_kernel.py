"""ringkit.ml.train_kernel — end-to-end ring training through the KERNEL forward/backward (SPEC-013
T6.3, Path B). A 2-layer ring MLP (Linear → sigmoid → Linear) trained on jittered XOR — the smallest
task that NEEDS a nonlinearity — entirely on device energy GEMMs:

    forward:  h = σ(X·W1ᵀ + b1) ; logits = h·W2ᵀ + b2            (ml.grad.linear_forward, kernel)
    loss:     softmax cross-entropy                              (ml.loss.cross_entropy_batch)
    backward: dW2,db2,dh <- linear_backward ; dh_pre <- sigmoid_backward ; dW1,db1 <- linear_backward
    step:     AdamW (Q16, ENERGY, float-free)                    (ml.optim)

This is the keystone (K1) re-run on the kernel path: it closes the scale gap that made the pure-Python
keystone infeasible (the Python 1808-token linear was 9.2 ms/token; the batched energy GEMM is 256x
faster and bit-exact). Honesty bar per keystone discipline: held-out generalization + a FAILING
random-label control, multi-seed (ring descent is SEED-FLAKY — reported, not hidden). No float.
"""
import time
from ringkit.core import native as rn
from ringkit.device import default_device
from ringkit.ml.loss import cross_entropy_batch, FRAC, ONE
from ringkit.ml.optim import AdamW, clip_grad_norm
from ringkit.ml import grad as G


def _xor_data(n, seed):
    """Jittered XOR in Q16: y = (x0>0) XOR (x1>0). Deterministic LCG jitter (no random import,
    no float). Returns (X flat [n*2], Y [n])."""
    proto = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    ylut = [0, 1, 1, 0]
    X, Y, s = [], [], seed & 0x7FFFFFFF
    for i in range(n):
        c = i & 3
        row = []
        for k in range(2):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            jit = (s % 13) - 6                                  # -6..6 in 1/16 units
            row.append((proto[c][k] << FRAC) + (jit << (FRAC - 4)))
        X.extend(row); Y.append(ylut[c])
    return X, Y


def _rand_labels(n, seed):
    """Random 2-class labels (the failing control): a net that 'learns' these cannot generalize."""
    out, s = [], seed & 0x7FFFFFFF
    for _ in range(n):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        out.append((s >> 8) & 1)
    return out


def _forward(X, W1, b1, W2, b2, N, K0, Hd, O, dev):
    hpre = G.linear_forward(X, W1, b1, N, Hd, K0, dev)          # [N*Hd]
    h = G.sigmoid_forward(hpre, dev)                            # [N*Hd]
    logits = G.linear_forward(h, W2, b2, N, O, Hd, dev)         # [N*O]
    return hpre, h, logits


def train(seed=1, epochs=200, lr=3277, Hd=8, n_train=80, n_test=40, labels="xor", dev=None):
    dev = dev if dev is not None else default_device()
    K0, O = 2, 2
    Xtr, Ytr = _xor_data(n_train, seed)
    Xte, Yte = _xor_data(n_test, seed + 9973)
    if labels == "random":
        Ytr = _rand_labels(n_train, seed + 555)                # control: shuffle the mapping away
    # Q16 weights (small deterministic init, per-seed, no random/float)
    def init(size, sd):
        out, s = [], (sd & 0x7FFFFFFF) or 1
        for _ in range(size):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            out.append(((s % 21) - 10) << (FRAC - 4))          # ~[-0.6, 0.6] Q16
        return out
    W1 = init(Hd * K0, seed * 7 + 1); b1 = [0] * Hd
    W2 = init(O * Hd, seed * 13 + 3); b2 = [0] * O
    opt = AdamW([Hd * K0, Hd, O * Hd, O], lr=lr, wd=7)

    def accuracy(X, Y, n):
        _, _, lg = _forward(X, W1, b1, W2, b2, n, K0, Hd, O, dev)
        return sum(1 for i in range(n)
                   if (0 if lg[i * O] >= lg[i * O + 1] else 1) == Y[i]) / n

    first_loss = last_loss = None
    t0 = time.perf_counter()
    for ep in range(epochs):
        hpre, h, logits = _forward(Xtr, W1, b1, W2, b2, n_train, K0, Hd, O, dev)
        bl = [logits[i * O:(i + 1) * O] for i in range(n_train)]
        loss, grads = cross_entropy_batch(bl, Ytr)             # grads[i][o] = dL/dlogit (Q16, /B)
        dLogits = [g for row in grads for g in row]            # [n_train*O]
        # backward (kernel energy GEMMs)
        dh, dW2, db2 = G.linear_backward(dLogits, h, W2, n_train, O, Hd, dev)
        dhpre = G.sigmoid_backward(dh, h, dev)
        _, dW1, db1 = G.linear_backward(dhpre, Xtr, W1, n_train, Hd, K0, dev)
        clip_grad_norm([dW1, db1, dW2, db2], ONE)
        opt.step([W1, b1, W2, b2], [dW1, db1, dW2, db2])
        if ep == 0:
            first_loss = loss
        last_loss = loss
    dt = time.perf_counter() - t0
    return {
        "seed": seed, "labels": labels,
        "first_loss": first_loss / ONE, "last_loss": last_loss / ONE,
        "train_acc": accuracy(Xtr, Ytr, n_train), "test_acc": accuracy(Xte, Yte, n_test),
        "epochs_per_s": epochs / dt if dt > 0 else 0.0,
    }


def _selftest():
    dev = default_device()
    seeds = [1, 7, 42, 101, 2024]
    real = [train(seed=s, labels="xor", dev=dev) for s in seeds]
    ctrl = [train(seed=s, labels="random", dev=dev) for s in seeds]
    real_test = [r["test_acc"] for r in real]
    ctrl_test = [c["test_acc"] for c in ctrl]
    mean_real = sum(real_test) / len(real_test)
    mean_ctrl = sum(ctrl_test) / len(ctrl_test)
    best = max(real_test)
    dropped = all(r["last_loss"] < r["first_loss"] for r in real)
    eps = real[0]["epochs_per_s"]
    for r in real:
        print(f"  [xor    seed {r['seed']:4d}] loss {r['first_loss']:.3f}->{r['last_loss']:.3f}  "
              f"train {r['train_acc']*100:5.1f}%  held-out {r['test_acc']*100:5.1f}%")
    for c in ctrl:
        print(f"  [random seed {c['seed']:4d}] held-out {c['test_acc']*100:5.1f}%  (control: must be ~chance)")
    print(f"  MEAN held-out: real {mean_real*100:.1f}%  vs  control {mean_ctrl*100:.1f}%  (best real {best*100:.1f}%)")
    print(f"  throughput: {eps:.1f} epochs/s on {dev} (forward+backward+AdamW, all kernel energy GEMMs)")
    spread = max(real_test) - min(real_test)
    if spread <= 0.05:
        print(f"  NOTE: all seeds converged here (held-out {min(real_test)*100:.0f}-{max(real_test)*100:.0f}%); "
              f"the K1 seed-flakiness appeared on the harder teacher target, not this XOR probe.")
    else:
        print(f"  NOTE: ring descent is SEED-FLAKY (held-out range {min(real_test)*100:.0f}-"
              f"{max(real_test)*100:.0f}%) — reported, not hidden (matches keystone K1).")
    # honesty bar: loss falls every seed; real generalizes above the failing control by a clear margin
    gate = dropped and (mean_real > mean_ctrl + 0.12) and (best >= 0.85)
    print(f"  loss fell every seed: {'PASS' if dropped else 'FAIL'}")
    print(f"  real > control + margin & best>=85%: {'PASS' if (mean_real>mean_ctrl+0.12 and best>=0.85) else 'FAIL'}")
    return gate


if __name__ == "__main__":
    print("ringkit.ml.train_kernel — end-to-end ring training on the KERNEL path (SPEC-013 T6.3):")
    print("RESULT:", "ALL PASS" if _selftest() else "FAIL")
