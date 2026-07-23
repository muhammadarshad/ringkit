"""ringkit.ml.loss — ring-native losses (our own F.cross_entropy), float-free (Q<frac> = 2-ring).

No torch, no numpy, no float. Fixed-point domain Q16 (ONE == 1.0). The workhorse is softmax
cross-entropy: its gradient is the clean closed form (softmax - onehot), so training needs no log
on the backward path; the loss VALUE uses a ring ln for monitoring only.

    loss, grad = cross_entropy(logits_q16, target)   # grad = d loss / d logit_j  (ENERGY, seed backward)
"""
from ringkit.core import native as rn

FRAC = 16
ONE = 1 << FRAC
LN2 = 45426                                       # round(ln2 * 2^16)


def _sd(n, d):
    if n >= 0:
        return rn.mf_floordiv(n + (d >> 1), d)
    return -rn.mf_floordiv((-n) + (d >> 1), d)


def exp_nonpos(x, frac=FRAC):
    """e^x for x<=0 in Q<frac> via Taylor. (0, ONE]."""
    one = 1 << frac
    if x > 0:
        raise ValueError("exp_nonpos: arg must be <= 0")
    if x <= -((frac + 3) << frac):
        return 0
    acc = one
    term = one
    for k in range(1, 40):
        term = _sd(rn.mul(term, x), (k << frac))     # term *= x/k  (x is Q<frac>)
        acc += term
        if term == 0:
            break
    return acc if acc > 0 else 0


def ln_fixed(x, frac=FRAC):
    """natural log of x>0 (Q<frac>) -> Q<frac>, signed. Range-reduce by powers of 2 to [ONE,2*ONE),
    then ln(m) = 2*atanh((m-1)/(m+1)). Float-free (bit_length shifts, isqrt-free)."""
    one = 1 << frac
    if x <= 0:
        raise ValueError("ln_fixed: x must be > 0")
    k = 0
    m = x
    while m >= (one << 1):
        m >>= 1
        k += 1
    while m < one:
        m <<= 1
        k -= 1
    # ln(m), m in [1,2): 2*atanh(w), w=(m-1)/(m+1) in [0,1/3)
    w = _sd((m - one) << frac, (m + one))
    w2 = rn.mul(w, w) >> frac
    term = w
    acc = w
    for j in (3, 5, 7, 9, 11, 13):
        term = rn.mul(term, w2) >> frac
        acc += _sd(term, j)
    return (acc << 1) + rn.mul(k, LN2)               # 2*atanh + k*ln2


def softmax(logits):
    """logits: Q16 ints -> Q16 probabilities (sum ~= ONE). Stable (subtract max)."""
    m = max(logits)
    ex = [exp_nonpos(l - m) for l in logits]         # each (0,ONE]
    tot = sum(ex) or 1
    return [_sd(e << FRAC, tot) for e in ex]


def cross_entropy(logits, target):
    """Softmax cross-entropy. logits: Q16 ints; target: class index.
    Returns (loss_q16, grad) where grad[j] = softmax[j] - onehot[j]  (Q16 ENERGY, seeds backward)."""
    p = softmax(logits)
    pt = p[target] if p[target] > 0 else 1
    loss = -ln_fixed(pt)                              # -log p_true, Q16 >= 0
    grad = [p[j] - (ONE if j == target else 0) for j in range(len(logits))]
    return loss, grad


def cross_entropy_batch(batch_logits, targets):
    """Mean CE over a batch. Returns (mean_loss_q16, [grad per sample], each averaged by 1/B)."""
    B = len(batch_logits)
    losses, grads = [], []
    for lg, t in zip(batch_logits, targets):
        l, g = cross_entropy(lg, t)
        losses.append(l)
        grads.append([_sd(gi, B) for gi in g])       # mean gradient
    return _sd(sum(losses), B), grads


def _selftest():
    ok = True
    n = 5
    # uniform logits -> softmax uniform (ONE/n each), loss = ln(n)
    p = softmax([0] * n)
    uni = all(abs(pi - ONE // n) <= 4 for pi in p)
    ok &= uni
    print(f"  softmax(uniform) ~= 1/n each: {'PASS' if uni else 'FAIL'} {p}")
    lnn = ln_fixed(n * ONE)
    loss, grad = cross_entropy([0] * n, 2)
    ce_uni = abs(loss - lnn) <= 40
    ok &= ce_uni
    print(f"  CE(uniform,t=2) = ln(5)={lnn}, got {loss}: {'PASS' if ce_uni else 'FAIL'}")
    # grad sums to ~0 (sum softmax - 1 = 0)
    gsum = sum(grad)
    gz = abs(gsum) <= 8
    ok &= gz
    print(f"  grad sums to ~0 (softmax-onehot): {'PASS' if gz else 'FAIL'} sum={gsum}")
    # loss decreases as the true-class logit grows
    l_lo, _ = cross_entropy([0, 0, 1 * ONE, 0, 0], 2)
    l_hi, _ = cross_entropy([0, 0, 5 * ONE, 0, 0], 2)
    mono = l_hi < l_lo
    ok &= mono
    print(f"  CE drops as target logit rises: {'PASS' if mono else 'FAIL'} ({l_lo} -> {l_hi})")
    # ln identity: ln(e) ~= 1  (e ~= 2.718*ONE)
    lne = ln_fixed(178145)                            # round(e*2^16)
    lne_ok = abs(lne - ONE) <= 40
    ok &= lne_ok
    print(f"  ln(e) ~= 1.0: {'PASS' if lne_ok else 'FAIL'} {lne}")
    return ok


if __name__ == "__main__":
    print("ringkit.ml.loss self-test (ring cross-entropy, float-free):")
    print("RESULT:", "ALL PASS" if _selftest() else "FAIL")
