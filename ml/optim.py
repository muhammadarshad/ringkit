"""
ring_optim.py — ring-native optimizer (SRD Phase 5), on ring_native + ring_autograd.

Finding from the T5.2 spike: the update rule is NOT the bottleneck — once gradients are
ENERGY (non-wrapping, from ring_autograd), a plain sign step converges from 100% of starts.
The parameter is an ARC (ring position, wraps mod 256); the gradient is ENERGY (signed).
Step against the gradient's sign. Multiplier-free.
"""
from ringkit.core import native as rn


def sign(g):
    """sign of an ENERGY gradient: -1 / 0 / +1."""
    if g > 0:
        return 1
    if g < 0:
        return -1
    return 0


def sgd_step(param_val, grad, lr=1):
    """One sign-SGD step on an ARC parameter: move lr steps against the gradient, wrap mod 256."""
    s = sign(grad)
    return (param_val - rn.mul(lr, s)) & 0xFF


FRAC = 16
ONE = 1 << FRAC


def _sd(n, d):
    """round(n/d), signed, d>0. Float-free."""
    if d <= 0:
        d = 1
    if n >= 0:
        return rn.mf_floordiv(n + (d >> 1), d)
    return -rn.mf_floordiv((-n) + (d >> 1), d)


def clip_grad_norm(grads, max_norm=ONE):
    """Ring grad clipping (our nn.utils.clip_grad_norm_): if the global L2 norm of all grads exceeds
    max_norm (Q16), scale every grad by max_norm/norm. Float-free (isqrt, mf_floordiv). In place."""
    ss = 0
    for g in grads:
        for gi in g:
            ss += rn.mul(gi, gi)                       # energy sum of squares (Q32 if g is Q16)
    norm = rn.isqrt(ss)                                # Q16 norm (sqrt of Q32)
    if norm <= max_norm or norm == 0:
        return norm
    for g in grads:
        for i in range(len(g)):
            g[i] = _sd(rn.mul(g[i], max_norm), norm)   # g *= max_norm/norm
    return norm


class AdamW:
    """Ring-native AdamW (our own optim.AdamW), ENERGY/Q16 domain, float-free.

    Params and grads are Q16 signed ints (value * 2^16); NO mod-256 wrap (weights are energy, not
    ARC). Moments m,v are Q16. Hyperparameters are Q16 rationals. Update per Kingma-Ba + decoupled
    weight decay (Loshchilov): sqrt via `rn.isqrt`, all divisions via shift-subtract `mf_floordiv`.

        m = b1*m + (1-b1)*g ;  v = b2*v + (1-b2)*g^2
        mhat = m/(1-b1^t) ;     vhat = v/(1-b2^t)
        theta -= lr*mhat/(sqrt(vhat)+eps) + lr*wd*theta
    """
    def __init__(self, sizes, lr=66, b1=58982, b2=65470, eps=655, wd=7):
        # defaults (Q16): lr=1e-3~66, b1=0.9~58982, b2=0.999~65470, wd=1e-4~7.
        # eps=655(~0.01): a fixed-point denom FLOOR — much larger than float Adam's 1e-8 because when
        # the Q16 2nd-moment v decays to ~0 and a fresh grad arrives, dividing by a tiny eps explodes
        # the step. 0.01 keeps lr*mhat/(sqrt(vhat)+eps) bounded. (Verified: removes the ep100 spike.)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = [[0] * s for s in sizes]
        self.v = [[0] * s for s in sizes]
        self.t = 0
        self.pw1 = ONE                                # b1^0
        self.pw2 = ONE                                # b2^0

    def step(self, params, grads):
        """params: list of Q16 flat lists (updated IN PLACE). grads: matching Q16 energy grads."""
        self.t += 1
        self.pw1 = rn.mul(self.pw1, self.b1) >> FRAC  # b1^t
        self.pw2 = rn.mul(self.pw2, self.b2) >> FRAC  # b2^t
        c1 = ONE - self.pw1                           # 1 - b1^t  (bias-correction denom)
        c2 = ONE - self.pw2
        b1, b2, lr, eps, wd = self.b1, self.b2, self.lr, self.eps, self.wd
        for pi in range(len(params)):
            p, g, m, v = params[pi], grads[pi], self.m[pi], self.v[pi]
            for i in range(len(p)):
                gi = g[i]
                m[i] = (rn.mul(b1, m[i]) + rn.mul(ONE - b1, gi)) >> FRAC
                g2 = rn.mul(gi, gi) >> FRAC
                v[i] = (rn.mul(b2, v[i]) + rn.mul(ONE - b2, g2)) >> FRAC
                mhat = _sd(m[i] << FRAC, c1)          # Q16
                vhat = _sd(v[i] << FRAC, c2)          # Q16 (>=0)
                sqrt_vhat = rn.isqrt(vhat << FRAC) if vhat > 0 else 0   # sqrt in Q16
                denom = sqrt_vhat + eps
                upd = _sd(rn.mul(lr, mhat), denom)    # lr*mhat/denom  (Q16 param units)
                decay = rn.mul(lr, rn.mul(wd, p[i]) >> FRAC) >> FRAC     # lr*wd*theta
                p[i] = p[i] - upd - decay


def coordinate_step(params, grads, loss_fn):
    """Coarse-to-fine, loss-gated coordinate step (SRD T5.5).

    Root cause of the SIN-training stall (derived by math): a +-1 step on a weight moves the
    pre-activation arg by its input coefficient x_i, so simultaneous steps overshoot the target
    angle (limit cycle). Fix: apply each parameter's gradient-sign step ONE AT A TIME and KEEP it
    only if it strictly reduces the loss. Overshoots are reverted; the fine (unit-coefficient /
    coprime) channel — the bias — lands the exact angle. This is the (state, r) codec as descent.

    params  : list of autograd Vars (ARC parameters)
    grads   : their ENERGY gradients
    loss_fn : zero-arg closure returning the current (non-wrapping) loss
    Returns number of accepted moves. Multiplier-free.
    """
    strict = 0
    for p, g in zip(params, grads):
        base = loss_fn()
        old = p.val
        s = sign(g)
        moved = False
        if s != 0:                                  # descent (against gradient): accept if NOT worse
            p.val = (old - s) & 0xFF                #   -> crosses SIN level-set plateaus
            l = loss_fn()
            if l <= base:
                moved = True
                if l < base:
                    strict += 1
            else:
                p.val = old
        if not moved:                               # opposite direction: strict improvement only
            d = s if s != 0 else 1
            p.val = (old + d) & 0xFF
            if loss_fn() < base:
                strict += 1
            else:
                p.val = old
    return strict


def _selftest():
    """AdamW minimizes a convex quadratic f(w)=sum (w-target)^2 ; grad=2(w-target). Q16, ring-only."""
    target = [2 * ONE, -1 * ONE, 3 * ONE, 0]
    w = [0, 0, 0, 0]
    opt = AdamW([len(w)], lr=3277, wd=0)             # lr~0.05, no decay for the pure convex test
    for _ in range(2000):
        grad = [rn.mul(2, w[i] - target[i]) for i in range(len(w))]   # Q16 energy grad
        opt.step([w], [[g for g in grad]])
    err = max(abs(w[i] - target[i]) for i in range(len(w)))
    ok = err <= (ONE >> 5)                            # within ~0.03
    print(f"  AdamW converges to target (err {err} <= {ONE>>5}): {'PASS' if ok else 'FAIL'}")
    print(f"    w={[round(x/ONE,3) for x in w]}  target={[round(x/ONE,3) for x in target]}")
    # weight decay pulls an un-gradiented param toward 0
    w2 = [5 * ONE]
    opt2 = AdamW([1], lr=3277, wd=6554)              # wd~0.1
    for _ in range(50):
        opt2.step([w2], [[0]])                        # zero grad -> only decay acts
    decayed = w2[0] < 5 * ONE
    print(f"  weight decay shrinks a zero-grad param: {'PASS' if decayed else 'FAIL'} {round(w2[0]/ONE,3)}")
    return ok and decayed


if __name__ == "__main__":
    print("ringkit.ml.optim AdamW self-test (ring-native, float-free):")
    print("RESULT:", "ALL PASS" if _selftest() else "FAIL")
