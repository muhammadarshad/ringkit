"""
ring_autograd.py — dual-ring reverse-mode autodiff (SRD Phase 4), on ring_native.

Decision (SRD 3.4): DUAL-RING. Each Var carries a VALUE (ARC, ring mod 256, may wrap) and
accumulates a GRAD (ENERGY, signed, NON-wrapping). This split is essential: folding the
gradient mod 256 destroys the descent signal for large errors (measured: 36% vs 100%
convergence). So values fold (qsm), gradients do not (mul on signed values).
Local gradients are ring closed-forms, taken SIGNED:
    add : d/da = d/db = 1
    sub : d/da = 1, d/db = -1
    mul : y=a*b -> d/da = signed(b), d/db = signed(a)   (product rule)
    neg : d/da = -1
    sin : d/dphi = signed(COS(phi))                     (rotational derivative)
    cos : d/dphi = -signed(SIN(phi))
Backward composes with mul (any-size shift-add) and plain accumulation (no fold).
Autograd = ADI: ARC value forward, ENERGY differential backward. Multiplier-free.
"""
from ringkit.core import native as rn


def _as_var(x):
    return x if isinstance(x, Var) else Var(x)


class Var:
    __slots__ = ("val", "grad", "_parents", "_backward")

    def __init__(self, val, parents=(), backward=None):
        self.val = int(val) & 0xFF
        self.grad = 0
        self._parents = parents
        self._backward = backward if backward is not None else (lambda: None)

    # ---- ops ----  (value: ARC, folds mod 256 via qsm ; grad: ENERGY, signed, no fold)
    def add(self, other):
        o = _as_var(other)
        out = Var((self.val + o.val) & 0xFF, (self, o))

        def bw():
            self.grad += out.grad
            o.grad += out.grad
        out._backward = bw
        return out

    def sub(self, other):
        o = _as_var(other)
        out = Var((self.val - o.val) & 0xFF, (self, o))

        def bw():
            self.grad += out.grad
            o.grad += -out.grad
        out._backward = bw
        return out

    def mul(self, other):
        o = _as_var(other)
        out = Var(rn.qsm(self.val, o.val) & 0xFF, (self, o))

        def bw():
            self.grad += rn.mul(out.grad, rn._signed(o.val))            # d/da = signed(b)
            o.grad += rn.mul(out.grad, rn._signed(self.val))            # d/db = signed(a)
        out._backward = bw
        return out

    def neg(self):
        out = Var(rn.ring_neg(self.val), (self,))

        def bw():
            self.grad += -out.grad
        out._backward = bw
        return out

    def sin(self):
        out = Var(rn.SIN(self.val), (self,))

        def bw():                                                       # d SIN = signed(COS)
            self.grad += rn.mul(out.grad, rn._signed(rn.COS(self.val)))
        out._backward = bw
        return out

    def cos(self):
        out = Var(rn.COS(self.val), (self,))

        def bw():                                                       # d COS = -signed(SIN)
            self.grad += rn.mul(out.grad, -rn._signed(rn.SIN(self.val)))
        out._backward = bw
        return out

    # ---- reverse-mode ----
    def backward(self, seed=1):
        topo, seen = [], set()

        def build(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    build(p)
                topo.append(v)
        build(self)
        for v in topo:
            v.grad = 0
        self.grad = int(seed)          # ENERGY seed, no fold
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Var(val={self.val}, grad={self.grad})"
