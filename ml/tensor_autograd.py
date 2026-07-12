"""
ringkit.ml.tensor_autograd — dual-ring reverse-mode autodiff over RingTensors (T4.5).

The array-level counterpart of `ml/autograd.py`. Same DUAL-RING discipline (SRD 3.4): each TVar
carries a VALUE (a RingTensor, ARC, folds mod 256) and accumulates a GRAD (a flat signed-int
buffer, ENERGY, NON-wrapping — folding the gradient mod 256 destroys the descent signal). Local
gradients are ring closed-forms taken SIGNED; backward composes with `rn.mul` (any-size shift-add)
and plain accumulation (no fold). Multiplier-free; no numpy, no float.

Supported: elementwise add/sub/mul (same-shape or scalar), sin/cos, sum() reduction, 2-D matmul.
    a = TVar([[1,2],[3,4]]); b = TVar([[5,6],[7,8]])
    loss = a.matmul(b).sin().sum(); loss.backward(); a.grad  # signed d loss / d a
"""
from ringkit.core import native as rn
from ringkit.array.tensor import RingTensor, matmul as _matmul


def _sz(shape):
    n = 1
    for s in shape:
        n = rn.mul(n, s)
    return n


def _as_tvar(x, shape):
    if isinstance(x, TVar):
        return x
    if isinstance(x, RingTensor):
        return TVar(x)
    # scalar broadcast to `shape`
    v = int(x) & 0xFF
    return TVar(RingTensor([v for _ in range(_sz(shape))], tuple(shape)))


class TVar:
    __slots__ = ("val", "grad", "shape", "unit", "_parents", "_backward")

    def __init__(self, val, parents=(), backward=None, unit="arc"):
        if not isinstance(val, RingTensor):
            val = val if isinstance(val, (list, tuple)) else [int(val) & 0xFF]
            val = RingTensor(val, unit=unit)
        self.val = val
        self.shape = val.shape
        self.unit = val.unit
        self.grad = [0 for _ in range(_sz(self.shape))]   # signed ENERGY, no fold
        self._parents = parents
        self._backward = backward if backward is not None else (lambda: None)

    def _check_same(self, o, op):
        if self.shape != o.shape:
            raise ValueError(f"{op}: shape mismatch {self.shape} vs {o.shape}")

    # ── elementwise ──
    def add(self, other):
        o = _as_tvar(other, self.shape)
        self._check_same(o, "add")
        out = TVar(self.val + o.val, (self, o))

        def bw():
            g = out.grad
            for i in range(len(g)):
                self.grad[i] += g[i]
                o.grad[i] += g[i]
        out._backward = bw
        return out

    def sub(self, other):
        o = _as_tvar(other, self.shape)
        self._check_same(o, "sub")
        out = TVar(self.val - o.val, (self, o))

        def bw():
            g = out.grad
            for i in range(len(g)):
                self.grad[i] += g[i]
                o.grad[i] += -g[i]
        out._backward = bw
        return out

    def mul(self, other):
        o = _as_tvar(other, self.shape)
        self._check_same(o, "mul")
        out = TVar(self.val.rmul(o.val), (self, o))
        a_d, b_d = self.val.data, o.val.data

        def bw():
            g = out.grad
            for i in range(len(g)):
                self.grad[i] += rn.mul(g[i], rn._signed(b_d[i]))   # d/da = signed(b)
                o.grad[i] += rn.mul(g[i], rn._signed(a_d[i]))      # d/db = signed(a)
        out._backward = bw
        return out

    def sin(self):
        out = TVar(RingTensor([rn.SIN(v) for v in self.val.data], self.shape, self.unit), (self,))
        a_d = self.val.data

        def bw():
            g = out.grad
            for i in range(len(g)):
                self.grad[i] += rn.mul(g[i], rn._signed(rn.COS(a_d[i])))   # d SIN = signed(COS)
        out._backward = bw
        return out

    def cos(self):
        out = TVar(RingTensor([rn.COS(v) for v in self.val.data], self.shape, self.unit), (self,))
        a_d = self.val.data

        def bw():
            g = out.grad
            for i in range(len(g)):
                self.grad[i] += rn.mul(g[i], -rn._signed(rn.SIN(a_d[i])))  # d COS = -signed(SIN)
        out._backward = bw
        return out

    # ── reduction ──
    def sum(self):
        """Sum all elements -> scalar TVar (shape (1,)). Gradient broadcasts 1 to every input."""
        acc = 0
        for v in self.val.data:
            acc = (acc + v) & 0xFF
        out = TVar(RingTensor([acc], (1,), self.unit), (self,))

        def bw():
            g0 = out.grad[0]
            for i in range(len(self.grad)):
                self.grad[i] += g0
        out._backward = bw
        return out

    # ── matmul (2-D) ──
    def matmul(self, other):
        o = _as_tvar(other, self.shape)
        if len(self.shape) != 2 or len(o.shape) != 2:
            raise ValueError("matmul: both operands must be 2-D")
        M, K = self.shape
        K2, N = o.shape
        if K != K2:
            raise ValueError(f"matmul: inner dims disagree {self.shape} @ {o.shape}")
        out = TVar(_matmul(self.val, o.val), (self, o))
        a_d, b_d = self.val.data, o.val.data

        def bw():
            g = out.grad                                   # dY, shape M x N (signed, flat)
            # dA[m,k] = sum_n dY[m,n] * signed(B[k,n])
            for m in range(M):
                for k in range(K):
                    acc = 0
                    for nn in range(N):
                        acc += rn.mul(g[rn.mul(m, N) + nn], rn._signed(b_d[rn.mul(k, N) + nn]))
                    self.grad[rn.mul(m, K) + k] += acc
            # dB[k,n] = sum_m signed(A[m,k]) * dY[m,n]
            for k in range(K):
                for nn in range(N):
                    acc = 0
                    for m in range(M):
                        acc += rn.mul(rn._signed(a_d[rn.mul(m, K) + k]), g[rn.mul(m, N) + nn])
                    o.grad[rn.mul(k, N) + nn] += acc
        out._backward = bw
        return out

    # ── reverse-mode ──
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
            for i in range(len(v.grad)):
                v.grad[i] = 0
        for i in range(len(self.grad)):
            self.grad[i] = int(seed)       # ENERGY seed, no fold
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"TVar(shape={self.shape}, val={list(self.val.data)}, grad={self.grad})"
