"""
ringkit.nn.layers — core layers of the engineer-facing framework: Layer base, Linear
(exact solve), Dense (invert-then-solve), Sequential. Torch-shaped outside, ring-native inside.
"""
from ringkit.core import native as _rn
from ringkit.linalg.solve import solve as _solve, modinv as _modinv
from ringkit.linalg import fit as _fitmod


def _as_rows(X):
    """Accept a single vector or a batch; always return a list of ring-int rows."""
    if not X:
        raise ValueError("empty input")
    if not isinstance(X[0], (list, tuple)):
        X = [X]
    return [[int(v) & 0xFF for v in row] for row in X]


def _matmul(X, W):
    """X (n x din) @ W (din x dout) -> (n x dout), ring, multiplier-free."""
    din = len(W)
    dout = len(W[0]) if din else 0
    out = []
    for row in X:
        out.append([sum(_rn.mul(row[i], W[i][j]) for i in range(din)) & 0xFF for j in range(dout)])
    return out


def _independent_rows(X, din):
    """Pick indices of `din` rows independent over Z256 (odd-pivot Gaussian). Fewer if rank<din."""
    basis = []          # list of (pivot_col, reduced_row)
    chosen = []
    for idx, raw in enumerate(X):
        r = [v & 0xFF for v in raw]
        for pc, brow in basis:
            if r[pc] & 0xFF:
                f = r[pc] & 0xFF
                r = [(r[k] - _rn.mul(f, brow[k])) & 0xFF for k in range(din)]
        piv = None
        for c in range(din):
            if r[c] & 1:                        # odd -> invertible pivot
                piv = c
                break
        if piv is not None:
            inv = _modinv(r[piv])
            r = [_rn.mul(v, inv) & 0xFF for v in r]
            basis.append((piv, r))
            chosen.append(idx)
            if len(chosen) == din:
                break
    return chosen


class Layer:
    @property
    def raw(self):
        """Escape hatch: the ring internals of this layer."""
        return {}

    def predict(self, X):
        return self(X)


class Linear(Layer):
    """A learnable ring-linear map y = x·W. `fit` recovers W EXACTLY by solving the ring system
    (the ring's superpower — no gradient descent needed for linear maps) whenever the data
    determines it; it generalizes perfectly to unseen inputs when a true linear rule exists."""

    def __init__(self, in_features, out_features):
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        # start at zero; fit() sets the real weights
        self.W = [[0 for _ in range(self.out_features)] for _ in range(self.in_features)]
        self._fitted = False

    def __call__(self, X):
        rows = _as_rows(X)
        if len(rows[0]) != self.in_features:
            raise ValueError(f"expected {self.in_features} features, got {len(rows[0])}")
        out = _matmul(rows, self.W)
        return out[0] if len(out) == 1 else out

    def fit(self, X, Y):
        """Learn the exact ring map from examples. X: (n x in), Y: (n x out). Needs `in`
        independent examples; extra examples are used to VERIFY the solution is consistent."""
        X = _as_rows(X)
        Y = _as_rows(Y)
        if len(X) != len(Y):
            raise ValueError(f"X/Y count mismatch: {len(X)} vs {len(Y)}")
        din, dout = self.in_features, self.out_features
        if len(X[0]) != din or len(Y[0]) != dout:
            raise ValueError(f"shape mismatch: X in={len(X[0])} (want {din}), Y out={len(Y[0])} (want {dout})")
        rows = _independent_rows(X, din)
        if len(rows) < din:
            raise ValueError(
                f"under-determined: found only {len(rows)} independent examples, need {din}. "
                "Provide more varied examples (or this map isn't linearly recoverable).")
        A = [X[r] for r in rows]
        W = [[0 for _ in range(dout)] for _ in range(din)]
        for j in range(dout):
            col = _solve(A, [Y[r][j] for r in rows])
            for i in range(din):
                W[i][j] = col[i]
        self.W = W
        self._fitted = True
        # honesty: verify on ALL provided examples (exact learning, not memorizing a subset)
        pred = _matmul(X, W)
        self.train_exact = all(pred[i] == Y[i] for i in range(len(X)))
        return self

    @property
    def raw(self):
        return {"W_ring": self.W, "in_features": self.in_features,
                "out_features": self.out_features, "fitted": self._fitted}


class Dense(Layer):
    """Nonlinear layer y = SIN(x·W + b). Learned by INVERT-THEN-SOLVE (invert the activation to
    its preimage, then solve the linear system exactly) — the ring finds the exact fit where
    gradient descent would stall in local minima. Needs in_features+1 varied examples per output."""

    def __init__(self, in_features, out_features):
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.params = [None for _ in range(self.out_features)]   # per-output [w..., b]
        self._fitted = False

    def __call__(self, X):
        rows = _as_rows(X)
        if self.params[0] is None:
            raise ValueError("Dense used before fit()")
        out = []
        for r in rows:
            o = []
            for j in range(self.out_features):
                p = self.params[j]
                acc = p[-1]
                for i in range(self.in_features):
                    acc = (acc + _rn.mul(r[i], p[i])) & 0xFF
                o.append(_rn.SIN(acc))
            out.append(o)
        return out[0] if len(out) == 1 else out

    def fit(self, X, Y):
        X = _as_rows(X)
        Y = _as_rows(Y)
        if len(X[0]) != self.in_features or len(Y[0]) != self.out_features:
            raise ValueError("Dense.fit: feature/output shape mismatch")
        params = []
        for j in range(self.out_features):
            p = _fitmod.fit(X, [Y[k][j] for k in range(len(Y))])
            if p is None:
                raise ValueError(
                    f"Dense: output {j} has no exact ring fit for this data "
                    "(target may be outside the SIN range, or examples too few/aligned).")
            params.append(p)
        self.params = params
        self._fitted = True
        pred = self(X)
        self.train_exact = all(pred[i] == Y[i] for i in range(len(X)))
        return self

    @property
    def raw(self):
        return {"params_ring": self.params, "activation": "SIN",
                "in_features": self.in_features, "out_features": self.out_features}


class Sequential(Layer):
    """Chain layers. predict runs them in order (torch-like)."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    @property
    def raw(self):
        return {"layers": [l.raw for l in self.layers]}
