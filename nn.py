"""
ringkit.nn — engineer-facing model framework. Torch-shaped on the outside, ring-native inside.

Write ordinary models. The Z256 ring machinery — fold-late (energy/phase), unit-safety (no
zero-divisor collapse), exact linear SOLVE instead of blind descent, and content-routing
attention — is handled for you. You never touch mod-256, vacuums, or strides.

    import ringkit as rk
    layer = rk.nn.Linear(in_features=4, out_features=2)
    layer.fit(X, Y)                 # learns the exact ring map (solve, not descent) when it can
    pred = layer.predict(X_test)    # generalizes to unseen inputs

    out, who = rk.nn.attention(queries, keys, values)   # content-based routing, not lookup

Escape hatch: every layer exposes `.raw` (the underlying ring weights) for power users who DO
want the internals. Regular engineers can ignore it entirely.
"""
from ringkit.core import native as _rn
from ringkit.linalg.solve import solve as _solve, is_invertible as _is_inv, modinv as _modinv
from ringkit.linalg import fit as _fitmod
from ringkit.ml import attention as _attn

# re-export content-based attention at the framework level (the real transformer primitive)
attention = _attn.attend
attention_scores = _attn.scores


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


def positional_encode(rows, start=0):
    """Exact ring RoPE: shift each row by its position (phase + m). Rotation-by-addition is exact
    on the ring for ANY position (no analytic-sine error) — this is the ring way to encode order."""
    return [[(v + (start + m)) & 0xFF for v in row] for m, row in enumerate(rows)]


class Attention(Layer):
    """Content-based attention layer. rope=True applies exact additive positional encoding to
    queries and keys first, so routing becomes position-aware (relative position matters)."""

    def __init__(self, rope=False):
        self.rope = rope

    def __call__(self, Q, K, V, hard=True):
        Q = _as_rows(Q)
        K = _as_rows(K)
        V = _as_rows(V)
        if self.rope:
            Q = positional_encode(Q)
            K = positional_encode(K)
        return _attn.attend(Q, K, V, hard=hard)

    @property
    def raw(self):
        return {"rope": self.rope, "kind": "content-attention"}


class TransformerBlock(Layer):
    """Self-attention (+ optional RoPE) with a ring residual, then an optional SIN feed-forward.
    Compose these into rk.nn.Transformer stacks. Ring internals stay hidden."""

    def __init__(self, dim, rope=True):
        self.dim = int(dim)
        self.attn = Attention(rope=rope)
        self.ffn = Dense(dim, dim)

    def __call__(self, X):
        rows = _as_rows(X)
        attended, _ = self.attn(rows, rows, rows)                      # self-attention
        res = [[(rows[i][d] + attended[i][d]) & 0xFF for d in range(self.dim)]
               for i in range(len(rows))]                              # ring residual add
        if self.ffn._fitted:
            out = self.ffn(res)
            return out if isinstance(out[0], list) else [out]
        return res

    @property
    def raw(self):
        return {"attn": self.attn.raw, "ffn": self.ffn.raw, "dim": self.dim}


class Transformer(Layer):
    """Ring-native transformer for in-context tasks. Two headline capabilities, both defining
    of a real transformer (and both impossible for a lookup):

      induction(seq)  — predict the token that followed the previous occurrence of the last token.
                        Needs CONTENT (match the token) AND POSITION (pick the most-recent match,
                        read the +1 token). Generalizes to tokens NEVER seen in training.
      recall(...)     — content-based key->value recall through a LEARNED query decoder (trained by
                        exact solve). Generalizes to unseen bindings; a random-trained decoder can't.

    Ring internals stay hidden; `.raw` exposes them.
    """

    def __init__(self, key_dim=1, rope=True):
        self.key_dim = int(key_dim)
        self.rope = rope
        self.decoder = Linear(self.key_dim, self.key_dim)   # learned query decoder (solve-trained)
        self._fitted = False

    def induction(self, seq, rope=None):
        """seq: list of tokens (ring ints). Returns (predicted_token, matched_position)."""
        if rope is None:
            rope = self.rope
        L = len(seq)
        if L < 2:
            raise ValueError("induction: need at least 2 tokens")
        q = seq[-1] & 0xFF
        # Hard attention with SEPARATE channels (the ring way): content dominates (weight >> any
        # position), position only breaks ties among equal-content matches -> the most-recent
        # occurrence, which is the induction answer. Cramming both into one scalar corrupts content.
        W = 256                                    # content weight > max position index
        best_score = None
        best_pos = 0
        for i in range(L - 1):
            tok = seq[i] & 0xFF
            d = (q - tok) & 0xFF
            d = d if d < 256 - d else 256 - d      # ring content distance
            score = -_rn.mul(W, d)                 # content match dominates (0 iff exact match)
            if rope:
                score = score + i                  # recency tie-break among content matches
            if best_score is None or score > best_score:
                best_score = score
                best_pos = i
        return seq[best_pos + 1] & 0xFF, best_pos

    def fit(self, queries_enc, keys_true):
        """Learn the query decoder so encoded queries land in key space (exact ring solve)."""
        self.decoder.fit(queries_enc, keys_true)
        self._fitted = True
        self.train_exact = self.decoder.train_exact
        return self

    def recall(self, keys, values, query_enc):
        """Decode the (possibly encoded) query, then content-attend over keys to read the value.
        Returns (value_vector, matched_position)."""
        q = self.decoder(query_enc) if self._fitted else list(query_enc)
        out, who = _attn.attend([q], _as_rows(keys), _as_rows(values), hard=True)
        return out[0], who[0]

    @property
    def raw(self):
        return {"decoder": self.decoder.raw, "rope": self.rope, "key_dim": self.key_dim}


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
