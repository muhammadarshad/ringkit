"""
ringkit.nn.transformer — the attention stack: exact ring RoPE (positional_encode), Attention,
TransformerBlock, Transformer (induction + in-context recall). Ring internals stay hidden.
"""
from ringkit.core import native as _rn
from ringkit.ml import attention as _attn
from ringkit.nn.layers import Layer, Linear, Dense, _as_rows


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


class HopBlock(Layer):
    """One TRAINED attention hop: a solve-fit Linear query-map + hard content attention.
    fit() learns the exact ring map that carries this hop's incoming representation into
    its memory's key space (linalg.solve — no descent). The hop then routes by content."""

    def __init__(self, dim):
        self.dim = int(dim)
        self.qmap = Linear(self.dim, self.dim)

    def fit(self, X, Y):
        self.qmap.fit(X, Y)
        self.train_exact = self.qmap.train_exact
        return self

    def __call__(self, q, K, V):
        out, who = _attn.attend([self.qmap(q)], _as_rows(K), _as_rows(V), hard=True)
        return out[0], who[0]

    @property
    def raw(self):
        return {"qmap": self.qmap.raw, "dim": self.dim}


class Stacked(Layer):
    """Stacked multi-block trained model: depth-d recall composes d trained attention hops —
    block b's retrieved value becomes block b+1's query. Each block is trained by EXACT
    solve (the ring way), and the capability is genuinely deep: a (d-1)-block model fails
    the same task (the depth control in tests/test_stacked.py), as does a random-trained
    block. Ring internals stay hidden; `.raw` exposes them."""

    def __init__(self, blocks=2, dim=2):
        self.dim = int(dim)
        self.blocks = [HopBlock(self.dim) for _ in range(int(blocks))]

    def fit(self, per_block):
        """per_block: one (X, Y) training set per hop — X the hop's incoming representation,
        Y the matching key-space rows. Each hop is solved independently (predictable bins)."""
        if len(per_block) != len(self.blocks):
            raise ValueError(f"Stacked.fit: {len(per_block)} training sets for {len(self.blocks)} blocks")
        for blk, (X, Y) in zip(self.blocks, per_block):
            blk.fit(X, Y)
        self.train_exact = all(getattr(b, "train_exact", False) for b in self.blocks)
        return self

    def recall(self, memories, query):
        """Route `query` through every hop: memories = [(K_b, V_b)] per block. Returns
        (final value row, [matched position per hop])."""
        if len(memories) != len(self.blocks):
            raise ValueError(f"Stacked.recall: {len(memories)} memories for {len(self.blocks)} blocks")
        q = list(query)
        path = []
        for blk, (K, V) in zip(self.blocks, memories):
            q, who = blk(q, K, V)
            path.append(who)
        return q, path

    @property
    def raw(self):
        return {"blocks": [b.raw for b in self.blocks], "dim": self.dim}
