"""RoPE attention + encoder layers + the Rotor RDT recurrence, native to ringkit (Z256, float-free).
Port of transformer/rope.py (RoPEAttention, RoPETransformerEncoderLayer) and transformer/recurrent.py
(RoPERDTEncoder). Internal domain Q16 (RingRational den=2^16); linears via QSM; softmax/gelu/sigmoid
via the ring transcendentals; QK-norm and pre-LN via ring layernorm. No float, no torch.
"""
from ringkit.core import native as rn
from ringkit.qcm import constants as C
from ringkit.qcm.tensor import QSMLinear, layernorm
from ringkit.qcm.activations import (softmax_fixed, sigmoid_fixed, gelu_fixed, FRAC, ONE)
from ringkit.qcm.rope import QuantumRoPE, QuantumRoPE4D
from ringkit.qcm.ssd import _to_q16, _mul_q16

# CRITICAL FIX 1: RDT keep-gate bias = -2.0 (sigmoid(-2)=0.119). In Q16.
GATE_BIAS_Q16 = -2 * ONE


def _dot_q16(a, b):
    acc = 0
    for i in range(len(a)):
        acc += rn.mul(a[i], b[i])
    return acc >> FRAC


def _inv_sqrt_q16(m):
    """1/sqrt(m) in Q16, float-free (ring isqrt)."""
    sq = rn.isqrt(m << (2 * FRAC))               # sqrt(m) in Q16
    if sq == 0:
        sq = 1
    return rn.mf_floordiv(ONE << FRAC, sq)        # ONE^2 / (sqrt(m)*ONE) = 1/sqrt(m)


class _Norm:
    """LayerNorm parameter holder (gamma, beta) — Q16 gains/offsets."""
    __slots__ = ("gamma", "beta")

    def __init__(self, dim, gamma=None, beta=None):
        self.gamma = gamma if gamma is not None else [ONE] * dim
        self.beta = beta if beta is not None else [0] * dim

    def __call__(self, vals_q16):
        return layernorm(vals_q16, self.gamma, self.beta)


class RoPEAttention:
    """Multi-head self-attention: QK-norm + QuantumRoPE + softmax attention. Q16, float-free.
    O(N^2) over tokens (dense global attention); route to a kernel for real N later."""
    __slots__ = ("embed_dim", "num_heads", "head_dim", "q_proj", "k_proj", "v_proj", "out_proj",
                 "q_norm", "k_norm", "rope", "_scale")

    def __init__(self, embed_dim=C.D_MODEL, num_heads=4, rope_4d=False,
                 q_proj=None, k_proj=None, v_proj=None, out_proj=None):
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.q_proj = q_proj or QSMLinear([[0] * embed_dim for _ in range(embed_dim)])
        self.k_proj = k_proj or QSMLinear([[0] * embed_dim for _ in range(embed_dim)])
        self.v_proj = v_proj or QSMLinear([[0] * embed_dim for _ in range(embed_dim)])
        self.out_proj = out_proj or QSMLinear([[0] * embed_dim for _ in range(embed_dim)])
        self.q_norm = _Norm(self.head_dim)
        self.k_norm = _Norm(self.head_dim)
        rope_cls = QuantumRoPE4D if rope_4d else QuantumRoPE
        self.rope = rope_cls(self.head_dim, max_z=32, max_x=128)
        self._scale = _inv_sqrt_q16(self.head_dim)

    def __call__(self, tokens_q16, D, H):
        N = D * H
        nh, hd, d = self.num_heads, self.head_dim, self.embed_dim
        q = [_to_q16(self.q_proj((x, -FRAC))) for x in tokens_q16]
        k = [_to_q16(self.k_proj((x, -FRAC))) for x in tokens_q16]
        v = [_to_q16(self.v_proj((x, -FRAC))) for x in tokens_q16]
        ctx = [[0] * d for _ in range(N)]
        for h in range(nh):
            base = h * hd
            qh = [self.q_norm([q[t][base + j] for j in range(hd)]) for t in range(N)]
            kh = [self.k_norm([k[t][base + j] for j in range(hd)]) for t in range(N)]
            vh = [[v[t][base + j] for j in range(hd)] for t in range(N)]
            qh = self.rope.forward(qh, D, H)
            kh = self.rope.forward(kh, D, H)
            for t1 in range(N):
                scores = [rn.mul(_dot_q16(qh[t1], kh[t2]), self._scale) >> FRAC for t2 in range(N)]
                w = softmax_fixed(scores)
                for j in range(hd):
                    acc = 0
                    for t2 in range(N):
                        acc += rn.mul(w[t2], vh[t2][j])
                    ctx[t1][base + j] = acc >> FRAC
        return [_to_q16(self.out_proj((ctx[t], -FRAC))) for t in range(N)]


class RoPEEncoderLayer:
    """Pre-LN transformer encoder layer with QuantumRoPE attention (Q16, float-free)."""
    __slots__ = ("attn", "linear1", "linear2", "norm1", "norm2", "d_model", "d_ff")

    def __init__(self, d_model=C.D_MODEL, nhead=4, dim_feedforward=512, rope_4d=False,
                 attn=None, linear1=None, linear2=None):
        self.d_model = int(d_model)
        self.d_ff = int(dim_feedforward)
        self.attn = attn or RoPEAttention(d_model, nhead, rope_4d=rope_4d)
        self.linear1 = linear1 or QSMLinear([[0] * d_model for _ in range(self.d_ff)])
        self.linear2 = linear2 or QSMLinear([[0] * self.d_ff for _ in range(d_model)])
        self.norm1 = _Norm(d_model)
        self.norm2 = _Norm(d_model)

    def __call__(self, tokens_q16, D, H):
        N = D * H
        normed = [self.norm1(x) for x in tokens_q16]
        a = self.attn(normed, D, H)
        x = [[tokens_q16[t][j] + a[t][j] for j in range(self.d_model)] for t in range(N)]
        out = []
        for t in range(N):
            ff = _to_q16(self.linear1((self.norm2(x[t]), -FRAC)))
            ff = [gelu_fixed(v) for v in ff]
            ff = _to_q16(self.linear2((ff, -FRAC)))
            out.append([x[t][j] + ff[j] for j in range(self.d_model)])
        return out


class RoPERDTEncoder:
    """Gated Recurrent Depth Transformer (Rotor body). One shared layer applied `depth` times with
    input re-injection (inject_gate) and a GRU keep-gate (CRITICAL FIX 1: gate bias -2.0). Q16."""
    __slots__ = ("depth", "d_model", "layer", "inject_gate", "depth_embed", "gate")

    def __init__(self, d_model=C.D_MODEL, nhead=4, dim_feedforward=512, depth=4, rope_4d=False,
                 layer=None, inject_gate=None, depth_embed=None, gate=None):
        self.depth = int(depth)
        self.d_model = int(d_model)
        self.layer = layer or RoPEEncoderLayer(d_model, nhead, dim_feedforward, rope_4d=rope_4d)
        self.inject_gate = inject_gate or QSMLinear([[0] * d_model for _ in range(d_model)])
        self.depth_embed = depth_embed or [[0] * d_model for _ in range(32)]   # Q16 table
        # gate: QSMLinear(2*d_model -> d_model), NO bias here; GATE_BIAS added pre-sigmoid (fix 1).
        self.gate = gate or QSMLinear([[0] * (2 * d_model) for _ in range(d_model)])

    def __call__(self, tokens_q16, D, H, depth=None):
        depth = self.depth if depth is None else depth
        N = D * H
        d = self.d_model
        x0 = [row[:] for row in tokens_q16]
        h = [row[:] for row in tokens_q16]
        for step in range(depth):
            de = self.depth_embed[step]
            h_in = []
            for t in range(N):
                alpha = [sigmoid_fixed(a) for a in _to_q16(self.inject_gate((x0[t], -FRAC)))]
                h_in.append([h[t][j] + _mul_q16(alpha[j], x0[t][j]) + de[j] for j in range(d)])
            h_new = self.layer(h_in, D, H)
            for t in range(N):
                pre = _to_q16(self.gate((h[t] + h_new[t], -FRAC)))
                g = [sigmoid_fixed(pre[j] + GATE_BIAS_Q16) for j in range(d)]   # FIX 1
                h[t] = [_mul_q16(g[j], h_new[t][j]) + _mul_q16(ONE - g[j], h[t][j]) for j in range(d)]
        return h


def _selftest():
    D, H = 2, 3
    N = D * H
    dm, nh = C.D_MODEL, 4
    def ql(M, K, seed):
        return QSMLinear([[((r * 2 + k + seed) % 5) - 2 for k in range(K)] for r in range(M)])
    toks = [[((t * 9 + j * 5) % 200 - 100) * (ONE >> 8) for j in range(dm)] for t in range(N)]
    attn = RoPEAttention(dm, nh, q_proj=ql(dm, dm, 1), k_proj=ql(dm, dm, 2),
                         v_proj=ql(dm, dm, 3), out_proj=ql(dm, dm, 4))
    a = attn(toks, D, H)
    a_ok = (len(a) == N and all(len(r) == dm for r in a))
    print(f"  RoPEAttention forward: {N} tokens x {dm}, valid={a_ok}")
    enc = RoPEEncoderLayer(dm, nh, 256, attn=attn, linear1=ql(256, dm, 5), linear2=ql(dm, 256, 6))
    e = enc(toks, D, H)
    e_ok = (len(e) == N and all(len(r) == dm for r in e))
    print(f"  RoPEEncoderLayer forward: valid={e_ok}")
    rdt = RoPERDTEncoder(dm, nh, 256, depth=2, layer=enc,
                         inject_gate=ql(dm, dm, 7), gate=ql(dm, 2 * dm, 8))
    r = rdt(toks, D, H)
    r_ok = (len(r) == N and all(len(row) == dm for row in r))
    print(f"  RoPERDTEncoder (depth 2) forward: valid={r_ok}")
    fix1 = (GATE_BIAS_Q16 == -2 * ONE)
    print(f"  CRITICAL FIX 1 gate bias == -2.0 (Q16 {GATE_BIAS_Q16}): {'PASS' if fix1 else 'FAIL'}")
    # softmax sanity: weights sum ~ONE
    from ringkit.qcm.activations import softmax_fixed as sm
    w = sm([3 * ONE, ONE, -2 * ONE, 0])
    ssum = sum(w)
    sok = abs(ssum - ONE) <= 8
    print(f"  softmax sums to ONE ({ssum} vs {ONE}): {'PASS' if sok else 'FAIL'}")
    return a_ok and e_ok and r_ok and fix1 and sok


if __name__ == "__main__":
    print("ringkit.qcm.attention self-test:")
    print("RESULT:", "ALL PASS" if _selftest() else "FAIL")
