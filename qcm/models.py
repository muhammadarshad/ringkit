"""QCM models, native to ringkit (Z256, float-free). Port of transformer/models.py.

MPRCMamba2 (Soliton) is complete here — the twin's base model: shared QCM front-end
(QuadrantRingProjector -> VacuumDepthEmbedding) -> num_layers x Mamba2SSD -> mean-pool -> head.
MPRCRDT (Rotor) and MPRCViT (Gluon) share the front-end/head and differ only in the body
(RoPE attention encoder / recurrent depth); their bodies are the next module (attention/encoder)
and are declared here with a clear NotImplementedError until that pass lands — no half-built code.

Everything runs on the ring substrate (QSM int8 linears, ring trig, ring transcendentals, Q16
RingRational domain). No float, no torch, no numpy, no emulation. Verified by ring-native forward
validity (finite valid Z-domain logits), never by a torch oracle.
"""
from ringkit.core import native as rn
from ringkit.device import default_device
from ringkit.qcm import constants as C
from ringkit.qcm.tensor import QSMLinear
from ringkit.qcm.activations import rmsnorm_fixed, FRAC, ONE
from ringkit.qcm.frontend import QuadrantRingProjector, VacuumDepthEmbedding
from ringkit.qcm.ssd import Mamba2SSD, _to_q16
from ringkit.qcm.attention import RoPERDTEncoder, RoPEEncoderLayer


def _mean_pool_q16(tokens, dev=None):
    """tokens: N (ints, exp) blocks -> mean vector in Q16. Sum over tokens on the device (colsum
    kernel); the /N (dim small floor-divs, trunc-toward-zero) is marshaling, not the hot loop."""
    dev = dev if dev is not None else default_device()
    N = len(tokens)
    dim = len(tokens[0][0])
    flat = [v for tok in tokens for v in _to_q16(tok)]          # [N*dim]
    s = dev.colsum(flat, N, dim)                                # sum over tokens [dim] (kernel)
    return [rn.mf_floordiv(s[j], N) if s[j] >= 0 else -rn.mf_floordiv(-s[j], N) for j in range(dim)]


class _Head:
    """RMSNorm + QSM classifier (native port of nn.Sequential(LayerNorm, Linear)). On-device."""
    __slots__ = ("norm_weight", "clf")

    def __init__(self, d_model, n_classes, norm_weight=None, clf=None):
        self.norm_weight = norm_weight if norm_weight is not None else [ONE] * d_model
        self.clf = clf if clf is not None else QSMLinear([[0] * d_model for _ in range(n_classes)])

    def __call__(self, pooled_q16, dev=None):
        yn = rmsnorm_fixed(pooled_q16, self.norm_weight, FRAC, 1, dev=dev)   # device rmsnorm kernel
        return self.clf((yn, -FRAC))                    # (logit ints, exp)


class MPRCMamba2:
    """Soliton — Mamba-2 SSD with the QCM physics front-end. Native, float-free.

    forward(grid_rows, D, H): grid_rows = D*H rows of W=D_MODEL Z256 ints (the photon/wave field).
    Returns (logit ints, exp) over n_classes.
    """
    def __init__(self, embed_dim=C.D_MODEL, num_layers=2, num_heads=4, n_classes=2,
                 rope_4d=False, use_gate=False, quadrant_proj=None, vacuum_emb=None,
                 layers=None, head=None):
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.n_classes = int(n_classes)
        self.quadrant_proj = quadrant_proj or QuadrantRingProjector(C.D_MODEL, embed_dim)
        self.vacuum_emb = vacuum_emb or VacuumDepthEmbedding(embed_dim, max_depth=C.D)
        self.layers = layers or [Mamba2SSD(d_model=embed_dim, num_heads=num_heads,
                                           rope_4d=rope_4d, use_gate=use_gate)
                                 for _ in range(self.num_layers)]
        self.head = head or _Head(embed_dim, n_classes)

    def encode_sequence(self, grid_rows, D, H):
        """Front-end + SSD layers -> N token blocks (the OCR/reasoning memory)."""
        z = self.quadrant_proj(grid_rows)
        z = self.vacuum_emb(z, grid_rows, D, H)
        for layer in self.layers:
            z = layer(z, D, H)
        return z

    def forward(self, grid_rows, D, H):
        z = self.encode_sequence(grid_rows, D, H)
        pooled = _mean_pool_q16(z)
        return self.head(pooled)

    __call__ = forward


class MPRCRDT:
    """Rotor — Gated Recurrent Depth Transformer with the QCM front-end. Native, float-free.
    Front-end -> RoPERDTEncoder (shared layer, `depth` recursive steps, gate_bias -2.0) -> mean-pool
    -> head. forward(grid_rows, D, H) -> (logit ints, exp)."""
    def __init__(self, embed_dim=C.D_MODEL, depth=4, num_heads=4, n_classes=2, rope_4d=False,
                 quadrant_proj=None, vacuum_emb=None, encoder=None, head=None):
        self.embed_dim = int(embed_dim)
        self.quadrant_proj = quadrant_proj or QuadrantRingProjector(C.D_MODEL, embed_dim)
        self.vacuum_emb = vacuum_emb or VacuumDepthEmbedding(embed_dim, max_depth=C.D)
        self.encoder = encoder or RoPERDTEncoder(embed_dim, num_heads, 512, depth, rope_4d=rope_4d)
        self.head = head or _Head(embed_dim, n_classes)

    def encode_sequence(self, grid_rows, D, H):
        z = self.quadrant_proj(grid_rows)
        z = self.vacuum_emb(z, grid_rows, D, H)
        return self.encoder([_to_q16(t) for t in z], D, H)

    def forward(self, grid_rows, D, H):
        z = self.encode_sequence(grid_rows, D, H)
        pooled = _mean_pool_q16([(row, 0) for row in z])
        return self.head(pooled)

    __call__ = forward


class MPRCViT:
    """Gluon — ViT with the QCM front-end and `depth` INDEPENDENT RoPE encoder layers. Native.
    (Register tokens omitted — the deployed medical path uses none.) forward(grid_rows,D,H)->logits."""
    def __init__(self, embed_dim=C.D_MODEL, depth=6, num_heads=4, n_classes=2, rope_4d=False,
                 quadrant_proj=None, vacuum_emb=None, layers=None, head=None):
        self.embed_dim = int(embed_dim)
        self.quadrant_proj = quadrant_proj or QuadrantRingProjector(C.D_MODEL, embed_dim)
        self.vacuum_emb = vacuum_emb or VacuumDepthEmbedding(embed_dim, max_depth=C.D)
        self.layers = layers or [RoPEEncoderLayer(embed_dim, num_heads, embed_dim * 4, rope_4d=rope_4d)
                                 for _ in range(int(depth))]
        self.head = head or _Head(embed_dim, n_classes)

    def encode_sequence(self, grid_rows, D, H):
        z = self.quadrant_proj(grid_rows)
        z = self.vacuum_emb(z, grid_rows, D, H)
        z = [_to_q16(t) for t in z]
        for layer in self.layers:
            z = layer(z, D, H)
        return z

    def forward(self, grid_rows, D, H):
        z = self.encode_sequence(grid_rows, D, H)
        pooled = _mean_pool_q16([(row, 0) for row in z])
        return self.head(pooled)

    __call__ = forward


def _selftest():
    D, H = 2, 3
    N = D * H
    W = C.D_MODEL
    # photon-like grid (Z256), one vacuum row
    grid = [[(d * 41 + h * 13 + w * 7) & 0xFF for w in range(W)] for d in range(D) for h in range(H)]
    grid[2] = [0] * W
    # small non-trivial weights so the forward exercises real paths
    def ql(M, K, seed):
        return QSMLinear([[((r * 3 + k + seed) % 5) - 2 for k in range(K)] for r in range(M)])
    qp = QuadrantRingProjector(W, C.D_MODEL); qp.proj = ql(C.D_MODEL, W * 4, 1)
    vde = VacuumDepthEmbedding(C.D_MODEL, max_depth=D)
    vde.depth_emb = [[(d * 5 + j) % 7 - 3 for j in range(C.D_MODEL)] for d in range(D)]
    vde.vacuum_emb = [2] * C.D_MODEL
    layers = [Mamba2SSD(d_model=C.D_MODEL, num_heads=4,
                        in_proj=ql(3 * C.D_MODEL + 4, C.D_MODEL, 2),
                        out_proj=ql(C.D_MODEL, C.D_MODEL, 3)) for _ in range(2)]
    head = _Head(C.D_MODEL, 2, clf=ql(2, C.D_MODEL, 4))
    model = MPRCMamba2(num_layers=2, num_heads=4, n_classes=2,
                       quadrant_proj=qp, vacuum_emb=vde, layers=layers, head=head)
    logits, exp = model.forward(grid, D, H)
    ok = (len(logits) == 2 and all(isinstance(v, int) for v in logits))
    print(f"  MPRCMamba2 (Soliton) end-to-end forward: logits={logits} exp={exp}, valid={ok}")
    pred = 0 if logits[0] >= logits[1] else 1
    print(f"  argmax prediction (binary twin head): class {pred}")
    return ok


if __name__ == "__main__":
    print("ringkit.qcm.models self-test:")
    print("RESULT:", "ALL PASS" if _selftest() else "FAIL")
