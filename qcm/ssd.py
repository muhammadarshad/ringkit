"""Mamba2SSD — the Soliton body, native to ringkit (Z256, float-free). Port of transformer/ssd.py.

Deployed default (what MPRCMamba2 uses): NON-CAUSAL VSSD — a global symmetric interaction, so the
`dA`/A_log decay is NOT on the path (it is only used by the causal branch). The scan is:

    proj = in_proj(x) -> q,k,v,delta         delta = softplus(delta)
    q,k = QuantumRoPE(q), QuantumRoPE(k)      (state transitions respect the Z256 ring metric)
    state = delta * (k (*) v)                 (*) = elementwise
    h = sum_tokens(state)                     (t->inf VSSD limit; global, exact)
    y = q (*) h
    y = RMSNorm(y)                            [* sigmoid(5-arm ADI gate) if use_gate]
    out = out_proj(y)

Internal numeric domain is Q16 (a RingRational with den=2^16): elementwise products use the exact
shift-add `rn.mul` then >>16; linears use the QSM int8 core (`QSMLinear`). No float, no torch.
The causal (log-domain cumsum) and lattice2d and 5-arm-gate variants are flagged for a later pass.
"""
from ringkit.core import native as rn
from ringkit.device import default_device
from ringkit.qcm import constants as C
from ringkit.qcm.tensor import QSMLinear
from ringkit.qcm.activations import softplus_fixed, sigmoid_list, softmax_fixed, rmsnorm_fixed, FRAC, ONE
from ringkit.qcm.rope import QuantumRoPE, QuantumRoPE4D


def _to_q16(block):
    """(ints, exp) block  ->  Q16 int list (value * 2^16), exact shift."""
    ints, exp = block
    sh = exp + FRAC
    if sh >= 0:
        return [int(v) << sh for v in ints]
    r = -sh
    return [int(v) >> r for v in ints]


def _mul_q16(a, b):
    """Q16 * Q16 -> Q16, exact (rn.mul then >>FRAC)."""
    return rn.mul(a, b) >> FRAC


class Mamba2SSD:
    """One Soliton SSD block. Weights supplied as native QSMLinear/int gains (loaded or fit); a
    zero/​identity default lets the forward run for a validity smoke. head_dim = d_model/num_heads."""
    __slots__ = ("d_model", "num_heads", "head_dim", "rope_4d", "use_gate", "use_lattice2d",
                 "in_proj", "out_proj", "norm_weight", "rope", "diff_steps",
                 "gate_proj", "gate_route", "arm_bias", "dev")

    def __init__(self, d_model=C.D_MODEL, num_heads=4, rope_4d=False, use_gate=False,
                 use_lattice2d=False, in_proj=None, out_proj=None, norm_weight=None,
                 gate_proj=None, gate_route=None, arm_bias=None, dev=None):
        self.dev = dev
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.rope_4d = bool(rope_4d)
        self.use_gate = bool(use_gate)
        self.use_lattice2d = bool(use_lattice2d)
        self.in_proj = in_proj if in_proj is not None else QSMLinear(
            [[0] * self.d_model for _ in range(3 * self.d_model + self.num_heads)])
        self.out_proj = out_proj if out_proj is not None else QSMLinear(
            [[0] * self.d_model for _ in range(self.d_model)])
        self.norm_weight = norm_weight if norm_weight is not None else [ONE] * self.d_model
        rope_cls = QuantumRoPE4D if rope_4d else QuantumRoPE
        self.rope = rope_cls(self.head_dim, max_z=32, max_x=128)
        # per-head heat-kernel diffusion time on the (D,H) torus: first half GLOBAL (-1 = Sum,
        # the t->inf limit, exact); second half LOCAL geometric ladder 1,2,4,8,16 (physics, derived).
        half = self.num_heads // 2
        self.diff_steps = [-1 if h < half else min(16, 1 << (h - half)) for h in range(self.num_heads)]
        # 5-arm ADI gate (center + R/M/P/C): derived-geometry volume knob, not an ML classifier.
        if use_gate:
            self.gate_proj = gate_proj or QSMLinear([[0] * self.d_model for _ in range(self.d_model)])
            self.gate_route = gate_route or QSMLinear([[0] * self.d_model for _ in range(5)])
            self.arm_bias = arm_bias or [[0] * self.d_model for _ in range(5)]   # Q16
        else:
            self.gate_proj = self.gate_route = self.arm_bias = None

    def __call__(self, tokens, D, H):
        """tokens: N=D*H (ints, exp) blocks (d_model each). Returns N (ints, exp) blocks."""
        N = D * H
        d, nh, hd = self.d_model, self.num_heads, self.head_dim
        # in_proj -> per-token Q16 [3d+nh]
        proj = [_to_q16(self.in_proj(tok)) for tok in tokens]
        # split into per-head q,k,v (Q16) and per-head delta (Q16 -> softplus)
        q_h = [[proj[t][h * hd:(h + 1) * hd] for t in range(N)] for h in range(nh)]
        k_h = [[proj[t][d + h * hd: d + (h + 1) * hd] for t in range(N)] for h in range(nh)]
        v_h = [[proj[t][2 * d + h * hd: 2 * d + (h + 1) * hd] for t in range(N)] for h in range(nh)]
        delta = [[softplus_fixed(proj[t][3 * d + h], FRAC) for t in range(N)] for h in range(nh)]
        # QuantumRoPE on q,k (per head over the D x H grid)
        q_h = [self.rope.forward(q_h[h], D, H) for h in range(nh)]
        k_h = [self.rope.forward(k_h[h], D, H) for h in range(nh)]
        # ── SSD scan composed from device kernels (Python only marshals; the arithmetic is kernels) ──
        dev = self.dev if self.dev is not None else default_device()
        y = [[0] * d for _ in range(N)]
        for h in range(nh):
            kf = [k_h[h][t][j] for t in range(N) for j in range(hd)]     # flatten head (marshal)
            vf = [v_h[h][t][j] for t in range(N) for j in range(hd)]
            dbc = [delta[h][t] for t in range(N) for j in range(hd)]     # delta broadcast over hd
            state = dev.emul(dev.emul(kf, vf, FRAC), dbc, FRAC)          # delta*(k*v)  [kernel]
            if self.use_lattice2d:                                       # 2-D heat kernel (per head)
                hstate = self._lattice_dev(dev, state, D, H, h, N, hd)
            else:
                hstate = dev.colsum(state, N, hd) * N                    # VSSD sum, broadcast to N
            yf = dev.emul([q_h[h][t][j] for t in range(N) for j in range(hd)], hstate, FRAC)  # q*h
            base = h * hd
            for t in range(N):
                y[t][base:base + hd] = yf[t * hd:(t + 1) * hd]
        # RMSNorm per token (device kernel), 5-arm ADI gate, out_proj (device gemv)
        yn = [rmsnorm_fixed(y[t], self.norm_weight, FRAC, 1, dev=dev) for t in range(N)]
        if self.use_gate:
            gate = self._adi_gate([_to_q16(tok) for tok in tokens], dev)
            yn = [dev.emul(yn[t], gate[t], FRAC) for t in range(N)]
        return [self.out_proj((yn[t], -FRAC)) for t in range(N)]

    def _lattice_dev(self, dev, state, D, H, h, N, hd):
        """Per-head (D,H)-torus propagator on-device. GLOBAL heads (diff_steps<0) = Sum broadcast
        (t->inf, exact); LOCAL heads = N * diffuse(state, t steps) via the device kernels. state and
        return are flat [N*hd]. Bit-exact to the old _lattice_state (diffuse >>3, then *N)."""
        if self.diff_steps[h] < 0:
            return dev.colsum(state, N, hd) * N              # global sum, broadcast to N
        cur = state
        for _ in range(self.diff_steps[h]):
            cur = dev.diffuse(cur, D, H, hd)                 # one 4-neighbour heat step [kernel]
        return dev.escale(cur, N, 0)                         # * N (integer; frac=0 -> exact)

    def _adi_gate(self, x_tokens, dev):
        """5-arm ADI gate: route each token over center + R/M/P/C, gate the SSD output. softmax +
        sigmoid via device kernels; the 5-arm bias mix stays a tiny per-token sum. Volume knob, not ML."""
        out = []
        for x in x_tokens:
            w = softmax_fixed(_to_q16(self.gate_route((x, -FRAC))))          # [5], device softmax
            gp = _to_q16(self.gate_proj((x, -FRAC)))                          # [d], device gemv
            mix = [sum(_mul_q16(w[a], self.arm_bias[a][j]) for a in range(5))
                   for j in range(self.d_model)]                             # 5-arm bias (tiny)
            g = [gp[j] + mix[j] for j in range(self.d_model)]
            out.append(sigmoid_list(g, FRAC, dev=dev))                       # device sigmoid
        return out

    @property
    def raw(self):
        return {"d_model": self.d_model, "num_heads": self.num_heads,
                "in_proj": self.in_proj.raw, "out_proj": self.out_proj.raw, "rope_4d": self.rope_4d}


def _selftest():
    from ringkit.qcm.tensor import QSMLinear as QL
    D, H = 2, 3
    N = D * H
    dm, nh = C.D_MODEL, 4
    # small non-trivial weights so output isn't trivially zero
    ip = QL([[((r + k) % 5) - 2 for k in range(dm)] for r in range(3 * dm + nh)])
    op = QL([[((r * 3 + k) % 5) - 2 for k in range(dm)] for r in range(dm)])
    tokens = [([(t * 13 + j * 7) % 200 - 100 for j in range(dm)], 0) for t in range(N)]
    def valid(out):
        return (len(out) == N and all(len(v) == dm for v, _ in out)
                and all(isinstance(x, int) for v, _ in out for x in v))
    # default (non-causal VSSD)
    out = Mamba2SSD(d_model=dm, num_heads=nh, in_proj=ip, out_proj=op)(tokens, D, H)
    ok0 = valid(out)
    print(f"  Mamba2SSD (non-causal VSSD) forward valid={ok0}; out[0][:4]={out[0][0][:4]}")
    # lattice2d (2-D toroidal diffusion physics)
    out2 = Mamba2SSD(d_model=dm, num_heads=nh, use_lattice2d=True, in_proj=ip, out_proj=op)(tokens, D, H)
    ok1 = valid(out2)
    print(f"  Mamba2SSD (lattice2d diffusion) forward valid={ok1}; out[0][:4]={out2[0][0][:4]}")
    # 5-arm ADI gate
    gp = QL([[((r + k) % 5) - 2 for k in range(dm)] for r in range(dm)])
    gr = QL([[((r + k) % 3) - 1 for k in range(dm)] for r in range(5)])
    ab = [[((a + j) % 5) - 2 for j in range(dm)] for a in range(5)]
    out3 = Mamba2SSD(d_model=dm, num_heads=nh, use_gate=True, in_proj=ip, out_proj=op,
                     gate_proj=gp, gate_route=gr, arm_bias=ab)(tokens, D, H)
    ok2 = valid(out3)
    print(f"  Mamba2SSD (5-arm ADI gate) forward valid={ok2}; out[0][:4]={out3[0][0][:4]}")
    # lattice2d + gate together
    out4 = Mamba2SSD(d_model=dm, num_heads=nh, use_lattice2d=True, use_gate=True, in_proj=ip,
                     out_proj=op, gate_proj=gp, gate_route=gr, arm_bias=ab)(tokens, D, H)
    ok3 = valid(out4)
    print(f"  Mamba2SSD (lattice2d + gate) forward valid={ok3}; out[0][:4]={out4[0][0][:4]}")
    return ok0 and ok1 and ok2 and ok3


if __name__ == "__main__":
    print("ringkit.qcm.ssd self-test:")
    print("RESULT:", "ALL PASS" if _selftest() else "FAIL")
