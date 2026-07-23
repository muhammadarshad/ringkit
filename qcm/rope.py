"""QuantumRoPE — Z256 anti-stride rotary position embedding, native port of transformer/rope.py.

The rotary tables are the integer ring SIN/COS (SCALE=21) of anti-stride arcs; positions map to arcs
by the modular-inverse walk (no 2*pi/256, no float trig, no Euler). Rotation of a head-dim vector:

    y[d] = ( x[d]*cos[d] + rotate_half(x)[d]*sin[d] ) / SCALE      (SCALE=21; signed, rounded)

Trainable magnitude lives in q/k; the arc is the fixed positional unit. The 2-axis QuantumRoPE splits
head_dim into a Z(depth) half and an X(height) half; QuantumRoPE4D splits into 4 ADI-locked chunks.
"""
from ringkit.core import native as rn
from ringkit.qcm import constants as C

_SCALE = C.SCALE                                 # 21


def _sdiv_scale(n):
    """round(n / SCALE), signed, float-free (mf_floordiv on |n|)."""
    if n >= 0:
        return rn.mf_floordiv(n + (_SCALE >> 1), _SCALE)
    return -rn.mf_floordiv((-n) + (_SCALE >> 1), _SCALE)


class QuantumRoPE:
    """2-axis (Z depth / X height) rotary. dim = head_dim, must be divisible by 4."""
    def __init__(self, dim, max_z=32, max_x=128):
        assert dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.dim = int(dim)
        self.max_z, self.max_x = int(max_z), int(max_x)
        half = self.dim // 2
        qd = half // 2
        # cos/sin tables: [z][x] -> dim signed ints. arc layout = [Z,Z | X,X] halves.
        self.cos = [[None] * self.max_x for _ in range(self.max_z)]
        self.sin = [[None] * self.max_x for _ in range(self.max_z)]
        for z in range(self.max_z):
            az = (z * C.ANTI_U) % 256
            for x in range(self.max_x):
                ax = (x * C.ANTI_X) % 256
                arc = [az] * qd + [az] * qd + [ax] * qd + [ax] * qd   # (dim,)
                self.cos[z][x] = [rn._signed(rn.COS(a)) for a in arc]
                self.sin[z][x] = [rn._signed(rn.SIN(a)) for a in arc]

    def _rotate_half(self, x):
        h = self.dim // 2
        q = h // 2
        xz, xx = x[:h], x[h:]
        rot_z = [-v for v in xz[q:]] + xz[:q]
        rot_x = [-v for v in xx[q:]] + xx[:q]
        return rot_z + rot_x

    def rotate(self, vec, z, x):
        cos = self.cos[z][x]; sin = self.sin[z][x]; rh = self._rotate_half(vec)
        return [_sdiv_scale(rn.mul(vec[d], cos[d]) + rn.mul(rh[d], sin[d])) for d in range(self.dim)]

    def forward(self, vecs, grid_z, grid_x):
        """vecs: N head-dim int vectors in token order n -> (z=n//grid_x, x=n%grid_x)."""
        return [self.rotate(vecs[n], n // grid_x, n % grid_x) for n in range(len(vecs))]


class QuantumRoPE4D:
    """Level-2 ADI 4-axis rotary. dim = head_dim, divisible by 8. Four chunks [R,M,P,C] with arcs
    a1=(z*ANTI_U), a2=(x*ANTI_X), delta1=a1-a2, a3=a1-(delta1+3), a4=a1-(delta1+8) (ADI recovery)."""
    def __init__(self, dim, max_z=32, max_x=128):
        assert dim % 8 == 0, "head_dim must be divisible by 8 for 4-axis ADI RoPE"
        self.dim = int(dim)
        self.max_z, self.max_x = int(max_z), int(max_x)
        per = self.dim // 4
        qd = per // 2
        self.cos = [[None] * self.max_x for _ in range(self.max_z)]
        self.sin = [[None] * self.max_x for _ in range(self.max_z)]
        for z in range(self.max_z):
            a1 = (z * C.ANTI_U) % 256
            for x in range(self.max_x):
                a2 = (x * C.ANTI_X) % 256
                d1 = (a1 - a2) % 256
                a3 = (a1 - (d1 + 3)) % 256
                a4 = (a1 - (d1 + 8)) % 256
                arc = ([a1] * qd + [a1] * qd + [a2] * qd + [a2] * qd
                       + [a3] * qd + [a3] * qd + [a4] * qd + [a4] * qd)  # (dim,)
                self.cos[z][x] = [rn._signed(rn.COS(a)) for a in arc]
                self.sin[z][x] = [rn._signed(rn.SIN(a)) for a in arc]

    def _rotate_half(self, x):
        c = self.dim // 4
        out = []
        for i in range(4):
            xc = x[i * c:(i + 1) * c]
            out += [-v for v in xc[c // 2:]] + xc[:c // 2]
        return out

    def rotate(self, vec, z, x):
        cos = self.cos[z][x]; sin = self.sin[z][x]; rh = self._rotate_half(vec)
        return [_sdiv_scale(rn.mul(vec[d], cos[d]) + rn.mul(rh[d], sin[d])) for d in range(self.dim)]

    def forward(self, vecs, grid_z, grid_x):
        return [self.rotate(vecs[n], n // grid_x, n % grid_x) for n in range(len(vecs))]


def mprc_grid_size(N):
    """(grid_h, grid_w) from sequence length N for standard MPRC shapes (native port)."""
    if C.L <= N <= C.L + 4:
        return C.D, C.H                          # 16, 113
    if N in (56, 60):
        return 7, 8
    if N in (64, 68):
        return 8, 8
    sq = rn.isqrt(N)
    return sq, rn.mf_floordiv(N, sq)


def _selftest():
    ok = True
    hd = 32
    rope = QuantumRoPE(hd, max_z=8, max_x=8)
    v = [((i * 7) % 41) - 20 for i in range(hd)]             # signed test vector
    y = rope.rotate(v, 0, 0)                                  # z=0,x=0 -> arcs 0 -> cos peak, sin 0
    # at (0,0): all arcs 0 -> COS=21, SIN=0 -> y = round(v*21/21) = v (identity rotation)
    id_ok = (y == v)
    print(f"  QuantumRoPE at pos(0,0) is identity: {'PASS' if id_ok else f'FAIL {y[:4]} vs {v[:4]}'}")
    yb = rope.forward([v, v, v, v], 2, 2)
    fwd_ok = (len(yb) == 4 and all(len(r) == hd for r in yb))
    print(f"  QuantumRoPE.forward over 4 tokens: {'PASS' if fwd_ok else 'FAIL'}")
    r4 = QuantumRoPE4D(hd, max_z=8, max_x=8)
    y4 = r4.rotate(v, 0, 0)                                   # (0,0)-> a1=a2=0,d1=0,a3=-3,a4=-8
    ok4 = (len(y4) == hd)
    print(f"  QuantumRoPE4D.rotate valid: {'PASS' if ok4 else 'FAIL'}")
    gs = mprc_grid_size(1808)
    grid_ok = (gs == (16, 113))
    print(f"  mprc_grid_size(1808)={gs}: {'PASS' if grid_ok else 'FAIL'}")
    return id_ok and fwd_ok and ok4 and grid_ok


if __name__ == "__main__":
    print("ringkit.qcm.rope self-test:")
    print("RESULT:", "ALL PASS" if _selftest() else "FAIL")
