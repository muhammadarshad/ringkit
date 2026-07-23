"""Shared QCM physics front-end (native), used by Soliton/Rotor/Gluon: quadrant ring projection
then vacuum-depth embedding. Native port of transformer/projections.py — float-free, Z256.

Input `grid` here is the photon/wave field already in the Z256 ARC domain (ints 0..255) — one row
per (depth d, height h) position, W=D_MODEL columns. (The torch original took float[0,1] and did
arc=grid*256; the ring lives directly on the arc, so no scaling.)
"""
from ringkit.core import native as rn
from ringkit.qcm import constants as C
from ringkit.qcm.tensor import QSMLinear


def _phase_sum(arc):
    """COS(arc)+SIN(arc) as a signed integer (the ring-native 'cos_unit+sin_unit', SCALE=21)."""
    return rn._signed(rn.COS(arc)) + rn._signed(rn.SIN(arc))


class QuadrantRingProjector:
    """Unpack each Z256 arc into 4 rectified ring quadrants (COS+, SIN+, -COS+, -SIN+), inject the
    Y/W anti-stride modulation, and project 4*W -> embed_dim via a QSM linear.

        q0 =  max(COS(arc), 0)   q1 =  max(SIN(arc), 0)
        q2 =  max(-COS(arc),0)   q3 =  max(-SIN(arc),0)     (all signed, SCALE=21)

    proj : QSMLinear(in_dim*4 -> embed_dim). in_dim = W = D_MODEL = 128.
    """
    def __init__(self, in_dim=C.D_MODEL, embed_dim=C.D_MODEL, proj=None):
        self.in_dim = int(in_dim)
        self.embed_dim = int(embed_dim)
        self.quads = C.MPRC_QUADRANTS                      # 4
        # anti-stride modulation (512,) = outer(phase_q[4], phase_y[128]), signed ints
        arc_y = [(i * C.ANTI_Y) % 256 for i in range(self.in_dim)]
        arc_q = [(q * C.ANTI_W) % 256 for q in range(self.quads)]
        phase_y = [_phase_sum(a) for a in arc_y]           # (128,)
        phase_q = [_phase_sum(a) for a in arc_q]           # (4,)
        self.modulation = [rn.mul(phase_q[q], phase_y[i])
                           for q in range(self.quads) for i in range(self.in_dim)]  # (512,)
        # projection weights: caller supplies a fitted/loaded QSMLinear, else zero-init placeholder
        self.proj = proj if proj is not None else QSMLinear(
            [[0] * (self.in_dim * self.quads) for _ in range(self.embed_dim)])

    def _quads_of(self, arc):
        c = rn._signed(rn.COS(arc)); s = rn._signed(rn.SIN(arc))
        return (c if c > 0 else 0, s if s > 0 else 0,
                -c if -c > 0 else 0, -s if -s > 0 else 0)

    def __call__(self, grid_rows):
        """grid_rows: list of N rows, each W=in_dim Z256 ints. Returns list of N (ints, exp) tokens."""
        out = []
        pin = self.in_dim * self.quads                     # 512
        for row in grid_rows:
            feat = [0] * pin
            for w in range(self.in_dim):
                q0, q1, q2, q3 = self._quads_of(row[w] & 0xFF)
                # channel-major quadrant layout matches torch cat([q0,q1,q2,q3], -1) over W
                feat[w] = q0
                feat[self.in_dim + w] = q1
                feat[2 * self.in_dim + w] = q2
                feat[3 * self.in_dim + w] = q3
            modded = [rn.mul(feat[j], self.modulation[j]) for j in range(pin)]
            out.append(self.proj(modded))                  # (ints, exp)
        return out


class VacuumDepthEmbedding:
    """Add a per-depth-row embedding, except VACUUM rows (all-zero field = phase singularity /
    padding) which take the learned vacuum embedding instead. Native port of the torch module.

    depth_emb : list of D rows (each embed_dim ints) — the learned depth table.
    vacuum_emb: embed_dim ints — the learned vacuum token.
    """
    def __init__(self, embed_dim=C.D_MODEL, max_depth=C.D, depth_emb=None, vacuum_emb=None):
        self.embed_dim = int(embed_dim)
        self.max_depth = int(max_depth)
        self.depth_emb = depth_emb if depth_emb is not None else \
            [[0] * self.embed_dim for _ in range(self.max_depth)]
        self.vacuum_emb = vacuum_emb if vacuum_emb is not None else [0] * self.embed_dim

    def __call__(self, tokens, grid_rows, D, H):
        """tokens: N (ints, exp) from the projector. grid_rows: the N raw Z256 rows (for the vacuum
        test). Returns N (ints, exp) with the depth/vacuum embedding added in the token's int domain."""
        out = []
        for d in range(D):
            for h in range(H):
                idx = d * H + h
                vals, exp = tokens[idx]
                row = grid_rows[idx]
                is_vac = (sum(int(v) & 0xFF for v in row) == 0)   # all-zero field = vacuum
                emb = self.vacuum_emb if is_vac else self.depth_emb[d]
                # add embedding in the token's integer domain (emb given at exp=0 -> shift to match)
                if exp > 0:
                    half = 1 << (exp - 1)
                    added = [vals[j] + ((emb[j] + half) >> exp if emb[j] >= 0
                                        else -((-emb[j] + half) >> exp)) for j in range(self.embed_dim)]
                else:
                    added = [vals[j] + emb[j] for j in range(self.embed_dim)]
                out.append((added, exp))
        return out


def _selftest():
    D, H, W = 2, 3, C.D_MODEL                               # tiny grid; W must be D_MODEL for proj
    N = D * H
    # random-ish deterministic grid rows (Z256), one all-zero (vacuum) row
    grid = [[(d * 37 + h * 11 + w * 5) & 0xFF for w in range(W)] for d in range(D) for h in range(H)]
    grid[1] = [0] * W                                       # a vacuum row
    proj = QuadrantRingProjector(in_dim=W, embed_dim=C.D_MODEL)
    # give the projector a non-zero identity-ish weight so output isn't trivially 0
    W_rows = [[1 if (j % (W * 4)) == r else 0 for j in range(W * 4)] for r in range(C.D_MODEL)]
    proj.proj = QSMLinear(W_rows)
    tok = proj(grid)
    vde = VacuumDepthEmbedding(embed_dim=C.D_MODEL, max_depth=D)
    vde.vacuum_emb = [7] * C.D_MODEL
    out = vde(tok, grid, D, H)
    ok = (len(out) == N and all(len(v) == C.D_MODEL for v, _ in out))
    print(f"  frontend forward: {N} tokens x {C.D_MODEL} dims, valid={ok}")
    print(f"  sample token[0][:6]={out[0][0][:6]} exp={out[0][1]}")
    return ok


if __name__ == "__main__":
    print("ringkit.qcm.frontend self-test:")
    print("RESULT:", "ALL PASS" if _selftest() else "FAIL")
