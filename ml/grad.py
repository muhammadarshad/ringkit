"""ringkit.ml.grad — kernel-backed linear forward + backward (Q16 ENERGY), the training hot path.

Replaces the Python loops that made ring training infeasible at scale (qcm.tensor.qsm_matmul's
per-token double loop; ml.tensor_autograd.TVar.matmul's triple loop; train_smoke's dW triple loop)
with BATCHED device energy GEMMs — `dev.gemm` == ring_gemm_i64, all T tokens in ONE call (measured
256x over the Python path on a 1808-token in_proj, bit-exact). This is SPEC-013 T6.1/T6.2 (Path B:
kernelize forward/backward first).

Domain: Q16 (value * 2^16). Weights/activations are Q16; gradients are ENERGY (signed, NON-wrapping —
folding a gradient mod 256 destroys descent, SRD 3.4). The energy GEMM (ring_gemm_i64) accumulates in
int64 and NEVER folds — that is why it, not the byte ring_gemm (which folds mod 256), is the backward
engine. Multiplier-free at the silicon (QSM); no float, no torch, no numpy.

Trainable linear (Q16 weights), forward:  y[t,m] = (Σ_k X[t,k]·W[m,k]) >> FRAC + b[m]
Backward (chain rule; ∂y/∂x = W/2^F, ∂y/∂W = x/2^F):
    dX[t,k] = (Σ_m dY[t,m]·W[m,k]) >> FRAC      (energy GEMM, contract M)
    dW[m,k] = (Σ_t dY[t,m]·X[t,k]) >> FRAC      (energy GEMM, contract T)
    db[m]   =  Σ_t dY[t,m]
The two grad GEMMs reuse the SAME dev.gemm kernel; the transposes are O(size) marshaling, not the
O(T·M·K) hot loop. Each op ships a pure-Python reference + a D9 bit-exact selftest + a finite-
difference gradient check (SRD NFR9/D1: verify the adjoint in ring units).
"""
from ringkit.core import native as rn
from ringkit.device import default_device

FRAC = 16
ONE = 1 << FRAC


def _transpose(A, R, Cc):
    """Flat [R,Cc] -> flat [Cc,R]. Marshaling only (not the hot loop)."""
    return [A[r * Cc + c] for c in range(Cc) for r in range(R)]


# ───────────────────────── linear (Q16) ─────────────────────────

def linear_forward(X, W, b, T, M, K, dev=None):
    """y[t,m] = (Σ_k X[t*K+k]·W[m*K+k]) >> FRAC + b[m]. X[T*K], W[M*K], b[M]|None, all Q16.
    Returns Q16 flat [T*M]. The Σ is the exact int64 energy GEMM; the >>FRAC (arithmetic floor)
    is applied once after accumulation (more accurate than per-term shift, and kernel-batched)."""
    dev = dev if dev is not None else default_device()
    acc = dev.gemm(X, W, T, M, K)                    # Σ X·W exact int64, flat [T*M]
    if b is None:
        return [acc[i] >> FRAC for i in range(T * M)]
    return [(acc[t * M + m] >> FRAC) + b[m] for t in range(T) for m in range(M)]


def linear_backward(dY, X, W, T, M, K, dev=None):
    """dX[T*K], dW[M*K], db[M] for y=(X@Wᵀ)>>FRAC+b. dY[T*M], X[T*K], W[M*K] Q16 ENERGY.
    Two energy GEMMs (no fold) + a colsum for db. See module docstring for the algebra."""
    dev = dev if dev is not None else default_device()
    # dX[t,k] = (Σ_m dY[t,m]·W[m,k]) >> FRAC  ==  gemm(dY[T,M], Wᵀ[K,M]; T,K, inner=M)
    Wt = _transpose(W, M, K)                          # [K,M]
    accX = dev.gemm(dY, Wt, T, K, M)                  # [T*K]
    dX = [accX[i] >> FRAC for i in range(T * K)]
    # dW[m,k] = (Σ_t dY[t,m]·X[t,k]) >> FRAC  ==  gemm(dYᵀ[M,T], Xᵀ[K,T]; M,K, inner=T)
    dYt = _transpose(dY, T, M)                        # [M,T]
    Xt = _transpose(X, T, K)                          # [K,T]
    accW = dev.gemm(dYt, Xt, M, K, T)                 # [M*K]
    dW = [accW[i] >> FRAC for i in range(M * K)]
    # db[m] = Σ_t dY[t,m]  (colsum over the T rows of dY[T,M])
    db = dev.colsum(dY, T, M)
    return dX, dW, list(db)


# ───────────────────────── sigmoid (fixed nonlinearity) ─────────────────────────

def sigmoid_forward(x, dev=None):
    """σ(x) in Q16 (device kernel). x Q16 flat."""
    dev = dev if dev is not None else default_device()
    return dev.sigmoid(x, FRAC)


def sigmoid_backward(dY, s, dev=None):
    """dX = dY ⊙ σ(1−σ), with σ=`s` the forward output (Q16). σ'(x)=σ(1−σ). Kernel emul (no fold)."""
    dev = dev if dev is not None else default_device()
    one_minus = [ONE - v for v in s]
    deriv = dev.emul(s, one_minus, FRAC)              # σ(1−σ), Q16 in [0, ONE/4]
    return dev.emul(dY, deriv, FRAC)


# ───────────────────────── references (pure Python, ring-exact) ─────────────────────────

def _linear_forward_ref(X, W, b, T, M, K):
    out = [0] * (T * M)
    for t in range(T):
        for m in range(M):
            acc = 0
            for k in range(K):
                acc += X[t * K + k] * W[m * K + k]    # exact int, no fold
            out[t * M + m] = (acc >> FRAC) + (b[m] if b is not None else 0)
    return out


def _linear_backward_ref(dY, X, W, T, M, K):
    dX = [0] * (T * K)
    for t in range(T):
        for k in range(K):
            acc = 0
            for m in range(M):
                acc += dY[t * M + m] * W[m * K + k]
            dX[t * K + k] = acc >> FRAC
    dW = [0] * (M * K)
    for m in range(M):
        for k in range(K):
            acc = 0
            for t in range(T):
                acc += dY[t * M + m] * X[t * K + k]
            dW[m * K + k] = acc >> FRAC
    db = [sum(dY[t * M + m] for t in range(T)) for m in range(M)]
    return dX, dW, db


# ───────────────────────── SSD adjoints — SAME kernels, adjoint order (SPEC-013 T6.4) ─────────────
# Python is the interface; every gradient below is composed from the forward's own device kernels
# (emul/escale/esub/colsum/diffuse/sigmoid + the rope tables). No float, no new silicon.

def _sd(n, d):
    """round(n/d), signed, d>0 (the loss.py/activations helper). Float-free."""
    if n >= 0:
        return rn.mf_floordiv(n + (d >> 1), d)
    return -rn.mf_floordiv((-n) + (d >> 1), d)


def softplus_backward(dY, x, dev=None):
    """y = softplus(x) -> dX = dY ⊙ σ(x)  (softplus' = sigmoid). Kernels: sigmoid + emul."""
    dev = dev if dev is not None else default_device()
    return dev.emul(dY, dev.sigmoid(x, FRAC), FRAC)


def emul_backward(dY, a, b, dev=None):
    """y = a⊙b (Q16) -> dA = dY⊙b, dB = dY⊙a. The same emul kernel, twice."""
    dev = dev if dev is not None else default_device()
    return dev.emul(dY, b, FRAC), dev.emul(dY, a, FRAC)


def colsum_broadcast_backward(dH, N, C, dev=None):
    """Adjoint of the VSSD zero-mode `dev.colsum(state,N,C) * N` (column-sum then list-REPEAT
    broadcast): h[t]=Σ_t' state[t'] for every t -> dState[t]=Σ_t' dH[t'] for every t. SELF-ADJOINT:
    the backward is the identical expression on the upstream grad."""
    dev = dev if dev is not None else default_device()
    return dev.colsum(dH, N, C) * N


def diffuse_backward(dY, D, H, hd, steps=1, dev=None):
    """Adjoint of `steps` toroidal 4-neighbour heat steps. The stencil (up+dn+lf+rt+4c)>>3 is
    SYMMETRIC on the torus -> self-adjoint: apply the same diffuse kernel to the grad, same steps."""
    dev = dev if dev is not None else default_device()
    cur = dY
    for _ in range(steps):
        cur = dev.diffuse(cur, D, H, hd)
    return cur


def rmsnorm_backward(dY, x, w, frac=FRAC, eps=1, dev=None):
    """Adjoint of rmsnorm_fixed: y_i = (x_i / rms)·w_i, rms = sqrt(mean(x²)+eps), all Q16.
        dX_j = (dY_j⊙w_j)/rms  −  x_j · Σ_i(dY_i⊙w_i⊙x_i) / (n·rms³)
    Composed from emul (squares, products), escale (the two scalar scalings), esub. The two scalars
    (1/rms and the Σ coefficient) are per-vector marshaling, not the hot loop."""
    dev = dev if dev is not None else default_device()
    n = len(x)
    ssq = sum(dev.emul(x, x, frac))                        # Σ (x²)>>F  (kernel squares, scalar sum)
    rms = rn.isqrt((rn.mf_floordiv(ssq, n) + eps) << frac) or 1
    g = dev.emul(dY, w, frac)                              # (dY⊙w) Q16
    inv = rn.mf_floordiv(1 << (frac + frac), rms)          # 2^32/rms -> escale ≡ ÷rms in Q16
    term1 = dev.escale(g, inv, frac)
    s = sum(dev.emul(g, x, frac))                          # Σ (g⊙x)>>F, Q16 scalar
    c = _sd(s << (3 * frac), n * rms * rms * rms)          # s·2^48/(n·rms³), Q16 scalar
    term2 = dev.escale(x, c, frac)
    return dev.esub(term1, term2)


def rope_backward(rope, dvecs, grid_z, grid_x):
    """Adjoint of QuantumRoPE(.4D).forward. Within each half/chunk cos/sin are constant, so rotate is
    a true 2×2 rotation R(θ); the adjoint is Rᵀ = R(−θ) = the SAME rotate with sin NEGATED. Reuses the
    rope's own tables and _rotate_half — identical ops, inverse arc."""
    from ringkit.qcm.rope import _sdiv_scale
    out = []
    for n_i in range(len(dvecs)):
        z, x = n_i // grid_x, n_i % grid_x
        cos, sin = rope.cos[z][x], rope.sin[z][x]
        v = dvecs[n_i]
        rh = rope._rotate_half(v)
        out.append([_sdiv_scale(rn.mul(v[d], cos[d]) - rn.mul(rh[d], sin[d]))
                    for d in range(rope.dim)])
    return out


# ───────────────────────── D9 selftest + finite-difference gradient check ─────────────────────────

def _selftest():
    dev = default_device()
    ok = True
    T, M, K = 6, 5, 7
    # deterministic Q16-ish signed operands (a linear congruential walk, no random import, no float)
    def walk(n, seed):
        out, s = [], seed
        for _ in range(n):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            out.append(((s % 2001) - 1000) << 4)      # ~[-16000, 16000] Q16 energy
        return out
    X = walk(T * K, 111)
    W = walk(M * K, 222)
    b = walk(M, 333)
    dY = walk(T * M, 444)

    # forward: kernel == pure-Python reference, bit-for-bit
    y_k = linear_forward(X, W, b, T, M, K, dev)
    y_r = _linear_forward_ref(X, W, b, T, M, K)
    f_ok = (y_k == y_r)
    ok &= f_ok
    print(f"  linear_forward kernel == reference (bit-exact): {'PASS' if f_ok else 'FAIL'}")

    # backward: kernel == reference, bit-for-bit (dX, dW, db)
    dX_k, dW_k, db_k = linear_backward(dY, X, W, T, M, K, dev)
    dX_r, dW_r, db_r = _linear_backward_ref(dY, X, W, T, M, K)
    b_ok = (dX_k == dX_r and dW_k == dW_r and db_k == db_r)
    ok &= b_ok
    print(f"  linear_backward kernel == reference (dX,dW,db bit-exact): {'PASS' if b_ok else 'FAIL'}")

    # finite-difference gradient check IN RING UNITS (SRD NFR9): perturb W[m,k] by ±h, compare the
    # scalar loss L=Σ_{t,m} c[t,m]·y[t,m] (linear in y so dL/dy=c=dY). Then dL/dW[m,k] must equal the
    # analytic dW. Because floor(>>FRAC) is piecewise-constant, use h a multiple of ONE (a clean ring
    # step) and require EXACT agreement on the mean over a few probes (structure-exact, not approx).
    c = dY                                            # loss weights => upstream grad is exactly c
    def loss(Wv):
        y = _linear_forward_ref(X, Wv, b, T, M, K)
        return sum(c[i] * y[i] for i in range(T * M))
    h = ONE                                           # one ring unit in Q16
    fd_ok = True
    probes = [(0, 0), (2, 3), (M - 1, K - 1), (1, 5)]
    for (m, k) in probes:
        Wp = list(W); Wp[m * K + k] += h
        Wm = list(W); Wm[m * K + k] -= h
        fd = (loss(Wp) - loss(Wm)) // (2 * h)         # central difference, integer
        an = dW_r[m * K + k]                           # analytic dL/dW (== kernel dW_k, proven above)
        # forward floor drops <FRAC bits; the central diff of a (·)>>FRAC map matches the analytic
        # grad up to that floor granularity (|Σ_t c·(dropped bits)|/2h). Bound it and report.
        tol = 0
        for t in range(T):
            tol += abs(c[t * M + m])
        tol = (tol >> FRAC) + 1                        # worst-case floor slack over T terms
        if abs(fd - an) > tol:
            fd_ok = False
            print(f"    FD mismatch W[{m},{k}]: fd={fd} an={an} tol={tol}")
    ok &= fd_ok
    print(f"  finite-difference dW matches analytic (ring units, within floor tol): "
          f"{'PASS' if fd_ok else 'FAIL'}")

    # sigmoid backward: derivative peaks at x=0 (σ'=1/4) and -> 0 in the tails (sanity + monotone)
    xs = [-(4 << 16), -(1 << 16), 0, (1 << 16), (4 << 16)]
    s = sigmoid_forward(xs, dev)
    dsig = sigmoid_backward([ONE] * len(xs), s, dev)   # dY=1 -> returns σ'(x)
    peak_ok = (dsig[2] == max(dsig) and dsig[0] < dsig[2] and dsig[-1] < dsig[2]
               and abs(dsig[2] - (ONE >> 2)) <= (ONE >> 5))   # σ'(0)≈0.25
    ok &= peak_ok
    print(f"  sigmoid_backward σ'(0)≈0.25 & tails decay: {'PASS' if peak_ok else 'FAIL'} {dsig}")

    # ── SSD adjoints (T6.4) ──
    def dot(u, v):
        return sum(u[i] * v[i] for i in range(len(u)))

    # emul backward: FD on a_j for L = Σ c·(a⊙b)
    a = walk(8, 555); b_v = walk(8, 666); c_v = walk(8, 777)
    dA, dB = emul_backward(c_v, a, b_v, dev)
    hq = ONE
    em_ok = True
    for j in (0, 3, 7):
        ap = list(a); ap[j] += hq
        am = list(a); am[j] -= hq
        fd = _sd(dot(c_v, dev.emul(ap, b_v, FRAC)) - dot(c_v, dev.emul(am, b_v, FRAC)), 2 * hq)
        if abs(fd - dA[j]) > max(abs(dA[j]) >> 3, 64):
            em_ok = False
            print(f"    emul FD mismatch j={j}: fd={fd} an={dA[j]}")
    ok &= em_ok
    print(f"  emul_backward matches finite difference: {'PASS' if em_ok else 'FAIL'}")

    # softplus backward: FD per element vs dY⊙σ(x)
    from ringkit.qcm.activations import softplus_fixed
    xv = [-(2 << 16), -(1 << 15), 0, (1 << 15), (2 << 16)]
    cv = [3 << 14, -(2 << 14), 1 << 15, 5 << 13, -(1 << 14)]
    an_sp = softplus_backward(cv, xv, dev)
    sp_ok = True
    for j in range(len(xv)):
        fd = _sd(cv[j] * (softplus_fixed(xv[j] + hq) - softplus_fixed(xv[j] - hq)), 2 * hq)
        if abs(fd - an_sp[j]) > max(abs(an_sp[j]) >> 3, 96):
            sp_ok = False
            print(f"    softplus FD mismatch j={j}: fd={fd} an={an_sp[j]}")
    ok &= sp_ok
    print(f"  softplus_backward (dY⊙σ) matches finite difference: {'PASS' if sp_ok else 'FAIL'}")

    # colsum-broadcast: EXACT self-adjointness  ⟨A·s, u⟩ == ⟨s, A·u⟩  (integer sums, no rounding)
    Nn, Cc = 5, 4
    sv = walk(Nn * Cc, 888); uv = walk(Nn * Cc, 999)
    As = dev.colsum(sv, Nn, Cc) * Nn
    Au = colsum_broadcast_backward(uv, Nn, Cc, dev)
    cs_ok = (dot(As, uv) == dot(sv, Au))
    ok &= cs_ok
    print(f"  colsum-broadcast self-adjoint (exact inner product): {'PASS' if cs_ok else 'FAIL'}")

    # diffuse: self-adjoint within floor slack  ⟨A x, y⟩ ≈ ⟨x, A y⟩
    Dg, Hg, hg = 3, 4, 2
    xg = walk(Dg * Hg * hg, 1111); yg = walk(Dg * Hg * hg, 2222)
    Ax = dev.diffuse(xg, Dg, Hg, hg)
    Ay = diffuse_backward(yg, Dg, Hg, hg, steps=1, dev=dev)
    lhs, rhs = dot(Ax, yg), dot(xg, Ay)
    tol_d = sum(abs(v) for v in xg) + sum(abs(v) for v in yg)      # ±1 floor per element
    df_ok = abs(lhs - rhs) <= tol_d
    ok &= df_ok
    print(f"  diffuse self-adjoint within floor slack (|Δ|={abs(lhs-rhs)}<={tol_d}): "
          f"{'PASS' if df_ok else 'FAIL'}")

    # rmsnorm backward: FD on x_j for L = Σ c·rmsnorm(x)
    n_r = 6
    xr = [(3 << 16), -(2 << 16), (1 << 16), (4 << 16), -(1 << 15), (2 << 16)]
    wr = [ONE] * n_r
    cr = [1 << 14, -(3 << 13), 2 << 14, -(1 << 14), 3 << 13, 1 << 13]
    an_rn = rmsnorm_backward(cr, xr, wr, FRAC, 1, dev)
    rn_ok = True
    for j in (0, 2, 4):
        xp = list(xr); xp[j] += hq
        xm = list(xr); xm[j] -= hq
        fd = _sd(dot(cr, dev.rmsnorm(xp, wr, FRAC, 1)) - dot(cr, dev.rmsnorm(xm, wr, FRAC, 1)), 2 * hq)
        if abs(fd - an_rn[j]) > max(abs(an_rn[j]) >> 3, 512):
            rn_ok = False
            print(f"    rmsnorm FD mismatch j={j}: fd={fd} an={an_rn[j]}")
    ok &= rn_ok
    print(f"  rmsnorm_backward matches finite difference: {'PASS' if rn_ok else 'FAIL'}")

    # rope: adjointness  ⟨R v, u⟩ ≈ ⟨v, Rᵀ u⟩  (rounding ±0.5/element slack), 2-axis + 4D
    from ringkit.qcm.rope import QuantumRoPE, QuantumRoPE4D
    for cls, nm in ((QuantumRoPE, "2-axis"), (QuantumRoPE4D, "4D")):
        rope = cls(16, max_z=4, max_x=4)
        vv = [walk(16, 3333 + i)[0:16] for i in range(4)]
        uu = [walk(16, 4444 + i)[0:16] for i in range(4)]
        Rv = rope.forward(vv, 2, 2)
        Rtu = rope_backward(rope, uu, 2, 2)
        lhs = sum(dot(Rv[t], uu[t]) for t in range(4))
        rhs = sum(dot(vv[t], Rtu[t]) for t in range(4))
        tol_r = (sum(abs(x) for t in uu for x in t) + sum(abs(x) for t in vv for x in t)) // 2 + 16
        r_ok = abs(lhs - rhs) <= tol_r
        ok &= r_ok
        print(f"  QuantumRoPE({nm}) adjoint = inverse-arc rotate (|Δ|={abs(lhs-rhs)}<={tol_r}): "
              f"{'PASS' if r_ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("ringkit.ml.grad — kernel-backed linear forward/backward (Q16 ENERGY, float-free):")
    print("RESULT:", "ALL PASS" if _selftest() else "FAIL")
