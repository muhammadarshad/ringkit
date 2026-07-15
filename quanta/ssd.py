"""Ring-native Mamba-2 State-Space-Duality layer (Soliton body), float-free.

Deployed config (gate_lat2d): non-causal + 2-D lattice heat-kernel + 5-arm ADI gate. The causal
`dA` (trapezoidal `A`) branch is NOT taken here, so `A_log` is unused — the state propagator is the
toroidal diffusion, not a cumulative decay.

    in_proj -> q,k,v,δ    δ = softplus(δ)     QuantumRoPE on q,k
    state = δ·(k⊙v)       h  = lattice_state(state)     y = q⊙h
    y = RMSNorm(y)        y *= sigmoid(gate(x))         out = out_proj(y)

lattice_state (per head over the D×H torus): GLOBAL heads = Σ_tokens (t→∞ limit, exact);
LOCAL heads = N · diffuse(state, t)  — t steps of mass-conserving 4-neighbour toroidal averaging
`(up+down+left+right+4·c)>>3`, the ring's own evolve, ×N so t→∞ recovers Σ (scale-matched).

softplus(x) = max(x,0) + ln(1+e^-|x|), with ln(1+u)=2·atanh(u/(2+u)) so the log only sees (1,2]
— fast series, no libm. Float-free: exp/sigmoid/layernorm (ract), shift-add MAC (infer), ring mul.
"""
from ringkit.core import native as rn
from ringkit.emulation import infer, ract
from ringkit.quanta._ringtrig import FRAC, ONE, _sd


def softplus_fixed(x, frac=FRAC):
    one = 1 << frac
    ax = -x if x < 0 else x
    lim = frac << frac                            # exact saturation: past frac·one, e^-|x| floors
    if ax > lim:                                  # to 0 in Q<frac> anyway -> softplus == max(x,0)
        ax = lim                                  # bit-identically; keeps exp off huge-arg bigints
    e = ract.exp_fixed(-ax, frac)                 # e^-|x| in (0, one]
    w = _sd(e << frac, (one << 1) + e)            # u/(2+u) in Q<frac>, <= 1/3
    w2 = rn.mul(w, w) >> frac
    term = w; acc = w                             # atanh(w) = w + w³/3 + w⁵/5 + ...
    for k in (3, 5, 7, 9, 11):
        term = rn.mul(term, w2) >> frac
        acc += rn.mf_floordiv(term, k)
    ln1p = acc << 1                               # 2·atanh
    return (x if x > 0 else 0) + ln1p


def _rot_half8(v):
    """QuantumRoPE4D rotate_half at head_dim=8: FOUR 2-dim ADI-axis chunks, each (-x1, x0).
    (The deployed 4D rope rotates within dim/4 chunks — NOT two 4-dim halves. The old halves
    form was self-consistently wrong vs the real torch model; caught by the webapp e2e anchor.)"""
    return [-v[1], v[0], -v[3], v[2], -v[5], v[4], -v[7], v[6]]


def _rope(vh, cosq, sinq, N, HD):
    out = []
    for t in range(N):
        v = vh[t]; rh = _rot_half8(v)
        out.append([_sd(rn.mul(v[d], cosq[t][d]), ONE) + _sd(rn.mul(rh[d], sinq[t][d]), ONE)
                    for d in range(HD)])
    return out


def _diffuse_step(cur, gh, gw, HD):
    """One mass-conserving 4-neighbour toroidal average over a gh×gw grid of HD-vectors, Q<frac>."""
    nxt = [[0] * HD for _ in range(gh * gw)]
    for r in range(gh):
        for cc in range(gw):
            i = r * gw + cc
            up = ((r - 1) % gh) * gw + cc; dn = ((r + 1) % gh) * gw + cc
            lf = r * gw + (cc - 1) % gw;   rt = r * gw + (cc + 1) % gw
            for d in range(HD):
                nxt[i][d] = (cur[up][d] + cur[dn][d] + cur[lf][d] + cur[rt][d] + (cur[i][d] << 2)) >> 3
    return nxt


def lattice_state(state, NH, N, HD, gh, gw, diff_steps):
    """state[hh] = [N][HD]  ->  h[hh] = [N][HD]. Global heads = Σ; local = N·diffuse(t)."""
    h = [None] * NH
    for hh in range(NH):
        if diff_steps[hh] < 0:                    # global: the t→∞ sum, broadcast to every token
            g = [sum(state[hh][t][d] for t in range(N)) for d in range(HD)]
            h[hh] = [g[:] for _ in range(N)]
    local = [hh for hh in range(NH) if diff_steps[hh] >= 0]
    if local:
        cur = {hh: [row[:] for row in state[hh]] for hh in local}
        for hh in local:
            if diff_steps[hh] == 0:
                h[hh] = [[rn.mul(cur[hh][t][d], N) for d in range(HD)] for t in range(N)]
        mt = max(diff_steps[hh] for hh in local)
        for step in range(1, mt + 1):
            for hh in local:
                cur[hh] = _diffuse_step(cur[hh], gh, gw, HD)
                if diff_steps[hh] == step:
                    h[hh] = [[rn.mul(cur[hh][t][d], N) for d in range(HD)] for t in range(N)]
    return h


def mamba2_ssd_layer(x, W, name, cosRq, sinRq, N, C, NH, HD, gh, gw):
    """One Soliton SSD block (gate_lat2d). `W(n)->Q<frac> weight`; `name` = layer weight prefix."""
    half = NH >> 1
    diff_steps = [-1 if hh < half else min(16, 1 << (hh - half)) for hh in range(NH)]
    proj = [infer.linear(r, W(name + "in_proj.weight"), W(name + "in_proj.bias"), 3 * C + NH, C, FRAC)
            for r in x]
    qh = [[proj[t][hh*HD:(hh+1)*HD] for t in range(N)] for hh in range(NH)]
    kh = [[proj[t][C + hh*HD: C + (hh+1)*HD] for t in range(N)] for hh in range(NH)]
    vh = [[proj[t][2*C + hh*HD: 2*C + (hh+1)*HD] for t in range(N)] for hh in range(NH)]
    dl = [[softplus_fixed(proj[t][3*C + hh], FRAC) for t in range(N)] for hh in range(NH)]
    qh = [_rope(qh[hh], cosRq, sinRq, N, HD) for hh in range(NH)]
    kh = [_rope(kh[hh], cosRq, sinRq, N, HD) for hh in range(NH)]
    state = [[[_sd(rn.mul(dl[hh][t], rn.mul(kh[hh][t][d], vh[hh][t][d]) >> FRAC), ONE)
               for d in range(HD)] for t in range(N)] for hh in range(NH)]
    h = lattice_state(state, NH, N, HD, gh, gw, diff_steps)
    y = [[rn.mul(qh[hh][t][d], h[hh][t][d]) >> FRAC for hh in range(NH) for d in range(HD)]
         for t in range(N)]
    y = [ract.rmsnorm_fixed(r, W(name + "norm.weight"), FRAC) for r in y]
    # 5-arm ADI gate on the layer INPUT x
    arm = W(name + "arm_bias")                    # flat 5*C
    gp_w = W(name + "gate_proj.weight"); gp_b = W(name + "gate_proj.bias")
    gr_w = W(name + "gate_route.weight"); gr_b = W(name + "gate_route.bias")
    gflat = []                                    # all tokens' gate args -> ONE sigmoid block call
    for t in range(N):
        route = infer.softmax(infer.linear(x[t], gr_w, gr_b, 5, C, FRAC), FRAC)   # [5] weights
        gp = infer.linear(x[t], gp_w, gp_b, C, C, FRAC)
        gflat.extend(gp[d] + sum(_sd(rn.mul(route[a], arm[a*C + d]), ONE) for a in range(5))
                     for d in range(C))
    sg = ract.sigmoid_list(gflat, FRAC)
    out = []
    for t in range(N):
        yt = [_sd(rn.mul(y[t][d], sg[t*C + d]), ONE) for d in range(C)]
        out.append(infer.linear(yt, W(name + "out_proj.weight"), W(name + "out_proj.bias"), C, C, FRAC))
    return out
