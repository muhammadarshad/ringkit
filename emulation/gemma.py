"""
ringkit.emulation.gemma — EMULATE Gemma2-2B autoregressively on the ring. NO float on the compute
path, NO FPU. (Owner stance: for models we don't control we are an emulation engine; the FPU is
replaced by ringkit QCM primitives — energy-QSM GEMV, integer Taylor exp, ring isqrt, CORDIC.)

Faithful Gemma2 forward (config from hpq.h G2_*):
  h = embed[token] * 48                              (embed_scale = sqrt(2304))
  per layer (26):
    r = h;  x = RMSNorm(h, pre_attn) with (1+gamma)
    q,k,v = q/k/o via onix linears (energy-QSM kernel + power-of-2 dequant)
    RoPE(q,k) NeoX half-split, theta 1e4, full head_dim 256
    GQA attention (8 q / 4 kv heads), score soft-cap 50, causal over the KV cache
    a = o_proj(ctx);  h = r + RMSNorm(a, post_attn)
    r = h; x = RMSNorm(h, pre_mlp);  m = down(gelu(gate(x)) * up(x));  h = r + RMSNorm(m, post_ff)
  logits[v] = softcap30( dot(RMSNorm(h, final), embed[v]) )     (tied LM head)

Weights: onix (int8 linears, energy-QSM kernel), embed.bin (f16, mmap, row-streamed + tied LM head),
norms.bin (f16 gammas). RoPE inv_freq is GEOMETRIC (r=theta^(-1/128), r^i by ring-mul); cos/sin via
a ring CORDIC (shifts + adds + baked atan table) — multiplier-free, no runtime trig. Float-free.
"""
from ringkit.core import native as rn
from ringkit.emulation import ract, onix
from ringkit.kernels.mprc.gemma import host as _k

FRAC = 16
ONE = 1 << FRAC


class G2:
    layers = 26; hidden = 2304; inter = 9216; vocab = 256000
    n_q = 8; n_kv = 4; head_dim = 256; group = 2      # 8 q / 4 kv -> 2 q per kv
    theta_ln = 603609            # ln(10000) in Q16 (RoPE base)
    embed_scale = 48             # sqrt(2304), an exact integer
    logit_cap = 30; attn_cap = 50
    eos = 1; bos = 2


# ── CORDIC constants (baked, one-time; like the square table) ────────────────────
_ATAN = [51472, 30386, 16055, 8150, 4091, 2047, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
_CORDIC_GAIN = 39797                 # 1/K in Q16
_TWO_PI, _PI, _HALF_PI = 411775, 205887, 102944


def _cordic(theta):
    """(cos, sin) of theta (Q16) in Q16, via CORDIC rotation — shifts + adds only, no multiply."""
    while theta > _PI:
        theta = theta - _TWO_PI
    while theta < -_PI:
        theta = theta + _TWO_PI
    flip = False
    if theta > _HALF_PI:
        theta = _PI - theta; flip = True
    elif theta < -_HALF_PI:
        theta = -_PI - theta; flip = True
    x = _CORDIC_GAIN; y = 0; z = theta
    for k in range(len(_ATAN)):
        dx = x >> k; dy = y >> k
        if z >= 0:
            x = x - dy; y = y + dx; z = z - _ATAN[k]
        else:
            x = x + dy; y = y - dx; z = z + _ATAN[k]
    if flip:
        x = -x
    return x, y


def _f16_to_fixed(u, frac=FRAC):
    """Decode a uint16 IEEE half float to a signed Q<frac> integer by INTEGER bit ops (no FPU)."""
    sign = (u >> 15) & 1
    exp = (u >> 10) & 0x1F
    man = u & 0x3FF
    if exp == 0:
        sh = frac - 24
        val = (man << sh) if sh >= 0 else (man >> (-sh))
    elif exp == 0x1F:
        val = 0
    else:
        mant = (1 << 10) | man
        sh = frac + (exp - 15) - 10
        val = (mant << sh) if sh >= 0 else (mant >> (-sh))
    return -val if sign else val


def rope_tables(pos, head_dim=G2.head_dim, frac=FRAC):
    """cos/sin rows (length head_dim/2) for position `pos`. inv_freq is geometric (r^i, r=theta^-1/128),
    cos/sin via ring CORDIC. Float-free, multiplier-free."""
    half = head_dim >> 1
    r = ract.exp_fixed(-rn.mf_floordiv(G2.theta_ln, half), frac)   # theta^(-1/half) = e^(-ln theta/half)
    cos_row = [0] * half; sin_row = [0] * half
    invf = ONE                                                     # r^0 = 1  (Q16)
    for i in range(half):
        ang = rn.mul(pos, invf)                                    # pos * inv_freq_i  (Q16)
        c, s = _cordic(ang)
        cos_row[i] = c; sin_row[i] = s
        invf = rn.mul(invf, r) >> frac                             # r^(i+1)
    return cos_row, sin_row


def _sd(n, d):
    return -rn.mf_floordiv(-n, d) if n < 0 else rn.mf_floordiv(n, d)


def rmsnorm_g2(x, gamma, frac=FRAC):
    """Gemma RMSNorm: x/rms * (1 + gamma). gamma is Q<frac>; weight' = ONE + gamma."""
    w = [ONE + g for g in gamma]
    return ract.rmsnorm_fixed(x, w, frac)


def apply_rope(vec, cos_row, sin_row, frac=FRAC):
    """NeoX half-split RoPE: pair (i, i+half) rotated by (cos_i, sin_i). Ring-mul only."""
    d = len(vec); half = d >> 1
    out = [0] * d
    for i in range(half):
        v0 = vec[i]; v1 = vec[i + half]; c = cos_row[i]; s = sin_row[i]
        out[i] = (rn.mul(v0, c) >> frac) - (rn.mul(v1, s) >> frac)
        out[i + half] = (rn.mul(v0, s) >> frac) + (rn.mul(v1, c) >> frac)
    return out


def proj(tensor, x, frac=FRAC):
    """Gemma linear over an ONIX int8 tensor with power-of-2 activation quant, faithful dequant:
       out = dot(xbar-128, x_s8) * act_scale * 2^s_row / z_row, act_scale = 2^a a power of two.
    Everything is a shift or an integer divide; the dot runs on the energy-QSM kernel. Float-free.
    `tensor` = (xbar, s_row, z_row, out_feat, in_feat)."""
    xbar, s_row, z_row, of, inf = tensor
    # max |x|
    mx = 0
    for v in x:
        av = -v if v < 0 else v
        if av > mx:
            mx = av
    if mx == 0:
        return [0] * of
    # a = ceil(log2(max|x_real| / 127)) with x_real = mx/2^frac  ->  smallest a: 127<<(a+frac) >= mx
    a = 0
    if (127 << frac) >= mx:
        while (127 << (a - 1 + frac)) >= mx:
            a = a - 1
    else:
        while (127 << (a + frac)) < mx:
            a = a + 1
    sh = frac + a                       # x_s8[i] = round(x_i / 2^sh), clamp [-127,127]
    xs = []
    for v in x:
        if sh > 0:
            rnd = 1 << (sh - 1)
            q = (v + rnd) >> sh if v >= 0 else -((-v + rnd) >> sh)
        elif sh == 0:
            q = v
        else:
            q = v << (-sh)
        if q > 127:
            q = 127
        elif q < -127:
            q = -127
        xs.append(q)
    dots = _k.qsm_dot(xbar, xs, of, inf)          # int64 energy dots (kernel), no fold
    out = []
    for r in range(of):
        acc = dots[r]
        shift = a + s_row[r] + frac               # out_Qfrac = acc * 2^(a+s) / z * 2^frac
        acc = acc << shift if shift >= 0 else acc >> (-shift)
        z = z_row[r] if z_row[r] else 1
        out.append(_sd(acc, z))
    return out


def attention(qh, k_cache, v_cache, scale, frac=FRAC):
    """One query head over its cached (past+current) K/V. score = softcap50((q·k)/sqrt(hd)); softmax;
    ctx = Σ w·v. Causal is implicit (cache holds only positions ≤ current). Float-free."""
    from ringkit.emulation import infer
    scores = []
    for k in k_cache:
        s = infer.dot(qh, k, frac)
        s = _sd(rn.mul(s, scale), ONE)                 # * 1/sqrt(hd)
        s = ract.softcap_fixed(s, G2.attn_cap, frac)   # 50*tanh(s/50)
        scores.append(s)
    w = infer.softmax(scores, frac)
    dv = len(v_cache[0])
    acc = [0] * dv
    for j in range(len(v_cache)):
        wj = w[j]; vj = v_cache[j]
        for d in range(dv):
            acc[d] += rn.mul(wj, vj[d])
    return [x >> frac for x in acc]


def layer_forward(W, li, h, cache, cos_row, sin_row, scale, frac=FRAC):
    hd = G2.head_dim; nq = G2.n_q; nkv = G2.n_kv
    res = h
    x = rmsnorm_g2(h, W.norm(li, "pre_attn"), frac)
    q = proj(W.lin(li, "q_proj"), x, frac)             # [nq*hd]
    k = proj(W.lin(li, "k_proj"), x, frac)             # [nkv*hd]
    v = proj(W.lin(li, "v_proj"), x, frac)             # [nkv*hd]
    qh = [apply_rope(q[i * hd:(i + 1) * hd], cos_row, sin_row, frac) for i in range(nq)]
    kh = [apply_rope(k[i * hd:(i + 1) * hd], cos_row, sin_row, frac) for i in range(nkv)]
    vh = [v[i * hd:(i + 1) * hd] for i in range(nkv)]
    for j in range(nkv):
        cache["k"][j].append(kh[j]); cache["v"][j].append(vh[j])
    ctx = []
    for i in range(nq):
        kv = rn.mf_floordiv(i, G2.group)               # q head i -> kv head i//2
        ctx.extend(attention(qh[i], cache["k"][kv], cache["v"][kv], scale, frac))
    a = proj(W.lin(li, "o_proj"), ctx, frac)           # [hidden]
    a = rmsnorm_g2(a, W.norm(li, "post_attn"), frac)
    h = [res[i] + a[i] for i in range(G2.hidden)]
    res = h
    x = rmsnorm_g2(h, W.norm(li, "pre_mlp"), frac)
    g = proj(W.lin(li, "gate_proj"), x, frac)
    u = proj(W.lin(li, "up_proj"), x, frac)
    act = [rn.mul(ract.gelu_fixed(g[i], frac), u[i]) >> frac for i in range(G2.inter)]
    m = proj(W.lin(li, "down_proj"), act, frac)        # [hidden]
    m = rmsnorm_g2(m, W.norm(li, "post_ff"), frac)
    return [res[i] + m[i] for i in range(G2.hidden)]


def forward_token(W, token, pos, cache, frac=FRAC):
    """One decode step: embed the token, run 26 layers, return the final-normed hidden (Q<frac>)."""
    from ringkit.emulation import infer
    h = [rn.mul(e, G2.embed_scale) for e in W.embed_row(token)]
    cos_row, sin_row = rope_tables(pos, G2.head_dim, frac)
    scale = infer.inv_sqrt(G2.head_dim, frac)
    for li in range(G2.layers):
        h = layer_forward(W, li, h, cache[li], cos_row, sin_row, scale, frac)
    return rmsnorm_g2(h, W.final_norm(), frac)


def new_cache():
    return [{"k": [[] for _ in range(G2.n_kv)], "v": [[] for _ in range(G2.n_kv)]}
            for _ in range(G2.layers)]


def generate(W, prompt_ids, n_new, frac=FRAC, verbose=False):
    """Greedy autoregressive generation. Returns the full id list (prompt + generated). Float-free:
    the LM-head argmax uses the tied f16 embedding via the kernel (monotone soft-cap ⇒ argmax of dot)."""
    cache = new_cache()
    out = list(prompt_ids)
    pos = 0
    hn = None
    for t in prompt_ids:                       # prefill
        hn = forward_token(W, t, pos, cache, frac)
        pos = pos + 1
    for _ in range(n_new):
        nt, _score = W.lm_argmax(hn)
        out.append(nt)
        if verbose:
            print("  ->", nt, flush=True)
        if nt == G2.eos:
            break
        hn = forward_token(W, nt, pos, cache, frac)
        pos = pos + 1
    return out
