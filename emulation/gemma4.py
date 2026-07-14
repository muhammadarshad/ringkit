"""
ringkit.emulation.gemma4 — EMULATE Gemma4-12B (gemma4_unified) autoregressively on the ring. NO
float on the compute path, NO FPU. Companion to `gemma.py` (Gemma2-2B); the pure ring `nn` stack is
untouched. Config from the HF `text_config` / mlx `gemma4_unified` reference (read as a spec only).

Gemma4 differs from Gemma2 in ways that are NOT cosmetic (each verified against the real 12B onix):

  * 48 layers, hidden 3840, inter 15360, vocab 262144, 16 Q heads.
  * RMSNorm has NO (1+gamma) offset — gamma is applied straight (`rmsnorm_fixed`, weight = gamma).
  * Sliding/global alternation: layer is GLOBAL iff (li % 6 == 5) → layers 5,11,…,47.
      local : 8 KV heads, head_dim 256, rope theta 1e4, FULL rotation (256).
      global: 1 KV head,  head_dim 512, rope theta 1e6, PARTIAL rotation (128 of 512) — and NO
              v_proj (attention_k_eq_v): V is a copy of the RAW k projection (before k-norm/RoPE).
  * Per-head Q-norm and K-norm (learnable gamma, NO offset) applied BEFORE RoPE; per-head V-norm
    (RMSNorm, NO scale) applied to V.
  * Attention scale = 1.0 and NO attention soft-cap (the per-head QK norms bound the magnitude).
  * FFN gate uses gelu_pytorch_tanh (the tanh approximation), NOT the sigmoid/fast GELU.
  * Per-layer learned residual scalar (`layer_scalars.bin`) multiplies the whole hidden at layer end.
  * Proportional RoPE (global): the freq exponent divides by the FULL head_dim (512) and the NeoX
    pair offset is head_dim/2 (256); only the first rot_dim/2 (64) pairs are rotated, the rest pass
    through. (For local layers this reduces to standard full-rotation NeoX RoPE.)
  * Output logit soft-cap 30 (monotone ⇒ argmax of the raw tied-f16 dot). EOS ∈ {1, 107}.

Reuses the proven Gemma2 leaves: `proj` (energy-QSM GEMV + power-of-2 dequant), `_f16_to_fixed`,
`_cordic`. RMSNorm/exp/tanh come from `ract`; dot/softmax from `infer`. Float-free, multiplier-free.

Scope: no sliding-window eviction (the demo prompts are ≪ 1024 tokens, so a growing cache matches
the reference exactly). Add windowing before exceeding 1024 positions.
"""
import ctypes
from ringkit.core import native as rn
from ringkit.emulation import ract, infer
from ringkit.emulation.gemma import _f16_to_fixed, _cordic, proj, FRAC, ONE
from ringkit.kernels.mprc.gemma import host as _kh

__all__ = ["G4", "gelu_tanh_fixed", "rope_tables", "apply_rope", "rmsnorm_g4",
           "layer_forward", "forward_token", "generate", "new_cache"]


class G4:
    layers = 48; hidden = 3840; inter = 15360; vocab = 262144
    n_q = 16
    head_dim_local = 256; head_dim_global = 512
    n_kv_local = 8; n_kv_global = 1
    global_period = 6                      # layer is global iff li % 6 == 5
    rot_local = 256; rot_global = 128      # rotated dims (partial for global)
    theta_ln_local = 603609                # ln(1e4) in Q16
    theta_ln_global = 905414               # ln(1e6) in Q16
    logit_cap = 30
    eos = (1, 107); bos = 2


def _is_global(li):
    """Layer is global-attention iff li % 6 == 5 (multiplier-free via mf_mod)."""
    return rn.mf_mod(li, G4.global_period) == G4.global_period - 1


def layer_geom(li):
    """(is_global, head_dim, n_kv, rot_dim, theta_ln, group) for layer li."""
    g = _is_global(li)
    if g:
        return True, G4.head_dim_global, G4.n_kv_global, G4.rot_global, G4.theta_ln_global, \
            rn.mf_floordiv(G4.n_q, G4.n_kv_global)
    return False, G4.head_dim_local, G4.n_kv_local, G4.rot_local, G4.theta_ln_local, \
        rn.mf_floordiv(G4.n_q, G4.n_kv_local)


# ── activations ──────────────────────────────────────────────────────────────
_SQRT_2_PI = 7978846        # sqrt(2/pi)   * 1e7  (rational integer, no float)
_TEN7 = 10000000
_GELU_C = 44715             # 0.044715     * 1e6
_MILLION = 1000000


def _sdiv(n, d):
    return -rn.mf_floordiv(-n, d) if n < 0 else rn.mf_floordiv(n, d)


def gelu_tanh_fixed(x, frac=FRAC):
    """gelu_pytorch_tanh in Q<frac>: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))). Float-free.
    Coefficients are rational integers (√(2/π)=7978846/1e7, 0.044715=44715/1e6)."""
    x2 = rn.mul(x, x) >> frac
    x3 = rn.mul(x2, x) >> frac                       # x³  (Q<frac>, signed)
    cube = _sdiv(rn.mul(_GELU_C, x3), _MILLION)      # 0.044715·x³
    inner = x + cube
    arg = _sdiv(rn.mul(_SQRT_2_PI, inner), _TEN7)    # √(2/π)·inner
    t = ract.tanh_fixed(arg, frac)                   # tanh(arg)
    half = (ONE + t) >> 1                            # 0.5·(1 + tanh)
    return rn.mul(x, half) >> frac                   # · x


# ── RoPE (proportional; full-rotation local is the rot_dim == head_dim case) ──
def rope_tables(pos, head_dim, rot_dim, theta_ln, frac=FRAC):
    """(cos_row, sin_row) of length rot_dim//2 for position `pos`. inv_freq is geometric with
    r = theta^(-1/(head_dim/2)) so freq_i = theta^(-2i/head_dim); cos/sin via ring CORDIC.
    Only rot_dim//2 pairs are produced (the partially-rotated global case); local rotates all."""
    half_full = head_dim >> 1                          # exponent divisor is the FULL head_dim/2
    n_rot = rot_dim >> 1
    r = ract.exp_fixed(-rn.mf_floordiv(theta_ln, half_full), frac)
    cos_row = [0] * n_rot; sin_row = [0] * n_rot
    invf = ONE
    for i in range(n_rot):
        ang = rn.mul(pos, invf)
        c, s = _cordic(ang)
        cos_row[i] = c; sin_row[i] = s
        invf = rn.mul(invf, r) >> frac
    return cos_row, sin_row


def apply_rope(vec, cos_row, sin_row, pair_off, frac=FRAC):
    """NeoX half-split RoPE: pair (i, i+pair_off) rotated by (cos_i, sin_i) for the len(cos_row)
    rotated pairs; dims outside the rotated span pass through unchanged. pair_off = head_dim//2."""
    out = list(vec)
    for i in range(len(cos_row)):
        v0 = vec[i]; v1 = vec[i + pair_off]; c = cos_row[i]; s = sin_row[i]
        out[i] = (rn.mul(v0, c) >> frac) - (rn.mul(v1, s) >> frac)
        out[i + pair_off] = (rn.mul(v0, s) >> frac) + (rn.mul(v1, c) >> frac)
    return out


def rmsnorm_g4(x, gamma, frac=FRAC):
    """Gemma4 RMSNorm: x/rms · gamma (NO 1+gamma offset). gamma is Q<frac>."""
    return ract.rmsnorm_fixed(x, gamma, frac)


def rmsnorm_noscale(x, frac=FRAC):
    """RMSNormNoScale (per-head V norm): x/rms, no learnable weight."""
    return ract.rmsnorm_fixed(x, [ONE] * len(x), frac)


def attention_g4(qh, k_cache, v_cache, frac=FRAC):
    """One query head over its cached K/V. Gemma4: score = q·k (scale 1.0, NO soft-cap); softmax;
    ctx = Σ w·v. Causal is implicit (cache holds only positions ≤ current)."""
    scores = [infer.dot(qh, k, frac) for k in k_cache]
    w = infer.softmax(scores, frac)
    dv = len(v_cache[0])
    acc = [0] * dv
    for j in range(len(v_cache)):
        wj = w[j]; vj = v_cache[j]
        for d in range(dv):
            acc[d] += rn.mul(wj, vj[d])
    return [a >> frac for a in acc]


def layer_forward(W, li, h, cache, pos, frac=FRAC):
    is_global, hd, n_kv, rot, theta_ln, group = layer_geom(li)
    nq = G4.n_q
    pair_off = hd >> 1

    # ── attention sublayer
    res = h
    x = rmsnorm_g4(h, W.norm(li, "pre_attn"), frac)
    q = proj(W.lin(li, "q_proj"), x, frac)                 # [nq*hd]
    k = proj(W.lin(li, "k_proj"), x, frac)                 # [n_kv*hd]
    if is_global:                                          # attention_k_eq_v: V = raw K projection
        v = list(k)
    else:
        v = proj(W.lin(li, "v_proj"), x, frac)             # [n_kv*hd]
    qh = [q[i * hd:(i + 1) * hd] for i in range(nq)]
    kh = [k[i * hd:(i + 1) * hd] for i in range(n_kv)]
    vh = [v[i * hd:(i + 1) * hd] for i in range(n_kv)]

    qg = W.norm(li, "q_norm"); kg = W.norm(li, "k_norm")   # per-head gammas (len hd), NO offset
    qh = [rmsnorm_g4(head, qg, frac) for head in qh]       # QK norm BEFORE RoPE
    kh = [rmsnorm_g4(head, kg, frac) for head in kh]
    vh = [rmsnorm_noscale(head, frac) for head in vh]      # V norm (no scale)

    cos_row, sin_row = rope_tables(pos, hd, rot, theta_ln, frac)
    if isinstance(cache, _kh.KVSlab):                      # C block path (bit-for-bit gated, D9)
        qf = _kh.rope([x for head in qh for x in head], cos_row, sin_row, pair_off, nq, hd, frac)
        kf = _kh.rope([x for head in kh for x in head], cos_row, sin_row, pair_off, n_kv, hd, frac)
        cache.insert([kf[j * hd:(j + 1) * hd] for j in range(n_kv)], vh)
        ctx = _kh.attention(qf, cache, nq, frac)
    else:
        qh = [apply_rope(head, cos_row, sin_row, pair_off, frac) for head in qh]
        kh = [apply_rope(head, cos_row, sin_row, pair_off, frac) for head in kh]
        for j in range(n_kv):
            cache["k"][j].append(kh[j]); cache["v"][j].append(vh[j])
        ctx = []
        for i in range(nq):
            kv = rn.mf_floordiv(i, group)
            ctx.extend(attention_g4(qh[i], cache["k"][kv], cache["v"][kv], frac))
    a = proj(W.lin(li, "o_proj"), ctx, frac)               # [hidden]
    a = rmsnorm_g4(a, W.norm(li, "post_attn"), frac)
    h = [res[i] + a[i] for i in range(G4.hidden)]

    # ── FFN sublayer
    res = h
    x = rmsnorm_g4(h, W.norm(li, "pre_mlp"), frac)
    g = proj(W.lin(li, "gate_proj"), x, frac)
    u = proj(W.lin(li, "up_proj"), x, frac)
    act = _kh.gelu_mul(g, u, frac)                         # ONE C block call (bit-for-bit, D9)
    if act is None:
        act = [rn.mul(gelu_tanh_fixed(g[i], frac), u[i]) >> frac for i in range(G4.inter)]
    m = proj(W.lin(li, "down_proj"), act, frac)            # [hidden]
    m = rmsnorm_g4(m, W.norm(li, "post_ff"), frac)
    h = [res[i] + m[i] for i in range(G4.hidden)]

    # ── per-layer learned residual scalar
    sc = W.layer_scalar(li)                                # Q<frac>
    if sc != ONE:
        h = [rn.mul(hi, sc) >> frac for hi in h]
    return h


def layer_forward_c(W, li, h, cache, pos, frac=FRAC):
    """C-RESIDENT layer: `h` is a ctypes int64 buffer and NEVER crosses the Python boundary —
    every op is a C block over resident buffers (the kit's speed model). Bit-identical to
    layer_forward (each block is gated at kernel load; composition verified in tests)."""
    ig, hd, n_kv, rot, tln, group = layer_geom(li)
    nq = G4.n_q
    pair_off = hd >> 1
    A = _kh.actbuf
    lib = _kh._lib
    HB = G4.hidden * 8

    # ── attention sublayer
    res = A("res", G4.hidden); ctypes.memmove(res, h, HB)
    x = A("x", G4.hidden)
    lib.rmsnorm_rows(x, h, _kh.pinned_i64(W.norm(li, "pre_attn")), 1, G4.hidden, frac, 1)
    q = A("q", nq * hd)
    k = A("k", n_kv * hd)
    v = A("v", n_kv * hd)
    if ig:                                                 # attention_k_eq_v: V = RAW K projection
        _kh.gemv_multi([(q, W.lin(li, "q_proj")), (k, W.lin(li, "k_proj"))], x, frac)
        ctypes.memmove(v, k, n_kv * hd * 8)
    else:
        _kh.gemv_multi([(q, W.lin(li, "q_proj")), (k, W.lin(li, "k_proj")),
                        (v, W.lin(li, "v_proj"))], x, frac)
    lib.rmsnorm_rows(q, q, _kh.pinned_i64(W.norm(li, "q_norm")), nq, hd, frac, 1)
    lib.rmsnorm_rows(k, k, _kh.pinned_i64(W.norm(li, "k_norm")), n_kv, hd, frac, 1)
    lib.rmsnorm_rows(v, v, _kh.ones_buf(hd, frac), n_kv, hd, frac, 1)
    cos_row, sin_row = rope_tables(pos, hd, rot, tln, frac)
    _kh.rope_buf(q, cos_row, sin_row, pair_off, nq, hd, frac)
    _kh.rope_buf(k, cos_row, sin_row, pair_off, n_kv, hd, frac)
    cache.insert_bufs(k, v)
    ctx = A("ctx", nq * hd); _kh.attn_into(ctx, q, cache, nq, frac)
    a = A("a", G4.hidden); _kh.gemv_into(a, W.lin(li, "o_proj"), ctx, frac)
    lib.rmsnorm_rows(a, a, _kh.pinned_i64(W.norm(li, "post_attn")), 1, G4.hidden, frac, 1)
    lib.add_into(h, res, a, G4.hidden)

    # ── FFN sublayer
    ctypes.memmove(res, h, HB)
    lib.rmsnorm_rows(x, h, _kh.pinned_i64(W.norm(li, "pre_mlp")), 1, G4.hidden, frac, 1)
    g = A("g", G4.inter)
    u = A("u", G4.inter)
    _kh.gemv_multi([(g, W.lin(li, "gate_proj")), (u, W.lin(li, "up_proj"))], x, frac)
    act = A("act", G4.inter); lib.gelu_mul_block(act, g, u, G4.inter, frac)
    m = A("m", G4.hidden); _kh.gemv_into(m, W.lin(li, "down_proj"), act, frac)
    lib.rmsnorm_rows(m, m, _kh.pinned_i64(W.norm(li, "post_ff")), 1, G4.hidden, frac, 1)
    lib.add_into(h, res, m, G4.hidden)

    # ── per-layer learned residual scalar
    sc = W.layer_scalar(li)
    if sc != ONE:
        lib.scale_q16(h, sc, G4.hidden, frac)


def forward_token(W, token, pos, cache, frac=FRAC):
    """One decode step: embed·√hidden, run 48 layers, return the final-normed hidden (Q<frac>)."""
    esc = W.embed_scale()                                  # Q<frac>: √3840
    if isinstance(cache[0], _kh.KVSlab):                   # C-resident path (kit speed model)
        h = _kh.actbuf("h", G4.hidden)
        _kh.embed_into(h, W.embed_row_bytes(token), G4.hidden, esc, frac)
        for li in range(G4.layers):
            layer_forward_c(W, li, h, cache[li], pos, frac)
        hn = _kh.actbuf("hn", G4.hidden)
        _kh._lib.rmsnorm_rows(hn, h, _kh.pinned_i64(W.final_norm()), 1, G4.hidden, frac, 1)
        return list(hn)                                    # single crossing, for the LM head
    h = [rn.mul(e, esc) >> frac for e in W.embed_row(token)]
    for li in range(G4.layers):
        h = layer_forward(W, li, h, cache[li], pos, frac)
    return rmsnorm_g4(h, W.final_norm(), frac)


def new_cache():
    if _kh.available():                # C-owned KV slabs (read in place by the attention block)
        return [_kh.KVSlab(layer_geom(li)[2], layer_geom(li)[1]) for li in range(G4.layers)]
    return [{"k": [[] for _ in range(layer_geom(li)[2])],
             "v": [[] for _ in range(layer_geom(li)[2])]}
            for li in range(G4.layers)]


def generate(W, prompt_ids, n_new, frac=FRAC, verbose=False):
    """Greedy autoregressive generation. Returns the full id list (prompt + generated). Float-free:
    LM head = argmax of the tied f16 embedding dot (monotone soft-cap ⇒ argmax preserved)."""
    cache = new_cache()
    out = list(prompt_ids)
    pos = 0
    hn = None
    for t in prompt_ids:
        hn = forward_token(W, t, pos, cache, frac)
        pos = pos + 1
    for _ in range(n_new):
        nt, _score = W.lm_argmax(hn)
        out.append(nt)
        if verbose:
            print("  ->", nt, flush=True)
        if nt in G4.eos:
            break
        hn = forward_token(W, nt, pos, cache, frac)
        pos = pos + 1
    return out
