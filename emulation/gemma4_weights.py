"""
ringkit.emulation.gemma4_weights — file-backed Gemma4-12B weight provider for gemma4.generate().
Streams the ONIX int8 linears (mmap, one tensor sliced at a time), the f16 embed.bin (mmap, tied LM
head), the f16 norms.bin (block + per-head Q/K gammas), and layer_scalars.bin — all as ring integers
by seek + integer decode (no torch/numpy/float on the load path).

norms.bin (magic NORM, ver 1): 16-byte header, then per layer
  [pre_attn | post_attn | pre_mlp | post_ff]  each `hidden` f16,
  then u32 q_hd, `q_hd` f16 q_norm gammas, u32 k_hd, `k_hd` f16 k_norm gammas
  (q_hd/k_hd = layer head_dim: 256 local, 512 global; 0 would mean no per-head norm — Gemma2 style);
  finally the model `final` norm (`hidden` f16).
layer_scalars.bin: `layers` little-endian f32 (the learned per-layer residual scalars).
embed.bin: row-major f16, [vocab, hidden]; row `t` at byte t*hidden*2.

Global layers (li % 6 == 5) carry NO v_proj tensor in the onix (attention_k_eq_v).
"""
import mmap
import os
from ringkit.emulation import onix, gemma, gemma4
from ringkit.kernels.mprc.gemma import host as _k

FRAC = gemma4.FRAC
_BLOCK_NAMES = ("pre_attn", "post_attn", "pre_mlp", "post_ff")


def _f16(u):
    return gemma._f16_to_fixed(u, FRAC)


class Gemma4Weights:
    def __init__(self, onix_path, embed_path, norms_path, scalars_path):
        self.onix_path = onix_path
        self._data_off, self._ents = onix.index(onix_path)
        self._of = open(onix_path, "rb")     # mmap'd once (reclaimable page cache, not RAM).
        # MAP_PRIVATE (copy-on-write, never written) so the buffer is writable-typed: the GEMV
        # block kernel then reads tensor slabs IN PLACE via zero-copy memoryviews — no Python
        # copies of weight memory, per the kit's C-owned-block model.
        self._onix = mmap.mmap(self._of.fileno(), 0, flags=mmap.MAP_PRIVATE)
        self._onix_mv = memoryview(self._onix)
        if os.environ.get("RINGKIT_GEMV") == "metal":     # GPU GEMV: map the onix once (no-copy)
            from ringkit.kernels.mprc.gemma import host as _gh
            _gh.metal_register_onix(self._onix)
        self._sz_cache = {}                  # cache only tiny s_row/z_row per tensor; slice xbar on demand
        self._ef = open(embed_path, "rb")
        self._emb = mmap.mmap(self._ef.fileno(), 0, prot=mmap.PROT_READ)
        self._embed_path = embed_path
        self._norms, self._qk, self._final = self._parse_norms(norms_path)
        self._scalars = self._parse_scalars(scalars_path)
        self._embed_scale = None             # lazy: √hidden in Q<frac>

    # ── linears (onix int8; xbar sliced from the shared mmap, s/z cached) ─────
    def lin(self, li, name):
        sub = "self_attn" if name in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
        full = f"model.layers.{li}.{sub}.{name}"
        e = self._ents[full]
        of, inf, xl, sl = e["out_feat"], e["in_feat"], e["xbar_len"], e["s_len"]
        base = self._data_off + e["offset"]
        xbar = self._onix_mv[base:base + of * inf]       # zero-copy view — C reads the slab in place
        sz = self._sz_cache.get(full)
        if sz is None:
            s_raw = self._onix[base + xl:base + xl + of]
            z_raw = self._onix[base + xl + sl:base + xl + sl + of]
            s_row = [v - 256 if v > 127 else v for v in s_raw]
            z_row = list(z_raw)
            sz = (s_row, z_row)
            self._sz_cache[full] = sz
        return xbar, sz[0], sz[1], of, inf

    # ── norms (block gammas + per-head Q/K gammas -> Q<frac>) ─────────────────
    def _parse_norms(self, path):
        b = open(path, "rb").read()
        assert b[0:4] == b"NORM", "norms.bin bad magic"
        n_layers = int.from_bytes(b[8:12], "little")
        hidden = int.from_bytes(b[12:16], "little")
        assert n_layers == gemma4.G4.layers, f"norms.bin n_layers={n_layers}"
        assert hidden == gemma4.G4.hidden, f"norms.bin hidden={hidden}"
        off = 16
        hb = hidden * 2
        blocks = []; qk = []
        for _l in range(n_layers):
            g = {}
            for nm in _BLOCK_NAMES:
                g[nm] = [_f16(int.from_bytes(b[off + 2 * j:off + 2 * j + 2], "little")) for j in range(hidden)]
                off += hb
            q_hd = int.from_bytes(b[off:off + 4], "little"); off += 4
            q_norm = [_f16(int.from_bytes(b[off + 2 * j:off + 2 * j + 2], "little")) for j in range(q_hd)]
            off += q_hd * 2
            k_hd = int.from_bytes(b[off:off + 4], "little"); off += 4
            k_norm = [_f16(int.from_bytes(b[off + 2 * j:off + 2 * j + 2], "little")) for j in range(k_hd)]
            off += k_hd * 2
            blocks.append(g); qk.append((q_norm, k_norm))
        final = [_f16(int.from_bytes(b[off + 2 * j:off + 2 * j + 2], "little")) for j in range(hidden)]
        return blocks, qk, final

    def _parse_scalars(self, path):
        raw = open(path, "rb").read()
        n = gemma4.G4.layers
        return [_f32_bits_to_fixed(int.from_bytes(raw[4 * i:4 * i + 4], "little"), FRAC)
                for i in range(n)]

    def norm(self, li, which):
        if which == "q_norm":
            return self._qk[li][0]
        if which == "k_norm":
            return self._qk[li][1]
        return self._norms[li][which]

    def final_norm(self):
        return self._final

    def layer_scalar(self, li):
        return self._scalars[li]

    def embed_scale(self):
        if self._embed_scale is None:
            self._embed_scale = rn_isqrt_q(gemma4.G4.hidden, FRAC)   # √hidden in Q<frac>
        return self._embed_scale

    # ── embeddings (f16 mmap) ─────────────────────────────────────────────────
    def embed_row(self, token):
        H = gemma4.G4.hidden
        base = token * H * 2
        raw = self._emb[base:base + H * 2]
        return [_f16(raw[2 * j] | (raw[2 * j + 1] << 8)) for j in range(H)]

    def embed_row_bytes(self, token):
        """Raw f16 row bytes (for the C embed block — decode happens in the kernel)."""
        H = gemma4.G4.hidden
        base = token * H * 2
        return self._emb[base:base + H * 2]

    def lm_argmax(self, hidden):
        """Greedy next token: argmax over the tied f16 embedding table. The kernel mmaps embed.bin
        READ-ONLY itself (zero-copy) — Python holds no embedding memory. shift=13 matches the f16
        decode used across the emulation kernel."""
        return _k.lm_argmax_file(hidden, self._embed_path, 0,
                                 gemma4.G4.vocab, gemma4.G4.hidden, shift=13)

    def close(self):
        try:
            self._emb.close(); self._ef.close()
            self._onix_mv.release()          # release the exported view before closing the mmap
            self._onix.close(); self._of.close()
        except Exception:
            pass


def rn_isqrt_q(n, frac):
    """√n in Q<frac>, float-free (ring isqrt): isqrt(n << 2·frac) = √n · 2^frac."""
    from ringkit.core import native as rn
    return rn.isqrt(n << (frac + frac))


def _f32_bits_to_fixed(bits, frac):
    """Decode an IEEE-754 single-precision uint32 (a load-time constant from layer_scalars.bin) to a
    signed Q<frac> integer by INTEGER bit-field ops alone — no FPU, no Python float ever materialized
    (mirrors `_f16_to_fixed` for the wider f32 field)."""
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    if exp == 0:
        val = 0                                   # subnormal ≈ 0 at Q<frac>
    elif exp == 0xFF:
        val = 0                                   # inf/nan → 0 (never expected here)
    else:
        mant = (1 << 23) | man
        sh = frac + (exp - 127) - 23
        val = (mant << sh) if sh >= 0 else (mant >> (-sh))
    return -val if sign else val


def default_paths():
    """Locate the Gemma4-12B files; returns (onix, embed, norms, scalars) or None if unreachable."""
    onix_cands = [os.path.expanduser("~/Projects/hpq-kernel-rust/gemma4_12b.onix")]
    w12 = [os.path.expanduser("~/Projects/mprc-scratchpad/hpq_kernel/weights_12b")]
    onix_p = next((p for p in onix_cands if os.path.exists(p)), None)
    w12_p = next((p for p in w12 if os.path.isdir(p)), None)
    if not onix_p or not w12_p:
        return None
    emb = os.path.join(w12_p, "embed.bin")
    nrm = os.path.join(w12_p, "norms.bin")
    scl = os.path.join(w12_p, "layer_scalars.bin")
    if not (os.path.exists(emb) and os.path.exists(nrm) and os.path.exists(scl)):
        return None
    return onix_p, emb, nrm, scl
