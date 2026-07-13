"""
ringkit.emulation.gemma_weights — file-backed Gemma2-2B weight provider for gemma.generate().
Reads the ONIX int8 linears, the f16 embed.bin (mmap, tied LM head), and the f16 norms.bin gammas,
all as ring integers by seek + integer decode (no torch/numpy/float on the load path).

norms.bin (magic NORM, ver 1): 16-byte header, then per layer
  [pre_attn | post_attn | pre_mlp | post_ff]  each `hidden` f16, then 8 bytes meta (q/k-norm hd = 0
  for Gemma2); finally the model `final` norm (`hidden` f16).
embed.bin: row-major f16, [vocab, hidden]; row `t` at byte t*hidden*2.
"""
import mmap
import os
from ringkit.emulation import onix, gemma
from ringkit.kernels.mprc.gemma import host as _k

FRAC = gemma.FRAC
_NAMES = ("pre_attn", "post_attn", "pre_mlp", "post_ff")


class Gemma2Weights:
    def __init__(self, onix_path, embed_path, norms_path):
        self.onix_path = onix_path
        self._data_off, self._ents = onix.index(onix_path)
        self._of = open(onix_path, "rb")     # shared, read-only mmap (reclaimable page cache, not RAM-resident)
        self._onix = mmap.mmap(self._of.fileno(), 0, prot=mmap.PROT_READ)
        self._sz_cache = {}                  # cache only the tiny s_row/z_row per tensor; slice xbar on demand
        # embed.bin mmap (read-only)
        self._ef = open(embed_path, "rb")
        self._emb = mmap.mmap(self._ef.fileno(), 0, prot=mmap.PROT_READ)
        self._embw = None                     # writable copy handle for the kernel (lazy)
        self._embed_path = embed_path
        self._norms, self._final = self._parse_norms(norms_path)

    # ── linears (onix int8; xbar sliced from the shared mmap, s/z cached) ─────────
    def lin(self, li, name):
        sub = "self_attn" if name in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
        full = f"model.layers.{li}.{sub}.{name}"
        e = self._ents[full]
        of, inf, xl, sl = e["out_feat"], e["in_feat"], e["xbar_len"], e["s_len"]
        base = self._data_off + e["offset"]
        xbar = self._onix[base:base + of * inf]          # transient bytes slice (one tensor, freed after use)
        sz = self._sz_cache.get(full)
        if sz is None:
            s_raw = self._onix[base + xl:base + xl + of]
            z_raw = self._onix[base + xl + sl:base + xl + sl + of]
            s_row = [v - 256 if v > 127 else v for v in s_raw]
            z_row = list(z_raw)
            sz = (s_row, z_row)
            self._sz_cache[full] = sz
        return xbar, sz[0], sz[1], of, inf

    # ── norms (f16 gammas -> Q<frac>) ─────────────────────────────────────────────
    def _parse_norms(self, path):
        b = open(path, "rb").read()
        assert b[0:4] == b"NORM", "norms.bin bad magic"
        n_layers = int.from_bytes(b[8:12], "little")
        hidden = int.from_bytes(b[12:16], "little")
        off = 16
        hb = hidden * 2
        layers = []
        for _l in range(n_layers):
            g = {}
            for nm in _NAMES:
                g[nm] = [gemma._f16_to_fixed(int.from_bytes(b[off + 2 * j:off + 2 * j + 2], "little"))
                         for j in range(hidden)]
                off += hb
            off += 8                          # q/k-norm meta (0 for Gemma2)
            layers.append(g)
        final = [gemma._f16_to_fixed(int.from_bytes(b[off + 2 * j:off + 2 * j + 2], "little"))
                 for j in range(hidden)]
        return layers, final

    def norm(self, li, which):
        return self._norms[li][which]

    def final_norm(self):
        return self._final

    # ── embeddings (f16 mmap) ─────────────────────────────────────────────────────
    def embed_row(self, token):
        H = gemma.G2.hidden
        base = token * H * 2
        raw = self._emb[base:base + H * 2]
        return [gemma._f16_to_fixed(raw[2 * j] | (raw[2 * j + 1] << 8)) for j in range(H)]

    def lm_argmax(self, hidden):
        """Greedy next token: argmax over the tied f16 embedding table. The kernel mmaps embed.bin
        READ-ONLY itself (zero-copy streaming) — Python holds no embedding memory."""
        return _k.lm_argmax_file(hidden, self._embed_path, 0,
                                 gemma.G2.vocab, gemma.G2.hidden, shift=13)

    def close(self):
        try:
            self._emb.close()
            if self._embw is not None:
                self._embw.close()
            self._ef.close()
            self._onix.close()
            self._of.close()
        except Exception:
            pass


def default_paths():
    """Locate the mounted Gemma2-2B files; returns (onix, embed, norms) or None if unreachable."""
    onix_cands = ["/sessions/dazzling-zen-euler/mnt/hpq-kernel-rust/gemma2_2b.onix",
                  os.path.expanduser("~/Projects/hpq-kernel-rust/gemma2_2b.onix")]
    w2b = ["/sessions/dazzling-zen-euler/mnt/mprc-scratchpad/hpq_kernel/weights_2b",
           os.path.expanduser("~/Projects/mprc-scratchpad/hpq_kernel/weights_2b")]
    onix_p = next((p for p in onix_cands if os.path.exists(p)), None)
    w2b_p = next((p for p in w2b if os.path.isdir(p)), None)
    if not onix_p or not w2b_p:
        return None
    emb = os.path.join(w2b_p, "embed.bin")
    nrm = os.path.join(w2b_p, "norms.bin")
    if not (os.path.exists(emb) and os.path.exists(nrm)):
        return None
    return onix_p, emb, nrm
