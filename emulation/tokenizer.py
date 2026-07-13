"""
ringkit.emulation.tokenizer — minimal Gemma BPE tokenizer (encode/decode) from a HF tokenizer.json.
String processing only (stdlib json); no arithmetic on ring values, no float, no external libs.

Gemma uses a byte-level-fallback BPE with a metaspace marker U+2581 ('▁') for spaces. encode():
normalize spaces -> '▁', seed with characters (byte_fallback for unknown chars), then greedily apply
the lowest-rank merge until none remain. A <bos> (id 2) is prepended. decode() maps ids -> pieces,
joins, restores spaces, and turns <0xHH> byte tokens back into UTF-8.
"""
import json

_SPACE = "▁"


class GemmaTokenizer:
    def __init__(self, path):
        d = json.load(open(path, encoding="utf-8"))
        m = d["model"]
        self.vocab = m["vocab"]                                   # piece -> id
        self.id2piece = {i: p for p, i in self.vocab.items()}
        raw = m.get("merges", [])
        pairs = [(x.split(" ") if isinstance(x, str) else list(x)) for x in raw]
        self.rank = {(a, b): r for r, (a, b) in enumerate(pairs)}
        self.bos = self.vocab.get("<bos>", 2)
        self.eos = self.vocab.get("<eos>", 1)

    # ── encode ────────────────────────────────────────────────────────────────────
    def _byte_fallback(self, ch):
        """Return the token id(s) for a single character, using <0xHH> byte tokens if needed."""
        if ch in self.vocab:
            return [self.vocab[ch]]
        return [self.vocab[f"<0x{byte:02X}>"] for byte in ch.encode("utf-8")]

    def encode(self, text, add_bos=True):
        norm = text.replace(" ", _SPACE)
        if not norm.startswith(_SPACE):
            norm = _SPACE + norm                                  # metaspace prefix
        # seed symbols as vocab pieces (single chars), leaving byte-fallback for the final id pass
        syms = list(norm)
        while True:
            best = None; bi = -1
            for i in range(len(syms) - 1):
                r = self.rank.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best):
                    best = r; bi = i
            if bi < 0:
                break
            syms[bi:bi + 2] = [syms[bi] + syms[bi + 1]]
        ids = [self.bos] if add_bos else []
        for s in syms:
            ids.extend(self._byte_fallback(s))
        return ids

    # ── decode ─────────────────────────────────────────────────────────────────────
    def decode(self, ids):
        out = bytearray()
        for i in ids:
            p = self.id2piece.get(i, "")
            if p in ("<bos>", "<eos>", "<pad>", "<unk>"):
                continue
            if p.startswith("<0x") and p.endswith(">") and len(p) == 6:
                out.append(int(p[3:5], 16))
            else:
                out.extend(p.replace(_SPACE, " ").encode("utf-8"))
        return out.decode("utf-8", errors="replace")


def default_path():
    import os
    cands = ["/sessions/dazzling-zen-euler/mnt/Projects/GemmaApp/GemmaApp/tokenizer.json",
             os.path.expanduser("~/Projects/GemmaApp/GemmaApp/tokenizer.json")]
    return next((p for p in cands if os.path.exists(p)), None)
