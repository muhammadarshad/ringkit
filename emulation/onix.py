"""
ringkit.onix — EMULATE a traditional model (Gemma) on the ring. torch/numpy/float-FREE.

STANCE (owner): for traditional models we cannot control we are an EMULATION ENGINE (like
TurboQuant/PolarQuant emulate/quantize) — we ingest their weights and run their compute on the
ring. For OUR MPRC architectures we are the operating system (native). Either way: NO FLOAT. The
FPU is replaced by ringkit's OWN QCM kernel — here the quarter-square product `rn.qsm` (ringkit's
signature QCM primitive), NOT a copy of hpq's `dot_qph`. hpq is a reference for WHAT the ops are;
the arithmetic is ringkit QCM-enabled code.

ONIX is ALREADY integer-quantized, and its dequant is shift + integer-divide + a multiplier-free
quarter-square dot — so it maps onto the ring with no FPU at all:
    xbar   : uint8 offset-binary weight, W_bar = W + 128        (an integer, already a ring value)
    s_row  : int8 per-row exponent -> scale 2^s = a bit SHIFT   (hpq: "dequant is a simple bit shift")
    z_row  : uint8 per-row divisor -> integer divide
    project: out[row] = dot(xbar[row]-128, x_s8) * 2^s_row[row] / z_row[row]   (dot = shift-add)

File layout (hpq.h / hpq_model.c): 256-byte header (magic "ONIX", n_tensors@8, index_off@68,
data_off@76); 192-byte index entries (name[128], offset@128, out_feat@136, n_blocks@140,
block_size@144, xbar_len@148, s_len@156); tensor data at data_off+offset = xbar | s_row | z_row.

We parse by file seek + int.from_bytes (no mmap-float, no torch/numpy). Weights are read as ring
integers directly — no float decode is even needed.
"""
from ringkit.core import native as rn

_HDR = 256
_ENT = 192


def _u32(b, o): return int.from_bytes(b[o:o + 4], "little")
def _u64(b, o): return int.from_bytes(b[o:o + 8], "little")


def index(path):
    """Parse the ONIX header + index. Returns (data_off, {name: entry}) with entry =
    (offset, out_feat, in_feat, block_size, xbar_len, s_len). Integer/byte ops only."""
    with open(path, "rb") as f:
        hdr = f.read(_HDR)
        if hdr[0:4] != b"ONIX":
            raise ValueError(f"not an ONIX file: magic {hdr[0:4]!r}")
        n = _u32(hdr, 8); idx_off = _u64(hdr, 68); data_off = _u64(hdr, 76)
        f.seek(idx_off)
        raw = f.read(n * _ENT)
    ents = {}
    for i in range(n):
        e = raw[i * _ENT:(i + 1) * _ENT]
        name = e[0:128].split(b"\x00", 1)[0].decode("latin1")
        ents[name] = {
            "offset": _u64(e, 128), "out_feat": _u32(e, 136),
            "in_feat": _u32(e, 140) * _u32(e, 144), "block_size": _u32(e, 144),
            "xbar_len": _u64(e, 148), "s_len": _u64(e, 156),
        }
    return data_off, ents


def tensor(path, name, rows=None):
    """Read a tensor's ring integers (xbar, s_row, z_row) by seek. `rows` limits how many output
    rows are read (proof/perf). Returns (xbar bytes, s_row int8 list, z_row list, out_feat, in_feat)."""
    data_off, ents = index(path)
    e = ents[name]
    of, inf, xl, sl = e["out_feat"], e["in_feat"], e["xbar_len"], e["s_len"]
    r = of if rows is None else min(rows, of)
    base = data_off + e["offset"]
    with open(path, "rb") as f:
        f.seek(base)
        xbar = f.read(r * inf)                       # first r rows of the weight matrix
        f.seek(base + xl)
        s_raw = f.read(of)                           # int8 per-row exponents
        f.seek(base + xl + sl)
        z_raw = f.read(of)                           # uint8 per-row divisors
    s_row = [v - 256 if v > 127 else v for v in s_raw[:r]]   # int8
    z_row = list(z_raw[:r])
    return xbar, s_row, z_row, of, inf


def linear(xbar, s_row, z_row, x_s8, out_feat, in_feat):
    """Full Gemma linear: out[row] = dot(xbar[row]-128, x)*2^s_row[row]/z_row[row] for all rows.
    The energy dot runs on ringkit's QCM energy-QSM kernel (C, int64 accumulate, no fold, quarter-
    square — the FPU replacement); per-row 2^s is a shift and /z an integer divide. Float-free."""
    from ringkit.kernels.mprc.gemma import host as _g
    dots = _g.qsm_dot(xbar, x_s8, out_feat, in_feat)     # int64 energy dots, no fold (C fast path)
    out = []
    for r in range(out_feat):
        acc = dots[r]
        acc = acc << s_row[r] if s_row[r] >= 0 else acc >> (-s_row[r])
        z = z_row[r] if z_row[r] else 1
        out.append(-rn.mf_floordiv(-acc, z) if acc < 0 else rn.mf_floordiv(acc, z))
    return out


def project_row(xbar, row, s, z, x_s8, in_feat):
    """One Gemma linear output: out = dot(xbar[row]-128, x) * 2^s / z. Ring/integer only — the FPU
    is replaced by ringkit's QCM quarter-square `rn.qsm` (exact for the int8 range |a+b|<=255<512),
    NOT hpq's dot_qph copy. 2^s is a shift; /z an integer divide."""
    base = rn.mul(row, in_feat)
    acc = 0
    for i in range(in_feat):
        acc += rn.qsm(xbar[base + i] - 128, x_s8[i])     # QCM quarter-square MAC (exact), no FPU
    if s >= 0:
        acc = acc << s
    else:
        acc = acc >> (-s)
    if z == 0:
        z = 1
    return -rn.mf_floordiv(-acc, z) if acc < 0 else rn.mf_floordiv(acc, z)
