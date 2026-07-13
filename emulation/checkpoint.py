"""
ringkit.checkpoint — load a REAL pretrained model into the ring, with ZERO torch / numpy / float.

The proof-of-kit (owner's bar): if ringkit can ingest a pretrained checkpoint using only integer and
byte operations — no torch, no numpy, no float ALU/FPU, no standard math — the kit is real.

HOW (all integer / bytes):
  * A torch `.pth` is a ZIP of a pickle (the state_dict structure) + raw storages (`data/<key>`).
    `zipfile` + a stub `Unpickler` (that returns markers instead of importing torch) recover, per
    tensor: name, storage key, dtype, shape. No torch imported.
  * Each weight is decoded from its RAW BYTES by INTEGER bit-ops only. An IEEE float is
    (sign, exponent, mantissa) — all integer fields extracted by shift/mask, never by the FPU:
        fp32 word u:  sign=(u>>31)&1   exp=(u>>23)&0xFF   mantissa=u&0x7FFFFF
    The biased EXPONENT is already a ring value in 0..255 — the weight's LOG-MAGNITUDE — so it maps
    to the ARC directly (ARC never leaves 256, C9). Sign+mantissa are the ENERGY refinement.
    No `float()`, no `struct.unpack('f')`, no math anywhere.

This module ingests weights into ring ARC values (the log-magnitude phase). Faithful dequantization
for a forward pass (mapping the full (sign, exp, mantissa) to the ring's (arc, energy) for exact
compute) is a separate, measured step; this proves INGESTION is torch/numpy/float-free.

Multiplier-free at the semantic level; only stdlib zipfile/pickle/io (bytes) + ringkit are imported.
"""
import io as _io
import pickle
import zipfile

from ringkit.rnp.tensor import RingTensor

# torch storage class -> (element byte size, dtype tag)
_STORAGE = {
    "FloatStorage": (4, "f32"), "HalfStorage": (2, "f16"), "BFloat16Storage": (2, "bf16"),
    "DoubleStorage": (8, "f64"), "LongStorage": (8, "i64"), "IntStorage": (4, "i32"),
    "ShortStorage": (2, "i16"), "CharStorage": (1, "i8"), "ByteStorage": (1, "u8"),
    "BoolStorage": (1, "u8"),
}


class _TensorRef:
    __slots__ = ("storage", "offset", "size", "stride")

    def __init__(self, storage, offset, size, stride):
        self.storage = storage      # ("STORAGE", storage_type, key, numel)
        self.offset = offset
        self.size = tuple(size)
        self.stride = tuple(stride)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *rest):
    return _TensorRef(storage, storage_offset, size, stride)


class _ODict(dict):
    """A dict subclass (has __dict__, so pickle BUILD works) standing in for OrderedDict."""


class _Stub:
    """Accept-anything placeholder for torch classes we never actually instantiate."""
    def __init__(self, *a, **k):
        pass
    def __setstate__(self, state):
        pass


def _rebuild_parameter(data, *rest):
    return data


class _RingUnpickler(pickle.Unpickler):
    """Parses the state_dict STRUCTURE without importing torch. Tensor DATA is not in the pickle."""

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if name == "OrderedDict":
            return _ODict
        if name.endswith("Storage"):
            return type(name, (), {})        # a marker class carrying the storage name; never torch
        return _Stub

    def persistent_load(self, pid):
        # torch pid: ('storage', <StorageType or name>, key, location, numel)
        tag = pid[0]
        if tag != "storage":
            raise ValueError(f"unexpected persistent id tag: {tag}")
        stype = pid[1] if isinstance(pid[1], str) else getattr(pid[1], "__name__", str(pid[1]))
        key = str(pid[2])
        numel = int(pid[4])
        return ("STORAGE", stype, key, numel)


def _flatten(obj, prefix=""):
    """Walk the (possibly nested) state_dict, yielding (name, _TensorRef)."""
    if isinstance(obj, _TensorRef):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}.{i}" if prefix else str(i))


def _bytes_to_arc(raw, elem, dtype):
    """Decode raw little-endian float bytes -> ring ARC values (log-magnitude), INTEGER ONLY.

    For f32/f16/bf16 the ARC is the biased exponent (already 0..255 for f32/bf16; f16's 5-bit
    exponent is shifted into the ring). No float, no FPU — the exponent is an integer bit-field."""
    n = len(raw) // elem
    out = bytearray(n)
    for i in range(n):
        u = int.from_bytes(raw[i * elem:(i + 1) * elem], "little")   # integer, no float
        if dtype == "f32":
            out[i] = (u >> 23) & 0xFF                                # 8-bit biased exponent = ARC
        elif dtype == "bf16":
            out[i] = (u >> 7) & 0xFF                                 # bf16 exponent is bits 7..14
        elif dtype == "f16":
            out[i] = ((u >> 10) & 0x1F) << 3                         # 5-bit exp -> spread onto the ring
        else:
            out[i] = u & 0xFF                                        # integer dtype: low byte
    return out


def _fp_to_fixed_word(u, dtype, frac):
    """One IEEE float word (as an INTEGER) -> signed fixed-point integer (value * 2**frac), by
    INTEGER shifts only. No FPU: `value = ±significand * 2**(exp-bias-mbits)`, so
    `value*2**frac = ±significand << (exp-bias-mbits+frac)` (negative shift = round-then-shift-down).
    ENERGY domain (unfolded); the ring ARC is this value mod 256, the energy is it >> 8 (fold late)."""
    if dtype == "f32":
        sign = (u >> 31) & 1; exp = (u >> 23) & 0xFF; man = u & 0x7FFFFF
        bias, mbits, emax = 127, 23, 0xFF
    elif dtype == "bf16":
        sign = (u >> 15) & 1; exp = (u >> 7) & 0xFF; man = u & 0x7F
        bias, mbits, emax = 127, 7, 0xFF
    elif dtype == "f16":
        sign = (u >> 15) & 1; exp = (u >> 10) & 0x1F; man = u & 0x3FF
        bias, mbits, emax = 15, 10, 0x1F
    else:
        return u if u < (1 << 63) else u - (1 << 64)     # integer dtype: pass through signed
    if exp == 0 or exp == emax:
        return 0                                         # zero/subnormal ~ 0; inf/nan -> 0
    sig = (1 << mbits) | man                             # implicit leading 1
    s = exp - bias - mbits + frac
    if s >= 0:
        val = sig << s
    else:
        sh = -s
        val = (sig + (1 << (sh - 1))) >> sh              # round to nearest, integer
    return -val if sign else val


def _to_fixed(raw, elem, dtype, frac):
    n = len(raw) // elem
    return [_fp_to_fixed_word(int.from_bytes(raw[i * elem:(i + 1) * elem], "little"), dtype, frac)
            for i in range(n)]


def load_fixed(path, frac=16, limit=None):
    """Load a .pth into FAITHFUL ring fixed-point (signed integers = value * 2**frac), torch/numpy/
    float-FREE. This is the inference-ready form: ring integer arithmetic (shift-add) on these
    reproduces the model's float compute up to the 2**-frac resolution. Returns name -> (values, shape)."""
    z = zipfile.ZipFile(path)
    names = z.namelist()
    root = names[0].split("/")[0]
    state = _RingUnpickler(_io.BytesIO(z.read(f"{root}/data.pkl"))).load()
    out = {}
    for name, ref in _flatten(state):
        if limit is not None and len(out) >= limit:
            break
        _, stype, key, numel = ref.storage
        elem, dtype = _STORAGE.get(stype, (1, "u8"))
        vals = _to_fixed(z.read(f"{root}/data/{key}"), elem, dtype, frac)
        shape = ref.size if ref.size else (len(vals),)
        n = 1
        for s in shape:
            n = n * s
        out[name] = (vals, tuple(shape) if n == len(vals) else (len(vals),))
    return out


def load_pth(path, limit=None):
    """Load a torch .pth into ring ARC tensors, torch/numpy/float-free.

    Returns dict name -> RingTensor (ARC = per-weight log-magnitude). `limit` caps how many tensors
    are decoded (for a quick proof on a large model)."""
    z = zipfile.ZipFile(path)
    names = z.namelist()
    root = names[0].split("/")[0]
    state = _RingUnpickler(_io.BytesIO(z.read(f"{root}/data.pkl"))).load()

    out = {}
    for count, (name, ref) in enumerate(_flatten(state)):
        if limit is not None and len(out) >= limit:
            break
        _, stype, key, numel = ref.storage
        elem, dtype = _STORAGE.get(stype, (1, "u8"))
        raw = z.read(f"{root}/data/{key}")
        arc = _bytes_to_arc(raw, elem, dtype)
        # shape may be scalar/empty; RingTensor needs a shape
        shape = ref.size if ref.size else (len(arc),)
        # guard: only build if the byte count matches the declared shape's element count
        n = 1
        for s in shape:
            n = n * s
        if n != len(arc):
            shape = (len(arc),)
        out[name] = RingTensor(list(arc), tuple(shape))
    return out


def summarize(path, limit=None):
    """Load and report totals — the proof line. Returns (n_tensors, n_params, dtypes)."""
    ten = load_pth(path, limit=limit)
    nparams = 0
    for t in ten.values():
        p = 1
        for s in t.shape:
            p = p * s
        nparams += p
    return len(ten), nparams, ten
