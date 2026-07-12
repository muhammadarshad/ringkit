"""
ringkit.rnp.tensor — RingTensor: production nD ndarray on the Z256 ring.

Charter-clean: pure-Python list backing (no numpy import, NFR2); every value is a ring value
0..255; all index/stride/size arithmetic is multiplier-free (rn.mul / rn.mf_floordiv). A `unit`
tag ('arc' | 'energy') selects the central tendency for reductions (circular vs geometric mean).

Production surface: nD shape/strides; basic indexing + slicing (int/slice/tuple/Ellipsis,
negatives, bounds-checked) with assignment; nD broadcasting; nD reductions (sum/mean/median/min/
max/prod/argmin/argmax, keepdims); nD transpose/swapaxes; reshape(-1)/flatten/ravel;
concatenate/stack; batched nD matmul (+ vector cases); numpy-style operators (+ - * @ .T);
iter/len/eq/repr. Errors are IndexError / ValueError with messages (no user-facing asserts).
No float, no '*'/'//'/'**'.

Note: basic indexing returns COPIES (not memory-sharing views). Augmented assignment on a slice
(`a[1:3] += 1`) works via __getitem__/__setitem__; holding a slice reference does not alias.
"""
import itertools
from ringkit.core import native as rn
from ringkit.stats import stats as rs
from ringkit.kernels import backend as _k     # silicon fast-path (zero-copy on the C buffer)

from ringkit.core.constants import TAU


# ── shape / stride helpers (multiplier-free) ─────────────────────────────────
def _size(shape):
    n = 1
    for s in shape:
        n = rn.mul(n, s)
    return n


def _row_major_strides(shape):
    n = len(shape)
    st = [0 for _ in range(n)]
    acc = 1
    for i in range(n - 1, -1, -1):
        st[i] = acc
        acc = rn.mul(acc, shape[i])
    return tuple(st)


def _broadcast_shape(a, b):
    ra, rb = list(reversed(a)), list(reversed(b))
    out = []
    for i in range(max(len(ra), len(rb))):
        da = ra[i] if i < len(ra) else 1
        db = rb[i] if i < len(rb) else 1
        if da == db or da == 1 or db == 1:
            out.append(da if db == 1 else db)
        else:
            raise ValueError(f"operands could not be broadcast: {tuple(a)} vs {tuple(b)}")
    return tuple(reversed(out))


def _norm_int(i, dim, axis):
    """Normalize a possibly-negative int index and bounds-check it."""
    j = i + dim if i < 0 else i
    if not (0 <= j < dim):
        raise IndexError(f"index {i} out of bounds for axis {axis} with size {dim}")
    return j


def _flatten(nested):
    if not isinstance(nested, (list, tuple)):
        return [nested], []
    if not nested or not isinstance(nested[0], (list, tuple)):
        return list(nested), [len(nested)]
    flat, subshape = [], []
    for row in nested:
        f, s = _flatten(row)
        flat.extend(f)
        subshape = s
    return flat, [len(nested)] + subshape


def _ranges(shape):
    return [range(s) for s in shape]


class RingTensor:
    __slots__ = ("data", "shape", "strides", "unit")

    def __init__(self, data, shape=None, unit="arc"):
        if shape is None:
            flat, shape = _flatten(data)
            self.data = bytearray(int(v) & 0xFF for v in flat)   # C-level contiguous buffer
        else:
            self.data = bytearray(int(v) & 0xFF for v in data)
        self.shape = tuple(int(s) for s in shape)
        if any(s < 0 for s in self.shape):
            raise ValueError(f"negative dimension in shape {self.shape}")
        self.strides = _row_major_strides(self.shape)
        if len(self.data) != _size(self.shape):
            raise ValueError(f"data length {len(self.data)} != shape size {_size(self.shape)} {self.shape}")
        if unit not in ("arc", "energy"):
            raise ValueError(f"unit must be 'arc' or 'energy', got {unit!r}")
        self.unit = unit

    # ── properties ──
    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return len(self.data)

    def _offset(self, multi):
        off = 0
        for i, st in zip(multi, self.strides):
            off += rn.mul(i, st)
        return off

    def copy(self):
        return RingTensor(self.data[:], self.shape, self.unit)

    def tolist(self):
        return _reshape_nested(self.data, self.shape)

    # ── indexing (basic: int / slice / tuple / Ellipsis, negatives, bounds-checked) ──
    def _expand(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        if Ellipsis in idx:
            e = idx.index(Ellipsis)
            fill = self.ndim - (len(idx) - 1)
            if fill < 0:
                raise IndexError("too many indices for tensor")
            idx = idx[:e] + tuple(slice(None) for _ in range(fill)) + idx[e + 1:]
        if len(idx) > self.ndim:
            raise IndexError(f"too many indices: tensor is {self.ndim}-d, got {len(idx)}")
        return idx + tuple(slice(None) for _ in range(self.ndim - len(idx)))

    def _dim_lists(self, idx):
        dim_lists, keep = [], []
        for d, ix in enumerate(idx):
            if isinstance(ix, slice):
                dim_lists.append(list(range(*ix.indices(self.shape[d]))))
                keep.append(True)
            else:
                dim_lists.append([_norm_int(int(ix), self.shape[d], d)])
                keep.append(False)
        return dim_lists, keep

    def __getitem__(self, idx):
        idx = self._expand(idx)
        dim_lists, keep = self._dim_lists(idx)
        out = bytearray(self.data[self._offset(m)] for m in itertools.product(*dim_lists))
        if not any(keep):
            return out[0]
        out_shape = tuple(len(dim_lists[d]) for d in range(self.ndim) if keep[d])
        return RingTensor(out, out_shape, self.unit)

    def __setitem__(self, idx, val):
        idx = self._expand(idx)
        dim_lists, _ = self._dim_lists(idx)
        positions = [self._offset(m) for m in itertools.product(*dim_lists)]
        if isinstance(val, RingTensor):
            vals = val.data
        elif isinstance(val, (list, tuple)):
            vals = [int(v) & 0xFF for v in val]
        else:
            vals = [int(val) & 0xFF for _ in positions]
        if len(vals) == 1 and len(positions) != 1:
            vals = [vals[0] for _ in positions]
        if len(vals) != len(positions):
            raise ValueError(f"cannot assign {len(vals)} values to {len(positions)} positions")
        for p, v in zip(positions, vals):
            self.data[p] = int(v) & 0xFF

    # ── shape ops ──
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = list(shape)
        if shape.count(-1) > 1:
            raise ValueError("can only specify one unknown (-1) dimension")
        if -1 in shape:
            known = 1
            for s in shape:
                if s != -1:
                    known = rn.mul(known, s)
            if known == 0 or rn.mf_mod(self.size, known) != 0:
                raise ValueError(f"cannot reshape size {self.size} into {tuple(shape)}")
            shape[shape.index(-1)] = rn.mf_floordiv(self.size, known)
        if _size(shape) != self.size:
            raise ValueError(f"cannot reshape size {self.size} into {tuple(shape)}")
        return RingTensor(self.data[:], tuple(shape), self.unit)

    def ravel(self):
        return RingTensor(self.data[:], (self.size,), self.unit)

    flatten = ravel

    def transpose(self, *axes):
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if sorted(axes) != list(range(self.ndim)):
            raise ValueError(f"invalid transpose axes {axes} for {self.ndim}-d tensor")
        new_shape = tuple(self.shape[a] for a in axes)
        out = []
        for m in itertools.product(*_ranges(new_shape)):
            old = [0 for _ in range(self.ndim)]
            for newpos, a in enumerate(axes):
                old[a] = m[newpos]
            out.append(self.data[self._offset(old)])
        return RingTensor(out, new_shape, self.unit)

    def swapaxes(self, a, b):
        axes = list(range(self.ndim))
        axes[a], axes[b] = axes[b], axes[a]
        return self.transpose(*axes)

    @property
    def T(self):
        return self.transpose()

    # ── elementwise (nD broadcasting) — compute maintained in the C buffer via kernels ──
    def _binary(self, other, op, unit=None):
        """op in 'ring_add'/'ring_sub'/'ring_mul'. Same-shape runs zero-copy through the C kernel;
        scalar/broadcast build the aligned buffers (gather in Python) then run the kernel."""
        u = unit or self.unit
        if not isinstance(other, RingTensor):
            o = int(other) & 0xFF
            ob = bytearray(o for _ in range(self.size))
            return RingTensor(_k.elementwise(op, self.data, ob), self.shape, u)
        if other.shape == self.shape:
            return RingTensor(_k.elementwise(op, self.data, other.data), self.shape, u)
        osh = _broadcast_shape(self.shape, other.shape)
        ab = bytearray(self.data[_bcast_off(self, m, osh)] for m in itertools.product(*_ranges(osh)))
        bb = bytearray(other.data[_bcast_off(other, m, osh)] for m in itertools.product(*_ranges(osh)))
        return RingTensor(_k.elementwise(op, ab, bb), osh, u)

    def radd(self, o):
        return self._binary(o, "ring_add")

    def rsub(self, o):
        return self._binary(o, "ring_sub")

    def rmul(self, o):
        return self._binary(o, "ring_mul", unit="energy")   # == qsm mod 256 (kernel-validated)

    def _binary_py(self, other, fn, unit=None):
        """Elementwise op via a Python fn (for ops not in the C kernel, e.g. division). Broadcasts."""
        u = unit or self.unit
        if not isinstance(other, RingTensor):
            o = int(other) & 0xFF
            return RingTensor(bytearray(fn(a, o) for a in self.data), self.shape, u)
        if other.shape == self.shape:
            return RingTensor(bytearray(fn(a, b) for a, b in zip(self.data, other.data)), self.shape, u)
        osh = _broadcast_shape(self.shape, other.shape)
        out = bytearray(fn(self.data[_bcast_off(self, m, osh)], other.data[_bcast_off(other, m, osh)])
                        for m in itertools.product(*_ranges(osh)))
        return RingTensor(out, osh, u)

    def rdiv(self, other):
        """Ring floor-division (mf_floordiv), elementwise + broadcast. Note: mod-256 division is
        exact only by odd (invertible) divisors; otherwise this is integer floor-division."""
        def _d(a, b):
            if b == 0:
                raise ZeroDivisionError("ring division by zero")
            return rn.mf_floordiv(a, b) & 0xFF
        return self._binary_py(other, _d)

    def __truediv__(self, o):
        return self.rdiv(o)

    def __floordiv__(self, o):
        return self.rdiv(o)

    def rneg(self):
        return RingTensor(bytearray(rn.ring_neg(a) for a in self.data), self.shape, self.unit)

    def apply(self, fn, unit=None):
        return RingTensor(bytearray(fn(a) & 0xFF for a in self.data), self.shape, unit or self.unit)

    def __add__(self, o): return self.radd(o)
    def __radd__(self, o): return self.radd(o)
    def __sub__(self, o): return self.rsub(o)
    def __mul__(self, o): return self.rmul(o)
    def __rmul__(self, o): return self.rmul(o)
    def __neg__(self): return self.rneg()
    def __matmul__(self, o): return matmul(self, o)

    # ── reductions (nD, keepdims, unit-aware) ──
    def _reduce(self, fn, axis, keepdims):
        if axis is None:
            r = fn(self.data)
            return RingTensor([r], (1,), self.unit) if keepdims else r
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        axes = tuple(a % self.ndim for a in axes)
        kept = [d for d in range(self.ndim) if d not in axes]
        out = []
        for km in itertools.product(*[range(self.shape[d]) for d in kept]):
            seq = []
            for rm in itertools.product(*[range(self.shape[d]) for d in axes]):
                full = [0 for _ in range(self.ndim)]
                for d, v in zip(kept, km):
                    full[d] = v
                for d, v in zip(axes, rm):
                    full[d] = v
                seq.append(self.data[self._offset(full)])
            out.append(fn(seq))
        if keepdims:
            out_shape = tuple(1 if d in axes else s for d, s in enumerate(self.shape))
        else:
            out_shape = tuple(s for d, s in enumerate(self.shape) if d not in axes)
        if not out_shape:
            return out[0]
        return RingTensor(out, out_shape, self.unit)

    def _ring_sum(self, seq):
        s = 0
        for v in seq:
            s = (s + v) & 0xFF
        return s

    def _ring_prod(self, seq):
        p = 1
        for v in seq:
            p = rn.qsm(p & 0xFF, v) & 0xFF
        return p

    def _center(self, seq):
        seq = list(seq)
        return rs.geometric_mean(seq) if self.unit == "energy" else rs.circular_mean(seq)

    def rsum(self, axis=None, keepdims=False):
        return self._reduce(self._ring_sum, axis, keepdims)

    def prod(self, axis=None, keepdims=False):
        return self._reduce(self._ring_prod, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return self._reduce(self._center, axis, keepdims)

    def median(self, axis=None, keepdims=False):
        return self._reduce(lambda s: rs.circular_median(list(s)), axis, keepdims)

    def min(self, axis=None, keepdims=False):
        return self._reduce(lambda s: min(s), axis, keepdims)

    def max(self, axis=None, keepdims=False):
        return self._reduce(lambda s: max(s), axis, keepdims)

    def argmin(self, axis=None, keepdims=False):
        return self._reduce(lambda s: min(range(len(s)), key=lambda i: s[i]), axis, keepdims)

    def argmax(self, axis=None, keepdims=False):
        return self._reduce(lambda s: max(range(len(s)), key=lambda i: s[i]), axis, keepdims)

    # ── dunders ──
    def __len__(self):
        if not self.shape:
            raise TypeError("len() of 0-d tensor")
        return self.shape[0]

    def __iter__(self):
        if self.ndim == 1:
            return iter(self.data)
        return (self[i] for i in range(self.shape[0]))

    def __eq__(self, other):
        return isinstance(other, RingTensor) and self.shape == other.shape and self.data == other.data

    __hash__ = None

    def __repr__(self):
        return f"RingTensor(shape={self.shape}, unit='{self.unit}', data={self.tolist()})"


# ── free helpers ──
def _bcast_off(t, out_multi, out_shape):
    off = 0
    nd, od = t.ndim, len(out_shape)
    for k in range(nd):
        oi = out_multi[od - nd + k]
        idx = 0 if t.shape[k] == 1 else oi
        off += rn.mul(idx, t.strides[k])
    return off


def _reshape_nested(flat, shape):
    if len(shape) <= 1:
        return list(flat)
    step = rn.mf_floordiv(len(flat), shape[0])
    return [_reshape_nested(flat[rn.mul(i, step):rn.mul(i + 1, step)], shape[1:]) for i in range(shape[0])]


def transpose(t, *axes):
    return t.transpose(*axes)


def _matmul2d(a, b):
    m, k = a.shape
    n = b.shape[1]
    out = []
    for i in range(m):
        row_i = [a[i, kk] for kk in range(k)]
        for j in range(n):
            acc = 0
            for kk in range(k):
                acc = (acc + rn.qsm(row_i[kk], b[kk, j])) & 0xFF
            out.append(acc)
    return RingTensor(out, (m, n), unit="energy")


def matmul(A, B):
    """Ring matmul (mod 256): QSM products + ring accumulation.
    Supports 1D vectors, 2D matrices, and batched nD (broadcast over leading dims)."""
    a1d, b1d = A.ndim == 1, B.ndim == 1
    a = A.reshape(1, A.shape[0]) if a1d else A
    b = B.reshape(B.shape[0], 1) if b1d else B
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"matmul inner dims mismatch: {A.shape} @ {B.shape}")
    if a.ndim == 2 and b.ndim == 2:
        C = _matmul2d(a, b)
    else:
        # batched: broadcast leading dims, matmul the trailing 2 axes
        abatch, bbatch = a.shape[:-2], b.shape[:-2]
        batch = _broadcast_shape(abatch, bbatch) if (abatch or bbatch) else ()
        m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
        blocks = []
        for bm in itertools.product(*_ranges(batch)):
            ai = _batch_slice(a, bm, 2)
            bi = _batch_slice(b, bm, 2)
            blocks.append(_matmul2d(ai, bi))
        out = []
        for blk in blocks:
            out.extend(blk.data)
        C = RingTensor(out, tuple(batch) + (m, n), unit="energy")
    if a1d and b1d:
        return C.data[0]
    if a1d:
        return C.reshape(*(C.shape[:-2] + (C.shape[-1],)))
    if b1d:
        return C.reshape(*(C.shape[:-2] + (C.shape[-2],)))
    return C


def _batch_slice(t, batch_multi, keep_last):
    """Index the leading (broadcast) dims of t with batch_multi, keep the last `keep_last` axes."""
    nd = t.ndim
    lead = nd - keep_last
    idx = []
    for d in range(lead):
        bd = t.shape[d]
        # align batch_multi (len = len(batch)) to the right of the lead dims
        bpos = d - (lead - len(batch_multi))
        idx.append(0 if bd == 1 or bpos < 0 else batch_multi[bpos])
    idx.extend(slice(None) for _ in range(keep_last))
    return t[tuple(idx)]


def concatenate(tensors, axis=0):
    ts = list(tensors)
    nd = ts[0].ndim
    for t in ts[1:]:
        if t.ndim != nd:
            raise ValueError("all tensors must have the same ndim to concatenate")
    out_dim = 0
    for t in ts:
        out_dim += t.shape[axis]
    out_shape = tuple(out_dim if d == axis else ts[0].shape[d] for d in range(nd))
    result = RingTensor([0 for _ in range(_size(out_shape))], out_shape, ts[0].unit)
    off = 0
    for t in ts:
        idx = tuple(slice(off, off + t.shape[axis]) if d == axis else slice(None) for d in range(nd))
        result[idx] = t
        off += t.shape[axis]
    return result


def stack(tensors, axis=0):
    ts = [t.reshape(*(t.shape[:axis] + (1,) + t.shape[axis:])) for t in tensors]
    return concatenate(ts, axis=axis)


def SIN(t):
    return t.apply(rn.SIN)


def COS(t):
    return t.apply(rn.COS)
