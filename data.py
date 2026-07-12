"""
ringkit.data — get ordinary data into and out of the ring, safely.

Engineers work with normal numbers and labels; this module maps them onto ring values (0..255)
and back, and provides the everyday plumbing (train/test split, batching, one-hot). The ring
specifics — mod-256 wrap, staying off the vacuum boundaries — are handled for you.

    import ringkit as rk
    Xr = rk.data.encode(X)                       # normal ints -> ring values
    (Xtr, Ytr), (Xte, Yte) = rk.data.split(Xr, Yr, test_frac=0.2)
    for xb, yb in rk.data.batches(Xtr, Ytr, size=32):
        ...
"""
import random as _random
from fractions import Fraction as _Fraction     # exact integer rationals (IO boundary, not std-math)
from ringkit.core import native as _rn


def _map(x, f):
    """Apply f elementwise over a scalar / vector / matrix, preserving structure."""
    if isinstance(x, (list, tuple)):
        return [_map(v, f) for v in x]
    return f(x)


def encode(values):
    """Map integers onto ring values 0..255 (wraps by mod-256). Scalar, vector, or matrix."""
    return _map(values, lambda v: int(v) & 0xFF)


def encode_range(values, lo, hi):
    """Scale numeric values in [lo, hi] onto the ring 0..255 (integer, round-to-nearest, no float)."""
    if hi <= lo:
        raise ValueError(f"encode_range: need hi > lo, got lo={lo}, hi={hi}")
    span = hi - lo

    def f(v):
        v = lo if v < lo else (hi if v > hi else v)
        num = _rn.mul(int(v) - lo, 255)                 # (v-lo)*255
        return _rn.mf_floordiv(_rn.mul(num, 2) + span, _rn.mul(span, 2)) & 0xFF   # round(num/span)
    return _map(values, f)


def one_hot(labels, num_classes):
    """Ring one-hot: label k -> vector with 1 at k, 0 elsewhere. Accepts a scalar or a list."""
    if isinstance(labels, (list, tuple)):
        return [one_hot(k, num_classes) for k in labels]
    k = int(labels)
    if not (0 <= k < num_classes):
        raise ValueError(f"one_hot: label {k} out of range [0,{num_classes})")
    return [1 if i == k else 0 for i in range(num_classes)]


def split(X, Y=None, test_frac=_Fraction(1, 5), seed=0):
    """Shuffle and split into train/test. Returns (Xtr, Xte) or ((Xtr,Ytr),(Xte,Yte)) if Y given."""
    n = len(X)
    # IO boundary: the engineer's fraction becomes an exact rational; from there the split
    # count is pure integer ring arithmetic (multiplier-free), never a float product.
    frac = _Fraction(test_frac).limit_denominator(1000000)
    if not (0 < frac < 1):
        raise ValueError(f"split: test_frac must be in (0,1), got {test_frac}")
    idx = list(range(n))
    _random.Random(seed).shuffle(idx)
    n_test = _rn.mf_floordiv(_rn.mul(n, frac.numerator), frac.denominator)
    test_i = set(idx[:n_test])
    Xtr = [X[i] for i in range(n) if i not in test_i]
    Xte = [X[i] for i in range(n) if i in test_i]
    if Y is None:
        return Xtr, Xte
    Ytr = [Y[i] for i in range(n) if i not in test_i]
    Yte = [Y[i] for i in range(n) if i in test_i]
    return (Xtr, Ytr), (Xte, Yte)


def batches(X, Y=None, size=32):
    """Yield (X_batch, Y_batch) — or just X_batch if Y is None — of at most `size` rows."""
    if size <= 0:
        raise ValueError(f"batches: size must be > 0, got {size}")
    n = len(X)
    i = 0
    while i < n:
        j = i + size
        if Y is None:
            yield X[i:j]
        else:
            yield X[i:j], Y[i:j]
        i = j
