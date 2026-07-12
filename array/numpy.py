"""
ringkit.array.numpy — our own numpy, ring-native (Z256). NOT the standard numpy.

A numpy-style ndarray namespace over RingTensor. Same feel (array/zeros/arange/eye, @/.T,
sum/prod/mean/min/max/argmin/argmax, reshape/transpose/concatenate/stack), but every element is
a ring value 0..255 and every op is ring-native + multiplier-free. Unit tag: 'arc' | 'energy'.

    import ringkit.array.numpy as rnp
    a = rnp.arange(12).reshape(3, 4);  b = rnp.eye(4);  c = a @ b;  d = rnp.sum(a, axis=0)
"""
from ringkit.core import native as rn
from ringkit.array.tensor import (
    RingTensor, matmul, transpose as _transpose, concatenate as _concat, stack as _stack,
    SIN as _SIN, COS as _COS,
)

TAU = 256
E_TAU = 2.844444444444444444
E_PI = 1.422222222222222222
E_PI2 = 0.711111111111111111

# ── creation ──
def array(data, unit="arc"):
    return RingTensor(data, unit=unit)


def _n(shape):
    n = 1
    for s in shape:
        n = rn.mul(n, s)
    return n


def full(shape, value, unit="arc"):
    v = int(value) & 0xFF
    return RingTensor([v for _ in range(_n(shape))], tuple(shape), unit)


def zeros(shape, unit="arc"):
    return full(shape, 0, unit)


def ones(shape, unit="arc"):
    return full(shape, 1, unit)


def empty(shape, unit="arc"):
    return zeros(shape, unit)


def zeros_like(t):
    return full(t.shape, 0, t.unit)


def ones_like(t):
    return full(t.shape, 1, t.unit)


def full_like(t, value):
    return full(t.shape, value, t.unit)


def arange(k, unit="arc"):
    return RingTensor([i & 0xFF for i in range(int(k))], (int(k),), unit)


def eye(n, unit="arc"):
    return RingTensor([1 if i == j else 0 for i in range(n) for j in range(n)], (n, n), unit)


identity = eye


# ── manipulation ──
def reshape(t, *shape):
    return t.reshape(*shape)


def ravel(t):
    return t.ravel()


def flatten(t):
    return t.ravel()


def transpose(t, *axes):
    return _transpose(t, *axes)


def swapaxes(t, a, b):
    return t.swapaxes(a, b)


def concatenate(ts, axis=0):
    return _concat(ts, axis)


def stack(ts, axis=0):
    return _stack(ts, axis)


def hstack(ts):
    return _concat(ts, 1 if ts[0].ndim > 1 else 0)


def vstack(ts):
    return _concat(ts, 0)


# ── linear algebra ──
def dot(a, b):
    return matmul(a, b)


def matmul_(a, b):
    return matmul(a, b)


# ── reductions ──
def sum(t, axis=None, keepdims=False):
    return t.rsum(axis, keepdims)


def prod(t, axis=None, keepdims=False):
    return t.prod(axis, keepdims)


def mean(t, axis=None, keepdims=False):
    return t.mean(axis, keepdims)


def median(t, axis=None, keepdims=False):
    return t.median(axis, keepdims)


def amin(t, axis=None, keepdims=False):
    return t.min(axis, keepdims)


def amax(t, axis=None, keepdims=False):
    return t.max(axis, keepdims)


def argmin(t, axis=None, keepdims=False):
    return t.argmin(axis, keepdims)


def argmax(t, axis=None, keepdims=False):
    return t.argmax(axis, keepdims)


# ── elementwise / trig ──
def add(a, b):
    return a.radd(b)


def subtract(a, b):
    return a.rsub(b)


def multiply(a, b):
    return a.rmul(b)


def negative(a):
    return a.rneg()


def sin(t):
    return _SIN(t)


def cos(t):
    return _COS(t)


def _rnd(n, m):                       # round(n/m), float-free
    return rn.mf_floordiv(rn.mul(n, 2) + m, rn.mul(m, 2))

def deg_to_arc2(d):                    # degrees -> double-cover arc (512). Lossless.
    return _rnd(rn.mul(int(d), 512), 360)     # round(d*512/360)

def arc2_to_deg(a):                    # double-cover arc -> degrees. Exact inverse.
    return _rnd(rn.mul(int(a), 360), 512)     # round(a*360/512)


def pi():
    """The ring position of pi: HALF = 128 (SIN(128)=0, COS(128)=-21). unit='arc'."""
    return RingTensor([128], (1,), unit="arc")


def tau():
    """The ring position of 2*pi = full turn: TAU = 256 == 0 (mod ring). unit='arc'."""
    return RingTensor([0], (1,), unit="arc")