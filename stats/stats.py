"""
ring_stats.py — ring-native central tendency for Z256 (QH4).

On a ring the linear average is wrong (mean of arcs 250 and 6 is ~0, not 128).
These are the circular / L1 forms, built on ring_native only.

  ring_dist(a,b)         circular L1 distance  min(|a-b|, 256-|a-b|)  in 0..128
  ARCTAN2(y,x)           ring atan2 -> arc 0..255 (direction of a vector)
  circular_mean(arcs)    resultant-vector mean direction (L2-style circular center)
  circular_median(arcs)  the arc minimizing total ring_dist (L1 / robust center)
  geometric_mean(vals)   multiplicative center: n=2 via isqrt(qsm); general via nth root
"""
from ringkit.core import native as rn

TAU = 256


def ring_dist(a, b):
    """Circular L1 distance on the ring: the shorter way around. 0..128."""
    d = (int(a) - int(b)) % TAU
    return d if d <= TAU - d else TAU - d


def ARCTAN2(y, x):
    """Ring atan2: arc 0..255 for the direction of vector (x ~ COS, y ~ SIN).
    Returns None for the zero vector (undefined direction)."""
    x = int(x)
    y = int(y)
    if x == 0 and y == 0:
        return None
    if x == 0:
        return rn.Q if y > 0 else rn.HALF + rn.Q    # +90 or +270 (192, no multiply)
    # base angle in first quadrant from |y|/|x|, as a SCALE-scaled tangent
    t = rn.mf_floordiv(rn.scale21(abs(y)), abs(x))   # SCALE*|y|//|x|
    base = rn.ARCTAN(t)                              # arc in [0,64)
    if x > 0 and y >= 0:
        return base % TAU                            # Q0
    if x < 0 and y >= 0:
        return (rn.HALF - base) % TAU                # Q1
    if x < 0 and y < 0:
        return (rn.HALF + base) % TAU                # Q2
    return (TAU - base) % TAU                         # Q3 (x>0, y<0)


def circular_mean(arcs):
    """Mean direction of ring arcs via the resultant vector (uses ring COS/SIN).
    Returns arc 0..255, or None if the resultant is the zero vector (e.g. antipodal).
    Raises ValueError on empty input."""
    arcs = list(arcs)
    if not arcs:
        raise ValueError("circular_mean: empty input")
    sx = sum(rn._signed(rn.COS(a)) for a in arcs)     # sum of x-components
    sy = sum(rn._signed(rn.SIN(a)) for a in arcs)     # sum of y-components
    return ARCTAN2(sy, sx)


def resultant_length(arcs):
    """Concentration |R| = sqrt(sx^2 + sy^2), ring-native (mul + general isqrt). Any arc count.
    Raises ValueError on empty input."""
    arcs = list(arcs)
    if not arcs:
        raise ValueError("resultant_length: empty input")
    sx = sum(rn._signed(rn.COS(a)) for a in arcs)
    sy = sum(rn._signed(rn.SIN(a)) for a in arcs)
    return rn.isqrt(rn.mul(sx, sx) + rn.mul(sy, sy))   # general isqrt handles any magnitude


def circular_median(arcs):
    """L1 circular center: the arc minimizing sum of ring_dist to the data.
    Returns the minimizing arc (ties -> smallest arc). Raises ValueError on empty input."""
    arcs = [int(a) % TAU for a in arcs]
    if not arcs:
        raise ValueError("circular_median: empty input")
    best_c, best_cost = 0, None
    for c in range(TAU):
        cost = 0
        for a in arcs:
            cost += ring_dist(c, a)
        if best_cost is None or cost < best_cost:
            best_c, best_cost = c, cost
    return best_c


def geometric_mean(vals):
    """Multiplicative center (n-th root of the product) for non-negative ring magnitudes.
    n=2 is fully multiplier-free: isqrt(qsm(a,b)). n>2 uses the product ring + integer n-th root.
    Raises ValueError on empty input or negative values."""
    vals = [int(v) for v in vals]
    n = len(vals)
    if n == 0:
        raise ValueError("geometric_mean: empty input")
    if any(v < 0 for v in vals):
        raise ValueError("geometric_mean: values must be non-negative ring magnitudes")
    if n == 1:
        return vals[0]
    if n == 2:
        return rn.isqrt_lut(rn.qsm(vals[0], vals[1]))   # sqrt(a*b), no multiply
    prod = 1
    for v in vals:
        prod = rn.mul(prod, v)                           # product ring (shift-add, no '*')
    return _iroot(prod, n)


def _iroot(x, n):
    """Integer n-th root: largest r with r**n <= x. Binary search, no '*'/'**' (rn.ipow)."""
    if x < 0:
        raise ValueError
    if x == 0:
        return 0
    lo, hi = 0, 1
    while rn.ipow(hi, n) <= x:
        hi <<= 1
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if rn.ipow(mid, n) <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo
