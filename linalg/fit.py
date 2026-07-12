"""
ring_fit.py — exact nonlinear fit by INVERT-THEN-SOLVE (on ring_native + ring_solve).

Math result (SRD T5.6): fitting act(W.x + b) = target is NOT a descent problem. On the ring the
activation is invertible (SIN -> ARCSIN gives a small preimage set per target), so:
  1. invert: each target's pre-activation arg lies in a finite preimage set,
  2. solve : pick one arg per point (for n = len(x)+1 points) and solve the LINEAR system exactly
             (ring_solve, needs an odd-determinant / invertible-mod-256 point set),
  3. verify: check the recovered params fit ALL data points.
This finds the exact solution where gradient descent hit local minima. No floats, multiplier-free.
"""
import itertools
from ringkit.core import native as rn
from ringkit.linalg import solve as rsolve


def sin_preimages(t):
    """All arcs a with SIN(a) == t (the ARCSIN preimage set)."""
    t &= 0xFF
    return [a for a in range(256) if rn.SIN(a) == t]


def _linear(params, x):
    acc = params[-1]                                   # bias
    for w, xi in zip(params[:-1], x):
        acc = (acc + rn.mul(w, xi)) & 0xFF
    return acc


def fit(data, targets, preimages=sin_preimages, activation=rn.SIN):
    """Recover params [w_0..w_{m-1}, b] with activation(W.x+b)==target for all points, exactly.
    Returns the param list, or None if unsatisfiable. n = m+1 points pin the params."""
    if not data:
        raise ValueError("fit: empty data")
    m = len(data[0])
    if any(len(x) != m for x in data):
        raise ValueError("fit: all data rows must have the same length")
    if len(targets) != len(data):
        raise ValueError(f"fit: {len(targets)} targets for {len(data)} data points")
    n = m + 1
    if len(data) < n:
        raise ValueError(f"fit: need at least {n} data points to pin {n} params, got {len(data)}")

    def fits(params):
        for x, t in zip(data, targets):
            if activation(_linear(params, x)) != (t & 0xFF):
                return False
        return True

    # order point-subsets by total preimage combinations (cheapest first)
    subsets = list(itertools.combinations(range(len(data)), n))
    subsets.sort(key=lambda s: _combo_size([preimages(targets[i]) for i in s]))
    for pts in subsets:
        A = [list(data[i]) + [1] for i in pts]
        if not rsolve.is_invertible(A):
            continue                                   # singular mod 2 -> skip whole subset
        pres = [preimages(targets[i]) for i in pts]
        for combo in itertools.product(*pres):
            sol = rsolve.solve(A, list(combo))
            if fits(sol):
                return sol
    return None


def _combo_size(pres):
    n = 1
    for p in pres:
        n = rn.mul(n, len(p)) if p else 0
    return n
