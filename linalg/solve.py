"""
ring_solve.py — ring-native EXACT linear solve mod 256 (on ring_native).

Math result (SRD): a linear layer is a linear system mod 256, not an optimization problem.
It has a closed-form solution whenever the coefficient determinant is odd (invertible mod 256).
Gradient descent stalls on it only because the wrapping squared-loss is non-convex — so we
SOLVE linear layers (exact, 100%) and reserve descent for the nonlinear parts.

    modinv(a)   inverse of an odd a mod 256, via Newton/Hensel doubling (multiplier-free)
    solve(A,b)  solve A x = b (mod 256) by elimination with odd (invertible) pivots

Multiplier-free: rn.mul only.
"""
from ringkit.core import native as rn


def modinv(a):
    """Inverse of odd a mod 256. Newton/Hensel: x <- x*(2 - a*x), doubling valid bits
    2 -> 4 -> 16 -> 256. Raises for even a (not invertible mod 2)."""
    a &= 0xFF
    if a & 1 == 0:
        raise ValueError("even value has no inverse mod 256")
    x = 1                                   # a*1 == 1 (mod 2)
    for _ in range(3):                      # 2^2 -> 2^4 -> 2^8
        x = rn.mul(x, (2 - rn.mul(a, x))) & 0xFF
    return x


def _check_square(A, b=None):
    n = len(A)
    if n == 0:
        raise ValueError("solve: empty system")
    if any(len(row) != n for row in A):
        raise ValueError(f"solve: A must be square {n}x{n}, got row lengths {[len(r) for r in A]}")
    if b is not None and len(b) != n:
        raise ValueError(f"solve: b length {len(b)} != n {n}")
    return n


def solve(A, b):
    """Solve A x = b (mod 256) via Gaussian elimination with odd pivots. Exact.
    Raises ValueError if the system is not square/consistent or has no odd (invertible) pivot
    in some column (singular mod 2 — no unique solution over Z256)."""
    n = _check_square(A, b)
    M = [[v & 0xFF for v in A[i]] + [b[i] & 0xFF] for i in range(n)]
    for col in range(n):
        piv = None
        for r in range(col, n):
            if M[r][col] & 1:               # odd -> invertible mod 256
                piv = r
                break
        if piv is None:
            raise ValueError(f"no invertible pivot in column {col} (singular mod 2)")
        M[col], M[piv] = M[piv], M[col]
        inv = modinv(M[col][col])
        M[col] = [rn.mul(v, inv) & 0xFF for v in M[col]]
        for r in range(n):
            if r != col and (M[r][col] & 0xFF):
                f = M[r][col] & 0xFF
                M[r] = [(M[r][k] - rn.mul(f, M[col][k])) & 0xFF for k in range(n + 1)]
    return [M[i][n] & 0xFF for i in range(n)]


def is_invertible(A):
    """True iff A is invertible mod 256 (every column has an odd pivot). No raise."""
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        return False
    try:
        solve(A, [0 for _ in range(n)])
        return True
    except ValueError:
        return False
