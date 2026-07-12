"""
ringkit.core.native — MPRC Z256 (QH4) ring-native math: the foundation / ISA.

All integer, mod 256, fully multiplier-free: no '*', '//', '**', '/' anywhere (verified by
AST audit). Multiply/divide/power are realized as mul / mf_floordiv / ipow (shift-add) and
qsm (quarter-square). Public forms validate their domains and raise ValueError / ZeroDivisionError
with messages — no silent out-of-range. Every identity is exhaustively tested over the 256 ring
in tests/test_native.py.

Sections
  1. Ring constants
  2. Multiplier-free primitives   (_SQ, qsm, isqrt_lut, scale21, mf_floordiv, ring_neg)
  3. Ring trig                    (_arch, SIN, COS, TAN, KS4)
  4. Ring complex / rotor         (qh_iota, iota_mul, polar_axis)
  5. ADI engine                   (derived_delta, recover, compress, evolve, mprc_axis_arcs)
  6. Time codec (kinematics)      (scale, encode, decode)
"""

# ── 1. Ring constants ────────────────────────────────────────────────────────
# Single-sourced in core/constants.py (frozen there — the ring's identity, not overridable).
from ringkit.core.constants import TAU, HALF, Q, Q2, SCALE, VACUUMS, RING_E, IOTA


# ── 2. Multiplier-free primitives ────────────────────────────────────────────
# _SQ[n] = n^2, built by odd-number accumulation (no multiply): n^2 = (n-1)^2 + (2n-1).
# Sized to the PRODUCT ring so qsm is total: accumulation |x+y| reaches 510 for ring pairs.
_NSQ = 512
_SQ = [0 for _ in range(_NSQ + 1)]
for _n in range(1, _NSQ + 1):
    _SQ[_n] = _SQ[_n - 1] + ((_n - 1) << 1) + 1


def ring_neg(x):
    """-x mod 256  (ring_neg(21) = 235)."""
    return (-x) & 0xFF


def qsm(x, y):
    """Exact product x*y with no multiply (quarter-square; mints the product ring C).
    accumulation s = |x+y|, differential d = |x-y|;  x*y = (s^2 - d^2) >> 2.
    Domain: |x+y| <= _NSQ (=512), which holds for all ring pairs (|x|,|y| <= 255)."""
    s = abs(x + y)
    d = abs(x - y)
    if s > _NSQ:
        raise ValueError(f"qsm: |x+y|={s} exceeds product-ring table {_NSQ} (x={x}, y={y})")
    return (_SQ[s] - _SQ[d]) >> 2


def isqrt_lut(m):
    """Largest r with r^2 <= m, via binary search over _SQ. No multiply.
    Domain: 0 <= m <= _NSQ^2 (=262144)."""
    if m < 0:
        raise ValueError(f"isqrt_lut: m must be >= 0, got {m}")
    if m > _SQ[_NSQ]:
        raise ValueError(f"isqrt_lut: m={m} exceeds table range {_SQ[_NSQ]}")
    lo, hi = 0, _NSQ
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if _SQ[mid] <= m:
            lo = mid
        else:
            hi = mid - 1
    return lo


def isqrt(m):
    """General integer sqrt: floor(sqrt(m)) for ANY m>=0. Digit-by-digit, no multiply
    (shifts/add/sub only). Use this for unbounded m; isqrt_lut is the fast table path for
    m <= _NSQ^2 used inside the trig kernel."""
    if m < 0:
        raise ValueError(f"isqrt: m must be >= 0, got {m}")
    if m == 0:
        return 0
    x = 0
    c = 1
    while (c << 2) <= m:
        c <<= 2                       # highest power of 4 <= m
    while c != 0:
        if m >= x + c:
            m -= x + c
            x = (x >> 1) + c
        else:
            x >>= 1
        c >>= 2
    return x


def scale21(r):
    """21*r as the XYZ scalar-axis sum (strides 4^0,4^1,4^2): (r<<4)+(r<<2)+r. No multiply."""
    return (r << 4) + (r << 2) + r


def mf_floordiv(n, d):
    """floor(n/d) for n>=0, d>0 via shift-subtract long division. No '//', no multiply."""
    if d == 0:
        raise ZeroDivisionError("mf_floordiv: division by zero")
    if d < 0 or n < 0:
        raise ValueError(f"mf_floordiv requires n>=0, d>0; got n={n}, d={d}")
    q = 0
    rem = 0
    for i in range(n.bit_length() - 1, -1, -1):
        rem = (rem << 1) | ((n >> i) & 1)
        if rem >= d:
            rem -= d
            q |= (1 << i)
    return q


def mul(a, b):
    """Product a*b with NO '*' — shift-and-add (repeated doubling). Any size, signed."""
    neg = (a < 0) ^ (b < 0)
    a = -a if a < 0 else a
    b = -b if b < 0 else b
    r = 0
    while b:
        if b & 1:
            r += a
        a <<= 1
        b >>= 1
    return -r if neg else r


def ipow(base, n):
    """base**n with NO '**'/'*' — repeated shift-add mul. n >= 0."""
    if int(n) < 0:
        raise ValueError(f"ipow: exponent must be >= 0, got {n}")
    r = 1
    for _ in range(int(n)):
        r = mul(r, base)
    return r


def ring_pow(base, exp):
    """base**exp (mod 256) — multiplier-free square-and-multiply, masked each step. exp >= 0.
    General ring growth for ANY base. (For the natural base RING_E=3 prefer ring_exp: it is the
    bijective, period-64, phase-locked exponential. ring_pow of an EVEN base collapses to 0.)"""
    if int(exp) < 0:
        raise ValueError(f"ring_pow: exponent must be >= 0, got {exp}")
    b = base & 0xFF
    r = 1
    e = int(exp)
    while e:
        if e & 1:
            r = mul(r, b) & 0xFF
        b = mul(b, b) & 0xFF
        e >>= 1
    return r


def mf_mod(n, d):
    """n mod d for n>=0, d>0 with no '%'/'*' — n - d*floor(n/d)."""
    if d <= 0 or n < 0:
        raise ValueError(f"mf_mod requires n>=0, d>0; got n={n}, d={d}")
    return n - mul(d, mf_floordiv(n, d))


# ── 2b. Ring-native e — the exponential base, earned (not a decimal) ──────────
# The continuous e (2.718..) is the base whose exponential is the eigenfunction of d/dx,
# and it is the base of exp/log. It has NO stable image on the ring: round(e*128)=348=92,
# and 92 is even (a zero-divisor) so 92^4 = 0 — continuous e collapses the orbit.
#
# The element that EARNS e's role on Z256 is the integer 3 — the canonical generator of the
# unit group (Z/256)* = {+-1} x <3> (verified: {+-1}x<3> is all 128 units). It earns it because:
#   * base of a discrete exp/log: n -> 3^n is a bijection Z/64 <-> <3>, EXACT integers, zero
#     residual (unlike E_hat*21 = 49.47, which is not exact);
#   * non-degenerate: 3 is odd -> a unit, it NEVER collapses (immune to the even black-hole);
#   * eigenfunction of the ring difference operator: Delta(3^n) = 3^(n+1)-3^n = 2*3^n  (exact,
#     the discrete counterpart of (e^x)' = e^x, eigenvalue g-1 = 2);
#   * phase-locked to rotation: ord(3) = 64 exactly, so 3^n resets to 1 precisely at the
#     64-step quarter turn (one iota) — amplitude growth locked to orthogonal rotation;
#   * canonical & rooted in the constants: 3 = master N, and SCALE=21 = 3*7.
# (The QH4 doc's "E_hat numerator 173" is exactly -3^45 mod 256 — it was a power of 3 all along.)
# RING_E = 3 (imported from core.constants): the exponential base / unit generator

_E_ORBIT = []                    # 3^k mod 256, k = 0..63  (the natural exponential's image)
_e_acc = 1
for _k in range(Q):              # Q = 64 = ord(3) = quarter turn
    _E_ORBIT.append(_e_acc)
    _e_acc = mul(_e_acc, RING_E) & 0xFF
_E_LOG = {v: k for k, v in enumerate(_E_ORBIT)}   # discrete log on the subgroup <3>


def ring_exp(n, sign=0):
    """Ring-native exponential: ((-1)^sign) * RING_E^n  (mod 256). Period 64 (one iota).
    Bijection Z/64 <-> <3> for sign=0. Multiplier-free, EXACT — the discrete counterpart of e^x."""
    v = _E_ORBIT[n & (Q - 1)]            # n mod 64 (Q is a power of two)
    return ring_neg(v) if (sign & 1) else v


def ring_log(u):
    """Ring-native natural log (discrete log base RING_E=3). Returns (k, sign) so that
    ring_exp(*ring_log(u)) == ring_exp(k, sign) == u, for any unit u (odd) — the exact inverse.
    Raises ValueError for even u — a zero-divisor has no log (it collapses to 0, not a unit)."""
    u &= 0xFF
    if (u & 1) == 0:
        raise ValueError(f"ring_log: {u} is even (a zero-divisor); it has no ring-e log — it collapses")
    if u in _E_LOG:
        return (_E_LOG[u], 0)
    return (_E_LOG[ring_neg(u)], 1)       # u = -3^k  (every odd unit is +-3^k)


# The rotor / iota: Z256 has NO scalar sqrt(-1) (no x with x^2 == 255), so "i" is not a ring
# number — it is the 2x2 rotor operator J = [[0,-1],[1,0]], with J^2 = -I and J^4 = I (the 4
# iotas). Growth (base RING_E=3, multiplicative) and rotation (J, structural) are therefore
# SEPARATE operators on the ring; standard math fuses them only because C has a scalar i.
# IOTA (imported from core.constants): J = [[0,-1],[1,0]] mod 256 : i^2 = -I, i^4 = I


def ring_cis(phi):
    """Ring-native Euler's formula:  e^{i*phi}  ==  (COS(phi), SIN(phi))  — the (real, imag)
    components on the radius-SCALE circle (rotation is structural via IOTA, not a scalar i).
    Euler's identity, ring form:  ring_cis(HALF) = (-SCALE, 0), i.e.  e^{i*pi} = -SCALE, so
    e^{i*pi} + SCALE == 0  (the ring's 'e^{i*pi}+1=0', in amplitude units).  ring_cis(Q) = (0, SCALE)
    is e^{i*pi/2} = i (a pure iota step)."""
    return (COS(phi), SIN(phi))


def rotate(phi, quarters=1):
    """Rotate an ANGLE by `quarters` iota steps (one step = 90 deg = 64). Exact; `quarters`
    may be negative. This is the structural rotor acting on the position (not a scalar product)."""
    return (phi + mul(Q, quarters)) & 0xFF


def cis_rotate(c, s, quarters=1):
    """Apply the IOTA rotor to a cis pair (c, s) = (cos, sin), `quarters` times. EXACT (i^4=1);
    one step sends (c, s) -> (-s, c). Composition law (verified exact for ALL angles):
        cis_rotate(*ring_cis(phi), k) == ring_cis(rotate(phi, k)).
    NOTE: only iota (quarter-turn) composition is exact. General angle-addition
    cis(a)(x)cis(b)=cis(a+b) is only APPROXIMATE off the cardinals, because SIN/COS use the
    semicircle _arch, not analytic sine — so it is deliberately not offered here."""
    k = quarters & 3
    for _ in range(k):
        c, s = ring_neg(s), c
    return (c, s)


# ── 3. Ring trig ─────────────────────────────────────────────────────────────
def _arch(pos, P):
    """One positive lobe over [0,P] (semicircle projection). P must be a power of two."""
    if pos <= 0 or pos >= P:
        return 0
    r = isqrt_lut(qsm(pos, P - pos))       # radius = sqrt(pos*(P-pos)), multiplier-free
    return (scale21(r) << 1) >> (P.bit_length() - 1)   # 21*2*r // P, shifts only


def SIN(phi):
    """Ring sine on Z256: zeros at {0,128}, peak +21 at 64, trough 235(=-21) at 192. Any int phi."""
    phi = int(phi) % TAU
    if phi < HALF:
        return _arch(phi, HALF)
    return ring_neg(_arch(phi - HALF, HALF))


def COS(phi):
    """Ring cosine = SIN(phi + quadrant). Zeros at {64,192}, peak +21 at 0."""
    return SIN((int(phi) + Q) % TAU)


def TAN(phi):
    c = COS(phi)
    if c == 0:
        return "VACUUM"
    return mf_floordiv(scale21(SIN(phi)), c)   # (SIN*21)//COS, no *, no //


def KS4(phi):
    """4-cycle sine: one lobe per quadrant, zeros at all four vacuums."""
    phi = int(phi) % TAU
    cycle = phi & (Q - 1)                       # phi % 64
    if cycle < Q2:
        return _arch(cycle, Q2)
    return ring_neg(_arch(cycle - Q2, Q2))


def _signed(x):
    """ring value 0..255 -> signed -128..127 (235 -> -21)."""
    x &= 0xFF
    return x - TAU if x > HALF else x


# NOTE: SIN512 (the 512 double-cover / spinor wave) was moved OUT of the core to the
# measurement layer (ring_measure.measure_sin(phi, 512)) — 512 is ENERGY-side
# (U-accumulator + overspill), returns amplitude not a ring position, and did not earn
# a place in the 256 core identity. The core stays ARC/identity only.


# ── 3b. Reciprocal trig (sign-correct, scaled by SCALE^2=441) ────────────────
# Unit-consistent with TAN (which carries SCALE): value 441 means 1.0.
# Return "VACUUM" at poles. Sign is carried correctly (unlike raw-ring floor div).
_SCALE2 = scale21(SCALE)                         # 21*21 = 441, multiplier-free


def SEC(phi):
    """secant = SCALE^2 // COS, sign-correct. VACUUM where COS=0 (phi in {64,192})."""
    c = _signed(COS(phi))
    if c == 0:
        return "VACUUM"
    mag = mf_floordiv(_SCALE2, abs(c))
    return mag if c > 0 else -mag


def CSC(phi):
    """cosecant = SCALE^2 // SIN, sign-correct. VACUUM where SIN=0 (phi in {0,128})."""
    s = _signed(SIN(phi))
    if s == 0:
        return "VACUUM"
    mag = mf_floordiv(_SCALE2, abs(s))
    return mag if s > 0 else -mag


def COT(phi):
    """cotangent = SCALE*COS // SIN, sign-correct. VACUUM where SIN=0 (phi in {0,128})."""
    s = _signed(SIN(phi))
    c = _signed(COS(phi))
    if s == 0:
        return "VACUUM"
    mag = mf_floordiv(scale21(abs(c)), abs(s))
    return -mag if (c < 0) ^ (s < 0) else mag


# ── 3c. Inverse trig (principal-branch reverse lookup) ───────────────────────
# Reverse the SIN/COS/TAN tables: given a signed value, return the arc phi (ring 0..255).
# Principal branches: ARCSIN [-64,64], ARCCOS [0,128], ARCTAN (-64,64).
# Coarse LUT -> returns the closest arc; SIN(ARCSIN(v)) reproduces representable v exactly.
def ARCSIN(v):
    """arcsin: signed value in [-SCALE,SCALE] -> principal arc. phi>=0 for v>=0, else mirror."""
    a = min(range(0, Q + 1), key=lambda p: abs(_signed(SIN(p)) - abs(int(v))))
    return a if v >= 0 else ring_neg(a)


def ARCCOS(v):
    """arccos: signed value in [-SCALE,SCALE] -> principal arc in [0,128]."""
    return min(range(0, HALF + 1), key=lambda p: abs(_signed(COS(p)) - int(v)))


def ARCTAN(v):
    """arctan: signed (SCALE-scaled) tangent -> principal arc in (-64,64) via the clean [0,64) branch."""
    v = int(v)
    a = min(range(0, Q), key=lambda p: abs(TAN(p) - abs(v)))   # TAN is well-defined on [0,64)
    return a if v >= 0 else ring_neg(a)


# ── 4. Ring complex / rotor ──────────────────────────────────────────────────
def qh_iota(phi):
    """Imaginary-unit power i0,i1,i2,i3 = quadrant index phi>>6 (4-cycle, values SCALE*{1,i,-1,-i})."""
    return (phi & 0xFF) >> 6


def iota_mul(phi):
    """Multiply by iota = quarter-turn rotation as a +64 arc shift. iota^4=1, iota^2=-1."""
    return (phi + Q) & 0xFF


def polar_axis(phi):
    """Polar axis = the 4 quadrant values as (quadrant, cos_sign, sin_sign).
    cos_sign = +1/-1 (right/left), sin_sign = +1/-1 (UP/DOWN); the (+/-, UP/DOWN) fold."""
    q = qh_iota(phi)
    cos_sign = 1 if q in (0, 3) else -1
    sin_sign = 1 if q in (0, 1) else -1
    return q, cos_sign, sin_sign


# ── 5. ADI engine (accumulation / differential) ──────────────────────────────
def derived_delta(delta1, k):
    """delta_k = (delta_1 + k^2 - 1) mod 256; consecutive gaps are the odd numbers 3,5,7,...
    k^2 via _SQ (no multiply)."""
    return (int(delta1) + _SQ[int(k)] - 1) % TAU


def recover(lead, delta1, n):
    """(lead, delta_1) -> full n-vector via the odd-increment rule. Exact."""
    lead = int(lead) % TAU
    return [lead] + [(lead - derived_delta(delta1, i)) % TAU for i in range(1, n)]


def compress(coords):
    """n-vector -> (Lambda, delta_1, lead); raises ValueError if not ADI-consistent."""
    coords = [int(c) % TAU for c in coords]
    a1 = coords[0]
    delta1 = (a1 - coords[1]) % TAU
    for i in range(1, len(coords)):
        if (a1 - coords[i]) % TAU != derived_delta(delta1, i):
            raise ValueError(f"ADI violation at delta_{i}")
    return sum(coords) % TAU, delta1, a1


def evolve(lam, delta1, lead, freqs, dt, n):
    """One ADI wave step, pure modular addition. Requires freqs[1:] equal. No '*'."""
    f0 = int(freqs[0])
    f1 = int(freqs[1])
    dt = int(dt)
    lam_step = (f0 + mul(n - 1, f1)) & 0xFF
    del_step = (f0 - f1) & 0xFF
    return (
        (int(lam) + mul(dt, lam_step)) & 0xFF,
        (int(delta1) + mul(dt, del_step)) & 0xFF,
        (int(lead) + mul(dt, f0)) & 0xFF,
    )


def mprc_axis_arcs(a1, a2):
    """Two primary arcs -> the 4 ADI-locked MPRC axis arcs (a1,a2,a3,a4)."""
    a1 %= TAU
    a2 %= TAU
    delta1 = (a1 - a2) % TAU
    a3 = (a1 - derived_delta(delta1, 2)) % TAU
    a4 = (a1 - derived_delta(delta1, 3)) % TAU
    return a1, a2, a3, a4


# ── 6. Time codec (kinematics: clock <-> ring) ───────────────────────────────
# Float-free integer codec. SCALE_n = 675 * 1024^n (675 = 86400/128 = seconds/half-ring;
# 1024 = 4*TAU sub-step). Uses integer '*' and '//' (float-free, NOT multiplier-free).
# Lossless within ONE ring traversal: t in [0, 128*SCALE_n) = [0, 86400*1024^n).
# Beyond that, state wraps by the ring modulus (by design). Vacuum nodes land at r=0.
_SCALE0 = 675
_RING_STEP = 1024


def scale(level=0):
    """SCALE_n = 675 * 1024^n  (n: 0=second, 1=ms, ...). No '*'/'**': 1024=2^10 -> shift, then mul."""
    if int(level) < 0:
        raise ValueError(f"scale: level must be >= 0, got {level}")
    p = 1
    for _ in range(int(level)):
        p <<= 10                       # * 1024
    return mul(_SCALE0, p)


def encode(t, level=0):
    """seconds -> (state in 0..255, residual r in 0..SCALE_n-1). Lossless within one ring. No '*'/'//'."""
    if int(t) < 0:
        raise ValueError(f"encode: t must be >= 0, got {t}")
    sc = scale(level)
    num = int(t) << 1                  # 2*t
    fd = mf_floordiv(num, sc)
    return fd & 0xFF, num - mul(sc, fd)   # (num//sc)%256 , num%sc


def decode(state, r, level=0):
    """(state, r) -> seconds. Exact inverse of encode (num always even -> >>1). No '*'/'//'."""
    if int(state) < 0 or int(r) < 0:
        raise ValueError(f"decode: state, r must be >= 0; got state={state}, r={r}")
    num = mul(int(state), scale(level)) + int(r)
    return num >> 1                    # // 2
