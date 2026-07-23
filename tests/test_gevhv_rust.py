"""Test for the GEVHV CPU silicon — the RUST backend (SPEC-014: CPU = Rust).

The GEVHV ops (react / measure / react_bound_scalar / offset_field / react_bound_vector)
live in ringkit/kernels/rust (ring_rust) behind the cpu_rust host; this suite gates them
BIT-FOR-BIT against the pure-Python judge ringkit.ml.gevhv (D9). Rust forces the ring
type-discipline the C draft left to vigilance: unsigned phases, explicit wrapping_sub,
no float — see kernels/rust/src/lib.rs GEVHV section.
Run: python -m ringkit.tests.test_gevhv_rust"""
import random
from ringkit.kernels.cpu_rust import host as rh
from ringkit.ml import gevhv as gv

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

random.seed(90210)
H, W = gv.H, gv.W
NSITES = H * W
rng = random.Random(90210)

def rand_grid():
    return [rng.randint(0, 255) for _ in range(NSITES)]

def rand_lut():
    return [rng.randint(0, 255) for _ in range(256)]

GRIDS = [rand_grid(), rand_grid(), [0] * NSITES, [128] * NSITES, [255] * NSITES]
LUTS = [rand_lut() for _ in GRIDS]

print("== availability: the Rust CPU backend built + self-tested (incl. GEVHV) at load ==")
avail = rh.available()
check("cpu_rust host.available() is True (ring_rust.dll built, selftest incl. _gevhv_selftest passed)", avail)
if not avail:
    # No fallback path to test if the backend didn't load — report and stop honestly.
    print("RESULT:", "ALL PASS" if not fails else f"FAIL ({len(fails)}): {fails}")
    raise SystemExit(0 if not fails else 1)

print("== no-regression: ml/gevhv.py is the judge, called never edited ==")
check("gv.H == 128, gv.W == 113", gv.H == 128 and gv.W == 113)
check("react(all-0, identity LUT) == all-0 (judge's own answer)",
      gv.react([0] * NSITES, list(range(256))) == [0] * NSITES)

print("== (a) react: rust == ml.gevhv.react, every adversarial + random manifold ==")
ok = True
for g, lut in zip(GRIDS, LUTS):
    if rh.gevhv_react(g, lut, H, W) != gv.react(g, lut):
        ok = False
check(f"gevhv_react bit-for-bit on {len(GRIDS)} manifolds", ok)

print("== (b) measure: rust u32 (Theorem G width) == ml.gevhv.measure, full + interior ==")
q = rand_grid()
ok = True
for g in GRIDS:
    for interior in (False, True):
        if rh.gevhv_measure(g, q, H, W, interior=interior) != gv.measure(g, q, interior=interior):
            ok = False
check("u32 measure matches the judge, full-grid and interior", ok)
e_int = rh.gevhv_measure([0] * NSITES, [128] * NSITES, H, W, interior=True)
e_full = rh.gevhv_measure([0] * NSITES, [128] * NSITES, H, W, interior=False)
check(f"adversarial interior energy == 13,986*128 = 1,790,208 (got {e_int})", e_int == 1790208)
check(f"adversarial full-grid energy == 14,464*128 = 1,851,392 (got {e_full})", e_full == 1851392)

print("== (c) react_bound_scalar: rust == judge, 50 random (s,t,lut) + odd/even-s rejection ==")
ok = True
for _ in range(50):
    g = rand_grid(); lut = rand_lut()
    s = rng.randrange(1, 256, 2); t = rng.randint(0, 255)
    if rh.gevhv_react_bound_scalar(g, lut, s, t, H, W) != gv.react_bound_scalar(g, lut, s, t):
        ok = False
check("50 random (s,t,lut) triples, bit-for-bit", ok)
check("even s is refused by the rust kernel (returns None), mirroring gv's ValueError",
      rh.gevhv_react_bound_scalar(GRIDS[0], LUTS[0], 2, 5, H, W) is None)
check("odd s is served normally",
      rh.gevhv_react_bound_scalar(GRIDS[0], LUTS[0], 3, 5, H, W) is not None)

print("== (d) offset_field + react_bound_vector: rust == judge, batch sharing one c ==")
v = rand_grid()
c = rh.gevhv_offset_field(v, H, W)
check("offset_field bit-for-bit", c == gv.offset_field(v))
ok = True
for g, lut in zip(GRIDS, LUTS):
    if rh.gevhv_react_bound_vector(g, lut, v, c, H, W) != gv.react_bound_vector(g, lut, v, c=c):
        ok = False
check("react_bound_vector bit-for-bit on all manifolds sharing one c", ok)

print("== (e) gevhv_scores: rust == ml.attention.scores (the Q·Kᵀ gemm-role replacement) ==")
from ringkit.ml.attention import scores as _scores
dim, nq, nk = 24, 6, 9
Q = [[rng.randint(0, 255) for _ in range(dim)] for _ in range(nq)]
K = [[rng.randint(0, 255) for _ in range(dim)] for _ in range(nk)]
qf = [v for r in Q for v in r]; kf = [v for r in K for v in r]
got = rh.gevhv_scores(qf, kf, nq, nk, dim)
want = [s for row in _scores(Q, K) for s in row]
check("gevhv_scores flat matrix == -Σ ring_distance, bit-for-bit", got == want)
check("negative energy: identical rows score 0, distinct rows score −dim·128",
      rh.gevhv_scores([0] * dim, [0] * dim, 1, 1, dim)[0] == 0
      and rh.gevhv_scores([0] * dim, [128] * dim, 1, 1, dim)[0] == -dim * 128)

print("== STAGE 2 — (f) gevhv_gemv_radix: Theorem C multiplier-free gemv == exact dot ==")
Mg, Kg = 7, 50
wv = [rng.randint(-1000, 1000) for _ in range(Mg * Kg)]
xv = [rng.randint(0, 255) for _ in range(Kg)]
gr = rh.gevhv_gemv_radix(wv, xv, Mg, Kg)
want_dot = [sum(wv[r * Kg + i] * xv[i] for i in range(Kg)) for r in range(Mg)]
check("gevhv_gemv_radix == Σ w·x, bit-for-bit (shift-add, no multiply)", gr == want_dot)
from ringkit.ml import gevhv as _gv
check("ml.gevhv.gemv_radix routes through the kernel and matches the reference",
      _gv.gemv_radix([wv[r * Kg:(r + 1) * Kg] for r in range(Mg)], xv) == want_dot)

print("== STAGE 2 — (g) gevhv_gemm_arc: multiplier-free arc GEMM == rn.mul reference ==")
from ringkit.core import native as _rn
Ma, Ka, Na = 5, 16, 6
A = [rng.randint(0, 255) for _ in range(Ma * Ka)]
B = [rng.randint(0, 255) for _ in range(Ka * Na)]
ca = list(rh.gevhv_gemm_arc(bytes(A), bytes(B), Ma, Ka, Na))
want_arc = [sum(_rn.mul(A[i * Ka + kk], B[kk * Na + j]) for kk in range(Ka)) & 0xFF
            for i in range(Ma) for j in range(Na)]
check("gevhv_gemm_arc == (A@B)&0xFF via rn.mul, bit-for-bit", ca == want_arc)

print("RESULT:", "ALL PASS" if not fails else f"FAIL ({len(fails)}): {fails}")
