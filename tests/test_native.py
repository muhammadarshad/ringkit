"""Exhaustive production tests for ringkit.core.native (the foundation).
math is an external oracle only (labeled), never imported by the library.
Run: python3 -m ringkit.tests.test_native"""
import math
import random
from ringkit.core import native as rn

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

def raises(exc, f):
    try: f(); return False
    except exc: return True
    except Exception: return False

print("== primitives vs oracle ==")
check("mul == * (signed, 0..40)", all(rn.mul(a, b) == a * b for a in range(-40, 41, 3) for b in range(-40, 41, 3)))
check("ipow == ** (0..12 ^ 0..5)", all(rn.ipow(b, n) == b ** n for b in range(13) for n in range(6)))
check("mf_floordiv == // (0..600 / 1..40)", all(rn.mf_floordiv(n, d) == n // d for n in range(0, 600, 7) for d in range(1, 41)))
check("mf_mod == % (0..600 mod 1..40)", all(rn.mf_mod(n, d) == n % d for n in range(0, 600, 7) for d in range(1, 41)))
check("qsm == x*y (ALL ring pairs)", all(rn.qsm(x, y) == x * y for x in range(256) for y in range(256)))
check("isqrt_lut == math.isqrt (0..262144)", all(rn.isqrt_lut(m) == math.isqrt(m) for m in range(0, 262145, 97)))
check("isqrt general == math.isqrt (0..10^7)", all(rn.isqrt(m) == math.isqrt(m) for m in range(0, 10**7, 9973)))
check("scale21 == 21*r", all(rn.scale21(r) == 21 * r for r in range(0, 300)))
check("ring_neg", all(rn.ring_neg(x) == (-x) % 256 for x in range(256)))
check("_SQ[n]==n^2", all(rn._SQ[n] == n * n for n in range(513)))

print("== trig identities over 256 ==")
sg = rn._signed
check("SIN zeros/peak/trough", rn.SIN(0)==0 and rn.SIN(128)==0 and rn.SIN(64)==21 and rn.SIN(192)==235)
check("COS = SIN(phi+64) over 256", all(rn.COS(p) == rn.SIN((p+64)%256) for p in range(256)))
check("SIN(-p) == -SIN(p) (ring)", all(rn.SIN((-p)%256) == rn.ring_neg(rn.SIN(p)) for p in range(256)))
check("SIN(p+128) == -SIN(p)", all(rn.SIN((p+128)%256) == rn.ring_neg(rn.SIN(p)) for p in range(256)))
check("SIN(p+256) == SIN(p)", all(rn.SIN((p+256)%256) == rn.SIN(p) for p in range(256)))
check("SIN bounded |.|<=21", all(abs(sg(rn.SIN(p))) <= 21 for p in range(256)))
check("Pythagorean at cardinals", all((lambda s,c: min(s,256-s)**2 + min(c,256-c)**2 == 441)(rn.SIN(v),rn.COS(v)) for v in (0,64,128,192)))
check("TAN VACUUM at {64,192}", rn.TAN(64)=="VACUUM" and rn.TAN(192)=="VACUUM" and rn.TAN(32)==21)
check("SEC VACUUM at {64,192}", rn.SEC(64)=="VACUUM" and rn.SEC(0)==21)
check("CSC/COT VACUUM at {0,128}", rn.CSC(0)=="VACUUM" and rn.COT(128)=="VACUUM" and rn.CSC(64)==21)
check("KS4 zeros at all 4 vacuums", all(rn.KS4(v)==0 for v in (0,64,128,192)))

print("== inverse trig (principal round-trip) ==")
check("SIN(ARCSIN(SIN)) principal", all(rn.SIN(rn.ARCSIN(sg(rn.SIN(p))))==rn.SIN(p) for p in range(0,65)))
check("COS(ARCCOS(COS)) principal", all(rn.COS(rn.ARCCOS(sg(rn.COS(p))))==rn.COS(p) for p in range(0,129)))
check("TAN(ARCTAN(TAN)) principal", all(rn.TAN(rn.ARCTAN(rn.TAN(p)))==rn.TAN(p) for p in range(0,64)))
check("ARCSIN(21)=64, ARCCOS(-21)=128", rn.ARCSIN(21)==64 and rn.ARCCOS(-21)==128)

print("== rotor / iota ==")
check("qh_iota quadrants", [rn.qh_iota(v) for v in (0,64,128,192)]==[0,1,2,3])
check("iota^4 = identity", all(rn.iota_mul(rn.iota_mul(rn.iota_mul(rn.iota_mul(p))))==p for p in range(256)))
check("iota^2 = -1 (shift128 == negate vec)", all((rn.COS((p+128)%256),rn.SIN((p+128)%256))==(rn.ring_neg(rn.COS(p)),rn.ring_neg(rn.SIN(p))) for p in range(256)))
check("COS == iota*SIN", all(rn.COS(p)==rn.SIN(rn.iota_mul(p)) for p in range(256)))
check("polar_axis cardinals", [rn.polar_axis(v) for v in (0,64,128,192)]==[(0,1,1),(1,-1,1),(2,-1,-1),(3,1,-1)])

print("== ADI ==")
check("recover(36,11,4)", rn.recover(36,11,4)==[36,25,22,17])
check("compress inverse", rn.compress([36,25,22,17])==(100,11,36))
check("derived_delta == old k*k form", all(rn.derived_delta(d,k)==(d+k*k-1)%256 for d in range(256) for k in range(20)))
check("mprc_axis_arcs(36,25)", rn.mprc_axis_arcs(36,25)==(36,25,22,17))
check("compress raises on non-ADI", raises(ValueError, lambda: rn.compress([1,2,4,99])))

print("== codec (in-range round-trip + vacuum nodes) ==")
ok=True
for lv in range(3):
    span = rn.mul(128, rn.scale(lv))
    for t in random.sample(range(span), 2000):
        if rn.decode(*rn.encode(t,lv),lv)!=t: ok=False; break
check("encode/decode round-trip in-range (lv 0..2)", ok)
check("vacuum nodes r=0 (06/12/18h)", all(rn.encode(t,0)[1]==0 for t in (21600,43200,64800)))

print("== ring-native e (exponential base = generator 3) ==")
check("RING_E == 3", rn.RING_E == 3)
check("exp o log == identity on all 128 units", all(rn.ring_exp(*rn.ring_log(u)) == u for u in range(1, 256, 2)))
check("ring_exp bijection on 64-orbit", len({rn.ring_exp(k) for k in range(64)}) == 64)
check("ord(3)=64 (phase-lock, no early reset)", rn.ring_exp(64) == 1 and all(rn.ring_exp(k) != 1 for k in range(1, 64)))
check("Delta(3^n)=2*3^n (eigenfunction of difference op)",
      all((rn.ring_exp(n + 1) - rn.ring_exp(n)) & 0xFF == rn.mul(2, rn.ring_exp(n)) & 0xFF for n in range(64)))
check("orbit never collapses (all units/odd)", all(rn.ring_exp(k) % 2 == 1 for k in range(64)))
check("doc's 173 == -3^45 (E_hat was a power of 3)", rn.ring_exp(45, 1) == 173)
check("ring_log raises on even (zero-divisor)", raises(ValueError, lambda: rn.ring_log(92)))
# ring_pow: general growth, matches ring_exp on base 3, collapses on even base
check("ring_pow(3,k) == ring_exp(k)", all(rn.ring_pow(3, k) == rn.ring_exp(k) for k in range(64)))
check("ring_pow matches masked ipow (random bases/exps)",
      all(rn.ring_pow(b, e) == rn.ipow(b, e) & 0xFF for b in (5, 21, 100, 173, 255) for e in (0, 1, 2, 7, 13)))
check("ring_pow of EVEN base collapses to 0 by exp 8", all(rn.ring_pow(a, 8) == 0 for a in range(2, 256, 2)))
check("ring_pow exp<0 -> ValueError", raises(ValueError, lambda: rn.ring_pow(3, -1)))
# ring_cis: our e^{i*phi}, Euler's formula + identity
check("e^{i*0} = (SCALE, 0)", rn.ring_cis(0) == (rn.SCALE, 0))
check("e^{i*pi/2} = (0, SCALE) = i", rn.ring_cis(rn.Q) == (0, rn.SCALE))
check("EULER IDENTITY e^{i*pi} = (-SCALE, 0)", rn.ring_cis(rn.HALF) == (rn.ring_neg(rn.SCALE), 0))
check("e^{i*pi} + SCALE == 0", (rn.ring_cis(rn.HALF)[0] + rn.SCALE) & 0xFF == 0)
# no scalar sqrt(-1); i is the rotor J with J^2=-I, J^4=I
check("no scalar i in Z256 (no x^2==255)", not any((rn.mul(x, x) & 0xFF) == 255 for x in range(256)))
_J = rn.IOTA
_J2 = ((rn.mul(_J[0][0], _J[0][0]) + rn.mul(_J[0][1], _J[1][0])) & 0xFF,
       (rn.mul(_J[0][0], _J[0][1]) + rn.mul(_J[0][1], _J[1][1])) & 0xFF,
       (rn.mul(_J[1][0], _J[0][0]) + rn.mul(_J[1][1], _J[1][0])) & 0xFF,
       (rn.mul(_J[1][0], _J[0][1]) + rn.mul(_J[1][1], _J[1][1])) & 0xFF)
check("IOTA^2 == -I  (i^2 = -1 as operator)", _J2 == (255, 0, 0, 255))
# rotor composition — EXACT for iota (quarter-turn) steps, all angles
check("cis_rotate(cis(phi),k) == cis(rotate(phi,k)) for ALL phi, k",
      all(rn.cis_rotate(*rn.ring_cis(p), k) == rn.ring_cis(rn.rotate(p, k)) for p in range(256) for k in range(4)))
check("rotate 4 quarters == identity (i^4=1)", all(rn.rotate(p, 4) == p for p in range(256)))
check("rotate 2 quarters == half-turn negation of the pair",
      all(rn.cis_rotate(*rn.ring_cis(p), 2) == (rn.ring_neg(rn.COS(p)), rn.ring_neg(rn.SIN(p))) for p in range(256)))
check("rotate accepts negative quarters (exact inverse)", all(rn.rotate(rn.rotate(p, 1), -1) == p for p in range(256)))

print("== error paths (production validation) ==")
check("qsm out-of-range -> ValueError", raises(ValueError, lambda: rn.qsm(300,300)))
check("isqrt neg -> ValueError", raises(ValueError, lambda: rn.isqrt_lut(-1)))
check("isqrt too big -> ValueError", raises(ValueError, lambda: rn.isqrt_lut(10**9)))
check("mf_floordiv zero -> ZeroDivisionError", raises(ZeroDivisionError, lambda: rn.mf_floordiv(5,0)))
check("mf_floordiv neg -> ValueError", raises(ValueError, lambda: rn.mf_floordiv(-5,2)))
check("ipow neg exp -> ValueError", raises(ValueError, lambda: rn.ipow(3,-1)))
check("scale neg level -> ValueError", raises(ValueError, lambda: rn.scale(-1)))
check("encode neg t -> ValueError", raises(ValueError, lambda: rn.encode(-5)))

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
