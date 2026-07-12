"""Production tests for ringkit.physics (measure + qcm). Verifies QCM constants against the
mounted-source math. Run: python3 -m ringkit.tests.test_physics"""
import math
from ringkit.core import native as rn
from ringkit.linalg import solve as sv
from ringkit.physics import measure as me
from ringkit.physics import qcm

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)
def raises(exc, f):
    try: f(); return False
    except exc: return True
    except Exception: return False

print("== measure (ENERGY rulers) ==")
check("measure_sin(512) == SIN512", all(me.measure_sin(p, 512) == me.SIN512(p) for p in range(512)))
check("peaks scale with N", me.measure_sin(64, 512) < me.measure_sin(128, 1024) < me.measure_sin(192, 1536))
check("SIN512 landmarks", me.SIN512(128) == 2688 and me.SIN512(384) == -2688 and me.SIN512(256) == 0)
check("N range guard", raises(ValueError, lambda: me.measure_sin(10, 4096)))
check("N must be even", raises(ValueError, lambda: me.measure_sin(10, 513)))
check("rings", me.layout() == {"CORE":256,"AXES(XYZU)":1024,"ACC_OVR":512,"WORKING":1536})

print("== qcm state / topology ==")
check("conjugate == ring_neg", all(qcm.conjugate(d) == rn.ring_neg(d) for d in range(256)))
check("quadrants UP+/UP-/DN+/DN-", [qcm.quadrant(v) for v in (0,64,128,192)] == ["UP+","UP-","DN+","DN-"])
check("product rule state*state(conj)=-1 (non-vacuum)", all(qcm.state(d)*qcm.state(qcm.conjugate(d)) == -1 for d in range(256) if not qcm.is_vacuum(d)))
check("is_vacuum {0,64,128,192}", all(qcm.is_vacuum(v) for v in (0,64,128,192)) and not qcm.is_vacuum(30))
check("7-prime walk covers all 252 non-vacuum", set(qcm.seven_prime_walk()) == set(range(256)) - qcm.VACUUMS)
check("hypervector 128x113=14464 bytes", qcm.HV_CELLS == 14464 and qcm.HV_BYTES == 14464 and len(qcm.hypervector(range(300))) == 14464)
check("manifold: midpoint/arms", qcm.midpoint(1000) == 500 and qcm.arms(500, 7) == (493, 507))

print("== QCM constants vs mounted-source math ==")
check("anti-strides (modinv)", [sv.modinv(s) for s in (3,5,7,9)] == [171,205,183,57])
check("128 singularity (all rings meet)", [rn.mul(128,s) & 0xFF for s in (3,5,7,9)] == [128,128,128,128])
N = 3
check("N=3 tree", (N+1)**2 == 16 and 16*(N*N-1) == 128 and (2*N)**2 == 36 and 256//(N+1) == 64)
check("252 = 4x7x9, 113 = 7D+1", 4*7*9 == 252 and 7*16+1 == 113 and math.gcd(113,256) == 1)
check("stride-7 = 36 bins, 9/quadrant", (lambda o: len(set(o))==36 and all(sum(1 for x in o if lo<=x<=hi)==9 for lo,hi in [(1,63),(65,127),(129,191),(193,255)]))([(7*k)&0xFF for k in range(1,37)]))

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
