"""Production test for ringkit.ml.kvpolar — the POLAR KV element (tick, mag), pure integer.

THE BAR IS FIDELITY, NOT ROUTING. An earlier routing benchmark passed EVERYTHING — including a
provably information-destroying even stride (256 ring values -> 32 codes) which still scored 1.00
at N=1024. 1-of-N routing cannot see information loss. So the bar here is KEY RECOVERY, and the
suite asserts that the known-bad codecs FAIL it. A benchmark whose known-bad control passes is not
a benchmark.

Run: python3 -m ringkit.tests.test_kvpolar"""
import ast
import os
import random
from ringkit.core import native as rn
from ringkit.linalg.solve import modinv
from ringkit.ml import kvpolar as kp

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)


def rdist(a, b):
    d = (a - b) & 0xFF
    return d if d < 256 - d else 256 - d


print("== 1. the element is PURE INTEGER and the magnitude is EXACT ==")
# float sqrt used ONLY as a labeled test oracle (charter D9: tests may use standard math as an oracle)
random.seed(2)
bad = 0
for _ in range(20000):
    sx = random.randrange(-128, 128)
    sy = random.randrange(-128, 128)
    m = rn.isqrt(rn.qsm(sx, sx) + rn.qsm(sy, sy))
    if m != int((sx * sx + sy * sy) ** 0.5):        # ORACLE
        bad += 1
check(f"integer magnitude == float oracle over 20000 pairs ({bad} disagreements) — no head_scale, "
      f"no calibration, no float in the codec", bad == 0)
t, m = kp.encode_pair(100, 0)
check(f"tick of (+x, 0) is 0: got {t}", t == 0)
t2, _ = kp.encode_pair(0, 100)
check(f"tick of (0, +y) is the quarter turn 64: got {t2}", t2 == 64)
t3, m3 = kp.encode_pair(100, 100)
check(f"tick of the diagonal is the eighth turn 32, mag = isqrt(2)*100 = 141: got ({t3}, {m3})",
      t3 == 32 and m3 == 141)

print("== 2. THE FIDELITY BAR — and the known-bad codecs MUST fail it (chance error = 64) ==")
D = 16
def stride_recovery(s, bits, trials=400):
    """Reconstruct the key after odd/even-stride precondition + uniform quantization."""
    try:
        inv = modinv(s)
    except Exception:
        inv = None                                   # even stride: NO inverse. The key is GONE.
    random.seed(1)
    e = n = 0
    for _ in range(trials):
        k = [random.randrange(256) for _ in range(D)]
        sh = 8 - bits
        enc = [((((rn.mul(x, s) & 0xFF) >> sh) << sh) + (1 << (sh - 1))) & 0xFF for x in k]
        rec = [rn.mul(x, inv) & 0xFF for x in enc] if inv is not None else list(enc)
        e += sum(rdist(rec[i], k[i]) for i in range(D))
        n += D
    return e / n

odd7 = stride_recovery(7, 4)
even8 = stride_recovery(8, 4)
check(f"CONTROL: the EVEN stride (x8, no modular inverse) fails — mean error {even8:.1f} of a "
      f"chance 64. The routing bar scored this 1.00.", even8 > 55)
check(f"AND SO DOES OUR OWN odd stride (x7) — mean error {odd7:.1f}, at chance. A stride is a "
      f"bijection but NOT an isometry: decode multiplies the error by modinv(s) and scatters it. "
      f"REJECTED.", odd7 > 55)

print("== 3. THE ARC POSITION IS NEVER QUANTIZED — it is exact at EVERY rate ==")
# The tick IS the ring's identity: an exact integer position on Z256, already the compressed form.
# Throwing bits away from it would destroy the exactness the framework rests on (exact additive
# RoPE, exact delta_tick, exact quadrant). Compression comes from the MAGNITUDE alone — which is
# the only part carrying redundancy, and the only part hpq quantizes either.
random.seed(4)
PAIRS = 4000
data = [(random.randrange(256), random.randrange(256)) for _ in range(PAIRS)]
def polar_err(mb):
    te = me = 0
    for x, y in data:
        t, m = kp.encode_pair(x, y)
        tq, mq = kp.quantize_element(t, m, mb)
        te += rdist(tq, t)
        me += abs(mq - m)
    return te / PAIRS, me / PAIRS

for mb in (8, 6, 4, 3, 2, 1):
    te, me = polar_err(mb)
    num, den = kp.bits_per_coord(mb)
    check(f"mag {mb}b -> {num}/{den} bits/coord: ARC error {te:.2f} (EXACT), mag error {me:.2f}",
          te == 0.0)
check("lossless at mag 8b: the element round-trips exactly", polar_err(8) == (0.0, 0.0))
# HONEST: protecting the ARC costs compression. 6.0 bits/coord (2.7x vs fp16) is the real figure;
# the 3.5 b/coord an earlier draft claimed was bought by QUANTIZING THE ARC POS, which is a
# violation of the ring, not a result. Further compression must come from the 64-chunk reading of
# the tick and the "4 diffusion" processing — NOT from crushing the angle.
num, den = kp.bits_per_coord(4)
check(f"HONEST: ARC-exact costs rate. mag 4b = {num}/{den} = 6.0 bits/coord = 2.7x vs fp16. The "
      f"3.5 b/coord claimed earlier came from quantizing the ARC pos — a violation, not a win.",
      (num, den) == (12, 2))

print("== 4. decode NEVER reconstructs — so the _arch SIN/COS error never enters scoring ==")
check("delta_tick is exact modular subtraction (the shorter way round)",
      kp.delta_tick(10, 250) == 16 and kp.delta_tick(250, 10) == 16 and kp.delta_tick(0, 128) == 128)
src = open(os.path.join(os.path.dirname(__file__), "..", "ml", "kvpolar.py")).read()
check("kvpolar never calls SIN/COS/ring_cis (no approximation in the decode path)",
      "SIN" not in src.replace("SIN/COS", "") and "ring_cis" not in src)

print("== 5. charter: multiplier-free + no standard-math imports (AST audit) ==")
tree = ast.parse(src)
bad_ops = [n.lineno for n in ast.walk(tree)
           if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.FloorDiv, ast.Pow, ast.Div))]
banned = {"numpy", "math", "scipy", "torch", "np"}
bad_imp = []
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        bad_imp += [a.name for a in n.names if a.name.split(".")[0] in banned]
    elif isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] in banned:
        bad_imp.append(n.module)
floats = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
check(f"no '*' '//' '**' '/' in kvpolar.py (found {bad_ops})", not bad_ops)
check(f"no standard-math imports (found {bad_imp})", not bad_imp)
check(f"no float literals (found {floats})", not floats)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
if fails:
    raise SystemExit(1)
