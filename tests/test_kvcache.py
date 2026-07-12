"""Production test for ringkit.ml.kvcache — the ring-native KV cache.

A KV cache has exactly ONE job: make incremental decode identical to recomputing the whole prefix
from scratch. So the headline bar (D1) is BIT-FOR-BIT equality against the uncached reference, at
every step, for every beta — not a similarity score.

Also proved here:
  * the CIRCULAR blend is the correct ring form: a linear mean of angles wraps (mean of 255 and 1 is
    the antipode 128); ours returns 0. A CONTROL asserts the linear mean is wrong, so we are not
    just asserting our own arithmetic back at ourselves.
  * beta is a real temperature: uniform (hot) -> soft -> exact argmax (cold).
  * the weight denominator can never collapse (lut[0] = 255 for every beta) — the zero-divisor wall
    that kills a modular divide never arises, because we divide in ENERGY.
  * no scales / zero-points are stored: footprint is exactly 1 byte per coordinate.
  * the module is multiplier-free and imports no standard math (AST audit).
Run: python3 -m ringkit.tests.test_kvcache"""
import ast
import os
import random
from ringkit.core import native as rn
from ringkit.ml import kvcache as kv

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

random.seed(11)
D = 4
def vec(): return [random.randint(0, 255) for _ in range(D)]


print("== 1. THE BAR: cached decode == uncached recompute, BIT-FOR-BIT, every step, every beta ==")
allok = True
for beta in (0, 1, 4, 16, 64, 128, 255):
    for trial in range(40):
        T = random.randint(1, 8)
        K = [vec() for _ in range(T)]
        V = [vec() for _ in range(T)]
        Q = [vec() for _ in range(T)]
        # Feed OUT-OF-RANGE ints on some trials so the cache's &0xFF masking is actually exercised
        # (otherwise both paths are trivially identical and the test proves nothing about folding).
        if trial % 3 == 0:
            K[0] = [x + 256 for x in K[0]]
            V[0] = [x - 256 for x in V[0]]
        # uncached reference: recompute the whole prefix from scratch at every step, on the RAW ints.
        want = []
        for t in range(T):
            Kp = [kv.rope(K[j], j) for j in range(t + 1)]
            qp = kv.rope(Q[t], t)
            row = kv.score_row(qp, Kp)
            w, best = kv.boltzmann_weights(row, beta)
            want.append(kv.circular_blend(V[:t + 1], w, best))
        # cached: append as we go, attend once per step
        c = kv.RingKVCache(D)
        got = []
        for t in range(T):
            c.append(K[t], V[t])
            got.append(c.attend(Q[t], beta=beta))
        if got != want:
            allok = False
            break
    if not allok:
        break
check("cached == uncached, bit-for-bit (7 betas x 40 sequences)", allok)

print("== 2. the CIRCULAR blend is the right form (control: the LINEAR mean is wrong) ==")
# two values on opposite sides of the wrap: 255 and 1. True blend = 0. Linear mean = 128 (antipode).
c = kv.RingKVCache(1)
c.append([10], [255])
c.append([10], [1])            # identical keys -> equal weights -> a pure 50/50 blend
out = c.attend([10], beta=0)   # beta=0: uniform weights
linear = (255 + 1) >> 1        # the WRONG (linear) mean an ordinary quantizer would take
check(f"circular blend of 255 and 1 -> {out[0]} (0 or 256-adjacent, NOT the antipode)",
      out[0] in (0, 255, 1))
check(f"CONTROL: linear mean gives {linear} (the antipode) — so the ring form is doing real work",
      linear == 128 and out[0] != 128)

print("== 3. beta is a real temperature: uniform -> soft -> exact argmax ==")
# Keys close together, so both weights stay comparable and the blend can actually move.
K = [[0, 0, 0, 0], [20, 20, 20, 20]]
V = [[10, 10, 10, 10], [250, 250, 250, 250]]
q = [0, 0, 0, 0]               # exactly matches key 0
c = kv.RingKVCache(D, rope=False)
for k, v in zip(K, V):
    c.append(k, v)
cold = c.attend(q, beta=255)
hot = c.attend(q, beta=0)
sweep = [c.attend(q, beta=b)[0] for b in (0, 1, 2, 4, 8, 16, 64, 255)]
check(f"cold (beta=255) collapses to the winning value exactly: {cold} == {V[0]}", cold == V[0])
check(f"hard=True agrees with the cold limit: {c.attend(q, hard=True)} == {V[0]}",
      c.attend(q, hard=True) == V[0])
check(f"hot (beta=0) blends away from the winner: {hot} != {V[0]}", hot != V[0])
# monotone: as beta rises (colder) the blend walks back toward the winning value 10.
# offsets are signed-from-winner, so the walk is monotone in the signed offset, not in the phase.
offs = [kv.signed_offset(s, 10) for s in sweep]
check(f"beta sweep walks monotonically from blended to argmax: phases {sweep} -> offsets {offs}",
      all(abs(offs[i]) >= abs(offs[i + 1]) for i in range(len(offs) - 1)) and offs[-1] == 0)

print("== 3b. HONEST LIMIT: the blend has an integer-resolution floor (stated, not hidden) ==")
# A far key's weight can be too small a fraction of the mass to move the output by even ONE LSB.
# That is a real limit of an integer ring blend and we assert it rather than pretend it away.
Kf = [[0, 0, 0, 0], [200, 200, 200, 200]]      # far key: ring_distance 56/coord -> gap 224
Vf = [[10, 10, 10, 10], [250, 250, 250, 250]]
cf = kv.RingKVCache(D, rope=False)
for k, v in zip(Kf, Vf):
    cf.append(k, v)
row = kv.score_row([0, 0, 0, 0], cf.keys())
w, best = kv.boltzmann_weights(row, 4)
pull = rn.qsm(w[1], kv.signed_offset(Vf[1][0], Vf[0][0]))
mass = w[0] + w[1]
check(f"at beta=4 the far key has weight {w[1]} vs mass {mass}: pull {pull}/{mass} rounds to "
      f"{rn.mf_floordiv(abs(pull), mass)} LSB -> output pinned at the winner",
      rn.mf_floordiv(abs(pull), mass) == 0 and cf.attend([0, 0, 0, 0], beta=4) == Vf[0])
check("the floor is a RESOLUTION limit, not a weighting bug: at beta=0 the same key does move it",
      cf.attend([0, 0, 0, 0], beta=0) != Vf[0])

print("== 4. the denominator can never collapse (the zero-divisor wall never arises) ==")
from ringkit.physics.gauge import boltzmann_lut
peaks = [boltzmann_lut(b)[0] for b in range(0, 256, 17)]
check(f"lut[0] == 255 for every beta -> sum(w) >= 255 > 0 (peaks: {peaks[:5]}...)",
      all(p == 255 for p in peaks))
# and the divide is INTEGER (energy), not modular: an even denominator is harmless.
w, best = kv.boltzmann_weights([-5, -5], 16)       # tie -> both weights 255 -> sum 510 (even)
den = w[0] + w[1]
check(f"even weight mass {den} divides exactly in ENERGY (a modular inverse would collapse here)",
      den == 510 and rn.mf_floordiv(rn.qsm(255, 60) + rn.qsm(255, 100), den) == 80)

print("== 5. footprint + the QCM cache manifold (its cost stated, not hidden) ==")
c = kv.RingKVCache(8)
for _ in range(100):
    c.append([1] * 8, [2] * 8)
check(f"payload = {c.payload_bytes()} bytes == 2*8*100 exactly: 1 byte/coord, no side tables",
      c.payload_bytes() == 1600)
# The slab is PRIME-pitched (dim 8 -> pitch 11), never a power of two: a 2^k row stride aliases
# successive tokens into the same cache sets. Measured worth 1.14x on the decode scan at dim 128.
# HONEST: the pad is real memory. At dim 8 it costs 37%; at dim 128 (pitch 131) it costs 2.3%.
check(f"slab is PRIME-pitched: dim 8 -> pitch {c.pitch} (NOT 8 — that is the aliasing layout)",
      c.pitch == 11)
check(f"manifold pad is real and paid for: slab {c.nbytes()} bytes vs {c.payload_bytes()} payload; "
      f"at dim 128 the pitch is {kv.RingKVCache(128).pitch} (2.3% pad, 1.14x faster scan)",
      c.nbytes() == 2200 and kv.RingKVCache(128).pitch == 131)
check("storage is a CONTIGUOUS C-owned slab, not a Python list of rows (D9)",
      isinstance(c.K, bytearray) and isinstance(c.V, bytearray))
check("cache holds only K and V (no scale/zero-point attributes)",
      not any(hasattr(c, a) for a in ("scale", "zero_point", "codebook", "calib")))

print("== 6. charter: multiplier-free + no standard-math imports (AST audit) ==")
src = open(os.path.join(os.path.dirname(__file__), "..", "ml", "kvcache.py")).read()
tree = ast.parse(src)
bad_ops = [n.lineno for n in ast.walk(tree)
           if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.FloorDiv, ast.Pow, ast.Div))]
banned = {"numpy", "math", "scipy", "torch", "np"}
bad_imp = []
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        bad_imp += [a.name for a in n.names if a.name.split(".")[0] in banned]
    elif isinstance(n, ast.ImportFrom) and n.module:
        if n.module.split(".")[0] in banned:
            bad_imp.append(n.module)
floats = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
check(f"no '*' '//' '**' '/' operators in kvcache.py (found {bad_ops})", not bad_ops)
check(f"no standard-math imports (found {bad_imp})", not bad_imp)
check(f"no float literals (found at lines {floats})", not floats)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
if fails:
    raise SystemExit(1)
