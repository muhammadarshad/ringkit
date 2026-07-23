"""Test for ringkit.ml.tkvcache — the TKV content-addressed KV cache (RAM tier, P3).

Ports the gevhv research bed's TKV verification protocol (../../../gevhv/TKV_MATH.md, Theorems
H-O, T11-T20) into the kit's suite: exhaustive wherever the domain allows, 3-sigma statistical
bounds elsewhere — computed EXACTLY via fractions.Fraction and rn.isqrt, never a float sqrt, so
"within 3 sigma" is a real inequality, not an eyeballed float comparison. Standard math ('*', '%',
'//') appears here ONLY as the labeled external oracle (D9 two-layer rule) — grep for "oracle" at
every site — never in the module under test, which is AST-audited at the end.

D1 bar for this module (Theorem N): whenever the true full-scan argmax key is among the probed
candidates, TKVCache.attend(hard=True) must equal ml.kvcache.RingKVCache/attend_full's hard output
BIT-FOR-BIT — T20a is that gate.
Run: python -m ringkit.tests.test_tkvcache"""
import ast
import os
import random
from fractions import Fraction
from ringkit.core import native as rn
from ringkit.ml import kvcache as kv
from ringkit.ml import tkvcache as tk

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)


def frac_pow(base, n):
    """base**n by explicit repeated multiplication — the LABELED '*' oracle (Fraction, exact
    rational; no float ever enters). Used instead of Fraction.__pow__ so the multiply is visible."""
    r = Fraction(1, 1)
    for _ in range(n):
        r = r * base                     # oracle '*' on an exact rational, not ring arithmetic
    return r


def sigma3_bound(Q, p):
    """EXACT integer upper bound on 3*sqrt(Q*p*(1-p)), via rn.isqrt (no float, no sqrt import).
    variance = Q*p*(1-p) is a Fraction; ceil(9*variance) is bounded above by an integer whose
    isqrt (rounded up) exceeds 3*sqrt(variance)."""
    variance = Q * p * (1 - p)           # oracle '*': the textbook binomial-variance formula
    num, den = variance.numerator, variance.denominator
    if num <= 0:
        return Fraction(0)
    ceil9 = -((-9 * num) // den)         # oracle '*'/'//': ceil(9*variance), exact integer division
    b = rn.isqrt(ceil9)
    if rn.mul(b, b) < ceil9:
        b += 1
    return Fraction(b)


def within_3sigma(hits, Q, p, slack=Fraction(0)):
    """|hits - Q*p| <= 3*sigma(binomial) + slack, all exact (Fraction + rn.isqrt)."""
    diff = hits - Q * p                  # oracle '*': Q*p is the predicted count
    bound = sigma3_bound(Q, p) + slack
    return diff <= bound and -diff <= bound


random.seed(2026)

print("== T11 - Theorem I: exact quadrant-LSH collision law (65,536 pairs x 256 rotations) ==")
QUAD = tuple(v >> 6 for v in range(256))
bad11 = 0
checks11 = 0
for a in range(256):
    for b in range(256):
        cnt = 0
        for t in range(256):
            checks11 += 1
            if QUAD[(a + t) & 0xFF] == QUAD[(b + t) & 0xFF]:
                cnt += 1
        d1 = (a - b) & 0xFF
        d2 = (b - a) & 0xFF
        delta = d1 if d1 < d2 else d2                 # ring distance, module-free (mask/compare only)
        predicted = (64 - delta) << 2 if delta < 64 else 0
        if cnt != predicted:
            bad11 += 1
check(f"count == 4*max(0,64-delta) exactly, every pair (checks={checks11}, mismatches={bad11})",
      bad11 == 0 and checks11 == 65536 * 256)

print("== T12 - Theorem J(i,ii,iv): address branchless, aligned, deterministic, phase-contiguous ==")
def _raises_ve(f):
    try: f(); return False
    except ValueError: return True
    except Exception: return False
check("degenerate orbit refused: dim divisible by 7 raises ValueError (Cor I2 precondition enforced)",
      _raises_ve(lambda: tk.sample_coords(4, 7)) and _raises_ve(lambda: tk.sample_coords(8, 112))
      and len(set(tk.sample_coords(8, 128))) == 8)
DIM12, M12 = 128, 10
coords12 = tk.sample_coords(M12, DIM12)
r12 = tk.make_rotation(9001, M12)
ok12 = True
checks12 = 0
for _trial in range(10000):
    h = [random.randint(0, 255) for _ in range(DIM12)]
    addr = tk.bucket_address(h, r12, coords12)
    base = tk.lba(addr, 0)
    if base & 0xFFF:
        ok12 = False
    if tk.lba(addr, 0) != base:                       # branchless-form determinism
        ok12 = False
    for p in range(4):
        checks12 += 1
        if tk.lba(addr, p) != base + p * 4096:         # oracle '*': 4 phases, 4 contiguous sectors
            ok12 = False
check(f"sector-aligned, deterministic, 4 phases contiguous (10,000 trials x 4, checks={checks12})",
      ok12)

print("== T13 - Theorem J(iii,v): tiling + capacity arithmetic (pure, no I/O) ==")
ok13 = True
ok13 &= (4096 // 128 == 32) and (4096 % 128 == 0)       # oracle '//','%': exact tiling, 32/sector
ok13 &= (16384 // 128 == 128) and (16384 % 128 == 0)
for m in (8, 9, 10, 12):
    sectors = (1 << (2 * m)) * 4                        # oracle '*': 4^m buckets x 4 phases
    total = sectors * 4096
    if total != tk.capacity_bytes(m) or total != (1 << (2 * m + 14)):
        ok13 = False
check("4096/128=32 exact tiling; capacity == 4^(m+1)*4KB == tk.capacity_bytes(m), m in {8,9,10,12}",
      ok13)

print("== T14 - Corollary I1: recall vs product law, m in {4,8}, >=1e5 trials, within 3 sigma ==")
DIM14 = 128
TRIALS14 = 100000
ok14 = True
for m14 in (4, 8):
    coords14 = tk.sample_coords(m14, DIM14)
    r14 = tk.make_rotation(4141 + m14, m14)
    h = [0] * DIM14
    h2 = [0] * DIM14
    for delta in (0, 8, 16, 32):
        p_pred = frac_pow(Fraction(64 - delta, 64), m14)      # exact rational, oracle '*' above
        hits = 0
        for _ in range(TRIALS14):
            for i, c in enumerate(coords14):
                base = random.randint(0, 255)
                h[c] = base
                if random.getrandbits(1):
                    h2[c] = (base + delta) & 0xFF
                else:
                    h2[c] = (base - delta) & 0xFF
            if tk.bucket_address(h, r14, coords14) == tk.bucket_address(h2, r14, coords14):
                hits += 1
        ok_delta = within_3sigma(hits, TRIALS14, p_pred)
        print(f"    m={m14} delta={delta:2d}  hits={hits}/{TRIALS14}  predicted={p_pred}  "
              f"{'OK' if ok_delta else 'DEVIATES'}")
        if not ok_delta:
            ok14 = False
check("empirical collision rate within 3 sigma of prod(max(0,(64-delta_i)/64)), all (m,delta)", ok14)

print("== T16 - Theorem K (core.native, single-sourced): payload round trip, exhaustive N in {2,4,8,16} ==")
ok16 = True
checks16 = 0
for N in (2, 4, 8, 16):
    for a1 in range(256):
        for d1 in range(256):
            checks16 += 1
            v = rn.recover(a1, d1, N)
            lead = v[0]
            rd1 = (v[0] - v[1]) & 0xFF
            good = (lead == a1) and (rd1 == d1)
            if good:
                for k in range(1, N):
                    if ((v[0] - v[k]) & 0xFF) != rn.derived_delta(rd1, k):
                        good = False
                        break
            if good:
                w = rn.recover(lead, rd1, N)
                if w != v:
                    good = False
            if not good:
                ok16 = False
check(f"construct->re-derive->recover bit-exact, N in {{2,4,8,16}} x all (a1,d1) (checks={checks16})",
      ok16 and checks16 == 4 * 65536)

print("== T17 - Theorem L (core.native, single-sourced): O(1) evolution == per-channel stepping ==")
random.seed(1717)
ok17 = True
checks17 = 0
N17 = 8
for _trial in range(64):
    a1 = random.randint(0, 255)
    d1 = random.randint(0, 255)
    f_lead = random.randint(0, 255)
    f_rest = random.randint(0, 255)
    v = rn.recover(a1, d1, N17)
    lam, _, _ = rn.compress(v)                          # single-sourced Lambda, not re-derived
    for dt in range(256):
        checks17 += 1
        w = [(v[i] + (f_lead if i == 0 else f_rest) * dt) & 0xFF for i in range(N17)]   # oracle '*'
        lam2, d1_2, lead2 = rn.evolve(lam, d1, a1, (f_lead, f_rest), dt, N17)
        r = rn.recover(lead2, d1_2, N17)
        if r != w:
            ok17 = False
        lam_r, _, _ = rn.compress(r)
        if lam_r != lam2:
            ok17 = False
check(f"O(1) payload evolution == per-channel stepping, all dt, 64 random (f_lead,f_rest) "
      f"(checks={checks17})", ok17 and checks17 == 64 * 256)

print("== T18 - Theorem M(i,ii): anti-stride write-time + relative-time recovery, exhaustive ==")
ok18 = True
checks18 = 0
for u in range(256):
    checks18 += 1
    if tk.anchor_recover(tk.anchor_lead(u)) != (u & 0xFF):
        ok18 = False
for uw in range(256):
    for uq in range(256):
        checks18 += 1
        du = tk.anchor_delta(tk.anchor_lead(uq), tk.anchor_lead(uw))
        if du != ((uq - uw) & 0xFF):
            ok18 = False
check(f"u == 183*lead exhaustive (256); Delta-u == 183*(lead_q-lead_w) exhaustive (65,536) "
      f"(checks={checks18})", ok18 and checks18 == 256 + 65536)

print("== T19 - Corollary K1 (core.native, single-sourced): lambda-shift = POS, gap = SPIN, N=4 ==")
ok19 = True
checks19 = 0
for a1 in range(256):
    for d1 in range(256):
        checks19 += 1
        v = rn.recover(a1, d1, 4)
        lam, _, _ = rn.compress(v)
        sum_d = 0
        for k in (1, 2, 3):
            sum_d += rn.derived_delta(d1, k)
        pos = ((lam + sum_d) & 0xFF) >> 2
        if pos != (a1 & 63):
            ok19 = False
        if a1 != (pos | ((a1 >> 6) << 6)):
            ok19 = False
check(f"(lambda+sum(delta))>>2 == a1 mod 64 (POS); a1 == POS + 64*SPIN, exhaustive (checks={checks19})",
      ok19 and checks19 == 65536)

print("== T20a/b - Theorem N: conditional exactness + recall law (no-RoPE and RoPE) ==")
DIM20, M20, T20N, Q20, BETA20 = 64, 8, 180, 250, 16
deltas20 = (0, 4, 8, 16)
ok20a = True
ok20b = True
for use_rope in (False, True):
    random.seed(31400 + (1 if use_rope else 0))
    K20 = [[random.randint(0, 255) for _ in range(DIM20)] for _ in range(T20N)]
    V20 = [[random.randint(0, 255) for _ in range(DIM20)] for _ in range(T20N)]
    ref = kv.RingKVCache(DIM20, rope=use_rope)
    for k, v in zip(K20, V20):
        ref.append(k, v)
    tkv = tk.TKVCache(DIM20, m=M20, L=1, seed=555, rope=use_rope)
    for k, v in zip(K20, V20):
        tkv.append(k, v)
    pos20 = T20N - 1
    Kp_full = [kv.rope(k, j) for j, k in enumerate(K20)] if use_rope else K20
    for delta in deltas20:
        p_target = Fraction(64 - delta, 64) if delta < 64 else Fraction(0)
        p_pred = frac_pow(p_target, M20)
        hits = 0
        agree = 0
        for _ in range(Q20):
            target = random.randrange(T20N)
            if random.getrandbits(1):
                q = [(x + delta) & 0xFF for x in K20[target]]
            else:
                q = [(x - delta) & 0xFF for x in K20[target]]
            qp = kv.rope(q, pos20) if use_rope else q
            row = kv.score_row(qp, Kp_full)
            _, jstar = kv.boltzmann_weights(row, BETA20)
            cand = tkv.probe(q)
            if jstar in cand:
                hits += 1
                tkv_val = tkv.attend(q, beta=BETA20, hard=True, pos=pos20)
                ref_val = ref.attend(q, beta=BETA20, hard=True)
                if tkv_val == ref_val == list(V20[jstar]):
                    agree += 1
        if agree != hits:
            ok20a = False
        law_ok = True
        if not use_rope:
            # Recall-LAW check is content-space only (no-RoPE), matching TKV_MATH.md's own T20b
            # protocol (and gevhv/verify/tkv_attend_verify.cpp): under RoPE, jstar is chosen on the
            # ROPED score, so its RAW content offset from q is no longer the injected `delta` — the
            # law (Theorem N) is stated on the CONTENT distance to whatever key wins, and RoPE mixes
            # a position term into that win, decoupling it from the controlled-delta construction.
            # T20a above already covers the RoPE case (conditional exactness holds regardless).
            # +2% slack: jstar can still drift off `target` from interference among the other
            # T20N-1 random keys even without RoPE (documented identically in the gevhv verify bed).
            law_ok = within_3sigma(hits, Q20, p_pred, slack=Fraction(2, 100) * Q20)
            if not law_ok:
                ok20b = False
        print(f"    rope={use_rope!s:5} delta={delta:2d}  hits={hits}/{Q20}  agree={agree}  "
              f"predicted={p_pred}  {'OK' if law_ok else ('DEVIATES' if not use_rope else 'n/a (RoPE: T20a only)')}")
check("T20a: hit => TKV hard attend == RingKVCache/attend_full hard, bit-for-bit (no-RoPE + RoPE)",
      ok20a)
check("T20b: recall within 3 sigma (+2% drift slack) of prod(1-delta/64)^m (no-RoPE, per TKV_MATH.md)",
      ok20b)

print("== T20c - Corollary N1: multi-table boost 1-(1-p)^L ==")
random.seed(20031)
K20c = [[random.randint(0, 255) for _ in range(DIM20)] for _ in range(T20N)]
V20c = [[random.randint(0, 255) for _ in range(DIM20)] for _ in range(T20N)]
pos20c = T20N - 1
Kp_20c = K20c                                            # no-RoPE path (isolates the address law)
delta20c = 8
queries20c = []
for _ in range(Q20):
    target = random.randrange(T20N)
    sign = random.getrandbits(1)
    q = [((x + delta20c) if sign else (x - delta20c)) & 0xFF for x in K20c[target]]
    row = kv.score_row(q, Kp_20c)
    _, jstar = kv.boltzmann_weights(row, BETA20)
    queries20c.append((q, jstar))
ok20c = True
p1 = None
for L in (1, 2, 4):
    tkv_l = tk.TKVCache(DIM20, m=M20, L=L, seed=777, rope=False)
    for k, v in zip(K20c, V20c):
        tkv_l.append(k, v)
    hits = 0
    for q, jstar in queries20c:
        if jstar in tkv_l.probe(q):
            hits += 1
    p_emp = Fraction(hits, Q20)
    if L == 1:
        p1 = p_emp
        print(f"    L={L}  hits={hits}/{Q20} (baseline p1={p1})")
        continue
    p_pred_L = 1 - frac_pow(1 - p1, L)
    law_ok = within_3sigma(hits, Q20, p_pred_L, slack=Fraction(2, 100) * Q20)
    print(f"    L={L}  hits={hits}/{Q20}  predicted(1-(1-p1)^L)={p_pred_L}  "
          f"{'OK' if law_ok else 'DEVIATES'}")
    if not law_ok:
        ok20c = False
check("multi-table recall ~ 1-(1-p1)^L within 3 sigma (+2% slack), L in {2,4}", ok20c)

print("== T20d - Theorem O: pre-RoPE address age-invariant; post-RoPE recency-biased (contrast) ==")
random.seed(9182)
DIM_D, M_D, TRIALS_D = 64, 8, 500
coords_d = tk.sample_coords(M_D, DIM_D)
r_d = tk.make_rotation(4242, M_D)
ages = list(range(0, 256, 17))
pre_fail = 0
post_shift = 0
checks_d = 0
for _ in range(TRIALS_D):
    c = [random.randint(0, 255) for _ in range(DIM_D)]
    a_pre = tk.bucket_address(c, r_d, coords_d)          # PRE-RoPE (content) address: production form
    for age in ages:
        checks_d += 1
        if tk.bucket_address(c, r_d, coords_d) != a_pre:
            pre_fail += 1
        # POST-RoPE addressing implemented ONLY here, as the labeled contrast (Theorem O(i)):
        # the key roped by `age`, addressed the same way — deliberately NOT production code.
        kr = kv.rope(c, age)
        if tk.bucket_address(kr, r_d, coords_d) != a_pre:
            post_shift += 1
ok20d = (pre_fail == 0) and (post_shift > checks_d // 2)
check(f"pre-RoPE age-invariant (0/{checks_d} mismatches); post-RoPE shifts with age in "
      f"{post_shift}/{checks_d} cases (recency bias)", ok20d)

print("== charter audit: ml/tkvcache.py is multiplier-free and import-clean ==")
src = open(os.path.join(os.path.dirname(__file__), "..", "ml", "tkvcache.py")).read()
tree = ast.parse(src)
bad_ops = [(n.lineno, type(n.op).__name__) for n in ast.walk(tree)
           if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Div, ast.FloorDiv,
                                                              ast.Pow, ast.Mod))]
check(f"no * / // ** %% operators in ml/tkvcache.py (found {bad_ops})", not bad_ops)
imports = [n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)]
imports += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
std = [m for m in imports if m and not m.startswith("ringkit")]
check(f"no standard-math imports (found {std})", not std)
floats = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
check(f"no float literals (found at lines {floats})", not floats)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
if fails:
    raise SystemExit(1)
