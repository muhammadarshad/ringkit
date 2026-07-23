"""Tests for ringkit.kernels.cuda (the GEVHV CUDA backend) — D9 silicon.

Bit-for-bit vs ringkit.ml.gevhv (the semantic judge) over random + adversarial manifolds,
batch independence (N=3, distinct content per manifold), and fused == composed(ml.gevhv).
Skip-as-pass with a printed reason when no CUDA device/toolchain exists (CI, no GPU); this
box carries an RTX 4060, so the real path must run and pass.
Run: python -m ringkit.tests.test_gevhv_cuda"""
import random

from ringkit.kernels.cuda import host
from ringkit.ml import gevhv as gv

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)


if not host.available():
    print("  SKIP  no CUDA device/toolchain here — backend correctly reports unavailable")
    check("available() is False and every op returns None (clean fallback contract)",
          host.react([0] * (gv.H * gv.W), [0] * 256, 1) is None
          and host.measure([0] * (gv.H * gv.W), [0] * (gv.H * gv.W), 1) is None
          and host.gevhv_scalar([0] * (gv.H * gv.W), [0] * 256, [0] * (gv.H * gv.W), 3, 0, 1) is None)
    print()
    print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
    raise SystemExit(0 if not fails else 1)

print("== availability ==")
check("GEVHV CUDA backend available (built, loaded, self-tested)", host.available())

random.seed(20260717)
H, W = gv.H, gv.W
SITES = H * W


def rand_grid(rng=random):
    return [rng.randrange(256) for _ in range(SITES)]


def flat(grids):
    out = []
    for g in grids:
        out.extend(g)
    return out


print("== per-op bit-for-bit: react (identity boundary) vs ml.gevhv.react ==")
lut = [random.randrange(256) for _ in range(256)]
grids = [rand_grid(), rand_grid(), [0] * SITES, [255] * SITES, [128] * SITES]
N = len(grids)
got = host.react(flat(grids), lut, N)
want = flat(gv.react(g, lut) for g in grids)
check(f"react: {N} manifolds (2 random + 3 adversarial) bit-exact", got == want)

print("== per-op bit-for-bit: react_bound_scalar vs ml.gevhv.react_bound_scalar ==")
ok = True
for s in (1, 255, 3, 183, 251):          # 1, 255 are edge units (self-inverse)
    for t in (0, 1, 42, 255):
        got = host.react_bound_scalar(flat(grids), lut, s, t, N)
        want = flat(gv.react_bound_scalar(g, lut, s, t) for g in grids)
        if got != want:
            ok = False
check("react_bound_scalar: 5 s x 4 t, batch of 5, all bit-exact", ok)
try:
    gv.absorb_lut(lut, 2, 0)
    check("even s raises ValueError in ml.gevhv.absorb_lut (sanity)", False)
except ValueError:
    check("even s raises ValueError in ml.gevhv.absorb_lut (sanity)", True)

print("== per-op bit-for-bit: react_bound_vector (shared offset field) vs ml.gevhv ==")
v = rand_grid()
c = gv.offset_field(v)
got = host.react_bound_vector(flat(grids), lut, v, N, c=c)
want = flat(gv.react_bound_vector(g, lut, v, c=c) for g in grids)
check("react_bound_vector: batch of 5 sharing one precomputed c, bit-exact", got == want)
got2 = host.react_bound_vector(flat(grids), lut, v, N)   # c derived internally (host.py)
check("react_bound_vector: c derived internally == precomputed-c path", got2 == want)

print("== per-op bit-for-bit: measure (full grid and interior-only) vs ml.gevhv.measure ==")
q = rand_grid()
for interior in (False, True):
    got = host.measure(flat(grids), q, N, interior=interior)
    want = [gv.measure(g, q, interior=interior) for g in grids]
    check(f"measure(interior={interior}): {N} manifolds bit-exact", got == want)

print("== adversarial measure bounds (Theorem G) ==")
g0 = [0] * SITES
q128 = [128] * SITES
e_int = host.measure(g0, q128, 1, interior=True)[0]
e_full = host.measure(g0, q128, 1, interior=False)[0]
INTERIOR = (H - 2) * (W - 2)
check(f"interior energy == {INTERIOR}*128 (got {e_int})", e_int == INTERIOR * 128 == 1790208)
check(f"full-grid energy == {SITES}*128 (got {e_full})", e_full == SITES * 128 == 1851392)

print("== batch independence: N=3 distinct manifolds, GPU batch == per-manifold singles ==")
batch = [rand_grid(), rand_grid(), rand_grid()]
singles_measure = [host.measure(m, q, 1)[0] for m in batch]
batch_measure = host.measure(flat(batch), q, 3)
check("measure: batched result == per-manifold singles (no cross term)",
      batch_measure == singles_measure)
s, t = 183, 42
singles_scalar = [host.gevhv_scalar(m, lut, q, s, t, 1)[0] for m in batch]
batch_scalar = host.gevhv_scalar(flat(batch), lut, q, s, t, 3)
check("gevhv_scalar fused: batched == per-manifold singles", batch_scalar == singles_scalar)

print("== fused == composed(ml.gevhv): gevhv_scalar ==")
ok = True
for s in (1, 255, 3, 183):
    for t in (0, 42, 255):
        for interior in (False, True):
            got = host.gevhv_scalar(flat(batch), lut, q, s, t, 3, interior=interior)
            want = [gv.gevhv_scalar(m, lut, q, s, t, interior=interior) for m in batch]
            if got != want:
                ok = False
check("gevhv_scalar: 4 s x 3 t x 2 interior-modes, batch of 3, all bit-exact vs ml.gevhv", ok)

print("== fused == composed(ml.gevhv): gevhv_vector ==")
ok = True
for interior in (False, True):
    got = host.gevhv_vector(flat(batch), lut, q, v, 3, c=c, interior=interior)
    want = [gv.gevhv_vector(m, lut, q, v, c=c, interior=interior) for m in batch]
    if got != want:
        ok = False
check("gevhv_vector: 2 interior-modes, batch of 3, bit-exact vs ml.gevhv", ok)

print("== fused kernel == unfused (react then measure) composition, on-device ==")
ok = True
for interior in (False, True):
    reacted = host.react_bound_scalar(flat(batch), lut, s, t, 3)
    unfused = host.measure(reacted, q, 3, interior=interior)
    fused = host.gevhv_scalar(flat(batch), lut, q, s, t, 3, interior=interior)
    if unfused != fused:
        ok = False
check("device unfused (react_bound_scalar + measure) == device fused, both interior modes", ok)

print("== D9 gate: GPU fused result also matches the FULL composed reference chain ==")
ok = True
for m in batch:
    ref = gv.measure(gv.react(gv.bind_scalar(m, s, t), lut), q)   # literal bind -> react -> measure
    fused_one = host.gevhv_scalar(m, lut, q, s, t, 1)[0]
    if ref != fused_one:
        ok = False
check("GPU gevhv_scalar == literal bind->react->measure composition (no shortcuts)", ok)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
raise SystemExit(0 if not fails else 1)
