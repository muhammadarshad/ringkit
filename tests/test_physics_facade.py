"""Facade test for ringkit.physics.Gauge — scientist-facing lattice gauge simulation.
Reads like ordinary simulation code; no lattice/ring internals surface.
Run: python3 -m ringkit.tests.test_physics_facade"""
import ringkit as rk

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

print("== create a field and read its state ==")
g = rk.physics.Gauge(size=(12, 12, 12), beta=60, seed=1)
a0, o0 = g.action(), g.order()
check("fresh random field is disordered (order ~0.5)", 0.45 < o0 < 0.6)
check("plaquette field has lattice size", len(g.plaquette()) == 12 * 12 * 12)

print("== thermalize COLD (beta=60): field orders ==")
g.thermalize(sweeps=40)
a1, o1 = g.action(), g.order()
print(f"    action {a0:.1f} -> {a1:.1f} | order {o0:.3f} -> {o1:.3f}")
check("cold thermalization lowers action (orders)", a1 < a0 * 0.7)
check("cold thermalization raises order", o1 > o0 + 0.2)

print("== thermalize HOT (beta=0): stays disordered ==")
gh = rk.physics.Gauge(size=(12, 12, 12), beta=0, seed=1)
ah0 = gh.action(); gh.thermalize(sweeps=40); ah1 = gh.action()
print(f"    hot action {ah0:.1f} -> {ah1:.1f}")
check("hot field stays disordered (action ~unchanged, high)", ah1 > a1 * 1.5)

print("== criticality scan locates the transition ==")
scan = rk.physics.Gauge.criticality([0, 8, 16, 32, 64], size=(10, 10, 10), sweeps=25, seed=3)
orders = [c for _, _, c in scan]
print("    (beta, action, order):", [(b, round(a, 1), round(c, 2)) for b, a, c in scan])
check("hot end disordered, cold end ordered", orders[0] < 0.6 and orders[-1] > 0.75)
check("order rises monotonically-ish with beta", all(orders[i] <= orders[i + 1] + 0.05 for i in range(len(orders) - 1)))

print("== escape hatch ==")
check("raw exposes grid/beta/shape", set(g.raw.keys()) == {"grid", "beta", "shape"})
check("repr is informative", "Gauge(" in repr(g))

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
