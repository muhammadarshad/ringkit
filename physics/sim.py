"""
ringkit.physics.sim — engineer/scientist-facing wrapper over the SU(256) gauge engine.

Run lattice-gauge simulations without touching the ring lattice math. Create a field, thermalize
it at a coupling, read its action/order, or scan for the phase transition. The Metropolis sweeps,
integer Boltzmann tables, checkerboard parity, and cache-blocked C kernels all run underneath.

    import ringkit as rk
    g = rk.physics.Gauge(size=(12, 12, 12), beta=60)   # cold coupling
    g.thermalize(sweeps=40)
    print(g.action(), g.order())                        # low action / high order -> ordered phase
    scan = rk.physics.Gauge.criticality([0, 8, 16, 32, 64])   # locate the transition
"""
import random as _random
from ringkit.core import native as _rn
from ringkit.physics import gauge as _gauge


class Gauge:
    """A ring-native SU(256) lattice gauge field. `beta` is the (integer) coupling: larger = colder
    (orders the field), 0 = hot (stays disordered). Ring internals live in `.raw`."""

    def __init__(self, size=(12, 12, 12), beta=40, seed=0):
        self.W, self.H, self.D = (int(s) for s in size)
        self.beta = int(beta)
        self._rng = _random.Random(seed)
        n = _rn.mul(_rn.mul(self.W, self.H), self.D)
        self.grid = bytearray(self._rng.randbytes(n))

    def action(self):
        """Mean local action (order parameter): low = ordered/aligned, high = disordered."""
        return _gauge.mean_action(self.grid, self.W, self.H, self.D)

    def order(self):
        """Neighbor-alignment order parameter in [0,1]: 1 = ordered, ~0.5 = disordered."""
        return _gauge.correlation(self.grid, 1, self.W, self.H, self.D)

    def thermalize(self, sweeps=40):
        """Run `sweeps` Metropolis sweeps at the current beta. Mutates the field; returns self.
        Sweeps run in batches so big lattices stay GPU-resident across the whole batch
        (unified memory): the grid crosses the bus once per batch, not once per sweep."""
        lut = _gauge.boltzmann_lut(self.beta)
        n = len(self.grid)
        remaining = int(sweeps)
        while remaining > 0:
            batch = 8 if remaining > 8 else remaining
            props = bytearray(self._rng.randbytes(_rn.mul(n, batch)))
            chances = bytearray(self._rng.randbytes(_rn.mul(n, batch)))
            _gauge.thermalize(self.grid, props, chances, lut, self.W, self.H, self.D, batch)
            remaining -= batch
        return self

    def plaquette(self):
        """The Wilson plaquette energy field over the lattice (a bytearray)."""
        return _gauge.plaquette(self.grid, self.W, self.H, self.D)

    @staticmethod
    def criticality(betas, size=(10, 10, 10), sweeps=30, seed=0):
        """Sweep coupling `betas`; for each, thermalize a fresh field and report
        (beta, mean_action, order). Locates the ordered<->disordered transition."""
        W, H, D = (int(s) for s in size)
        return _gauge.criticality_scan(betas, W, H, D, therm=sweeps, seed=seed)

    @property
    def raw(self):
        return {"grid": self.grid, "beta": self.beta, "shape": (self.W, self.H, self.D)}

    def __repr__(self):
        return f"Gauge(shape={(self.W, self.H, self.D)}, beta={self.beta}, action={self.action():.1f})"
