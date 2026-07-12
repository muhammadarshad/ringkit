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
        # Persistent GPU session (unified memory): the lattice lives on the device across
        # calls; self.grid syncs lazily, only when an observable actually reads it.
        self._sess = _gauge.session_for(self.grid, self.W, self.H, self.D)
        self._stale = False                       # True -> device copy is newer than self.grid

    def _sync(self):
        if self._stale and self._sess is not None:
            if self._sess.read_into(self.grid) != 0:
                self._sess = None                 # session died: grid still holds last sync
            self._stale = False
        return self.grid

    def action(self):
        """Mean local action (order parameter): low = ordered/aligned, high = disordered."""
        return _gauge.mean_action(self._sync(), self.W, self.H, self.D)

    def order(self):
        """Neighbor-alignment order parameter in [0,1]: 1 = ordered, ~0.5 = disordered."""
        return _gauge.correlation(self._sync(), 1, self.W, self.H, self.D)

    def thermalize(self, sweeps=40):
        """Run `sweeps` Metropolis sweeps at the current beta. Mutates the field; returns self.
        Randoms are DERIVED on the compute device (counter RNG, rk_mix32 spec): on the unified-
        memory GPU only a 256-byte LUT crosses the bus — the lattice stays device-resident in
        the persistent session, syncing back only when an observable reads it. The per-call
        seed comes from this Gauge's seeded stream, so runs stay reproducible."""
        lut = _gauge.boltzmann_lut(self.beta)
        seed = self._rng.getrandbits(32)
        if self._sess is not None:
            if self._sess.thermalize_rng(seed, 0, lut, int(sweeps)) == 0:
                self._stale = True
                return self
            self._sess = None                     # fall through to the routed host path
        _gauge.thermalize_rng(self.grid, seed, lut, self.W, self.H, self.D, int(sweeps))
        return self

    def profile(self, rmax=10):
        """C(R) for R=1..rmax — the mass-gap observable (alignment vs distance)."""
        return _gauge.correlation_profile(self._sync(), self.W, self.H, self.D, rmax)

    def phase(self):
        """'confined' (mass gap: alignment dead by R=5) or 'deconfined' (long-range order)."""
        return _gauge.phase_of(self.profile())

    def plaquette(self):
        """The Wilson plaquette energy field over the lattice (a bytearray)."""
        return _gauge.plaquette(self._sync(), self.W, self.H, self.D)

    @staticmethod
    def criticality(betas, size=(10, 10, 10), sweeps=30, seed=0):
        """Sweep coupling `betas`; for each, thermalize a fresh field and report
        (beta, mean_action, order). Locates the ordered<->disordered transition."""
        W, H, D = (int(s) for s in size)
        return _gauge.criticality_scan(betas, W, H, D, therm=sweeps, seed=seed)

    @property
    def raw(self):
        return {"grid": self._sync(), "beta": self.beta, "shape": (self.W, self.H, self.D)}

    def __repr__(self):
        return f"Gauge(shape={(self.W, self.H, self.D)}, beta={self.beta}, action={self.action():.1f})"
