"""
ringkit.kernels.mprc.lattice.host — SILICON host for the SU(256) lattice gauge engine (D9).

Holds everything about the gauge engine that is legally hardware-flavored: the ctypes
build/load of gauge.c, the bit-for-bit pure-Python fallback (the reference of record lives
beside the kernel it validates, exactly like kernels.backend), and the measurement
observables (correlation / mean_action) whose normalized outputs are floats — labeled
MEASUREMENT IO, not ring quantities. Charter D9: this layer uses hardware ops on purpose
and is excluded from the semantic AST audit.

The physics-facing semantic surface is ringkit.physics.gauge, which re-exports these forms.
"""
import ctypes
import os
import subprocess
from ringkit.kernels.backend import _arch_flags, _BUILD, so_path

# CPU threads for the *_mt kernels: bins are predictable (checkerboard slabs own disjoint
# sites; boundary reads are opposite-parity) so threads run lock-free with no merge step.
# Thread only above the floor — spawn cost dominates tiny lattices.
NTHREADS = min(os.cpu_count() or 1, 16)
MT_MIN_NODES = 1 << 14

_DIR = os.path.dirname(__file__)
_SO = so_path("gauge")                    # arch-keyed: x86_64/arm64 interpreters coexist
_C = os.path.join(_DIR, "gauge.c")
_U8 = ctypes.POINTER(ctypes.c_uint8)
_lib = None
_tried = False

# Metropolis sweeps route to the Metal GPU at/above this node count — measured crossover on
# M1 Pro (2026-07-12): 24^3 C wins, 32^3 metal wins (2.9x at 48^3 rising to 8.4x at 160^3).
# The sweep is compute-dense (6-neighbor ring distance + LUT per node), which is Metal's win
# condition; the bandwidth-trivial plaquette stays on C (metal measured ~10x SLOWER there).
GAUGE_METAL_MIN_NODES = 1 << 15
_metal_ok = None                          # D9 gate: metal serves only after a bit-for-bit self-test


def build():
    os.makedirs(_BUILD, exist_ok=True)
    tmp = _SO + ".tmp"
    subprocess.run(["cc", "-O3", "-funroll-loops", "-shared", "-fPIC",
                    *_arch_flags(), "-o", tmp, _C], check=True)
    os.replace(tmp, _SO)                      # fresh inode -> a re-CDLL loads the new image


def _bind(lib):
    for name in ("plaquette", "plaquette_blocked"):
        fn = getattr(lib, name)
        fn.argtypes = [_U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long]
        fn.restype = None
    lib.metropolis_sweep.argtypes = [_U8, _U8, _U8, _U8,
                                     ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_int]
    lib.metropolis_sweep.restype = None
    lib.metropolis_sweep_rng.argtypes = [_U8, ctypes.c_uint, ctypes.c_uint, _U8,
                                         ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_int]
    lib.metropolis_sweep_rng.restype = None
    lib.metropolis_sweep_mt.argtypes = [_U8, _U8, _U8, _U8, ctypes.c_long, ctypes.c_long,
                                        ctypes.c_long, ctypes.c_int, ctypes.c_int]
    lib.metropolis_sweep_mt.restype = None
    lib.metropolis_sweep_rng_mt.argtypes = [_U8, ctypes.c_uint, ctypes.c_uint, _U8,
                                            ctypes.c_long, ctypes.c_long, ctypes.c_long,
                                            ctypes.c_int, ctypes.c_int]
    lib.metropolis_sweep_rng_mt.restype = None
    lib.plaquette_mt.argtypes = [_U8, _U8, ctypes.c_long, ctypes.c_long, ctypes.c_long,
                                 ctypes.c_int]
    lib.plaquette_mt.restype = None
    _PL = ctypes.POINTER(ctypes.c_long)
    for nm in ("action_sums", "correlation_sums"):
        fn = getattr(lib, nm)
        fn.argtypes = ([_U8, ctypes.c_long, ctypes.c_long, ctypes.c_long, _PL, _PL]
                       if nm == "action_sums"
                       else [_U8, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_long,
                             _PL, _PL])
        fn.restype = None
    return lib


def _threads_for(n):
    return NTHREADS if n >= MT_MIN_NODES else 1


def _load():
    global _lib, _tried
    if _lib is not None or _tried:
        return _lib
    _tried = True
    try:
        # Rebuild BEFORE the first CDLL when the artifact predates its source: dyld caches
        # images by path, so replacing the file after loading cannot take effect in-process.
        if not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(_C):
            build()
        _lib = _bind(ctypes.CDLL(_SO))
    except Exception:
        _lib = None
    return _lib


def available():
    return _load() is not None


def _ptr(ba):
    return (ctypes.c_uint8 * len(ba)).from_buffer(ba)


def _py_plaquette(g, W, H, D):
    e = bytearray(len(g))
    for k in range(1, D - 1):
        for j in range(1, H - 1):
            base = (k * H + j) * W
            for i in range(1, W - 1):
                c = base + i
                pos = (g[c] + g[c + 1]) & 0xFF
                neg = (g[c + W] + g[c - 1]) & 0xFF
                e[c] = (pos - neg) & 0xFF
    return e


def plaquette(g, W, H, D, blocked=True, force_python=False):
    """Wilson plaquette action over the W x H x D lattice. Returns an energy bytearray (C mem).
    blocked=True uses 64-depth cache tiling. Falls back to pure Python if no C kernel."""
    if len(g) != W * H * D:
        raise ValueError(f"plaquette: len(g)={len(g)} != W*H*D={W * H * D}")
    lib = None if force_python else _load()
    if lib is None:
        return _py_plaquette(g, W, H, D)
    gb = g if isinstance(g, bytearray) else bytearray(g)
    e = bytearray(len(g))
    nt = _threads_for(len(g))
    if nt > 1:
        lib.plaquette_mt(_ptr(e), _ptr(gb), W, H, D, nt)
        return e
    fn = lib.plaquette_blocked if blocked else lib.plaquette
    fn(_ptr(e), _ptr(gb), W, H, D)
    return e


def _py_sweep(grid, prop, chance, lut, W, H, D, parity):
    sk = W * H
    def cd(a, b):
        d = (a - b) & 0xFF
        return d if d < 256 - d else 256 - d
    for k in range(1, D - 1):
        for j in range(1, H - 1):
            base = k * sk + j * W
            for i in range(1, W - 1):
                if ((i + j + k) & 1) != parity:
                    continue
                c = base + i
                old = grid[c]; nv = (old + prop[c]) & 0xFF
                nbs = (grid[c+1], grid[c-1], grid[c+W], grid[c-W], grid[c+sk], grid[c-sk])
                So = sum(cd(old, x) for x in nbs)
                Sn = sum(cd(nv, x) for x in nbs)
                dS = Sn - So
                if dS <= 0 or chance[c] < lut[min(dS, 255)]:
                    grid[c] = nv


def _metal_gauge_ready():
    """Load-time eligibility gate for the Metal sweep: bit-for-bit vs the Python reference
    on a fixed 8^3 lattice, once per process. D9's bar, enforced at the door."""
    global _metal_ok
    if _metal_ok is not None:
        return _metal_ok
    try:
        from ringkit.kernels.apple.metal import host as mh
        if not mh.available():
            _metal_ok = False
            return False
        W = H = D = 8
        n = W * H * D
        g = bytearray((i & 0xFF) for i in range(n))
        prop = bytearray(((i << 3) + 5) & 0xFF for i in range(n))
        chance = bytearray(((i << 1) + 17) & 0xFF for i in range(n))
        lut = bytearray(255 - d if d < 255 else 0 for d in range(256))
        want = bytearray(g)
        _py_sweep(want, prop, chance, lut, W, H, D, 0)
        _py_sweep(want, prop, chance, lut, W, H, D, 1)
        got = bytearray(g)
        _metal_ok = mh.gauge_sweep(got, prop, chance, lut, W, H, D) == 0 and got == want
    except Exception:
        _metal_ok = False
    return _metal_ok


def sweep(grid, prop, chance, lut, W, H, D, force_python=False):
    """One full Metropolis sweep (both checkerboard parities), in place on `grid` (bytearray).
    Routes to the Metal GPU for lattices >= GAUGE_METAL_MIN_NODES (measured crossover; falls
    through to C on any failure), else the C kernel, else the Python reference."""
    if not force_python and len(grid) >= GAUGE_METAL_MIN_NODES and _metal_gauge_ready():
        from ringkit.kernels.apple.metal import host as mh
        if mh.gauge_sweep(grid, prop, chance, lut, W, H, D) == 0:
            return grid
    lib = None if force_python else _load()
    if lib is None:
        _py_sweep(grid, prop, chance, lut, W, H, D, 0)
        _py_sweep(grid, prop, chance, lut, W, H, D, 1)
        return grid
    pb, cb, lb = bytearray(prop), bytearray(chance), bytearray(lut)
    nt = _threads_for(len(grid))
    for parity in (0, 1):
        lib.metropolis_sweep_mt(_ptr(grid), _ptr(pb), _ptr(cb), _ptr(lb), W, H, D, parity, nt)
    return grid


def _rand32(seed, sweep, idx):
    """Reference of record for the counter RNG (rk_mix32) — MUST match gauge.c and gauge.metal
    bit-for-bit (gated). Randoms are derived from (seed, sweep, node), never stored."""
    x = (idx + (sweep + 1) * 0x9E3779B9) & 0xFFFFFFFF
    x ^= (seed * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def _py_sweep_rng(grid, seed, sweep, lut, W, H, D, parity):
    sk = W * H
    def cd(a, b):
        d = (a - b) & 0xFF
        return d if d < 256 - d else 256 - d
    for k in range(1, D - 1):
        for j in range(1, H - 1):
            base = k * sk + j * W
            for i in range(1, W - 1):
                if ((i + j + k) & 1) != parity:
                    continue
                c = base + i
                x = _rand32(seed, sweep, c)
                pr = x & 0xFF
                ch = (x >> 8) & 0xFF
                old = grid[c]; nv = (old + pr) & 0xFF
                nbs = (grid[c+1], grid[c-1], grid[c+W], grid[c-W], grid[c+sk], grid[c-sk])
                So = sum(cd(old, v) for v in nbs)
                Sn = sum(cd(nv, v) for v in nbs)
                dS = Sn - So
                if dS <= 0 or ch < lut[min(dS, 255)]:
                    grid[c] = nv


_metal_rng_ok = None


def _metal_rng_ready():
    """Eligibility gate for the on-GPU RNG path: metal thermalize_rng must equal the Python
    reference on a fixed 8^3 lattice / 2 sweeps, once per process."""
    global _metal_rng_ok
    if _metal_rng_ok is not None:
        return _metal_rng_ok
    try:
        from ringkit.kernels.apple.metal import host as mh
        if not mh.available():
            _metal_rng_ok = False
            return False
        W = H = D = 8
        n = W * H * D
        g = bytearray((i & 0xFF) for i in range(n))
        lut = bytearray(255 - d if d < 255 else 0 for d in range(256))
        want = bytearray(g)
        for s in (0, 1):
            _py_sweep_rng(want, 12345, s, lut, W, H, D, 0)
            _py_sweep_rng(want, 12345, s, lut, W, H, D, 1)
        got = bytearray(g)
        _metal_rng_ok = (mh.thermalize_rng(got, 12345, 0, lut, W, H, D, 2) == 0
                         and got == want)
    except Exception:
        _metal_rng_ok = False
    return _metal_rng_ok


def thermalize_rng(grid, seed, lut, W, H, D, sweeps, force_python=False):
    """Run `sweeps` full sweeps with DERIVED randoms (rk_mix32 counter RNG): nothing random
    is generated CPU-side or transferred — the unified-GPU path moves only grid + lut.
    Routes metal (floor + gate) -> C -> Python reference; all three are the same spec."""
    seed &= 0xFFFFFFFF
    n = W * H * D
    if (not force_python and n >= GAUGE_METAL_MIN_NODES and _metal_rng_ready()):
        from ringkit.kernels.apple.metal import host as mh
        if mh.thermalize_rng(grid, seed, 0, lut, W, H, D, sweeps) == 0:
            return grid
    lib = None if force_python else _load()
    if lib is None:
        for s in range(sweeps):
            _py_sweep_rng(grid, seed, s, lut, W, H, D, 0)
            _py_sweep_rng(grid, seed, s, lut, W, H, D, 1)
        return grid
    lb = lut if isinstance(lut, bytearray) else bytearray(lut)
    nt = _threads_for(len(grid))
    for s in range(sweeps):
        for parity in (0, 1):
            lib.metropolis_sweep_rng_mt(_ptr(grid), seed, s, _ptr(lb), W, H, D, parity, nt)
    return grid


def thermalize(grid, props, chances, lut, W, H, D, sweeps):
    """Run `sweeps` full sweeps over concatenated per-sweep random arrays (sweeps*n each).
    Fused GPU path when eligible (floor + metal self-test + 16-byte offset alignment, since
    per-sweep buffer offsets are s*n); falls back to per-sweep dispatch, same semantics."""
    n = W * H * D
    if (sweeps >= 2 and n >= GAUGE_METAL_MIN_NODES and (n & 15) == 0
            and _metal_gauge_ready()):
        from ringkit.kernels.apple.metal import host as mh
        if mh.thermalize(grid, props, chances, lut, W, H, D, sweeps) == 0:
            return grid
    for s in range(sweeps):
        o = s * n
        sweep(grid, props[o:o + n], chances[o:o + n], lut, W, H, D)
    return grid


def session_for(grid, W, H, D):
    """A persistent-GPU session for this lattice, or None when ineligible (no metal, below
    the routing floor, or the rng path failed its bit-for-bit gate). Caller owns sync points."""
    n = W * H * D
    if n < GAUGE_METAL_MIN_NODES or len(grid) != n or not _metal_rng_ready():
        return None
    try:
        from ringkit.kernels.apple.metal import host as mh
        return mh.GaugeSession(grid, W, H, D)
    except Exception:
        return None


def correlation(grid, R, W, H, D):
    """Order parameter: mean alignment of links R apart along the i-axis, normalized 0..1.
    1 = perfectly aligned (ordered), ~0.5 = random, 0 = anti-aligned. Ring-distance based;
    the normalized value is a float MEASUREMENT output (IO), not a ring quantity."""
    lib = _load()
    if lib is not None:
        gb = grid if isinstance(grid, bytearray) else bytearray(grid)
        tot = ctypes.c_long(0); n = ctypes.c_long(0)
        lib.correlation_sums(_ptr(gb), R, W, H, D, ctypes.byref(tot), ctypes.byref(n))
        return tot.value / (n.value * 128) if n.value else 0.0
    sk = W * H
    def cd(a, b):
        d = (a - b) & 0xFF
        return d if d < 256 - d else 256 - d
    tot = 0; n = 0
    for k in range(1, D - 1):
        for j in range(1, H - 1):
            base = k * sk + j * W
            for i in range(1, W - 1 - R):
                c = base + i
                tot += 128 - cd(grid[c], grid[c + R])
                n += 1
    return tot / (n * 128) if n else 0.0


# Confinement reading on the C(R) profile — MEASUREMENT constants (this layer owns float
# IO). Our correlation normalizes RANDOM alignment to 0.5 (the sources' <cos> maps it to 0),
# so the mass-gap signature is the EXCESS over 0.5 dying out: confined = no alignment
# beyond random by R = PHASE_R; deconfined = long-range excess persists.
PHASE_R = 5
PHASE_EXCESS_BELOW = 0.05


def correlation_profile(grid, W, H, D, rmax=10):
    """C(R) for R = 1..rmax — the mass-gap observable (QCM paper's main physics readout).
    Float MEASUREMENT outputs (IO)."""
    return [correlation(grid, R, W, H, D) for R in range(1, int(rmax) + 1)]


def phase_of(profile):
    """Read the phase from a C(R) profile: 'confined' if the alignment EXCESS over the
    random baseline (0.5) has died by R=PHASE_R (mass gap), else 'deconfined'. A labeled
    measurement interpretation, not a ring value."""
    R = min(PHASE_R, len(profile))
    return "confined" if profile[R - 1] - 0.5 < PHASE_EXCESS_BELOW else "deconfined"


def mean_action(grid, W, H, D):
    """Average local ring-action (sum of neighbor ring-distances) — order parameter.
    Low = ordered/aligned (cold), high = disordered (hot). Float MEASUREMENT output (IO)."""
    lib = _load()
    if lib is not None:
        gb = grid if isinstance(grid, bytearray) else bytearray(grid)
        tot = ctypes.c_long(0); n = ctypes.c_long(0)
        lib.action_sums(_ptr(gb), W, H, D, ctypes.byref(tot), ctypes.byref(n))
        return tot.value / n.value if n.value else 0
    sk = W * H
    def cd(a, b):
        d = (a - b) & 0xFF
        return d if d < 256 - d else 256 - d
    tot = 0; n = 0
    for k in range(1, D - 1):
        for j in range(1, H - 1):
            base = k * sk + j * W
            for i in range(1, W - 1):
                c = base + i
                tot += cd(grid[c], grid[c+1]) + cd(grid[c], grid[c+W]) + cd(grid[c], grid[c+sk])
                n += 3
    return tot / n if n else 0
