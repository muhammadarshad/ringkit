"""ringkit.kernels.residency.host — loader for the NVMe residency kernel (Phase B, D9 silicon
layer). Pure PyO3 (no C, no ctypes ABI): the crate `ring_residency` (kernels/residency) is a
Python extension module built by `maturin develop --release`. This host builds it on first use
if the import fails, and SELF-TESTS byte-for-bit round-trip against a plain-buffered-file Python
reference before serving (D9) — no key/value/attention/position semantics live here; this module
is generic byte-range residency only. Consumers (a KV-cache tier, weight residency, content-
addressed memory) are later phases built ON this one, not baked into it.
"""
import os
import random
import subprocess
import tempfile

_DIR = os.path.dirname(__file__)
_RUST_DIR = _DIR  # the crate's Cargo.toml lives directly in kernels/residency

SECTOR = 4096

_lib = None
_tried = False


def build():
    """`maturin develop --release` -> installs ring_residency into the active Python env.
    Raises on failure. Mirrors kernels/cpu_rust/host.py's build-on-first-use convention exactly,
    including the VIRTUAL_ENV shim for a stock (venv-less) interpreter."""
    import sys
    env = dict(os.environ)
    env.setdefault("VIRTUAL_ENV", sys.prefix)
    subprocess.run(["maturin", "develop", "--release"], cwd=_RUST_DIR, check=True,
                   capture_output=True, text=True, env=env)


def _load():
    global _lib, _tried
    if _lib is not None or _tried:
        return _lib
    _tried = True
    try:
        import ring_residency
    except ImportError:
        try:
            build()
            import ring_residency  # noqa: F811 - retry the import after building
        except Exception:
            _lib = None
            return None
    except Exception:
        _lib = None
        return None
    try:
        if ring_residency.ring_residency_probe() != 42 or not _selftest(ring_residency):
            _lib = None
            return None
    except Exception:
        _lib = None
        return None
    _lib = ring_residency
    return _lib


def available():
    return _load() is not None


# ── the D9 judge: ordinary buffered file I/O — the semantic reference this kernel must match ──

def _pyref_write(path, offset, data):
    with open(path, "r+b") as f:
        f.seek(offset)
        f.write(data)


def _pyref_read(path, offset, length):
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(length)


def _selftest(rr):
    """D9 gate: native reads/writes must be byte-for-byte identical to a plain buffered Python
    reference, over randomized sector-aligned offsets/lengths/patterns — plus every alignment-
    and truncation-guard rejection the design commits to. Any failure here means _load() reports
    the backend unavailable; there is no silent wrong answer."""
    rng = random.Random(20260721)
    tmpdir = tempfile.mkdtemp(prefix="ringkit_residency_selftest_")
    native_path = os.path.join(tmpdir, "native.bin")
    ref_path = os.path.join(tmpdir, "ref.bin")
    n_sectors = 8
    size = n_sectors * SECTOR
    try:
        if rr.sector_size() != SECTOR:
            return False
        res = rr.Residency(native_path, size)
        with open(ref_path, "wb") as f:
            f.write(bytes(size))

        # randomized aligned round-trip
        for _ in range(30):
            n_sec = rng.randint(1, n_sectors)
            offset = rng.randint(0, n_sectors - n_sec) * SECTOR
            length = n_sec * SECTOR
            pattern = bytes(rng.randrange(256) for _ in range(length))
            res.write(offset, pattern)
            _pyref_write(ref_path, offset, pattern)
            got = bytes(res.read(offset, length))
            want = _pyref_read(ref_path, offset, length)
            if got != want:
                return False

        def _must_raise(fn):
            try:
                fn()
                return False
            except Exception:
                return True

        # misaligned offset / length must raise, on both write and read
        if not _must_raise(lambda: res.write(1, bytes(SECTOR))):
            return False
        if not _must_raise(lambda: res.write(0, bytes(SECTOR - 1))):
            return False
        if not _must_raise(lambda: res.read(1, SECTOR)):
            return False
        if not _must_raise(lambda: res.read(0, SECTOR - 1)):
            return False

        # non-sector-multiple size at open must raise
        if not _must_raise(lambda: rr.Residency(os.path.join(tmpdir, "bad.bin"), SECTOR - 1)):
            return False

        # opening an EXISTING larger file with a smaller size must refuse (no silent truncation)
        if not _must_raise(lambda: rr.Residency(native_path, SECTOR)):
            return False

        res.close()
        return True
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── public surface ────────────────────────────────────────────────────────────────────────

def open_residency(path, size):
    """Open (or create) a byte-addressable residency file of exactly `size` bytes (a sector
    multiple). Returns a native Residency object (its own .write/.read/.close methods), or None
    if the backend is unavailable. A loaded backend's own errors (bad alignment, refusing to
    truncate an existing larger file) propagate as real exceptions — those are caller mistakes,
    not "unavailable", and are not swallowed into a None."""
    lib = _load()
    if lib is None:
        return None
    return lib.Residency(path, size)
