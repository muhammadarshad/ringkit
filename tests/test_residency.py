"""Test for the NVMe residency kernel (Phase B, D9 silicon layer).

Generic byte-addressable file residency lives in ringkit/kernels/residency (crate
`ring_residency`) behind the residency host; this suite gates it against a plain buffered-file
Python reference. No key/value/attention semantics are tested here — those are later phases.
Run: python -m ringkit.tests.test_residency
"""
import os
import random
import shutil
import tempfile

from ringkit.kernels.residency import host as rh

fails = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)


def must_raise(fn):
    try:
        fn()
        return False
    except Exception:
        return True


print("== availability: the residency kernel built + self-tested at load ==")
avail = rh.available()
check("residency host.available() is True (ring_residency built, selftest passed)", avail)
if not avail:
    # No fallback path to test if the backend didn't load — report and stop honestly.
    print("RESULT:", "ALL PASS" if not fails else f"FAIL ({len(fails)}): {fails}")
    raise SystemExit(0 if not fails else 1)

SECTOR = rh.SECTOR
tmpdir = tempfile.mkdtemp(prefix="ringkit_residency_test_")
rng = random.Random(424242)

try:
    print("== (a) sector_size() reports the platform sector size ==")
    check(f"sector_size() == {SECTOR}", rh._load().sector_size() == SECTOR)

    print("== (b) open + randomized aligned round-trip vs a plain-buffered Python reference ==")
    n_sectors = 12
    size = n_sectors * SECTOR
    native_path = os.path.join(tmpdir, "residency.bin")
    ref_path = os.path.join(tmpdir, "reference.bin")
    res = rh.open_residency(native_path, size)
    check("open_residency returns a live handle", res is not None)
    with open(ref_path, "wb") as f:
        f.write(bytes(size))

    ok = True
    for _ in range(60):
        n_sec = rng.randint(1, n_sectors)
        offset = rng.randint(0, n_sectors - n_sec) * SECTOR
        length = n_sec * SECTOR
        pattern = bytes(rng.randrange(256) for _ in range(length))
        res.write(offset, pattern)
        with open(ref_path, "r+b") as f:
            f.seek(offset)
            f.write(pattern)
        got = bytes(res.read(offset, length))
        with open(ref_path, "rb") as f:
            f.seek(offset)
            want = f.read(length)
        if got != want:
            ok = False
    check(f"60 randomized aligned writes/reads, byte-for-byte == reference", ok)

    print("== (c) whole-file reread after multiple overlapping writes matches the reference ==")
    with open(ref_path, "rb") as f:
        want_whole = f.read()
    got_whole = bytes(res.read(0, size))
    check("full-file read matches the reference after all writes", got_whole == want_whole)

    print("== (d) alignment violations raise on both write and read ==")
    check("write with misaligned offset raises", must_raise(lambda: res.write(1, bytes(SECTOR))))
    check("write with misaligned length raises", must_raise(lambda: res.write(0, bytes(SECTOR - 1))))
    check("read with misaligned offset raises", must_raise(lambda: res.read(1, SECTOR)))
    check("read with misaligned length raises", must_raise(lambda: res.read(0, SECTOR - 1)))

    print("== (e) open() guards: non-sector-multiple size, and no silent truncation ==")
    check("open() with a non-sector-multiple size raises",
          must_raise(lambda: rh.open_residency(os.path.join(tmpdir, "bad.bin"), SECTOR - 1)))
    check("re-opening an existing larger file with a smaller size raises (refuses to truncate)",
          must_raise(lambda: rh.open_residency(native_path, SECTOR)))

    print("== (f) unavailable-vs-caller-error distinction ==")
    check("a loaded backend's own errors propagate (not swallowed to None)",
          must_raise(lambda: rh.open_residency(os.path.join(tmpdir, "bad2.bin"), 1)))

    res.close()
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

print("RESULT:", "ALL PASS" if not fails else f"FAIL ({len(fails)}): {fails}")
