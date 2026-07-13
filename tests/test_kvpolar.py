"""MOVED. The polar (Euclidean) KV codec was removed as an MPRC anti-pattern — Cartesian->polar via
atan2 + the L2 magnitude sqrt(x^2+y^2) is foreign standard math and lossy. The ring-native element
is ADI (accumulation, differential); its tests live in tests/test_kvadi.py. This file remains only
so nothing that references the old path errors; it asserts the shim still forwards to kvadi."""
from ringkit.ml import kvpolar as shim   # deprecated shim -> kvadi

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

check("kvpolar is now a shim forwarding to the ADI element (no Euclidean)",
      shim.decode_pair(*shim.encode_pair(30, 30)) == (30, 30))
check("the removed Euclidean quantizer is gone", not hasattr(shim, "quantize_element"))

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
