"""Aggregate runner for the whole ringkit test suite. Run: python3 -m ringkit.tests.run_all"""
import subprocess
import sys

MODULES = [
    "test_constants", "test_native", "test_stats", "test_calculus", "test_linalg",
    "test_tensor", "test_rnp", "test_rmath", "test_physics", "test_ml", "test_tensor_autograd",
    "test_attention", "test_nn_facade", "test_data_facade", "test_physics_facade",
    "test_kernels", "test_gauge", "test_metal",
]


def main():
    all_pass = True
    for m in MODULES:
        r = subprocess.run([sys.executable, "-m", f"ringkit.tests.{m}"],
                           capture_output=True, text=True)
        results = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT")]
        status = results[-1] if results else "NO RESULT / ERROR"
        ok = "ALL PASS" in status
        all_pass = all_pass and ok
        print(f"  {m:16s} {status}")
    print("=" * 44)
    print("ECOSYSTEM:", "ALL GREEN" if all_pass else "FAILURES PRESENT")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
