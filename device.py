"""ringkit.device — explicit backend selection (the .device() pattern, like torch's device).

You CHOOSE the compute backend; every op dispatches to that device's C/CUDA/Metal kernel. Python is
the interface only — the real engineering is the kernels. A selected device that is unavailable, or is
missing a requested kernel, RAISES: no silent Python fallback (the kit is kernel-fast by contract).

    import ringkit as rk
    rk.devices()                      # {name: available?}  on this machine
    dev = rk.device("cuda")           # "cpu" | "cpu+simd" | "cuda" | "metal"; None -> best available
    dev.sigmoid(xs, frac)             # dispatched to the device's sigmoid kernel
    dev.rmsnorm(x, w, frac, eps)      # ...

Devices map to the kernel backends already in the tree:
  cuda        -> kernels/nvidia/cuda (nvcc)          [gemv, gemm, elementwise, sigmoid/exp/rmsnorm]
  metal       -> kernels/apple/metal (Apple GPU)     [elementwise; more as ported]
  cpu / cpu+simd -> kernels/mprc/gemma + backend cpu-c C kernels   [built where a C toolchain exists]
The cpu backends need the cross-platform C build (Win/Mac/Linux) to be present.
"""

CPU, CPU_SIMD, CUDA, METAL = "cpu", "cpu+simd", "cuda", "metal"
ALL = (CPU, CPU_SIMD, CUDA, METAL)


def _cuda_host():
    try:
        from ringkit.kernels.nvidia.cuda import host as ch
        return ch if ch.available() else None
    except Exception:
        return None


def _metal_host():
    try:
        from ringkit.kernels.apple.metal import host as mh
        return mh if mh.available() else None
    except Exception:
        return None


def _cpu_host():
    """The CPU backend. SPEC-014: native Rust (`kernels.cpu_rust`) — cross-platform via cargo (fixes
    the C backend's cpu:False), SIMD (LLVM), multi-thread (rayon). Falls back to the mprc C kernel
    only where Rust is absent but the C toolchain built."""
    try:
        from ringkit.kernels.cpu_rust import host as rh
        if rh.available():
            return rh
    except Exception:
        pass
    try:
        from ringkit.kernels.mprc.gemma import host as kh
        return kh if kh.available() else None
    except Exception:
        return None


def devices():
    """{device: available?} on this machine — the honest picture (built + self-tested + hardware present)."""
    cpu_ok = _cpu_host() is not None
    return {CPU: cpu_ok, CPU_SIMD: cpu_ok,
            CUDA: _cuda_host() is not None, METAL: _metal_host() is not None}


class Device:
    """A selected compute backend. Its methods dispatch to that backend's kernel; a missing kernel or
    unavailable device raises (never a Python fallback)."""
    __slots__ = ("name", "_h")

    def __init__(self, name):
        if name not in ALL:
            raise ValueError(f"unknown device {name!r}; choose from {ALL}")
        self.name = name
        self._h = _cpu_host() if name in (CPU, CPU_SIMD) else (_cuda_host() if name == CUDA else _metal_host())
        if self._h is None:
            raise RuntimeError(
                f"device {name!r} is unavailable on this machine (devices: {devices()}). "
                f"Build its backend (CUDA=nvcc, cpu=cross-platform C, metal=Apple) or pick an available one.")

    def _k(self, fn):
        f = getattr(self._h, fn, None)
        if f is None:
            raise RuntimeError(
                f"device {self.name!r} has no '{fn}' kernel yet — write it at the low level "
                f"(no Python fallback).")
        return f

    def _vec(self, fn, *args):
        r = self._k(fn)(*args)
        if r is None:
            raise RuntimeError(f"device {self.name!r}: '{fn}' kernel returned None (regime unsupported "
                               f"on this backend). Extend the kernel — refusing the Python path.")
        return r

    # ── unified op surface (present on cuda + cpu backends with the same names) ──
    def sigmoid(self, xs, frac=16):
        return self._vec("sigmoid_vec", xs, frac)

    def exp_nonpos(self, xs, frac=16):
        return self._vec("exp_vec", xs, frac)

    def rmsnorm(self, x, weight, frac=16, eps=1):
        return self._vec("rmsnorm", x, weight, frac, eps)

    def gemv(self, W, x, M, K):
        """Exact energy GEMV out[j] = Sum_i W[j*K+i]*x[i]. cuda: gemv_i64; cpu: qsm_gemv_exact wrapper."""
        if self.name == CUDA:
            return self._vec("gemv_i64", W, x, M, K)
        return self._vec("gemv", W, x, M, K)      # cpu backend's uniform gemv (added in phase 2)

    def gemm(self, X, W, T, M, K):
        """Batched exact energy GEMM: out[t*M+m] = Sum_k X[t*K+k]*W[m*K+k] — all T tokens in ONE
        call (the batching that replaces per-token gemv; ~1000x fewer launches)."""
        return self._vec("gemm_i64", X, W, T, M, K)

    # ── Q16 energy primitives the forward/backward compose from ──
    def emul(self, a, b, frac=16):
        """Q16 elementwise (a*b)>>frac (arithmetic FLOOR, == _mul_q16 / infer.linear)."""
        return self._vec("emul_q16", a, b, frac)

    def eadd(self, a, b):
        """Elementwise a+b (int64 energy, no fold)."""
        return self._vec("eadd", a, b)

    def esub(self, a, b):
        """Elementwise a-b (int64 energy, no fold)."""
        return self._vec("esub", a, b)

    def escale(self, a, sc, frac=16):
        """Q16 scale by scalar sc: (a*sc)>>frac."""
        return self._vec("escale_q16", a, sc, frac)

    def colsum(self, rows_flat, R, C):
        """Column-sum over R rows of length C: out[c] = Sum_r rows_flat[r*C+c] (reduction/mean-pool)."""
        return self._vec("colsum", rows_flat, R, C)

    def diffuse(self, grid_flat, D, H, hd):
        """One toroidal 4-neighbour heat step over a (D,H) grid of hd-vectors (SSD lattice2d)."""
        return self._vec("diffuse", grid_flat, D, H, hd)

    def relu(self, a):
        """max(a[i], 0) elementwise (frontend quadrant rectifier)."""
        return self._vec("relu", a)

    def gather(self, lut, idx):
        """out[i] = lut[idx[i]] — LUT lookup by u8 arc byte (trig table gather)."""
        return self._vec("gather", lut, idx)

    @property
    def raw(self):
        """The underlying backend host module (escape hatch for backend-specific kernels)."""
        return self._h

    def __repr__(self):
        return f"Device({self.name!r})"


_DEFAULT = None


def device(name=None):
    """Select a device. None -> best available (cuda > metal > cpu+simd > cpu). Raises if none/unavailable."""
    if name is not None:
        return Device(name)
    av = devices()
    for n in (CUDA, METAL, CPU_SIMD, CPU):
        if av.get(n):
            return Device(n)
    raise RuntimeError("no compute backend available on this machine — build a kernel (CUDA or CPU C).")


def default_device():
    """The process default device (best available), cached."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = device(None)
    return _DEFAULT


def _selftest():
    ok = True
    av = devices()
    print(f"  devices: {av}")
    # at least one real backend must be present (no python device — compute is kernels only)
    ok &= any(av.values())
    print(f"  >=1 kernel backend available: {'PASS' if any(av.values()) else 'FAIL'}")
    d = device(None)
    print(f"  best available device: {d}")
    # dispatch a real op to the device and check it matches our NATIVE reference (qcm.activations,
    # the promoted float-free scalar impl) -- NOT emulation (which is for public models only).
    from ringkit.qcm.activations import sigmoid_fixed as _native_sigmoid
    fr = 16
    xs = [-(3 << 16), 0, (2 << 16)]
    got = d.sigmoid(xs, fr)
    want = [_native_sigmoid(v, fr) for v in xs]
    ok &= (got == want)
    print(f"  {d}.sigmoid dispatched to kernel == reference: {'PASS' if got == want else 'FAIL'}")
    # explicit unavailable device raises (honest, no fallback)
    unavailable = next((n for n in ALL if not av.get(n)), None)
    if unavailable is not None:
        try:
            device(unavailable)
            raised = False
        except RuntimeError:
            raised = True
        ok &= raised
        print(f"  device({unavailable!r}) raises (unavailable, no fallback): {'PASS' if raised else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("ringkit.device self-test (.device() backend selection):")
    print("RESULT:", "ALL PASS" if _selftest() else "FAIL")
