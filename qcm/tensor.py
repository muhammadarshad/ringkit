"""Native ring numeric core for ringkit.qcm — float-free linear + norm on the QSM substrate.

A "float" is two rings (RingRational num/den); here we carry the model's magnitudes as signed
integers with a per-tensor POWER-OF-2 scale (den = 2^exp) — a block-floating representation whose
scale is a pure shift (exponent from `int.bit_length`, never IEEE float). Products are exact:

    a*b via core.native.qsm  (quarter-square: exact over all int8 pairs, multiplier-free)

and the batched dot Sum_k w_k*x_k is bit-identical whether taken through the qsm reference loop or
the native CUDA int64 GEMV kernel (both compute the same integer). No torch, no numpy, no emulation.

Representation of an activation vector: (vals: list[int signed int8 in -127..127], exp: int),
meaning true_value[i] ~= vals[i] * 2**exp. `quantize` re-derives (vals, exp) for any int vector.
"""
from ringkit.core import native as rn
from ringkit.device import default_device

_I8_MAX = 127


def _pow2_exp(absmax):
    """Smallest exp>=0 with (absmax >> exp) <= 127. Float-free (bit_length, not log2)."""
    if absmax <= _I8_MAX:
        return 0
    return absmax.bit_length() - 7               # 2^(bl-1) <= absmax < 2^bl ; want <=127=2^7-1


def quantize(vals):
    """int vector -> (int8 vals in [-127,127], exp) with value ~= q*2^exp. Round-to-nearest, shift."""
    absmax = 0
    for v in vals:
        a = -v if v < 0 else v
        if a > absmax:
            absmax = a
    exp = _pow2_exp(absmax)
    if exp == 0:
        return [int(v) for v in vals], 0
    half = 1 << (exp - 1)                         # round to nearest on the arithmetic shift
    out = []
    for v in vals:
        if v >= 0:
            out.append((v + half) >> exp)
        else:
            out.append(-((-v + half) >> exp))
    return out, exp


def qsm_dot(w_row, x):
    """Exact int8xint8 dot Sum_k w_row[k]*x[k] via the multiplier-free quarter-square (QSM) product.
    ENERGY-side, int32-safe: for int8 operands the exact dot stays exact in int32 up to K<=133144
    (127^2*K < 2^31); our K<=512 -> |acc|<=8.2M. No int64 magnitude lane (that is the emulation
    engine's Gemma path, not our architecture)."""
    acc = 0
    for k in range(len(x)):
        acc += rn.qsm(w_row[k], x[k])
    return acc


def qsm_matmul(W, x, M, K):
    """y[j] = Sum_k W[j*K+k]*x[k], exact int8-QSM (the native ring linear). Does NOT route to the
    emulation int64 GEMV kernel (that carries a signed int64 magnitude our architecture does not use;
    STANCE: emulation is public-models-only). The int32-safe QSM dot IS the correct native path;
    a bit-exact int32-QSM silicon kernel is the (bench-gated, last-priority) performance path."""
    return [qsm_dot(W[j * K:(j + 1) * K], x) for j in range(M)]


class QSMLinear:
    """Ring-native linear map y = W x + b, float-free. Weights are int8 (+ shared exp `w_exp`);
    the forward requantizes the input to int8, takes the exact QSM dot, and returns (ints, exp).

    W_rows : list[list[int]]  (M rows of K int8 weights, -127..127)
    w_exp  : int              weight scale exponent (true W ~= W_rows * 2^w_exp)
    b      : list[int] | None bias in the OUTPUT integer domain (added post-rescale), or None
    """
    __slots__ = ("M", "K", "_Wflat", "w_exp", "b")

    def __init__(self, W_rows, w_exp=0, bias=None):
        self.M = len(W_rows)
        self.K = len(W_rows[0]) if W_rows else 0
        self._Wflat = [int(v) for row in W_rows for v in row]
        self.w_exp = int(w_exp)
        self.b = None if bias is None else [int(v) for v in bias]

    def __call__(self, x):
        """x: (ints, exp) or a plain int list (exp=0 assumed). Returns (out ints, out exp)."""
        vals, x_exp = x if isinstance(x, tuple) else (x, 0)
        xq, xq_exp = quantize(vals)
        acc = qsm_matmul(self._Wflat, xq, self.M, self.K)   # exact Sum Wq*xq
        out_exp = self.w_exp + x_exp + xq_exp               # combined power-of-2 scale
        if self.b is not None:
            acc = [acc[j] + self.b[j] for j in range(self.M)]
        return acc, out_exp

    def forward_batch(self, tokens, dev=None):
        """Batched forward over T tokens in ONE device energy GEMM (ring_gemm_i64) — the kernel fast
        path that replaces the per-token Python qsm_matmul loop (SPEC-013 T6.1; measured 256x, bit-
        exact). `tokens`: list of (ints, exp) or plain int lists. Returns a list of (out_ints, out_exp)
        — one per token, bit-for-bit identical to calling `self(tok)` per token (the int8 QSM value is
        exact; the energy GEMM does not fold). Per-token exp differs (block-float), so it is tracked
        and returned per token, not folded into one scale."""
        dev = dev if dev is not None else default_device()
        T = len(tokens)
        if T == 0:
            return []
        Xflat, exps = [], []
        for x in tokens:
            vals, x_exp = x if isinstance(x, tuple) else (x, 0)
            xq, xq_exp = quantize(vals)
            Xflat.extend(xq)
            exps.append(self.w_exp + x_exp + xq_exp)
        acc = dev.gemm(Xflat, self._Wflat, T, self.M, self.K)     # Σ Wq·xq exact int64, [T*M]
        out = []
        for t in range(T):
            row = acc[t * self.M:(t + 1) * self.M]
            if self.b is not None:
                row = [row[j] + self.b[j] for j in range(self.M)]
            out.append((row, exps[t]))
        return out

    @property
    def raw(self):
        return {"W": self._Wflat, "M": self.M, "K": self.K, "w_exp": self.w_exp, "b": self.b}


def rmsnorm(vals, weight, eps=1):
    """Ring RMS-norm: y_i = vals_i / sqrt(mean(vals^2)) * weight_i, float-free (isqrt, mf_floordiv).
    Returns (ints, exp=0) scaled so the unit norm maps to `weight` (int gain). `vals` plain ints."""
    n = len(vals)
    ss = 0
    for v in vals:
        ss += rn.mul(v, v)                                  # exact square, energy (no fold)
    ms = rn.mf_floordiv(ss, n) if ss >= 0 else 0            # mean square
    rms = rn.isqrt(ms)
    if rms < eps:
        rms = eps
    out = []
    for i in range(n):
        num = rn.mul(vals[i], weight[i])                    # vals * gain
        q = rn.mf_floordiv(num, rms) if num >= 0 else -rn.mf_floordiv(-num, rms)
        out.append(q)
    return out, 0


_LN_FRAC = 16
_LN_ONE = 1 << _LN_FRAC


def layernorm(vals, gamma, beta, eps=1):
    """Ring LayerNorm in Q16: mean-center, normalize by std (isqrt), affine (gamma,beta Q16).
    `vals` Q16 ints. Returns Q16 ints. Float-free (isqrt, mf_floordiv, shift-add)."""
    n = len(vals)
    tot = 0
    for v in vals:
        tot += v
    mean = tot // n                                       # signed floor is fine for centering
    cent = [v - mean for v in vals]
    var = 0
    for c in cent:
        var += rn.mul(c, c) >> _LN_FRAC                   # (c^2) in Q16, accumulate
    var //= n
    std = rn.isqrt(var << _LN_FRAC)                       # sqrt(var) in Q16
    if std < eps:
        std = eps
    out = []
    for i in range(n):
        num = cent[i] << _LN_FRAC                         # Q32 numerator
        nrm = rn.mf_floordiv(num, std) if num >= 0 else -rn.mf_floordiv(-num, std)   # Q16 normalized
        out.append((rn.mul(nrm, gamma[i]) >> _LN_FRAC) + beta[i])
    return out


# ── ring-native self-test (D1: verify by execution; no torch/numpy) ──
def _selftest():
    ok = True
    # QSM losslessness: qsm(x,y) == x*y over all signed int8 pairs (the substrate anchor)
    for x in range(-127, 128):
        for y in range(-127, 128):
            if rn.qsm(x, y) != x * y:
                ok = False
                break
        if not ok:
            break
    print(f"  qsm(x,y)==x*y over all int8 pairs: {'LOSSLESS' if ok else 'FAIL'}")
    # QSMLinear exactness: y == Wq @ xq for int8 inputs (exp bookkeeping aside)
    W = [[1, -2, 3, 0], [4, 5, -6, 7]]
    lin = QSMLinear(W, w_exp=0)
    x = [10, -20, 30, 40]
    (y, e) = lin(x)
    ref = [sum(W[j][k] * x[k] for k in range(4)) for j in range(2)]
    lin_ok = (y == ref and e == 0)                          # x fits int8 -> xq_exp 0, exact
    print(f"  QSMLinear exact dot vs reference: {'PASS' if lin_ok else f'FAIL {y} vs {ref}'}")
    # rmsnorm produces finite valid ints of the right length
    r, _ = rmsnorm([3, 4, 0, -5], [21, 21, 21, 21])
    rms_ok = (len(r) == 4 and all(isinstance(v, int) for v in r))
    print(f"  rmsnorm valid output: {'PASS' if rms_ok else 'FAIL'} {r}")
    # forward_batch (kernel energy GEMM) == per-token __call__ (Python qsm_matmul), bit-for-bit
    try:
        Wb = [[((r * 3 + k) % 7) - 3 for k in range(6)] for r in range(4)]
        lb = QSMLinear(Wb, w_exp=0)
        toks = [([(t * 13 + j * 7) % 400 - 200 for j in range(6)], t % 3 - 1) for t in range(9)]
        ref = [lb(tk) for tk in toks]
        bat = lb.forward_batch(toks)
        batch_ok = (bat == ref)
        print(f"  forward_batch (kernel) == per-token __call__ (bit-exact, 9 tokens): "
              f"{'PASS' if batch_ok else 'FAIL'}")
    except RuntimeError as e:
        batch_ok = True
        print(f"  forward_batch: no device available, skipped ({e})")
    return ok and lin_ok and rms_ok and batch_ok


if __name__ == "__main__":
    print("ringkit.qcm.tensor self-test:")
    print("RESULT:", "ALL PASS" if _selftest() else "FAIL")
