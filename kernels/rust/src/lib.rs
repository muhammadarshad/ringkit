//! ringkit Rust kernel host — SPEC-014 P1/P2, migrated to PURE PyO3 ("no C in Rust").
//!
//! Every op is a `#[pyfunction]` taking/returning real Rust types (Vec<u8>/Vec<i32>/Vec<i64>,
//! usize, bool) — NO out-params, NO raw pointers, NO `#[no_mangle]`, NO `extern "C"`, NO
//! `std::os::raw::*` at the API. The gauge engine's parallel path (previously a raw-pointer
//! `SendMutU8` trick) is reimplemented with SAFE Rust: the mutable output buffer is split into
//! disjoint per-thread slabs via `split_at_mut`, and any cross-slab NEIGHBOUR reads go through an
//! immutable snapshot taken before the pass starts (sound because the checkerboard-parity
//! invariant guarantees a same-pass write never lands on a site any other thread reads this pass —
//! see `sweep_dispatch` below). D9: each op is bit-for-bit == the pure-Python ring reference
//! (verified by the Python host at load). No float.
//!
//! Ring discipline: the ENERGY GEMM accumulates in i64 and NEVER folds mod 256 (a gradient/distance
//! that wraps destroys descent/ranking). The byte GEMM folds mod 256 (ARC value side). No C ABI: the
//! crate is a pure Python extension module (pyo3), built via `maturin develop --release`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

/// (a*b)>>s, ARITHMETIC FLOOR (== Python `(a*b)>>frac`, _mul_q16, k_emul). i128 exact.
#[inline]
fn mul_shr_floor(a: i64, b: i64, s: u32) -> i64 {
    ((a as i128 * b as i128) >> s) as i64
}

/// (a*b)>>s, TRUNCATE toward zero (magnitude shift, re-sign; == ract._sdiv-style, k_rmsnorm mulshr).
#[inline]
fn mul_shr_trunc(a: i64, b: i64, s: u32) -> i64 {
    let neg = (a < 0) ^ (b < 0);
    let r = (((a as i128).unsigned_abs() * (b as i128).unsigned_abs()) >> s) as i64;
    if neg { -r } else { r }
}

/// floor(sqrt(x)) for x>=0 (== rn.isqrt), on i128.
#[inline]
fn isqrt128(x: i128) -> i128 {
    if x <= 0 {
        return 0;
    }
    let mut lo: i128 = 0;
    let mut hi: i128 = 1i128 << 64; // sqrt(i128 upper range) headroom
    while lo < hi {
        let mid = (lo + hi + 1) >> 1;
        if mid <= x / mid { lo = mid; } else { hi = mid - 1; }
    }
    lo
}

/// e^x in Q<frac> (== qcm.activations.exp_fixed / ract.exp_fixed): range-reduce (halve) + 12-term
/// integer Taylor + re-square; e^-|x| = 1/e^|x|. Positive intermediates; i128 products are exact.
fn exp_fixed(x: i64, frac: u32) -> i64 {
    let one: i64 = 1 << frac;
    let neg = x < 0;
    let ax = if neg { -x } else { x };
    let half = one >> 1;
    let mut m = 0u32;
    let mut red = ax;
    while red > half { red >>= 1; m += 1; }
    let mut term = one;
    let mut acc = one;
    for k in 1..=12i64 {
        term = mul_shr_floor(term, red, frac) / k; // mf_floordiv (term,red>=0)
        acc += term;
        if term == 0 { break; }
    }
    for _ in 0..m { acc = mul_shr_floor(acc, acc, frac); }
    if neg { acc = (1i64 << (frac + frac)) / acc; }
    acc
}

/// 1/(1+e^-x) in Q<frac> (== qcm.activations.sigmoid_fixed). Saturates |x| at frac<<frac.
fn sigmoid_fixed(x: i64, frac: u32) -> i64 {
    let one: i64 = 1 << frac;
    let lim: i64 = (frac as i64) << frac;
    let x = x.clamp(-lim, lim);
    let e = exp_fixed(-x, frac);
    (1i64 << (frac + frac)) / (one + e)
}

// ═══════════════════════════════════════════════════════════════════════════════
// FULL CPU DEVICE OP SURFACE (SPEC-014: CPU = Rust). Bit-for-bit == the ring references
// (qcm.activations / emulation.ract): energy GEMM/GEMV (i64, no fold), Q16 elementwise
// (arithmetic-floor `(a*b)>>frac`), colsum, toroidal diffuse, relu, gather, and the fixed-point
// sigmoid/exp/rmsnorm. i128 intermediates match Python's arbitrary precision exactly.
// ═══════════════════════════════════════════════════════════════════════════════

/// Batched exact ENERGY GEMM: out[t*M+m] = Σ_k X[t*K+k]·W[m*K+k], i64 accumulate, NO fold.
#[pyfunction]
fn ring_gemm_i64(x: Vec<i32>, w: Vec<i32>, t: usize, m: usize, k: usize) -> PyResult<Vec<i64>> {
    if x.len() != t * k || w.len() != m * k {
        return Err(PyValueError::new_err("ring_gemm_i64: length mismatch"));
    }
    Ok((0..t)
        .into_par_iter()
        .flat_map(|ti| {
            let xr = &x[ti * k..ti * k + k];
            (0..m)
                .map(|mi| {
                    let wr = &w[mi * k..mi * k + k];
                    let mut acc: i64 = 0;
                    for ki in 0..k { acc += xr[ki] as i64 * wr[ki] as i64; }
                    acc
                })
                .collect::<Vec<i64>>()
        })
        .collect())
}

/// Batched energy GEMV: out[j] = Σ_i W[j*K+i]*x[i] (i64, no fold). W,x are i32.
#[pyfunction]
fn ring_gemv_i64(w: Vec<i32>, x: Vec<i32>, m: usize, k: usize) -> PyResult<Vec<i64>> {
    if w.len() != m * k || x.len() != k {
        return Err(PyValueError::new_err("ring_gemv_i64: length mismatch"));
    }
    Ok((0..m)
        .into_par_iter()
        .map(|j| {
            let wr = &w[j * k..j * k + k];
            let mut acc: i64 = 0;
            for i in 0..k { acc += wr[i] as i64 * x[i] as i64; }
            acc
        })
        .collect())
}

/// Vector sigmoid (rayon over the elements). o[i] = sigmoid_fixed(x[i]).
#[pyfunction]
fn ring_sigmoid(x: Vec<i64>, frac: i32) -> Vec<i64> {
    let f = frac as u32;
    x.par_iter().map(|&xi| sigmoid_fixed(xi, f)).collect()
}

/// Vector exp (softmax/nonpos domain, but valid either sign). o[i] = exp_fixed(x[i]).
#[pyfunction]
fn ring_exp(x: Vec<i64>, frac: i32) -> Vec<i64> {
    let f = frac as u32;
    x.par_iter().map(|&xi| exp_fixed(xi, f)).collect()
}

/// Single-vector RMSNorm (== ract.rmsnorm_fixed): x/sqrt(mean(x^2)+eps)*w.
#[pyfunction]
fn ring_rmsnorm(x: Vec<i64>, w: Vec<i64>, frac: i32, eps: i64) -> PyResult<Vec<i64>> {
    let n = x.len();
    if n == 0 || w.len() != n {
        return Err(PyValueError::new_err("ring_rmsnorm: bad length"));
    }
    let f = frac as u32;
    let mut ssq: i128 = 0;
    for &v in &x {
        let a = (v as i128).unsigned_abs() as i128;
        ssq += (a * a) >> f; // (x*x)>>frac accumulated (energy, no fold)
    }
    let ms = ssq / (n as i128) + eps as i128;
    let mut rms = isqrt128(ms << f);
    if rms == 0 { rms = 1; }
    let mut os = vec![0i64; n];
    for i in 0..n {
        let axf = ((x[i] as i128).unsigned_abs() as i128) << f;
        let q = (axf / rms) as i64;
        let norm = if x[i] < 0 { -q } else { q };
        os[i] = mul_shr_trunc(norm, w[i], f); // norm*w>>frac (trunc, == k_rmsnorm mulshr)
    }
    Ok(os)
}

/// Q16 elementwise: op 0 = emul ((a*b)>>frac floor), 1 = eadd (a+b), 2 = esub (a-b). i64, no fold.
#[pyfunction]
fn ring_ew_q16(op: i32, a: Vec<i64>, b: Vec<i64>, frac: i32) -> PyResult<Vec<i64>> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("ring_ew_q16: length mismatch"));
    }
    let f = frac as u32;
    match op {
        0 => Ok(a.par_iter().zip(b.par_iter()).map(|(&x, &y)| mul_shr_floor(x, y, f)).collect()),
        1 => Ok(a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x + y).collect()),
        2 => Ok(a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x - y).collect()),
        _ => Err(PyValueError::new_err(format!("ring_ew_q16: unknown op {op}"))),
    }
}

/// Q16 scale by scalar sc: o[i] = (a[i]*sc)>>frac (floor).
#[pyfunction]
fn ring_escale(a: Vec<i64>, sc: i64, frac: i32) -> Vec<i64> {
    let f = frac as u32;
    a.par_iter().map(|&ai| mul_shr_floor(ai, sc, f)).collect()
}

/// Column-sum over R rows of length C: out[c] = Σ_r in[r*C+c].
#[pyfunction]
fn ring_colsum(inp: Vec<i64>, r: usize, c: usize) -> PyResult<Vec<i64>> {
    if inp.len() != r * c {
        return Err(PyValueError::new_err("ring_colsum: length mismatch"));
    }
    Ok((0..c)
        .into_par_iter()
        .map(|cc| {
            let mut acc: i64 = 0;
            for rr in 0..r { acc += inp[rr * c + cc]; }
            acc
        })
        .collect())
}

/// One toroidal 4-neighbour heat step over a (D,H) grid of hd-vectors: o = (up+dn+lf+rt+4*c)>>3.
#[pyfunction]
fn ring_diffuse(inp: Vec<i64>, d: usize, h: usize, hd: usize) -> PyResult<Vec<i64>> {
    let tot = d * h * hd;
    if inp.len() != tot {
        return Err(PyValueError::new_err("ring_diffuse: length mismatch"));
    }
    Ok((0..tot)
        .into_par_iter()
        .map(|idx| {
            let j = idx % hd;
            let cell = idx / hd;
            let r = cell / h;
            let c = cell % h;
            let up = (((r + d - 1) % d) * h + c) * hd + j;
            let dn = (((r + 1) % d) * h + c) * hd + j;
            let lf = (r * h + (c + h - 1) % h) * hd + j;
            let rt = (r * h + (c + 1) % h) * hd + j;
            (inp[up] + inp[dn] + inp[lf] + inp[rt] + (inp[idx] << 2)) >> 3
        })
        .collect())
}

/// relu: o[i] = max(a[i], 0).
#[pyfunction]
fn ring_relu(a: Vec<i64>) -> Vec<i64> {
    a.par_iter().map(|&ai| ai.max(0)).collect()
}

/// gather: o[i] = lut[idx[i]] (idx are u8 arc bytes).
#[pyfunction]
fn ring_gather(lut: Vec<i64>, idx: &[u8]) -> PyResult<Vec<i64>> {
    idx.par_iter()
        .map(|&ix| {
            lut.get(ix as usize)
                .copied()
                .ok_or_else(|| PyValueError::new_err("ring_gather: index outside lut"))
        })
        .collect()
}

/// Liveness probe (P1): returns 42. Confirms the extension module loaded.
#[pyfunction]
fn ring_rust_probe() -> i32 {
    42
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPU TIER, BYTE ELEMENTWISE (kernels/backend "cpu-c" registry slot, SPEC-014). Hardware wrapping
// u8 arithmetic mod 256 — bit-for-bit == kernels/backend's `_PY` reference `(a*b)&0xFF` etc.
// ═══════════════════════════════════════════════════════════════════════════════

#[inline]
fn ew_u8(op: i32, a: u8, b: u8) -> u8 {
    match op {
        0 => a.wrapping_mul(b),
        1 => a.wrapping_add(b),
        _ => a.wrapping_sub(b),
    }
}

fn ring_ew_u8_checked(py: Python<'_>, op: i32, a: &[u8], b: &[u8]) -> PyResult<Py<PyBytes>> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("ring ew: length mismatch"));
    }
    let out: Vec<u8> = a.iter().zip(b.iter()).map(|(&ai, &bi)| ew_u8(op, ai, bi)).collect();
    Ok(PyBytes::new_bound(py, &out).into())
}

/// op: 0=mul. Single-call scalar op — the exact surface kernels/backend's cpu-c tier binds.
/// `a`,`b` are borrowed zero-copy from the caller's bytes/bytearray (no per-element boxing).
#[pyfunction]
fn ring_mul(py: Python<'_>, a: &[u8], b: &[u8]) -> PyResult<Py<PyBytes>> {
    ring_ew_u8_checked(py, 0, a, b)
}
#[pyfunction]
fn ring_add(py: Python<'_>, a: &[u8], b: &[u8]) -> PyResult<Py<PyBytes>> {
    ring_ew_u8_checked(py, 1, a, b)
}
#[pyfunction]
fn ring_sub(py: Python<'_>, a: &[u8], b: &[u8]) -> PyResult<Py<PyBytes>> {
    ring_ew_u8_checked(py, 2, a, b)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Manifold staple sweep, u8 ring-native (`G:\quantum\research\qcm\sud256_qvk.jl`: W_opt,H_opt=128,113).
// The HyperVector manifold is [N][H][W], W contiguous inner (SIMD lane), H middle (prime — cache-bank
// avoidance), N outer. Per interior site: staple = 4-neighbour ring sum, then LUT-Boltzmann transform.
// ═══════════════════════════════════════════════════════════════════════════════

/// One manifold's staple sweep (shared by the single-thread and rayon paths → bit-identical).
/// `o`,`g` are one manifold [H×W] (W contiguous inner → the inner `i` loop auto-vectorizes to SIMD).
#[inline]
fn staple_one(o: &mut [u8], g: &[u8], l: &[u8], w: usize, h: usize) {
    for i in 0..w {
        o[i] = g[i];
        o[(h - 1) * w + i] = g[(h - 1) * w + i];
    }
    for j in 1..h - 1 {
        let row = j * w;
        o[row] = g[row]; // left boundary
        o[row + w - 1] = g[row + w - 1]; // right boundary
        for i in 1..w - 1 {
            let idx = row + i;
            let s = g[idx - 1] as u16 + g[idx + 1] as u16 + g[idx - w] as u16 + g[idx + w] as u16;
            let d = ((g[idx] as u16 + s) & 0xFF) as usize;
            o[idx] = l[d];
        }
    }
}

fn staple_check(grid_len: usize, lut_len: usize, w: usize, h: usize, n: usize) -> PyResult<()> {
    if w < 2 || h < 2 {
        return Err(PyValueError::new_err("ring_manifold_staple_u8: w,h must be >= 2"));
    }
    if lut_len != 256 {
        return Err(PyValueError::new_err("ring_manifold_staple_u8: lut must have 256 entries"));
    }
    if grid_len != n * h * w {
        return Err(PyValueError::new_err("ring_manifold_staple_u8: grid length != n*h*w"));
    }
    Ok(())
}

#[pyfunction]
fn ring_manifold_staple_u8(py: Python<'_>, grid: &[u8], lut: &[u8], w: usize, h: usize, n: usize) -> PyResult<Py<PyBytes>> {
    staple_check(grid.len(), lut.len(), w, h, n)?;
    let stride = h * w;
    let mut out = vec![0u8; grid.len()];
    for nn in 0..n {
        staple_one(&mut out[nn * stride..(nn + 1) * stride], &grid[nn * stride..(nn + 1) * stride], lut, w, h);
    }
    Ok(PyBytes::new_bound(py, &out).into())
}

/// Multi-threaded manifold staple sweep (rayon over the N independent manifolds — separate
/// in/out chunks, race-free, no locks). Bit-IDENTICAL to `ring_manifold_staple_u8` (both call
/// `staple_one`).
#[pyfunction]
fn ring_manifold_staple_u8_mt(py: Python<'_>, grid: &[u8], lut: &[u8], w: usize, h: usize, n: usize) -> PyResult<Py<PyBytes>> {
    staple_check(grid.len(), lut.len(), w, h, n)?;
    let stride = h * w;
    let mut out = vec![0u8; grid.len()];
    out.par_chunks_mut(stride)
        .zip(grid.par_chunks(stride))
        .for_each(|(o_m, g_m)| staple_one(o_m, g_m, lut, w, h));
    Ok(PyBytes::new_bound(py, &out).into())
}

// ═══════════════════════════════════════════════════════════════════════════════
// GEVHV — the bind→react→measure operator (GEVHV_MATH.md Theorems E/F/F2/G; the judge is
// ringkit/ml/gevhv.py, reproduced BIT-FOR-BIT and gated at host load by kernels/cpu_rust/host.py).
// Phases are u8, staple sums / LUT indices / bind params / ENERGY are u32, the ring wrap is an
// explicit `wrapping_sub` (never an accidental promotion), no float. Manifold: H rows × W cols,
// row-major, flat off = row*w + col.
// ═══════════════════════════════════════════════════════════════════════════════

fn gevhv_dims_ok(n: usize, h: usize, w: usize) -> bool {
    n == h * w
}

// The four ring/energy helpers this section used to own locally (staple5, cdist, has_interior,
// the react/measure inner loops) now live in the verified `gevhv` crate (G:\quantum\research\
// gevhv\backends\gevhv) — the same operator, self-tested independently there. These wrappers
// keep the PyO3 boundary (arg validation, byte<->Z8 conversion) and delegate the computation.

#[inline]
fn to_z8_vec(s: &[u8]) -> Vec<zring::Z8> {
    s.iter().map(|&b| zring::Z8(b)).collect()
}

#[inline]
fn to_z8_lut(s: &[u8]) -> [zring::Z8; 256] {
    let mut out = [zring::Z8::ZERO; 256];
    out.iter_mut().zip(s.iter()).for_each(|(o, &b)| *o = zring::Z8(b));
    out
}

#[inline]
fn from_z8_vec(v: &[zring::Z8]) -> Vec<u8> {
    v.iter().map(|z| z.0).collect()
}

/// react — L[Σ5 mod 256] at interior sites, identity at boundary. `g`,`lut` are borrowed
/// zero-copy from the caller's bytes/bytearray; the result is built as Python `bytes` in one
/// bulk copy (no per-element boxing — this keeps large manifolds fast, D1).
#[pyfunction]
fn gevhv_react(py: Python<'_>, g: &[u8], lut: &[u8], h: usize, w: usize) -> PyResult<Py<PyBytes>> {
    if !gevhv_dims_ok(g.len(), h, w) || lut.len() != 256 {
        return Err(PyValueError::new_err("gevhv_react: bad dims"));
    }
    let gz = to_z8_vec(g);
    let lutz = to_z8_lut(lut);
    let out = gevhv::react(&gz, &lutz, h, w);
    Ok(PyBytes::new_bound(py, &from_z8_vec(&out)).into())
}

/// measure — u32 ENERGY. Raises ValueError outside Theorem G's proven int-exact domain.
/// `gevhv::measure` returns ZPsi (the unbounded winding energy); `.recover()` is the exact same
/// magnitude this used to accumulate directly, cast down under the same bound this function has
/// always enforced (h*w <= 16_777_215 keeps it well inside u32 range).
#[pyfunction]
fn gevhv_measure(g: &[u8], q: &[u8], h: usize, w: usize, interior: bool) -> PyResult<u32> {
    if !gevhv_dims_ok(g.len(), h, w) || !gevhv_dims_ok(q.len(), h, w) {
        return Err(PyValueError::new_err("gevhv_measure: bad dims"));
    }
    if h * w > 16_777_215 {
        return Err(PyValueError::new_err("gevhv_measure: beyond the proven int-exact domain"));
    }
    let gz = to_z8_vec(g);
    let qz = to_z8_vec(q);
    Ok(gevhv::measure(&gz, &qz, h, w, interior).recover() as u32)
}

/// gevhv_scores — the attention/similarity GEMM-role replacement: GEVHV's measure batched over
/// the whole Q×K grid. score[i,j] = -Σ_d cdist(q[i,d], k[j,d]) — ring-L1 similarity, signed ENERGY
/// (never folded); higher (closer to 0) = better match. This is the ring-native stand-in for the
/// Q·Kᵀ gemv, NOT a dot product (GEVHV is transform-and-measure, not bilinear): bind (the position
/// gauge, e.g. RoPE) is applied by the caller at insert, react is identity for plain attention,
/// and this is the measure stage over every (query,key) pair. == ml.attention.scores bit-for-bit.
/// `gevhv::scores` returns +energy (unnegated ZPsi); negated here to match this op's contract.
#[pyfunction]
fn gevhv_scores(q: &[u8], k: &[u8], nq: usize, nk: usize, dim: usize) -> PyResult<Vec<i64>> {
    if q.len() != nq * dim || k.len() != nk * dim {
        return Err(PyValueError::new_err("gevhv_scores: bad dims (q must be nq*dim, k must be nk*dim)"));
    }
    let qz = to_z8_vec(q);
    let kz = to_z8_vec(k);
    Ok(gevhv::scores(&qz, &kz, nq, nk, dim).into_iter().map(|e| -(e.recover() as i64)).collect())
}

/// mul_free — a·b with NO hardware multiply: shift-and-add (repeated doubling). The MPRC thesis
/// op (multipliers are the silicon bottleneck; Theorem C bypasses them). Full product (<= 65025).
#[inline]
fn mul_free(a: u32, b: u32) -> u32 {
    let mut acc = 0u32;
    let (mut aa, mut bb) = (a, b);
    while bb != 0 {
        if bb & 1 == 1 { acc = acc.wrapping_add(aa); }
        aa <<= 1;
        bb >>= 1;
    }
    acc
}

/// gevhv_gemv_radix — Theorem C, the MULTIPLIER-FREE gemv (the gemm/gemv-arithmetic replacement):
/// out[r] = Σ_i 2^i · b_i,  b_i = Σ_{j : bit_i(x[j])=1} w[r,j]. Eight independent bit-plane
/// accumulators, one shift per plane — shifts and adds ONLY, no '*'. Exactly equals Σ_j w[r,j]·x[j]
/// (bit-for-bit == ring_gemv_i64). w signed i32, x u8 activation; i64 out (energy, unfolded).
#[pyfunction]
fn gevhv_gemv_radix(w: Vec<i32>, x: &[u8], m: usize, k: usize) -> PyResult<Vec<i64>> {
    if w.len() != m * k || x.len() != k {
        return Err(PyValueError::new_err("gevhv_gemv_radix: bad dims (w=m*k, x=k)"));
    }
    let mut out = vec![0i64; m];
    out.par_iter_mut().enumerate().for_each(|(r, o)| {
        let wr = &w[r * k..r * k + k];
        let mut b = [0i64; 8];
        for j in 0..k {
            let mut xj = x[j];
            let mut i = 0usize;
            while xj != 0 {
                if xj & 1 == 1 {
                    b[i] += wr[j] as i64;
                }
                xj >>= 1;
                i += 1;
            }
        }
        let mut acc = 0i64;
        for (i, &bi) in b.iter().enumerate() {
            acc += bi << i;
        }
        *o = acc;
    });
    Ok(out)
}

/// gevhv_gemm_arc — the ARC-side ring GEMM C[i,j] = (Σ_k A[i,k]·B[k,j]) & 0xFF, MULTIPLIER-FREE
/// (mul_free shift-add per Theorem C; the mod-256 fold licensed exact by Corollary B1). Pure Rust
/// replacement for ring_gemm.c's shiftadd variant — gives rnp.matmul a native Windows linear map
/// with no hardware '*' and no C toolchain. Bit-for-bit == the rn.mul reference.
#[pyfunction]
fn gevhv_gemm_arc(py: Python<'_>, a: &[u8], b: &[u8], m: usize, k: usize, n: usize) -> PyResult<Py<PyBytes>> {
    if a.len() != m * k || b.len() != k * n {
        return Err(PyValueError::new_err("gevhv_gemm_arc: bad dims (a=m*k, b=k*n)"));
    }
    let mut c = vec![0u8; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(i, crow)| {
        let arow = &a[i * k..i * k + k];
        for (j, cij) in crow.iter_mut().enumerate() {
            let mut acc = 0u32;
            for kk in 0..k {
                acc = acc.wrapping_add(mul_free(arow[kk] as u32, b[kk * n + j] as u32));
            }
            *cij = (acc & 0xFF) as u8;
        }
    });
    Ok(PyBytes::new_bound(py, &c).into())
}

/// react_bound_scalar (Theorem F): absorbed LUT interior + pointwise affine boundary. Raises
/// ValueError on even s (not a unit of Z256) — validated HERE, before crossing into `gevhv`:
/// its own `absorb_lut` enforces the same rule via a panic, which PyO3 would otherwise turn
/// into a PanicException instead of this ValueError, so the check stays in the wrapper to keep
/// the existing exception contract exact.
#[pyfunction]
fn gevhv_react_bound_scalar(py: Python<'_>, g: &[u8], lut: &[u8], s: u32, t: u32, h: usize, w: usize) -> PyResult<Py<PyBytes>> {
    if !gevhv_dims_ok(g.len(), h, w) || lut.len() != 256 {
        return Err(PyValueError::new_err("gevhv_react_bound_scalar: bad dims"));
    }
    if s & 1 == 0 {
        return Err(PyValueError::new_err(format!("gevhv_react_bound_scalar: s={s} is even (zero-divisor)")));
    }
    let gz = to_z8_vec(g);
    let lutz = to_z8_lut(lut);
    let out = gevhv::react_bound_scalar(&gz, &lutz, zring::Z8(s as u8), zring::Z8(t as u8), h, w);
    Ok(PyBytes::new_bound(py, &from_z8_vec(&out)).into())
}

/// offset_field (Theorem F2): c = Σ5(v) mod 256 interior, 0 boundary.
#[pyfunction]
fn gevhv_offset_field(py: Python<'_>, v: &[u8], h: usize, w: usize) -> PyResult<Py<PyBytes>> {
    if !gevhv_dims_ok(v.len(), h, w) {
        return Err(PyValueError::new_err("gevhv_offset_field: bad dims"));
    }
    let vz = to_z8_vec(v);
    let out = gevhv::offset_field(&vz, h, w);
    Ok(PyBytes::new_bound(py, &from_z8_vec(&out)).into())
}

/// react_bound_vector (Theorem F2): bind_vector boundary + L[(Σ5(g)+c) mod 256] interior.
/// `c` is the caller-supplied offset field (shared across a batch).
#[pyfunction]
fn gevhv_react_bound_vector(py: Python<'_>, g: &[u8], lut: &[u8], v: &[u8], c: &[u8], h: usize, w: usize) -> PyResult<Py<PyBytes>> {
    if !gevhv_dims_ok(g.len(), h, w) || lut.len() != 256 || !gevhv_dims_ok(v.len(), h, w) || !gevhv_dims_ok(c.len(), h, w) {
        return Err(PyValueError::new_err("gevhv_react_bound_vector: bad dims"));
    }
    let gz = to_z8_vec(g);
    let lutz = to_z8_lut(lut);
    let vz = to_z8_vec(v);
    let cz = to_z8_vec(c);
    let out = gevhv::react_bound_vector(&gz, &lutz, &vz, &cz, h, w);
    Ok(PyBytes::new_bound(py, &from_z8_vec(&out)).into())
}

// ═══════════════════════════════════════════════════════════════════════════════
// SU(256) LATTICE GAUGE ENGINE (kernels/mprc/lattice, SPEC-014). A bit-for-bit port of the
// original C/unsafe-Rust gauge kernels, now SAFE Rust: the parallel path splits the mutable
// output buffer into disjoint k-slabs via `split_at_mut` and reads neighbours from an immutable
// SNAPSHOT taken before the pass. This is sound because of the checkerboard-parity invariant: a
// single call only ever WRITES sites of one parity, and every neighbour of a site (6-neighbor
// stencil) is the OTHER parity, so no site written this call is ever read as a neighbour this
// call — reading a pre-pass snapshot is therefore bit-identical to reading the live (in-place)
// buffer, and it lets every slab's writes be provably disjoint (no raw pointers, no unsafe Send).
// D9: the Rust path is gated at host load by `_rust_selftest` against the pure-Python reference
// (`_py_plaquette` / `_py_sweep` / `_py_sweep_rng`) in kernels/mprc/lattice/host.py.
// ═══════════════════════════════════════════════════════════════════════════════

/// circular (ring L1) distance min(|a-b|, 256-|a-b|) over Z256 — the U(1) local action term.
#[inline]
fn cdist(a: u8, b: u8) -> i64 {
    let d = a.wrapping_sub(b) as i64; // 0..255, uint8 wraparound subtraction (== (a-b)&0xFF)
    let e = 256 - d;
    if d < e { d } else { e }
}

/// Counter-based per-node RNG (rk_mix32): bit-for-bit identical to the Python reference `_rand32`
/// in kernels/mprc/lattice/host.py (lowbias32 mix, wrapping u32 math).
#[inline]
fn rk_mix32(seed: u32, sweep: u32, idx: u32) -> u32 {
    let mut x = idx.wrapping_add(sweep.wrapping_add(1).wrapping_mul(0x9E3779B9));
    x ^= seed.wrapping_mul(0x85EBCA6B);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846CA68B);
    x ^= x >> 16;
    x
}

/// Static k-slab split of [lo,hi) into `nthreads` disjoint contiguous ranges (last gets the
/// ragged remainder).
fn k_bounds(lo: i64, hi: i64, nthreads: i32) -> Vec<(i64, i64)> {
    let span = hi - lo;
    let mut nt = (nthreads.max(1)) as i64;
    if nt > span {
        nt = span.max(1);
    }
    let chunk = span / nt;
    let rem = span % nt;
    let mut v = Vec::with_capacity(nt as usize);
    let mut k = lo;
    for t in 0..nt {
        let len = chunk + if t < rem { 1 } else { 0 };
        v.push((k, k + len));
        k += len;
    }
    v
}

/// One k-slab [k0,k1) of the Wilson plaquette action, writing into `e_out` at LOCAL offset
/// `out_k0` (== k0 for the full-buffer/single-thread call; == 0 for a slab that itself begins at
/// global k0). Reads `g` (never written) at GLOBAL offsets — safe to share across slabs since it
/// is a plain immutable borrow.
fn lat_slab(e_out: &mut [u8], g: &[u8], w: i64, h: i64, k0: i64, k1: i64, out_k0: i64) {
    let sk = w * h;
    for k in k0..k1 {
        let k_out = k - k0 + out_k0;
        for j in 1..h - 1 {
            let base = k * sk + j * w;
            let base_out = k_out * sk + j * w;
            for i in 1..w - 1 {
                let c = (base + i) as usize;
                let co = (base_out + i) as usize;
                let pos = g[c].wrapping_add(g[c + 1]); // right + up
                let neg = g[c + w as usize].wrapping_add(g[c - 1]); // left + down
                e_out[co] = pos.wrapping_sub(neg);
            }
        }
    }
}

fn plaquette_core(g: &[u8], w: i64, h: i64, d: i64) -> Vec<u8> {
    let n = (w * h * d) as usize;
    let mut e = vec![0u8; n];
    lat_slab(&mut e, g, w, h, 1, d - 1, 1);
    e
}

fn plaquette_blocked_core(g: &[u8], w: i64, h: i64, d: i64) -> Vec<u8> {
    let n = (w * h * d) as usize;
    let mut e = vec![0u8; n];
    let mut kb = 1i64;
    while kb < d - 1 {
        let kmax = if kb + 64 < d - 1 { kb + 64 } else { d - 1 };
        lat_slab(&mut e, g, w, h, kb, kmax, kb);
        kb += 64;
    }
    e
}

/// Split `e`'s CORE region [lo*sk, hi*sk) (global byte offsets) into one disjoint mutable slab
/// per `bounds` entry, in order. The head [0, lo*sk) and tail [hi*sk, len) are left untouched
/// (boundary layers this kernel never writes) and simply dropped from the split.
fn split_core_mut<'a>(e: &'a mut [u8], bounds: &[(i64, i64)], sk: i64) -> Vec<&'a mut [u8]> {
    let lo = bounds[0].0;
    let hi = bounds[bounds.len() - 1].1;
    let (_head, mid_and_tail) = e.split_at_mut((lo * sk) as usize);
    let mid_len = ((hi - lo) * sk) as usize;
    let (mid, _tail) = mid_and_tail.split_at_mut(mid_len);
    let mut rest = mid;
    let mut slabs = Vec::with_capacity(bounds.len());
    for &(k0, k1) in bounds {
        let take = ((k1 - k0) * sk) as usize;
        let (chunk, remainder) = rest.split_at_mut(take);
        slabs.push(chunk);
        rest = remainder;
    }
    slabs
}

fn plaquette_mt_core(g: &[u8], w: i64, h: i64, d: i64, nthreads: i32) -> Vec<u8> {
    let n = (w * h * d) as usize;
    let mut e = vec![0u8; n];
    let bounds = k_bounds(1, d - 1, nthreads);
    if bounds.len() <= 1 {
        if let Some(&(k0, k1)) = bounds.first() {
            lat_slab(&mut e, g, w, h, k0, k1, k0);
        }
        return e;
    }
    let sk = w * h;
    {
        let slabs = split_core_mut(&mut e, &bounds, sk);
        slabs
            .into_par_iter()
            .zip(bounds.par_iter())
            .for_each(|(slab, &(k0, k1))| lat_slab(slab, g, w, h, k0, k1, 0));
    }
    e
}

/// One checkerboard-parity pass over k in [k0,k1) (GLOBAL indices), reading neighbours from
/// `read` (the snapshot for the mt path, or the live grid itself for the sequential path — both
/// sound, see the module doc) and writing into `write` at LOCAL offset `out_k0`. `rng`=false
/// reads prop/chance arrays (GLOBAL indices, read-only, safe to share); `rng`=true derives them
/// from rk_mix32(seed,sweep,c).
#[allow(clippy::too_many_arguments)]
fn sweep_range(
    write: &mut [u8], read: &[u8], prop: Option<&[u8]>, chance: Option<&[u8]>, lut: &[u8],
    w: i64, h: i64, parity: i32, k0: i64, k1: i64, out_k0: i64,
    rng: bool, seed: u32, sweep_idx: u32,
) {
    let sk = w * h;
    for k in k0..k1 {
        let k_out = k - k0 + out_k0;
        for j in 1..h - 1 {
            let base = k * sk + j * w;
            let base_out = k_out * sk + j * w;
            for i in 1..w - 1 {
                if (((i + j + k) & 1) as i32) != parity {
                    continue;
                }
                let c = (base + i) as usize;
                let co = (base_out + i) as usize;
                let (pr, ch): (u8, u8) = if rng {
                    let x = rk_mix32(seed, sweep_idx, c as u32);
                    ((x & 0xFF) as u8, ((x >> 8) & 0xFF) as u8)
                } else {
                    (prop.unwrap()[c], chance.unwrap()[c])
                };
                let old = read[c];
                let nv = old.wrapping_add(pr);
                let r = read[c + 1];
                let l = read[c - 1];
                let u = read[c + w as usize];
                let dn = read[c - w as usize];
                let f = read[c + sk as usize];
                let bk = read[c - sk as usize];
                let so = cdist(old, r) + cdist(old, l) + cdist(old, u) + cdist(old, dn) + cdist(old, f) + cdist(old, bk);
                let sn = cdist(nv, r) + cdist(nv, l) + cdist(nv, u) + cdist(nv, dn) + cdist(nv, f) + cdist(nv, bk);
                let d_s = sn - so;
                let lut_idx = if d_s > 255 { 255 } else { d_s.max(0) };
                let accept = d_s <= 0 || (ch as i64) < (lut[lut_idx as usize] as i64);
                write[co] = if accept { nv } else { old };
            }
        }
    }
}

/// Sequential (single-thread) sweep: reads and writes the same live buffer. Sound in plain safe
/// Rust because reads/writes are done element-by-element (u8 is Copy) — never a live overlapping
/// borrow — and semantically identical to reading a pre-pass snapshot (see module doc): a
/// neighbour is always the OTHER checkerboard parity, so it is never written earlier in this
/// same call.
#[allow(clippy::too_many_arguments)]
fn sweep_seq(
    grid: &mut [u8], prop: Option<&[u8]>, chance: Option<&[u8]>, lut: &[u8],
    w: i64, h: i64, parity: i32, k0: i64, k1: i64, rng: bool, seed: u32, sweep_idx: u32,
) {
    // SAFETY/soundness note: we can't pass `grid` as both `read` and `write` to `sweep_range`
    // (that would need two live borrows of the same slice). Instead take a private snapshot only
    // when nthreads==1 would otherwise need aliasing — here we sidestep it entirely by processing
    // one site at a time with plain indexing (Copy reads then a write), which is what
    // `sweep_range` already does; we just need `read`/`write` to be the SAME slice. Rust allows
    // this via a single `&mut [u8]` and indexing (`grid[c]`), so route through a tiny local loop
    // instead of `sweep_range`.
    let sk = w * h;
    for k in k0..k1 {
        for j in 1..h - 1 {
            let base = k * sk + j * w;
            for i in 1..w - 1 {
                if (((i + j + k) & 1) as i32) != parity {
                    continue;
                }
                let c = (base + i) as usize;
                let (pr, ch): (u8, u8) = if rng {
                    let x = rk_mix32(seed, sweep_idx, c as u32);
                    ((x & 0xFF) as u8, ((x >> 8) & 0xFF) as u8)
                } else {
                    (prop.unwrap()[c], chance.unwrap()[c])
                };
                let old = grid[c];
                let nv = old.wrapping_add(pr);
                let r = grid[c + 1];
                let l = grid[c - 1];
                let u = grid[c + w as usize];
                let dn = grid[c - w as usize];
                let f = grid[c + sk as usize];
                let bk = grid[c - sk as usize];
                let so = cdist(old, r) + cdist(old, l) + cdist(old, u) + cdist(old, dn) + cdist(old, f) + cdist(old, bk);
                let sn = cdist(nv, r) + cdist(nv, l) + cdist(nv, u) + cdist(nv, dn) + cdist(nv, f) + cdist(nv, bk);
                let d_s = sn - so;
                let lut_idx = if d_s > 255 { 255 } else { d_s.max(0) };
                let accept = d_s <= 0 || (ch as i64) < (lut[lut_idx as usize] as i64);
                grid[c] = if accept { nv } else { old };
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sweep_dispatch(
    mut grid: Vec<u8>, prop: Option<&[u8]>, chance: Option<&[u8]>, lut: &[u8],
    w: i64, h: i64, d: i64, parity: i32, nthreads: i32, rng: bool, seed: u32, sweep_idx: u32,
) -> Vec<u8> {
    let bounds = k_bounds(1, d - 1, nthreads);
    if bounds.len() <= 1 {
        if let Some(&(k0, k1)) = bounds.first() {
            sweep_seq(&mut grid, prop, chance, lut, w, h, parity, k0, k1, rng, seed, sweep_idx);
        }
        return grid;
    }
    // Multi-slab: take an immutable snapshot BEFORE mutating (sound — see module doc), then
    // split the live grid into disjoint per-thread mutable slabs and write only into those.
    let snapshot = grid.clone();
    let sk = w * h;
    {
        let slabs = split_core_mut(&mut grid, &bounds, sk);
        slabs.into_par_iter().zip(bounds.par_iter()).for_each(|(slab, &(k0, k1))| {
            sweep_range(slab, &snapshot, prop, chance, lut, w, h, parity, k0, k1, 0, rng, seed, sweep_idx);
        });
    }
    grid
}

// The `Vec<u8>` extraction/return path boxes every element as a Python int (measured ~8ns/elem
// each way) — fine for small buffers, but a 128x113x256 lattice (3.7M nodes) blows well past the
// D9 throughput floor tests/test_gauge.py asserts. `&[u8]` borrows the caller's bytes/bytearray
// zero-copy on the way in; `PyBytes::new_bound` builds the return with one bulk memcpy on the
// way out (measured ~40x faster round-trip on the full 128x113x256 lattice) — still pure PyO3,
// still safe Rust, just the fast idiomatic pyo3 byte-buffer path instead of the generic Vec<T>
// per-element one.

#[pyfunction]
fn plaquette(py: Python<'_>, g: &[u8], w: i64, h: i64, d: i64) -> PyResult<Py<PyBytes>> {
    if g.len() != (w * h * d) as usize {
        return Err(PyValueError::new_err("plaquette: len(g) != w*h*d"));
    }
    Ok(PyBytes::new_bound(py, &plaquette_core(g, w, h, d)).into())
}

#[pyfunction]
fn plaquette_blocked(py: Python<'_>, g: &[u8], w: i64, h: i64, d: i64) -> PyResult<Py<PyBytes>> {
    if g.len() != (w * h * d) as usize {
        return Err(PyValueError::new_err("plaquette_blocked: len(g) != w*h*d"));
    }
    Ok(PyBytes::new_bound(py, &plaquette_blocked_core(g, w, h, d)).into())
}

#[pyfunction]
fn plaquette_mt(py: Python<'_>, g: &[u8], w: i64, h: i64, d: i64, nthreads: i32) -> PyResult<Py<PyBytes>> {
    if g.len() != (w * h * d) as usize {
        return Err(PyValueError::new_err("plaquette_mt: len(g) != w*h*d"));
    }
    Ok(PyBytes::new_bound(py, &plaquette_mt_core(g, w, h, d, nthreads)).into())
}

#[pyfunction]
fn metropolis_sweep(py: Python<'_>, grid: &[u8], prop: &[u8], chance: &[u8], lut: &[u8], w: i64, h: i64, d: i64, parity: i32) -> PyResult<Py<PyBytes>> {
    if lut.len() != 256 || prop.len() != grid.len() || chance.len() != grid.len() {
        return Err(PyValueError::new_err("metropolis_sweep: bad dims"));
    }
    let out = sweep_dispatch(grid.to_vec(), Some(prop), Some(chance), lut, w, h, d, parity, 1, false, 0, 0);
    Ok(PyBytes::new_bound(py, &out).into())
}

#[pyfunction]
fn metropolis_sweep_rng(py: Python<'_>, grid: &[u8], seed: u32, sweep: u32, lut: &[u8], w: i64, h: i64, d: i64, parity: i32) -> PyResult<Py<PyBytes>> {
    if lut.len() != 256 {
        return Err(PyValueError::new_err("metropolis_sweep_rng: lut must have 256 entries"));
    }
    let out = sweep_dispatch(grid.to_vec(), None, None, lut, w, h, d, parity, 1, true, seed, sweep);
    Ok(PyBytes::new_bound(py, &out).into())
}

#[pyfunction]
fn metropolis_sweep_mt(py: Python<'_>, grid: &[u8], prop: &[u8], chance: &[u8], lut: &[u8], w: i64, h: i64, d: i64, parity: i32, nthreads: i32) -> PyResult<Py<PyBytes>> {
    if lut.len() != 256 || prop.len() != grid.len() || chance.len() != grid.len() {
        return Err(PyValueError::new_err("metropolis_sweep_mt: bad dims"));
    }
    let out = sweep_dispatch(grid.to_vec(), Some(prop), Some(chance), lut, w, h, d, parity, nthreads, false, 0, 0);
    Ok(PyBytes::new_bound(py, &out).into())
}

#[pyfunction]
fn metropolis_sweep_rng_mt(py: Python<'_>, grid: &[u8], seed: u32, sweep: u32, lut: &[u8], w: i64, h: i64, d: i64, parity: i32, nthreads: i32) -> PyResult<Py<PyBytes>> {
    if lut.len() != 256 {
        return Err(PyValueError::new_err("metropolis_sweep_rng_mt: lut must have 256 entries"));
    }
    let out = sweep_dispatch(grid.to_vec(), None, None, lut, w, h, d, parity, nthreads, true, seed, sweep);
    Ok(PyBytes::new_bound(py, &out).into())
}

/// Σ neighbor ring-distances (mean_action's numerator/denominator) — MEASUREMENT reduction, host
/// does the final (float) divide.
#[pyfunction]
fn action_sums(grid: &[u8], w: i64, h: i64, d: i64) -> PyResult<(i64, i64)> {
    if grid.len() != (w * h * d) as usize {
        return Err(PyValueError::new_err("action_sums: len(grid) != w*h*d"));
    }
    let sk = w * h;
    let mut tot: i64 = 0;
    let mut n: i64 = 0;
    for k in 1..d - 1 {
        for j in 1..h - 1 {
            let base = k * sk + j * w;
            for i in 1..w - 1 {
                let c = (base + i) as usize;
                let gc = grid[c];
                tot += cdist(gc, grid[c + 1]) + cdist(gc, grid[c + w as usize]) + cdist(gc, grid[c + sk as usize]);
                n += 3;
            }
        }
    }
    Ok((tot, n))
}

/// Σ (128 - ring-distance(grid[c], grid[c+R])) over the i-axis — correlation's reduction.
#[pyfunction]
fn correlation_sums(grid: &[u8], r: i64, w: i64, h: i64, d: i64) -> PyResult<(i64, i64)> {
    if grid.len() != (w * h * d) as usize {
        return Err(PyValueError::new_err("correlation_sums: len(grid) != w*h*d"));
    }
    let sk = w * h;
    let mut tot: i64 = 0;
    let mut n: i64 = 0;
    for k in 1..d - 1 {
        for j in 1..h - 1 {
            let base = k * sk + j * w;
            for i in 1..w - 1 - r {
                let c = (base + i) as usize;
                tot += 128 - cdist(grid[c], grid[c + r as usize]);
                n += 1;
            }
        }
    }
    Ok((tot, n))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Module registration — EXACTLY the op surface the Python hosts (cpu_rust/host.py,
// kernels/backend/__init__.py, kernels/mprc/lattice/host.py) bind by name.
// ═══════════════════════════════════════════════════════════════════════════════

#[pymodule]
fn ring_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ring_gemm_i64, m)?)?;
    m.add_function(wrap_pyfunction!(ring_gemv_i64, m)?)?;
    m.add_function(wrap_pyfunction!(ring_sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(ring_exp, m)?)?;
    m.add_function(wrap_pyfunction!(ring_rmsnorm, m)?)?;
    m.add_function(wrap_pyfunction!(ring_ew_q16, m)?)?;
    m.add_function(wrap_pyfunction!(ring_escale, m)?)?;
    m.add_function(wrap_pyfunction!(ring_colsum, m)?)?;
    m.add_function(wrap_pyfunction!(ring_diffuse, m)?)?;
    m.add_function(wrap_pyfunction!(ring_relu, m)?)?;
    m.add_function(wrap_pyfunction!(ring_gather, m)?)?;
    m.add_function(wrap_pyfunction!(ring_mul, m)?)?;
    m.add_function(wrap_pyfunction!(ring_add, m)?)?;
    m.add_function(wrap_pyfunction!(ring_sub, m)?)?;
    m.add_function(wrap_pyfunction!(ring_rust_probe, m)?)?;
    m.add_function(wrap_pyfunction!(ring_manifold_staple_u8, m)?)?;
    m.add_function(wrap_pyfunction!(ring_manifold_staple_u8_mt, m)?)?;
    m.add_function(wrap_pyfunction!(gevhv_react, m)?)?;
    m.add_function(wrap_pyfunction!(gevhv_measure, m)?)?;
    m.add_function(wrap_pyfunction!(gevhv_scores, m)?)?;
    m.add_function(wrap_pyfunction!(gevhv_gemv_radix, m)?)?;
    m.add_function(wrap_pyfunction!(gevhv_gemm_arc, m)?)?;
    m.add_function(wrap_pyfunction!(gevhv_react_bound_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(gevhv_offset_field, m)?)?;
    m.add_function(wrap_pyfunction!(gevhv_react_bound_vector, m)?)?;
    m.add_function(wrap_pyfunction!(plaquette, m)?)?;
    m.add_function(wrap_pyfunction!(plaquette_blocked, m)?)?;
    m.add_function(wrap_pyfunction!(plaquette_mt, m)?)?;
    m.add_function(wrap_pyfunction!(metropolis_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(metropolis_sweep_rng, m)?)?;
    m.add_function(wrap_pyfunction!(metropolis_sweep_mt, m)?)?;
    m.add_function(wrap_pyfunction!(metropolis_sweep_rng_mt, m)?)?;
    m.add_function(wrap_pyfunction!(action_sums, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_sums, m)?)?;
    Ok(())
}
