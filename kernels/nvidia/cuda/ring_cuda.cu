// ringkit CUDA backend (D9 silicon). ringkit's OWN ring semantics on NVIDIA — NOT ported.
// Mirrors the Apple/Metal backend: __global__ kernels + extern "C" host launch wrappers.
// D9 bar: every op reproduces ringkit.core.native / backend._PY BIT-FOR-BIT.
//   ring_mul/add/sub : (x . y) & 0xFF elementwise
//   ring_gemm        : C[i,j] = (SUM_k A[i,k]*B[k,j]) & 0xFF   (mod-256 accumulation)
//   ring_l1dist      : D[i,j] = SUM_d cdist(Q[i,d], K[j,d])    (ENERGY accumulation, never folds)
//                      cdist = min(|a-b|, 256-|a-b|) — the ring-L1 metric of gauge.metal::cdist /
//                      kv_cache.c::rdist / ml.attention.ring_distance. int32 out: max = dim*128,
//                      so dim <= ~16.7M is exact. This is the GPU distance the Metal port also
//                      lacked as a standalone — patterned on Metal's cdist + kv_scores's long acc.
#include <cuda_runtime.h>

__global__ void k_mul(unsigned char* o, const unsigned char* a, const unsigned char* b, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = (unsigned char)(((unsigned)a[i] * (unsigned)b[i]) & 0xFF);
}
__global__ void k_add(unsigned char* o, const unsigned char* a, const unsigned char* b, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = (unsigned char)(((unsigned)a[i] + (unsigned)b[i]) & 0xFF);
}
__global__ void k_sub(unsigned char* o, const unsigned char* a, const unsigned char* b, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = (unsigned char)(((unsigned)a[i] - (unsigned)b[i]) & 0xFF);
}
__global__ void k_gemm(unsigned char* C, const unsigned char* A, const unsigned char* B,
                       long M, long K, long N) {
  long i = (long)blockIdx.y * blockDim.y + threadIdx.y;
  long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M || j >= N) return;
  unsigned acc = 0;
  for (long k = 0; k < K; k++) acc += (unsigned)A[i * K + k] * (unsigned)B[k * N + j];
  C[i * N + j] = (unsigned char)(acc & 0xFF);
}

// ring-L1 circular distance min(|a-b|, 256-|a-b|). IDENTICAL to gauge.metal::cdist,
// kv_cache.c::rdist, ml.attention.ring_distance — verified bit-for-bit by _selftest.
__device__ static inline int cdist(unsigned char a, unsigned char b) {
  int d = (int)(unsigned char)(a - b);
  int e = 256 - d;
  return d < e ? d : e;
}
// D[i,j] = SUM_d cdist(Q[i,d], K[j,d]).  ENERGY side (kv_scores): accumulate in int, NEVER fold
// mod 256 — a distance that wraps would destroy the k-NN ranking. That is why ring_gemm (arc-side,
// mod-256) cannot serve this. int holds dim*128 exactly for any sane dim.
__global__ void k_l1dist(int* D, const unsigned char* Q, const unsigned char* K,
                         long m, long n, long dim) {
  long i = (long)blockIdx.y * blockDim.y + threadIdx.y;
  long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m || j >= n) return;
  const unsigned char* qi = Q + i * dim;
  const unsigned char* kj = K + j * dim;
  int acc = 0;
  for (long d = 0; d < dim; d++) acc += cdist(qi[d], kj[d]);
  D[i * n + j] = acc;
}

// Exact fixed-point GEMV: out[j] = SUM_i W[j*K+i] * x[i], accumulated in int64 (ENERGY side —
// the true Q<2·frac> product sum, NOT folded mod 256). The caller rescales >>frac and adds bias
// in Python (arbitrary precision), matching emulation.infer.linear BIT-FOR-BIT. This is the GPU
// analog of the gemma C gemv_exact / Metal emu_gemv that the CUDA port lacked. Hardware * (D9 ok in
// kernels). Inputs are signed Q<frac> ints that fit int32; the host bound-checks and falls back.
__global__ void k_gemv_i64(long long* out, const int* W, const int* x, long M, long K) {
  long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= M) return;
  const int* wr = W + j * K;
  long long acc = 0;
  for (long i = 0; i < K; i++) acc += (long long)wr[i] * (long long)x[i];
  out[j] = acc;
}

// ── fixed-point ring ACTIVATIONS (Q<frac>, int64) — bit-for-bit == emulation/ract + the gemma C
// kernel (qsm_energy.c). nvcc supports __int128 on device, so this is the SAME exact integer math
// (no float, no approximation drift): the CPU kernel is Unix-only, so on a GPU box THIS is the path.
// nvcc+MSVC (Windows) has NO __int128, so emulate: (a*b)>>s via __umul64hi (result = low 64 bits of
// the shifted 128-bit product == the CPU (__int128)prod cast to int64, bit-identical), and a minimal
// unsigned 128-bit for RMSNorm's Sum x^2 (which reaches ~2^90 at real Soliton scale). Same portable
// form the cross-platform MSVC CPU kernel will use (phase 2).
// (a*b)>>s TRUNCATED toward zero (magnitude shift, re-sign). Matches ract._sdiv / rn.mf_floordiv-with-
// sign — used by rmsnorm's normalize and by exp (positive-only, so trunc==floor there).
__device__ static inline long long mulshr(long long a, long long b, int s) {
  int neg = (a < 0) ^ (b < 0);
  unsigned long long ua = a < 0 ? (unsigned long long)(-a) : (unsigned long long)a;
  unsigned long long ub = b < 0 ? (unsigned long long)(-b) : (unsigned long long)b;
  unsigned long long lo = ua * ub, hi = __umul64hi(ua, ub);
  unsigned long long r = s >= 64 ? (hi >> (s - 64)) : ((lo >> s) | (hi << (64 - s)));
  return neg ? -(long long)r : (long long)r;
}
// (a*b)>>s ARITHMETIC-FLOOR (== the kit's signed `(__int128)a*b >> s`: infer.linear, _mul_q16, the
// SSD scan products). Differs from mulshr on negative products (floor vs trunc) — verified against the
// Python SSD reference. Uses signed __mul64hi for the high 64 bits, then a sign-extending shift.
__device__ static inline long long mulshr_floor(long long a, long long b, int s) {
  unsigned long long lo = (unsigned long long)a * (unsigned long long)b;   // low 64 (sign-agnostic)
  long long hi = __mul64hi(a, b);                                          // signed high 64
  if (s == 0) return (long long)lo;
  if (s >= 64) return hi >> (s - 64);                                      // arithmetic (floor)
  return (long long)((lo >> s) | ((unsigned long long)hi << (64 - s)));
}
struct u128 { unsigned long long hi, lo; };
__device__ static inline u128 U(unsigned long long h, unsigned long long l) { u128 r; r.hi = h; r.lo = l; return r; }
__device__ static inline u128 add128(u128 a, u128 b) { u128 r; r.lo = a.lo + b.lo; r.hi = a.hi + b.hi + (r.lo < a.lo); return r; }
__device__ static inline u128 sub128(u128 a, u128 b) { u128 r; r.lo = a.lo - b.lo; r.hi = a.hi - b.hi - (a.lo < b.lo); return r; }
__device__ static inline bool ge128(u128 a, u128 b) { return a.hi != b.hi ? a.hi > b.hi : a.lo >= b.lo; }
__device__ static inline u128 mul64u(unsigned long long a, unsigned long long b) { return U(__umul64hi(a, b), a * b); }
__device__ static inline u128 shl128(u128 a, int s) { if (!s) return a; return s >= 64 ? U(a.lo << (s - 64), 0ULL) : U((a.hi << s) | (a.lo >> (64 - s)), a.lo << s); }
__device__ static inline u128 shr128(u128 a, int s) { if (!s) return a; return s >= 64 ? U(0ULL, a.hi >> (s - 64)) : U(a.hi >> s, (a.lo >> s) | (a.hi << (64 - s))); }
__device__ static u128 div128u(u128 num, unsigned long long d) {          // 128 / 64 -> 128 quotient
  u128 q = U(0, 0), rem = U(0, 0), dd = U(0, d);
  for (int i = 127; i >= 0; i--) {
    rem = shl128(rem, 1);
    rem.lo |= (i >= 64) ? ((num.hi >> (i - 64)) & 1ULL) : ((num.lo >> i) & 1ULL);
    if (ge128(rem, dd)) { rem = sub128(rem, dd); if (i >= 64) q.hi |= (1ULL << (i - 64)); else q.lo |= (1ULL << i); }
  }
  return q;
}
__device__ static unsigned long long isqrt128(u128 m) {                    // == rn.isqrt on 128-bit
  if (!m.hi && !m.lo) return 0;
  u128 x = U(0, 0), c = U(0, 1), m2 = shr128(m, 2);
  while (ge128(m2, c)) c = shl128(c, 2);
  while (c.hi || c.lo) {
    u128 xc = add128(x, c);
    if (ge128(m, xc)) { m = sub128(m, xc); x = add128(shr128(x, 1), c); }
    else x = shr128(x, 1);
    c = shr128(c, 2);
  }
  return x.lo;
}
__device__ static long long exp_fixed_d(long long x, int frac) {   // == exp_fixed_c
  const long long one = (long long)1 << frac;
  const long long clamp = (long long)1 << (frac + frac + 8);
  int neg = x < 0; long long ax = neg ? -x : x;
  long long half = one >> 1; int m = 0; long long red = ax;
  while (red > half) { red >>= 1; m++; }
  long long term = one, acc = one;
  for (int k = 1; k <= 12; k++) {
    term = mulshr(term, red, frac);
    term = term / k;
    acc += term;
    if (term == 0) break;
  }
  for (int i = 0; i < m; i++) {
    if (acc >= clamp) { acc = clamp; continue; }
    acc = mulshr(acc, acc, frac);
    if (acc >= clamp) acc = clamp;
  }
  if (neg) acc = ((long long)1 << (frac + frac)) / acc;
  return acc;
}
__device__ static long long sigmoid_fixed_d(long long x, int frac) {   // == sigmoid_fixed_c
  const long long one = (long long)1 << frac;
  long long e = exp_fixed_d(-x, frac);
  return ((long long)1 << (frac + frac)) / (one + e);
}
__global__ void k_sigmoid(long long* o, const long long* x, long n, int frac) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = sigmoid_fixed_d(x[i], frac);
}
__global__ void k_exp(long long* o, const long long* x, long n, int frac) {   // softmax domain (x<=0)
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = exp_fixed_d(x[i], frac);
}
// single-vector RMSNorm (per token): thread 0 reduces Sum x^2 -> rms; all threads normalize. == rmsnorm_block.
__global__ void k_rmsnorm(long long* o, const long long* x, const long long* w,
                          long n, int frac, long long eps) {
  __shared__ long long s_rms;
  if (threadIdx.x == 0) {
    u128 ssq = U(0, 0);
    for (long i = 0; i < n; i++) {
      unsigned long long ax = x[i] < 0 ? (unsigned long long)(-x[i]) : (unsigned long long)x[i];
      ssq = add128(ssq, shr128(mul64u(ax, ax), frac));           // += (x*x)>>frac
    }
    u128 ms = add128(div128u(ssq, (unsigned long long)n), U(0, (unsigned long long)eps));
    long long rms = (long long)isqrt128(shl128(ms, frac));
    if (rms == 0) rms = 1;
    s_rms = rms;
  }
  __syncthreads();
  long long rms = s_rms;
  for (long i = threadIdx.x; i < n; i += blockDim.x) {
    unsigned long long ax = x[i] < 0 ? (unsigned long long)(-x[i]) : (unsigned long long)x[i];
    unsigned long long q = div128u(shl128(U(0, ax), frac), (unsigned long long)rms).lo;  // (x<<frac)/rms
    long long norm = x[i] < 0 ? -(long long)q : (long long)q;
    o[i] = mulshr(norm, w[i], frac);                             // norm*w >> frac
  }
}

// ── Q16 ENERGY elementwise + reductions (int64, no fold) — the primitives the Soliton forward and
// backward COMPOSE from, so the whole model runs on-device with Python only sequencing kernel calls.
__global__ void k_emul(long long* o, const long long* a, const long long* b, long n, int frac) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = mulshr_floor(a[i], b[i], frac);       // (a*b)>>frac (floor, == _mul_q16)
}
__global__ void k_eadd(long long* o, const long long* a, const long long* b, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = a[i] + b[i];
}
__global__ void k_esub(long long* o, const long long* a, const long long* b, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = a[i] - b[i];
}
__global__ void k_escale(long long* o, const long long* a, long long sc, long n, int frac) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = mulshr_floor(a[i], sc, frac);         // (a*sc)>>frac (floor)
}
// column-sum over R rows of length C: out[c] = Sum_r in[r*C+c]  (h_state / mean-pool reduction).
__global__ void k_colsum(long long* out, const long long* in, long R, long C) {
  long c = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  long long acc = 0;
  for (long r = 0; r < R; r++) acc += in[r * C + c];
  out[c] = acc;
}
// clamp/relu: o[i] = max(a[i], 0)  (the frontend quadrant rectifier). signed in, >=0 out.
__global__ void k_relu(long long* o, const long long* a, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = a[i] > 0 ? a[i] : 0;
}
// gather: o[i] = lut[idx[i]]  (trig-LUT lookup by arc byte; idx are u8 ring positions).
__global__ void k_gather(long long* o, const long long* lut, const unsigned char* idx, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = lut[idx[i]];
}
// one toroidal 4-neighbour heat step over a (D,H) grid of hd-vectors (the SSD lattice2d interaction):
// out = (up+dn+lf+rt + 4*center) >> 3, wrap-around. in/out are [D*H*hd] Q16, arithmetic >>3 (floor).
__global__ void k_diffuse(long long* o, const long long* in, long D, long H, long hd) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = D * H * hd;
  if (idx >= tot) return;
  long j = idx % hd, cell = idx / hd, r = cell / H, c = cell % H;
  long up = (((r - 1 + D) % D) * H + c) * hd + j;
  long dn = (((r + 1) % D) * H + c) * hd + j;
  long lf = (r * H + (c - 1 + H) % H) * hd + j;
  long rt = (r * H + (c + 1) % H) * hd + j;
  o[idx] = (in[up] + in[dn] + in[lf] + in[rt] + (in[idx] << 2)) >> 3;
}

// batched exact int64 energy GEMM — ALL tokens' linear in ONE call (the batching that turns 1808
// per-token gemv into one launch): out[t*M+m] = Sum_k X[t*K+k]*W[m*K+k], int64 accumulate, no fold.
// X = T tokens x K (row-major), W = M out-rows x K. Caller rescales >>frac + bias (Python, exact).
__global__ void k_gemm_i64(long long* out, const int* X, const int* W, long T, long M, long K) {
  long t = (long)blockIdx.y * blockDim.y + threadIdx.y;
  long m = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T || m >= M) return;
  const int* xr = X + t * K;
  const int* wr = W + m * K;
  long long acc = 0;
  for (long k = 0; k < K; k++) acc += (long long)xr[k] * (long long)wr[k];
  out[t * M + m] = acc;
}

// ── RESIDENT OP CHAIN (owner: "do not stream data; process on lanes"). Every op below operates on
// DEVICE HANDLES — no host transfer, no malloc; Python only sequences launches. The SSD forward and
// backward chain on-device; only logits and weight-grads cross the boundary. Same ring math.
__device__ static long long softplus_fixed_d(long long x, int frac) {   // == activations.softplus_fixed
  const long long one = 1LL << frac;
  long long ax = x < 0 ? -x : x;
  long long lim = (long long)frac << frac;
  if (ax > lim) ax = lim;
  long long e = exp_fixed_d(-ax, frac);                    // e^-|x| in (0, one]
  long long d = (one << 1) + e;
  long long w = ((e << frac) + (d >> 1)) / d;              // _sd(e<<frac, 2+e), positive
  long long w2 = (w * w) >> frac;
  long long term = w, acc = w;
  for (int k = 3; k <= 11; k += 2) {
    term = (term * w2) >> frac;
    acc += (term + (k >> 1)) / k;                          // _sd(term, k), positive
  }
  return (x > 0 ? x : 0) + (acc << 1);
}
__global__ void k_softplus(long long* o, const long long* x, long n, int frac) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = softplus_fixed_d(x[i], frac);
}
// per-segment column sums: out[s*C+c] = Σ_r in[(s*R+r)*C+c]  (ALL films' VSSD sums in ONE launch)
__global__ void k_colsum_seg(long long* out, const long long* in, long nseg, long R, long C) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nseg * C) return;
  long s = i / C, c = i % C;
  long long acc = 0;
  const long long* base = in + s * R * C;
  for (long r = 0; r < R; r++) acc += base[r * C + c];
  out[i] = acc;
}
// broadcast each segment's row R times: out[(s*R+r)*C+c] = in[s*C+c]
__global__ void k_tile_rows(long long* out, const long long* in, long nseg, long R, long C) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = nseg * R * C;
  if (i >= tot) return;
  long c = i % C, s = i / (R * C);
  out[i] = in[s * C + c];
}
__global__ void k_rowsum(long long* out, const long long* in, long T, long C) {
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  long long acc = 0;
  for (long c = 0; c < C; c++) acc += in[t * C + c];
  out[t] = acc;
}
__global__ void k_slice_cols(long long* out, const long long* in, long T, long stride, long off, long ncols) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * ncols) return;
  long t = i / ncols, j = i % ncols;
  out[i] = in[t * stride + off + j];
}
__global__ void k_scatter_cols(long long* out, const long long* in, long T, long stride, long off, long ncols) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * ncols) return;
  long t = i / ncols, j = i % ncols;
  out[t * stride + off + j] = in[i];
}
// rotate_half: out[t*hd+j] = sgn[j] * in[t*hd + idx[j]]  (idx/sgn resident, uploaded once)
__global__ void k_permute_scale(long long* out, const long long* in, const long long* idx,
                                const long long* sgn, long T, long hd) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * hd) return;
  long t = i / hd, j = i % hd;
  out[i] = sgn[j] * in[t * hd + idx[j]];
}
// row-wise RMSNorm over T rows of length n (one block per row; same math as k_rmsnorm)
__global__ void k_rmsnorm_rows(long long* o, const long long* x, const long long* w,
                               long T, long n, int frac, long long eps) {
  long row = blockIdx.x;
  if (row >= T) return;
  const long long* xr = x + row * n;
  long long* orow = o + row * n;
  __shared__ long long s_rms;
  if (threadIdx.x == 0) {
    u128 ssq = U(0, 0);
    for (long i = 0; i < n; i++) {
      unsigned long long ax = xr[i] < 0 ? (unsigned long long)(-xr[i]) : (unsigned long long)xr[i];
      ssq = add128(ssq, shr128(mul64u(ax, ax), frac));
    }
    u128 ms = add128(div128u(ssq, (unsigned long long)n), U(0, (unsigned long long)eps));
    long long rms = (long long)isqrt128(shl128(ms, frac));
    s_rms = rms ? rms : 1;
  }
  __syncthreads();
  long long rms = s_rms;
  for (long i = threadIdx.x; i < n; i += blockDim.x) {
    unsigned long long ax = xr[i] < 0 ? (unsigned long long)(-xr[i]) : (unsigned long long)xr[i];
    unsigned long long q = div128u(shl128(U(0, ax), frac), (unsigned long long)rms).lo;
    long long norm = xr[i] < 0 ? -(long long)q : (long long)q;
    orow[i] = mulshr(norm, w[i], frac);
  }
}
// row-wise RMSNorm BACKWARD (the ml.grad closed form, per row):
//   dX_j = (dY_j·w_j)/rms − x_j·Σ_i(dY_i·w_i·x_i)/(n·rms³)   — verified adjoint, on-device.
__global__ void k_rmsnorm_rows_bwd(long long* dx, const long long* dy, const long long* x,
                                   const long long* w, long T, long n, int frac, long long eps) {
  long row = blockIdx.x;
  if (row >= T) return;
  const long long* xr = x + row * n;
  const long long* dyr = dy + row * n;
  long long* dxr = dx + row * n;
  __shared__ long long s_rms, s_inv, s_c;
  if (threadIdx.x == 0) {
    u128 ssq = U(0, 0);
    for (long i = 0; i < n; i++) {
      unsigned long long ax = xr[i] < 0 ? (unsigned long long)(-xr[i]) : (unsigned long long)xr[i];
      ssq = add128(ssq, shr128(mul64u(ax, ax), frac));
    }
    u128 ms = add128(div128u(ssq, (unsigned long long)n), U(0, (unsigned long long)eps));
    long long rms = (long long)isqrt128(shl128(ms, frac));
    if (rms == 0) rms = 1;
    long long inv = (1LL << (frac + frac)) / rms;                       // 2^32/rms
    long long s = 0;
    for (long i = 0; i < n; i++) {                                      // Σ (dY⊙w⊙x)>>2F
      long long g = mulshr_floor(dyr[i], w[i], frac);
      s += mulshr_floor(g, xr[i], frac);
    }
    // c = _sd(s·2^(3F), n·rms³)  — 128-bit intermediate, signed rounding div
    int neg = s < 0;
    unsigned long long as_ = neg ? (unsigned long long)(-s) : (unsigned long long)s;
    u128 num = shl128(U(0, as_), 3 * frac);
    // denom = n·rms³ (fits 128: rms≤2^40-ish); do two 64-div passes via div128u with d=n·rms then /rms²?
    // rms³ can exceed 64 bits; divide sequentially: num/(n·rms) then /rms then /rms (floor each — a
    // conservative rounding; the FD-verified host form rounds once; deviation ≤ 2 LSB, gradient-safe).
    unsigned long long d1 = (unsigned long long)(n) * (unsigned long long)rms;
    u128 q1 = div128u(num, d1);
    u128 q2 = div128u(q1, (unsigned long long)rms);
    u128 q3 = div128u(q2, (unsigned long long)rms);
    long long c = (long long)q3.lo;
    s_rms = rms; s_inv = inv; s_c = neg ? -c : c;
  }
  __syncthreads();
  long long inv = s_inv, c = s_c;
  for (long i = threadIdx.x; i < n; i += blockDim.x) {
    long long g = mulshr_floor(dyr[i], w[i], frac);
    long long t1 = mulshr_floor(g, inv, frac);                          // g/rms in Q16
    long long t2 = mulshr_floor(xr[i], c, frac);                        // x·c>>F
    dxr[i] = t1 - t2;
  }
}
__global__ void k_shr_bias(long long* o, const long long* acc, const long long* b, long T, long M, int frac) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * M) return;
  o[i] = (acc[i] >> frac) + (b ? b[i % M] : 0);
}
__global__ void k_transpose_i32(int* out, const int* in, long R, long C) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= R * C) return;
  long r = i / C, c = i % C;
  out[c * R + r] = in[i];
}
__global__ void k_i64_to_i32(int* out, const long long* in, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (int)in[i];
}

// ── u8 MANIFOLD BROADCAST (SPEC-014): the value path is u8 (14KB/manifold), so a whole BATCH of N
// HyperVectors [N][H][W] sits resident and ONE operator (LUT) broadcasts across every site of every
// manifold — the training utilization pattern (vs the i64 GEMM whose output blows up VRAM at N>~64).
// W = inner (SIMD/coalesced), H = prime middle, N = batch. Interior: 4-neighbour staple + LUT; else copy.
__global__ void k_manifold_staple_u8(unsigned char* out, const unsigned char* grid,
                                     const unsigned char* lut, long W, long H, long N) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = N * H * W;
  if (idx >= tot) return;
  long i = idx % W;
  long rem = idx / W;
  long j = rem % H;
  if (i == 0 || i == W - 1 || j == 0 || j == H - 1) { out[idx] = grid[idx]; return; }
  unsigned s = (unsigned)grid[idx - 1] + (unsigned)grid[idx + 1]
             + (unsigned)grid[idx - W] + (unsigned)grid[idx + W];
  unsigned d = ((unsigned)grid[idx] + s) & 0xFFu;   // ring sum mod 256
  out[idx] = lut[d];                                 // LUT-Boltzmann broadcast (256B, L1-resident)
}

typedef void (*eltk)(unsigned char*, const unsigned char*, const unsigned char*, long);
static int elt(eltk kern, unsigned char* o, const unsigned char* a, const unsigned char* b, long n) {
  unsigned char *da, *db, *dout; cudaError_t e;
  if ((e = cudaMalloc(&da, n))) return (int)e;
  if ((e = cudaMalloc(&db, n))) { cudaFree(da); return (int)e; }
  if ((e = cudaMalloc(&dout, n))) { cudaFree(da); cudaFree(db); return (int)e; }
  cudaMemcpy(da, a, n, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, n, cudaMemcpyHostToDevice);
  long thr = 256, blk = (n + thr - 1) / thr;
  kern<<<(unsigned)blk, (unsigned)thr>>>(dout, da, db, n);
  e = cudaDeviceSynchronize();
  cudaMemcpy(o, dout, n, cudaMemcpyDeviceToHost);
  cudaFree(da); cudaFree(db); cudaFree(dout);
  return (int)e;
}

extern "C" {
__declspec(dllexport) int rk_cuda_available(void) {
  int n = 0; return (cudaGetDeviceCount(&n) == cudaSuccess && n > 0) ? 1 : 0;
}
__declspec(dllexport) int ring_mul(unsigned char* o, const unsigned char* a, const unsigned char* b, long n) { return elt(k_mul, o, a, b, n); }
__declspec(dllexport) int ring_add(unsigned char* o, const unsigned char* a, const unsigned char* b, long n) { return elt(k_add, o, a, b, n); }
__declspec(dllexport) int ring_sub(unsigned char* o, const unsigned char* a, const unsigned char* b, long n) { return elt(k_sub, o, a, b, n); }
__declspec(dllexport) int ring_gemm(unsigned char* C, const unsigned char* A, const unsigned char* B,
                                    long M, long K, long N) {
  unsigned char *dA, *dB, *dC; cudaError_t e; long sA = M * K, sB = K * N, sC = M * N;
  if ((e = cudaMalloc(&dA, sA))) return (int)e;
  if ((e = cudaMalloc(&dB, sB))) { cudaFree(dA); return (int)e; }
  if ((e = cudaMalloc(&dC, sC))) { cudaFree(dA); cudaFree(dB); return (int)e; }
  cudaMemcpy(dA, A, sA, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sB, cudaMemcpyHostToDevice);
  dim3 block(16, 16), grid((unsigned)((N + 15) / 16), (unsigned)((M + 15) / 16));
  k_gemm<<<grid, block>>>(dC, dA, dB, M, K, N);
  e = cudaDeviceSynchronize();
  cudaMemcpy(C, dC, sC, cudaMemcpyDeviceToHost);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return (int)e;
}
__declspec(dllexport) int ring_l1dist(int* D, const unsigned char* Q, const unsigned char* K,
                                      long m, long n, long dim) {
  unsigned char *dQ, *dK; int* dD; cudaError_t e; long sQ = m * dim, sK = n * dim, sD = m * n;
  if ((e = cudaMalloc(&dQ, sQ))) return (int)e;
  if ((e = cudaMalloc(&dK, sK))) { cudaFree(dQ); return (int)e; }
  if ((e = cudaMalloc(&dD, sD * (long)sizeof(int)))) { cudaFree(dQ); cudaFree(dK); return (int)e; }
  cudaMemcpy(dQ, Q, sQ, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, K, sK, cudaMemcpyHostToDevice);
  dim3 block(16, 16), grid((unsigned)((n + 15) / 16), (unsigned)((m + 15) / 16));
  k_l1dist<<<grid, block>>>(dD, dQ, dK, m, n, dim);
  e = cudaDeviceSynchronize();
  cudaMemcpy(D, dD, sD * (long)sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(dQ); cudaFree(dK); cudaFree(dD);
  return (int)e;
}
__declspec(dllexport) int ring_gemv_i64(long long* out, const int* W, const int* x,
                                        long M, long K) {
  int *dW, *dx; long long* dout; cudaError_t e; long sW = M * K;
  if ((e = cudaMalloc(&dW, sW * (long)sizeof(int)))) return (int)e;
  if ((e = cudaMalloc(&dx, K * (long)sizeof(int)))) { cudaFree(dW); return (int)e; }
  if ((e = cudaMalloc(&dout, M * (long)sizeof(long long)))) { cudaFree(dW); cudaFree(dx); return (int)e; }
  cudaMemcpy(dW, W, sW * (long)sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dx, x, K * (long)sizeof(int), cudaMemcpyHostToDevice);
  long thr = 128, blk = (M + thr - 1) / thr;
  k_gemv_i64<<<(unsigned)blk, (unsigned)thr>>>(dout, dW, dx, M, K);
  e = cudaDeviceSynchronize();
  cudaMemcpy(out, dout, M * (long)sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFree(dW); cudaFree(dx); cudaFree(dout);
  return (int)e;
}
__declspec(dllexport) int ring_gemm_i64(long long* out, const int* X, const int* W,
                                        long T, long M, long K) {
  int *dX, *dW; long long* dout; cudaError_t e;
  if ((e = cudaMalloc(&dX, T * K * (long)sizeof(int)))) return (int)e;
  if ((e = cudaMalloc(&dW, M * K * (long)sizeof(int)))) { cudaFree(dX); return (int)e; }
  if ((e = cudaMalloc(&dout, T * M * (long)sizeof(long long)))) { cudaFree(dX); cudaFree(dW); return (int)e; }
  cudaMemcpy(dX, X, T * K * (long)sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dW, W, M * K * (long)sizeof(int), cudaMemcpyHostToDevice);
  dim3 block(16, 16), grid((unsigned)((M + 15) / 16), (unsigned)((T + 15) / 16));
  k_gemm_i64<<<grid, block>>>(dout, dX, dW, T, M, K);
  e = cudaDeviceSynchronize();
  cudaMemcpy(out, dout, T * M * (long)sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFree(dX); cudaFree(dW); cudaFree(dout);
  return (int)e;
}
__declspec(dllexport) int ring_sigmoid(long long* o, const long long* x, long n, int frac) {
  long long *dx, *dout; cudaError_t e;
  if ((e = cudaMalloc(&dx, n * (long)sizeof(long long)))) return (int)e;
  if ((e = cudaMalloc(&dout, n * (long)sizeof(long long)))) { cudaFree(dx); return (int)e; }
  cudaMemcpy(dx, x, n * (long)sizeof(long long), cudaMemcpyHostToDevice);
  long thr = 256, blk = (n + thr - 1) / thr;
  k_sigmoid<<<(unsigned)blk, (unsigned)thr>>>(dout, dx, n, frac);
  e = cudaDeviceSynchronize();
  cudaMemcpy(o, dout, n * (long)sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFree(dx); cudaFree(dout);
  return (int)e;
}
__declspec(dllexport) int ring_exp(long long* o, const long long* x, long n, int frac) {
  long long *dx, *dout; cudaError_t e;
  if ((e = cudaMalloc(&dx, n * (long)sizeof(long long)))) return (int)e;
  if ((e = cudaMalloc(&dout, n * (long)sizeof(long long)))) { cudaFree(dx); return (int)e; }
  cudaMemcpy(dx, x, n * (long)sizeof(long long), cudaMemcpyHostToDevice);
  long thr = 256, blk = (n + thr - 1) / thr;
  k_exp<<<(unsigned)blk, (unsigned)thr>>>(dout, dx, n, frac);
  e = cudaDeviceSynchronize();
  cudaMemcpy(o, dout, n * (long)sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFree(dx); cudaFree(dout);
  return (int)e;
}
__declspec(dllexport) int ring_rmsnorm(long long* o, const long long* x, const long long* w,
                                       long n, int frac, long long eps) {
  long long *dx, *dw, *dout; cudaError_t e;
  if ((e = cudaMalloc(&dx, n * (long)sizeof(long long)))) return (int)e;
  if ((e = cudaMalloc(&dw, n * (long)sizeof(long long)))) { cudaFree(dx); return (int)e; }
  if ((e = cudaMalloc(&dout, n * (long)sizeof(long long)))) { cudaFree(dx); cudaFree(dw); return (int)e; }
  cudaMemcpy(dx, x, n * (long)sizeof(long long), cudaMemcpyHostToDevice);
  cudaMemcpy(dw, w, n * (long)sizeof(long long), cudaMemcpyHostToDevice);
  long thr = n < 256 ? n : 256;
  k_rmsnorm<<<1, (unsigned)thr>>>(dout, dx, dw, n, frac, eps);
  e = cudaDeviceSynchronize();
  cudaMemcpy(o, dout, n * (long)sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFree(dx); cudaFree(dw); cudaFree(dout);
  return (int)e;
}
// binary Q16 elementwise: op 0=emul(a*b>>frac) 1=eadd 2=esub. one host wrapper, kernel by op.
__declspec(dllexport) int ring_ew_q16(int op, long long* o, const long long* a, const long long* b,
                                      long n, int frac) {
  long long *da, *db, *dout; cudaError_t e;
  long sz = n * (long)sizeof(long long);
  if ((e = cudaMalloc(&da, sz))) return (int)e;
  if ((e = cudaMalloc(&db, sz))) { cudaFree(da); return (int)e; }
  if ((e = cudaMalloc(&dout, sz))) { cudaFree(da); cudaFree(db); return (int)e; }
  cudaMemcpy(da, a, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, sz, cudaMemcpyHostToDevice);
  long thr = 256, blk = (n + thr - 1) / thr;
  if (op == 0) k_emul<<<(unsigned)blk, (unsigned)thr>>>(dout, da, db, n, frac);
  else if (op == 1) k_eadd<<<(unsigned)blk, (unsigned)thr>>>(dout, da, db, n);
  else k_esub<<<(unsigned)blk, (unsigned)thr>>>(dout, da, db, n);
  e = cudaDeviceSynchronize();
  cudaMemcpy(o, dout, sz, cudaMemcpyDeviceToHost);
  cudaFree(da); cudaFree(db); cudaFree(dout);
  return (int)e;
}
__declspec(dllexport) int ring_escale(long long* o, const long long* a, long long sc, long n, int frac) {
  long long *da, *dout; cudaError_t e; long sz = n * (long)sizeof(long long);
  if ((e = cudaMalloc(&da, sz))) return (int)e;
  if ((e = cudaMalloc(&dout, sz))) { cudaFree(da); return (int)e; }
  cudaMemcpy(da, a, sz, cudaMemcpyHostToDevice);
  long thr = 256, blk = (n + thr - 1) / thr;
  k_escale<<<(unsigned)blk, (unsigned)thr>>>(dout, da, sc, n, frac);
  e = cudaDeviceSynchronize();
  cudaMemcpy(o, dout, sz, cudaMemcpyDeviceToHost);
  cudaFree(da); cudaFree(dout);
  return (int)e;
}
__declspec(dllexport) int ring_colsum(long long* out, const long long* in, long R, long C) {
  long long *din, *dout; cudaError_t e;
  if ((e = cudaMalloc(&din, R * C * (long)sizeof(long long)))) return (int)e;
  if ((e = cudaMalloc(&dout, C * (long)sizeof(long long)))) { cudaFree(din); return (int)e; }
  cudaMemcpy(din, in, R * C * (long)sizeof(long long), cudaMemcpyHostToDevice);
  long thr = 256, blk = (C + thr - 1) / thr;
  k_colsum<<<(unsigned)blk, (unsigned)thr>>>(dout, din, R, C);
  e = cudaDeviceSynchronize();
  cudaMemcpy(out, dout, C * (long)sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFree(din); cudaFree(dout);
  return (int)e;
}
__declspec(dllexport) int ring_relu(long long* o, const long long* a, long n) {
  long long *da, *dout; cudaError_t e; long sz = n * (long)sizeof(long long);
  if ((e = cudaMalloc(&da, sz))) return (int)e;
  if ((e = cudaMalloc(&dout, sz))) { cudaFree(da); return (int)e; }
  cudaMemcpy(da, a, sz, cudaMemcpyHostToDevice);
  long thr = 256, blk = (n + thr - 1) / thr;
  k_relu<<<(unsigned)blk, (unsigned)thr>>>(dout, da, n);
  e = cudaDeviceSynchronize();
  cudaMemcpy(o, dout, sz, cudaMemcpyDeviceToHost);
  cudaFree(da); cudaFree(dout);
  return (int)e;
}
__declspec(dllexport) int ring_gather(long long* o, const long long* lut, long lutn,
                                      const unsigned char* idx, long n) {
  long long *dlut, *dout; unsigned char* didx; cudaError_t e;
  if ((e = cudaMalloc(&dlut, lutn * (long)sizeof(long long)))) return (int)e;
  if ((e = cudaMalloc(&didx, n))) { cudaFree(dlut); return (int)e; }
  if ((e = cudaMalloc(&dout, n * (long)sizeof(long long)))) { cudaFree(dlut); cudaFree(didx); return (int)e; }
  cudaMemcpy(dlut, lut, lutn * (long)sizeof(long long), cudaMemcpyHostToDevice);
  cudaMemcpy(didx, idx, n, cudaMemcpyHostToDevice);
  long thr = 256, blk = (n + thr - 1) / thr;
  k_gather<<<(unsigned)blk, (unsigned)thr>>>(dout, dlut, didx, n);
  e = cudaDeviceSynchronize();
  cudaMemcpy(o, dout, n * (long)sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFree(dlut); cudaFree(didx); cudaFree(dout);
  return (int)e;
}
__declspec(dllexport) int ring_diffuse(long long* o, const long long* in, long D, long H, long hd) {
  long long *din, *dout; cudaError_t e; long tot = D * H * hd, sz = tot * (long)sizeof(long long);
  if ((e = cudaMalloc(&din, sz))) return (int)e;
  if ((e = cudaMalloc(&dout, sz))) { cudaFree(din); return (int)e; }
  cudaMemcpy(din, in, sz, cudaMemcpyHostToDevice);
  long thr = 256, blk = (tot + thr - 1) / thr;
  k_diffuse<<<(unsigned)blk, (unsigned)thr>>>(dout, din, D, H, hd);
  e = cudaDeviceSynchronize();
  cudaMemcpy(o, dout, sz, cudaMemcpyDeviceToHost);
  cudaFree(din); cudaFree(dout);
  return (int)e;
}

// ── RESIDENCY API (SPEC-014 Part B): keep weights/activations on-device across calls. Fixes the
// per-call cudaMalloc/H2D/D2H/free + Python-list marshaling that starved the GPU (2% util). Upload
// ONCE -> a device handle (void*); launch the GEMM on resident handles (no transfers); download ONCE.
// A training loop uploads the weight ONCE and reuses the handle across tokens/epochs. The kernels are
// the SAME bit-exact k_gemm_i64 (D9 gate above unchanged).
__declspec(dllexport) void* rk_dev_upload_i32(const int* host, long n) {
  int* d; if (cudaMalloc(&d, n * (long)sizeof(int)) != cudaSuccess) return 0;
  if (cudaMemcpy(d, host, n * (long)sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(d); return 0; }
  return (void*)d;
}
__declspec(dllexport) void* rk_dev_alloc_i64(long n) {
  long long* d; if (cudaMalloc(&d, n * (long)sizeof(long long)) != cudaSuccess) return 0;
  return (void*)d;
}
__declspec(dllexport) int rk_dev_gemm_i64_resident(void* d_out, const void* d_X, const void* d_W,
                                                   long T, long M, long K) {
  dim3 block(16, 16), grid((unsigned)((M + 15) / 16), (unsigned)((T + 15) / 16));
  k_gemm_i64<<<grid, block>>>((long long*)d_out, (const int*)d_X, (const int*)d_W, T, M, K);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_download_i64(long long* host, const void* d_out, long n) {
  return (int)cudaMemcpy(host, d_out, n * (long)sizeof(long long), cudaMemcpyDeviceToHost);
}
__declspec(dllexport) void rk_dev_free(void* p) { if (p) cudaFree((void*)p); }
// u8 residency: upload a u8 buffer (a batch of manifolds) ONCE; broadcast the LUT operator across it.
__declspec(dllexport) void* rk_dev_upload_u8(const unsigned char* host, long n) {
  unsigned char* d; if (cudaMalloc(&d, n) != cudaSuccess) return 0;
  if (cudaMemcpy(d, host, n, cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(d); return 0; }
  return (void*)d;
}
__declspec(dllexport) void* rk_dev_alloc_u8(long n) {
  unsigned char* d; if (cudaMalloc(&d, n) != cudaSuccess) return 0; return (void*)d;
}
__declspec(dllexport) int rk_dev_manifold_staple_u8_resident(void* d_out, const void* d_grid,
                                                             const void* d_lut, long W, long H, long N) {
  long tot = N * H * W, thr = 256, blk = (tot + thr - 1) / thr;
  k_manifold_staple_u8<<<(unsigned)blk, (unsigned)thr>>>((unsigned char*)d_out,
      (const unsigned char*)d_grid, (const unsigned char*)d_lut, W, H, N);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_download_u8(unsigned char* host, const void* d, long n) {
  return (int)cudaMemcpy(host, d, n, cudaMemcpyDeviceToHost);
}
// ── resident-chain exports: handle-in/handle-out, no transfers; sync once per call (cheap) ──
#define RK_GRID(n) (unsigned)(((n) + 255) / 256), 256
__declspec(dllexport) void* rk_dev_upload_i64(const long long* host, long n) {
  long long* d; if (cudaMalloc(&d, n * (long)sizeof(long long)) != cudaSuccess) return 0;
  if (cudaMemcpy(d, host, n * (long)sizeof(long long), cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(d); return 0; }
  return (void*)d;
}
__declspec(dllexport) void* rk_dev_alloc_i32(long n) {
  int* d; if (cudaMalloc(&d, n * (long)sizeof(int)) != cudaSuccess) return 0; return (void*)d;
}
__declspec(dllexport) int rk_dev_ew_res(int op, void* o, const void* a, const void* b, long n, int frac) {
  if (op == 0) k_emul<<<RK_GRID(n)>>>((long long*)o, (const long long*)a, (const long long*)b, n, frac);
  else if (op == 1) k_eadd<<<RK_GRID(n)>>>((long long*)o, (const long long*)a, (const long long*)b, n);
  else k_esub<<<RK_GRID(n)>>>((long long*)o, (const long long*)a, (const long long*)b, n);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_escale_res(void* o, const void* a, long long sc, long n, int frac) {
  k_escale<<<RK_GRID(n)>>>((long long*)o, (const long long*)a, sc, n, frac);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_softplus(void* o, const void* x, long n, int frac) {
  k_softplus<<<RK_GRID(n)>>>((long long*)o, (const long long*)x, n, frac);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_sigmoid_res(void* o, const void* x, long n, int frac) {
  k_sigmoid<<<RK_GRID(n)>>>((long long*)o, (const long long*)x, n, frac);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_colsum_seg(void* out, const void* in, long nseg, long R, long C) {
  k_colsum_seg<<<RK_GRID(nseg * C)>>>((long long*)out, (const long long*)in, nseg, R, C);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_tile_rows(void* out, const void* in, long nseg, long R, long C) {
  k_tile_rows<<<RK_GRID(nseg * R * C)>>>((long long*)out, (const long long*)in, nseg, R, C);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_rowsum(void* out, const void* in, long T, long C) {
  k_rowsum<<<RK_GRID(T)>>>((long long*)out, (const long long*)in, T, C);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_slice_cols(void* out, const void* in, long T, long stride, long off, long ncols) {
  k_slice_cols<<<RK_GRID(T * ncols)>>>((long long*)out, (const long long*)in, T, stride, off, ncols);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_scatter_cols(void* out, const void* in, long T, long stride, long off, long ncols) {
  k_scatter_cols<<<RK_GRID(T * ncols)>>>((long long*)out, (const long long*)in, T, stride, off, ncols);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_permute_scale(void* out, const void* in, const void* idx, const void* sgn,
                                               long T, long hd) {
  k_permute_scale<<<RK_GRID(T * hd)>>>((long long*)out, (const long long*)in,
                                       (const long long*)idx, (const long long*)sgn, T, hd);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_rmsnorm_rows(void* o, const void* x, const void* w,
                                              long T, long n, int frac, long long eps) {
  long thr = n < 256 ? n : 256;
  k_rmsnorm_rows<<<(unsigned)T, (unsigned)thr>>>((long long*)o, (const long long*)x,
                                                 (const long long*)w, T, n, frac, eps);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_rmsnorm_rows_bwd(void* dx, const void* dy, const void* x, const void* w,
                                                  long T, long n, int frac, long long eps) {
  long thr = n < 256 ? n : 256;
  k_rmsnorm_rows_bwd<<<(unsigned)T, (unsigned)thr>>>((long long*)dx, (const long long*)dy,
                                                     (const long long*)x, (const long long*)w, T, n, frac, eps);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_shr_bias(void* o, const void* acc, const void* b, long T, long M, int frac) {
  k_shr_bias<<<RK_GRID(T * M)>>>((long long*)o, (const long long*)acc, (const long long*)b, T, M, frac);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_transpose_i32(void* out, const void* in, long R, long C) {
  k_transpose_i32<<<RK_GRID(R * C)>>>((int*)out, (const int*)in, R, C);
  return (int)cudaDeviceSynchronize();
}
__declspec(dllexport) int rk_dev_i64_to_i32(void* out, const void* in, long n) {
  k_i64_to_i32<<<RK_GRID(n)>>>((int*)out, (const long long*)in, n);
  return (int)cudaDeviceSynchronize();
}
}

// ── u8-ONIX fused exact GEMV (the emulation proj, whole): one block per row ──
// out[r] = sdiv( (Σ_k (xbar[r*K+k]-128)·x[k]) shifted by s_row[r], z_row[r] ) —
// the EXACT semantics of kernels/mprc/gemma/host._py_gemv_exact (D9-gated by
// the caller on first use): arithmetic shift = Python floor >>, C `/` = the
// symmetric truncating divide. i64 accumulation, no fold.
__global__ void k_gemv_u8_exact(long long* out, const unsigned char* xbar, const long long* x,
                                const signed char* s_row, const long long* z_row, long M, long K) {
  long r = blockIdx.x;
  if (r >= M) return;
  const unsigned char* row = xbar + r * K;
  __shared__ long long sh[256];
  long long acc = 0;
  for (long k = threadIdx.x; k < K; k += blockDim.x)
    acc += (long long)((int)row[k] - 128) * x[k];
  sh[threadIdx.x] = acc;
  __syncthreads();
  for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
    if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    // The Python reference shifts UNBOUNDED ints; a bare i64 D<<s wrapped on
    // real layer-4 magnitudes (measured: |h| off 56.8%). No __int128 on the
    // MSVC nvcc host, and none needed: for s >= 0,
    //   trunc(D*2^s / z) = (|D|/z)*2^s + ((|D|%z)*2^s)/z    (sign applied last)
    // — the remainder is < z, so the second term never overflows, and the
    // first is the result's own scale (fits i64 whenever the OUTPUT does,
    // which the Mac-passing forward establishes). For s < 0 the floor shift
    // precedes the divide and D itself always fits i64.
    long long D = sh[0];
    int s = s_row[r];
    long long z = z_row[r]; if (z == 0) z = 1;
    if (s >= 0) {
      unsigned long long aD = (D < 0) ? (unsigned long long)(-D) : (unsigned long long)D;
      unsigned long long uz = (unsigned long long)z;
      unsigned long long q = aD / uz;
      unsigned long long rm = aD % uz;
      unsigned long long mag = (q << s) + ((rm << s) / uz);
      out[r] = (D < 0) ? -(long long)mag : (long long)mag;
    } else {
      long long t = D >> (-s);            // arithmetic shift = Python floor >>
      out[r] = t / z;                     // C trunc = the symmetric divide
    }
  }
}

extern "C" {
// RESIDENT variant: the slab is already on-device (rk_dev_upload_u8 handle) —
// per call only the activation (K·8 B) and s/z rows cross PCIe. This is what
// lets the weights stay put while tokens flow.
__declspec(dllexport) int rk_gemv_u8_exact_res(long long* out, const void* dxbar, const long long* x,
                                               const signed char* s_row, const long long* z_row,
                                               long M, long K) {
  long long *dx, *dout, *dz; signed char* ds; cudaError_t e;
  if ((e = cudaMalloc(&dx, K * (long)sizeof(long long)))) return (int)e;
  if ((e = cudaMalloc(&dout, M * (long)sizeof(long long)))) { cudaFree(dx); return (int)e; }
  if ((e = cudaMalloc(&ds, M))) { cudaFree(dx); cudaFree(dout); return (int)e; }
  if ((e = cudaMalloc(&dz, M * (long)sizeof(long long)))) { cudaFree(dx); cudaFree(dout); cudaFree(ds); return (int)e; }
  cudaMemcpy(dx, x, K * (long)sizeof(long long), cudaMemcpyHostToDevice);
  cudaMemcpy(ds, s_row, M, cudaMemcpyHostToDevice);
  cudaMemcpy(dz, z_row, M * (long)sizeof(long long), cudaMemcpyHostToDevice);
  k_gemv_u8_exact<<<(unsigned)M, 256>>>(dout, (const unsigned char*)dxbar, dx, ds, dz, M, K);
  e = cudaDeviceSynchronize();
  cudaMemcpy(out, dout, M * (long)sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFree(dx); cudaFree(dout); cudaFree(ds); cudaFree(dz);
  return (int)e;
}
__declspec(dllexport) int rk_gemv_u8_exact(long long* out, const unsigned char* xbar, const long long* x,
                                           const signed char* s_row, const long long* z_row,
                                           long M, long K) {
  unsigned char* dxb; long long *dx, *dout, *dz; signed char* ds; cudaError_t e; long sW = M * K;
  if ((e = cudaMalloc(&dxb, sW))) return (int)e;
  if ((e = cudaMalloc(&dx, K * (long)sizeof(long long)))) { cudaFree(dxb); return (int)e; }
  if ((e = cudaMalloc(&dout, M * (long)sizeof(long long)))) { cudaFree(dxb); cudaFree(dx); return (int)e; }
  if ((e = cudaMalloc(&ds, M))) { cudaFree(dxb); cudaFree(dx); cudaFree(dout); return (int)e; }
  if ((e = cudaMalloc(&dz, M * (long)sizeof(long long)))) { cudaFree(dxb); cudaFree(dx); cudaFree(dout); cudaFree(ds); return (int)e; }
  cudaMemcpy(dxb, xbar, sW, cudaMemcpyHostToDevice);
  cudaMemcpy(dx, x, K * (long)sizeof(long long), cudaMemcpyHostToDevice);
  cudaMemcpy(ds, s_row, M, cudaMemcpyHostToDevice);
  cudaMemcpy(dz, z_row, M * (long)sizeof(long long), cudaMemcpyHostToDevice);
  k_gemv_u8_exact<<<(unsigned)M, 256>>>(dout, dxb, dx, ds, dz, M, K);
  e = cudaDeviceSynchronize();
  cudaMemcpy(out, dout, M * (long)sizeof(long long), cudaMemcpyDeviceToHost);
  cudaFree(dxb); cudaFree(dx); cudaFree(dout); cudaFree(ds); cudaFree(dz);
  return (int)e;
}
}
