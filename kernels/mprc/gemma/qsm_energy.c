/* ringkit silicon — ENERGY-domain quarter-square GEMV for model emulation (charter D9).
 *
 * Emulating a traditional model (Gemma) needs the FULL int accumulation (the ENERGY/distance side),
 * NOT the mod-256 ARC fold that ring_gemm does. out[row] = sum_k (xbar[row*K+k]-128) * x[k], with
 * xbar uint8 offset-binary (W+128) and x int8, accumulated in int64 and NEVER folded.
 *
 * The product is ringkit's QCM quarter-square identity  a*b = (SQ[|a+b|] - SQ[|a-b|]) >> 2  — a
 * table read + shift + add, NO hardware multiply. This is ringkit's own QSM (the FPU replacement),
 * written from the identity, NOT a copy of hpq's dot_qph. Even the square table is built by
 * odd-number accumulation (n^2 = (n-1)^2 + 2n-1), so there is no multiply anywhere.
 *
 * |a| <= 128, |b| <= 127  ->  |a+b|,|a-b| <= 255, so SQ[0..255] suffices (sized 512 for headroom). */
#include <stdint.h>
#include <stddef.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>

static int64_t SQ[512];
static int sq_ready = 0;

static void build_sq(void) {
    SQ[0] = 0;
    for (int n = 1; n < 512; n++) SQ[n] = SQ[n - 1] + ((int64_t)(n << 1) - 1);   /* n^2, no multiply */
    sq_ready = 1;
}

/* out[row] = sum_k (xbar[row*K+k] - 128) * x[k]   (int64, unfolded ENERGY). Multiplier-free QSM. */
void qsm_dot(int64_t * restrict out, const uint8_t * restrict xbar,
             const int8_t * restrict x, long M, long K) {
    if (!sq_ready) build_sq();
    for (long r = 0; r < M; r++) {
        const uint8_t * restrict w = xbar + r * K;
        int64_t acc = 0;
        for (long k = 0; k < K; k++) {
            int a = (int)w[k] - 128;          /* signed weight  [-128,127] */
            int b = (int)x[k];                /* int8 activation [-127,127] */
            int s = a + b; if (s < 0) s = -s;
            int d = a - b; if (d < 0) d = -d;
            acc += (SQ[s] - SQ[d]) >> 2;       /* a*b, exact (s,d same parity -> divisible by 4) */
        }
        out[r] = acc;
    }
}

/* ── Fused EXACT digit-decomposition GEMV (the whole `proj` in one block call) ────────────────
 * The ring does NOT quantize the model: the Q<frac> activation vector x (int64) is decomposed
 * EXACTLY into power-of-2-scaled int8 digit passes (residual re-encoded until ZERO — each pass
 * peels >=7 bits, so <=10 passes for any int64), then ONE memory sweep of the weight slab
 * accumulates every pass's QSM dot per row, and the row is finished in place:
 *     out[r] = sd( (SUM_p dot_p << (a_p - a_min)) << (a_min + s_row[r] + frac), z_row[r] )
 * BIT-FOR-BIT equal to the Python semantic reference emulation/gemma.py::proj (D9; __int128 for
 * the combine so no overflow departs from Python's bigints; >> on negatives is arithmetic/floor,
 * same as Python). Rows are disjoint blocks (split/merge-free by construction — the MPP axis).
 * Returns the number of digit passes, or -1 on error. */
#define GEMV_MAXP 16

static int ge127(long e, int64_t mx) {            /* 127*2^e >= mx (exact for integer mx) */
    if (e >= 0) return (e >= 56) || (((int64_t)127 << e) >= mx);
    return (e <= -8) ? (0 >= mx) : ((int64_t)(127 >> (-e)) >= mx);
}

/* Exact digit decomposition of x into xs_scratch (np passes); a_pass[p] receives each scale. */
static long gemv_decompose(const int64_t * restrict x, long K, int frac,
                           int8_t * restrict xs_scratch, int64_t * restrict r_scratch,
                           long * restrict a_pass) {
    int64_t mx = 0;
    for (long k = 0; k < K; k++) {
        int64_t av = x[k] < 0 ? -x[k] : x[k];
        r_scratch[k] = x[k];
        if (av > mx) mx = av;
    }
    if (mx == 0) return 0;
    long np = 0;
    while (np < GEMV_MAXP) {
        /* a = smallest a with 127*2^(a+frac) >= mx  (same walk as the Python reference) */
        long a = 0;
        if (ge127(frac, mx))       { while (ge127(a - 1 + frac, mx)) a--; }
        else                       { while (!ge127(a + frac, mx)) a++; }
        long sh = frac + a;
        int8_t * restrict xs = xs_scratch + np * K;
        if (sh > 0) {
            int64_t rnd = (int64_t)1 << (sh - 1);
            for (long k = 0; k < K; k++) {
                int64_t v = r_scratch[k];
                int64_t q = v >= 0 ? ((v + rnd) >> sh) : -((-v + rnd) >> sh);
                if (q > 127) q = 127; else if (q < -127) q = -127;
                xs[k] = (int8_t)q;
            }
        } else if (sh == 0) {
            for (long k = 0; k < K; k++) {
                int64_t q = r_scratch[k];
                if (q > 127) q = 127; else if (q < -127) q = -127;
                xs[k] = (int8_t)q;
            }
        } else {
            for (long k = 0; k < K; k++) xs[k] = (int8_t)(r_scratch[k] << (-sh));
        }
        a_pass[np++] = a;
        if (sh <= 0) break;                       /* pure left shift: already exact */
        mx = 0;
        for (long k = 0; k < K; k++) {
            int64_t rv = r_scratch[k] - ((int64_t)xs[k] << sh);
            r_scratch[k] = rv;
            if (rv < 0) rv = -rv;
            if (rv > mx) mx = rv;
        }
        if (mx == 0) break;                       /* residual zero: decomposition complete */
    }
    return np;
}

/* Rows [r0, r1): one sweep of the slab per row, all passes' dots together, row finished in
 * place. Rows are DISJOINT blocks — the specialised-MPP split axis, merge-free by construction. */
static void gemv_rows(int64_t * restrict out, const uint8_t * restrict xbar,
                      long K, const int32_t * restrict s_row, const int64_t * restrict z_row,
                      int frac, const int8_t * restrict xs_scratch,
                      const long * restrict a_pass, long np, long a_min, long r0, long r1) {
    for (long r = r0; r < r1; r++) {
        const uint8_t * restrict w = xbar + (size_t)r * K;
        int64_t dot[GEMV_MAXP];
        for (long p = 0; p < np; p++) dot[p] = 0;
        for (long k = 0; k < K; k++) {
            int a = (int)w[k] - 128;
            for (long p = 0; p < np; p++) {
                int b = (int)xs_scratch[p * K + k];
                int s = a + b; if (s < 0) s = -s;
                int d = a - b; if (d < 0) d = -d;
                dot[p] += (SQ[s] - SQ[d]) >> 2;   /* a*b via QSM, exact */
            }
        }
        __int128 acc = 0;
        for (long p = 0; p < np; p++) acc += (__int128)dot[p] << (a_pass[p] - a_min);
        long shift = a_min + (long)s_row[r] + frac;
        __int128 t = shift >= 0 ? (acc << shift) : (acc >> (-shift));
        int64_t z = z_row[r] ? z_row[r] : 1;
        __int128 q = t < 0 ? -((-t) / z) : t / z; /* symmetric divide, == Python _sd */
        out[r] = (int64_t)q;
    }
}

typedef struct {
    int64_t *out; const uint8_t *xbar; long K;
    const int32_t *s_row; const int64_t *z_row; int frac;
    const int8_t *xs; const long *a_pass; long np; long a_min;
    long r0, r1;
} gemv_job;

static void *gemv_worker(void *arg) {
    gemv_job *j = (gemv_job *)arg;
    gemv_rows(j->out, j->xbar, j->K, j->s_row, j->z_row, j->frac,
              j->xs, j->a_pass, j->np, j->a_min, j->r0, j->r1);
    return NULL;
}

long qsm_gemv_exact(int64_t * restrict out, const uint8_t * restrict xbar,
                    const int64_t * restrict x, long M, long K,
                    const int32_t * restrict s_row, const int64_t * restrict z_row,
                    int frac, int8_t * restrict xs_scratch, int64_t * restrict r_scratch) {
    if (!sq_ready) build_sq();
    long a_pass[GEMV_MAXP];
    long np = gemv_decompose(x, K, frac, xs_scratch, r_scratch, a_pass);
    if (np == 0) {
        for (long r = 0; r < M; r++) out[r] = 0;
        return 0;
    }
    long a_min = a_pass[0];
    for (long p = 1; p < np; p++) if (a_pass[p] < a_min) a_min = a_pass[p];
    gemv_rows(out, xbar, K, s_row, z_row, frac, xs_scratch, a_pass, np, a_min, 0, M);
    return np;
}

/* Specialised-MPP variant: decompose ONCE, then split the row range into nthreads disjoint
 * blocks (lock-free, no merge step — each block owns its out rows). Bit-identical to the
 * single-thread path by construction. */
long qsm_gemv_exact_mt(int64_t * restrict out, const uint8_t * restrict xbar,
                       const int64_t * restrict x, long M, long K,
                       const int32_t * restrict s_row, const int64_t * restrict z_row,
                       int frac, int8_t * restrict xs_scratch, int64_t * restrict r_scratch,
                       long nthreads) {
    if (!sq_ready) build_sq();
    long a_pass[GEMV_MAXP];
    long np = gemv_decompose(x, K, frac, xs_scratch, r_scratch, a_pass);
    if (np == 0) {
        for (long r = 0; r < M; r++) out[r] = 0;
        return 0;
    }
    long a_min = a_pass[0];
    for (long p = 1; p < np; p++) if (a_pass[p] < a_min) a_min = a_pass[p];
    if (nthreads > M) nthreads = M;
    if (nthreads <= 1) {
        gemv_rows(out, xbar, K, s_row, z_row, frac, xs_scratch, a_pass, np, a_min, 0, M);
        return np;
    }
    pthread_t tid[64];
    gemv_job job[64];
    if (nthreads > 64) nthreads = 64;
    long chunk = (M + nthreads - 1) / nthreads;
    long nt = 0;
    for (long t = 0; t < nthreads; t++) {
        long r0 = t * chunk, r1 = r0 + chunk;
        if (r0 >= M) break;
        if (r1 > M) r1 = M;
        job[nt] = (gemv_job){out, xbar, K, s_row, z_row, frac,
                             xs_scratch, a_pass, np, a_min, r0, r1};
        if (pthread_create(&tid[nt], NULL, gemv_worker, &job[nt]) != 0) {
            gemv_rows(out, xbar, K, s_row, z_row, frac, xs_scratch, a_pass, np, a_min, r0, r1);
            continue;                              /* run this block inline on spawn failure */
        }
        nt++;
    }
    for (long t = 0; t < nt; t++) pthread_join(tid[t], NULL);
    return np;
}

/* ── Fixed-point activation BLOCK kernels (gelu_pytorch_tanh + RMSNorm) ──────────────────────
 * Bit-for-bit replicas of the Python semantic references (emulation/ract.py, emulation/gemma4.py):
 * rn.mul is the exact integer product and mf_floordiv is floor-division, so hardware * and / give
 * identical values (D9 kernels may use hardware multiply); Python's arbitrary-precision e^|x| in
 * exp_fixed only ever appears as a DIVISOR (2^(2f)/acc or 2^(2f)/(one+acc)), so any acc above
 * 2^(2f) yields quotient 0 — the kernel saturates acc at 2^(2f+8), which is exactly equivalent.
 * All intermediates that can exceed 63 bits go through __int128; >> on negatives is arithmetic
 * (floor), matching Python. */

static inline int64_t sdiv_i128(__int128 n, int64_t d) {      /* truncate toward zero, == _sdiv */
    return n < 0 ? -(int64_t)((-n) / d) : (int64_t)(n / d);
}

static int64_t exp_fixed_c(int64_t x, int frac) {             /* == ract.exp_fixed (saturated) */
    const int64_t one = (int64_t)1 << frac;
    const int64_t clamp = (int64_t)1 << (frac + frac + 8);    /* > 2^(2f): divisor-only regime */
    int neg = x < 0;
    int64_t ax = neg ? -x : x;
    int64_t half = one >> 1;
    int m = 0;
    int64_t red = ax;
    while (red > half) { red >>= 1; m++; }
    int64_t term = one, acc = one;
    for (int k = 1; k <= 12; k++) {
        term = (int64_t)(((__int128)term * red) >> frac);
        term = term / k;                                      /* term >= 0: floor == trunc */
        acc += term;
        if (term == 0) break;
    }
    for (int i = 0; i < m; i++) {
        if (acc >= clamp) { acc = clamp; continue; }          /* saturated: stays saturated */
        acc = (int64_t)(((__int128)acc * acc) >> frac);
        if (acc >= clamp) acc = clamp;
    }
    if (neg) acc = ((int64_t)1 << (frac + frac)) / acc;       /* e^-|x| = 1/e^|x|, both >= 0 */
    return acc;
}

static int64_t sigmoid_fixed_c(int64_t x, int frac) {         /* == ract.sigmoid_fixed */
    const int64_t one = (int64_t)1 << frac;
    int64_t e = exp_fixed_c(-x, frac);
    return ((int64_t)1 << (frac + frac)) / (one + e);
}

static int64_t tanh_fixed_c(int64_t x, int frac) {            /* == ract.tanh_fixed */
    const int64_t one = (int64_t)1 << frac;
    return (sigmoid_fixed_c(x << 1, frac) << 1) - one;
}

static int64_t gelu_tanh_c(int64_t x, int frac) {             /* == gemma4.gelu_tanh_fixed */
    const int64_t one = (int64_t)1 << frac;
    int64_t x2 = (int64_t)(((__int128)x * x) >> frac);
    int64_t x3 = (int64_t)(((__int128)x2 * x) >> frac);
    int64_t cube = sdiv_i128((__int128)44715 * x3, 1000000);          /* 0.044715·x³ */
    int64_t inner = x + cube;
    int64_t arg = sdiv_i128((__int128)7978846 * inner, 10000000);     /* √(2/π)·inner */
    int64_t t = tanh_fixed_c(arg, frac);
    int64_t half = (one + t) >> 1;
    return (int64_t)(((__int128)x * half) >> frac);
}

/* out[i] = (gelu_tanh(g[i]) * u[i]) >> frac — the whole gated-FFN activation in one block call. */
void gelu_mul_block(int64_t * restrict out, const int64_t * restrict g,
                    const int64_t * restrict u, long n, int frac) {
    for (long i = 0; i < n; i++)
        out[i] = (int64_t)(((__int128)gelu_tanh_c(g[i], frac) * u[i]) >> frac);
}

/* Elementwise sigmoid block — == ract.sigmoid_fixed for EVERY input (the divisor saturation is
 * exactly the Python clamp's fixed point on both tails). One call gates a whole token batch. */
void sigmoid_block(int64_t * restrict out, const int64_t * restrict x, long n, int frac) {
    for (long i = 0; i < n; i++)
        out[i] = sigmoid_fixed_c(x[i], frac);
}

/* Elementwise exp block — == ract.exp_fixed on the softmax domain x <= 0 (where e^x is a
 * PURE DIVISOR result <= one, so the saturation equivalence holds; positive args would need
 * Python's arbitrary-precision growth and are the caller's job to exclude). */
void exp_block(int64_t * restrict out, const int64_t * restrict x, long n, int frac) {
    for (long i = 0; i < n; i++)
        out[i] = exp_fixed_c(x[i], frac);
}

static uint64_t isqrt_c(unsigned __int128 m) {                /* == rn.isqrt, digit-by-digit */
    if (m == 0) return 0;
    unsigned __int128 x = 0, c = 1;
    while (c <= (m >> 2)) c <<= 2;    /* wrap-proof: (c << 2) <= m overflows to 0 for m >= 2^126
                                       * and spins forever; shifting m down instead cannot wrap */
    while (c != 0) {
        if (m >= x + c) { m -= x + c; x = (x >> 1) + c; }
        else x >>= 1;
        c >>= 2;
    }
    return (uint64_t)x;
}

/* RMSNorm block: x / isqrt(mean(x²)+eps) · w, all Q<frac> — == ract.rmsnorm_fixed. */
void rmsnorm_block(int64_t * restrict out, const int64_t * restrict x,
                   const int64_t * restrict w, long n, int frac, int64_t eps) {
    __int128 ssq = 0;                                         /* int64 wrapped on huge-but-legit
                                                               * Q<frac> activations (|x| ~ 2^45,
                                                               * Soliton y_prenorm); exact to |x|
                                                               * < 2^58 at n <= 2^7 — host guards */
    for (long i = 0; i < n; i++)
        ssq += ((__int128)x[i] * x[i]) >> frac;               /* per-element shift, then sum */
    __int128 ms = ssq / n + eps;                              /* ssq >= 0: floor == trunc */
    int64_t rms = (int64_t)isqrt_c((unsigned __int128)ms << frac);
    if (rms == 0) rms = 1;
    const int64_t one = (int64_t)1 << frac;
    for (long i = 0; i < n; i++) {
        int64_t norm = sdiv_i128((__int128)x[i] * one, rms);  /* == (x<<frac)/rms, sign-safe */
        out[i] = sdiv_i128((__int128)norm * w[i], one);
    }
}

/* ── Hardware-* BRIDGE variant of the exact GEMV (kit precedent: ring_gemm's 3 gated variants —
 * bridge / shiftadd / QSM table). With a hardware multiply the exact integer dot needs NO digit
 * decomposition: out[r] = sd(floor((Σ_k (xbar-128)·x_k) · 2^s_row[r]), z_row[r]) directly — ONE
 * sweep, auto-vectorizable, BIT-IDENTICAL to the QSM digit path by construction (both equal the
 * exact dot; gated against it in the load selftest). QSM remains the silicon/reference path. */
typedef struct {
    int64_t *out; const uint8_t *xbar; const int64_t *x; const int32_t *x32; long K;
    const int32_t *s_row; const int64_t *z_row; int frac;
    long r0, r1;
} bridge_job;

static void bridge_rows(bridge_job *j) {
    /* int32 fast path (x32 != NULL): max|x| < 2^31 holds for every real Q16 activation, so
     * products are <= 2^38 and the int64 accumulator is exact for K <= 2^25 — and the widening
     * u8->i32 multiply-accumulate VECTORIZES (NEON smlal / AVX2 pmuldq). Four partial
     * accumulators keep the MLA pipes full. Falls back to the scalar __int128 accumulator when
     * the range check fails (same exact dot). */
    const long K = j->K;
    for (long r = j->r0; r < j->r1; r++) {
        const uint8_t * restrict w = j->xbar + (size_t)r * K;
        __int128 acc;
        if (j->x32) {
            const int32_t * restrict x = j->x32;
            int64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            long k = 0;
            for (; k + 4 <= K; k += 4) {
                a0 += (int64_t)((int32_t)w[k]     - 128) * x[k];
                a1 += (int64_t)((int32_t)w[k + 1] - 128) * x[k + 1];
                a2 += (int64_t)((int32_t)w[k + 2] - 128) * x[k + 2];
                a3 += (int64_t)((int32_t)w[k + 3] - 128) * x[k + 3];
            }
            for (; k < K; k++)
                a0 += (int64_t)((int32_t)w[k] - 128) * x[k];
            acc = (a0 + a1) + (a2 + a3);
        } else {
            const int64_t * restrict x = j->x;
            acc = 0;
            for (long k = 0; k < K; k++)
                acc += (__int128)((int64_t)w[k] - 128) * x[k];
        }
        long s = j->s_row[r];
        __int128 t = s >= 0 ? (acc << s) : (acc >> (-s));
        int64_t z = j->z_row[r] ? j->z_row[r] : 1;
        j->out[r] = (int64_t)(t < 0 ? -((-t) / z) : t / z);
    }
}

/* Narrow Q<frac> int64 activations to int32 for the vectorized/GPU paths. Returns 1 when every
 * element fits (products <= 2^38 -> int64/long accumulators exact), else 0 (caller falls back). */
long rk_narrow32(const int64_t * restrict x, int32_t * restrict out, long K) {
    for (long k = 0; k < K; k++) {
        int64_t av = x[k] < 0 ? -x[k] : x[k];
        if (av >= ((int64_t)1 << 31)) return 0;
        out[k] = (int32_t)x[k];
    }
    return 1;
}

/* ── Resident-activation helpers: the hidden vector lives in C buffers between blocks ──────── */

void add_into(int64_t * restrict dst, const int64_t * restrict a,
              const int64_t * restrict b, long n) {           /* dst = a + b (residual add) */
    for (long i = 0; i < n; i++) dst[i] = a[i] + b[i];
}

void scale_q16(int64_t * restrict h, int64_t sc, long n, int frac) {  /* h = (h·sc) >> frac */
    for (long i = 0; i < n; i++)
        h[i] = (int64_t)(((__int128)h[i] * sc) >> frac);
}

/* RMSNorm over `rows` independent rows of length n sharing gamma w (per-head Q/K/V norms use
 * rows = n_heads; the full-hidden norms use rows = 1). == ract.rmsnorm_fixed per row. */
void rmsnorm_block(int64_t * restrict out, const int64_t * restrict x,
                   const int64_t * restrict w, long n, int frac, int64_t eps);
void rmsnorm_rows(int64_t * restrict out, const int64_t * restrict x,
                  const int64_t * restrict w, long rows, long n, int frac, int64_t eps) {
    for (long r = 0; r < rows; r++)
        rmsnorm_block(out + r * n, x + r * n, w, n, frac, eps);
}

/* Embedding row: f16 -> Q<frac>, scaled by esc (Q<frac>): out[i] = (f16_fixed·esc) >> frac. */
static inline int64_t f16_fixed(uint16_t h, int shift);
void embed_row_block(int64_t * restrict out, const uint16_t * restrict row,
                     long n, int64_t esc, int frac) {
    for (long i = 0; i < n; i++)
        out[i] = (int64_t)(((__int128)f16_fixed(row[i], frac) * esc) >> frac);
}

static void *bridge_worker(void *arg) { bridge_rows((bridge_job *)arg); return NULL; }

long qsm_gemv_bridge_mt(int64_t * restrict out, const uint8_t * restrict xbar,
                        const int64_t * restrict x, long M, long K,
                        const int32_t * restrict s_row, const int64_t * restrict z_row,
                        int frac, int32_t * restrict x32_scratch, long nthreads) {
    /* one range scan; narrow x to the vectorizable int32 form when every element fits */
    const int32_t *x32 = NULL;
    if (x32_scratch) {
        int fits = 1;
        for (long k = 0; k < K; k++) {
            int64_t av = x[k] < 0 ? -x[k] : x[k];
            if (av >= ((int64_t)1 << 31)) { fits = 0; break; }
        }
        if (fits) {
            for (long k = 0; k < K; k++) x32_scratch[k] = (int32_t)x[k];
            x32 = x32_scratch;
        }
    }
    if (nthreads > M) nthreads = M;
    if (nthreads <= 1) {
        bridge_job j = {out, xbar, x, x32, K, s_row, z_row, frac, 0, M};
        bridge_rows(&j);
        return 1;
    }
    if (nthreads > 64) nthreads = 64;
    pthread_t tid[64];
    bridge_job job[64];
    long chunk = (M + nthreads - 1) / nthreads;
    long nt = 0;
    for (long t = 0; t < nthreads; t++) {
        long r0 = t * chunk, r1 = r0 + chunk;
        if (r0 >= M) break;
        if (r1 > M) r1 = M;
        job[nt] = (bridge_job){out, xbar, x, x32, K, s_row, z_row, frac, r0, r1};
        if (pthread_create(&tid[nt], NULL, bridge_worker, &job[nt]) != 0) {
            bridge_rows(&job[nt]);
            continue;
        }
        nt++;
    }
    for (long t = 0; t < nt; t++) pthread_join(tid[t], NULL);
    return 1;
}

/* ── Attention + RoPE BLOCK kernels over C-resident KV slabs ─────────────────────────────────
 * Bit-for-bit replicas of emulation/gemma4.py::attention_g4 / apply_rope and emulation/infer.py::
 * dot / softmax. K/V live in C-owned slabs ([kv_head, cap, hd] int64, host-grown); one call
 * computes ALL query heads, thread-split over heads (disjoint ctx rows — merge-free). */

static int64_t dot_q16(const int64_t * restrict a, const int64_t * restrict b, long n, int frac) {
    __int128 acc = 0;                                 /* == infer.dot: exact Σ a·b, then >> frac */
    for (long i = 0; i < n; i++) acc += (__int128)a[i] * b[i];
    return (int64_t)(acc >> frac);
}

static void softmax_c(int64_t * restrict sw, long n, int frac) {   /* == infer.softmax, in place */
    int64_t m = sw[0];
    for (long j = 1; j < n; j++) if (sw[j] > m) m = sw[j];
    int64_t z = 0;
    for (long j = 0; j < n; j++) { sw[j] = exp_fixed_c(sw[j] - m, frac); z += sw[j]; }
    if (z == 0) z = 1;
    for (long j = 0; j < n; j++) sw[j] = (sw[j] << frac) / z;      /* nonneg: floor == trunc */
}

static void attn_head(int64_t * restrict ctx, const int64_t * restrict qh,
                      const int64_t * restrict kh, const int64_t * restrict vh,
                      long nkeys, long hd, int frac, int64_t * restrict sw) {
    for (long j = 0; j < nkeys; j++) sw[j] = dot_q16(qh, kh + j * hd, hd, frac);
    softmax_c(sw, nkeys, frac);
    for (long d = 0; d < hd; d++) {                   /* Σ w·v exact, ONE final >> frac */
        __int128 acc = 0;
        for (long j = 0; j < nkeys; j++) acc += (__int128)sw[j] * vh[j * hd + d];
        ctx[d] = (int64_t)(acc >> frac);
    }
}

typedef struct {
    int64_t *ctx; const int64_t *q; const int64_t *kslab; const int64_t *vslab;
    long nkv, hd, nkeys, cap, group; int frac; int64_t *sw;
    long h0, h1;
} attn_job;

static void *attn_worker(void *arg) {
    attn_job *j = (attn_job *)arg;
    for (long h = j->h0; h < j->h1; h++) {
        long kv = h / j->group;
        attn_head(j->ctx + h * j->hd, j->q + h * j->hd,
                  j->kslab + kv * j->cap * j->hd, j->vslab + kv * j->cap * j->hd,
                  j->nkeys, j->hd, j->frac, j->sw + h * j->nkeys);
    }
    return NULL;
}

/* ctx[nq*hd] <- attention over the KV slabs for ALL nq query heads. sw is nq*nkeys scratch. */
void attn_block(int64_t * restrict ctx, const int64_t * restrict q,
                const int64_t * restrict kslab, const int64_t * restrict vslab,
                long nq, long nkv, long hd, long nkeys, long cap, int frac,
                int64_t * restrict sw, long nthreads) {
    long group = nq / nkv;
    if (nthreads > nq) nthreads = nq;
    if (nthreads <= 1) {
        attn_job j = {ctx, q, kslab, vslab, nkv, hd, nkeys, cap, group, frac, sw, 0, nq};
        attn_worker(&j);
        return;
    }
    if (nthreads > 64) nthreads = 64;
    pthread_t tid[64];
    attn_job job[64];
    long chunk = (nq + nthreads - 1) / nthreads;
    long nt = 0;
    for (long t = 0; t < nthreads; t++) {
        long h0 = t * chunk, h1 = h0 + chunk;
        if (h0 >= nq) break;
        if (h1 > nq) h1 = nq;
        job[nt] = (attn_job){ctx, q, kslab, vslab, nkv, hd, nkeys, cap, group, frac, sw, h0, h1};
        if (pthread_create(&tid[nt], NULL, attn_worker, &job[nt]) != 0) {
            attn_worker(&job[nt]);
            continue;
        }
        nt++;
    }
    for (long t = 0; t < nt; t++) pthread_join(tid[t], NULL);
}

/* NeoX half-split RoPE over nh heads IN PLACE: pair (i, i+pair_off) for the n_rot rotated pairs;
 * dims outside the span untouched. == gemma4.apply_rope (rn.mul exact product, >> frac floor). */
void rope_block(int64_t * restrict vec, const int64_t * restrict cos_row,
                const int64_t * restrict sin_row, long n_rot, long pair_off,
                long nh, long hd, int frac) {
    for (long h = 0; h < nh; h++) {
        int64_t * restrict v = vec + h * hd;
        for (long i = 0; i < n_rot; i++) {
            int64_t v0 = v[i], v1 = v[i + pair_off], c = cos_row[i], s = sin_row[i];
            v[i]            = (int64_t)(((__int128)v0 * c) >> frac)
                            - (int64_t)(((__int128)v1 * s) >> frac);
            v[i + pair_off] = (int64_t)(((__int128)v0 * s) >> frac)
                            + (int64_t)(((__int128)v1 * c) >> frac);
        }
    }
}

/* ── Tied LM-head argmax over the f16 embedding table ────────────────────────────
 * logit[v] = softcap * tanh(dot(hidden, embed[v]) / softcap); softcap>0 is monotone, so
 * argmax_v logit == argmax_v dot(hidden, embed[v]). We therefore only need the raw dot's argmax.
 * embed is raw f16 (uint16), decoded to a fixed-point integer (value << SHIFT) by INTEGER bit ops
 * (no FPU); hidden is Q<frac> int32. A fixed linear scale of embed preserves the argmax ordering.
 * This is the D9 hardware-bridge variant (hardware * permitted in kernels), self-tested bit-for-bit. */
static inline int64_t f16_fixed(uint16_t h, int shift) {
    int sign = (h >> 15) & 1;
    int exp  = (h >> 10) & 0x1F;
    int man  = h & 0x3FF;
    int64_t val;
    if (exp == 0) {                          /* subnormal: man * 2^-24 */
        int sh = shift - 24;
        val = sh >= 0 ? ((int64_t)man << sh) : ((int64_t)man >> (-sh));
    } else if (exp == 0x1F) {
        val = 0;                             /* inf/nan -> 0 (absent in weights) */
    } else {
        int64_t mant = (int64_t)((1 << 10) | man);   /* 1.man scaled by 1024 */
        int sh = shift + (exp - 15) - 10;            /* value = mant * 2^(exp-15) / 1024 */
        val = sh >= 0 ? (mant << sh) : (mant >> (-sh));
    }
    return sign ? -val : val;
}

/* Returns best token id; writes its raw dot to *best. hidden:[H] int32 Q<frac>, emb:[V*H] f16. */
long lm_argmax(const int32_t * restrict hidden, const uint16_t * restrict emb,
               long V, long H, int shift, int64_t * restrict best) {
    int64_t bs = (int64_t)1 << 62; bs = -bs;    /* -inf */
    long bi = 0;
    for (long v = 0; v < V; v++) {
        const uint16_t * restrict row = emb + (size_t)v * H;
        int64_t acc = 0;
        for (long i = 0; i < H; i++) acc += (int64_t)hidden[i] * f16_fixed(row[i], shift);
        if (acc > bs) { bs = acc; bi = v; }
    }
    *best = bs;
    return bi;
}

/* Streaming variant: the kernel mmaps embed.bin READ-ONLY itself (zero-copy, reclaimable page
 * cache — the on-device streaming path). Python holds no embedding memory. `off` is a byte offset
 * to the [V*H] f16 table within the file. Returns best id, or -1 on I/O error. */
long lm_argmax_file(const int32_t * restrict hidden, const char *path, long off,
                    long V, long H, int shift, int64_t * restrict best) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    size_t map_len = (size_t)off + (size_t)V * H * 2;
    if ((size_t)st.st_size < map_len) map_len = st.st_size;
    void *base = mmap(NULL, map_len, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) return -1;
    const uint16_t *emb = (const uint16_t *)((const uint8_t *)base + off);
    long bi = lm_argmax(hidden, emb, V, H, shift, best);
    munmap(base, map_len);
    return bi;
}
