/* ringkit ring GEMM — C = A(MxK) @ B(KxN) mod 256, row-major uint8.
 * SILICON layer (charter D9), validated bit-for-bit vs the multiplier-free Python reference.
 *
 * THREE variants, because the charter's no-multiply rule is architectural (multipliers are
 * the silicon bottleneck) and the kernel campaign must MEASURE the thesis, not assume it:
 *   ring_gemm_mul      — hardware-`*` bridge: what today's commodity ALUs reward. D9 quarantine.
 *   ring_gemm_qsm      — QUARTER-SQUARE form: a*b = floor((a+b)^2/4) - floor((|a-b|)^2/4).
 *                        Two loads from a 511-entry uint16 table (1 KB, L1-resident) + adds.
 *                        ZERO multiplies at runtime; the table itself is built by odd-number
 *                        accumulation (adds + shifts only).
 *   ring_gemm_shiftadd — the ring's own mul (shift-and-add, rn.mul's form) hoisted: per k-row
 *                        the scalar A[i,k] is constant, so its set bits become whole-row
 *                        vector shift+add passes. ZERO multiplies, and it auto-vectorizes.
 *
 * Threading: rows of C are PREDICTABLE BINS — each thread owns a disjoint row range, writes
 * nothing shared, needs no merge. Bit-identical to single-threaded (gated in tests). */
#include <stdint.h>
#include <pthread.h>

/* ── quarter-square table: q[t] = floor(t*t/4), t in [0, 510] ─────────────── */

static uint16_t QSQ[511];
static int QSQ_READY = 0;

void ring_gemm_init(void) {                 /* adds + shifts only (odd-number accumulation) */
    uint32_t sq = 0;                        /* sq = t^2, built as sum of odd numbers        */
    QSQ[0] = 0;
    for (int t = 1; t <= 510; t++) {
        sq += (uint32_t)(t + t - 1);
        QSQ[t] = (uint16_t)(sq >> 2);
    }
    QSQ_READY = 1;
}

static inline uint8_t qsm_mul(uint8_t a, uint8_t b) {
    int s = (int)a + (int)b;
    int d = (int)a - (int)b;
    if (d < 0) d = -d;
    return (uint8_t)(QSQ[s] - QSQ[d]);
}

/* ── row-range cores (each thread runs one of these over its own rows) ────── */

static void _gemm_mul_rows(uint8_t * restrict C, const uint8_t * restrict A,
                           const uint8_t * restrict B, long M, long K, long N,
                           long i0, long i1) {
    (void)M;
    for (long i = i0; i < i1; i++) {
        uint8_t * restrict c = C + i * N;
        const uint8_t * restrict a = A + i * K;
        for (long j = 0; j < N; j++) c[j] = 0;
        for (long k = 0; k < K; k++) {
            uint8_t av = a[k];
            if (!av) continue;
            const uint8_t * restrict b = B + k * N;
            for (long j = 0; j < N; j++)
                c[j] = (uint8_t)(c[j] + (uint8_t)(av * b[j]));
        }
    }
}

static void _gemm_qsm_rows(uint8_t * restrict C, const uint8_t * restrict A,
                           const uint8_t * restrict B, long M, long K, long N,
                           long i0, long i1) {
    (void)M;
    for (long i = i0; i < i1; i++) {
        uint8_t * restrict c = C + i * N;
        const uint8_t * restrict a = A + i * K;
        for (long j = 0; j < N; j++) c[j] = 0;
        for (long k = 0; k < K; k++) {
            uint8_t av = a[k];
            if (!av) continue;
            const uint8_t * restrict b = B + k * N;
            for (long j = 0; j < N; j++)
                c[j] = (uint8_t)(c[j] + qsm_mul(av, b[j]));
        }
    }
}

static void _gemm_shiftadd_rows(uint8_t * restrict C, const uint8_t * restrict A,
                                const uint8_t * restrict B, long M, long K, long N,
                                long i0, long i1) {
    (void)M;
    for (long i = i0; i < i1; i++) {
        uint8_t * restrict c = C + i * N;
        const uint8_t * restrict a = A + i * K;
        for (long j = 0; j < N; j++) c[j] = 0;
        for (long k = 0; k < K; k++) {
            uint8_t av = a[k];
            if (!av) continue;
            const uint8_t * restrict b = B + k * N;
            /* av * b[j] mod 256 = sum over set bits s of av of (b[j] << s), all mod 256.
             * av is loop-constant, so each set bit is one whole-row vector shift+add pass. */
            for (int s = 0; s < 8; s++) {
                if (!(av & (1u << s))) continue;
                for (long j = 0; j < N; j++)
                    c[j] = (uint8_t)(c[j] + (uint8_t)(b[j] << s));
            }
        }
    }
}

/* ── single-thread entry points ───────────────────────────────────────────── */

void ring_gemm_mul(uint8_t *C, const uint8_t *A, const uint8_t *B, long M, long K, long N) {
    _gemm_mul_rows(C, A, B, M, K, N, 0, M);
}

void ring_gemm_qsm(uint8_t *C, const uint8_t *A, const uint8_t *B, long M, long K, long N) {
    if (!QSQ_READY) ring_gemm_init();
    _gemm_qsm_rows(C, A, B, M, K, N, 0, M);
}

void ring_gemm_shiftadd(uint8_t *C, const uint8_t *A, const uint8_t *B, long M, long K, long N) {
    _gemm_shiftadd_rows(C, A, B, M, K, N, 0, M);
}

/* ── threaded entry points: static row bins, no locks, no merge ───────────── */

typedef struct {
    uint8_t *C;
    const uint8_t *A, *B;
    long M, K, N, i0, i1;
    int variant;                            /* 0 mul, 1 qsm, 2 shiftadd */
} rk_gjob;

static void *_gemm_worker(void *arg) {
    rk_gjob *j = (rk_gjob *)arg;
    if (j->variant == 0)
        _gemm_mul_rows(j->C, j->A, j->B, j->M, j->K, j->N, j->i0, j->i1);
    else if (j->variant == 1)
        _gemm_qsm_rows(j->C, j->A, j->B, j->M, j->K, j->N, j->i0, j->i1);
    else
        _gemm_shiftadd_rows(j->C, j->A, j->B, j->M, j->K, j->N, j->i0, j->i1);
    return (void *)0;
}

#define RK_MAX_THREADS 64

static void _gemm_mt(uint8_t *C, const uint8_t *A, const uint8_t *B,
                     long M, long K, long N, int variant, int nthreads) {
    if (variant == 1 && !QSQ_READY) ring_gemm_init();
    if (nthreads > RK_MAX_THREADS) nthreads = RK_MAX_THREADS;
    if (nthreads > M) nthreads = (int)M;
    if (nthreads <= 1) {
        _gemm_worker(&(rk_gjob){C, A, B, M, K, N, 0, M, variant});
        return;
    }
    pthread_t tid[RK_MAX_THREADS];
    rk_gjob job[RK_MAX_THREADS];
    long chunk = M / nthreads, rem = M % nthreads, i = 0;
    for (int t = 0; t < nthreads; t++) {
        long len = chunk + (t < rem ? 1 : 0);
        job[t] = (rk_gjob){C, A, B, M, K, N, i, i + len, variant};
        i += len;
        pthread_create(&tid[t], 0, _gemm_worker, &job[t]);
    }
    for (int t = 0; t < nthreads; t++)
        pthread_join(tid[t], 0);
}

void ring_gemm_mul_mt(uint8_t *C, const uint8_t *A, const uint8_t *B,
                      long M, long K, long N, int nthreads) {
    _gemm_mt(C, A, B, M, K, N, 0, nthreads);
}

void ring_gemm_qsm_mt(uint8_t *C, const uint8_t *A, const uint8_t *B,
                      long M, long K, long N, int nthreads) {
    _gemm_mt(C, A, B, M, K, N, 1, nthreads);
}

void ring_gemm_shiftadd_mt(uint8_t *C, const uint8_t *A, const uint8_t *B,
                           long M, long K, long N, int nthreads) {
    _gemm_mt(C, A, B, M, K, N, 2, nthreads);
}
