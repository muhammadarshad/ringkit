/* QCM-style ring kernel — the "C backend" for ring_numpy.
 * Techniques from the QCM paper: strict uint8 (no 64-bit promotion), contiguous
 * L1-resident batch, restrict pointers -> compiler auto-vectorizes to 8-bit SIMD.
 * The SILICON layer; the ring SEMANTICS (multiplier-free) are the spec it implements. */
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define D 14464   /* 128 x 113 hypervector batch, ~14.5 KB, L1-resident */

static void ring_mul(uint8_t * restrict c, const uint8_t * restrict a,
                     const uint8_t * restrict b, int n) {
    for (int i = 0; i < n; i++) c[i] = (uint8_t)(a[i] * b[i]);  /* 8-bit, wraps mod 256 */
}

int main(void) {
    static uint8_t A[D], B[D], C[D];
    for (int i = 0; i < D; i++) { A[i] = (uint8_t)(i * 7 + 1); B[i] = (uint8_t)(i * 3 + 2); }
    long iters = 2000000L;
    volatile uint64_t sink = 0;                    /* consume results -> defeat dead-code elim */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long k = 0; k < iters; k++) {
        B[k & (D - 1)] ^= (uint8_t)k;              /* per-iteration data dependency */
        ring_mul(C, A, B, D);
        sink += C[k & (D - 1)];
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double u = (double)iters * D;
    printf("C backend: %.0f M updates in %.3f s -> %.1f MUPS (sink=%llu)\n",
           u / 1e6, sec, u / 1e6 / sec, (unsigned long long)sink);
    return 0;
}
