/* ringkit silicon — KV cache scoring on the QCM CACHE MANIFOLD (charter D9).
 *
 * The cache is ONE contiguous uint8 slab, but it is NOT laid out on a power-of-two row stride.
 * kernels/mprc/qcm/cache_manifold.c already proved the silicon truth this repo is built on: a
 * POWER-OF-TWO leading dimension aliases successive rows into the SAME cache sets (conflict
 * misses), and a PRIME leading dimension does not. A KV cache is the worst possible offender —
 * dim is almost always 64/128/256, exactly the aliasing case — so token j lives at
 *
 *      K[j * pitch .. j * pitch + dim - 1],     pitch = the next PRIME >= dim
 *
 * The pad bytes are never read. This is the manifold, not a micro-optimisation: the ring's
 * traversal of memory is prime-strided, and laying the cache out on 2^k is what "violating QCM"
 * would actually mean.
 *
 * Traversal is SEQUENTIAL over tokens. A stride-7 "quantum walk" variant (j -> (j+7) mod n) was
 * built and measured against it per D1: identical scores (bijection), but ~2% SLOWER — the
 * hardware prefetcher wins a pure linear scan (commit 5f755df has the numbers). The QCM win in
 * this file is the MANIFOLD (prime pitch), not the hop order; the losing variant was removed
 * once its measurement was on record.
 *
 * SILICON layer: uses hardware * and - on purpose; must reproduce the multiplier-free semantic
 * reference BIT-FOR-BIT (host.py self-tests at load and refuses to serve on any disagreement). */
#include <stdint.h>

static inline int rdist(uint8_t a, uint8_t b) {
    int d = (int)(uint8_t)(a - b);
    int e = 256 - d;
    return d < e ? d : e;
}

/* score[j] = -sum_d rdist(q[d], K[j][d]) — signed ENERGY in long, so the ranking can never wrap. */
void kv_scores(long * restrict out, const uint8_t * restrict K, const uint8_t * restrict q,
               long n, long dim, long pitch) {
    for (long j = 0; j < n; j++) {
        const uint8_t * restrict k = K + j * pitch;      /* PRIME pitch: no cache-set aliasing */
        long s = 0;
        for (long d = 0; d < dim; d++) s -= (long)rdist(q[d], k[d]);
        out[j] = s;
    }
}

/* Circular value blend around the winner (the rest of the attend hot path, on the ENERGY side).
 * out[d] = ref[d] + trunc( sum_j w[j] * signed_offset(V[j][d], ref[d]) / sum_j w[j] )   (mod 256)
 * signed_offset lives in [-127,128]; the divide truncates toward zero, matching the Python
 * reference's sign-split mf_floordiv bit-for-bit. den > 0 is guaranteed by the caller (lut[0]=255);
 * guarded anyway. Values are ARC (uint8); the weighted sum is ENERGY (signed long, never folded). */
void kv_blend(uint8_t * restrict out, const uint8_t * restrict V, const long * restrict w,
              long n, long dim, long pitch, long best) {
    const uint8_t * restrict ref = V + best * pitch;
    long den = 0;
    for (long j = 0; j < n; j++) den += w[j];
    if (den <= 0) { for (long d = 0; d < dim; d++) out[d] = ref[d]; return; }
    for (long d = 0; d < dim; d++) {
        int rd = (int)ref[d];
        long num = 0;
        for (long j = 0; j < n; j++) {
            long wj = w[j];
            if (wj) {
                int off = (int)(uint8_t)(V[j * pitch + d] - rd);   /* signed_offset in [-127,128] */
                if (off > 128) off -= 256;
                num += wj * (long)off;                             /* qsm = exact product */
            }
        }
        out[d] = (uint8_t)(rd + num / den);                        /* trunc toward zero */
    }
}

/* Fused scan: score every key AND take the argmax in ONE pass, no intermediate array. */
long kv_argmax(const uint8_t * restrict K, const uint8_t * restrict q,
               long n, long dim, long pitch, long * restrict best_out) {
    long best_j = 0, best = 0;
    for (long j = 0; j < n; j++) {
        const uint8_t * restrict k = K + j * pitch;
        long s = 0;
        for (long d = 0; d < dim; d++) s -= (long)rdist(q[d], k[d]);
        if (j == 0 || s > best) { best = s; best_j = j; }
    }
    if (best_out) *best_out = best;
    return best_j;
}
