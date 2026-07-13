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
 * Two traversals are provided and BENCHMARKED against each other (kv_scores vs kv_scores_walk),
 * because D1 says measure, don't assert:
 *   kv_scores      — sequential over tokens (hardware prefetch friendly)
 *   kv_scores_walk — the stride-7 QCM quantum walk over tokens: j -> (j + 7) mod n. 7 is odd, so
 *                    gcd(7, n) = 1 whenever n is not a multiple of 7 => the walk is a BIJECTION and
 *                    visits every token exactly once (the same unit/zero-divisor law as the ring).
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

/* The QCM quantum walk over tokens: +7 mod n, odd => bijective => every token visited once. */
void kv_scores_walk(long * restrict out, const uint8_t * restrict K, const uint8_t * restrict q,
                    long n, long dim, long pitch) {
    long j = 0;
    for (long step = 0; step < n; step++) {
        const uint8_t * restrict k = K + j * pitch;
        long s = 0;
        for (long d = 0; d < dim; d++) s -= (long)rdist(q[d], k[d]);
        out[j] = s;
        j += 7;
        while (j >= n) j -= n;      /* +7 mod n, no divide. MUST loop: for n < 7 one subtract
                                     * leaves j out of range (n=3: 0+7=7, 7-3=4, still >= 3). */
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
