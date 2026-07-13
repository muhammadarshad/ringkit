/* ringkit silicon — the ADI KV element: ARC-only encode/decode (charter D9).
 *
 * ARC vs VALUE (RingTopology): the ARC POSITION is the angle/degree (256 ~ 360 deg) and NEVER
 * leaves 0..255 — so this whole kernel is uint8 mod-256. Energy (distance / SI units) does NOT
 * enter here; it only appears on the SCORING side (kv_scores, signed long). ADI is an arc-space
 * re-basis: differential (forward differences) and its inverse, accumulation (prefix sum).
 *
 *   encode:  delta[i] = row[i+1] - row[i]           (mod 256)   lead = row[0]
 *   decode:  row[0] = lead;  row[i] = row[i-1] + delta[i-1]     (mod 256)   (Fundamental Thm of Ring Calculus)
 *
 * Exact & reversible for ANY dimension N. Batched over a prime-pitched slab of R rows (token j at
 * offset j*pitch, pad bytes never touched — the QCM cache manifold). SILICON layer: hardware * and
 * - on purpose; must reproduce ringkit.ml.kvadi BIT-FOR-BIT (host self-tests at load). */
#include <stdint.h>

/* One row -> (lead, delta[0..dim-2]). ARC, mod 256. */
void adi_encode_batch(uint8_t * restrict leads, uint8_t * restrict deltas,
                      const uint8_t * restrict rows, long R, long dim, long pitch) {
    for (long r = 0; r < R; r++) {
        const uint8_t * restrict row = rows + r * pitch;
        uint8_t * restrict d = deltas + r * pitch;
        leads[r] = row[0];
        for (long i = 0; i + 1 < dim; i++) d[i] = (uint8_t)(row[i + 1] - row[i]);
    }
}

/* (lead, delta) -> row, by accumulation. ARC, mod 256. Exact inverse of adi_encode_batch. */
void adi_decode_batch(uint8_t * restrict rows, const uint8_t * restrict leads,
                      const uint8_t * restrict deltas, long R, long dim, long pitch) {
    for (long r = 0; r < R; r++) {
        uint8_t * restrict row = rows + r * pitch;
        const uint8_t * restrict d = deltas + r * pitch;
        row[0] = leads[r];
        for (long i = 1; i < dim; i++) row[i] = (uint8_t)(row[i - 1] + d[i - 1]);
    }
}
