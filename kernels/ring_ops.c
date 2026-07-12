/* ringkit silicon backend — elementwise ring ops on uint8 batches (mod 256).
 * Strict uint8 (no 64-bit promotion), restrict pointers -> compiler auto-vectorizes to 8-bit
 * SIMD (the QCM technique). SILICON layer (charter D9): uses hardware ops on purpose, validated
 * bit-for-bit against the multiplier-free ring semantics in ringkit.core.native.
 *
 * Two kernel families:
 *   ring_*      : straight loop (compiler auto-unrolls/vectorizes under -O3 -march=native)
 *   ring_*_u64  : EXPLICIT 64-lane unroll. The ring is 256 = 4 x 64, so a 256-batch is 4 fixed
 *                 64-wide passes (compile-time trip count) — the Julia unroll-by-64 x4 pattern.
 * Data buffers are owned by the caller (a Python bytearray's C memory, passed zero-copy). */
#include <stdint.h>

#define OP_MUL(x, y) ((uint8_t)((x) * (y)))
#define OP_ADD(x, y) ((uint8_t)((x) + (y)))
#define OP_SUB(x, y) ((uint8_t)((x) - (y)))

#define DEFINE_OP(NAME, OP)                                                                     \
void NAME(uint8_t * restrict c, const uint8_t * restrict a, const uint8_t * restrict b, long n) {\
    for (long i = 0; i < n; i++) c[i] = OP(a[i], b[i]);                                          \
}                                                                                               \
void NAME##_u64(uint8_t * restrict c, const uint8_t * restrict a, const uint8_t * restrict b, long n) {\
    long i = 0;                                                                                 \
    for (; i + 64 <= n; i += 64) {              /* fixed 64-lane chunk, unrolled */             \
        for (int j = 0; j < 64; j++) c[i + j] = OP(a[i + j], b[i + j]);                         \
    }                                                                                           \
    for (; i < n; i++) c[i] = OP(a[i], b[i]);   /* remainder */                                 \
}

DEFINE_OP(ring_mul, OP_MUL)
DEFINE_OP(ring_add, OP_ADD)
DEFINE_OP(ring_sub, OP_SUB)
