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
