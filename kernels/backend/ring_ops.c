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
#include <pthread.h>

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

/* Specialised-MPP block variant: the range [0,n) is split into disjoint contiguous blocks, one
 * per thread — lock-free, merge-free (outputs never overlap), bit-identical to the scalar loop.
 * numpy's uint8 ufunc is single-threaded, so one core is bandwidth-bound at ~one memory channel;
 * splitting the blocks across cores uses AGGREGATE memory bandwidth and beats it. Threshold-gated:
 * below MT_MIN elements the thread spawn costs more than the copy. */
#define EW_MT_MIN (1 << 18)
typedef struct { uint8_t *c; const uint8_t *a; const uint8_t *b; long lo, hi; int op; } _ew_job;

static void _ew_run(_ew_job *j) {
    uint8_t * restrict c = j->c; const uint8_t * restrict a = j->a; const uint8_t * restrict b = j->b;
    if (j->op == 0)      for (long i = j->lo; i < j->hi; i++) c[i] = OP_MUL(a[i], b[i]);
    else if (j->op == 1) for (long i = j->lo; i < j->hi; i++) c[i] = OP_ADD(a[i], b[i]);
    else                 for (long i = j->lo; i < j->hi; i++) c[i] = OP_SUB(a[i], b[i]);
}
static void *_ew_worker(void *arg) { _ew_run((_ew_job *)arg); return 0; }

/* op: 0 mul, 1 add, 2 sub. Disjoint blocks over C-owned memory, no merge. Per-call spawn — kept
 * for reference; on a bandwidth op the spawn cost dominates (measured). Prefer ring_ew_pool. */
void ring_ew_mt(uint8_t * restrict c, const uint8_t * restrict a, const uint8_t * restrict b,
                long n, int op, long nthreads) {
    if (nthreads <= 1 || n < EW_MT_MIN) {
        _ew_job j = {c, a, b, 0, n, op};
        _ew_run(&j);
        return;
    }
    if (nthreads > 64) nthreads = 64;
    pthread_t tid[64];
    _ew_job job[64];
    long chunk = (n + nthreads - 1) / nthreads;
    long k = 0;
    for (long t = 0; t < nthreads; t++) {
        long lo = t * chunk, hi = lo + chunk;
        if (lo >= n) break;
        if (hi > n) hi = n;
        job[k] = (_ew_job){c, a, b, lo, hi, op};
        if (pthread_create(&tid[k], 0, _ew_worker, &job[k]) != 0) {
            _ew_run(&job[k]);                 /* inline on spawn failure */
            continue;
        }
        k++;
    }
    for (long t = 0; t < k; t++) pthread_join(tid[t], 0);
}

/* ── PERSISTENT WORKER POOL: threads spawned ONCE, reused every call (torch/ATen pattern). ─────
 * The block-move done right for a bandwidth op: amortize spawn to zero, so what remains is the
 * aggregate multi-core DRAM bandwidth the specialised MPP is after. Blocks stay disjoint/merge-
 * free; a generation counter + two condvars form the dispatch/complete barrier. */
#define POOL_MAX 64
static struct {
    int n;                       /* workers (0 = uninitialized) */
    pthread_t th[POOL_MAX];
    pthread_mutex_t mx;
    pthread_cond_t go, done;
    long gen;                    /* dispatch generation; workers wake when this advances */
    int remaining;               /* workers still finishing this generation */
    /* current job broadcast to all workers */
    uint8_t *c; const uint8_t *a; const uint8_t *b; long n_elem; int op; long nblocks;
    long seen[POOL_MAX];         /* per-worker last gen processed */
} P;

static void *_pool_worker(void *arg) {
    long id = (long)arg;
    for (;;) {
        pthread_mutex_lock(&P.mx);
        while (P.seen[id] == P.gen) pthread_cond_wait(&P.go, &P.mx);
        long g = P.gen;
        uint8_t *c = P.c; const uint8_t *a = P.a; const uint8_t *b = P.b;
        long n = P.n_elem, nb = P.nblocks; int op = P.op;
        P.seen[id] = g;
        pthread_mutex_unlock(&P.mx);
        if (g < 0) return 0;                          /* shutdown sentinel */
        if (id < nb) {
            long chunk = (n + nb - 1) / nb;
            long lo = id * chunk, hi = lo + chunk;
            if (hi > n) hi = n;
            if (lo < n) { _ew_job j = {c, a, b, lo, hi, op}; _ew_run(&j); }
        }
        pthread_mutex_lock(&P.mx);
        if (--P.remaining == 0) pthread_cond_signal(&P.done);
        pthread_mutex_unlock(&P.mx);
    }
}

static void _pool_init(int nthreads) {
    if (nthreads > POOL_MAX) nthreads = POOL_MAX;
    pthread_mutex_init(&P.mx, 0);
    pthread_cond_init(&P.go, 0);
    pthread_cond_init(&P.done, 0);
    P.gen = 0;
    for (int i = 0; i < nthreads; i++) { P.seen[i] = 0; }
    P.n = nthreads;
    for (long i = 0; i < nthreads; i++) pthread_create(&P.th[i], 0, _pool_worker, (void *)i);
}

/* op: 0 mul, 1 add, 2 sub. Reuses the persistent pool; single-thread fallback below threshold. */
void ring_ew_pool(uint8_t * restrict c, const uint8_t * restrict a, const uint8_t * restrict b,
                  long n, int op, long nthreads) {
    if (nthreads <= 1 || n < EW_MT_MIN) {
        _ew_job j = {c, a, b, 0, n, op};
        _ew_run(&j);
        return;
    }
    if (P.n == 0) _pool_init((int)nthreads);
    pthread_mutex_lock(&P.mx);
    P.c = c; P.a = a; P.b = b; P.n_elem = n; P.op = op;
    P.nblocks = P.n < nthreads ? P.n : nthreads;
    P.remaining = P.n;                                /* every worker consumes the generation */
    P.gen++;
    pthread_cond_broadcast(&P.go);
    while (P.remaining > 0) pthread_cond_wait(&P.done, &P.mx);
    pthread_mutex_unlock(&P.mx);
}
