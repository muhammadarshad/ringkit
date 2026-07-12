/* ringkit SU(256) gauge engine core — the Wilson plaquette action on a Z256 lattice.
 * energy[i,j,k] = (grid[i,j,k] + grid[i+1,j,k]) - (grid[i,j+1,k] + grid[i-1,j,k])  (mod 256)
 * i.e. plaquette = (right + up) - (left + down), pure uint8 (the SU(256) closed group).
 * Row-major index c = (k*H + j)*W + i, so i is stride-1 (the SIMD axis), i+1->c+1, i-1->c-1, j+1->c+W.
 * SILICON layer (charter D9): hardware uint8 ops, validated bit-for-bit vs the ring reference.
 *
 * plaquette_blocked adds CACHE BLOCKING: the depth (k) loop is tiled in 64-slabs so each slab's
 * working set stays L2-resident (the Julia "Depth==256 -> 4 x 64" unroll). 256 = 4 x 64.
 *
 * MULTITHREADING (the *_mt entry points): the lattice bins are PREDICTABLE — checkerboard
 * parity means same-parity sites never neighbor, and slab boundaries only READ opposite-parity
 * values — so k-slabs are statically partitioned across threads with no locks and no merge
 * step. Each thread completes its own bin; the only barrier is the parity edge (pthread_join).
 * Threaded results are BIT-IDENTICAL to single-threaded (gated in tests). */
#include <stdint.h>
#include <pthread.h>

/* ── plaquette ──────────────────────────────────────────────────────────── */

static inline void _slab(uint8_t * restrict e, const uint8_t * restrict g,
                         long W, long H, long k0, long k1) {
    for (long k = k0; k < k1; k++)
        for (long j = 1; j < H - 1; j++) {
            long base = (k * H + j) * W;
            for (long i = 1; i < W - 1; i++) {
                long c = base + i;
                uint8_t pos = (uint8_t)(g[c] + g[c + 1]);      /* right + up   */
                uint8_t neg = (uint8_t)(g[c + W] + g[c - 1]);  /* left + down  */
                e[c] = (uint8_t)(pos - neg);
            }
        }
}

void plaquette(uint8_t * restrict e, const uint8_t * restrict g, long W, long H, long D) {
    _slab(e, g, W, H, 1, D - 1);
}

void plaquette_blocked(uint8_t * restrict e, const uint8_t * restrict g, long W, long H, long D) {
    for (long kb = 1; kb < D - 1; kb += 64) {           /* 64-depth cache tiles (lock working set) */
        long kmax = kb + 64 < D - 1 ? kb + 64 : D - 1;
        _slab(e, g, W, H, kb, kmax);
    }
}

/* ── Metropolis sweep (checkerboard) ────────────────────────────────────── */

/* circular (ring L1) distance min(|a-b|, 256-|a-b|) — the U(1) local action term */
static inline int _cdist(uint8_t a, uint8_t b) {
    int d = (int)((a - b) & 0xFF);
    int e = 256 - d;
    return d < e ? d : e;
}

/* Counter-based per-node RNG (rk_mix32): randoms are DERIVED from (seed, sweep, node index),
 * never stored or transferred. Spec (identical in the Python reference and the Metal shader —
 * bit-for-bit gated):
 *   x = (idx + (sweep+1) * 0x9E3779B9) mod 2^32
 *   x ^= (seed * 0x85EBCA6B) mod 2^32
 *   x ^= x>>16;  x *= 0x7FEB352D;  x ^= x>>15;  x *= 0x846CA68B;  x ^= x>>16   (lowbias32)
 *   prop = x & 0xFF;  chance = (x >> 8) & 0xFF */
static inline uint32_t rk_mix32(uint32_t seed, uint32_t sweep, uint32_t idx) {
    uint32_t x = idx + (sweep + 1u) * 0x9E3779B9u;
    x ^= seed * 0x85EBCA6Bu;
    x ^= x >> 16;  x *= 0x7FEB352Du;
    x ^= x >> 15;  x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

/* one parity pass over k in [k0, k1) — the single core both st and mt paths run.
 * rng != 0 -> derive prop/chance via rk_mix32(seed, sweep, c); else read the arrays. */
static void _sweep_range(uint8_t * restrict grid, const uint8_t * restrict prop,
                         const uint8_t * restrict chance, const uint8_t * restrict lut,
                         long W, long H, int parity, long k0, long k1,
                         int rng, uint32_t seed, uint32_t sweep) {
    long sk = W * H;
    for (long k = k0; k < k1; k++)
        for (long j = 1; j < H - 1; j++) {
            long base = k * sk + j * W;
            for (long i = 1; i < W - 1; i++) {
                long c = base + i;
                if ((int)((i + j + k) & 1) != parity) continue;
                uint8_t pr, ch;
                if (rng) {
                    uint32_t x = rk_mix32(seed, sweep, (uint32_t)c);
                    pr = (uint8_t)(x & 0xFF);
                    ch = (uint8_t)((x >> 8) & 0xFF);
                } else {
                    pr = prop[c];
                    ch = chance[c];
                }
                uint8_t old = grid[c];
                uint8_t nv = (uint8_t)(old + pr);
                uint8_t r = grid[c + 1], l = grid[c - 1];
                uint8_t u = grid[c + W], dn = grid[c - W];
                uint8_t f = grid[c + sk], bk = grid[c - sk];
                int So = _cdist(old, r) + _cdist(old, l) + _cdist(old, u) + _cdist(old, dn) + _cdist(old, f) + _cdist(old, bk);
                int Sn = _cdist(nv, r) + _cdist(nv, l) + _cdist(nv, u) + _cdist(nv, dn) + _cdist(nv, f) + _cdist(nv, bk);
                int dS = Sn - So;
                int accept = (dS <= 0) || (ch < lut[dS > 255 ? 255 : dS]);
                grid[c] = accept ? nv : old;
            }
        }
}

void metropolis_sweep(uint8_t * restrict grid, const uint8_t * restrict prop,
                      const uint8_t * restrict chance, const uint8_t * restrict lut,
                      long W, long H, long D, int parity) {
    _sweep_range(grid, prop, chance, lut, W, H, parity, 1, D - 1, 0, 0, 0);
}

void metropolis_sweep_rng(uint8_t * restrict grid, uint32_t seed, uint32_t sweep,
                          const uint8_t * restrict lut,
                          long W, long H, long D, int parity) {
    _sweep_range(grid, 0, 0, lut, W, H, parity, 1, D - 1, 1, seed, sweep);
}

/* ── threaded entry points: static k-slab bins, no locks, no merge ──────── */

typedef struct {
    uint8_t *grid;
    const uint8_t *prop, *chance, *lut;
    long W, H, k0, k1;
    int parity, rng;
    uint32_t seed, sweep;
} rk_job;

static void *_sweep_worker(void *arg) {
    rk_job *j = (rk_job *)arg;
    _sweep_range(j->grid, j->prop, j->chance, j->lut, j->W, j->H,
                 j->parity, j->k0, j->k1, j->rng, j->seed, j->sweep);
    return (void *)0;
}

#define RK_MAX_THREADS 64

static void _sweep_mt(uint8_t *grid, const uint8_t *prop, const uint8_t *chance,
                      const uint8_t *lut, long W, long H, long D, int parity,
                      int nthreads, int rng, uint32_t seed, uint32_t sweep) {
    long lo = 1, hi = D - 1, span = hi - lo;
    if (nthreads > RK_MAX_THREADS) nthreads = RK_MAX_THREADS;
    if (nthreads > span) nthreads = (int)span;
    if (nthreads <= 1) {
        _sweep_range(grid, prop, chance, lut, W, H, parity, lo, hi, rng, seed, sweep);
        return;
    }
    pthread_t tid[RK_MAX_THREADS];
    rk_job job[RK_MAX_THREADS];
    long chunk = span / nthreads, rem = span % nthreads, k = lo;
    for (int t = 0; t < nthreads; t++) {
        long len = chunk + (t < rem ? 1 : 0);
        job[t] = (rk_job){grid, prop, chance, lut, W, H, k, k + len,
                          parity, rng, seed, sweep};
        k += len;
        pthread_create(&tid[t], 0, _sweep_worker, &job[t]);
    }
    for (int t = 0; t < nthreads; t++)
        pthread_join(tid[t], 0);                         /* the parity barrier */
}

void metropolis_sweep_mt(uint8_t *grid, const uint8_t *prop, const uint8_t *chance,
                         const uint8_t *lut, long W, long H, long D, int parity,
                         int nthreads) {
    _sweep_mt(grid, prop, chance, lut, W, H, D, parity, nthreads, 0, 0, 0);
}

void metropolis_sweep_rng_mt(uint8_t *grid, uint32_t seed, uint32_t sweep,
                             const uint8_t *lut, long W, long H, long D, int parity,
                             int nthreads) {
    _sweep_mt(grid, 0, 0, lut, W, H, D, parity, nthreads, 1, seed, sweep);
}

typedef struct {
    uint8_t *e;
    const uint8_t *g;
    long W, H, k0, k1;
} rk_pjob;

static void *_plaq_worker(void *arg) {
    rk_pjob *j = (rk_pjob *)arg;
    _slab(j->e, j->g, j->W, j->H, j->k0, j->k1);
    return (void *)0;
}

void plaquette_mt(uint8_t *e, const uint8_t *g, long W, long H, long D, int nthreads) {
    long lo = 1, hi = D - 1, span = hi - lo;
    if (nthreads > RK_MAX_THREADS) nthreads = RK_MAX_THREADS;
    if (nthreads > span) nthreads = (int)span;
    if (nthreads <= 1) {
        _slab(e, g, W, H, lo, hi);
        return;
    }
    pthread_t tid[RK_MAX_THREADS];
    rk_pjob job[RK_MAX_THREADS];
    long chunk = span / nthreads, rem = span % nthreads, k = lo;
    for (int t = 0; t < nthreads; t++) {
        long len = chunk + (t < rem ? 1 : 0);
        job[t] = (rk_pjob){e, g, W, H, k, k + len};
        k += len;
        pthread_create(&tid[t], 0, _plaq_worker, &job[t]);
    }
    for (int t = 0; t < nthreads; t++)
        pthread_join(tid[t], 0);
}
