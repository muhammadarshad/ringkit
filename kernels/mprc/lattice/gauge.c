/* ringkit SU(256) gauge engine core — the Wilson plaquette action on a Z256 lattice.
 * energy[i,j,k] = (grid[i,j,k] + grid[i+1,j,k]) - (grid[i,j+1,k] + grid[i-1,j,k])  (mod 256)
 * i.e. plaquette = (right + up) - (left + down), pure uint8 (the SU(256) closed group).
 * Row-major index c = (k*H + j)*W + i, so i is stride-1 (the SIMD axis), i+1->c+1, i-1->c-1, j+1->c+W.
 * SILICON layer (charter D9): hardware uint8 ops, validated bit-for-bit vs the ring reference.
 *
 * plaquette_blocked adds CACHE BLOCKING: the depth (k) loop is tiled in 64-slabs so each slab's
 * working set stays L2-resident (the Julia "Depth==256 -> 4 x 64" unroll). 256 = 4 x 64. */
#include <stdint.h>

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

/* circular (ring L1) distance min(|a-b|, 256-|a-b|) — the U(1) local action term */
static inline int _cdist(uint8_t a, uint8_t b) {
    int d = (int)((a - b) & 0xFF);
    int e = 256 - d;
    return d < e ? d : e;
}

/* One checkerboard Metropolis sweep of the ring U(1) gauge field.
 * Local action at a site = sum of ring-distance to its 6 neighbors (align-with-neighbors).
 * Propose new = old + prop[c]; dS = S_new - S_old; accept if dS<=0 (downhill) or
 * chance[c] < lut[dS] (uphill tunneling, integer Boltzmann LUT). Branchless store. Pure uint8.
 * parity selects the (i+j+k)&1 sublattice (checkerboard -> no data races / detailed balance). */
void metropolis_sweep(uint8_t * restrict grid, const uint8_t * restrict prop,
                      const uint8_t * restrict chance, const uint8_t * restrict lut,
                      long W, long H, long D, int parity) {
    long sk = W * H;
    for (long k = 1; k < D - 1; k++)
        for (long j = 1; j < H - 1; j++) {
            long base = k * sk + j * W;
            for (long i = 1; i < W - 1; i++) {
                long c = base + i;
                if ((int)((i + j + k) & 1) != parity) continue;
                uint8_t old = grid[c];
                uint8_t nv = (uint8_t)(old + prop[c]);
                uint8_t r = grid[c + 1], l = grid[c - 1];
                uint8_t u = grid[c + W], dn = grid[c - W];
                uint8_t f = grid[c + sk], bk = grid[c - sk];
                int So = _cdist(old, r) + _cdist(old, l) + _cdist(old, u) + _cdist(old, dn) + _cdist(old, f) + _cdist(old, bk);
                int Sn = _cdist(nv, r) + _cdist(nv, l) + _cdist(nv, u) + _cdist(nv, dn) + _cdist(nv, f) + _cdist(nv, bk);
                int dS = Sn - So;
                int accept = (dS <= 0) || (chance[c] < lut[dS > 255 ? 255 : dS]);
                grid[c] = accept ? nv : old;
            }
        }
}

/* Counter-based per-node RNG (rk_mix32): randoms are DERIVED from (seed, sweep, node index),
 * never stored or transferred — the unified-GPU thermalize needs only grid + LUT on the bus.
 * Spec (identical in the Python reference and the Metal shader — bit-for-bit gated):
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

void metropolis_sweep_rng(uint8_t * restrict grid, uint32_t seed, uint32_t sweep,
                          const uint8_t * restrict lut,
                          long W, long H, long D, int parity) {
    long sk = W * H;
    for (long k = 1; k < D - 1; k++)
        for (long j = 1; j < H - 1; j++) {
            long base = k * sk + j * W;
            for (long i = 1; i < W - 1; i++) {
                long c = base + i;
                if ((int)((i + j + k) & 1) != parity) continue;
                uint32_t x = rk_mix32(seed, sweep, (uint32_t)c);
                uint8_t pr = (uint8_t)(x & 0xFF);
                uint8_t ch = (uint8_t)((x >> 8) & 0xFF);
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
