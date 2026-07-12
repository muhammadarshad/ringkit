// ringkit — Metal gauge kernels (D9 silicon): Wilson plaquette + checkerboard Metropolis
// sweep on the Z256 lattice. Semantics mirror kernels/mprc/lattice/gauge.c EXACTLY and are
// validated bit-for-bit against it before serving. Checkerboard parity = no same-parity
// neighbors, so one parity dispatch has no read/write hazards; parities run as ordered passes.
#include <metal_stdlib>
using namespace metal;

struct GaugeParams { uint W; uint H; uint D; uint parity; };

kernel void plaquette(device uchar* e            [[buffer(0)]],
                      const device uchar* g      [[buffer(1)]],
                      constant GaugeParams& p    [[buffer(2)]],
                      uint3 t [[thread_position_in_grid]]) {
    uint i = t.x + 1, j = t.y + 1, k = t.z + 1;
    if (i >= p.W - 1 || j >= p.H - 1 || k >= p.D - 1) return;
    uint c = (k * p.H + j) * p.W + i;
    uchar pos = (uchar)(g[c] + g[c + 1]);        // right + up
    uchar neg = (uchar)(g[c + p.W] + g[c - 1]);  // left + down
    e[c] = (uchar)(pos - neg);
}

static inline int cdist(uchar a, uchar b) {
    int d = (int)((uchar)(a - b));
    int e2 = 256 - d;
    return d < e2 ? d : e2;
}

kernel void metropolis_sweep(device uchar* grid          [[buffer(0)]],
                             const device uchar* prop    [[buffer(1)]],
                             const device uchar* chance  [[buffer(2)]],
                             const device uchar* lut     [[buffer(3)]],
                             constant GaugeParams& p     [[buffer(4)]],
                             uint3 t [[thread_position_in_grid]]) {
    uint i = t.x + 1, j = t.y + 1, k = t.z + 1;
    if (i >= p.W - 1 || j >= p.H - 1 || k >= p.D - 1) return;
    if (((i + j + k) & 1u) != p.parity) return;
    uint sk = p.W * p.H;
    uint c = k * sk + j * p.W + i;
    uchar old = grid[c];
    uchar nv = (uchar)(old + prop[c]);
    uchar r = grid[c + 1],    l  = grid[c - 1];
    uchar u = grid[c + p.W],  dn = grid[c - p.W];
    uchar f = grid[c + sk],   bk = grid[c - sk];
    int So = cdist(old, r) + cdist(old, l) + cdist(old, u)
           + cdist(old, dn) + cdist(old, f) + cdist(old, bk);
    int Sn = cdist(nv, r) + cdist(nv, l) + cdist(nv, u)
           + cdist(nv, dn) + cdist(nv, f) + cdist(nv, bk);
    int dS = Sn - So;
    bool accept = (dS <= 0) || (chance[c] < lut[dS > 255 ? 255 : dS]);
    grid[c] = accept ? nv : old;
}

// Counter-based per-node RNG — spec identical to rk_mix32 in gauge.c and the Python
// reference (bit-for-bit gated). Randoms are derived, never transferred: only grid + LUT
// cross the unified-memory bus for a whole thermalize batch.
struct RngParams { uint W; uint H; uint D; uint parity; uint seed; uint sweep; };

static inline uint rk_mix32(uint seed, uint sweep, uint idx) {
    uint x = idx + (sweep + 1u) * 0x9E3779B9u;
    x ^= seed * 0x85EBCA6Bu;
    x ^= x >> 16;  x *= 0x7FEB352Du;
    x ^= x >> 15;  x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

kernel void metropolis_sweep_rng(device uchar* grid          [[buffer(0)]],
                                 const device uchar* lut     [[buffer(1)]],
                                 constant RngParams& p       [[buffer(2)]],
                                 uint3 t [[thread_position_in_grid]]) {
    uint i = t.x + 1, j = t.y + 1, k = t.z + 1;
    if (i >= p.W - 1 || j >= p.H - 1 || k >= p.D - 1) return;
    if (((i + j + k) & 1u) != p.parity) return;
    uint sk = p.W * p.H;
    uint c = k * sk + j * p.W + i;
    uint x = rk_mix32(p.seed, p.sweep, c);
    uchar pr = (uchar)(x & 0xFFu);
    uchar ch = (uchar)((x >> 8) & 0xFFu);
    uchar old = grid[c];
    uchar nv = (uchar)(old + pr);
    uchar r = grid[c + 1],    l  = grid[c - 1];
    uchar u = grid[c + p.W],  dn = grid[c - p.W];
    uchar f = grid[c + sk],   bk = grid[c - sk];
    int So = cdist(old, r) + cdist(old, l) + cdist(old, u)
           + cdist(old, dn) + cdist(old, f) + cdist(old, bk);
    int Sn = cdist(nv, r) + cdist(nv, l) + cdist(nv, u)
           + cdist(nv, dn) + cdist(nv, f) + cdist(nv, bk);
    int dS = Sn - So;
    bool accept = (dS <= 0) || (ch < lut[dS > 255 ? 255 : dS]);
    grid[c] = accept ? nv : old;
}
