// ringkit — Metal emulation GEMV (D9 silicon): the exact integer dot of the hardware-* bridge
// (kernels/mprc/gemma/qsm_energy.c::bridge_rows int32 path) on the unified GPU.
//   out[r] = sd( (Σ_k (onix[r*K+k]-128) · x[k]) · 2^s_row[r], z_row[r] )
// The onix weight slab is the file's own pages (no-copy MTLBuffer over the host mmap — unified
// memory, nothing materialized); x is the int32-narrowed activation vector (range-checked on the
// host: products ≤ 2^38, the long accumulator is exact — out-of-range falls back to CPU).
// One THREADGROUP per output row; lanes stride K (coalesced byte reads) and tree-reduce.
#include <metal_stdlib>
using namespace metal;

struct EmuGemvParams { uint M; uint K; int frac; };

kernel void emu_gemv(device long* out              [[buffer(0)]],
                     const device uchar* xbar      [[buffer(1)]],   // onix buffer @ tensor offset
                     const device int* x           [[buffer(2)]],
                     const device int* s_row       [[buffer(3)]],
                     const device long* z_row      [[buffer(4)]],
                     constant EmuGemvParams& p     [[buffer(5)]],
                     uint row  [[threadgroup_position_in_grid]],
                     uint lid  [[thread_position_in_threadgroup]],
                     uint tpg  [[threads_per_threadgroup]]) {
    threadgroup long partial[256];
    if (row >= p.M) return;
    const device uchar* w = xbar + (ulong)row * p.K;
    long acc = 0;
    for (uint k = lid; k < p.K; k += tpg)
        acc += (long)((int)w[k] - 128) * (long)x[k];
    partial[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg >> 1; s > 0; s >>= 1) {
        if (lid < s) partial[lid] += partial[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        long a = partial[0];
        int  s = s_row[row];
        long t = s >= 0 ? (a << s) : (a >> (-s));            // >> on long: arithmetic (floor)
        long z = z_row[row] ? z_row[row] : 1;
        out[row] = t < 0 ? -((-t) / z) : t / z;              // symmetric divide == _sd
    }
}
