// ringkit — Metal ring GEMM (D9 silicon): C = A(MxK) @ B(KxN) mod 256, row-major uchar.
// Two variants, mirroring ring_gemm.c and gated bit-for-bit against the same reference:
//   gemm_mul — hardware-`*` bridge (what the GPU's integer ALUs reward).
//   gemm_qsm — MULTIPLIER-FREE quarter-square form: two table reads + adds per product.
//              This measures the charter's bottleneck thesis on LUT-capable GPU fabric.
// One thread per output element; thread.x runs along j so B row reads coalesce.
#include <metal_stdlib>
using namespace metal;

struct GemmParams { uint M; uint K; uint N; };

kernel void gemm_mul(device uchar* C            [[buffer(0)]],
                     const device uchar* A      [[buffer(1)]],
                     const device uchar* B      [[buffer(2)]],
                     constant GemmParams& p     [[buffer(3)]],
                     const device ushort* qsq   [[buffer(4)]],   // unused here; uniform ABI
                     uint2 t [[thread_position_in_grid]]) {
    uint j = t.x, i = t.y;
    if (i >= p.M || j >= p.N) return;
    const device uchar* a = A + i * p.K;
    uchar acc = 0;
    for (uint k = 0; k < p.K; k++)
        acc = (uchar)(acc + (uchar)(a[k] * B[k * p.N + j]));
    C[i * p.N + j] = acc;
}

kernel void gemm_qsm(device uchar* C            [[buffer(0)]],
                     const device uchar* A      [[buffer(1)]],
                     const device uchar* B      [[buffer(2)]],
                     constant GemmParams& p     [[buffer(3)]],
                     const device ushort* qsq   [[buffer(4)]],   // q[t] = floor(t^2/4), t<=510
                     uint2 t [[thread_position_in_grid]]) {
    uint j = t.x, i = t.y;
    if (i >= p.M || j >= p.N) return;
    const device uchar* a = A + i * p.K;
    uchar acc = 0;
    for (uint k = 0; k < p.K; k++) {
        uint av = a[k], bv = B[k * p.N + j];
        uint s = av + bv;
        uint d = av > bv ? av - bv : bv - av;
        acc = (uchar)(acc + (uchar)(qsq[s] - qsq[d]));           // zero multiplies
    }
    C[i * p.N + j] = acc;
}
