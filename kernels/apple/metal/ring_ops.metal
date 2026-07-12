// ringkit — Metal ring ops (D9 silicon). uchar arithmetic wraps mod 256 natively, which IS the
// ring semantics; validated bit-for-bit against the Python reference before serving.
#include <metal_stdlib>
using namespace metal;

kernel void ring_mul(device uchar* o        [[buffer(0)]],
                     const device uchar* a  [[buffer(1)]],
                     const device uchar* b  [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    o[i] = (uchar)(a[i] * b[i]);
}

kernel void ring_add(device uchar* o        [[buffer(0)]],
                     const device uchar* a  [[buffer(1)]],
                     const device uchar* b  [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    o[i] = (uchar)(a[i] + b[i]);
}

kernel void ring_sub(device uchar* o        [[buffer(0)]],
                     const device uchar* a  [[buffer(1)]],
                     const device uchar* b  [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    o[i] = (uchar)(a[i] - b[i]);
}
