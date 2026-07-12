// ringkit — Metal shim (D9 silicon): a tiny ObjC layer exposing a C ABI for ctypes.
// Compiles the .metal sources at runtime (newLibraryWithSource — no metallib toolchain needed)
// and dispatches ring + gauge kernels. Buffers are copied in/out (bytearrays are not
// page-aligned, so no-copy MTLBuffer wrapping is not possible); the copy cost is measured
// and documented in tests, not hidden. The gauge sweep runs BOTH parities GPU-resident in
// one round trip (two ordered encoder passes), so the grid crosses the bus once per sweep.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string.h>

#define RK_ABI 6

static id<MTLDevice> g_dev = nil;
static id<MTLCommandQueue> g_queue = nil;
// 0 mul, 1 add, 2 sub, 3 plaquette, 4 metropolis_sweep, 5 metropolis_sweep_rng,
// 6 gemm_mul, 7 gemm_qsm
static id<MTLComputePipelineState> g_pso[8] = {nil};
static id<MTLBuffer> g_qsq = nil;           // quarter-square table q[t]=floor(t^2/4), t<=510

typedef struct { unsigned int W, H, D, parity; } RKGaugeParams;
typedef struct { unsigned int W, H, D, parity, seed, sweep; } RKRngParams;

int rk_metal_abi_version(void) { return RK_ABI; }

int rk_metal_init(const char *src_utf8) {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) {
            NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
            if (all.count == 0) return -1;
            g_dev = all[0];
        }
        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:src_utf8];
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) return -2;
        g_queue = [g_dev newCommandQueue];
        if (!g_queue) return -3;
        const char *names[8] = {"ring_mul", "ring_add", "ring_sub",
                                "plaquette", "metropolis_sweep", "metropolis_sweep_rng",
                                "gemm_mul", "gemm_qsm"};
        for (int k = 0; k < 8; k++) {
            id<MTLFunction> fn = [lib newFunctionWithName:
                                  [NSString stringWithUTF8String:names[k]]];
            if (!fn) return -4;
            g_pso[k] = [g_dev newComputePipelineStateWithFunction:fn error:&err];
            if (!g_pso[k]) return -5;
        }
        /* quarter-square table, built by odd-number accumulation (adds + shifts only) */
        g_qsq = [g_dev newBufferWithLength:511 * sizeof(unsigned short)
                                   options:MTLResourceStorageModeShared];
        if (!g_qsq) return -6;
        unsigned short *q = (unsigned short *)g_qsq.contents;
        unsigned int sq = 0;
        q[0] = 0;
        for (int t = 1; t <= 510; t++) {
            sq += (unsigned int)(t + t - 1);
            q[t] = (unsigned short)(sq >> 2);
        }
        return 0;
    }
}

typedef struct { unsigned int M, K, N; } RKGemmParams;

// variant: 0 = mul (hardware-* bridge), 1 = qsm (multiplier-free LUT). Returns 0 on success.
int rk_metal_gemm(int variant, unsigned char *C, const unsigned char *A,
                  const unsigned char *B, long M, long K, long N) {
    int pidx = variant == 0 ? 6 : 7;
    if (variant < 0 || variant > 1 || !g_pso[pidx] || M < 1 || K < 1 || N < 1) return -1;
    @autoreleasepool {
        id<MTLBuffer> ba = [g_dev newBufferWithBytes:A length:M * K
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> bb = [g_dev newBufferWithBytes:B length:K * N
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> bc = [g_dev newBufferWithLength:M * N
                                              options:MTLResourceStorageModeShared];
        if (!ba || !bb || !bc) return -2;
        RKGemmParams p = {(unsigned int)M, (unsigned int)K, (unsigned int)N};
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_pso[pidx]];
        [enc setBuffer:bc offset:0 atIndex:0];
        [enc setBuffer:ba offset:0 atIndex:1];
        [enc setBuffer:bb offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc setBuffer:g_qsq offset:0 atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)N, (NSUInteger)M, 1)
       threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return -3;
        memcpy(C, bc.contents, (size_t)(M * N));
        return 0;
    }
}

const char *rk_metal_device_name(void) {
    return g_dev ? g_dev.name.UTF8String : "";
}

// op: 0 = mul, 1 = add, 2 = sub. Returns 0 on success.
int rk_metal_elementwise(int op, unsigned char *out,
                         const unsigned char *a, const unsigned char *b, long n) {
    if (op < 0 || op > 2 || !g_pso[op] || n <= 0) return -1;
    @autoreleasepool {
        id<MTLBuffer> ba = [g_dev newBufferWithBytes:a length:n
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> bb = [g_dev newBufferWithBytes:b length:n
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo = [g_dev newBufferWithLength:n
                                              options:MTLResourceStorageModeShared];
        if (!ba || !bb || !bo) return -2;
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_pso[op]];
        [enc setBuffer:bo offset:0 atIndex:0];
        [enc setBuffer:ba offset:0 atIndex:1];
        [enc setBuffer:bb offset:0 atIndex:2];
        NSUInteger tg = g_pso[op].maxTotalThreadsPerThreadgroup;
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return -3;
        memcpy(out, bo.contents, (size_t)n);
        return 0;
    }
}

static void _dispatch3d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                        long W, long H, long D) {
    [enc dispatchThreads:MTLSizeMake((NSUInteger)(W - 2), (NSUInteger)(H - 2), (NSUInteger)(D - 2))
   threadsPerThreadgroup:MTLSizeMake(8, 8, 4)];
}

// Wilson plaquette over the interior. Returns 0 on success.
int rk_metal_plaquette(unsigned char *e, const unsigned char *g, long W, long H, long D) {
    long n = W * H * D;
    if (!g_pso[3] || W < 3 || H < 3 || D < 3) return -1;
    @autoreleasepool {
        id<MTLBuffer> bg = [g_dev newBufferWithBytes:g length:n
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> be = [g_dev newBufferWithLength:n
                                              options:MTLResourceStorageModeShared];
        if (!bg || !be) return -2;
        RKGaugeParams p = {(unsigned int)W, (unsigned int)H, (unsigned int)D, 0};
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_pso[3]];
        [enc setBuffer:be offset:0 atIndex:0];
        [enc setBuffer:bg offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        _dispatch3d(enc, g_pso[3], W, H, D);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return -3;
        memcpy(e, be.contents, (size_t)n);
        return 0;
    }
}

static int _thermalize_rng_on(id<MTLBuffer> bgrid, unsigned int seed, unsigned int sweep0,
                              const unsigned char *lut, long W, long H, long D, long sweeps) {
    if (!g_pso[5] || W < 3 || H < 3 || D < 3 || sweeps < 1) return -1;
    @autoreleasepool {
        id<MTLBuffer> blut = [g_dev newBufferWithBytes:lut length:256
                                               options:MTLResourceStorageModeShared];
        if (!blut) return -2;
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        for (long s = 0; s < sweeps; s++) {
            for (unsigned int parity = 0; parity < 2; parity++) {
                RKRngParams p = {(unsigned int)W, (unsigned int)H, (unsigned int)D,
                                 parity, seed, (unsigned int)(sweep0 + s)};
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:g_pso[5]];
                [enc setBuffer:bgrid offset:0 atIndex:0];
                [enc setBuffer:blut offset:0 atIndex:1];
                [enc setBytes:&p length:sizeof(p) atIndex:2];
                _dispatch3d(enc, g_pso[5], W, H, D);
                [enc endEncoding];
            }
        }
        [cb commit];
        [cb waitUntilCompleted];
        return cb.status == MTLCommandBufferStatusCompleted ? 0 : -3;
    }
}

// The unified-GPU endgame: a batch of sweeps whose randoms are DERIVED on-GPU (rk_mix32),
// so ONLY grid (n bytes, once each way) + lut (256 bytes) ever cross the bus. sweep0 is the
// starting sweep index (callers continue the counter across batches).
int rk_metal_thermalize_rng(unsigned char *grid, unsigned int seed, unsigned int sweep0,
                            const unsigned char *lut, long W, long H, long D, long sweeps) {
    long n = W * H * D;
    @autoreleasepool {
        id<MTLBuffer> bgrid = [g_dev newBufferWithBytes:grid length:n
                                                options:MTLResourceStorageModeShared];
        if (!bgrid) return -2;
        int rc = _thermalize_rng_on(bgrid, seed, sweep0, lut, W, H, D, sweeps);
        if (rc == 0) memcpy(grid, bgrid.contents, (size_t)n);
        return rc;
    }
}

/* ── PERSISTENT GPU SESSIONS: the lattice lives on the device across facade calls; the host
 *    copy syncs only on observable reads. Unified memory: contents ptr is host-coherent after
 *    waitUntilCompleted, so read/write are plain memcpy — no blit pass needed. ────────────── */

#define RK_MAX_SESSIONS 32
static id<MTLBuffer> g_sess[RK_MAX_SESSIONS];
static long g_sess_len[RK_MAX_SESSIONS];

long rk_metal_session_create(const unsigned char *grid, long n) {
    if (!g_dev || n < 1) return -1;
    for (long i = 0; i < RK_MAX_SESSIONS; i++) {
        if (g_sess[i] == nil) {
            g_sess[i] = [g_dev newBufferWithBytes:grid length:n
                                          options:MTLResourceStorageModeShared];
            if (!g_sess[i]) return -2;
            g_sess_len[i] = n;
            return i;
        }
    }
    return -3;                                        /* table full */
}

static int _sess_ok(long sid, long n) {
    return sid >= 0 && sid < RK_MAX_SESSIONS && g_sess[sid] != nil
        && (n < 0 || n == g_sess_len[sid]);
}

int rk_metal_session_thermalize_rng(long sid, unsigned int seed, unsigned int sweep0,
                                    const unsigned char *lut,
                                    long W, long H, long D, long sweeps) {
    if (!_sess_ok(sid, W * H * D)) return -1;
    return _thermalize_rng_on(g_sess[sid], seed, sweep0, lut, W, H, D, sweeps);
}

int rk_metal_session_read(long sid, unsigned char *out, long n) {
    if (!_sess_ok(sid, n)) return -1;
    memcpy(out, g_sess[sid].contents, (size_t)n);
    return 0;
}

int rk_metal_session_write(long sid, const unsigned char *grid, long n) {
    if (!_sess_ok(sid, n)) return -1;
    memcpy(g_sess[sid].contents, grid, (size_t)n);
    return 0;
}

int rk_metal_session_free(long sid) {
    if (sid < 0 || sid >= RK_MAX_SESSIONS) return -1;
    g_sess[sid] = nil;                                /* ARC releases the buffer */
    g_sess_len[sid] = 0;
    return 0;
}

// A BATCH of full sweeps, GPU-resident (unified memory): the grid crosses the bus once per
// batch instead of once per sweep, and all 2*sweeps parity passes queue in ONE command buffer.
// props/chances are concatenated per-sweep arrays (sweeps * n bytes each).
int rk_metal_thermalize(unsigned char *grid, const unsigned char *props,
                        const unsigned char *chances, const unsigned char *lut,
                        long W, long H, long D, long sweeps) {
    long n = W * H * D;
    if (!g_pso[4] || W < 3 || H < 3 || D < 3 || sweeps < 1) return -1;
    @autoreleasepool {
        id<MTLBuffer> bgrid = [g_dev newBufferWithBytes:grid length:n
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bprop = [g_dev newBufferWithBytes:props length:n * sweeps
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bchan = [g_dev newBufferWithBytes:chances length:n * sweeps
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> blut = [g_dev newBufferWithBytes:lut length:256
                                               options:MTLResourceStorageModeShared];
        if (!bgrid || !bprop || !bchan || !blut) return -2;
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        for (long s = 0; s < sweeps; s++) {
            for (unsigned int parity = 0; parity < 2; parity++) {
                RKGaugeParams p = {(unsigned int)W, (unsigned int)H, (unsigned int)D, parity};
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:g_pso[4]];
                [enc setBuffer:bgrid offset:0 atIndex:0];
                [enc setBuffer:bprop offset:(NSUInteger)(s * n) atIndex:1];
                [enc setBuffer:bchan offset:(NSUInteger)(s * n) atIndex:2];
                [enc setBuffer:blut offset:0 atIndex:3];
                [enc setBytes:&p length:sizeof(p) atIndex:4];
                _dispatch3d(enc, g_pso[4], W, H, D);
                [enc endEncoding];
            }
        }
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return -3;
        memcpy(grid, bgrid.contents, (size_t)n);
        return 0;
    }
}

// One FULL Metropolis sweep (parity 0 then 1) GPU-resident: grid crosses the bus once each way.
// Ordered encoder passes give the same parity-sequential semantics as the C kernel. Returns 0.
int rk_metal_gauge_sweep(unsigned char *grid, const unsigned char *prop,
                         const unsigned char *chance, const unsigned char *lut,
                         long W, long H, long D) {
    long n = W * H * D;
    if (!g_pso[4] || W < 3 || H < 3 || D < 3) return -1;
    @autoreleasepool {
        id<MTLBuffer> bgrid = [g_dev newBufferWithBytes:grid length:n
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bprop = [g_dev newBufferWithBytes:prop length:n
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bchan = [g_dev newBufferWithBytes:chance length:n
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> blut = [g_dev newBufferWithBytes:lut length:256
                                               options:MTLResourceStorageModeShared];
        if (!bgrid || !bprop || !bchan || !blut) return -2;
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        for (unsigned int parity = 0; parity < 2; parity++) {
            RKGaugeParams p = {(unsigned int)W, (unsigned int)H, (unsigned int)D, parity};
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:g_pso[4]];
            [enc setBuffer:bgrid offset:0 atIndex:0];
            [enc setBuffer:bprop offset:0 atIndex:1];
            [enc setBuffer:bchan offset:0 atIndex:2];
            [enc setBuffer:blut offset:0 atIndex:3];
            [enc setBytes:&p length:sizeof(p) atIndex:4];
            _dispatch3d(enc, g_pso[4], W, H, D);
            [enc endEncoding];
        }
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return -3;
        memcpy(grid, bgrid.contents, (size_t)n);
        return 0;
    }
}
