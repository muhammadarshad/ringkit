// ringkit — Metal shim (D9 silicon): a tiny ObjC layer exposing a C ABI for ctypes.
// Compiles the .metal source at runtime (newLibraryWithSource — no metallib toolchain needed)
// and dispatches elementwise ring kernels. v1 copies buffers in/out (bytearrays are not
// page-aligned, so no-copy MTLBuffer wrapping is not possible); the copy cost is measured
// and documented in tests, not hidden.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string.h>

static id<MTLDevice> g_dev = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_pso[3] = {nil, nil, nil};   // mul, add, sub

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
        const char *names[3] = {"ring_mul", "ring_add", "ring_sub"};
        for (int k = 0; k < 3; k++) {
            id<MTLFunction> fn = [lib newFunctionWithName:
                                  [NSString stringWithUTF8String:names[k]]];
            if (!fn) return -4;
            g_pso[k] = [g_dev newComputePipelineStateWithFunction:fn error:&err];
            if (!g_pso[k]) return -5;
        }
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
