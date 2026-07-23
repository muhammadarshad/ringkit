// ringkit CUDA backend for GEVHV (D9 silicon) — ringkit's OWN port of the gevhv research
// kernels (G:\quantum\research\gevhv\kernels\gevhv_fused.cu is the WHAT; this is the HOW).
// D9 bar: every op here reproduces ringkit.ml.gevhv BIT-FOR-BIT (gated at host load by
// kernels/cuda/host.py._selftest). Hardware '*' is allowed in this layer (D9 two-layer rule);
// the absorbed LUT (L') and the vector-bind offset field (c) are precomputed ONCE, host-side,
// by calling ml.gevhv.absorb_lut / ml.gevhv.offset_field directly (the multiplier-free
// semantic reference) — this backend never re-derives that arithmetic, it only consumes it.
//
//   REACT           rho(g)_ij = lut[Sigma5(g)_ij mod 256] interior, IDENTITY boundary
//                   (== ml.gevhv.react)
//   REACT (scalar)  rho(phi_{s,t}(g)): interior via absorbed table L' (Theorem F, computed
//                   host-side), boundary via pointwise phi = s*g+t (the boundary clause)
//                   (== ml.gevhv.react_bound_scalar)
//   REACT (vector)  rho(g+v): interior via L[(Sigma5(g)+c) mod 256] with the shared offset
//                   field c = Sigma5(v) (Theorem F2, computed host-side), boundary via g+v
//                   (== ml.gevhv.react_bound_vector)
//   MEASURE         E_n = SUM_sites cdist(g_ij, q_ij), full grid or interior-only, ENERGY-
//                   accumulated in int32 (Theorem G) (== ml.gevhv.measure)
//   FUSED           react_bound_{scalar,vector} + measure in ONE pass, no intermediate
//                   manifold ever materialized (== ml.gevhv.gevhv_scalar / gevhv_vector)
//
// Batched: G is [N][H][W] u8, row-major, H=128 rows W=113 cols (prime — QCM geometry). The
// query field q and the LUT(s)/offset-field are broadcast across the whole batch (Theorem
// F2's batch independence). Persistent device buffers are NOT required for correctness —
// every host wrapper below does its own malloc/copy/launch/sync/copy-back/free; correctness
// first, no timing claims (CORRECTION_REFOCUS).
#include <cstdint>
#include <cuda_runtime.h>

typedef unsigned char u8;

__device__ __forceinline__ u8 cdist8(u8 a, u8 b) {
  int d = (int)(u8)(a - b);
  int e = 256 - d;
  return (u8)(d < e ? d : e);
}

// ---------------- react kernels (write the whole [N][H][W] manifold) ----------------

// identity-boundary react (== ml.gevhv.react). Flattened over N*H*W; interior sites never
// read across a manifold seam because W < H*W and i,j are re-derived from the LOCAL site
// offset (idx % sites), so idx-1/idx+1/idx-W/idx+W stay inside the same manifold for any
// site with 0 < i < W-1 and 0 < j < H-1.
__global__ void k_react(u8* out, const u8* G, const u8* lut, long H, long W, long N) {
  long sites = H * W, tot = N * sites;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tot) return;
  long site = idx % sites;
  int i = (int)(site % W), j = (int)(site / W);
  if (i == 0 || i == W - 1 || j == 0 || j == H - 1) { out[idx] = G[idx]; return; }
  unsigned s5 = (unsigned)G[idx] + G[idx - 1] + G[idx + 1] + G[idx - W] + G[idx + W];
  out[idx] = lut[s5 & 0xFF];
}

// scalar-bound react (== ml.gevhv.react_bound_scalar). Lp is the ALREADY-ABSORBED table
// L'[u] = L[(s*u + 5t) mod 256], computed host-side by ml.gevhv.absorb_lut. s, t are still
// needed here for the boundary clause's pointwise bind (hardware '*' — D9-legal in kernels).
__global__ void k_react_bound_scalar(u8* out, const u8* G, const u8* Lp, long H, long W, long N,
                                     int s, int t) {
  long sites = H * W, tot = N * sites;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tot) return;
  long site = idx % sites;
  int i = (int)(site % W), j = (int)(site / W);
  if (i == 0 || i == W - 1 || j == 0 || j == H - 1) {
    out[idx] = (u8)(s * (int)G[idx] + t);
    return;
  }
  unsigned s5 = (unsigned)G[idx] + G[idx - 1] + G[idx + 1] + G[idx - W] + G[idx + W];
  out[idx] = Lp[s5 & 0xFF];
}

// vector-bound react (== ml.gevhv.react_bound_vector). L is the RAW lut (no absorption for
// the vector form); v, c are per-site (length H*W), shared/broadcast across the whole batch —
// index by the LOCAL site, not the flat batch index.
__global__ void k_react_bound_vector(u8* out, const u8* G, const u8* L, const u8* v, const u8* c,
                                     long H, long W, long N) {
  long sites = H * W, tot = N * sites;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tot) return;
  long site = idx % sites;
  int i = (int)(site % W), j = (int)(site / W);
  if (i == 0 || i == W - 1 || j == 0 || j == H - 1) {
    out[idx] = (u8)((int)G[idx] + (int)v[site]);
    return;
  }
  unsigned s5 = (unsigned)G[idx] + G[idx - 1] + G[idx + 1] + G[idx - W] + G[idx + W];
  out[idx] = L[(s5 + c[site]) & 0xFF];
}

// ---------------- measure (batched L1-ring energy reduction) ----------------
// One block-column per manifold (blockIdx.y = n), grid-stride over its sites, block reduction
// into shared memory, atomicAdd into E[n]. interior!=0 restricts the sum to the (H-2)(W-2)
// reacted sites (== ml.gevhv.measure(interior=True)); interior==0 sums the full grid,
// INCLUDING the boundary bind value (the boundary clause — exactly what gevhv_scalar/vector's
// default interior=False composition measures).
__global__ void k_measure(int* E, const u8* G, const u8* q, long H, long W, int interior) {
  long n = blockIdx.y;
  long sites = H * W;
  const u8* g = G + n * sites;
  __shared__ int sh[256];
  int acc = 0;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < sites;
       idx += (long)gridDim.x * blockDim.x) {
    if (interior) {
      int i = (int)(idx % W), j = (int)(idx / W);
      if (i == 0 || i == W - 1 || j == 0 || j == H - 1) continue;
    }
    acc += cdist8(g[idx], q[idx]);
  }
  sh[threadIdx.x] = acc; __syncthreads();
  for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
    if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(&E[n], sh[0]);
}

// ---------------- fused GEVHV (react_bound_{scalar,vector} + measure, one pass) ----------------
// No intermediate [N][H][W] manifold is ever materialized; only N ints leave the device.
// == ml.gevhv.gevhv_scalar / gevhv_vector (the composed reference), bit-for-bit.
__global__ void k_gevhv_fused_scalar(const u8* G, const u8* Lp, const u8* q, int* E,
                                     long H, long W, int s, int t, int interior) {
  long n = blockIdx.y;
  long sites = H * W;
  const u8* g = G + n * sites;
  __shared__ int sh[256];
  int acc = 0;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < sites;
       idx += (long)gridDim.x * blockDim.x) {
    int i = (int)(idx % W), j = (int)(idx / W);
    bool boundary = (i == 0 || i == W - 1 || j == 0 || j == H - 1);
    if (interior && boundary) continue;
    u8 r;
    if (boundary) r = (u8)(s * (int)g[idx] + t);                 // boundary clause (pointwise bind)
    else {
      unsigned s5 = (unsigned)g[idx] + g[idx - 1] + g[idx + 1] + g[idx - W] + g[idx + W];
      r = Lp[s5 & 0xFF];                                          // bind absorbed: zero extra cost
    }
    acc += cdist8(r, q[idx]);
  }
  sh[threadIdx.x] = acc; __syncthreads();
  for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
    if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(&E[n], sh[0]);
}

__global__ void k_gevhv_fused_vector(const u8* G, const u8* L, const u8* v, const u8* c,
                                     const u8* q, int* E, long H, long W, int interior) {
  long n = blockIdx.y;
  long sites = H * W;
  const u8* g = G + n * sites;
  __shared__ int sh[256];
  int acc = 0;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < sites;
       idx += (long)gridDim.x * blockDim.x) {
    int i = (int)(idx % W), j = (int)(idx / W);
    bool boundary = (i == 0 || i == W - 1 || j == 0 || j == H - 1);
    if (interior && boundary) continue;
    u8 r;
    if (boundary) r = (u8)((int)g[idx] + (int)v[idx]);
    else {
      unsigned s5 = (unsigned)g[idx] + g[idx - 1] + g[idx + 1] + g[idx - W] + g[idx + W];
      r = L[(s5 + c[idx]) & 0xFF];                                // offset field (Theorem F2)
    }
    acc += cdist8(r, q[idx]);
  }
  sh[threadIdx.x] = acc; __syncthreads();
  for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
    if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(&E[n], sh[0]);
}

// ---------------- host wrappers (extern "C", ctypes-callable) ----------------
extern "C" {

__declspec(dllexport) int rk_cuda_available(void) {
  int n = 0;
  return (cudaGetDeviceCount(&n) == cudaSuccess && n > 0) ? 1 : 0;
}

__declspec(dllexport) int rk_gevhv_react(u8* outG, const u8* G, const u8* lut,
                                         long H, long W, long N) {
  long sites = H * W, tot = N * sites;
  u8 *dG, *dLut, *dOut; cudaError_t e;
  if ((e = cudaMalloc(&dG, tot))) return (int)e;
  if ((e = cudaMalloc(&dLut, 256))) { cudaFree(dG); return (int)e; }
  if ((e = cudaMalloc(&dOut, tot))) { cudaFree(dG); cudaFree(dLut); return (int)e; }
  cudaMemcpy(dG, G, tot, cudaMemcpyHostToDevice);
  cudaMemcpy(dLut, lut, 256, cudaMemcpyHostToDevice);
  long thr = 256, blk = (tot + thr - 1) / thr;
  k_react<<<(unsigned)blk, (unsigned)thr>>>(dOut, dG, dLut, H, W, N);
  e = cudaDeviceSynchronize();
  cudaMemcpy(outG, dOut, tot, cudaMemcpyDeviceToHost);
  cudaFree(dG); cudaFree(dLut); cudaFree(dOut);
  return (int)e;
}

__declspec(dllexport) int rk_gevhv_react_bound_scalar(u8* outG, const u8* G, const u8* lutAbs,
                                                       long H, long W, long N, int s, int t) {
  long sites = H * W, tot = N * sites;
  u8 *dG, *dLp, *dOut; cudaError_t e;
  if ((e = cudaMalloc(&dG, tot))) return (int)e;
  if ((e = cudaMalloc(&dLp, 256))) { cudaFree(dG); return (int)e; }
  if ((e = cudaMalloc(&dOut, tot))) { cudaFree(dG); cudaFree(dLp); return (int)e; }
  cudaMemcpy(dG, G, tot, cudaMemcpyHostToDevice);
  cudaMemcpy(dLp, lutAbs, 256, cudaMemcpyHostToDevice);
  long thr = 256, blk = (tot + thr - 1) / thr;
  k_react_bound_scalar<<<(unsigned)blk, (unsigned)thr>>>(dOut, dG, dLp, H, W, N, s, t);
  e = cudaDeviceSynchronize();
  cudaMemcpy(outG, dOut, tot, cudaMemcpyDeviceToHost);
  cudaFree(dG); cudaFree(dLp); cudaFree(dOut);
  return (int)e;
}

__declspec(dllexport) int rk_gevhv_react_bound_vector(u8* outG, const u8* G, const u8* lut,
                                                       const u8* v, const u8* c,
                                                       long H, long W, long N) {
  long sites = H * W, tot = N * sites;
  u8 *dG, *dL, *dv, *dc, *dOut; cudaError_t e;
  if ((e = cudaMalloc(&dG, tot))) return (int)e;
  if ((e = cudaMalloc(&dL, 256))) { cudaFree(dG); return (int)e; }
  if ((e = cudaMalloc(&dv, sites))) { cudaFree(dG); cudaFree(dL); return (int)e; }
  if ((e = cudaMalloc(&dc, sites))) { cudaFree(dG); cudaFree(dL); cudaFree(dv); return (int)e; }
  if ((e = cudaMalloc(&dOut, tot))) { cudaFree(dG); cudaFree(dL); cudaFree(dv); cudaFree(dc); return (int)e; }
  cudaMemcpy(dG, G, tot, cudaMemcpyHostToDevice);
  cudaMemcpy(dL, lut, 256, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, v, sites, cudaMemcpyHostToDevice);
  cudaMemcpy(dc, c, sites, cudaMemcpyHostToDevice);
  long thr = 256, blk = (tot + thr - 1) / thr;
  k_react_bound_vector<<<(unsigned)blk, (unsigned)thr>>>(dOut, dG, dL, dv, dc, H, W, N);
  e = cudaDeviceSynchronize();
  cudaMemcpy(outG, dOut, tot, cudaMemcpyDeviceToHost);
  cudaFree(dG); cudaFree(dL); cudaFree(dv); cudaFree(dc); cudaFree(dOut);
  return (int)e;
}

__declspec(dllexport) int rk_gevhv_measure(int* E, const u8* G, const u8* q,
                                           long H, long W, long N, int interior) {
  long sites = H * W, tot = N * sites;
  u8 *dG, *dq; int* dE; cudaError_t e;
  if ((e = cudaMalloc(&dG, tot))) return (int)e;
  if ((e = cudaMalloc(&dq, sites))) { cudaFree(dG); return (int)e; }
  if ((e = cudaMalloc(&dE, N * (long)sizeof(int)))) { cudaFree(dG); cudaFree(dq); return (int)e; }
  cudaMemcpy(dG, G, tot, cudaMemcpyHostToDevice);
  cudaMemcpy(dq, q, sites, cudaMemcpyHostToDevice);
  cudaMemset(dE, 0, N * (long)sizeof(int));
  dim3 grid((unsigned)((sites + 255) / 256), (unsigned)N);
  k_measure<<<grid, 256>>>(dE, dG, dq, H, W, interior);
  e = cudaDeviceSynchronize();
  cudaMemcpy(E, dE, N * (long)sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(dG); cudaFree(dq); cudaFree(dE);
  return (int)e;
}

__declspec(dllexport) int rk_gevhv_fused_scalar(int* E, const u8* G, const u8* lutAbs, const u8* q,
                                                long H, long W, long N, int s, int t, int interior) {
  long sites = H * W, tot = N * sites;
  u8 *dG, *dLp, *dq; int* dE; cudaError_t e;
  if ((e = cudaMalloc(&dG, tot))) return (int)e;
  if ((e = cudaMalloc(&dLp, 256))) { cudaFree(dG); return (int)e; }
  if ((e = cudaMalloc(&dq, sites))) { cudaFree(dG); cudaFree(dLp); return (int)e; }
  if ((e = cudaMalloc(&dE, N * (long)sizeof(int)))) { cudaFree(dG); cudaFree(dLp); cudaFree(dq); return (int)e; }
  cudaMemcpy(dG, G, tot, cudaMemcpyHostToDevice);
  cudaMemcpy(dLp, lutAbs, 256, cudaMemcpyHostToDevice);
  cudaMemcpy(dq, q, sites, cudaMemcpyHostToDevice);
  cudaMemset(dE, 0, N * (long)sizeof(int));
  dim3 grid((unsigned)((sites + 255) / 256), (unsigned)N);
  k_gevhv_fused_scalar<<<grid, 256>>>(dG, dLp, dq, dE, H, W, s, t, interior);
  e = cudaDeviceSynchronize();
  cudaMemcpy(E, dE, N * (long)sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(dG); cudaFree(dLp); cudaFree(dq); cudaFree(dE);
  return (int)e;
}

__declspec(dllexport) int rk_gevhv_fused_vector(int* E, const u8* G, const u8* lut, const u8* v,
                                                const u8* c, const u8* q,
                                                long H, long W, long N, int interior) {
  long sites = H * W, tot = N * sites;
  u8 *dG, *dL, *dv, *dc, *dq; int* dE; cudaError_t e;
  if ((e = cudaMalloc(&dG, tot))) return (int)e;
  if ((e = cudaMalloc(&dL, 256))) { cudaFree(dG); return (int)e; }
  if ((e = cudaMalloc(&dv, sites))) { cudaFree(dG); cudaFree(dL); return (int)e; }
  if ((e = cudaMalloc(&dc, sites))) { cudaFree(dG); cudaFree(dL); cudaFree(dv); return (int)e; }
  if ((e = cudaMalloc(&dq, sites))) { cudaFree(dG); cudaFree(dL); cudaFree(dv); cudaFree(dc); return (int)e; }
  if ((e = cudaMalloc(&dE, N * (long)sizeof(int)))) {
    cudaFree(dG); cudaFree(dL); cudaFree(dv); cudaFree(dc); cudaFree(dq); return (int)e;
  }
  cudaMemcpy(dG, G, tot, cudaMemcpyHostToDevice);
  cudaMemcpy(dL, lut, 256, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, v, sites, cudaMemcpyHostToDevice);
  cudaMemcpy(dc, c, sites, cudaMemcpyHostToDevice);
  cudaMemcpy(dq, q, sites, cudaMemcpyHostToDevice);
  cudaMemset(dE, 0, N * (long)sizeof(int));
  dim3 grid((unsigned)((sites + 255) / 256), (unsigned)N);
  k_gevhv_fused_vector<<<grid, 256>>>(dG, dL, dv, dc, dq, dE, H, W, interior);
  e = cudaDeviceSynchronize();
  cudaMemcpy(E, dE, N * (long)sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(dG); cudaFree(dL); cudaFree(dv); cudaFree(dc); cudaFree(dq); cudaFree(dE);
  return (int)e;
}

}  // extern "C"
