// ringkit CUDA backend (D9 silicon). ringkit's OWN ring semantics on NVIDIA — NOT ported.
// Mirrors the Apple/Metal backend: __global__ kernels + extern "C" host launch wrappers.
// D9 bar: every op reproduces ringkit.core.native / backend._PY BIT-FOR-BIT.
//   ring_mul/add/sub : (x . y) & 0xFF elementwise
//   ring_gemm        : C[i,j] = (SUM_k A[i,k]*B[k,j]) & 0xFF   (mod-256 accumulation)
#include <cuda_runtime.h>

__global__ void k_mul(unsigned char* o, const unsigned char* a, const unsigned char* b, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = (unsigned char)(((unsigned)a[i] * (unsigned)b[i]) & 0xFF);
}
__global__ void k_add(unsigned char* o, const unsigned char* a, const unsigned char* b, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = (unsigned char)(((unsigned)a[i] + (unsigned)b[i]) & 0xFF);
}
__global__ void k_sub(unsigned char* o, const unsigned char* a, const unsigned char* b, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = (unsigned char)(((unsigned)a[i] - (unsigned)b[i]) & 0xFF);
}
__global__ void k_gemm(unsigned char* C, const unsigned char* A, const unsigned char* B,
                       long M, long K, long N) {
  long i = (long)blockIdx.y * blockDim.y + threadIdx.y;
  long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M || j >= N) return;
  unsigned acc = 0;
  for (long k = 0; k < K; k++) acc += (unsigned)A[i * K + k] * (unsigned)B[k * N + j];
  C[i * N + j] = (unsigned char)(acc & 0xFF);
}

typedef void (*eltk)(unsigned char*, const unsigned char*, const unsigned char*, long);
static int elt(eltk kern, unsigned char* o, const unsigned char* a, const unsigned char* b, long n) {
  unsigned char *da, *db, *dout; cudaError_t e;
  if ((e = cudaMalloc(&da, n))) return (int)e;
  if ((e = cudaMalloc(&db, n))) { cudaFree(da); return (int)e; }
  if ((e = cudaMalloc(&dout, n))) { cudaFree(da); cudaFree(db); return (int)e; }
  cudaMemcpy(da, a, n, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, n, cudaMemcpyHostToDevice);
  long thr = 256, blk = (n + thr - 1) / thr;
  kern<<<(unsigned)blk, (unsigned)thr>>>(dout, da, db, n);
  e = cudaDeviceSynchronize();
  cudaMemcpy(o, dout, n, cudaMemcpyDeviceToHost);
  cudaFree(da); cudaFree(db); cudaFree(dout);
  return (int)e;
}

extern "C" {
__declspec(dllexport) int rk_cuda_available(void) {
  int n = 0; return (cudaGetDeviceCount(&n) == cudaSuccess && n > 0) ? 1 : 0;
}
__declspec(dllexport) int ring_mul(unsigned char* o, const unsigned char* a, const unsigned char* b, long n) { return elt(k_mul, o, a, b, n); }
__declspec(dllexport) int ring_add(unsigned char* o, const unsigned char* a, const unsigned char* b, long n) { return elt(k_add, o, a, b, n); }
__declspec(dllexport) int ring_sub(unsigned char* o, const unsigned char* a, const unsigned char* b, long n) { return elt(k_sub, o, a, b, n); }
__declspec(dllexport) int ring_gemm(unsigned char* C, const unsigned char* A, const unsigned char* B,
                                    long M, long K, long N) {
  unsigned char *dA, *dB, *dC; cudaError_t e; long sA = M * K, sB = K * N, sC = M * N;
  if ((e = cudaMalloc(&dA, sA))) return (int)e;
  if ((e = cudaMalloc(&dB, sB))) { cudaFree(dA); return (int)e; }
  if ((e = cudaMalloc(&dC, sC))) { cudaFree(dA); cudaFree(dB); return (int)e; }
  cudaMemcpy(dA, A, sA, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sB, cudaMemcpyHostToDevice);
  dim3 block(16, 16), grid((unsigned)((N + 15) / 16), (unsigned)((M + 15) / 16));
  k_gemm<<<grid, block>>>(dC, dA, dB, M, K, N);
  e = cudaDeviceSynchronize();
  cudaMemcpy(C, dC, sC, cudaMemcpyDeviceToHost);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return (int)e;
}
}
