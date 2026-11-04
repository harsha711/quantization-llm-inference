#include "kernels.cuh"

namespace {

__global__ void dequant_w4a16_kernel(const uint32_t* __restrict__ qweight,
                                     const __half* __restrict__ scales, __half* __restrict__ W,
                                     int N, int K, int group_size) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int k8 = blockIdx.y;
  if (n >= N) return;
  const int k0 = k8 * 8;
  if (k0 >= K) return;

  const uint32_t packed = qweight[static_cast<size_t>(k8) * N + n];
  const float s = __half2float(scales[static_cast<size_t>(k0 / group_size) * N + n]);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int q = static_cast<int>((packed >> (4 * j)) & 0xFu);
    W[static_cast<size_t>(k0 + j) * N + n] = __float2half_rn(static_cast<float>(q - 8) * s);
  }
}

// Write every 4-byte word so the traffic goes through the L1/L2 path rather than
// a memset engine that can bypass the cache.
__global__ void l2_flush_kernel(uint32_t* buf, size_t nwords) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < nwords;
       i += stride)
    buf[i] = 0;
}

}  // namespace

void launch_dequant_w4a16(const uint32_t* qweight, const __half* scales, __half* W, int N, int K,
                          int group_size, cudaStream_t stream) {
  constexpr int kThreads = 256;
  dim3 block(kThreads);
  dim3 grid((N + kThreads - 1) / kThreads, K / 8);
  dequant_w4a16_kernel<<<grid, block, 0, stream>>>(qweight, scales, W, N, K, group_size);
  CUDA_CHECK(cudaGetLastError());
}

void launch_l2_flush(void* buf, size_t nbytes, cudaStream_t stream) {
  const size_t nwords = nbytes / sizeof(uint32_t);
  constexpr int kThreads = 256;
  int blocks = 2048;
  l2_flush_kernel<<<blocks, kThreads, 0, stream>>>(static_cast<uint32_t*>(buf), nwords);
  CUDA_CHECK(cudaGetLastError());
}
