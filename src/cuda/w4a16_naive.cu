// Version 0: correctness baseline for the fused W4A16 kernel.
//
// One thread computes one output C[m, n]. Threads in a warp take consecutive n, so
// their qweight and scale loads are coalesced, and they all read the same A element
// (a broadcast). There is no shared-memory tiling and no data reuse across threads,
// so this is expected to be far slower than cuBLAS for anything but tiny M.
#include "kernels.cuh"

namespace {

__global__ void w4a16_naive_kernel(const __half* __restrict__ A,
                                   const uint32_t* __restrict__ qweight,
                                   const __half* __restrict__ scales, __half* __restrict__ C,
                                   int M, int N, int K, int group_size) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  if (n >= N || m >= M) return;

  const __half* a_row = A + static_cast<size_t>(m) * K;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += 8) {
    const uint32_t packed = qweight[static_cast<size_t>(k0 / 8) * N + n];
    // group_size is a multiple of 8, so all 8 rows in this word share one scale.
    const float s = __half2float(scales[static_cast<size_t>(k0 / group_size) * N + n]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (k0 + j >= K) break;
      const int q = static_cast<int>((packed >> (4 * j)) & 0xFu);
      // Same reconstructed FP16 as host dequant / the unfused dequant kernel.
      const float w = __half2float(__float2half_rn(static_cast<float>(q - 8) * s));
      acc = fmaf(__half2float(a_row[k0 + j]), w, acc);
    }
  }
  C[static_cast<size_t>(m) * N + n] = __float2half_rn(acc);
}

}  // namespace

void launch_w4a16_naive(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                        int M, int N, int K, int group_size, cudaStream_t stream) {
  constexpr int kThreads = 256;
  dim3 block(kThreads);
  dim3 grid((N + kThreads - 1) / kThreads, M);
  w4a16_naive_kernel<<<grid, block, 0, stream>>>(A, qweight, scales, C, M, N, K, group_size);
  CUDA_CHECK(cudaGetLastError());
}
