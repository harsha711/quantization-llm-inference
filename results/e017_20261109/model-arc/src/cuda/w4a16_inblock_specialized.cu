#include "kernels.cuh"

namespace {

template <bool HasTail>
__global__ void w4a16_inblock_specialized_kernel(const __half* __restrict__ A,
                                    const uint32_t* __restrict__ qweight,
                                    const __half* __restrict__ scales, __half* __restrict__ C,
                                    int N, int K, int group_size) {
  const int lane = threadIdx.x;
  const int slice = threadIdx.y;
  const int n = blockIdx.x * 32 + lane;
  const int m = blockIdx.y;
  const int words = (K + 7) / 8;
  const int begin = words * slice / 8;
  const int end = words * (slice + 1) / 8;
  float acc = 0.f;
  if (n < N) {
    const __half* a_row = A + static_cast<size_t>(m) * K;
    for (int k8 = begin; k8 < end; ++k8) {
      const int k0 = k8 * 8;
      const uint32_t packed = qweight[static_cast<size_t>(k8) * N + n];
      const float s = __half2float(scales[static_cast<size_t>(k0 / group_size) * N + n]);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if constexpr (HasTail) {
          if (k0 + j >= K) break;
        }
        const int q = static_cast<int>((packed >> (4 * j)) & 0xFu);
        const float w = __half2float(__float2half_rn(static_cast<float>(q - 8) * s));
        acc = fmaf(__half2float(a_row[k0 + j]), w, acc);
      }
    }
  }

  __shared__ float partial[8][32];
  partial[slice][lane] = acc;
  // Even empty slices and lanes beyond N must publish zero and reach the barrier.
  __syncthreads();
  if (slice == 0 && n < N) {
    float sum = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) sum += partial[s][lane];
    C[static_cast<size_t>(m) * N + n] = __float2half_rn(sum);
  }
}

}  // namespace

void launch_w4a16_inblock_specialized(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                          int M, int N, int K, int group_size, cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid((N + 31) / 32, M);
  if (K % 8 == 0)
    w4a16_inblock_specialized_kernel<false><<<grid, block, 0, stream>>>(
        A, qweight, scales, C, N, K, group_size);
  else
    w4a16_inblock_specialized_kernel<true><<<grid, block, 0, stream>>>(
        A, qweight, scales, C, N, K, group_size);
  CUDA_CHECK(cudaGetLastError());
}
