#include "kernels.cuh"

namespace {

template <int Rows, bool HasKTail, bool HasMTail>
__global__ void w4a16_inblock_rows_kernel(const __half* __restrict__ A,
                                        const uint32_t* __restrict__ qweight,
                                        const __half* __restrict__ scales,
                                        __half* __restrict__ C,
                                        int M, int N, int K, int group_size) {
  const int lane = threadIdx.x;
  const int slice = threadIdx.y;
  const int n = blockIdx.x * 32 + lane;
  const int m = blockIdx.y * Rows;
  const int words = (K + 7) / 8;
  const int begin = words * slice / 8;
  const int end = words * (slice + 1) / 8;
  float acc[Rows] = {};
  if (n < N) {
    for (int k8 = begin; k8 < end; ++k8) {
      const int k0 = k8 * 8;
      const uint32_t packed = qweight[static_cast<size_t>(k8) * N + n];
      const float s = __half2float(scales[static_cast<size_t>(k0 / group_size) * N + n]);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if constexpr (HasKTail) {
          if (k0 + j >= K) break;
        }
        const int q = static_cast<int>((packed >> (4 * j)) & 0xFu);
        const float w = __half2float(__float2half_rn(static_cast<float>(q - 8) * s));
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
          if constexpr (HasMTail) {
            if (m + r >= M) continue;
          }
          acc[r] = fmaf(__half2float(A[static_cast<size_t>(m + r) * K + k0 + j]), w, acc[r]);
        }
      }
    }
  }

  __shared__ float partial[Rows][8][32];
#pragma unroll
  for (int r = 0; r < Rows; ++r) partial[r][slice][lane] = acc[r];
  // Invalid rows/columns and empty slices must also publish zeros and reach the barrier.
  __syncthreads();
  if (slice == 0 && n < N) {
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
      if constexpr (HasMTail) {
        if (m + r >= M) continue;
      }
      float sum = 0.f;
#pragma unroll
      for (int s = 0; s < 8; ++s) sum += partial[r][s][lane];
      C[static_cast<size_t>(m + r) * N + n] = __float2half_rn(sum);
    }
  }
}

template <int Rows>
void launch_rows(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                 int M, int N, int K, int group_size, cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid((N + 31) / 32, (M + Rows - 1) / Rows);
  if (K % 8 == 0 && M % Rows == 0)
    w4a16_inblock_rows_kernel<Rows, false, false><<<grid, block, 0, stream>>>(A, qweight, scales, C, M, N, K, group_size);
  else if (K % 8 == 0)
    w4a16_inblock_rows_kernel<Rows, false, true><<<grid, block, 0, stream>>>(A, qweight, scales, C, M, N, K, group_size);
  else if (M % Rows == 0)
    w4a16_inblock_rows_kernel<Rows, true, false><<<grid, block, 0, stream>>>(A, qweight, scales, C, M, N, K, group_size);
  else
    w4a16_inblock_rows_kernel<Rows, true, true><<<grid, block, 0, stream>>>(A, qweight, scales, C, M, N, K, group_size);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launch_w4a16_inblock_m2(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                             int M, int N, int K, int group_size, cudaStream_t stream) {
  launch_rows<2>(A, qweight, scales, C, M, N, K, group_size, stream);
}

void launch_w4a16_inblock_m4(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                             int M, int N, int K, int group_size, cudaStream_t stream) {
  launch_rows<4>(A, qweight, scales, C, M, N, K, group_size, stream);
}
