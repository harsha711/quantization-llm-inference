// v0 — our first fused W4A16 kernel (bandwidth-roofline series, docs/OURS.md).
//
// C[m, n] = sum_k A[m, k] * W_hat[k, n], with W_hat rebuilt from the packed 4-bit
// codes in registers. Plain and correct; speed comes in v1+.
//
// Data (all row-major, full contract in include/w4a16/quant.h):
//   A        M x K             __half  activations
//   qweight  ceil(K/8) x N     uint32  word (k8, n) packs rows k = 8*k8 + j, j = 0..7,
//                                      row j in bits [4j, 4j+4)
//   scales   ceil(K/group) x N __half  one scale per (group of `group_size` rows, column)
//   C        M x N             __half  output
//
// Mapping: one thread computes one output C[m, n].
//   n = blockIdx.x * blockDim.x + threadIdx.x   (consecutive threads -> consecutive
//                                                 columns -> one warp reads 32 adjacent
//                                                 qweight words = 128 B, coalesced)
//   m = blockIdx.y                               (every thread in the block needs the
//                                                 same A[m, k] -> broadcast)
// So the grid is ceil(N / 256) x M blocks of 256 threads. At M = 1 that is only
// N / 256 blocks for 36 SMs — the weakness v1 attacks.
//
// Reconstruction must match the CPU reference bit-for-bit:
//   W_hat = round_to_fp16((q - 8) * s)  -> __float2half_rn, then back to float for the FMA.
// Accumulate in FP32, write C once as FP16.
#include "kernels.cuh"

namespace {

constexpr int kThreads = 256;

__global__ void w4a16_v0_kernel(const __half* __restrict__ A,
                                const uint32_t* __restrict__ qweight,
                                const __half* __restrict__ scales, __half* __restrict__ C,
                                int M, int N, int K, int group_size) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  if (n >= N || m >= M) return;  // last block may hang off the right edge

  const __half* a = A + static_cast<size_t>(m) * K;
  const int full_words = K / 8;  // words whose 8 rows are all < K
  float acc = 0.f;

  // Full words: the trip count 8 is a compile-time constant, so the compiler unrolls
  // it — 8 independent A loads, constant shift amounts, no loop branch. (A first
  // version used a run-time bound min(8, K - k0) here and was ~28% slower; see
  // docs/OURS.md, v0.)
  for (int k8 = 0; k8 < full_words; ++k8) {
    const int k0 = 8 * k8;
    const uint32_t packed = qweight[static_cast<size_t>(k8) * N + n];
    // group_size is a multiple of 8, so all 8 rows of this word share one scale.
    const float s = __half2float(scales[static_cast<size_t>(k0 / group_size) * N + n]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int q = static_cast<int>((packed >> (4 * j)) & 0xFu);
      const float w = __half2float(__float2half_rn(static_cast<float>(q - 8) * s));
      acc = fmaf(__half2float(a[k0 + j]), w, acc);
    }
  }

  // Tail: when K % 8 != 0 the final word holds fewer than 8 real rows.
  const int k0 = 8 * full_words;
  if (k0 < K) {
    const uint32_t packed = qweight[static_cast<size_t>(full_words) * N + n];
    const float s = __half2float(scales[static_cast<size_t>(k0 / group_size) * N + n]);
    for (int j = 0; j < K - k0; ++j) {
      const int q = static_cast<int>((packed >> (4 * j)) & 0xFu);
      const float w = __half2float(__float2half_rn(static_cast<float>(q - 8) * s));
      acc = fmaf(__half2float(a[k0 + j]), w, acc);
    }
  }
  C[static_cast<size_t>(m) * N + n] = __float2half_rn(acc);
}

}  // namespace

void launch_w4a16_v0(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                     int M, int N, int K, int group_size, cudaStream_t stream) {
  const dim3 block(kThreads);
  const dim3 grid((N + kThreads - 1) / kThreads, M);
  w4a16_v0_kernel<<<grid, block, 0, stream>>>(A, qweight, scales, C, M, N, K, group_size);
  CUDA_CHECK(cudaGetLastError());
}
