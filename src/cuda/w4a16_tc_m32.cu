// E048: tc with a 32-row M tile. Each warp computes two m16 row tiles that share every unpacked B
// fragment, so each weight is loaded and unpacked once per 32 rows instead of once per 16. Otherwise
// as E044's tc: 8 warps split K into 64-wide chunks, same fragment maps and fast unpack, FP32 partials
// reduced in shared memory in fixed warp order. Launch rule (E048 predeclaration): M >= 32 uses this
// kernel, M < 32 the unchanged tc launcher.
#include "w4a16_tc_common.cuh"

namespace {

using namespace tc_common;

template <bool HasMTail>
__global__ void __launch_bounds__(kWarps * 32)
    w4a16_tc_m32_kernel(const __half* __restrict__ A, const uint32_t* __restrict__ qweight,
                        const __half* __restrict__ scales, __half* __restrict__ C, int M, int N,
                        int K, int group_size) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int t = lane & 3, g = lane >> 2;
  const int n0 = blockIdx.x * 32, m0 = blockIdx.y * 32;
  const int chunks = K / 64, chunks_per_group = group_size / 64;
  const int c_begin = chunks * warp / kWarps, c_end = chunks * (warp + 1) / kWarps;
  // Rows of the two m16 tiles: tile h covers m0 + 16h + {g, g+8}.
  int r[2][2];
  bool v[2][2];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      r[h][e] = m0 + 16 * h + g + 8 * e;
      v[h][e] = !HasMTail || r[h][e] < M;
    }
  float acc[2][kTiles][4] = {};
  for (int c = c_begin; c < c_end; ++c) {
    const int k0 = c * 64 + 16 * t;
    uint4 alo[2][2], ahi[2][2];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        alo[h][e] = ahi[h][e] = make_uint4(0, 0, 0, 0);
        if (v[h][e]) {
          const uint4* p = reinterpret_cast<const uint4*>(A + static_cast<size_t>(r[h][e]) * K + k0);
          alo[h][e] = p[0];
          ahi[h][e] = p[1];
        }
      }
    const int grp = c / chunks_per_group;
#pragma unroll
    for (int i = 0; i < kTiles; ++i) {
      const int n = n0 + 8 * i + g;
      const uint32_t w0 = qweight[static_cast<size_t>(c * 8 + 2 * t) * N + n];
      const uint32_t w1 = qweight[static_cast<size_t>(c * 8 + 2 * t + 1) * N + n];
      const __half sh = scales[static_cast<size_t>(grp) * N + n];
      const float s = __half2float(sh);
      const __half2 s2 = __half2half2(sh);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t b0 = unpack2<true>(w0, j, s2, s), b1 = unpack2<true>(w1, j, s2, s);
#pragma unroll
        for (int h = 0; h < 2; ++h)
          mma16816(acc[h][i], comp(alo[h][0], j), comp(alo[h][1], j), comp(ahi[h][0], j),
                   comp(ahi[h][1], j), b0, b1);
      }
    }
  }

  __shared__ float partial[kWarps][32][32];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int i = 0; i < kTiles; ++i) {
      const int col = 8 * i + 2 * t, row = 16 * h + g;
      partial[warp][row][col] = acc[h][i][0];
      partial[warp][row][col + 1] = acc[h][i][1];
      partial[warp][row + 8][col] = acc[h][i][2];
      partial[warp][row + 8][col + 1] = acc[h][i][3];
    }
  // Warps with no chunks publish zeros; every thread reaches the barrier.
  __syncthreads();
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    const int idx = (threadIdx.x + part * kWarps * 32) * 2, row = idx / 32, col = idx % 32;
    if (!HasMTail || m0 + row < M) {
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        s0 += partial[w][row][col];
        s1 += partial[w][row][col + 1];
      }
      *reinterpret_cast<__half2*>(C + static_cast<size_t>(m0 + row) * N + n0 + col) =
          __floats2half2_rn(s0, s1);
    }
  }
}

}  // namespace

void launch_w4a16_tc_m32(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                         int M, int N, int K, int group_size, cudaStream_t stream) {
  // Launcher precondition (E044 qualification): the half2 store also needs a 4-byte-aligned C.
  if (K % 64 || group_size % 64 || N % 32 || reinterpret_cast<uintptr_t>(A) % 16 ||
      reinterpret_cast<uintptr_t>(C) % 4)
    return launch_w4a16_inblock_m4_vec(A, qweight, scales, C, M, N, K, group_size, stream);
  if (M < 32) return launch_w4a16_tc(A, qweight, scales, C, M, N, K, group_size, stream);
  const dim3 grid(N / 32, (M + 31) / 32), block(kWarps * 32);
  if (M % 32 == 0)
    w4a16_tc_m32_kernel<false><<<grid, block, 0, stream>>>(A, qweight, scales, C, M, N, K, group_size);
  else
    w4a16_tc_m32_kernel<true><<<grid, block, 0, stream>>>(A, qweight, scales, C, M, N, K, group_size);
  CUDA_CHECK(cudaGetLastError());
}
