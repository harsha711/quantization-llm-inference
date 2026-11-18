// E047: tc with K split across blocks (grid z) so small-M grids fill the GPU.
// Same tile (16 rows x 32 columns), 8 warps, fragment maps and fast unpack as E044's tc. Block z of S
// takes chunks [chunks*z/S, chunks*(z+1)/S) of K and splits them over its 8 warps as tc does; the warp
// partials are reduced in shared memory in fixed warp order, the block writes its FP32 tile to a
// workspace, and the last-arriving block of each output tile sums the S tiles in fixed z order
// (0..S-1) and stores C. Output is therefore bitwise reproducible run to run (arrival order never
// enters the arithmetic); against tc only the FP32 addition order differs (FP64 tolerance applies).
//
// Launch rule, fixed before timing (E047 predeclaration): S = 4 if the tc grid (N/32 x ceil(M/16)
// blocks) is below 3 waves of the resident-block capacity (SMs x occupancy of this kernel) and K has
// at least 4 x 8 chunks of 64; otherwise the unchanged tc launcher (S = 1).
#include "w4a16_tc_common.cuh"

namespace {

using namespace tc_common;
constexpr int kSplit = 4;

template <bool HasMTail>
__global__ void __launch_bounds__(kWarps * 32)
    w4a16_tc_splitk_kernel(const __half* __restrict__ A, const uint32_t* __restrict__ qweight,
                           const __half* __restrict__ scales, __half* __restrict__ C, int M, int N,
                           int K, int group_size, float* __restrict__ ws,
                           unsigned* __restrict__ counters) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int t = lane & 3, g = lane >> 2;
  const int n0 = blockIdx.x * 32, m0 = blockIdx.y * 16;
  const int S = gridDim.z, z = blockIdx.z;
  const int chunks = K / 64, chunks_per_group = group_size / 64;
  const int s_begin = chunks * z / S, s_len = chunks * (z + 1) / S - s_begin;
  const int c_begin = s_begin + s_len * warp / kWarps, c_end = s_begin + s_len * (warp + 1) / kWarps;
  const int r0 = m0 + g, r1 = r0 + 8;
  const bool v0 = !HasMTail || r0 < M, v1 = !HasMTail || r1 < M;
  float acc[kTiles][4] = {};
  for (int c = c_begin; c < c_end; ++c) {
    const int k0 = c * 64 + 16 * t;
    const uint4 zz = make_uint4(0, 0, 0, 0);
    uint4 a0lo = zz, a0hi = zz, a1lo = zz, a1hi = zz;
    if (v0) {
      const uint4* p = reinterpret_cast<const uint4*>(A + static_cast<size_t>(r0) * K + k0);
      a0lo = p[0]; a0hi = p[1];
    }
    if (v1) {
      const uint4* p = reinterpret_cast<const uint4*>(A + static_cast<size_t>(r1) * K + k0);
      a1lo = p[0]; a1hi = p[1];
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
      for (int j = 0; j < 4; ++j)
        mma16816(acc[i], comp(a0lo, j), comp(a1lo, j), comp(a0hi, j), comp(a1hi, j),
                 unpack2<true>(w0, j, s2, s), unpack2<true>(w1, j, s2, s));
    }
  }

  __shared__ float partial[kWarps][16][32];
  __shared__ bool is_last;
#pragma unroll
  for (int i = 0; i < kTiles; ++i) {
    const int col = 8 * i + 2 * t;
    partial[warp][g][col] = acc[i][0];
    partial[warp][g][col + 1] = acc[i][1];
    partial[warp][g + 8][col] = acc[i][2];
    partial[warp][g + 8][col + 1] = acc[i][3];
  }
  __syncthreads();
  const int idx = threadIdx.x * 2, row = idx / 32, col = idx % 32;
  float s0 = 0.f, s1 = 0.f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    s0 += partial[w][row][col];
    s1 += partial[w][row][col + 1];
  }
  // Tile partial -> workspace (L2), then the classic threadfence/counter handoff to the last block.
  const size_t tile = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
  float2* tile_ws = reinterpret_cast<float2*>(ws + tile * S * 512);
  __stcg(tile_ws + z * 256 + threadIdx.x, make_float2(s0, s1));
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) is_last = atomicAdd(counters + tile, 1u) == static_cast<unsigned>(S - 1);
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  float t0 = 0.f, t1 = 0.f;
  for (int zi = 0; zi < S; ++zi) {
    const float2 v = __ldcg(tile_ws + zi * 256 + threadIdx.x);
    t0 += v.x;
    t1 += v.y;
  }
  if (!HasMTail || m0 + row < M)
    *reinterpret_cast<__half2*>(C + static_cast<size_t>(m0 + row) * N + n0 + col) =
        __floats2half2_rn(t0, t1);
  if (threadIdx.x == 0) counters[tile] = 0;  // ready for the next launch on this stream
}

// shortcut: one process-wide workspace, grown on demand (cudaMalloc synchronizes, so the first launch
// of a larger shape must be outside timing) and shared by all launches: concurrent launches on
// different streams would race on it; give each stream its own workspace if that is ever needed.
float* g_ws = nullptr;
unsigned* g_counters = nullptr;
size_t g_ws_floats = 0, g_counter_n = 0;

}  // namespace

// Path for this shape: 0 vec fallback, 1 unchanged tc, kSplit split-K (the predeclared rule).
int w4a16_tc_splitk_factor(const __half* A, const __half* C, int M, int N, int K, int group_size) {
  // Launcher precondition (E044 qualification): the half2 store also needs a 4-byte-aligned C.
  if (K % 64 || group_size % 64 || N % 32 || reinterpret_cast<uintptr_t>(A) % 16 ||
      reinterpret_cast<uintptr_t>(C) % 4)
    return 0;
  static int slots = 0;  // shortcut: queried once per process, assumes one device
  if (!slots) {
    int dev = 0, sms = 0, occ = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, w4a16_tc_splitk_kernel<false>,
                                                             kWarps * 32, 0));
    slots = sms * occ;
  }
  const size_t tiles = static_cast<size_t>(N / 32) * ((M + 15) / 16);
  return tiles >= 3 * static_cast<size_t>(slots) || K / 64 < kSplit * kWarps ? 1 : kSplit;
}

void launch_w4a16_tc_splitk(const __half* A, const uint32_t* qweight, const __half* scales,
                            __half* C, int M, int N, int K, int group_size, cudaStream_t stream) {
  const int split = w4a16_tc_splitk_factor(A, C, M, N, K, group_size);
  if (split == 0)
    return launch_w4a16_inblock_m4_vec(A, qweight, scales, C, M, N, K, group_size, stream);
  if (split == 1) return launch_w4a16_tc(A, qweight, scales, C, M, N, K, group_size, stream);
  const size_t tiles = static_cast<size_t>(N / 32) * ((M + 15) / 16);
  if (tiles * kSplit * 512 > g_ws_floats) {
    CUDA_CHECK(cudaFree(g_ws));
    g_ws_floats = tiles * kSplit * 512;
    CUDA_CHECK(cudaMalloc(&g_ws, g_ws_floats * sizeof(float)));
  }
  if (tiles > g_counter_n) {
    CUDA_CHECK(cudaFree(g_counters));
    g_counter_n = tiles;
    CUDA_CHECK(cudaMalloc(&g_counters, g_counter_n * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(g_counters, 0, g_counter_n * sizeof(unsigned)));
  }
  const dim3 grid(N / 32, (M + 15) / 16, kSplit), block(kWarps * 32);
  if (M % 16 == 0)
    w4a16_tc_splitk_kernel<false><<<grid, block, 0, stream>>>(A, qweight, scales, C, M, N, K,
                                                               group_size, g_ws, g_counters);
  else
    w4a16_tc_splitk_kernel<true><<<grid, block, 0, stream>>>(A, qweight, scales, C, M, N, K,
                                                              group_size, g_ws, g_counters);
  CUDA_CHECK(cudaGetLastError());
}

unsigned long long w4a16_tc_splitk_unpack_selftest() {
  unsigned long long* d = nullptr;
  unsigned long long h = 0;
  CUDA_CHECK(cudaMalloc(&d, sizeof(h)));
  CUDA_CHECK(cudaMemset(d, 0, sizeof(h)));
  tc_common::unpack_selftest_kernel<<<256, 256>>>(d);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d));
  return h;
}
