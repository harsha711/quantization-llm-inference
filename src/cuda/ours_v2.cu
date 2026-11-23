// v2 — 128-bit loads: each lane reads 4 neighbouring packed words in one request.
//
// Why (docs/OURS.md, v1 profile): after v1 every SM is busy, but DRAM is only ~55%
// used. Each thread asks memory for 4 bytes at a time (one packed word), so too few
// bytes are in flight to cover DRAM latency (Little's law: bytes in flight needed =
// bandwidth x latency). v2 asks for 16 bytes per request.
//
// Layout consequence. qweight is (k8, n) row-major, so the 4 words that sit next to
// each other in memory are 4 neighbouring COLUMNS of the same 8 rows — not 4 more
// rows of one column:
//
//   qweight row k8:  ... | n0 | n0+1 | n0+2 | n0+3 | ...     <- one uint4 = lane l
//                          \______ 16 B ______/
//
// So a lane computes 4 outputs, a warp covers 128 columns, and there are 4x fewer
// column tiles than in v1. The K split (grid.z) has to win back the blocks this loses;
// the split rule below is re-fitted for that (docs/OURS.md, v2).
//
// Per packed word k8, lane l of column tile t (n0 = 4 * (32 t + l)) loads:
//   qweight[k8*N + n0 .. n0+3]   one 16 B load   (the warp reads 512 contiguous bytes)
//   scales [g *N + n0 .. n0+3]   one  8 B load   (g = 8*k8 / group_size)
//   A      [m*K + 8*k8 .. +7]    one 16 B load   (same address for every lane: broadcast)
// and does 4 columns x 8 rows = 32 FMAs, reusing the 8 activations across 4 columns.
//
// Fast path needs N % 4 == 0 and K % 8 == 0, so that every row of qweight/scales/A
// starts on a 16 B / 8 B boundary. Other shapes fall back to v1. All real model
// shapes qualify.
//
// Everything else is v1: 8 warps split the block's K range into contiguous runs,
// fixed-order in-block reduction, and the last-arriving block of a tile finishes the
// cross-block sum (deterministic, one kernel, no memset).
// Numerics unchanged: W_hat = fp16((q - 8) * s), FP32 accumulation, rows summed in
// order j = 0..7 within each word.
#include <algorithm>
#include <cstdlib>

#include "kernels.cuh"

namespace {

constexpr int kLanes = 32;
constexpr int kWarps = 8;
constexpr int kCols = 4;                    // columns per lane (one uint4 of words)
constexpr int kTileCols = kLanes * kCols;   // 128 columns per block
constexpr int kMaxSplits = 64;

__device__ __forceinline__ float2 half2_bits_to_float2(uint32_t u) {
  return __half22float2(*reinterpret_cast<const __half2*>(&u));
}

__global__ void __launch_bounds__(kLanes * kWarps)
    w4a16_v2_kernel(const __half* __restrict__ A, const uint32_t* __restrict__ qweight,
                    const __half* __restrict__ scales, __half* __restrict__ C,
                    float* __restrict__ partials, unsigned* __restrict__ counters, int M, int N,
                    int K, int group_size, int splits) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tile = blockIdx.x;
  const int m = blockIdx.y;
  const int z = blockIdx.z;
  const int n0 = (tile * kLanes + lane) * kCols;  // first of this lane's 4 columns
  const bool active = n0 < N;  // N % 4 == 0: a lane's 4 columns are all in or all out

  // This block's packed-word range, then this warp's contiguous run inside it.
  const int words = K / 8;
  const int w_begin = static_cast<int>(static_cast<long long>(words) * z / splits);
  const int w_end = static_cast<int>(static_cast<long long>(words) * (z + 1) / splits);
  const int span = w_end - w_begin;
  const int first = w_begin + span * warp / kWarps;
  const int last = w_begin + span * (warp + 1) / kWarps;

  float acc[kCols] = {0.f, 0.f, 0.f, 0.f};
  if (active) {
    const uint4* a_words = reinterpret_cast<const uint4*>(A + static_cast<size_t>(m) * K);
    for (int k8 = first; k8 < last; ++k8) {
      const uint4 q4 = *reinterpret_cast<const uint4*>(qweight + static_cast<size_t>(k8) * N + n0);
      const uint2 s4 = *reinterpret_cast<const uint2*>(
          scales + static_cast<size_t>(8 * k8 / group_size) * N + n0);
      const uint4 a8 = a_words[k8];

      // 8 activations, converted once and reused for all 4 columns.
      float a[8];
      const float2 a01 = half2_bits_to_float2(a8.x), a23 = half2_bits_to_float2(a8.y);
      const float2 a45 = half2_bits_to_float2(a8.z), a67 = half2_bits_to_float2(a8.w);
      a[0] = a01.x; a[1] = a01.y; a[2] = a23.x; a[3] = a23.y;
      a[4] = a45.x; a[5] = a45.y; a[6] = a67.x; a[7] = a67.y;

      const float2 s01 = half2_bits_to_float2(s4.x), s23 = half2_bits_to_float2(s4.y);
      const float s[kCols] = {s01.x, s01.y, s23.x, s23.y};
      const uint32_t q[kCols] = {q4.x, q4.y, q4.z, q4.w};

#pragma unroll
      for (int c = 0; c < kCols; ++c) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int qq = static_cast<int>((q[c] >> (4 * j)) & 0xFu);
          const float w = __half2float(__float2half_rn(static_cast<float>(qq - 8) * s[c]));
          acc[c] = fmaf(a[j], w, acc[c]);
        }
      }
    }
  }

  // 1. In-block reduction over the 8 warps, fixed order y = 0..7.
  __shared__ float4 part[kWarps][kLanes];
  part[warp][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
  __syncthreads();
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  if (warp == 0) {
#pragma unroll
    for (int y = 0; y < kWarps; ++y) {
      const float4 p = part[y][lane];
      sum.x += p.x; sum.y += p.y; sum.z += p.z; sum.w += p.w;
    }
  }

  const size_t slot = static_cast<size_t>(m) * N + n0;  // multiple of 4
  auto store_c = [&](float4 v) {
    const __half2 lo = __floats2half2_rn(v.x, v.y);
    const __half2 hi = __floats2half2_rn(v.z, v.w);
    uint2 out;
    out.x = *reinterpret_cast<const uint32_t*>(&lo);
    out.y = *reinterpret_cast<const uint32_t*>(&hi);
    *reinterpret_cast<uint2*>(C + slot) = out;  // 8 B, aligned since N % 4 == 0
  };

  if (splits == 1) {
    if (warp == 0 && active) store_c(sum);
    return;
  }

  // 2. Cross-block reduction (same protocol as v1): publish, fence, count; the
  //    last-arriving block of this (m, tile) adds the S partials in order z = 0..S-1.
  float4* parts4 = reinterpret_cast<float4*>(partials);
  const size_t mn4 = static_cast<size_t>(M) * N / 4;
  if (warp == 0 && active) parts4[static_cast<size_t>(z) * mn4 + slot / 4] = sum;
  __threadfence();
  __syncthreads();

  __shared__ bool is_last;
  unsigned* counter = counters + static_cast<size_t>(m) * gridDim.x + tile;
  if (lane == 0 && warp == 0) is_last = atomicAdd(counter, 1u) == static_cast<unsigned>(splits - 1);
  __syncthreads();
  if (!is_last) return;

  __threadfence();
  if (warp == 0 && active) {
    float4 total = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int zz = 0; zz < splits; ++zz) {
      const float4 p = __ldcg(&parts4[static_cast<size_t>(zz) * mn4 + slot / 4]);  // L2
      total.x += p.x; total.y += p.y; total.z += p.z; total.w += p.w;
    }
    store_c(total);
  }
  if (lane == 0 && warp == 0) *counter = 0u;  // ready for the next launch
}

// Workspace persists across launches, grown on demand (first, untimed launch only).
float* g_partials = nullptr;
size_t g_partials_elems = 0;
unsigned* g_counters = nullptr;
size_t g_counter_elems = 0;

void ensure_workspace(size_t partial_elems, size_t counter_elems) {
  if (partial_elems > g_partials_elems) {
    if (g_partials) CUDA_CHECK(cudaFree(g_partials));
    CUDA_CHECK(cudaMalloc(&g_partials, partial_elems * sizeof(float)));
    g_partials_elems = partial_elems;
  }
  if (counter_elems > g_counter_elems) {
    if (g_counters) CUDA_CHECK(cudaFree(g_counters));
    CUDA_CHECK(cudaMalloc(&g_counters, counter_elems * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(g_counters, 0, counter_elems * sizeof(unsigned)));
    g_counter_elems = counter_elems;
  }
}

int sm_count() {
  static int cached = 0;
  if (cached == 0) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, dev));
  }
  return cached;
}

bool aligned(const void* p, size_t bytes) { return reinterpret_cast<uintptr_t>(p) % bytes == 0; }

}  // namespace

bool w4a16_v2_fast_path(const __half* A, const uint32_t* qweight, const __half* scales,
                        const __half* C, int N, int K) {
  return N % kCols == 0 && K % 8 == 0 && aligned(A, 16) && aligned(qweight, 16) &&
         aligned(scales, 8) && aligned(C, 8);
}

// Split rule (docs/OURS.md, v2 "split sweep"): aim for about 4 blocks per SM,
//   S = round(4 * SMs / (tiles * M)), clamped to [1, words / 8].
// Measured at M = 1 on c8, the best S of {1..32} landed at 128-144 blocks on all
// three large shapes (4096^2: S=4 or 5, 1536x8960: S=2, 8960x1536: S=12). v1's
// balance rule does not transfer: with 32 tiles it sees no imbalance at S = 1 and
// launches 32 blocks for 36 SMs (47 us vs 26.6 us at S = 4).
// W4A16_V2_SPLITS=<S> forces S (also used by v3, which shares v2's grid).
int w4a16_v2_splits(int M, int N, int K) {
  if (const char* env = std::getenv("W4A16_V2_SPLITS"))
    return std::max(1, std::min(std::atoi(env), kMaxSplits));
  constexpr int kTargetBlocksPerSm = 4;
  const long long base = static_cast<long long>((N + kTileCols - 1) / kTileCols) * M;
  const long long target = static_cast<long long>(kTargetBlocksPerSm) * sm_count();
  const int s_max = std::max(1, std::min(kMaxSplits, (K / 8) / kWarps));  // >= 1 word per warp
  const long long s = (target + base / 2) / base;
  return static_cast<int>(std::max(1LL, std::min<long long>(s, s_max)));
}

void w4a16_v2_workspace(int M, int N, int splits, float** partials, unsigned** counters) {
  if (splits > 1)
    ensure_workspace(static_cast<size_t>(splits) * M * N,
                     static_cast<size_t>(M) * ((N + kTileCols - 1) / kTileCols));
  *partials = g_partials;
  *counters = g_counters;
}

void launch_w4a16_v2(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                     int M, int N, int K, int group_size, cudaStream_t stream) {
  if (!w4a16_v2_fast_path(A, qweight, scales, C, N, K)) {
    launch_w4a16_v1(A, qweight, scales, C, M, N, K, group_size, stream);
    return;
  }
  const int splits = w4a16_v2_splits(M, N, K);
  float* partials = nullptr;
  unsigned* counters = nullptr;
  w4a16_v2_workspace(M, N, splits, &partials, &counters);
  const dim3 block(kLanes, kWarps);
  const dim3 grid((N + kTileCols - 1) / kTileCols, M, splits);
  w4a16_v2_kernel<<<grid, block, 0, stream>>>(A, qweight, scales, C, partials, counters, M, N, K,
                                             group_size, splits);
  CUDA_CHECK(cudaGetLastError());
}
