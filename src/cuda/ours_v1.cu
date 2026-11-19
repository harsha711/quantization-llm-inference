// v1 — split K across warps AND blocks so every shape fills the GPU (docs/OURS.md).
//
// Problem v1 fixes (ncu, docs/OURS.md "Profile"): with one thread per output column
// the grid has only N/256 blocks — 6 blocks for 36 SMs at N=1536 — so most SMs idle
// and too few loads are in flight to keep DRAM busy.
//
// Mapping
//   block  = 32 lanes x 8 warps. Lane -> one of 32 adjacent columns (coalesced 128 B
//            qweight reads, as in v0). Warp y -> the y-th contiguous run of packed
//            words in this block's K range (as inblock_specialized).
//   grid.x = ceil(N / 32) column tiles
//   grid.y = M rows
//   grid.z = S K-splits, chosen at launch by the balance rule above
//            w4a16_v1_splits() (busiest SM's block count x work per block).
//
// Reduction (deterministic: the summation order never depends on timing)
//   1. In-block: the 8 warps' partial sums go through shared memory and are added
//      in fixed order y = 0..7.
//   2. Across the S blocks of a tile (only if S > 1): each block writes its 32 sums
//      to an FP32 workspace, fences, and bumps a per-(m, tile) counter. The block
//      that arrives last adds the S partials in fixed order z = 0..S-1, writes FP16,
//      and resets the counter to 0 for the next launch. One kernel, no memset.
//
// Numerics: W_hat = fp16((q - 8) * s) exactly as v0; FP32 accumulation. Splitting K
// changes the order of the FP32 additions, so results are checked against the FP64
// tolerance, not bitwise against v0.
#include <algorithm>
#include <cstdlib>

#include "kernels.cuh"

namespace {

constexpr int kLanes = 32;
constexpr int kWarps = 8;
constexpr int kMaxSplits = 32;

// Ablation knobs (docs/OURS.md, v1): Contiguous = each warp takes one contiguous run
// of words (as inblock_specialized) instead of every 8th word; HasTail = false
// compiles out the partial-final-word check when K % 8 == 0.
template <bool Contiguous, bool HasTail>
__global__ void __launch_bounds__(kLanes * kWarps)
    w4a16_v1_kernel(const __half* __restrict__ A, const uint32_t* __restrict__ qweight,
                    const __half* __restrict__ scales, __half* __restrict__ C,
                    float* __restrict__ partials, unsigned* __restrict__ counters, int M, int N,
                    int K, int group_size, int splits) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tile = blockIdx.x;
  const int n = tile * kLanes + lane;
  const int m = blockIdx.y;
  const int z = blockIdx.z;

  // This block's packed-word range [w_begin, w_end) of the K dimension.
  const int words = (K + 7) / 8;
  const int w_begin = static_cast<int>(static_cast<long long>(words) * z / splits);
  const int w_end = static_cast<int>(static_cast<long long>(words) * (z + 1) / splits);

  float acc = 0.f;
  if (n < N) {  // out-of-range lanes still reach every barrier below
    const __half* a = A + static_cast<size_t>(m) * K;
    const int span = w_end - w_begin;
    const int first = Contiguous ? w_begin + span * warp / kWarps : w_begin + warp;
    const int last = Contiguous ? w_begin + span * (warp + 1) / kWarps : w_end;
    const int step = Contiguous ? 1 : kWarps;
    for (int k8 = first; k8 < last; k8 += step) {
      const int k0 = 8 * k8;
      const uint32_t packed = qweight[static_cast<size_t>(k8) * N + n];
      const float s = __half2float(scales[static_cast<size_t>(k0 / group_size) * N + n]);
      if (!HasTail || k0 + 8 <= K) {  // full word: constant trip count, unrolled (v0 lesson)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int q = static_cast<int>((packed >> (4 * j)) & 0xFu);
          const float w = __half2float(__float2half_rn(static_cast<float>(q - 8) * s));
          acc = fmaf(__half2float(a[k0 + j]), w, acc);
        }
      } else {  // partial final word
        for (int j = 0; j < K - k0; ++j) {
          const int q = static_cast<int>((packed >> (4 * j)) & 0xFu);
          const float w = __half2float(__float2half_rn(static_cast<float>(q - 8) * s));
          acc = fmaf(__half2float(a[k0 + j]), w, acc);
        }
      }
    }
  }

  // 1. In-block reduction over the 8 warps, fixed order.
  __shared__ float part[kWarps][kLanes];
  part[warp][lane] = acc;
  __syncthreads();
  float sum = 0.f;
  if (warp == 0) {
#pragma unroll
    for (int y = 0; y < kWarps; ++y) sum += part[y][lane];
  }

  if (splits == 1) {
    if (warp == 0 && n < N) C[static_cast<size_t>(m) * N + n] = __float2half_rn(sum);
    return;
  }

  // 2. Cross-block reduction: last-arriving block of this (m, tile) finishes it.
  const size_t slot = static_cast<size_t>(m) * N + n;
  if (warp == 0 && n < N) partials[static_cast<size_t>(z) * M * N + slot] = sum;
  __threadfence();  // make this block's partials visible device-wide before counting
  __syncthreads();

  __shared__ bool is_last;
  unsigned* counter = counters + static_cast<size_t>(m) * gridDim.x + tile;
  if (lane == 0 && warp == 0) is_last = atomicAdd(counter, 1u) == static_cast<unsigned>(splits - 1);
  __syncthreads();
  if (!is_last) return;

  __threadfence();
  if (warp == 0 && n < N) {
    float total = 0.f;
    for (int zz = 0; zz < splits; ++zz)
      total += __ldcg(&partials[static_cast<size_t>(zz) * M * N + slot]);  // L2, not stale L1
    C[slot] = __float2half_rn(total);
  }
  if (lane == 0 && warp == 0) *counter = 0u;  // ready for the next launch
}

// Workspace persists across launches; grown (never shrunk) on demand. Growing
// synchronizes the device, which only happens on a shape's first (untimed) launch.
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

constexpr int kAutoMaxSplits = 8;  // S = 12 measured slower on 4096^2 (docs/OURS.md, v1)

}  // namespace

// Split rule (docs/OURS.md, v1): kernel time ~ (blocks on the busiest SM) x (work per
// block) = ceil(tiles*M*S / SMs) / S. Pick the S in [1, 8] that minimizes it; ties go
// to the smaller S (less reduction work). Splitting is only worth its cross-block
// reduction when the predicted load drops >= 10% below S = 1: without that margin the
// rule chose S = 5 for 1536x8960 (2.5% predicted gain) and measured 30.7 us vs 27.7 us
// at S = 1. The first rule — "just exceed one wave of resident blocks" — picked S = 5
// for 8960x1536 (30.7 us) where S = 3 gives 27.3 us.
int w4a16_v1_splits(int M, int N, int K) {
  // Experiment override: W4A16_V1_SPLITS=<S> forces S splits (clamped to [1, 32]).
  if (const char* env = std::getenv("W4A16_V1_SPLITS"))
    return std::max(1, std::min(std::atoi(env), kMaxSplits));
  const long long base = static_cast<long long>((N + kLanes - 1) / kLanes) * M;
  const int sms = sm_count();
  const int words = (K + 7) / 8;
  const int s_max = std::max(1, std::min(kAutoMaxSplits, words / kWarps));  // >= 1 word per warp
  auto load = [&](int s) { return static_cast<double>((base * s + sms - 1) / sms) / s; };
  const double cutoff = 0.9 * load(1);
  int best_s = 1;
  double best_load = load(1);
  for (int s = 2; s <= s_max; ++s) {
    const double l = load(s);
    if (l <= cutoff + 1e-9 && l < best_load - 1e-9) {
      best_load = l;
      best_s = s;
    }
  }
  return best_s;
}

void launch_w4a16_v1_variant(const __half* A, const uint32_t* qweight, const __half* scales,
                             __half* C, int M, int N, int K, int group_size, cudaStream_t stream,
                             bool contiguous, bool specialize_tail) {
  const int splits = w4a16_v1_splits(M, N, K);
  const int tiles = (N + kLanes - 1) / kLanes;
  if (splits > 1)
    ensure_workspace(static_cast<size_t>(splits) * M * N, static_cast<size_t>(M) * tiles);
  const dim3 block(kLanes, kWarps);
  const dim3 grid(tiles, M, splits);
  const bool tail = !specialize_tail || K % 8 != 0;
#define V1_LAUNCH(C_, T_)                                                                  \
  w4a16_v1_kernel<C_, T_><<<grid, block, 0, stream>>>(A, qweight, scales, C, g_partials, \
                                                      g_counters, M, N, K, group_size, splits)
  if (contiguous && tail) V1_LAUNCH(true, true);
  else if (contiguous) V1_LAUNCH(true, false);
  else if (tail) V1_LAUNCH(false, true);
  else V1_LAUNCH(false, false);
#undef V1_LAUNCH
  CUDA_CHECK(cudaGetLastError());
}

void launch_w4a16_v1(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                     int M, int N, int K, int group_size, cudaStream_t stream) {
  // Final v1: contiguous word run per warp + tail check compiled out when K % 8 == 0.
  // (First measured version was interleaved + run-time tail check: 15-20% slower.)
  launch_w4a16_v1_variant(A, qweight, scales, C, M, N, K, group_size, stream, true, true);
}
