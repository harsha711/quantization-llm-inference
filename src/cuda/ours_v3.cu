// v3 — fast 4-bit -> FP16 unpack ("magic number" trick), on top of v2.
//
// Why: in a latency-bound loop every instruction per weight costs time (the v0 and v1
// lessons). v2 rebuilds each weight with roughly: shift, mask, int->float, subtract 8,
// multiply by the scale, round to FP16, back to FP32, FMA.
//
// The trick. FP16 1024.0 is 0x6400, and between 1024 and 2048 FP16 numbers are
// spaced exactly 1 apart, so the low 10 bits are an integer added to 1024:
//
//     0x6400 | q      ==  1024 + q          (q = 0..15, exact)
//
// One 32-bit AND/OR on the packed word therefore produces TWO FP16 values at once
// (a half2), with no int->float conversion. Our packing puts row j in bits [4j, 4j+4):
//
//   (word & 0x000F000F) | 0x64006400  ->  (1024 + q0,      1024 + q4)
//   (word & 0x00F000F0) | 0x64006400  ->  (1024 + 16 q1,   1024 + 16 q5)
//   then word >> 8 gives rows (2, 6) and (3, 7) the same way.
//
//   lo - 1032              = q - 8                      (__hsub2)
//   hi * (1/16) + (-72)    = 64 + q - 72 = q - 8        (__hfma2)
//
// Every step is exact in FP16. Then W_hat = (q - 8) * s with one __hmul2: the exact
// product rounded once to FP16, which is the same number v2 gets from
// __float2half_rn((float)(q - 8) * s). So W_hat is bit-identical to v2, and v3 keeps
// v2's FMA order (rows 0..7 in sequence, FP32 accumulation), so its OUTPUT is
// bit-identical to v2 at the same split S. The bench checks that bitwise.
//
// Grid, split rule, reduction: exactly v2 (one idea per version).
#include "kernels.cuh"

namespace {

constexpr int kLanes = 32;
constexpr int kWarps = 8;
constexpr int kCols = 4;
constexpr int kTileCols = kLanes * kCols;

__device__ __forceinline__ __half2 bits_to_half2(uint32_t u) {
  return *reinterpret_cast<const __half2*>(&u);
}

// Packed word (8 x 4-bit) -> 8 FP32 copies of W_hat = fp16((q - 8) * s), in row order.
__device__ __forceinline__ void unpack8(uint32_t word, __half2 s2, float w[8]) {
  const __half2 k1032 = bits_to_half2(0x64086408u);  // {1032, 1032}
  const __half2 k1_16 = bits_to_half2(0x2C002C00u);  // {1/16, 1/16}
  const __half2 km72 = bits_to_half2(0xD480D480u);   // {-72, -72}
#pragma unroll
  for (int p = 0; p < 2; ++p) {  // p = 0: rows (0,4),(1,5); p = 1: rows (2,6),(3,7)
    const uint32_t v = word >> (8 * p);
    const __half2 lo = bits_to_half2((v & 0x000F000Fu) | 0x64006400u);
    const __half2 hi = bits_to_half2((v & 0x00F000F0u) | 0x64006400u);
    const float2 wa = __half22float2(__hmul2(__hsub2(lo, k1032), s2));        // rows 2p,   2p+4
    const float2 wb = __half22float2(__hmul2(__hfma2(hi, k1_16, km72), s2));  // rows 2p+1, 2p+5
    w[2 * p] = wa.x;
    w[2 * p + 4] = wa.y;
    w[2 * p + 1] = wb.x;
    w[2 * p + 5] = wb.y;
  }
}

__global__ void __launch_bounds__(kLanes * kWarps)
    w4a16_v3_kernel(const __half* __restrict__ A, const uint32_t* __restrict__ qweight,
                    const __half* __restrict__ scales, __half* __restrict__ C,
                    float* __restrict__ partials, unsigned* __restrict__ counters, int M, int N,
                    int K, int group_size, int splits) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tile = blockIdx.x;
  const int m = blockIdx.y;
  const int z = blockIdx.z;
  const int n0 = (tile * kLanes + lane) * kCols;
  const bool active = n0 < N;

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

      float a[8];
      const float2 a01 = __half22float2(bits_to_half2(a8.x));
      const float2 a23 = __half22float2(bits_to_half2(a8.y));
      const float2 a45 = __half22float2(bits_to_half2(a8.z));
      const float2 a67 = __half22float2(bits_to_half2(a8.w));
      a[0] = a01.x; a[1] = a01.y; a[2] = a23.x; a[3] = a23.y;
      a[4] = a45.x; a[5] = a45.y; a[6] = a67.x; a[7] = a67.y;

      const __half2 s01 = bits_to_half2(s4.x), s23 = bits_to_half2(s4.y);
      const __half2 s2[kCols] = {__low2half2(s01), __high2half2(s01), __low2half2(s23),
                                 __high2half2(s23)};
      const uint32_t q[kCols] = {q4.x, q4.y, q4.z, q4.w};

#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        float w[8];
        unpack8(q[c], s2[c], w);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[c] = fmaf(a[j], w[j], acc[c]);  // v2's order
      }
    }
  }

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

  const size_t slot = static_cast<size_t>(m) * N + n0;
  auto store_c = [&](float4 v) {
    const __half2 lo = __floats2half2_rn(v.x, v.y);
    const __half2 hi = __floats2half2_rn(v.z, v.w);
    uint2 out;
    out.x = *reinterpret_cast<const uint32_t*>(&lo);
    out.y = *reinterpret_cast<const uint32_t*>(&hi);
    *reinterpret_cast<uint2*>(C + slot) = out;
  };

  if (splits == 1) {
    if (warp == 0 && active) store_c(sum);
    return;
  }

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
      const float4 p = __ldcg(&parts4[static_cast<size_t>(zz) * mn4 + slot / 4]);
      total.x += p.x; total.y += p.y; total.z += p.z; total.w += p.w;
    }
    store_c(total);
  }
  if (lane == 0 && warp == 0) *counter = 0u;
}

}  // namespace

void launch_w4a16_v3(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                     int M, int N, int K, int group_size, cudaStream_t stream) {
  if (!w4a16_v2_fast_path(A, qweight, scales, C, N, K)) {
    launch_w4a16_v1(A, qweight, scales, C, M, N, K, group_size, stream);
    return;
  }
  const int splits = w4a16_v2_splits(M, N, K);  // same grid as v2
  float* partials = nullptr;
  unsigned* counters = nullptr;
  w4a16_v2_workspace(M, N, splits, &partials, &counters);  // shared with v2; never concurrent
  const dim3 block(kLanes, kWarps);
  const dim3 grid((N + kTileCols - 1) / kTileCols, M, splits);
  w4a16_v3_kernel<<<grid, block, 0, stream>>>(A, qweight, scales, C, partials, counters, M, N, K,
                                             group_size, splits);
  CUDA_CHECK(cudaGetLastError());
}
