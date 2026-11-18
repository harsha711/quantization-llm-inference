// Device helpers of E044's tc kernel, copied unchanged from src/cuda/w4a16_tc.cu so that later rungs
// (E047 onward) reuse them without editing the parent's source. Same fragment maps and fast unpack.
#pragma once
#include <cstring>

#include "kernels.cuh"

namespace tc_common {

constexpr int kWarps = 8;
constexpr int kTiles = 4;  // n8 tiles per warp -> 32 columns

__device__ __forceinline__ uint32_t h2_bits(__half2 h) { uint32_t u; memcpy(&u, &h, 4); return u; }
__device__ __forceinline__ __half2 bits_h2(uint32_t u) { __half2 h; memcpy(&h, &u, 4); return h; }

// Codes at nibbles 2j (low half) and 2j+1 (high half) of word -> FP16((q-8)*s) for each.
template <bool Fast>
__device__ __forceinline__ uint32_t unpack2(uint32_t word, int j, __half2 s2, float s) {
  const uint32_t x = word >> (8 * j);
  if constexpr (Fast) {
    const uint32_t v = (x & 0xFu) | ((x << 12) & 0xF0000u) | 0x64006400u;
    return h2_bits(__hmul2(__hsub2(bits_h2(v), __float2half2_rn(1032.f)), s2));
  } else {
    const int q0 = static_cast<int>(x & 0xFu), q1 = static_cast<int>((x >> 4) & 0xFu);
    return h2_bits(__halves2half2(__float2half_rn(static_cast<float>(q0 - 8) * s),
                                  __float2half_rn(static_cast<float>(q1 - 8) * s)));
  }
}

__device__ __forceinline__ uint32_t comp(const uint4& v, int j) {
  return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w;
}

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

namespace {  // internal linkage: each including file gets its own copy
// Every finite FP16 scale x every code at every nibble position: fast unpack == contract formula.
__global__ void unpack_selftest_kernel(unsigned long long* bad) {
  const int sb = blockIdx.x * blockDim.x + threadIdx.x;
  if (sb >= 65536) return;
  const __half sh = __ushort_as_half(static_cast<unsigned short>(sb));
  if (__hisinf(sh) || __hisnan(sh)) return;
  const float s = __half2float(sh);
  const __half2 s2 = __half2half2(sh);
  for (uint32_t q = 0; q < 16; ++q) {
    const uint32_t word = (q | ((15u - q) << 4)) * 0x01010101u;
    for (int j = 0; j < 4; ++j)
      if (unpack2<true>(word, j, s2, s) != unpack2<false>(word, j, s2, s)) atomicAdd(bad, 1ull);
  }
}
}  // namespace

}  // namespace tc_common
