// E044: our W4A16 GEMM on FP16 Tensor Cores (mma.sync.m16n8k16, FP32 accumulate).
// Block: 8 warps split K into 64-wide chunks; tile 16 rows x 32 columns; FP32 partials reduced in
// shared memory in fixed warp order. Fragment layouts follow the PTX ISA m16n8k16 description.
//
// Inside a chunk, thread t = lane%4 owns K positions [16t, 16t+16): packed words 2t and 2t+1 of its
// column, and the same 16 activations (two 16-byte loads) of its rows. MMA step j feeds logical
// k = 2t+sub from word 2t nibble 2j+sub and logical k = 2t+8+sub from word 2t+1 nibble 2j+sub; A uses
// the identical map, so the dot product is exact and only the FP32 addition order differs.
#include <cstring>

#include "kernels.cuh"

namespace {

constexpr int kWarps = 8;
constexpr int kTiles = 4;  // n8 tiles per warp -> 32 columns

__device__ __forceinline__ uint32_t h2_bits(__half2 h) { uint32_t u; memcpy(&u, &h, 4); return u; }
__device__ __forceinline__ __half2 bits_h2(uint32_t u) { __half2 h; memcpy(&h, &u, 4); return h; }

// Codes at nibbles 2j (low half) and 2j+1 (high half) of word -> FP16((q-8)*s) for each.
template <bool Fast>
__device__ __forceinline__ uint32_t unpack2(uint32_t word, int j, __half2 s2, float s) {
  const uint32_t x = word >> (8 * j);
  if constexpr (Fast) {
    // FP16 1024+q exactly; minus 1032 gives q-8 exactly; FP16 multiply rounds the exact product once,
    // which equals FP16(FP32(q-8)*s) because that FP32 product is exact. Checked exhaustively.
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

template <bool Fast, bool HasMTail>
__global__ void __launch_bounds__(kWarps * 32)
    w4a16_tc_kernel(const __half* __restrict__ A, const uint32_t* __restrict__ qweight,
                    const __half* __restrict__ scales, __half* __restrict__ C, int M, int N, int K,
                    int group_size) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int t = lane & 3, g = lane >> 2;
  const int n0 = blockIdx.x * 32, m0 = blockIdx.y * 16;
  const int chunks = K / 64, chunks_per_group = group_size / 64;
  const int c_begin = chunks * warp / kWarps, c_end = chunks * (warp + 1) / kWarps;
  const int r0 = m0 + g, r1 = r0 + 8;
  const bool v0 = !HasMTail || r0 < M, v1 = !HasMTail || r1 < M;
  float acc[kTiles][4] = {};
  for (int c = c_begin; c < c_end; ++c) {
    const int k0 = c * 64 + 16 * t;
    const uint4 z = make_uint4(0, 0, 0, 0);
    uint4 a0lo = z, a0hi = z, a1lo = z, a1hi = z;
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
                 unpack2<Fast>(w0, j, s2, s), unpack2<Fast>(w1, j, s2, s));
    }
  }

  __shared__ float partial[kWarps][16][32];
#pragma unroll
  for (int i = 0; i < kTiles; ++i) {
    const int col = 8 * i + 2 * t;
    partial[warp][g][col] = acc[i][0];
    partial[warp][g][col + 1] = acc[i][1];
    partial[warp][g + 8][col] = acc[i][2];
    partial[warp][g + 8][col + 1] = acc[i][3];
  }
  // Warps with no chunks publish zeros; every thread reaches the barrier.
  __syncthreads();
  const int idx = threadIdx.x * 2, row = idx / 32, col = idx % 32;
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

template <bool Fast>
void launch_tc(const __half* A, const uint32_t* qweight, const __half* scales, __half* C, int M,
               int N, int K, int group_size, cudaStream_t stream) {
  if (K % 64 || group_size % 64 || N % 32 || reinterpret_cast<uintptr_t>(A) % 16)
    return launch_w4a16_inblock_m4_vec(A, qweight, scales, C, M, N, K, group_size, stream);
  const dim3 grid(N / 32, (M + 15) / 16), block(kWarps * 32);
  if (M % 16 == 0)
    w4a16_tc_kernel<Fast, false><<<grid, block, 0, stream>>>(A, qweight, scales, C, M, N, K, group_size);
  else
    w4a16_tc_kernel<Fast, true><<<grid, block, 0, stream>>>(A, qweight, scales, C, M, N, K, group_size);
  CUDA_CHECK(cudaGetLastError());
}

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

void launch_w4a16_tc(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                     int M, int N, int K, int group_size, cudaStream_t stream) {
  launch_tc<true>(A, qweight, scales, C, M, N, K, group_size, stream);
}

void launch_w4a16_tc_cvt(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                         int M, int N, int K, int group_size, cudaStream_t stream) {
  launch_tc<false>(A, qweight, scales, C, M, N, K, group_size, stream);
}

unsigned long long w4a16_tc_unpack_selftest() {
  unsigned long long* d = nullptr;
  unsigned long long h = 0;
  CUDA_CHECK(cudaMalloc(&d, sizeof(h)));
  CUDA_CHECK(cudaMemset(d, 0, sizeof(h)));
  unpack_selftest_kernel<<<256, 256>>>(d);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d));
  return h;
}
