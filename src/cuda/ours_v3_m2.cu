// E053: v3 with two rows sharing each weight load/unpack. Per-row arithmetic and split rule stay fixed.
#include "kernels.cuh"

namespace {
constexpr int kLanes = 32, kWarps = 8, kCols = 4, kTileCols = kLanes*kCols;

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

template <bool HasMTail>
__global__ void __launch_bounds__(kLanes*kWarps)
w4a16_v3_m2_kernel(const __half* __restrict__ A, const uint32_t* __restrict__ qweight,
                   const __half* __restrict__ scales, __half* __restrict__ C,
                   float* __restrict__ partials, unsigned* __restrict__ counters,
                   int M, int N, int K, int group_size, int splits) {
  const int lane = threadIdx.x, warp = threadIdx.y, tile = blockIdx.x;
  const int m = 2*blockIdx.y, z = blockIdx.z;
  const int n0 = (tile*kLanes+lane)*kCols;
  const bool active = n0 < N;
  const int words = K/8;
  const int begin = static_cast<int>(static_cast<long long>(words)*z/splits);
  const int end = static_cast<int>(static_cast<long long>(words)*(z+1)/splits);
  const int first = begin+(end-begin)*warp/kWarps;
  const int last = begin+(end-begin)*(warp+1)/kWarps;
  float acc[2][kCols] = {};
  if (active) {
    const uint4* a_words = reinterpret_cast<const uint4*>(A+static_cast<size_t>(m)*K);
    for (int k8 = first; k8 < last; ++k8) {
      const uint4 q4 = *reinterpret_cast<const uint4*>(qweight+static_cast<size_t>(k8)*N+n0);
      const uint2 s4 = *reinterpret_cast<const uint2*>(scales+static_cast<size_t>(8*k8/group_size)*N+n0);
      float a[2][8] = {};
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        if (!HasMTail || m+r < M) {
          const uint4 a8 = a_words[static_cast<size_t>(r)*words+k8];
          const float2 a01 = __half22float2(bits_to_half2(a8.x));
          const float2 a23 = __half22float2(bits_to_half2(a8.y));
          const float2 a45 = __half22float2(bits_to_half2(a8.z));
          const float2 a67 = __half22float2(bits_to_half2(a8.w));
          a[r][0]=a01.x; a[r][1]=a01.y; a[r][2]=a23.x; a[r][3]=a23.y;
          a[r][4]=a45.x; a[r][5]=a45.y; a[r][6]=a67.x; a[r][7]=a67.y;
        }
      }
      const __half2 s01 = bits_to_half2(s4.x), s23 = bits_to_half2(s4.y);
      const __half2 s2[kCols] = {__low2half2(s01), __high2half2(s01), __low2half2(s23), __high2half2(s23)};
      const uint32_t q[kCols] = {q4.x,q4.y,q4.z,q4.w};
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        float w[8];
        unpack8(q[c],s2[c],w);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
          for (int r = 0; r < 2; ++r)
            if (!HasMTail || m+r < M) acc[r][c] = fmaf(a[r][j],w[j],acc[r][c]);
        }
      }
    }
  }
  __shared__ float4 part[2][kWarps][kLanes];
#pragma unroll
  for (int r = 0; r < 2; ++r)
    part[r][warp][lane] = make_float4(acc[r][0],acc[r][1],acc[r][2],acc[r][3]);
  __syncthreads();
  float4 sum[2] = {make_float4(0,0,0,0),make_float4(0,0,0,0)};
  if (warp == 0) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
      for (int y = 0; y < kWarps; ++y) {
        const float4 p = part[r][y][lane];
        sum[r].x+=p.x; sum[r].y+=p.y; sum[r].z+=p.z; sum[r].w+=p.w;
      }
    }
  }
  const size_t slot = static_cast<size_t>(m)*N+n0;
  auto store_c = [&](float4 v,int r) {
    const __half2 lo = __floats2half2_rn(v.x,v.y), hi = __floats2half2_rn(v.z,v.w);
    uint2 out;
    out.x = *reinterpret_cast<const uint32_t*>(&lo);
    out.y = *reinterpret_cast<const uint32_t*>(&hi);
    *reinterpret_cast<uint2*>(C+slot+static_cast<size_t>(r)*N) = out;
  };
  if (splits == 1) {
    if (warp == 0 && active) {
#pragma unroll
      for (int r = 0; r < 2; ++r)
        if (!HasMTail || m+r < M) store_c(sum[r],r);
    }
    return;
  }
  float4* parts4 = reinterpret_cast<float4*>(partials);
  const size_t mn4 = static_cast<size_t>(M)*N/4;
  if (warp == 0 && active) {
#pragma unroll
    for (int r = 0; r < 2; ++r)
      if (!HasMTail || m+r < M)
        parts4[static_cast<size_t>(z)*mn4+(slot+static_cast<size_t>(r)*N)/4] = sum[r];
  }
  __threadfence();
  __syncthreads();
  __shared__ bool is_last;
  unsigned* counter = counters+static_cast<size_t>(blockIdx.y)*gridDim.x+tile;
  if (lane == 0 && warp == 0) is_last = atomicAdd(counter,1u) == static_cast<unsigned>(splits-1);
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  if (warp == 0 && active) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      if (!HasMTail || m+r < M) {
        float4 total = make_float4(0,0,0,0);
        for (int zz = 0; zz < splits; ++zz) {
          const float4 p = __ldcg(&parts4[static_cast<size_t>(zz)*mn4+(slot+static_cast<size_t>(r)*N)/4]);
          total.x+=p.x; total.y+=p.y; total.z+=p.z; total.w+=p.w;
        }
        store_c(total,r);
      }
    }
  }
  if (lane == 0 && warp == 0) *counter = 0u;
}
}  // namespace

void launch_w4a16_v3_m2(const __half* A, const uint32_t* qweight, const __half* scales,
                        __half* C, int M, int N, int K, int group_size, cudaStream_t stream) {
  if (M == 1 || !w4a16_v2_fast_path(A,qweight,scales,C,N,K)) {
    launch_w4a16_v3(A,qweight,scales,C,M,N,K,group_size,stream);
    return;
  }
  const int splits = w4a16_v2_splits(M,N,K);
  float* partials = nullptr;
  unsigned* counters = nullptr;
  w4a16_v2_workspace(M,N,splits,&partials,&counters);
  const dim3 block(kLanes,kWarps), grid((N+kTileCols-1)/kTileCols,(M+1)/2,splits);
  if (M%2 == 0)
    w4a16_v3_m2_kernel<false><<<grid,block,0,stream>>>(A,qweight,scales,C,partials,counters,M,N,K,group_size,splits);
  else
    w4a16_v3_m2_kernel<true><<<grid,block,0,stream>>>(A,qweight,scales,C,partials,counters,M,N,K,group_size,splits);
  CUDA_CHECK(cudaGetLastError());
}
