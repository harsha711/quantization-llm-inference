// E057 part B: measured GPU roofs (descriptive microbenchmarks, not a GEMM kernel).
//   e057_gpu [--small|--tiny]
// (1) bandwidth: cudaMemcpyAsync device-to-device (bytes = read + write) and a read-only kernel
//     (16-byte loads, per-warp shuffle reduction, one atomic per warp) over a buffer >> 32 MiB L2;
// (2) tensor rate: a kernel issuing only mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 (the
//     instruction shape of our tc kernels) from registers, NACC independent accumulators per warp.
// Correctness (outside timing): copy mismatches = 0; read-kernel sum = closed form; one-tile MMA vs
// FP64 reference with the PTX fragment layouts; every timed-MMA accumulator equals iters/16 exactly.
// Output: CSV lines "kind,..." on stdout, phase markers "PHASE name begin|end epoch_ms".
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                     \
  do {                                                                                            \
    cudaError_t e_ = (x);                                                                         \
    if (e_ != cudaSuccess) {                                                                      \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(3);                                                                               \
    }                                                                                             \
  } while (0)

static long long epoch_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}
static void phase(const char* name, const char* what) {
  std::printf("PHASE %s %s %lld\n", name, what, epoch_ms());
  std::fflush(stdout);
}
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

__global__ void fill_index(uint32_t* p, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = (uint32_t)i;
}
__global__ void count_mismatch(const uint32_t* p, size_t n, unsigned long long* bad) {
  unsigned long long b = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    b += p[i] != (uint32_t)i;
  if (b) atomicAdd(bad, b);
}

template <int UNROLL>
__global__ void read_kernel(const uint4* __restrict__ p, size_t n16, unsigned long long* out) {
  unsigned long long s = 0;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  for (; i + (UNROLL - 1) * stride < n16; i += UNROLL * stride) {
    uint4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) v[u] = p[i + u * stride];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) s += (unsigned long long)v[u].x + v[u].y + v[u].z + v[u].w;
  }
  for (; i < n16; i += stride) {
    uint4 v = p[i];
    s += (unsigned long long)v.x + v.y + v.z + v.w;
  }
  for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Peak kernel: fragments come from global memory (not compile-time constants), NACC independent chains.
template <int NACC>
__global__ void mma_peak(const uint32_t* in, float* out, int iters) {
  const uint32_t a0 = in[0], a1 = in[1], a2 = in[2], a3 = in[3], b0 = in[4], b1 = in[5];
  float acc[NACC][4];
#pragma unroll
  for (int j = 0; j < NACC; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < NACC; ++j) mma16816(acc[j], a0, a1, a2, a3, b0, b1);
  }
  float* o = out + ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * NACC * 4;
#pragma unroll
  for (int j = 0; j < NACC; ++j)
#pragma unroll
    for (int k = 0; k < 4; ++k) o[j * 4 + k] = acc[j][k];
}

// One-tile check with the PTX ISA m16n8k16 .f16 fragment layouts. A 16x16 row-major, B 16x8 stored B[k*8+n],
// C/D 16x8 row-major FP32.
__global__ void mma_tile(const uint16_t* A, const uint16_t* B, const float* C, float* D) {
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  uint32_t a[4], b[2];
  for (int r = 0; r < 4; ++r) {  // a_r holds elements 2r, 2r+1
    uint32_t lo, hi;
    for (int h = 0; h < 2; ++h) {
      int i = 2 * r + h, row = g + 8 * ((i >> 1) & 1), col = t * 2 + (i & 1) + 8 * (i >> 2);
      (h ? hi : lo) = A[row * 16 + col];
    }
    a[r] = lo | (hi << 16);
  }
  for (int r = 0; r < 2; ++r) {
    uint32_t lo, hi;
    for (int h = 0; h < 2; ++h) {
      int i = 2 * r + h, k = t * 2 + (i & 1) + 8 * (i >> 1);
      (h ? hi : lo) = B[k * 8 + g];
    }
    b[r] = lo | (hi << 16);
  }
  float d[4];
  for (int i = 0; i < 4; ++i) d[i] = C[(g + 8 * (i >> 1)) * 8 + t * 2 + (i & 1)];
  mma16816(d, a[0], a[1], a[2], a[3], b[0], b[1]);
  for (int i = 0; i < 4; ++i) D[(g + 8 * (i >> 1)) * 8 + t * 2 + (i & 1)] = d[i];
}

static uint16_t f2h(float f) {  // round-to-nearest-even FP16 for |f| in normal range (inputs in [-1,1])
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int e = (int)((x >> 23) & 0xff) - 127 + 15;
  uint32_t m = x & 0x7fffffu;
  if (e <= 0) return (uint16_t)sign;  // flush tiny values (not produced below)
  uint32_t hm = m >> 13, rem = m & 0x1fffu;
  uint32_t h = sign | ((uint32_t)e << 10) | hm;
  if (rem > 0x1000u || (rem == 0x1000u && (hm & 1))) ++h;
  return (uint16_t)h;
}
static double h2d(uint16_t h) {
  int e = (h >> 10) & 0x1f, m = h & 0x3ff;
  double v = e ? std::ldexp(1.0 + m / 1024.0, e - 15) : std::ldexp(m / 1024.0, -14);
  return (h & 0x8000) ? -v : v;
}

int main(int argc, char** argv) {
  const bool tiny = argc > 1 && !std::strcmp(argv[1], "--tiny");
  const bool small = argc > 1 && !std::strcmp(argv[1], "--small");
  int dev = 0;
  CK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  int sm_clock_khz = 0, mem_clock_khz = 0, bus_bits = 0;
  cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, dev);
  cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
  cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
  int drv = 0, rt = 0;
  cudaDriverGetVersion(&drv);
  cudaRuntimeGetVersion(&rt);
  std::printf("device,%s,sm_%d%d,SMs=%d,L2=%d,attr_clock_khz=%d,attr_mem_clock_khz=%d,bus_bits=%d,driver=%d,runtime=%d,mode=%s\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize, sm_clock_khz,
              mem_clock_khz, bus_bits, drv, rt, tiny ? "tiny" : small ? "small" : "full");
  const int SMS = prop.multiProcessorCount;
  int failures = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));

  // ---------------- (1) bandwidth ----------------
  const size_t bytes = tiny ? (1u << 20) : small ? (256ull << 20) : (1ull << 30);
  const int warm = tiny ? 1 : 5, reps = tiny ? 2 : small ? 20 : 100;
  const size_t words = bytes / 4;
  uint32_t *src, *dst;
  unsigned long long* dacc;
  CK(cudaMalloc(&src, bytes));
  CK(cudaMalloc(&dst, bytes));
  CK(cudaMalloc(&dacc, sizeof(unsigned long long)));
  fill_index<<<SMS * 8, 256>>>(src, words);
  CK(cudaMemset(dst, 0, bytes));
  CK(cudaDeviceSynchronize());

  phase("copy", "begin");
  {
    std::vector<double> ms;
    for (int r = 0; r < warm + reps; ++r) {
      CK(cudaEventRecord(e0));
      CK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice));
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      float t;
      CK(cudaEventElapsedTime(&t, e0, e1));
      if (r >= warm) ms.push_back(t);
    }
    CK(cudaMemset(dacc, 0, sizeof(unsigned long long)));
    count_mismatch<<<SMS * 8, 256>>>(dst, words, dacc);
    unsigned long long bad = 0;
    CK(cudaMemcpy(&bad, dacc, sizeof bad, cudaMemcpyDeviceToHost));
    if (bad) ++failures;
    double med = median(ms);
    std::printf("copy,bytes_per_copy=%zu,counted_bytes=%zu,warm=%d,reps=%d,median_ms=%.6f,min_ms=%.6f,max_ms=%.6f,GBps=%.3f,mismatches=%llu\n",
                bytes, 2 * bytes, warm, reps, med, *std::min_element(ms.begin(), ms.end()),
                *std::max_element(ms.begin(), ms.end()), 2.0 * bytes / (med * 1e-3) / 1e9, bad);
    for (double v : ms) std::fprintf(stderr, "copy_sample_ms %.6f\n", v);
  }
  phase("copy", "end");

  phase("read", "begin");
  {
    const unsigned long long n = words;
    const unsigned long long expect = n * (n - 1) / 2;  // sum of p[i] = i
    const int bps_list[] = {2, 4, 8, 16};
    for (int unroll : {2, 4, 8}) {
      for (int bps : bps_list) {
        const int blocks = SMS * bps, threads = 256;
        std::vector<double> ms;
        bool ok = true;
        for (int r = 0; r < warm + reps; ++r) {
          CK(cudaMemsetAsync(dacc, 0, sizeof(unsigned long long)));
          CK(cudaEventRecord(e0));
          if (unroll == 2) read_kernel<2><<<blocks, threads>>>((const uint4*)src, bytes / 16, dacc);
          if (unroll == 4) read_kernel<4><<<blocks, threads>>>((const uint4*)src, bytes / 16, dacc);
          if (unroll == 8) read_kernel<8><<<blocks, threads>>>((const uint4*)src, bytes / 16, dacc);
          CK(cudaEventRecord(e1));
          CK(cudaEventSynchronize(e1));
          CK(cudaGetLastError());
          float t;
          CK(cudaEventElapsedTime(&t, e0, e1));
          unsigned long long got = 0;
          CK(cudaMemcpy(&got, dacc, sizeof got, cudaMemcpyDeviceToHost));
          ok = ok && got == expect;
          if (r >= warm) ms.push_back(t);
        }
        if (!ok) ++failures;
        double med = median(ms);
        std::printf("read,bytes=%zu,unroll=%d,blocks_per_sm=%d,blocks=%d,threads=%d,warm=%d,reps=%d,median_ms=%.6f,min_ms=%.6f,max_ms=%.6f,GBps=%.3f,sum_ok=%d\n",
                    bytes, unroll, bps, blocks, threads, warm, reps, med, *std::min_element(ms.begin(), ms.end()),
                    *std::max_element(ms.begin(), ms.end()), bytes / (med * 1e-3) / 1e9, ok ? 1 : 0);
      }
    }
  }
  phase("read", "end");
  CK(cudaFree(src));
  CK(cudaFree(dst));

  // ---------------- (2) one-tile MMA correctness ----------------
  {
    std::mt19937 rng(57);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<uint16_t> A(256), B(128);
    std::vector<float> C(128), D(128);
    for (auto& v : A) v = f2h(U(rng));
    for (auto& v : B) v = f2h(U(rng));
    for (auto& v : C) v = U(rng);
    uint16_t *dA, *dB;
    float *dC, *dD;
    CK(cudaMalloc(&dA, 512));
    CK(cudaMalloc(&dB, 256));
    CK(cudaMalloc(&dC, 512));
    CK(cudaMalloc(&dD, 512));
    CK(cudaMemcpy(dA, A.data(), 512, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B.data(), 256, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dC, C.data(), 512, cudaMemcpyHostToDevice));
    mma_tile<<<1, 32>>>(dA, dB, dC, dD);
    CK(cudaGetLastError());
    CK(cudaMemcpy(D.data(), dD, 512, cudaMemcpyDeviceToHost));
    double maxerr = 0, maxrel = 0;
    bool ok = true;
    for (int i = 0; i < 16; ++i)
      for (int j = 0; j < 8; ++j) {
        double ref = C[i * 8 + j], mag = std::fabs(C[i * 8 + j]);
        for (int k = 0; k < 16; ++k) {
          double p = h2d(A[i * 16 + k]) * h2d(B[k * 8 + j]);
          ref += p;
          mag += std::fabs(p);
        }
        double err = std::fabs(D[i * 8 + j] - ref);
        maxerr = std::max(maxerr, err);
        maxrel = std::max(maxrel, err / mag);
        if (!(err <= 1e-5 * mag + 1e-6)) ok = false;  // FP32 accumulation of exact FP16 products
      }
    if (!ok) ++failures;
    std::printf("mma_tile,max_abs_err=%.3e,max_err_over_abs_sum=%.3e,pass=%d\n", maxerr, maxrel, ok ? 1 : 0);
    CK(cudaFree(dA));
    CK(cudaFree(dB));
    CK(cudaFree(dC));
    CK(cudaFree(dD));
  }

  // ---------------- (3) tensor rate ----------------
  phase("mma", "begin");
  {
    // every A/B element = 2^-4 (FP16 0x2C00): each MMA adds 16 * 2^-8 = 2^-4 to every accumulator, exactly.
    const uint32_t h = 0x2C002C00u;
    std::vector<uint32_t> in(6, h);
    uint32_t* din;
    CK(cudaMalloc(&din, 24));
    CK(cudaMemcpy(din, in.data(), 24, cudaMemcpyHostToDevice));
    const double target_mma = tiny ? 1e5 : 2.4e8;  // about 20 ms per launch near 47 TFLOP/s
    const int mwarm = tiny ? 1 : 3, mreps = tiny ? 2 : small ? 10 : 30;
    double best = 0;
    int best_nacc = 0, best_wpb = 0, best_bps = 0;
    auto run = [&](int nacc, int wpb, int bps, int reps_, const char* tag) {
      const int blocks = SMS * bps, threads = 32 * wpb;
      const long long warps = (long long)blocks * wpb;
      int iters = (int)std::max(1.0, std::min(16777216.0, std::round(target_mma / (warps * nacc))));
      float* dout;
      const size_t nout = (size_t)blocks * threads * nacc * 4;
      CK(cudaMalloc(&dout, nout * sizeof(float)));
      std::vector<double> ms;
      for (int r = 0; r < mwarm + reps_; ++r) {
        CK(cudaEventRecord(e0));
        if (nacc == 1) mma_peak<1><<<blocks, threads>>>(din, dout, iters);
        if (nacc == 2) mma_peak<2><<<blocks, threads>>>(din, dout, iters);
        if (nacc == 4) mma_peak<4><<<blocks, threads>>>(din, dout, iters);
        if (nacc == 8) mma_peak<8><<<blocks, threads>>>(din, dout, iters);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaGetLastError());
        float t;
        CK(cudaEventElapsedTime(&t, e0, e1));
        if (r >= mwarm) ms.push_back(t);
      }
      std::vector<float> out(nout);
      CK(cudaMemcpy(out.data(), dout, nout * sizeof(float), cudaMemcpyDeviceToHost));
      const float expect = (float)iters / 16.0f;
      size_t bad = 0;
      for (float v : out) bad += !(v == expect);
      if (bad) ++failures;
      CK(cudaFree(dout));
      const double med = median(ms), mmas = (double)warps * nacc * iters;
      const double tflops = mmas * 4096.0 / (med * 1e-3) / 1e12;
      std::printf("%s,nacc=%d,warps_per_block=%d,blocks_per_sm=%d,blocks=%d,iters=%d,mma_per_launch=%.0f,warm=%d,reps=%d,median_ms=%.6f,min_ms=%.6f,max_ms=%.6f,TFLOPs=%.4f,bad_outputs=%zu\n",
                  tag, nacc, wpb, bps, blocks, iters, mmas, mwarm, reps_, med, *std::min_element(ms.begin(), ms.end()),
                  *std::max_element(ms.begin(), ms.end()), tflops, bad);
      std::fflush(stdout);
      return tflops;
    };
    for (int nacc : {1, 2, 4, 8})
      for (int wpb : {4, 8})
        for (int bps : {1, 2, 4, 8}) {
          double t = run(nacc, wpb, bps, mreps, "mma");
          if (t > best) best = t, best_nacc = nacc, best_wpb = wpb, best_bps = bps;
        }
    phase("mma", "end");
    // Confirmation: the best configuration again, longer (selection bias check, clock sampling window).
    phase("mma_confirm", "begin");
    run(best_nacc, best_wpb, best_bps, tiny ? 2 : small ? 20 : 200, "mma_confirm");
    phase("mma_confirm", "end");
    CK(cudaFree(din));
  }
  CK(cudaDeviceSynchronize());
  std::printf("failures,%d\n", failures);
  return failures ? 1 : 0;
}
