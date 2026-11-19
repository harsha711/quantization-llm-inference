// Achievable DRAM read bandwidth on this GPU: the ceiling behind `pct_floor`.
//
//   w4a16_bwprobe [--mib 256,1024] [--iters 20]
//
// Streams a buffer much larger than L2 with 128-bit loads and reports the best
// median GB/s over a few load-unroll depths. A W4A16 kernel at M=1 is a read
// stream of its weights, so this is the number it can at best approach. The
// datasheet peak (448 GB/s on the RTX 5060 Ti) is usually not reachable.
//
// The last line is machine-readable: `ceiling_gbps=<value>`.
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "kernels.cuh"

namespace {

template <int Unroll>
__global__ void read_stream(const uint4* __restrict__ in, size_t n, uint32_t* out) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint32_t acc = 0;
  for (; i + (Unroll - 1) * stride < n; i += Unroll * stride) {
    uint4 v[Unroll];
#pragma unroll
    for (int u = 0; u < Unroll; ++u) v[u] = in[i + u * stride];  // independent loads in flight
#pragma unroll
    for (int u = 0; u < Unroll; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  for (; i < n; i += stride) {
    const uint4 v = in[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  // Data-dependent store so the compiler cannot drop the loads.
  if (acc == 0x9e3779b9u) out[0] = acc;
}

template <int Unroll>
double measure(const uint4* buf, size_t n, uint32_t* out, int blocks, int threads, int iters) {
  for (int w = 0; w < 3; ++w) read_stream<Unroll><<<blocks, threads>>>(buf, n, out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  std::vector<double> ms(iters);
  for (int it = 0; it < iters; ++it) {
    CUDA_CHECK(cudaEventRecord(a));
    read_stream<Unroll><<<blocks, threads>>>(buf, n, out);
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float t = 0;
    CUDA_CHECK(cudaEventElapsedTime(&t, a, b));
    ms[it] = t;
  }
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  std::sort(ms.begin(), ms.end());
  const double median_s = ms[iters / 2] * 1e-3;
  return static_cast<double>(n) * sizeof(uint4) / median_s / 1e9;
}

template <int Unroll>
int full_grid(int threads) {
  int dev = 0, sms = 0, per_sm = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, read_stream<Unroll>, threads, 0));
  return sms * per_sm;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<size_t> mibs = {256, 1024};
  int iters = 20;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--mib" && i + 1 < argc) {
      mibs.clear();
      std::stringstream ss(argv[++i]);
      std::string tok;
      while (std::getline(ss, tok, ',')) mibs.push_back(std::stoull(tok));
    } else if (a == "--iters" && i + 1 < argc) {
      iters = std::stoi(argv[++i]);
    } else {
      std::fprintf(stderr, "usage: %s [--mib 256,1024] [--iters 20]\n", argv[0]);
      return EXIT_FAILURE;
    }
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s (sm_%d%d, %d SMs, L2 %.1f MiB)\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount, prop.l2CacheSize / 1048576.0);

  constexpr int kThreads = 256;
  uint32_t* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, sizeof(uint32_t)));
  double best = 0;
  std::printf("%8s %7s %8s %10s\n", "MiB", "unroll", "blocks", "GB/s");
  for (size_t mib : mibs) {
    const size_t bytes = mib << 20;
    if (bytes < 4 * static_cast<size_t>(prop.l2CacheSize))
      std::printf("warning: %zu MiB is under 4x L2; result may include cache hits\n", mib);
    uint4* buf = nullptr;
    CUDA_CHECK(cudaMalloc(&buf, bytes));
    CUDA_CHECK(cudaMemset(buf, 0x5a, bytes));
    const size_t n = bytes / sizeof(uint4);
    const double g1 = measure<1>(buf, n, out, full_grid<1>(kThreads), kThreads, iters);
    const double g4 = measure<4>(buf, n, out, full_grid<4>(kThreads), kThreads, iters);
    const double g8 = measure<8>(buf, n, out, full_grid<8>(kThreads), kThreads, iters);
    std::printf("%8zu %7d %8d %10.1f\n", mib, 1, full_grid<1>(kThreads), g1);
    std::printf("%8zu %7d %8d %10.1f\n", mib, 4, full_grid<4>(kThreads), g4);
    std::printf("%8zu %7d %8d %10.1f\n", mib, 8, full_grid<8>(kThreads), g8);
    best = std::max({best, g1, g4, g8});
    CUDA_CHECK(cudaFree(buf));
  }
  CUDA_CHECK(cudaFree(out));
  std::printf("ceiling_gbps=%.1f\n", best);
  return EXIT_SUCCESS;
}
