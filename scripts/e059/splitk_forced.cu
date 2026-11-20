// E059 part (b): forced-S launcher for E047's UNCHANGED tc_splitk kernel (diagnostic split sweep).
// E047's source is included verbatim (src/cuda/w4a16_tc_splitk.cu is not edited and is not compiled
// separately into the E059 library); this file only adds a host launcher whose grid z is the requested S
// instead of E047's fixed rule (S = 4 or tc). Workspace/counter handling is E047's launcher code with S
// in place of kSplit. Device code is unchanged (the analyzer requires its SASS to equal the same-job
// w4a16_bench build). S = 1 runs the split kernel with one block per tile (not the tc kernel).
#include "../../src/cuda/w4a16_tc_splitk.cu"

extern "C" int e059_tc_splitk(const void* A, const void* qweight, const void* scales, void* C, int M,
                              int N, int K, int group_size, int S, void* stream) {
  const __half* a = static_cast<const __half*>(A);
  __half* c = static_cast<__half*>(C);
  // E047's launcher preconditions (factor 0 = inblock_m4_vec fallback): refuse instead of falling back.
  if (S < 1 || w4a16_tc_splitk_factor(a, c, M, N, K, group_size) == 0) return -1;
  const size_t tiles = static_cast<size_t>(N / 32) * ((M + 15) / 16);
  if (tiles * S * 512 > g_ws_floats) {
    CUDA_CHECK(cudaFree(g_ws));
    g_ws_floats = tiles * S * 512;
    CUDA_CHECK(cudaMalloc(&g_ws, g_ws_floats * sizeof(float)));
  }
  if (tiles > g_counter_n) {
    CUDA_CHECK(cudaFree(g_counters));
    g_counter_n = tiles;
    CUDA_CHECK(cudaMalloc(&g_counters, g_counter_n * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(g_counters, 0, g_counter_n * sizeof(unsigned)));
  }
  const dim3 grid(N / 32, (M + 15) / 16, S), block(tc_common::kWarps * 32);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (M % 16 == 0)
    w4a16_tc_splitk_kernel<false><<<grid, block, 0, s>>>(a, static_cast<const uint32_t*>(qweight),
                                                          static_cast<const __half*>(scales), c, M, N, K,
                                                          group_size, g_ws, g_counters);
  else
    w4a16_tc_splitk_kernel<true><<<grid, block, 0, s>>>(a, static_cast<const uint32_t*>(qweight),
                                                         static_cast<const __half*>(scales), c, M, N, K,
                                                         group_size, g_ws, g_counters);
  return static_cast<int>(cudaGetLastError());
}

// Resident blocks per SM of the split kernel (recorded; E047's rule uses it).
extern "C" int e059_tc_splitk_occupancy() {
  int occ = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, w4a16_tc_splitk_kernel<false>,
                                                           tc_common::kWarps * 32, 0));
  return occ;
}
