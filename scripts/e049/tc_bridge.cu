// E049: C entry points for the unchanged E044 `tc` and E048 `tc_m32` launchers (project-layout codes/scales).
#include "kernels.cuh"

extern "C" int e049_tc(const void* A, const void* qweight, const void* scales, void* C, int M, int N,
                       int K, int group_size, void* stream) {
  launch_w4a16_tc(static_cast<const __half*>(A), static_cast<const uint32_t*>(qweight),
                  static_cast<const __half*>(scales), static_cast<__half*>(C), M, N, K, group_size,
                  static_cast<cudaStream_t>(stream));
  return static_cast<int>(cudaGetLastError());
}

extern "C" int e049_tc_m32(const void* A, const void* qweight, const void* scales, void* C, int M, int N,
                           int K, int group_size, void* stream) {
  launch_w4a16_tc_m32(static_cast<const __half*>(A), static_cast<const uint32_t*>(qweight),
                      static_cast<const __half*>(scales), static_cast<__half*>(C), M, N, K, group_size,
                      static_cast<cudaStream_t>(stream));
  return static_cast<int>(cudaGetLastError());
}
