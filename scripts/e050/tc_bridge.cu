// E050: C entry points for the unchanged E044 `tc` launcher and the `tc_ladder` dispatch rung.
// The rung is this dispatch only: no device code changes (tc/tc_m32 sources unchanged).
#include "kernels.cuh"

extern "C" int e050_tc(const void* A, const void* qweight, const void* scales, void* C, int M, int N,
                       int K, int group_size, void* stream) {
  launch_w4a16_tc(static_cast<const __half*>(A), static_cast<const uint32_t*>(qweight),
                  static_cast<const __half*>(scales), static_cast<__half*>(C), M, N, K, group_size,
                  static_cast<cudaStream_t>(stream));
  return static_cast<int>(cudaGetLastError());
}

// tc_ladder: M >= 64 runs the E048 tc_m32 launcher, M < 64 the E044 tc launcher (threshold from E049, post hoc).
extern "C" int e050_tc_ladder(const void* A, const void* qweight, const void* scales, void* C, int M,
                              int N, int K, int group_size, void* stream) {
  auto launch = M >= 64 ? launch_w4a16_tc_m32 : launch_w4a16_tc;
  launch(static_cast<const __half*>(A), static_cast<const uint32_t*>(qweight),
         static_cast<const __half*>(scales), static_cast<__half*>(C), M, N, K, group_size,
         static_cast<cudaStream_t>(stream));
  return static_cast<int>(cudaGetLastError());
}
