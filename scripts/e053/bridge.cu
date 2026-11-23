#include "kernels.cuh"

extern "C" int e053_v3(const void* a, const void* q, const void* s, void* c,
                       int m, int n, int k, int group, void* stream) {
  launch_w4a16_v3(static_cast<const __half*>(a), static_cast<const uint32_t*>(q),
                  static_cast<const __half*>(s), static_cast<__half*>(c), m, n, k, group,
                  static_cast<cudaStream_t>(stream));
  return static_cast<int>(cudaGetLastError());
}

extern "C" int e053_tc_ladder(const void* a, const void* q, const void* s, void* c,
                              int m, int n, int k, int group, void* stream) {
  (m >= 64 ? launch_w4a16_tc_m32 : launch_w4a16_tc)(
      static_cast<const __half*>(a), static_cast<const uint32_t*>(q),
      static_cast<const __half*>(s), static_cast<__half*>(c), m, n, k, group,
      static_cast<cudaStream_t>(stream));
  return static_cast<int>(cudaGetLastError());
}

#ifdef E053_CANDIDATE
extern "C" int e053_v3_m2(const void* a, const void* q, const void* s, void* c,
                          int m, int n, int k, int group, void* stream) {
  launch_w4a16_v3_m2(static_cast<const __half*>(a), static_cast<const uint32_t*>(q),
                     static_cast<const __half*>(s), static_cast<__half*>(c), m, n, k, group,
                     static_cast<cudaStream_t>(stream));
  return static_cast<int>(cudaGetLastError());
}
#endif
