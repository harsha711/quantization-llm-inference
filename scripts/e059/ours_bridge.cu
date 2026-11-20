// E059: C entry points for the teammate's UNCHANGED v1/v2/v3 launchers and read-only L2 flush
// (origin/samarth/bandwidth-roofline commit cd39962, extracted verbatim to teammate_src/). Minimal adapter:
// it only refuses the silent v1 fallback of v2/v3 and the S-forcing environment overrides, so that every
// timed v3 call is the v3 kernel with its own split rule.
#include <cstdlib>

#include "teammate_src/kernels.cuh"

namespace {
bool overridden() { return std::getenv("W4A16_V1_SPLITS") || std::getenv("W4A16_V2_SPLITS"); }
}  // namespace

extern "C" int e059_v1(const void* A, const void* q, const void* s, void* C, int M, int N, int K,
                       int group_size, void* stream) {
  if (overridden()) return -2;
  launch_w4a16_v1(static_cast<const __half*>(A), static_cast<const uint32_t*>(q),
                  static_cast<const __half*>(s), static_cast<__half*>(C), M, N, K, group_size,
                  static_cast<cudaStream_t>(stream));
  return static_cast<int>(cudaGetLastError());
}

extern "C" int e059_v2_fast_path(const void* A, const void* q, const void* s, const void* C, int N, int K) {
  return w4a16_v2_fast_path(static_cast<const __half*>(A), static_cast<const uint32_t*>(q),
                            static_cast<const __half*>(s), static_cast<const __half*>(C), N, K);
}

extern "C" int e059_v2_splits(int M, int N, int K) { return w4a16_v2_splits(M, N, K); }
extern "C" int e059_v1_splits(int M, int N, int K) { return w4a16_v1_splits(M, N, K); }

#define E059_V(name, launcher)                                                                      \
  extern "C" int name(const void* A, const void* q, const void* s, void* C, int M, int N, int K,     \
                      int group_size, void* stream) {                                               \
    if (overridden()) return -2;                                                                    \
    if (!e059_v2_fast_path(A, q, s, C, N, K)) return -1; /* v1 fallback refused */                  \
    launcher(static_cast<const __half*>(A), static_cast<const uint32_t*>(q),                         \
             static_cast<const __half*>(s), static_cast<__half*>(C), M, N, K, group_size,            \
             static_cast<cudaStream_t>(stream));                                                    \
    return static_cast<int>(cudaGetLastError());                                                    \
  }
E059_V(e059_v2, launch_w4a16_v2)
E059_V(e059_v3, launch_w4a16_v3)
#undef E059_V

// Their read-only flush (their scripts/ours/marlin_compare.py "read_flush"): reads nbytes of a never-written buffer.
extern "C" int e059_flush_read(const void* buf, size_t nbytes, void* sink, void* stream) {
  launch_l2_flush_read(buf, nbytes, static_cast<uint32_t*>(sink), static_cast<cudaStream_t>(stream));
  return static_cast<int>(cudaGetLastError());
}
