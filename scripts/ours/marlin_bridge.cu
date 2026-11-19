// C ABI over our kernels for scripts/ours/marlin_compare.py (docs/OURS.md, "Marlin
// under the read flush"). Modelled on Harsha's scripts/e004/bridge.cu, which is left
// unchanged; this one adds v1/v2/v3 and the read-only flush.
#include <algorithm>
#include <exception>
#include <numeric>

#include "kernels.cuh"
#include "w4a16/fp16_host.h"
#include "w4a16/reference.h"
#include "w4a16/synthetic.h"

extern "C" int ours_generate(int m, int n, int k, uint64_t seed, uint16_t* a, uint16_t* w,
                             uint32_t* q, uint16_t* s, uint16_t* wh) {
  try {
    auto p = w4a16::make_problem(m, n, k, 128, seed);
    std::copy(p.A.begin(), p.A.end(), a);
    std::copy(p.W.begin(), p.W.end(), w);
    std::copy(p.qw.qweight.begin(), p.qw.qweight.end(), q);
    std::copy(p.qw.scales.begin(), p.qw.scales.end(), s);
    auto h = w4a16::dequantize_all_fp16(p.qw);
    std::copy(h.begin(), h.end(), wh);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}

// FP64 reference C = A * W for all m rows (W given as FP16 bits).
extern "C" void ours_reference(const uint16_t* a, const uint16_t* w, int m, int n, int k,
                               double* out) {
  std::vector<float> wf(static_cast<size_t>(k) * n);
  for (size_t i = 0; i < wf.size(); ++i) wf[i] = w4a16::half_to_float(w[i]);
  std::vector<int> rows(m);
  std::iota(rows.begin(), rows.end(), 0);
  auto r = w4a16::reference_gemm_rows(a, wf.data(), m, n, k, rows);
  std::copy(r.begin(), r.end(), out);
}

// Math mode 16 = CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION (FP32 partial sums),
// as E004 used so cuBLAS passes the FP64 check. Set here rather than from Python so it
// goes through the same libcublas copy that created the handle (torch loads its own).
extern "C" void* ours_handle() {
  cublasHandle_t h;
  CUBLAS_CHECK(cublasCreate(&h));
  CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  return h;
}
extern "C" void ours_destroy(void* h) { CUBLAS_CHECK(cublasDestroy((cublasHandle_t)h)); }
extern "C" int ours_l2() {
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  return p.l2CacheSize;
}
extern "C" void ours_flush_store(void* buf, size_t bytes, void* stream) {
  launch_l2_flush(buf, bytes, (cudaStream_t)stream);
}
extern "C" void ours_flush_read(const void* buf, size_t bytes, void* sink, void* stream) {
  launch_l2_flush_read(buf, bytes, static_cast<uint32_t*>(sink), (cudaStream_t)stream);
}

// Workspace growth inside v1/v2/v3 synchronizes the device; marlin_compare.py makes
// one untimed call per (kernel, shape, M) before any graph capture, so no growth
// happens while capturing.
extern "C" int ours_v2_splits(int m, int n, int k) { return w4a16_v2_splits(m, n, k); }

// IDs are this file's ABI, mirrored in marlin_compare.py (OURS_IDS).
extern "C" void ours_launch(int id, const __half* a, const uint32_t* q, const __half* s,
                            const __half* w, __half* c, int m, int n, int k, void* stream_ptr,
                            void* handle_ptr) {
  auto stream = (cudaStream_t)stream_ptr;
  switch (id) {
    case 0: launch_w4a16_v1(a, q, s, c, m, n, k, 128, stream); break;
    case 1: launch_w4a16_v2(a, q, s, c, m, n, k, 128, stream); break;
    case 2: launch_w4a16_v3(a, q, s, c, m, n, k, 128, stream); break;
    case 3: {
      auto h = (cublasHandle_t)handle_ptr;
      CUBLAS_CHECK(cublasSetStream(h, stream));
      launch_cublas_fp16(h, a, w, c, m, n, k);
      break;
    }
    case 4: launch_w4a16_inblock_specialized(a, q, s, c, m, n, k, 128, stream); break;
    default: std::fprintf(stderr, "unknown ours_launch id %d\n", id); std::exit(1);
  }
  CUDA_CHECK(cudaGetLastError());
}
