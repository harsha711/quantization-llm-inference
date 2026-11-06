// E004 adapter only. The retained CUDA kernels are compiled without source changes.
#include <algorithm>
#include <exception>
#include <numeric>
#include "kernels.cuh"
#include "w4a16/fp16_host.h"
#include "w4a16/reference.h"
#include "w4a16/synthetic.h"

extern "C" int e004_generate(int m, int n, int k, uint64_t seed, uint16_t* a,
                              uint16_t* w, uint32_t* q, uint16_t* s, uint16_t* wh) {
  try {
    auto p = w4a16::make_problem(m, n, k, 128, seed);
    std::copy(p.A.begin(), p.A.end(), a);
    std::copy(p.W.begin(), p.W.end(), w);
    std::copy(p.qw.qweight.begin(), p.qw.qweight.end(), q);
    std::copy(p.qw.scales.begin(), p.qw.scales.end(), s);
    auto h = w4a16::dequantize_all_fp16(p.qw);
    std::copy(h.begin(), h.end(), wh);
    return 0;
  } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}

extern "C" void e004_reference(const uint16_t* a, const uint16_t* w, int m, int n, int k,
                                double* out) {
  std::vector<float> wf(static_cast<size_t>(k)*n);
  for (size_t i=0; i<wf.size(); ++i) wf[i] = w4a16::half_to_float(w[i]);
  std::vector<int> rows(m);
  std::iota(rows.begin(), rows.end(), 0);
  auto r = w4a16::reference_gemm_rows(a, wf.data(), m, n, k, rows);
  std::copy(r.begin(), r.end(), out);
}

extern "C" void* e004_handle() {
  cublasHandle_t h;
  CUBLAS_CHECK(cublasCreate(&h));
  return h;
}
extern "C" void e004_destroy(void* h) { CUBLAS_CHECK(cublasDestroy((cublasHandle_t)h)); }
extern "C" int e004_l2() {
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  return p.l2CacheSize;
}
extern "C" void e004_flush(void* p, size_t bytes, void* stream) {
  launch_l2_flush(p, bytes, (cudaStream_t)stream);
}

// IDs are a fixed bridge ABI, recorded by bench.py.
extern "C" void e004_launch(int id, const __half* a, const uint32_t* q, const __half* s,
                            const __half* w, __half* wh, __half* c, int m, int n, int k,
                            void* stream_ptr, void* handle_ptr) {
  auto stream = (cudaStream_t)stream_ptr;
  auto h = (cublasHandle_t)handle_ptr;
  switch(id) {
    case 0: launch_w4a16_naive(a,q,s,c,m,n,k,128,stream); break;
    case 1: launch_w4a16_inblock_specialized(a,q,s,c,m,n,k,128,stream); break;
    case 2: launch_w4a16_inblock_m2(a,q,s,c,m,n,k,128,stream); break;
    case 3: launch_w4a16_inblock_m4(a,q,s,c,m,n,k,128,stream); break;
    case 4: case 5: case 6:
      CUBLAS_CHECK(cublasSetStream(h, stream));
      if (id==6) launch_dequant_w4a16(q,s,wh,n,k,128,stream);
      launch_cublas_fp16(h,a,id==4 ? w : wh,c,m,n,k);
      break;
    case 7: launch_dequant_w4a16(q,s,wh,n,k,128,stream); break;
    case 8: launch_w4a16_inblock(a,q,s,c,m,n,k,128,stream); break;
    default: std::fprintf(stderr,"unknown E004 bridge selector\n"); std::exit(1);
  }
  CUDA_CHECK(cudaGetLastError());
}
