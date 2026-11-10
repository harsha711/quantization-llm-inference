// E005 changes host launch policy only. All configurations share device code.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#define marlin e005_marlin
#define marlin_gemm e005_unused_torch_gemm
#include "marlin_template.h"
#include "marlin.cu"
#undef marlin_gemm
#undef marlin

extern "C" int e005_call(int mode, int dry, const void* a, const void* b,
    const void* s, void* c, void* tmp, void* locks, int m, int n, int k,
    void* stream, int64_t* info) {
  try {
    STD_TORCH_CHECK(mode >= 0 && mode <= 2 && m > 0 && k >= 256 &&
                    k % 128 == 0 && n > 0 && n % 64 == 0);
    int dev, sms;
    STD_TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
    STD_TORCH_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess);
    e005_marlin::e005_mode = mode;
    e005_marlin::e005_small_layout = false;
    e005_marlin::e005_dry_run = dry;
    e005_marlin::marlin_mm(a,b,c,tmp,nullptr,nullptr,const_cast<void*>(s),
        nullptr,nullptr,m,n,k,k,locks,vllm::kFloat16,vllm::kU4B8,
        vllm::kFloat16,vllm::kFloat16,false,false,k/128,128,dev,
        (cudaStream_t)stream,-1,-1,sms,false,true,false);
    std::copy(e005_marlin::e005_info,e005_marlin::e005_info+20,info);
    e005_marlin::e005_mode = 0;
    e005_marlin::e005_small_layout = false;
    return (int)cudaGetLastError();
  } catch (const std::exception& e) {
    e005_marlin::e005_mode = 0;
    e005_marlin::e005_small_layout = false;
    std::fprintf(stderr,"E005 error: %s\n",e.what());
    return -1;
  }
}
