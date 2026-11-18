#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                           \
  do {                                                                             \
    cudaError_t err_ = (expr);                                                     \
    if (err_ != cudaSuccess) {                                                     \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #expr,           \
                   cudaGetErrorString(err_));                                      \
      std::exit(EXIT_FAILURE);                                                     \
    }                                                                              \
  } while (0)

#define CUBLAS_CHECK(expr)                                                         \
  do {                                                                             \
    cublasStatus_t st_ = (expr);                                                   \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                            \
      std::fprintf(stderr, "%s:%d: %s: cuBLAS status %d\n", __FILE__, __LINE__,    \
                   #expr, static_cast<int>(st_));                                  \
      std::exit(EXIT_FAILURE);                                                     \
    }                                                                              \
  } while (0)

// All matrices row-major. qweight/scales use the logical layout in w4a16/quant.h.
void launch_w4a16_naive(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                        int M, int N, int K, int group_size, cudaStream_t stream);

void launch_w4a16_inblock(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                          int M, int N, int K, int group_size, cudaStream_t stream);

void launch_w4a16_inblock_specialized(const __half* A, const uint32_t* qweight,
                                      const __half* scales, __half* C, int M, int N, int K,
                                      int group_size, cudaStream_t stream);

void launch_w4a16_inblock_m2(const __half* A, const uint32_t* qweight, const __half* scales,
                             __half* C, int M, int N, int K, int group_size, cudaStream_t stream);
void launch_w4a16_inblock_m4(const __half* A, const uint32_t* qweight, const __half* scales,
                             __half* C, int M, int N, int K, int group_size, cudaStream_t stream);

// E043: same mapping/arithmetic as inblock_specialized / inblock_m4, wider operand loads.
void launch_w4a16_inblock_vec(const __half* A, const uint32_t* qweight, const __half* scales,
                              __half* C, int M, int N, int K, int group_size, cudaStream_t stream);
void launch_w4a16_inblock_m4_vec(const __half* A, const uint32_t* qweight, const __half* scales,
                                 __half* C, int M, int N, int K, int group_size, cudaStream_t stream);

// E044: our Tensor Core W4A16 GEMM (fast and contract-formula unpack); exhaustive unpack self-test.
void launch_w4a16_tc(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                     int M, int N, int K, int group_size, cudaStream_t stream);
void launch_w4a16_tc_cvt(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                         int M, int N, int K, int group_size, cudaStream_t stream);
unsigned long long w4a16_tc_unpack_selftest();

// E047: tc with K split over S blocks and a deterministic last-block reduction (fixed z order).
void launch_w4a16_tc_splitk(const __half* A, const uint32_t* qweight, const __half* scales,
                            __half* C, int M, int N, int K, int group_size, cudaStream_t stream);
int w4a16_tc_splitk_factor(const __half* A, const __half* C, int M, int N, int K, int group_size);
unsigned long long w4a16_tc_splitk_unpack_selftest();

// E048: tc with a 32-row M tile (two m16 tiles per warp share each unpacked weight) for M >= 32.
void launch_w4a16_tc_m32(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                         int M, int N, int K, int group_size, cudaStream_t stream);

void launch_dequant_w4a16(const uint32_t* qweight, const __half* scales, __half* W, int N, int K,
                          int group_size, cudaStream_t stream);

void launch_l2_flush(void* buf, size_t nbytes, cudaStream_t stream);

// C = A * W with FP16 inputs/outputs and FP32 accumulation.
void launch_cublas_fp16(cublasHandle_t handle, const __half* A, const __half* W, __half* C, int M,
                        int N, int K);
