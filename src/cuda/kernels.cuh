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

// Bandwidth-roofline series (docs/OURS.md). Same contract as launch_w4a16_naive.
void launch_w4a16_v0(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                     int M, int N, int K, int group_size, cudaStream_t stream);
void launch_w4a16_v1(const __half* A, const uint32_t* qweight, const __half* scales, __half* C,
                     int M, int N, int K, int group_size, cudaStream_t stream);
int w4a16_v1_splits(int M, int N, int K);  // K-splits v1 will launch for this shape
// v1 ablation: contiguous = one contiguous word run per warp; specialize_tail = compile
// out the partial-word check when K % 8 == 0. launch_w4a16_v1 = (true, true).
void launch_w4a16_v1_variant(const __half* A, const uint32_t* qweight, const __half* scales,
                             __half* C, int M, int N, int K, int group_size, cudaStream_t stream,
                             bool contiguous, bool specialize_tail);

void launch_dequant_w4a16(const uint32_t* qweight, const __half* scales, __half* W, int N, int K,
                          int group_size, cudaStream_t stream);

void launch_l2_flush(void* buf, size_t nbytes, cudaStream_t stream);
// Read-only variant: evicts L2 without leaving dirty lines (flush_read.cu).
void launch_l2_flush_read(const void* buf, size_t nbytes, uint32_t* sink, cudaStream_t stream);

// C = A * W with FP16 inputs/outputs and FP32 accumulation.
void launch_cublas_fp16(cublasHandle_t handle, const __half* A, const __half* W, __half* C, int M,
                        int N, int K);
