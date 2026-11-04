#include "kernels.cuh"

// cuBLAS is column-major. A row-major X (r x c) is the column-major X^T, so we
// compute C^T (N x M) = W^T (N x K) * A^T (K x M) and no transposes are needed.
void launch_cublas_fp16(cublasHandle_t handle, const __half* A, const __half* W, __half* C, int M,
                        int N, int K) {
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, N, A,
                            CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}
