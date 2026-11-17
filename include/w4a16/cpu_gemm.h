#pragma once
// E041: fused W4A16 CPU GEMM. C = A * W_hat with W_hat = FP16((q-8)*s) reconstructed on the fly,
// FP32 accumulation in ascending k per output, FP16 output. OpenMP splits independent 64-column
// blocks, so every output's arithmetic is identical for any thread count.
#include <cstdint>

#include "w4a16/quant.h"

namespace w4a16 {

// A is M x K FP16 bits (row-major), C is M x N FP16 bits (row-major).
void cpu_w4a16_gemm(const uint16_t* A, const QuantizedWeights& qw, int M, uint16_t* C);

}  // namespace w4a16
