#pragma once
// INT4 weight format (logical layout). Host dequant, GPU dequant, and the fused
// kernel must reconstruct the same FP16 values.
//
//   W is K x N row-major FP16, split into groups of `group_size` consecutive
//   rows along K. Each (group, column) has one stored FP16 scale.
//
//   Encoding (symmetric, zero-point 8, 15 used levels):
//     s     = round_to_fp16(max_{k in group} |W[k, n]| / 7)
//     z     = round_to_nearest_even(W[k, n] / s)     if s > 0, else 0
//     z     = clamp(z, -7, +7)                       // never -8
//     q     = z + 8                                  // stored code in {1, ..., 15}
//     W_hat = round_to_fp16((q - 8) * s)             // FP16 reconstructed weight
//
//   Code 0 is unused. Scale uses 7 rather than 8 so the representable range is
//   symmetric: the extreme stored values 1 and 15 decode to -7s and +7s.
//   Rounding of W/s is C++ nearbyint (round-ties-to-even at the default mode).
//   round_to_fp16 is IEEE RN, matching CUDA __float2half_rn.
//
//   The product (q - 8) * s is exact in FP32 (integer |z| <= 7 times an FP16),
//   but is not always an FP16, so the reconstructed weight is the rounded FP16,
//   then converted back to FP32 for accumulation. Host and device must agree
//   bit-for-bit on that FP16.
//
//   Packing: qweight is ceil(K/8) x N uint32, row-major. Word (k8, n) holds rows
//   k = 8*k8 + j for j = 0..7, with row j in bits [4j, 4j+4).
//   scales is ceil(K/group_size) x N FP16, row-major. The last group may be partial.
//   group_size is a multiple of 8; unused final-word nibbles are zero and never read.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace w4a16 {

struct QuantizedWeights {
  int K = 0;
  int N = 0;
  int group_size = 0;
  std::vector<uint32_t> qweight;  // ceil(K/8) x N
  std::vector<uint16_t> scales;   // ceil(K/group_size) x N, FP16 bits
};

inline size_t fp16_bytes(int rows, int cols) { return static_cast<size_t>(rows) * cols * 2; }
inline size_t qweight_bytes(int K, int N) { return static_cast<size_t>((K + 7) / 8) * N * 4; }
inline size_t scales_bytes(int K, int N, int group) {
  return static_cast<size_t>((K + group - 1) / group) * N * 2;
}

void validate_shape(int K, int N, int group_size);

QuantizedWeights quantize_int4_groupwise(const uint16_t* W_fp16, int K, int N, int group_size);

std::vector<uint32_t> pack_int4(const std::vector<uint8_t>& q, int K, int N);

inline int unpack_q(const QuantizedWeights& qw, int k, int n) {
  uint32_t word = qw.qweight[static_cast<size_t>(k / 8) * qw.N + n];
  return static_cast<int>((word >> (4 * (k % 8))) & 0xFu);
}

uint16_t dequant_fp16(const QuantizedWeights& qw, int k, int n);
float dequant(const QuantizedWeights& qw, int k, int n);

std::vector<uint16_t> dequantize_all_fp16(const QuantizedWeights& qw);
std::vector<float> dequantize_all(const QuantizedWeights& qw);

}  // namespace w4a16
