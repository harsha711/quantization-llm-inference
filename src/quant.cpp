#include "w4a16/quant.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "w4a16/fp16_host.h"

namespace w4a16 {

void validate_shape(int K, int N, int group_size) {
  if (K <= 0 || N <= 0 || group_size <= 0)
    throw std::invalid_argument("K, N and group_size must be positive");
  if (group_size % 8 != 0)
    throw std::invalid_argument("group_size must be a multiple of 8 (one packed word never spans groups)");
  if (K % group_size != 0)
    throw std::invalid_argument("K=" + std::to_string(K) + " is not a multiple of group_size=" +
                                std::to_string(group_size));
}

std::vector<uint32_t> pack_int4(const std::vector<uint8_t>& q, int K, int N) {
  std::vector<uint32_t> packed(static_cast<size_t>(K / 8) * N, 0u);
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n) {
      uint32_t v = q[static_cast<size_t>(k) * N + n] & 0xFu;
      packed[static_cast<size_t>(k / 8) * N + n] |= v << (4 * (k % 8));
    }
  return packed;
}

QuantizedWeights quantize_int4_groupwise(const uint16_t* W_fp16, int K, int N, int group_size) {
  validate_shape(K, N, group_size);
  const int groups = K / group_size;
  QuantizedWeights qw;
  qw.K = K;
  qw.N = N;
  qw.group_size = group_size;
  qw.scales.resize(static_cast<size_t>(groups) * N);
  std::vector<uint8_t> q(static_cast<size_t>(K) * N);

  for (int g = 0; g < groups; ++g)
    for (int n = 0; n < N; ++n) {
      float amax = 0.f;
      for (int k = g * group_size; k < (g + 1) * group_size; ++k)
        amax = std::max(amax, std::fabs(half_to_float(W_fp16[static_cast<size_t>(k) * N + n])));
      uint16_t s_bits = float_to_half(amax / 7.f);
      float s = half_to_float(s_bits);
      qw.scales[static_cast<size_t>(g) * N + n] = s_bits;
      for (int k = g * group_size; k < (g + 1) * group_size; ++k) {
        size_t idx = static_cast<size_t>(k) * N + n;
        int v = 0;
        if (s > 0.f) {
          float r = std::nearbyint(half_to_float(W_fp16[idx]) / s);
          v = static_cast<int>(std::min(7.f, std::max(-7.f, r)));
        }
        q[idx] = static_cast<uint8_t>(v + 8);
      }
    }
  qw.qweight = pack_int4(q, K, N);
  return qw;
}

uint16_t dequant_fp16(const QuantizedWeights& qw, int k, int n) {
  float s = half_to_float(qw.scales[static_cast<size_t>(k / qw.group_size) * qw.N + n]);
  return float_to_half(static_cast<float>(unpack_q(qw, k, n) - 8) * s);
}

float dequant(const QuantizedWeights& qw, int k, int n) { return half_to_float(dequant_fp16(qw, k, n)); }

std::vector<uint16_t> dequantize_all_fp16(const QuantizedWeights& qw) {
  std::vector<uint16_t> W(static_cast<size_t>(qw.K) * qw.N);
  for (int k = 0; k < qw.K; ++k)
    for (int n = 0; n < qw.N; ++n) W[static_cast<size_t>(k) * qw.N + n] = dequant_fp16(qw, k, n);
  return W;
}

std::vector<float> dequantize_all(const QuantizedWeights& qw) {
  std::vector<uint16_t> h = dequantize_all_fp16(qw);
  std::vector<float> W(h.size());
  for (size_t i = 0; i < h.size(); ++i) W[i] = half_to_float(h[i]);
  return W;
}

}  // namespace w4a16
