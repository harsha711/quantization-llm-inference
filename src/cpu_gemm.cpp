#include "w4a16/cpu_gemm.h"

#include <algorithm>
#include <vector>

#include "w4a16/fp16_host.h"

namespace w4a16 {

void cpu_w4a16_gemm(const uint16_t* A, const QuantizedWeights& qw, int M, uint16_t* C) {
  const int K = qw.K, N = qw.N, G = qw.group_size, NB = 64;
  const int nblocks = (N + NB - 1) / NB;
#pragma omp parallel
  {
    // Per-thread scratch: one group of reconstructed weights for one column block, and the
    // block's FP32 accumulators. W_hat never exists in memory beyond this tile.
    std::vector<float> tile(static_cast<size_t>(G) * NB), acc(static_cast<size_t>(M) * NB);
#pragma omp for schedule(static)
    for (int b = 0; b < nblocks; ++b) {
      const int n0 = b * NB, nb = std::min(NB, N - n0);
      std::fill(acc.begin(), acc.end(), 0.0f);
      for (int k0 = 0; k0 < K; k0 += G) {
        const int kg = std::min(G, K - k0);
        for (int n = 0; n < nb; ++n) {
          // 16 possible FP16 reconstructions per (group, column): identical to dequant_fp16.
          const float s = half_to_float(qw.scales[static_cast<size_t>(k0 / G) * N + n0 + n]);
          float lut[16];
          for (int q = 0; q < 16; ++q) lut[q] = half_to_float(float_to_half(static_cast<float>(q - 8) * s));
          for (int kk = 0; kk < kg; ++kk) {
            const int k = k0 + kk;
            const uint32_t word = qw.qweight[static_cast<size_t>(k / 8) * N + n0 + n];
            tile[static_cast<size_t>(kk) * NB + n] = lut[(word >> (4 * (k % 8))) & 0xFu];
          }
        }
        for (int m = 0; m < M; ++m) {
          float* c = acc.data() + static_cast<size_t>(m) * NB;
          const uint16_t* a = A + static_cast<size_t>(m) * K + k0;
          for (int kk = 0; kk < kg; ++kk) {
            const float av = half_to_float(a[kk]);
            const float* w = tile.data() + static_cast<size_t>(kk) * NB;
            for (int n = 0; n < nb; ++n) c[n] += av * w[n];
          }
        }
      }
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < nb; ++n)
          C[static_cast<size_t>(m) * N + n0 + n] = float_to_half(acc[static_cast<size_t>(m) * NB + n]);
    }
  }
}

}  // namespace w4a16
