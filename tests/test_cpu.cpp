#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "w4a16/fp16_host.h"
#include "w4a16/quant.h"
#include "w4a16/reference.h"
#include "w4a16/synthetic.h"

using namespace w4a16;

static int g_failures = 0;
#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                    \
    }                                                                  \
  } while (0)

static void test_half_roundtrip() {
  for (uint32_t h = 0; h < 65536; ++h) {
    bool nan = ((h >> 10) & 0x1f) == 31 && (h & 0x3ff);
    if (nan) continue;
    CHECK(float_to_half(half_to_float(static_cast<uint16_t>(h))) == h);
  }
}

static void test_half_rounding() {
  CHECK(float_to_half(1.0f) == 0x3c00);
  CHECK(float_to_half(1.0f + 0x1.0p-11f) == 0x3c00);        // tie, round to even
  CHECK(float_to_half(1.0f + 3 * 0x1.0p-11f) == 0x3c02);    // tie, round to even (up)
  CHECK(float_to_half(65504.f) == 0x7bff);
  CHECK(float_to_half(65520.f) == 0x7c00);                  // rounds to +inf
  CHECK(float_to_half(-2.f) == 0xc000);
  CHECK(float_to_half(0x1.0p-24f) == 0x0001);               // smallest subnormal
  CHECK(float_to_half(0x1.0p-25f) == 0x0000);               // tie to even (zero)
  CHECK(float_to_half(0x1.1p-25f) == 0x0001);
  CHECK(half_to_float(0x0001) == 0x1.0p-24f);
  CHECK(std::isinf(half_to_float(0x7c00)));
}

static void test_pack_unpack() {
  const int K = 64, N = 5;
  std::vector<uint8_t> q(K * N);
  SplitMix64 rng(42);
  for (auto& v : q) v = static_cast<uint8_t>(rng.next() & 0xF);
  QuantizedWeights qw;
  qw.K = K;
  qw.N = N;
  qw.group_size = 32;
  qw.qweight = pack_int4(q, K, N);
  CHECK(qw.qweight.size() == static_cast<size_t>(K / 8) * N);
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n) CHECK(unpack_q(qw, k, n) == q[k * N + n]);
  // Bit layout: row k = 8*k8 + j lives in bits [4j, 4j+4) of word (k8, n).
  std::vector<uint8_t> one(K * N, 0);
  one[9 * N + 2] = 0xA;  // k8 = 1, j = 1
  auto p = pack_int4(one, K, N);
  CHECK(p[1 * N + 2] == 0xA0u);
}

static void test_quantization() {
  const int K = 256, N = 33, G = 64;
  Problem p = make_problem(1, N, K, G, 7);
  const auto& qw = p.qw;
  for (int g = 0; g < K / G; ++g)
    for (int n = 0; n < N; ++n) {
      float amax = 0.f;
      for (int k = g * G; k < (g + 1) * G; ++k)
        amax = std::fmax(amax, std::fabs(half_to_float(p.W[k * N + n])));
      float s = half_to_float(qw.scales[g * N + n]);
      CHECK(qw.scales[g * N + n] == float_to_half(amax / 7.f));
      for (int k = g * G; k < (g + 1) * G; ++k) {
        float w = half_to_float(p.W[k * N + n]);
        int q = unpack_q(qw, k, n);
        CHECK(q >= 1 && q <= 15);  // code 0 unused; z clamped to [-7, 7]
        float w_f32 = static_cast<float>(q - 8) * s;
        CHECK(dequant_fp16(qw, k, n) == float_to_half(w_f32));
        CHECK(std::fabs(w - w_f32) <= 0.5f * s * (1.f + 1e-6f));
      }
    }
  std::vector<uint16_t> zeros(16 * 2, 0);
  auto qz = quantize_int4_groupwise(zeros.data(), 16, 2, 8);
  for (int k = 0; k < 16; ++k)
    for (int n = 0; n < 2; ++n) CHECK(unpack_q(qz, k, n) == 8 && dequant(qz, k, n) == 0.f);

  // Forced overflow of W/s rounds to 8 and must clamp to 7 (stored code 15), not -8/0.
  std::vector<uint16_t> hi(8, float_to_half(7.f));
  hi[0] = float_to_half(8.f);  // amax=8, s=fp16(8/7); 8/s > 7
  auto qh = quantize_int4_groupwise(hi.data(), 8, 1, 8);
  for (int k = 0; k < 8; ++k) CHECK(unpack_q(qh, k, 0) >= 1 && unpack_q(qh, k, 0) <= 15);
  CHECK(unpack_q(qh, 0, 0) == 15);
}

static void test_reference_gemm() {
  // A = [[1, 2, 3], [4, 5, 6]], B = [[1, 0], [0, 1], [1, -1]]
  std::vector<uint16_t> A;
  for (float v : {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}) A.push_back(float_to_half(v));
  std::vector<float> B = {1, 0, 0, 1, 1, -1};
  auto C = reference_gemm_rows(A.data(), B.data(), 2, 2, 3, {0, 1});
  CHECK(C.size() == 4);
  CHECK(C[0] == 4 && C[1] == -1 && C[2] == 10 && C[3] == -1);
  auto C1 = reference_gemm_rows(A.data(), B.data(), 2, 2, 3, {1});
  CHECK(C1[0] == 10 && C1[1] == -1);
}

static void test_compare_and_rows() {
  auto rows = pick_check_rows(1000, 5);
  CHECK(rows.size() == 5 && rows.front() == 0 && rows.back() == 999);
  CHECK(pick_check_rows(3, 5).size() == 3);

  std::vector<double> ref = {1.0, -2.0, 0.5, 4.0};
  std::vector<uint16_t> got;
  for (double r : ref) got.push_back(float_to_half(static_cast<float>(r)));
  auto st = compare_rows(got.data(), ref, {0, 1}, 2, 1e-3, 1e-3);
  CHECK(st.pass() && st.max_abs == 0.0);
  got[3] = float_to_half(4.1f);
  CHECK(!compare_rows(got.data(), ref, {0, 1}, 2, 1e-3, 1e-3).pass());
  got[3] = 0x7e00;  // NaN
  CHECK(!compare_rows(got.data(), ref, {0, 1}, 2, 1e-3, 1e-3).pass());
}

static void test_memory_sizes() {
  CHECK(fp16_bytes(4096, 4096) == 32ull * 1024 * 1024);
  CHECK(qweight_bytes(4096, 4096) == 8ull * 1024 * 1024);
  CHECK(scales_bytes(4096, 4096, 128) == 256ull * 1024);
}

static void test_determinism() {
  Problem a = make_problem(4, 16, 64, 32, 123);
  Problem b = make_problem(4, 16, 64, 32, 123);
  Problem c = make_problem(4, 16, 64, 32, 124);
  CHECK(a.A == b.A && a.W == b.W && a.qw.qweight == b.qw.qweight && a.qw.scales == b.qw.scales);
  CHECK(a.A != c.A && a.W != c.W);
}

int main() {
  test_half_roundtrip();
  test_half_rounding();
  test_pack_unpack();
  test_quantization();
  test_reference_gemm();
  test_compare_and_rows();
  test_memory_sizes();
  test_determinism();
  if (g_failures) {
    std::printf("test_cpu: %d check(s) FAILED\n", g_failures);
    return EXIT_FAILURE;
  }
  std::printf("test_cpu: all checks passed\n");
  return EXIT_SUCCESS;
}
