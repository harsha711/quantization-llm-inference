#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace w4a16 {

// Reference C[r, :] = sum_k A[r, k] * B[k, :] for each r in `rows`, accumulated in
// FP64. A is M x K FP16 bits, B is K x N FP32. Returns rows.size() x N values.
std::vector<double> reference_gemm_rows(const uint16_t* A, const float* B, int M, int N, int K,
                                        const std::vector<int>& rows);

// All rows if M <= max_rows, otherwise max_rows evenly spaced rows including the
// first and last. Keeps the O(M*N*K) CPU check affordable for large M.
std::vector<int> pick_check_rows(int M, int max_rows);

struct ErrorStats {
  size_t count = 0;
  size_t num_bad = 0;     // elements with |got - ref| > rtol*|ref| + atol, or non-finite
  double max_abs = 0.0;
  double rel_l2 = 0.0;    // ||got - ref||_2 / ||ref||_2
  double ref_rms = 0.0;
  double atol = 0.0;      // atol_frac * ref_rms
  bool pass() const { return count > 0 && num_bad == 0; }
};

// got is the full M x N FP16 output; ref comes from reference_gemm_rows with the same rows.
ErrorStats compare_rows(const uint16_t* got, const std::vector<double>& ref,
                        const std::vector<int>& rows, int N, double rtol, double atol_frac);

}  // namespace w4a16
