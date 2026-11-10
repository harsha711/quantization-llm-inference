#include "w4a16/reference.h"

#include <cmath>

#include "w4a16/fp16_host.h"

namespace w4a16 {

std::vector<double> reference_gemm_rows(const uint16_t* A, const float* B, int M, int N, int K,
                                        const std::vector<int>& rows) {
  (void)M;
  std::vector<double> C(rows.size() * static_cast<size_t>(N), 0.0);
#pragma omp parallel for schedule(dynamic)
  for (long long ri = 0; ri < static_cast<long long>(rows.size()); ++ri) {
    double* c = C.data() + ri * N;
    const uint16_t* a = A + static_cast<size_t>(rows[ri]) * K;
    for (int k = 0; k < K; ++k) {
      double av = half_to_float(a[k]);
      const float* b = B + static_cast<size_t>(k) * N;
      for (int n = 0; n < N; ++n) c[n] += av * static_cast<double>(b[n]);
    }
  }
  return C;
}

std::vector<int> pick_check_rows(int M, int max_rows) {
  std::vector<int> rows;
  if (max_rows <= 0 || M <= 0) return rows;
  if (M <= max_rows) {
    for (int i = 0; i < M; ++i) rows.push_back(i);
    return rows;
  }
  if (max_rows == 1) {
    rows.push_back(0);
    return rows;
  }
  for (int i = 0; i < max_rows; ++i)
    rows.push_back(static_cast<int>(static_cast<long long>(i) * (M - 1) / (max_rows - 1)));
  return rows;
}

ErrorStats compare_rows(const uint16_t* got, const std::vector<double>& ref,
                        const std::vector<int>& rows, int N, double rtol, double atol_frac) {
  ErrorStats st;
  double sum_ref2 = 0.0;
  for (double r : ref) sum_ref2 += r * r;
  st.count = ref.size();
  st.ref_rms = st.count ? std::sqrt(sum_ref2 / st.count) : 0.0;
  st.atol = atol_frac * st.ref_rms;

  double sum_err2 = 0.0;
  for (size_t ri = 0; ri < rows.size(); ++ri)
    for (int n = 0; n < N; ++n) {
      double r = ref[ri * N + n];
      double g = half_to_float(got[static_cast<size_t>(rows[ri]) * N + n]);
      double e = std::fabs(g - r);
      if (!std::isfinite(g) || e > rtol * std::fabs(r) + st.atol) ++st.num_bad;
      if (std::isfinite(e)) {
        if (e > st.max_abs) st.max_abs = e;
        sum_err2 += e * e;
      }
    }
  st.rel_l2 = sum_ref2 > 0 ? std::sqrt(sum_err2 / sum_ref2) : 0.0;
  return st;
}

}  // namespace w4a16
