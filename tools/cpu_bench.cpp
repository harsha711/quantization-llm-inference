// E041: time the fused W4A16 CPU GEMM at the thread count set by OMP_NUM_THREADS.
// cpu_bench M N K GROUP SEED REPS
// Prints one CSV line: threads,M,N,K,group,seed,reps,median_ms,min_ms,max_ms,fp64_pass,num_bad,rel_l2,max_abs,fnv1a
// The FP64 check (all rows if M <= 64, else 64 sampled rows, CONTRACT.md §2) and the output hash run outside timing.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "w4a16/cpu_gemm.h"
#include "w4a16/reference.h"
#include "w4a16/synthetic.h"

using namespace w4a16;

int main(int argc, char** argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: cpu_bench M N K GROUP SEED REPS\n");
    return 2;
  }
  const int M = std::atoi(argv[1]), N = std::atoi(argv[2]), K = std::atoi(argv[3]), G = std::atoi(argv[4]);
  const unsigned long long seed = std::strtoull(argv[5], nullptr, 10);
  const int reps = std::atoi(argv[6]);
  if (M <= 0 || reps <= 0) return 2;
  validate_shape(K, N, G);
  Problem p = make_problem(M, N, K, G, seed);
  std::vector<uint16_t> C(static_cast<size_t>(M) * N);

  cpu_w4a16_gemm(p.A.data(), p.qw, M, C.data());  // warmup
  std::vector<double> ms;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    cpu_w4a16_gemm(p.A.data(), p.qw, M, C.data());
    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::vector<double> s = ms;
  std::sort(s.begin(), s.end());
  const double med = s.size() % 2 ? s[s.size() / 2] : 0.5 * (s[s.size() / 2 - 1] + s[s.size() / 2]);

  std::vector<int> rows = pick_check_rows(M, 64);
  std::vector<float> What = dequantize_all(p.qw);
  ErrorStats st = compare_rows(C.data(), reference_gemm_rows(p.A.data(), What.data(), M, N, K, rows), rows, N, 1e-3, 1e-3);
  unsigned long long h = 1469598103934665603ULL;  // FNV-1a over all output bits: equal across thread counts
  for (uint16_t v : C) {
    h = (h ^ (v & 0xFF)) * 1099511628211ULL;
    h = (h ^ (v >> 8)) * 1099511628211ULL;
  }
  std::printf("%d,%d,%d,%d,%d,%llu,%d,%.6f,%.6f,%.6f,%d,%zu,%.6e,%.6e,%016llx\n", omp_get_max_threads(), M, N, K, G,
              seed, reps, med, s.front(), s.back(), st.pass() ? 1 : 0, st.num_bad, st.rel_l2, st.max_abs, h);
  for (double v : ms) std::fprintf(stderr, "sample_ms %.6f\n", v);
  return st.pass() ? 0 : 1;
}
