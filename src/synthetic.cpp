#include "w4a16/synthetic.h"

#include <cmath>

#include "w4a16/fp16_host.h"

namespace w4a16 {

Problem make_problem(int M, int N, int K, int group_size, uint64_t seed) {
  validate_shape(K, N, group_size);
  Problem p;
  p.M = M;
  p.N = N;
  p.K = K;
  p.group_size = group_size;
  p.seed = seed;

  SplitMix64 rng_a(seed * 2 + 0);
  SplitMix64 rng_w(seed * 2 + 1);
  p.A.resize(static_cast<size_t>(M) * K);
  for (auto& x : p.A) x = float_to_half(static_cast<float>(rng_a.normal()));
  const double w_std = 1.0 / std::sqrt(static_cast<double>(K));
  p.W.resize(static_cast<size_t>(K) * N);
  for (auto& x : p.W) x = float_to_half(static_cast<float>(rng_w.normal() * w_std));

  p.qw = quantize_int4_groupwise(p.W.data(), K, N, group_size);
  return p;
}

}  // namespace w4a16
