#pragma once
// Deterministic synthetic inputs. Uses its own PRNG rather than <random>
// distributions, whose output differs between standard library implementations.
#include <cmath>
#include <cstdint>
#include <vector>

#include "w4a16/quant.h"

namespace w4a16 {

struct SplitMix64 {
  uint64_t state;
  explicit SplitMix64(uint64_t seed) : state(seed) {}

  uint64_t next() {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }  // [0, 1)

  double normal() {  // Box-Muller
    double u1 = 1.0 - uniform();
    double u2 = uniform();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
};

struct Problem {
  int M = 0, N = 0, K = 0, group_size = 0;
  uint64_t seed = 0;
  std::vector<uint16_t> A;  // M x K FP16, entries ~ N(0, 1)
  std::vector<uint16_t> W;  // K x N FP16 original weights, entries ~ N(0, 1/K)
  QuantizedWeights qw;      // INT4 quantization of W
};

// W is scaled by 1/sqrt(K) so each output has roughly unit variance and stays far
// from the FP16 overflow threshold (65504) for any K.
Problem make_problem(int M, int N, int K, int group_size, uint64_t seed);

}  // namespace w4a16
