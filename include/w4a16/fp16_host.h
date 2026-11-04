#pragma once
// Host-side IEEE binary16 <-> binary32 conversion. FP16 values are stored on the
// host as raw uint16_t bits so host code does not depend on CUDA headers.
#include <cstdint>
#include <cstring>

namespace w4a16 {

inline float half_to_float(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      int s = 0;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        ++s;
      }
      mant &= 0x3ffu;
      f = sign | (static_cast<uint32_t>(113 - s) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7f800000u | (mant << 13);
  } else {
    f = sign | ((exp + 112) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

// Round-to-nearest-even, matching __float2half_rn.
inline uint16_t float_to_half(float value) {
  uint32_t x;
  std::memcpy(&x, &value, sizeof(x));
  uint32_t sign = (x >> 16) & 0x8000u;
  uint32_t exp = (x >> 23) & 0xffu;
  uint32_t mant = x & 0x7fffffu;

  if (exp == 0xff) return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
  int e = static_cast<int>(exp) - 127 + 15;
  if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00u);

  if (e >= 1) {
    uint32_t h = (static_cast<uint32_t>(e) << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h += 1;  // carry into exponent is correct
    return static_cast<uint16_t>(sign | h);
  }

  if (exp == 0) return static_cast<uint16_t>(sign);
  int shift = 14 - e;
  if (shift > 24) return static_cast<uint16_t>(sign);
  uint32_t full = mant | 0x800000u;
  uint32_t h = full >> shift;
  uint32_t rem = full & ((1u << shift) - 1u);
  uint32_t halfway = 1u << (shift - 1);
  if (rem > halfway || (rem == halfway && (h & 1u))) h += 1;
  return static_cast<uint16_t>(sign | h);
}

}  // namespace w4a16
