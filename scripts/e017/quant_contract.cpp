// Host-only adapter to validate the NumPy port against the retained quantizer.
#include <algorithm>
#include <cfenv>
#include "w4a16/quant.h"
extern "C" int e017_quant_contract(const uint16_t* weight,int k,int n,
    uint32_t* packed,uint16_t* scales,uint16_t* reconstructed) {
  if (!weight || !packed || !scales || !reconstructed || k<=0 || n<=0 || k%128 || n%64 || std::fegetround()!=FE_TONEAREST) return -1;
  const auto q=w4a16::quantize_int4_groupwise(weight,k,n,128);
  const auto wh=w4a16::dequantize_all_fp16(q);
  std::copy(q.qweight.begin(),q.qweight.end(),packed);
  std::copy(q.scales.begin(),q.scales.end(),scales);
  std::copy(wh.begin(),wh.end(),reconstructed);
  return 0;
}
