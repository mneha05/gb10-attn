#include "gb10_arm/attention.hpp"

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

namespace gb10::arm {

bool sve_compiled() {
#if defined(__ARM_FEATURE_SVE)
  return true;
#else
  return false;
#endif
}

float dot_sve(const float* a, const float* b, std::size_t n) {
#if defined(__ARM_FEATURE_SVE)
  std::size_t i = 0;
  svfloat32_t acc = svdup_f32(0.0f);

  while (i < n) {
    const svbool_t pg = svwhilelt_b32(i, n);
    const svfloat32_t va = svld1(pg, a + i);
    const svfloat32_t vb = svld1(pg, b + i);
    acc = svmla_f32_m(pg, acc, va, vb);
    i += svcntw();
  }

  return svaddv_f32(svptrue_b32(), acc);
#else
  return dot_neon(a, b, n);
#endif
}

}  // namespace gb10::arm
