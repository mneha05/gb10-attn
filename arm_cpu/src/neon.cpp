#include "gb10_arm/attention.hpp"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace gb10::arm {

bool neon_compiled() {
#if defined(__aarch64__)
  return true;
#else
  return false;
#endif
}

float dot_neon(const float* a, const float* b, std::size_t n) {
#if defined(__aarch64__)
  std::size_t i = 0;
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);

  for (; i + 8 <= n; i += 8) {
    const float32x4_t a0 = vld1q_f32(a + i);
    const float32x4_t a1 = vld1q_f32(a + i + 4);
    const float32x4_t b0 = vld1q_f32(b + i);
    const float32x4_t b1 = vld1q_f32(b + i + 4);
    acc0 = vfmaq_f32(acc0, a0, b0);
    acc1 = vfmaq_f32(acc1, a1, b1);
  }

  float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
#else
  return dot_scalar(a, b, n);
#endif
}

}  // namespace gb10::arm
