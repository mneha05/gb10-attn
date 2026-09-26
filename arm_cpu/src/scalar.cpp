#include "gb10_arm/attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace gb10::arm {

float dot_scalar(const float* a, const float* b, std::size_t n) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

void attention_scores(const float* query,
                      const float* keys,
                      float* scores,
                      std::size_t tokens,
                      std::size_t head_dim,
                      DotFn dot) {
  for (std::size_t token = 0; token < tokens; ++token) {
    scores[token] = dot(query, keys + token * head_dim, head_dim);
  }
}

void softmax_inplace(float* scores, std::size_t n) {
  if (n == 0) return;

  float max_value = -std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < n; ++i) {
    max_value = std::max(max_value, scores[i]);
  }

  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    scores[i] = std::exp(scores[i] - max_value);
    sum += scores[i];
  }

  const float inv = 1.0f / sum;
  for (std::size_t i = 0; i < n; ++i) {
    scores[i] *= inv;
  }
}

}  // namespace gb10::arm
