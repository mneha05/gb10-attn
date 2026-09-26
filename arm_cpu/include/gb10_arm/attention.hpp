#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace gb10::arm {

struct CpuFeatures {
  bool neon{false};
  bool dotprod{false};
  bool sve{false};
  bool sve2{false};
  bool sme{false};
  bool sme2{false};
  std::size_t sve_vector_bits{0};
  std::size_t sme_vector_bits{0};
};

using DotFn = float (*)(const float*, const float*, std::size_t);

float dot_scalar(const float* a, const float* b, std::size_t n);
float dot_neon(const float* a, const float* b, std::size_t n);
float dot_sve(const float* a, const float* b, std::size_t n);

bool neon_compiled();
bool sve_compiled();

CpuFeatures detect_cpu_features();
std::string describe_cpu_features(const CpuFeatures& features);

DotFn resolve_dot_backend(const std::string& requested,
                          const CpuFeatures& features,
                          std::string* selected_backend);

void attention_scores(const float* query,
                      const float* keys,
                      float* scores,
                      std::size_t tokens,
                      std::size_t head_dim,
                      DotFn dot);

void softmax_inplace(float* scores, std::size_t n);

}  // namespace gb10::arm
