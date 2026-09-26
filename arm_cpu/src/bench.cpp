#include "gb10_arm/attention.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

struct Args {
  std::size_t tokens = 2048;
  std::size_t dim = 128;
  int iterations = 400;
  std::string backend = "auto";
  bool csv = false;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string x = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value after " << x << "\n";
        std::exit(2);
      }
      return argv[++i];
    };

    if (x == "--tokens") a.tokens = std::stoull(value());
    else if (x == "--dim") a.dim = std::stoull(value());
    else if (x == "--iters") a.iterations = std::stoi(value());
    else if (x == "--backend") a.backend = value();
    else if (x == "--csv") a.csv = true;
    else if (x == "--help") {
      std::cout << "arm_attn_bench [--tokens N] [--dim N] [--iters N] "
                   "[--backend auto|scalar|neon|sve] [--csv]\n";
      std::exit(0);
    }
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace gb10::arm;
  const Args args = parse(argc, argv);

  std::vector<float> query(args.dim);
  std::vector<float> keys(args.tokens * args.dim);
  std::vector<float> reference(args.tokens);
  std::vector<float> scores(args.tokens);

  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
  for (auto& x : query) x = dist(rng);
  for (auto& x : keys) x = dist(rng);

  const CpuFeatures features = detect_cpu_features();
  std::string selected;
  DotFn dot = resolve_dot_backend(args.backend, features, &selected);
  if (!dot) {
    std::cerr << "requested backend unavailable: " << args.backend << "\n"
              << describe_cpu_features(features) << "\n";
    return 3;
  }

  attention_scores(query.data(), keys.data(), reference.data(),
                   args.tokens, args.dim, dot_scalar);
  attention_scores(query.data(), keys.data(), scores.data(),
                   args.tokens, args.dim, dot);

  float max_abs = 0.0f;
  for (std::size_t i = 0; i < args.tokens; ++i) {
    max_abs = std::max(max_abs, std::abs(reference[i] - scores[i]));
  }
  if (max_abs > 1e-4f) {
    std::cerr << "correctness failure max_abs=" << max_abs << "\n";
    return 4;
  }

  for (int i = 0; i < 10; ++i) {
    attention_scores(query.data(), keys.data(), scores.data(),
                     args.tokens, args.dim, dot);
    softmax_inplace(scores.data(), args.tokens);
  }

  const auto begin = std::chrono::steady_clock::now();
  double checksum = 0.0;
  for (int it = 0; it < args.iterations; ++it) {
    attention_scores(query.data(), keys.data(), scores.data(),
                     args.tokens, args.dim, dot);
    softmax_inplace(scores.data(), args.tokens);
    checksum += scores[static_cast<std::size_t>(it) % args.tokens];
  }
  const auto end = std::chrono::steady_clock::now();

  const double seconds =
      std::chrono::duration<double>(end - begin).count();
  const double calls =
      static_cast<double>(args.iterations) * args.tokens;
  const double flops = calls * (2.0 * args.dim);
  const double key_bytes =
      calls * static_cast<double>(args.dim) * sizeof(float);
  const double gflops = flops / seconds / 1e9;
  const double effective_gbs = key_bytes / seconds / 1e9;
  const double ns_per_dot = seconds * 1e9 / calls;

  if (args.csv) {
    std::cout << "backend,tokens,dim,iters,seconds,ns_per_dot,gflops,"
                 "effective_key_gbs,max_abs,checksum\n";
    std::cout << selected << "," << args.tokens << "," << args.dim << ","
              << args.iterations << "," << seconds << "," << ns_per_dot << ","
              << gflops << "," << effective_gbs << "," << max_abs << ","
              << checksum << "\n";
  } else {
    std::cout << "GB10 Arm attention microbenchmark\n"
              << "features  " << describe_cpu_features(features) << "\n"
              << "backend   " << selected << "\n"
              << "shape     tokens=" << args.tokens
              << " head_dim=" << args.dim << "\n"
              << "time      " << std::fixed << std::setprecision(3)
              << seconds * 1e3 << " ms\n"
              << "dot       " << ns_per_dot << " ns/call\n"
              << "compute   " << gflops << " GFLOP/s\n"
              << "key read  " << effective_gbs
              << " GB/s effective input rate\n"
              << "max_abs   " << max_abs << "\n"
              << "checksum  " << checksum << "\n";
  }

  return 0;
}
