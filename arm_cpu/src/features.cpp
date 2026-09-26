#include "gb10_arm/attention.hpp"

#include <sstream>

#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <asm/hwcap.h>
#include <linux/prctl.h>
#endif

namespace gb10::arm {

CpuFeatures detect_cpu_features() {
  CpuFeatures f;

#if defined(__linux__) && defined(__aarch64__)
  const unsigned long hwcap = getauxval(AT_HWCAP);
  const unsigned long hwcap2 = getauxval(AT_HWCAP2);

#ifdef HWCAP_ASIMD
  f.neon = (hwcap & HWCAP_ASIMD) != 0;
#else
  f.neon = true;
#endif

#ifdef HWCAP_ASIMDDP
  f.dotprod = (hwcap & HWCAP_ASIMDDP) != 0;
#endif
#ifdef HWCAP_SVE
  f.sve = (hwcap & HWCAP_SVE) != 0;
#endif
#ifdef HWCAP2_SVE2
  f.sve2 = (hwcap2 & HWCAP2_SVE2) != 0;
#endif
#ifdef HWCAP2_SME
  f.sme = (hwcap2 & HWCAP2_SME) != 0;
#endif
#ifdef HWCAP2_SME2
  f.sme2 = (hwcap2 & HWCAP2_SME2) != 0;
#endif

#ifdef PR_SVE_GET_VL
  if (f.sve) {
    const long vl = prctl(PR_SVE_GET_VL);
    if (vl >= 0) {
      f.sve_vector_bits =
          static_cast<std::size_t>(vl & PR_SVE_VL_LEN_MASK) * 8;
    }
  }
#endif

#ifdef PR_SME_GET_VL
  if (f.sme) {
    const long vl = prctl(PR_SME_GET_VL);
    if (vl >= 0) {
      f.sme_vector_bits =
          static_cast<std::size_t>(vl & PR_SME_VL_LEN_MASK) * 8;
    }
  }
#endif
#endif

  return f;
}

std::string describe_cpu_features(const CpuFeatures& f) {
  std::ostringstream out;
  out << "neon=" << f.neon
      << " dotprod=" << f.dotprod
      << " sve=" << f.sve
      << " sve2=" << f.sve2
      << " sme=" << f.sme
      << " sme2=" << f.sme2;

  if (f.sve_vector_bits) out << " sve_vl=" << f.sve_vector_bits << "b";
  if (f.sme_vector_bits) out << " sme_vl=" << f.sme_vector_bits << "b";
  return out.str();
}

DotFn resolve_dot_backend(const std::string& requested,
                          const CpuFeatures& features,
                          std::string* selected_backend) {
  auto select = [&](DotFn fn, const char* name) {
    if (selected_backend) *selected_backend = name;
    return fn;
  };

  if (requested == "scalar") return select(dot_scalar, "scalar");

  if (requested == "neon") {
    if (features.neon && neon_compiled()) return select(dot_neon, "neon");
    return nullptr;
  }

  if (requested == "sve") {
    if (features.sve && sve_compiled()) return select(dot_sve, "sve");
    return nullptr;
  }

  if (requested == "auto") {
    if (features.sve && sve_compiled()) return select(dot_sve, "sve");
    if (features.neon && neon_compiled()) return select(dot_neon, "neon");
    return select(dot_scalar, "scalar");
  }

  return nullptr;
}

}  // namespace gb10::arm
