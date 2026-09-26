cmake_minimum_required(VERSION 3.20)
project(gb10_arm_attn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(GB10_ARM_ENABLE_SVE "Compile the SVE backend on AArch64" ON)

add_library(gb10_arm STATIC
  src/scalar.cpp
  src/neon.cpp
  src/sve.cpp
  src/features.cpp
)
target_include_directories(gb10_arm PUBLIC include)
target_compile_options(gb10_arm PRIVATE -O3 -Wall -Wextra -Wpedantic)

# Keep the reference backend genuinely scalar so the Neon/SVE speedup is
# measured against scalar code rather than compiler auto-vectorization.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(src/scalar.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-tree-vectorize"
  )
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" GB10_PROCESSOR)
if(GB10_PROCESSOR MATCHES "aarch64|arm64")
  set_source_files_properties(src/neon.cpp PROPERTIES
    COMPILE_OPTIONS "-march=armv8.2-a+simd"
  )

  if(GB10_ARM_ENABLE_SVE)
    set_source_files_properties(src/sve.cpp PROPERTIES
      COMPILE_OPTIONS "-march=armv8.2-a+sve2"
    )
  endif()
endif()

add_executable(arm_attn_bench src/bench.cpp)
target_link_libraries(arm_attn_bench PRIVATE gb10_arm)
target_compile_options(arm_attn_bench PRIVATE -O3 -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME scalar_smoke
  COMMAND arm_attn_bench --backend scalar --tokens 64 --dim 128 --iters 2
)
