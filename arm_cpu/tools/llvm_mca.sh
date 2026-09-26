#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/build-mca}"
mkdir -p "$OUT"

CLANG="${CLANG:-clang++}"
LLVM_MCA="${LLVM_MCA:-llvm-mca}"

"$CLANG"   --target=aarch64-linux-gnu   --sysroot="${AARCH64_SYSROOT:-/usr/aarch64-linux-gnu}"   -std=c++20 -O3 -march=armv9.2-a+sve2   -I"$ROOT/include"   -S "$ROOT/src/neon.cpp"   -o "$OUT/neon.s"

"$LLVM_MCA" -mtriple=aarch64 -mcpu="${LLVM_MCA_CPU:-cortex-x925}"   "$OUT/neon.s" > "$OUT/neon.mca.txt"

echo "wrote $OUT/neon.mca.txt"
