#!/bin/bash
#SBATCH --job-name=gb10-arm-attn
#SBATCH --partition=gb10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=arm_cpu/results/%x-%j.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results

echo "=== provenance ==="
date -Is
hostname
uname -a
lscpu
grep -m1 '^Features' /proc/cpuinfo || true

cmake -S . -B build-gb10 -G Ninja   -DCMAKE_BUILD_TYPE=Release   -DGB10_ARM_ENABLE_SVE=ON
cmake --build build-gb10 -j4

for backend in scalar neon sve auto; do
  echo
  echo "=== $backend ==="
  ./build-gb10/arm_attn_bench     --backend "$backend"     --tokens 2048     --dim 128     --iters 600     --csv || true
done

if command -v perf >/dev/null 2>&1; then
  echo
  echo "=== perf / auto backend ==="
  perf stat -e cycles,instructions,branches,branch-misses,cache-misses     ./build-gb10/arm_attn_bench       --backend auto --tokens 2048 --dim 128 --iters 120
fi
