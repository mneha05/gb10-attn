#!/bin/bash
#SBATCH --job-name=gb10-arm-attn
#SBATCH --partition=gb10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=gb10-arm-attn-%j.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$REPO_ROOT/arm_cpu/build-gb10"
RESULTS="$REPO_ROOT/arm_cpu/results"
mkdir -p "$RESULTS"
cd "$REPO_ROOT"

echo "=== provenance ==="
date -Is
hostname
uname -a
lscpu
grep -m1 '^Features' /proc/cpuinfo || true

cmake -S arm_cpu -B "$BUILD" -G Ninja   -DCMAKE_BUILD_TYPE=Release   -DGB10_ARM_ENABLE_SVE=ON
cmake --build "$BUILD" -j4

CSV="$RESULTS/arm-attn-${SLURM_JOB_ID}.csv"
echo "backend,tokens,dim,iters,seconds,ns_per_dot,gflops,effective_key_gbs,max_abs,checksum" > "$CSV"

for backend in scalar neon sve auto; do
  echo
  echo "=== $backend ==="
  "$BUILD/arm_attn_bench"     --backend "$backend"     --tokens 2048     --dim 128     --iters 600     --csv | tee /tmp/arm-attn.csv

  tail -n 1 /tmp/arm-attn.csv >> "$CSV" || true
done

if command -v perf >/dev/null 2>&1; then
  echo
  echo "=== perf / auto backend ==="
  perf stat -e cycles,instructions,branches,branch-misses,cache-misses     "$BUILD/arm_attn_bench"       --backend auto --tokens 2048 --dim 128 --iters 120       2>&1 | tee "$RESULTS/perf-${SLURM_JOB_ID}.txt"
fi

echo
echo "CSV: $CSV"
