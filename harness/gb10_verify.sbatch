#!/usr/bin/env bash
#SBATCH -A mithuna
#SBATCH -p gb10
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH -t 00:20:00
#SBATCH -J gb10-verify
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Second run. Three questions:
#   1. what is GB10's L2, and does the corrected peak read 273 GB/s?
#   2. what do the kernels sustain once reads actually miss cache?
#   3. what exactly faults at (batch=64, ctx=2048)?

set -Eeuo pipefail

export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$HOME/miniforge3/envs/gb10/bin:$CUDA_HOME/bin:$PATH:/apps/cuda/12.8/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0;12.1"
export HETERO="$HOME/hetero-serve"

REPO="$HOME/gb10-attn"
OUT="$REPO/results/$(hostname -s)-$(date +%Y%m%d-%H%M%S)-verify"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/console.log") 2>&1
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

cd "$REPO"
say "provenance"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | tee "$OUT/gpu.txt"
git -C "$REPO" rev-parse --short HEAD | sed 's/^/gb10-attn /' | tee "$OUT/provenance.txt"
git -C "$HETERO" rev-parse --short HEAD | sed 's/^/hetero-serve /' | tee -a "$OUT/provenance.txt"

# ---- 1. corrected probe: L2 + peak bandwidth -------------------------------
say "probe (corrected peak + L2)"
python harness/probe_arch.py > "$OUT/arch.json" 2>"$OUT/arch.err" || true
grep -E 'l2_cache|peak_GBs|capability|device_name' "$OUT/arch.json" || cat "$OUT/arch.err"

# ---- 2. cold-cache sweep ---------------------------------------------------
# Only shapes known to survive; the faulting ones are handled in step 3.
say "cold-cache sweep"
for B in 1 4 16 64 128; do
  for C in 512 2048; do
    # skip the two shapes that fault, so one crash cannot eat the sweep
    if [ "$C" = "2048" ] && { [ "$B" = "64" ] || [ "$B" = "128" ]; }; then
      echo "  skip b=$B c=$C (known fault -- see sanitizer step)"; continue
    fi
    echo "  b=$B c=$C cold"
    ( cd "$HETERO" && python scripts/bench_kernel.py --batch "$B" --context "$C" \
        --iters 50 --cold-pools 0 ) > "$OUT/cold_b${B}_c${C}.txt" 2>&1 \
      || echo "    (failed)"
    # hot comparison at the same shape, to quantify how much was cache
    ( cd "$HETERO" && python scripts/bench_kernel.py --batch "$B" --context "$C" \
        --iters 50 --cold-pools 1 ) > "$OUT/hot_b${B}_c${C}.txt" 2>&1 \
      || echo "    (failed)"
  done
done

# ---- 3. the crash ----------------------------------------------------------
say "crash bisect (CUDA_LAUNCH_BLOCKING=1)"
CUDA_LAUNCH_BLOCKING=1 python harness/repro_crash.py --bisect \
  > "$OUT/crash_bisect.txt" 2>&1 || true
tail -25 "$OUT/crash_bisect.txt"

say "compute-sanitizer memcheck on the faulting shape"
if command -v compute-sanitizer >/dev/null 2>&1; then
  CUDA_LAUNCH_BLOCKING=1 timeout 420 compute-sanitizer --tool memcheck \
      --error-exitcode 0 --print-limit 20 \
      python harness/repro_crash.py --batch 64 --context 2048 \
      > "$OUT/sanitizer.txt" 2>&1 || true
  grep -iE "invalid|error|out of bounds|Leaked|ERROR SUMMARY" "$OUT/sanitizer.txt" | head -25
else
  echo "compute-sanitizer not on PATH" | tee "$OUT/sanitizer.txt"
fi

say "done"
ls -la "$OUT"
