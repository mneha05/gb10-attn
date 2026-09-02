#!/usr/bin/env bash
#SBATCH -A mithuna
#SBATCH -p gb10
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH -t 00:30:00
#SBATCH -J gb10-ncu
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Turn "v3 reaches 78% of bandwidth" into "78%, and here is what the other
# 22% is". BLOCKED until RCAC enables performance counters -- see
# docs/rcac-ticket-gpu-counters.md. This job checks that first and exits
# early rather than burning the allocation.
#
# The three numbers that decide the story:
#   dram__throughput...            how close to the bus we actually get
#   lts__t_sector_hit_rate         is L2 helping, or are we truly streaming?
#   sm__warps_active / waves       is there enough parallelism to saturate?

set -Eeuo pipefail
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$HOME/miniforge3/envs/gb10/bin:$CUDA_HOME/bin:$PATH:/apps/cuda/12.8/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0;12.1"
export HETERO="$HOME/hetero-serve"

REPO="$HOME/gb10-attn"
OUT="$REPO/results/$(hostname -s)-$(date +%Y%m%d-%H%M%S)-ncu"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/console.log") 2>&1
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

cd "$REPO"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | tee "$OUT/gpu.txt"

# ---- gate: are counters actually available? --------------------------------
say "counter permission check"
command -v ncu >/dev/null 2>&1 || { echo "ncu not on PATH"; exit 1; }
ncu --version | head -2
if ! ncu --metrics gpu__time_duration.sum --target-processes all \
        python -c "import torch; torch.randn(1024, device='cuda').sum().item()" \
        > "$OUT/permcheck.txt" 2>&1; then
  if grep -q "ERR_NVGPUCTRPERM" "$OUT/permcheck.txt"; then
    echo
    echo "BLOCKED: ERR_NVGPUCTRPERM -- the driver still restricts performance"
    echo "counters to admin users on this node. Nothing to do until RCAC applies"
    echo "NVreg_RestrictProfilingToAdminUsers=0 (see docs/rcac-ticket-gpu-counters.md)."
    echo "Releasing the allocation rather than burning 30 minutes."
    exit 0
  fi
  echo "ncu failed for some other reason:"; tail -20 "$OUT/permcheck.txt"; exit 1
fi
echo "counters available -- proceeding"

# Metrics as an array: the first attempt at this built one long backslash-
# continued string and fed it through `tr`, which mangled it silently.
METRICS=(
  dram__bytes_read.sum
  dram__bytes_write.sum
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  lts__t_sector_hit_rate.pct
  lts__t_sectors_srcunit_tex_op_read.sum
  l1tex__t_sector_hit_rate.pct
  sm__warps_active.avg.pct_of_peak_sustained_active
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  launch__waves_per_multiprocessor
  launch__occupancy_limit_registers
  launch__occupancy_limit_shared_mem
  gpu__time_duration.sum
)
M=$(IFS=,; echo "${METRICS[*]}")

# Shapes chosen to bracket the bandwidth story: one comfortably above L2
# (streaming, the honest roofline point) and one at L2 capacity (where the
# hot-cache inflation peaked), so the L2 hit rate should differ sharply.
for SHAPE in "16 2048" "16 512"; do
  set -- $SHAPE; B=$1; C=$2
  say "profiling b=$B ctx=$C"
  ncu --target-processes all -f --csv --page raw \
      --metrics "$M" \
      --kernel-name-base demangled \
      --kernel-name regex:"paged_attention" \
      python "$HETERO/scripts/bench_kernel.py" --batch "$B" --context "$C" --iters 3 \
      > "$OUT/ncu_b${B}_c${C}.csv" 2>"$OUT/ncu_b${B}_c${C}.err" || \
      { echo "  failed:"; tail -5 "$OUT/ncu_b${B}_c${C}.err"; continue; }
  echo "  rows: $(wc -l < "$OUT/ncu_b${B}_c${C}.csv")"
done

say "summary"
python tools/summarize_ncu.py "$OUT" | tee "$OUT/summary.txt" || echo "(summarizer failed)"

say "done"; ls "$OUT"
