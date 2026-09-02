#!/usr/bin/env bash
#SBATCH -A mithuna
#SBATCH -p gb10
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH -t 00:45:00
#SBATCH -J gb10-ext
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Post-int32-fix run:
#   A. does the 20-point sweep that used to fault now pass end to end?
#   B. where is the NEW ceiling, and does it land where predicted?
#   C. roofline across the region the fix unlocked (b=64/128, ctx up to 4096)
#   D. CUDA unit tests, which only ever run on a GPU node

set -Eeuo pipefail
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$HOME/miniforge3/envs/gb10/bin:$CUDA_HOME/bin:$PATH:/apps/cuda/12.8/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0;12.1"
export HETERO="$HOME/hetero-serve"

REPO="$HOME/gb10-attn"
OUT="$REPO/results/$(hostname -s)-$(date +%Y%m%d-%H%M%S)-extended"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/console.log") 2>&1
say() { printf '\n\033[1m== %s\033[0m  (t+%ds)\n' "$*" "$SECONDS"; }

cd "$REPO"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | tee "$OUT/gpu.txt"
{ echo "gb10-attn $(git -C "$REPO" rev-parse --short HEAD)"
  echo "hetero-serve $(git -C "$HETERO" rev-parse --short HEAD)"; } | tee "$OUT/provenance.txt"

say "A. 20-point sweep (previously 8 of these faulted)"
CUDA_LAUNCH_BLOCKING=1 python harness/repro_crash.py --bisect > "$OUT/sweep_after_fix.txt" 2>&1 || true
grep -E "^  b=|smallest|none faulted" "$OUT/sweep_after_fix.txt" | head -30

say "B. where is the new ceiling?"
python - > "$OUT/headroom.txt" 2>&1 <<'PY'
import sys; sys.path.insert(0, __import__('os').environ['HETERO'])
import torch
from heteroserve.config import KVConfig, ModelConfig
from heteroserve.kv.torch_blocks import TorchBlockAllocator
cfg = ModelConfig()
print(f"{'blocks':>8} {'elem/layer':>16} {'frac of int32':>14}")
for nb in (7282, 20000, 50000, 87000, 87400, 100000):
    kv = KVConfig(block_size=16, num_blocks=nb, dtype="float16")
    per = 2 * nb * 16 * cfg.kv_heads * cfg.head_dim
    print(f"{nb:8d} {per:16,d} {per/(2**31-1):13.2%}")
print("\nold monolithic limit was 7,282 blocks (whole pool over 2**31)")
print(f"new per-layer limit is {(2**31-1)//(2*16*cfg.kv_heads*cfg.head_dim):,} blocks")
PY
cat "$OUT/headroom.txt"

say "C. roofline in the newly unlocked region"
for B in 64 128; do
  for C in 2048 4096; do
    echo "  b=$B c=$C"
    ( cd "$HETERO" && timeout 500 python scripts/bench_kernel.py --batch "$B" --context "$C" \
        --iters 30 --cold-pools 1 ) > "$OUT/ext_b${B}_c${C}.txt" 2>&1 || echo "    (failed/timeout)"
    grep -E "v3 kernel|minimum traffic|cache:" "$OUT/ext_b${B}_c${C}.txt" | head -4 || true
  done
done

say "D. CUDA unit tests (skipped on the login node)"
( cd "$HETERO" && timeout 900 python -m pytest -q ) > "$OUT/pytest_gpu.txt" 2>&1 || true
tail -6 "$OUT/pytest_gpu.txt"

say "done"; ls "$OUT"
