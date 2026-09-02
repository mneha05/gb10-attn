#!/usr/bin/env bash
# One allocation, one command. Land on a GB10 node, run this, collect artifacts,
# release the node.
#
#   sinteractive -A mithuna -p gb10 -c 20 --gres=gpu:1 -t 30:00
#   ./harness/gb10_burst.sh
#
# Everything that does not need the GPU (editing, reading, analysis) belongs on
# the login node or your laptop. This script assumes it is the only thing you
# will run while holding the allocation, and it is written to finish inside a
# 30 minute window: each phase is time-boxed and phases are skippable.
#
#   PHASES=probe,correct,bench ./harness/gb10_burst.sh    # skip the slow ncu pass
#   BUDGET_S=900 ./harness/gb10_burst.sh                  # tighter wall clock
#
# Artifacts land in results/<host>-<timestamp>/ which is what you copy back.

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HETERO="${HETERO:-$REPO/hetero-serve}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${OUT:-$REPO/results/$(hostname -s)-$STAMP}"
PHASES="${PHASES:-probe,correct,bench,ncu}"
BUDGET_S="${BUDGET_S:-1500}"          # 25 min of a 30 min allocation
START=$SECONDS

mkdir -p "$OUT"
# Tee everything; the log is an artifact too.
exec > >(tee -a "$OUT/console.log") 2>&1

say()  { printf '\n\033[1m== %s\033[0m  (t+%ds)\n' "$*" "$((SECONDS-START))"; }
warn() { printf '\033[33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
has()  { printf '%s' ",$PHASES," | grep -q ",$1,"; }
left() { echo $(( BUDGET_S - (SECONDS - START) )); }
budget_ok() {
  if [ "$(left)" -le "${1:-60}" ]; then
    warn "skipping $2 -- only $(left)s of budget left"; return 1
  fi; return 0
}

trap 'echo; echo "interrupted at t+$((SECONDS-START))s; partial artifacts in $OUT"' INT TERM

# ---------------------------------------------------------------- preflight --
say "preflight"
command -v nvidia-smi >/dev/null 2>&1 || die \
"no nvidia-smi -- you are almost certainly still on the rowdy login node.
 The login node has no GPU and no CUDA; that is expected. Get a compute node:
   sinteractive -A mithuna -p gb10 -c 20 --gres=gpu:1 -t 30:00"

nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
           --format=csv,noheader | tee "$OUT/gpu.txt"

ARCH="$(uname -m)"
echo "uname -m: $ARCH" | tee "$OUT/uname.txt"
[ "$ARCH" = "aarch64" ] || warn \
"expected aarch64 (GB10 is Grace-based). Got '$ARCH'. If you pip-installed or
 pulled an NGC container on an x86 box, it will not run here -- use the
 aarch64/sbsa variants."

CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')"
echo "compute capability (packed): $CAP"
if [ "$CAP" = "121" ]; then
  # Build for the real part *and* keep sm_120 PTX, because that is what
  # TRT-LLM's dispatch layer thinks it is talking to. See docs/sm121-dispatch.md.
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0;12.1}"
  echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
else
  warn "compute_cap is $CAP, not 121 -- this is not a GB10; results are not GB10 data."
fi

command -v ninja >/dev/null 2>&1 || warn "ninja missing; torch cpp_extension needs it (pip install ninja)"
[ -d "$HETERO" ] || die "hetero-serve not found at $HETERO (set HETERO=/path/to/hetero-serve)"

{
  echo "commit_gb10attn=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo "commit_heteroserve=$(git -C "$HETERO" rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo "slurm_job=${SLURM_JOB_ID:-none}  node=$(hostname -s)  stamp=$STAMP"
} | tee "$OUT/provenance.txt"

# -------------------------------------------------------------------- probe --
if has probe; then
  say "probe: what arch does each layer report"
  python "$REPO/harness/probe_arch.py" > "$OUT/arch.json" || warn "probe failed"
  tail -8 "$OUT/arch.json" || true
fi

# ------------------------------------------------------------- correctness --
# Never time a kernel that is wrong. bench_kernel.py already validates against
# the torch reference and reports FAILED instead of a number, but run the
# smallest shape first so a compile break costs seconds, not the allocation.
if has correct && budget_ok 120 correctness; then
  say "correctness (small shape, first compile happens here)"
  ( cd "$HETERO" && python scripts/bench_kernel.py --batch 2 --context 64 --iters 3 ) \
    > "$OUT/correctness.txt" 2>&1 || warn "correctness/compile failed -- see correctness.txt"
  grep -iE 'fail|error|mismatch|backend|ninja' "$OUT/correctness.txt" | head -20 || true
fi

# -------------------------------------------------------------------- bench --
# Decode is memory-bound, so sweep batch (the knob that turns latency-bound into
# bandwidth-bound) and context. Prefill is the compute-bound contrast.
if has bench && budget_ok 240 bench; then
  say "bench sweep"
  : > "$OUT/bench_index.txt"
  for B in 1 4 16 64 128; do
    for C in 512 2048; do
      [ "$(left)" -gt 90 ] || { warn "budget: stopping sweep at B=$B C=$C"; break 2; }
      f="$OUT/bench_b${B}_c${C}.txt"
      echo "  decode b=$B c=$C"
      ( cd "$HETERO" && python scripts/bench_kernel.py \
          --batch "$B" --context "$C" --iters 50 ) > "$f" 2>&1 \
        || warn "bench b=$B c=$C failed"
      echo "$f" >> "$OUT/bench_index.txt"
    done
  done
  if [ "$(left)" -gt 120 ]; then
    say "bench prefill (compute-bound contrast)"
    ( cd "$HETERO" && python scripts/bench_kernel.py \
        --batch 4 --context 2048 --prefill 512 --iters 20 ) \
      > "$OUT/bench_prefill.txt" 2>&1 || warn "prefill bench failed"
  fi
fi

# ---------------------------------------------------------------------- ncu --
# The roofline evidence. On GB10 "dram" counters are the LPDDR5X controller --
# unified memory, so unlike the T4 there is no PCIe hop hiding in the numbers.
if has ncu && budget_ok 300 ncu; then
  if ! command -v ncu >/dev/null 2>&1; then
    warn "ncu not on PATH -- try 'module load nsight-compute' (or cuda) and rerun with PHASES=ncu"
  else
    say "nsight compute (roofline counters)"
    METRICS='dram__bytes_read.sum,dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__waves_per_multiprocessor,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
lts__t_sector_hit_rate.pct'
    # --target-processes all: the kernel is JIT-built by a torch subprocess.
    # -f overwrite, --csv for the machine-readable copy we actually analyse.
    ( cd "$HETERO" && ncu --target-processes all -f \
        --metrics "$(echo "$METRICS" | tr -d ' \')" \
        --kernel-name-base demangled \
        --csv --page raw \
        python scripts/bench_kernel.py --batch 64 --context 2048 --iters 3 ) \
      > "$OUT/ncu_b64_c2048.csv" 2>"$OUT/ncu.err" \
      || warn "ncu failed -- see ncu.err (profiling may need --allow-legacy or perf counter perms)"
    head -3 "$OUT/ncu.err" 2>/dev/null || true
  fi
fi

# ------------------------------------------------------------------- wrapup --
say "done"
echo "budget used: $((SECONDS-START))s of ${BUDGET_S}s"
echo "artifacts:"
ls -la "$OUT"
echo
echo "copy back with:"
echo "  scp -r <user>@rowdy.rcac.purdue.edu:$OUT ./results/"
echo "You can release the allocation now (exit)."
