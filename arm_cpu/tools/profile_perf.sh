#!/usr/bin/env bash
set -euo pipefail

BENCH="${1:-./build/arm_attn_bench}"
shift || true

exec perf stat   -e task-clock,cycles,instructions,branches,branch-misses   -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses   -- "$BENCH" "$@"
