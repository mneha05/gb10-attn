#!/usr/bin/env bash
set -euo pipefail

BENCH="${1:-./build/arm_attn_bench}"
shift || true

if [[ ! -e /sys/bus/event_source/devices/cs_etm/type ]]; then
  echo "CoreSight ETM perf source is not exposed by this kernel."
  echo "Check CONFIG_CORESIGHT*, permissions, and /sys/bus/event_source/devices."
  exit 2
fi

OUT="${CORESIGHT_OUT:-coresight-perf.data}"

perf record -o "$OUT" -e cs_etm//u -- "$BENCH" "$@"

echo "wrote $OUT"
perf report --stdio --dump -i "$OUT" | head -n 120
