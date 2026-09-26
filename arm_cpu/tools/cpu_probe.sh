#!/usr/bin/env bash
set -euo pipefail

echo "=== uname ==="
uname -a
echo
echo "=== lscpu ==="
lscpu || true
echo
echo "=== Arm feature flags ==="
grep -m1 -E '^Features|^flags' /proc/cpuinfo || true
echo
echo "=== toolchain ==="
command -v clang++ >/dev/null && clang++ --version | head -n 1 || true
command -v g++ >/dev/null && g++ --version | head -n 1 || true
command -v perf >/dev/null && perf --version || true
command -v bpftrace >/dev/null && bpftrace --version || true
echo
echo "=== CoreSight sources ==="
ls -1 /sys/bus/event_source/devices 2>/dev/null | grep -E 'cs_etm|coresight' || true
