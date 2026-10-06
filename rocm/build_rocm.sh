#!/usr/bin/env bash
set -euo pipefail
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${HIPCC:-$ROCM_PATH/bin/hipcc}"
ARCH="${AMD_GPU_ARCH:-gfx942}"
OUT="${1:-build/rocm}"
mkdir -p "$OUT"
"$HIPCC" -O3 -std=c++17 --offload-arch="$ARCH" rocm/hip/paged_attn_v1_hip.cpp -o "$OUT/paged_attn_v1_hip"
echo "built $OUT/paged_attn_v1_hip for $ARCH"
