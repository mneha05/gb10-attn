#!/usr/bin/env bash
set -euo pipefail
NPROC="${NPROC_PER_NODE:-2}"
torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" \
  distributed_train/ddp_smoke.py --steps "${STEPS:-40}" --batch "${BATCH:-32}"
