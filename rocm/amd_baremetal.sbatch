#!/bin/bash
#SBATCH --job-name=rocm-attn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=rocm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
bash rocm/baremetal_probe.sh "rocm-provenance-${SLURM_JOB_ID}.txt"
AMD_GPU_ARCH="${AMD_GPU_ARCH:-gfx942}" bash rocm/build_rocm.sh
./build/rocm/paged_attn_v1_hip
