#!/bin/bash
#SBATCH --job-name=rocm-ddp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --output=rocm-ddp-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
bash rocm/baremetal_probe.sh "rocm-ddp-provenance-${SLURM_JOB_ID}.txt"
bash distributed_train/run_ddp.sh
