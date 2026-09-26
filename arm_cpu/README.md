# GB10 Arm CPU attention lab

A CPU-side companion to the CUDA paged-attention work in this repository.

The DGX Spark / GB10 system combines the Blackwell GPU with a 20-core Arm CPU
(10 Cortex-X925 + 10 Cortex-A725). This directory isolates the attention
score inner loop and measures how the Arm cores handle it with progressively
more specialized vector paths.

## Implemented backends

| backend | implementation | status |
|---|---|---|
| scalar | portable FP32 reference | runs everywhere |
| Neon | AArch64 Advanced SIMD, dual 128-bit FMA accumulators | implemented |
| SVE | vector-length-agnostic FP32 FMA loop | implemented |
| SME/SME2 | Linux HWCAP + vector-length capability probe | detected, not falsely claimed as a direct handwritten kernel |

The runtime dispatcher checks Linux HWCAP/HWCAP2 and only selects a compiled
backend that the machine reports as supported.

## Build

```bash
cmake -S arm_cpu -B arm_cpu/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build arm_cpu/build
arm_cpu/build/arm_attn_bench --backend auto
```

Example benchmark matrix:

```bash
for backend in scalar neon sve auto; do
  arm_cpu/build/arm_attn_bench     --backend "$backend" --tokens 2048 --dim 128 --iters 600 --csv
done
```

Every optimized backend is compared against the scalar reference before timing;
the benchmark exits non-zero when the maximum absolute error exceeds `1e-4`.

## GB10 hardware run

```bash
sbatch arm_cpu/gb10_arm_bench.sbatch
```

The job records CPU provenance, feature flags, scalar/Neon/SVE CSV measurements
and a `perf stat` pass when hardware counters are available.

**No GB10 Arm benchmark numbers are claimed in this README until that job has
actually been run on the node.**

## Linux performance tooling

```bash
arm_cpu/tools/profile_perf.sh   arm_cpu/build/arm_attn_bench --backend neon --tokens 2048 --dim 128

sudo bpftrace arm_cpu/tools/attention_sched.bt

arm_cpu/tools/profile_coresight.sh   arm_cpu/build/arm_attn_bench --backend auto

arm_cpu/tools/llvm_mca.sh
```

## ML runtime lanes

- `kleidiai/` builds Arm's KleidiAI microkernel library for a future
  handwritten-vs-library comparison.
- `executorch/` exports the attention-score probe to an ExecuTorch XNNPACK
  `.pte` program for Arm CPU / Android deployment experiments.

## Why this is separate from the CUDA benchmark

The CUDA results in the root of this repo are measured GPU bandwidth results.
The CPU lab is deliberately independent so CPU GFLOP/s, vector dispatch and PMU
counter claims cannot be confused with the existing GPU measurements.
