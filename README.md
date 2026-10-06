# gb10-attn — paged attention on NVIDIA GB10 (SM 12.1)

A measured characterization of paged-attention decode on GB10 (DGX Spark class,
sm_121, unified LPDDR5X), plus two bugs the port surfaced.

All numbers below were produced on Purdue RCAC `rowdy`, partition `gb10`, node
`c000`/`c001`, and the raw artifacts are committed under [results/](results/).

## Headline

**The v3 context-split (FlashDecoding) kernel sustains 82–85% of GB10's
273 GB/s memory bandwidth** at large working sets, ~14–16× PyTorch SDPA on the
same paged layout. Peak observed: **231.3 GB/s (84.7%)** at batch=64,
ctx=2048 — a shape that crashed before the int32 fix below.

Full writeup: [docs/REPORT.md](docs/REPORT.md).

That number is lower than the first run reported, and the difference is the
interesting part.

## AMD ROCm + distributed training lane

This repo now includes an AMD portability and training lane under [`rocm/`](rocm/) and [`distributed_train/`](distributed_train/).

Verified in GitHub Actions run [#37526744469](https://github.com/mneha05/gb10-attn/actions/runs/37526744469):

- AMD's ROCm 7.2.4 `hipcc` compiled the standalone HIP port of `paged_attn_v1` for **gfx942**.
- The CI runner did not expose an AMD GPU, so the HIP binary reports a compile-only smoke result there; AMD hardware runtime/performance remains a separate bare-metal validation step.
- A real **2-process PyTorch DDP** training job completed with `world_size=2`, Gloo backend on CPU CI, loss decreasing from **1.4313 to 0.3015**, and **0.0 parameter checksum spread** between ranks.
- The same DDP code automatically selects the NCCL backend when a GPU runtime is present; on ROCm PyTorch that routes to RCCL.
- `rocm/baremetal_probe.sh` captures CPU/NUMA, PCIe, huge pages, IOMMU, `hipconfig`, `rocminfo`, and `rocm-smi` provenance.
- Slurm launchers are included for a single-node AMD HIP kernel validation and a 4-GPU ROCm DDP run.

This is evidence of hands-on **HIP/ROCm porting** and **distributed-training mechanics**. It does not claim AMD hardware benchmark numbers until the Slurm lane is run on an actual AMD GPU node.

## Arm CPU optimization lane

The same GB10 node also exposes a 20-core Arm CPU (10 Cortex-X925 + 10
Cortex-A725), so this repo now includes a CPU-side attention lab under
[`arm_cpu/`](arm_cpu/README.md).

Implemented and CI-validated:

- scalar FP32 attention-score reference,
- **Neon / Advanced SIMD** Q·K dot-product kernel,
- vector-length-agnostic **SVE** kernel,
- Linux HWCAP dispatch for Neon, DotProd, SVE/SVE2 and **SME/SME2** capability,
- AArch64 cross-build + QEMU correctness checks,
- object-code checks proving Neon and SVE instructions are emitted,
- `perf`, eBPF/bpftrace, **CoreSight ETM**, and LLVM/`llvm-mca` profiling harnesses,
- optional **KleidiAI** build lane,
- **ExecuTorch XNNPACK** attention export for Arm CPU / Android experiments,
- an RCAC `gb10` Slurm job that records CPU provenance and benchmark CSVs.

The Arm CI proves correctness and instruction generation. It does **not**
invent GB10 CPU speedup numbers: those are only added after the Slurm job is
actually run on the hardware.

## Two bugs found by moving Turing code to Blackwell

### 1. The roofline denominator was 2× too large

`peak_bandwidth_gbs()` computed `clock × 2 × width / 8`. That is correct on the
T4 this code grew up on and wrong on GB10:

| part | memory | reported clock | correct formula | peak |
|---|---|---|---|---|
| Tesla T4 | GDDR6 | 5001 MHz | clock **× 2** × 256/8 | 320 GB/s |
| NVIDIA GB10 | LPDDR5X | 8533 MHz | clock × 256/8 | **273 GB/s** |

`cudaDevAttrMemoryClockRate` reports a clock needing the DDR doubling on GDDR6,
but already reports the *effective transfer rate* on LPDDR5X. Nothing in the API
distinguishes them. The old formula invented a 546 GB/s bus, so a kernel at 84%
of bandwidth reported as 42%. Confirmed on-device: the probe emits
`peak_GBs_no_double: 273.1` against NVIDIA's published 273 GB/s.

Fixed upstream in `hetero-serve` — prefer the vendor figure for known parts,
fall back to the old formula for unknown ones, and always return the provenance.

### 2. The benchmark was timing L2, not DRAM

GB10's L2 is **24.0 MiB**. The benchmark re-read one KV region 50 times, so
after the first iteration everything hit cache. The tell was arithmetic:

```
batch=4  ctx=2048   working set 25.2 MB (= 1.00 x L2)   459.0 GB/s = 168% of peak
```

168% of a 273 GB/s bus is not a measurement. Inflation peaked exactly where the
working set matched L2 capacity, and vanished above ~4× L2.

`--cold-pools N` now allocates N disjoint KV regions and rotates the block table
through them while timing, so a region is evicted before it is read again.

| working set | ÷ L2 | hot (1 region) | cold (4 regions) |
|---|---|---|---|
| 1.6 MB | 0.06 | 72.7 (26.6%) | 72.6 (26.6%) |
| 6.3 MB | 0.25 | 284.0 (104%) | 182.7 (66.9%) |
| 6.3 MB | 0.25 | 294.9 (108%) | 206.4 (75.6%) |
| 25.2 MB | 1.00 | 459.0 (168%) | 208.8 (76.5%) |
| 25.2 MB | 1.00 | 408.7 (150%) | 215.7 (79.0%) |
| 100.7 MB | 4.00 | 216.9 (79.4%) | — |
| 201.3 MB | 8.00 | 221.8 (81.2%) | — |

The validation that matters: cold small-working-set numbers (76–79%) agree with
the large working sets that were *naturally* cold anyway (78–81%). Two
independent routes to the same answer.

## An int32 overflow in the KV pool

At larger shapes the allocator faults with `CUDA error: an illegal memory
access` — inside `write_kv`, before any attention kernel launches.

The trigger is not batch or context. The pool is one contiguous
`[n_layer, 2, num_blocks, block_size, H, D]` tensor, and the fault appears
exactly when its **element count crosses 2³¹**. For gpt2 geometry each block
contributes 294,912 elements, so the limit is 7,282 blocks.

Predicted vs observed across a 20-point sweep: **20/20, no mismatches.**

| | blocks | elements | |
|---|---|---|---|
| largest passing | 6,528 (b=96, ctx=1024) | 1,925,185,536 | < 2³¹ |
| — | — | **2,147,483,647** | int32 max |
| smallest failing | 8,448 (b=64, ctx=2048) | 2,491,416,576 | > 2³¹ |

Mechanism: `pool[layer, 0]` for a late layer sits at a storage offset beyond 2³¹
*elements* (layer 11 begins 2.28 G elements in), so any int32 offset arithmetic
wraps. Reproducer: [harness/repro_crash.py](harness/repro_crash.py) (`--bisect`).
**Fixed** by allocating one tensor per layer; re-running the same 20-point
sweep afterwards, none faulted. New ceiling: 87,381 blocks per layer, 12.0×
the old limit — exactly the factor `n_layer` predicts.

## SM121 vs SM120 dispatch

[docs/sm121-dispatch.md](docs/sm121-dispatch.md), verified against
TensorRT-LLM `870d460c` and confirmed on hardware. TRT-LLM masks 12.1 → 12.0 in
three independent layers: the defaulted `getSMVersion(queryRealSmArch=false)`,
a second normalization inside `getXMMAKernelsV2()`, and an `-arch=sm_120f`
family target in the XQA JIT. Consequence: `mSM` is never 121 on a GB10, so
every `mSM == kSM_121` test is unreachable — though all are redundant
`||` disjuncts, so behaviour is correct and this is dead code, not a bug.

Stock PyTorch reaches the same place independently: `libtorch_cuda.so` carries
476 sm_120 cubins, exactly 1 sm_121, and **no PTX at all**, so GB10 runs on
sm_120 cubins via minor-version binary compatibility. (That one sm_121 cubin is
13.3 MB / 360 kernels, of which 260 are 256-byte empty stubs left by
`enable_2x_kernel_for_sm89` / `enable_3x_kernel_for_sm9x` guards.)

## What did *not* need porting

Nothing. All five kernels compiled and ran correctly on sm_121 with no source
changes, at `-gencode arch=compute_120,code=sm_120 -gencode
arch=compute_121,code=sm_121`. They use only warp shuffles, `__expf`, WMMA and
dynamic shared memory — no inline PTX, no `mma.sync`, no `cp.async`, no
`ldmatrix`, no cooperative groups, nothing cluster-dependent (which GB10 lacks).
The "port" was a non-event; the measurement was where the work turned out to be.

## Layout

```
kernels/   five paged-attention kernels as standalone .cu/.h, extracted
           verbatim from hetero-serve by tools/extract_kernels.py
harness/   probe_arch.py     what arch/L2/bandwidth does each layer report
           repro_crash.py    minimal int32-overflow reproducer (--bisect)
           gb10_burst.sh     build + correctness + sweep + ncu, time-boxed
           gb10*.sbatch      batch forms (preferred over holding a node)
docs/      sm121-dispatch.md            TRT-LLM SM121 handling, source-verified
           rcac-ticket-gpu-counters.md  ready-to-send ERR_NVGPUCTRPERM ticket
results/   one directory per job, committed as evidence
```

## Reproducing on rowdy

The login node has no GPU and no CUDA — expected, not a problem:

```bash
sbatch harness/gb10_verify.sbatch      # queues, runs, releases itself
```

Two things that will silently ruin a run:

- **Use CUDA 13.2.** `/usr/local/cuda-13.2` is the only toolkit on rowdy that
  can target sm_121 at all; the Lmod `cuda/12.4|12.5|12.8` modules stop at
  sm_120 and will mis-target without saying so.
- **It is aarch64.** Anything pip-installed or pulled from NGC must be the
  aarch64/sbsa variant.

## Known gaps

- **Nsight Compute is blocked** by `ERR_NVGPUCTRPERM` — a cluster driver
  setting, not an sm_121 incompatibility (Nsight connects to the device fine).
  Ticket drafted in `docs/`. Until it is granted there is no occupancy or
  DRAM-counter attribution, only wall-clock bandwidth.
- **No TensorRT-LLM comparison yet.** Everything here is these kernels vs
  PyTorch, not vs TRT-LLM. An aarch64 TRT-LLM container is staged in scratch;
  serving and NVFP4/FP8 characterization are not done.
- The int32 pool overflow is diagnosed but not fixed.
- fp16 only, one layer, gpt2 geometry.
