# Paged attention on NVIDIA GB10: a measured characterization

**Neha Mahesh** · September 2026
Hardware: NVIDIA GB10 (sm_121, 48 SMs, 128 GB unified LPDDR5X, 273 GB/s)
Cluster: Purdue RCAC `rowdy`, partition `gb10`, nodes `c000`/`c001`, aarch64
Software: CUDA 13.2, PyTorch 2.14.0+cu130, driver 595.58.03

Raw artifacts for every number here are committed under [`../results/`](../results/),
one directory per Slurm job, each carrying the git SHA that produced it.

---

## 1. What this is, and what it is not

The plan was to *port* a set of paged-attention kernels from Turing (sm_75) to
GB10 and find out what breaks. Almost nothing broke. All five kernels compiled
and ran correctly on sm_121 with **zero source changes**.

So this is not a porting writeup. It is a measurement writeup, and the
interesting results are two bugs that only became visible *because* the hardware
changed — both of which had been silently shaping results on the original
hardware's terms, and neither of which is in a kernel.

The honest one-line summary: **a context-split FlashDecoding kernel sustains
82–85% of GB10's memory bandwidth in the large-working-set regime, and getting
to a number I trusted required fixing the measurement twice.**

## 2. Methodology

Decode attention is memory-bandwidth-bound: it reads every cached K and V
exactly once and does almost no arithmetic per byte. So the meaningful score is
**achieved GB/s against the memory bus**, not a speedup ratio against an
arbitrary baseline.

- **Traffic model.** `bytes_moved = 2 · batch · context · heads · head_dim ·
  itemsize` — the minimum a correct decode *must* read. Anything above this is
  the kernel re-reading data it should have kept.
- **Baselines.** A naive gather-then-einsum path (what a non-fused engine does)
  and PyTorch SDPA on the same gathered layout. SDPA is the honest strong
  baseline; beating one's own slow einsum proves nothing.
- **Correctness gate.** Every path is checked against a torch reference before
  it is timed. A path that disagrees is reported as FAILED, never timed. All
  results below passed at ≤1.1e-06 max absolute error.
- **Kernels.** v1 naive fused (shared-memory scores, tree reduction); v2 online
  softmax, one warp per (sequence, head), warp shuffles; v3 = v2 plus a context
  split (FlashDecoding), originally motivated by Nsight on a T4 showing v2
  occupancy-starved at 0.1 waves/SM.

Every measurement ran as a Slurm batch job rather than an interactive
allocation, so nothing sat idle holding a shared GB10.

## 3. The roofline denominator was 2× too large

The benchmark computed peak bandwidth as `clock × 2 × width / 8`. On GB10 that
produced **546 GB/s** — against a real bus of 273 GB/s.

The formula is not simply wrong; it is *conditionally* wrong, which is why it
survived:

| part | memory | reported clock | correct | peak |
|---|---|---|---|---|
| Tesla T4 | GDDR6 | 5001 MHz | clock **× 2** × 256/8 | 320 GB/s ✓ |
| NVIDIA GB10 | LPDDR5X | 8533 MHz | clock × 256/8 | **273 GB/s** ✓ |

`cudaDevAttrMemoryClockRate` reports a clock still needing the DDR doubling on
GDDR6, but already reports the **effective transfer rate** on LPDDR5X
(LPDDR5X-8533 *is* 8533 MT/s). Nothing in the CUDA API distinguishes the two
conventions.

Confirmed on-device — the probe now emits both, side by side:

```json
"peak_GBs_no_double": 273.1,     // matches NVIDIA's published figure
"peak_GBs_doubled":   546.1,     // what the old formula claimed
"peak_GBs":           273.0,
"peak_GBs_source":    "vendor spec (clock is already the effective rate)"
```

Consequence: a kernel sitting at 84% of bandwidth reported as 42% — the single
number that decides whether a kernel is finished being optimised was off by
half, in the direction that invites wasted work.

**Fix:** prefer the vendor's published figure for known parts, fall back to the
old formula for unknown ones, and return provenance alongside the number so a
reader can tell a spec sheet from a guess. Upstreamed to `hetero-serve`.

## 4. The benchmark was measuring L2, not DRAM

With the denominator corrected, several readings were still impossible:

```
batch=4  ctx=2048   working set 25.2 MB   459.0 GB/s = 168% of a 273 GB/s bus
```

**GB10's L2 is 24.0 MiB (25,165,824 bytes).** The benchmark allocated one KV
region and re-read it 50 times, so every iteration after the first was served
from cache. Inflation peaked *exactly* where the working set matched L2
capacity, and disappeared above roughly 4× L2:

| working set | ÷ L2 | hot (1 region) | cold (4 regions) |
|---|---|---|---|
| 1.6 MB | 0.06 | 72.7 (26.6%) | 72.6 (26.6%) |
| 6.3 MB | 0.25 | 284.0 (**104%**) | 182.7 (66.9%) |
| 6.3 MB | 0.25 | 294.9 (**108%**) | 206.4 (75.6%) |
| 25.2 MB | 1.00 | 459.0 (**168%**) | 208.8 (76.5%) |
| 25.2 MB | 1.00 | 408.7 (**150%**) | 215.7 (79.0%) |
| 100.7 MB | 4.00 | 216.9 (79.4%) | — |
| 201.3 MB | 8.00 | 221.8 (81.2%) | — |

**Fix:** `--cold-pools N` allocates N disjoint KV regions and rotates the block
table through them across timing iterations, so a region is evicted before it is
read again. They share one pool tensor and differ only in which blocks they
occupy, so the cost is memory, not a second allocator. `--cold-pools 0` sizes N
so the rotation footprint is ~4× L2.

### The large-working-set regime

Fixing the int32 overflow (§5) unlocked shapes that previously crashed, and
those turn out to be the most trustworthy measurements in the project: at
16–32× L2 there is no cache ambiguity left to argue about.

| batch | ctx | working set | ÷ L2 | v3 | % of 273 |
|---|---|---|---|---|---|
| 64 | 2048 | 402.7 MB | 16× | 231.3 GB/s | **84.7%** |
| 64 | 4096 | 805.3 MB | 32× | 227.6 GB/s | **83.4%** |
| 128 | 2048 | 805.3 MB | 32× | 225.2 GB/s | **82.5%** |

Both of the first two shapes faulted before the fix, so the project's best
numbers exist only because the allocator bug was found and repaired.

**The validation that matters:** cold-mode small working sets (76–79%)
independently agree with large working sets that were *naturally* cold anyway
(78–81%). Two different routes to the same answer is the reason to believe it.

A note on the shape of the curve: it is non-monotonic in working-set size
because two effects fight. Small batches are parallelism-starved (b=1 sits at
27% regardless of cache), while large working sets exceed L2. The best hot
number appears where the two happen to align — which is precisely why the hot
measurement was untrustworthy.

## 5. An int32 overflow in the KV pool

At larger shapes the allocator faulted with `CUDA error: an illegal memory
access` — raised inside `write_kv`, **before any attention kernel launched**.

The trigger is neither batch nor context. The pool was one contiguous
`[n_layer, 2, num_blocks, block_size, H, D]` tensor, and the fault appears
exactly when its total element count crosses **2³¹**. For GPT-2 geometry each
block contributes 294,912 elements, putting the wall at 7,282 blocks.

Predicted vs observed over a 20-point (batch, context) sweep: **20/20, no
mismatches.**

| | blocks | elements | |
|---|---|---|---|
| largest passing | 6,528 (b=96, ctx=1024) | 1,925,185,536 | < 2³¹ |
| | | **2,147,483,647** | int32 max |
| smallest failing | 8,448 (b=64, ctx=2048) | 2,491,416,576 | > 2³¹ |

The mechanism is storage *offset*, not tensor size: `pool[layer, 0]` for a high
layer index begins beyond 2³¹ **elements** into the allocation — layer 11 of 12
starts 2.28 G elements in — so int32 offset arithmetic wraps.
`compute-sanitizer` localised the fault to `cudaLaunchKernel`.

**Fix:** allocate one tensor per layer. The largest offset any single view can
carry drops by a factor of `n_layer`, moving the wall out by the same factor. A
`_LayeredPool` proxy preserves the previous indexing surface exactly
(`pool[layer]`, `pool[layer, kv]`, `pool[:, :, ids]`, and assignment), verified
against the monolithic tensor form-by-form, so none of the ~30 call sites across
the engine, benchmarks and test suite had to change.

This raises the ceiling rather than removing it — a single layer must still stay
under 2³¹ elements (~87k blocks here, ~12× the old limit) — so
`capacity_headroom()` reports the remaining margin instead of letting the next
person discover it as an illegal access.

**Verified.** Re-running the same 20-point sweep after the fix: *none faulted*.
The measured new ceiling is **87,381 blocks per layer**, 12.0× the old 7,282 —
exactly the factor `n_layer` predicts.

## 6. Nsight attribution — blocked

Not done, and not for a reason worth hiding: Nsight Compute on this cluster
fails with `ERR_NVGPUCTRPERM`. The driver restricts performance counters to
admin users (`NVreg_RestrictProfilingToAdminUsers=1`).

Worth stating precisely, because the obvious guess is wrong: **Nsight connects
to the sm_121 device and launches the application correctly.** Only counter
collection is refused. This is a cluster policy setting, not a tool or
architecture compatibility problem. A support request is drafted in
[`rcac-ticket-gpu-counters.md`](rcac-ticket-gpu-counters.md).

Until that lands, every bandwidth figure here is derived from wall-clock time
and a traffic model — which is sound, but cannot answer *why* the kernel stops
at ~79% rather than ~95%. The job that will answer it
([`../harness/gb10_ncu.sbatch`](../harness/gb10_ncu.sbatch)) is written and gated
on a permission check, so it exits immediately rather than consuming an
allocation if counters are still restricted.

## 7. TensorRT-LLM comparison — in progress

Everything above is these kernels versus PyTorch. That is a real result but a
weak reference: SDPA over a gathered paged layout is not what anyone actually
serves with.

Staged so far: an aarch64 TensorRT-LLM container (14 GB) and
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` (19 GB), both in cluster scratch.
NVFP4 matters specifically here — Blackwell has the FP4 tensor cores, and
TRT-LLM's GEMM dispatch carries explicit `120 || 121` arms for those paths.

Not yet measured. Until it is, no claim in this report is "versus NVIDIA's
stack."

## 8. How TensorRT-LLM dispatches on sm_121

Full analysis in [`sm121-dispatch.md`](sm121-dispatch.md), verified against
TensorRT-LLM `870d460c` and confirmed against hardware.

A GB10 is compute capability 12.1. TensorRT-LLM deliberately reports **12.0**,
so the existing sm_120 kernel tables get reused, and it does so in **three
independent layers**:

1. `getSMVersion(bool queryRealSmArch = false)` returns 120 for a real 121. The
   parameter defaults to false, so every unqualified call site sees 120. Exactly
   two callers opt out — `ncclUtils.cpp` and `cublasScaledMM.cpp` — because they
   need the true part.
2. `getXMMAKernelsV2()` normalises 121 → 120 *again*, independently, at FMHA
   kernel-table lookup.
3. The XQA JIT emits `-arch=sm_120f` — a CUDA 12.9+ *family-conditional* target,
   the mechanism that makes the reuse sound rather than merely convenient.

Consequence: `mSM` is never 121 on a GB10, so every `mSM == kSM_121` test in
`attentionOp.cpp` and `fmhaRunner.cpp` is statically unreachable. **These are all
redundant `||` disjuncts in ORs already containing 120**, so behaviour is correct
and this is dead code, not a bug — worth documenting, not worth filing.

One genuine tripwire: `xqa/mha.cu:92` branches on `__CUDA_ARCH__ == 1210`, which
cannot fire under the `sm_120f` JIT (where `__CUDA_ARCH__` is 1200). Both arms
currently set identical constants, so nothing misbehaves today — but anyone who
later gives sm_121 different tuning constants there will find the branch
silently ineffective.

**Stock PyTorch arrives at the same strategy independently.** `libtorch_cuda.so`
(330 MB) carries 476 sm_120 cubins, exactly **1** sm_121, and **no PTX at all** —
so GB10 runs on sm_120 cubins via minor-version binary compatibility, with
nothing to JIT. That single sm_121 cubin is 13.3 MB / 360 kernels, of which
**260 are 256-byte empty stubs** left behind by `enable_2x_kernel_for_sm89` and
`enable_3x_kernel_for_sm9x` guards compiling their bodies out.
`torch.cuda.get_arch_list()` does not advertise sm_121 even though it ships.

## 9. What did not need porting

Nothing. All five kernels built and ran correctly on sm_121 at
`-gencode arch=compute_120,code=sm_120 -gencode arch=compute_121,code=sm_121`.

A portability scan explains why: they use only warp shuffles, `__expf`, WMMA and
dynamic shared memory. No inline PTX, no `mma.sync`, no `cp.async`, no
`ldmatrix`, no cooperative groups, and nothing depending on cluster launch or
distributed shared memory — which matters, because workstation Blackwell
(sm_120/sm_121) lacks that hardware, a difference TRT-LLM's own sources call out
explicitly.

The lesson is not "porting is easy." It is that *portable-by-construction
kernels port for free, and the fragile part was the measurement harness around
them* — the two real bugs were a bandwidth formula and a pool allocator, neither
of which is CUDA.

## 10. Limitations

- fp16 only; NVFP4 and FP8 paths not measured.
- One attention layer, GPT-2 geometry (12 heads, head_dim 64, block_size 16).
- No Nsight counters, so no occupancy or DRAM attribution (§6).
- No TensorRT-LLM numbers yet (§7); "vs NVIDIA's stack" is not yet earned.
- Single node, single GPU; nothing about multi-GPU or NVLink-C2C.
- The int32 ceiling is raised, not removed (§5): 87,381 blocks per layer.

## 11. Resume bullet

> Characterized paged-attention decode on NVIDIA GB10 (Grace-Blackwell,
> sm_121): measured a context-split FlashDecoding kernel at 82–85% of the
> 273 GB/s unified-memory bandwidth, ~14–16× PyTorch SDPA. Found and fixed two
> latent measurement bugs surfaced by the Turing→Blackwell move — a 2× error in
> the roofline denominator caused by GDDR6-vs-LPDDR5X memory-clock reporting
> semantics, and an int32 storage-offset overflow that faulted the KV pool past
> 2³¹ elements (root-caused to a model that predicted 20/20 observed
> pass/fail shapes, fixed via per-layer allocation). Documented TensorRT-LLM's
> three-layer sm_121→sm_120 dispatch masking from source and confirmed it on
> hardware.

Shorter, if one line is all there is:

> Measured a FlashDecoding kernel at 82–85% of NVIDIA GB10's 273 GB/s memory
> bandwidth on sm_121; found and fixed a 2× roofline-formula bug and an int32
> KV-pool overflow surfaced by the Turing→Blackwell port.
