# How TensorRT-LLM dispatches on GB10 (SM 12.1)

Source-verified against `NVIDIA/TensorRT-LLM` @ `870d460c` (2026-09-01).
Every claim below is a file:line you can open. Nothing here is measured yet —
this is the static picture that the GB10 measurements will be interpreted against.

## The short version

A GB10 is compute capability **12.1**. TensorRT-LLM's C++ layer mostly pretends
it is **12.0**, on purpose, so the existing sm_120 kernel tables and cubins get
reused. This is not a bug and not a workaround bolted on late — it is a
deliberate, documented policy with an explicit opt-out, and it is implemented
**independently in at least three layers**.

## Layer 1 — the global mask in `getSMVersion()`

`cpp/include/tensorrt_llm/common/cudaUtils.h:289-306`

```cpp
/// @param queryRealSmArch Whether to query the real SM architecture. example usage:
/// use real sm arch when do LUT tuning and use fake sm arch when reuse sm120 code
/// on sm121 devices.
inline int getSMVersion(bool queryRealSmArch = false)
{
    ...
    int sm = sm_major * 10 + sm_minor;
    if (sm == 121 && !queryRealSmArch)
    {
        return 120;
    }
    return sm;
}
```

The parameter **defaults to false**, so every unqualified `getSMVersion()` call
site on a GB10 receives `120`. In the whole tree only two callers opt out:

| caller | why it needs the truth |
|---|---|
| `cpp/tensorrt_llm/common/ncclUtils.cpp:156` | NCCL topology/transport selection |
| `cpp/tensorrt_llm/thop/cublasScaledMM.cpp:75` | picks a different cuBLAS algo list on a real 121 |

### Consequence: `mSM` is never 121

Both attention entry points cache the *masked* value at construction:

- `cpp/tensorrt_llm/common/attentionOp.h:597` — `int mSM = ...::getSMVersion();`
- `cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.h:94` — same

So every downstream `mSM == kSM_121` test is **statically unreachable on GB10**:

- `attentionOp.cpp:2886, 2894, 2900`
- `fmhaRunner.cpp:88`, `fmhaRunner.cpp:362` (`isSm120f`)
- `fused_multihead_attention_v2.cpp:289`

Important honesty note: **all of these are harmless.** Each is a `|| mSM == 121`
disjunct in an OR that already contains `mSM == 120`, so masked-to-120 takes the
same branch. The behaviour is correct; the `121` arms are dead redundancy. Do
not file this as a bug — it is at most a readability observation, and NVIDIA
wrote it defensively on purpose.

## Layer 2 — a second normalization at kernel-table lookup

`cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_v2.cpp:637-651`

```cpp
FusedMultiHeadAttentionXMMAKernelV2 const* getXMMAKernelsV2(..., unsigned int sm)
{
    if (sm == kSM_121) { sm = kSM_120; }
    if (sm == kSM_103) { sm = kSM_100; }   // SM103 uses SM100 FMHA v2 kernels
    ...
}
```

Belt and braces: even if a caller *did* pass a real 121, the FMHA kernel-metadata
lookup folds it to 120 before matching `kernelMeta.mSM`. This is why the mask in
layer 1 is safe to be defaulted-on.

## Layer 3 — the JIT compiles for a *family* target

`cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper/src/nvrtcWrapper.cpp:76-82`

```cpp
// SM121 uses the same cubin target as SM120 (sm_120f) for compatibility.
if (SM == 120 || SM == 121) { return "-arch=sm_120f"; }
```

Note the `f` suffix — a CUDA 12.9+ *family-conditional* target. `sm_120f` cubins
are valid across the sm_120 family including sm_121, which is the mechanism that
makes the whole reuse strategy sound rather than merely convenient.

### A real (if benign) consequence worth measuring

`cpp/kernels/xqa/mha.cu:92` and `cpp/kernels/xqa/utils.cuh:49` branch on the
*device-side* macro:

```cpp
#if __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210
```

When XQA is built through the JIT path above, `-arch=sm_120f` means
`__CUDA_ARCH__ == 1200`, so the `1210` arm never fires. Here too the two arms
set identical values (`preferedKHeadPartBytes = 64`, `cacheVTileSeqLen = 32`),
so there is no behavioural difference today — but it is a live tripwire: anyone
who later gives sm_121 different tuning constants there will find the branch
silently ineffective under the JIT. **This is the one worth a question upstream**,
phrased as "is the 1210 arm intended to be reachable?", not as a bug report.

## Where sm_121 *is* treated as genuinely distinct

The reuse is not total. Real warp-specialized FMHA sources exist that name
sm_121 explicitly and encode hardware differences:

- `cpp/kernels/fmha_v2/src/fmha/warpspec_sm120/dma_sync_mma.h` — TMA producer.
  Line 54 notes these parts **lack cluster-launch hardware**; line 75 records
  that the `.tile` qualifier is *required* on sm_120/sm_121 PTX; line 66 and
  line 266 both cite GB10 (sm_121) reproducers.
- `cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu:656` —
  "NOT on workstation Blackwell (SM120/SM121) which lacks cluster launch hardware."
- `cpp/tensorrt_llm/kernels/cutlass_kernels/fp4_gemm/fp4_gemm_template.h:485,566`
  and the MoE GEMM dispatchers — NVFP4 paths carry explicit `120 || 121` arms.

So the accurate framing is: **sm_121 shares sm_120's *kernels*, while the
sm_120-family as a whole is distinguished from datacenter Blackwell (sm_100) by
the absence of cluster launch / distributed shared memory.** That second
distinction is the one with real performance consequences, and it is the one our
port has to respect.

## What this means for our custom op

1. If we query the SM through TRT-LLM's own utility we will be told 120. Build
   for **both** `12.0` and `12.1` (`TORCH_CUDA_ARCH_LIST="12.0;12.1"`) so we can
   measure whether an sm_121-native cubin differs at all from the sm_120f one.
   That A/B is a publishable result in itself and nobody has reported it.
2. Do not write anything depending on cluster launch / distributed shared memory
   or `cp.async.bulk` cluster variants — GB10 lacks the hardware.
3. Our extracted kernels use only warp shuffles, `__expf`, WMMA and dynamic
   shared memory (see `../kernels/`), none of which is cluster-dependent, so the
   port surface is small. The WMMA prefill is the only kernel with a
   hardware-generation assumption baked into a comment (`sm_75`'s 48 KB shared
   budget), and that is a *shared-memory carveout* question on Blackwell, not a
   correctness one.

## Open questions to answer with measurements

- Does an sm_121-native build of our kernels differ measurably from sm_120f?
- What is the real achieved fraction of 273 GB/s for TRT-LLM decode, per batch
  size, FP8 vs NVFP4?
- Since GB10 is unified LPDDR5X, the T4's PCIe H2D tax simply does not exist
  here. The T4 study found 33% lost host-side; the GB10 equivalent question is
  whether the *unified* path introduces its own contention when CPU and GPU
  share the same 273 GB/s. That is a genuinely new measurement.
