# gb10-attn — paged attention on GB10 (SM 12.1)

Porting the `hetero-serve` paged-attention kernels from Turing (sm_75) to
GB10 (sm_121), and characterizing what TensorRT-LLM actually achieves against
GB10's 273 GB/s memory bandwidth.

## Status

| phase | state |
|---|---|
| 0. extract kernels from hetero-serve into standalone TUs | **done** |
| 1. characterize stock TRT-LLM decode roofline | harness written, **not yet run** |
| 2. port kernels to SM121 | static analysis **done**, build **not yet attempted** |
| 3. integrate as TRT-LLM custom op + head-to-head | not started |
| 4. upstream findings | one candidate identified, unverified |

**No GB10 measurement has been taken yet.** Every number in this repo will be
labelled with the node and commit that produced it; there are currently none.

## Layout

```
kernels/     five paged-attention kernels as standalone .cu/.h
             (extracted verbatim from hetero-serve by tools/extract_kernels.py)
tools/       extract_kernels.py -- re-run to re-sync from upstream
harness/     probe_arch.py   what SM does each layer report?
             gb10_burst.sh   one allocation, one command, all artifacts
docs/        sm121-dispatch.md  source-verified TRT-LLM SM121 dispatch analysis
results/     one directory per allocation, committed as evidence
```

## The kernels

Extracted from `github.com/mneha05/hetero-serve`, where they live as raw
strings compiled at runtime by `torch.utils.cpp_extension.load_inline`.

| file | what it is | port risk |
|---|---|---|
| `paged_attn_v1` | naive fused; scores in shared mem, tree reduction | low |
| `paged_attn_v2` | online softmax, one warp per (seq, head), shuffles | low |
| `paged_attn_v3` | v2 + context split (FlashDecoding) | low |
| `paged_attn_prefill` | causal paged prefill, S query tokens | low |
| `paged_attn_wmma` | WMMA tensor-core prefill, 16×16×16 fragments | **medium** |

Portability scan (see `docs/`): no inline PTX, no `mma.sync`, no `cp.async`, no
`ldmatrix`, no cooperative groups, no `__CUDA_ARCH__` guards. v1/v2/v3/prefill
use only warp shuffles, `__expf`, and dynamic shared memory — all stable from
sm_75 through sm_121. The WMMA prefill is the only one carrying a
hardware-generation assumption (`sm_75`'s 48 KB shared budget) and is the one
place a Blackwell shared-memory carveout question arises.

Upstream passed no `-arch`/`-gencode`; `load_inline` inferred the target from
the live device. On GB10 that inference is exactly the thing under study, so the
port makes it explicit.

## Running on rowdy

The login node has no GPU, no CUDA, no `nvidia-smi` — that is expected and does
not mean the hardware is unreachable. Get a compute node first:

```bash
sinteractive -A mithuna -p gb10 -c 20 --gres=gpu:1 -t 30:00
./harness/gb10_burst.sh
```

Allocations are 30 minutes, so the workflow is deliberately batch-shaped: all
editing and analysis happens off the node, and the node is only held for
compile-run-profile bursts. `gb10_burst.sh` time-boxes each phase and writes
everything to `results/<host>-<stamp>/` so you can collect and release.

GB10 is Grace-based **aarch64**. Anything pip-installed or pulled from NGC must
be the aarch64/sbsa variant; an x86 wheel or container will not run.

## Prior work

The Turing baseline — including the Nsight finding that v2 was occupancy-starved
at 0.1 waves/SM, which is what motivated v3's context split — lives in
`hetero-serve`. Its `scripts/bench_kernel.py` already reports achieved GB/s
against device peak rather than a speedup ratio, and that is the metric carried
forward here.
