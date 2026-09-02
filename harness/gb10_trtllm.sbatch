#!/usr/bin/env bash
#SBATCH -A mithuna
#SBATCH -p gb10
#SBATCH --gres=gpu:1
#SBATCH -c 20
#SBATCH -t 00:45:00
#SBATCH -J gb10-trtllm
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Phase 1: what does NVIDIA's own stack achieve on this box?
#
# Everything measured so far is our kernels vs PyTorch. That is a weak
# baseline -- PyTorch SDPA on a gathered paged layout is not what anyone
# serves with. This measures TensorRT-LLM on the same hardware, on an NVFP4
# checkpoint, so the roofline has a real reference point.
#
# Deliberately staged: import -> engine build -> offline bench -> serve.
# Each stage is a plausible failure point on aarch64/sm_121 and each writes
# its own artifact, so a failure at stage 3 still leaves stages 1-2 as
# results rather than losing the whole allocation.

set -Eeuo pipefail

SIF=/scratch/rowdy/mahesh54/containers/tritonserver_25_08-trtllm-python-py3.sif
MODEL=/scratch/rowdy/mahesh54/models/nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
REPO="$HOME/gb10-attn"
OUT="$REPO/results/$(hostname -s)-$(date +%Y%m%d-%H%M%S)-trtllm"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/console.log") 2>&1
say() { printf '\n\033[1m== %s\033[0m  (t+%ds)\n' "$*" "$SECONDS"; }

export APPTAINER_CACHEDIR=/scratch/rowdy/mahesh54/.apptainer_cache
# --nv exposes the driver (libcuda.so.1) which the container lacks on its own.
# -B binds scratch so the container can see model and output paths.
AP=(apptainer exec --nv -B /scratch/rowdy/mahesh54:/scratch/rowdy/mahesh54 "$SIF")

nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$OUT/gpu.txt"
echo "model: $MODEL ($(du -sh "$MODEL" | cut -f1))" | tee "$OUT/provenance.txt"

say "1. does TRT-LLM import, and what SM does it think this is?"
"${AP[@]}" python3 - > "$OUT/trtllm_env.txt" 2>&1 <<'PY' || true
import torch, platform
print("arch     ", platform.machine())
print("torch    ", torch.__version__, "cuda", torch.version.cuda)
print("device   ", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
try:
    import tensorrt_llm as t
    print("trtllm   ", t.__version__)
    # The masking behaviour, straight from the horse's mouth: TRT-LLM's C++
    # getSMVersion() reports 120 for a real 121. See docs/sm121-dispatch.md.
    from tensorrt_llm._utils import get_sm_version
    print("trtllm reports SM:", get_sm_version())
except Exception as e:
    print("trtllm import FAILED:", type(e).__name__, e)
PY
cat "$OUT/trtllm_env.txt"

say "2. model config sanity (is the NVFP4 quantisation what we think?)"
"${AP[@]}" python3 -c "
import json,glob,os
for f in ['config.json','hf_quant_config.json','quant_config.json']:
    p=os.path.join('$MODEL',f)
    if os.path.exists(p):
        print('--',f); print(json.dumps(json.load(open(p)),indent=2)[:900])
" > "$OUT/model_config.txt" 2>&1 || true
head -40 "$OUT/model_config.txt"

say "3. offline throughput (trtllm-bench if present, else the LLM API)"
"${AP[@]}" bash -lc "command -v trtllm-bench trtllm-serve" > "$OUT/cli.txt" 2>&1 || true
cat "$OUT/cli.txt"

# The LLM API is the more portable entry point and does not need a separate
# engine-build step for a checkpoint TRT-LLM can load directly.
for BATCH in 1 8 32; do
  say "   decode throughput, batch=$BATCH"
  MODEL="$MODEL" BATCH="$BATCH" timeout 900 "${AP[@]}" python3 - \
      > "$OUT/bench_bs${BATCH}.txt" 2>&1 <<'PY' || echo "   (failed -- see bench_bs${BATCH}.txt)"
import os, time, torch
from tensorrt_llm import LLM, SamplingParams
model, bs = os.environ["MODEL"], int(os.environ["BATCH"])
# Short prompt, long generation: we want the decode phase to dominate, since
# that is the memory-bandwidth-bound regime the whole project is about.
prompts = ["Explain why memory bandwidth limits LLM decoding."] * bs
sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)
llm = LLM(model=model, max_batch_size=max(bs, 8))
llm.generate(prompts[:1], SamplingParams(max_tokens=8))       # warm up
torch.cuda.synchronize(); t0 = time.perf_counter()
outs = llm.generate(prompts, sp)
torch.cuda.synchronize(); dt = time.perf_counter() - t0
toks = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"batch={bs} generated={toks} tokens in {dt:.3f}s")
print(f"throughput = {toks/dt:.1f} tok/s   per-seq = {toks/dt/bs:.1f} tok/s")
PY
  grep -E "throughput|generated|Error|error" "$OUT/bench_bs${BATCH}.txt" | head -5 || true
done

say "done"; ls "$OUT"
