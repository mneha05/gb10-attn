#!/usr/bin/env python3
"""What compute capability does this box actually report, and to whom?

On GB10 the answer differs by layer, which is the whole point of this probe:

  * the driver / CUDA runtime report the real SM 12.1
  * torch reports 12.1 and will emit sm_121 cubins
  * TensorRT-LLM's C++ layer *deliberately* reports 120, because
    tensorrt_llm::common::getSMVersion() special-cases
        if (sm == 121 && !queryRealSmArch) return 120;
    so that the sm_120 kernel tables get reused on sm_121 parts.
    (cpp/include/tensorrt_llm/common/cudaUtils.h)

Two callers opt out of the mask by passing queryRealSmArch=true --
ncclUtils.cpp and cublasScaledMM.cpp -- because they need the true part.

Run this on a GB10 compute node and commit the output next to the benchmarks;
it is the evidence for the dispatch writeup.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys


def sh(*cmd: str) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def probe_torch() -> dict:
    try:
        import torch
    except ImportError as exc:
        return {"available": False, "error": f"{exc}"}
    d: dict = {
        "available": True,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        # What torch was *compiled* to emit. If sm_121 is absent, torch will
        # fall back to PTX-JIT from the highest sm_120 PTX it has -- which is
        # exactly the silent-reuse path we are documenting.
        "compiled_arch_list": getattr(torch.cuda, "get_arch_list", lambda: [])(),
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    }
    if d["cuda_available"]:
        major, minor = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)
        d.update({
            "device_name": props.name,
            "capability": f"{major}.{minor}",
            "sm_arch_int": major * 10 + minor,          # 121 on GB10
            "trtllm_would_report": 120 if major * 10 + minor == 121 else major * 10 + minor,
            "multi_processor_count": props.multi_processor_count,
            "total_memory_GiB": round(props.total_memory / 2**30, 2),
        })
        # GB10 is unified LPDDR5X: the "GPU memory" is the system's 128 GB, and
        # there is no PCIe hop for H2D. Record both so the roofline denominator
        # is not guessed later.
        for attr in ("memory_clock_rate", "memory_bus_width", "l2_cache_size"):
            if hasattr(props, attr):
                d[attr] = getattr(props, attr)
        if d.get("memory_clock_rate") and d.get("memory_bus_width"):
            # kHz * bits -> GB/s, DDR (x2)
            gbs = d["memory_clock_rate"] * 1e3 * 2 * d["memory_bus_width"] / 8 / 1e9
            d["derived_peak_GBs"] = round(gbs, 1)
    return d


def main() -> int:
    report = {
        "host": platform.node(),
        "machine": platform.machine(),          # expect aarch64 on GB10
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "nvidia_smi": sh("nvidia-smi",
                         "--query-gpu=name,compute_cap,memory.total,driver_version",
                         "--format=csv,noheader"),
        "nvcc": sh("nvcc", "--version"),
        "torch": probe_torch(),
    }

    print(json.dumps(report, indent=2, default=str))

    t = report["torch"]
    if t.get("sm_arch_int") == 121:
        print("\n"
              "GB10 confirmed: real SM 12.1.\n"
              "  torch/nvcc target : sm_121\n"
              "  TRT-LLM getSMVersion() reports : 120  (deliberate mask)\n"
              "  -> any custom op built against TRT-LLM's reported SM will be\n"
              "     compiled and dispatched as sm_120. Build ours for BOTH.",
              file=sys.stderr)
    elif t.get("cuda_available"):
        print(f"\nNote: this is not a GB10 (SM {t.get('capability')}).", file=sys.stderr)
    else:
        print("\nNo CUDA device visible. On rowdy this means you are still on the "
              "login node -- get a compute node first:\n"
              "  sinteractive -A mithuna -p gb10 -c 20 --gres=gpu:1 -t 30:00",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
