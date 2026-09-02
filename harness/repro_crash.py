#!/usr/bin/env python3
"""Smallest reproducer for the illegal memory access at large (batch, context).

Observed on GB10 (sm_121), torch 2.14.0+cu130, CUDA 13.2:

    b=64  ctx=512   ok
    b=64  ctx=2048  CUDA error: an illegal memory access was encountered
    b=128 ctx=2048  same

The traceback lands in TorchBlockAllocator.write_kv -> _as_tensor, i.e. during
*setup*, before any attention kernel launches. So this is very likely KV-pool
indexing rather than the attention kernels -- but CUDA errors are reported
asynchronously, so the reported site is not necessarily the guilty launch.
Run with CUDA_LAUNCH_BLOCKING=1 to make the attribution trustworthy.

    CUDA_LAUNCH_BLOCKING=1 python harness/repro_crash.py --bisect
    compute-sanitizer --tool memcheck python harness/repro_crash.py --batch 64 --context 2048

Exit 0 = the shape survived, 1 = it faulted.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

HETERO = os.environ.get("HETERO", str(Path.home() / "hetero-serve"))
sys.path.insert(0, HETERO)

import numpy as np  # noqa: E402


def attempt(batch: int, context: int, block_size: int = 16,
            dtype: str = "float16", verbose: bool = True) -> tuple[bool, str]:
    """Allocate a KV pool and write into it. No attention. Returns (ok, note)."""
    import torch
    from heteroserve.config import KVConfig, ModelConfig
    from heteroserve.kv.torch_blocks import TorchBlockAllocator

    cfg = ModelConfig()
    blocks_needed = batch * (context // block_size + 4)
    kv = KVConfig(block_size=block_size,
                  num_blocks=max(64, blocks_needed), dtype=dtype)

    note = (f"b={batch} ctx={context} blocks={blocks_needed} "
            f"pool={kv.num_blocks} blk={block_size}")
    if verbose:
        print(f"  {note} ...", end="", flush=True)
    try:
        alloc = TorchBlockAllocator(kv, cfg, device="cuda:0")
        rng = np.random.default_rng(0)
        for _ in range(batch):
            a = alloc.allocate([int(t) for t in rng.integers(1, 40000, size=context)])
            k = rng.standard_normal(
                (cfg.n_layer, cfg.n_head, context, cfg.head_dim)).astype(np.float32)
            alloc.write_kv(a.block_ids, 0, k, k)
        torch.cuda.synchronize()
    except Exception as exc:
        if verbose:
            print(f" FAULT: {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
        return False, f"{note}: {type(exc).__name__}"
    if verbose:
        print(" ok")
    return True, note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--block-size", dest="block_size", type=int, default=16)
    ap.add_argument("--bisect", action="store_true",
                    help="search for the smallest faulting shape")
    args = ap.parse_args()

    print(f"CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING', 'unset')}")

    if not args.bisect:
        ok, _ = attempt(args.batch, args.context, args.block_size)
        return 0 if ok else 1

    # A fault is sticky: once a context is poisoned every later CUDA call in
    # this process fails too. So each shape is probed in a fresh subprocess.
    import subprocess
    def probe(b, c):
        r = subprocess.run(
            [sys.executable, __file__, "--batch", str(b), "--context", str(c),
             "--block-size", str(args.block_size)],
            capture_output=True, text=True, env={**os.environ})
        return r.returncode == 0, (r.stdout + r.stderr)

    print("\n=== sweep: which (batch, context) fault? ===")
    grid = {}
    for b in (16, 32, 64, 96, 128):
        for c in (512, 1024, 1536, 2048):
            ok, _ = probe(b, c)
            grid[(b, c)] = ok
            print(f"  b={b:4d} ctx={c:5d}  {'ok' if ok else 'FAULT'}")

    faults = sorted([k for k, v in grid.items() if not v], key=lambda t: t[0] * t[1])
    print("\n=== smallest faulting shape ===")
    if not faults:
        print("  none faulted -- not reproduced at these shapes")
        return 0
    b, c = faults[0]
    print(f"  b={b} ctx={c}  (blocks={b*(c//args.block_size+4)})")
    print("\n=== that shape's output ===")
    _, out = probe(b, c)
    print(out[-2500:])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
