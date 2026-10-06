#!/usr/bin/env python3
import argparse
import json
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def backend_for_runtime() -> str:
    # ROCm PyTorch intentionally exposes the CUDA-compatible torch API surface;
    # NCCL backend calls are serviced by RCCL on AMD systems.
    return "nccl" if torch.cuda.is_available() else "gloo"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()

    backend = backend_for_runtime()
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.manual_seed(1234)

    if torch.cuda.is_available():
        device = torch.device("cuda", rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    model = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 4)).to(device)
    ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    first_loss = None
    for _ in range(args.steps):
        # Same synthetic batch on every rank keeps this CI smoke deterministic.
        g = torch.Generator(device=device).manual_seed(777)
        x = torch.randn(args.batch, 16, generator=g, device=device)
        y = ((x[:, 0] > 0).long() + 2 * (x[:, 1] > 0).long()) % 4
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(ddp(x), y)
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
        loss.backward()
        opt.step()

    checksum = torch.tensor(
        [sum(float(p.detach().float().sum().cpu()) for p in ddp.module.parameters())],
        device=device,
        dtype=torch.float64,
    )
    gathered = [torch.zeros_like(checksum) for _ in range(world)]
    dist.all_gather(gathered, checksum)
    checks = [float(t.cpu()) for t in gathered]

    final = torch.tensor([float(loss.detach().cpu())], device=device, dtype=torch.float64)
    dist.all_reduce(final, op=dist.ReduceOp.SUM)
    final_mean = float((final / world).cpu())

    if rank == 0:
        spread = max(checks) - min(checks)
        result = {
            "world_size": world,
            "backend": backend,
            "device": str(device),
            "first_loss": first_loss,
            "final_mean_loss": final_mean,
            "parameter_checksum_spread": spread,
        }
        print(json.dumps(result, sort_keys=True))
        if spread > 1e-8:
            raise SystemExit(f"DDP parameters diverged: spread={spread}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
