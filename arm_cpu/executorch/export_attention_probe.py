#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower


class AttentionScoreProbe(torch.nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.scale = head_dim ** -0.5

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(keys, query.unsqueeze(-1)).squeeze(-1)
        return torch.softmax(scores * self.scale, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("attention_xnnpack.pte"))
    args = parser.parse_args()

    torch.manual_seed(2026)
    model = AttentionScoreProbe(args.dim).eval()
    example_inputs = (
        torch.randn(args.dim),
        torch.randn(args.tokens, args.dim),
    )

    exported = torch.export.export(model, example_inputs)
    program = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    args.out.write_bytes(program.buffer)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
