#!/usr/bin/env python3
"""Fail-fast NCCL collective probe for a local tensor-parallel group."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=30))
    try:
        value = torch.tensor([local_rank + 1.0], device=f"cuda:{local_rank}")
        dist.all_reduce(value)
        torch.cuda.synchronize()
        expected = world_size * (world_size + 1) / 2
        actual = value.item()
        if actual != expected:
            raise RuntimeError(
                f"NCCL all_reduce returned {actual}, expected {expected}"
            )
        if local_rank == 0:
            print(
                f"NCCL collective probe passed: world_size={world_size}, "
                f"all_reduce={actual:.1f}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
