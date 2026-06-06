#!/usr/bin/env python3
"""DDP Scaling Efficiency on 8xRTX 4090
========================================

Uses torchrun to launch multi-process DDP training simulation.
Measures throughput scaling across 2/4/8 GPUs.

Usage:
  torchrun --nproc_per_node=2 benchmark_ddp_scaling_4090.py
  torchrun --nproc_per_node=4 benchmark_ddp_scaling_4090.py
  torchrun --nproc_per_node=8 benchmark_ddp_scaling_4090.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.distributed as dist
import torch.nn as nn
import time
import json

def main():
    # Initialize DDP
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)

    print(f"[Rank {rank}] World size={world_size}, GPU={local_rank}, "
          f"Device={torch.cuda.get_device_name(local_rank)}")

    # Model configurations to test
    configs = [
        ("50M", 4096, 2048, 3),       # ~50M params, gradient ~200MB
        ("8M", 2048, 1024, 2),       # ~8M params
    ]

    results = []

    for name, H, inner, n_layers in configs:
        # Simple MLP model
        class SimpleModel(nn.Module):
            def __init__(self, H, inner, n_layers):
                super().__init__()
                layers = []
                for _ in range(n_layers):
                    layers.append(nn.Linear(H, inner))
                    layers.append(nn.ReLU())
                    layers.append(nn.Linear(inner, H))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        model = SimpleModel(H, inner, n_layers).cuda(local_rank)
        n_params = sum(p.numel() for p in model.parameters())
        model_size_mb = n_params * 4 / 1e6  # FP32

        # Wrap with DDP
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Data
        batch_size = 64
        x = torch.randn(batch_size, H, device=f'cuda:{local_rank}')
        y = torch.randn(batch_size, H, device=f'cuda:{local_rank}')

        # Warmup
        for _ in range(10):
            output = model(x)
            loss = ((output - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        dist.barrier()

        # Measure
        n_steps = 50
        dist.barrier()
        t_start = time.perf_counter()

        for _ in range(n_steps):
            output = model(x)
            loss = ((output - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dist.barrier()
        t_end = time.perf_counter()

        total_ms = (t_end - t_start) * 1000
        per_step_ms = total_ms / n_steps
        throughput = batch_size * world_size / (per_step_ms / 1000)

        # Only rank 0 prints
        if rank == 0:
            # Compute single-GPU baseline (estimated)
            single_gpu_ms = per_step_ms * world_size  # naive estimate
            # Actually need to measure separately, use theoretical
            # For now compute scaling vs ideal
            ideal_time = per_step_ms  # with perfect scaling, time = single_gpu_time/world_size
            # We measure total time, ideal would be single_gpu_time/world_size
            # But we don't have single_gpu_time, so compute relative
            print(f"  {name} ({n_params/1e6:.1f}M params): "
                  f"{per_step_ms:.2f}ms/step, "
                  f"{throughput:.0f} samples/s, "
                  f"model={model_size_mb:.1f}MB")

            results.append({
                "name": name, "world_size": world_size,
                "n_params_m": round(n_params/1e6, 1),
                "model_mb": round(model_size_mb, 1),
                "per_step_ms": round(per_step_ms, 2),
                "throughput": round(throughput, 0),
                "batch_size": batch_size,
            })

    # Save results (rank 0 only)
    if rank == 0:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(out_dir, 'ddp_scaling_results.json')

        # Append to existing results or create new
        all_results = []
        try:
            with open(out_path) as f:
                all_results = json.load(f)
        except FileNotFoundError:
            pass

        all_results.extend(results)
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {out_path}")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()