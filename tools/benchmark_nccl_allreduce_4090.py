#!/usr/bin/env python3
"""NCCL AllReduce Bandwidth on 8xRTX 4090
==========================================

Multi-process NCCL benchmark via torchrun.

Usage:
  torchrun --nproc_per_node=2 --master_port=29510 benchmark_nccl_allreduce_4090.py
  torchrun --nproc_per_node=4 --master_port=29510 benchmark_nccl_allreduce_4090.py
  torchrun --nproc_per_node=8 --master_port=29510 benchmark_nccl_allreduce_4090.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.distributed as dist
import time
import json

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"NCCL AllReduce Benchmark: {world_size} GPUs")

    # Test sizes: from 1KB to 512MB
    sizes = [
        ("1KB", 256),           # 256 FP32 elements = 1KB
        ("4KB", 1024),
        ("16KB", 4096),
        ("64KB", 16384),
        ("256KB", 65536),
        ("1MB", 262144),
        ("4MB", 1048576),
        ("16MB", 4194304),
        ("64MB", 16777216),
        ("256MB", 67108864),
    ]

    results = []

    for name, n_elements in sizes:
        tensor = torch.randn(n_elements, device=f'cuda:{local_rank}', dtype=torch.float32)
        size_bytes = n_elements * 4
        size_mb = size_bytes / 1e6

        # Warmup
        for _ in range(10):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.barrier()

        # Measure
        n_iters = 50
        dist.barrier()
        t_start = time.perf_counter()
        for _ in range(n_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.barrier()
        t_end = time.perf_counter()

        total_ms = (t_end - t_start) * 1000
        per_iter_ms = total_ms / n_iters

        # Effective bandwidth: AllReduce sends/receives (world_size-1)*size/world_size data
        # For ring algorithm: 2*(world_size-1) steps, each sends size/(world_size-1)
        # Total data moved per node = 2*size (for ring)
        # Effective bus BW = size_bytes / per_iter_ms * 1e3 / 1e9  (GB/s)
        eff_bw = size_bytes / per_iter_ms * 1e3 / 1e9  # single direction effective
        alg_bw = size_bytes * world_size / (per_iter_ms * 1e3) / 1e9  # algorithm bandwidth

        if rank == 0:
            print(f"  {name} ({size_mb:.2f}MB): {per_iter_ms:.3f}ms/iter, "
                  f"eff_bw={eff_bw:.2f} GB/s, alg_bw={alg_bw:.2f} GB/s")

        results.append({
            "world_size": world_size,
            "size_name": name,
            "size_mb": round(size_mb, 3),
            "per_iter_ms": round(per_iter_ms, 3),
            "eff_bw_gb_s": round(eff_bw, 2),
            "alg_bw_gb_s": round(alg_bw, 2),
        })

    # Save (rank 0)
    if rank == 0:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(out_dir, 'nccl_allreduce_results.json')
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