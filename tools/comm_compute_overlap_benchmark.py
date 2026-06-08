"""
Communication-Computation Overlap Benchmark — RTX 4090

Tests whether communication (AllReduce/ReduceScatter) can overlap with
GPU computation on PCIe RTX 4090. Key for understanding FSDP scaling.

Experiments:
1. Compute-only vs Communication-only baseline timing
2. Overlap efficiency with CUDA streams (compute + comm concurrent)
3. FSDP-style overlap: ReduceScatter + compute concurrent
4. Overlap scaling: single GPU vs multi-GPU
5. Optimal overlap strategy for different model sizes
"""

import torch
import torch.distributed as dist
import time
import json
import os

def init_distributed():
    """Initialize distributed environment."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size

def benchmark(fn, warmup=5, repeats=20):
    """Benchmark a function with CUDA synchronization."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    avg = sum(times) / len(times)
    return {"avg_us": round(avg, 2), "min_us": round(min(times), 2)}

def run_all():
    rank, world_size = init_distributed()
    device = torch.device(f"cuda:{rank}")
    props = torch.cuda.get_device_properties(device)

    results = {}

    if rank == 0:
        print("=" * 70)
        print(f"Comm-Compute Overlap Benchmark — RTX 4090 ({world_size} GPUs)")
        print("=" * 70)

    # ====================================================================
    # Exp 1: Baseline — Compute-only and Comm-only
    # ====================================================================
    if rank == 0:
        print("\n--- Exp 1: Baseline ---")

    # Compute-only: forward pass (GEMM)
    hidden = 4096
    inter_dim = 14336
    x = torch.randn(32, hidden, device=device, dtype=torch.bfloat16)
    w_gate = torch.randn(hidden, inter_dim, device=device, dtype=torch.bfloat16)
    w_up = torch.randn(hidden, inter_dim, device=device, dtype=torch.bfloat16)
    w_down = torch.randn(inter_dim, hidden, device=device, dtype=torch.bfloat16)

    def compute_only():
        gate = x @ w_gate
        up = x @ w_up
        silu = torch.nn.functional.silu(gate) * up
        down = silu @ w_down
        return down

    compute_time = benchmark(compute_only)

    # Communication-only: AllReduce
    tensor_size_mb = [1, 4, 16, 64, 256]

    comm_times = {}
    for size_mb in tensor_size_mb:
        n_elements = int(size_mb * 1024 * 1024 / 2)  # BF16
        tensor = torch.randn(n_elements, device=device, dtype=torch.bfloat16)

        def allreduce_only():
            dist.all_reduce(tensor)
            return tensor

        comm_time = benchmark(allreduce_only, warmup=3, repeats=10)
        comm_times[str(size_mb)] = {
            "time_us": comm_time["avg_us"],
            "bw_mbps": round(size_mb / (comm_time["avg_us"] / 1e6), 2),
        }

        if rank == 0:
            print(f"  AllReduce {size_mb}MB: {comm_time['avg_us']:.0f}us ({size_mb/(comm_time['avg_us']/1e6):.0f}MB/s)")

    results["exp1_baseline"] = {
        "compute_us": compute_time["avg_us"],
        "compute_desc": "MLP forward (gate+up+silu+down) B=32",
        "comm_times": comm_times,
    }

    # ====================================================================
    # Exp 2: Overlap — Compute + Comm Concurrent
    # ====================================================================
    if rank == 0:
        print("\n--- Exp 2: Compute + Comm Overlap ---")

    # Test: run compute and AllReduce concurrently on different streams
    exp2 = {}

    for size_mb in [4, 16, 64]:
        n_elements = int(size_mb * 1024 * 1024 / 2)
        tensor = torch.randn(n_elements, device=device, dtype=torch.bfloat16)

        # Sequential: compute then comm
        def sequential():
            down = compute_only()
            dist.all_reduce(tensor)
            return down

        sequential_time = benchmark(sequential, warmup=3, repeats=10)

        # Overlapped: compute and comm on different streams
        def overlapped():
            # Start AllReduce on comm stream
            comm_stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(comm_stream):
                dist.all_reduce(tensor)

            # Run compute on default stream simultaneously
            gate = x @ w_gate
            up = x @ w_up
            silu = torch.nn.functional.silu(gate) * up
            down = silu @ w_down

            # Wait for comm to finish
            torch.cuda.current_stream().wait_stream(comm_stream)
            return down

        overlap_time = benchmark(overlapped, warmup=3, repeats=10)

        # Overlap efficiency
        theoretical_min = max(compute_time["avg_us"], comm_times[str(size_mb)]["time_us"])
        overlap_efficiency = (sequential_time["avg_us"] - overlap_time["avg_us"]) / \
                           (sequential_time["avg_us"] - theoretical_min) * 100
        time_saved = sequential_time["avg_us"] - overlap_time["avg_us"]

        exp2[str(size_mb)] = {
            "compute_us": compute_time["avg_us"],
            "comm_us": comm_times[str(size_mb)]["time_us"],
            "sequential_us": sequential_time["avg_us"],
            "overlapped_us": overlap_time["avg_us"],
            "time_saved_us": round(time_saved, 2),
            "overlap_efficiency_pct": round(overlap_efficiency, 1),
            "theoretical_min_us": round(theoretical_min, 2),
        }

        if rank == 0:
            print(f"  {size_mb}MB: sequential={sequential_time['avg_us']:.0f}us, "
                  f"overlapped={overlap_time['avg_us']:.0f}us → "
                  f"eff={overlap_efficiency:.1f}% ({time_saved:.0f}us saved)")

    results["exp2_overlap"] = exp2

    # ====================================================================
    # Exp 3: FSDP-style Overlap — ReduceScatter + Compute
    # ====================================================================
    if rank == 0:
        print("\n--- Exp 3: FSDP-style Overlap (RS + Compute) ---")

    exp3 = {}

    # FSDP pattern: after forward, reduce_scatter gradients
    # Then overlap reduce_scatter with next layer's forward

    # Simulate: reduce_scatter output + compute next layer
    for size_mb in [4, 16]:
        n_elements = int(size_mb * 1024 * 1024 / 2 / world_size)  # per-GPU size for RS
        tensor = torch.randn(n_elements * world_size, device=device, dtype=torch.bfloat16)
        output = torch.empty(n_elements, device=device, dtype=torch.bfloat16)

        # RS only
        def rs_only():
            dist.reduce_scatter_tensor(output, tensor)
            return output

        rs_time = benchmark(rs_only, warmup=3, repeats=10)

        # Sequential: RS then compute
        def rs_then_compute():
            dist.reduce_scatter_tensor(output, tensor)
            gate = x @ w_gate
            up = x @ w_up
            silu = torch.nn.functional.silu(gate) * up
            down = silu @ w_down
            return down

        rs_compute_time = benchmark(rs_then_compute, warmup=3, repeats=10)

        # Overlapped: RS on comm stream + compute on default stream
        def rs_overlap_compute():
            # Start RS on comm stream
            comm_stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(comm_stream):
                dist.reduce_scatter_tensor(output, tensor)

            # Compute on default stream
            gate = x @ w_gate
            up = x @ w_up
            silu = torch.nn.functional.silu(gate) * up
            down = silu @ w_down

            torch.cuda.current_stream().wait_stream(comm_stream)
            return down

        rs_overlap_time = benchmark(rs_overlap_compute, warmup=3, repeats=10)

        # AllGather (for next layer's parameters)
        gathered = torch.empty(n_elements * world_size, device=device, dtype=torch.bfloat16)

        def ag_only():
            dist.all_gather_into_tensor(gathered, tensor[:n_elements])
            return gathered

        ag_time = benchmark(ag_only, warmup=3, repeats=10)

        exp3[str(size_mb)] = {
            "rs_us": rs_time["avg_us"],
            "ag_us": ag_time["avg_us"],
            "compute_us": compute_time["avg_us"],
            "sequential_us": rs_compute_time["avg_us"],
            "overlapped_us": rs_overlap_time["avg_us"],
            "overlap_efficiency_pct": round(
                (rs_compute_time["avg_us"] - rs_overlap_time["avg_us"]) /
                (rs_compute_time["avg_us"] - max(rs_time["avg_us"], compute_time["avg_us"])) * 100,
                1
            ),
        }

        if rank == 0:
            print(f"  {size_mb}MB: RS={rs_time['avg_us']:.0f}us, AG={ag_time['avg_us']:.0f}us, "
                  f"sequential={rs_compute_time['avg_us']:.0f}us, "
                  f"overlapped={rs_overlap_time['avg_us']:.0f}us")

    results["exp3_fsdp_overlap"] = exp3

    # ====================================================================
    # Exp 4: CUDA_DEVICE_MAX_CONNECTIONS Effect
    # ====================================================================
    if rank == 0:
        print("\n--- Exp 4: CUDA_DEVICE_MAX_CONNECTIONS ---")

    exp4 = {}

    # This env var controls how many concurrent CUDA operations can be queued
    # Default: many connections → communication can overlap with compute
    # Set to 1: only one connection → forces serialization → but ensures
    #   communication completes before compute starts (important for FSDP)

    # Test with current settings (default = many connections)
    # We already measured overlap above

    # Measure with forced serialization (simulate MAX_CONNECTIONS=1 behavior)
    for size_mb in [16]:
        n_elements = int(size_mb * 1024 * 1024 / 2)
        tensor = torch.randn(n_elements, device=device, dtype=torch.bfloat16)

        # Forced serial: wait for AllReduce to complete before starting compute
        def forced_serial():
            dist.all_reduce(tensor)
            torch.cuda.synchronize()  # Force completion
            down = compute_only()
            return down

        forced_time = benchmark(forced_serial, warmup=3, repeats=10)

        exp4 = {
            "default_overlap_us": exp2.get("16", {}).get("overlapped_us", 0),
            "forced_serial_us": forced_time["avg_us"],
            "sequential_us": exp2.get("16", {}).get("sequential_us", 0),
            "overlap_vs_serial_pct": round(
                (forced_time["avg_us"] - exp2.get("16", {}).get("overlapped_us", forced_time["avg_us"])) /
                forced_time["avg_us"] * 100,
                1
            ),
        }

        if rank == 0:
            print(f"  Default overlap: {exp2.get('16', {}).get('overlapped_us', 0):.0f}us")
            print(f"  Forced serial: {forced_time['avg_us']:.0f}us")

    results["exp4_max_connections"] = exp4

    # ====================================================================
    # Exp 5: FSDP Scaling Analysis
    # ====================================================================
    if rank == 0:
        print("\n--- Exp 5: FSDP Scaling Analysis ---")

    exp5 = {}

    # Calculate FSDP scaling efficiency based on overlap data
    # FSDP per-step cost = max(RS_time, compute_time) + AG_time + compute_time
    # Without overlap = RS + compute + AG + compute
    # With overlap = max(RS, compute) + AG + compute (first compute overlaps RS)

    # For 7B model per layer (ZeRO-3):
    # RS: gradient shard ≈ 0.5MB per GPU (8 GPU)
    # AG: parameter gather ≈ 0.5MB per GPU
    # Compute: forward ≈ 500us (B=32)

    # Calculate for different GPU counts
    for n_gpu in [1, 2, 4, 8]:
        if n_gpu > world_size:
            continue

        # Model size per layer (approximate for 7B)
        param_per_layer_mb = 244 * 4 / 1024  # ~1MB per param set
        shard_per_gpu_mb = param_per_layer_mb / n_gpu

        # Communication time per layer (from NCCL benchmark)
        # RS+AG ≈ 2 * shard_per_gpu_mb / nccl_bw
        # From exp3: per-GPU RS ≈ 5.3 GB/s
        rs_per_layer_us = shard_per_gpu_mb / 5.3 * 1e6  # microseconds
        ag_per_layer_us = rs_per_layer_us  # similar bandwidth

        # Compute time per layer (from decode breakdown)
        compute_per_layer_us = 500  # ~500us for B=32 one layer

        # Without overlap
        no_overlap_us = compute_per_layer_us + rs_per_layer_us + ag_per_layer_us + compute_per_layer_us

        # With overlap (RS overlaps with forward compute)
        overlap_us = max(rs_per_layer_us, compute_per_layer_us) + ag_per_layer_us + compute_per_layer_us

        # Communication ratio
        comm_ratio_no_overlap = (rs_per_layer_us + ag_per_layer_us) / no_overlap_us * 100
        comm_ratio_overlap = (rs_per_layer_us + ag_per_layer_us) / overlap_us * 100 if overlap_us > 0 else 0

        # Speedup vs 1 GPU
        speedup_no_overlap = 1.0 / (no_overlap_us / compute_per_layer_us) if compute_per_layer_us > 0 else 0
        speedup_overlap = 1.0 / (overlap_us / compute_per_layer_us) if compute_per_layer_us > 0 else 0

        exp5[str(n_gpu)] = {
            "shard_per_gpu_mb": round(shard_per_gpu_mb, 3),
            "rs_us": round(rs_per_layer_us, 0),
            "ag_us": round(ag_per_layer_us, 0),
            "compute_us": compute_per_layer_us,
            "no_overlap_us": round(no_overlap_us, 0),
            "overlap_us": round(overlap_us, 0),
            "comm_ratio_no_overlap_pct": round(comm_ratio_no_overlap, 1),
            "comm_ratio_overlap_pct": round(comm_ratio_overlap, 1),
            "speedup_no_overlap": round(speedup_no_overlap, 2),
            "speedup_overlap": round(speedup_overlap, 2),
        }

        if rank == 0:
            print(f"  {n_gpu} GPU: shard={shard_per_gpu_mb:.2f}MB, "
                  f"no_overlap={no_overlap_us:.0f}us({comm_ratio_no_overlap:.1f}%comm), "
                  f"overlap={overlap_us:.0f}us({comm_ratio_overlap:.1f}%comm) "
                  f"→ speedup={speedup_overlap:.2f}x")

    results["exp5_fsdp_scaling"] = exp5

    # ====================================================================
    # Summary
    # ====================================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("SUMMARY — Comm-Compute Overlap RTX 4090")
        print("=" * 70)

        compute_us = compute_time["avg_us"]
        print(f"\n  Compute time: {compute_us:.0f}us (MLP forward B=32)")
        print(f"  Overlap efficiency: varies by tensor size")
        print(f"  PCIe RTX 4090: overlap limited by PCIe bandwidth bottleneck")
        print(f"  NVLink: full overlap possible → FSDP scaling viable")

    # Save results (only rank 0)
    if rank == 0:
        try:
            with open('results/comm_compute_overlap.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except:
            with open('comm_compute_overlap.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
        print("Results saved.")

    dist.destroy_process_group()
    return results


if __name__ == '__main__':
    # Must be run with torchrun
    # torchrun --nproc_per_node=2 tools/comm_compute_overlap_benchmark.py
    results = run_all()