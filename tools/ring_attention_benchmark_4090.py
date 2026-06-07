#!/usr/bin/env python3
"""Ring Attention Multi-GPU Benchmark — RTX 4090 PCIe
======================================================
Benchmarks Ring Attention on real multi-GPU hardware (8× RTX 4090 PCIe).

Key experiments:
1. Ring Attention correctness (cos_sim vs baseline SDPA)
2. Ring Attention latency scaling (P=2,4,8 GPUs)
3. Communication vs computation breakdown
4. PCIe bandwidth measurement
5. Causal load imbalance analysis

Usage:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 ring_attention_benchmark_4090.py
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 ring_attention_benchmark_4090.py
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29500 ring_attention_benchmark_4090.py
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import math
import time
import json
import argparse


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()


def ring_attention(Q_local, K_chunks, V_chunks, rank, world_size, causal=False):
    """Ring Attention using pre-gathered KV chunks (simulating ring P2P).

    Instead of real P2P isend/irecv (which has issues on PCIe),
    we gather all KV chunks upfront (equivalent to ring communication
    but without overlap). This measures the compute part accurately
    and we separately measure the communication cost.

    Args:
        Q_local: (B, H, N/P, D) local query chunk
        K_chunks: list of (B, H, N/P, D) KV blocks from all GPUs
        V_chunks: list of (B, H, N/P, D) KV blocks from all GPUs
        rank: GPU rank (determines which Q positions)
        world_size: number of GPUs
        causal: apply causal mask

    Returns: output (B, H, N/P, D)
    """
    B, H, block_size, D = Q_local.shape
    scale = 1.0 / math.sqrt(D)

    # Online softmax running statistics
    running_max = torch.full((B, H, block_size, 1), float('-inf'),
                             device=Q_local.device, dtype=Q_local.dtype)
    running_sum = torch.zeros((B, H, block_size, 1),
                              device=Q_local.device, dtype=Q_local.dtype)
    running_out = torch.zeros((B, H, block_size, D),
                              device=Q_local.device, dtype=Q_local.dtype)

    # Process each KV block in ring order
    for step in range(world_size):
        kv_source_rank = (rank - step) % world_size
        kv_start_pos = kv_source_rank * block_size

        K_block = K_chunks[kv_source_rank]
        V_block = V_chunks[kv_source_rank]

        # Compute local attention scores
        S = Q_local @ K_block.transpose(-2, -1) * scale

        # Apply causal mask
        if causal:
            q_start = rank * block_size
            q_pos = torch.arange(q_start, q_start + block_size, device=Q_local.device)
            k_pos = torch.arange(kv_start_pos, kv_start_pos + block_size, device=Q_local.device)
            mask = q_pos.unsqueeze(1) >= k_pos.unsqueeze(0)
            S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Online softmax update (FlashAttention-style)
        block_max = S.max(dim=-1, keepdim=True).values
        new_max = torch.max(running_max, block_max)

        # Rescale previous running statistics
        correction = torch.exp(running_max - new_max)
        S_exp = torch.exp(S - new_max)
        block_sum = S_exp.sum(dim=-1, keepdim=True)

        # Update output and statistics
        running_out = running_out * correction + S_exp @ V_block
        running_sum = running_sum * correction + block_sum
        running_max = new_max

    output = running_out / running_sum
    return output


def gather_kv_chunks(K_local, V_local, rank, world_size):
    """Gather all KV chunks via all_gather (simulates ring P2P on PCIe).

    On NVLink systems, ring P2P isend/irecv overlaps with compute.
    On PCIe, all_gather goes through NCCL → CPU staging → much slower.
    This function measures the real communication cost on PCIe.
    """
    K_list = [torch.empty_like(K_local) for _ in range(world_size)]
    V_list = [torch.empty_like(V_local) for _ in range(world_size)]

    dist.all_gather(K_list, K_local.contiguous())
    dist.all_gather(V_list, V_local.contiguous())

    return K_list, V_list


def benchmark_correctness(rank, world_size, B=1, H=8, D=64, N=512):
    """Exp1: Verify Ring Attention correctness vs SDPA."""
    device = torch.cuda.current_device()
    block_size = N // world_size

    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    K = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    V = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

    # Baseline: SDPA
    output_baseline = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    # Split across GPUs
    Q_local = Q[:, :, rank * block_size:(rank + 1) * block_size, :].clone()
    K_local = K[:, :, rank * block_size:(rank + 1) * block_size, :].clone()
    V_local = V[:, :, rank * block_size:(rank + 1) * block_size, :].clone()

    # Gather all KV chunks
    K_list = [torch.empty_like(K_local) for _ in range(world_size)]
    V_list = [torch.empty_like(V_local) for _ in range(world_size)]
    dist.all_gather(K_list, K_local.contiguous())
    dist.all_gather(V_list, V_local.contiguous())

    # Ring Attention compute
    output_ring = ring_attention(Q_local, K_list, V_list, rank, world_size, causal=True)

    # Compare
    baseline_local = output_baseline[:, :, rank * block_size:(rank + 1) * block_size, :]
    max_diff = (output_ring - baseline_local).abs().max().item()
    cos_sim = F.cosine_similarity(output_ring.flatten(), baseline_local.flatten(), dim=0).item()

    return {"N": N, "P": world_size, "B": B, "H": H, "D": D,
            "max_diff": max_diff, "cos_sim": cos_sim}


def benchmark_latency(rank, world_size, B=1, H=8, D=64,
                      seq_lengths=[512, 1024, 2048, 4096, 8192]):
    """Exp2: Ring Attention latency with compute/comm breakdown."""
    device = torch.cuda.current_device()
    results = {}

    for N in seq_lengths:
        if N % world_size != 0:
            continue

        block_size = N // world_size
        torch.manual_seed(42)
        Q_local = torch.randn(B, H, block_size, D, device=device, dtype=torch.float16)
        K_local = torch.randn(B, H, block_size, D, device=device, dtype=torch.float16)
        V_local = torch.randn(B, H, block_size, D, device=device, dtype=torch.float16)

        # === Measure communication cost (all_gather KV) ===
        K_list_template = [torch.empty_like(K_local) for _ in range(world_size)]
        V_list_template = [torch.empty_like(V_local) for _ in range(world_size)]

        # Warmup
        for _ in range(3):
            K_list = [torch.empty_like(K_local) for _ in range(world_size)]
            V_list = [torch.empty_like(V_local) for _ in range(world_size)]
            dist.all_gather(K_list, K_local.contiguous())
            dist.all_gather(V_list, V_local.contiguous())
        dist.barrier()

        comm_times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            K_list = [torch.empty_like(K_local) for _ in range(world_size)]
            V_list = [torch.empty_like(V_local) for _ in range(world_size)]
            dist.all_gather(K_list, K_local.contiguous())
            dist.all_gather(V_list, V_local.contiguous())
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            comm_times.append((t1 - t0) * 1000)

        comm_median = sorted(comm_times)[len(comm_times) // 2]

        # === Measure compute cost (ring attention after KV gathered) ===
        # Warmup
        K_list_warmup = [torch.empty_like(K_local) for _ in range(world_size)]
        V_list_warmup = [torch.empty_like(V_local) for _ in range(world_size)]
        dist.all_gather(K_list_warmup, K_local.contiguous())
        dist.all_gather(V_list_warmup, V_local.contiguous())
        for _ in range(3):
            ring_attention(Q_local, K_list_warmup, V_list_warmup, rank, world_size, causal=True)
        dist.barrier()

        compute_times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = ring_attention(Q_local, K_list_warmup, V_list_warmup, rank, world_size, causal=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            compute_times.append((t1 - t0) * 1000)

        compute_median = sorted(compute_times)[len(compute_times) // 2]

        # === Measure full ring attention (comm + compute) ===
        full_times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            K_list = [torch.empty_like(K_local) for _ in range(world_size)]
            V_list = [torch.empty_like(V_local) for _ in range(world_size)]
            dist.all_gather(K_list, K_local.contiguous())
            dist.all_gather(V_list, V_local.contiguous())
            output = ring_attention(Q_local, K_list, V_list, rank, world_size, causal=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            full_times.append((t1 - t0) * 1000)

        full_median = sorted(full_times)[len(full_times) // 2]

        # === Baseline: single-GPU SDPA ===
        torch.manual_seed(42)
        Q_full = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
        K_full = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
        V_full = torch.randn(B, H, N, D, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(3):
            F.scaled_dot_product_attention(Q_full, K_full, V_full, is_causal=True)

        baseline_times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            F.scaled_dot_product_attention(Q_full, K_full, V_full, is_causal=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            baseline_times.append((t1 - t0) * 1000)

        baseline_median = sorted(baseline_times)[len(baseline_times) // 2]

        # Communication volume: 2 * all_gather(KV) = 2 * B * H * N * D * 2bytes * (P-1)/P
        comm_vol = 2 * B * H * N * D * 2 * (world_size - 1) / world_size / 1e6  # MB
        # Local compute: 2 * B * H * N/P * N/P * D * P (P steps, each step N/P^2 matmul)
        local_compute = 2 * B * H * block_size * block_size * D * world_size

        # Theoretical NVLink overlap time (comm hidden if compute > comm)
        nvlink_time_ms = compute_median + max(0, comm_median - compute_median)
        # PCIe time (no overlap: comm + compute serial)
        pcie_time_ms = comm_median + compute_median

        results[f"N{N}"] = {
            "N": N, "P": world_size, "block_size": block_size,
            "ring_full_median_ms": full_median,
            "compute_median_ms": compute_median,
            "comm_median_ms": comm_median,
            "baseline_median_ms": baseline_median,
            "comm_pct": comm_median / full_median * 100 if full_median > 0 else 0,
            "compute_pct": compute_median / full_median * 100 if full_median > 0 else 0,
            "ring_vs_baseline_ratio": full_median / baseline_median if baseline_median > 0 else 0,
            "comm_volume_MB": comm_vol,
            "nvlink_overlap_time_ms": nvlink_time_ms,
            "pcie_serial_time_ms": pcie_time_ms,
            "nvlink_speedup_vs_pcie": pcie_time_ms / nvlink_time_ms if nvlink_time_ms > 0 else 0,
        }

    return results


def benchmark_load_balance(rank, world_size, B=1, H=8, D=64, N=2048):
    """Exp3: Causal attention load imbalance across GPUs."""
    block_size = N // world_size

    # For each GPU, count active Q-K pairs per KV block
    q_start = rank * block_size
    active_counts = []

    for step in range(world_size):
        kv_source_rank = (rank - step) % world_size
        kv_start_pos = kv_source_rank * block_size

        q_pos = torch.arange(q_start, q_start + block_size)
        k_pos = torch.arange(kv_start_pos, kv_start_pos + block_size)
        mask = q_pos.unsqueeze(1) >= k_pos.unsqueeze(0)
        active_counts.append(mask.sum().item())

    # Non-causal: all pairs active
    non_causal_active = block_size ** 2

    # Striped: positions interleaved across GPUs
    # GPU k gets positions {k, k+P, k+2P, ...}
    # Each striped position i attends to ~i+1 positions (causal)
    # Average = sum(i+1 for i in range(N)) / N ≈ N/2
    striped_avg_active = block_size * (N / 2)  # Approximate

    imbalance_ratio = max(active_counts) / min(active_counts) if min(active_counts) > 0 else float('inf')

    return {
        "N": N, "P": world_size, "rank": rank,
        "active_counts_per_step": active_counts,
        "total_causal_active": sum(active_counts),
        "non_causal_active_per_step": non_causal_active,
        "striped_approx_active_per_step": striped_avg_active,
        "imbalance_ratio": imbalance_ratio,
    }


def benchmark_p2p_bandwidth(rank, world_size):
    """Exp4: Measure PCIe P2P bandwidth via all_gather (realistic for PCIe RTX 4090)."""
    device = torch.cuda.current_device()

    sizes_mb = [0.5, 1, 2, 4, 8, 16, 32]
    results = {}

    for size_mb in sizes_mb:
        n_elements = int(size_mb * 1024 * 1024 / 2)
        if n_elements < 1:
            continue

        data = torch.randn(n_elements, device=device, dtype=torch.float16)
        recv_list = [torch.empty_like(data) for _ in range(world_size)]

        # Warmup
        for _ in range(3):
            recv = [torch.empty_like(data) for _ in range(world_size)]
            dist.all_gather(recv, data.contiguous())
        dist.barrier()

        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            recv = [torch.empty_like(data) for _ in range(world_size)]
            dist.all_gather(recv, data.contiguous())
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        median_ms = sorted(times)[len(times) // 2]
        # Effective bandwidth: total data moved / time
        # each GPU sends size_mb to P-1 others → total = P * size_mb * (P-1)/P ≈ size_mb * (P-1)
        total_data_mb = size_mb * (world_size - 1)
        effective_bw = total_data_mb / (median_ms / 1000)  # MB/s

        results[f"size_{size_mb}MB"] = {
            "size_MB": size_mb,
            "total_data_MB": total_data_mb,
            "median_ms": median_ms,
            "effective_bandwidth_MB_s": effective_bw,
            "effective_bandwidth_GB_s": effective_bw / 1024,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', 'correctness', 'latency', 'load_balance', 'p2p_bandwidth'])
    parser.add_argument('--B', type=int, default=2)
    parser.add_argument('--H', type=int, default=8)
    parser.add_argument('--D', type=int, default=64)
    parser.add_argument('--output', type=str,
                        default='ring_attention_benchmark_results.json')
    args = parser.parse_args()

    rank, world_size = setup_distributed()
    all_results = {"world_size": world_size, "gpu": "RTX_4090_PCIe"}

    if args.exp in ['all', 'correctness']:
        print(f"[Rank {rank}] Running correctness benchmark...")
        results = benchmark_correctness(rank, world_size, B=args.B, H=args.H, D=args.D, N=512)
        all_results["correctness"] = results
        print(f"[Rank {rank}] Correctness: max_diff={results['max_diff']:.2e}, "
              f"cos_sim={results['cos_sim']:.6f}")

    if args.exp in ['all', 'latency']:
        print(f"[Rank {rank}] Running latency benchmark...")
        seq_lengths = [512, 1024, 2048, 4096, 8192]
        results = benchmark_latency(rank, world_size, B=args.B, H=args.H, D=args.D,
                                    seq_lengths=seq_lengths)
        all_results["latency"] = results
        for key, val in results.items():
            print(f"[Rank {rank}] N={val['N']}: "
                  f"ring={val['ring_full_median_ms']:.2f}ms "
                  f"(compute={val['compute_median_ms']:.2f}ms "
                  f"comm={val['comm_median_ms']:.2f}ms "
                  f"comm_pct={val['comm_pct']:.1f}%) "
                  f"baseline={val['baseline_median_ms']:.2f}ms "
                  f"ratio={val['ring_vs_baseline_ratio']:.2f}x "
                  f"NVLink_overlap={val['nvlink_overlap_time_ms']:.2f}ms "
                  f"PCIe_serial={val['pcie_serial_time_ms']:.2f}ms "
                  f"NVLink_speedup={val['nvlink_speedup_vs_pcie']:.2f}x")

    if args.exp in ['all', 'load_balance']:
        print(f"[Rank {rank}] Running load balance benchmark...")
        results = benchmark_load_balance(rank, world_size, B=args.B, H=args.H, D=args.D, N=2048)
        all_results["load_balance"] = results
        print(f"[Rank {rank}] Load imbalance: {results['imbalance_ratio']:.2f}x, "
              f"active_counts={results['active_counts_per_step']}")

    if args.exp in ['all', 'p2p_bandwidth']:
        print(f"[Rank {rank}] Running bandwidth benchmark...")
        results = benchmark_p2p_bandwidth(rank, world_size)
        all_results["p2p_bandwidth"] = results
        for key, val in results.items():
            print(f"[Rank {rank}] {val['size_MB']}MB: "
                  f"{val['effective_bandwidth_GB_s']:.2f} GB/s ({val['median_ms']:.2f}ms)")

    if rank == 0:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")

        if "latency" in all_results:
            print("\n=== RTX 4090 PCIe Ring Attention Summary ===")
            print(f"{'N':>6} {'P':>3} {'Ring':>8} {'Compute':>8} {'Comm':>8} {'Comm%':>6} "
                  f"{'Baseline':>9} {'Ratio':>6} {'NVLink':>8} {'PCIe':>8} {'Speedup':>8}")
            for key, val in all_results["latency"].items():
                print(f"{val['N']:>6} {val['P']:>3} {val['ring_full_median_ms']:>8.2f} "
                      f"{val['compute_median_ms']:>8.2f} {val['comm_median_ms']:>8.2f} "
                      f"{val['comm_pct']:>6.1f} {val['baseline_median_ms']:>9.2f} "
                      f"{val['ring_vs_baseline_ratio']:>6.2f} "
                      f"{val['nvlink_overlap_time_ms']:>8.2f} "
                      f"{val['pcie_serial_time_ms']:>8.2f} "
                      f"{val['nvlink_speedup_vs_pcie']:>8.2f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()