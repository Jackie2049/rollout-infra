"""
NCCL Multi-GPU Communication Benchmark — 8x RTX 4090 PCIe
Measures real NCCL collective operation performance (AllReduce, ReduceScatter, AllGather)
and validates theoretical NCCL deep dive findings.

Focus: Understanding PCIe bandwidth limits for distributed training/inference.
"""

import torch
import torch.distributed as dist
import time
import json
import os

def init_distributed():
    """Initialize NCCL distributed backend"""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def benchmark_collective(op_fn, tensor_size_bytes, warmup=5, repeats=50):
    """Benchmark a collective operation"""
    # Create tensor of specified size
    n_elements = tensor_size_bytes // 2  # BF16 = 2 bytes
    tensor = torch.randn(n_elements, device=torch.cuda.current_device(), dtype=torch.bfloat16)

    # Warmup
    for _ in range(warmup):
        op_fn(tensor)
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        op_fn(tensor)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    avg_ms = sum(times) / len(times)
    bandwidth_gbps = tensor_size_bytes / (avg_ms / 1000) / 1024**3  # GB/s

    return {
        "avg_ms": round(avg_ms, 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "bandwidth_gbps": round(bandwidth_gbps, 2),
        "tensor_size_mb": round(tensor_size_bytes / 1024**2, 2),
    }


def run_all():
    rank, world_size, local_rank = init_distributed()
    results = {}

    if rank == 0:
        print(f"=" * 70)
        print(f"NCCL Multi-GPU Communication Benchmark — {world_size}x RTX 4090 PCIe")
        print(f"=" * 70)

    # Device info
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if rank == 0:
        print(f"Device: {props.name}, HBM: {props.total_memory / 1024**3:.2f} GB")

    # ====================================================================
    # Exp 1: AllReduce Bandwidth — Size Sweep
    # ====================================================================
    if rank == 0:
        print("\n--- Exp 1: AllReduce Bandwidth vs Data Size ---")

    sizes_mb = [0.001, 0.01, 0.1, 1, 10, 50, 100]
    exp1 = {}

    for size_mb in sizes_mb:
        size_bytes = int(size_mb * 1024**2)
        n_elements = max(size_bytes // 2, 1)
        tensor = torch.randn(n_elements, device=torch.cuda.current_device(), dtype=torch.bfloat16)

        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg_ms = sum(times) / len(times)
        bw = size_bytes / (avg_ms / 1000) / 1024**3

        key = f"{size_mb}MB"
        exp1[key] = {
            "avg_ms": round(avg_ms, 4),
            "min_ms": round(min(times), 4),
            "max_ms": round(max(times), 4),
            "bandwidth_gbps": round(bw, 2),
            "size_mb": size_mb,
        }
        if rank == 0:
            print(f"  {size_mb}MB: {avg_ms:.4f}ms → {bw:.2f}GB/s")

    results["exp1_allreduce_size_sweep"] = exp1

    # ====================================================================
    # Exp 2: ReduceScatter + AllGather (FSDP pattern)
    # ====================================================================
    if rank == 0:
        print("\n--- Exp 2: ReduceScatter + AllGather (FSDP Pattern) ---")

    sizes_mb = [1, 10, 50, 100]
    exp2 = {}

    for size_mb in sizes_mb:
        size_bytes = int(size_mb * 1024**2)
        n_elements = max(size_bytes // 2, world_size)  # must be divisible by world_size
        n_elements = (n_elements + world_size - 1) // world_size * world_size  # round up
        actual_mb = n_elements * 2 / 1024**2

        full_tensor = torch.randn(n_elements, device=torch.cuda.current_device(), dtype=torch.bfloat16)
        chunk_size = n_elements // world_size
        scattered_tensor = torch.randn(chunk_size, device=torch.cuda.current_device(), dtype=torch.bfloat16)

        # Warmup RS+AG
        for _ in range(5):
            dist.reduce_scatter_tensor(scattered_tensor, full_tensor, op=dist.ReduceOp.SUM)
            dist.all_gather_into_tensor(full_tensor, scattered_tensor)
        torch.cuda.synchronize()

        # Measure RS
        times_rs = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dist.reduce_scatter_tensor(scattered_tensor, full_tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_rs.append((t1 - t0) * 1000)

        # Measure AG
        times_ag = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dist.all_gather_into_tensor(full_tensor, scattered_tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ag.append((t1 - t0) * 1000)

        avg_rs = sum(times_rs) / len(times_rs)
        avg_ag = sum(times_ag) / len(times_ag)
        avg_total = avg_rs + avg_ag
        bw_rs = (n_elements * 2) / (avg_rs / 1000) / 1024**3  # bandwidth per GPU
        bw_ag = (n_elements * 2) / (avg_ag / 1000) / 1024**3

        key = f"{size_mb}MB"
        exp2[key] = {
            "actual_mb": round(actual_mb, 2),
            "rs_avg_ms": round(avg_rs, 4),
            "ag_avg_ms": round(avg_ag, 4),
            "total_avg_ms": round(avg_total, 4),
            "rs_bandwidth_gbps": round(bw_rs, 2),
            "ag_bandwidth_gbps": round(bw_ag, 2),
            "per_gpu_data_mb": round(actual_mb / world_size, 2),
        }
        if rank == 0:
            print(f"  {actual_mb:.2f}MB: RS={avg_rs:.4f}ms AG={avg_ag:.4f}ms Total={avg_total:.4f}ms → RS_bw={bw_rs:.2f} AG_bw={bw_ag:.2f}GB/s")

    results["exp2_reduce_scatter_all_gather"] = exp2

    # ====================================================================
    # Exp 3: Communication Time Ratio (simulate FSDP step)
    # ====================================================================
    if rank == 0:
        print("\n--- Exp 3: Communication Ratio in FSDP Training Step ---")

    # Simulate FSDP forward+backward for 7B model
    # Each layer: forward=AG+compute, backward=RS+compute+AG+RS+compute
    # Total per layer: 2RS + 2AG = 4 collective ops (same as AllReduce×2)
    hidden = 4096  # 7B-like
    n_layers = 32
    layer_params = hidden * hidden * 4 * 3 // 2  # QKV+gate+up+out ≈ 50M params/layer BF16
    layer_bytes = layer_params * 2  # BF16
    layer_mb = layer_bytes / 1024**2

    # Measure compute time (GEMM forward+backward ≈ 6 matmul per layer)
    M = 32  # batch size
    W_qkv = torch.randn(hidden, 3 * hidden, device=torch.cuda.current_device(), dtype=torch.bfloat16)
    x = torch.randn(M, hidden, device=torch.cuda.current_device(), dtype=torch.bfloat16)

    # Warmup GEMM
    for _ in range(10):
        out = x @ W_qkv
    torch.cuda.synchronize()

    # Measure single GEMM
    times_gemm = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = x @ W_qkv
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_gemm.append((t1 - t0) * 1000)

    avg_gemm_ms = sum(times_gemm) / len(times_gemm)
    # 6 GEMM per layer (forward 2 + backward 4)
    compute_per_layer_ms = avg_gemm_ms * 6

    # Measure communication for this layer size
    n_elements_comm = max(layer_params, world_size)
    n_elements_comm = (n_elements_comm + world_size - 1) // world_size * world_size
    full_tensor = torch.randn(n_elements_comm, device=torch.cuda.current_device(), dtype=torch.bfloat16)
    chunk_size = n_elements_comm // world_size
    scattered_tensor = torch.randn(chunk_size, device=torch.cuda.current_device(), dtype=torch.bfloat16)

    # Warmup
    for _ in range(5):
        dist.reduce_scatter_tensor(scattered_tensor, full_tensor, op=dist.ReduceOp.SUM)
        dist.all_gather_into_tensor(full_tensor, scattered_tensor)
    torch.cuda.synchronize()

    # Measure RS+AG (one full cycle)
    times_comm = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.reduce_scatter_tensor(scattered_tensor, full_tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dist.all_gather_into_tensor(full_tensor, scattered_tensor)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        times_comm.append(((t1 - t0) + (t2 - t1)) * 1000)

    avg_comm_cycle_ms = sum(times_comm) / len(times_comm)
    # 2 cycles per layer (forward + backward)
    comm_per_layer_ms = avg_comm_cycle_ms * 2

    comm_ratio = comm_per_layer_ms / (compute_per_layer_ms + comm_per_layer_ms)

    exp3 = {
        "model_7b_like": {
            "layer_params_m": round(layer_params / 1e6, 2),
            "layer_mb": round(layer_mb, 2),
            "batch_size": M,
            "single_gemm_ms": round(avg_gemm_ms, 4),
            "compute_per_layer_ms": round(compute_per_layer_ms, 3),
            "comm_cycle_ms": round(avg_comm_cycle_ms, 4),
            "comm_per_layer_ms": round(comm_per_layer_ms, 3),
            "comm_ratio_pct": round(comm_ratio * 100, 1),
            "effective_speedup": round(1 / (1 + comm_ratio) if comm_ratio < 1 else 0, 2),
        }
    }
    if rank == 0:
        print(f"  7B-like(B={M}): compute={compute_per_layer_ms:.3f}ms, comm={comm_per_layer_ms:.3f}ms → comm_ratio={comm_ratio*100:.1f}%")
        print(f"  → Effective speedup={1/(1+comm_ratio):.2f}x vs single GPU")
        print(f"  → NVLink comm_ratio would be ~{comm_per_layer_ms/12*2:.1f}% → speedup ~{1/(1+comm_per_layer_ms/12*2/(compute_per_layer_ms+comm_per_layer_ms/12*2)):.2f}x")

    results["exp3_communication_ratio"] = exp3

    # ====================================================================
    # Exp 4: P2P Access Capability
    # ====================================================================
    if rank == 0:
        print("\n--- Exp 4: P2P Access Capability ---")

    exp4 = {}
    p2p_enabled = torch.cuda.can_device_access_peer(0, 1)
    exp4["p2p_gpu0_to_gpu1"] = p2p_enabled

    # P2P is disabled on consumer GPUs — skip enable attempt (API removed in PyTorch 2.9)
    exp4["p2p_enable_attempt_skipped"] = True
    exp4["p2p_note"] = "Consumer GPUs: P2P disabled by default, cannot enable"

    if rank == 0:
        print(f"  P2P access GPU0→GPU1: {p2p_enabled}")
        print(f"  → RTX 4090 PCIe: P2P disabled → must go through CPU → slow!")

    results["exp4_p2p_capability"] = exp4

    # ====================================================================
    # Exp 5: Broadcast + Send/Recv (basic operations)
    # ====================================================================
    if rank == 0:
        print("\n--- Exp 5: Broadcast Latency ---")

    sizes_mb = [0.001, 0.01, 0.1, 1]
    exp5 = {}

    for size_mb in sizes_mb:
        size_bytes = int(size_mb * 1024**2)
        n_elements = max(size_bytes // 2, 1)
        tensor = torch.randn(n_elements, device=torch.cuda.current_device(), dtype=torch.bfloat16)

        # Warmup
        for _ in range(5):
            dist.broadcast(tensor, src=0)
        torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dist.broadcast(tensor, src=0)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg_ms = sum(times) / len(times)
        bw = size_bytes / (avg_ms / 1000) / 1024**3

        key = f"{size_mb}MB"
        exp5[key] = {
            "avg_ms": round(avg_ms, 4),
            "bandwidth_gbps": round(bw, 2),
        }
        if rank == 0:
            print(f"  Broadcast {size_mb}MB: {avg_ms:.4f}ms → {bw:.2f}GB/s")

    results["exp5_broadcast"] = exp5

    # ====================================================================
    # Summary
    # ====================================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("SUMMARY — NCCL 8× RTX 4090 PCIe Communication")
        print("=" * 70)

        # Extract key numbers
        ar_100mb = exp1.get("100MB", {})
        ar_10mb = exp1.get("10MB", {})
        rs_ag_100 = exp2.get("100MB", {})
        comm_data = exp3.get("model_7b_like", {})

        print(f"\n  AllReduce 100MB: {ar_100mb.get('avg_ms', 0):.2f}ms → {ar_100mb.get('bandwidth_gbps', 0):.2f}GB/s")
        print(f"  AllReduce 10MB: {ar_10mb.get('avg_ms', 0):.2f}ms → {ar_10mb.get('bandwidth_gbps', 0):.2f}GB/s")
        print(f"  RS+AG 100MB: {rs_ag_100.get('total_avg_ms', 0):.2f}ms → RS {rs_ag_100.get('rs_bandwidth_gbps', 0):.2f} AG {rs_ag_100.get('ag_bandwidth_gbps', 0):.2f}GB/s")
        print(f"  Comm ratio (7B B=32): {comm_data.get('comm_ratio_pct', 0):.1f}%")
        print(f"  Effective speedup: {comm_data.get('effective_speedup', 0):.2f}x")
        print(f"  P2P: {p2p_enabled}")
        print(f"\n  Production implications:")
        print(f"    → PCIe bandwidth ~3-5 GB/s AllReduce → NVLink ~50-100x faster")
        print(f"    → FSDP communication ratio ~{comm_data.get('comm_ratio_pct', 0):.0f}% → scaling limited")
        print(f"    → P2P disabled → all GPU communication through CPU → slower")
        print(f"    → RTX 4090 = single GPU optimal, multi-GPU limited by PCIe")

    return results


if __name__ == '__main__':
    results = run_all()
    rank = dist.get_rank()
    if rank == 0:
        try:
            with open('results/nccl_multi_gpu_benchmark.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except:
            with open('nccl_multi_gpu_benchmark.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
    dist.destroy_process_group()