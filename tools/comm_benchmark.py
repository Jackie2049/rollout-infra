"""
PyTorch 分布式通信 Benchmark — CPU 版本

测量集合通信操作在不同数据规模下的延迟。
GPU 到位后可改为 CUDA 后端获取真实 GPU 通信性能。

用法:
    torchrun --nproc_per_node=4 tools/comm_benchmark.py
    torchrun --nproc_per_node=2 tools/comm_benchmark.py --backend gloo
"""

import argparse
import os
import time

import torch
import torch.distributed as dist


def benchmark_op(op_fn, tensor, warmup=5, num_iters=20):
    """测量一个通信操作的延迟"""
    # Warmup
    for _ in range(warmup):
        op_fn(tensor)
        dist.barrier()

    # Benchmark
    times = []
    for _ in range(num_iters):
        dist.barrier()
        start = time.perf_counter()
        op_fn(tensor)
        dist.barrier()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    return avg_ms, min_ms, max_ms


def run_allreduce_benchmark(rank, world_size):
    """AllReduce benchmark"""
    if rank == 0:
        print("\n" + "=" * 70)
        print("AllReduce Benchmark")
        print("=" * 70)
        print(f"  {'Size (KB)':>12} {'Elements':>12} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'BW (GB/s)':>10}")
        print("-" * 70)

    sizes_kb = [1, 4, 16, 64, 256, 1024, 4096, 16384]
    results = []

    for size_kb in sizes_kb:
        num_elements = size_kb * 1024 // 4  # float32
        tensor = torch.randn(num_elements, dtype=torch.float32)

        def op(t):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        avg_ms, min_ms, max_ms = benchmark_op(op, tensor)
        data_gb = num_elements * 4 / (1024**3)
        # AllReduce 通信量 = 2 * (N-1)/N * data_size
        comm_gb = 2 * (world_size - 1) / world_size * data_gb
        bandwidth = comm_gb / (avg_ms / 1000) if avg_ms > 0 else 0

        results.append((size_kb, num_elements, avg_ms, min_ms, max_ms, bandwidth))

        if rank == 0:
            print(f"  {size_kb:>10} KB {num_elements:>12,} {avg_ms:>10.3f} {min_ms:>10.3f} {max_ms:>10.3f} {bandwidth:>10.3f}")

    return results


def run_reduce_scatter_benchmark(rank, world_size):
    """ReduceScatter benchmark"""
    if rank == 0:
        print("\n" + "=" * 70)
        print("ReduceScatter Benchmark")
        print("=" * 70)
        print(f"  {'Size (KB)':>12} {'Elements':>12} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
        print("-" * 70)

    sizes_kb = [1, 4, 16, 64, 256, 1024, 4096]

    for size_kb in sizes_kb:
        # Total elements must be divisible by world_size
        num_elements = (size_kb * 1024 // 4 // world_size) * world_size
        tensor = torch.randn(num_elements, dtype=torch.float32)
        output = torch.randn(num_elements // world_size, dtype=torch.float32)

        def op(t, out=output):
            dist.reduce_scatter_tensor(out, t)

        avg_ms, min_ms, max_ms = benchmark_op(op, tensor)

        if rank == 0:
            print(f"  {size_kb:>10} KB {num_elements:>12,} {avg_ms:>10.3f} {min_ms:>10.3f} {max_ms:>10.3f}")


def run_allgather_benchmark(rank, world_size):
    """AllGather benchmark"""
    if rank == 0:
        print("\n" + "=" * 70)
        print("AllGather Benchmark")
        print("=" * 70)
        print(f"  {'Size (KB)':>12} {'Elements':>12} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
        print("-" * 70)

    sizes_kb = [1, 4, 16, 64, 256, 1024, 4096]

    for size_kb in sizes_kb:
        chunk_elements = size_kb * 1024 // 4
        input_tensor = torch.randn(chunk_elements, dtype=torch.float32)
        output_tensor = torch.randn(chunk_elements * world_size, dtype=torch.float32)

        def op(inp=input_tensor, out=output_tensor):
            dist.all_gather_into_tensor(out, inp)

        avg_ms, min_ms, max_ms = benchmark_op(op, input_tensor)

        if rank == 0:
            print(f"  {size_kb:>10} KB {chunk_elements:>12,} {avg_ms:>10.3f} {min_ms:>10.3f} {max_ms:>10.3f}")


def run_broadcast_benchmark(rank, world_size):
    """Broadcast benchmark"""
    if rank == 0:
        print("\n" + "=" * 70)
        print("Broadcast Benchmark")
        print("=" * 70)
        print(f"  {'Size (KB)':>12} {'Elements':>12} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
        print("-" * 70)

    sizes_kb = [1, 4, 16, 64, 256, 1024, 4096]

    for size_kb in sizes_kb:
        num_elements = size_kb * 1024 // 4
        tensor = torch.randn(num_elements, dtype=torch.float32)

        def op(t):
            dist.broadcast(t, src=0)

        avg_ms, min_ms, max_ms = benchmark_op(op, tensor)

        if rank == 0:
            print(f"  {size_kb:>10} KB {num_elements:>12,} {avg_ms:>10.3f} {min_ms:>10.3f} {max_ms:>10.3f}")


def main():
    parser = argparse.ArgumentParser(description="Communication Benchmark")
    parser.add_argument("--backend", default="gloo", help="通信后端 (gloo/nccl)")
    parser.add_argument("--warmup", type=int, default=5, help="预热迭代数")
    parser.add_argument("--iters", type=int, default=20, help="测试迭代数")
    args = parser.parse_args()

    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print("=" * 70)
        print("PyTorch 分布式通信 Benchmark")
        print("=" * 70)
        print(f"  Backend: {args.backend}")
        print(f"  World size: {world_size}")
        print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"  Warmup: {args.warmup}, Iters: {args.iters}")

    run_allreduce_benchmark(rank, world_size)
    run_reduce_scatter_benchmark(rank, world_size)
    run_allgather_benchmark(rank, world_size)
    run_broadcast_benchmark(rank, world_size)

    if rank == 0:
        print("\n" + "=" * 70)
        print("Benchmark 完成!")
        print("=" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
