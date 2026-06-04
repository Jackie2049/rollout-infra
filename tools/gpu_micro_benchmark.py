#!/usr/bin/env python3
"""GPU Micro-Benchmark + Profiling 工具

使用 CUDA Events 精确测量 GPU 操作时间:
1. GEMM (compute-bound): 不同 M/N/K 的 TFLOPS
2. Memory Copy (memory-bound): HBM 带宽
3. Attention: Prefill vs Decode 性能对比
4. Kernel Launch Overhead
5. Batch Size 对 throughput 的影响 (实测 memory-bound 特征)

用法 (GPU 服务器):
  source /root/miniconda3/bin/activate myconda
  export HF_HUB_OFFLINE=1
  python gpu_micro_benchmark.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import time
import torch
import torch.nn.functional as F
from collections import OrderedDict

GPU_NAME = torch.cuda.get_device_name(0)
print(f"GPU: {GPU_NAME}")
print(f"CUDA: {torch.version.cuda}, Compute: {torch.cuda.get_device_capability()}")


# ============================================================
# 精确计时工具
# ============================================================

class CUDATimer:
    """使用 CUDA Events 的精确计时器"""
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()

    @property
    def ms(self):
        return self.start.elapsed_time(self.end)


def bench_op(fn, warmup=10, rep=100):
    """Benchmark an operation using CUDA events"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / rep  # ms


# ============================================================
# 实验 1: GEMM 性能 (compute-bound)
# ============================================================

def exp1_gemm():
    print("\n" + "=" * 60)
    print("实验1: GEMM 性能 (Compute-Bound)")
    print("=" * 60)

    results = []
    print(f"\n{'M=N=K':<12} {'时间(ms)':<12} {'TFLOPS':<10} {'%Peak':<8} {'Bound'}")
    print("-" * 55)

    peak_tflops = 14.7  # A16 FP16 Tensor Core

    for size in [128, 256, 512, 1024, 2048, 4096]:
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)

        ms = bench_op(lambda: torch.matmul(a, b))
        flops = 2.0 * size * size * size
        tflops = flops / ms / 1e9
        pct = tflops / peak_tflops * 100

        bound = "compute" if pct > 30 else "overhead"
        results.append({"size": size, "ms": round(ms, 4), "tflops": round(tflops, 2)})
        print(f"{size:<12} {ms:<12.4f} {tflops:<10.2f} {pct:<8.1f}% {bound}")

    return results


# ============================================================
# 实验 2: Memory Bandwidth (memory-bound)
# ============================================================

def exp2_memory_bandwidth():
    print("\n" + "=" * 60)
    print("实验2: Memory Bandwidth (Memory-Bound)")
    print("=" * 60)

    results = []
    print(f"\n{'操作':<20} {'数据量(MB)':<12} {'时间(ms)':<12} {'带宽(GB/s)':<12}")
    print("-" * 56)

    n = 1 << 24  # 16M elements = 64MB for fp32

    # Read bandwidth (sum)
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    ms = bench_op(lambda: x.sum())
    bw = n * 4 / ms / 1e6  # read only
    results.append({"op": "sum (read)", "bw_gbs": round(bw, 1)})
    print(f"{'sum (read)':<20} {n*4/1e6:<12.0f} {ms:<12.3f} {bw:<12.1f}")

    # Write bandwidth (zero)
    ms = bench_op(lambda: torch.zeros(n, device="cuda", dtype=torch.float32))
    bw = n * 4 / ms / 1e6  # write only
    results.append({"op": "zeros (write)", "bw_gbs": round(bw, 1)})
    print(f"{'zeros (write)':<20} {n*4/1e6:<12.0f} {ms:<12.3f} {bw:<12.1f}")

    # Copy bandwidth (read+write)
    ms = bench_op(lambda: x.clone())
    bw = 2 * n * 4 / ms / 1e6  # read + write
    results.append({"op": "clone (R+W)", "bw_gbs": round(bw, 1)})
    print(f"{'clone (R+W)':<20} {n*4/1e6:<12.0f} {ms:<12.3f} {bw:<12.1f}")

    # Element-wise (R+R+W)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    ms = bench_op(lambda: x + y)
    bw = 3 * n * 4 / ms / 1e6
    results.append({"op": "add (R+R+W)", "bw_gbs": round(bw, 1)})
    print(f"{'add (R+R+W)':<20} {n*4/1e6:<12.0f} {ms:<12.3f} {bw:<12.1f}")

    return results


# ============================================================
# 实验 3: Attention 性能 (Prefill vs Decode)
# ============================================================

def exp3_attention():
    print("\n" + "=" * 60)
    print("实验3: Attention — Prefill vs Decode")
    print("=" * 60)

    results = []
    num_heads = 8
    head_dim = 64
    batch = 1
    dtype = torch.float16

    print(f"\nnum_heads={num_heads}, head_dim={head_dim}, batch={batch}")

    # Prefill: full attention (compute-bound for long seq)
    print(f"\n--- Prefill (full SDPA) ---")
    print(f"{'seq_len':<12} {'时间(ms)':<12} {'TFLOPS':<10} {'compute/seq':<12}")
    print("-" * 48)

    for seq_len in [128, 256, 512, 1024, 2048]:
        q = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)

        ms = bench_op(lambda: F.scaled_dot_product_attention(q, k, v))
        # FLOPS: 4 * B * H * S^2 * D (QK^T + softmax * V)
        flops = 4 * batch * num_heads * seq_len * seq_len * head_dim
        tflops = flops / ms / 1e9
        per_seq = ms / seq_len * 1000  # us per seq_len token

        results.append({"type": "prefill", "seq": seq_len, "ms": round(ms, 4), "tflops": round(tflops, 2)})
        print(f"{seq_len:<12} {ms:<12.3f} {tflops:<10.2f} {per_seq:<12.1f} us")

    # Decode: single token attention (memory-bound)
    print(f"\n--- Decode (KV cache read, q_len=1) ---")
    print(f"{'kv_len':<12} {'时间(ms)':<12} {'带宽(GB/s)':<12} {'note'}")
    print("-" * 48)

    for kv_len in [128, 256, 512, 1024, 2048]:
        q = torch.randn(batch, num_heads, 1, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, num_heads, kv_len, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(batch, num_heads, kv_len, head_dim, device="cuda", dtype=dtype)

        ms = bench_op(lambda: F.scaled_dot_product_attention(q, k, v))
        # Memory: read K+V = 2 * B * H * kv_len * D * 2 bytes
        mem_bytes = 2 * batch * num_heads * kv_len * head_dim * 2
        bw = mem_bytes / ms / 1e6

        results.append({"type": "decode", "kv_len": kv_len, "ms": round(ms, 4), "bw_gbs": round(bw, 1)})
        print(f"{kv_len:<12} {ms:<12.3f} {bw:<12.1f} {'memory-bound' if bw < 100 else ''}")

    return results


# ============================================================
# 实验 4: Kernel Launch Overhead
# ============================================================

def exp4_kernel_overhead():
    print("\n" + "=" * 60)
    print("实验4: Kernel Launch Overhead")
    print("=" * 60)

    results = []
    n = 1  # minimal work

    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")

    # Measure many launches
    reps = 1000
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(reps):
        torch.add(x, y)
    end.record()
    torch.cuda.synchronize()

    us_per_launch = start.elapsed_time(end) / reps * 1000  # us

    print(f"\n  Empty kernel (n=1): {us_per_launch:.1f} us/launch")
    results.append({"overhead_us": round(us_per_launch, 1)})

    # Compare with actual work
    for size in [1, 100, 1000, 10000, 100000]:
        a = torch.randn(size, device="cuda")
        b = torch.randn(size, device="cuda")
        ms = bench_op(lambda: torch.add(a, b), warmup=5, rep=1000)
        us = ms * 1000
        useful_us = size * 4 / 170 / 1000  # estimated at 170 GB/s
        overhead_pct = max(0, (us - useful_us) / us * 100) if us > 0 else 0
        print(f"  n={size:>6d}: {us:.1f} us total, ~{useful_us:.1f} us useful, overhead={overhead_pct:.0f}%")

    return results


# ============================================================
# 实验 5: Batch Size 对 Decode Throughput 的影响
# ============================================================

def exp5_batch_decode():
    print("\n" + "=" * 60)
    print("实验5: Batch Decode Throughput (memory-bound 验证)")
    print("=" * 60)

    results = []
    num_heads = 8
    head_dim = 64
    kv_len = 512
    dtype = torch.float16

    print(f"\nnum_heads={num_heads}, head_dim={head_dim}, kv_len={kv_len}")
    print(f"\n{'batch':<8} {'时间(ms)':<12} {'吞吐(tok/s)':<14} {'KV读取(GB/s)':<14} {'带宽效率'}")
    print("-" * 68)

    for batch in [1, 2, 4, 8, 16, 32, 64]:
        q = torch.randn(batch, num_heads, 1, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, num_heads, kv_len, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(batch, num_heads, kv_len, head_dim, device="cuda", dtype=dtype)

        ms = bench_op(lambda: F.scaled_dot_product_attention(q, k, v))

        total_tokens = batch * kv_len
        throughput = total_tokens / ms * 1000

        # KV read: 2 * B * H * kv_len * D * 2 bytes
        kv_bytes = 2 * batch * num_heads * kv_len * head_dim * 2
        kv_bw = kv_bytes / ms / 1e6

        # 带宽效率 (相对实测峰值 170 GB/s)
        eff = kv_bw / 170 * 100

        results.append({
            "batch": batch, "ms": round(ms, 3),
            "throughput": round(throughput, 0), "kv_bw": round(kv_bw, 1),
            "efficiency": round(eff, 1),
        })
        print(f"{batch:<8} {ms:<12.3f} {throughput:<14.0f} {kv_bw:<14.1f} {eff:.0f}%")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GPU Micro-Benchmark + Profiling")
    print("=" * 60)

    all_results = OrderedDict()

    all_results["gemm"] = exp1_gemm()
    all_results["memory_bw"] = exp2_memory_bandwidth()
    all_results["attention"] = exp3_attention()
    all_results["kernel_overhead"] = exp4_kernel_overhead()
    all_results["batch_decode"] = exp5_batch_decode()

    print("\n" + "=" * 60)
    print("关键洞察")
    print("=" * 60)
    print("""
  1. GEMM: 大矩阵接近峰值 TFLOPS, 小矩阵受 kernel launch 影响
  2. Memory: add/clone 带宽 ~170 GB/s, sum (read-only) 更高
  3. Attention:
     - Prefill compute-bound: TFLOPS 随 seq_len 增长
     - Decode memory-bound: 吞吐取决于 KV cache 读取带宽
  4. Kernel Launch: ~5-10 us/launch, 对小操作影响大
  5. Batch Decode: 批量 attention 线性扩展到 batch~32, 之后饱和
""")

    with open("/root/gpu_micro_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Results saved to gpu_micro_benchmark_results.json")
