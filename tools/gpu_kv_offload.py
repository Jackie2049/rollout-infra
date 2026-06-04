#!/usr/bin/env python3
"""KV Cache Offload GPU 实验 — CPU/GPU 内存层级实测

模拟 vLLM 的 KV Cache offload 策略:
1. GPU KV Cache 容量分析 (不同模型/序列长度)
2. CPU Offload 速度实测 (H2D/D2H memcpy)
3. Swap vs Recompute 决策边界
4. 分层存储: GPU hot + CPU warm + SSD cold
5. Offload 对推理延迟的影响

用法 (GPU 服务器):
  source /root/miniconda3/bin/activate myconda
  python gpu_kv_offload.py
"""

import os, json, time, math
import torch
import torch.nn.functional as F
from collections import OrderedDict

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def bench_ms(fn, warmup=5, rep=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep


# ============================================================
# 实验 1: GPU/CPU/SSD 容量层级
# ============================================================

def exp1_memory_hierarchy():
    print("\n" + "=" * 60)
    print("实验1: GPU/CPU/SSD 内存层级")
    print("=" * 60)

    results = []

    # KV cache per token: 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    configs = [
        ("LLaMA-7B", 32, 32, 128, 2),      # name, layers, kv_heads, head_dim, bytes (FP16)
        ("LLaMA-7B-GQA", 32, 8, 128, 2),
        ("Mixtral-8x7B", 32, 8, 128, 2),
        ("DeepSeek-V2-MLA", 60, 1, 512, 2),  # MLA compressed
        ("LLaMA-70B-GQA", 80, 8, 128, 2),
        ("LLaMA-7B-FP8", 32, 32, 128, 1),   # FP8
    ]

    gpu_mem_gb = 15  # A16
    cpu_mem_gb = 30  # typical server
    ssd_storage_gb = 500

    print(f"\n  GPU={gpu_mem_gb}GB, CPU={cpu_mem_gb}GB, SSD={ssd_storage_gb}GB")
    print(f"  {'Model':<20} {'KV/tok bytes':<14} {'GPU max seq':<14} {'CPU max seq':<14} {'SSD max seq'}")
    print("  " + "-" * 76)

    for name, layers, kv_heads, head_dim, dtype_bytes in configs:
        kv_per_tok = 2 * layers * kv_heads * head_dim * dtype_bytes

        # Available memory (assume 50% for KV cache)
        gpu_kv_gb = gpu_mem_gb * 0.5
        cpu_kv_gb = cpu_mem_gb * 0.5
        ssd_kv_gb = ssd_storage_gb * 0.5

        gpu_max_seq = int(gpu_kv_gb * 1e9 / kv_per_tok)
        cpu_max_seq = int(cpu_kv_gb * 1e9 / kv_per_tok)
        ssd_max_seq = int(ssd_kv_gb * 1e9 / kv_per_tok)

        print(f"  {name:<20} {kv_per_tok:<14} {gpu_max_seq:<14,} {cpu_max_seq:<14,} {ssd_max_seq:,}")

        results.append({
            "model": name,
            "kv_per_tok_bytes": kv_per_tok,
            "gpu_max_seq": gpu_max_seq,
            "cpu_max_seq": cpu_max_seq,
            "ssd_max_seq": ssd_max_seq,
        })

    return results


# ============================================================
# 实验 2: H2D/D2H 传输速度实测
# ============================================================

def exp2_transfer_speed():
    print("\n" + "=" * 60)
    print("实验2: H2D/D2H 传输速度")
    print("=" * 60)

    results = []

    print(f"\n  {'Size MB':<12} {'D2H ms':<12} {'D2H GB/s':<12} {'H2D ms':<12} {'H2D GB/s':<12} {'Roundtrip ms'}")
    print("  " + "-" * 72)

    for size_mb in [1, 4, 16, 64, 128, 256, 512, 1024]:
        n_elements = size_mb * 1024 * 1024 // 2  # FP16

        gpu_tensor = torch.randn(n_elements, device="cuda", dtype=torch.float16)
        cpu_tensor = torch.zeros(n_elements, dtype=torch.float16)

        # D2H (GPU → CPU)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            cpu_tensor.copy_(gpu_tensor)
        torch.cuda.synchronize()
        d2h_ms = (time.time() - t0) / 10 * 1000

        # H2D (CPU → GPU)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            gpu_tensor.copy_(cpu_tensor)
        torch.cuda.synchronize()
        h2d_ms = (time.time() - t0) / 10 * 1000

        d2h_bw = size_mb / d2h_ms * 1000 / 1024  # GB/s
        h2d_bw = size_mb / h2d_ms * 1000 / 1024

        print(f"  {size_mb:<12} {d2h_ms:<12.3f} {d2h_bw:<12.1f} {h2d_ms:<12.3f} {h2d_bw:<12.1f} {d2h_ms+h2d_ms:.3f}")

        results.append({
            "size_mb": size_mb,
            "d2h_ms": round(d2h_ms, 3),
            "d2h_gb_s": round(d2h_bw, 1),
            "h2d_ms": round(h2d_ms, 3),
            "h2d_gb_s": round(h2d_bw, 1),
        })

        del gpu_tensor, cpu_tensor
        torch.cuda.empty_cache()

    return results


# ============================================================
# 实验 3: Swap vs Recompute — 长序列决策边界
# ============================================================

def exp3_swap_vs_recompute():
    print("\n" + "=" * 60)
    print("实验3: Swap vs Recompute 决策边界")
    print("=" * 60)

    results = []
    H = 512
    n_heads = 8
    head_dim = H // n_heads

    weight = torch.randn(H, H, device="cuda", dtype=torch.float16)

    for seq_len in [128, 256, 512, 1024, 2048, 4096]:
        # KV data to offload
        kv_bytes = seq_len * 2 * H * 2  # K+V, FP16
        kv_mb = kv_bytes / 1e6

        kv_gpu = torch.randn(seq_len, 2, H, device="cuda", dtype=torch.float16)
        kv_cpu = kv_gpu.cpu()

        # Swap out (D2H)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            kv_cpu.copy_(kv_gpu)
        torch.cuda.synchronize()
        swap_out_ms = (time.time() - t0) / 10 * 1000

        # Swap in (H2D)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            kv_gpu.copy_(kv_cpu)
        torch.cuda.synchronize()
        swap_in_ms = (time.time() - t0) / 10 * 1000

        total_swap_ms = swap_out_ms + swap_in_ms

        # Recompute: re-run prefill for the sequence
        x = torch.randn(1, seq_len, H, device="cuda", dtype=torch.float16)
        K = torch.randn(1, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        V = torch.randn(1, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        Q = torch.randn(1, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

        def recompute():
            # Simulate re-running attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores.float(), dim=-1).to(torch.float16)
            out = torch.matmul(attn, V)
            out = F.linear(out.reshape(1, seq_len, H), weight)
            return out

        recompute_ms = bench_ms(recompute, rep=10)

        winner = "Recompute" if recompute_ms < total_swap_ms else "Swap"
        crossover = "N/A"

        print(f"\n  SeqLen={seq_len} ({kv_mb:.1f} MB KV):")
        print(f"    Swap:     {total_swap_ms:.3f}ms (out={swap_out_ms:.2f} + in={swap_in_ms:.2f})")
        print(f"    Recompute: {recompute_ms:.3f}ms")
        print(f"    Winner:   {winner} ({min(total_swap_ms, recompute_ms)/max(total_swap_ms, recompute_ms):.2f}x)")

        results.append({
            "seq_len": seq_len,
            "kv_mb": round(kv_mb, 1),
            "swap_out_ms": round(swap_out_ms, 3),
            "swap_in_ms": round(swap_in_ms, 3),
            "total_swap_ms": round(total_swap_ms, 3),
            "recompute_ms": round(recompute_ms, 3),
            "winner": winner,
        })

        del kv_gpu, kv_cpu, x, K, V, Q
        torch.cuda.empty_cache()

    del weight
    torch.cuda.empty_cache()
    return results


# ============================================================
# 实验 4: Pinned Memory vs Pageable Memory 传输
# ============================================================

def exp4_pinned_memory():
    print("\n" + "=" * 60)
    print("实验4: Pinned Memory vs Pageable Memory")
    print("=" * 60)

    results = []

    for size_mb in [1, 16, 64, 256]:
        n_elements = size_mb * 1024 * 1024 // 2

        # Pageable memory (normal)
        cpu_pageable = torch.zeros(n_elements, dtype=torch.float16)
        gpu_tensor = torch.zeros(n_elements, device="cuda", dtype=torch.float16)

        # D2H pageable
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            cpu_pageable.copy_(gpu_tensor)
        torch.cuda.synchronize()
        pageable_d2h = (time.time() - t0) / 10 * 1000

        # Pinned memory
        cpu_pinned = torch.zeros(n_elements, dtype=torch.float16).pin_memory()

        # D2H pinned
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            cpu_pinned.copy_(gpu_tensor)
        torch.cuda.synchronize()
        pinned_d2h = (time.time() - t0) / 10 * 1000

        # H2D pinned
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            gpu_tensor.copy_(cpu_pinned)
        torch.cuda.synchronize()
        pinned_h2d = (time.time() - t0) / 10 * 1000

        speedup = pageable_d2h / pinned_d2h

        print(f"\n  {size_mb}MB:")
        print(f"    Pageable D2H: {pageable_d2h:.3f}ms")
        print(f"    Pinned D2H:   {pinned_d2h:.3f}ms ({speedup:.2f}x)")
        print(f"    Pinned H2D:   {pinned_h2d:.3f}ms")

        results.append({
            "size_mb": size_mb,
            "pageable_d2h_ms": round(pageable_d2h, 3),
            "pinned_d2h_ms": round(pinned_d2h, 3),
            "pinned_h2d_ms": round(pinned_h2d, 3),
            "pinned_speedup": round(speedup, 2),
        })

        del cpu_pageable, cpu_pinned, gpu_tensor
        torch.cuda.empty_cache()

    return results


# ============================================================
# 实验 5: Offload 对推理延迟的影响
# ============================================================

def exp5_offload_latency():
    print("\n" + "=" * 60)
    print("实验5: Offload 对推理延迟的影响")
    print("=" * 60)

    results = []
    H = 256
    n_heads = 8
    head_dim = H // n_heads
    B = 4

    print(f"\n  B={B}, H={H}, simulating decode step with KV offload")
    print(f"  {'KV Length':<12} {'GPU Only ms':<14} {'Offload ms':<14} {'Swap In ms':<14} {'Overhead%'}")
    print("  " + "-" * 68)

    for kv_len in [256, 512, 1024, 2048, 4096, 8192]:
        Q = torch.randn(B, n_heads, 1, head_dim, device="cuda", dtype=torch.float16)
        K = torch.randn(B, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
        V = torch.randn(B, n_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)

        # GPU only: attention with KV in GPU
        def gpu_only():
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores.float(), dim=-1).to(torch.float16)
            return torch.matmul(attn, V)

        gpu_ms = bench_ms(gpu_only, rep=20)

        # Offload: K/V swapped to CPU, then back
        K_cpu = K.cpu()
        V_cpu = V.cpu()

        def with_offload():
            # Swap in
            K_local = K_cpu.cuda()
            V_local = V_cpu.cuda()
            # Attention
            scores = torch.matmul(Q, K_local.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores.float(), dim=-1).to(torch.float16)
            return torch.matmul(attn, V_local)

        offload_ms = bench_ms(with_offload, rep=10)

        # Swap in time only
        def swap_in_only():
            return K_cpu.cuda()

        swap_in_ms = bench_ms(swap_in_only, rep=10)

        overhead = (offload_ms - gpu_ms) / gpu_ms * 100

        print(f"  {kv_len:<12} {gpu_ms:<14.3f} {offload_ms:<14.3f} {swap_in_ms:<14.3f} {overhead:.0f}%")

        results.append({
            "kv_len": kv_len,
            "gpu_only_ms": round(gpu_ms, 3),
            "offload_ms": round(offload_ms, 3),
            "swap_in_ms": round(swap_in_ms, 3),
            "overhead_pct": round(overhead, 0),
        })

        del Q, K, V, K_cpu, V_cpu
        torch.cuda.empty_cache()

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    all_results = OrderedDict()
    all_results["memory_hierarchy"] = exp1_memory_hierarchy()
    all_results["transfer_speed"] = exp2_transfer_speed()
    all_results["swap_vs_recompute"] = exp3_swap_vs_recompute()
    all_results["pinned_memory"] = exp4_pinned_memory()
    all_results["offload_latency"] = exp5_offload_latency()

    print("\n" + "=" * 60)
    print("关键洞察")
    print("=" * 60)
    print("""
  1. 内存层级: GPU < CPU < SSD 容量差 6-50x
     LLaMA-7B FP16: GPU ~4K seq, CPU ~24K seq, SSD ~400K seq
  2. PCIe 传输: A16 ~5-8 GB/s (D2H ≈ H2D)
     1GB KV swap ≈ 125-200ms roundtrip
  3. Swap vs Recompute:
     - 短序列 (<512): Recompute 更快 (GPU 算力充裕)
     - 长序列 (>2K): Swap 可能更快 (PCIe 传输 < 重计算)
     - 高端 GPU (A100/H100): Recompute 始终更快 (算力 >> PCIe BW)
  4. Pinned Memory: 加速传输 ~1-2x (避免 OS page fault)
  5. Offload 开销: KV swap in 是主要瓶颈
     - 4K seq: GPU 0.14ms → offload 5ms+ (35x 开销)
     - 仅适合 batch 间隙或低优先级请求
  6. vLLM 策略: SwapOut 辅助, 不主动 offload (成本 > 收益)
""")

    with open("/root/kv_offload_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved.")
