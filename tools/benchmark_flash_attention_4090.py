#!/usr/bin/env python3
"""FlashAttention Throughput & Memory Savings on RTX 4090
=========================================================

Measures the actual performance benefit of FlashAttention (via PyTorch SDPA)
vs naive attention implementation, and validates the IO savings theory.

Key questions:
A. How much speedup does FlashAttention provide at different sequence lengths?
B. Does speedup scale with seq_len² as predicted (IO savings)?
C. Memory usage comparison: FlashAttention vs naive (no KV cache for backward)
D. Backward pass savings (FlashAttention recomputes attention matrix)
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
import json

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print("=" * 60)

device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)
HBM_GB = props.total_memory / 1e9
print(f"HBM: {HBM_GB:.1f} GB")
print("=" * 60)


# ============================================================
# 1. Naive Attention (explicit softmax)
# ============================================================

def naive_attention(q, k, v):
    """Standard attention without FlashAttention optimization."""
    # q: [B, H, S_q, D], k: [B, H, S_k, D], v: [B, H, S_k, D]
    scale = q.shape[-1] ** -0.5
    attn_weights = torch.matmul(q * scale, k.transpose(-2, -1))  # [B, H, S_q, S_k]
    attn_weights = F.softmax(attn_weights, dim=-1)
    output = torch.matmul(attn_weights, v)  # [B, H, S_q, D]
    return output


# ============================================================
# Benchmark: Forward Only
# ============================================================

def bench_forward():
    print("\n1. Forward Attention: FlashAttention vs Naive")

    results = []
    configs = [
        # Decode-style (Q=1 token, KV=seq_len)
        ("decode B=1 S=512", 1, 32, 1, 512, 128),
        ("decode B=4 S=512", 4, 32, 1, 512, 128),
        ("decode B=32 S=512", 32, 32, 1, 512, 128),
        ("decode B=128 S=512", 128, 32, 1, 512, 128),
        ("decode B=1 S=2K", 1, 32, 1, 2048, 128),
        ("decode B=32 S=2K", 32, 32, 1, 2048, 128),
        ("decode B=128 S=2K", 128, 32, 1, 2048, 128),
        # Prefill-style (Q=seq_len, KV=seq_len)
        ("prefill B=1 S=1K", 1, 32, 1024, 1024, 128),
        ("prefill B=1 S=2K", 1, 32, 2048, 2048, 128),
        ("prefill B=1 S=4K", 1, 32, 4096, 4096, 128),
        ("prefill B=4 S=1K", 4, 32, 1024, 1024, 128),
        ("prefill B=4 S=4K", 4, 32, 4096, 4096, 128),
    ]

    for desc, B, H, S_q, S_k, D in configs:
        q = torch.randn(B, H, S_q, D, device='cuda')
        k = torch.randn(B, H, S_k, D, device='cuda')
        v = torch.randn(B, H, S_k, D, device='cuda')

        # Warmup both
        for _ in range(5):
            _ = F.scaled_dot_product_attention(q, k, v)
            _ = naive_attention(q, k, v)
        torch.cuda.synchronize()

        n = 20
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        # FlashAttention (SDPA)
        s.record()
        for _ in range(n):
            _ = F.scaled_dot_product_attention(q, k, v)
        e.record()
        torch.cuda.synchronize()
        flash_ms = s.elapsed_time(e) / n

        # Naive
        s.record()
        for _ in range(n):
            _ = naive_attention(q, k, v)
        e.record()
        torch.cuda.synchronize()
        naive_ms = s.elapsed_time(e) / n

        speedup = naive_ms / flash_ms if flash_ms > 0 else 0

        print(f"  {desc}: naive={naive_ms:.4f}ms flash={flash_ms:.4f}ms speedup={speedup:.2f}x")

        results.append({
            "desc": desc, "B": B, "H": H, "S_q": S_q, "S_k": S_k, "D": D,
            "naive_ms": round(naive_ms, 4),
            "flash_ms": round(flash_ms, 4),
            "speedup": round(speedup, 2),
        })

    return results


# ============================================================
# Benchmark: Memory Usage
# ============================================================

def bench_memory():
    print("\n2. Memory Usage: FlashAttention vs Naive")

    results = []
    configs = [
        (1, 32, 512, 128),
        (1, 32, 1024, 128),
        (1, 32, 2048, 128),
        (1, 32, 4096, 128),
        (1, 32, 8192, 128),
        (4, 32, 2048, 128),
        (4, 32, 4096, 128),
    ]

    for B, H, S, D in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Naive attention (stores full attention matrix [B, H, S, S])
        q = torch.randn(B, H, S, D, device='cuda', requires_grad=True)
        k = torch.randn(B, H, S, D, device='cuda', requires_grad=True)
        v = torch.randn(B, H, S, D, device='cuda', requires_grad=True)

        naive_peak = torch.cuda.max_memory_allocated() / 1e6  # input tensors only

        out_naive = naive_attention(q, k, v)
        naive_peak = torch.cuda.max_memory_allocated() / 1e6

        # Theoretical naive memory for attention matrix
        attn_matrix_bytes = B * H * S * S * 4  # FP32 attention weights
        attn_matrix_mb = attn_matrix_bytes / 1e6

        del q, k, v, out_naive
        torch.cuda.empty_cache()

        # FlashAttention (never stores full attention matrix)
        torch.cuda.reset_peak_memory_stats()
        q = torch.randn(B, H, S, D, device='cuda', requires_grad=True)
        k = torch.randn(B, H, S, D, device='cuda', requires_grad=True)
        v = torch.randn(B, H, S, D, device='cuda', requires_grad=True)

        out_flash = F.scaled_dot_product_attention(q, k, v)
        flash_peak = torch.cuda.max_memory_allocated() / 1e6

        mem_saved_pct = (naive_peak - flash_peak) / naive_peak * 100 if naive_peak > 0 else 0

        print(f"  B={B} H={H} S={S} D={D}: "
              f"naive_peak={naive_peak:.0f}MB flash_peak={flash_peak:.0f}MB "
              f"attn_matrix_theory={attn_matrix_mb:.0f}MB saved={mem_saved_pct:.1f}%")

        results.append({
            "B": B, "H": H, "S": S, "D": D,
            "naive_peak_mb": round(naive_peak, 0),
            "flash_peak_mb": round(flash_peak, 0),
            "attn_matrix_theory_mb": round(attn_matrix_mb, 0),
            "saved_pct": round(mem_saved_pct, 1),
        })

        del q, k, v, out_flash
        torch.cuda.empty_cache()

    return results


# ============================================================
# Benchmark: Forward + Backward
# ============================================================

def bench_backward():
    print("\n3. Forward + Backward: FlashAttention vs Naive")

    results = []
    configs = [
        (1, 32, 512, 128),
        (1, 32, 1024, 128),
        (1, 32, 2048, 128),
        (1, 32, 4096, 128),
        (4, 32, 1024, 128),
        (4, 32, 2048, 128),
    ]

    for B, H, S, D in configs:
        # FlashAttention fwd+bwd
        torch.cuda.empty_cache()
        q = torch.randn(B, H, S, D, device='cuda', requires_grad=True)
        k = torch.randn(B, H, S, D, device='cuda', requires_grad=True)
        v = torch.randn(B, H, S, D, device='cuda', requires_grad=True)

        for _ in range(5):
            out = F.scaled_dot_product_attention(q, k, v)
            out.sum().backward()
        torch.cuda.synchronize()

        n = 10
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            out = F.scaled_dot_product_attention(q, k, v)
            out.sum().backward()
        e.record()
        torch.cuda.synchronize()
        flash_fwbwd_ms = s.elapsed_time(e) / n

        flash_peak = torch.cuda.max_memory_allocated() / 1e6

        q.grad = None; k.grad = None; v.grad = None
        del q, k, v, out
        torch.cuda.empty_cache()

        # Naive fwd+bwd
        torch.cuda.reset_peak_memory_stats()
        q = torch.randn(B, H, S, D, device='cuda', requires_grad=True)
        k = torch.randn(B, H, S, D, device='cuda', requires_grad=True)
        v = torch.randn(B, H, S, D, device='cuda', requires_grad=True)

        for _ in range(5):
            out = naive_attention(q, k, v)
            out.sum().backward()
        torch.cuda.synchronize()

        s.record()
        for _ in range(n):
            out = naive_attention(q, k, v)
            out.sum().backward()
        e.record()
        torch.cuda.synchronize()
        naive_fwbwd_ms = s.elapsed_time(e) / n

        naive_peak = torch.cuda.max_memory_allocated() / 1e6

        speedup = naive_fwbwd_ms / flash_fwbwd_ms if flash_fwbwd_ms > 0 else 0
        mem_saved_pct = (naive_peak - flash_peak) / naive_peak * 100 if naive_peak > 0 else 0

        print(f"  B={B} H={H} S={S}: "
              f"naive_fwbwd={naive_fwbwd_ms:.3f}ms flash_fwbwd={flash_fwbwd_ms:.3f}ms "
              f"speedup={speedup:.2f}x mem_saved={mem_saved_pct:.1f}%")

        results.append({
            "B": B, "H": H, "S": S, "D": D,
            "naive_fwbwd_ms": round(naive_fwbwd_ms, 3),
            "flash_fwbwd_ms": round(flash_fwbwd_ms, 3),
            "speedup": round(speedup, 2),
            "naive_peak_mb": round(naive_peak, 0),
            "flash_peak_mb": round(flash_peak, 0),
            "mem_saved_pct": round(mem_saved_pct, 1),
        })

        q.grad = None; k.grad = None; v.grad = None
        del q, k, v, out
        torch.cuda.empty_cache()

    return results


# ============================================================
# Run
# ============================================================

forward_results = bench_forward()
memory_results = bench_memory()
backward_results = bench_backward()

print("\n" + "=" * 60)
print("SUMMARY: FlashAttention on RTX 4090")
print("=" * 60)
avg_fwd_speedup = sum(r["speedup"] for r in forward_results) / len(forward_results)
avg_bwd_speedup = sum(r["speedup"] for r in backward_results) / len(backward_results)
avg_mem_saved = sum(r["saved_pct"] for r in memory_results) / len(memory_results)
print(f"Average forward speedup: {avg_fwd_speedup:.2f}x")
print(f"Average fwd+bwd speedup: {avg_bwd_speedup:.2f}x")
print(f"Average memory saved: {avg_mem_saved:.1f}%")

# Save
out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, 'flash_attention_benchmark_results.json')
with open(out_path, 'w') as f:
    json.dump({
        "forward": forward_results,
        "memory": memory_results,
        "forward_backward": backward_results,
    }, f, indent=2)
print(f"Results saved to {out_path}")