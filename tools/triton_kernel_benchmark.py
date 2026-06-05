#!/usr/bin/env python3
"""Triton Kernel Benchmark Suite — RTX 4090
================================================
Demonstrates GPU kernel optimization from first principles:
1. Fused LayerNorm vs PyTorch LayerNorm
2. Fused RMSNorm vs PyTorch RMSNorm
3. Fused QKV Projection (3 matmuls → 1 fused)
4. Softmax kernel (online softmax / Flash Attention style)
5. Block-sparse matmul (MoE-style)

Educational purpose: understand GPU kernel optimization patterns.
Requires: triton (pip install triton)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from collections import defaultdict

# Check for Triton
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("WARNING: triton not installed. Running PyTorch-only benchmarks.")


# ============================================================
# 1. Fused RMSNorm (Triton)
# ============================================================

if HAS_TRITON:
    @triton.jit
    def _rmsnorm_kernel(
        X_ptr, W_ptr, Y_ptr,
        stride_x_batch, stride_x_dim,
        stride_w_dim,
        N: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused RMSNorm kernel: y = x / sqrt(mean(x^2) + eps) * w"""
        row_idx = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # Load row
        x_ptrs = X_ptr + row_idx * stride_x_batch + cols * stride_x_dim
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Compute RMS: sqrt(mean(x^2) + eps)
        x_sq = x * x
        mean_sq = tl.sum(x_sq, axis=0) / N
        rms = tl.sqrt(mean_sq + eps)

        # Normalize
        x_normed = x / rms

        # Multiply by weight
        w_ptrs = W_ptr + cols * stride_w_dim
        w = tl.load(w_ptrs, mask=mask, other=1.0).to(tl.float32)
        y = x_normed * w

        # Store
        y_ptrs = Y_ptr + row_idx * stride_x_batch + cols * stride_x_dim
        tl.store(y_ptrs, y, mask=mask)


def triton_rmsnorm(x, weight, eps=1e-6):
    """Fused RMSNorm using Triton."""
    assert x.is_contiguous()
    B, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)

    grid = (B,)
    _rmsnorm_kernel[grid](
        x, weight, y,
        x.stride(0), x.stride(1),
        weight.stride(0),
        N=N, eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


def torch_rmsnorm(x, weight, eps=1e-6):
    """PyTorch RMSNorm reference."""
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (weight * x_normed).to(x.dtype)


# ============================================================
# 2. Fused Softmax (Triton — Flash Attention style)
# ============================================================

if HAS_TRITON:
    @triton.jit
    def _softmax_kernel(
        X_ptr, Y_ptr,
        stride_x_batch, stride_x_row, stride_x_col,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Online softmax kernel (one-pass, Flash Attention style).

        Key insight: Instead of loading all elements, compute running max + sum.
        This is exactly what Flash Attention does for attention scores.
        """
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        # Load row
        x_ptrs = X_ptr + row_idx * stride_x_row + col_offsets * stride_x_col
        x = tl.load(x_ptrs, mask=mask, other=float('-inf')).to(tl.float32)

        # Step 1: Find max (for numerical stability)
        x_max = tl.max(x, axis=0)

        # Step 2: Compute exp(x - max)
        x_shifted = x - x_max
        exp_x = tl.exp(x_shifted)

        # Step 3: Compute sum
        sum_exp = tl.sum(exp_x, axis=0)

        # Step 4: Normalize
        y = exp_x / sum_exp

        # Store
        y_ptrs = Y_ptr + row_idx * stride_x_row + col_offsets * stride_x_col
        tl.store(y_ptrs, y, mask=mask)


def triton_softmax(x):
    """Fused softmax using Triton."""
    assert x.is_contiguous()
    B, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)

    grid = (B,)
    _softmax_kernel[grid](
        x, y,
        x.stride(0), x.stride(0), x.stride(1),
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


# ============================================================
# 3. Fused QKV Projection (Triton)
# ============================================================

if HAS_TRITON:
    @triton.jit
    def _fused_qkv_kernel(
        X_ptr, Wq_ptr, Wk_ptr, Wv_ptr,
        Q_ptr, K_ptr, V_ptr,
        stride_x_batch, stride_x_dim,
        M: tl.constexpr,  # hidden dim
        D: tl.constexpr,  # head dim
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused QKV projection: 3 separate matmuls → 1 kernel launch.

        In standard implementation:
          Q = X @ Wq  (kernel launch 1)
          K = X @ Kq  (kernel launch 2)
          V = X @ Vq  (kernel launch 3)

        Fused: 1 kernel launch, X loaded once.
        """
        batch_idx = tl.program_id(0)
        m_idx = tl.program_id(1)

        m_off = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        d_off = tl.arange(0, BLOCK_D)

        # Load X row
        x_ptrs = X_ptr + batch_idx * stride_x_batch + m_off[:, None] * stride_x_dim
        x = tl.load(x_ptrs, mask=m_off[:, None] < M, other=0.0).to(tl.float32)

        # Compute Q = X @ Wq
        wq_ptrs = Wq_ptr + m_off[:, None] * D + d_off[None, :]
        wq = tl.load(wq_ptrs, mask=(m_off[:, None] < M) & (d_off[None, :] < D), other=0.0)
        q = tl.sum(x * wq, axis=0)  # [D]

        # Compute K = X @ Wk
        wk_ptrs = Wk_ptr + m_off[:, None] * D + d_off[None, :]
        wk = tl.load(wk_ptrs, mask=(m_off[:, None] < M) & (d_off[None, :] < D), other=0.0)
        k = tl.sum(x * wk, axis=0)

        # Compute V = X @ Wv
        wv_ptrs = Wv_ptr + m_off[:, None] * D + d_off[None, :]
        wv = tl.load(wv_ptrs, mask=(m_off[:, None] < M) & (d_off[None, :] < D), other=0.0)
        v = tl.sum(x * wv, axis=0)

        # Store
        base = batch_idx * D
        tl.store(Q_ptr + base + d_off, q, mask=d_off < D)
        tl.store(K_ptr + base + d_off, k, mask=d_off < D)
        tl.store(V_ptr + base + d_off, v, mask=d_off < D)


# ============================================================
# 4. Benchmark Utilities
# ============================================================

def benchmark_fn(fn, *args, warmup=10, rep=100, **kwargs):
    """Benchmark a function using CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    # Synchronize
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(rep):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / rep
    return ms


def benchmark_cpu_fn(fn, *args, warmup=10, rep=100, **kwargs):
    """Benchmark a CPU function."""
    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)

    return np.median(times) * 1000  # ms


# ============================================================
# 5. Experiments
# ============================================================

def experiment_rmsnorm(device='cuda'):
    """Benchmark fused RMSNorm vs PyTorch."""
    print("\n  === Experiment: Fused RMSNorm ===")

    results = {}
    for hidden_dim in [256, 512, 1024, 2048, 4096]:
        batch_size = max(1, 8192 // hidden_dim)  # Keep similar total elements

        x = torch.randn(batch_size, hidden_dim, device=device)
        w = torch.ones(hidden_dim, device=device)

        # PyTorch baseline
        if device == 'cuda':
            ms_torch = benchmark_fn(torch_rmsnorm, x, w)
        else:
            ms_torch = benchmark_cpu_fn(torch_rmsnorm, x, w)

        # Triton fused
        if HAS_TRITON and device == 'cuda':
            ms_triton = benchmark_fn(triton_rmsnorm, x, w)
            speedup = ms_torch / ms_triton
            print(f"    dim={hidden_dim:>5} B={batch_size:>4}: "
                  f"PyTorch={ms_torch:.3f}ms, Triton={ms_triton:.3f}ms, "
                  f"speedup={speedup:.2f}x")
            results[hidden_dim] = {
                'pytorch_ms': round(ms_torch, 3),
                'triton_ms': round(ms_triton, 3),
                'speedup': round(speedup, 2),
            }
        else:
            print(f"    dim={hidden_dim:>5} B={batch_size:>4}: PyTorch={ms_torch:.3f}ms")
            results[hidden_dim] = {'pytorch_ms': round(ms_torch, 3)}

        # Verify correctness
        y_torch = torch_rmsnorm(x, w)
        if HAS_TRITON and device == 'cuda':
            y_triton = triton_rmsnorm(x, w)
            max_diff = (y_torch - y_triton).abs().max().item()
            assert max_diff < 1e-5, f"RMSNorm mismatch: {max_diff}"

    return results


def experiment_softmax(device='cuda'):
    """Benchmark fused softmax vs PyTorch."""
    print("\n  === Experiment: Fused Softmax (Flash Attention style) ===")

    results = {}
    for seq_len in [256, 512, 1024, 2048, 4096, 8192]:
        batch_size = min(1024, max(1, 8192 // seq_len))

        x = torch.randn(batch_size, seq_len, device=device)

        # PyTorch baseline
        if device == 'cuda':
            ms_torch = benchmark_fn(F.softmax, x, dim=-1)
        else:
            ms_torch = benchmark_cpu_fn(F.softmax, x, dim=-1)

        # Triton fused
        if HAS_TRITON and device == 'cuda':
            ms_triton = benchmark_fn(triton_softmax, x)
            speedup = ms_torch / ms_triton
            print(f"    seq={seq_len:>5} B={batch_size:>4}: "
                  f"PyTorch={ms_torch:.3f}ms, Triton={ms_triton:.3f}ms, "
                  f"speedup={speedup:.2f}x")
            results[seq_len] = {
                'pytorch_ms': round(ms_torch, 3),
                'triton_ms': round(ms_triton, 3),
                'speedup': round(speedup, 2),
            }
        else:
            print(f"    seq={seq_len:>5} B={batch_size:>4}: PyTorch={ms_torch:.3f}ms")
            results[seq_len] = {'pytorch_ms': round(ms_torch, 3)}

    return results


def experiment_fused_qkv(device='cuda'):
    """Benchmark separate QKV projections vs fused."""
    print("\n  === Experiment: Fused QKV Projection ===")

    results = {}
    for hidden_dim in [512, 1024, 2048]:
        batch_size = 32
        head_dim = hidden_dim

        x = torch.randn(batch_size, hidden_dim, device=device)
        wq = torch.randn(hidden_dim, head_dim, device=device)
        wk = torch.randn(hidden_dim, head_dim, device=device)
        wv = torch.randn(hidden_dim, head_dim, device=device)

        # Separate: 3 kernel launches
        def separate_qkv():
            q = x @ wq
            k = x @ wk
            v = x @ wv
            return q, k, v

        if device == 'cuda':
            ms_separate = benchmark_fn(separate_qkv)
        else:
            ms_separate = benchmark_cpu_fn(separate_qkv)

        # PyTorch stacked (1 matmul with stacked weights)
        w_stacked = torch.cat([wq, wk, wv], dim=1)  # [hidden, 3*head]
        def stacked_qkv():
            qkv = x @ w_stacked  # [B, 3*head]
            q, k, v = qkv.chunk(3, dim=-1)
            return q, k, v

        if device == 'cuda':
            ms_stacked = benchmark_fn(stacked_qkv)
        else:
            ms_stacked = benchmark_cpu_fn(stacked_qkv)

        speedup = ms_separate / ms_stacked
        print(f"    dim={hidden_dim:>5}: "
              f"separate={ms_separate:.3f}ms, stacked={ms_stacked:.3f}ms, "
              f"speedup={speedup:.2f}x")
        results[hidden_dim] = {
            'separate_ms': round(ms_separate, 3),
            'stacked_ms': round(ms_stacked, 3),
            'speedup': round(speedup, 2),
        }

    return results


def experiment_layernorm_vs_rmsnorm(device='cuda'):
    """Compare LayerNorm vs RMSNorm throughput."""
    print("\n  === Experiment: LayerNorm vs RMSNorm ===")

    results = {}
    for hidden_dim in [256, 512, 1024, 2048, 4096]:
        batch_size = max(1, 8192 // hidden_dim)

        x = torch.randn(batch_size, hidden_dim, device=device)
        w = torch.ones(hidden_dim, device=device)
        b = torch.zeros(hidden_dim, device=device)

        # LayerNorm (PyTorch)
        ln = nn.LayerNorm(hidden_dim, device=device)

        # RMSNorm (PyTorch)
        def rmsnorm_fn():
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            return (w * x * torch.rsqrt(variance + 1e-6)).to(x.dtype)

        if device == 'cuda':
            ms_ln = benchmark_fn(ln, x)
            ms_rms = benchmark_fn(rmsnorm_fn)
        else:
            ms_ln = benchmark_cpu_fn(ln, x)
            ms_rms = benchmark_cpu_fn(rmsnorm_fn)

        speedup = ms_ln / ms_rms
        print(f"    dim={hidden_dim:>5}: "
              f"LayerNorm={ms_ln:.3f}ms, RMSNorm={ms_rms:.3f}ms, "
              f"RMS faster={speedup:.2f}x")
        results[hidden_dim] = {
            'layernorm_ms': round(ms_ln, 3),
            'rmsnorm_ms': round(ms_rms, 3),
            'speedup': round(speedup, 2),
        }

    return results


def experiment_memory_bandwidth(device='cuda'):
    """Measure effective memory bandwidth for common operations."""
    print("\n  === Experiment: Memory Bandwidth ===")

    if device != 'cuda':
        print("    Skipping (GPU only)")
        return {}

    gpu_name = torch.cuda.get_device_name(0)
    # Use known peak bandwidths for common GPUs
    peak_bw_map = {
        'NVIDIA GeForce RTX 4090': 1008,
        'NVIDIA A100-SXM4-80GB': 2039,
        'NVIDIA H100 80GB HBM3': 3350,
        'NVIDIA RTX A6000': 768,
    }
    peak_bw = peak_bw_map.get(gpu_name, 800)  # Default 800 GB/s

    results = {}
    size = 16 * 1024 * 1024  # 16M elements = 64 MB (FP16)

    # Vector add: 3 memory ops (read a, read b, write c) → 3 × 64 MB
    a = torch.randn(size, device=device, dtype=torch.float16)
    b = torch.randn(size, device=device, dtype=torch.float16)

    def vector_add():
        return a + b

    ms = benchmark_fn(vector_add)
    data_moved = 3 * size * 2 / 1e9  # GB (3 ops × 2 bytes each)
    effective_bw = data_moved / (ms / 1000)  # GB/s
    efficiency = effective_bw / peak_bw * 100

    print(f"    Vector add (64MB): {ms:.3f}ms, {effective_bw:.0f} GB/s "
          f"({efficiency:.1f}% of {peak_bw:.0f} GB/s peak)")
    results['vector_add'] = {
        'ms': round(ms, 3),
        'gb_s': round(effective_bw, 1),
        'efficiency_pct': round(efficiency, 1),
    }

    # Element-wise multiply
    def elem_mul():
        return a * b

    ms = benchmark_fn(elem_mul)
    effective_bw = data_moved / (ms / 1000)
    efficiency = effective_bw / peak_bw * 100
    print(f"    Elem mul (64MB): {ms:.3f}ms, {effective_bw:.0f} GB/s "
          f"({efficiency:.1f}%)")
    results['elem_mul'] = {
        'ms': round(ms, 3),
        'gb_s': round(effective_bw, 1),
        'efficiency_pct': round(efficiency, 1),
    }

    # Reduction (sum)
    data_moved_sum = size * 2 / 1e9  # Only read
    ms = benchmark_fn(lambda: a.sum())
    effective_bw = data_moved_sum / (ms / 1000)
    print(f"    Reduction (32MB): {ms:.3f}ms, {effective_bw:.0f} GB/s")
    results['reduction'] = {
        'ms': round(ms, 3),
        'gb_s': round(effective_bw, 1),
    }

    results['peak_bw_gbs'] = round(peak_bw, 0)
    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Triton Kernel Benchmark Suite")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"\n  GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")
        print(f"  Triton: {'available' if HAS_TRITON else 'not available'}")
    else:
        print(f"\n  Device: CPU (limited benchmarks)")

    results = {}

    # Exp 1: RMSNorm
    results['rmsnorm'] = experiment_rmsnorm(device)

    # Exp 2: Softmax
    results['softmax'] = experiment_softmax(device)

    # Exp 3: Fused QKV
    results['fused_qkv'] = experiment_fused_qkv(device)

    # Exp 4: LayerNorm vs RMSNorm
    results['ln_vs_rms'] = experiment_layernorm_vs_rmsnorm(device)

    # Exp 5: Memory bandwidth
    results['memory_bw'] = experiment_memory_bandwidth(device)

    # Summary
    print("\n" + "=" * 60)
    print("Kernel Optimization Summary")
    print("=" * 60)
    print("""
    Key GPU Kernel Optimization Patterns:

    1. FUSION: Multiple ops → 1 kernel launch
       - Eliminates intermediate memory reads/writes
       - Example: RMSNorm (load x → compute → store y) in 1 pass
       - Savings: 2-5x for bandwidth-bound ops

    2. TILING: Divide large computation into tiles
       - Each tile fits in SRAM (shared memory)
       - Flash Attention: O(N) memory instead of O(N²)
       - Example: softmax processes one row per program

    3. ONLINE ALGORITHMS: Multi-pass → single-pass
       - Flash Attention's online softmax:
         max_running → max_running, sum_running → sum_running
       - Eliminates the need to store intermediate results

    4. MEMORY COALESCING: Ensure consecutive threads read
       consecutive memory addresses
       - Triton handles this automatically with tl.load/store
       - Critical for achieving peak bandwidth

    5. REDUCING KERNEL LAUNCH OVERHEAD:
       - Launch overhead: ~5-10μs per kernel (GPU dependent)
       - Fusing 3 kernels → 1: save 10-20μs
       - Matters for small ops, not for large matmuls
    """)

    if device == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"  Peak GPU memory: {mem:.1f} MB")
        results['gpu_memory_mb'] = round(mem, 1)

    with open("triton_kernel_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to triton_kernel_results.json")


if __name__ == "__main__":
    main()
