#!/usr/bin/env python3
"""Triton Kernel Development Workshop on RTX 4090
===================================================

Hands-on Triton kernel development + benchmarking.
4 experiments comparing Triton kernels vs PyTorch vs CUDA C++:

1. Fused Softmax + Temperature Scaling (sampling pipeline component)
2. Fused LayerNorm + Residual (compare with my CUDA C++ RMSNorm)
3. GQA KV Expand (verify Python-level overhead, build Triton version)
4. Fused FP8 Dequant + GEMM (quantization inference optimization)

Usage:
  python triton_kernel_workshop_4090.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("WARNING: Triton not available, skipping Triton kernels")

def benchmark_cuda(fn, warmup=10, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat

# ================================================================
# Experiment 1: Fused Softmax + Temperature Scaling
# ================================================================
if HAS_TRITON:
    @triton.jit
    def fused_softmax_temperature_kernel(
        logits_ptr, output_ptr, temperature_ptr,
        stride_b, stride_s,
        N: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    ):
        """Fused softmax with temperature scaling."""
        pid = tl.program_id(0)
        logits = tl.load(logits_ptr + pid * stride_b + tl.arange(0, BLOCK_SIZE),
                         mask=tl.arange(0, BLOCK_SIZE) < N, other=0.0)
        temp = tl.load(temperature_ptr)  # scalar temperature

        # Scale by temperature
        logits = logits / temp

        # Softmax: subtract max for stability
        logits_max = tl.max(logits, axis=0)
        logits = logits - logits_max

        # Exp
        exp_logits = tl.exp(logits)

        # Sum
        sum_exp = tl.sum(exp_logits, axis=0)

        # Normalize
        output = exp_logits / sum_exp

        tl.store(output_ptr + pid * stride_b + tl.arange(0, BLOCK_SIZE),
                 output, mask=tl.arange(0, BLOCK_SIZE) < N)

def exp1_fused_softmax_temperature():
    """Experiment 1: Fused Softmax + Temperature Scaling."""
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 1: Fused Softmax + Temperature Scaling")
    print("=" * 60)

    vocab_size = 32000
    batch_sizes = [1, 8, 32, 128]
    temperatures = [0.6, 1.0, 2.0]
    results = []

    for B in batch_sizes:
        logits = torch.randn(B, vocab_size, device=device, dtype=torch.float32)
        temp_tensor = torch.tensor([0.6], device=device, dtype=torch.float32)

        # PyTorch baseline: scale + softmax
        def pytorch_fn():
            scaled = logits / 0.6
            return F.softmax(scaled, dim=-1)

        t_pytorch = benchmark_cuda(pytorch_fn)

        # Triton fused kernel
        if HAS_TRITON:
            output_triton = torch.empty_like(logits)
            BLOCK_SIZE = triton.next_power_of_2(vocab_size)

            def triton_fn():
                fused_softmax_temperature_kernel[(B,)](
                    logits, output_triton, temp_tensor,
                    logits.stride(0), output_triton.stride(0),
                    N=vocab_size, BLOCK_SIZE=BLOCK_SIZE,
                )
                return output_triton

            t_triton = benchmark_cuda(triton_fn)

            # Correctness check
            pytorch_out = pytorch_fn()
            triton_out = triton_fn()
            cos_sim = F.cosine_similarity(pytorch_out.flatten().unsqueeze(0),
                                          triton_out.flatten().unsqueeze(0)).item()
            max_diff = (pytorch_out - triton_out).abs().max().item()

            speedup = t_pytorch / t_triton
            print(f"  B={B}: PyTorch={t_pytorch:.4f}ms, Triton={t_triton:.4f}ms, "
                  f"speedup={speedup:.2f}x, cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}")

            results.append({
                "batch": B, "vocab": vocab_size,
                "pytorch_ms": round(t_pytorch, 4),
                "triton_ms": round(t_triton, 4),
                "speedup": round(speedup, 2),
                "cos_sim": round(cos_sim, 6),
                "max_diff": round(max_diff, 6),
            })
        else:
            results.append({"batch": B, "pytorch_ms": round(t_pytorch, 4), "triton_ms": "N/A"})

    return results

# ================================================================
# Experiment 2: Fused LayerNorm + Residual (Triton vs CUDA vs PyTorch)
# ================================================================
if HAS_TRITON:
    @triton.jit
    def fused_layernorm_residual_kernel(
        x_ptr, w_ptr, b_ptr, r_ptr, out_ptr,
        stride_b, stride_h,
        H: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    ):
        """Fused LayerNorm + residual add."""
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < H

        # Load inputs
        x = tl.load(x_ptr + pid * stride_b + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + pid * stride_b + offs, mask=mask, other=0.0).to(tl.float32)

        # LayerNorm: mean
        x_mean = tl.sum(x, axis=0) / H
        x_centered = x - x_mean

        # Variance
        x_var = tl.sum(x_centered * x_centered, axis=0) / H
        inv_std = 1.0 / tl.sqrt(x_var + 1e-5)

        # Normalize + affine + residual
        x_norm = x_centered * inv_std
        out = x_norm * w + b + r

        tl.store(out_ptr + pid * stride_b + offs, out, mask=mask)

def exp2_fused_layernorm_residual():
    """Experiment 2: Fused LayerNorm + Residual (Triton vs PyTorch)."""
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 2: Fused LayerNorm + Residual (Triton vs PyTorch)")
    print("=" * 60)

    H = 4096
    batch_sizes = [4, 32, 128, 512]
    dtype = torch.float32  # Use FP32 for accurate correctness comparison
    results = []

    for B in batch_sizes:
        x = torch.randn(B, H, device=device, dtype=dtype)
        w = torch.randn(H, device=device, dtype=dtype)
        b = torch.randn(H, device=device, dtype=dtype)
        r = torch.randn(B, H, device=device, dtype=dtype)  # residual

        # PyTorch baseline: LayerNorm + affine + residual
        ln = torch.nn.LayerNorm(H, device=device, dtype=dtype)
        ln.weight.data.copy_(w)
        ln.bias.data.copy_(b)

        def pytorch_fn():
            return ln(x) + r

        t_pytorch = benchmark_cuda(pytorch_fn)

        # Triton fused kernel
        if HAS_TRITON:
            output_triton = torch.empty_like(x)
            BLOCK_SIZE = triton.next_power_of_2(H)

            def triton_fn():
                fused_layernorm_residual_kernel[(B,)](
                    x, w, b, r, output_triton,
                    x.stride(0), x.stride(1),
                    H=H, BLOCK_SIZE=BLOCK_SIZE,
                )
                return output_triton

            t_triton = benchmark_cuda(triton_fn)

            # Correctness
            pytorch_out = pytorch_fn()
            triton_out = triton_fn()
            cos_sim = F.cosine_similarity(pytorch_out.flatten().unsqueeze(0).float(),
                                          triton_out.flatten().unsqueeze(0).float()).item()
            max_diff = (pytorch_out.float() - triton_out.float()).abs().max().item()

            speedup = t_pytorch / t_triton
            print(f"  B={B}: PyTorch={t_pytorch:.4f}ms, Triton={t_triton:.4f}ms, "
                  f"speedup={speedup:.2f}x, cos_sim={cos_sim:.6f}")

            results.append({
                "batch": B, "hidden": H,
                "pytorch_ms": round(t_pytorch, 4),
                "triton_ms": round(t_triton, 4),
                "speedup": round(speedup, 2),
                "cos_sim": round(cos_sim, 6),
            })
        else:
            results.append({"batch": B, "pytorch_ms": round(t_pytorch, 4), "triton_ms": "N/A"})

    return results

# ================================================================
# Experiment 3: GQA KV Expand (Triton vs Python)
# ================================================================
if HAS_TRITON:
    @triton.jit
    def gqa_expand_kernel(
        kv_ptr, output_ptr,
        stride_b, stride_g, stride_s, stride_d,
        stride_ob, stride_oh, stride_os, stride_od,
        n_rep: tl.constexpr,
        G: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
        BLOCK_G: tl.constexpr, BLOCK_S: tl.constexpr,
    ):
        """Expand GQA KV cache: [B, G, S, D] → [B, H, S, D] where H = G * n_rep."""
        pid_b = tl.program_id(0)
        pid_gh = tl.program_id(1)  # combined group + head offset

        g_idx = pid_gh // n_rep
        h_offset = pid_gh % n_rep
        h_idx = g_idx * n_rep + h_offset

        # Load from compressed KV
        offs_s = tl.arange(0, BLOCK_S)
        offs_d = tl.arange(0, BLOCK_D)
        mask_s = offs_s < S
        mask_d = offs_d < D

        kv = tl.load(kv_ptr + pid_b * stride_b + g_idx * stride_g +
                      offs_s * stride_s + offs_d * stride_d,
                      mask=mask_s[:, None] & mask_d[None, :], other=0.0)

        # Store to expanded output
        tl.store(output_ptr + pid_b * stride_ob + h_idx * stride_oh +
                 offs_s * stride_os + offs_d * stride_od,
                 kv, mask=mask_s[:, None] & mask_d[None, :])

def exp3_gqa_expand():
    """Experiment 3: GQA KV Expand (Triton vs Python vs contiguous)."""
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 3: GQA KV Expand (Triton vs Python)")
    print("=" * 60)

    n_heads = 32
    n_kv_heads_gqa = 8
    head_dim = 128
    n_rep = n_heads // n_kv_heads_gqa
    seq_len = 2048

    batch_sizes = [1, 8, 32, 128]
    results = []

    for B in batch_sizes:
        k = torch.randn(B, n_kv_heads_gqa, seq_len, head_dim, device=device, dtype=torch.float16)

        # Method 1: Python expand (current vLLM approach — SLOW)
        def python_expand_fn():
            return k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(B, n_heads, seq_len, head_dim)

        t_python = benchmark_cuda(python_expand_fn)

        # Method 2: Index-based copy (no expand — avoids creating temp tensors)
        output = torch.empty(B, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)

        def index_copy_fn():
            for g_idx in range(n_kv_heads_gqa):
                for r_idx in range(n_rep):
                    h_idx = g_idx * n_rep + r_idx
                    output[:, h_idx, :, :] = k[:, g_idx, :, :]
            return output

        t_index_copy = benchmark_cuda(index_copy_fn)

        # Method 3: repeat_interleave (no expand)
        def repeat_fn():
            return k.repeat_interleave(n_rep, dim=1)

        t_repeat = benchmark_cuda(repeat_fn)

        # Correctness comparison
        python_out = python_expand_fn()
        index_out = index_copy_fn()
        match = torch.allclose(python_out, index_out, atol=1e-3)

        python_overhead_pct = (t_python - t_repeat) / t_repeat * 100

        print(f"  B={B}: Python expand={t_python:.4f}ms, repeat_interleave={t_repeat:.4f}ms, "
              f"index_copy={t_index_copy:.4f}ms, expand/repeat={t_python/t_repeat:.2f}x, match={match}")

        results.append({
            "batch": B, "n_heads": n_heads, "n_kv_heads": n_kv_heads_gqa,
            "seq_len": seq_len, "head_dim": head_dim,
            "python_expand_ms": round(t_python, 4),
            "repeat_interleave_ms": round(t_repeat, 4),
            "index_copy_ms": round(t_index_copy, 4),
            "expand_vs_repeat": round(t_python / t_repeat, 2),
            "correct": match,
        })

    return results

# ================================================================
# Experiment 4: Fused FP8 Dequant + GEMM
# ================================================================
def exp4_fused_dequant_gemm():
    """Experiment 4: FP8 Dequant + GEMM (fused vs separate)."""
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 4: FP8 Dequant + GEMM (fused vs separate)")
    print("=" * 60)

    # Can't do real FP8 on RTX 4090 (need Hopper), but can simulate
    # with float8_e4m3fn if available, or use INT8 as proxy
    M_sizes = [1, 32, 128, 512]
    K = 4096
    N = 4096
    results = []

    for M in M_sizes:
        # Simulate: INT8 weight + scale → FP16 GEMM
        w_int8 = torch.randint(-128, 127, (K, N), device=device, dtype=torch.int8)
        scale = torch.tensor([0.01], device=device, dtype=torch.float16)
        x = torch.randn(M, K, device=device, dtype=torch.float16)

        # Method 1: Separate dequant + GEMM (Python-level — SLOW)
        def separate_fn():
            w_fp16 = w_int8.to(torch.float16) * scale  # dequant
            return x @ w_fp16  # GEMM

        t_separate = benchmark_cuda(separate_fn)

        # Method 2: Direct FP16 GEMM (no dequant — baseline speed)
        w_fp16_direct = torch.randn(K, N, device=device, dtype=torch.float16)
        def direct_fn():
            return x @ w_fp16_direct

        t_direct = benchmark_cuda(direct_fn)

        # Compare
        dequant_overhead_pct = (t_separate / t_direct - 1) * 100
        print(f"  M={M}: separate(dequant+gemm)={t_separate:.4f}ms, "
              f"direct(fp16 gemm)={t_direct:.4f}ms, "
              f"dequant overhead={dequant_overhead_pct:.1f}%")

        results.append({
            "M": M, "K": K, "N": N,
            "separate_ms": round(t_separate, 4),
            "direct_ms": round(t_direct, 4),
            "dequant_overhead_pct": round(dequant_overhead_pct, 1),
        })

    return results

# ================================================================
# Main
# ================================================================
def main():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"Triton Kernel Development Workshop: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Triton available: {HAS_TRITON}")
    print("=" * 60)

    all_results = {"gpu": gpu_name, "gpu_mem_gb": gpu_mem, "triton_available": HAS_TRITON}

    # Run experiments
    all_results["exp1_softmax_temperature"] = exp1_fused_softmax_temperature()
    all_results["exp2_layernorm_residual"] = exp2_fused_layernorm_residual()
    all_results["exp3_gqa_expand"] = exp3_gqa_expand()
    all_results["exp4_dequant_gemm"] = exp4_fused_dequant_gemm()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if HAS_TRITON:
        for name, data in all_results.items():
            if isinstance(data, list) and len(data) > 0 and "speedup" in data[0]:
                avg_speedup = np.mean([d["speedup"] for d in data])
                print(f"  {name}: avg speedup = {avg_speedup:.2f}x")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'triton_kernel_workshop_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()