#!/usr/bin/env python3
"""Triton Fused QKV Projection Kernel v2 — tl.dot() Tiled Matmul

Production-style Triton QKV kernel using tl.dot() for proper tiled GEMM.
This addresses the root cause of v1's 0.19-0.57x slowdown:
- v1 used naive for-loop row-level matmul → cuBLAS tiled GEMM beats it 5x
- v2 uses tl.dot() for tiled matmul → mathematically same algorithm as cuBLAS

4 experiments:
1. Correctness: Triton tiled vs PyTorch (cos_sim + max_diff)
2. Performance: v2(tl.dot) vs v1(naive) vs PyTorch sequential/stacked
3. Block size tuning: BLOCK_M/BLOCK_K/BLOCK_N sweep
4. Decode vs prefill: QKV in real inference scenarios

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u tools/triton_fused_qkv_v2.py
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("WARNING: Triton not available. Exiting.")
    sys.exit(1)


def measure_time(fn, warmup=5, repeat=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return np.median(times), np.mean(times), np.std(times)


# ============================================================
# Triton Tiled Matmul Kernel using tl.dot()
# ============================================================

@triton.jit
def fused_qkv_matmul_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_x_m, stride_x_k,
    stride_w_n, stride_w_k,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled matmul kernel for fused QKV projection using tl.dot().
    x: [M, K] input (flattened B*S × D)
    w: [N, K] weight (stacked QKV weights)
    out: [M, N] output (stacked QKV outputs)

    Computes: out = x @ w^T  (equivalent to F.linear(x, w))
    Uses tl.dot() for proper tiled GEMM — same algorithm as cuBLAS.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in FP32
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Tiled matmul: accumulate over K dimension
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load x block: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load w block: [BLOCK_K, BLOCK_N] (transposed layout)
        # w[n, k] → w^T[k, n] → we load w at (offs_n, offs_k) as (offs_k, offs_n)
        w_ptrs = w_ptr + offs_n[None, :] * stride_w_n + offs_k[:, None] * stride_w_k
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # tl.dot: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] → [BLOCK_M, BLOCK_N]
        # Both inputs must be same dtype, accumulation in FP32
        acc += tl.dot(x_vals, w_vals, allow_tf32=False)

    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_tiled_qkv(x, wq, wk, wv, num_heads_q, num_heads_kv, head_dim,
                      BLOCK_M=32, BLOCK_N=32, BLOCK_K=32):
    """Fused QKV using tl.dot() tiled matmul — production approach."""
    B, S, D = x.shape
    M = B * S
    K = D

    # Stack weights: [N_TOTAL, K]
    w_stacked = torch.cat([wq, wk, wv], dim=0)
    N_TOTAL = w_stacked.shape[0]

    # Flatten input: [M, K]
    x_flat = x.reshape(M, D).contiguous()

    # Allocate output: [M, N_TOTAL]
    out_flat = torch.empty(M, N_TOTAL, device=x.device, dtype=x.dtype)

    # Launch kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_TOTAL, BLOCK_N))
    fused_qkv_matmul_kernel[grid](
        x_flat, w_stacked, out_flat,
        M, N_TOTAL, K,
        stride_x_m=D, stride_x_k=1,
        stride_w_n=D, stride_w_k=1,
        stride_out_m=N_TOTAL, stride_out_n=1,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # Split output into Q, K, V
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim
    q = out_flat[:, :q_size].reshape(B, S, num_heads_q, head_dim)
    k = out_flat[:, q_size:q_size+kv_size].reshape(B, S, num_heads_kv, head_dim)
    v = out_flat[:, q_size+kv_size:].reshape(B, S, num_heads_kv, head_dim)

    return q, k, v


def pytorch_sequential_qkv(x, wq, wk, wv, num_heads_q, num_heads_kv, head_dim):
    """Sequential QKV: 3 separate matmuls."""
    B, S, D = x.shape
    q = torch.nn.functional.linear(x, wq).reshape(B, S, num_heads_q, head_dim)
    k = torch.nn.functional.linear(x, wk).reshape(B, S, num_heads_kv, head_dim)
    v = torch.nn.functional.linear(x, wv).reshape(B, S, num_heads_kv, head_dim)
    return q, k, v


def pytorch_stacked_qkv(x, wq, wk, wv, num_heads_q, num_heads_kv, head_dim):
    """Stacked QKV: single matmul with stacked weights."""
    B, S, D = x.shape
    w_stacked = torch.cat([wq, wk, wv], dim=0)
    out = torch.nn.functional.linear(x, w_stacked)
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim
    q = out[:, :, :q_size].reshape(B, S, num_heads_q, head_dim)
    k = out[:, :, q_size:q_size+kv_size].reshape(B, S, num_heads_kv, head_dim)
    v = out[:, :, q_size+kv_size:].reshape(B, S, num_heads_kv, head_dim)
    return q, k, v


# ============================================================
# Experiments
# ============================================================

def exp1_correctness(device):
    """Verify Triton tiled QKV correctness."""
    print("\n" + "="*60)
    print("Exp 1: Triton Tiled QKV Correctness (tl.dot())")
    print("="*60)

    results = {}
    configs = [
        (4, 128, 256, 8, 4, 32, "MHA_256"),
        (4, 128, 256, 8, 8, 32, "GQA_equal"),
        (16, 512, 512, 16, 4, 32, "large_GQA"),
        (1, 64, 256, 8, 2, 32, "small"),
        (1, 1, 256, 8, 4, 32, "decode_B1"),
    ]

    for B, S, D, H_Q, H_KV, D_HEAD, label in configs:
        x = torch.randn(B, S, D, device=device, dtype=torch.float16)
        wq = torch.randn(H_Q * D_HEAD, D, device=device, dtype=torch.float16)
        wk = torch.randn(H_KV * D_HEAD, D, device=device, dtype=torch.float16)
        wv = torch.randn(H_KV * D_HEAD, D, device=device, dtype=torch.float16)

        # PyTorch reference
        q_ref, k_ref, v_ref = pytorch_sequential_qkv(x, wq, wk, wv, H_Q, H_KV, D_HEAD)

        # Triton tiled
        q_tri, k_tri, v_tri = triton_tiled_qkv(x, wq, wk, wv, H_Q, H_KV, D_HEAD)

        # Compare (cast to float32 for comparison)
        for name, ref, tri in [("Q", q_ref, q_tri), ("K", k_ref, k_tri), ("V", v_ref, v_tri)]:
            cos = torch.nn.functional.cosine_similarity(
                ref.float().reshape(-1), tri.float().reshape(-1), dim=0).item()
            diff = (ref.float() - tri.float()).abs().max().item()
            results[f"{label}_{name}"] = {'cos_sim': cos, 'max_diff': diff}
            print(f"  {label} {name}: cos={cos:.6f}, diff={diff:.6f}")

    return results


def exp2_performance(device):
    """Benchmark: Triton tiled vs sequential vs stacked."""
    print("\n" + "="*60)
    print("Exp 2: QKV Performance Comparison — tl.dot() vs Naive vs PyTorch")
    print("="*60)

    results = {}
    configs = [
        # (B, S, D, H_Q, H_KV, D_HEAD, label)
        (1, 64, 256, 8, 4, 32, "B1_S64_D256"),
        (4, 128, 256, 8, 4, 32, "B4_S128_D256"),
        (16, 256, 256, 8, 4, 32, "B16_S256_D256"),
        (32, 256, 256, 8, 4, 32, "B32_S256_D256"),
        (64, 256, 256, 8, 4, 32, "B64_S256_D256"),
        (4, 128, 512, 16, 4, 32, "B4_D512_GQA4"),
        (4, 128, 1024, 16, 8, 64, "B4_D1024_GQA8"),
    ]

    for B, S, D, H_Q, H_KV, D_HEAD, label in configs:
        x = torch.randn(B, S, D, device=device, dtype=torch.float16)
        wq = torch.randn(H_Q * D_HEAD, D, device=device, dtype=torch.float16)
        wk = torch.randn(H_KV * D_HEAD, D, device=device, dtype=torch.float16)
        wv = torch.randn(H_KV * D_HEAD, D, device=device, dtype=torch.float16)

        # PyTorch sequential
        med_seq, _, _ = measure_time(
            lambda: pytorch_sequential_qkv(x, wq, wk, wv, H_Q, H_KV, D_HEAD))

        # PyTorch stacked
        med_stack, _, _ = measure_time(
            lambda: pytorch_stacked_qkv(x, wq, wk, wv, H_Q, H_KV, D_HEAD))

        # Triton tiled (tl.dot)
        # Auto-select block sizes based on problem size
        M = B * S
        N = (H_Q + 2 * H_KV) * D_HEAD
        BM = min(32, triton.next_power_of_2(M)) if M < 32 else 32
        BN = min(64, triton.next_power_of_2(N)) if N < 64 else 64
        BK = min(32, triton.next_power_of_2(D))

        med_tri, _, _ = measure_time(
            lambda: triton_tiled_qkv(x, wq, wk, wv, H_Q, H_KV, D_HEAD,
                                     BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK))

        tri_speedup = med_seq / med_tri
        stack_speedup = med_seq / med_stack

        results[label] = {
            'config': {'B': B, 'S': S, 'D': D, 'H_Q': H_Q, 'H_KV': H_KV, 'D_HEAD': D_HEAD,
                       'M': M, 'N': N, 'K': D, 'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK},
            'sequential_ms': med_seq,
            'stacked_ms': med_stack,
            'stacked_speedup': stack_speedup,
            'triton_tiled_ms': med_tri,
            'triton_tiled_speedup': tri_speedup,
        }
        print(f"  {label}: Seq={med_seq:.3f}ms, Stack={med_stack:.3f}ms({stack_speedup:.2f}x), "
              f"Triton_dot={med_tri:.3f}ms({tri_speedup:.2f}x)")

    return results


def exp3_block_tuning(device):
    """Block size tuning: sweep BLOCK_M/BLOCK_K/BLOCK_N."""
    print("\n" + "="*60)
    print("Exp 3: Block Size Tuning for tl.dot() Kernel")
    print("="*60)

    results = {}
    B, S, D, H_Q, H_KV, D_HEAD = 16, 256, 256, 8, 4, 32

    x = torch.randn(B, S, D, device=device, dtype=torch.float16)
    wq = torch.randn(H_Q * D_HEAD, D, device=device, dtype=torch.float16)
    wk = torch.randn(H_KV * D_HEAD, D, device=device, dtype=torch.float16)
    wv = torch.randn(H_KV * D_HEAD, D, device=device, dtype=torch.float16)

    # PyTorch baseline
    med_seq, _, _ = measure_time(
        lambda: pytorch_sequential_qkv(x, wq, wk, wv, H_Q, H_KV, D_HEAD))

    block_configs = [
        # (BM, BK, BN)
        (16, 16, 16),
        (16, 32, 32),
        (32, 32, 32),
        (32, 32, 64),
        (32, 64, 64),
        (64, 32, 32),
        (64, 32, 64),
        (64, 64, 64),
        (128, 32, 32),
        (128, 32, 64),
    ]

    for BM, BK, BN in block_configs:
        label = f"BM{BM}_BK{BK}_BN{BN}"
        try:
            med_tri, _, _ = measure_time(
                lambda: triton_tiled_qkv(x, wq, wk, wv, H_Q, H_KV, D_HEAD,
                                         BLOCK_M=BM, BLOCK_K=BK, BLOCK_N=BN))
            speedup = med_seq / med_tri
            results[label] = {
                'BLOCK_M': BM, 'BLOCK_K': BK, 'BLOCK_N': BN,
                'triton_ms': med_tri, 'speedup': speedup,
            }
            print(f"  {label}: {med_tri:.3f}ms ({speedup:.2f}x)")
        except Exception as e:
            results[label] = {'error': str(e)}
            print(f"  {label}: FAILED ({e})")

    return results


def exp4_decode_prefill(device):
    """QKV projection in decode vs prefill scenarios."""
    print("\n" + "="*60)
    print("Exp 4: QKV Projection — Decode vs Prefill (tl.dot())")
    print("="*60)

    results = {}
    D = 256
    H_Q = 8
    H_KV = 4
    D_HEAD = 32

    scenarios = [
        (1, 1, "decode_B1_S1"),
        (1, 8, "decode_B1_S8"),
        (1, 32, "decode_B1_S32"),
        (4, 128, "prefill_B4_S128"),
        (16, 512, "prefill_B16_S512"),
        (64, 1024, "prefill_B64_S1024"),
    ]

    for B, S, label in scenarios:
        x = torch.randn(B, S, D, device=device, dtype=torch.float16)
        wq = torch.randn(H_Q * D_HEAD, D, device=device, dtype=torch.float16)
        wk = torch.randn(H_KV * D_HEAD, D, device=device, dtype=torch.float16)
        wv = torch.randn(H_KV * D_HEAD, D, device=device, dtype=torch.float16)

        med_seq, _, _ = measure_time(
            lambda: pytorch_sequential_qkv(x, wq, wk, wv, H_Q, H_KV, D_HEAD))
        med_stack, _, _ = measure_time(
            lambda: pytorch_stacked_qkv(x, wq, wk, wv, H_Q, H_KV, D_HEAD))

        M = B * S
        N = (H_Q + 2 * H_KV) * D_HEAD
        BM = min(32, triton.next_power_of_2(M)) if M < 32 else 32
        BN = min(64, triton.next_power_of_2(N)) if N < 64 else 64

        med_tri, _, _ = measure_time(
            lambda: triton_tiled_qkv(x, wq, wk, wv, H_Q, H_KV, D_HEAD,
                                     BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=32))

        is_decode = (B * S <= 32)

        results[label] = {
            'B': B, 'S': S, 'M': M, 'N': N,
            'sequential_ms': med_seq,
            'stacked_ms': med_stack,
            'triton_tiled_ms': med_tri,
            'triton_speedup': med_seq / med_tri,
            'stacked_speedup': med_seq / med_stack,
            'is_decode': is_decode,
        }
        print(f"  {label}: Seq={med_seq:.3f}ms, Stack={med_stack:.3f}ms({med_seq/med_stack:.2f}x), "
              f"Triton={med_tri:.3f}ms({med_seq/med_tri:.2f}x), decode={is_decode}")

    return results


def main():
    device = torch.device('cuda:0')
    print("=" * 70)
    print("Triton Fused QKV Projection Kernel v2 — tl.dot() Tiled Matmul")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"\nKey: v1(naive for-loop)=0.19-0.57x vs v2(tl.dot tiled)=? vs cuBLAS")

    all_results = {}
    all_results['exp1_correctness'] = exp1_correctness(device)
    all_results['exp2_performance'] = exp2_performance(device)
    all_results['exp3_block_tuning'] = exp3_block_tuning(device)
    all_results['exp4_decode_prefill'] = exp4_decode_prefill(device)

    # Summary
    print("\n" + "=" * 70)
    print("Summary: QKV v2 (tl.dot()) vs v1 (naive) vs PyTorch")
    print("=" * 70)

    if 'exp1_correctness' in all_results:
        e1 = all_results['exp1_correctness']
        all_pass = all(v['cos_sim'] > 0.999 for v in e1.values() if isinstance(v, dict) and 'cos_sim' in v)
        print(f"  Exp1 — Correctness: {'ALL PASS' if all_pass else 'FAILED'}")
        for key, val in e1.items():
            if isinstance(val, dict) and 'cos_sim' in val:
                print(f"    {key}: cos={val['cos_sim']:.6f}, diff={val['max_diff']:.6f}")

    if 'exp2_performance' in all_results:
        e2 = all_results['exp2_performance']
        print("  Exp2 — Performance:")
        for key, val in e2.items():
            if isinstance(val, dict) and 'triton_tiled_speedup' in val:
                print(f"    {key}: Triton_dot {val['triton_tiled_speedup']:.2f}x, "
                      f"Stack {val['stacked_speedup']:.2f}x")

    if 'exp3_block_tuning' in all_results:
        e3 = all_results['exp3_block_tuning']
        print("  Exp3 — Block Tuning:")
        best = None
        for key, val in e3.items():
            if isinstance(val, dict) and 'speedup' in val:
                print(f"    {key}: {val['speedup']:.2f}x")
                if best is None or val['speedup'] > best['speedup']:
                    best = val
        if best:
            print(f"    → Best: BM={best['BLOCK_M']},BK={best['BLOCK_K']},BN={best['BLOCK_N']} "
                  f"({best['speedup']:.2f}x)")

    if 'exp4_decode_prefill' in all_results:
        e4 = all_results['exp4_decode_prefill']
        print("  Exp4 — Decode vs Prefill:")
        for key, val in e4.items():
            if isinstance(val, dict) and 'triton_speedup' in val:
                print(f"    {key}: Triton {val['triton_speedup']:.2f}x, "
                      f"Stack {val['stacked_speedup']:.2f}x, "
                      f"decode={val['is_decode']}")

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', 'triton_fused_qkv_v2.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()