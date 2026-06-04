#!/usr/bin/env python3
"""Triton Kernel 入门实验

测试 Triton 是否可用, 并学习基本 kernel 编写:
1. Vector Add kernel
2. Fused Bias + ReLU kernel
3. Softmax kernel
4. 性能对比: Triton vs PyTorch
"""
import torch
import triton
import triton.language as tl
import time

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Triton: {triton.__version__}")


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def bias_relu_kernel(input_ptr, bias_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets % 768, mask=mask)  # bias broadcast
    result = tl.maximum(x + bias, 0.0)  # ReLU
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Load row
    row = tl.load(input_ptr + row_start + offsets, mask=mask, other=float('-inf'))

    # Subtract max for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Exp
    numerator = tl.exp(row_minus_max)
    # Sum
    denominator = tl.sum(numerator, axis=0)
    # Normalize
    softmax_output = numerator / denominator
    tl.store(output_ptr + row_start + offsets, softmax_output, mask=mask)


def bench(fn, warmup=10, rep=100):
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
# 实验 1: Vector Add — Triton vs PyTorch
# ============================================================
print("\n" + "=" * 60)
print("实验1: Vector Add (Triton vs PyTorch)")
print("=" * 60)

for N in [1024, 1024*1024, 64*1024*1024]:
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.randn(N, device="cuda", dtype=torch.float32)
    output = torch.empty_like(x)

    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    triton_ms = bench(lambda: add_kernel[grid](x, y, output, N, BLOCK_SIZE=BLOCK))
    pytorch_ms = bench(lambda: torch.add(x, y, out=output))

    correct = torch.allclose(output, x + y)
    bw = N * 4 * 3 / triton_ms / 1e6  # GB/s (read 2 + write 1)

    print(f"\n  N={N//1024}K: Triton={triton_ms:.4f}ms, PyTorch={pytorch_ms:.4f}ms, "
          f"ratio={triton_ms/pytorch_ms:.2f}, BW={bw:.0f}GB/s, correct={correct}")


# ============================================================
# 实验 2: Fused Bias + ReLU — Triton vs PyTorch
# ============================================================
print("\n" + "=" * 60)
print("实验2: Fused Bias + ReLU")
print("=" * 60)

for B, S, H in [(1, 1024, 768), (8, 512, 1024), (32, 256, 2048)]:
    N = B * S * H
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    bias = torch.randn(H, device="cuda", dtype=torch.float32)
    output = torch.empty_like(x)

    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    triton_ms = bench(lambda: bias_relu_kernel[grid](x, bias, output, N, BLOCK_SIZE=BLOCK))

    # PyTorch unfused
    pytorch_unfused_ms = bench(lambda: torch.maximum(x + bias, torch.zeros_like(x)))

    # PyTorch fused
    pytorch_fused_ms = bench(lambda: F.relu(x + bias))

    speedup = pytorch_unfused_ms / triton_ms

    print(f"\n  ({B}x{S}x{H}): Triton={triton_ms:.4f}ms, "
          f"PyTorch unfused={pytorch_unfused_ms:.4f}ms, "
          f"speedup={speedup:.2f}x")


# ============================================================
# 实验 3: Softmax — Triton vs PyTorch
# ============================================================
print("\n" + "=" * 60)
print("实验3: Softmax (Triton vs PyTorch)")
print("=" * 60)

import torch.nn.functional as F

for B, N in [(1, 1024), (32, 4096), (128, 32768)]:
    x = torch.randn(B, N, device="cuda", dtype=torch.float32)
    output = torch.empty_like(x)

    BLOCK = triton.next_power_of_2(N)

    triton_ms = bench(lambda: softmax_kernel[(B,)](x, output, B, N, BLOCK_SIZE=BLOCK))
    pytorch_ms = bench(lambda: F.softmax(x, dim=-1))

    correct = torch.allclose(output, F.softmax(x, dim=-1), atol=1e-5)

    print(f"\n  ({B}x{N}): Triton={triton_ms:.4f}ms, PyTorch={pytorch_ms:.4f}ms, "
          f"ratio={triton_ms/pytorch_ms:.2f}, correct={correct}")


print("\nDone!")
