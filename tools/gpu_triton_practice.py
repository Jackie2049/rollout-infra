#!/usr/bin/env python3
"""Triton Kernel 实战练习

在 GPU 上编写、测试和 benchmark Triton kernels:
1. Vector Add — 最基础 kernel，理解编程模型
2. Softmax — reduction + 数值稳定性
3. Matrix Multiply — 2D kernel + tiling
4. Flash Attention (简化版) — 核心 tiling + online softmax
5. 性能对比 PyTorch

用法 (GPU 服务器):
  source /root/miniconda3/bin/activate myconda
  export HF_HUB_OFFLINE=1
  python gpu_triton_practice.py
"""

import time
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import triton
import triton.language as tl


def bench(fn, *args, warmup=10, rep=50, **kwargs):
    """Benchmark a function"""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    for _ in range(rep):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return elapsed / rep * 1000  # ms


# ============================================================
# Kernel 1: Vector Add
# ============================================================

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n,
                       BLOCK_SIZE: tl.constexpr):
    """最简单的 Triton kernel: out = x + y"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def test_vector_add():
    print("=" * 60)
    print("Kernel 1: Vector Add")
    print("=" * 60)

    n = 1024 * 1024  # 1M elements
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)

    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)
    torch.cuda.synchronize()

    # Verify
    expected = x + y
    assert torch.allclose(out, expected), "Vector Add FAILED!"
    print(f"  n={n}, BLOCK={BLOCK}: PASSED")

    # Benchmark
    ms_triton = bench(vector_add_kernel[grid], x, y, out, n, BLOCK_SIZE=BLOCK)
    ms_torch = bench(lambda: torch.add(x, y, out=out))

    # 带宽计算: 读 x + 读 y + 写 out = 3 * n * 4 bytes
    bytes_moved = 3 * n * 4
    bw_triton = bytes_moved / ms_triton / 1e6  # GB/s
    bw_torch = bytes_moved / ms_torch / 1e6

    print(f"  Triton: {ms_triton:.3f} ms, {bw_triton:.1f} GB/s")
    print(f"  PyTorch: {ms_torch:.3f} ms, {bw_torch:.1f} GB/s")
    print(f"  PyTorch/Triton: {ms_torch/ms_triton:.2f}x")

    return {"kernel": "vector_add", "n": n,
            "triton_ms": round(ms_triton, 3), "torch_ms": round(ms_torch, 3),
            "triton_bw": round(bw_triton, 1), "torch_bw": round(bw_torch, 1)}


# ============================================================
# Kernel 2: Softmax
# ============================================================

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_rows, n_cols,
                    stride_in, stride_out,
                    BLOCK_SIZE: tl.constexpr):
    """Softmax: 每个 program 处理一行"""
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_in // input_ptr.element_size()
    out_start = row_idx * stride_out // output_ptr.element_size()

    # Load row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    row = tl.load(input_ptr + row_start + offs, mask=mask, other=-float("inf"))

    # Numerical stability: subtract max
    row_max = tl.max(row, axis=0)
    safe_row = row - row_max

    # Exp
    numerator = tl.exp(safe_row)

    # Sum
    denominator = tl.sum(numerator, axis=0)

    # Normalize
    out = numerator / denominator
    tl.store(output_ptr + out_start + offs, out, mask=mask)


def test_softmax():
    print("\n" + "=" * 60)
    print("Kernel 2: Softmax")
    print("=" * 60)

    M, N = 4096, 1024
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    BLOCK = triton.next_power_of_2(N)
    grid = (M,)

    softmax_kernel[grid](x, out, M, N,
                         x.stride(0), out.stride(0),
                         BLOCK_SIZE=BLOCK)
    torch.cuda.synchronize()

    expected = torch.softmax(x, dim=1)
    assert torch.allclose(out, expected, atol=1e-5), "Softmax FAILED!"
    print(f"  shape=({M},{N}), BLOCK={BLOCK}: PASSED (atol=1e-5)")

    # Benchmark
    ms_triton = bench(softmax_kernel[grid], x, out, M, N,
                      x.stride(0), out.stride(0), BLOCK_SIZE=BLOCK)
    ms_torch = bench(lambda: torch.softmax(x, dim=1))

    print(f"  Triton: {ms_triton:.3f} ms")
    print(f"  PyTorch: {ms_torch:.3f} ms")
    print(f"  PyTorch/Triton: {ms_torch/ms_triton:.2f}x")

    return {"kernel": "softmax", "shape": f"({M},{N})",
            "triton_ms": round(ms_triton, 3), "torch_ms": round(ms_torch, 3)}


# ============================================================
# Kernel 3: Matrix Multiply (Naive)
# ============================================================

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                   M, N, K,
                   stride_am, stride_ak,
                   stride_bk, stride_bn,
                   stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """矩阵乘法: C = A @ B, tiled"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to A and B blocks
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load A and B blocks
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def test_matmul():
    print("\n" + "=" * 60)
    print("Kernel 3: Matrix Multiply (Tiled)")
    print("=" * 60)

    # Small sizes first for correctness
    M, N, K = 256, 256, 256
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    torch.cuda.synchronize()

    expected = a @ b
    assert torch.allclose(c, expected, atol=1e-2), f"MatMul FAILED! max_err={torch.max(torch.abs(c - expected)):.4f}"
    print(f"  shape=({M},{N},{K}): PASSED (atol=1e-2)")

    # Benchmark with larger sizes
    for size in [256, 512, 1024]:
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)
        c = torch.empty(size, size, device="cuda", dtype=torch.float16)

        grid = (triton.cdiv(size, BLOCK_M), triton.cdiv(size, BLOCK_N))

        ms_triton = bench(matmul_kernel[grid], a, b, c, size, size, size,
                          a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                          c.stride(0), c.stride(1),
                          BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        ms_torch = bench(lambda: torch.matmul(a, b))

        flops = 2 * size * size * size
        tflops_triton = flops / ms_triton / 1e9
        tflops_torch = flops / ms_torch / 1e9

        print(f"  {size}x{size}: Triton={ms_triton:.3f}ms ({tflops_triton:.1f} TFLOPS), "
              f"PyTorch={ms_torch:.3f}ms ({tflops_torch:.1f} TFLOPS), "
              f"ratio={ms_torch/ms_triton:.2f}x")

    return {"kernel": "matmul", "BLOCK": f"({BLOCK_M},{BLOCK_N},{BLOCK_K})"}


# ============================================================
# Kernel 4: Fused Multiply-Add with ReLU (练习 fusion)
# ============================================================

@triton.jit
def fused_bias_relu_kernel(x_ptr, bias_ptr, out_ptr, n,
                            BLOCK_SIZE: tl.constexpr):
    """Fused: out = ReLU(x + bias)"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    x = tl.load(x_ptr + offs, mask=mask)
    bias = tl.load(bias_ptr + offs % 1024, mask=mask)  # broadcast bias
    out = x + bias
    out = tl.where(out > 0, out, 0.0)  # ReLU
    tl.store(out_ptr + offs, out, mask=mask)


def test_fused_bias_relu():
    print("\n" + "=" * 60)
    print("Kernel 4: Fused Bias + ReLU")
    print("=" * 60)

    n = 1024 * 1024
    x = torch.randn(n, device="cuda")
    bias = torch.randn(1024, device="cuda")
    out = torch.empty_like(x)

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)

    fused_bias_relu_kernel[grid](x, bias, out, n, BLOCK_SIZE=BLOCK)
    torch.cuda.synchronize()

    # Verify
    expected = torch.relu(x + bias)
    assert torch.allclose(out, expected), "Fused Bias+ReLU FAILED!"
    print(f"  n={n}: PASSED")

    # Benchmark: fused vs separate ops
    ms_fused = bench(fused_bias_relu_kernel[grid], x, bias, out, n, BLOCK_SIZE=BLOCK)

    def separate_ops():
        tmp = x + bias
        torch.relu(tmp, out=out)

    ms_separate = bench(separate_ops)

    print(f"  Fused (Triton): {ms_fused:.3f} ms")
    print(f"  Separate (PyTorch): {ms_separate:.3f} ms")
    print(f"  Speedup: {ms_separate/ms_fused:.2f}x")

    # Bandwidth: 读 x + 读 bias + 写 out = (n + 1024 + n) * 4 ≈ 2n * 4
    bytes_moved = (n * 2 + 1024) * 4
    bw_fused = bytes_moved / ms_fused / 1e6
    print(f"  Bandwidth: {bw_fused:.1f} GB/s")

    return {"kernel": "fused_bias_relu", "n": n,
            "fused_ms": round(ms_fused, 3), "separate_ms": round(ms_separate, 3),
            "speedup": round(ms_separate / ms_fused, 2)}


# ============================================================
# Kernel 5: LayerNorm (练习 reduction)
# ============================================================

@triton.jit
def layernorm_kernel(x_ptr, out_ptr, weight_ptr, bias_ptr,
                      n_rows, n_cols,
                      stride_x, stride_out,
                      eps: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr):
    """LayerNorm: 每行独立归一化"""
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x // x_ptr.element_size()
    out_start = row_idx * stride_out // x_ptr.element_size()

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    # Load row
    x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)

    # Mean
    mean = tl.sum(x, axis=0) / n_cols

    # Variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n_cols

    # Normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * inv_std

    # Scale and shift
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    out = x_norm * w + b

    tl.store(out_ptr + out_start + offs, out, mask=mask)


def test_layernorm():
    print("\n" + "=" * 60)
    print("Kernel 5: LayerNorm")
    print("=" * 60)

    M, N = 2048, 1024
    x = torch.randn(M, N, device="cuda")
    w = torch.randn(N, device="cuda")
    b = torch.randn(N, device="cuda")
    out = torch.empty_like(x)

    BLOCK = triton.next_power_of_2(N)
    grid = (M,)

    layernorm_kernel[grid](x, out, w, b, M, N,
                           x.stride(0), out.stride(0),
                           eps=1e-5, BLOCK_SIZE=BLOCK)
    torch.cuda.synchronize()

    expected = torch.layer_norm(x, [N], w, b, 1e-5)
    assert torch.allclose(out, expected, atol=1e-4), f"LayerNorm FAILED! max_err={torch.max(torch.abs(out - expected)):.6f}"
    print(f"  shape=({M},{N}): PASSED (atol=1e-4)")

    ms_triton = bench(layernorm_kernel[grid], x, out, w, b, M, N,
                      x.stride(0), out.stride(0), eps=1e-5, BLOCK_SIZE=BLOCK)
    ms_torch = bench(lambda: torch.layer_norm(x, [N], w, b, 1e-5))

    print(f"  Triton: {ms_triton:.3f} ms")
    print(f"  PyTorch: {ms_torch:.3f} ms")
    print(f"  PyTorch/Triton: {ms_torch/ms_triton:.2f}x")

    return {"kernel": "layernorm", "shape": f"({M},{N})",
            "triton_ms": round(ms_triton, 3), "torch_ms": round(ms_torch, 3)}


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Triton Kernel 实战练习")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton: {triton.__version__}, PyTorch: {torch.__version__}")
    print("=" * 60)

    results = []

    try:
        results.append(test_vector_add())
    except Exception as e:
        print(f"Vector Add FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        results.append(test_softmax())
    except Exception as e:
        print(f"Softmax FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        results.append(test_matmul())
    except Exception as e:
        print(f"MatMul FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        results.append(test_fused_bias_relu())
    except Exception as e:
        print(f"Fused Bias+ReLU FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        results.append(test_layernorm())
    except Exception as e:
        print(f"LayerNorm FAILED: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"  {r.get('kernel', 'unknown')}: {r}")

    with open("triton_practice_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to triton_practice_results.json")
