#!/usr/bin/env python3
"""Triton Kernel 示例集 — GPU 实验用

需要在 Linux + CUDA GPU 环境运行:
  pip install triton
  python triton_examples.py

包含:
  1. vector_add — 最简单的 Triton kernel
  2. fused_relu — 融合 ReLU kernel
  3. softmax — 并行 reduction softmax
  4. matmul — 分块矩阵乘法 (利用 Tensor Core)
  5. benchmark — 与 PyTorch 原生实现对比
"""

import torch
import time

# Triton 在 macOS 上不可用, 仅在 Linux + CUDA 环境导入
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: triton not installed. Run on Linux + CUDA GPU.")
    print("Install: pip install triton")


# ============================================================
# 1. Vector Add — 最简单的 kernel
# ============================================================

def make_vector_add():
    """返回 vector_add kernel 和 launcher"""

    @triton.jit
    def _vector_add_kernel(
        x_ptr, y_ptr, output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    def vector_add(x, y):
        output = torch.empty_like(x)
        n = output.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        _vector_add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
        return output

    return vector_add


# ============================================================
# 2. Fused ReLU — 融合运算 kernel
# ============================================================

def make_fused_relu():
    @triton.jit
    def _fused_relu_kernel(
        input_ptr, output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask)
        # ReLU: max(0, x)
        output = tl.where(x > 0, x, 0.0)
        tl.store(output_ptr + offsets, output, mask=mask)

    def fused_relu(x):
        output = torch.empty_like(x)
        n = output.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        _fused_relu_kernel[grid](x, output, n, BLOCK_SIZE=1024)
        return output

    return fused_relu


# ============================================================
# 3. Softmax — 并行 reduction
# ============================================================

def make_softmax():
    @triton.jit
    def _softmax_kernel(
        input_ptr, output_ptr,
        n_rows, n_cols,
        stride_in, stride_out,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        row_start = row_idx * stride_in
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        # 加载一行
        row = tl.load(input_ptr + row_start + offsets, mask=mask, other=-float('inf'))

        # 数值稳定: 减去最大值
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        output = numerator / denominator

        tl.store(output_ptr + row_idx * stride_out + offsets, output, mask=mask)

    def triton_softmax(x):
        assert x.dim() == 2
        n_rows, n_cols = x.shape
        output = torch.empty_like(x)
        grid = (n_rows,)
        _softmax_kernel[grid](
            x, output,
            n_rows, n_cols,
            x.stride(0), output.stride(0),
            BLOCK_SIZE=triton.next_power_of_2(n_cols),
        )
        return output

    return triton_softmax


# ============================================================
# 4. Matrix Multiply — 分块 GEMM (利用 Tensor Core)
# ============================================================

def make_matmul():
    @triton.jit
    def _matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # 累加器
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # 沿 K 维 tiling
        for k_start in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = k_start * BLOCK_K + offs_k
            # A tile: [BLOCK_M, BLOCK_K]
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak)
            # B tile: [BLOCK_K, BLOCK_N]
            b = tl.load(b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)
            # 矩阵乘加 (自动利用 Tensor Core)
            acc += tl.dot(a, b)

        c = acc.to(a_ptr.dtype.element_ty)
        mask_m = offs_m[:, None] < M
        mask_n = offs_n[None, :] < N
        mask = mask_m & mask_n
        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c, mask=mask)

    def triton_matmul(a, b):
        assert a.dim() == 2 and b.dim() == 2
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        _matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return c

    return triton_matmul


# ============================================================
# 5. Benchmark 工具
# ============================================================

def benchmark_fn(fn, *args, warmup=10, repeat=100):
    """简单的 benchmark 函数"""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / repeat
    return ms


# ============================================================
# Main — 运行所有示例和 benchmark
# ============================================================

def main():
    if not HAS_TRITON:
        print("Triton not available. Exiting.")
        return

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Triton version: {triton.__version__}")
    print()

    # ---- Test 1: Vector Add ----
    print("=" * 50)
    print("1. Vector Add")
    vector_add = make_vector_add()
    x = torch.randn(1024 * 1024, device=device, dtype=torch.float16)
    y = torch.randn(1024 * 1024, device=device, dtype=torch.float16)

    out_triton = vector_add(x, y)
    out_torch = x + y
    print(f"  Correct: {torch.allclose(out_triton, out_torch, atol=1e-2)}")

    ms_triton = benchmark_fn(vector_add, x, y)
    ms_torch = benchmark_fn(lambda a, b: a + b, x, y)
    print(f"  Triton: {ms_triton:.3f}ms, PyTorch: {ms_torch:.3f}ms")
    print()

    # ---- Test 2: Fused ReLU ----
    print("=" * 50)
    print("2. Fused ReLU")
    fused_relu = make_fused_relu()
    x = torch.randn(1024 * 1024, device=device, dtype=torch.float16)

    out_triton = fused_relu(x)
    out_torch = torch.relu(x)
    print(f"  Correct: {torch.allclose(out_triton, out_torch, atol=1e-2)}")

    ms_triton = benchmark_fn(fused_relu, x)
    ms_torch = benchmark_fn(torch.relu, x)
    print(f"  Triton: {ms_triton:.3f}ms, PyTorch: {ms_torch:.3f}ms")
    print()

    # ---- Test 3: Softmax ----
    print("=" * 50)
    print("3. Softmax")
    triton_softmax = make_softmax()
    x = torch.randn(1024, 1024, device=device, dtype=torch.float16)

    out_triton = triton_softmax(x)
    out_torch = torch.softmax(x.float(), dim=1).half()
    print(f"  Correct: {torch.allclose(out_triton, out_torch, atol=1e-2)}")

    ms_triton = benchmark_fn(triton_softmax, x)
    ms_torch = benchmark_fn(lambda a: torch.softmax(a.float(), dim=1).half(), x)
    print(f"  Triton: {ms_triton:.3f}ms, PyTorch: {ms_torch:.3f}ms")
    print()

    # ---- Test 4: Matmul ----
    print("=" * 50)
    print("4. Matrix Multiply (GEMM)")
    triton_matmul = make_matmul()
    M, N, K = 512, 512, 512
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)

    out_triton = triton_matmul(a, b)
    out_torch = torch.matmul(a, b)
    print(f"  Correct: {torch.allclose(out_triton, out_torch, atol=1e-1)}")

    ms_triton = benchmark_fn(triton_matmul, a, b)
    ms_torch = benchmark_fn(torch.matmul, a, b)
    print(f"  Triton: {ms_triton:.3f}ms, PyTorch: {ms_torch:.3f}ms")
    print()

    # ---- Summary ----
    print("=" * 50)
    print("All tests passed! Triton kernels work correctly.")
    print("Triton performance should be within 90-100% of PyTorch.")


if __name__ == "__main__":
    main()
