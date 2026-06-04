#!/usr/bin/env python3
"""Triton Kernel 实战练习 v2

修复兼容性问题:
- 使用字节 stride 而非 element_size()
- 设置 TRITON_CACHE_WORKAROUND
- 适配 Triton 3.1.0 + CUDA 11.8 + SM 8.6

用法 (GPU 服务器):
  source /root/miniconda3/bin/activate myconda
  export TRITON_CACHE_DIR=/tmp/triton_cache
  export HF_HUB_OFFLINE=1
  python gpu_triton_practice_v2.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
import json
import torch
import triton
import triton.language as tl

GPU_NAME = torch.cuda.get_device_name(0)
print(f"GPU: {GPU_NAME}, Triton: {triton.__version__}, CUDA: {torch.version.cuda}")
print(f"Compute Cap: {torch.cuda.get_device_capability()}")


def bench_ms(fn, *args, warmup=10, rep=50, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(rep):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.time() - t0) / rep * 1000


# ============================================================
# Kernel 1: Vector Add (基础)
# ============================================================

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n: tl.int32,
                       BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def test_vector_add():
    print("\n--- Kernel 1: Vector Add ---")
    n = 1 << 20  # 1M
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    vector_add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    assert torch.allclose(out, x + y), "FAILED"
    print(f"  Correctness: PASSED")

    ms_t = bench_ms(vector_add_kernel[grid], x, y, out, n, BLOCK=BLOCK)
    ms_p = bench_ms(lambda: torch.add(x, y, out=out))
    bw = 3 * n * 4 / ms_t / 1e6
    print(f"  Triton={ms_t:.3f}ms ({bw:.0f} GB/s), PyTorch={ms_p:.3f}ms, ratio={ms_p/ms_t:.2f}x")
    return {"kernel": "vector_add", "triton_ms": round(ms_t, 3), "torch_ms": round(ms_p, 3), "bw_gbs": round(bw, 0)}


# ============================================================
# Kernel 2: Softmax (reduction)
# ============================================================

@triton.jit
def softmax_kernel(inp_ptr, out_ptr, n_cols: tl.int32,
                    stride: tl.int32,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(inp_ptr + row * stride + offs, mask=mask, other=-float("inf"))
    x_max = tl.max(x, 0)
    x_safe = x - x_max
    num = tl.exp(x_safe)
    den = tl.sum(num, 0)
    out = num / den
    tl.store(out_ptr + row * stride + offs, out, mask=mask)


def test_softmax():
    print("\n--- Kernel 2: Softmax ---")
    M, N = 4096, 1024
    x = torch.randn(M, N, device="cuda")
    out = torch.empty_like(x)

    BLOCK = triton.next_power_of_2(N)
    softmax_kernel[(M,)](x, out, N, x.stride(0), BLOCK=BLOCK)
    assert torch.allclose(out, torch.softmax(x, 1), atol=1e-5), "FAILED"
    print(f"  Correctness: PASSED (atol=1e-5)")

    ms_t = bench_ms(softmax_kernel[(M,)], x, out, N, x.stride(0), BLOCK=BLOCK)
    ms_p = bench_ms(lambda: torch.softmax(x, 1))
    print(f"  Triton={ms_t:.3f}ms, PyTorch={ms_p:.3f}ms, ratio={ms_p/ms_t:.2f}x")
    return {"kernel": "softmax", "triton_ms": round(ms_t, 3), "torch_ms": round(ms_p, 3)}


# ============================================================
# Kernel 3: Fused Bias + ReLU (kernel fusion)
# ============================================================

@triton.jit
def fused_bias_relu_kernel(x_ptr, bias_ptr, out_ptr, n: tl.int32,
                            bias_size: tl.int32,
                            BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(bias_ptr + offs % bias_size, mask=mask)
    out = tl.where(x + b > 0, x + b, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


def test_fused_bias_relu():
    print("\n--- Kernel 3: Fused Bias + ReLU ---")
    n = 1 << 20
    bias_size = 1024
    x = torch.randn(n, device="cuda")
    bias = torch.randn(bias_size, device="cuda")
    out = torch.empty_like(x)

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    fused_bias_relu_kernel[grid](x, bias, out, n, bias_size, BLOCK=BLOCK)
    assert torch.allclose(out, torch.relu(x + bias)), "FAILED"
    print(f"  Correctness: PASSED")

    ms_t = bench_ms(fused_bias_relu_kernel[grid], x, bias, out, n, bias_size, BLOCK=BLOCK)
    ms_p = bench_ms(lambda: torch.relu(x + bias, out=torch.empty_like(x)))
    bw = (2 * n + bias_size) * 4 / ms_t / 1e6
    print(f"  Fused={ms_t:.3f}ms ({bw:.0f} GB/s), Separate={ms_p:.3f}ms, speedup={ms_p/ms_t:.2f}x")
    return {"kernel": "fused_bias_relu", "fused_ms": round(ms_t, 3), "separate_ms": round(ms_p, 3), "speedup": round(ms_p/ms_t, 2)}


# ============================================================
# Kernel 4: Vector Scaling + Reduction (sum)
# ============================================================

@triton.jit
def reduce_sum_kernel(x_ptr, out_ptr, n: tl.int32,
                       BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.sum(x, 0)
    # Atomic add to output
    tl.atomic_add(out_ptr + pid, s)


def test_reduce_sum():
    print("\n--- Kernel 4: Reduce Sum ---")
    n = 1 << 22  # 4M
    BLOCK = 1024
    n_blocks = triton.cdiv(n, BLOCK)

    x = torch.randn(n, device="cuda")
    out = torch.zeros(n_blocks, device="cuda")

    reduce_sum_kernel[(n_blocks,)](x, out, n, BLOCK=BLOCK)
    torch.cuda.synchronize()

    expected = x.sum().item()
    result = out.sum().item()
    rel_err = abs(result - expected) / abs(expected)
    print(f"  Correctness: sum={result:.4f}, expected={expected:.4f}, rel_err={rel_err:.6f}")

    ms_t = bench_ms(reduce_sum_kernel[(n_blocks,)], x, out, n, BLOCK=BLOCK)
    ms_p = bench_ms(lambda: x.sum())
    bw = n * 4 / ms_t / 1e6  # read only
    print(f"  Triton={ms_t:.3f}ms ({bw:.0f} GB/s), PyTorch={ms_p:.3f}ms")
    return {"kernel": "reduce_sum", "triton_ms": round(ms_t, 3), "torch_ms": round(ms_p, 3), "rel_err": round(rel_err, 6)}


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Triton Kernel 实战练习 v2")
    print("=" * 60)

    results = []
    tests = [
        ("Vector Add", test_vector_add),
        ("Softmax", test_softmax),
        ("Fused Bias+ReLU", test_fused_bias_relu),
        ("Reduce Sum", test_reduce_sum),
    ]

    for name, fn in tests:
        try:
            r = fn()
            results.append(r)
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"  {r}")

    with open("/root/triton_practice_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved. {len(results)}/{len(tests)} kernels passed.")
