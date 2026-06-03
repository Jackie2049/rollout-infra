"""Triton Kernel 实战 — Fused LayerNorm + Residual + GELU

实现一个融合的 LayerNorm kernel，对比:
1. PyTorch eager 分解执行 (3个独立 kernel)
2. Triton 融合 kernel (1个 kernel)

这是推理优化的核心技术 — 融合多个小操作为一个 kernel，
减少 kernel launch overhead 和 HBM 读写次数。

使用方法:
    python triton_layernorm_bench.py   # 需要在 GPU 上运行

要求:
    pip install triton
"""

import torch
import time

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def benchmark(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


# ============================================================
# PyTorch Eager 实现
# ============================================================
def torch_fused_layernorm(x, residual, weight, bias, eps=1e-5):
    """PyTorch eager: 3个独立 kernel calls.
    1. x + residual (elementwise add)
    2. LayerNorm (mean, var, normalize, scale, shift)
    3. GELU activation
    """
    hidden = x + residual
    mean = hidden.mean(dim=-1, keepdim=True)
    var = hidden.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (hidden - mean) / torch.sqrt(var + eps)
    output = normalized * weight + bias
    return torch.nn.functional.gelu(output)


# ============================================================
# Triton 融合 Kernel 实现
# ============================================================
if HAS_TRITON:
    @triton.jit
    def _layernorm_gelu_kernel(
        X_ptr, R_ptr, W_ptr, B_ptr, Out_ptr,
        stride_x_batch, stride_x_seq, stride_x_hidden,
        stride_r_batch, stride_r_seq, stride_r_hidden,
        N: tl.constexpr,  # hidden dim (compile-time constant)
        BLOCK_SIZE: tl.constexpr,
        eps: tl.constexpr,
    ):
        """Fused LayerNorm + Residual + GELU kernel.

        Each program instance processes one row (one token's hidden state).
        """
        # Program ID → which row to process
        row_idx = tl.program_id(0)
        batch_idx = row_idx // stride_x_seq
        seq_idx = row_idx % stride_x_seq  # This doesn't work for multi-dim, simplified

        # Compute base pointers for this row
        row_start_x = X_ptr + row_idx * stride_x_hidden
        row_start_r = R_ptr + row_idx * stride_r_hidden
        row_start_out = Out_ptr + row_idx * stride_x_hidden

        # Load x + residual
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        x = tl.load(row_start_x + offsets, mask=mask, other=0.0)
        r = tl.load(row_start_r + offsets, mask=mask, other=0.0)
        hidden = x + r

        # Compute mean
        mean = tl.sum(hidden, axis=0) / N

        # Compute variance
        centered = hidden - mean
        var = tl.sum(centered * centered, axis=0) / N

        # Normalize
        rstd = 1.0 / tl.sqrt(var + eps)
        normalized = centered * rstd

        # Scale and shift
        w = tl.load(W_ptr + offsets, mask=mask, other=1.0)
        b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
        output = normalized * w + b

        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # Triton doesn't have tanh, use sigmoid approximation
        # GELU(x) ≈ x * sigmoid(1.702 * x)
        gelu = output * tl.sigmoid(1.702 * output)

        # Store result
        tl.store(row_start_out + offsets, gelu, mask=mask)

    def triton_fused_layernorm(x, residual, weight, bias, eps=1e-5):
        """Triton fused kernel wrapper."""
        assert x.is_contiguous()
        assert residual.is_contiguous()

        batch_seq = x.shape[0] * x.shape[1]
        hidden = x.shape[2]
        output = torch.empty_like(x)

        # Launch: one program per row
        grid = (batch_seq,)
        BLOCK_SIZE = triton.next_power_of_2(hidden)

        _layernorm_gelu_kernel[grid](
            x, residual, weight, bias, output,
            x.stride(0), x.stride(1), x.stride(2),
            residual.stride(0), residual.stride(1), residual.stride(2),
            N=hidden,
            BLOCK_SIZE=BLOCK_SIZE,
            eps=eps,
        )
        return output


def main():
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"=== Triton Kernel 实战 — {props.name} ===\n")

    if not HAS_TRITON:
        print("ERROR: triton not installed. Run: pip install triton")
        return

    # ============================================================
    # 实验 1: 正确性验证
    # ============================================================
    print("=" * 60)
    print("实验 1: 正确性验证 (Triton vs PyTorch)")
    print("=" * 60)

    torch.manual_seed(42)
    batch, seq, hidden = 4, 8, 768
    x = torch.randn(batch, seq, hidden, dtype=torch.float32, device=device)
    residual = torch.randn(batch, seq, hidden, dtype=torch.float32, device=device)
    weight = torch.randn(hidden, dtype=torch.float32, device=device)
    bias = torch.randn(hidden, dtype=torch.float32, device=device)

    out_torch = torch_fused_layernorm(x, residual, weight, bias)
    out_triton = triton_fused_layernorm(x, residual, weight, bias)

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        out_torch.flatten().unsqueeze(0),
        out_triton.flatten().unsqueeze(0)
    ).item()
    max_diff = (out_torch - out_triton).abs().max().item()
    mean_diff = (out_torch - out_triton).abs().mean().item()

    print(f"  余弦相似度: {cos_sim:.6f} (1.0 = 完美)")
    print(f"  最大绝对误差: {max_diff:.6f}")
    print(f"  平均绝对误差: {mean_diff:.6f}")

    if cos_sim > 0.99:
        print("  ✅ 正确性验证通过")
    else:
        print("  ❌ 正确性验证失败，需要调试")
        return

    # ============================================================
    # 实验 2: 性能对比
    # ============================================================
    print()
    print("=" * 60)
    print("实验 2: 性能对比 — Eager vs Triton Fused")
    print("=" * 60)

    configs = [
        (1, 1, 768,    "Batch=1, Seq=1 (Decode)"),
        (1, 128, 768,  "Batch=1, Seq=128"),
        (4, 1, 768,    "Batch=4, Seq=1"),
        (4, 128, 768,  "Batch=4, Seq=128"),
        (16, 1, 768,   "Batch=16, Seq=1"),
        (16, 128, 768, "Batch=16, Seq=128"),
        (32, 512, 768, "Batch=32, Seq=512"),
        (32, 1024, 768, "Batch=32, Seq=1024"),
    ]

    print(f"  {'配置':<30} {'Eager (ms)':>12} {'Triton (ms)':>12} {'Speedup':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")

    for batch, seq, hidden, label in configs:
        x = torch.randn(batch, seq, hidden, dtype=torch.float32, device=device)
        residual = torch.randn(batch, seq, hidden, dtype=torch.float32, device=device)
        weight = torch.randn(hidden, dtype=torch.float32, device=device)
        bias = torch.randn(hidden, dtype=torch.float32, device=device)

        t_eager = benchmark(
            lambda: torch_fused_layernorm(x, residual, weight, bias))
        t_triton = benchmark(
            lambda: triton_fused_layernorm(x, residual, weight, bias))

        speedup = t_eager / t_triton
        print(f"  {label:<30} {t_eager:>12.3f} {t_triton:>12.3f} {speedup:>10.2f}x")

        del x, residual

    # ============================================================
    # 实验 3: Kernel 数量对比
    # ============================================================
    print()
    print("=" * 60)
    print("实验 3: Kernel Fusion 分析")
    print("=" * 60)
    print("""
PyTorch Eager 分解执行 (每步调用多个 kernel):
  1. residual = x + residual          → 1 kernel (elementwise add)
  2. mean = hidden.mean(dim=-1)       → 1 kernel (reduce)
  3. centered = hidden - mean          → 1 kernel (broadcast sub)
  4. var = (centered^2).mean(dim=-1)  → 2 kernels (square + reduce)
  5. rstd = 1/sqrt(var + eps)         → 1 kernel (elementwise)
  6. normalized = centered * rstd      → 1 kernel (broadcast mul)
  7. output = normalized * w + b       → 2 kernels (mul + add)
  8. gelu = gelu(output)               → ~3 kernels (sigmoid approximation)

  总计: ~12 kernel launches

Triton 融合:
  1. 全部操作在 1 个 kernel 中完成
  - 只读写 HBM 各 1 次 (x, residual, weight, bias → output)
  - 中间结果全部在 SRAM (shared memory / registers)

  总计: 1 kernel launch

性能提升来源:
  1. 减少 kernel launch overhead: 12 × ~9us → 1 × ~9us = 节省 ~99us
  2. 减少 HBM 读写: eager 模式每个中间结果都要写回 HBM
     融合模式中间结果全部在 SRAM 中传递
  3. 更好的内存访问模式: 编译器可以优化 load/store
    """)

    # ============================================================
    # 实验 4: 不同 hidden dim 的影响
    # ============================================================
    print("=" * 60)
    print("实验 4: Hidden Dimension 的影响 (Batch=4, Seq=128)")
    print("=" * 60)

    print(f"  {'Hidden':>8} {'Eager (ms)':>12} {'Triton (ms)':>12} {'Speedup':>10}")
    for hidden in [256, 512, 768, 1024, 2048, 4096]:
        batch, seq = 4, 128
        x = torch.randn(batch, seq, hidden, dtype=torch.float32, device=device)
        residual = torch.randn(batch, seq, hidden, dtype=torch.float32, device=device)
        weight = torch.randn(hidden, dtype=torch.float32, device=device)
        bias = torch.randn(hidden, dtype=torch.float32, device=device)

        t_eager = benchmark(
            lambda: torch_fused_layernorm(x, residual, weight, bias))
        t_triton = benchmark(
            lambda: triton_fused_layernorm(x, residual, weight, bias))

        speedup = t_eager / t_triton
        print(f"  {hidden:>8} {t_eager:>12.3f} {t_triton:>12.3f} {speedup:>10.2f}x")

        del x, residual

    print()
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print("""
Triton Kernel Fusion 的核心价值:
1. 减少 kernel launches: 12 → 1, 节省 ~99us (对于小 batch 最重要)
2. 减少 HBM 读写: 中间结果留在 SRAM, 减少 memory bandwidth 压力
3. 自动调优: @triton.autotune 可以搜索最优 BLOCK_SIZE
4. 开发效率: Python 语法比 CUDA C++ 简单得多

实际应用:
- vLLM 用 Triton 写了很多融合 kernel (fused Layernorm, RMSNorm, etc.)
- FlashAttention 的 Triton 实现性能接近 CUDA 版本
- PyTorch 2.0 torch.compile 底层用 Triton 生成融合 kernel
    """)


if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
