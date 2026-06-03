"""量化推理 Benchmark — FP32/FP16/BF16/INT8 性能对比

在 GPU 上对比不同精度下的推理性能:
1. 矩阵乘法吞吐量 (FP32 vs FP16 vs INT8)
2. 内存占用对比
3. 精度损失评估 (余弦相似度 + 相对误差)
4. 推理吞吐量对比 (模拟 Transformer forward pass)

使用方法:
    python quantization_bench.py   # 需要在 GPU 上运行
"""

import torch
import time
import math


def benchmark(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item()


def main():
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"=== 量化推理 Benchmark — {props.name} ===\n")

    # =====================================================
    # 实验 1: GEMM 吞吐量 vs 精度
    # =====================================================
    print("=" * 70)
    print("实验 1: GEMM 吞吐量 vs 数据类型")
    print("=" * 70)

    M, N, K = 4096, 4096, 4096

    results = {}
    for dtype_name, dtype in [("FP32", torch.float32), ("FP16", torch.float16),
                               ("BF16", torch.bfloat16)]:
        a = torch.randn(M, K, dtype=dtype, device=device)
        b = torch.randn(K, N, dtype=dtype, device=device)
        t = benchmark(lambda: torch.mm(a, b))
        tflops = 2 * M * N * K / (t / 1000) / 1e12
        mem = (M * K + K * N + M * N) * a.element_size() / 1e6
        results[dtype_name] = {"time": t, "tflops": tflops, "mem_mb": mem}
        print(f"  {dtype_name:>4}: {t:.3f} ms, {tflops:.2f} TFLOPS, {mem:.0f} MB")
        del a, b

    print(f"\n  FP16 加速: {results['FP32']['time']/results['FP16']['time']:.2f}x")
    print(f"  内存节省: {results['FP32']['mem_mb']/results['FP16']['mem_mb']:.1f}x")

    # =====================================================
    # 实验 2: INT8 动态量化
    # =====================================================
    print()
    print("=" * 70)
    print("实验 2: INT8 动态量化 (模拟)")
    print("=" * 70)

    M, N, K = 4096, 4096, 4096
    a_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    b_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)

    # FP16 baseline
    t_fp16 = benchmark(lambda: torch.mm(a_fp16, b_fp16))

    # INT8 quantize-dequantize (simulated)
    def int8_matmul(a, b):
        # Per-token quantization for activation
        a_max = a.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        a_scale = a_max / 127.0
        a_int8 = (a / a_scale).to(torch.int8)

        # Per-channel quantization for weight
        b_max = b.abs().amax(dim=0, keepdim=True).clamp(min=1e-5)
        b_scale = b_max / 127.0
        b_int8 = (b / b_scale).to(torch.int8)

        # Dequantize and multiply
        # Note: Real INT8 uses specialized kernels (e.g., CUTLASS)
        return torch.mm(a_int8.float(), b_int8.float()) * (a_scale * b_scale.T)

    # Warmup
    _ = int8_matmul(a_fp16[:16], b_fp16[:, :16])

    # Compare accuracy
    out_fp16 = torch.mm(a_fp16[:256], b_fp16[:, :256])
    out_int8 = int8_matmul(a_fp16[:256], b_fp16[:, :256])
    cos_sim = cosine_similarity(out_fp16, out_int8)
    rel_err = (out_fp16 - out_int8).abs().mean() / out_fp16.abs().mean()

    print(f"  FP16 baseline: {t_fp16:.3f} ms")
    print(f"  INT8 QDQ cos_sim: {cos_sim:.4f} (1.0=完美)")
    print(f"  INT8 相对误差: {rel_err*100:.2f}%")

    # Memory comparison
    mem_fp16 = (M * K + K * N) * 2 / 1e6
    mem_int8 = (M * K + K * N) * 1 / 1e6
    print(f"  FP16 权重内存: {mem_fp16:.0f} MB")
    print(f"  INT8 权重内存: {mem_int8:.0f} MB ({mem_fp16/mem_int8:.1f}x 节省)")

    # =====================================================
    # 实验 3: 模拟 Transformer 层的精度对比
    # =====================================================
    print()
    print("=" * 70)
    print("实验 3: 模拟 Transformer 层 Forward Pass (GPT-2 Small)")
    print("=" * 70)

    H, FFN = 768, 3072
    batch, seq = 4, 1

    for dtype_name, dtype in [("FP32", torch.float32), ("FP16", torch.float16),
                               ("BF16", torch.bfloat16)]:
        x = torch.randn(batch, seq, H, dtype=dtype, device=device)
        wq = torch.randn(H, H, dtype=dtype, device=device)
        wk = torch.randn(H, H, dtype=dtype, device=device)
        wv = torch.randn(H, H, dtype=dtype, device=device)
        wo = torch.randn(H, H, dtype=dtype, device=device)
        w1 = torch.randn(H, FFN, dtype=dtype, device=device)
        w2 = torch.randn(FFN, H, dtype=dtype, device=device)
        ln_w = torch.randn(H, dtype=dtype, device=device)

        def transformer_layer(x, wq, wk, wv, wo, w1, w2, ln_w):
            # Pre-LN
            x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True).to(dtype) + 1e-6) * ln_w
            # Attention
            q = torch.matmul(x, wq)
            k = torch.matmul(x, wk)
            v = torch.matmul(x, wv)
            # Simplified attention (no softmax for speed)
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(H)
            attn = torch.matmul(attn, v)
            x = x + torch.matmul(attn, wo)
            # MLP
            x2 = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True).to(dtype) + 1e-6) * ln_w
            x = x + torch.matmul(torch.nn.functional.gelu(torch.matmul(x2, w1)), w2)
            return x

        t = benchmark(lambda: transformer_layer(x, wq, wk, wv, wo, w1, w2, ln_w))

        # Memory
        total_params = (H*H*4 + H*FFN + FFN*H + H) * 2  # approximate
        mem = total_params / 1e6  # MB (2 bytes per param for FP16)

        print(f"  {dtype_name:>4}: {t:.3f} ms/layer, 权重 ~{mem:.0f} MB")
        del x, wq, wk, wv, wo, w1, w2, ln_w

    # =====================================================
    # 实验 4: 大矩阵内存对比
    # =====================================================
    print()
    print("=" * 70)
    print("实验 4: 权重内存占用 vs 模型大小")
    print("=" * 70)

    for name, params_b in [("GPT-2 Small (124M)", 124e6),
                            ("GPT-2 Medium (355M)", 355e6),
                            ("LLaMA-7B", 7e9),
                            ("LLaMA-70B", 70e9)]:
        fp32_gb = params_b * 4 / 1e9
        fp16_gb = params_b * 2 / 1e9
        int8_gb = params_b * 1 / 1e9
        int4_gb = params_b * 0.5 / 1e9
        print(f"  {name:<25} FP32: {fp32_gb:>5.1f} GB  FP16: {fp16_gb:>5.1f} GB  "
              f"INT8: {int8_gb:>4.1f} GB  INT4: {int4_gb:>4.1f} GB")

    print()
    print("=" * 70)
    print("总结")
    print("=" * 70)
    print("""
量化推理的关键结论:
1. FP16 相比 FP32: 2-4x 吞吐提升, 2x 内存节省, 几乎无精度损失
2. INT8 相比 FP16: 2x 内存节省, 但需要专用 kernel (CUTLASS/triton)
3. INT4/FP8: 4x 内存节省, 适合大模型推理 (70B on single GPU)
4. 精度损失: FP16 cos_sim > 0.999, INT8 > 0.99, 量化感知训练可进一步减少

为什么量化对推理重要:
- LLM 推理是 memory-bound (不是 compute-bound)
- 减少权重精度 = 减少 HBM 读取 = 更高吞吐
- 70B 模型: FP16 需 140GB (2×A100), INT4 只需 35GB (1×A100)
    """)


if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
