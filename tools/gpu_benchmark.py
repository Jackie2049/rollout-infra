#!/usr/bin/env python3
"""GPU 快速基准测试 — 验证 PyTorch + CUDA 环境 + 性能数据采集

用法:
  python gpu_benchmark.py           # 运行全部测试
  python gpu_benchmark.py --quick   # 快速模式 (只验证环境)

测试内容:
  1. 环境验证 (CUDA, GPU 型号, PyTorch 版本)
  2. 带宽测试 (HBM 读写速度)
  3. 算力测试 (GEMM TFLOPS)
  4. 混合精度对比 (FP32/FP16/BF16)
  5. 显存占用分析
  6. 矩阵乘法 scaling (不同 size)
"""

import sys
import time
import argparse


def section(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def check_env():
    """验证 GPU + PyTorch 环境"""
    section("1. 环境验证")

    import torch
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("  ERROR: CUDA not available!")
        return False

    print(f"  CUDA version:   {torch.version.cuda}")
    print(f"  cuDNN version:  {torch.backends.cudnn.version()}")
    print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print(f"  GPU count:   {torch.cuda.device_count()}")
    print(f"  GPU memory:  {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # 检查 BF16 支持
    bf16 = torch.cuda.is_bf16_supported()
    print(f"  BF16 support:   {bf16}")

    # Tensor Core 检查
    props = torch.cuda.get_device_properties(0)
    major = props.major
    minor = props.minor
    print(f"  Compute capability: {major}.{minor}")
    print(f"  Tensor Cores: {'Yes (SM 7.0+)' if major >= 7 else 'No'}")

    return True


def benchmark_bandwidth():
    """测试 HBM 带宽"""
    section("2. HBM 带宽测试")

    import torch

    device = torch.device('cuda')
    sizes = [1, 4, 16, 64, 256, 1024]  # MB

    print(f"  {'Size (MB)':<12} {'Read (GB/s)':<15} {'Write (GB/s)':<15}")
    print(f"  {'-'*42}")

    for size_mb in sizes:
        n = size_mb * 1024 * 1024 // 4  # float32 elements

        # Write 测试 (CPU → GPU)
        src = torch.randn(n, dtype=torch.float32, device='cpu')
        dst = torch.empty(n, dtype=torch.float32, device=device)

        # Warmup
        for _ in range(3):
            dst.copy_(src)
        torch.cuda.synchronize()

        # Measure write
        start = time.perf_counter()
        for _ in range(10):
            dst.copy_(src)
        torch.cuda.synchronize()
        write_time = (time.perf_counter() - start) / 10
        write_bw = size_mb / 1024 / write_time  # GB/s

        # Read 测试 (GPU → CPU)
        src_gpu = torch.randn(n, dtype=torch.float32, device=device)
        dst_cpu = torch.empty(n, dtype=torch.float32, device='cpu')

        for _ in range(3):
            dst_cpu.copy_(src_gpu)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(10):
            dst_cpu.copy_(src_gpu)
        torch.cuda.synchronize()
        read_time = (time.perf_counter() - start) / 10
        read_bw = size_mb / 1024 / read_time  # GB/s

        print(f"  {size_mb:<12} {read_bw:<15.1f} {write_bw:<15.1f}")

    # GPU 内部拷贝带宽
    print(f"\n  GPU Internal Copy (GPU→GPU):")
    for size_mb in [1, 64, 256, 1024]:
        n = size_mb * 1024 * 1024 // 4
        src = torch.randn(n, dtype=torch.float32, device=device)
        dst = torch.empty_like(src)

        for _ in range(3):
            dst.copy_(src)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(20):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 20
        bw = size_mb / 1024 / elapsed
        print(f"    {size_mb}MB: {bw:.1f} GB/s")


def benchmark_gemm():
    """测试 GEMM (矩阵乘法) TFLOPS"""
    section("3. GEMM 算力测试")

    import torch

    device = torch.device('cuda')

    sizes = [128, 256, 512, 1024, 2048, 4096]
    dtypes = [
        ("FP32", torch.float32),
        ("FP16", torch.float16),
    ]

    # 检查 BF16
    if torch.cuda.is_bf16_supported():
        dtypes.append(("BF16", torch.bfloat16))

    for dtype_name, dtype in dtypes:
        print(f"\n  {dtype_name}:")
        print(f"    {'M=N=K':<12} {'Time (ms)':<12} {'TFLOPS':<12} {'Efficiency':<12}")
        print(f"    {'-'*48}")

        for n in sizes:
            a = torch.randn(n, n, device=device, dtype=dtype)
            b = torch.randn(n, n, device=device, dtype=dtype)

            # Warmup
            for _ in range(5):
                c = torch.mm(a, b)
            torch.cuda.synchronize()

            # Measure
            repeats = max(10, 1000 // n)  # 小矩阵多测几次
            start = time.perf_counter()
            for _ in range(repeats):
                c = torch.mm(a, b)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / repeats

            # FLOPS = 2 * M * N * K (乘加各一次)
            flops = 2 * n * n * n
            tflops = flops / elapsed / 1e12
            # 理论峰值: A16 = ~7.1 TFLOPS (FP16), ~0.35 TFLOPS (FP32)
            # 这里只显示实际值
            print(f"    {n:<12} {elapsed*1000:<12.3f} {tflops:<12.2f}")


def benchmark_mixed_precision():
    """混合精度训练对比"""
    section("4. 混合精度训练对比")

    import torch
    import torch.nn as nn

    device = torch.device('cuda')
    hidden = 1024
    num_layers = 4
    batch = 32
    seq = 128

    results = {}

    for dtype_name, dtype in [("FP32", torch.float32), ("FP16", torch.float16)]:
        if dtype_name == "BF16" and not torch.cuda.is_bf16_supported():
            continue
        try:
            if dtype_name == "BF16":
                dtype = torch.bfloat16
        except:
            pass

        # 创建简单模型
        layers = nn.Sequential(*[
            nn.Linear(hidden, hidden) for _ in range(num_layers)
        ]).to(device).to(dtype)

        x = torch.randn(batch, seq, hidden, device=device, dtype=dtype)

        # Warmup
        for _ in range(10):
            y = layers(x)
            loss = y.sum()
            loss.backward()
        torch.cuda.synchronize()

        # Measure
        mem_before = torch.cuda.memory_allocated() / 1e9
        start = time.perf_counter()
        steps = 50
        for _ in range(steps):
            y = layers(x)
            loss = y.sum()
            loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        mem_after = torch.cuda.max_memory_allocated() / 1e9

        throughput = batch * seq * steps / elapsed
        results[dtype_name] = {
            "time": elapsed,
            "throughput": throughput,
            "memory": mem_after,
        }

        print(f"  {dtype_name}:")
        print(f"    Time:      {elapsed:.3f}s ({elapsed/steps*1000:.1f}ms/step)")
        print(f"    Throughput: {throughput:.0f} samples/s")
        print(f"    Peak Memory: {mem_after:.2f} GB")

    if "FP16" in results and "FP32" in results:
        speedup = results["FP32"]["time"] / results["FP16"]["time"]
        print(f"\n  FP16 Speedup over FP32: {speedup:.2f}x")


def benchmark_memory():
    """显存分析"""
    section("5. 显存分析")

    import torch
    import torch.nn as nn

    device = torch.device('cuda')

    # 不同大小的模型显存占用
    configs = [
        ("Tiny (125M)",  512,  12, 8),
        ("Small (350M)", 768,  16, 12),
        ("Medium (7B partial)", 1024, 4, 8),
    ]

    print(f"  {'Config':<25} {'Params':<12} {'Model Mem':<12} {'+Optimizer':<12} {'+Grad':<12}")
    print(f"  {'-'*73}")

    for name, hidden, layers, heads in configs:
        # 估算参数量
        params = 0
        for _ in range(layers):
            # attention: 4 * hidden^2 (QKV + output projection)
            params += 4 * hidden * hidden
            # FFN: 8/3 * hidden^2 (SwiGLU)
            params += int(8/3 * hidden * hidden)

        params_b = params / 1e9

        # BF16 模型
        model_mem = params * 2 / 1e9  # 2 bytes per param
        # + AdamW optimizer (FP32: m + v)
        opt_mem = params * 8 / 1e9
        # + gradients (BF16)
        grad_mem = params * 2 / 1e9

        total = model_mem + opt_mem + grad_mem

        print(f"  {name:<25} {params_b:<12.3f} {model_mem:<12.2f} "
              f"{model_mem+opt_mem:<12.2f} {total:<12.2f}")

    print(f"\n  可用显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"  建议: 对于 A16 15GB, 适合 Tiny/Small 模型训练或 7B 推理")


def main():
    parser = argparse.ArgumentParser(description="GPU 快速基准测试")
    parser.add_argument("--quick", action="store_true", help="只验证环境")
    args = parser.parse_args()

    if not check_env():
        sys.exit(1)

    if args.quick:
        print("\n  Quick mode: environment check passed.")
        return

    benchmark_bandwidth()
    benchmark_gemm()
    benchmark_mixed_precision()
    benchmark_memory()

    print(f"\n{'='*60}")
    print("  基准测试完成!")


if __name__ == "__main__":
    main()
