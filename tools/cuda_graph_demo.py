"""CUDA Graph 实验 — 验证 kernel launch overhead 的优化效果

演示 CUDA Graph 如何通过录制-重放机制减少 kernel launch 开销，
特别适用于推理 decode 阶段的小 batch 场景。

使用方法:
    python cuda_graph_demo.py          # 需要在 GPU 上运行
    python cuda_graph_demo.py --cpu    # CPU 模式对比

实验内容:
    1. 测量不同数量 kernel 的 launch overhead
    2. 对比 eager 执行 vs CUDA Graph 的延迟
    3. 分析 CUDA Graph 在 decode 场景的加速效果
"""

import argparse
import time
import torch


def benchmark(fn, warmup=10, iters=100):
    """Benchmark a function and return average time in ms."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def experiment_kernel_launch_overhead(device, n_kernels_list):
    """Experiment 1: Measure kernel launch overhead scaling."""
    print("=" * 60)
    print("实验 1: Kernel Launch Overhead 随 kernel 数量增长")
    print("=" * 60)

    x = torch.randn(1024, device=device)
    results = {}

    for n_kernels in n_kernels_list:
        def run_n_kernels(x=x, n=n_kernels):
            for _ in range(n):
                x = x + 1.0
            return x

        t = benchmark(run_n_kernels)
        overhead_per_kernel = t / n_kernels
        results[n_kernels] = t
        print(f"  {n_kernels:>3} kernels: {t:.3f} ms "
              f"({overhead_per_kernel * 1000:.1f} us/kernel)")

    return results


def experiment_cuda_graph_basic(device, seq_len, n_layers):
    """Experiment 2: Eager vs CUDA Graph for a simple model forward pass."""
    print()
    print("=" * 60)
    print("实验 2: Eager vs CUDA Graph — 模拟 Decode Step")
    print("=" * 60)
    print(f"  配置: seq_len={seq_len}, n_layers={n_layers}, hidden=768")

    # Simulate a transformer decode step: n_layers of (layer_norm + linear + add)
    hidden = 768
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Create weights and input (static shapes for CUDA Graph)
    weights = [torch.randn(hidden, hidden, dtype=dtype, device=device)
               for _ in range(n_layers)]
    biases = [torch.randn(hidden, dtype=dtype, device=device)
              for _ in range(n_layers)]
    ln_weights = [torch.randn(hidden, dtype=dtype, device=device)
                  for _ in range(n_layers)]
    ln_biases = [torch.randn(hidden, dtype=dtype, device=device)
                 for _ in range(n_layers)]

    def decode_step(x, weights=weights, biases=biases, ln_weights=ln_weights,
                    ln_biases=ln_biases):
        """Simulate a single decode step through n transformer layers."""
        for i in range(len(weights)):
            # LayerNorm approximation (simplified)
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + 1e-5) * ln_weights[i] + ln_biases[i]
            # Linear + residual
            x = x + torch.matmul(x, weights[i]) + biases[i]
        return x

    # Input
    x = torch.randn(1, seq_len, hidden, dtype=dtype, device=device)

    # Eager execution
    t_eager = benchmark(lambda: decode_step(x))
    print(f"  Eager:     {t_eager:.3f} ms")

    if device.type == "cuda":
        # CUDA Graph: record
        g = torch.cuda.CUDAGraph()
        x_static = x.clone()

        # Warmup for recording
        with torch.cuda.graph(g):
            x_static = decode_step(x_static)

        # Replay
        t_graph = benchmark(lambda: g.replay())
        print(f"  CUDA Graph: {t_graph:.3f} ms")
        print(f"  加速比:     {t_eager / t_graph:.2f}x")
        print(f"  节省:       {(t_eager - t_graph) * 1000:.1f} us/step")
        return {"eager": t_eager, "graph": t_graph, "speedup": t_eager / t_graph}
    else:
        print("  (CUDA Graph 仅支持 GPU，跳过)")
        return {"eager": t_eager, "graph": None, "speedup": 1.0}


def experiment_batch_scaling(device):
    """Experiment 3: CUDA Graph speedup at different batch sizes."""
    print()
    print("=" * 60)
    print("实验 3: CUDA Graph 加速比 vs Batch Size")
    print("=" * 60)

    if device.type != "cuda":
        print("  (CUDA Graph 仅支持 GPU，跳过)")
        return

    hidden = 768
    n_layers = 4
    dtype = torch.float16

    # Pre-create weights
    weights = [torch.randn(hidden, hidden, dtype=dtype, device=device)
               for _ in range(n_layers)]
    biases = [torch.randn(hidden, dtype=dtype, device=device)
              for _ in range(n_layers)]

    def forward(x):
        for i in range(n_layers):
            x = x + torch.matmul(x, weights[i]) + biases[i]
        return x

    print(f"  {'Batch':>6} {'Eager (ms)':>12} {'Graph (ms)':>12} {'Speedup':>10}")
    print(f"  {'-'*6:>6} {'-'*12:>12} {'-'*12:>12} {'-'*10:>10}")

    for batch in [1, 2, 4, 8, 16, 32]:
        x = torch.randn(batch, 1, hidden, dtype=dtype, device=device)

        # Eager
        t_eager = benchmark(lambda: forward(x))

        # CUDA Graph
        g = torch.cuda.CUDAGraph()
        x_static = x.clone()
        with torch.cuda.graph(g):
            x_static = forward(x_static)
        t_graph = benchmark(lambda: g.replay())

        speedup = t_eager / t_graph
        print(f"  {batch:>6} {t_eager:>12.3f} {t_graph:>12.3f} {speedup:>10.2f}x")

        del x, x_static, g


def experiment_kernel_count_graph(device):
    """Experiment 4: CUDA Graph speedup vs number of kernels."""
    print()
    print("=" * 60)
    print("实验 4: CUDA Graph 加速比 vs Kernel 数量")
    print("=" * 60)

    if device.type != "cuda":
        print("  (CUDA Graph 仅支持 GPU，跳过)")
        return

    x = torch.randn(1, 128, device=device)
    print(f"  {'Kernels':>8} {'Eager (ms)':>12} {'Graph (ms)':>12} {'Speedup':>10}")
    print(f"  {'-'*8:>8} {'-'*12:>12} {'-'*12:>12} {'-'*10:>10}")

    for n_kernels in [10, 20, 50, 100, 200, 500]:
        def run_n(x=x, n=n_kernels):
            for _ in range(n):
                x = x + 1.0
            return x

        # Eager
        t_eager = benchmark(run_n)

        # CUDA Graph
        g = torch.cuda.CUDAGraph()
        x_static = x.clone()
        with torch.cuda.graph(g):
            x_static = run_n(x_static)
        t_graph = benchmark(lambda: g.replay())

        speedup = t_eager / t_graph
        print(f"  {n_kernels:>8} {t_eager:>12.3f} {t_graph:>12.3f} {speedup:>10.2f}x")

        del g


def main():
    parser = argparse.ArgumentParser(description="CUDA Graph 实验")
    parser.add_argument("--cpu", action="store_true", help="使用 CPU 模式")
    args = parser.parse_args()

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("=== CUDA Graph 实验 — CPU 模式 (无 CUDA Graph) ===\n")
    else:
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        print(f"=== CUDA Graph 实验 — {props.name} ===")
        print(f"    显存: {props.total_memory / 1e9:.1f} GB\n")

    # Experiment 1: Kernel launch overhead
    experiment_kernel_launch_overhead(device, [1, 5, 10, 20, 50, 100])

    # Experiment 2: Basic CUDA Graph
    experiment_cuda_graph_basic(device, seq_len=1, n_layers=12)

    # Experiment 3: Batch scaling
    experiment_batch_scaling(device)

    # Experiment 4: Kernel count
    experiment_kernel_count_graph(device)

    print()
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print("""
CUDA Graph 的核心价值:
1. 减少 kernel launch overhead: 从每个 kernel ~9us 降到录制-重放的固定成本
2. 小 batch (decode) 场景收益最大: launch 开销占比高
3. 大 batch 场景收益递减: 计算时间远大于 launch 时间
4. 限制: 需要静态 shape，不支持动态控制流
5. vLLM V1 默认开启 CUDA Graph (enforce_eager=False)
    """)


if __name__ == "__main__":
    main()
