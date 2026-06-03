"""GPU Profiling Experiment - A16

Profile key GPU operations to understand performance characteristics.
"""
import torch
import time

def benchmark(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000  # ms

def main():
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"=== GPU: {props.name} ===")
    print(f"  SM count: {props.multi_processor_count}")
    print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
    print(f"  L2 cache: {props.L2_cache_size / 1024 / 1024:.0f} MB, Warp: {props.warp_size}")
    print()

    # 1. GEMM (FP16)
    print("=== GEMM (FP16) Benchmarks ===")
    for M, N, K in [(1024, 1024, 1024), (4096, 4096, 4096), (8192, 8192, 8192)]:
        a = torch.randn(M, K, dtype=torch.float16, device=device)
        b = torch.randn(K, N, dtype=torch.float16, device=device)
        t = benchmark(lambda: torch.mm(a, b))
        tflops = 2 * M * N * K / (t / 1000) / 1e12
        print(f"  [{M}x{K}]x[{K}x{N}]: {t:.3f} ms, {tflops:.2f} TFLOPS")
        del a, b
    print()

    # FP32 GEMM
    print("=== GEMM (FP32) Benchmarks ===")
    for M, N, K in [(1024, 1024, 1024), (4096, 4096, 4096)]:
        a = torch.randn(M, K, dtype=torch.float32, device=device)
        b = torch.randn(K, N, dtype=torch.float32, device=device)
        t = benchmark(lambda: torch.mm(a, b))
        tflops = 2 * M * N * K / (t / 1000) / 1e12
        print(f"  [{M}x{K}]x[{K}x{N}]: {t:.3f} ms, {tflops:.2f} TFLOPS")
        del a, b
    print()

    # 2. Memory Bandwidth
    print("=== Memory Bandwidth ===")
    for size_mb in [1, 10, 100, 500, 1000]:
        n = size_mb * 1024 * 1024 // 4
        x = torch.randn(n, device=device)
        t = benchmark(lambda: x.clone())
        # clone = read + write
        bw = size_mb * 2 / (t / 1000) / 1e3
        print(f"  Copy {size_mb:>4}MB: {t:.3f} ms, {bw:.1f} GB/s")
        del x
    print()

    # 3. Kernel Launch Overhead
    print("=== Kernel Launch Overhead ===")
    x = torch.randn(1, device=device)
    t = benchmark(lambda: x + 1.0, warmup=50, iters=1000)
    print(f"  Single element add: {t*1000:.1f} us")
    del x

    for log2n in range(10, 25, 2):
        n = 2 ** log2n
        x = torch.randn(n, device=device)
        t = benchmark(lambda: x * 2.0, warmup=20, iters=200)
        bw = n * 4 * 3 / (t / 1000) / 1e9
        print(f"  Scale {n:>10,}: {t*1000:>7.1f} us, {bw:.1f} GB/s")
        del x
    print()

    # 4. Attention
    print("=== Attention (SDPA, FP16) ===")
    batch, heads, dim = 4, 32, 64
    try:
        for seq_len in [512, 1024, 2048, 4096, 8192]:
            q = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
            k = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
            v = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
            t = benchmark(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
            flops = batch * heads * seq_len * seq_len * dim * 4
            tflops = flops / (t / 1000) / 1e12
            print(f"  SDPA S={seq_len:>5}: {t:.3f} ms, {tflops:.2f} TFLOPS")
            del q, k, v
    except Exception as e:
        print(f"  Failed: {e}")
    print()

    # 5. Reductions
    print("=== Reductions ===")
    for log2n in [20, 22, 24]:
        n = 2 ** log2n
        x = torch.randn(n, dtype=torch.float32, device=device)
        t = benchmark(lambda: x.sum())
        bw = n * 4 / (t / 1000) / 1e9
        print(f"  Sum 2^{log2n}={n:>10,}: {t:.3f} ms, {bw:.1f} GB/s")
        del x
    print()

    # 6. GEMM batch size scaling
    print("=== GEMM Batch Scaling (M=1, N=4096, K=4096, FP16) ===")
    for batch in [1, 4, 16, 64, 256]:
        a = torch.randn(batch, 4096, 4096, dtype=torch.float16, device=device)
        b = torch.randn(4096, 4096, dtype=torch.float16, device=device)
        t = benchmark(lambda: torch.bmm(a, b.unsqueeze(0).expand(batch, -1, -1)))
        total_flops = 2 * batch * 4096 * 4096 * 4096
        tflops = total_flops / (t / 1000) / 1e12
        print(f"  B={batch:>3}: {t:.3f} ms, {tflops:.2f} TFLOPS")
        del a, b
    print()

    print("=== Profiling Complete ===")

if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
