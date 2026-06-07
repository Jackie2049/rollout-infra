#!/usr/bin/env python3
"""CUTLASS GEMM Benchmark — RTX 4090

Compare CUTLASS basic_gemm (FP32) vs cuBLAS (via PyTorch) on RTX 4090.
Also benchmark FP16/BF16 matmul via PyTorch + CUDA Events.

Usage:
  On GPU server: python tools/cutlass_gemm_benchmark.py

Runs on: RTX 4090 (SM89), CUDA 12.8, PyTorch 2.9.0+cu128
"""

import torch
import numpy as np
import time
import json
import os
import sys
import subprocess

def warmup_gpu(device, n=10):
    """Warmup GPU to avoid cold-start measurement artifacts."""
    for _ in range(n):
        a = torch.randn(2048, 2048, device=device)
        b = torch.randn(2048, 2048, device=device)
        _ = a @ b
    torch.cuda.synchronize()

def benchmark_pytorch_gemm(device, M, N, K, dtype, n_iters=100, n_warmup=10):
    """Benchmark PyTorch matmul (cuBLAS backend) with CUDA Events."""
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)

    # Warmup
    for _ in range(n_warmup):
        _ = a @ b
    torch.cuda.synchronize()

    # Benchmark with CUDA Events (most accurate)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(n_iters):
        start_event.record()
        c = a @ b
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    avg_ms = np.mean(times)
    std_ms = np.std(times)

    # Compute TFLOPS
    # GEMM FLOPs = 2 * M * N * K (multiply + add)
    flops = 2.0 * M * N * K
    tflops = flops / (avg_ms * 1e-3) / 1e12

    return {
        'M': M, 'N': N, 'K': K,
        'dtype': str(dtype),
        'avg_ms': avg_ms,
        'std_ms': std_ms,
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'tflops': tflops,
        'flops': flops,
    }

def run_cutlass_gemm(binary_path, M, N, K):
    """Run CUTLASS 00_basic_gemm binary and check result."""
    result = subprocess.run(
        [binary_path, str(M), str(N), str(K)],
        capture_output=True, text=True, timeout=30
    )
    return result.stdout.strip(), result.returncode

def benchmark_cutlass_gemm(binary_path, M, N, K, n_iters=10):
    """Benchmark CUTLASS basic_gemm by running it multiple times."""
    # First verify correctness
    output, rc = run_cutlass_gemm(binary_path, M, N, K)
    if rc != 0:
        return {'error': f'CUTLASS GEMM failed: {output}', 'M': M, 'N': N, 'K': K}

    # CUTLASS basic_gemm doesn't have timing, so we time the subprocess
    # This is less accurate but gives relative comparison
    # For precise CUTLASS timing we'd need to modify the example or use cutlass_profiler
    times = []
    for _ in range(n_iters):
        start = time.time()
        run_cutlass_gemm(binary_path, M, N, K)
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)

    avg_ms = np.mean(times)
    flops = 2.0 * M * N * K
    tflops = flops / (avg_ms * 1e-3) / 1e12

    return {
        'M': M, 'N': N, 'K': K,
        'dtype': 'float32',
        'avg_ms': avg_ms,
        'tflops': tflops,
        'status': output,
    }

def main():
    device = torch.device('cuda')

    print("=" * 70)
    print("CUTLASS GEMM Benchmark — RTX 4090 (SM89)")
    print("=" * 70)

    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}, Memory: {gpu_mem:.1f} GB")
    print(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")

    # Warmup
    warmup_gpu(device)

    # Find CUTLASS binary
    cutlass_binary = os.path.expanduser("~/CUTLASS/build/examples/00_basic_gemm/00_basic_gemm")
    has_cutlass = os.path.exists(cutlass_binary)

    # Benchmark sizes
    sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        # Decode sizes (small M)
        (1, 4096, 4096),
        (8, 4096, 4096),
        (32, 4096, 4096),
        (128, 4096, 4096),
        # Non-square
        (4096, 1024, 4096),
        (1024, 4096, 4096),
    ]

    # PyTorch benchmarks
    results = {}
    for dtype_name, dtype in [('fp32', torch.float32), ('fp16', torch.float16), ('bf16', torch.bfloat16)]:
        print(f"\n--- PyTorch {dtype_name} matmul (cuBLAS) ---")
        results[dtype_name] = []
        for M, N, K in sizes:
            r = benchmark_pytorch_gemm(device, M, N, K, dtype)
            results[dtype_name].append(r)
            print(f"  M={M:5d} N={N:5d} K={K:5d}: {r['avg_ms']:.3f}ms ± {r['std_ms']:.3f}ms, "
                  f"{r['tflops']:.2f} TFLOPS")

    # CUTLASS FP32 benchmark
    if has_cutlass:
        print(f"\n--- CUTLASS basic_gemm (FP32) ---")
        results['cutlass_fp32'] = []
        for M, N, K in sizes:
            r = benchmark_cutlass_gemm(cutlass_binary, M, N, K)
            if 'error' in r:
                print(f"  M={M:5d} N={N:5d} K={K:5d}: ERROR - {r['error']}")
            else:
                results['cutlass_fp32'].append(r)
                print(f"  M={M:5d} N={N:5d} K={K:5d}: {r['avg_ms']:.3f}ms, {r['tflops']:.2f} TFLOPS")

    # HBM bandwidth measurement
    print(f"\n--- HBM Bandwidth (memory copy) ---")
    sizes_bw = [1<<20, 1<<22, 1<<24, 1<<26]  # 1MB to 64MB
    bw_results = []
    for nbytes in sizes_bw:
        data = torch.randn(nbytes // 4, dtype=torch.float32, device=device)
        dst = torch.empty_like(data)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(5):
            dst.copy_(data)
        torch.cuda.synchronize()

        times = []
        for _ in range(50):
            start_event.record()
            dst.copy_(data)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))

        avg_ms = np.mean(times)
        bw_gbps = (nbytes / (avg_ms * 1e-3)) / 1e9
        bw_results.append({'nbytes': nbytes, 'ms': avg_ms, 'gbps': bw_gbps})
        print(f"  {int(nbytes/(1<<20)):3d} MB: {avg_ms:.3f}ms, {bw_gbps:.1f} GB/s")

    # Summary
    print("\n" + "=" * 70)
    print("Summary — RTX 4090 GEMM Performance")
    print("=" * 70)

    # Peak TFLOPS at largest square size
    fp32_peak = max(r['tflops'] for r in results['fp32'] if r['M'] >= 2048)
    fp16_peak = max(r['tflops'] for r in results['fp16'] if r['M'] >= 2048)
    bf16_peak = max(r['tflops'] for r in results['bf16'] if r['M'] >= 2048)
    fp32_peak_hw = 82.6  # RTX 4090 FP32 peak
    fp16_peak_hw = 165.2  # RTX 4090 FP16 peak

    print(f"  FP32 peak achieved: {fp32_peak:.2f} TFLOPS ({fp32_peak/fp32_peak_hw*100:.1f}% of {fp32_peak_hw} TFLOPS hw peak)")
    print(f"  FP16 peak achieved: {fp16_peak:.2f} TFLOPS ({fp16_peak/fp16_peak_hw*100:.1f}% of {fp16_peak_hw} TFLOPS hw peak)")
    print(f"  BF16 peak achieved: {bf16_peak:.2f} TFLOPS ({bf16_peak/fp16_peak_hw*100:.1f}% of {fp16_peak_hw} TFLOPS hw peak)")
    print(f"  FP16/FP32 ratio: {fp16_peak/fp32_peak:.2f}x")

    # Decode performance (small batch)
    decode_r = [r for r in results['fp16'] if r['M'] <= 32]
    for r in decode_r:
        ai = 2 * r['M'] * r['N'] * r['K'] / (r['M'] * r['K'] * 2 + r['K'] * r['N'] * 2 + r['M'] * r['N'] * 2)
        mem_bound_pct = r['tflops'] / fp16_peak_hw * 100
        print(f"  Decode M={r['M']}: {r['tflops']:.2f} TFLOPS ({mem_bound_pct:.1f}% peak), AI={ai:.1f}")

    # Save results
    output_path = os.path.expanduser("~/rollout-infra/results/cutlass_gemm_benchmark.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    main()
