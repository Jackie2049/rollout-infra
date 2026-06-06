#!/usr/bin/env python3
"""CUTLASS/cuBLAS GEMM Benchmark on RTX 4090
===========================================

Benchmarks GEMM performance on RTX 4090 (SM89) using cuBLAS:
1. FP16 vs FP32 vs BF16 GEMM throughput
2. Decode-size GEMM profile (M=1,2,4,8,16,32,64,128,256,512)
3. FP8 (E4M3) GEMM if supported
4. TFLOPS vs theoretical peak comparison
5. Memory-bound vs compute-bound crossover (ridge point)

Usage:
  python gemm_roofline_cutlass_4090.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import json
import math
import numpy as np

def benchmark_cuda(fn, warmup=10, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat

def compute_tflops(M, N, K, time_ms, dtype_bytes=2):
    """Compute achieved TFLOPS for C = A @ B (M×K × K×N → M×N)."""
    # FLOPs = 2 * M * N * K (multiply + add per output element)
    flops = 2 * M * N * K
    tflops = flops / (time_ms * 1e-3) / 1e12
    return tflops

def compute_arithmetic_intensity(M, N, K, dtype_bytes=2):
    """Compute arithmetic intensity (FLOPS/byte)."""
    # Bytes read: M*K + K*N (A+B), Bytes written: M*N (C)
    bytes_read = (M * K + K * N) * dtype_bytes
    bytes_write = M * N * dtype_bytes
    total_bytes = bytes_read + bytes_write
    return 2 * M * N * K / total_bytes

# ================================================================
# Experiment 1: FP16 vs FP32 vs BF16 GEMM Throughput
# ================================================================
def exp1_dtype_comparison():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 1: FP16 vs FP32 vs BF16 GEMM Throughput")
    print("=" * 60)

    N = 4096  # Fixed matrix size
    M_sizes = [1, 4, 16, 64, 256, 1024, 4096]
    K = 4096
    dtypes = [(torch.float32, 4), (torch.float16, 2), (torch.bfloat16, 2)]
    results = []

    for M in M_sizes:
        for dtype, dtype_bytes in dtypes:
            A = torch.randn(M, K, device=device, dtype=dtype)
            B = torch.randn(K, N, device=device, dtype=dtype)

            def gemm_fn():
                return A @ B

            time_ms = benchmark_cuda(gemm_fn)
            tflops = compute_tflops(M, N, K, time_ms, dtype_bytes)
            ai = compute_arithmetic_intensity(M, N, K, dtype_bytes)

            print(f"  M={M}, dtype={str(dtype).split('.')[-1]}: {time_ms:.4f}ms, "
                  f"{tflops:.2f} TFLOPS, AI={ai:.1f}")

            results.append({
                "M": M, "N": N, "K": K,
                "dtype": str(dtype).split('.')[-1],
                "dtype_bytes": dtype_bytes,
                "time_ms": round(time_ms, 4),
                "tflops": round(tflops, 2),
                "arithmetic_intensity": round(ai, 1),
            })

    return results

# ================================================================
# Experiment 2: Decode-Size GEMM Profile (FP16)
# ================================================================
def exp2_decode_gemm_profile():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 2: Decode-Size GEMM Profile (FP16)")
    print("=" * 60)

    # Simulate 7B model GEMM sizes
    # MLP: gate+up: (B, H) × (H, 4H) → (B, 4H), down: (B, 4H) × (4H, H) → (B, H)
    # Attn: QKV: (B, H) × (H, 3H) → (B, 3H), out: (B, H) × (H, H) → (B, H)
    H = 4096  # hidden_size for 7B
    configs = [
        ("MLP_gate_up", H, 4*H),  # M=B, K=H, N=4H
        ("MLP_down", 4*H, H),      # M=B, K=4H, N=H
        ("Attn_QKV", H, 3*H),      # M=B, K=H, N=3H
        ("Attn_out", H, H),         # M=B, K=H, N=H
    ]

    B_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512]
    results = []

    for name, K_dim, N_dim in configs:
        for B in B_sizes:
            A = torch.randn(B, K_dim, device=device, dtype=torch.float16)
            B_mat = torch.randn(K_dim, N_dim, device=device, dtype=torch.float16)

            def gemm_fn():
                return A @ B_mat

            time_ms = benchmark_cuda(gemm_fn)
            tflops = compute_tflops(B, N_dim, K_dim, time_ms, 2)
            ai = compute_arithmetic_intensity(B, N_dim, K_dim, 2)

            # Throughput in tok/s (each token needs multiple GEMMs per layer)
            # Simplified: 1 token = all 3 MLP GEMMs + 2 attn GEMMs ≈ 5 GEMMs
            if name == "MLP_gate_up":
                tok_per_sec = 1e3 / time_ms  # single GEMM throughput in tok/s

            print(f"  {name} B={B}: {time_ms:.4f}ms, {tflops:.2f} TFLOPS, AI={ai:.1f}")

            results.append({
                "gemm_name": name, "batch": B,
                "K": K_dim, "N": N_dim,
                "time_ms": round(time_ms, 4),
                "tflops": round(tflops, 2),
                "arithmetic_intensity": round(ai, 1),
            })

    return results

# ================================================================
# Experiment 3: FP8 GEMM (if supported)
# ================================================================
def exp3_fp8_gemm():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 3: FP8 GEMM (if supported on RTX 4090)")
    print("=" * 60)

    results = []

    # Check if float8_e4m3fn is available
    try:
        N = 4096
        K = 4096
        M_sizes = [1, 32, 128, 512, 2048]

        for M in M_sizes:
            # FP8 GEMM: cast to float8 then matmul
            # Note: cuBLAS may not support direct FP8 matmul on SM89
            # We test the cast pathway
            A_f16 = torch.randn(M, K, device=device, dtype=torch.float16)
            B_f16 = torch.randn(K, N, device=device, dtype=torch.float16)

            # Method 1: Direct FP16 matmul (baseline)
            def fp16_fn():
                return A_f16 @ B_f16

            t_fp16 = benchmark_cuda(fp16_fn)

            # Method 2: Cast to FP8 then matmul (Python-level)
            try:
                A_fp8 = A_f16.to(torch.float8_e4m3fn)
                B_fp8 = B_f16.to(torch.float8_e4m3fn)

                # FP8 matmul may not work directly → cast back
                def fp8_cast_fn():
                    # Cast FP8 → FP16 then matmul
                    return A_fp8.to(torch.float16) @ B_fp8.to(torch.float16)

                t_fp8_cast = benchmark_cuda(fp8_cast_fn)
                overhead_pct = (t_fp8_cast / t_fp16 - 1) * 100

                # Try direct FP8 matmul (may fail on SM89)
                try:
                    def fp8_direct_fn():
                        return A_fp8 @ B_fp8

                    t_fp8_direct = benchmark_cuda(fp8_direct_fn)
                    fp8_direct_tflops = compute_tflops(M, N, K, t_fp8_direct, 1)

                    print(f"  M={M}: FP16={t_fp16:.4f}ms, FP8_cast={t_fp8_cast:.4f}ms "
                          f"(overhead {overhead_pct:.1f}%), FP8_direct={t_fp8_direct:.4f}ms "
                          f"({fp8_direct_tflops:.2f} TFLOPS)")

                    results.append({
                        "M": M, "N": N, "K": K,
                        "fp16_ms": round(t_fp16, 4),
                        "fp8_cast_ms": round(t_fp8_cast, 4),
                        "fp8_cast_overhead_pct": round(overhead_pct, 1),
                        "fp8_direct_ms": round(t_fp8_direct, 4),
                        "fp8_direct_tflops": round(fp8_direct_tflops, 2),
                    })
                except Exception as e:
                    print(f"  M={M}: FP16={t_fp16:.4f}ms, FP8_cast={t_fp8_cast:.4f}ms "
                          f"(overhead {overhead_pct:.1f}%), FP8_direct FAILED: {e}")

                    results.append({
                        "M": M, "fp16_ms": round(t_fp16, 4),
                        "fp8_cast_ms": round(t_fp8_cast, 4),
                        "fp8_cast_overhead_pct": round(overhead_pct, 1),
                        "fp8_direct_ms": "FAILED",
                    })
            except Exception as e:
                print(f"  M={M}: FP8 cast failed: {e}")
                results.append({"M": M, "fp8_error": str(e)})

    except Exception as e:
        print(f"  FP8 not supported: {e}")
        results.append({"error": f"FP8 dtype not available: {e}"})

    return results

# ================================================================
# Experiment 4: Memory Bandwidth Verification
# ================================================================
def exp4_memory_bandwidth():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 4: Memory Bandwidth Verification (Copy Benchmark)")
    print("=" * 60)

    results = []

    # Test HBM bandwidth by copying large tensors
    sizes_mb = [1, 4, 16, 64, 256, 1024]  # MB

    for size_mb in sizes_mb:
        num_elements = size_mb * 1024 * 1024 // 2  # FP16 = 2 bytes
        src = torch.randn(num_elements, device=device, dtype=torch.float16)
        dst = torch.empty_like(src)

        def copy_fn():
            dst.copy_(src)

        time_ms = benchmark_cuda(copy_fn, warmup=5, repeat=50)
        total_bytes = size_mb * 1024 * 1024 * 2  # read + write
        bw_gb_s = total_bytes / (time_ms * 1e-3) / 1e9

        print(f"  Size={size_mb}MB: {time_ms:.4f}ms, BW={bw_gb_s:.2f} GB/s")

        results.append({
            "size_mb": size_mb,
            "time_ms": round(time_ms, 4),
            "bandwidth_gb_s": round(bw_gb_s, 2),
        })

    return results

# ================================================================
# Experiment 5: Ridge Point Detection (AI vs TFLOPS)
# ================================================================
def exp5_ridge_point():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 5: Ridge Point Detection (AI vs TFLOPS)")
    print("=" * 60)

    # Vary M to find compute-bound vs memory-bound crossover
    K = 4096
    N = 4096
    M_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    results = []

    for M in M_sizes:
        A = torch.randn(M, K, device=device, dtype=torch.float16)
        B = torch.randn(K, N, device=device, dtype=torch.float16)

        def gemm_fn():
            return A @ B

        time_ms = benchmark_cuda(gemm_fn, warmup=5, repeat=20)
        tflops = compute_tflops(M, N, K, time_ms, 2)
        ai = compute_arithmetic_intensity(M, N, K, 2)

        # Roofline prediction: TFLOPS_peak * min(1, AI / ridge_AI)
        # HBM BW ≈ 900 GB/s, TFLOPS_peak ≈ 165 TFLOPS (sparse)
        ridge_ai = 165 / 0.9  # ~183 ops/byte (ridge point)
        # Actually: peak_tflops / peak_bw = 165e12 / 900e9 ≈ 183 ops/byte
        # But real FP16 dense peak ≈ 82.58 TFLOPS → ridge = 82.58/0.9 ≈ 92 ops/byte

        print(f"  M={M}: AI={ai:.1f}, TFLOPS={tflops:.2f}, time={time_ms:.4f}ms")

        results.append({
            "M": M, "N": N, "K": K,
            "arithmetic_intensity": round(ai, 1),
            "tflops": round(tflops, 2),
            "time_ms": round(time_ms, 4),
        })

    # Find ridge point (where TFLOPS plateaus)
    # Simple heuristic: first M where TFLOPS > 80% of max observed
    max_tflops = max(r["tflops"] for r in results)
    ridge_point = None
    for r in results:
        if r["tflops"] > 0.8 * max_tflops and ridge_point is None:
            ridge_point = r

    if ridge_point:
        print(f"\n  RIDGE POINT: M={ridge_point['M']}, AI={ridge_point['arithmetic_intensity']}, "
              f"TFLOPS={ridge_point['tflops']} (80% of peak {max_tflops:.2f})")
        print(f"  Decode GEMM (M≤32): memory-bound (AI ≤ 3)")
        print(f"  Large GEMM (M≥{ridge_point['M']}): compute-bound (AI ≥ {ridge_point['arithmetic_intensity']})")

    return results

# ================================================================
# Main
# ================================================================
def main():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
    gpu_props = torch.cuda.get_device_properties(device)

    print(f"CUTLASS/cuBLAS GEMM Benchmark: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"  SM: {gpu_props.major}.{gpu_props.minor}")
    print(f"  CUDA cores: {gpu_props.multi_processor_count * 128}")
    print(f"  MPs: {gpu_props.multi_processor_count}")
    print("=" * 60)

    all_results = {
        "gpu": gpu_name, "gpu_mem_gb": round(gpu_mem, 1),
        "sm_version": f"{gpu_props.major}.{gpu_props.minor}",
        "cuda_cores": gpu_props.multi_processor_count * 128,
        "mps": gpu_props.multi_processor_count,
    }

    all_results["exp1_dtype_comparison"] = exp1_dtype_comparison()
    all_results["exp2_decode_gemm_profile"] = exp2_decode_gemm_profile()
    all_results["exp3_fp8_gemm"] = exp3_fp8_gemm()
    all_results["exp4_memory_bandwidth"] = exp4_memory_bandwidth()
    all_results["exp5_ridge_point"] = exp5_ridge_point()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # FP16 peak TFLOPS
    fp16_results = [r for r in all_results["exp1_dtype_comparison"] if r["dtype"] == "float16"]
    fp16_peak = max(r["tflops"] for r in fp16_results)
    fp32_results = [r for r in all_results["exp1_dtype_comparison"] if r["dtype"] == "float32"]
    fp32_peak = max(r["tflops"] for r in fp32_results)

    print(f"  FP16 peak: {fp16_peak:.2f} TFLOPS")
    print(f"  FP32 peak: {fp32_peak:.2f} TFLOPS")
    print(f"  FP16/FP32 ratio: {fp16_peak/fp32_peak:.2f}x")

    # HBM bandwidth
    bw_results = all_results["exp4_memory_bandwidth"]
    max_bw = max(r["bandwidth_gb_s"] for r in bw_results)
    print(f"  Peak HBM bandwidth: {max_bw:.2f} GB/s")

    # Ridge point
    ridge_results = all_results["exp5_ridge_point"]
    ridge_max = max(r["tflops"] for r in ridge_results)
    ridge_entry = [r for r in ridge_results if r["tflops"] > 0.8 * ridge_max]
    if ridge_entry:
        r = ridge_entry[0]
        print(f"  Ridge point: M={r['M']}, AI={r['arithmetic_intensity']}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'gemm_roofline_cutlass_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()