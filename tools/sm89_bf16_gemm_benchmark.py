#!/usr/bin/env python3
"""
SM89 BF16 GEMM Benchmark — RTX 4090 LLM-relevant shapes
Benchmark cuBLAS BF16 GEMM for decode (memory-bound) and prefill (compute-bound) shapes
Compute TFLOPS, arithmetic intensity, peak utilization, and roofline analysis

RTX 4090: SM89, 128 SMs, 16384 CUDA cores
BF16 Tensor Core peak: 165.2 TFLOPS (dense HMMA, same rate as FP16)
FP16 Tensor Core peak: 165.2 TFLOPS (dense HMMA)
FP32 peak (TF32): 82.6 TFLOPS
HBM bandwidth: 890.8 GB/s (实测, GEMM access pattern)
实测cuBLAS BF16 peak: 167.14 TFLOPS (101% of 165.2)

Ridge point: AI ≈ 2*165.2/0.891 ≈ 370 FLOPS/byte (compute→memory crossover)
"""

import torch
import time
import json
import argparse
import numpy as np

def calculate_tflops(M, N, K, time_ms):
    """Calculate TFLOPS for GEMM: 2*M*N*K FLOPS"""
    return 2.0 * M * N * K / (time_ms * 1e6)

def calculate_arithmetic_intensity(M, N, K):
    """AI = FLOPS/bytes for BF16 GEMM"""
    flops = 2.0 * M * N * K
    bytes_accessed = M * K * 2 + K * N * 2 + M * N * 2  # BF16 = 2 bytes
    return flops / bytes_accessed

def calculate_roofline(M, N, K, peak_tflops, hbm_bw_gbs):
    """Compute roofline prediction"""
    ai = calculate_arithmetic_intensity(M, N, K)
    ridge = 2 * peak_tflops / hbm_bw_gbs * 1000  # FLOPS/byte

    if ai < ridge:  # memory-bound
        predicted_tflops = ai * hbm_bw_gbs * 1000 / 2  # GB/s * AI * 1000 / 2 = TFLOPS
        bound = "memory"
    else:  # compute-bound
        predicted_tflops = peak_tflops
        bound = "compute"

    return predicted_tflops, bound, ridge

def benchmark_gemm(M, N, K, warmup=20, runs=200):
    """Benchmark BF16 GEMM: C = A @ B with proper synchronization"""
    device = torch.device('cuda')

    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(warmup):
        C = torch.mm(A, B)
    torch.cuda.synchronize()

    # For small shapes (M < 128), use per-GEMM sync timing to avoid async batching
    # For large shapes, use batched event timing for efficiency
    if M < 128:
        times = []
        for _ in range(runs):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            C = torch.mm(A, B)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        ms = np.median(times)
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(runs):
            C = torch.mm(A, B)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / runs

    return ms

def run_benchmark():
    device = torch.device('cuda')

    # Device info
    device_name = torch.cuda.get_device_name()
    device_props = torch.cuda.get_device_properties(device)

    # Measured values
    PEAK_TFLOPS = 165.2  # BF16 Tensor Core dense peak for RTX 4090
    HBM_BW_GBS = 890.8   # 实测 GEMM access pattern bandwidth

    results = {
        "device": device_name,
        "peak_tflops_bf16": PEAK_TFLOPS,
        "hbm_bw_gbs": HBM_BW_GBS,
        "ridge_flops_per_byte": 2 * PEAK_TFLOPS / HBM_BW_GBS * 1000,
    }

    # LLM-relevant GEMM shapes
    shapes = [
        # Decode shapes (memory-bound, small M = batch size)
        # LLaMA-7B: hidden=4096, intermediate=11008 (≈10240 rounded)
        ("decode_B=1_attn",    1,    4096, 4096),
        ("decode_B=1_gate",    1,    11008, 4096),
        ("decode_B=4_attn",    4,    4096, 4096),
        ("decode_B=8_attn",    8,    4096, 4096),
        ("decode_B=16_attn",   16,   4096, 4096),
        ("decode_B=32_attn",   32,   4096, 4096),
        ("decode_B=32_gate",   32,   11008, 4096),
        ("decode_B=64_attn",   64,   4096, 4096),
        ("decode_B=128_attn",  128,  4096, 4096),
        ("decode_B=128_gate",  128,  11008, 4096),
        ("decode_B=256_attn",  256,  4096, 4096),

        # Prefill shapes (compute-bound, large M = seq_len * batch)
        ("prefill_S=256",      256,  4096, 4096),
        ("prefill_S=512",      512,  4096, 4096),
        ("prefill_S=1024",     1024, 4096, 4096),
        ("prefill_S=2048",     2048, 4096, 4096),
        ("prefill_S=4096",     4096, 4096, 4096),
        ("prefill_S=8192",     8192, 4096, 4096),

        # MoE-like shapes (small M per expert)
        ("moe_M=4_gate",       4,    11008, 4096),
        ("moe_M=8_gate",       8,    11008, 4096),
        ("moe_M=16_gate",      16,   11008, 4096),

        # Square matrix sweep (for roofline analysis)
        ("square_256",         256,  256, 256),
        ("square_512",         512,  512, 512),
        ("square_1024",        1024, 1024, 1024),
        ("square_2048",        2048, 2048, 2048),
        ("square_4096",        4096, 4096, 4096),
        ("square_8192",        8192, 8192, 8192),
    ]

    shape_results = []

    print(f"SM89 BF16 GEMM Benchmark — {device_name}")
    print(f"Peak BF16: {PEAK_TFLOPS} TFLOPS, HBM BW: {HBM_BW_GBS} GB/s")
    print(f"Ridge point: {2*PEAK_TFLOPS/HBM_BW_GBS*1000:.1f} FLOPS/byte")
    print()
    print(f"{'Shape':<25} {'M':>6} {'N':>6} {'K':>6} {'AI':>8} {'ms':>8} {'TFLOPS':>8} {'Peak%':>7} {'Bound':>8} {'Roofline%':>10}")
    print("-" * 105)

    for name, M, N, K in shapes:
        ai = calculate_arithmetic_intensity(M, N, K)
        roofline_pred, bound, ridge = calculate_roofline(M, N, K, PEAK_TFLOPS, HBM_BW_GBS)

        try:
            ms = benchmark_gemm(M, N, K)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{name:<25} {M:>6} {N:>6} {K:>6} OOM!")
                torch.cuda.empty_cache()
                continue
            raise

        tflops = calculate_tflops(M, N, K, ms)
        peak_pct = tflops / PEAK_TFLOPS * 100
        roofline_pct = tflops / roofline_pred * 100

        print(f"{name:<25} {M:>6} {N:>6} {K:>6} {ai:>8.1f} {ms:>8.4f} {tflops:>8.2f} {peak_pct:>7.1f}% {bound:>8} {roofline_pct:>10.1f}%")

        shape_results.append({
            "name": name, "M": M, "N": N, "K": K,
            "ai": ai, "time_ms": ms, "tflops": tflops,
            "peak_pct": peak_pct, "bound": bound,
            "roofline_pct": roofline_pct,
            "ridge": ridge,
        })

        torch.cuda.empty_cache()

    results["shapes"] = shape_results

    # Summary analysis
    decode_shapes = [s for s in shape_results if "decode" in s["name"]]
    prefill_shapes = [s for s in shape_results if "prefill" in s["name"]]
    moe_shapes = [s for s in shape_results if "moe" in s["name"]]

    print()
    print("=== Summary ===")

    if decode_shapes:
        avg_decode_peak = np.mean([s["peak_pct"] for s in decode_shapes])
        min_decode_peak = np.min([s["peak_pct"] for s in decode_shapes])
        print(f"Decode: avg {avg_decode_peak:.1f}% peak, min {min_decode_peak:.1f}% peak → {bound}")
        print(f"  B=1: {decode_shapes[0]['tflops']:.2f} TFLOPS ({decode_shapes[0]['peak_pct']:.1f}% peak)")
        b32 = [s for s in decode_shapes if "B=32_attn" in s["name"]]
        if b32:
            print(f"  B=32: {b32[0]['tflops']:.2f} TFLOPS ({b32[0]['peak_pct']:.1f}% peak)")

    if prefill_shapes:
        avg_prefill_peak = np.mean([s["peak_pct"] for s in prefill_shapes])
        max_prefill_peak = np.max([s["peak_pct"] for s in prefill_shapes])
        print(f"Prefill: avg {avg_prefill_peak:.1f}% peak, max {max_prefill_peak:.1f}% peak → compute-bound")

    if moe_shapes:
        avg_moe_peak = np.mean([s["peak_pct"] for s in moe_shapes])
        print(f"MoE: avg {avg_moe_peak:.1f}% peak → extremely memory-bound")

    # Key insight
    print()
    print("=== Key Insights ===")
    b1 = [s for s in shape_results if "B=1_attn" in s["name"]]
    if b1:
        print(f"Decode B=1: {b1[0]['peak_pct']:.1f}% peak → 98%+ Tensor Core idle!")
        print(f"  INT4 can increase to ~{b1[0]['peak_pct']*2:.1f}% peak (2x bandwidth saving)")

    s4096 = [s for s in shape_results if "S=4096" in s["name"]]
    if s4096:
        print(f"Prefill S=4096: {s4096[0]['peak_pct']:.1f}% peak → near-optimal compute utilization")

    # Find crossover
    square_shapes = [s for s in shape_results if "square" in s["name"]]
    if square_shapes:
        for s in square_shapes:
            if s["bound"] == "compute":
                prev = [p for p in square_shapes if p["M"] < s["M"]]
                if prev and prev[-1]["bound"] == "memory":
                    print(f"Crossover: M={prev[-1]['M']}→{s['M']} (AI={prev[-1]['ai']:.1f}→{s['ai']:.1f})")
                    break

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/sm89_bf16_gemm_benchmark.json")
    args = parser.parse_args()

    results = run_benchmark()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")