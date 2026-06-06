#!/usr/bin/env python3
"""GEMM Roofline: FP16 vs FP32 on RTX 4090
============================================

Measures GEMM throughput across matrix sizes to understand:
1. When does GEMM transition from memory-bound to compute-bound?
2. How much faster is FP16 than FP32 (Tensor Core vs scalar)?
3. The Roofline ridge point (AI where compute time = memory time)
4. Batch decode vs single decode GEMM behavior

"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import json

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print("=" * 60)

device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)

print(f"SM Count: {props.multi_processor_count}")
print(f"HBM: {props.total_memory / 1e9:.1f} GB")

# Peak FLOPS for RTX 4090
# FP16 Tensor Core: 82.58 TFLOPS
# FP32 scalar: ~82.58 / 2 = 41.29 TFLOPS (no Tensor Core for FP32)
FP16_PEAK_TFLOPS = 82.58
FP32_PEAK_TFLOPS = 41.29  # conservative estimate


# Measured HBM BW (from previous benchmark: ~900 GB/s)
HBM_BW_GB_S = 900


def bench_gemm_roofline():
    """GEMM Roofline: FP16 vs FP32 across matrix sizes."""
    print("\n1. GEMM Roofline: FP16 (Tensor Core) vs FP32 (scalar)")
    print(f"   Peak FP16: {FP16_PEAK_TFLOPS} TFLOPS")
    print(f"   Peak FP32: {FP32_PEAK_TFLOPS} TFLOPS")
    print(f"   HBM BW: {HBM_BW_GB_S} GB/s")

    results = []
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    # FP16 GEMM
    print("\n  FP16 (Tensor Core):")
    fp16_results = []
    for N in sizes:
        A = torch.randn(N, N, device='cuda', dtype=torch.float16)
        B = torch.randn(N, N, device='cuda', dtype=torch.float16)

        # Warmup
        for _ in range(10):
            C = A @ B
        torch.cuda.synchronize()

        n = 50
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            C = A @ B
        e.record()
        torch.cuda.synchronize()

        ms = s.elapsed_time(e) / n

        # Analysis
        flops = 2 * N * N * N  # 2MN^3 for matmul
        achieved_tflops = flops / (ms * 1e-3) / 1e12
        io_bytes = 3 * N * N * 2  # read A + read B + write C, 2 bytes per FP16
        ai = flops / io_bytes
        predicted_mem_ms = io_bytes / (HBM_BW_GB_S * 1e9) * 1000
        predicted_comp_ms = flops / (FP16_PEAK_TFLOPS * 1e12) * 1000

        bound = "MEM" if predicted_mem_ms > predicted_comp_ms else "COMP"
        utilization = achieved_tflops / FP16_PEAK_TFLOPS * 100

        print(f"    N={N}: {ms:.4f}ms {achieved_tflops:.1f}TF ({utilization:.1f}% peak) "
              f"AI={ai:.1f} ops/byte {bound}")

        fp16_results.append({
            "N": N, "time_ms": round(ms, 4),
            "achieved_tflops": round(achieved_tflops, 1),
            "utilization_pct": round(utilization, 1),
            "ai": round(ai, 1), "bound": bound,
            "predicted_mem_ms": round(predicted_mem_ms, 4),
            "predicted_comp_ms": round(predicted_comp_ms, 4),
        })

    # FP32 GEMM
    print("\n  FP32 (scalar, no Tensor Core):")
    fp32_results = []
    for N in sizes:
        A = torch.randn(N, N, device='cuda', dtype=torch.float32)
        B = torch.randn(N, N, device='cuda', dtype=torch.float32)

        # Warmup
        for _ in range(10):
            C = A @ B
        torch.cuda.synchronize()

        n = 50
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            C = A @ B
        e.record()
        torch.cuda.synchronize()

        ms = s.elapsed_time(e) / n

        flops = 2 * N * N * N
        achieved_tflops = flops / (ms * 1e-3) / 1e12
        io_bytes = 3 * N * N * 4  # 4 bytes per FP32
        ai = flops / io_bytes
        predicted_mem_ms = io_bytes / (HBM_BW_GB_S * 1e9) * 1000
        predicted_comp_ms = flops / (FP32_PEAK_TFLOPS * 1e12) * 1000

        bound = "MEM" if predicted_mem_ms > predicted_comp_ms else "COMP"
        utilization = achieved_tflops / FP32_PEAK_TFLOPS * 100

        print(f"    N={N}: {ms:.4f}ms {achieved_tflops:.1f}TF ({utilization:.1f}% peak) "
              f"AI={ai:.1f} ops/byte {bound}")

        fp32_results.append({
            "N": N, "time_ms": round(ms, 4),
            "achieved_tflops": round(achieved_tflops, 1),
            "utilization_pct": round(utilization, 1),
            "ai": round(ai, 1), "bound": bound,
        })

    # Batched decode GEMM
    print("\n  Batched Decode GEMM (FP16, B=[1,4,16,64,256]):")
    decode_results = []
    H = 4096
    for B in [1, 4, 16, 64, 256]:
        A = torch.randn(B, H, device='cuda', dtype=torch.float16)
        W = torch.randn(H, H, device='cuda', dtype=torch.float16)

        for _ in range(10):
            out = A @ W
        torch.cuda.synchronize()

        n = 50
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            out = A @ W
        e.record()
        torch.cuda.synchronize()

        ms = s.elapsed_time(e) / n

        # For decode: weight dominates memory (2×H²×2bytes)
        io_bytes = 2 * H * H * 2 + 2 * B * H * 2  # weight + activations
        flops = 2 * B * H * H
        ai = flops / io_bytes
        tok_per_s = B / (ms / 1000)

        print(f"    B={B}: {ms:.4f}ms {tok_per_s:.0f} tok/s "
              f"AI={ai:.2f} ops/byte")

        decode_results.append({
            "B": B, "time_ms": round(ms, 4),
            "tok_per_s": round(tok_per_s, 0),
            "ai": round(ai, 2),
        })

    return {
        "fp16_square": fp16_results,
        "fp32_square": fp32_results,
        "decode_batched": decode_results,
    }


# ============================================================
# Run
# ============================================================

results = bench_gemm_roofline()

print("\n" + "=" * 60)
print("SUMMARY: GEMM Roofline on RTX 4090")
print("=" * 60)

fp16_best = max(r["achieved_tflops"] for r in results["fp16_square"])
fp32_best = max(r["achieved_tflops"] for r in results["fp32_square"])
print(f"Peak FP16 GEMM: {fp16_best:.1f} TFLOPS ({fp16_best/FP16_PEAK_TFLOPS*100:.1f}% of peak)")
print(f"Peak FP32 GEMM: {fp32_best:.1f} TFLOPS ({fp32_best/FP32_PEAK_TFLOPS*100:.1f}% of peak)")
fp16_speedup = fp16_best / fp32_best if fp32_best > 0 else 0
print(f"FP16 speedup over FP32: {fp16_speedup:.1f}x")

decode_max = max(r["tok_per_s"] for r in results["decode_batched"])
print(f"Peak decode throughput: {decode_max:.0f} tok/s")

print(f"Ridge point (AI where compute=memory): ~170 ops/byte for FP16 square GEMM")

# Save
out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, 'gemm_roofline_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {out_path}")