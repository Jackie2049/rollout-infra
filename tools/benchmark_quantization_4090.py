#!/usr/bin/env python3
"""Quantization Inference Benchmark on RTX 4090
=================================================

Measures the actual inference speedup from weight quantization (FP16 vs INT8 vs FP8)
on RTX 4090, and validates the quality impact.

Key questions:
A. How much speedup does INT8/FP8 weight-only quantization provide?
B. What's the quality impact (cosine similarity, max diff)?
C. Does quantization benefit scale with model size?
D. Memory savings vs speedup tradeoff

FP16 baseline: [B, H] @ [H, H] → weight 2×H² bytes (FP16)
INT8 weight-only: weight H² bytes, dequantize per-row → memory 2x savings
FP8 (E4M3): weight H²/2 bytes → memory 4x savings (if supported)
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
import json
import math

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print(f"FP8 support: {torch.float8_e4m3fn in [getattr(torch, k) for k in dir(torch) if 'float8' in k]}")
print("=" * 60)


# ============================================================
# 1. FP16 Baseline (matmul decode step)
# ============================================================

def bench_fp16_baseline():
    print("\n1. FP16 Baseline (reference)")

    results = []
    for B, H in [(1, 4096), (4, 4096), (16, 4096), (32, 4096), (128, 4096),
                 (1, 2048), (32, 2048), (128, 2048)]:
        x = torch.randn(B, H, device='cuda', dtype=torch.float16)
        w = torch.randn(H, H, device='cuda', dtype=torch.float16)
        w_mem = H * H * 2 / 1e6  # FP16 = 2 bytes/param

        # Warmup
        for _ in range(10):
            _ = x @ w
        torch.cuda.synchronize()

        n = 50
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            _ = x @ w
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / n

        print(f"  B={B} H={H}: {ms:.4f}ms weight={w_mem:.0f}MB")

        results.append({
            "B": B, "H": H, "dtype": "FP16",
            "time_ms": round(ms, 4),
            "weight_mb": round(w_mem, 0),
        })

    return results


# ============================================================
# 2. INT8 Weight-Only Quantization (simulate)
# ============================================================

def bench_int8_weight_only():
    """INT8 weight-only: store weights as INT8, dequantize per-row before matmul."""
    print("\n2. INT8 Weight-Only Quantization")

    results = []
    for B, H in [(1, 4096), (4, 4096), (16, 4096), (32, 4096), (128, 4096),
                 (1, 2048), (32, 2048), (128, 2048)]:
        # FP16 reference weight
        w_fp16 = torch.randn(H, H, device='cuda', dtype=torch.float16)

        # Quantize: per-channel INT8 (scale per row)
        w_max = w_fp16.abs().max(dim=1, keepdim=True).values  # [H, 1]
        scale = w_max / 127.0  # scale to INT8 range
        w_int8 = (w_fp16 / scale).round().clamp(-128, 127).to(torch.int8)  # [H, H] int8
        # Store scale as FP16
        scale_fp16 = scale.squeeze(1).to(torch.float16)  # [H]

        # Dequantize: w_dequant = w_int8 * scale (per-row)
        # This is what weight-only quantization does before matmul
        w_dequant = w_int8.to(torch.float16) * scale_fp16.unsqueeze(1)  # [H, H] fp16

        # Quality check
        cos_sim = F.cosine_similarity(w_fp16.flatten().unsqueeze(0),
                                       w_dequant.flatten().unsqueeze(0)).item()
        max_diff = (w_fp16 - w_dequant).abs().max().item()
        rel_diff = max_diff / w_fp16.abs().max().item()

        # Memory savings
        w_mem_fp16 = H * H * 2 / 1e6
        w_mem_int8 = H * H * 1 / 1e6 + H * 2 / 1e6  # int8 weight + fp16 scale
        mem_saved = (1 - w_mem_int8 / w_mem_fp16) * 100

        x = torch.randn(B, H, device='cuda', dtype=torch.float16)

        # Warmup dequantize + matmul
        for _ in range(10):
            w_dq = w_int8.to(torch.float16) * scale_fp16.unsqueeze(1)
            _ = x @ w_dq
        torch.cuda.synchronize()

        # Measure dequantize + matmul
        n = 50
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            w_dq = w_int8.to(torch.float16) * scale_fp16.unsqueeze(1)
            _ = x @ w_dq
        e.record()
        torch.cuda.synchronize()
        int8_ms = s.elapsed_time(e) / n

        # Also measure just the dequantize overhead
        s.record()
        for _ in range(n):
            _ = w_int8.to(torch.float16) * scale_fp16.unsqueeze(1)
        e.record()
        torch.cuda.synchronize()
        dequant_ms = s.elapsed_time(e) / n

        print(f"  B={B} H={H}: total={int8_ms:.4f}ms dequant={dequant_ms:.4f}ms "
              f"mem_saved={mem_saved:.1f}% cos_sim={cos_sim:.6f} rel_diff={rel_diff:.4f}")

        results.append({
            "B": B, "H": H, "dtype": "INT8_weight_only",
            "total_ms": round(int8_ms, 4),
            "dequant_ms": round(dequant_ms, 4),
            "weight_mb": round(w_mem_int8, 0),
            "mem_saved_pct": round(mem_saved, 1),
            "cos_sim": round(cos_sim, 6),
            "max_diff": round(max_diff, 4),
            "rel_diff": round(rel_diff, 4),
        })

    return results


# ============================================================
# 3. FP8 (E4M3) Weight Quantization (if supported)
# ============================================================

def bench_fp8_weight():
    """FP8 E4M3 weight: store as float8_e4m3fn, dequantize to FP16 before matmul."""
    print("\n3. FP8 E4M3 Weight Quantization")

    # Check if FP8 is available
    try:
        _ = torch.tensor([1.0], dtype=torch.float8_e4m3fn, device='cuda')
    except RuntimeError:
        print("  FP8 not supported on this GPU/driver, skipping")
        return []

    results = []
    for B, H in [(1, 4096), (4, 4096), (16, 4096), (32, 4096), (128, 4096),
                 (1, 2048), (32, 2048), (128, 2048)]:
        # FP16 reference
        w_fp16 = torch.randn(H, H, device='cuda', dtype=torch.float16)

        # Quantize to FP8 E4M3
        w_fp8 = w_fp16.to(torch.float8_e4m3fn)  # hardware-supported cast

        # Dequantize back to FP16
        w_dequant = w_fp8.to(torch.float16)

        # Quality check
        cos_sim = F.cosine_similarity(w_fp16.flatten().unsqueeze(0),
                                       w_dequant.flatten().unsqueeze(0)).item()
        max_diff = (w_fp16 - w_dequant).abs().max().item()
        rel_diff = max_diff / w_fp16.abs().max().item()

        # Memory savings
        w_mem_fp16 = H * H * 2 / 1e6
        w_mem_fp8 = H * H * 1 / 1e6  # FP8 = 1 byte/param
        mem_saved = (1 - w_mem_fp8 / w_mem_fp16) * 100

        x = torch.randn(B, H, device='cuda', dtype=torch.float16)

        # Warmup
        for _ in range(10):
            w_dq = w_fp8.to(torch.float16)
            _ = x @ w_dq
        torch.cuda.synchronize()

        # Measure
        n = 50
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            w_dq = w_fp8.to(torch.float16)
            _ = x @ w_dq
        e.record()
        torch.cuda.synchronize()
        fp8_ms = s.elapsed_time(e) / n

        # Dequantize overhead only
        s.record()
        for _ in range(n):
            _ = w_fp8.to(torch.float16)
        e.record()
        torch.cuda.synchronize()
        dequant_ms = s.elapsed_time(e) / n

        print(f"  B={B} H={H}: total={fp8_ms:.4f}ms dequant={dequant_ms:.4f}ms "
              f"mem_saved={mem_saved:.1f}% cos_sim={cos_sim:.6f} rel_diff={rel_diff:.4f}")

        results.append({
            "B": B, "H": H, "dtype": "FP8_E4M3",
            "total_ms": round(fp8_ms, 4),
            "dequant_ms": round(dequant_ms, 4),
            "weight_mb": round(w_mem_fp8, 0),
            "mem_saved_pct": round(mem_saved, 1),
            "cos_sim": round(cos_sim, 6),
            "max_diff": round(max_diff, 4),
            "rel_diff": round(rel_diff, 4),
        })

    return results


# ============================================================
# 4. Groupwise INT4 Quantization (AWQ-style)
# ============================================================

def bench_int4_awq_style():
    """INT4 groupwise quantization (AWQ/Marlin style, simulate with FP16 packing)."""
    print("\n4. INT4 Groupwise Quantization (AWQ-style, simulated)")

    GROUP_SIZE = 128  # AWQ default group size
    results = []

    for B, H in [(1, 4096), (4, 4096), (32, 4096), (128, 4096),
                 (1, 2048), (32, 2048), (128, 2048)]:
        # FP16 reference
        w_fp16 = torch.randn(H, H, device='cuda', dtype=torch.float16)

        # Simulate INT4 groupwise quantization
        # Each group of 128 elements has a scale (FP16) and 4-bit values
        n_groups = H // GROUP_SIZE
        # Scale per group per row: [H, n_groups]
        # Quantized values: [H, H] stored as 4-bit (packed 2 per byte)
        # But we simulate by storing quantized in FP16 (real kernel packs differently)

        # Per-group quantization
        w_reshaped = w_fp16.reshape(H, n_groups, GROUP_SIZE)  # [H, n_groups, 128]
        group_max = w_reshaped.abs().max(dim=2, keepdim=True).values  # [H, n_groups, 1]
        group_scale = group_max / 7.0  # 4-bit: range [-8, 7]
        w_int4 = (w_reshaped / group_scale).round().clamp(-8, 7)  # quantized
        w_dequant = (w_int4 * group_scale).reshape(H, H).to(torch.float16)  # dequantized

        # Quality check
        cos_sim = F.cosine_similarity(w_fp16.flatten().unsqueeze(0),
                                       w_dequant.flatten().unsqueeze(0)).item()
        max_diff = (w_fp16 - w_dequant).abs().max().item()
        rel_diff = max_diff / w_fp16.abs().max().item()

        # Memory: INT4 = 0.5 bytes/param, scale = 2 bytes * n_groups per row
        w_mem_fp16 = H * H * 2 / 1e6
        w_mem_int4 = H * H * 0.5 / 1e6 + H * n_groups * 2 / 1e6  # weights + scales
        mem_saved = (1 - w_mem_int4 / w_mem_fp16) * 100

        x = torch.randn(B, H, device='cuda', dtype=torch.float16)

        # Warmup: groupwise dequantize + matmul
        for _ in range(10):
            w_dq = (w_int4 * group_scale).reshape(H, H).to(torch.float16)
            _ = x @ w_dq
        torch.cuda.synchronize()

        n = 50
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            w_dq = (w_int4 * group_scale).reshape(H, H).to(torch.float16)
            _ = x @ w_dq
        e.record()
        torch.cuda.synchronize()
        int4_ms = s.elapsed_time(e) / n

        print(f"  B={B} H={H}: total={int4_ms:.4f}ms mem_saved={mem_saved:.1f}% "
              f"cos_sim={cos_sim:.6f} rel_diff={rel_diff:.4f}")

        results.append({
            "B": B, "H": H, "dtype": "INT4_AWQ_sim",
            "total_ms": round(int4_ms, 4),
            "weight_mb": round(w_mem_int4, 0),
            "mem_saved_pct": round(mem_saved, 1),
            "cos_sim": round(cos_sim, 6),
            "max_diff": round(max_diff, 4),
            "rel_diff": round(rel_diff, 4),
            "GROUP_SIZE": GROUP_SIZE,
        })

    return results


# ============================================================
# Run All Benchmarks
# ============================================================

fp16_results = bench_fp16_baseline()
int8_results = bench_int8_weight_only()
fp8_results = bench_fp8_weight()
int4_results = bench_int4_awq_style()

print("\n" + "=" * 60)
print("SUMMARY: Quantization Inference on RTX 4090")
print("=" * 60)

# Calculate speedups (INT8 and FP8 total vs FP16 baseline)
fp16_dict = {f"{r['B']}_{r['H']}": r['time_ms'] for r in fp16_results}

if int8_results:
    for r in int8_results:
        key = f"{r['B']}_{r['H']}"
        if key in fp16_dict:
            sp = fp16_dict[key] / r['total_ms']
            print(f"  INT8 B={r['B']} H={r['H']}: speedup={sp:.2f}x, "
                  f"mem_saved={r['mem_saved_pct']}%, cos_sim={r['cos_sim']}")

if fp8_results:
    for r in fp8_results:
        key = f"{r['B']}_{r['H']}"
        if key in fp16_dict:
            sp = fp16_dict[key] / r['total_ms']
            print(f"  FP8 B={r['B']} H={r['H']}: speedup={sp:.2f}x, "
                  f"mem_saved={r['mem_saved_pct']}%, cos_sim={r['cos_sim']}")

if int4_results:
    for r in int4_results:
        key = f"{r['B']}_{r['H']}"
        if key in fp16_dict:
            sp = fp16_dict[key] / r['total_ms']
            print(f"  INT4 B={r['B']} H={r['H']}: speedup={sp:.2f}x, "
                  f"mem_saved={r['mem_saved_pct']}%, cos_sim={r['cos_sim']}")

# Save
out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, 'quantization_benchmark_results.json')
with open(out_path, 'w') as f:
    json.dump({
        "fp16": fp16_results,
        "int8_weight_only": int8_results,
        "fp8_e4m3": fp8_results,
        "int4_awq_sim": int4_results,
    }, f, indent=2)
print(f"Results saved to {out_path}")