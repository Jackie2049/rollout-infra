#!/usr/bin/env python3
"""Quantization Inference Benchmark — RTX 4090
=================================================
Measures real quantized inference performance with:
1. INT8 dynamic quantization (PyTorch native)
2. FP8 E4M3 / FP8 E5M2 (RTX 4090 SM89 supports FP8)
3. INT4 weight-only quantization (GPTQ-style simulation)
4. Mixed-precision: FP8 weights + FP16 compute
5. Quantization error analysis (cos_sim, MSE per dtype)
6. Memory savings vs throughput tradeoff
"""

import torch
import torch.nn as nn
import time
import json
import math


def benchmark_fn(name, fn, n_runs=50, warmup=10):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    median = sorted(times)[len(times) // 2]
    return {"name": name, "median_ms": median, "min_ms": min(times)}


def quantize_to_int8_linear(linear_layer):
    """Apply PyTorch dynamic INT8 quantization to a linear layer."""
    # PyTorch dynamic quant: weight INT8, activation FP32 computed
    quantized = torch.quantization.quantize_dynamic(
        linear_layer, {nn.Linear}, dtype=torch.qint8
    )
    return quantized


def simulate_int4_weights(weight, group_size=128):
    """Simulate INT4 weight-only quantization (GPTQ-style).

    Group quantization: each group of `group_size` elements shares
    one scale and zero-point.
    """
    orig_shape = weight.shape
    weight_flat = weight.flatten()

    # Pad to group_size
    n_elements = weight_flat.numel()
    n_groups = math.ceil(n_elements / group_size)
    padded = torch.zeros(n_groups * group_size, device=weight.device, dtype=weight.dtype)
    padded[:n_elements] = weight_flat

    # Group-wise quantization
    groups = padded.reshape(n_groups, group_size)
    w_max = groups.abs().max(dim=-1).values.clamp(min=1e-5)
    scale = w_max / 7.0  # INT4 range: -8 to 7
    zero_point = torch.zeros_like(scale)

    # Quantize
    groups_q = torch.round(groups / scale.unsqueeze(-1)).clamp(-8, 7)
    # Dequantize
    groups_deq = groups_q * scale.unsqueeze(-1)

    dequantized = groups_deq.reshape(-1)[:n_elements].reshape(orig_shape)
    return dequantized, scale, zero_point


def run_experiment():
    device = "cuda"
    results = {}

    # Model sizes to test
    model_configs = {
        "2M": {"d": 256, "heads": 8, "layers": 4, "vocab": 32000},
        "25M": {"d": 512, "heads": 16, "layers": 8, "vocab": 32000},
        "125M": {"d": 1024, "heads": 16, "layers": 16, "vocab": 32000},
    }

    # Experiment 1: FP8 GEMM throughput
    print("=== Exp 1: FP8 vs FP16 vs INT8 GEMM Throughput ===")

    for label, cfg in model_configs.items():
        d = cfg["d"]
        batch_sizes = [1, 8, 32, 128, 256]

        gemm_results = {}
        for B in batch_sizes:
            # FP16 baseline
            A_fp16 = torch.randn(B, d, device=device, dtype=torch.float16)
            W_fp16 = torch.randn(d, d, device=device, dtype=torch.float16)

            fp16_res = benchmark_fn(
                f"FP16_B{B}", lambda: A_fp16 @ W_fp16
            )

            # FP8 E4M3: store as FP8, dequantize to FP16 for compute
            # This is how FP8 is actually used in inference (not direct GEMM)
            W_fp8_e4m3_stored = W_fp16.to(torch.float8_e4m3fn)
            A_fp8_e4m3_stored = A_fp16.to(torch.float8_e4m3fn)

            # FP8 weight-only: dequantize weight then compute in FP16
            fp8_wonly_res = benchmark_fn(
                f"FP8_WONLY_B{B}", lambda: A_fp16 @ W_fp8_e4m3_stored.to(torch.float16)
            )

            # FP8 full: dequantize both then compute
            fp8_full_res = benchmark_fn(
                f"FP8_FULL_B{B}",
                lambda: A_fp8_e4m3_stored.to(torch.float16) @ W_fp8_e4m3_stored.to(torch.float16)
            )

            # FP8 E5M2: same pattern
            W_fp8_e5m2_stored = W_fp16.to(torch.float8_e5m2)
            fp8_e5m2_wonly_res = benchmark_fn(
                f"FP8_E5M2_WONLY_B{B}", lambda: A_fp16 @ W_fp8_e5m2_stored.to(torch.float16)
            )

            # INT8 matmul (simulate: quantize then dequantize)
            # Note: cuBLAS INT8 gemm requires specific conditions
            # We test the cast-based approach first
            A_int8_cast = A_fp16.to(torch.int8).to(torch.float16)
            W_int8_cast = W_fp16.to(torch.int8).to(torch.float16)

            int8_cast_res = benchmark_fn(
                f"INT8_CAST_B{B}", lambda: A_int8_cast @ W_int8_cast
            )

            # INT4 weight-only (dequantized weight, FP16 activation)
            W_int4_deq, _, _ = simulate_int4_weights(W_fp16, group_size=128)
            int4_res = benchmark_fn(
                f"INT4_WONLY_B{B}", lambda: A_fp16 @ W_int4_deq
            )

            # Calculate TFLOPS (2*d^2*B FLOPS for matmul)
            flops = 2 * B * d * d
            fp16_tflops = flops / fp16_res["median_ms"] / 1e9 * 1e3

            gemm_results[f"B{B}"] = {
                "fp16_ms": fp16_res["median_ms"],
                "fp8_e4m3_wonly_ms": fp8_wonly_res["median_ms"],
                "fp8_e4m3_full_ms": fp8_full_res["median_ms"],
                "fp8_e5m2_wonly_ms": fp8_e5m2_wonly_res["median_ms"],
                "int8_cast_ms": int8_cast_res["median_ms"],
                "int4_wonly_ms": int4_res["median_ms"],
                "fp16_tflops": fp16_tflops,
                "fp8_wonly_vs_fp16_ratio": fp8_wonly_res["median_ms"] / fp16_res["median_ms"] if fp16_res["median_ms"] > 0 else 0,
                "int4_vs_fp16_ratio": int4_res["median_ms"] / fp16_res["median_ms"] if fp16_res["median_ms"] > 0 else 0,
            }

            print(f"  {label} B={B}: FP16={fp16_res['median_ms']:.3f}ms "
                  f"FP8_WONLY={fp8_wonly_res['median_ms']:.3f}ms "
                  f"(ratio={fp8_wonly_res['median_ms']/fp16_res['median_ms']:.2f}x) "
                  f"FP8_FULL={fp8_full_res['median_ms']:.3f}ms "
                  f"INT4_WONLY={int4_res['median_ms']:.3f}ms "
                  f"(ratio={int4_res['median_ms']/fp16_res['median_ms']:.2f}x)")

        results[f"gemm_{label}"] = gemm_results

    # Experiment 2: Quantization error analysis
    print("\n=== Exp 2: Quantization Error Analysis ===")

    for label, cfg in model_configs.items():
        d = cfg["d"]
        W_fp16 = torch.randn(d, d, device=device, dtype=torch.float16)
        W_fp32 = W_fp16.float()

        # FP8 E4M3 error
        W_fp8_e4m3 = W_fp16.to(torch.float8_e4m3fn).to(torch.float16)
        fp8_e4m3_cos = torch.nn.functional.cosine_similarity(
            W_fp32.flatten(), W_fp8_e4m3.float().flatten(), dim=0
        ).item()
        fp8_e4m3_mse = ((W_fp32 - W_fp8_e4m3.float()) ** 2).mean().item()

        # FP8 E5M2 error
        W_fp8_e5m2 = W_fp16.to(torch.float8_e5m2).to(torch.float16)
        fp8_e5m2_cos = torch.nn.functional.cosine_similarity(
            W_fp32.flatten(), W_fp8_e5m2.float().flatten(), dim=0
        ).item()
        fp8_e5m2_mse = ((W_fp32 - W_fp8_e5m2.float()) ** 2).mean().item()

        # INT8 error (weight-only)
        W_int8 = torch.round(W_fp32 * 127 / W_fp32.abs().max()).clamp(-128, 127).to(torch.float16)
        W_int8_deq = W_int8 * (W_fp32.abs().max() / 127)
        int8_cos = torch.nn.functional.cosine_similarity(
            W_fp32.flatten(), W_int8_deq.float().flatten(), dim=0
        ).item()
        int8_mse = ((W_fp32 - W_int8_deq.float()) ** 2).mean().item()

        # INT4 error (group-wise)
        W_int4_deq, _, _ = simulate_int4_weights(W_fp16, group_size=128)
        int4_cos = torch.nn.functional.cosine_similarity(
            W_fp32.flatten(), W_int4_deq.float().flatten(), dim=0
        ).item()
        int4_mse = ((W_fp32 - W_int4_deq.float()) ** 2).mean().item()

        # INT4 group_size sweep
        int4_gs_sweep = {}
        for gs in [32, 64, 128, 256]:
            W_gs_deq, _, _ = simulate_int4_weights(W_fp16, group_size=gs)
            gs_cos = torch.nn.functional.cosine_similarity(
                W_fp32.flatten(), W_gs_deq.float().flatten(), dim=0
            ).item()
            gs_mse = ((W_fp32 - W_gs_deq.float()) ** 2).mean().item()
            int4_gs_sweep[f"gs{gs}"] = {"cos_sim": gs_cos, "mse": gs_mse}

        results[f"error_{label}"] = {
            "fp8_e4m3": {"cos_sim": fp8_e4m3_cos, "mse": fp8_e4m3_mse},
            "fp8_e5m2": {"cos_sim": fp8_e5m2_cos, "mse": fp8_e5m2_mse},
            "int8_weight_only": {"cos_sim": int8_cos, "mse": int8_mse},
            "int4_group128": {"cos_sim": int4_cos, "mse": int4_mse},
            "int4_gs_sweep": int4_gs_sweep,
        }

        print(f"  {label} d={d}: FP8_E4M3 cos={fp8_e4m3_cos:.4f} MSE={fp8_e4m3_mse:.6f} "
              f"FP8_E5M2 cos={fp8_e5m2_cos:.4f} MSE={fp8_e5m2_mse:.6f} "
              f"INT8 cos={int8_cos:.4f} MSE={int8_mse:.6f} "
              f"INT4 cos={int4_cos:.4f} MSE={int4_mse:.6f}")

    # Experiment 3: Memory savings
    print("\n=== Exp 3: Memory Savings Analysis ===")

    for label, cfg in model_configs.items():
        d = cfg["d"]
        vocab = cfg["vocab"]
        n_layers = cfg["layers"]
        d_ff = 4 * d

        # Count parameter bytes per dtype
        # Per layer: Q/K/V/O proj (4×d²) + MLP up/down/gate (3×d×d_ff) + norms
        layer_params = 4 * d * d + 3 * d * d_ff + 2 * d  # ~approx
        total_params = vocab * d + n_layers * layer_params + d  # embed + layers + final_ln

        fp16_bytes = total_params * 2  # 2 bytes per param
        fp8_bytes = total_params * 1    # 1 byte per param
        int8_bytes = total_params * 1   # 1 byte per param
        int4_bytes = total_params * 0.5  # 0.5 bytes per param (packed)

        memory_results = {
            "total_params": total_params,
            "fp16_memory_MB": fp16_bytes / 1e6,
            "fp8_memory_MB": fp8_bytes / 1e6,
            "int8_memory_MB": int8_bytes / 1e6,
            "int4_memory_MB": int4_bytes / 1e6,
            "fp8_vs_fp16_savings": (1 - fp8_bytes / fp16_bytes) * 100,
            "int8_vs_fp16_savings": (1 - int8_bytes / fp16_bytes) * 100,
            "int4_vs_fp16_savings": (1 - int4_bytes / fp16_bytes) * 100,
            "fits_in_24GB_fp16": fp16_bytes / 1e9 <= 24,
            "fits_in_24GB_int4": int4_bytes / 1e9 <= 24,
        }

        results[f"memory_{label}"] = memory_results

        print(f"  {label}: FP16={fp16_bytes/1e6:.1f}MB "
              f"FP8={fp8_bytes/1e6:.1f}MB(saves{(1-fp8_bytes/fp16_bytes)*100:.0f}%) "
              f"INT4={int4_bytes/1e6:.1f}MB(saves{(1-int4_bytes/fp16_bytes)*100:.0f}%) "
              f"24GB fits: FP16={fp16_bytes/1e9<=24} INT4={int4_bytes/1e9<=24}")

    # Experiment 4: Decode throughput (memory-bound scenario)
    print("\n=== Exp 4: Decode Throughput — Quantized vs FP16 ===")

    for label, cfg in model_configs.items():
        d = cfg["d"]
        B_values = [1, 8, 32, 128]

        decode_results = {}
        for B in B_values:
            # Simulate decode: single token (B, 1) × weight matrix
            x_fp16 = torch.randn(B, 1, d, device=device, dtype=torch.float16)
            W_fp16 = torch.randn(d, d, device=device, dtype=torch.float16)

            # FP16 decode
            fp16_decode = benchmark_fn(
                f"FP16_decode_B{B}",
                lambda: x_fp16 @ W_fp16
            )

            # INT4 weight-only decode (dequantized weight)
            W_int4_deq, _, _ = simulate_int4_weights(W_fp16, group_size=128)
            int4_decode = benchmark_fn(
                f"INT4_decode_B{B}",
                lambda: x_fp16 @ W_int4_deq
            )

            # FP8 decode (weight-only: dequantize then compute)
            W_fp8_stored = W_fp16.to(torch.float8_e4m3fn)
            fp8_decode = benchmark_fn(
                f"FP8_decode_B{B}",
                lambda: x_fp16 @ W_fp8_stored.to(torch.float16)
            )

            decode_results[f"B{B}"] = {
                "fp16_ms": fp16_decode["median_ms"],
                "int4_wonly_ms": int4_decode["median_ms"],
                "fp8_ms": fp8_decode["median_ms"],
                "int4_vs_fp16": int4_decode["median_ms"] / fp16_decode["median_ms"],
                "fp8_vs_fp16": fp8_decode["median_ms"] / fp16_decode["median_ms"],
            }

            print(f"  {label} B={B}: FP16={fp16_decode['median_ms']:.3f}ms "
                  f"INT4_WONLY={int4_decode['median_ms']:.3f}ms "
                  f"(ratio={int4_decode['median_ms']/fp16_decode['median_ms']:.2f}x) "
                  f"FP8={fp8_decode['median_ms']:.3f}ms "
                  f"(ratio={fp8_decode['median_ms']/fp16_decode['median_ms']:.2f}x)")

        results[f"decode_{label}"] = decode_results

    # Experiment 5: KV Cache quantization impact
    print("\n=== Exp 5: KV Cache Quantization ===")

    S_values = [512, 1024, 2048, 4096, 8192]
    B = 32
    d = 512  # 25M model dimension
    n_heads = 16
    d_head = d // n_heads

    kv_results = {}
    for S in S_values:
        # FP16 KV cache
        K_fp16 = torch.randn(B, n_heads, S, d_head, device=device, dtype=torch.float16)
        V_fp16 = torch.randn(B, n_heads, S, d_head, device=device, dtype=torch.float16)
        Q = torch.randn(B, n_heads, 1, d_head, device=device, dtype=torch.float16)

        fp16_kv_bytes = 2 * B * n_heads * S * d_head * 2  # K+V, 2 bytes each

        # FP16 attention
        fp16_attn = benchmark_fn(
            f"FP16_attn_S{S}",
            lambda: torch.softmax(Q @ K_fp16.transpose(-2, -1) / math.sqrt(d_head), dim=-1) @ V_fp16
        )

        # FP8 KV: store as FP8, dequantize to FP16 for attention compute
        K_fp8 = K_fp16.to(torch.float8_e4m3fn)
        V_fp8 = V_fp16.to(torch.float8_e4m3fn)
        fp8_kv_bytes = 2 * B * n_heads * S * d_head * 1  # K+V, 1 byte each

        # FP8 attention (dequantize KV then compute in FP16)
        fp8_attn = benchmark_fn(
            f"FP8_attn_S{S}",
            lambda: torch.softmax(
                Q @ K_fp8.to(torch.float16).transpose(-2, -1) / math.sqrt(d_head),
                dim=-1
            ) @ V_fp8.to(torch.float16)
        )

        # INT8 KV (weight-only style)
        K_int8_scale = K_fp16.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
        V_int8_scale = V_fp16.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
        K_int8 = torch.round(K_fp16 / K_int8_scale).clamp(-128, 127).to(torch.int8)
        V_int8 = torch.round(V_fp16 / V_int8_scale).clamp(-128, 127).to(torch.int8)
        K_int8_deq = K_int8.to(torch.float16) * K_int8_scale.to(torch.float16)
        V_int8_deq = V_int8.to(torch.float16) * V_int8_scale.to(torch.float16)
        int8_kv_bytes = 2 * B * n_heads * S * d_head * 1

        int8_attn = benchmark_fn(
            f"INT8_attn_S{S}",
            lambda: torch.softmax(
                Q @ K_int8_deq.transpose(-2, -1) / math.sqrt(d_head),
                dim=-1
            ) @ V_int8_deq
        )

        # Error analysis for KV
        fp8_kv_cos = torch.nn.functional.cosine_similarity(
            K_fp16.flatten().float(), K_fp8.to(torch.float16).flatten().float(), dim=0
        ).item()
        int8_kv_cos = torch.nn.functional.cosine_similarity(
            K_fp16.flatten().float(), K_int8_deq.flatten().float(), dim=0
        ).item()

        kv_results[f"S{S}"] = {
            "fp16_attn_ms": fp16_attn["median_ms"],
            "fp8_attn_ms": fp8_attn["median_ms"],
            "int8_attn_ms": int8_attn["median_ms"],
            "fp8_vs_fp16": fp8_attn["median_ms"] / fp16_attn["median_ms"],
            "int8_vs_fp16": int8_attn["median_ms"] / fp16_attn["median_ms"],
            "fp16_kv_MB": fp16_kv_bytes / 1e6,
            "fp8_kv_MB": fp8_kv_bytes / 1e6,
            "int8_kv_MB": int8_kv_bytes / 1e6,
            "fp8_kv_savings_pct": (1 - fp8_kv_bytes / fp16_kv_bytes) * 100,
            "int8_kv_savings_pct": (1 - int8_kv_bytes / fp16_kv_bytes) * 100,
            "fp8_kv_cos_sim": fp8_kv_cos,
            "int8_kv_cos_sim": int8_kv_cos,
        }

        print(f"  S={S}: FP16={fp16_attn['median_ms']:.3f}ms "
              f"FP8={fp8_attn['median_ms']:.3f}ms(ratio={fp8_attn['median_ms']/fp16_attn['median_ms']:.2f}x) "
              f"INT8={int8_attn['median_ms']:.3f}ms(ratio={int8_attn['median_ms']/fp16_attn['median_ms']:.2f}x) "
              f"KV FP16={fp16_kv_bytes/1e6:.1f}MB FP8={fp8_kv_bytes/1e6:.1f}MB "
              f"saves{(1-fp8_kv_bytes/fp16_kv_bytes)*100:.0f}% "
              f"FP8_cos={fp8_kv_cos:.4f} INT8_cos={int8_kv_cos:.4f}")

    results["kv_cache"] = kv_results

    # Save results
    with open("results/quantization_inference_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/quantization_inference_benchmark.json")

    return results


if __name__ == "__main__":
    run_experiment()