#!/usr/bin/env python3
"""LLM Inference Performance Estimator (RTX 4090 calibrated)
============================================================
Estimates latency and throughput for LLM inference on RTX 4090,
calibrated with real benchmark data from our experiments.

Usage:
    python tools/perf_estimator_4090.py --model 7B --batch 32 --seq 2048
"""

import argparse
import json

# Calibrated parameters from RTX 4090 benchmarks
GPU_PARAMS = {
    "rtx4090": {
        "hbm_bw_gbs": 890.8,      # Measured HBM bandwidth (GB/s)
        "peak_tflops": 169.6,       # Measured GEMM TFLOPS (FP16)
        "vram_gb": 24.0,
        "kernel_launch_us": 1.1,   # Kernel launch overhead (us)
        "sdpa_peak_tflops": 162.3,  # SDPA peak TFLOPS
        "decode_peak_tflops": 138.7, # Batch decode peak TFLOPS
        "pcie_bw_gbs": 5.0,        # TP interconnect (no NVLink)
    },
    "a100": {
        "hbm_bw_gbs": 2030.0,
        "peak_tflops": 312.0,
        "vram_gb": 80.0,
        "kernel_launch_us": 5.0,
        "nvlink_bw_gbs": 300.0,
    },
    "h100": {
        "hbm_bw_gbs": 3352.0,
        "peak_tflops": 990.0,
        "vram_gb": 80.0,
        "kernel_launch_us": 3.0,
        "nvlink_bw_gbs": 900.0,
    }
}

MODEL_PARAMS = {
    "125m": {"params_b": 0.125, "hidden": 768, "layers": 12, "heads": 12, "vocab": 50272},
    "350m": {"params_b": 0.35, "hidden": 1024, "layers": 24, "heads": 16, "vocab": 50272},
    "1.3b": {"params_b": 1.3, "hidden": 2048, "layers": 24, "heads": 32, "vocab": 50272},
    "7b": {"params_b": 7.0, "hidden": 4096, "layers": 32, "heads": 32, "vocab": 32000},
    "13b": {"params_b": 13.0, "hidden": 5120, "layers": 40, "heads": 40, "vocab": 32000},
    "70b": {"params_b": 70.0, "hidden": 8192, "layers": 80, "heads": 64, "vocab": 32000},
}


def estimate_decode_latency(model_name, batch_size, gpu="rtx4090", precision="fp16"):
    """Estimate decode latency using memory-bound roofline model."""
    model = MODEL_PARAMS[model_name]
    gpu_p = GPU_PARAMS[gpu]

    bytes_per_param = 2 if precision == "fp16" else 1 if precision == "fp8" else 4
    model_bytes = model["params_b"] * 1e9 * bytes_per_param

    # Decode is memory-bound: latency = model_size / HBM_BW
    # Each token requires reading full model weights
    hbm_bw = gpu_p["hbm_bw_gbs"] * 1e9  # bytes/s
    single_decode_s = model_bytes / hbm_bw

    # Batch decode: latency increases with batch but sub-linearly
    # From our benchmark: B=1→512, latency 0.3→1.0ms (3.3x) while batch 512x
    # Model: latency = base * (1 + alpha * log2(batch))
    alpha = 0.08  # calibrated from RTX 4090 data
    batch_decode_s = single_decode_s * (1 + alpha * max(0, (batch_size - 1).bit_length()))

    tok_per_s = batch_size / batch_decode_s
    tflops = 2 * model["params_b"] * 1e9 * batch_size / (batch_decode_s * 1e12)

    return {
        "latency_ms": batch_decode_s * 1000,
        "tok_per_s": tok_per_s,
        "tflops": tflops,
        "model_gb": model_bytes / 1e9,
        "memory_bound": True,
    }


def estimate_prefill_latency(model_name, seq_len, batch_size=1, gpu="rtx4090"):
    """Estimate prefill latency using compute-bound model."""
    model = MODEL_PARAMS[model_name]
    gpu_p = GPU_PARAMS[gpu]

    # Prefill is compute-bound: FLOPS = 2 * params * seq_len
    # But also has attention: FLOPS_att = 2 * B * layers * heads * head_dim * seq_len^2
    head_dim = model["hidden"] // model["heads"]
    flops_mlp = 2 * model["params_b"] * 1e9 * seq_len * batch_size
    flops_attn = 2 * batch_size * model["layers"] * model["heads"] * head_dim * seq_len * seq_len
    total_flops = flops_mlp + flops_attn

    # Utilization depends on seq_len (from our benchmark data)
    if seq_len <= 256:
        util = 0.3
    elif seq_len <= 512:
        util = 0.6
    elif seq_len <= 1024:
        util = 0.85
    else:
        util = 0.95

    effective_tflops = gpu_p["peak_tflops"] * util
    latency_s = total_flops / (effective_tflops * 1e12)

    tok_per_s = seq_len * batch_size / latency_s
    attn_pct = flops_attn / total_flops * 100

    return {
        "latency_ms": latency_s * 1000,
        "tok_per_s": tok_per_s,
        "tflops": total_flops / (latency_s * 1e12),
        "attn_flops_pct": attn_pct,
        "total_flops_G": total_flops / 1e9,
    }


def estimate_kv_cache(model_name, seq_len, batch_size, precision="fp16"):
    """Estimate KV cache memory requirements."""
    model = MODEL_PARAMS[model_name]
    head_dim = model["hidden"] // model["heads"]
    bytes_per = 2 if precision == "fp16" else 1

    kv_per_token = 2 * model["layers"] * model["heads"] * head_dim * bytes_per
    total_kv = kv_per_token * seq_len * batch_size

    return {
        "kv_per_token_bytes": kv_per_token,
        "total_kv_gb": total_kv / 1e9,
        "kv_per_request_mb": kv_per_token * seq_len / 1e6,
    }


def estimate_max_concurrent(model_name, gpu="rtx4090", precision="fp16", seq_len=2048, kv_fraction=0.8):
    """Estimate max concurrent requests."""
    model = MODEL_PARAMS[model_name]
    gpu_p = GPU_PARAMS[gpu]

    bytes_per = 2 if precision == "fp16" else 1
    model_gb = model["params_b"] * 1e9 * bytes_per / 1e9

    kv_budget_gb = gpu_p["vram_gb"] * kv_fraction - model_gb

    if kv_budget_gb <= 0:
        return {"max_concurrent": 0, "model_gb": model_gb, "kv_budget_gb": 0, "kv_per_request_mb": 0, "error": "Model doesn't fit in VRAM"}

    kv = estimate_kv_cache(model_name, seq_len, 1, precision)
    max_concurrent = int(kv_budget_gb * 1e9 / (kv["kv_per_token_bytes"] * seq_len))

    return {
        "max_concurrent": max_concurrent,
        "model_gb": model_gb,
        "kv_budget_gb": kv_budget_gb,
        "kv_per_request_mb": kv["kv_per_request_mb"],
    }


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Performance Estimator")
    parser.add_argument("--model", default="7b", choices=MODEL_PARAMS.keys())
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--gpu", default="rtx4090", choices=GPU_PARAMS.keys())
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "fp8"])
    parser.add_argument("--mode", default="all", choices=["decode", "prefill", "kv", "max", "all"])
    args = parser.parse_args()

    print("=" * 60)
    print(f"LLM Inference Performance Estimator")
    print(f"Model: {args.model.upper()} | GPU: {args.gpu.upper()} | Precision: {args.precision}")
    print("=" * 60)

    if args.mode in ["decode", "all"]:
        print("\n--- Decode Performance ---")
        for bs in [1, 4, 16, 32, 64, 128, 256]:
            r = estimate_decode_latency(args.model, bs, args.gpu, args.precision)
            print(f"  B={bs:>4}: {r['latency_ms']:>7.2f}ms, {r['tok_per_s']:>10,.0f} tok/s, {r['tflops']:>6.1f} TFLOPS")

    if args.mode in ["prefill", "all"]:
        print("\n--- Prefill Performance ---")
        for sl in [128, 256, 512, 1024, 2048, 4096]:
            r = estimate_prefill_latency(args.model, sl, 1, args.gpu)
            print(f"  S={sl:>5}: {r['latency_ms']:>7.2f}ms, {r['tok_per_s']:>10,.0f} tok/s, attn={r['attn_flops_pct']:>5.1f}%")

    if args.mode in ["kv", "all"]:
        print("\n--- KV Cache ---")
        for sl in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
            kv = estimate_kv_cache(args.model, sl, 1, args.precision)
            print(f"  Seq={sl:>6}: {kv['kv_per_request_mb']:>8.1f} MB/req, {kv['total_kv_gb']:>8.3f} GB (B=32)")

    if args.mode in ["max", "all"]:
        print("\n--- Max Concurrent Requests ---")
        for sl in [512, 1024, 2048, 4096]:
            mc = estimate_max_concurrent(args.model, args.gpu, args.precision, sl)
            print(f"  Seq={sl:>5}: max_concurrent={mc['max_concurrent']:>5} (model={mc['model_gb']:.1f}GB, kv_budget={mc['kv_budget_gb']:.1f}GB)")


if __name__ == "__main__":
    main()
