#!/usr/bin/env python3
"""
DeepSeek-V3 Serving Cost Estimator
====================================
Estimates GPU requirements and cost for serving DeepSeek-V3 (671B MoE).

Based on:
- Model: 671B total params, 37B active per token, 256 experts, top-8
- Memory: weights + KV cache + activation overhead
- Throughput: from deepseek_v3_pipeline_sim.py analysis
- Hardware: H800/H100/B200 pricing

Usage:
    python tools/deepseek_v3_cost_estimator.py --tp 8 --ep 8
    python tools/deepseek_v3_cost_estimator.py --model deepseek-v3 --gpu h800
"""

import argparse
import math


# ─── Model Specs ─────────────────────────────────────────────────────────────

MODELS = {
    "deepseek-v3": {
        "name": "DeepSeek-V3",
        "total_params_B": 671,      # Total parameters (billions)
        "active_params_B": 37,      # Active per token (billions)
        "hidden": 7168,
        "num_layers": 61,
        "num_experts": 256,
        "num_topk": 8,
        "intermediate_size": 18432,
        "num_attention_heads": 128,
        "num_kv_heads": 1,          # MLA: 1 KV head (compressed)
        "head_dim_qk": 576,
        "head_dim_v": 512,
        "kv_compress_dim": 512,     # MLA compression
        "vocab_size": 129280,
        "weight_quant": "fp8",      # FP8 quantized weights
        "bytes_per_param": 1,       # FP8 = 1 byte
        "max_seq_len": 131072,
    },
    "deepseek-v2": {
        "name": "DeepSeek-V2",
        "total_params_B": 236,
        "active_params_B": 21,
        "hidden": 5120,
        "num_layers": 60,
        "num_experts": 160,
        "num_topk": 6,
        "intermediate_size": 1536,
        "num_attention_heads": 128,
        "num_kv_heads": 1,          # MLA: 1 KV head
        "head_dim_qk": 256,
        "head_dim_v": 512,
        "kv_compress_dim": 512,
        "vocab_size": 102400,
        "weight_quant": "bf16",
        "bytes_per_param": 2,
        "max_seq_len": 131072,
    },
    "mixtral-8x22b": {
        "name": "Mixtral 8x22B",
        "total_params_B": 141,
        "active_params_B": 39,
        "hidden": 6144,
        "num_layers": 56,
        "num_experts": 8,
        "num_topk": 2,
        "intermediate_size": 16384,
        "num_attention_heads": 48,
        "num_kv_heads": 8,       # GQA: 8 KV heads
        "head_dim_qk": 128,
        "head_dim_v": 128,
        "kv_compress_dim": None,  # No MLA
        "vocab_size": 32768,
        "weight_quant": "bf16",
        "bytes_per_param": 2,
        "max_seq_len": 65536,
    },
}


# ─── GPU Specs ────────────────────────────────────────────────────────────────

GPUS = {
    "h800": {
        "name": "H800 80GB",
        "memory_gb": 80,
        "fp8_tflops": 1979,
        "bf16_tflops": 989,
        "memory_bw_gbs": 3352,
        "nvlink_bw_gbs": 900,
        "price_usd_per_hour": 3.50,  # Cloud pricing estimate
        "tpu_equiv": None,
    },
    "h100": {
        "name": "H100 80GB SXM",
        "memory_gb": 80,
        "fp8_tflops": 1979,
        "bf16_tflops": 989,
        "memory_bw_gbs": 3352,
        "nvlink_bw_gbs": 900,
        "price_usd_per_hour": 3.00,
    },
    "b200": {
        "name": "B200 192GB",
        "memory_gb": 192,
        "fp8_tflops": 4500,
        "bf16_tflops": 2250,
        "memory_bw_gbs": 8000,
        "nvlink_bw_gbs": 1800,
        "price_usd_per_hour": 6.00,
    },
    "a100-80": {
        "name": "A100 80GB",
        "memory_gb": 80,
        "fp8_tflops": 0,  # No FP8
        "bf16_tflops": 312,
        "memory_bw_gbs": 2039,
        "nvlink_bw_gbs": 600,
        "price_usd_per_hour": 1.50,
    },
}


def estimate_memory(model, num_gpus, seq_len, batch_size, kv_quant="bf16"):
    """Estimate memory requirements per GPU."""
    m = MODELS[model]

    # Weight memory (split across GPUs with TP)
    total_weight_bytes = m["total_params_B"] * 1e9 * m["bytes_per_param"]
    weight_per_gpu_gb = total_weight_bytes / num_gpus / 1e9

    # KV cache per request (accounting for TP parallelism)
    if m["kv_compress_dim"]:
        # MLA: 1 KV head, cannot split across TP GPUs
        kv_dim = m["kv_compress_dim"]
    else:
        num_kv = m.get("num_kv_heads", m["num_attention_heads"])
        kv_heads_per_gpu = max(1, num_kv // num_gpus)
        kv_dim = m["head_dim_v"] * kv_heads_per_gpu

    if kv_quant == "fp8":
        kv_bytes_per_token = kv_dim + 16 + kv_dim // 4  # FP8 format approx
    else:
        kv_bytes_per_token = kv_dim * 2  # BF16

    kv_per_request_mb = seq_len * kv_bytes_per_token * m["num_layers"] / 1e6

    # Total KV cache (all requests)
    total_kv_mb = kv_per_request_mb * batch_size

    # Activation memory (rough estimate: 2x hidden per layer per token)
    activation_mb = batch_size * seq_len * m["hidden"] * 2 * 4 / 1e6  # rough

    # Overhead (optimizer states, fragmentation, etc.)
    overhead_gb = 2.0  # Fixed overhead

    total_per_gpu_gb = (
        weight_per_gpu_gb
        + total_kv_mb / 1024
        + activation_mb / 1024
        + overhead_gb
    )

    return {
        "weight_per_gpu_gb": weight_per_gpu_gb,
        "kv_per_request_mb": kv_per_request_mb,
        "total_kv_mb": total_kv_mb,
        "activation_mb": activation_mb,
        "overhead_gb": overhead_gb,
        "total_per_gpu_gb": total_per_gpu_gb,
    }


def estimate_throughput(model, gpu, num_gpus, batch_size, seq_len):
    """Estimate decode throughput based on pipeline analysis."""
    g = GPUS[gpu]
    m = MODELS[model]

    if model == "deepseek-v3":
        # From our pipeline simulator analysis:
        # H800: expert weight loading is bottleneck at ~2523 us per MoE layer
        # Attention: ~190 us per layer (compute-bound)
        # EP: ~17 us per layer
        experts_per_gpu = m["num_experts"] // num_gpus if num_gpus > 1 else m["num_experts"]

        # Expert weight per GPU
        expert_weight_gb = (
            2 * m["hidden"] * m["intermediate_size"] * experts_per_gpu
            * m["bytes_per_param"] / 1e9
        )

        # Expert weight loading time
        expert_load_us = expert_weight_gb * 1e9 / (g["memory_bw_gbs"] * 1e9) * 1e6

        # Expert compute time
        tokens_per_gpu = batch_size * m["num_topk"] / num_gpus
        expert_flops_gflops = 2 * tokens_per_gpu * m["hidden"] * m["intermediate_size"] * 2 / 1e9
        expert_compute_us = expert_flops_gflops / (g["fp8_tflops"] * 1e3) * 1e6

        expert_latency_us = max(expert_load_us, expert_compute_us)

        # MLA attention (compute-bound)
        attn_flops_gflops = (
            2 * seq_len * (m["head_dim_qk"] + m["head_dim_v"])
            * m["num_attention_heads"] * batch_size / 1e9
        )
        # FlashMLA measured: 660 TFLOPS compute-bound
        attn_tflops = 660 if gpu in ("h800", "h100") else 660 * g["fp8_tflops"] / 1979
        attn_latency_us = attn_flops_gflops / (attn_tflops * 1e3) * 1e6

        # EP communication
        tokens_per_link = batch_size * m["num_topk"] / num_gpus
        ep_bytes = tokens_per_link * m["hidden"] * 1  # FP8
        ep_bw = g["nvlink_bw_gbs"] * 0.8  # 80% utilization (DeepEP measured)
        if num_gpus > 1:
            ep_latency_us = ep_bytes / (ep_bw * 1e9) * 1e6 * 2  # dispatch + combine
        else:
            ep_latency_us = 0

        # Per-layer latency
        attn_layer_us = attn_latency_us
        moe_layer_us = expert_latency_us + ep_latency_us + 5  # 5 us router overhead

        # Total: 1 dense + 60 MoE layers
        total_us = attn_layer_us + 60 * (attn_layer_us + moe_layer_us)
        throughput = batch_size / (total_us / 1e6)
    else:
        # Generic roofline model for other MoE models
        active_params = m["active_params_B"] * 1e9
        total_flops = 2 * active_params * batch_size  # batch forward
        compute_time = total_flops / (g["bf16_tflops"] * 1e12)  # seconds

        # Memory time (weight read, once per batch)
        weight_bytes = m["total_params_B"] * 1e9 * m["bytes_per_param"] / num_gpus
        mem_time = weight_bytes / (g["memory_bw_gbs"] * 1e9)

        time_per_batch = max(compute_time, mem_time)
        throughput = batch_size / time_per_batch

        total_us = time_per_batch * 1e6

    return {
        "per_step_ms": total_us / 1000,
        "tokens_per_second": throughput,
        "attn_per_layer_us": attn_latency_us if model == "deepseek-v3" else 0,
        "expert_per_layer_us": expert_latency_us if model == "deepseek-v3" else 0,
        "ep_per_layer_us": ep_latency_us if model == "deepseek-v3" else 0,
    }


def estimate_cost(model, gpu, num_gpus, seq_len, batch_size, target_tps=None):
    """Full cost estimation."""
    g = GPUS[gpu]

    mem = estimate_memory(model, num_gpus, seq_len, batch_size)
    perf = estimate_throughput(model, gpu, num_gpus, batch_size, seq_len)

    # Check if model fits
    fits = mem["total_per_gpu_gb"] <= g["memory_gb"]
    memory_util = mem["total_per_gpu_gb"] / g["memory_gb"] * 100

    # Max batch size (KV cache limited)
    available_for_kv = g["memory_gb"] - mem["weight_per_gpu_gb"] - mem["overhead_gb"]
    max_kv_mb = available_for_kv * 1024
    max_batch = int(max_kv_mb / mem["kv_per_request_mb"]) if mem["kv_per_request_mb"] > 0 else batch_size

    # Cost calculation
    cost_per_hour = num_gpus * g["price_usd_per_hour"]
    cost_per_million_tokens = cost_per_hour / (perf["tokens_per_second"] * 3600) * 1e6

    result = {
        "model": MODELS[model]["name"],
        "gpu": g["name"],
        "num_gpus": num_gpus,
        "config": {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "max_batch": max_batch,
        },
        "memory": mem,
        "performance": perf,
        "cost": {
            "gpus": num_gpus,
            "cost_per_hour_usd": round(cost_per_hour, 2),
            "cost_per_mtok_usd": round(cost_per_million_tokens, 4),
            "tokens_per_dollar": round(perf["tokens_per_second"] * 3600 / cost_per_hour, 0),
        },
        "fits_in_memory": fits,
        "memory_util_pct": round(memory_util, 1),
    }

    # If target throughput specified, calculate needed GPUs
    if target_tps and perf["tokens_per_second"] > 0:
        replicas = math.ceil(target_tps / perf["tokens_per_second"])
        result["scaling"] = {
            "target_tps": target_tps,
            "replicas_needed": replicas,
            "total_gpus": replicas * num_gpus,
            "total_cost_per_hour_usd": round(replicas * cost_per_hour, 2),
        }

    return result


def print_report(result):
    """Print a formatted cost report."""
    print(f"\n{'=' * 65}")
    print(f"  {result['model']} Serving on {result['num_gpus']}x {result['gpu']}")
    print(f"{'=' * 65}")

    c = result["config"]
    print(f"\n  Configuration:")
    print(f"    Sequence Length:    {c['seq_len']:>10,} tokens")
    print(f"    Batch Size:         {c['batch_size']:>10,}")
    print(f"    Max Batch (KV):     {c['max_batch']:>10,}")

    m = result["memory"]
    print(f"\n  Memory (per GPU):")
    print(f"    Weights:            {m['weight_per_gpu_gb']:>10.1f} GB")
    print(f"    KV Cache:           {m['total_kv_mb']:>10.1f} MB")
    print(f"    Activations:        {m['activation_mb']:>10.1f} MB")
    print(f"    Overhead:           {m['overhead_gb']:>10.1f} GB")
    print(f"    ─────────────────────────────────")
    print(f"    Total:              {m['total_per_gpu_gb']:>10.1f} GB")
    print(f"    Utilization:        {result['memory_util_pct']:>10.1f}%")
    print(f"    Fits:               {'Yes' if result['fits_in_memory'] else 'NO - Need more GPUs!'}")

    p = result["performance"]
    print(f"\n  Performance:")
    print(f"    Step Latency:       {p['per_step_ms']:>10.1f} ms")
    print(f"    Throughput:         {p['tokens_per_second']:>10,} tok/s")
    if p["attn_per_layer_us"]:
        print(f"    Attn/Layer:         {p['attn_per_layer_us']:>10.1f} μs")
        print(f"    Expert/Layer:       {p['expert_per_layer_us']:>10.1f} μs")
        print(f"    EP/Layer:           {p['ep_per_layer_us']:>10.1f} μs")

    cost = result["cost"]
    print(f"\n  Cost:")
    print(f"    GPUs:               {cost['gpus']:>10,}")
    print(f"    $/hour:             ${cost['cost_per_hour_usd']:>10.2f}")
    print(f"    $/M tokens:         ${cost['cost_per_mtok_usd']:>10.4f}")
    print(f"    Tokens/$:           {cost['tokens_per_dollar']:>10,.0f}")

    if "scaling" in result:
        s = result["scaling"]
        print(f"\n  Scaling to {s['target_tps']:,} tok/s:")
        print(f"    Replicas:           {s['replicas_needed']:>10,}")
        print(f"    Total GPUs:         {s['total_gpus']:>10,}")
        print(f"    Total $/hour:       ${s['total_cost_per_hour_usd']:>10.2f}")


def compare_gpus(model, seq_len, batch_size):
    """Compare serving cost across different GPUs."""
    print(f"\n{'=' * 85}")
    print(f"  GPU Comparison — {MODELS[model]['name']} (B={batch_size}, S={seq_len})")
    print(f"{'=' * 85}")
    print(f"{'GPU':<15} {'#GPUs':>5} {'Mem%':>6} {'tok/s':>10} {'$/Mtok':>10} {'tok/$':>10} {'Fits':>5}")
    print("-" * 85)

    for gpu_name, g in GPUS.items():
        for num_gpus in [8, 16, 32]:
            try:
                result = estimate_cost(model, gpu_name, num_gpus, seq_len, batch_size)
                fits = "Yes" if result["fits_in_memory"] else "No"
                print(f"{g['name']:<15} {num_gpus:>5} "
                      f"{result['memory_util_pct']:>5.0f}% "
                      f"{result['performance']['tokens_per_second']:>10,.0f} "
                      f"${result['cost']['cost_per_mtok_usd']:>9.4f} "
                      f"{result['cost']['tokens_per_dollar']:>10,.0f} "
                      f"{fits:>5}")
            except Exception:
                pass
            if result["fits_in_memory"]:
                break  # Don't need more GPUs if it already fits


def compare_models(gpu, num_gpus, seq_len, batch_size):
    """Compare serving cost across different models."""
    print(f"\n{'=' * 85}")
    print(f"  Model Comparison — {GPUS[gpu]['name']} × {num_gpus}")
    print(f"{'=' * 85}")
    print(f"{'Model':<18} {'Params':>8} {'Mem/GPU':>8} {'tok/s':>10} {'$/Mtok':>10} {'tok/$':>10}")
    print("-" * 85)

    for model_name, m in MODELS.items():
        try:
            result = estimate_cost(model_name, gpu, num_gpus, seq_len, batch_size)
            if result["fits_in_memory"]:
                print(f"{m['name']:<18} {m['total_params_B']:>7.0f}B "
                      f"{result['memory']['total_per_gpu_gb']:>7.1f}GB "
                      f"{result['performance']['tokens_per_second']:>10,.0f} "
                      f"${result['cost']['cost_per_mtok_usd']:>9.4f} "
                      f"{result['cost']['tokens_per_dollar']:>10,.0f}")
            else:
                print(f"{m['name']:<18} {m['total_params_B']:>7.0f}B "
                      f"{'OOM':>8} {'-':>10} {'-':>10} {'-':>10}")
        except Exception:
            print(f"{m['name']:<18} {m['total_params_B']:>7.0f}B {'Error':>8}")


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V3 Serving Cost Estimator")
    parser.add_argument("--model", default="deepseek-v3", choices=MODELS.keys())
    parser.add_argument("--gpu", default="h800", choices=GPUS.keys())
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-tps", type=int, default=None, help="Target throughput")
    parser.add_argument("--compare-gpus", action="store_true")
    parser.add_argument("--compare-models", action="store_true")
    args = parser.parse_args()

    if args.compare_gpus:
        compare_gpus(args.model, args.seq_len, args.batch_size)
        return

    if args.compare_models:
        compare_models(args.gpu, args.num_gpus, args.seq_len, args.batch_size)
        return

    result = estimate_cost(
        args.model, args.gpu, args.num_gpus,
        args.seq_len, args.batch_size, args.target_tps,
    )
    print_report(result)

    # Also show sequence length scaling
    print(f"\n{'─' * 65}")
    print(f"  Sequence Length Scaling (B={args.batch_size}, {args.num_gpus}× {GPUS[args.gpu]['name']})")
    print(f"{'─' * 65}")
    print(f"{'SeqLen':>8} {'Mem/GPU':>8} {'tok/s':>10} {'$/Mtok':>10} {'MaxBatch':>10}")
    print("-" * 55)
    for sl in [1024, 4096, 8192, 32768, 65536, 131072]:
        try:
            r = estimate_cost(args.model, args.gpu, args.num_gpus, sl, args.batch_size)
            fits = "Yes" if r["fits_in_memory"] else "No"
            print(f"{sl:>8,} {r['memory']['total_per_gpu_gb']:>7.1f}GB "
                  f"{r['performance']['tokens_per_second']:>10,.0f} "
                  f"${r['cost']['cost_per_mtok_usd']:>9.4f} "
                  f"{r['config']['max_batch']:>10,}")
        except Exception:
            print(f"{sl:>8,} {'Error':>8}")


if __name__ == "__main__":
    main()
