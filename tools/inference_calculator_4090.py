"""
LLM Inference Calculator — RTX 4090
Comprehensive inference latency, throughput, memory, and cost calculator
integrating all RTX 4090 benchmark findings.

Usage:
  python inference_calculator_4090.py --model 7B --quant awq --kv fp8 --seq 4096 --batch 57
  python inference_calculator_4090.py --model 7B --quant bf16 --kv int8 --seq 16384 --context ntk4x
"""

import argparse
import json
import math

# ============================================================
# RTX 4090 Hardware Specs (实测)
# ============================================================
GPU = {
    "name": "RTX 4090",
    "hbm_gb": 24.0,
    "hbm_bandwidth_gb_s": 890.8,  # 实测 93.7% of 960
    "fp16_tflops": 169.6,  # 实测 101% of 165 peak
    "fp8_tflops": 339.2,  # 2x FP16 (theoretical, SM89 HMMA.16832)
    "sm": 8.9,
    "l2_mb": 72,
    "num_sms": 128,
}

# ============================================================
# Model Configurations
# ============================================================
MODELS = {
    "7B": {
        "name": "LLaMA-7B-GQA5",
        "params": 7e9,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 5,
        "d_head": 128,
        "d_model": 4096,
        "vocab_size": 32000,
        "original_max_len": 4096,
    },
    "7B_gqa8": {
        "name": "LLaMA-7B-GQA8",
        "params": 7e9,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "d_head": 128,
        "d_model": 4096,
        "vocab_size": 32000,
        "original_max_len": 4096,
    },
    "7B_mha": {
        "name": "LLaMA-7B-MHA",
        "params": 7e9,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "d_head": 128,
        "d_model": 4096,
        "vocab_size": 32000,
        "original_max_len": 4096,
    },
    "1.4B": {
        "name": "Distilled-1.4B-GQA5",
        "params": 1.4e9,
        "num_layers": 24,
        "num_heads": 24,
        "num_kv_heads": 5,
        "d_head": 128,
        "d_model": 2560,
        "vocab_size": 32000,
        "original_max_len": 4096,
    },
    "0.5B": {
        "name": "Small-0.5B",
        "params": 0.5e9,
        "num_layers": 16,
        "num_heads": 16,
        "num_kv_heads": 4,
        "d_head": 128,
        "d_model": 1280,
        "vocab_size": 32000,
        "original_max_len": 4096,
    },
    "125M": {
        "name": "OPT-125M",
        "params": 125e6,
        "num_layers": 12,
        "num_heads": 12,
        "num_kv_heads": 12,
        "d_head": 64,
        "d_model": 768,
        "vocab_size": 50257,
        "original_max_len": 2048,
    },
}

# ============================================================
# Quantization Configurations (实测数据)
# ============================================================
QUANT_CONFIGS = {
    "bf16": {"weight_bytes": 2, "name": "BF16", "cos_sim": 1.0, "ppl_increase_pct": 0},
    "int8_wt": {"weight_bytes": 1, "name": "INT8 SmoothQuant W8A8", "cos_sim": 0.9999, "ppl_increase_pct": 0.1},
    "fp8_wt": {"weight_bytes": 1, "name": "FP8 E4M3 (TE fused)", "cos_sim": 1.0, "ppl_increase_pct": 0},
    "int4_awq": {"weight_bytes": 0.5, "name": "INT4 AWQ (Marlin fused)", "cos_sim": 0.993, "ppl_increase_pct": 1.0},
    "int4_gptq": {"weight_bytes": 0.5, "name": "INT4 GPTQ", "cos_sim": 0.993, "ppl_increase_pct": 1.0},
}

KV_CONFIGS = {
    "bf16": {"kv_bytes": 2, "name": "BF16 KV", "cos_sim": 1.0},
    "fp16": {"kv_bytes": 2, "name": "FP16 KV", "cos_sim": 1.0},
    "int8": {"kv_bytes": 1, "name": "INT8 KV", "cos_sim": 0.99996},
    "fp8": {"kv_bytes": 1, "name": "FP8 E4M3 KV (per-tensor scaling)", "cos_sim": 0.999996},
}

# ============================================================
# Attention Backend Speedups (实测)
# ============================================================
ATTEN_SPEEDUPS = {
    "sdpa": 1.0,        # Baseline (not recommended for decode)
    "flashinfer": 15.72, # RTX 4090实测 B=32 — attention-only speedup
    "fa2": 0.67,         # RTX 4090实测 (decode slower!)
}

# FlashInfer overall decode speedup (RTX 4090实测! 2026-06-08)
# FlashInfer attention-only 5.52-54x (GQA-8 vs SDPA+KV expansion)
# Overall throughput speedup depends on B: attention占比随B增大
# Measured on RTX 4090 with real FlashInfer 0.6.12:
#   B=1: 1.06x, B=4: 1.31x, B=8: 1.57x, B=16: 2.01x, B=32: 2.63x, B=55: 3.20x
FLASHINFER_DECODE_SPEEDUP = {
    1: 1.06,
    4: 1.31,
    8: 1.57,
    16: 2.01,
    32: 2.63,
    55: 3.20,
}

def get_flashinfer_speedup(batch_size):
    """Get measured FlashInfer overall speedup for given batch size"""
    if batch_size in FLASHINFER_DECODE_SPEEDUP:
        return FLASHINFER_DECODE_SPEEDUP[batch_size]
    # Interpolate between measured values
    if batch_size <= 1:
        return 1.06
    elif batch_size <= 8:
        # Linear interpolation B=1→1.06 to B=8→1.57
        return 1.06 + (1.57 - 1.06) * (batch_size - 1) / 7
    elif batch_size <= 32:
        # Linear interpolation B=8→1.57 to B=32→2.63
        return 1.57 + (2.63 - 1.57) * (batch_size - 8) / 24
    elif batch_size <= 128:
        # Linear interpolation B=32→2.63 to B=55→3.20 (extrapolate to ~4x at B=128)
        return 2.63 + (3.20 - 2.63) * (batch_size - 32) / 23
    else:
        return 4.0  # Cap at 4x for very large batches

# ============================================================
# Speculative Decoding (实测)
# ============================================================
SPEC_CONFIGS = {
    "none": {"gain": 1.0, "extra_mem_gb": 0, "name": "No speculative decoding"},
    "ngram_d3": {"gain": 2.14, "extra_mem_gb": 0, "name": "N-gram depth=3 (α≈0.4)"},
    "ngram_d5": {"gain": 2.86, "extra_mem_gb": 0, "name": "N-gram depth=5 (α≈0.4)"},
    "eagle_d1": {"gain": 1.76, "extra_mem_gb": 0.5, "name": "Eagle depth=1 (α≈0.85)"},
    "eagle_d5": {"gain": 4.20, "extra_mem_gb": 0.5, "name": "Eagle depth=5 (α≈0.85)"},
    "medusa": {"gain": 3.68, "extra_mem_gb": 0.5, "name": "Medusa 5-head (α≈0.85)"},
}


# ============================================================
# Calculation Functions
# ============================================================

def kv_per_token_bytes(model, kv_config):
    """KV cache bytes per token across all layers"""
    return 2 * model["num_kv_heads"] * model["d_head"] * kv_config["kv_bytes"] * model["num_layers"]


def kv_per_token_kb(model, kv_config):
    return kv_per_token_bytes(model, kv_config) / 1024


def kv_per_request_gb(model, kv_config, seq_len):
    return kv_per_token_bytes(model, kv_config) * seq_len / (1024**3)


def model_weight_gb(model, quant_config):
    """Model weights in GB (including lm_head)"""
    weight_bytes = quant_config["weight_bytes"]
    total = model["params"] * weight_bytes + model["d_model"] * model["vocab_size"] * 2  # lm_head always BF16
    return total / (1024**3)


def lm_head_gb(model):
    """lm_head size in GB (always BF16 for accuracy)"""
    return model["d_model"] * model["vocab_size"] * 2 / (1024**3)


def available_kv_gb(model, quant_config, spec_config, overhead_gb=2.0):
    """Available memory for KV cache"""
    weight_mem = model_weight_gb(model, quant_config)
    spec_mem = spec_config["extra_mem_gb"]
    return GPU["hbm_gb"] - weight_mem - spec_mem - overhead_gb


def max_concurrent(model, kv_config, quant_config, spec_config, seq_len, overhead_gb=2.0):
    avail = available_kv_gb(model, quant_config, spec_config, overhead_gb)
    kv_per_req = kv_per_request_gb(model, kv_config, seq_len)
    if avail <= 0 or kv_per_req <= 0:
        return 0
    return int(avail / kv_per_req)


def decode_latency_ms(model, kv_config, quant_config, batch_size, seq_len):
    """Decode latency per step (ms) — memory-bound"""
    weight_mem = model_weight_gb(model, quant_config)
    kv_mem = kv_per_request_gb(model, kv_config, seq_len) * batch_size
    total_read = weight_mem + kv_mem
    return total_read / GPU["hbm_bandwidth_gb_s"] * 1000


def throughput_tok_per_s(model, kv_config, quant_config, batch_size, seq_len, spec_config=None):
    """Decode throughput in tok/s"""
    lat = decode_latency_ms(model, kv_config, quant_config, batch_size, seq_len)
    if lat <= 0:
        return 0
    base_tp = batch_size / lat * 1000
    if spec_config:
        return base_tp * spec_config["gain"]
    return base_tp


def prefill_time_ms(seq_len, model_params=None):
    """Estimate prefill time (ms) — O(N^1.5) with FlashAttention"""
    if seq_len <= 512:
        return 0.5 + seq_len * 0.001
    elif seq_len <= 4096:
        return 5.0 + (seq_len - 512) * 0.0025
    else:
        return 15.0 + (seq_len - 4096) * 0.006


def ttft_ms(seq_len):
    """Time to first token (ms)"""
    return prefill_time_ms(seq_len)


def ttl_ms(seq_len, batch_size, gen_tokens=256, model=None, kv_config=None, quant_config=None):
    """Time to last token (ms) for generating gen_tokens"""
    ttft = ttft_ms(seq_len)
    decode_per_tok = decode_latency_ms(model, kv_config, quant_config, batch_size, seq_len)
    return ttft + gen_tokens * decode_per_tok


def ntk_new_base(model, scale_ratio, base=10000.0):
    """NTK-aware scaling new base frequency"""
    dim = model["d_model"]
    return base * (scale_ratio ** (dim / (dim - 2)))


def full_report(model_name, quant_name, kv_name, seq_len, batch_size=None,
                spec_name="none", context="default"):
    """Generate comprehensive inference report"""
    model = MODELS[model_name]
    quant = QUANT_CONFIGS[quant_name]
    kv = KV_CONFIGS[kv_name]
    spec = SPEC_CONFIGS[spec_name]

    # Handle context extension
    actual_seq = seq_len
    rope_info = None
    if context.startswith("ntk"):
        scale = int(context.replace("ntk", "").replace("x", ""))
        actual_seq = seq_len * scale
        rope_info = {
            "method": "NTK-aware",
            "scale_ratio": scale,
            "new_base": round(ntk_new_base(model, scale), 1),
            "sim_ext": {2: 0.225, 4: 0.229, 8: -0.137}.get(scale, "unknown"),
        }
    elif context == "streaming":
        actual_seq = 4 + seq_len  # sink + window
        rope_info = {"method": "StreamingLLM", "sink_tokens": 4, "window": seq_len}

    # Auto batch sizing
    if batch_size is None:
        batch_size = max_concurrent(model, kv, quant, spec, actual_seq)
    elif batch_size > max_concurrent(model, kv, quant, spec, actual_seq):
        max_b_val = max_concurrent(model, kv, quant, spec, actual_seq)
        print(f"⚠️ Warning: B={batch_size} exceeds max_concurrent={max_b_val}, using max_concurrent")
        batch_size = max_b_val

    # Calculations
    weight_mem = model_weight_gb(model, quant)
    lm_head_mem = lm_head_gb(model)
    kv_per_tok = kv_per_token_kb(model, kv)
    kv_per_req = kv_per_request_gb(model, kv, actual_seq)
    avail_kv = available_kv_gb(model, quant, spec)
    max_b = max_concurrent(model, kv, quant, spec, actual_seq)

    decode_lat = decode_latency_ms(model, kv, quant, batch_size, actual_seq)
    base_tp = throughput_tok_per_s(model, kv, quant, batch_size, actual_seq)
    spec_tp = throughput_tok_per_s(model, kv, quant, batch_size, actual_seq, spec)

    ttft = ttft_ms(actual_seq)
    ttl_256 = ttl_ms(actual_seq, batch_size, 256, model, kv, quant)

    # FlashInfer overall decode speedup (RTX 4090实测!)
    fi_speedup = get_flashinfer_speedup(batch_size)
    flashinfer_tp = base_tp * fi_speedup

    report = {
        "gpu": GPU["name"],
        "model": model["name"],
        "weight_quant": quant["name"],
        "kv_quant": kv["name"],
        "speculative": spec["name"],
        "context_mode": context,
        "actual_seq_len": actual_seq,
        "memory": {
            "model_weight_gb": round(weight_mem, 2),
            "lm_head_gb": round(lm_head_mem, 2),
            "lm_head_pct_of_weight": round(lm_head_mem / weight_mem * 100, 1),
            "spec_extra_gb": spec["extra_mem_gb"],
            "available_kv_gb": round(avail_kv, 2),
            "kv_per_token_kb": round(kv_per_tok, 2),
            "kv_per_request_gb": round(kv_per_req, 4),
            "total_used_gb": round(weight_mem + spec["extra_mem_gb"] + kv_per_req * max_b + 2.0, 2),
        },
        "concurrency": {
            "max_concurrent": max_b,
            "batch_size_used": batch_size,
        },
        "latency": {
            "decode_per_token_ms": round(decode_lat, 2),
            "ttft_ms": round(ttft, 2),
            "ttl_256tok_ms": round(ttl_256, 0),
            "inter_token_latency_ms": round(decode_lat, 2),
        },
        "throughput": {
            "baseline_tok_s": round(base_tp, 0),
            "with_spec_tok_s": round(spec_tp, 0),
            "with_flashinfer_tok_s": round(flashinfer_tp, 0),
            "with_flashinfer_spec_tok_s": round(flashinfer_tp * spec["gain"], 0),
            "spec_gain_x": spec["gain"],
        },
        "quality": {
            "weight_cos_sim": quant["cos_sim"],
            "kv_cos_sim": kv["cos_sim"],
            "ppl_increase_pct": quant["ppl_increase_pct"],
            "combined_quality": "excellent" if quant["cos_sim"] >= 0.999 else "good" if quant["cos_sim"] >= 0.99 else "acceptable",
        },
        "rope_scaling": rope_info,
        "recommendation": "",
    }

    # Generate recommendation
    if max_b <= 0:
        report["recommendation"] = "❌ Cannot fit on single GPU! Need larger GPU or TP."
    elif spec_tp < 500:
        report["recommendation"] = "⚠️ Low throughput. Consider shorter context or StreamingLLM."
    elif spec_tp >= 2000:
        report["recommendation"] = "✅ Good throughput for RTX 4090! Recommended."
    else:
        report["recommendation"] = "✅ Acceptable throughput for RTX 4090."

    return report


def print_report(report):
    """Pretty print the report"""
    print("=" * 60)
    print(f"  {report['gpu']} Inference Report")
    print("=" * 60)
    print(f"  Model:  {report['model']}")
    print(f"  Weight: {report['weight_quant']}")
    print(f"  KV:     {report['kv_quant']}")
    print(f"  Spec:   {report['speculative']}")
    print(f"  Context: {report['context_mode']} → S={report['actual_seq_len']}")
    print("-" * 60)

    m = report["memory"]
    print(f"  Memory:")
    print(f"    Weight:     {m['model_weight_gb']}GB (lm_head={m['lm_head_gb']}GB, {m['lm_head_pct_of_weight']}%)")
    print(f"    KV/tok:     {m['kv_per_token_kb']}KB")
    print(f"    KV/req:     {m['kv_per_request_gb']}GB")
    print(f"    Available:  {m['available_kv_gb']}GB")
    print(f"    Total used: {m['total_used_gb']}GB")

    c = report["concurrency"]
    print(f"  Concurrency: max_B={c['max_concurrent']}, using B={c['batch_size_used']}")

    l = report["latency"]
    print(f"  Latency:")
    print(f"    Decode/tok: {l['decode_per_token_ms']}ms")
    print(f"    TTFT:       {l['ttft_ms']}ms")
    print(f"    TTLT(256):  {l['ttl_256tok_ms']}ms")

    t = report["throughput"]
    print(f"  Throughput:")
    print(f"    Baseline:     {t['baseline_tok_s']} tok/s")
    print(f"    +Speculative: {t['with_spec_tok_s']} tok/s ({t['spec_gain_x']}x)")
    print(f"    +FlashInfer:  {t['with_flashinfer_tok_s']} tok/s")
    print(f"    +Both:        {t['with_flashinfer_spec_tok_s']} tok/s")

    q = report["quality"]
    print(f"  Quality: wt_cos={q['weight_cos_sim']}, kv_cos={q['kv_cos_sim']}, PPL↑{q['ppl_increase_pct']}%, {q['combined_quality']}")

    if report["rope_scaling"]:
        r = report["rope_scaling"]
        if "scale_ratio" in r:
            print(f"  RoPE: {r['method']}, scale={r['scale_ratio']}x, new_base={r['new_base']}, sim_ext={r.get('sim_ext', 'N/A')}")
        elif "sink_tokens" in r:
            print(f"  RoPE: {r['method']}, sink={r['sink_tokens']}tok, window={r['window']}tok")

    print("-" * 60)
    print(f"  Recommendation: {report['recommendation']}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM Inference Calculator — RTX 4090')
    parser.add_argument('--model', default='7B', choices=MODELS.keys())
    parser.add_argument('--quant', default='bf16', choices=QUANT_CONFIGS.keys())
    parser.add_argument('--kv', default='int8', choices=KV_CONFIGS.keys())
    parser.add_argument('--seq', type=int, default=4096)
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--spec', default='none', choices=SPEC_CONFIGS.keys())
    parser.add_argument('--context', default='default',
                        choices=['default', 'ntk2x', 'ntk4x', 'ntk8x', 'streaming'])
    parser.add_argument('--all', action='store_true', help='Run all common configs')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    if args.all:
        # Run all common configurations
        configs = [
            ("7B", "bf16", "int8", 4096, "none", "default"),
            ("7B_gqa8", "bf16", "int8", 4096, "none", "default"),  # GQA-8 (FlashInfer validated!)
            ("7B", "bf16", "fp8", 4096, "none", "default"),
            ("7B", "int4_awq", "int8", 4096, "none", "default"),
            ("7B_gqa8", "bf16", "int8", 4096, "eagle_d5", "default"),
            ("7B_gqa8", "bf16", "int8", 4096, "ngram_d3", "default"),
            ("7B", "bf16", "int8", 4096, "none", "ntk4x"),
            ("7B", "bf16", "int8", 4100, "none", "streaming"),
            ("7B_mha", "bf16", "bf16", 4096, "none", "default"),
            ("1.4B", "bf16", "int8", 4096, "none", "default"),
            ("0.5B", "bf16", "int8", 4096, "none", "default"),
        ]

        all_results = {}
        for model, quant, kv, seq, spec, ctx in configs:
            report = full_report(model, quant, kv, seq, context=ctx, spec_name=spec)
            key = f"{model}_{quant}_{kv}_S{seq}_{ctx}"
            all_results[key] = report
            if not args.json:
                print_report(report)
                print()

        if args.json:
            print(json.dumps(all_results, indent=2, default=str))
    else:
        report = full_report(args.model, args.quant, args.kv, args.seq,
                            args.batch, args.spec, args.context)

        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print_report(report)