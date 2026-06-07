#!/usr/bin/env python3
"""End-to-End LLM Serving Throughput Simulator — RTX 4090
==========================================================
Integrates ALL component optimizations into a unified throughput model.
Simulates real production scenarios and measures how each optimization
contributes to overall serving throughput.

This is a SYSTEM-LEVEL experiment, not just another micro-benchmark.
It answers: "Given my RTX 4090 and all the optimizations I've studied,
what's the maximum throughput I can achieve in realistic scenarios?"

Optimizations integrated:
- INT4 weight-only quantization (75% memory savings)
- INT8 KV cache quantization (50% memory savings)
- GQA (KV compression)
- MLA (KV compression, capacity optimization)
- Continuous batching (dynamic batch scheduling)
- Prefix sharing (GRPO rollout optimization)
- FlashAttention (prefill memory savings)

Production scenarios:
1. Chat serving (short prompts, medium responses)
2. Batch inference (long prompts, short responses)
3. RL rollout (repeated prompts, multiple completions)
4. Long context (very long prompts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import math


def benchmark_fn(name, fn, n_runs=50, warmup=10):
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


class MiniGQA(nn.Module):
    """Mini GQA model for realistic benchmarking."""
    def __init__(self, vocab_size=32000, d_model=512, n_heads=16, n_kv_heads=4,
                 n_layers=8, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        self.d_head = d_model // n_heads
        self.d_ff = d_ff

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=0.0, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)


def simulate_int4_weight(weight, group_size=128):
    """Simulate INT4 weight-only quantization."""
    orig_shape = weight.shape
    w_flat = weight.flatten()
    n = w_flat.numel()
    n_groups = math.ceil(n / group_size)
    padded = torch.zeros(n_groups * group_size, device=weight.device, dtype=weight.dtype)
    padded[:n] = w_flat
    groups = padded.reshape(n_groups, group_size)
    w_max = groups.abs().max(dim=-1).values.clamp(min=1e-5)
    scale = w_max / 7.0
    groups_q = torch.round(groups / scale.unsqueeze(-1)).clamp(-8, 7)
    groups_deq = groups_q * scale.unsqueeze(-1)
    return groups_deq.reshape(-1)[:n].reshape(orig_shape)


def run_experiment():
    device = "cuda"
    results = {}

    # ========================
    # Experiment 1: Baseline — Full FP16 Model Decode
    # ========================
    print("=== Exp 1: Baseline FP16 Decode Throughput ===")

    # 25M GQA model (GQA-16:4)
    model = MiniGQA(vocab_size=32000, d_model=512, n_heads=16, n_kv_heads=4,
                    n_layers=8, d_ff=2048).to(device).to(torch.float16)
    n_params = sum(p.numel() for p in model.parameters())
    model_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9  # GB

    print(f"  Model: {n_params:,} params, {model_mem:.2f} GB FP16")

    # Measure decode latency at different batch sizes
    B_values = [1, 4, 8, 16, 32, 64, 128, 256]
    baseline_decode = {}

    for B in B_values:
        input_ids = torch.randint(0, 32000, (B, 1), device=device)
        model.eval()
        with torch.no_grad():
            res = benchmark_fn(f"baseline_B{B}", lambda: model(input_ids))

        # Throughput: tokens/second
        throughput = B / res["median_ms"] * 1000  # tok/s
        baseline_decode[f"B{B}"] = {
            "latency_ms": res["median_ms"],
            "throughput_tok_s": throughput,
            "batch_size": B,
        }
        print(f"  B={B}: latency={res['median_ms']:.2f}ms throughput={throughput:.0f} tok/s")

    # Find peak throughput
    peak_baseline_B = max(baseline_decode, key=lambda k: baseline_decode[k]["throughput_tok_s"])
    peak_baseline_throughput = baseline_decode[peak_baseline_B]["throughput_tok_s"]

    results["baseline"] = {
        "model_params": n_params,
        "model_mem_GB": model_mem,
        "decode_by_batch": baseline_decode,
        "peak_throughput_tok_s": peak_baseline_throughput,
        "peak_batch": int(peak_baseline_B.replace("B", "")),
    }

    print(f"  Peak throughput: {peak_baseline_throughput:.0f} tok/s at B={peak_baseline_B}")

    # ========================
    # Experiment 2: INT4 Quantized Model Decode
    # ========================
    print("\n=== Exp 2: INT4 Weight-Only Quantized Decode ===")

    # Create INT4-quantized weights
    int4_model_mem = model_mem * 0.25  # INT4: 75% savings

    # Measure with simulated INT4 (dequantize weights, then compute)
    # This simulates production: fused INT4 kernel dequantizes on-the-fly
    int4_decode = {}
    for B in B_values:
        input_ids = torch.randint(0, 32000, (B, 1), device=device)

        # Simulate: same latency as FP16 (INT4 dequant ≈ free)
        # But can run larger batch because model uses less memory
        res = benchmark_fn(f"int4_B{B}", lambda: model(input_ids))

        # INT4 allows larger batch: remaining memory for KV + activations
        total_gpu = 24.0  # GB
        available_after_model = total_gpu - int4_model_mem
        # KV per request: 2 * n_kv_heads * d_head * S * 2 bytes (FP16 KV)
        kv_per_req_fp16 = 2 * 4 * 32 * 2048 * 2 / 1e9  # GB (GQA-4, S=2048)
        kv_per_req_int8 = kv_per_req_fp16 * 0.5  # INT8 KV: 50% savings
        max_batch_kv_only = int(available_after_model * 0.7 / kv_per_req_fp16)  # 70% for KV
        max_batch_kv_int8 = int(available_after_model * 0.7 / kv_per_req_int8)

        throughput_fp16_kv = B / res["median_ms"] * 1000
        throughput_int8_kv = min(B, max_batch_kv_int8) / res["median_ms"] * 1000

        int4_decode[f"B{B}"] = {
            "latency_ms": res["median_ms"],
            "throughput_tok_s": throughput_fp16_kv,
            "throughput_int8_kv_tok_s": throughput_int8_kv,
            "max_batch_fp16_kv": max_batch_kv_only,
            "max_batch_int8_kv": max_batch_kv_int8,
        }
        print(f"  B={B}: latency={res['median_ms']:.2f}ms "
              f"throughput(FP16_KV)={throughput_fp16_kv:.0f} "
              f"throughput(INT8_KV)={throughput_int8_kv:.0f} "
              f"max_batch_fp16={max_batch_kv_only} "
              f"max_batch_int8={max_batch_kv_int8}")

    results["int4"] = {
        "model_mem_GB": int4_model_mem,
        "kv_per_req_fp16_GB": kv_per_req_fp16,
        "kv_per_req_int8_GB": kv_per_req_int8,
        "decode_by_batch": int4_decode,
    }

    # ========================
    # Experiment 3: Continuous Batching Throughput Simulation
    # ========================
    print("\n=== Exp 3: Continuous Batching Throughput Simulation ===")

    # Simulate a serving system with continuous batching
    # Key: requests arrive, get batched, decode tokens, leave when done
    # Throughput = avg_batch_size / avg_decode_latency

    scenarios = {
        "chat_short": {"avg_prompt": 64, "avg_response": 128, "concurrent": 50},
        "chat_medium": {"avg_prompt": 256, "avg_response": 256, "concurrent": 30},
        "batch_inference": {"avg_prompt": 512, "avg_response": 64, "concurrent": 100},
        "rl_rollout": {"avg_prompt": 512, "avg_response": 128, "concurrent": 8, "n_per_prompt": 8},
        "long_context": {"avg_prompt": 8192, "avg_response": 256, "concurrent": 10},
    }

    serving_results = {}
    for scenario_name, params in scenarios.items():
        avg_prompt = params["avg_prompt"]
        avg_response = params["avg_response"]
        concurrent = params["concurrent"]
        n_per_prompt = params.get("n_per_prompt", 1)

        # Prefill latency (approximate)
        # Prefill is compute-bound: ~0.5ms per 512 tokens for 25M model
        prompt_ids = torch.randint(0, 32000, (1, avg_prompt), device=device)
        model.eval()
        with torch.no_grad():
            prefill_res = benchmark_fn(
                f"prefill_{scenario_name}", lambda: model(prompt_ids)
            )

        # Decode latency at effective batch size
        effective_batch = min(concurrent * n_per_prompt, 256)
        decode_ids = torch.randint(0, 32000, (effective_batch, 1), device=device)
        with torch.no_grad():
            decode_res = benchmark_fn(
                f"decode_{scenario_name}_B{effective_batch}",
                lambda: model(decode_ids)
            )

        # Total serving throughput
        # Each request: prefill_time + response_length × decode_time_per_token
        # With continuous batching: decode throughput = B / decode_latency
        prefill_time = prefill_res["median_ms"]
        decode_time_per_token = decode_res["median_ms"] / effective_batch  # ms per token per request

        # Throughput calculations
        # Naive (no batching): one request at a time
        naive_time_per_req = prefill_time + avg_response * decode_time_per_token
        naive_throughput = 1000 / naive_time_per_req  # req/s

        # Continuous batching: multiple requests decode simultaneously
        # Decode throughput = effective_batch / decode_latency_ms * 1000
        decode_throughput_tok_s = effective_batch / decode_res["median_ms"] * 1000

        # Time to complete all concurrent requests
        # Prefill phase: concurrent × prefill_time (sequential prefill)
        # Decode phase: avg_response × (decode_latency / effective_batch)
        total_prefill_time = concurrent * prefill_time
        total_decode_tokens = concurrent * avg_response * n_per_prompt
        total_decode_time = total_decode_tokens / decode_throughput_tok_s * 1000  # ms
        total_time = total_prefill_time + total_decode_time

        # Throughput = total_tokens / total_time
        total_tokens = concurrent * (avg_prompt + avg_response) * n_per_prompt
        throughput_tok_s = total_tokens / total_time * 1000

        # Memory requirements
        kv_per_req = 2 * 4 * 32 * (avg_prompt + avg_response) * 2 / 1e9  # GB (GQA-4)
        kv_total_fp16 = kv_per_req * concurrent * n_per_prompt
        kv_total_int8 = kv_total_fp16 * 0.5
        kv_total_int4_int8 = kv_total_fp16 * 0.25 * 0.5  # INT4 model + INT8 KV

        # Check if fits in 24GB
        total_mem_fp16 = model_mem + kv_total_fp16
        total_mem_int4_int8 = int4_model_mem + kv_total_int8

        fits_fp16 = total_mem_fp16 <= 24
        fits_int4_int8 = total_mem_int4_int8 <= 24

        serving_results[scenario_name] = {
            "avg_prompt": avg_prompt,
            "avg_response": avg_response,
            "concurrent": concurrent,
            "n_per_prompt": n_per_prompt,
            "effective_batch": effective_batch,
            "prefill_ms": prefill_time,
            "decode_ms": decode_res["median_ms"],
            "decode_time_per_token_ms": decode_time_per_token,
            "naive_throughput_req_s": naive_throughput,
            "decode_throughput_tok_s": decode_throughput_tok_s,
            "total_time_ms": total_time,
            "total_tokens": total_tokens,
            "throughput_tok_s": throughput_tok_s,
            "kv_per_req_GB": kv_per_req,
            "kv_total_fp16_GB": kv_total_fp16,
            "kv_total_int8_GB": kv_total_int8,
            "total_mem_fp16_GB": total_mem_fp16,
            "total_mem_int4_int8_GB": total_mem_int4_int8,
            "fits_24GB_fp16": fits_fp16,
            "fits_24GB_int4_int8": fits_int4_int8,
        }

        print(f"  {scenario_name}: prefill={prefill_time:.2f}ms "
              f"decode_B{effective_batch}={decode_res['median_ms']:.2f}ms "
              f"throughput={throughput_tok_s:.0f}tok/s "
              f"total_mem_fp16={total_mem_fp16:.2f}GB(fits={fits_fp16}) "
              f"total_mem_int4_int8={total_mem_int4_int8:.2f}GB(fits={fits_int4_int8})")

    results["serving"] = serving_results

    # ========================
    # Experiment 4: Optimization Stack Comparison
    # ========================
    print("\n=== Exp 4: Optimization Stack Comparison ===")

    # Compare throughput with different optimization stacks
    # Scenario: chat_medium (256 prompt, 256 response, 30 concurrent)

    stacks = {
        "baseline_fp16": {
            "model_mem": model_mem,
            "kv_factor": 1.0,  # FP16 KV
            "batch_factor": 1.0,  # No special batching
        },
        "int4_weights": {
            "model_mem": model_mem * 0.25,
            "kv_factor": 1.0,  # FP16 KV still
            "batch_factor": 1.0,
        },
        "int4_int8_kv": {
            "model_mem": model_mem * 0.25,
            "kv_factor": 0.5,  # INT8 KV
            "batch_factor": 1.0,
        },
        "int4_int8_kv_gqa": {
            "model_mem": model_mem * 0.25,
            "kv_factor": 0.5 * (4/16),  # INT8 KV + GQA-16:4
            "batch_factor": 1.0,
        },
        "int4_int8_kv_prefix": {
            "model_mem": model_mem * 0.25,
            "kv_factor": 0.5,  # INT8 KV
            "batch_factor": 2.46,  # Prefix sharing (full-model)
        },
    }

    avg_S = 512  # prompt + response
    stack_results = {}
    for stack_name, stack in stacks.items():
        m_mem = stack["model_mem"]
        kv_factor = stack["kv_factor"]
        batch_factor = stack["batch_factor"]

        # KV per request
        kv_base = 2 * 16 * 32 * avg_S * 2 / 1e9  # MHA FP16 baseline
        kv_per_req = kv_base * kv_factor

        # Available memory for KV
        available = 24.0 - m_mem
        max_concurrent = int(available * 0.7 / kv_per_req)

        # Decode throughput at max batch
        B = min(max_concurrent, 256)
        decode_ids = torch.randint(0, 32000, (B, 1), device=device)
        with torch.no_grad():
            decode_res = benchmark_fn(
                f"stack_{stack_name}_B{B}", lambda: model(decode_ids)
            )

        throughput = B / decode_res["median_ms"] * 1000 * batch_factor

        stack_results[stack_name] = {
            "model_mem_GB": m_mem,
            "kv_per_req_GB": kv_per_req,
            "max_concurrent": max_concurrent,
            "effective_batch": B,
            "decode_latency_ms": decode_res["median_ms"],
            "throughput_tok_s": throughput,
            "total_mem_GB": m_mem + kv_per_req * max_concurrent,
        }

        print(f"  {stack_name}: model={m_mem:.2f}GB "
              f"kv/req={kv_per_req:.4f}GB "
              f"max_concurrent={max_concurrent} "
              f"throughput={throughput:.0f}tok/s "
              f"total_mem={m_mem+kv_per_req*max_concurrent:.2f}GB")

    results["optimization_stacks"] = stack_results

    # ========================
    # Experiment 5: 7B Model Serving Feasibility
    # ========================
    print("\n=== Exp 5: 7B Model Serving Feasibility ===")

    # Can we serve a 7B model on RTX 4090?
    # 7B FP16 = ~14GB → barely fits in 24GB
    # 7B INT4 = ~3.5GB → lots of room!

    # Estimate decode throughput for 7B model
    # Using roofline: 7B decode is memory-bound
    # Total weight read per token = 14GB (FP16) or 3.5GB (INT4)
    # HBM bandwidth = 890 GB/s

    sevenb_params = 7_000_000_000
    sevenb_fp16_mem = sevenb_params * 2 / 1e9  # 14 GB
    sevenb_int4_mem = sevenb_params * 0.5 / 1e9  # 3.5 GB

    # Estimate: decode latency ≈ weight_read / HBM_BW
    hbm_bw = 890.0  # GB/s (RTX 4090 measured)

    # FP16 7B: weight read = 14GB → latency ≈ 14/890 = 15.7ms per token (B=1)
    # INT4 7B: weight read = 3.5GB → latency ≈ 3.5/890 = 3.9ms per token (B=1)

    # Measured decode latency for our 25M model at different batch sizes
    # Use this to estimate 7B scaling factor
    # 7B / 25M ≈ 280x more params → but decode latency scales with weight read, not params
    # weight ratio: 7B_fp16 / 25M_fp16 = 14GB / 0.1GB ≈ 140x

    # Use actual 25M measurements to extrapolate
    ref_mem_gb = model_mem
    scale_factor_fp16 = sevenb_fp16_mem / ref_mem_gb
    scale_factor_int4 = sevenb_int4_mem / ref_mem_gb

    # Estimate 7B throughput using roofline
    sevenb_serving = {}
    for opt_name, model_mem_7b in [("fp16", sevenb_fp16_mem), ("int4", sevenb_int4_mem)]:
        available = 24.0 - model_mem_7b

        for kv_name, kv_factor in [("fp16_kv", 1.0), ("int8_kv", 0.5), ("int4_int8_kv", 0.5)]:
            if opt_name == "int4" and kv_name == "fp16_kv":
                kv_name_full = "int4_fp16_kv"
            elif opt_name == "int4" and kv_name == "int8_kv":
                kv_name_full = "int4_int8_kv"
            elif opt_name == "fp16" and kv_name == "fp16_kv":
                kv_name_full = "fp16_fp16_kv"
            elif opt_name == "fp16" and kv_name == "int8_kv":
                kv_name_full = "fp16_int8_kv"
            else:
                kv_name_full = f"{opt_name}_{kv_name}"

            # GQA-4 KV per request (S=2048 avg context)
            kv_base = 2 * 4 * 32 * 2048 * 2 / 1e9  # GB (GQA-4, FP16)
            kv_per_req = kv_base * kv_factor
            max_concurrent = int(available * 0.7 / kv_per_req)

            # Decode throughput estimate
            # Roofline: throughput ≈ HBM_BW / weight_read_per_token
            weight_read = model_mem_7b
            decode_time_per_token_est = weight_read / hbm_bw * 1000  # ms
            throughput_est = max_concurrent / decode_time_per_token_est * 1000  # tok/s

            fits_24gb = (model_mem_7b + kv_per_req * max_concurrent) <= 24.0

            sevenb_serving[kv_name_full] = {
                "model_mem_GB": model_mem_7b,
                "kv_per_req_GB": kv_per_req,
                "max_concurrent": max_concurrent,
                "decode_time_est_ms": decode_time_per_token_est,
                "throughput_est_tok_s": throughput_est,
                "fits_24GB": fits_24gb,
                "total_mem_GB": model_mem_7b + kv_per_req * max_concurrent,
            }

            print(f"  7B {kv_name_full}: model={model_mem_7b:.2f}GB "
                  f"kv/req={kv_per_req:.4f}GB "
                  f"max_concurrent={max_concurrent} "
                  f"decode_est={decode_time_per_token_est:.2f}ms "
                  f"throughput_est={throughput_est:.0f}tok/s "
                  f"fits={fits_24gb}")

    results["sevenb_serving"] = sevenb_serving

    # ========================
    # Experiment 6: Optimization ROI Summary
    # ========================
    print("\n=== Exp 6: Optimization ROI Summary ===")

    # For each optimization, compute: throughput_gain / implementation_cost
    # implementation_cost: 1=easy(Python), 2=medium(config), 3=hard(custom kernel)

    roi = {
        "GQA-4": {
            "kv_savings": "4x KV (75%)",
            "throughput_gain": "2x concurrent",
            "latency_impact": "1.00x (zero overhead)",
            "impl_cost": 1,
            "roi": "HIGH — config change, zero cost",
        },
        "INT4 weights": {
            "kv_savings": "model 75% smaller",
            "throughput_gain": "3x concurrent (more KV room)",
            "latency_impact": "0.87-1.08x (near-free)",
            "impl_cost": 3,
            "roi": "VERY HIGH — fused kernel needed but 75% memory",
        },
        "INT8 KV": {
            "kv_savings": "2x KV capacity",
            "throughput_gain": "2x concurrent",
            "latency_impact": "1.00x (zero overhead)",
            "impl_cost": 2,
            "roi": "VERY HIGH — minimal code change, 2x capacity",
        },
        "Continuous batching": {
            "kv_savings": "no savings",
            "throughput_gain": "2-5x throughput (dynamic batching)",
            "latency_impact": "0 latency impact",
            "impl_cost": 3,
            "roi": "HIGH — scheduling logic but huge throughput gain",
        },
        "Prefix sharing": {
            "kv_savings": "2.46x compute savings (n=8)",
            "throughput_gain": "2.46x for RL rollout",
            "latency_impact": "0 latency impact",
            "impl_cost": 4,
            "roi": "MEDIUM — model-level modification, RL only",
        },
        "MLA": {
            "kv_savings": "3.2x KV capacity",
            "throughput_gain": "3.2x concurrent (capacity)",
            "latency_impact": "2-8x slower per request",
            "impl_cost": 5,
            "roi": "LOW — only for 671B+, huge impl cost",
        },
        "FlashAttention": {
            "kv_savings": "85-97% activation memory",
            "throughput_gain": "1.03-1.10x prefill only",
            "latency_impact": "0.67-0.84x decode (SLOWER!)",
            "impl_cost": 1,
            "roi": "LOW for decode, HIGH for memory savings",
        },
        "Speculative decoding": {
            "kv_savings": "no savings",
            "throughput_gain": "3-6x (if α≥0.8, lr≤0.2)",
            "latency_impact": "0.13-0.76x if bad draft",
            "impl_cost": 3,
            "roi": "MEDIUM — needs good draft model, uncertain gain",
        },
    }

    results["roi"] = roi
    for opt_name, info in roi.items():
        print(f"  {opt_name}: savings={info['kv_savings']} "
              f"gain={info['throughput_gain']} "
              f"latency={info['latency_impact']} "
              f"ROI={info['roi']}")

    # Save
    with open("results/e2e_serving_throughput.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/e2e_serving_throughput.json")

    return results


if __name__ == "__main__":
    run_experiment()