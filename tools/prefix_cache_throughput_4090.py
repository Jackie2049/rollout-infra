#!/usr/bin/env python3
"""Prefix Cache Throughput Benchmark on RTX 4090
===============================================

Benchmarks prefix cache impact on LLM inference throughput:
1. Prefix cache hit rate vs throughput — simulate multi-turn patterns
2. Block alignment overhead — vLLM block_size=16 vs token-level sharing
3. Recompute vs Cache — what's the cost of recomputing prefix vs caching?
4. Multi-turn conversation patterns — 1/2/3/5 turn cumulative savings
5. GRPO n_samples prefix sharing — n=2/4/8/16 compute savings

This connects vLLM/SGLang prefix caching theory to real GPU measurements.

Usage:
  python prefix_cache_throughput_4090.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import json
import math

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

# ================================================================
# Experiment 1: Prefix Cache Hit Rate vs Throughput
# ================================================================
def exp1_prefix_cache_hit_rate():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 1: Prefix Cache Hit Rate vs Throughput")
    print("=" * 60)

    # Simulate: each request has prefix_len tokens + suffix_len new tokens
    # prefix_len = shared system prompt / conversation history
    # suffix_len = new query/response tokens
    # Throughput benefit = prefix_len / (prefix_len + suffix_len) * cache_hit_rate

    # Simulate with actual GEMM operations (matmul = attention computation)
    # Prefix computation: prefix_len tokens → KV computed (expensive)
    # Suffix computation: suffix_len tokens → attend to prefix+suffix KV
    # With cache: skip prefix KV computation → only compute suffix

    H = 4096   # hidden_size (7B model)
    N = 4096   # output dim (attention projection)

    # Test different prefix ratios
    prefix_ratios = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]  # prefix/(prefix+suffix)
    B_sizes = [1, 8, 32, 128]
    total_seq = 512  # fixed total sequence length

    results = []

    for B in B_sizes:
        for prefix_ratio in prefix_ratios:
            prefix_len = int(total_seq * prefix_ratio)
            suffix_len = total_seq - prefix_len

            if suffix_len == 0:
                continue  # no new tokens to process

            # Method 1: No cache — compute full sequence from scratch
            # Forward: all total_seq tokens → compute KV + attention
            x_full = torch.randn(B, total_seq, H, device=device, dtype=torch.float16)
            w_proj = torch.randn(H, N, device=device, dtype=torch.float16)

            def no_cache_fn():
                # Full sequence projection (simulates attention computation)
                return x_full @ w_proj

            t_no_cache = benchmark_cuda(no_cache_fn, warmup=5, repeat=50)

            # Method 2: With cache — only compute suffix tokens
            # Prefix KV already cached → only need suffix forward pass
            x_suffix = torch.randn(B, suffix_len, H, device=device, dtype=torch.float16)

            def cached_fn():
                # Only suffix tokens → shorter sequence → less compute
                return x_suffix @ w_proj

            t_cached = benchmark_cuda(cached_fn, warmup=5, repeat=50)

            # Savings calculation
            compute_savings_pct = (1 - suffix_len / total_seq) * 100
            time_savings_pct = (1 - t_cached / t_no_cache) * 100

            # Theoretical time savings (memory-bound):
            # time ∝ total_bytes → savings ≈ compute_savings_pct
            # But actual savings depends on batch size and memory behavior

            # Throughput: tokens/second of NEW content (not cached)
            # No cache: B * total_seq / t_no_cache * 1000
            # With cache: B * suffix_len / t_cached * 1000
            throughput_no_cache = B * total_seq / (t_no_cache * 1e-3)
            throughput_cached_new = B * suffix_len / (t_cached * 1e-3)
            throughput_cached_total = B * total_seq / (t_cached * 1e-3)  # including cached prefix

            print(f"  B={B}, prefix={prefix_ratio:.0%}: no_cache={t_no_cache:.4f}ms, "
                  f"cached={t_cached:.4f}ms, time_savings={time_savings_pct:.1f}%, "
                  f"compute_savings={compute_savings_pct:.1f}%")

            results.append({
                "B": B,
                "prefix_ratio": prefix_ratio,
                "prefix_len": prefix_len,
                "suffix_len": suffix_len,
                "total_seq": total_seq,
                "no_cache_ms": round(t_no_cache, 4),
                "cached_ms": round(t_cached, 4),
                "compute_savings_pct": round(compute_savings_pct, 1),
                "time_savings_pct": round(time_savings_pct, 1),
                "throughput_no_cache": round(throughput_no_cache, 0),
                "throughput_cached_new": round(throughput_cached_new, 0),
                "throughput_cached_total": round(throughput_cached_total, 0),
            })

    return results

# ================================================================
# Experiment 2: Block Alignment Overhead (vLLAM-style)
# ================================================================
def exp2_block_alignment():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 2: Block Alignment Overhead (vLLM-style)")
    print("=" * 60)

    # vLLM uses block_size=16 → prefix must align to block boundaries
    # If two requests share 127 tokens → vLLM shares only 112 (7 blocks × 16)
    # The remaining 15 tokens (127 - 112) are recomputed → wasted!

    H = 4096
    N = 4096
    B = 32

    block_size = 16  # vLLM default

    # Test: shared prefix length vs what vLLM can actually cache
    # Scenario: 2 requests with shared prefix of different lengths
    prefix_lengths = [15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024]

    results = []

    for prefix_len in prefix_lengths:
        # What vLLM caches: floor(prefix_len / block_size) * block_size
        cached_len = (prefix_len // block_size) * block_size
        wasted_len = prefix_len - cached_len
        cache_efficiency = cached_len / prefix_len if prefix_len > 0 else 0
        waste_pct = wasted_len / prefix_len * 100 if prefix_len > 0 else 0

        # With block alignment: we need to recompute the wasted part
        # Without block alignment (token-level): we cache everything
        # Difference = recomputing wasted_len tokens per request

        # Simulate: attention computation on wasted tokens vs saved tokens
        if wasted_len > 0:
            # Cost of recomputing wasted prefix tokens
            x_wasted = torch.randn(B, wasted_len, H, device=device, dtype=torch.float16)
            w = torch.randn(H, N, device=device, dtype=torch.float16)
            def wasted_fn():
                return x_wasted @ w
            t_wasted = benchmark_cuda(wasted_fn, warmup=5, repeat=50)

            # Cost of NOT recomputing (if token-level sharing worked)
            # These tokens would be 0ms if cached
            wasted_cost_pct = t_wasted  # ms per request for wasted computation
        else:
            t_wasted = 0
            wasted_cost_pct = 0

        print(f"  prefix_len={prefix_len}: cached={cached_len}, wasted={wasted_len} "
              f"({waste_pct:.1f}% waste), recompute_cost={t_wasted:.4f}ms")

        results.append({
            "prefix_len": prefix_len,
            "block_size": block_size,
            "cached_len": cached_len,
            "wasted_len": wasted_len,
            "cache_efficiency": round(cache_efficiency, 4),
            "waste_pct": round(waste_pct, 1),
            "recompute_cost_ms": round(t_wasted, 4),
        })

    return results

# ================================================================
# Experiment 3: Recompute vs Cache (Cost Comparison)
# ================================================================
def exp3_recompute_vs_cache():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 3: Recompute vs Cache Cost Comparison")
    print("=" * 60)

    # When prefix is not cached → recompute it (extra forward pass)
    # When prefix IS cached → no recompute, but need to READ cached KV
    # Which is cheaper? (Recompute ∝ compute_cost, Read ∝ HBM bandwidth)

    H = 4096
    N = 4096
    B_sizes = [1, 4, 8, 16, 32, 64, 128]
    prefix_len = 128  # shared prefix length

    w_proj = torch.randn(H, N, device=device, dtype=torch.float16)

    results = []

    for B in B_sizes:
        # Cost 1: Recompute prefix (forward pass for prefix_len tokens)
        x_prefix = torch.randn(B, prefix_len, H, device=device, dtype=torch.float16)

        def recompute_fn():
            return x_prefix @ w_proj

        t_recompute = benchmark_cuda(recompute_fn, warmup=5, repeat=50)

        # Cost 2: Read cached prefix KV from memory
        # KV cache size: 2 × B × prefix_len × H × 2 bytes (FP16)
        # = 2 × B × prefix_len × 8192 bytes
        kv_size_bytes = 2 * B * prefix_len * H * 2  # FP16

        # Simulate KV read by copying KV tensors
        kv_cache = torch.randn(2, B, prefix_len, H, device=device, dtype=torch.float16)
        kv_output = torch.empty_like(kv_cache)

        def kv_read_fn():
            kv_output.copy_(kv_cache)

        t_kv_read = benchmark_cuda(kv_read_fn, warmup=5, repeat=50)

        # Compute savings
        # Recompute: pure compute (memory-bound for small B, compute-bound for large B)
        # KV read: pure memory (copy = HBM bandwidth)
        savings_pct = (1 - t_kv_read / t_recompute) * 100

        # TFLOPS for recompute
        tflops_recompute = 2 * B * prefix_len * H * N / (t_recompute * 1e-3) / 1e12

        # HBM bandwidth for KV read
        kv_bw = kv_size_bytes / (t_kv_read * 1e-3) / 1e9  # GB/s

        print(f"  B={B}: recompute={t_recompute:.4f}ms({tflops_recompute:.2f} TFLOPS), "
              f"KV_read={t_kv_read:.4f}ms({kv_bw:.1f} GB/s), "
              f"savings={savings_pct:.1f}%")

        results.append({
            "B": B,
            "prefix_len": prefix_len,
            "recompute_ms": round(t_recompute, 4),
            "kv_read_ms": round(t_kv_read, 4),
            "recompute_tflops": round(tflops_recompute, 2),
            "kv_read_bw_gb_s": round(kv_bw, 1),
            "savings_pct": round(savings_pct, 1),
            "kv_size_mb": round(kv_size_bytes / 1e6, 2),
        })

    return results

# ================================================================
# Experiment 4: Multi-Turn Conversation Cumulative Savings
# ================================================================
def exp4_multi_turn_savings():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 4: Multi-Turn Conversation Cumulative Savings")
    print("=" * 60)

    # Simulate multi-turn conversation:
    # Turn 1: system_prompt(512) + user_query(64) + response(256)
    # Turn 2: system_prompt(512) + user1_query(64) + response1(256) + user2_query(64) + response2(256)
    # Turn 3: ... (cumulative history grows)
    # Each turn: new_tokens = user_query + response, cached = all previous tokens

    H = 4096
    N = 4096
    B = 32  # concurrent users

    system_prompt_len = 512
    user_query_len = 64
    response_len = 256
    w_proj = torch.randn(H, N, device=device, dtype=torch.float16)

    results = []

    for n_turns in [1, 2, 3, 5, 10]:
        # Cumulative context length after n_turns
        total_len = system_prompt_len + n_turns * (user_query_len + response_len)
        # New tokens to compute this turn
        new_len = user_query_len + response_len
        # Cached tokens (all previous)
        cached_len = total_len - new_len
        # Prefix ratio
        prefix_ratio = cached_len / total_len

        # Compute time for full forward (no cache)
        x_full = torch.randn(B, total_len, H, device=device, dtype=torch.float16)
        def full_fn():
            return x_full @ w_proj
        t_full = benchmark_cuda(full_fn, warmup=5, repeat=50)

        # Compute time for just new tokens (with cache)
        x_new = torch.randn(B, new_len, H, device=device, dtype=torch.float16)
        def new_fn():
            return x_new @ w_proj
        t_new = benchmark_cuda(new_fn, warmup=5, repeat=50)

        # Cumulative savings over n_turns (each turn caches previous)
        # Total tokens without cache: Σ total_len_i for each turn
        # Total tokens with cache: Σ new_len_i for each turn (first turn has no cache)
        total_no_cache = sum(system_prompt_len + t * (user_query_len + response_len)
                           for t in range(1, n_turns + 1))
        total_with_cache = system_prompt_len + user_query_len + response_len  # turn 1 (no cache)
        for t in range(2, n_turns + 1):
            total_with_cache += user_query_len + response_len  # only new tokens

        cumulative_savings = (1 - total_with_cache / total_no_cache) * 100

        print(f"  n_turns={n_turns}: total={total_len}, cached={cached_len} "
              f"({prefix_ratio:.1%}), t_full={t_full:.4f}ms, "
              f"t_new={t_new:.4f}ms, cumulative_savings={cumulative_savings:.1f}%")

        results.append({
            "n_turns": n_turns,
            "total_len": total_len,
            "cached_len": cached_len,
            "new_len": new_len,
            "prefix_ratio": round(prefix_ratio, 3),
            "full_forward_ms": round(t_full, 4),
            "new_only_ms": round(t_new, 4),
            "cumulative_savings_pct": round(cumulative_savings, 1),
            "total_no_cache_tokens": total_no_cache,
            "total_with_cache_tokens": total_with_cache,
        })

    return results

# ================================================================
# Experiment 5: GRPO n_samples Prefix Sharing
# ================================================================
def exp5_grpo_prefix_sharing():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 5: GRPO n_samples Prefix Sharing")
    print("=" * 60)

    # GRPO: each prompt sampled n times → all n responses share prefix
    # Without sharing: n independent forward passes → n× prefix cost
    # With sharing: 1× prefix + n× suffix → savings = (n-1)/n × prefix_ratio

    H = 4096
    N = 4096
    prompt_len = 512
    response_len = 256

    n_samples = [2, 4, 8, 16]
    B = 16  # number of prompts per batch

    w_proj = torch.randn(H, N, device=device, dtype=torch.float16)

    results = []

    for n in n_samples:
        total_len = prompt_len + response_len
        prefix_ratio = prompt_len / total_len

        # Without sharing: n forward passes for each prompt
        # Total: n × total_len tokens per prompt
        total_no_sharing = n * total_len

        # With sharing: 1 prefix forward + n suffix forwards
        # Prefix: prompt_len tokens (computed once)
        # Suffix: response_len tokens per sample (n times)
        total_with_sharing = prompt_len + n * response_len

        compute_savings = (1 - total_with_sharing / total_no_sharing) * 100

        # Actual GPU timing
        # No sharing: n × forward(total_len)
        x_full = torch.randn(B * n, total_len, H, device=device, dtype=torch.float16)
        def no_sharing_fn():
            return x_full @ w_proj
        t_no_sharing = benchmark_cuda(no_sharing_fn, warmup=5, repeat=50)

        # With sharing: 1 × forward(prompt_len) + n × forward(response_len)
        x_prefix = torch.randn(B, prompt_len, H, device=device, dtype=torch.float16)
        x_suffix = torch.randn(B * n, response_len, H, device=device, dtype=torch.float16)

        # Combined: prefix (B tokens) + suffix (B*n tokens) = flat layout
        x_combined = torch.randn(B + B * n, max(prompt_len, response_len),
                                 H, device=device, dtype=torch.float16)

        def sharing_prefix_fn():
            return x_prefix @ w_proj

        def sharing_suffix_fn():
            return x_suffix @ w_proj

        t_prefix = benchmark_cuda(sharing_prefix_fn, warmup=5, repeat=50)
        t_suffix = benchmark_cuda(sharing_suffix_fn, warmup=5, repeat=50)
        t_with_sharing = t_prefix + t_suffix

        time_savings = (1 - t_with_sharing / t_no_sharing) * 100

        # Throughput
        throughput_no_sharing = B * n * total_len / (t_no_sharing * 1e-3)
        throughput_with_sharing = B * n * response_len / (t_suffix * 1e-3)

        print(f"  n={n}: compute_savings={compute_savings:.1f}%, "
              f"time_no_sharing={t_no_sharing:.4f}ms, "
              f"time_with_sharing={t_with_sharing:.4f}ms, "
              f"time_savings={time_savings:.1f}%")

        results.append({
            "n_samples": n,
            "B": B,
            "prompt_len": prompt_len,
            "response_len": response_len,
            "prefix_ratio": round(prefix_ratio, 3),
            "compute_savings_pct": round(compute_savings, 1),
            "time_no_sharing_ms": round(t_no_sharing, 4),
            "time_with_sharing_ms": round(t_with_sharing, 4),
            "time_savings_pct": round(time_savings, 1),
            "throughput_no_sharing": round(throughput_no_sharing, 0),
            "throughput_with_sharing_new": round(throughput_with_sharing, 0),
            "total_no_sharing_tokens": total_no_sharing,
            "total_with_sharing_tokens": total_with_sharing,
        })

    return results

# ================================================================
# Main
# ================================================================
def main():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9

    print(f"Prefix Cache Throughput Benchmark: {gpu_name} ({gpu_mem:.1f} GB)")
    print("=" * 60)

    all_results = {"gpu": gpu_name, "gpu_mem_gb": round(gpu_mem, 1)}

    all_results["exp1_prefix_cache_hit_rate"] = exp1_prefix_cache_hit_rate()
    all_results["exp2_block_alignment"] = exp2_block_alignment()
    all_results["exp3_recompute_vs_cache"] = exp3_recompute_vs_cache()
    all_results["exp4_multi_turn_savings"] = exp4_multi_turn_savings()
    all_results["exp5_grpo_prefix_sharing"] = exp5_grpo_prefix_sharing()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Exp 1: hit rate impact
    hit_rate = all_results["exp1_prefix_cache_hit_rate"]
    b32_75 = next(r for r in hit_rate if r["B"] == 32 and r["prefix_ratio"] == 0.75)
    print(f"  75% prefix ratio (B=32): {b32_75['time_savings_pct']}% time savings "
          f"(compute savings {b32_75['compute_savings_pct']}%)")

    # Exp 2: block alignment waste
    block = all_results["exp2_block_alignment"]
    worst_waste = max(r for r in block if r["waste_pct"] > 0 and r["prefix_len"] < 256)
    print(f"  Worst block waste: prefix_len={worst_waste['prefix_len']}, "
          f"{worst_waste['waste_pct']}% wasted, {worst_waste['recompute_cost_ms']}ms recompute")

    # Exp 3: recompute vs cache
    recomp = all_results["exp3_recompute_vs_cache"]
    b1 = next(r for r in recomp if r["B"] == 1)
    b128 = next(r for r in recomp if r["B"] == 128)
    print(f"  Recompute vs Cache: B=1 savings {b1['savings_pct']}%, "
          f"B=128 savings {b128['savings_pct']}%")

    # Exp 4: multi-turn
    multi = all_results["exp4_multi_turn_savings"]
    mt5 = next(r for r in multi if r["n_turns"] == 5)
    print(f"  5-turn conversation: {mt5['cumulative_savings_pct']}% cumulative savings")

    # Exp 5: GRPO
    grpo = all_results["exp5_grpo_prefix_sharing"]
    g8 = next(r for r in grpo if r["n_samples"] == 8)
    print(f"  GRPO n=8: {g8['compute_savings_pct']}% compute savings, "
          f"{g8['time_savings_pct']}% time savings")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'prefix_cache_throughput_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()