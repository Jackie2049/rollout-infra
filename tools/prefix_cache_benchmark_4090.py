#!/usr/bin/env python3
"""RTX 4090 Prefix Caching Benchmark
=====================================
Measures prefix caching effectiveness:
1. KV Cache reuse with shared prefixes
2. Memory savings from prefix deduplication
3. Batch prefill speedup with cached prefixes
4. RadixAttention-style tree sharing simulation
"""

import torch
import time
import json
import math

def benchmark(fn, warmup=5, repeat=50):
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


def main():
    print("=" * 60)
    print("RTX 4090 Prefix Caching Benchmark")
    print("=" * 60)

    props = torch.cuda.get_device_properties(0)
    print(f"\n  GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")

    results = {}

    # ===== Exp 1: Prefill Cost With and Without Cache =====
    print("\n" + "=" * 60)
    print("Experiment 1: Prefill Cost — Cached vs Uncached Prefix")
    print("=" * 60)

    # Simulate: system prompt (prefix) + user prompt (suffix)
    # Without cache: prefill entire sequence (prefix + suffix)
    # With cache: only prefill suffix (prefix KV already cached)
    prefix_lens = [64, 128, 256, 512, 1024]
    suffix_lens = [64, 128, 256]
    H = 4096  # hidden dim (7B model)
    num_layers = 32

    print(f"\n  {'Prefix':>8} {'Suffix':>8} {'Full(ms)':>10} {'Cache(ms)':>10} {'Saved%':>8} {'SavedKV(MB)':>12}")
    print("  " + "-" * 65)

    for pl in prefix_lens:
        for sl in suffix_lens:
            full_len = pl + sl

            # Full prefill: simulate attention + MLP for entire sequence
            # Attention: O(B * L * H) per layer (simplified, ignore O(L^2))
            # MLP: O(B * L * 4H) per layer
            # We simulate with matmul: hidden @ weight for each layer
            hidden_full = torch.randn(1, full_len, H, device='cuda', dtype=torch.float16)
            weight = torch.randn(H, H, device='cuda', dtype=torch.float16)

            ms_full = benchmark(lambda: hidden_full @ weight)

            # Cached prefill: only suffix
            hidden_suffix = torch.randn(1, sl, H, device='cuda', dtype=torch.float16)
            ms_cached = benchmark(lambda: hidden_suffix @ weight)

            saved_pct = (1 - ms_cached / ms_full) * 100
            # KV cache memory saved: 2 * layers * prefix_len * hidden * 2 bytes (FP16)
            kv_per_token = 2 * num_layers * H * 2  # bytes
            saved_kv_mb = pl * kv_per_token / 1e6

            print(f"  {pl:>8} {sl:>8} {ms_full:>10.3f} {ms_cached:>10.3f} {saved_pct:>7.1f}% {saved_kv_mb:>11.1f}")

            results[f"exp1_p{pl}_s{sl}"] = {
                "prefix_len": pl, "suffix_len": sl,
                "full_ms": round(ms_full, 3), "cached_ms": round(ms_cached, 3),
                "saved_pct": round(saved_pct, 1),
                "saved_kv_mb": round(saved_kv_mb, 1),
            }

            del hidden_full, hidden_suffix
            torch.cuda.empty_cache()

    # ===== Exp 2: Batch Sharing with Common Prefix =====
    print("\n" + "=" * 60)
    print("Experiment 2: Batch Sharing — Common Prefix Deduplication")
    print("=" * 60)

    prefix_len = 256
    batch_sizes = [1, 4, 8, 16, 32]
    suffix_len = 128

    print(f"\n  Common prefix={prefix_len}, suffix={suffix_len}")
    print(f"  {'Batch':>6} {'NoCache(ms)':>12} {'Cache(ms)':>12} {'Speedup':>8} {'MemSaved(MB)':>14}")
    print("  " + "-" * 60)

    for bs in batch_sizes:
        full_len = prefix_len + suffix_len

        # No cache: each request processes full sequence independently
        hidden_full = torch.randn(bs, full_len, H, device='cuda', dtype=torch.float16)
        weight = torch.randn(H, H, device='cuda', dtype=torch.float16)
        ms_nocache = benchmark(lambda: hidden_full @ weight)

        # With cache: prefix computed once, suffix computed per request
        hidden_prefix = torch.randn(1, prefix_len, H, device='cuda', dtype=torch.float16)
        ms_prefix = benchmark(lambda: hidden_prefix @ weight)
        hidden_suffix = torch.randn(bs, suffix_len, H, device='cuda', dtype=torch.float16)
        ms_suffix = benchmark(lambda: hidden_suffix @ weight)
        ms_cache = ms_prefix + ms_suffix

        speedup = ms_nocache / ms_cache
        # Memory saved: (batch_size - 1) * prefix_len * kv_per_token
        kv_per_token = 2 * num_layers * H * 2
        mem_saved = (bs - 1) * prefix_len * kv_per_token / 1e6

        print(f"  {bs:>6} {ms_nocache:>12.3f} {ms_cache:>12.3f} {speedup:>8.2f}x {mem_saved:>13.1f}")

        results[f"exp2_b{bs}"] = {
            "batch_size": bs, "no_cache_ms": round(ms_nocache, 3),
            "cache_ms": round(ms_cache, 3), "speedup": round(speedup, 2),
            "mem_saved_mb": round(mem_saved, 1),
        }

        del hidden_full, hidden_suffix, hidden_prefix
        torch.cuda.empty_cache()

    # ===== Exp 3: Multi-turn Conversation Savings =====
    print("\n" + "=" * 60)
    print("Experiment 3: Multi-turn Conversation (Cumulative Savings)")
    print("=" * 60)

    turn_lens = [128, 256, 512, 1024]  # tokens per turn
    num_turns = 5

    for turn_len in turn_lens:
        print(f"\n  Turn length={turn_len} tokens:")
        print(f"  {'Turn':>6} {'NewTokens':>10} {'CachedTokens':>13} {'CacheRatio%':>12} {'CumKV(MB)':>12}")
        print("  " + "-" * 60)

        cum_new = 0
        cum_cached = 0
        kv_per_token = 2 * num_layers * H * 2

        for turn in range(1, num_turns + 1):
            new_tokens = turn_len  # new turn tokens
            cached_tokens = (turn - 1) * turn_len  # all previous turns

            cum_new += new_tokens
            cum_cached += cached_tokens
            total = cum_new + cum_cached
            cache_ratio = cum_cached / total * 100 if total > 0 else 0
            cum_kv = cum_cached * kv_per_token / 1e6

            print(f"  {turn:>6} {new_tokens:>10} {cached_tokens:>13} {cache_ratio:>11.1f}% {cum_kv:>11.1f}")

            results[f"exp3_t{turn_len}_turn{turn}"] = {
                "turn_len": turn_len, "turn": turn,
                "new_tokens": new_tokens, "cached_tokens": cached_tokens,
                "cache_ratio": round(cache_ratio, 1),
                "cum_kv_mb": round(cum_kv, 1),
            }

    # ===== Exp 4: RadixAttention Tree Sharing =====
    print("\n" + "=" * 60)
    print("Experiment 4: RadixAttention Tree Sharing Simulation")
    print("=" * 60)

    # Simulate a tree of prompts with shared prefixes
    # Root (system prompt 256 tok) -> 2 branches -> each 2 sub-branches
    # Total 4 leaf requests
    system_prompt = 256
    branch_lens = [64, 128, 64, 128]  # different branch lengths
    leaf_lens = [32, 64, 32, 64]      # leaf-specific tokens

    total_tokens_no_sharing = sum((system_prompt + b + l) for b, l in zip(branch_lens, leaf_lens))
    total_tokens_with_sharing = system_prompt + sum(b + l for b, l in zip(branch_lens, leaf_lens))
    # RadixAttention: shared system prompt computed once
    savings_pct = (1 - total_tokens_with_sharing / total_tokens_no_sharing) * 100
    kv_per_token = 2 * num_layers * H * 2
    saved_kv = (3 * system_prompt) * kv_per_token / 1e6  # 4 requests sharing 1 prefix

    print(f"\n  Tree structure: 1 root (sys={system_prompt}) -> 4 leaves")
    print(f"  No sharing: {total_tokens_no_sharing} tokens across 4 requests")
    print(f"  With RadixAttention: {total_tokens_with_sharing} tokens (prefix computed once)")
    print(f"  Savings: {savings_pct:.1f}% compute, {saved_kv:.1f} MB KV cache")

    results["exp4_tree"] = {
        "total_no_sharing": total_tokens_no_sharing,
        "total_with_sharing": total_tokens_with_sharing,
        "savings_pct": round(savings_pct, 1),
        "saved_kv_mb": round(saved_kv, 1),
    }

    # ===== Exp 5: KV Cache Block Management =====
    print("\n" + "=" * 60)
    print("Experiment 5: KV Cache Block Management Analysis")
    print("=" * 60)

    block_sizes = [8, 16, 32, 64]
    total_vram = 24.0  # GB
    model_size = 14.0  # GB (7B FP16)
    kv_budget = (total_vram * 0.8 - model_size) * 1e9  # bytes

    print(f"\n  Model: 7B FP16 ({model_size} GB), KV budget: {kv_budget/1e9:.1f} GB")
    print(f"  {'BlockSize':>10} {'Blocks':>8} {'MaxSeq':>8} {'WasteRate%':>11} {'FragmentMB':>12}")
    print("  " + "-" * 55)

    for bs in block_sizes:
        kv_per_block = bs * kv_per_token  # bytes per block
        num_blocks = int(kv_budget / kv_per_block)
        max_seq = num_blocks * bs  # if all blocks for 1 request

        # Internal fragmentation: average waste per request
        # Assume requests uniformly distributed in [128, 2048] tokens
        avg_waste = 0  # calculate below
        for req_len in range(128, 2049, 128):
            blocks_needed = math.ceil(req_len / bs)
            waste = blocks_needed * bs - req_len
            avg_waste += waste
        avg_waste /= len(range(128, 2049, 128))
        waste_rate = avg_waste / ((128 + 2048) / 2) * 100
        fragment_mb = avg_waste * kv_per_token / 1e6

        print(f"  {bs:>10} {num_blocks:>8} {max_seq:>8} {waste_rate:>10.1f}% {fragment_mb:>11.1f}")

        results[f"exp5_bs{bs}"] = {
            "block_size": bs, "num_blocks": num_blocks,
            "max_seq": max_seq, "waste_rate": round(waste_rate, 1),
            "fragment_mb": round(fragment_mb, 1),
        }

    # ===== Exp 6: Prefix Sharing in RLHF Rollout =====
    print("\n" + "=" * 60)
    print("Experiment 6: RLHF Rollout Prefix Sharing (GRPO n=8)")
    print("=" * 60)

    prompt_len = 512
    response_lens = [128, 256, 512, 1024]
    n_samples = 8  # GRPO: generate n responses per prompt

    print(f"\n  GRPO: prompt={prompt_len}, n={n_samples}")
    print(f"  {'RespLen':>8} {'NoShare(tok)':>13} {'Share(tok)':>12} {'Save%':>7} {'KVSave(MB)':>12}")
    print("  " + "-" * 55)

    for resp_len in response_lens:
        total_no_share = n_samples * (prompt_len + resp_len)
        total_share = prompt_len + n_samples * resp_len  # prompt computed once
        save_pct = (1 - total_share / total_no_share) * 100
        kv_save = (n_samples - 1) * prompt_len * kv_per_token / 1e6

        print(f"  {resp_len:>8} {total_no_share:>13} {total_share:>12} {save_pct:>6.1f}% {kv_save:>11.1f}")

        results[f"exp6_resp{resp_len}"] = {
            "resp_len": resp_len,
            "total_no_share": total_no_share,
            "total_share": total_share,
            "save_pct": round(save_pct, 1),
            "kv_save_mb": round(kv_save, 1),
        }

    with open("prefix_cache_4090_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to prefix_cache_4090_results.json")


if __name__ == "__main__":
    main()
