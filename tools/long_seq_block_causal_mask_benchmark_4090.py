"""Long-Sequence Block-Causal Mask Benchmark — RTX 4090

Validates that LSE merge (FlashAttention backend) becomes faster than
SDPA math backend for long prefix sequences (4K+).

Short-seq result: LSE merge 0.54x slower (2 FA calls + LSE overhead > math for small matrices)
Long-seq expectation: LSE merge faster (math O(N²) vs FlashAttn O(N) IO)

Benchmarks pure attention layer timing for:
- prefix_len: 256, 512, 1024, 2048, 4096, 6144, 8192
- suffix_len: 128 (fixed, typical GRPO response length)
"""

import json
import time
import torch
import math

DEVICE = "cuda:0"
DTYPE = torch.float16

def warmup(device, n=100):
    x = torch.randn(256, 256, device=device, dtype=DTYPE)
    for _ in range(n):
        y = x @ x
    torch.cuda.synchronize()


def single_sdpa_block_causal(q, prefix_k, prefix_v, suffix_k, suffix_v,
                              num_heads, num_kv_heads, head_dim, prefix_len):
    """Single SDPA call with block-causal float mask (math backend)."""
    g = num_heads // num_kv_heads
    suffix_len = q.shape[2]

    # Expand GQA KV
    if g > 1:
        prefix_k = prefix_k.repeat_interleave(g, dim=1)
        prefix_v = prefix_v.repeat_interleave(g, dim=1)
        suffix_k = suffix_k.repeat_interleave(g, dim=1)
        suffix_v = suffix_v.repeat_interleave(g, dim=1)

    full_k = torch.cat([prefix_k, suffix_k], dim=2)
    full_v = torch.cat([prefix_v, suffix_v], dim=2)

    total_kv_len = prefix_len + suffix_len
    model_dtype = q.dtype
    mask_2d = torch.zeros(suffix_len, total_kv_len, dtype=model_dtype, device=q.device)
    suffix_idx = torch.arange(suffix_len, device=q.device)
    kv_idx = torch.arange(total_kv_len, device=q.device)
    future_positions = kv_idx.unsqueeze(0) > (suffix_idx.unsqueeze(1) + prefix_len)
    mask_2d[future_positions] = float('-inf')
    attn_mask = mask_2d.unsqueeze(0).unsqueeze(0).expand(1, num_heads, -1, -1).contiguous()

    out = torch.nn.functional.scaled_dot_product_attention(
        q, full_k, full_v, attn_mask=attn_mask, is_causal=False
    )
    out = out.permute(0, 2, 1, 3).reshape(1, suffix_len, num_heads * head_dim).to(model_dtype)
    return out


def lse_merge_attn(q, prefix_k, prefix_v, suffix_k, suffix_v, num_heads, num_kv_heads, head_dim):
    """Block-causal attention via two FlashAttention calls + LSE merge."""
    from flash_attn import flash_attn_func

    g = num_heads // num_kv_heads
    suffix_len = q.shape[2]
    prefix_len = prefix_k.shape[2]

    # Expand GQA KV if needed
    if g > 1:
        prefix_k = prefix_k.repeat_interleave(g, dim=1)
        prefix_v = prefix_v.repeat_interleave(g, dim=1)
        suffix_k = suffix_k.repeat_interleave(g, dim=1)
        suffix_v = suffix_v.repeat_interleave(g, dim=1)

    # FlashAttention format: [batch, seqlen, num_heads, head_dim]
    q_fa = q.permute(0, 2, 1, 3)
    p_k_fa = prefix_k.permute(0, 2, 1, 3)
    p_v_fa = prefix_v.permute(0, 2, 1, 3)
    s_k_fa = suffix_k.permute(0, 2, 1, 3)
    s_v_fa = suffix_v.permute(0, 2, 1, 3)

    # Call 1: Q_suffix × K_prefix (full attention)
    out_prefix, lse_prefix, _ = flash_attn_func(
        q_fa, p_k_fa, p_v_fa, causal=False, return_attn_probs=True
    )
    # Call 2: Q_suffix × K_suffix (causal)
    out_suffix, lse_suffix, _ = flash_attn_func(
        q_fa, s_k_fa, s_v_fa, causal=True, return_attn_probs=True
    )

    # LSE merge (stable)
    lse_p = lse_prefix.squeeze(0).transpose(0, 1)  # [suffix_len, num_heads]
    lse_s = lse_suffix.squeeze(0).transpose(0, 1)

    max_lse = torch.maximum(lse_p, lse_s)
    exp_p = torch.exp(lse_p - max_lse)
    exp_s = torch.exp(lse_s - max_lse)
    total = exp_p + exp_s

    out_p = out_prefix.squeeze(0)  # [suffix_len, num_heads, head_dim]
    out_s = out_suffix.squeeze(0)

    alpha_p = (exp_p / total).unsqueeze(-1)
    alpha_s = (exp_s / total).unsqueeze(-1)
    merged = alpha_p * out_p + alpha_s * out_s
    merged = merged.reshape(1, suffix_len, num_heads * head_dim)

    return merged


def main():
    print("=" * 60)
    print("Long-Sequence Block-Causal Mask Benchmark — RTX 4090")
    print("LSE merge (FlashAttn) vs SDPA math backend")
    print("=" * 60)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}, Memory: {gpu_mem:.1f} GB\n")

    warmup(DEVICE)

    # Model config: GQA-4 (20 query heads, 4 kv heads, head_dim=128)
    num_heads = 20
    num_kv_heads = 4
    head_dim = 128
    suffix_len = 128  # typical GRPO short response

    results = []

    for prefix_len in [256, 512, 1024, 2048, 4096, 6144, 8192]:
        total_len = prefix_len + suffix_len
        print(f"--- prefix={prefix_len}, suffix={suffix_len}, total={total_len} ---")

        # Create test tensors
        q_test = torch.randn(1, num_heads, suffix_len, head_dim, device=DEVICE, dtype=DTYPE)
        pk_test = torch.randn(1, num_kv_heads, prefix_len, head_dim, device=DEVICE, dtype=DTYPE)
        pv_test = torch.randn(1, num_kv_heads, prefix_len, head_dim, device=DEVICE, dtype=DTYPE)
        sk_test = torch.randn(1, num_kv_heads, suffix_len, head_dim, device=DEVICE, dtype=DTYPE)
        sv_test = torch.randn(1, num_kv_heads, suffix_len, head_dim, device=DEVICE, dtype=DTYPE)

        # Precision check (first run)
        try:
            sdpa_out = single_sdpa_block_causal(q_test, pk_test, pv_test, sk_test, sv_test,
                                                 num_heads, num_kv_heads, head_dim, prefix_len)
            lse_out = lse_merge_attn(q_test, pk_test, pv_test, sk_test, sv_test,
                                     num_heads, num_kv_heads, head_dim)
            cos_sim = torch.nn.functional.cosine_similarity(
                sdpa_out.flatten().unsqueeze(0).float(),
                lse_out.flatten().unsqueeze(0).float()
            ).item()
            max_diff = (sdpa_out.float() - lse_out.float()).abs().max().item()
            print(f"  Precision: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}")
        except Exception as e:
            print(f"  Precision check failed: {e}")
            cos_sim = 0
            max_diff = float('inf')

        # Performance: SDPA math backend
        reps = 20
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(reps):
                single_sdpa_block_causal(q_test, pk_test, pv_test, sk_test, sv_test,
                                         num_heads, num_kv_heads, head_dim, prefix_len)
        torch.cuda.synchronize()
        sdpa_ms = (time.perf_counter() - t0) * 1000 / reps

        # Performance: LSE merge
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(reps):
                lse_merge_attn(q_test, pk_test, pv_test, sk_test, sv_test,
                               num_heads, num_kv_heads, head_dim)
        torch.cuda.synchronize()
        lse_ms = (time.perf_counter() - t0) * 1000 / reps

        speedup = sdpa_ms / lse_ms if lse_ms > 0 else 0
        print(f"  SDPA math: {sdpa_ms:.3f}ms, LSE merge: {lse_ms:.3f}ms, "
              f"speedup={speedup:.2f}x")

        results.append({
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "total_len": total_len,
            "cos_sim_lse_vs_sdpa": round(cos_sim, 6),
            "max_diff_lse_vs_sdpa": round(max_diff, 4),
            "sdpa_math_ms": round(sdpa_ms, 3),
            "lse_merge_ms": round(lse_ms, 3),
            "speedup_lse_vs_sdpa": round(speedup, 2),
        })

        # Free memory for next test
        del q_test, pk_test, pv_test, sk_test, sv_test
        if 'sdpa_out' in dir(): del sdpa_out
        if 'lse_out' in dir(): del lse_out
        torch.cuda.empty_cache()

    # Save results
    all_results = {
        "gpu": gpu_name,
        "gpu_mem_gb": round(gpu_mem, 1),
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "suffix_len": suffix_len,
        "results": results,
    }

    with open("long_seq_block_causal_mask_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nPrefix → SDPA math vs LSE merge (attention-only):")
    for r in results:
        direction = "LSE faster" if r["speedup_lse_vs_sdpa"] > 1 else "SDPA faster"
        print(f"  prefix={r['prefix_len']}: SDPA={r['sdpa_math_ms']}ms, "
              f"LSE={r['lse_merge_ms']}ms, ratio={r['speedup_lse_vs_sdpa']}x ({direction})")

    # Find crossover
    for r in results:
        if r["speedup_lse_vs_sdpa"] > 1:
            print(f"\n** Crossover point: prefix_len={r['prefix_len']} **")
            break

    print(f"\nResults saved to long_seq_block_causal_mask_results.json")


if __name__ == "__main__":
    main()