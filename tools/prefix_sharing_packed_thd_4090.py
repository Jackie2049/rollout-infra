"""Prefix Sharing Packed THD Micro-Benchmark — RTX 4090

5 experiments simulating the prefix-sharing architecture from prefix-0501 project:
  Exp1: Packed THD KV injection overhead (provider KV store -> reuser KV inject)
  Exp2: Prefix-sharing savings with varying n_samples (GRPO)
  Exp3: GQA 24:4 prefix KV injection (Qwen3.6-27B config)
  Exp4: Packed vs unpacked attention throughput comparison
  Exp5: Prefix-Last Restore overhead (logprob recovery)

This benchmark simulates the One-Forward + KV Injection + Prefix-Last Restore
architecture that prefix-0501 uses for training-time prefix sharing.
"""

import json
import time
import torch

DEVICE = "cuda:0"
DTYPE = torch.float16


def warmup(device, n=50):
    x = torch.randn(256, 256, device=device, dtype=DTYPE)
    for _ in range(n):
        y = x @ x
    torch.cuda.synchronize()


def gqa_sdpa(q_bshd, k_bshd, v_bshd, num_heads, num_kv_heads, is_causal=False):
    """GQA attention. Input/output in (B, S, H, D) format."""
    q = q_bshd.permute(0, 2, 1, 3)  # (B, H_q, S, D)
    k = k_bshd.permute(0, 2, 1, 3)  # (B, H_kv, S, D)
    v = v_bshd.permute(0, 2, 1, 3)  # (B, H_kv, S, D)
    if num_heads != num_kv_heads:
        g = num_heads // num_kv_heads
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    return out.permute(0, 2, 1, 3)  # (B, S, H_q, D)


# ---------------------------------------------------------------------------
# Exp1: KV Injection Overhead
# ---------------------------------------------------------------------------
def exp1_kv_injection():
    results = []
    num_heads = 32
    num_kv_heads = 4
    head_dim = 128

    for prefix_len in [64, 128, 256, 512, 1024]:
        suffix_len = 128
        total_len = prefix_len + suffix_len
        B = 8

        # --- Full forward: all B sequences compute independently ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        all_k = torch.randn(B, total_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
        all_v = torch.randn(B, total_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
        q = torch.randn(B, suffix_len, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
        out = gqa_sdpa(q, all_k, all_v, num_heads, num_kv_heads)
        torch.cuda.synchronize()
        full_ms = (time.perf_counter() - t0) * 1000

        # --- PS forward: 1 provider + B-1 reusers with KV injection ---
        provider_prefix_k = torch.randn(1, prefix_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
        provider_prefix_v = torch.randn(1, prefix_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # Reusers only compute suffix KV
        reuser_suffix_k = torch.randn(B-1, suffix_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
        reuser_suffix_v = torch.randn(B-1, suffix_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
        # KV injection: concat provider prefix + reuser suffix
        injected_k = torch.cat([provider_prefix_k.expand(B-1, -1, -1, -1), reuser_suffix_k], dim=1)
        injected_v = torch.cat([provider_prefix_v.expand(B-1, -1, -1, -1), reuser_suffix_v], dim=1)
        q_ps = torch.randn(B-1, suffix_len, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
        out_ps = gqa_sdpa(q_ps, injected_k, injected_v, num_heads, num_kv_heads)
        torch.cuda.synchronize()
        ps_ms = (time.perf_counter() - t0) * 1000

        compute_savings_pct = (B - 1) / B * prefix_len / total_len * 100
        time_savings_pct = (1 - ps_ms / full_ms) * 100 if full_ms > 0 else 0

        results.append({
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "total_len": total_len,
            "B": B,
            "full_forward_ms": round(full_ms, 3),
            "ps_forward_ms": round(ps_ms, 3),
            "compute_savings_pct": round(compute_savings_pct, 1),
            "time_savings_pct": round(time_savings_pct, 1),
            "speedup": round(full_ms / ps_ms, 2) if ps_ms > 0 else 0,
        })

    return results


# ---------------------------------------------------------------------------
# Exp2: GRPO n_samples Savings
# ---------------------------------------------------------------------------
def exp2_grpo_savings():
    results = []
    num_heads = 32
    num_kv_heads = 4
    head_dim = 128
    prefix_len = 512
    suffix_len = 256
    total_len = prefix_len + suffix_len

    for n_samples in [2, 4, 8, 16]:
        # Full forward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        all_k = torch.randn(n_samples, total_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
        all_v = torch.randn(n_samples, total_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
        q = torch.randn(n_samples, suffix_len, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
        out = gqa_sdpa(q, all_k, all_v, num_heads, num_kv_heads)
        torch.cuda.synchronize()
        full_ms = (time.perf_counter() - t0) * 1000

        # PS forward: 1 provider + (n-1) reusers
        provider_prefix_k = torch.randn(1, prefix_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
        provider_prefix_v = torch.randn(1, prefix_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reuser_suffix_k = torch.randn(n_samples-1, suffix_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
        reuser_suffix_v = torch.randn(n_samples-1, suffix_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
        injected_k = torch.cat([provider_prefix_k.expand(n_samples-1, -1, -1, -1), reuser_suffix_k], dim=1)
        injected_v = torch.cat([provider_prefix_v.expand(n_samples-1, -1, -1, -1), reuser_suffix_v], dim=1)
        q_ps = torch.randn(n_samples-1, suffix_len, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
        out_ps = gqa_sdpa(q_ps, injected_k, injected_v, num_heads, num_kv_heads)
        torch.cuda.synchronize()
        ps_ms = (time.perf_counter() - t0) * 1000

        compute_savings = (n_samples - 1) / n_samples * prefix_len / total_len * 100
        time_savings = (1 - ps_ms / full_ms) * 100 if full_ms > 0 else 0

        results.append({
            "n_samples": n_samples,
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "compute_savings_pct": round(compute_savings, 1),
            "time_savings_pct": round(time_savings, 1),
            "speedup": round(full_ms / ps_ms, 2) if ps_ms > 0 else 0,
            "full_ms": round(full_ms, 3),
            "ps_ms": round(ps_ms, 3),
        })

    return results


# ---------------------------------------------------------------------------
# Exp3: Qwen3.6 GQA 24:4 Prefix KV Injection
# ---------------------------------------------------------------------------
def exp3_qwen36_gqa():
    results = []
    num_heads = 24
    num_kv_heads = 4
    head_dim = 256

    for seq_len in [256, 512, 1024, 2048]:
        B = 8
        prefix_ratio = 0.75
        prefix_len = int(seq_len * prefix_ratio)
        suffix_len = seq_len - prefix_len

        # Full forward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        all_k = torch.randn(B, seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
        all_v = torch.randn(B, seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
        q = torch.randn(B, suffix_len, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
        out = gqa_sdpa(q, all_k, all_v, num_heads, num_kv_heads)
        torch.cuda.synchronize()
        full_ms = (time.perf_counter() - t0) * 1000

        # PS forward
        provider_prefix_k = torch.randn(1, prefix_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
        provider_prefix_v = torch.randn(1, prefix_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reuser_suffix_k = torch.randn(B-1, suffix_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
        reuser_suffix_v = torch.randn(B-1, suffix_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
        injected_k = torch.cat([provider_prefix_k.expand(B-1, -1, -1, -1), reuser_suffix_k], dim=1)
        injected_v = torch.cat([provider_prefix_v.expand(B-1, -1, -1, -1), reuser_suffix_v], dim=1)
        q_ps = torch.randn(B-1, suffix_len, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
        out_ps = gqa_sdpa(q_ps, injected_k, injected_v, num_heads, num_kv_heads)
        torch.cuda.synchronize()
        ps_ms = (time.perf_counter() - t0) * 1000

        # KV injection overhead (cat + expand only)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _injected = torch.cat([provider_prefix_k.expand(B-1, -1, -1, -1), reuser_suffix_k], dim=1)
        torch.cuda.synchronize()
        inject_ms = (time.perf_counter() - t0) * 1000

        full_attn_ratio = 16 / 64  # Qwen3.6: 16 full attn / 64 total layers

        results.append({
            "seq_len": seq_len,
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "B": B,
            "full_ms": round(full_ms, 3),
            "ps_ms": round(ps_ms, 3),
            "inject_overhead_ms": round(inject_ms, 3),
            "speedup_per_full_attn_layer": round(full_ms / ps_ms, 2) if ps_ms > 0 else 0,
            "full_attn_layer_ratio": round(full_attn_ratio, 3),
            "estimated_total_speedup": round((1 + full_attn_ratio * (full_ms / ps_ms - 1)), 2) if ps_ms > 0 else 1.0,
        })

    return results


# ---------------------------------------------------------------------------
# Exp4: Packed vs Unpacked Attention Throughput
# ---------------------------------------------------------------------------
def exp4_packed_vs_unpacked():
    results = []
    num_heads = 24
    num_kv_heads = 4
    head_dim = 256

    for B in [2, 4, 8, 16]:
        seq_len = 512

        # Unpacked BSH: all B sequences compute full length
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            k_full = torch.randn(B, seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
            v_full = torch.randn(B, seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
            q_full = torch.randn(B, 128, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
            out = gqa_sdpa(q_full, k_full, v_full, num_heads, num_kv_heads)
        torch.cuda.synchronize()
        full_bsh_ms = (time.perf_counter() - t0) * 1000 / 10

        # PS packed: 1 provider (full seq) + B-1 reusers (suffix only = 128 tokens)
        suffix_len = 128
        prefix_len = seq_len - suffix_len

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            # Provider full KV (injected to all reusers)
            p_k = torch.randn(1, seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
            p_v = torch.randn(1, seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
            # Reuser suffix KV
            r_k = torch.randn(B-1, suffix_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
            r_v = torch.randn(B-1, suffix_len, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
            # Inject: provider prefix + reuser suffix
            inj_k = torch.cat([p_k[:, :prefix_len].expand(B-1, -1, -1, -1), r_k], dim=1)
            inj_v = torch.cat([p_v[:, :prefix_len].expand(B-1, -1, -1, -1), r_v], dim=1)
            q_ps = torch.randn(B-1, suffix_len, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
            out = gqa_sdpa(q_ps, inj_k, inj_v, num_heads, num_kv_heads)
        torch.cuda.synchronize()
        ps_packed_ms = (time.perf_counter() - t0) * 1000 / 10

        # Token reduction
        full_tokens = B * seq_len
        ps_tokens = seq_len + (B-1) * suffix_len  # provider full + reusers suffix
        token_reduction_pct = (1 - ps_tokens / full_tokens) * 100

        results.append({
            "B": B,
            "seq_len": seq_len,
            "full_bsh_ms": round(full_bsh_ms, 3),
            "ps_packed_ms": round(ps_packed_ms, 3),
            "speedup": round(full_bsh_ms / ps_packed_ms, 2) if ps_packed_ms > 0 else 0,
            "full_tokens": full_tokens,
            "ps_tokens": ps_tokens,
            "token_reduction_pct": round(token_reduction_pct, 1),
        })

    return results


# ---------------------------------------------------------------------------
# Exp5: Prefix-Last Restore Overhead
# ---------------------------------------------------------------------------
def exp5_prefix_last_restore():
    results = []
    vocab_size = 32000

    for B in [2, 4, 8, 16, 32]:
        seq_len = 512
        total_tokens = B * seq_len

        logits = torch.randn(1, total_tokens, vocab_size, device=DEVICE, dtype=torch.float32)
        labels = torch.randint(0, vocab_size, (1, total_tokens), device=DEVICE)

        # Base: compute all logprobs (log_softmax + gather)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        all_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        base_lp = all_logprobs[0].gather(1, labels[0].unsqueeze(1)).squeeze(1)
        torch.cuda.synchronize()
        base_ms = (time.perf_counter() - t0) * 1000

        # Prefix-Last Restore: for each reuser, compute 1 logprob from provider prefix-last
        n_reusers = B - 1
        prefix_len = 384  # 75% of 512
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        restored = base_lp.clone()
        provider_prefix_last_pos = prefix_len - 1
        for i in range(n_reusers):
            reuse_pos = (i + 1) * seq_len + prefix_len
            provider_logits_slice = logits[0, provider_prefix_last_pos:provider_prefix_last_pos+1, :]
            reuse_label = labels[0, reuse_pos:reuse_pos+1]
            lp = torch.nn.functional.log_softmax(provider_logits_slice, dim=-1)
            restored_value = lp.gather(1, reuse_label.unsqueeze(1))
            restored[reuse_pos] = restored_value.squeeze()
        torch.cuda.synchronize()
        restore_ms = (time.perf_counter() - t0) * 1000

        results.append({
            "B": B,
            "n_reusers": n_reusers,
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "base_logprob_ms": round(base_ms, 3),
            "prefix_last_restore_ms": round(restore_ms, 3),
            "restore_overhead_pct": round(restore_ms / base_ms * 100, 1) if base_ms > 0 else 0,
            "restore_per_reuser_ms": round(restore_ms / n_reusers, 3) if n_reusers > 0 else 0,
        })

    # Full vocab test (248320) for B=8
    vocab_full = 248320
    logits_f = torch.randn(1, 8 * 512, vocab_full, device=DEVICE, dtype=torch.float32)
    labels_f = torch.randint(0, vocab_full, (1, 8 * 512), device=DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    lp_f = torch.nn.functional.log_softmax(logits_f, dim=-1)
    base_f = lp_f[0].gather(1, labels_f[0].unsqueeze(1)).squeeze(1)
    torch.cuda.synchronize()
    full_ms = (time.perf_counter() - t0) * 1000

    # Restore for Qwen3.6 vocab
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    restored_f = base_f.clone()
    for i in range(7):
        reuse_pos = (i + 1) * 512 + 384
        p_logits = logits_f[0, 383:384, :]
        r_label = labels_f[0, reuse_pos:reuse_pos+1]
        lp_r = torch.nn.functional.log_softmax(p_logits, dim=-1)
        restored_f[reuse_pos] = lp_r.gather(1, r_label.unsqueeze(1)).squeeze()
    torch.cuda.synchronize()
    restore_f_ms = (time.perf_counter() - t0) * 1000

    results.append({
        "B": 8,
        "n_reusers": 7,
        "vocab_size": vocab_full,
        "seq_len": 512,
        "base_logprob_ms": round(full_ms, 3),
        "prefix_last_restore_ms": round(restore_f_ms, 3),
        "restore_overhead_pct": round(restore_f_ms / full_ms * 100, 1) if full_ms > 0 else 0,
        "note": "Qwen3.6 full vocab size (248320)",
    })

    return results


def main():
    print("=" * 60)
    print("Prefix Sharing Packed THD Micro-Benchmark — RTX 4090")
    print("=" * 60)

    gpu_name = torch.cuda.get_device_name(DEVICE)
    gpu_mem = torch.cuda.get_device_properties(DEVICE).total_memory / 1e9
    print(f"GPU: {gpu_name}, Memory: {gpu_mem:.1f} GB")

    warmup(DEVICE)

    all_results = {"gpu": gpu_name, "gpu_mem_gb": round(gpu_mem, 1)}

    # Exp1
    print("\n--- Exp1: KV Injection Overhead ---")
    r1 = exp1_kv_injection()
    for r in r1:
        print(f"  prefix={r['prefix_len']}: full={r['full_forward_ms']}ms, ps={r['ps_forward_ms']}ms, "
              f"savings={r['time_savings_pct']}%, speedup={r['speedup']}x")
    all_results["exp1_kv_injection"] = r1

    # Exp2
    print("\n--- Exp2: GRPO n_samples Savings ---")
    r2 = exp2_grpo_savings()
    for r in r2:
        print(f"  n={r['n_samples']}: compute={r['compute_savings_pct']}%, "
              f"time={r['time_savings_pct']}%, speedup={r['speedup']}x")
    all_results["exp2_grpo_savings"] = r2

    # Exp3
    print("\n--- Exp3: Qwen3.6 GQA 24:4 KV Injection ---")
    r3 = exp3_qwen36_gqa()
    for r in r3:
        print(f"  seq={r['seq_len']}: full={r['full_ms']}ms, ps={r['ps_ms']}ms, "
              f"inject={r['inject_overhead_ms']}ms, per-layer={r['speedup_per_full_attn_layer']}x, "
              f"total_est={r['estimated_total_speedup']}x")
    all_results["exp3_qwen36_gqa"] = r3

    # Exp4
    print("\n--- Exp4: Packed vs Unpacked Throughput ---")
    r4 = exp4_packed_vs_unpacked()
    for r in r4:
        print(f"  B={r['B']}: full={r['full_bsh_ms']}ms, ps={r['ps_packed_ms']}ms, "
              f"speedup={r['speedup']}x, token_reduction={r['token_reduction_pct']}%")
    all_results["exp4_packed_vs_unpacked"] = r4

    # Exp5
    print("\n--- Exp5: Prefix-Last Restore Overhead ---")
    r5 = exp5_prefix_last_restore()
    for r in r5:
        if "note" in r:
            print(f"  B={r['B']}: vocab={r['vocab_size']}, base={r['base_logprob_ms']}ms, "
                  f"restore={r['prefix_last_restore_ms']}ms ({r['note']})")
        else:
            print(f"  B={r['B']}: base={r['base_logprob_ms']}ms, restore={r['prefix_last_restore_ms']}ms, "
                  f"overhead={r['restore_overhead_pct']}%")
    all_results["exp5_prefix_last_restore"] = r5

    # Save
    with open("prefix_sharing_packed_thd_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to prefix_sharing_packed_thd_results.json")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best_grpo = max(r2, key=lambda x: x["speedup"])
    print(f"GRPO best: n={best_grpo['n_samples']}, speedup={best_grpo['speedup']}x, savings={best_grpo['time_savings_pct']}%")
    best_qwen = r3[-1]
    print(f"Qwen3.6-27B: seq=2048, full-attn speedup={best_qwen['speedup_per_full_attn_layer']}x, "
          f"estimated total={best_qwen['estimated_total_speedup']}x (25% layers benefit)")
    best_restore = max([r for r in r5 if "note" not in r], key=lambda x: x["B"])
    print(f"Prefix-Last Restore: B={best_restore['B']}, overhead={best_restore['restore_overhead_pct']}% of logprob compute")


if __name__ == "__main__":
    main()