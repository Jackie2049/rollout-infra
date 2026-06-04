#!/usr/bin/env python3
"""RTX 4090 Attention Variant Benchmark
=========================================
Compares different attention mechanisms:
1. MHA (Multi-Head Attention) — standard baseline
2. GQA (Grouped-Query Attention) — KV head reduction
3. MQA (Multi-Query Attention) — single KV head
4. MLA (Multi-head Latent Attention) — DeepSeek-V2 style compression
"""

import torch
import torch.nn.functional as F
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
    print("RTX 4090 Attention Variant Benchmark")
    print("=" * 60)

    props = torch.cuda.get_device_properties(0)
    print(f"\n  GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")

    results = {}

    # Common params (7B-like model)
    H = 4096      # hidden dim
    n_heads = 32   # query heads
    head_dim = H // n_heads  # 128
    n_layers = 32
    batch = 8

    # ===== Exp 1: GQA vs MHA vs MQA — Decode =====
    print("\n" + "=" * 60)
    print("Experiment 1: Attention Variant — Decode (B=8)")
    print("=" * 60)

    seq_lens = [512, 1024, 2048, 4096]
    kv_configs = {
        "MHA (32 KV heads)": 32,
        "GQA-8 (8 KV heads)": 8,
        "GQA-4 (4 KV heads)": 4,
        "MQA (1 KV head)": 1,
    }

    print(f"\n  {'Config':>22}", end="")
    for sl in seq_lens:
        print(f"  {'S='+str(sl):>10}", end="")
    print(f"  {'KV/B(bytes)':>12}")
    print("  " + "-" * 80)

    for name, n_kv_heads in kv_configs.items():
        print(f"  {name:>22}", end="")
        for sl in seq_lens:
            # Simulate decode: batch x n_heads x 1 x head_dim @ n_kv_heads x sl x head_dim
            # For GQA: Q has n_heads, K/V have n_kv_heads, each KV head serves n_heads/n_kv_heads Q heads
            q = torch.randn(batch * n_heads, 1, head_dim, device='cuda', dtype=torch.float16)

            if n_kv_heads == n_heads:
                # MHA: standard
                k = torch.randn(batch * n_kv_heads, sl, head_dim, device='cuda', dtype=torch.float16)
                v = torch.randn(batch * n_kv_heads, sl, head_dim, device='cuda', dtype=torch.float16)
            else:
                # GQA/MQA: fewer KV heads, repeat for Q heads
                k = torch.randn(batch * n_kv_heads, sl, head_dim, device='cuda', dtype=torch.float16)
                v = torch.randn(batch * n_kv_heads, sl, head_dim, device='cuda', dtype=torch.float16)

            def attn_fn(q=q, k=k, v=v, n_qh=n_heads, n_kvh=n_kv_heads, bs=batch):
                # Reshape for grouped attention
                q = q.view(bs, n_qh, 1, head_dim)
                k = k.view(bs, n_kvh, sl, head_dim)
                v = v.view(bs, n_kvh, sl, head_dim)
                if n_kvh < n_qh:
                    # Expand KV heads
                    n_rep = n_qh // n_kvh
                    k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bs, n_qh, sl, head_dim)
                    v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bs, n_qh, sl, head_dim)
                # Scaled dot-product attention
                out = F.scaled_dot_product_attention(q, k, v)
                return out

            ms = benchmark(attn_fn)
            tflops = 2 * batch * n_heads * sl * head_dim * 2 / (ms * 1e-3 * 1e12)
            print(f"  {ms:>8.2f}ms", end="")

            del q, k, v
            torch.cuda.empty_cache()

        # KV cache per batch entry: 2 * n_kv_heads * head_dim * seq_len * 2 bytes (FP16)
        # But we report per-token for easier comparison
        kv_per_tok = 2 * n_kv_heads * head_dim * 2  # bytes
        print(f"  {kv_per_tok:>11,}")

        results[f"exp1_{name}"] = {"kv_per_tok_bytes": kv_per_tok, "n_kv_heads": n_kv_heads}

    # ===== Exp 2: GQA vs MHA — Prefill =====
    print("\n" + "=" * 60)
    print("Experiment 2: Attention Variant — Prefill")
    print("=" * 60)

    print(f"\n  {'Config':>22}", end="")
    for sl in [256, 512, 1024, 2048]:
        print(f"  {'S='+str(sl):>10}", end="")
    print()
    print("  " + "-" * 65)

    for name, n_kv_heads in kv_configs.items():
        print(f"  {name:>22}", end="")
        for sl in [256, 512, 1024, 2048]:
            q = torch.randn(batch, n_heads, sl, head_dim, device='cuda', dtype=torch.float16)
            k = torch.randn(batch, n_kv_heads, sl, head_dim, device='cuda', dtype=torch.float16)
            v = torch.randn(batch, n_kv_heads, sl, head_dim, device='cuda', dtype=torch.float16)

            def attn_fn(q=q, k=k, v=v, n_qh=n_heads, n_kvh=n_kv_heads, bs=batch, s=sl):
                q = q.view(bs, n_qh, s, head_dim)
                k = k.view(bs, n_kvh, s, head_dim)
                v = v.view(bs, n_kvh, s, head_dim)
                if n_kvh < n_qh:
                    n_rep = n_qh // n_kvh
                    k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bs, n_qh, s, head_dim)
                    v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bs, n_qh, s, head_dim)
                return F.scaled_dot_product_attention(q, k, v)

            ms = benchmark(attn_fn)
            # Prefill FLOPS: 2 * B * n_heads * S * (S * head_dim) * 2 (Q@K + attn@V)
            # But SDPA uses FlashAttention so FLOPS = 2*B*n_heads*S*S*head_dim + 2*B*n_heads*S*head_dim
            flops = 2 * batch * n_heads * sl * sl * head_dim  # dominant term
            tflops = flops / (ms * 1e-3 * 1e12)
            print(f"  {tflops:>8.1f}T", end="")

            results[f"exp2_{name}_s{sl}"] = {
                "ms": round(ms, 3), "tflops": round(tflops, 1),
            }

            del q, k, v
            torch.cuda.empty_cache()
        print()

    # ===== Exp 3: MLA (Latent Attention) Compression =====
    print("\n" + "=" * 60)
    print("Experiment 3: MLA Compression Simulation")
    print("=" * 60)

    # MLA: compress KV from (n_heads, head_dim) to latent_dim
    # KV = n_heads * head_dim = 32 * 128 = 4096 → compress to latent_dim
    latent_dims = [256, 512, 1024, 2048]  # MLA latent sizes

    print(f"\n  MHA KV per token: {2 * 32 * 128 * 2:,} bytes = {(2 * 32 * 128 * 2)/1024:.1f} KB")
    print(f"  Compression ratios:")
    print(f"  {'LatentDim':>12} {'MLA KB/tok':>12} {'Ratio':>8} {'Decode ms':>12}")
    print("  " + "-" * 50)

    sl = 2048
    for latent_dim in latent_dims:
        # MLA: KV compressed to latent_dim, upsampled on-the-fly
        # Simulate: store latent (smaller), then upsample via matmul
        kv_latent = torch.randn(batch, sl, latent_dim, device='cuda', dtype=torch.float16)
        up_proj = torch.randn(latent_dim, 2 * n_heads * head_dim, device='cuda', dtype=torch.float16)

        def mla_decode(kv_latent=kv_latent, up_proj=up_proj):
            # Upsample latent to full KV
            kv_full = kv_latent @ up_proj  # [B, S, 2*n_heads*head_dim]
            return kv_full

        ms = benchmark(mla_decode)
        mla_kb = latent_dim * 2 / 1024  # 2 bytes per FP16
        mha_kb = 2 * n_heads * head_dim * 2 / 1024
        ratio = mha_kb / mla_kb

        print(f"  {latent_dim:>12} {mla_kb:>10.1f}KB {ratio:>7.1f}x {ms:>10.3f}ms")

        results[f"exp3_mla_d{latent_dim}"] = {
            "latent_dim": latent_dim, "mla_kb": round(mla_kb, 1),
            "compression_ratio": round(ratio, 1), "decode_ms": round(ms, 3),
        }

        del kv_latent, up_proj
        torch.cuda.empty_cache()

    # ===== Exp 4: KV Cache Memory Analysis =====
    print("\n" + "=" * 60)
    print("Experiment 4: KV Cache Memory — 7B Model on RTX 4090")
    print("=" * 60)

    total_vram = 24.0  # GB
    model_size = 14.0  # GB (7B FP16)
    kv_budget = (total_vram * 0.8 - model_size) * 1e9  # bytes

    configs = [
        ("MHA-32", 32, head_dim, 2),
        ("GQA-8", 8, head_dim, 2),
        ("GQA-4", 4, head_dim, 2),
        ("MQA", 1, head_dim, 2),
        ("MLA-256", 1, 256, 2),    # MLA stores latent, no per-head
        ("MLA-512", 1, 512, 2),
    ]

    print(f"\n  Model: 7B FP16 ({model_size} GB), KV budget: {kv_budget/1e9:.1f} GB")
    print(f"  {'Config':>10} {'KV/tok(B)':>10} {'Seq=512':>10} {'Seq=2K':>10} {'Seq=8K':>10}")
    print("  " + "-" * 55)

    for name, n_kv, d_kv, bpe in configs:
        # MLA: KV per token = 2 * layers * latent_dim * bytes
        # Standard: KV per token = 2 * layers * n_kv_heads * head_dim * bytes
        if name.startswith("MLA"):
            kv_per_tok = 2 * n_layers * d_kv * bpe
        else:
            kv_per_tok = 2 * n_layers * n_kv * d_kv * bpe

        for sl in [512, 2048, 8192]:
            max_reqs = int(kv_budget / (kv_per_tok * sl))
            if name == configs[0][0] and sl == 512:
                print(f"  {name:>10} {kv_per_tok:>10,} {max_reqs:>10}", end="")
            elif sl == 512:
                print(f"  {name:>10} {kv_per_tok:>10,} {max_reqs:>10}", end="")
            else:
                print(f" {max_reqs:>10}", end="")
        print()

    # ===== Exp 5: FlashAttention vs Manual Attention =====
    print("\n" + "=" * 60)
    print("Experiment 5: SDPA (FlashAttention) vs Manual Attention")
    print("=" * 60)

    for sl in [512, 1024, 2048, 4096]:
        q = torch.randn(batch, n_heads, sl, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch, n_heads, sl, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch, n_heads, sl, head_dim, device='cuda', dtype=torch.float16)

        # SDPA (FlashAttention)
        def sdpa_fn(q=q, k=k, v=v):
            return F.scaled_dot_product_attention(q, k, v)

        ms_sdpa = benchmark(sdpa_fn)

        # Manual attention (for comparison, only safe for smaller seq)
        def manual_fn(q=q, k=k, v=v):
            scale = 1.0 / math.sqrt(head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            return torch.matmul(attn, v)

        if sl <= 2048:
            ms_manual = benchmark(manual_fn)
            speedup = ms_manual / ms_sdpa
            print(f"  S={sl:>5}: SDPA={ms_sdpa:.3f}ms, Manual={ms_manual:.3f}ms, Speedup={speedup:.1f}x")
        else:
            print(f"  S={sl:>5}: SDPA={ms_sdpa:.3f}ms, Manual=OOM")

        del q, k, v
        torch.cuda.empty_cache()

    with open("attn_variant_4090_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to attn_variant_4090_results.json")


if __name__ == "__main__":
    main()
