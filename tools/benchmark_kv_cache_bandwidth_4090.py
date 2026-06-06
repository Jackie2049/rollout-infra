#!/usr/bin/env python3
"""KV Cache Memory Bandwidth Benchmark on RTX 4090
===================================================

Measures the memory bandwidth for reading KV caches at different
configurations (MHA/GQA/MQA/MLA) during decode. Since decode is
memory-bound, KV cache read bandwidth directly determines throughput.

Key question: How does KV cache size affect decode throughput?

Tests:
1. KV cache read bandwidth (memcpy-like) vs KV size
2. Attention decode: actual KV read + compute vs pure compute
3. GQA/MQA KV expansion overhead (expand + SDPA)
4. MLA latent upsample bandwidth vs full KV read
5. KV cache per-token bandwidth at different seq lengths

Usage:
  python benchmark_kv_cache_bandwidth_4090.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
import time
import json

def benchmark_cuda_events(fn, warmup=10, repeat=100):
    """Benchmark using CUDA events for accurate GPU timing."""
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
    device = 'cuda:0'
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"KV Cache Memory Bandwidth Benchmark: {gpu_name}, {total_mem:.1f} GB")

    results = []

    # Common params (7B-like model)
    H = 4096       # hidden dim
    n_heads = 32    # query heads
    head_dim = H // n_heads  # 128
    n_layers = 32

    # ================================================================
    # Experiment 1: Raw HBM Read Bandwidth — Memcpy Benchmark
    # ================================================================
    print("\n" + "=" * 60)
    print("Experiment 1: HBM Read Bandwidth — Data Size Scaling")
    print("=" * 60)

    # Measure pure memory read bandwidth at different sizes
    # This gives us the baseline for KV cache read speed
    sizes_mb = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    print(f"\n  {'Size(MB)':>10} {'Read(ms)':>10} {'BW(GB/s)':>10} {'%Peak':>8}")
    print("  " + "-" * 40)

    # Known peak HBM BW from previous benchmark: ~877 GB/s
    peak_bw = 877.0  # GB/s

    hbm_results = []
    for size_mb in sizes_mb:
        n_elements = int(size_mb * 1e6 / 2)  # FP16 = 2 bytes
        data = torch.randn(n_elements, device=device, dtype=torch.float16)
        actual_mb = data.nelement() * 2 / 1e6

        def read_fn(d=data):
            # Simple read: sum reduction forces reading all elements
            return d.sum()

        ms = benchmark_cuda_events(read_fn)
        bw = actual_mb / ms  # MB/ms = GB/s
        pct = bw / peak_bw * 100

        print(f"  {actual_mb:>9.1f} {ms:>9.4f} {bw:>9.1f} {pct:>7.1f}%")
        hbm_results.append({
            "size_mb": round(actual_mb, 1), "ms": round(ms, 4),
            "bw_gb_s": round(bw, 1), "pct_peak": round(pct, 1),
        })

        del data
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    results.append({"experiment": "hbm_read_bandwidth", "peak_bw": peak_bw, "data": hbm_results})

    # ================================================================
    # Experiment 2: KV Cache Read — Different Attention Variants
    # ================================================================
    print("\n" + "=" * 60)
    print("Experiment 2: KV Cache Read — MHA/GQA/MQA/MLA at Decode")
    print("=" * 60)

    # During decode, we read the full KV cache for each layer
    # KV per layer per batch = 2 * n_kv_heads * head_dim * seq_len * batch * 2bytes
    # For MHA:   2 * 32 * 128 * S * B * 2 = 8192 * S * B bytes/layer
    # For GQA-8: 2 * 8 * 128 * S * B * 2 = 2048 * S * B bytes/layer
    # For MQA:   2 * 1 * 128 * S * B * 2 = 256 * S * B bytes/layer
    # For MLA-256: 2 * 256 * S * B * 2 = 512 * S * B bytes/layer (latent only)

    seq_len = 2048
    batch = 8  # Use smaller batch to avoid OOM with per-layer measurement

    kv_configs = {
        "MHA-32": 32,
        "GQA-8": 8,
        "GQA-4": 4,
        "MQA-1": 1,
    }

    print(f"\n  B={batch}, S={seq_len} (per-layer measurement)")
    print(f"  {'Config':>10} {'KV/layer(MB)':>14} {'Read ms':>10} {'BW(GB/s)':>10}")
    print("  " + "-" * 50)

    kv_read_results = []
    for name, n_kv in kv_configs.items():
        kv_per_layer_bytes = 2 * n_kv * head_dim * seq_len * batch * 2
        kv_layer_mb = kv_per_layer_bytes / 1e6

        # Allocate per-layer KV cache (K + V)
        k = torch.randn(batch, n_kv, seq_len, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, n_kv, seq_len, head_dim, device=device, dtype=torch.float16)
        actual_mb = (k.nelement() + v.nelement()) * 2 / 1e6

        def kv_read_fn(k=k, v=v):
            return k.sum() + v.sum()

        ms = benchmark_cuda_events(kv_read_fn)
        bw = actual_mb / ms

        print(f"  {name:>10} {actual_mb:>13.1f} {ms:>9.4f} {bw:>9.1f}")
        kv_read_results.append({
            "name": name, "n_kv_heads": n_kv,
            "kv_per_layer_mb": round(kv_layer_mb, 1),
            "actual_mb": round(actual_mb, 1),
            "read_ms": round(ms, 4),
            "bw_gb_s": round(bw, 1),
        })

        del k, v
        torch.cuda.empty_cache()

    # MLA latent read (per-layer)
    for latent_dim in [256, 512]:
        name = f"MLA-{latent_dim}"
        kv_per_layer_bytes = 2 * latent_dim * seq_len * batch * 2
        kv_layer_mb = kv_per_layer_bytes / 1e6

        kv_latent = torch.randn(batch, seq_len, latent_dim, device=device, dtype=torch.float16)
        actual_mb = kv_latent.nelement() * 2 / 1e6

        def kv_read_fn(kv=kv_latent):
            return kv.sum()

        ms = benchmark_cuda_events(kv_read_fn)
        bw = actual_mb / ms

        print(f"  {name:>10} {actual_mb:>13.1f} {ms:>9.4f} {bw:>9.1f}")
        kv_read_results.append({
            "name": name, "latent_dim": latent_dim,
            "kv_per_layer_mb": round(kv_layer_mb, 1),
            "actual_mb": round(actual_mb, 1),
            "read_ms": round(ms, 4),
            "bw_gb_s": round(bw, 1),
        })

        del kv_latent
        torch.cuda.empty_cache()

    results.append({"experiment": "kv_cache_read", "batch": batch, "seq_len": seq_len, "data": kv_read_results})

    # ================================================================
    # Experiment 3: KV Read vs Batch Size — Memory-Bound Scaling
    # ================================================================
    print("\n" + "=" * 60)
    print("Experiment 3: KV Read Time vs Batch Size (MHA, S=2048)")
    print("=" * 60)

    seq_len = 2048
    print(f"\n  {'Batch':>8} {'KV/layer MB':>12} {'Read ms':>10} {'BW GB/s':>10}")
    print("  " + "-" * 45)

    batch_scale_results = []
    for B in [1, 4, 8, 16, 32, 64, 128, 256]:
        kv_layer_bytes = 2 * n_heads * head_dim * seq_len * B * 2

        if kv_layer_bytes / 1e9 > 2:
            continue

        k = torch.randn(B, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(B, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        actual_mb = (k.nelement() + v.nelement()) * 2 / 1e6

        def read_fn(k=k, v=v):
            return k.sum() + v.sum()

        ms = benchmark_cuda_events(read_fn)
        bw = actual_mb / ms

        print(f"  {B:>8} {actual_mb:>11.1f} {ms:>9.4f} {bw:>9.1f}")

        batch_scale_results.append({
            "batch": B, "layer_mb": round(actual_mb, 1),
            "read_ms": round(ms, 4), "bw_gb_s": round(bw, 1),
        })

        del k, v
        torch.cuda.empty_cache()

    results.append({"experiment": "kv_batch_scaling", "seq_len": seq_len, "data": batch_scale_results})

    # ================================================================
    # Experiment 4: KV Read vs Seq Length — Long Context Impact
    # ================================================================
    print("\n" + "=" * 60)
    print("Experiment 4: KV Read Time vs Seq Length (MHA-32, per-layer)")
    print("=" * 60)

    batch = 8  # Smaller batch to fit longer seq
    print(f"\n  B={batch}")
    print(f"  {'SeqLen':>8} {'KV/layer MB':>12} {'Read ms':>10} {'BW GB/s':>10}")
    print("  " + "-" * 45)

    seq_scale_results = []
    for S in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        kv_layer_bytes = 2 * n_heads * head_dim * S * batch * 2

        if kv_layer_bytes / 1e9 > 2:
            continue

        k = torch.randn(batch, n_heads, S, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, n_heads, S, head_dim, device=device, dtype=torch.float16)
        actual_mb = (k.nelement() + v.nelement()) * 2 / 1e6

        def read_fn(k=k, v=v):
            return k.sum() + v.sum()

        ms = benchmark_cuda_events(read_fn)
        bw = actual_mb / ms

        print(f"  {S:>8} {actual_mb:>11.1f} {ms:>9.4f} {bw:>9.1f}")

        seq_scale_results.append({
            "seq_len": S, "layer_mb": round(actual_mb, 1),
            "read_ms": round(ms, 4), "bw_gb_s": round(bw, 1),
        })

        del k, v
        torch.cuda.empty_cache()

    results.append({"experiment": "kv_seq_scaling", "batch": batch, "data": seq_scale_results})

    # ================================================================
    # Experiment 5: GQA/MQA KV Expansion Overhead
    # ================================================================
    print("\n" + "=" * 60)
    print("Experiment 5: GQA/MQA KV Expansion Overhead (B=8)")
    print("=" * 60)

    seq_len = 2048
    batch = 8

    gqa_results = []
    for name, n_kv in [("MHA-32", 32), ("GQA-8", 8), ("GQA-4", 4), ("MQA-1", 1)]:
        # Step 1: Read compressed KV
        k_comp = torch.randn(batch, n_kv, seq_len, head_dim, device=device, dtype=torch.float16)
        v_comp = torch.randn(batch, n_kv, seq_len, head_dim, device=device, dtype=torch.float16)

        if n_kv < n_heads:
            # Step 2: Expand KV heads
            n_rep = n_heads // n_kv

            def expand_fn(k=k_comp, v=v_comp, nr=n_rep):
                k_exp = k.unsqueeze(2).expand(-1, -1, nr, -1, -1).reshape(batch, n_heads, seq_len, head_dim)
                v_exp = v.unsqueeze(2).expand(-1, -1, nr, -1, -1).reshape(batch, n_heads, seq_len, head_dim)
                return k_exp, v_exp

            expand_ms = benchmark_cuda_events(expand_fn)

            # Step 3: Attention with expanded KV
            q = torch.randn(batch, n_heads, 1, head_dim, device=device, dtype=torch.float16)

            def full_decode_fn(q=q, k=k_comp, v=v_comp, nr=n_rep):
                k_exp = k.unsqueeze(2).expand(-1, -1, nr, -1, -1).reshape(batch, n_heads, seq_len, head_dim)
                v_exp = v.unsqueeze(2).expand(-1, -1, nr, -1, -1).reshape(batch, n_heads, seq_len, head_dim)
                return F.scaled_dot_product_attention(q, k_exp, v_exp)

            total_ms = benchmark_cuda_events(full_decode_fn)
            attn_ms = total_ms - expand_ms
            expand_pct = expand_ms / total_ms * 100

            print(f"  {name:>10} {expand_ms:>9.4f} {attn_ms:>9.4f} {total_ms:>9.4f} {expand_pct:>7.1f}%")
        else:
            # MHA: no expansion needed
            q = torch.randn(batch, n_heads, 1, head_dim, device=device, dtype=torch.float16)

            def mha_fn(q=q, k=k_comp, v=v_comp):
                return F.scaled_dot_product_attention(q, k, v)

            total_ms = benchmark_cuda_events(mha_fn)

            print(f"  {name:>10} {0:>9} {total_ms:>9.4f} {total_ms:>9.4f} {0:>7}%")

        gqa_results.append({
            "name": name, "n_kv_heads": n_kv,
            "expand_ms": round(expand_ms if n_kv < n_heads else 0, 4),
            "attn_ms": round(attn_ms if n_kv < n_heads else total_ms, 4),
            "total_ms": round(total_ms, 4),
            "expand_pct": round(expand_pct if n_kv < n_heads else 0, 1),
        })

        del k_comp, v_comp
        if 'q' in dir():
            del q
        torch.cuda.empty_cache()

    results.append({"experiment": "gqa_expand_overhead", "batch": batch, "seq_len": seq_len, "data": gqa_results})

    # ================================================================
    # Experiment 6: MLA Upsample vs Full KV Read
    # ================================================================
    print("\n" + "=" * 60)
    print("Experiment 6: MLA Latent Upsample vs Full KV Read")
    print("=" * 60)

    seq_len = 2048
    batch = 8

    print(f"\n  B={batch}, S={seq_len}")
    print(f"  {'Method':>15} {'KB/tok':>10} {'ms':>10} {'BW GB/s':>10} {'vs MHA':>10}")
    print("  " + "-" * 60)

    # MHA: full KV read
    kv_mha = torch.randn(batch, 32, seq_len, head_dim, device=device, dtype=torch.float16)
    def mha_read_fn(kv=kv_mha):
        return kv.sum()
    mha_ms = benchmark_cuda_events(mha_read_fn)
    mha_bw = kv_mha.nelement() * 2 / 1e6 / mha_ms
    mha_kb_per_tok = 2 * 32 * head_dim * 2 / 1024

    print(f"  {'MHA-32 read':>15} {mha_kb_per_tok:>9.1f} {mha_ms:>9.4f} {mha_bw:>9.1f} {'1.0x':>10}")

    mla_up_results = []
    mla_up_results.append({
        "name": "MHA-32_read", "kb_per_tok": round(mha_kb_per_tok, 1),
        "ms": round(mha_ms, 4), "bw_gb_s": round(mha_bw, 1), "ratio": 1.0,
    })

    del kv_mha

    # MLA: latent read + upsample
    for latent_dim in [256, 512, 1024]:
        # Read latent KV (smaller)
        kv_latent = torch.randn(batch, seq_len, latent_dim, device=device, dtype=torch.float16)
        # Upsample: latent → full KV
        up_proj = torch.randn(latent_dim, 2 * n_heads * head_dim, device=device, dtype=torch.float16)

        def mla_upsample_fn(kv=kv_latent, proj=up_proj):
            # Read latent + upsample via matmul
            return kv_latent @ up_proj

        mla_ms = benchmark_cuda_events(mla_upsample_fn)
        mla_kb = latent_dim * 2 / 1024  # per-token KB
        ratio = mha_kb_per_tok / mla_kb

        # Calculate effective bandwidth (latent data volume)
        latent_bytes = kv_latent.nelement() * 2
        mla_bw = latent_bytes / 1e6 / mla_ms

        print(f"  {'MLA-%d upsample' % latent_dim:>15} {mla_kb:>9.1f} {mla_ms:>9.4f} {mla_bw:>9.1f} {ratio:>9.1f}x")

        mla_up_results.append({
            "name": f"MLA-{latent_dim}_upsample",
            "latent_dim": latent_dim,
            "kb_per_tok": round(mla_kb, 1),
            "ms": round(mla_ms, 4),
            "bw_gb_s": round(mla_bw, 1),
            "ratio": round(ratio, 1),
        })

        del kv_latent, up_proj
        torch.cuda.empty_cache()

    results.append({"experiment": "mla_upsample_vs_mha", "batch": batch, "seq_len": seq_len, "data": mla_up_results})

    # ================================================================
    # Summary: Decode Throughput Model
    # ================================================================
    print("\n" + "=" * 60)
    print("Summary: Decode Throughput Model (7B, B=32)")
    print("=" * 60)

    # 7B decode per-step data volume breakdown:
    # 1. Model weights: 14GB (FP16) — read once per step
    # 2. KV cache per step: varies by attention variant
    # 3. Activations: tiny (~batch * hidden_dim * 2bytes)

    model_weight_mb = 14000  # 7B FP16 = 14GB
    batch = 32
    peak_hbm = 877.0  # GB/s from previous benchmark

    print(f"\n  Peak HBM: {peak_hbm} GB/s, Model weight: {model_weight_mb} MB")
    print(f"\n  {'Config':>10} {'KV MB/step':>12} {'Total MB':>10} {'Est ms':>10} {'tok/s':>10}")
    print("  " + "-" * 55)

    summary_data = []
    for name, n_kv in [("MHA-32", 32), ("GQA-8", 8), ("GQA-4", 4), ("MQA-1", 1)]:
        kv_per_step = 2 * n_kv * head_dim * 2048 * batch * 2 * n_layers / 1e6  # MB per step
        total = model_weight_mb + kv_per_step
        # est_ms = total_MB / (peak_GB/s * 1000 MB/GB) * 1000 ms/s
        est_ms = total / (peak_hbm * 1000) * 1000  # = total / peak_hbm directly
        throughput = batch / (est_ms / 1000)

        print(f"  {name:>10} {kv_per_step:>11.1f} {total:>9.1f} {est_ms:>9.2f} {throughput:>9.0f}")
        summary_data.append({
            "name": name, "kv_per_step_mb": round(kv_per_step, 1),
            "total_mb": round(total, 1), "est_ms": round(est_ms, 2),
            "throughput_tok_s": round(throughput, 0),
        })

    for latent_dim in [256, 512]:
        name = f"MLA-{latent_dim}"
        kv_per_step = 2 * latent_dim * 2048 * batch * 2 * n_layers / 1e6
        total = model_weight_mb + kv_per_step
        est_ms = total / 877
        throughput = batch / (est_ms / 1000)
        print(f"  {name:>10} {kv_per_step:>11.1f} {total:>9.1f} {est_ms:>9.2f} {throughput:>9.0f}")
        summary_data.append({
            "name": name, "kv_per_step_mb": round(kv_per_step, 1),
            "total_mb": round(total, 1), "est_ms": round(est_ms, 2),
            "throughput_tok_s": round(throughput, 0),
        })

    results.append({"experiment": "decode_throughput_model", "peak_hbm": peak_hbm, "data": summary_data})

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kv_cache_bandwidth_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()