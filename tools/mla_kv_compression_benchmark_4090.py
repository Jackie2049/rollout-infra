#!/usr/bin/env python3
"""MLA (Multi-head Latent Attention) KV Compression Simulation — RTX 4090
=========================================================================
DeepSeek-V3's key innovation: compress KV cache via low-rank projection.
Original: n_heads × d_head per token → MLA: d_latent per token (much smaller!)
Plus RoPE decoupled: rope_head separately stored.

Key measurements:
1. MLA vs MHA vs GQA KV memory comparison
2. MLA projection latency (down-project + up-project)
3. MLA decode throughput (attention with compressed KV)
4. Upsample matmul overhead analysis
5. RoPE decoupled attention accuracy verification
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


def run_experiment():
    device = "cuda"
    results = {}

    # DeepSeek-V3 MLA parameters (approximate)
    # d_model=7168, n_heads=128, d_head=256, d_latent=512 (KV compressed)
    # rope_d_head=64 (decoupled RoPE)
    # Plus simulation with smaller configs for RTX 4090 feasibility

    configs = {
        # Smaller configs for practical benchmarking
        "GQA_4": {"d": 256, "n_heads": 8, "n_kv_heads": 4, "d_head": 32, "d_latent": 64, "rope_d": 16},
        "GQA_8": {"d": 512, "n_heads": 16, "n_kv_heads": 8, "d_head": 32, "d_latent": 128, "rope_d": 16},
        # DeepSeek-V3 style (scaled down)
        "DS_V3_style": {"d": 1024, "n_heads": 32, "n_kv_heads": 32, "d_head": 256, "d_latent": 512, "rope_d": 64},
    }

    # ========================
    # Exp 1: KV Memory Comparison
    # ========================
    print("=== Exp 1: KV Memory Comparison (MHA vs GQA vs MLA) ===")

    S_values = [512, 1024, 2048, 4096, 8192, 16384]
    B = 1  # Single request for clarity

    kv_memory = {}
    for label, cfg in configs.items():
        n_heads = cfg["n_heads"]
        n_kv_heads = cfg["n_kv_heads"]
        d_head = cfg["d_head"]
        d_latent = cfg["d_latent"]
        rope_d = cfg["rope_d"]

        for S in S_values:
            # MHA: n_heads × d_head × 2 (K+V) per token
            mha_kv_bytes = 2 * n_heads * d_head * S * 2  # FP16, 2 bytes

            # GQA: n_kv_heads × d_head × 2 per token
            gqa_kv_bytes = 2 * n_kv_heads * d_head * S * 2  # FP16

            # MLA: d_latent × 2 (compressed K+V) + n_heads × rope_d × 2 (RoPE decoupled) per token
            # In practice: compressed_latent + rope_latent
            mla_compressed_bytes = 2 * d_latent * S * 2  # compressed KV (FP16)
            mla_rope_bytes = 2 * n_heads * rope_d * S * 2  # RoPE head (FP16)
            mla_kv_bytes = mla_compressed_bytes + mla_rope_bytes

            compression_ratio_mha = mha_kv_bytes / mla_kv_bytes
            compression_ratio_gqa = gqa_kv_bytes / mla_kv_bytes

            kv_memory[f"{label}_S{S}"] = {
                "S": S, "config": label,
                "mha_kv_MB": mha_kv_bytes / 1e6,
                "gqa_kv_MB": gqa_kv_bytes / 1e6,
                "mla_kv_MB": mla_kv_bytes / 1e6,
                "mla_compressed_MB": mla_compressed_bytes / 1e6,
                "mla_rope_MB": mla_rope_bytes / 1e6,
                "compression_vs_mha": compression_ratio_mha,
                "compression_vs_gqa": compression_ratio_gqa,
            }

            print(f"  {label} S={S}: MHA={mha_kv_bytes/1e6:.1f}MB "
                  f"GQA={gqa_kv_bytes/1e6:.1f}MB "
                  f"MLA={mla_kv_bytes/1e6:.1f}MB "
                  f"(compressed={mla_compressed_bytes/1e6:.1f}MB+rope={mla_rope_bytes/1e6:.1f}MB) "
                  f"vs_MHA={compression_ratio_mha:.1f}x vs_GQA={compression_ratio_gqa:.1f}x")

    results["kv_memory"] = kv_memory

    # ========================
    # Exp 2: MLA Projection Latency
    # ========================
    print("\n=== Exp 2: MLA Projection Latency ===")

    for label, cfg in configs.items():
        d = cfg["d"]
        d_latent = cfg["d_latent"]
        n_heads = cfg["n_heads"]
        d_head = cfg["d_head"]
        rope_d = cfg["rope_d"]

        # Down-projection: d_model → d_latent (compress KV)
        W_down_k = torch.randn(d, d_latent, device=device, dtype=torch.float16)
        W_down_v = torch.randn(d, d_latent, device=device, dtype=torch.float16)

        # Up-projection: d_latent → n_heads × d_head (decompress for attention)
        # MLA: separate K and V up-projections
        W_up_k = torch.randn(d_latent, n_heads * d_head, device=device, dtype=torch.float16)
        W_up_v = torch.randn(d_latent, n_heads * d_head, device=device, dtype=torch.float16)

        # RoPE projection: d_model → n_heads × rope_d
        W_rope = torch.randn(d, n_heads * rope_d, device=device, dtype=torch.float16)

        for S in [1, 128, 512, 2048]:
            x_input = torch.randn(1, S, d, device=device, dtype=torch.float16)

            # Down-project K (compress)
            down_k_res = benchmark_fn(
                f"{label}_down_k_S{S}", lambda: x_input @ W_down_k
            )
            # Down-project V (compress)
            down_v_res = benchmark_fn(
                f"{label}_down_v_S{S}", lambda: x_input @ W_down_v
            )
            # Up-project K (decompress)
            compressed_k = x_input @ W_down_k  # Pre-compute for up-proj benchmark
            up_k_res = benchmark_fn(
                f"{label}_up_k_S{S}", lambda: compressed_k @ W_up_k
            )
            # Up-project V (decompress)
            compressed_v = x_input @ W_down_v
            up_v_res = benchmark_fn(
                f"{label}_up_v_S{S}", lambda: compressed_v @ W_up_v
            )
            # RoPE projection
            rope_res = benchmark_fn(
                f"{label}_rope_S{S}", lambda: x_input @ W_rope
            )

            proj_total_ms = down_k_res["median_ms"] + down_v_res["median_ms"] + \
                           up_k_res["median_ms"] + up_v_res["median_ms"] + rope_res["median_ms"]

            results[f"projection_{label}_S{S}"] = {
                "down_k_ms": down_k_res["median_ms"],
                "down_v_ms": down_v_res["median_ms"],
                "up_k_ms": up_k_res["median_ms"],
                "up_v_ms": up_v_res["median_ms"],
                "rope_ms": rope_res["median_ms"],
                "total_ms": proj_total_ms,
                "S": S,
            }

            print(f"  {label} S={S}: down_K={down_k_res['median_ms']:.3f}ms "
                  f"down_V={down_v_res['median_ms']:.3f}ms "
                  f"up_K={up_k_res['median_ms']:.3f}ms "
                  f"up_V={up_v_res['median_ms']:.3f}ms "
                  f"rope={rope_res['median_ms']:.3f}ms "
                  f"total={proj_total_ms:.3f}ms")

    # ========================
    # Exp 3: MLA Decode Throughput
    # ========================
    print("\n=== Exp 3: MLA Decode Throughput (MHA vs GQA vs MLA) ===")

    for label, cfg in configs.items():
        d = cfg["d"]
        n_heads = cfg["n_heads"]
        n_kv_heads = cfg["n_kv_heads"]
        d_head = cfg["d_head"]
        d_latent = cfg["d_latent"]
        rope_d = cfg["rope_d"]
        scale = 1.0 / math.sqrt(d_head)

        B_values = [1, 8, 32, 64]  # Limit to B=64 to avoid OOM on large configs
        S = 2048  # Typical context length

        decode_results = {}
        for B in B_values:
            # === MHA decode ===
            Q_mha = torch.randn(B, n_heads, 1, d_head, device=device, dtype=torch.float16)
            K_mha = torch.randn(B, n_heads, S, d_head, device=device, dtype=torch.float16)
            V_mha = torch.randn(B, n_heads, S, d_head, device=device, dtype=torch.float16)

            mha_attn = benchmark_fn(
                f"MHA_B{B}", lambda: torch.softmax(Q_mha @ K_mha.transpose(-2, -1) * scale, dim=-1) @ V_mha
            )

            # === GQA decode ===
            Q_gqa = torch.randn(B, n_heads, 1, d_head, device=device, dtype=torch.float16)
            K_gqa = torch.randn(B, n_kv_heads, S, d_head, device=device, dtype=torch.float16)
            V_gqa = torch.randn(B, n_kv_heads, S, d_head, device=device, dtype=torch.float16)

            # GQA: expand K/V to match Q heads
            n_rep = n_heads // n_kv_heads
            K_gqa_exp = K_gqa.unsqueeze(2).expand(B, n_kv_heads, n_rep, S, d_head).reshape(B, n_heads, S, d_head)
            V_gqa_exp = V_gqa.unsqueeze(2).expand(B, n_kv_heads, n_rep, S, d_head).reshape(B, n_heads, S, d_head)

            gqa_attn = benchmark_fn(
                f"GQA_B{B}", lambda: torch.softmax(Q_gqa @ K_gqa_exp.transpose(-2, -1) * scale, dim=-1) @ V_gqa_exp
            )

            # === MLA decode ===
            # Step 1: Read compressed latent from cache
            compressed_k_cache = torch.randn(B, S, d_latent, device=device, dtype=torch.float16)
            compressed_v_cache = torch.randn(B, S, d_latent, device=device, dtype=torch.float16)
            rope_cache = torch.randn(B, S, n_heads * rope_d, device=device, dtype=torch.float16)

            # Step 2: Up-project for attention
            W_up_k = torch.randn(d_latent, n_heads * d_head, device=device, dtype=torch.float16)
            W_up_v = torch.randn(d_latent, n_heads * d_head, device=device, dtype=torch.float16)

            # New token query
            Q_new = torch.randn(B, 1, d, device=device, dtype=torch.float16)
            W_q_proj = torch.randn(d, n_heads * d_head, device=device, dtype=torch.float16)
            W_q_rope = torch.randn(d, n_heads * rope_d, device=device, dtype=torch.float16)

            def mla_decode_step():
                # Project new query
                q = Q_new @ W_q_proj  # (B, 1, n_heads*d_head)
                q = q.view(B, 1, n_heads, d_head).transpose(1, 2)  # (B, n_heads, 1, d_head)
                q_rope = Q_new @ W_q_rope  # (B, 1, n_heads*rope_d)
                q_rope = q_rope.view(B, 1, n_heads, rope_d).transpose(1, 2)  # (B, n_heads, 1, rope_d)

                # Up-project cached KV
                k_up = compressed_k_cache @ W_up_k  # (B, S, n_heads*d_head)
                k_up = k_up.view(B, S, n_heads, d_head).transpose(1, 2)  # (B, n_heads, S, d_head)
                v_up = compressed_v_cache @ W_up_v  # (B, S, n_heads*d_head)
                v_up = v_up.view(B, S, n_heads, d_head).transpose(1, 2)  # (B, n_heads, S, d_head)

                # RoPE cached positions
                k_rope = rope_cache.view(B, S, n_heads, rope_d).transpose(1, 2)  # (B, n_heads, S, rope_d)

                # Concatenate: [q_nope, q_rope] × [k_nope, k_rope]
                # In MLA: attention = softmax(q_nope @ k_nope.T + q_rope @ k_rope.T) / sqrt(d)
                # Simplified: just use full d_head (nope+rope concatenated in practice)
                attn_scores = q @ k_up.transpose(-2, -1) * scale
                attn_weights = torch.softmax(attn_scores, dim=-1)
                attn_out = attn_weights @ v_up
                return attn_out

            mla_attn = benchmark_fn(
                f"MLA_B{B}", mla_decode_step
            )

            # MLA without up-projection (just attention on compressed)
            # This simulates what happens if we do attention in latent space
            q_latent = torch.randn(B, 1, d_latent, device=device, dtype=torch.float16)
            compressed_k_only = torch.randn(B, S, d_latent, device=device, dtype=torch.float16)

            mla_compressed_only = benchmark_fn(
                f"MLA_compressed_B{B}",
                lambda: torch.softmax(q_latent @ compressed_k_only.transpose(-2, -1) / math.sqrt(d_latent), dim=-1)
            )

            decode_results[f"B{B}"] = {
                "mha_ms": mha_attn["median_ms"],
                "gqa_ms": gqa_attn["median_ms"],
                "mla_ms": mla_attn["median_ms"],
                "mla_compressed_ms": mla_compressed_only["median_ms"],
                "mla_vs_mha": mla_attn["median_ms"] / mha_attn["median_ms"],
                "mla_vs_gqa": mla_attn["median_ms"] / gqa_attn["median_ms"],
            }

            print(f"  {label} B={B}: MHA={mha_attn['median_ms']:.3f}ms "
                  f"GQA={gqa_attn['median_ms']:.3f}ms "
                  f"MLA={mla_attn['median_ms']:.3f}ms "
                  f"(MLA/MHA={mla_attn['median_ms']/mha_attn['median_ms']:.2f}x "
                  f"MLA/GQA={mla_attn['median_ms']/gqa_attn['median_ms']:.2f}x)")

        results[f"decode_{label}"] = decode_results

    # ========================
    # Exp 4: Upsample Matmul Overhead
    # ========================
    print("\n=== Exp 4: Upsample Matmul Overhead (MLA vs reading full KV) ===")

    # Key question: is MLA up-projection cheaper than reading full KV from HBM?
    # MLA: read compressed (d_latent) → up-project (matmul) → full (n_heads×d_head)
    # MHA: read full (n_heads×d_head) directly from HBM

    for label, cfg in configs.items():
        d_latent = cfg["d_latent"]
        n_heads = cfg["n_heads"]
        d_head = cfg["d_head"]
        rope_d = cfg["rope_d"]

        # Bytes to read from HBM per attention step (B=1, S=2048)
        S = 2048
        B = 1

        mha_kv_read_bytes = 2 * n_heads * d_head * S * 2  # K+V FP16
        gqa_kv_read_bytes = 2 * cfg["n_kv_heads"] * d_head * S * 2  # K+V FP16
        mla_kv_read_bytes = 2 * d_latent * S * 2 + 2 * n_heads * rope_d * S * 2  # compressed+rope FP16

        # Upsample matmul sizes
        # K: (1, S, d_latent) × (d_latent, n_heads*d_head) → (1, S, n_heads*d_head)
        # V: same
        up_k_flops = 2 * B * S * d_latent * (n_heads * d_head)
        up_v_flops = 2 * B * S * d_latent * (n_heads * d_head)

        total_up_flops = up_k_flops + up_v_flops

        # Time estimates using HBM BW (890 GB/s) and compute (167 TFLOPS)
        mha_read_time_est = mha_kv_read_bytes / (890e9) * 1e3  # ms
        gqa_read_time_est = gqa_kv_read_bytes / (890e9) * 1e3
        mla_read_time_est = mla_kv_read_bytes / (890e9) * 1e3
        up_compute_time_est = total_up_flops / (167e12) * 1e3  # ms

        # Measured
        compressed_k = torch.randn(B, S, d_latent, device=device, dtype=torch.float16)
        W_up_k = torch.randn(d_latent, n_heads * d_head, device=device, dtype=torch.float16)
        compressed_v = torch.randn(B, S, d_latent, device=device, dtype=torch.float16)
        W_up_v = torch.randn(d_latent, n_heads * d_head, device=device, dtype=torch.float16)

        up_k_bench = benchmark_fn(f"{label}_up_k_S{S}", lambda: compressed_k @ W_up_k)
        up_v_bench = benchmark_fn(f"{label}_up_v_S{S}", lambda: compressed_v @ W_up_v)

        # Full KV read benchmark
        K_full = torch.randn(B, n_heads, S, d_head, device=device, dtype=torch.float16)
        V_full = torch.randn(B, n_heads, S, d_head, device=device, dtype=torch.float16)
        Q = torch.randn(B, n_heads, 1, d_head, device=device, dtype=torch.float16)

        mha_attn_bench = benchmark_fn(
            f"{label}_mha_attn_S{S}",
            lambda: torch.softmax(Q @ K_full.transpose(-2, -1) * (1/math.sqrt(d_head)), dim=-1) @ V_full
        )

        overhead_results = {
            "mha_kv_read_bytes": mha_kv_read_bytes,
            "gqa_kv_read_bytes": gqa_kv_read_bytes,
            "mla_kv_read_bytes": mla_kv_read_bytes,
            "mha_kv_read_MB": mha_kv_read_bytes / 1e6,
            "gqa_kv_read_MB": gqa_kv_read_bytes / 1e6,
            "mla_kv_read_MB": mla_kv_read_bytes / 1e6,
            "mla_vs_mha_read_ratio": mha_kv_read_bytes / mla_kv_read_bytes,
            "mla_vs_gqa_read_ratio": gqa_kv_read_bytes / mla_kv_read_bytes,
            "up_k_flops": up_k_flops,
            "up_v_flops": up_v_flops,
            "total_up_flops": total_up_flops,
            "mha_read_time_est_ms": mha_read_time_est,
            "mla_read_time_est_ms": mla_read_time_est,
            "up_compute_time_est_ms": up_compute_time_est,
            "up_k_measured_ms": up_k_bench["median_ms"],
            "up_v_measured_ms": up_v_bench["median_ms"],
            "up_total_measured_ms": up_k_bench["median_ms"] + up_v_bench["median_ms"],
            "mha_attn_measured_ms": mha_attn_bench["median_ms"],
            "mla_total_est_ms": mla_read_time_est + up_k_bench["median_ms"] + up_v_bench["median_ms"],
        }

        results[f"overhead_{label}"] = overhead_results

        print(f"  {label}: MHA read={mha_kv_read_bytes/1e6:.1f}MB "
              f"GQA read={gqa_kv_read_bytes/1e6:.1f}MB "
              f"MLA read={mla_kv_read_bytes/1e6:.1f}MB "
              f"(vs_MHA={mha_kv_read_bytes/mla_kv_read_bytes:.1f}x省) "
              f"up_K={up_k_bench['median_ms']:.3f}ms "
              f"up_V={up_v_bench['median_ms']:.3f}ms "
              f"MHA_attn={mha_attn_bench['median_ms']:.3f}ms "
              f"MLA_total_est={mla_read_time_est+up_k_bench['median_ms']+up_v_bench['median_ms']:.3f}ms")

    # ========================
    # Exp 5: RoPE Decoupled Attention Accuracy
    # ========================
    print("\n=== Exp 5: RoPE Decoupled Attention Accuracy ===")

    for label, cfg in configs.items():
        d = cfg["d"]
        n_heads = cfg["n_heads"]
        d_head = cfg["d_head"]
        rope_d = cfg["rope_d"]
        d_latent = cfg["d_latent"]

        S = 64
        B = 4
        d_nope = d_head - rope_d  # Non-RoPE part of head

        # Standard attention (baseline)
        Q_std = torch.randn(B, n_heads, S, d_head, device=device, dtype=torch.float32)
        K_std = torch.randn(B, n_heads, S, d_head, device=device, dtype=torch.float32)
        V_std = torch.randn(B, n_heads, S, d_head, device=device, dtype=torch.float32)

        std_attn = torch.softmax(Q_std @ K_std.transpose(-2, -1) / math.sqrt(d_head), dim=-1) @ V_std

        # MLA-style: split into nope + rope parts
        Q_nope = Q_std[..., :d_nope]  # (B, n_heads, S, d_nope)
        Q_rope = Q_std[..., d_nope:]  # (B, n_heads, S, rope_d)
        K_nope = K_std[..., :d_nope]
        K_rope = K_std[..., d_nope:]

        # MLA attention: score = Q_nope @ K_nope.T + Q_rope @ K_rope.T
        score_nope = Q_nope @ K_nope.transpose(-2, -1)  # (B, n_heads, S, S)
        score_rope = Q_rope @ K_rope.transpose(-2, -1)  # (B, n_heads, S, S)
        total_score = (score_nope + score_rope) / math.sqrt(d_head)

        # Standard score (for comparison)
        std_score = Q_std @ K_std.transpose(-2, -1) / math.sqrt(d_head)

        # Check that split scores = full score
        score_cos = F.cosine_similarity(
            total_score.flatten(), std_score.flatten(), dim=0
        ).item()
        score_diff = (total_score - std_score).abs().max().item()

        # MLA attention output
        mla_attn = torch.softmax(total_score, dim=-1) @ V_std

        # Check output matches
        out_cos = F.cosine_similarity(
            mla_attn.flatten(), std_attn.flatten(), dim=0
        ).item()
        out_diff = (mla_attn - std_attn).abs().max().item()

        # MLA with RoPE applied (simplified sinusoidal)
        positions = torch.arange(S, device=device, dtype=torch.float32)
        freq = 1.0 / (10000 ** (torch.arange(0, rope_d, 2, device=device, dtype=torch.float32) / rope_d))
        cos_vals = torch.cos(positions.unsqueeze(-1) * freq.unsqueeze(0))
        sin_vals = torch.sin(positions.unsqueeze(-1) * freq.unsqueeze(0))

        # Apply RoPE to rope parts
        Q_rope_even = Q_rope[..., 0::2]
        Q_rope_odd = Q_rope[..., 1::2]
        K_rope_even = K_rope[..., 0::2]
        K_rope_odd = K_rope[..., 1::2]

        # RoPE rotation
        Q_rope_rot_even = Q_rope_even * cos_vals - Q_rope_odd * sin_vals
        Q_rope_rot_odd = Q_rope_even * sin_vals + Q_rope_odd * cos_vals
        K_rope_rot_even = K_rope_even * cos_vals - K_rope_odd * sin_vals
        K_rope_rot_odd = K_rope_even * sin_vals + K_rope_odd * cos_vals

        # Reconstruct rotated rope vectors
        Q_rope_rot = torch.stack([Q_rope_rot_even, Q_rope_rot_odd], dim=-1).flatten(-2)
        K_rope_rot = torch.stack([K_rope_rot_even, K_rope_rot_odd], dim=-1).flatten(-2)

        # MLA with RoPE attention
        score_rope_rot = Q_rope_rot @ K_rope_rot.transpose(-2, -1)
        total_score_rope = (score_nope + score_rope_rot) / math.sqrt(d_head)
        mla_attn_rope = torch.softmax(total_score_rope, dim=-1) @ V_std

        # Compare with standard + RoPE applied to full vector
        # Standard RoPE (applied to full d_head)
        cos_vals_full = torch.cos(positions.unsqueeze(-1) * (1.0 / (10000 ** (torch.arange(0, d_head, 2, device=device, dtype=torch.float32) / d_head))).unsqueeze(0))
        sin_vals_full = torch.sin(positions.unsqueeze(-1) * (1.0 / (10000 ** (torch.arange(0, d_head, 2, device=device, dtype=torch.float32) / d_head))).unsqueeze(0))

        Q_std_even = Q_std[..., 0::2]
        Q_std_odd = Q_std[..., 1::2]
        K_std_even = K_std[..., 0::2]
        K_std_odd = K_std[..., 1::2]

        Q_std_rot_even = Q_std_even * cos_vals_full - Q_std_odd * sin_vals_full
        Q_std_rot_odd = Q_std_even * sin_vals_full + Q_std_odd * cos_vals_full
        K_std_rot_even = K_std_even * cos_vals_full - K_std_odd * sin_vals_full
        K_std_rot_odd = K_std_even * sin_vals_full + K_std_odd * cos_vals_full

        Q_std_rot = torch.stack([Q_std_rot_even, Q_std_rot_odd], dim=-1).flatten(-2)
        K_std_rot = torch.stack([K_std_rot_even, K_std_rot_odd], dim=-1).flatten(-2)

        std_score_rope = Q_std_rot @ K_std_rot.transpose(-2, -1) / math.sqrt(d_head)
        std_attn_rope = torch.softmax(std_score_rope, dim=-1) @ V_std

        # Compare MLA+RoPE vs standard+RoPE
        rope_out_cos = F.cosine_similarity(
            mla_attn_rope.flatten(), std_attn_rope.flatten(), dim=0
        ).item()
        rope_out_diff = (mla_attn_rope - std_attn_rope).abs().max().item()

        results[f"accuracy_{label}"] = {
            "d_nope": d_nope,
            "rope_d": rope_d,
            "score_split_cos_sim": score_cos,
            "score_split_max_diff": score_diff,
            "output_cos_sim": out_cos,
            "output_max_diff": out_diff,
            "rope_output_cos_sim": rope_out_cos,
            "rope_output_max_diff": rope_out_diff,
        }

        print(f"  {label}: score_split cos={score_cos:.6f} diff={score_diff:.6e} "
              f"output cos={out_cos:.6f} diff={out_diff:.6e} "
              f"MLA+RoPE vs Std+RoPE cos={rope_out_cos:.6f} diff={rope_out_diff:.6e}")

    # ========================
    # Exp 6: MLA Capacity Scaling (Concurrent Requests)
    # ========================
    print("\n=== Exp 6: MLA Capacity Scaling (KV Cache per 24GB GPU) ===")

    gpu_memory_bytes = 24 * 1e9  # 24 GB
    # Assume model weights + overhead take 30% of GPU
    available_kv_bytes = gpu_memory_bytes * 0.7  # 70% for KV cache

    capacity_results = {}
    for label, cfg in configs.items():
        n_heads = cfg["n_heads"]
        n_kv_heads = cfg["n_kv_heads"]
        d_head = cfg["d_head"]
        d_latent = cfg["d_latent"]
        rope_d = cfg["rope_d"]

        for avg_S in [2048, 4096, 8192, 16384]:
            mha_per_req = 2 * n_heads * d_head * avg_S * 2
            gqa_per_req = 2 * n_kv_heads * d_head * avg_S * 2
            mla_per_req = 2 * d_latent * avg_S * 2 + 2 * n_heads * rope_d * avg_S * 2

            mha_max_req = int(available_kv_bytes / mha_per_req)
            gqa_max_req = int(available_kv_bytes / gqa_per_req)
            mla_max_req = int(available_kv_bytes / mla_per_req)

            capacity_results[f"{label}_S{avg_S}"] = {
                "avg_S": avg_S,
                "mha_max_concurrent": mha_max_req,
                "gqa_max_concurrent": gqa_max_req,
                "mla_max_concurrent": mla_max_req,
                "mla_vs_mha_capacity": mla_max_req / mha_max_req if mha_max_req > 0 else 0,
                "mla_vs_gqa_capacity": mla_max_req / gqa_max_req if gqa_max_req > 0 else 0,
            }

            print(f"  {label} avg_S={avg_S}: MHA={mha_max_req}req "
                  f"GQA={gqa_max_req}req "
                  f"MLA={mla_max_req}req "
                  f"(MLA/MHA={mla_max_req/mha_max_req:.1f}x MLA/GQA={mla_max_req/gqa_max_req:.1f}x)")

    results["capacity"] = capacity_results

    # Save
    with open("results/mla_kv_compression_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/mla_kv_compression_benchmark.json")

    return results


if __name__ == "__main__":
    run_experiment()