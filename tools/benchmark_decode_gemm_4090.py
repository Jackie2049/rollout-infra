#!/usr/bin/env python3
"""Decode GEMM Micro-Benchmark on RTX 4090
============================================

Measures per-layer decode GEMM timing for realistic model sizes.
Decode = memory-bound, B=1 single token generation path.

Tests: QKV proj + MLP (2 GEMMs per attention + 3 GEMMs per MLP)
With B=1,4,8,32,128,256,512

Usage:
  python benchmark_decode_gemm_4090.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import time
import json

def benchmark_gemm(M, K, N, dtype, device, n_iters=200, n_warmup=20):
    """Benchmark A[M,K] × B[K,N] → C[M,N]"""
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)

    for _ in range(n_warmup):
        C = torch.mm(A, B)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    per_ms = (t1 - t0) * 1000 / n_iters
    # FLOPS = 2 * M * K * N
    flops = 2 * M * K * N
    tflops = flops / (per_ms * 1e-3) / 1e12
    # Data volume = M*K + K*N + M*N elements
    data_bytes = (M*K + K*N + M*N) * torch.tensor([], dtype=dtype).element_size()
    bw_gb_s = data_bytes / (per_ms * 1e-3) / 1e9

    return per_ms, tflops, bw_gb_s

def main():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print(f"Decode GEMM Micro-Benchmark: {torch.cuda.get_device_name(device)}")

    # Model configs (typical LLaMA-style dimensions)
    # Attention: Q_proj, K_proj, V_proj → [B,H] × [H, H_q/k/v]
    # Attention out: [B,H_q] × [H_q, H]  (or [B,H] × [H,H] for MHA)
    # MLP: gate [B,H] × [H, inner], up [B,H] × [H, inner], down [B,inner] × [inner, H]

    configs = [
        ("7B-MHA", 4096, 4096, 11008),    # hidden=4096, inner=11008 (4*H*2/3)
        ("7B-GQA8", 4096, 4096, 11008),   # same but with GQA (KV proj smaller)
        ("70B", 8192, 8192, 28672),        # hidden=8192, inner=28672
    ]

    batch_sizes = [1, 4, 8, 32, 128, 256, 512]
    dtype = torch.float16

    results = []

    for name, H, H_q, inner in configs:
        print(f"\n=== {name}: H={H}, H_q={H_q}, inner={inner} ===")

        for B in batch_sizes:
            layer_times = {}

            # Q projection: [B, H] × [H, H_q]
            ms, tf, bw = benchmark_gemm(B, H, H_q, dtype, device)
            layer_times["q_proj"] = {"ms": round(ms, 4), "tflops": round(tf, 2), "bw_gb_s": round(bw, 2)}

            # K projection: [B, H] × [H, H] (or smaller for GQA)
            kv_dim = H if "MHA" in name else H // 4  # GQA: 8 KV heads
            ms, tf, bw = benchmark_gemm(B, H, kv_dim, dtype, device)
            layer_times["k_proj"] = {"ms": round(ms, 4), "tflops": round(tf, 2), "bw_gb_s": round(bw, 2)}

            # V projection (same dims as K)
            layer_times["v_proj"] = layer_times["k_proj"]

            # Attention output: [B, H_q] × [H_q, H]
            ms, tf, bw = benchmark_gemm(B, H_q, H, dtype, device)
            layer_times["attn_out"] = {"ms": round(ms, 4), "tflops": round(tf, 2), "bw_gb_s": round(bw, 2)}

            # MLP gate: [B, H] × [H, inner]
            ms, tf, bw = benchmark_gemm(B, H, inner, dtype, device)
            layer_times["mlp_gate"] = {"ms": round(ms, 4), "tflops": round(tf, 2), "bw_gb_s": round(bw, 2)}

            # MLP up: [B, H] × [H, inner]
            layer_times["mlp_up"] = layer_times["mlp_gate"]

            # MLP down: [B, inner] × [inner, H]
            ms, tf, bw = benchmark_gemm(B, inner, H, dtype, device)
            layer_times["mlp_down"] = {"ms": round(ms, 4), "tflops": round(tf, 2), "bw_gb_s": round(bw, 2)}

            # Total per layer
            attn_time = layer_times["q_proj"]["ms"] + layer_times["k_proj"]["ms"] + \
                        layer_times["v_proj"]["ms"] + layer_times["attn_out"]["ms"]
            mlp_time = layer_times["mlp_gate"]["ms"] + layer_times["mlp_up"]["ms"] + \
                       layer_times["mlp_down"]["ms"]
            layer_total = attn_time + mlp_time

            print(f"  B={B}: attn={attn_time:.3f}ms, mlp={mlp_time:.3f}ms, "
                  f"total={layer_total:.3f}ms/layer")

            results.append({
                "name": name, "batch_size": B,
                "hidden": H, "inner": inner,
                "layer_times": layer_times,
                "attn_total_ms": round(attn_time, 3),
                "mlp_total_ms": round(mlp_time, 3),
                "layer_total_ms": round(layer_total, 3),
            })

    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'decode_gemm_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print decode throughput estimates
    print("\n=== Decode Throughput Estimate (32 layers, 7B MHA) ===")
    for r in results:
        if r["name"] == "7B-MHA":
            B = r["batch_size"]
            layer_ms = r["layer_total_ms"]
            total_ms = layer_ms * 32
            throughput = B / (total_ms / 1000)
            print(f"  B={B}: {layer_ms:.3f}ms/layer, {total_ms:.1f}ms total, "
                  f"{throughput:.0f} tok/s")

if __name__ == '__main__':
    main()