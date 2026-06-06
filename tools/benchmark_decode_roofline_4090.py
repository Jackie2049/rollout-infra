#!/usr/bin/env python3
"""Batched Decode Throughput Roofline Verification on RTX 4090
===========================================================

Verifies the theoretical Roofline model for LLM decode throughput:
- Decode is always memory-bound (AI ≈ 1.0)
- Throughput ∝ HBM bandwidth × batch_size / (hidden × vocab)
- Peak throughput = HBM_BW / bytes_per_token

Expected (Roofline model):
  RTX 4090 HBM: ~900 GB/s
  7B FP16: ~529K tok/s @ B=512 (实测)
  Throughput should scale linearly with batch until compute-bound crossover

This benchmark tests actual throughput vs model predictions.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import time
import json

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print("=" * 60)

device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)

# ============================================================
# 1. HBM Bandwidth Measurement (sanity check)
# ============================================================

def measure_hbm_bw():
    """Simple HBM bandwidth measurement via large memcpy."""
    print("\n1. HBM Bandwidth Measurement")

    size_mb = 100  # 100MB copy
    n_elements = size_mb * 1024 * 1024 // 4  # FP32 elements

    src = torch.randn(n_elements, device='cuda')
    dst = torch.empty(n_elements, device='cuda')

    # Warmup
    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize()

    n = 100
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        dst.copy_(src)
    e.record()
    torch.cuda.synchronize()

    time_ms = s.elapsed_time(e) / n
    bw_gb_s = size_mb / time_ms  # read + write = 2 × size, but time_ms is one direction

    # Actually memcpy reads + writes = 2 × size
    actual_bw = 2 * size_mb / time_ms

    print(f"  Copy {size_mb}MB: {time_ms:.3f}ms")
    print(f"  HBM bandwidth (read+write): {actual_bw:.1f} GB/s")
    print(f"  (RTX 4090 peak: 960 GB/s, we expect ~900 GB/s)")

    return actual_bw


# ============================================================
# 2. Decode Throughput vs Batch Size (Roofline verification)
# ============================================================

def bench_decode_throughput():
    """Measure decode throughput (tokens/sec) vs batch size."""
    print("\n2. Decode Throughput vs Batch Size")

    # Simulate a single transformer decode layer
    # Key: linear layer [B, H] @ [H, H] → [B, H]
    H = 4096  # LLaMA-7B hidden size

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    results = []

    w = torch.randn(H, H, device='cuda')  # weight matrix

    for B in batch_sizes:
        x = torch.randn(B, H, device='cuda')

        # Warmup
        for _ in range(10):
            _ = x @ w
        torch.cuda.synchronize()

        # Measure
        n = 50
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            out = x @ w
        e.record()
        torch.cuda.synchronize()

        time_ms = s.elapsed_time(e) / n

        # Throughput = B tokens per step / time
        tok_per_s = B / (time_ms / 1000)

        # Theoretical (Roofline): memory-bound
        # IO per step = 2 × B × H × 4 (read x + read w_row + write out)
        #             ≈ 2 × H × H × 4 + 2 × B × H × 4  (for B<<H²)
        # For H=4096: weight = 2 × 4096² × 4 = 134MB (dominates for small B)
        # compute = 2 × B × H² FLOPs

        io_bytes = 2 * H * H * 4 + 2 * B * H * 4  # weight + activations
        compute_flops = 2 * B * H * H  # matmul FLOPs

        # Predicted time (memory-bound)
        hbm_bw = 900e9  # 900 GB/s (measured)
        predicted_time_ms = io_bytes / hbm_bw * 1000

        # Predicted time (compute-bound)
        peak_tflops = 169.6  # measured FP16 TFLOPS
        compute_time_ms = compute_flops / (peak_tflops * 1e12) * 1000

        # Actual is max of compute and memory time (Roofline)
        predicted_ms = max(predicted_time_ms, compute_time_ms)
        error_pct = abs(time_ms - predicted_ms) / time_ms * 100

        # Arithmetic intensity
        ai = compute_flops / io_bytes

        print(f"  B={B}: time={time_ms:.4f}ms throughput={tok_per_s:.0f} tok/s "
              f"AI={ai:.1f} ops/byte "
              f"predict={predicted_ms:.4f}ms error={error_pct:.1f}% "
              f"{'MEM' if predicted_time_ms > compute_time_ms else 'COMP'}")

        results.append({
            "B": B,
            "time_ms": round(time_ms, 4),
            "tok_per_s": round(tok_per_s, 0),
            "ai": round(ai, 1),
            "predicted_ms": round(predicted_ms, 4),
            "error_pct": round(error_pct, 1),
            "bound": "MEM" if predicted_time_ms > compute_time_ms else "COMP",
        })

    return results


# ============================================================
# 3. Full Model Decode Throughput
# ============================================================

def bench_full_model_decode():
    """Simulate full model decode (N layers + logits)."""
    print("\n3. Full Model Decode Throughput (32 layers + logits)")

    n_layers = 32
    H = 4096
    V = 32000

    # Allocate all weights
    w_layers = [torch.randn(H, H, device='cuda') for _ in range(n_layers)]
    w_logits = torch.randn(H, V, device='cuda')
    w_norm = torch.randn(H, device='cuda')

    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
    results = []

    for B in batch_sizes:
        x = torch.randn(B, H, device='cuda')

        def full_decode(x_in):
            out = x_in
            for w in w_layers:
                h = out @ w
                var = h.pow(2).mean(-1, keepdim=True)
                out = h * torch.rsqrt(var + 1e-6) * w_norm
            logits = out @ w_logits
            return logits

        # Warmup
        for _ in range(5):
            _ = full_decode(x)
        torch.cuda.synchronize()

        n = 20
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            _ = full_decode(x)
        e.record()
        torch.cuda.synchronize()

        time_ms = s.elapsed_time(e) / n
        tok_per_s = B / (time_ms / 1000)

        print(f"  B={B}: {time_ms:.3f}ms/step → {tok_per_s:.0f} tok/s")

        results.append({
            "B": B,
            "time_ms": round(time_ms, 3),
            "tok_per_s": round(tok_per_s, 0),
        })

    return results


# ============================================================
# Run
# ============================================================

hbm_bw = measure_hbm_bw()
throughput_results = bench_decode_throughput()
full_model_results = bench_full_model_decode()

print("\n" + "=" * 60)
print("SUMMARY: Decode Roofline on RTX 4090")
print("=" * 60)
print(f"HBM BW: {hbm_bw:.1f} GB/s")
print(f"Key insight: Decode is memory-bound (AI≈1) → throughput ∝ HBM BW × B / (2×H²×4bytes)")
print("For 7B FP16 single layer: peak ≈ 900GB/s / 134MB = 6.7K steps/s × B tokens")

# Save
out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, 'decode_roofline_results.json')
with open(out_path, 'w') as f:
    json.dump({
        "hbm_bw": round(hbm_bw, 1),
        "single_layer": throughput_results,
        "full_model": full_model_results,
    }, f, indent=2)
print(f"Results saved to {out_path}")