"""
FlashInfer vs SDPA vs Triton Decode Attention Benchmark — RTX 4090 (SM89)

Compares 3 decode attention backends:
1. FlashInfer BatchDecodeWithPagedKVCacheWrapper (production kernel)
2. PyTorch SDPA with is_causal=False (correct baseline)
3. Triton custom decode kernel (our implementation)

Tests: correctness (cos_sim), latency, throughput (tok/s), GQA support
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import time

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
print(f"Device: {props.name} SM={props.major}.{props.minor}")

# Model config
H = 2560
num_heads = 20
num_kv_heads = 5
d_head = H // num_heads
S = 512

# =====================================================================
# Helper: benchmark function
# =====================================================================
def benchmark_fn(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    latencies = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    return np.mean(latencies), np.std(latencies)


# =====================================================================
# SECTION 1: SDPA Decode (is_causal=False — correct baseline)
# =====================================================================
print("\n" + "=" * 70)
print("SDPA Decode Attention (is_causal=False)")
print("=" * 70)

batch_sizes = [1, 4, 8, 16, 32]
sdpa_results = []

for B in batch_sizes:
    M = B * S  # total tokens
    # Create KV cache (S positions)
    k_cache = torch.randn(B, S, num_kv_heads, d_head, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn(B, S, num_kv_heads, d_head, dtype=torch.bfloat16, device=device)

    # Create Q (1 position per batch, at position S)
    q = torch.randn(B, 1, num_heads, d_head, dtype=torch.bfloat16, device=device)

    def sdpa_decode():
        # Expand KV for SDPA (requires same num_heads)
        k_expanded = k_cache.unsqueeze(2).expand(B, S, num_heads // num_kv_heads, num_kv_heads, d_head)
        k_expanded = k_expanded.reshape(B, S, num_heads, d_head)
        v_expanded = v_cache.unsqueeze(2).expand(B, S, num_heads // num_kv_heads, num_kv_heads, d_head)
        v_expanded = v_expanded.reshape(B, S, num_heads, d_head)
        return F.scaled_dot_product_attention(q, k_expanded, v_expanded, is_causal=False)

    mean_ms, std_ms = benchmark_fn(sdpa_decode, warmup=5, iters=30)
    tok_per_s = B / (mean_ms * 1e-3)  # decode: B tokens per step

    sdpa_results.append({
        "batch": B, "seq_len": S,
        "ms": round(mean_ms, 4), "std_ms": round(std_ms, 4),
        "tok_per_s": round(tok_per_s, 1),
    })
    print(f"  B={B}: {mean_ms:.3f}ms ± {std_ms:.3f}ms → {tok_per_s:.1f} tok/s")


# =====================================================================
# SECTION 2: FlashInfer Decode (if available)
# =====================================================================
FLASHINFER_AVAILABLE = False
try:
    import flashinfer
    FLASHINFER_AVAILABLE = True
    print(f"\nFlashInfer version: {flashinfer.__version__}")
except ImportError:
    print("\nFlashInfer not available — skipping FlashInfer benchmark")

fi_results = []
if FLASHINFER_AVAILABLE:
    print("\n" + "=" * 70)
    print("FlashInfer BatchDecodeWithPagedKVCacheWrapper")
    print("=" * 70)

    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    # Create workspace buffer
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)

    for B in batch_sizes:
        M = B * S

        # Create paged KV cache (page_size=16, NHD layout)
        page_size = 16
        num_pages_per_seq = (S + page_size - 1) // page_size
        total_pages = B * num_pages_per_seq

        k_data = torch.randn(total_pages, page_size, num_kv_heads, d_head, dtype=torch.bfloat16, device=device)
        v_data = torch.randn(total_pages, page_size, num_kv_heads, d_head, dtype=torch.bfloat16, device=device)

        # Page table: logical → physical mapping
        page_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
        paged_kv_indptr = torch.arange(0, total_pages + num_pages_per_seq, num_pages_per_seq, dtype=torch.int32, device=device)
        paged_kv_last_page_len = torch.full((B,), page_size, dtype=torch.int32, device=device)

        # Create wrapper
        wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")

        # Q tensor
        q = torch.randn(B, num_heads, d_head, dtype=torch.bfloat16, device=device)

        def fi_decode():
            wrapper.begin_forward(
                paged_kv_indptr,
                page_indices,
                paged_kv_last_page_len,
                num_heads,
                num_kv_heads,
                d_head,
            )
            o = wrapper.forward(
                q,
                k_data,
                v_data,
                paged_kv_indptr,
                page_indices,
                paged_kv_last_page_len,
            )
            wrapper.end_forward()
            return o

        mean_ms, std_ms = benchmark_fn(fi_decode, warmup=5, iters=30)
        tok_per_s = B / (mean_ms * 1e-3)

        fi_results.append({
            "batch": B, "seq_len": S,
            "ms": round(mean_ms, 4), "std_ms": round(std_ms, 4),
            "tok_per_s": round(tok_per_s, 1),
        })
        print(f"  B={B}: {mean_ms:.3f}ms ± {std_ms:.3f}ms → {tok_per_s:.1f} tok/s")

        # Correctness check: compare with SDPA
        # (reuse the first batch for correctness)
        if B == batch_sizes[0]:
            # Get SDPA output for correctness
            k_for_sdpa = k_data.reshape(total_pages, page_size, num_kv_heads, d_head)
            k_full = k_for_sdpa[:num_pages_per_seq].reshape(S, num_kv_heads, d_head)
            k_expanded = k_full.unsqueeze(1).expand(S, num_heads // num_kv_heads, num_kv_heads, d_head).reshape(S, num_heads, d_head)
            v_full = v_data[:num_pages_per_seq].reshape(S, num_kv_heads, d_head)
            v_expanded = v_full.unsqueeze(1).expand(S, num_heads // num_kv_heads, num_kv_heads, d_head).reshape(S, num_heads, d_head)
            q_for_sdpa = q.unsqueeze(1)  # B, 1, num_heads, d_head
            sdpa_out = F.scaled_dot_product_attention(q_for_sdpa, k_expanded.unsqueeze(0), v_expanded.unsqueeze(0), is_causal=False)

            fi_out = fi_decode()

            cos_sim = F.cosine_similarity(
                sdpa_out.flatten().unsqueeze(0).float(),
                fi_out.flatten().unsqueeze(0).float()
            ).item()
            print(f"  Correctness: cos_sim(FI vs SDPA) = {cos_sim:.6f}")


# =====================================================================
# SECTION 3: Triton Decode (if available)
# =====================================================================
TRITON_AVAILABLE = False
try:
    import triton
    TRITON_AVAILABLE = True
    print(f"\nTriton version: {triton.__version__}")
except ImportError:
    print("\nTriton not available — skipping Triton benchmark")

# Note: Our custom Triton decode kernel requires specific implementation
# For now, we'll just compare SDPA and FlashInfer results
# Triton benchmark would need to import our triton_decode_attn_benchmark script

triton_results = []


# =====================================================================
# Save Results
# =====================================================================
all_results = {
    "device_info": {
        "name": props.name,
        "sm": f"{props.major}.{props.minor}",
        "mp_count": props.multi_processor_count,
        "memory_GB": round(props.total_memory / 1e9, 2),
    },
    "model_config": {
        "H": H, "num_heads": num_heads, "num_kv_heads": num_kv_heads,
        "d_head": d_head, "S": S,
    },
    "sdpa_decode": sdpa_results,
    "flashinfer_decode": fi_results,
    "flashinfer_available": FLASHINFER_AVAILABLE,
    "triton_available": TRITON_AVAILABLE,
}

os.makedirs("results", exist_ok=True)
output_path = "results/flashinfer_decode_benchmark.json"
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {output_path}")


# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if FLASHINFER_AVAILABLE and fi_results:
    print("\nFlashInfer vs SDPA Decode Comparison:")
    for i, B in enumerate(batch_sizes):
        sdpa_ms = sdpa_results[i]["ms"]
        fi_ms = fi_results[i]["ms"]
        speedup = sdpa_ms / fi_ms
        print(f"  B={B}: SDPA={sdpa_ms:.3f}ms, FI={fi_ms:.3f}ms → {speedup:.2f}x")
else:
    print("\nFlashInfer not available — only SDPA results")
    for r in sdpa_results:
        print(f"  B={r['batch']}: {r['ms']:.3f}ms → {r['tok_per_s']:.1f} tok/s")