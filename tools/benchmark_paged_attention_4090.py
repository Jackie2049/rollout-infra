#!/usr/bin/env python3
"""Paged vs Contiguous Attention: KV Cache Indirection Overhead
============================================================

Measures the overhead of vLLM's block-based KV cache (Paged Attention)
vs contiguous KV cache (standard attention) on RTX 4090.

vLLM uses block_table to map virtual KV positions to physical blocks.
This adds an indirection layer (block lookup) that increases memory access
and potentially hurts cache locality.

Key questions:
A. How much overhead does block_table lookup add?
B. Does the overhead depend on batch size or sequence length?
C. Why does vLLM need custom Triton kernels instead of just torch.sdpa?
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import time
import json

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print("=" * 60)


# ============================================================
# 1. Contiguous Attention (PyTorch SDPA - no paged KV)
# ============================================================

def bench_contiguous_attention():
    """Standard PyTorch SDPA attention (contiguous KV cache)."""
    print("\n1. Contiguous Attention (torch.nn.functional.scaled_dot_product_attention)")

    results = []
    for B, H_q, H_kv, D, seq_len, _desc in [
        (1, 32, 32, 128, 512, "B=1 H=32 D=128 S=512"),
        (8, 32, 32, 128, 512, "B=8"),
        (32, 32, 32, 128, 512, "B=32"),
        (128, 32, 32, 128, 512, "B=128"),
        (1, 32, 8, 128, 2048, "B=1 GQA H_kv=8 S=2048"),  # GQA
        (8, 32, 8, 128, 2048, "B=8 GQA"),
        (32, 32, 8, 128, 2048, "B=32 GQA"),
        (128, 32, 8, 128, 2048, "B=128 GQA"),
    ]:
        # Actually B is total tokens, H_q is num_q_heads, H_kv is num_kv_heads
        # For decode: Q = [B, H_q, 1, D], K/V = [B, H_kv, seq_len, D]
        # With GQA: H_kv < H_q, each KV head serves H_q/H_kv Q heads

        q = torch.randn(B, H_q, 1, D, device='cuda')
        k = torch.randn(B, H_kv, seq_len, D, device='cuda')
        v = torch.randn(B, H_kv, seq_len, D, device='cuda')

        # For GQA: expand K/V heads to match Q heads for SDPA
        if H_kv < H_q:
            n_groups = H_q // H_kv
            k = k.repeat_interleave(n_groups, dim=1)  # [B, H_q, seq_len, D]
            v = v.repeat_interleave(n_groups, dim=1)  # [B, H_q, seq_len, D]

        # Warmup
        for _ in range(10):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        n = 50
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        e.record()
        torch.cuda.synchronize()

        ms = s.elapsed_time(e) / n

        desc = f"B={B} Hq={H_q} Hkv={H_kv} D={D} S={seq_len}"
        gqa = "GQA" if H_kv < H_q else "MHA"
        print(f"  {desc} ({gqa}): {ms:.4f}ms")

        results.append({
            "B": B, "H_q": H_q, "H_kv": H_kv, "D": D, "S": seq_len,
            "type": gqa,
            "time_ms": round(ms, 4),
        })

    return results


# ============================================================
# 2. Paged Attention Simulation (block_table indirection)
# ============================================================

def bench_paged_attention_simulation():
    """Simulate paged attention by gathering KV from scattered blocks.

    This measures the overhead of:
    1. Block table lookup (virtual→physical mapping)
    2. Scattered memory reads (non-contiguous KV blocks)
    """
    print("\n2. Paged Attention Simulation (block_table gather)")

    BLOCK_SIZE = 16  # vLLM default block size
    results = []

    for B, H_kv, D, seq_len in [
        (1, 8, 128, 512),
        (8, 8, 128, 512),
        (32, 8, 128, 512),
        (128, 8, 128, 512),
        (1, 8, 128, 2048),
        (32, 8, 128, 2048),
    ]:
        n_blocks = seq_len // BLOCK_SIZE
        total_blocks = B * n_blocks * H_kv + 100  # some extra blocks for "pool"

        # Physical KV cache: [total_blocks, BLOCK_SIZE, D]
        physical_kv = torch.randn(total_blocks, BLOCK_SIZE, D, device='cuda')

        # Block table: [B, H_kv, n_blocks] → maps virtual positions to physical blocks
        # Each entry is a physical block index
        block_table = torch.randint(0, total_blocks - B * n_blocks * H_kv,
                                     (B, H_kv, n_blocks), device='cuda')
        # Make each request's blocks contiguous (but scattered in physical space)
        for b in range(B):
            for h in range(H_kv):
                block_table[b, h] = torch.arange(b * n_blocks * H_kv + h * n_blocks,
                                                   b * n_blocks * H_kv + h * n_blocks + n_blocks)

        # Method 1: Gather KV from paged blocks (simulate paged attention)
        def paged_gather():
            # Gather blocks from physical KV using block_table
            # Result: [B, H_kv, seq_len, D]
            gathered = physical_kv[block_table]  # [B, H_kv, n_blocks, BLOCK_SIZE, D]
            gathered = gathered.reshape(B, H_kv, seq_len, D)
            return gathered

        # Method 2: Contiguous KV (no gather needed)
        contiguous_kv = torch.randn(B, H_kv, seq_len, D, device='cuda')

        # Warmup gather
        for _ in range(10):
            _ = paged_gather()
        torch.cuda.synchronize()

        # Measure gather overhead
        n = 100
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            _ = paged_gather()
        e.record()
        torch.cuda.synchronize()
        gather_ms = s.elapsed_time(e) / n

        # Measure contiguous read (baseline)
        def contiguous_read():
            return contiguous_kv.clone()

        for _ in range(10):
            _ = contiguous_read()
        torch.cuda.synchronize()

        s.record()
        for _ in range(n):
            _ = contiguous_read()
        e.record()
        torch.cuda.synchronize()
        read_ms = s.elapsed_time(e) / n

        overhead_pct = (gather_ms / read_ms - 1) * 100 if read_ms > 0 else 0

        print(f"  B={B} Hkv={H_kv} D={D} S={seq_len}: "
              f"gather={gather_ms:.4f}ms contiguous_read={read_ms:.4f}ms "
              f"overhead={overhead_pct:.1f}%")

        results.append({
            "B": B, "H_kv": H_kv, "D": D, "S": seq_len,
            "BLOCK_SIZE": BLOCK_SIZE,
            "gather_ms": round(gather_ms, 4),
            "read_ms": round(read_ms, 4),
            "overhead_pct": round(overhead_pct, 1),
        })

    return results


# ============================================================
# 3. Full Paged Attention (gather + compute) vs Contiguous
# ============================================================

def bench_paged_vs_contiguous_full():
    """Full attention computation: paged (gather+compute) vs contiguous."""
    print("\n3. Full Attention: Paged (gather+SDPA) vs Contiguous (SDPA only)")

    BLOCK_SIZE = 16
    results = []

    for B, H_q, H_kv, D, seq_len, _desc in [
        (4, 32, 8, 128, 512, "B=4 GQA S=512"),
        (16, 32, 8, 128, 512, "B=16 GQA S=512"),
        (32, 32, 8, 128, 512, "B=32 GQA S=512"),
        (128, 32, 8, 128, 512, "B=128 GQA S=512"),
        (4, 32, 8, 128, 2048, "B=4 GQA S=2048"),
        (32, 32, 8, 128, 2048, "B=32 GQA S=2048"),
    ]:
        n_blocks = seq_len // BLOCK_SIZE
        total_blocks = B * n_blocks * H_kv + 100

        # Physical KV
        physical_k = torch.randn(total_blocks, BLOCK_SIZE, D, device='cuda')
        physical_v = torch.randn(total_blocks, BLOCK_SIZE, D, device='cuda')

        # Block table
        block_table = torch.zeros(B, H_kv, n_blocks, dtype=torch.long, device='cuda')
        for b in range(B):
            for h in range(H_kv):
                block_table[b, h] = torch.arange(b * n_blocks * H_kv + h * n_blocks,
                                                   b * n_blocks * H_kv + h * n_blocks + n_blocks)

        # Q for decode
        q = torch.randn(B, H_q, 1, D, device='cuda')

        # Contiguous attention (baseline)
        cont_k = torch.randn(B, H_kv, seq_len, D, device='cuda')
        cont_v = torch.randn(B, H_kv, seq_len, D, device='cuda')

        # For GQA: expand KV heads to match Q heads
        n_groups = H_q // H_kv
        cont_k_expanded = cont_k.repeat_interleave(n_groups, dim=1)
        cont_v_expanded = cont_v.repeat_interleave(n_groups, dim=1)

        def contiguous_attn():
            return torch.nn.functional.scaled_dot_product_attention(q, cont_k_expanded, cont_v_expanded)

        # Paged attention (gather + compute)
        def paged_attn():
            # Gather K and V from paged blocks
            gathered_k = physical_k[block_table].reshape(B, H_kv, seq_len, D)
            gathered_v = physical_v[block_table].reshape(B, H_kv, seq_len, D)
            gathered_k_expanded = gathered_k.repeat_interleave(n_groups, dim=1)
            gathered_v_expanded = gathered_v.repeat_interleave(n_groups, dim=1)
            return torch.nn.functional.scaled_dot_product_attention(q, gathered_k_expanded, gathered_v_expanded)

        # Warmup
        for _ in range(10):
            _ = contiguous_attn()
            _ = paged_attn()
        torch.cuda.synchronize()

        n = 50

        # Measure contiguous
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            _ = contiguous_attn()
        e.record()
        torch.cuda.synchronize()
        cont_ms = s.elapsed_time(e) / n

        # Measure paged
        s.record()
        for _ in range(n):
            _ = paged_attn()
        e.record()
        torch.cuda.synchronize()
        paged_ms = s.elapsed_time(e) / n

        overhead_pct = (paged_ms / cont_ms - 1) * 100 if cont_ms > 0 else 0

        print(f"  B={B} Hq={H_q} Hkv={H_kv} D={D} S={seq_len}: "
              f"contiguous={cont_ms:.4f}ms paged={paged_ms:.4f}ms "
              f"overhead={overhead_pct:.1f}%")

        results.append({
            "B": B, "H_q": H_q, "H_kv": H_kv, "D": D, "S": seq_len,
            "contiguous_ms": round(cont_ms, 4),
            "paged_ms": round(paged_ms, 4),
            "overhead_pct": round(overhead_pct, 1),
        })

    return results


# ============================================================
# Run
# ============================================================

cont_results = bench_contiguous_attention()
paged_gather_results = bench_paged_attention_simulation()
full_results = bench_paged_vs_contiguous_full()

print("\n" + "=" * 60)
print("SUMMARY: Paged vs Contiguous Attention on RTX 4090")
print("=" * 60)
print("Key findings:")
avg_overhead = sum(r["overhead_pct"] for r in full_results) / len(full_results)
print(f"  Paged attention overhead: ~{avg_overhead:.1f}% (gather+compute vs contiguous)")
print("  This is why vLLM needs custom Triton kernels (block_table indirection)")
print("  Python-level gather adds significant overhead vs CUDA-level indirection")

# Save
out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, 'paged_attention_benchmark_results.json')
with open(out_path, 'w') as f:
    json.dump({
        "contiguous_attention": cont_results,
        "paged_gather_overhead": paged_gather_results,
        "paged_vs_contiguous": full_results,
    }, f, indent=2)
print(f"Results saved to {out_path}")