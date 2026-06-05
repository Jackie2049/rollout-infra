#!/usr/bin/env python3
"""Tiled Attention — FlashAttention-inspired implementation
=============================================================
Implements attention with tiling to demonstrate FlashAttention concepts:
1. Naive O(N²) memory attention
2. Tiled O(N) memory attention (online softmax)
3. Numerical accuracy verification
4. Performance comparison

Educational purpose: understand FlashAttention's tiling & online softmax.
NOT a production implementation — real FlashAttention needs CUDA/Triton kernels.
"""

import torch
import torch.nn.functional as F
import math
import time


def naive_attention(Q, K, V):
    """Standard O(N²) memory attention."""
    d_k = Q.size(-1)
    S = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    P = F.softmax(S, dim=-1)
    O = P @ V
    return O


def tiled_attention(Q, K, V, block_size=64):
    """Tiled O(N) memory attention with online softmax.

    Demonstrates FlashAttention's core algorithm:
    1. Process Q in blocks of B_r rows
    2. For each Q block, iterate over K,V blocks
    3. Use online softmax to accumulate without storing full attention matrix

    Args:
        Q: (B, H, N, d) query
        K: (B, H, N, d) key
        V: (B, H, N, d) value
        block_size: tile size (B_r = B_c = block_size)
    """
    B, H, N, d = Q.shape
    d_k = d

    # Output and running statistics
    O = torch.zeros_like(Q)          # (B, H, N, d)
    m = torch.full((B, H, N, 1), float('-inf'), device=Q.device, dtype=Q.dtype)  # running max
    l = torch.zeros((B, H, N, 1), device=Q.device, dtype=Q.dtype)  # running sum

    # Outer loop: iterate over Q blocks
    for i_start in range(0, N, block_size):
        i_end = min(i_start + block_size, N)
        Q_i = Q[:, :, i_start:i_end, :]  # (B, H, B_r, d)

        # Initialize running stats for this Q block
        m_i = torch.full((B, H, i_end - i_start, 1), float('-inf'),
                          device=Q.device, dtype=Q.dtype)
        l_i = torch.zeros((B, H, i_end - i_start, 1),
                           device=Q.device, dtype=Q.dtype)
        O_i = torch.zeros((B, H, i_end - i_start, d),
                           device=Q.device, dtype=Q.dtype)

        # Inner loop: iterate over K, V blocks
        for j_start in range(0, N, block_size):
            j_end = min(j_start + block_size, N)
            K_j = K[:, :, j_start:j_end, :]  # (B, H, B_c, d)
            V_j = V[:, :, j_start:j_end, :]  # (B, H, B_c, d)

            # Compute local attention scores
            S_ij = Q_i @ K_j.transpose(-2, -1) / math.sqrt(d_k)  # (B, H, B_r, B_c)

            # Online softmax update
            m_ij = S_ij.max(dim=-1, keepdim=True).values  # (B, H, B_r, 1)
            m_new = torch.max(m_i, m_ij)

            # Correction factor for previous statistics
            exp_diff = torch.exp(m_i - m_new)  # 0 where m_i was -inf
            l_new = exp_diff * l_i + torch.exp(S_ij - m_new).sum(dim=-1, keepdim=True)

            # Update output
            P_ij = torch.exp(S_ij - m_new)  # (B, H, B_r, B_c)
            O_new = (exp_diff * l_i * O_i + P_ij @ V_j) / l_new

            # Update running stats
            m_i = m_new
            l_i = l_new
            O_i = O_new

        # Write final output for this Q block
        O[:, :, i_start:i_end, :] = O_i
        m[:, :, i_start:i_end, :] = m_i
        l[:, :, i_start:i_end, :] = l_i

    return O


def run_experiments(device='cuda'):
    print("=" * 70)
    print("Tiled Attention — FlashAttention Concept Verification")
    print(f"Device: {device}")
    print("=" * 70)

    results = {}

    # ----------------------------------------------------------
    # Experiment 1: Numerical Accuracy
    # ----------------------------------------------------------
    print("\n--- Experiment 1: Numerical Accuracy ---")

    torch.manual_seed(42)
    B, H, N, d = 2, 4, 256, 64
    Q = torch.randn(B, H, N, d, device=device)
    K = torch.randn(B, H, N, d, device=device)
    V = torch.randn(B, H, N, d, device=device)

    O_naive = naive_attention(Q, K, V)

    for bs in [16, 32, 64, 128]:
        O_tiled = tiled_attention(Q, K, V, block_size=bs)
        max_err = (O_naive - O_tiled).abs().max().item()
        mean_err = (O_naive - O_tiled).abs().mean().item()
        cos_sim = F.cosine_similarity(O_naive.flatten(), O_tiled.flatten(), dim=0).item()
        print(f"  block_size={bs:3d}: max_err={max_err:.2e}, mean_err={mean_err:.2e}, "
              f"cos_sim={cos_sim:.8f}")
        results[f'acc_bs{bs}'] = {'max_err': max_err, 'cos_sim': cos_sim}

    # ----------------------------------------------------------
    # Experiment 2: Memory Usage
    # ----------------------------------------------------------
    print("\n--- Experiment 2: Memory Usage ---")

    for N in [512, 1024, 2048]:
        torch.cuda.empty_cache()
        Q = torch.randn(1, 8, N, 64, device=device)
        K = torch.randn(1, 8, N, 64, device=device)
        V = torch.randn(1, 8, N, 64, device=device)

        # Naive memory
        torch.cuda.reset_peak_memory_stats()
        O1 = naive_attention(Q, K, V)
        mem_naive = torch.cuda.max_memory_allocated() / 1e6

        # Tiled memory
        torch.cuda.reset_peak_memory_stats()
        O2 = tiled_attention(Q, K, V, block_size=64)
        mem_tiled = torch.cuda.max_memory_allocated() / 1e6

        print(f"  N={N:4d}: naive={mem_naive:.1f}MB, tiled={mem_tiled:.1f}MB, "
              f"ratio={mem_naive/mem_tiled:.2f}x")

        results[f'mem_N{N}'] = {
            'naive_mb': mem_naive, 'tiled_mb': mem_tiled,
            'ratio': mem_naive / mem_tiled,
        }

        del Q, K, V, O1, O2

    # ----------------------------------------------------------
    # Experiment 3: Performance vs Sequence Length
    # ----------------------------------------------------------
    print("\n--- Experiment 3: Performance vs Sequence Length ---")

    for N in [256, 512, 1024, 2048]:
        Q = torch.randn(1, 8, N, 64, device=device)
        K = torch.randn(1, 8, N, 64, device=device)
        V = torch.randn(1, 8, N, 64, device=device)

        # Warmup
        for _ in range(5):
            naive_attention(Q, K, V)
            tiled_attention(Q, K, V, block_size=64)
        torch.cuda.synchronize()

        # Naive timing
        t0 = time.time()
        for _ in range(50):
            naive_attention(Q, K, V)
        torch.cuda.synchronize()
        t_naive = (time.time() - t0) / 50 * 1000

        # Tiled timing
        t0 = time.time()
        for _ in range(50):
            tiled_attention(Q, K, V, block_size=64)
        torch.cuda.synchronize()
        t_tiled = (time.time() - t0) / 50 * 1000

        # SDPA (FlashAttention) timing
        t0 = time.time()
        for _ in range(50):
            F.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()
        t_sdpa = (time.time() - t0) / 50 * 1000

        print(f"  N={N:4d}: naive={t_naive:.3f}ms, tiled={t_tiled:.3f}ms, "
              f"SDPA={t_sdpa:.3f}ms, tiled/naive={t_tiled/t_naive:.2f}x, "
              f"SDPA/naive={t_sdpa/t_naive:.2f}x")

        results[f'perf_N{N}'] = {
            'naive_ms': t_naive, 'tiled_ms': t_tiled, 'sdpa_ms': t_sdpa,
        }

    # ----------------------------------------------------------
    # Experiment 4: Block Size Effect
    # ----------------------------------------------------------
    print("\n--- Experiment 4: Block Size Effect ---")

    N = 1024
    Q = torch.randn(1, 8, N, 64, device=device)
    K = torch.randn(1, 8, N, 64, device=device)
    V = torch.randn(1, 8, N, 64, device=device)
    O_ref = naive_attention(Q, K, V)

    for bs in [16, 32, 64, 128, 256, 512]:
        O_t = tiled_attention(Q, K, V, block_size=bs)
        max_err = (O_ref - O_t).abs().max().item()

        # Timing
        for _ in range(3):
            tiled_attention(Q, K, V, block_size=bs)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            tiled_attention(Q, K, V, block_size=bs)
        torch.cuda.synchronize()
        t = (time.time() - t0) / 20 * 1000

        print(f"  bs={bs:3d}: time={t:.3f}ms, max_err={max_err:.2e}")
        results[f'bs_{bs}'] = {'time_ms': t, 'max_err': max_err}

    # ----------------------------------------------------------
    # Experiment 5: Causal Mask Verification
    # ----------------------------------------------------------
    print("\n--- Experiment 5: Causal Mask (mathematical note) ---")
    print("""
  FlashAttention with causal mask:
  - Only compute S_ij where row >= col (lower triangle)
  - For tile (i, j): if j_end <= i_start, skip (all masked)
  - If j_start >= i_end, fully visible (no mask needed)
  - Otherwise, apply mask within the tile

  Memory savings with causal:
    Full attention: O(N²) → O(N²/2) (triangle)
    Tiled: still O(N) regardless of mask

  Real FlashAttention supports causal in the CUDA kernel
  by simply skipping masked tiles → 2x speedup for causal.
    """)

    return results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    results = run_experiments(device=device)

    import json
    with open('tiled_attention_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to tiled_attention_results.json")
