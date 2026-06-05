#!/usr/bin/env python3
"""Fused Top-nσ Triton Kernel + Benchmark
=========================================
Implements Top-nσ sampling as a single fused Triton kernel:
  threshold = max(logits) - n * std(logits)
  logits[logits < threshold] = -inf

This replaces 3 separate ops (max, std, masked_fill) with 1 kernel launch.

Paper: Tang et al., ACL 2025, arXiv:2411.07641

Experiments:
1. Triton kernel correctness verification
2. Latency benchmark vs PyTorch baseline
3. Vocab size scaling (1K → 128K)
4. Batch size scaling
"""

import torch
import torch.nn.functional as F
import time
import json
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("WARNING: Triton not available, running PyTorch-only experiments")


# ============================================================
# PyTorch Baseline
# ============================================================

def top_n_sigma_pytorch(logits, n=2.0):
    """PyTorch baseline: 3 separate ops."""
    threshold = logits.max(dim=-1, keepdim=True).values - n * logits.std(dim=-1, keepdim=True)
    return logits.masked_fill(logits < threshold, float('-inf'))


# ============================================================
# Triton Fused Kernel
# ============================================================

if HAS_TRITON:
    @triton.jit
    def _top_n_sigma_kernel(
        logits_ptr, output_ptr, n_val: tl.constexpr,
        VOCAB_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused Top-nσ kernel: compute threshold and mask in one pass.

        Each program instance handles one (batch, head) row.
        Two-pass approach:
          Pass 1: compute sum, sum_sq, max
          Pass 2: apply threshold
        """
        row_idx = tl.program_id(0)

        # Pointers for this row
        row_start = row_idx * VOCAB_SIZE

        # Pass 1: compute statistics
        running_max = float('-inf')
        running_sum = 0.0
        running_sum_sq = 0.0

        for block_start in range(0, VOCAB_SIZE, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < VOCAB_SIZE
            vals = tl.load(logits_ptr + row_start + offsets, mask=mask, other=0.0)

            running_max = tl.maximum(running_max, tl.max(vals, axis=0))
            running_sum += tl.sum(vals, axis=0)
            running_sum_sq += tl.sum(vals * vals, axis=0)

        # Compute mean and std
        mean = running_sum / VOCAB_SIZE
        variance = running_sum_sq / VOCAB_SIZE - mean * mean
        std = tl.sqrt(tl.maximum(variance, 1e-10))

        # Threshold
        threshold = running_max - n_val * std

        # Pass 2: apply threshold
        for block_start in range(0, VOCAB_SIZE, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < VOCAB_SIZE
            vals = tl.load(logits_ptr + row_start + offsets, mask=mask, other=0.0)

            # Apply threshold
            out_vals = tl.where(vals < threshold, float('-inf'), vals)
            tl.store(output_ptr + row_start + offsets, out_vals, mask=mask)

    def top_n_sigma_triton(logits, n=2.0):
        """Triton fused Top-nσ."""
        orig_shape = logits.shape
        # Reshape to 2D: (B*H, V)
        x = logits.reshape(-1, orig_shape[-1])
        n_rows, vocab_size = x.shape
        output = torch.empty_like(x)

        # Choose block size
        BLOCK_SIZE = min(triton.next_power_of_2(vocab_size), 4096)

        grid = (n_rows,)
        _top_n_sigma_kernel[grid](
            x, output,
            n_val=n,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output.reshape(orig_shape)


# ============================================================
# Experiments
# ============================================================

def experiment1_correctness(device='cuda'):
    """Verify Triton kernel matches PyTorch baseline."""
    print("\n" + "="*70)
    print("Experiment 1: Correctness Verification")
    print("="*70)

    results = {}

    for vocab_size in [1024, 4096, 32000]:
        for n_val in [1.0, 2.0, 3.0]:
            torch.manual_seed(42)
            logits = torch.randn(4, vocab_size, device=device)

            out_pytorch = top_n_sigma_pytorch(logits.clone(), n=n_val)

            if HAS_TRITON:
                out_triton = top_n_sigma_triton(logits.clone(), n=n_val)
                max_err = (out_pytorch - out_triton).abs().max().item()
                cos_sim = F.cosine_similarity(
                    out_pytorch.flatten(), out_triton.flatten(), dim=0
                ).item()

                # Check threshold match
                thresh_pytorch = logits.max(dim=-1).values - n_val * logits.std(dim=-1)
                n_survive_pytorch = (out_pytorch > float('-inf')).sum(dim=-1).float().mean()
                n_survive_triton = (out_triton > float('-inf')).sum(dim=-1).float().mean()

                match = max_err < 1e-5
                print(f"  V={vocab_size:5d}, n={n_val:.1f}: max_err={max_err:.2e}, "
                      f"cos={cos_sim:.6f}, survive_pt={n_survive_pytorch:.0f} "
                      f"survive_tr={n_survive_triton:.0f} {'✓' if match else '✗'}")

                results[f'V{vocab_size}_n{n_val}'] = {
                    'max_err': max_err, 'cos_sim': cos_sim, 'match': match,
                }

    return results


def experiment2_latency(device='cuda'):
    """Benchmark Triton vs PyTorch latency."""
    print("\n" + "="*70)
    print("Experiment 2: Latency Benchmark")
    print("="*70)

    if not HAS_TRITON:
        print("  Triton not available, skipping")
        return {}

    results = {}

    for vocab_size in [1024, 4096, 8192, 32000, 128000]:
        for batch_size in [1, 8, 32, 128]:
            torch.manual_seed(42)
            logits = torch.randn(batch_size, vocab_size, device=device)

            # Warmup
            for _ in range(10):
                top_n_sigma_pytorch(logits.clone(), n=2.0)
                top_n_sigma_triton(logits.clone(), n=2.0)
            torch.cuda.synchronize()

            # PyTorch baseline
            n_trials = 100
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n_trials):
                top_n_sigma_pytorch(logits.clone(), n=2.0)
            torch.cuda.synchronize()
            t_pytorch = (time.time() - t0) / n_trials * 1e6  # microseconds

            # Triton
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n_trials):
                top_n_sigma_triton(logits.clone(), n=2.0)
            torch.cuda.synchronize()
            t_triton = (time.time() - t0) / n_trials * 1e6

            speedup = t_pytorch / t_triton

            results[f'V{vocab_size}_B{batch_size}'] = {
                'pytorch_us': t_pytorch,
                'triton_us': t_triton,
                'speedup': speedup,
            }

            if batch_size == 32 or (vocab_size == 128000 and batch_size in [1, 128]):
                print(f"  V={vocab_size:6d}, B={batch_size:3d}: "
                      f"pytorch={t_pytorch:8.1f}us, triton={t_triton:8.1f}us, "
                      f"speedup={speedup:.2f}x")

    # Summary table for key configs
    print(f"\n  LLM-typical configs (B=32):")
    print(f"  {'Vocab':>7} | {'PyTorch':>10} | {'Triton':>10} | {'Speedup':>8}")
    print(f"  {'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for vocab_size in [4096, 32000, 128000]:
        key = f'V{vocab_size}_B32'
        if key in results:
            r = results[key]
            print(f"  {vocab_size:7d} | {r['pytorch_us']:8.1f}us | "
                  f"{r['triton_us']:8.1f}us | {r['speedup']:6.2f}x")

    return results


def experiment3_kernel_ops_comparison(device='cuda'):
    """Compare: how many ops does fused kernel save?"""
    print("\n" + "="*70)
    print("Experiment 3: Operation Count Comparison")
    print("="*70)

    print("""
    PyTorch baseline (3 kernel launches):
      1. max(logits, dim=-1)          → 1 reduction kernel
      2. std(logits, dim=-1)          → 2 reduction kernels (mean + var)
      3. threshold = max - n * std    → 1 elementwise kernel
      4. masked_fill(logits < thresh) → 1 elementwise kernel
      Total: ~5 kernel launches + intermediate tensors

    Triton fused (1 kernel launch):
      1. _top_n_sigma_kernel          → 1 kernel, 2 passes (statistics + mask)
      Total: 1 kernel launch, 0 intermediate tensors

    Memory savings:
      PyTorch: max_tensor + std_tensor + threshold_tensor + mask_tensor = 4 intermediates
      Triton:  0 intermediates (everything in registers/SRAM)
    """)

    # Measure memory allocation difference
    vocab_size = 32000
    batch_size = 32
    logits = torch.randn(batch_size, vocab_size, device=device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    _ = top_n_sigma_pytorch(logits.clone(), n=2.0)
    torch.cuda.synchronize()
    pytorch_peak = torch.cuda.max_memory_allocated()

    if HAS_TRITON:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        _ = top_n_sigma_triton(logits.clone(), n=2.0)
        torch.cuda.synchronize()
        triton_peak = torch.cuda.max_memory_allocated()

        print(f"  V={vocab_size}, B={batch_size}:")
        print(f"    PyTorch peak memory: {pytorch_peak/1e6:.1f} MB")
        print(f"    Triton peak memory:  {triton_peak/1e6:.1f} MB")
        print(f"    Memory saving: {(pytorch_peak - triton_peak)/1e6:.1f} MB "
              f"({(1 - triton_peak/pytorch_peak)*100:.1f}%)")

    return {'pytorch_peak': pytorch_peak}


def experiment4_end_to_end_sampling(device='cuda'):
    """Full sampling pipeline benchmark: logits → sample."""
    print("\n" + "="*70)
    print("Experiment 4: End-to-End Sampling Pipeline")
    print("="*70)

    if not HAS_TRITON:
        print("  Triton not available, skipping")
        return {}

    results = {}
    vocab_size = 32000

    for batch_size in [1, 16, 64, 256]:
        logits = torch.randn(batch_size, vocab_size, device=device) * 2.0

        # Warmup
        for _ in range(10):
            filtered = top_n_sigma_pytorch(logits.clone(), n=2.0)
            probs = F.softmax(filtered, dim=-1)
            _ = torch.multinomial(probs, 1)
            filtered = top_n_sigma_triton(logits.clone(), n=2.0)
            probs = F.softmax(filtered, dim=-1)
            _ = torch.multinomial(probs, 1)
        torch.cuda.synchronize()

        n_trials = 100

        # PyTorch pipeline
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_trials):
            filtered = top_n_sigma_pytorch(logits.clone(), n=2.0)
            probs = F.softmax(filtered, dim=-1)
            tokens = torch.multinomial(probs, 1)
        torch.cuda.synchronize()
        t_pytorch = (time.time() - t0) / n_trials * 1e6

        # Triton pipeline
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_trials):
            filtered = top_n_sigma_triton(logits.clone(), n=2.0)
            probs = F.softmax(filtered, dim=-1)
            tokens = torch.multinomial(probs, 1)
        torch.cuda.synchronize()
        t_triton = (time.time() - t0) / n_trials * 1e6

        speedup = t_pytorch / t_triton
        results[f'B{batch_size}'] = {
            'pytorch_us': t_pytorch, 'triton_us': t_triton, 'speedup': speedup
        }
        print(f"  B={batch_size:3d}: pytorch={t_pytorch:.1f}us, triton={t_triton:.1f}us, "
              f"speedup={speedup:.2f}x (full pipeline: filter→softmax→sample)")

    return results


def run_all_experiments(device='cuda'):
    print("="*70)
    print("Fused Top-nσ Triton Kernel Benchmark")
    print("Paper: Tang et al., ACL 2025 (arXiv:2411.07641)")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton: {'available' if HAS_TRITON else 'NOT available'}")
    print("="*70)

    all_results = {}
    all_results['exp1_correctness'] = experiment1_correctness(device)
    all_results['exp2_latency'] = experiment2_latency(device)
    all_results['exp3_ops'] = experiment3_kernel_ops_comparison(device)
    all_results['exp4_e2e'] = experiment4_end_to_end_sampling(device)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if HAS_TRITON:
        exp2 = all_results.get('exp2_latency', {})
        if exp2:
            print("\n  Key latency results (B=32):")
            for v in [32000, 128000]:
                key = f'V{v}_B32'
                if key in exp2:
                    r = exp2[key]
                    print(f"    V={v}: {r['speedup']:.2f}x speedup "
                          f"({r['pytorch_us']:.0f}us → {r['triton_us']:.0f}us)")

        exp4 = all_results.get('exp4_e2e', {})
        if exp4:
            print("\n  End-to-end pipeline speedup:")
            for k, v in sorted(exp4.items()):
                print(f"    {k}: {v['speedup']:.2f}x")

    def convert(obj):
        if isinstance(obj, torch.Tensor): return obj.item()
        elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert(v) for v in obj]
        elif isinstance(obj, bool): return obj
        return obj

    with open('top_n_sigma_triton_results.json', 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print("\nResults saved to top_n_sigma_triton_results.json")
    return all_results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    run_all_experiments(device=device)
