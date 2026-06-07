#!/usr/bin/env python3
"""Sequence Length Scaling Experiment — RTX 4090

How training throughput, memory, and convergence scale with sequence length.

Test prompt lengths from short (1+1=, 4 tokens) to long (999+999=, many tokens)
and measure:
1. Throughput scaling (prefill theory: ∝ N², but small N may be memory-bound)
2. Peak GPU memory scaling (KV ∝ N, activation ∝ N)
3. Convergence quality (does longer context help?)
4. Attention pattern analysis (how does attention change with length)

Usage: On GPU server:
  CUDA_VISIBLE_DEVICES=0 python -u tools/sequence_length_scaling.py --model_size 2.28m
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.mini_grpo_training import (
    MiniGQATransformer, VOCAB_SIZE, TOKENS,
    generate_sft_dataset,
)


def generate_arithmetic_long(max_digits=1):
    """Generate arithmetic problem with variable digit count.
    max_digits=1: a+b= (4 tokens, short)
    max_digits=2: aa+bb= (6 tokens, medium)
    max_digits=3: aaa+bbb= (8 tokens, long)
    Returns (full_tokens, prompt_len, correct_sum)
    """
    a = np.random.randint(0, 10 ** max_digits)
    b = np.random.randint(0, 10 ** max_digits)

    # Convert to digit tokens
    a_str = str(a)
    b_str = str(b)

    prompt_tokens = []
    for d in a_str:
        prompt_tokens.append(TOKENS[d])
    prompt_tokens.append(TOKENS['+'])
    for d in b_str:
        prompt_tokens.append(TOKENS[d])
    prompt_tokens.append(TOKENS['='])

    correct_sum = a + b

    # Generate answer tokens
    sum_str = str(correct_sum)
    answer_tokens = []
    for d in sum_str:
        answer_tokens.append(TOKENS[d])
    answer_tokens.append(TOKENS['<eos>'])

    full_tokens = prompt_tokens + answer_tokens
    prompt_len = len(prompt_tokens)

    return full_tokens, prompt_len, correct_sum


def generate_padded_dataset(max_digits, num_samples=500, max_len=64):
    """Generate dataset with padding to fixed length for consistent measurement."""
    dataset = []
    for _ in range(num_samples):
        tokens, prompt_len, correct_sum = generate_arithmetic_long(max_digits)
        # Pad to max_len
        if len(tokens) < max_len:
            tokens = tokens + [TOKENS['<pad>']] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        dataset.append((tokens, prompt_len, correct_sum, max_len))
    return dataset


def train_with_seqlen(max_digits, model_size, num_steps, max_len=64,
                       lr=2e-3, dtype=torch.bfloat16):
    """Train with a specific sequence length configuration."""
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    np.random.seed(42)

    hd = 64 if model_size == '76k' else (256 if model_size == '2.28m' else 512)
    nl = 2 if model_size == '76k' else (4 if model_size == '2.28m' else 8)
    nh = 4 if model_size == '76k' else (8 if model_size == '2.28m' else 16)
    nkv = 2 if model_size == '76k' else (4 if model_size == '2.28m' else 8)

    model = MiniGQATransformer(
        hidden_dim=hd, num_layers=nl, num_heads=nh,
        num_kv_heads=nkv, vocab_size=VOCAB_SIZE).to(device=device, dtype=dtype)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset = generate_padded_dataset(max_digits, 500, max_len)

    torch.cuda.reset_peak_memory_stats()

    losses = []
    per_step_times = []
    start_time = time.time()

    for step in range(num_steps):
        idx = np.random.randint(len(dataset))
        tokens, prompt_len, correct_sum, seq_len = dataset[idx]
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        targets = input_ids.clone()

        step_start = time.time()
        logits = model(input_ids)

        # Only compute loss on non-padded answer tokens
        # Find answer positions (after prompt, before padding)
        answer_start = prompt_len - 1  # Shift position for next-token prediction
        # Find end of real tokens (exclude padding)
        real_end = prompt_len
        for t in tokens[prompt_len:]:
            if t != TOKENS['<pad>']:
                real_end += 1
            else:
                break

        if real_end > answer_start + 1:
            loss = F.cross_entropy(
                logits[:, answer_start:real_end-1, :].reshape(-1, VOCAB_SIZE),
                targets[:, answer_start+1:real_end].reshape(-1))
        else:
            # Degenerate case: no answer tokens to predict
            loss = F.cross_entropy(
                logits[:, answer_start, :].reshape(-1, VOCAB_SIZE),
                targets[:, answer_start+1].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
        per_step_times.append(time.time() - step_start)

    elapsed = time.time() - start_time
    throughput = num_steps / elapsed
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Eval
    model.eval()
    model_fp32 = model.float()
    correct = 0
    for _ in range(100):
        tokens, prompt_len, correct_sum = generate_arithmetic_long(max_digits)
        tokens_padded = tokens + [TOKENS['<pad>']] * (max_len - len(tokens)) if len(tokens) < max_len else tokens[:max_len]
        input_ids = torch.tensor([tokens_padded], dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model_fp32(input_ids)[0, prompt_len-1, :].argmax().item()
        if pred == TOKENS[str(correct_sum)[0]]:
            correct += 1
    eval_acc = correct / 100

    avg_digit_count = len(str(np.random.randint(0, 10**max_digits))) * 2 + 3

    result = {
        'max_digits': max_digits,
        'model_size': model_size,
        'seq_len': max_len,
        'avg_prompt_len': avg_digit_count,
        'num_steps': num_steps,
        'total_time_s': elapsed,
        'throughput_steps_s': throughput,
        'avg_step_time_ms': np.mean(per_step_times) * 1000,
        'final_loss': losses[-1],
        'min_loss': min(losses),
        'eval_accuracy': eval_acc,
        'peak_gpu_mem_gb': peak_mem,
        'losses_trajectory': losses[:10] + losses[-10:],
    }

    print(f"  digits={max_digits} seq={max_len}: "
          f"{throughput:.1f} steps/s, "
          f"avg_step={np.mean(per_step_times)*1000:.1f}ms, "
          f"loss={losses[-1]:.3f}, "
          f"eval={eval_acc:.1%}, "
          f"mem={peak_mem:.4f}GB")

    return result


def measure_forward_time(model_size, seq_len, dtype=torch.bfloat16, num_iters=50):
    """Measure pure forward pass time for different sequence lengths."""
    device = torch.device('cuda:0')

    hd = 64 if model_size == '76k' else (256 if model_size == '2.28m' else 512)
    nl = 2 if model_size == '76k' else (4 if model_size == '2.28m' else 8)
    nh = 4 if model_size == '76k' else (8 if model_size == '2.28m' else 16)
    nkv = 2 if model_size == '76k' else (4 if model_size == '2.28m' else 8)

    model = MiniGQATransformer(
        hidden_dim=hd, num_layers=nl, num_heads=nh,
        num_kv_heads=nkv, vocab_size=VOCAB_SIZE).to(device=device, dtype=dtype)
    model.eval()

    # Warmup
    for _ in range(5):
        x = torch.randint(0, VOCAB_SIZE, (1, seq_len), device=device)
        model(x)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        x = torch.randint(0, VOCAB_SIZE, (1, seq_len), device=device)
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    avg_time_ms = elapsed / num_iters * 1000

    # Memory
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, VOCAB_SIZE, (1, seq_len), device=device)
    with torch.no_grad():
        model(x)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # KV cache size estimate
    kv_bytes = 2 * nl * nkv * (hd // nh) * seq_len * 2  # BF16, 2 for K+V
    kv_mb = kv_bytes / 1e6

    return {
        'seq_len': seq_len,
        'avg_forward_ms': avg_time_ms,
        'peak_gpu_mem_gb': peak_mem,
        'kv_cache_mb': kv_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--model_size', default='2.28m',
                        choices=['76k', '2.28m'])
    parser.add_argument('--output', default='sequence_length_scaling_results.json')
    args = parser.parse_args()

    print("=" * 70)
    print("Sequence Length Scaling Experiment — RTX 4090")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}

    # Experiment 1: Forward pass time scaling
    print(f"\n--- Exp 1: Forward Pass Scaling ({args.model_size}) ---")
    seq_lengths = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for sl in seq_lengths:
        r = measure_forward_time(args.model_size, sl)
        key = f'forward_seq{sl}_{args.model_size}'
        results[key] = r
        print(f"  seq={sl}: forward={r['avg_forward_ms']:.2f}ms, "
              f"mem={r['peak_gpu_mem_gb']:.4f}GB, "
              f"KV={r['kv_cache_mb']:.1f}MB")

    # Experiment 2: Training with different digit widths
    print(f"\n--- Exp 2: Training Convergence ({args.model_size}) ---")
    for max_digits in [1, 2, 3]:
        # max_digits=1: 1-digit (a+b, ~4 tok prompt)
        # max_digits=2: 2-digit (aa+bb, ~6 tok prompt)
        # max_digits=3: 3-digit (aaa+bbb, ~8 tok prompt)
        max_len = max(16, (max_digits * 2 + 3 + 4) * 2)  # Ensure enough room
        result = train_with_seqlen(max_digits, args.model_size, args.num_steps, max_len)
        key = f'train_digits{max_digits}_seq{max_len}_{args.model_size}'
        results[key] = result

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nForward pass scaling:")
    baseline = results[f'forward_seq4_{args.model_size}']
    baseline_time = baseline['avg_forward_ms']
    for sl in seq_lengths:
        key = f'forward_seq{sl}_{args.model_size}'
        r = results[key]
        ratio = r['avg_forward_ms'] / baseline_time
        # Expected: prefill ∝ N² → ratio should be (sl/4)²
        expected_ratio = (sl / 4) ** 2
        print(f"  seq={sl}: {r['avg_forward_ms']:.2f}ms (ratio={ratio:.2f}, "
              f"expected_N²={expected_ratio:.2f}, "
              f"KV={r['kv_cache_mb']:.1f}MB, "
              f"mem={r['peak_gpu_mem_gb']:.4f}GB)")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()