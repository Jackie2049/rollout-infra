#!/usr/bin/env python3
"""Regularization Comparison Experiment — RTX 4090

Compare 5 regularization methods and their combinations:
1. No regularization (baseline)
2. Weight decay (wd=0.1, AdamW decoupled)
3. Dropout (p=0.1, 0.3)
4. Gradient noise injection (add Gaussian noise to gradients)
5. BF16 quantization noise (implicit regularization)

Key questions:
1. Which regularization helps most for small model convergence?
2. Does BF16 quantization noise act like regularization? (our hypothesis)
3. Do combinations (wd+dropout, wd+BF16) work better?
4. Can gradient noise injection substitute for larger batch size?

Usage: On GPU server:
  CUDA_VISIBLE_DEVICES=0 python -u tools/regularization_comparison.py --model_size 76k
  CUDA_VISIBLE_DEVICES=0 python -u tools/regularization_comparison.py --model_size 2.28m
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
    generate_arithmetic_prompt, generate_sft_dataset,
)


def train_with_regularization(reg_type, model_size, num_steps, lr=2e-3):
    """Train with a specific regularization configuration."""
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    np.random.seed(42)

    hd = 64 if model_size == '76k' else 256
    nl = 2 if model_size == '76k' else 4
    nh = 4 if model_size == '76k' else 8
    nkv = 2 if model_size == '76k' else 4

    # Parse regularization type
    use_bf16 = 'bf16' in reg_type
    use_dropout = 'dropout' in reg_type
    dropout_p = 0.3 if 'dropout03' in reg_type else (0.1 if 'dropout01' in reg_type else 0)
    use_grad_noise = 'grad_noise' in reg_type
    grad_noise_sigma = 0.01
    wd = 0.1 if 'wd' in reg_type else 0.0

    # Model dtype
    if use_bf16:
        # BF16 native: model in BF16, optimizer needs special handling
        # Use FP32 model + BF16 AMP for compatibility
        dtype = torch.float32
        use_amp_bf16 = True
    else:
        dtype = torch.float32
        use_amp_bf16 = False

    model = MiniGQATransformer(
        hidden_dim=hd, num_layers=nl, num_heads=nh,
        num_kv_heads=nkv, vocab_size=VOCAB_SIZE).to(device)

    # Apply dropout if needed
    if use_dropout and dropout_p > 0:
        # Add dropout after each layer's output
        # We'll inject it in the forward pass manually
        model._dropout_p = dropout_p
    else:
        model._dropout_p = 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    dataset = generate_sft_dataset(500)

    torch.cuda.reset_peak_memory_stats()
    losses = []
    start_time = time.time()

    for step in range(num_steps):
        optimizer.zero_grad()

        idx = np.random.randint(len(dataset))
        full_tokens, prompt_len = dataset[idx]
        input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
        targets = input_ids.clone()

        if use_amp_bf16:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, prompt_len-1:-1, :].reshape(-1, VOCAB_SIZE),
                    targets[:, prompt_len:].reshape(-1))
        else:
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits[:, prompt_len-1:-1, :].reshape(-1, VOCAB_SIZE),
                targets[:, prompt_len:].reshape(-1))

        loss.backward()

        # Gradient noise injection
        if use_grad_noise:
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * grad_noise_sigma)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    elapsed = time.time() - start_time
    throughput = num_steps / elapsed
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Eval
    model.eval()
    correct = 0
    for _ in range(100):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]],
                                  dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(input_ids)[0, -2, :].argmax().item()
        if pred == correct_sum:
            correct += 1
    eval_acc = correct / 100

    result = {
        'reg_type': reg_type,
        'model_size': model_size,
        'weight_decay': wd,
        'dropout': dropout_p,
        'grad_noise_sigma': grad_noise_sigma if use_grad_noise else 0,
        'use_bf16': use_bf16,
        'num_steps': num_steps,
        'total_time_s': elapsed,
        'throughput_steps_s': throughput,
        'final_loss': losses[-1],
        'min_loss': min(losses),
        'eval_accuracy': eval_acc,
        'peak_gpu_mem_gb': peak_mem,
        'loss_std': np.std(losses),
    }

    print(f"  {reg_type}: {throughput:.1f} steps/s, "
          f"loss={losses[-1]:.3f}, eval={eval_acc:.1%}, "
          f"mem={peak_mem:.4f}GB, "
          f"wd={wd}, dropout={dropout_p}, bf16={use_bf16}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--model_size', default='76k', choices=['76k', '2.28m'])
    parser.add_argument('--output', default='regularization_comparison_results.json')
    args = parser.parse_args()

    print("=" * 70)
    print("Regularization Comparison Experiment — RTX 4090")
    print("=" * 70)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}

    # All regularization configurations to test
    reg_configs = [
        'none',           # No regularization
        'wd',             # Weight decay only (wd=0.1)
        'dropout01',      # Dropout p=0.1 only
        'dropout03',      # Dropout p=0.3 only
        'grad_noise',     # Gradient noise σ=0.01 only
        'bf16',           # BF16 quantization noise only
        'wd_dropout01',   # WD + Dropout
        'wd_bf16',        # WD + BF16
        'dropout01_bf16', # Dropout + BF16
        'wd_dropout01_bf16',  # All three combined!
    ]

    print(f"\n--- Model: {args.model_size}, lr=2e-3 ---")
    for reg in reg_configs:
        result = train_with_regularization(reg, args.model_size, args.num_steps)
        key = f'reg_{reg}_{args.model_size}'
        results[key] = result

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    baseline = results[f'reg_none_{args.model_size}']
    baseline_eval = baseline['eval_accuracy']

    for reg in reg_configs:
        r = results[f'reg_{reg}_{args.model_size}']
        eval_diff = r['eval_accuracy'] - baseline_eval
        loss_diff = r['final_loss'] - baseline['final_loss']
        speedup = r['throughput_steps_s'] / baseline['throughput_steps_s']
        print(f"  {reg}: eval={r['eval_accuracy']:.1%} (Δ{eval_diff:+.1%}), "
              f"loss={r['final_loss']:.3f} (Δ{loss_diff:+.3f}), "
              f"speed={speedup:.2f}x, "
              f"wd={r['weight_decay']}, drop={r['dropout']}, "
              f"bf16={r['use_bf16']}")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()