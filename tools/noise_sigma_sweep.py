#!/usr/bin/env python3
"""Noise σ Sweep Experiment — RTX 4090

Scan different gradient noise σ values for SFT training:
σ = 0 (baseline), 0.001, 0.005, 0.01, 0.05, 0.1, 0.5

Goal: Find optimal noise level for 2.28M model at 300 steps.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u tools/noise_sigma_sweep.py --model_size 2.28m
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


def evaluate_model(model, device, num_samples=100):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    for _ in range(num_samples):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]],
                                  dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(input_ids)[0, -2, :].argmax().item()
        if pred == correct_sum:
            correct += 1
    model.train()
    return correct / num_samples


def train_with_noise(model_size, num_steps, lr=2e-3, noise_sigma=0.0):
    """Train SFT with specific noise sigma."""
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    np.random.seed(42)

    hd = 64 if model_size == '76k' else 256
    nl = 2 if model_size == '76k' else 4
    nh = 4 if model_size == '76k' else 8
    nkv = 2 if model_size == '76k' else 4

    model = MiniGQATransformer(
        hidden_dim=hd, num_layers=nl, num_heads=nh,
        num_kv_heads=nkv, vocab_size=VOCAB_SIZE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset = generate_sft_dataset(500)

    torch.cuda.reset_peak_memory_stats()
    losses = []
    eval_trajectory = []
    start_time = time.time()

    for step in range(num_steps):
        optimizer.zero_grad()
        idx = np.random.randint(len(dataset))
        full_tokens, prompt_len = dataset[idx]
        input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
        targets = input_ids.clone()

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits[:, prompt_len-1:-1, :].reshape(-1, VOCAB_SIZE),
            targets[:, prompt_len:].reshape(-1))

        loss.backward()

        if noise_sigma > 0:
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * noise_sigma)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 50 == 0 or step == num_steps - 1:
            eval_acc = evaluate_model(model, device, 50)
            eval_trajectory.append({'step': step, 'eval': eval_acc})

    elapsed = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    final_eval = evaluate_model(model, device, 100)

    return {
        'noise_sigma': noise_sigma,
        'model_size': model_size,
        'num_steps': num_steps,
        'final_loss': losses[-1],
        'min_loss': min(losses),
        'final_eval': final_eval,
        'peak_eval': max(e['eval'] for e in eval_trajectory),
        'eval_trajectory': eval_trajectory,
        'peak_gpu_mem_gb': peak_mem,
        'total_time_s': elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=300)
    parser.add_argument('--model_size', default='2.28m', choices=['76k', '2.28m'])
    parser.add_argument('--output', default='noise_sigma_sweep.json')
    args = parser.parse_args()

    print("=" * 70)
    print("Noise σ Sweep Experiment — RTX 4090")
    print("=" * 70)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_size}, Steps: {args.num_steps}")

    sigma_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    results = {}

    for sigma in sigma_values:
        label = f"σ={sigma}" if sigma > 0 else "baseline"
        print(f"\n[{sigma_values.index(sigma)+1}/{len(sigma_values)}] {label}")
        r = train_with_noise(args.model_size, args.num_steps, noise_sigma=sigma)
        results[f'sigma_{sigma}'] = r
        print(f"  eval={r['final_eval']:.1%}, peak={r['peak_eval']:.1%}, "
              f"loss={r['final_loss']:.3f}, time={r['total_time_s']:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Optimal σ for 2.28M SFT")
    print("=" * 70)

    baseline = results['sigma_0']
    print(f"  Baseline (σ=0): eval={baseline['final_eval']:.1%}, "
          f"peak={baseline['peak_eval']:.1%}")

    for sigma in sigma_values[1:]:
        r = results[f'sigma_{sigma}']
        delta = r['final_eval'] - baseline['final_eval']
        print(f"  σ={sigma}: eval={r['final_eval']:.1%} (Δ{delta:+.1%}), "
              f"peak={r['peak_eval']:.1%}, loss={r['final_loss']:.3f}")

    # Find best
    best_sigma = max(sigma_values,
                     key=lambda s: results[f'sigma_{s}']['final_eval'])
    best_result = results[f'sigma_{best_sigma}']
    print(f"\n  Best σ = {best_sigma}: eval = {best_result['final_eval']:.1%}")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()