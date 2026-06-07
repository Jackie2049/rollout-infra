#!/usr/bin/env python3
"""Batch Size Scaling + Convergence Experiment — RTX 4090

How training convergence and throughput scale with effective batch size
(using gradient accumulation as batch size proxy).

Test GA=1/2/4/8/16/32/64 with FIXED total compute budget (1000 micro-steps).

Usage: On GPU server:
  CUDA_VISIBLE_DEVICES=0 python -u tools/batch_size_scaling.py --model_size 76k
  CUDA_VISIBLE_DEVICES=0 python -u tools/batch_size_scaling.py --model_size 2.28m
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


def train_with_batch_size(grad_accum_steps, model_size, num_micro_steps=1000, lr=2e-3):
    """Train with a specific effective batch size (via gradient accumulation)."""
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    np.random.seed(42)

    hd = 64 if model_size == '76k' else 256
    nl = 2 if model_size == '76k' else 4
    nh = 4 if model_size == '76k' else 8
    nkv = 2 if model_size == '76k' else 4

    # Use FP32 model + BF16 AMP (compatible with Adam optimizer)
    model = MiniGQATransformer(
        hidden_dim=hd, num_layers=nl, num_heads=nh,
        num_kv_heads=nkv, vocab_size=VOCAB_SIZE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset = generate_sft_dataset(500)

    torch.cuda.reset_peak_memory_stats()

    num_optimizer_steps = num_micro_steps // grad_accum_steps
    losses = []
    eval_checkpoints = []
    grad_norms = []
    step_times = []

    start_time = time.time()

    for opt_step in range(num_optimizer_steps):
        optimizer.zero_grad()
        accum_loss = 0.0

        step_start = time.time()
        for micro in range(grad_accum_steps):
            idx = np.random.randint(len(dataset))
            full_tokens, prompt_len = dataset[idx]
            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
            targets = input_ids.clone()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, prompt_len-1:-1, :].reshape(-1, VOCAB_SIZE),
                    targets[:, prompt_len:].reshape(-1))
            loss = loss / grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm))
        optimizer.step()
        losses.append(accum_loss * grad_accum_steps)
        step_times.append(time.time() - step_start)

        # Eval at key milestones
        if opt_step in [0, 4, 9, 19, 49, 99] or opt_step == num_optimizer_steps - 1:
            model.eval()
            correct = 0
            for _ in range(50):
                prompt_tokens, correct_sum = generate_arithmetic_prompt()
                input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]],
                                          dtype=torch.long, device=device)
                with torch.no_grad():
                    pred = model(input_ids)[0, -2, :].argmax().item()
                if pred == correct_sum:
                    correct += 1
            eval_acc = correct / 50
            eval_checkpoints.append({
                'opt_step': opt_step,
                'micro_step': opt_step * grad_accum_steps,
                'loss': losses[-1],
                'eval_accuracy': eval_acc,
            })
            model.train()

    elapsed = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Final eval
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
    final_eval = correct / 100

    result = {
        'grad_accum_steps': grad_accum_steps,
        'effective_batch_size': grad_accum_steps,
        'model_size': model_size,
        'num_micro_steps': num_micro_steps,
        'num_optimizer_steps': num_optimizer_steps,
        'total_time_s': elapsed,
        'throughput_opt_steps_s': num_optimizer_steps / elapsed,
        'throughput_micro_steps_s': num_micro_steps / elapsed,
        'avg_opt_step_time_ms': np.mean(step_times) * 1000,
        'final_loss': losses[-1],
        'min_loss': min(losses),
        'final_eval_accuracy': final_eval,
        'peak_eval_accuracy': max(e['eval_accuracy'] for e in eval_checkpoints),
        'peak_gpu_mem_gb': peak_mem,
        'mean_grad_norm': np.mean(grad_norms),
        'loss_variance': np.std(losses),
        'eval_trajectory': eval_checkpoints,
    }

    print(f"  GA={grad_accum_steps} (bs={grad_accum_steps}): "
          f"{num_optimizer_steps} opt_steps in {elapsed:.1f}s, "
          f"final_eval={final_eval:.1%}, "
          f"peak_eval={result['peak_eval_accuracy']:.1%}, "
          f"loss={losses[-1]:.3f}, "
          f"mem={peak_mem:.4f}GB")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_micro_steps', type=int, default=1000)
    parser.add_argument('--model_size', default='76k', choices=['76k', '2.28m'])
    parser.add_argument('--output', default='batch_size_scaling_results.json')
    args = parser.parse_args()

    print("=" * 70)
    print("Batch Size Scaling + Convergence Experiment — RTX 4090")
    print("=" * 70)
    print(f"Total compute budget: {args.num_micro_steps} micro-steps (fixed)")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}
    ga_values = [1, 2, 4, 8, 16, 32, 64]

    print(f"\n--- Model: {args.model_size}, lr=2e-3 ---")
    for ga in ga_values:
        if args.num_micro_steps % ga != 0:
            continue
        result = train_with_batch_size(ga, args.model_size, args.num_micro_steps)
        key = f'ga{ga}_{args.model_size}'
        results[key] = result

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Convergence vs Batch Size (same total compute)")
    print("=" * 70)

    for ga in ga_values:
        key = f'ga{ga}_{args.model_size}'
        if key not in results:
            continue
        r = results[key]
        print(f"  GA={ga}: opt_steps={r['num_optimizer_steps']}, "
              f"eval={r['final_eval_accuracy']:.1%}, "
              f"peak_eval={r['peak_eval_accuracy']:.1%}, "
              f"loss={r['final_loss']:.3f}, "
              f"micro/s={r['throughput_micro_steps_s']:.0f}")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()