#!/usr/bin/env python3
"""Learning Rate Schedule + Batch Size Scaling Experiment — RTX 4090

Compare 4 LR schedules × 3 batch size scaling rules on RTX 4090:
1. LR Schedules: constant, cosine, warmup+constant, warmup+cosine
2. Batch Size Scaling: no scaling (baseline), linear scaling (lr∝bs), sqrt scaling (lr∝√bs)
3. Also: AdamW vs SGD comparison for scaling rules

Key questions:
1. Does warmup help small model training?
2. Linear vs sqrt scaling: which works for AdamW?
3. Cosine vs constant: which converges better in 100 steps?
4. Does AdamW need different scaling than SGD?

Usage: On GPU server:
  CUDA_VISIBLE_DEVICES=0 python -u tools/lr_schedule_benchmark.py --model_size 76k
  CUDA_VISIBLE_DEVICES=0 python -u tools/lr_schedule_benchmark.py --model_size 2.28m
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
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


def get_lr_schedule(schedule_type, base_lr, num_steps, warmup_steps=10):
    """Return a list of LR values for each step."""
    lrs = []
    for step in range(num_steps):
        if schedule_type == 'constant':
            lrs.append(base_lr)
        elif schedule_type == 'cosine':
            # Cosine decay from base_lr to 0
            progress = step / num_steps
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        elif schedule_type == 'warmup_constant':
            if step < warmup_steps:
                lr = base_lr * (step + 1) / warmup_steps
            else:
                lr = base_lr
            lrs.append(lr)
        elif schedule_type == 'warmup_cosine':
            if step < warmup_steps:
                lr = base_lr * (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / (num_steps - warmup_steps)
                lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")
    return lrs


def train_with_schedule(schedule_type, base_lr, model_size, num_steps,
                         optimizer_type='adamw', grad_accum_steps=1,
                         warmup_steps=10, dtype=torch.bfloat16):
    """Train with a specific LR schedule and optimizer."""
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

    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.1)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9,
                                     weight_decay=0.1)
    elif optimizer_type == 'sgd_no_momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, weight_decay=0.1)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    dataset = generate_sft_dataset(500)
    lrs = get_lr_schedule(schedule_type, base_lr, num_steps, warmup_steps)

    torch.cuda.reset_peak_memory_stats()

    losses = []
    step_times = []
    start_time = time.time()

    for step in range(num_steps):
        # Update LR
        for pg in optimizer.param_groups:
            pg['lr'] = lrs[step]

        optimizer.zero_grad()
        accum_loss = 0.0

        for micro in range(grad_accum_steps):
            idx = np.random.randint(len(dataset))
            full_tokens, prompt_len = dataset[idx]
            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
            targets = input_ids.clone()

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits[:, prompt_len-1:-1, :].reshape(-1, VOCAB_SIZE),
                targets[:, prompt_len:].reshape(-1))
            loss = loss / grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(accum_loss * grad_accum_steps)

    elapsed = time.time() - start_time
    throughput = num_steps / elapsed
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Eval (cast to FP32 for fair comparison)
    model.eval()
    model_fp32 = model.float()
    correct = 0
    for _ in range(100):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]],
                                  dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model_fp32(input_ids)[0, -2, :].argmax().item()
        if pred == correct_sum:
            correct += 1
    eval_acc = correct / 100

    return {
        'schedule': schedule_type,
        'optimizer': optimizer_type,
        'base_lr': base_lr,
        'grad_accum_steps': grad_accum_steps,
        'effective_batch_size': grad_accum_steps,
        'model_size': model_size,
        'num_steps': num_steps,
        'total_time_s': elapsed,
        'throughput_steps_s': throughput,
        'final_loss': losses[-1],
        'min_loss': min(losses),
        'max_loss': max(losses),
        'mean_loss_last10': np.mean(losses[-10:]),
        'eval_accuracy': eval_acc,
        'peak_gpu_mem_gb': peak_mem,
        'lr_trajectory': lrs[:5] + lrs[-5:],  # sample LRs
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--model_size', default='76k',
                        choices=['76k', '2.28m'])
    parser.add_argument('--output', default='lr_schedule_results.json')
    args = parser.parse_args()

    print("=" * 70)
    print("LR Schedule + Batch Size Scaling Experiment — RTX 4090")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}

    # Experiment 1: LR Schedules (AdamW, base_lr=2e-3)
    print(f"\n--- Exp 1: LR Schedules ({args.model_size}, AdamW, lr=2e-3) ---")
    base_lr = 2e-3
    for schedule in ['constant', 'cosine', 'warmup_constant', 'warmup_cosine']:
        result = train_with_schedule(schedule, base_lr, args.model_size,
                                      args.num_steps, 'adamw')
        key = f'schedule_{schedule}_{args.model_size}'
        results[key] = result
        print(f"  {schedule}: loss={result['final_loss']:.3f}, "
              f"eval={result['eval_accuracy']:.1%}, "
              f"last10_mean={result['mean_loss_last10']:.3f}")

    # Experiment 2: Batch Size Scaling with AdamW
    print(f"\n--- Exp 2: Batch Size Scaling ({args.model_size}, AdamW) ---")
    base_lr = 2e-3
    for ga in [1, 2, 4, 8]:
        # No scaling
        result_no = train_with_schedule('warmup_constant', base_lr, args.model_size,
                                          args.num_steps, 'adamw', ga)
        key = f'scaling_no_ga{ga}_{args.model_size}'
        results[key] = result_no

        # Linear scaling: lr = base_lr * ga
        result_lin = train_with_schedule('warmup_constant', base_lr * ga, args.model_size,
                                           args.num_steps, 'adamw', ga)
        key = f'scaling_linear_ga{ga}_{args.model_size}'
        results[key] = result_lin

        # Sqrt scaling: lr = base_lr * sqrt(ga)
        result_sqrt = train_with_schedule('warmup_constant', base_lr * math.sqrt(ga),
                                            args.model_size, args.num_steps, 'adamw', ga)
        key = f'scaling_sqrt_ga{ga}_{args.model_size}'
        results[key] = result_sqrt

        print(f"  GA={ga}: no_scale={result_no['eval_accuracy']:.1%}, "
              f"linear(lr={base_lr*ga:.1e})={result_lin['eval_accuracy']:.1%}, "
              f"sqrt(lr={base_lr*math.sqrt(ga):.1e})={result_sqrt['eval_accuracy']:.1%}")

    # Experiment 3: AdamW vs SGD scaling
    print(f"\n--- Exp 3: AdamW vs SGD Scaling ({args.model_size}) ---")
    for opt in ['adamw', 'sgd']:
        for ga in [1, 4, 8]:
            lr_adamw = 2e-3
            lr_sgd = 0.1  # SGD needs much higher LR
            lr = lr_adamw if opt == 'adamw' else lr_sgd
            lr_scaled = lr * math.sqrt(ga)  # sqrt scaling for both
            result = train_with_schedule('warmup_constant', lr_scaled, args.model_size,
                                          args.num_steps, opt, ga)
            key = f'{opt}_sqrt_ga{ga}_{args.model_size}'
            results[key] = result
            print(f"  {opt} GA={ga}: loss={result['final_loss']:.3f}, "
                  f"eval={result['eval_accuracy']:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nExp 1: LR Schedules (AdamW, lr=2e-3)")
    for schedule in ['constant', 'cosine', 'warmup_constant', 'warmup_cosine']:
        r = results[f'schedule_{schedule}_{args.model_size}']
        print(f"  {schedule}: eval={r['eval_accuracy']:.1%}, "
              f"loss={r['final_loss']:.3f}, "
              f"last10={r['mean_loss_last10']:.3f}")

    print("\nExp 2: Batch Size Scaling (AdamW)")
    for ga in [1, 2, 4, 8]:
        no_r = results[f'scaling_no_ga{ga}_{args.model_size}']
        lin_r = results[f'scaling_linear_ga{ga}_{args.model_size}']
        sqrt_r = results[f'scaling_sqrt_ga{ga}_{args.model_size}']
        print(f"  GA={ga}: no={no_r['eval_accuracy']:.1%}, "
              f"linear={lin_r['eval_accuracy']:.1%}, "
              f"sqrt={sqrt_r['eval_accuracy']:.1%}")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()