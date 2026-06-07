#!/usr/bin/env python3
"""Reward Function Design Experiment — Quick approach using monkey-patching.

Run each reward function separately on different GPUs, using the existing
grpo_training_step by monkey-patching compute_reward.

Usage: On GPU server with 4 GPUs available:
  CUDA_VISIBLE_DEVICES=0 python tools/reward_fn_quick.py --reward_fn binary --gpu 0 &
  CUDA_VISIBLE_DEVICES=1 python tools/reward_fn_quick.py --reward_fn graded --gpu 1 &
  CUDA_VISIBLE_DEVICES=2 python tools/reward_fn_quick.py --reward_fn shaped --gpu 2 &
  CUDA_VISIBLE_DEVICES=3 python tools/reward_fn_quick.py --reward_fn curriculum --gpu 3 &
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.mini_grpo_training import (
    MiniGQATransformer, VOCAB_SIZE, TOKENS, IDX_TO_TOKEN,
    generate_arithmetic_prompt, grpo_training_step, sft_training_step,
    generate_sft_dataset, compute_reward,
)


# ============================================================
# Reward Functions
# ============================================================

def compute_reward_binary(generated_tokens, correct_sum):
    """Binary: 1 if correct, 0 otherwise (sparse reward)."""
    if len(generated_tokens) == 0:
        return 0.0
    first_token = generated_tokens[0]
    token_str = IDX_TO_TOKEN.get(first_token, '<unk>')
    if token_str == str(correct_sum):
        return 1.0
    return 0.0


def compute_reward_graded(generated_tokens, correct_sum):
    """Graded: 1.0 correct, 0.3 ±1, 0.1 ±2 (default)."""
    if len(generated_tokens) == 0:
        return 0.0
    first_token = generated_tokens[0]
    token_str = IDX_TO_TOKEN.get(first_token, '<unk>')
    if token_str == str(correct_sum):
        return 1.0
    try:
        val = int(token_str)
        if abs(val - correct_sum) == 1:
            return 0.3
        elif abs(val - correct_sum) <= 2:
            return 0.1
    except (ValueError, TypeError):
        pass
    return 0.0


def compute_reward_shaped(generated_tokens, correct_sum):
    """Shaped: correct=1.0, any_digit=0.2, ±1=0.5, ±2=0.3."""
    if len(generated_tokens) == 0:
        return 0.0
    first_token = generated_tokens[0]
    token_str = IDX_TO_TOKEN.get(first_token, '<unk>')
    if token_str == str(correct_sum):
        return 1.0
    try:
        val = int(token_str)
        if abs(val - correct_sum) == 1:
            return 0.5  # Higher partial credit
        elif abs(val - correct_sum) <= 2:
            return 0.3
        else:
            return 0.2  # Any digit gets small reward (exploration bonus)
    except (ValueError, TypeError):
        pass
    return 0.0


def compute_reward_curriculum(generated_tokens, correct_sum, step=0, total_steps=300):
    """Curriculum: shaped → graded → binary as training progresses."""
    progress = step / total_steps
    if progress < 0.33:
        return compute_reward_shaped(generated_tokens, correct_sum)
    elif progress < 0.67:
        return compute_reward_graded(generated_tokens, correct_sum)
    else:
        return compute_reward_binary(generated_tokens, correct_sum)


REWARD_FUNCTIONS = {
    'binary': compute_reward_binary,
    'graded': compute_reward_graded,
    'shaped': compute_reward_shaped,
}


def eval_model(model, device, n_eval=200):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    for _ in range(n_eval):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]], dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(input_ids)[0, -2, :].argmax().item()
        if pred == correct_sum:
            correct += 1
        total += 1
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--sft_steps', type=int, default=200)
    parser.add_argument('--num_steps', type=int, default=300)
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--reward_fn', required=True,
                        choices=['binary', 'graded', 'shaped', 'curriculum'])
    parser.add_argument('--output', default='reward_fn_results.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print(f"Reward Function Experiment: {args.reward_fn}")
    print("=" * 70)

    # Monkey-patch the reward function
    import tools.mini_grpo_training as training_module

    if args.reward_fn == 'curriculum':
        # For curriculum, we need step-dependent reward
        # We'll patch it in the training loop
        original_compute_reward = training_module.compute_reward
    else:
        training_module.compute_reward = REWARD_FUNCTIONS[args.reward_fn]
        print(f"Patched compute_reward to {args.reward_fn}")

    # Train model with SFT warmup + GRPO
    torch.manual_seed(42)
    np.random.seed(42)
    model = MiniGQATransformer(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)

    # SFT warmup
    print("\n--- Phase 1: SFT Warmup ---")
    sft_dataset = generate_sft_dataset(500)
    sft_optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(args.sft_steps):
        sft_training_step(model, sft_dataset, sft_optimizer, device)
        if step % 50 == 0:
            acc = eval_model(model, device, n_eval=50)
            print(f"  SFT step {step}: eval_acc={acc:.1%}")

    # Switch to GRPO optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # GRPO training
    print(f"\n--- Phase 2: GRPO Training (reward={args.reward_fn}) ---")
    metrics_history = []

    for step in range(args.num_steps):
        prompts = [generate_arithmetic_prompt() for _ in range(8)]

        # For curriculum: patch reward function based on step
        if args.reward_fn == 'curriculum':
            def curriculum_reward(tokens, correct_sum, _step=step, _total=args.num_steps):
                return compute_reward_curriculum(tokens, correct_sum, _step, _total)
            training_module.compute_reward = curriculum_reward

        metrics = grpo_training_step(
            model, prompts, args.n_samples, 3, optimizer, device)
        metrics_history.append(metrics)

        if step % 50 == 0 or step == args.num_steps - 1:
            acc = eval_model(model, device, n_eval=100)
            print(f"  Step {step:3d}: reward={metrics['reward_mean']:.3f}, "
                  f"acc={metrics['accuracy']:.1%}, eval={acc:.1%}")

    # Final eval
    final_eval = eval_model(model, device, n_eval=200)
    print(f"\nFinal eval_acc ({args.reward_fn}): {final_eval:.1%}")

    # Save results
    rewards = [m['reward_mean'] for m in metrics_history]
    results = {
        'reward_fn': args.reward_fn,
        'sft_steps': args.sft_steps,
        'grpo_steps': args.num_steps,
        'final_eval_accuracy': final_eval,
        'initial_reward': rewards[0],
        'final_reward': rewards[-1],
        'peak_reward': max(rewards),
        'peak_accuracy': max(m['accuracy'] for m in metrics_history),
        'final_accuracy': metrics_history[-1]['accuracy'],
        'final_loss': metrics_history[-1]['loss'],
    }

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (float, np.floating)):
            return float(obj)
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()