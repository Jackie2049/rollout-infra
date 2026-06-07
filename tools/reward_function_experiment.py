#!/usr/bin/env python3
"""Reward Function Design Impact on RL Training — RTX 4090

Compare 4 reward functions and their impact on GRPO convergence:
1. Binary: 1 if correct, 0 otherwise (sparse, hard to learn from)
2. Graded: 1.0 correct, 0.3 ±1, 0.1 ±2 (dense, easier to learn from)
3. Shaped: 1.0 correct + 0.2 for generating any valid digit (densest)
4. Curriculum: starts with shaped → transitions to binary (curriculum learning)

Key questions:
1. Does dense reward help GRPO converge faster?
2. Does reward shaping cause reward hacking?
3. Is curriculum learning the best approach?
4. How does reward design interact with SFT warmstart?

All methods use SFT warmstart for fair comparison (since SFT is decisive).

Usage: On GPU server: python tools/reward_function_experiment.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import os
import sys
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.mini_grpo_training import (
    MiniGQATransformer, VOCAB_SIZE, TOKENS, IDX_TO_TOKEN,
    generate_arithmetic_prompt, grpo_training_step, sft_training_step,
    generate_sft_dataset,
)


# ============================================================
# Reward Functions
# ============================================================

def compute_reward_binary(generated_tokens, correct_sum):
    """Binary: 1 if correct, 0 otherwise."""
    if len(generated_tokens) == 0:
        return 0.0
    first_token = generated_tokens[0]
    token_str = IDX_TO_TOKEN.get(first_token, '<unk>')
    if token_str == str(correct_sum):
        return 1.0
    return 0.0


def compute_reward_graded(generated_tokens, correct_sum):
    """Graded: 1.0 correct, 0.3 ±1, 0.1 ±2."""
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
    """Shaped: 1.0 correct + 0.2 for generating any valid digit."""
    if len(generated_tokens) == 0:
        return 0.0
    first_token = generated_tokens[0]
    token_str = IDX_TO_TOKEN.get(first_token, '<unk>')
    # Base reward for generating any digit (exploration bonus)
    base = 0.0
    try:
        int(token_str)  # Is it a digit?
        base = 0.2
    except (ValueError, TypeError):
        pass
    # Correctness bonus
    if token_str == str(correct_sum):
        return 1.0
    # Proximity bonus
    try:
        val = int(token_str)
        if abs(val - correct_sum) == 1:
            return base + 0.3
        elif abs(val - correct_sum) <= 2:
            return base + 0.1
    except (ValueError, TypeError):
        pass
    return base


def compute_reward_curriculum(generated_tokens, correct_sum, step, total_steps=300):
    """Curriculum: shaped at start → binary at end."""
    progress = step / total_steps
    # Phase 1 (0-50%): shaped reward (easy to learn)
    # Phase 2 (50-80%): graded reward (transition)
    # Phase 3 (80-100%): binary reward (strict)
    if progress < 0.5:
        return compute_reward_shaped(generated_tokens, correct_sum)
    elif progress < 0.8:
        return compute_reward_graded(generated_tokens, correct_sum)
    else:
        return compute_reward_binary(generated_tokens, correct_sum)


# ============================================================
# Custom GRPO training step with configurable reward
# ============================================================

def grpo_training_step_custom(model, prompts, n_samples, max_response_len,
                               optimizer, device, reward_fn, step=None, total_steps=300):
    """GRPO training step with custom reward function."""
    model.train()
    all_rewards = []
    all_log_probs = []
    all_advantages = []
    group_count = 0
    correct_count = 0
    total_count = 0

    for prompt_tokens, correct_sum in prompts:
        group_rewards = []
        group_log_probs = []

        for _ in range(n_samples):
            input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]],
                                      dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids)

            # Sample response
            response_tokens = []
            current_ids = input_ids.clone()
            total_log_prob = 0.0

            for resp_step in range(max_response_len):
                with torch.no_grad():
                    logits = model(current_ids)
                    next_logits = logits[:, -1, :]
                    probs = F.softmax(next_logits, dim=-1)
                    sampled_token = torch.multinomial(probs, num_samples=1).item()

                log_prob = F.log_softmax(next_logits, dim=-1)[0, sampled_token].item()
                total_log_prob += log_prob
                response_tokens.append(sampled_token)

                if sampled_token == TOKENS['<eos>']:
                    break

                current_ids = torch.cat([current_ids,
                    torch.tensor([[sampled_token]], dtype=torch.long, device=device)], dim=1)

            # Compute reward with custom function
            if reward_fn == 'curriculum':
                reward = compute_reward_curriculum(
                    response_tokens, correct_sum, step=step, total_steps=total_steps)
            elif reward_fn == 'binary':
                reward = compute_reward_binary(response_tokens, correct_sum)
            elif reward_fn == 'graded':
                reward = compute_reward_graded(response_tokens, correct_sum)
            elif reward_fn == 'shaped':
                reward = compute_reward_shaped(response_tokens, correct_sum)
            else:
                reward = compute_reward_binary(response_tokens, correct_sum)

            group_rewards.append(reward)
            group_log_probs.append(total_log_prob)

            if reward == 1.0:
                correct_count += 1
            total_count += 1

        if len(group_rewards) > 1:
            mean_r = np.mean(group_rewards)
            std_r = np.std(group_rewards) if np.std(group_rewards) > 0 else 1.0

            for i, (r, lp) in enumerate(zip(group_rewards, group_log_probs)):
                advantage = (r - mean_r) / std_r
                all_advantages.append(advantage)
                all_rewards.append(r)
                all_log_probs.append(lp)
            group_count += 1

    # Compute loss
    if len(all_advantages) > 0:
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32, device=device)
        log_probs_tensor = torch.tensor(all_log_probs, dtype=torch.float32, device=device)

        loss = -(advantages_tensor * log_probs_tensor).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return {
        'loss': loss.item() if len(all_advantages) > 0 else 0.0,
        'reward_mean': np.mean(all_rewards) if all_rewards else 0.0,
        'reward_std': np.std(all_rewards) if all_rewards else 0.0,
        'accuracy': correct_count / total_count if total_count > 0 else 0.0,
        'advantage_mean': np.mean(all_advantages) if all_advantages else 0.0,
        'num_valid': len(all_advantages),
        'num_groups': group_count,
    }


def eval_model(model, device, n_eval=200, reward_fn='binary'):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    for _ in range(n_eval):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]],
                                  dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(input_ids)[0, -2, :].argmax().item()
        if pred == correct_sum:
            correct += 1
        total += 1
    return correct / total


def run_experiment(reward_fn, device, args, gpu_id=None):
    """Run one reward function experiment."""
    if gpu_id is not None:
        device = torch.device(f'cuda:{gpu_id}')

    torch.manual_seed(42)
    np.random.seed(42)
    model = MiniGQATransformer(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)

    # SFT warmup
    print(f"\n--- SFT warmup for reward={reward_fn} ---")
    sft_dataset = generate_sft_dataset(500)
    sft_optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(args.sft_steps):
        sft_training_step(model, sft_dataset, sft_optimizer, device)

    # GRPO training with custom reward
    print(f"\n--- GRPO training with reward={reward_fn} ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics_history = []
    for step in range(args.num_steps):
        prompts = [generate_arithmetic_prompt() for _ in range(args.num_prompts_per_step)]
        metrics = grpo_training_step_custom(
            model, prompts, args.n_samples, args.max_response_len,
            optimizer, device, reward_fn=reward_fn,
            step=step, total_steps=args.num_steps)
        metrics_history.append(metrics)

        if step % 50 == 0 or step == args.num_steps - 1:
            eval_acc = eval_model(model, device, n_eval=100)
            print(f"  Step {step:3d} | reward={reward_fn}: loss={metrics['loss']:.4f}, "
                  f"reward={metrics['reward_mean']:.3f}, "
                  f"acc={metrics['accuracy']:.1%}, eval={eval_acc:.1%}")

    # Final evaluation
    final_eval = eval_model(model, device, n_eval=200)
    print(f"\n  FINAL eval_acc ({reward_fn}): {final_eval:.1%}")

    return {
        'reward_fn': reward_fn,
        'metrics_history': metrics_history,
        'final_eval_accuracy': final_eval,
        'sft_steps': args.sft_steps,
        'grpo_steps': args.num_steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--sft_steps', type=int, default=200)
    parser.add_argument('--num_steps', type=int, default=300)
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--max_response_len', type=int, default=8)
    parser.add_argument('--num_prompts_per_step', type=int, default=8)
    parser.add_argument('--output', default='reward_function_results.json')
    parser.add_argument('--reward_fn', default='all',
                        choices=['binary', 'graded', 'shaped', 'curriculum', 'all'])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    results = {}

    reward_fns = ['binary', 'graded', 'shaped', 'curriculum'] if args.reward_fn == 'all' else [args.reward_fn]

    print("=" * 70)
    print("Reward Function Design Impact on RL Training — RTX 4090")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Reward functions: {reward_fns}")
    print(f"SFT warmup: {args.sft_steps} steps → GRPO: {args.num_steps} steps")
    print()

    for rf in reward_fns:
        result = run_experiment(rf, device, args)
        results[rf] = {
            'final_eval_accuracy': result['final_eval_accuracy'],
            'reward_fn': rf,
            'sft_steps': args.sft_steps,
            'grpo_steps': args.num_steps,
        }

        # Extract key metrics from history
        rewards = [m['reward_mean'] for m in result['metrics_history']]
        accs = [m['accuracy'] for m in result['metrics_history']]
        losses = [m['loss'] for m in result['metrics_history']]

        results[rf]['initial_reward'] = rewards[0]
        results[rf]['final_reward'] = rewards[-1]
        results[rf]['peak_reward'] = max(rewards)
        results[rf]['initial_accuracy'] = accs[0]
        results[rf]['final_accuracy'] = accs[-1]
        results[rf]['peak_accuracy'] = max(accs)
        results[rf]['final_loss'] = losses[-1]

        # Convergence speed: steps to reach 80% eval
        # (not stored in metrics, need separate eval)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("Reward Function Design Summary")
    print("=" * 70)
    for rf in reward_fns:
        r = results[rf]
        print(f"  {rf:12s}: eval_acc={r['final_eval_accuracy']:.1%}, "
              f"peak_reward={r['peak_reward']:.3f}, "
              f"final_reward={r['final_reward']:.3f}")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy values
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