#!/usr/bin/env python3
"""Pure GRPO Reward Function Experiment — No SFT Warmup

Compare 4 reward functions with PURE GRPO (no SFT warmup).
Previous experiment showed: when SFT is perfect, all rewards reach 100% → irrelevant.
This experiment tests: when model starts from scratch, reward design matters!

Run on GPU server (4 GPUs, one per reward function):
  CUDA_VISIBLE_DEVICES=0 python tools/reward_fn_pure_grpo.py --reward_fn binary &
  CUDA_VISIBLE_DEVICES=1 python tools/reward_fn_pure_grpo.py --reward_fn graded &
  CUDA_VISIBLE_DEVICES=2 python tools/reward_fn_pure_grpo.py --reward_fn shaped &
  CUDA_VISIBLE_DEVICES=3 python tools/reward_fn_pure_grpo.py --reward_fn curriculum &
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
    generate_arithmetic_prompt,
)


# ============================================================
# Reward Functions (same as reward_fn_quick.py)
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
    """Shaped: 1.0 correct, 0.5 ±1, 0.3 ±2, 0.2 any digit."""
    if len(generated_tokens) == 0:
        return 0.0
    first_token = generated_tokens[0]
    token_str = IDX_TO_TOKEN.get(first_token, '<unk>')
    if token_str == str(correct_sum):
        return 1.0
    try:
        val = int(token_str)
        if abs(val - correct_sum) == 1:
            return 0.5
        elif abs(val - correct_sum) <= 2:
            return 0.3
        else:
            return 0.2
    except (ValueError, TypeError):
        pass
    return 0.0


def compute_reward_curriculum(generated_tokens, correct_sum, step=0, total_steps=600):
    """Curriculum: shaped→graded→binary as training progresses."""
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


# ============================================================
# Pure GRPO Training Step (custom, with configurable reward)
# ============================================================

def grpo_training_step_pure(model, prompts, n_samples, max_response_len,
                             optimizer, device, reward_fn_name,
                             step=None, total_steps=600):
    """GRPO training step with custom reward function (no SFT warmup).
    Uses REINFORCE-style: collect log_probs with grad, then compute loss."""
    model.train()
    all_log_probs_with_grad = []  #log_probs with gradient tracking
    all_advantages = []
    all_rewards = []
    group_count = 0
    correct_count = 0
    total_count = 0

    for prompt_tokens, correct_sum in prompts:
        group_rewards = []
        group_log_probs = []

        for _ in range(n_samples):
            input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]],
                                      dtype=torch.long, device=device)
            # Sample response WITH gradient tracking for log_probs
            response_tokens = []
            current_ids = input_ids.clone()
            total_log_prob = 0.0

            for resp_step in range(max_response_len):
                logits = model(current_ids)
                next_logits = logits[:, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                sampled_token = torch.multinomial(probs, num_samples=1).item()

                # Compute log_prob WITH gradient (key for REINFORCE)
                log_prob = F.log_softmax(next_logits, dim=-1)[0, sampled_token]
                total_log_prob += log_prob  #This is a tensor with grad!
                response_tokens.append(sampled_token)

                if sampled_token == TOKENS['<eos>']:
                    break

                current_ids = torch.cat([current_ids,
                    torch.tensor([[sampled_token]], dtype=torch.long, device=device)], dim=1)

            # Compute reward
            if reward_fn_name == 'curriculum':
                reward = compute_reward_curriculum(
                    response_tokens, correct_sum, step=step, total_steps=total_steps)
            elif reward_fn_name == 'binary':
                reward = compute_reward_binary(response_tokens, correct_sum)
            elif reward_fn_name == 'graded':
                reward = compute_reward_graded(response_tokens, correct_sum)
            elif reward_fn_name == 'shaped':
                reward = compute_reward_shaped(response_tokens, correct_sum)
            else:
                reward = compute_reward_binary(response_tokens, correct_sum)

            group_rewards.append(reward)
            group_log_probs.append(total_log_prob)  #Tensor with grad

            if reward == 1.0:
                correct_count += 1
            total_count += 1

        # Group normalization (GRPO-style)
        if len(group_rewards) > 1:
            mean_r = np.mean(group_rewards)
            std_r = np.std(group_rewards)
            if std_r == 0:
                std_r = 1.0

            for r, lp in zip(group_rewards, group_log_probs):
                advantage = (r - mean_r) / std_r
                all_advantages.append(advantage)
                all_rewards.append(r)
                all_log_probs_with_grad.append(lp)  #Keep tensor with grad
            group_count += 1

    # Compute loss and update (log_probs already have grad)
    loss_val = 0.0
    if len(all_advantages) > 0:
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32, device=device)
        #Stack log_probs (they're tensors with grad)
        log_probs_tensor = torch.stack(all_log_probs_with_grad)

        loss = -(advantages_tensor * log_probs_tensor).mean()
        loss_val = loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return {
        'loss': loss_val,
        'reward_mean': np.mean(all_rewards) if all_rewards else 0.0,
        'reward_std': np.std(all_rewards) if all_rewards else 0.0,
        'accuracy': correct_count / total_count if total_count > 0 else 0.0,
        'advantage_mean': np.mean(all_advantages) if all_advantages else 0.0,
        'num_valid': len(all_advantages),
        'num_groups': group_count,
    }


def eval_model(model, device, n_eval=200):
    """Evaluate model accuracy (greedy)."""
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
    parser.add_argument('--num_steps', type=int, default=600)  #More steps since no SFT
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--reward_fn', required=True,
                        choices=['binary', 'graded', 'shaped', 'curriculum'])
    parser.add_argument('--output', default='reward_fn_pure_grpo_results.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print(f"Pure GRPO Reward Function Experiment: {args.reward_fn}")
    print("(NO SFT warmup — model starts from random init)")
    print("=" * 70)

    # Train from scratch (no SFT)
    torch.manual_seed(42)
    np.random.seed(42)
    model = MiniGQATransformer(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Baseline eval
    baseline_acc = eval_model(model, device, n_eval=100)
    print(f"\nBaseline eval (random init): {baseline_acc:.1%}")

    # Pure GRPO training
    print(f"\n--- Pure GRPO Training (reward={args.reward_fn}, lr={args.lr}) ---")
    metrics_history = []

    for step in range(args.num_steps):
        prompts = [generate_arithmetic_prompt() for _ in range(8)]
        metrics = grpo_training_step_pure(
            model, prompts, args.n_samples, 3, optimizer, device,
            reward_fn_name=args.reward_fn, step=step, total_steps=args.num_steps)
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
        'warmup': 'none',  #No SFT warmup
        'lr': args.lr,
        'grpo_steps': args.num_steps,
        'baseline_eval': baseline_acc,
        'final_eval_accuracy': final_eval,
        'initial_reward': rewards[0],
        'final_reward': rewards[-1],
        'peak_reward': max(rewards),
        'peak_accuracy': max(m['accuracy'] for m in metrics_history),
        'final_accuracy': metrics_history[-1]['accuracy'],
        'final_loss': metrics_history[-1]['loss'],
        'metrics_history': metrics_history,
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