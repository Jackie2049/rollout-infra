#!/usr/bin/env python3
"""GRPO Advantage = Adaptive Gradient Noise Verification — RTX 4090

Verify the theoretical connection:
  GRPO A=(r-μ)/σ ≈ adaptive gradient noise injection

Compare 5 strategies:
1. SFT baseline (no RL, no noise)
2. GRPO with σ-normalization (A=(r-μ)/σ) — standard GRPO
3. GRPO without σ-normalization (A=r-μ) — unnormalized
4. SFT + fixed gradient noise (σ=0.01) — our best regularization
5. SFT + adaptive gradient noise (σ proportional to loss variance) — mimics GRPO

If GRPO σ-normalization ≈ adaptive gradient noise,
then strategy 5 should match strategy 2's convergence quality.

Usage: On GPU server:
  CUDA_VISIBLE_DEVICES=0 python -u tools/grpo_advantage_noise_verification.py --model_size 76k
  CUDA_VISIBLE_DEVICES=0 python -u tools/grpo_advantage_noise_verification.py --model_size 2.28m
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
    """Evaluate model accuracy on arithmetic task."""
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


def train_sft_baseline(model_size, num_steps, lr=2e-3):
    """Strategy 1: SFT baseline (no noise)."""
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 25 == 0 or step == num_steps - 1:
            eval_acc = evaluate_model(model, device, 50)
            eval_trajectory.append({'step': step, 'eval': eval_acc})

    elapsed = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    final_eval = evaluate_model(model, device, 100)

    return {
        'strategy': 'sft_baseline',
        'model_size': model_size,
        'final_loss': losses[-1],
        'final_eval': final_eval,
        'peak_eval': max(e['eval'] for e in eval_trajectory),
        'eval_trajectory': eval_trajectory,
        'peak_gpu_mem_gb': peak_mem,
        'total_time_s': elapsed,
    }


def train_sft_fixed_noise(model_size, num_steps, lr=2e-3, noise_sigma=0.01):
    """Strategy 4: SFT + fixed gradient noise."""
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

        # Fixed gradient noise injection
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * noise_sigma)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 25 == 0 or step == num_steps - 1:
            eval_acc = evaluate_model(model, device, 50)
            eval_trajectory.append({'step': step, 'eval': eval_acc})

    elapsed = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    final_eval = evaluate_model(model, device, 100)

    return {
        'strategy': 'sft_fixed_noise',
        'noise_sigma': noise_sigma,
        'model_size': model_size,
        'final_loss': losses[-1],
        'final_eval': final_eval,
        'peak_eval': max(e['eval'] for e in eval_trajectory),
        'eval_trajectory': eval_trajectory,
        'peak_gpu_mem_gb': peak_mem,
        'total_time_s': elapsed,
    }


def train_sft_adaptive_noise(model_size, num_steps, lr=2e-3, base_sigma=0.01):
    """Strategy 5: SFT + adaptive gradient noise (σ scales with loss variance).

    Mimics GRPO: σ proportional to reward/group_std.
    Here we use loss variance over recent steps as proxy for reward variance.
    """
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
    noise_sigmas = []
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

        # Adaptive gradient noise: σ scales with recent loss variance
        if len(losses) >= 5:
            recent_std = np.std(losses[-5:])
            # Scale noise: more variance → more exploration noise
            adaptive_sigma = base_sigma * max(recent_std, 0.1)
        else:
            adaptive_sigma = base_sigma

        noise_sigmas.append(adaptive_sigma)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * adaptive_sigma)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 25 == 0 or step == num_steps - 1:
            eval_acc = evaluate_model(model, device, 50)
            eval_trajectory.append({'step': step, 'eval': eval_acc})

    elapsed = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    final_eval = evaluate_model(model, device, 100)

    return {
        'strategy': 'sft_adaptive_noise',
        'base_sigma': base_sigma,
        'mean_sigma': np.mean(noise_sigmas),
        'max_sigma': max(noise_sigmas),
        'model_size': model_size,
        'final_loss': losses[-1],
        'final_eval': final_eval,
        'peak_eval': max(e['eval'] for e in eval_trajectory),
        'eval_trajectory': eval_trajectory,
        'peak_gpu_mem_gb': peak_mem,
        'total_time_s': elapsed,
    }


def train_grpo_normalized(model_size, num_steps, n_samples=4, lr=2e-3):
    """Strategy 2: GRPO with σ-normalization (standard).

    Uses n different dataset samples per group (simplified GRPO).
    Reward = continuous (softmax probability of correct answer) — avoids
    binary reward causing zero advantages when all samples same reward.
    Advantage A = (r - μ_r) / σ_r.
    Loss = -Σ A_i * log π(response_i).
    """
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

        group_rewards = []
        group_log_probs = []

        for sample_idx in range(n_samples):
            idx = np.random.randint(len(dataset))
            full_tokens, prompt_len = dataset[idx]
            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)

            logits = model(input_ids)

            # Get log probabilities for response tokens
            response_logits = logits[:, prompt_len-1:-1, :]
            log_probs = F.log_softmax(response_logits, dim=-1)
            target_tokens = input_ids[:, prompt_len:]
            token_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
            total_log_prob = token_log_probs.sum()

            # Continuous reward: softmax probability of correct answer
            # This ensures different samples almost always have different rewards
            answer_logits = logits[0, prompt_len-1, :]
            answer_probs = F.softmax(answer_logits, dim=-1)
            correct_answer_token = full_tokens[prompt_len]
            reward = answer_probs[correct_answer_token].item()
            group_rewards.append(reward)
            group_log_probs.append(total_log_prob)

        # Compute advantages with σ-normalization (GRPO standard)
        rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32, device=device)
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_r) / std_r

        # Compute loss: -Σ A_i * log π(response)
        total_loss = torch.tensor(0.0, device=device)
        for i in range(n_samples):
            adv_i = advantages[i]
            log_prob_i = group_log_probs[i]
            total_loss = total_loss - adv_i * log_prob_i

        total_loss = total_loss / n_samples

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(total_loss.item())

        if step % 25 == 0 or step == num_steps - 1:
            eval_acc = evaluate_model(model, device, 50)
            eval_trajectory.append({'step': step, 'eval': eval_acc})

    elapsed = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    final_eval = evaluate_model(model, device, 100)

    return {
        'strategy': 'grpo_normalized',
        'n_samples': n_samples,
        'reward_type': 'continuous_softmax_prob',
        'model_size': model_size,
        'final_loss': losses[-1],
        'final_eval': final_eval,
        'peak_eval': max(e['eval'] for e in eval_trajectory),
        'eval_trajectory': eval_trajectory,
        'peak_gpu_mem_gb': peak_mem,
        'total_time_s': elapsed,
    }


def train_grpo_unnormalized(model_size, num_steps, n_samples=4, lr=2e-3):
    """Strategy 3: GRPO without σ-normalization (A=r-μ only).

    Same as train_grpo_normalized but advantages = r - μ (no σ division).
    Uses continuous reward (softmax prob) to avoid zero-advantage problem.
    """
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

        group_rewards = []
        group_log_probs = []

        for sample_idx in range(n_samples):
            idx = np.random.randint(len(dataset))
            full_tokens, prompt_len = dataset[idx]
            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)

            logits = model(input_ids)

            response_logits = logits[:, prompt_len-1:-1, :]
            log_probs = F.log_softmax(response_logits, dim=-1)
            target_tokens = input_ids[:, prompt_len:]
            token_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
            total_log_prob = token_log_probs.sum()

            # Continuous reward: softmax probability of correct answer
            answer_logits = logits[0, prompt_len-1, :]
            answer_probs = F.softmax(answer_logits, dim=-1)
            correct_answer_token = full_tokens[prompt_len]
            reward = answer_probs[correct_answer_token].item()
            group_rewards.append(reward)
            group_log_probs.append(total_log_prob)

        # Compute advantages WITHOUT σ-normalization
        rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32, device=device)
        mean_r = rewards_tensor.mean()
        advantages = rewards_tensor - mean_r  # No division by std!

        total_loss = torch.tensor(0.0, device=device)
        for i in range(n_samples):
            adv_i = advantages[i]
            log_prob_i = group_log_probs[i]
            total_loss = total_loss - adv_i * log_prob_i

        total_loss = total_loss / n_samples

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(total_loss.item())

        if step % 25 == 0 or step == num_steps - 1:
            eval_acc = evaluate_model(model, device, 50)
            eval_trajectory.append({'step': step, 'eval': eval_acc})

    elapsed = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    final_eval = evaluate_model(model, device, 100)

    return {
        'strategy': 'grpo_unnormalized',
        'n_samples': n_samples,
        'reward_type': 'continuous_softmax_prob',
        'model_size': model_size,
        'final_loss': losses[-1],
        'final_eval': final_eval,
        'peak_eval': max(e['eval'] for e in eval_trajectory),
        'eval_trajectory': eval_trajectory,
        'peak_gpu_mem_gb': peak_mem,
        'total_time_s': elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--model_size', default='76k', choices=['76k', '2.28m'])
    parser.add_argument('--output', default='grpo_advantage_noise_verification.json')
    args = parser.parse_args()

    print("=" * 70)
    print("GRPO Advantage = Adaptive Gradient Noise Verification")
    print("=" * 70)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_size}, Steps: {args.num_steps}")
    print(f"All strategies use FP32 (no AMP/BF16) for clean noise comparison")

    results = {}

    # Strategy 1: SFT baseline
    print(f"\n[1/5] SFT baseline ({args.model_size})")
    r1 = train_sft_baseline(args.model_size, args.num_steps)
    results['sft_baseline'] = r1
    print(f"  eval={r1['final_eval']:.1%}, peak={r1['peak_eval']:.1%}, "
          f"loss={r1['final_loss']:.3f}, time={r1['total_time_s']:.1f}s")

    # Strategy 2: GRPO normalized (A=(r-μ)/σ)
    print(f"\n[2/5] GRPO normalized (A=(r-μ)/σ, n=4)")
    r2 = train_grpo_normalized(args.model_size, args.num_steps)
    results['grpo_normalized'] = r2
    print(f"  eval={r2['final_eval']:.1%}, peak={r2['peak_eval']:.1%}, "
          f"loss={r2['final_loss']:.3f}, time={r2['total_time_s']:.1f}s")

    # Strategy 3: GRPO unnormalized (A=r-μ)
    print(f"\n[3/5] GRPO unnormalized (A=r-μ, n=4)")
    r3 = train_grpo_unnormalized(args.model_size, args.num_steps)
    results['grpo_unnormalized'] = r3
    print(f"  eval={r3['final_eval']:.1%}, peak={r3['peak_eval']:.1%}, "
          f"loss={r3['final_loss']:.3f}, time={r3['total_time_s']:.1f}s")

    # Strategy 4: SFT + fixed gradient noise
    print(f"\n[4/5] SFT + fixed noise (σ=0.01)")
    r4 = train_sft_fixed_noise(args.model_size, args.num_steps)
    results['sft_fixed_noise'] = r4
    print(f"  eval={r4['final_eval']:.1%}, peak={r4['peak_eval']:.1%}, "
          f"loss={r4['final_loss']:.3f}, time={r4['total_time_s']:.1f}s")

    # Strategy 5: SFT + adaptive gradient noise
    print(f"\n[5/5] SFT + adaptive noise (σ∝loss_var)")
    r5 = train_sft_adaptive_noise(args.model_size, args.num_steps)
    results['sft_adaptive_noise'] = r5
    print(f"  eval={r5['final_eval']:.1%}, peak={r5['peak_eval']:.1%}, "
          f"loss={r5['final_loss']:.3f}, mean_sigma={r5['mean_sigma']:.4f}, "
          f"time={r5['total_time_s']:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: GRPO σ-normalization ≈ Adaptive Gradient Noise?")
    print("=" * 70)

    baseline_eval = r1['final_eval']

    strategies = [
        ('sft_baseline', 'SFT baseline (no noise)'),
        ('grpo_normalized', 'GRPO A=(r-μ)/σ'),
        ('grpo_unnormalized', 'GRPO A=r-μ (no σ)'),
        ('sft_fixed_noise', 'SFT+fixed σ=0.01'),
        ('sft_adaptive_noise', 'SFT+adaptive σ∝loss_var'),
    ]
    for key, label in strategies:
        r = results[key]
        delta = r['final_eval'] - baseline_eval
        print(f"  {label}: eval={r['final_eval']:.1%} (Δ{delta:+.1%}), "
              f"peak={r['peak_eval']:.1%}, "
              f"loss={r['final_loss']:.3f}")

    print("\nVerification hypothesis:")
    print(f"  If GRPO σ-normalization ≈ adaptive gradient noise:")
    print(f"  → GRPO normalized eval ≈ SFT adaptive noise eval")
    print(f"  → Actual: GRPO normalized {r2['final_eval']:.1%} "
          f"vs SFT adaptive noise {r5['final_eval']:.1%}")
    print(f"  → Δ = {abs(r2['final_eval'] - r5['final_eval']):.1%}")

    # Key comparison: normalized vs unnormalized GRPO
    print(f"\nσ-normalization effect:")
    print(f"  GRPO normalized eval: {r2['final_eval']:.1%}")
    print(f"  GRPO unnormalized eval: {r3['final_eval']:.1%}")
    print(f"  → σ-normalization Δ = {r2['final_eval'] - r3['final_eval']:+.1%}")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()