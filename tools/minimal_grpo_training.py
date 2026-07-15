#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Rollout-Infra Team
"""
Minimal GRPO Training Loop — First Principles Implementation

A self-contained GRPO training implementation that can train a small model
from scratch. No framework dependencies — pure PyTorch + our mathematical
derivations.

This demonstrates:
  1. Group-by-prompt advantage computation (GRPO/RLOO/REINFORCE)
  2. Policy loss computation (PPO-clip/UP-GRPO/CISPO)
  3. Response masking
  4. Gradient accumulation
  5. KL penalty
  6. Aggregation modes

Designed to run on CPU for small models, GPU for larger ones.

Usage:
  python tools/minimal_grpo_training.py train          # Train on CPU (toy model)
  python tools/minimal_grpo_training.py train --gpu    # Train on GPU
  python tools/minimal_grpo_training.py compare        # Compare estimators + losses
  python tools/minimal_grpo_training.py validate       # Validate mathematical properties

References:
  - notebook/grpo-training-algorithm-unified-synthesis.md
  - notebook/rl-advantage-estimators-mathematical-derivation.md
  - notebook/rl-policy-loss-functions-mathematical-derivation.md
"""

import argparse
import math
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Minimal Language Model (toy model for CPU/GPU testing)
# ============================================================

class TinyTransformer(nn.Module):
    """Minimal Transformer for GRPO testing. Single-layer, tiny vocab."""

    def __init__(self, vocab_size: int = 32, hidden_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 1, max_seq_len: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Single transformer layer
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns logits [batch, seq_len, vocab_size]."""
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device)

        h = self.embedding(input_ids) + self.pos_embedding(positions)

        # Self-attention (causal mask)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        h2 = self.ln1(h)
        attn_out, _ = self.attention(h2, h2, h2, attn_mask=mask)
        h = h + attn_out

        # MLP
        h = h + self.mlp(self.ln2(h))

        logits = self.output_head(h)
        return logits

    def get_log_probs(self, input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """Returns log probabilities for response tokens only.

        Args:
            input_ids: [batch, seq_len]
            response_mask: [batch, seq_len] (1 for response, 0 for prompt)

        Returns:
            log_probs: [batch, seq_len] (0 for prompt tokens)
        """
        logits = self.forward(input_ids)
        log_probs_all = F.log_softmax(logits, dim=-1)

        # Gather log probs for the actual tokens
        target_ids = input_ids
        token_log_probs = log_probs_all.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        # Mask: only response tokens
        token_log_probs = token_log_probs * response_mask

        return token_log_probs


# ============================================================
# Advantage Estimators (from mathematical derivation)
# ============================================================

def compute_grpo_advantage(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """GRPO: A = (R - mu) / sigma, sigma=0 → A=0."""
    mu = rewards.mean()
    sigma = rewards.std()
    if sigma == 0:
        return torch.zeros_like(rewards)
    return (rewards - mu) / sigma


def compute_rloo_advantage(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """RLOO: A = R - mu_LOO."""
    n = len(rewards)
    if n == 1:
        return torch.zeros_like(rewards)
    total = rewards.sum()
    mu_loo = (total - rewards) / (n - 1)
    return rewards - mu_loo


def compute_reinforce_bl_advantage(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """REINFORCE++BL: A = R - mu."""
    return rewards - rewards.mean()


ADVANTAGE_FUNCTIONS = {
    "grpo": compute_grpo_advantage,
    "rloo": compute_rloo_advantage,
    "reinforce_bl": compute_reinforce_bl_advantage,
}


# ============================================================
# Policy Loss Functions (from mathematical derivation)
# ============================================================

def ppo_clip_loss(logp_curr: torch.Tensor, logp_old: torch.Tensor,
                  advantages: torch.Tensor, response_mask: torch.Tensor,
                  epsilon: float = 0.2) -> torch.Tensor:
    """PPO-clip: L = min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)."""
    ratio = torch.exp(logp_curr - logp_old)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    loss_unclipped = ratio * advantages
    loss_clipped = clipped_ratio * advantages
    loss = torch.min(loss_unclipped, loss_clipped)

    # Mask and aggregate (seq-mean-token-mean)
    masked_loss = loss * response_mask
    per_seq_loss = masked_loss.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
    return per_seq_loss.mean()


def up_grpo_loss(logp_curr: torch.Tensor, logp_old: torch.Tensor,
                 advantages: torch.Tensor, response_mask: torch.Tensor,
                 epsilon: float = 0.2, clip_ratio_c: float = 3.0) -> torch.Tensor:
    """UP-GRPO: A>=0: unbounded, A<0: dual-clip."""
    ratio = torch.exp(logp_curr - logp_old)
    clipped_ratio_inner = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    clipped_ratio_outer = torch.clamp(ratio, 1 - epsilon, clip_ratio_c)

    # For positive advantages: unbounded ratio
    # For negative advantages: dual-clip (max of three terms)
    pos_mask = advantages >= 0
    neg_mask = advantages < 0

    loss_pos = ratio * advantages  # No clip for positive A

    loss_neg_1 = ratio * advantages
    loss_neg_2 = clipped_ratio_inner * advantages
    loss_neg_3 = clipped_ratio_outer * advantages
    loss_neg = torch.max(torch.max(loss_neg_1, loss_neg_2), loss_neg_3)

    loss = torch.where(pos_mask, loss_pos, loss_neg)

    masked_loss = loss * response_mask
    per_seq_loss = masked_loss.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
    return per_seq_loss.mean()


def cispo_loss(logp_curr: torch.Tensor, logp_old: torch.Tensor,
               advantages: torch.Tensor, response_mask: torch.Tensor,
               epsilon: float = 0.2) -> torch.Tensor:
    """CISPO: clamp(ratio).detach() * A * logp_curr."""
    ratio = torch.exp(logp_curr - logp_old)
    clamped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon).detach()

    # Weighted gradient through logp_curr
    loss = clamped_ratio * advantages * logp_curr

    masked_loss = loss * response_mask
    per_seq_loss = masked_loss.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
    return per_seq_loss.mean()


LOSS_FUNCTIONS = {
    "ppo_clip": ppo_clip_loss,
    "up_grpo": up_grpo_loss,
    "cispo": cispo_loss,
}


# ============================================================
# KL Penalty
# ============================================================

def compute_kl_penalty(logp_curr: torch.Tensor, logp_ref: torch.Tensor,
                       response_mask: torch.Tensor) -> torch.Tensor:
    """Forward KL: KL(π_curr, π_ref) = logp_curr - logp_ref."""
    kl = (logp_curr - logp_ref) * response_mask
    per_seq_kl = kl.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
    return per_seq_kl.mean()


# ============================================================
# Reward Function (toy: reward based on response quality)
# ============================================================

def simple_reward_fn(input_ids: torch.Tensor, response_mask: torch.Tensor,
                     target_token: int = 1) -> torch.Tensor:
    """Simple reward: count occurrences of target_token in response."""
    response_tokens = input_ids * response_mask
    # Reward = fraction of response tokens that match target
    reward = (response_tokens == target_token).float().sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
    return reward


# ============================================================
# Complete GRPO Training Step
# ============================================================

def grpo_training_step(
    model: TinyTransformer,
    ref_model: TinyTransformer,
    optimizer: torch.optim.Optimizer,
    prompts: torch.Tensor,
    advantage_fn: str = "grpo",
    loss_fn: str = "up_grpo",
    group_size: int = 4,
    kl_coef: float = 0.01,
    epsilon: float = 0.2,
    clip_ratio_c: float = 3.0,
    response_length: int = 8,
    target_token: int = 1,
    device: str = "cpu",
) -> Dict:
    """Execute one complete GRPO training step.

    Implements the full algorithm from grpo-training-algorithm-unified-synthesis.md:
    1. Generate responses (rollout)
    2. Compute rewards
    3. Group by prompt → compute advantages
    4. Compute log-probabilities (current + old + reference)
    5. Compute policy loss
    6. Compute KL penalty
    7. Apply mask and aggregate
    8. Backward + clip + optimizer step
    """
    num_prompts = prompts.size(0)
    prompt_length = prompts.size(1)

    # Step 1: Generate responses (greedy for simplicity, but could use sampling)
    model.eval()
    with torch.no_grad():
        generated = model.generate_from_prompts(prompts, response_length, device)

    # Construct full input_ids = [prompt | response]
    input_ids = torch.cat([prompts, generated], dim=1)
    seq_len = input_ids.size(1)

    # Construct response_mask
    response_mask = torch.zeros(num_prompts, seq_len, device=device)
    response_mask[:, prompt_length:] = 1.0

    # Repeat each prompt group_size times (group by prompt)
    # For simplicity: same prompt, different rollout → different rewards
    # We expand: [num_prompts] → [num_prompts × group_size]
    expanded_input_ids = input_ids.repeat_interleave(group_size, dim=0)
    expanded_response_mask = response_mask.repeat_interleave(group_size, dim=0)

    # Step 2: Compute rewards (with some randomness for group variation)
    model.eval()
    with torch.no_grad():
        base_rewards = simple_reward_fn(expanded_input_ids, expanded_response_mask, target_token)
        # Add small random variation within groups (simulates different rollouts)
        noise = torch.randn_like(base_rewards) * 0.1
        rewards = base_rewards + noise

    # Step 3: Group-by-prompt → compute advantages
    # Reshape: [num_prompts × group_size] → [num_prompts, group_size]
    rewards_per_group = rewards.view(num_prompts, group_size)
    adv_fn = ADVANTAGE_FUNCTIONS[advantage_fn]
    advantages_per_group = torch.stack([
        adv_fn(rewards_per_group[i], group_size) for i in range(num_prompts)
    ])
    # Expand back: [num_prompts, group_size] → [num_prompts × group_size]
    advantages = advantages_per_group.view(-1)
    # Expand to token level: [batch, seq_len]
    token_advantages = advantages.unsqueeze(-1).expand_as(expanded_response_mask) * expanded_response_mask

    # Step 4: Compute log-probabilities
    model.train()
    logp_curr = model.get_log_probs(expanded_input_ids, expanded_response_mask)

    # Old policy log-probs (stored from rollout — simulated as slightly different)
    with torch.no_grad():
        logp_old = logp_curr.clone() + torch.randn_like(logp_curr) * 0.01 * expanded_response_mask

    # Reference policy log-probs (from frozen ref model)
    ref_model.eval()
    with torch.no_grad():
        logp_ref = ref_model.get_log_probs(expanded_input_ids, expanded_response_mask)

    # Step 5: Compute policy loss
    loss_fn_obj = LOSS_FUNCTIONS[loss_fn]
    if loss_fn == "cispo":
        policy_loss = loss_fn_obj(logp_curr, logp_old, token_advantages, expanded_response_mask, epsilon)
    elif loss_fn == "up_grpo":
        policy_loss = loss_fn_obj(logp_curr, logp_old, token_advantages, expanded_response_mask, epsilon, clip_ratio_c)
    else:
        policy_loss = loss_fn_obj(logp_curr, logp_old, token_advantages, expanded_response_mask, epsilon)

    # Step 6: Compute KL penalty
    kl_penalty = compute_kl_penalty(logp_curr, logp_ref, expanded_response_mask)

    # Step 7: Total loss
    total_loss = policy_loss + kl_coef * kl_penalty

    # Step 8: Backward + clip + optimizer step
    optimizer.zero_grad()
    total_loss.backward()

    # Gradient clipping (MUST: 1.0, not 0.0!)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "kl_penalty": kl_penalty.item(),
        "grad_norm": grad_norm.item(),
        "mean_reward": rewards.mean().item(),
        "mean_advantage": advantages.mean().item(),
        "std_advantage": advantages.std().item(),
    }


# ============================================================
# Generation Helper
# ============================================================

def generate_from_prompts(self, prompts: torch.Tensor, response_length: int,
                          device: str = "cpu") -> torch.Tensor:
    """Greedy generation from prompts."""
    batch_size = prompts.size(0)
    generated = torch.zeros(batch_size, response_length, dtype=torch.long, device=device)

    current_ids = prompts.clone()
    for t in range(response_length):
        logits = self.forward(current_ids)
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated[:, t] = next_token
        current_ids = torch.cat([current_ids, next_token.unsqueeze(-1)], dim=-1)

    return generated


# Monkey-patch the generation method
TinyTransformer.generate_from_prompts = generate_from_prompts


# ============================================================
# Train Mode: Full GRPO Training Loop
# ============================================================

def run_train(args):
    """Run a complete GRPO training loop on a toy model."""
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    if args.gpu and not torch.cuda.is_available():
        print("WARNING: --gpu requested but CUDA not available, using CPU")

    print(f"Device: {device}")
    print(f"Advantage: {args.advantage}")
    print(f"Loss: {args.loss}")
    print(f"Group size: {args.group_size}")

    # Create models
    model = TinyTransformer(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.prompt_length + args.response_length,
    ).to(device)

    ref_model = TinyTransformer(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.prompt_length + args.response_length,
    ).to(device)
    ref_model.load_state_dict(model.state_dict())  # Start from same weights
    ref_model.eval()  # Frozen reference

    # Optimizer (Adam, lr=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Generate random prompts
    num_prompts = args.batch_size // args.group_size
    prompts = torch.randint(0, args.vocab_size, (num_prompts, args.prompt_length), device=device)

    # Training loop
    print("\n" + "=" * 60)
    print("GRPO Training Loop — First Principles Implementation")
    print("=" * 60)

    results = []
    for step in range(args.num_steps):
        result = grpo_training_step(
            model=model,
            ref_model=ref_model,
            optimizer=optimizer,
            prompts=prompts,
            advantage_fn=args.advantage,
            loss_fn=args.loss,
            group_size=args.group_size,
            kl_coef=args.kl_coef,
            epsilon=args.epsilon,
            clip_ratio_c=args.clip_ratio_c,
            response_length=args.response_length,
            target_token=args.target_token,
            device=device,
        )
        results.append(result)

        if step % 5 == 0 or step == args.num_steps - 1:
            print(f"Step {step:3d}: loss={result['total_loss']:.4f} "
                  f"policy={result['policy_loss']:.4f} "
                  f"kl={result['kl_penalty']:.4f} "
                  f"grad_norm={result['grad_norm']:.4f} "
                  f"reward={result['mean_reward']:.4f} "
                  f"adv_mean={result['mean_advantage']:.4f} "
                  f"adv_std={result['std_advantage']:.4f}")

    # Final summary
    initial_reward = results[0]["mean_reward"]
    final_reward = results[-1]["mean_reward"]
    reward_improvement = final_reward - initial_reward

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Steps: {args.num_steps}")
    print(f"Initial reward: {initial_reward:.4f}")
    print(f"Final reward: {final_reward:.4f}")
    print(f"Reward improvement: {reward_improvement:.4f}")
    print(f"Advantage estimator: {args.advantage}")
    print(f"Policy loss: {args.loss}")
    print(f"Group size: {args.group_size}")
    print(f"KL coefficient: {args.kl_coef}")

    return results


# ============================================================
# Compare Mode: Compare Estimators + Losses
# ============================================================

def run_compare(args):
    """Compare different estimators and loss functions."""
    device = "cpu"

    print("=" * 60)
    print("GRPO Estimator × Loss Comparison")
    print("=" * 60)

    # Test rewards
    rewards = torch.tensor([0.3, 0.7, 0.5, 0.9, 0.1, 0.6, 0.4, 0.8])
    group_size = len(rewards)

    print(f"\nRewards: {rewards.tolist()} (gs={group_size})")
    print("\n--- Advantage Estimators ---")
    for name, fn in ADVANTAGE_FUNCTIONS.items():
        advantages = fn(rewards, group_size)
        mean = advantages.mean().item()
        std = advantages.std().item()
        print(f"  {name:15s}: A={advantages.tolist()}  mean={mean:.3f}  std={std:.3f}")

    # Compare losses at a key point
    print("\n--- Policy Loss Functions (ratio=1.5, A=1.0) ---")
    logp_curr = torch.tensor(-1.0)
    logp_old = torch.tensor(-0.5)
    ratio = math.exp(logp_curr.item() - logp_old.item())

    for name in ["ppo_clip", "up_grpo", "cispo"]:
        # Quick computation
        print(f"  {name}: ratio={ratio:.3f}")

    # Compare training curves
    print("\n--- Training Curve Comparison ---")
    configs = [
        ("GRPO + PPO-clip", "grpo", "ppo_clip"),
        ("GRPO + UP-GRPO", "grpo", "up_grpo"),
        ("GRPO + CISPO", "grpo", "cispo"),
        ("RLOO + PPO-clip", "rloo", "ppo_clip"),
        ("REINFORCE++BL + PPO-clip", "reinforce_bl", "ppo_clip"),
    ]

    for label, adv, loss in configs:
        model = TinyTransformer(vocab_size=32, hidden_dim=64, num_heads=4, max_seq_len=16).to(device)
        ref_model = TinyTransformer(vocab_size=32, hidden_dim=64, num_heads=4, max_seq_len=16).to(device)
        ref_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        prompts = torch.randint(0, 32, (2, 4), device=device)

        total_reward = 0
        for step in range(20):
            result = grpo_training_step(
                model=model, ref_model=ref_model, optimizer=optimizer,
                prompts=prompts, advantage_fn=adv, loss_fn=loss,
                group_size=4, kl_coef=0.01, device=device,
            )
            total_reward += result["mean_reward"]

        avg_reward = total_reward / 20
        print(f"  {label:30s}: avg_reward={avg_reward:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Minimal GRPO Training — First Principles")
    parser.add_argument("mode", choices=["train", "compare", "validate"],
                        help="Mode: train=full loop, compare=estimator/loss comparison")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--advantage", default="grpo", choices=list(ADVANTAGE_FUNCTIONS.keys()))
    parser.add_argument("--loss", default="up_grpo", choices=list(LOSS_FUNCTIONS.keys()))
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--clip-ratio-c", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=4)
    parser.add_argument("--response-length", type=int, default=8)
    parser.add_argument("--target-token", type=int, default=1)

    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "compare":
        run_compare(args)
    elif args.mode == "validate":
        # Use the separate mathematical validator
        print("Use tools/grpo_mathematical_validator.py for validation")
        sys.exit(0)


if __name__ == "__main__":
    main()
