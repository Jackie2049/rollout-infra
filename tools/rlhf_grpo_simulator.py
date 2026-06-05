#!/usr/bin/env python3
"""RLHF / GRPO Training Simulator
===================================
Demonstrates RLHF from first principles with actual training:
1. Reward Model training (from preference data)
2. PPO (Proximal Policy Optimization) for RL fine-tuning
3. GRPO (Group Relative Policy Optimization) - no reward model needed
4. Comparison: PPO vs GRPO vs DPO

Builds on the DPO pipeline (dpo_train_pipeline.py).
Educational purpose: understand RLHF algorithms end-to-end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple
from collections import defaultdict


# ============================================================
# 1. Policy Model (MiniGPT for RL)
# ============================================================

class MiniGPT(nn.Module):
    """Small GPT model for policy/value/reward."""
    def __init__(self, vocab_size=256, d_model=128, n_head=4, n_layer=3,
                 max_len=128, tie_weights=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head,
                dim_feedforward=4 * d_model,
                batch_first=True, dropout=0.1
            ) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def get_log_probs(self, sequences, actions):
        """Get log probability of actions given sequences (for policy gradient)."""
        logits = self(sequences)  # [B, T, V]
        # actions: [B, T] — the tokens to evaluate
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather log probs for taken actions
        action_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        return action_log_probs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=32, temperature=0.8, top_k=20):
        """Generate continuations."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def generate_with_logprobs(self, idx, max_new_tokens=32, temperature=0.8):
        """Generate and collect log probabilities (for PPO)."""
        all_log_probs = []
        all_tokens = []

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)

            idx_next = torch.multinomial(probs, num_samples=1)
            token_log_prob = log_probs.gather(1, idx_next)

            all_tokens.append(idx_next)
            all_log_probs.append(token_log_prob)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx, all_tokens, all_log_probs


class ValueHead(nn.Module):
    """Value function head for PPO (predicts V(s))."""
    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, hidden_states):
        """hidden_states: [B, T, d_model] → [B, T]"""
        return self.net(hidden_states).squeeze(-1)


class RewardModel(nn.Module):
    """Reward model: sequence → scalar reward.

    In practice: Bradley-Terry model trained on preferences.
    Here: simple head on top of pretrained model.
    """
    def __init__(self, vocab_size=256, d_model=128, n_head=4, n_layer=2, max_len=128):
        super().__init__()
        self.backbone = MiniGPT(vocab_size, d_model, n_head, n_layer, max_len, tie_weights=False)
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, sequences):
        """Return reward for each sequence: [B, T] → [B]"""
        B, T = sequences.shape
        pos = torch.arange(T, device=sequences.device).unsqueeze(0)
        x = self.backbone.tok_emb(sequences) + self.backbone.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=sequences.device)
        for block in self.backbone.blocks:
            x = block(x, src_mask=mask, is_causal=True)
        x = self.backbone.ln_f(x)
        # Use last token's hidden state for reward
        reward = self.reward_head(x[:, -1, :]).squeeze(-1)
        return reward


class PolicyWithValue(nn.Module):
    """Policy network with value head for PPO."""
    def __init__(self, vocab_size=256, d_model=128, n_head=4, n_layer=3, max_len=128):
        super().__init__()
        self.policy = MiniGPT(vocab_size, d_model, n_head, n_layer, max_len)
        self.value_head = ValueHead(d_model)

    def get_value(self, sequences):
        """Get state values: [B, T] → [B, T]"""
        B, T = sequences.shape
        pos = torch.arange(T, device=sequences.device).unsqueeze(0)
        x = self.policy.tok_emb(sequences) + self.policy.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=sequences.device)
        for block in self.policy.blocks:
            x = block(x, src_mask=mask, is_causal=True)
        x = self.policy.ln_f(x)
        values = self.value_head(x)
        return values


# ============================================================
# 2. Environment & Reward
# ============================================================

class SimpleTextEnv:
    """Simple text environment with programmable reward.

    Reward design:
    - Sequences containing "preferred" token patterns get higher reward
    - This simulates a "helpful/harmless" reward model
    """
    def __init__(self, vocab_size=256, seq_len=32, device='cpu'):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device

        # Define "preferred" patterns (specific token ranges)
        self.good_tokens = list(range(10, 40))  # "good" tokens
        self.great_tokens = list(range(10, 20))  # "great" tokens
        self.bad_tokens = list(range(100, 120))  # "bad" tokens

    def compute_reward(self, sequences):
        """Compute reward for generated sequences.

        Args:
            sequences: [B, T] token IDs
        Returns:
            rewards: [B] scalar rewards
        """
        B = sequences.shape[0]
        rewards = torch.zeros(B, device=self.device)

        for i in range(B):
            seq = sequences[i].tolist()
            # Good token ratio
            good_count = sum(1 for t in seq if t in self.good_tokens)
            great_count = sum(1 for t in seq if t in self.great_tokens)
            bad_count = sum(1 for t in seq if t in self.bad_tokens)

            # Reward = good ratio + great bonus - bad penalty
            good_ratio = good_count / len(seq)
            great_bonus = great_count / len(seq) * 0.5
            bad_penalty = bad_count / len(seq) * 0.3

            # Diversity bonus
            unique_ratio = len(set(seq)) / len(seq)
            diversity_bonus = unique_ratio * 0.2

            rewards[i] = good_ratio + great_bonus - bad_penalty + diversity_bonus

        return rewards

    def create_prompts(self, batch_size=16):
        """Create prompt sequences for generation."""
        prompts = []
        for _ in range(batch_size):
            # Random prompt tokens
            prompt = torch.randint(10, self.vocab_size // 2, (8,))
            prompts.append(prompt)
        return torch.stack(prompts).to(self.device)


# ============================================================
# 3. PPO Training
# ============================================================

@dataclass
class PPOConfig:
    """PPO configuration."""
    learning_rate: float = 3e-4
    ppo_epochs: int = 4          # PPO update epochs per batch
    clip_range: float = 0.2      # PPO clipping range
    value_coef: float = 0.5      # Value loss coefficient
    entropy_coef: float = 0.01   # Entropy bonus coefficient
    gamma: float = 1.0           # Discount factor (no discount for text)
    gae_lambda: float = 0.95     # GAE lambda
    kl_coef: float = 0.1         # KL penalty coefficient
    max_new_tokens: int = 24     # Generation length
    batch_size: int = 32         # Batch size for generation
    mini_batch_size: int = 8     # Mini-batch for PPO update


class PPOTrainer:
    """PPO trainer for RLHF."""

    def __init__(self, policy, ref_policy, reward_fn, config, device='cpu'):
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_fn = reward_fn
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate
        )

    def compute_advantages(self, rewards, values, dones=None):
        """Compute GAE advantages."""
        B, T = values.shape
        advantages = torch.zeros_like(values)
        last_gae = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, min(t, rewards.shape[1]-1)] + \
                    self.config.gamma * next_value - values[:, t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae
            advantages[:, t] = last_gae

        return advantages

    def ppo_loss(self, old_log_probs, new_log_probs, advantages, values, returns):
        """Compute PPO clipped objective loss."""
        # Ratio: pi_new / pi_old
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_range,
                           1 + self.config.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy = -(new_log_probs * torch.exp(new_log_probs)).mean()

        # Total loss
        loss = policy_loss + self.config.value_coef * value_loss - \
               self.config.entropy_coef * entropy

        return loss, policy_loss, value_loss, entropy

    def train_step(self, prompts):
        """One PPO training step: generate → compute rewards → update."""
        cfg = self.config
        B = prompts.shape[0]

        # ---- Phase 1: Rollout (generate with old policy) ----
        self.policy.eval()
        sequences = self.policy.policy.generate(
            prompts, max_new_tokens=cfg.max_new_tokens, temperature=0.8
        )

        gen_tokens = sequences[:, prompts.shape[1]:]  # Generated part only
        seq_len = gen_tokens.shape[1]

        # ---- Phase 2: Compute rewards ----
        rewards = self.reward_fn(sequences)  # [B]

        # ---- Phase 3: Compute values & advantages ----
        values = self.policy.get_value(sequences)  # [B, T]
        # Expand scalar rewards to per-token rewards (dense reward → sparse)
        token_rewards = torch.zeros_like(values)
        token_rewards[:, -1] = rewards  # Reward at last position

        advantages = self.compute_advantages(token_rewards, values)
        returns = advantages + values.detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()  # Detach from computation graph
        returns = returns.detach()

        # ---- Phase 4: Compute old log probs ----
        with torch.no_grad():
            old_log_probs = self.policy.policy.get_log_probs(
                sequences, gen_tokens
            )[:, -seq_len:].detach()  # [B, seq_len]

            ref_log_probs = self.ref_policy.get_log_probs(
                sequences, gen_tokens
            )[:, -seq_len:].detach()

        # ---- Phase 5: PPO update ----
        self.policy.train()
        total_loss = 0
        n_updates = 0

        for _ in range(cfg.ppo_epochs):
            # Mini-batch updates
            indices = torch.randperm(B)
            for start in range(0, B, cfg.mini_batch_size):
                end = min(start + cfg.mini_batch_size, B)
                mb_idx = indices[start:end]

                mb_sequences = sequences[mb_idx]
                mb_gen_tokens = gen_tokens[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx, -seq_len:]
                mb_returns = returns[mb_idx, -seq_len:]

                # New log probs
                new_log_probs = self.policy.policy.get_log_probs(
                    mb_sequences, mb_gen_tokens
                )[:, -seq_len:]

                # New values
                new_values = self.policy.get_value(mb_sequences)[:, -seq_len:]

                # KL penalty
                mb_ref_log_probs = ref_log_probs[mb_idx]
                kl_penalty = (new_log_probs - mb_ref_log_probs).mean()

                # PPO loss
                loss, p_loss, v_loss, entropy = self.ppo_loss(
                    mb_old_log_probs, new_log_probs,
                    mb_advantages, new_values, mb_returns
                )
                loss = loss + cfg.kl_coef * kl_penalty

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        avg_loss = total_loss / max(n_updates, 1)
        return avg_loss, rewards.mean().item(), sequences


# ============================================================
# 4. GRPO Training (Group Relative Policy Optimization)
# ============================================================

@dataclass
class GRPOConfig:
    """GRPO configuration."""
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    kl_coef: float = 0.05        # KL to reference policy
    n_samples: int = 4            # Number of samples per prompt (group size)
    max_new_tokens: int = 24
    batch_size: int = 8           # Number of prompts per batch
    ppo_epochs: int = 2


class GRPOTrainer:
    """GRPO trainer — no value function needed!

    Key insight: Instead of learning V(s), GRPO uses group statistics:
    - Generate n_samples responses per prompt
    - Compute rewards for each
    - Normalize rewards within group (relative ranking)
    - Use as advantage estimate

    This eliminates the critic network entirely.
    """

    def __init__(self, policy, ref_policy, reward_fn, config, device='cpu'):
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_fn = reward_fn
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate
        )

    def train_step(self, prompts):
        """One GRPO training step."""
        cfg = self.config
        n_prompts = prompts.shape[0]
        n_samples = cfg.n_samples

        # ---- Phase 1: Generate n_samples per prompt ----
        self.policy.eval()
        all_sequences = []
        all_gen_tokens = []

        for i in range(n_prompts):
            prompt = prompts[i:i+1].expand(n_samples, -1)  # [n_samples, prompt_len]
            seqs = self.policy.generate(
                prompt, max_new_tokens=cfg.max_new_tokens, temperature=0.9
            )
            all_sequences.append(seqs)
            gen = seqs[:, prompts.shape[1]:]
            all_gen_tokens.append(gen)

        # Stack: [n_prompts * n_samples, T]
        all_sequences = torch.cat(all_sequences, dim=0)
        all_gen_tokens = torch.cat(all_gen_tokens, dim=0)

        # ---- Phase 2: Compute rewards ----
        rewards = self.reward_fn(all_sequences)  # [n_prompts * n_samples]
        seq_len = all_gen_tokens.shape[1]

        # ---- Phase 3: Group-normalize rewards (GRPO key!) ----
        # Reshape: [n_prompts, n_samples]
        rewards_grouped = rewards.view(n_prompts, n_samples)

        # Normalize within each group
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards_grouped - group_mean) / group_std
        advantages = advantages.view(-1).detach()  # [n_prompts * n_samples]

        # ---- Phase 4: Compute log probs ----
        self.policy.train()
        with torch.no_grad():
            old_log_probs = self.policy.get_log_probs(
                all_sequences, all_gen_tokens
            )[:, -seq_len:].detach()
            ref_log_probs = self.ref_policy.get_log_probs(
                all_sequences, all_gen_tokens
            )[:, -seq_len:].detach()

        # ---- Phase 5: GRPO update (clipped + KL) ----
        total_loss = 0
        for _ in range(cfg.ppo_epochs):
            new_log_probs = self.policy.get_log_probs(
                all_sequences, all_gen_tokens
            )[:, -seq_len:]

            # Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Per-token advantage (broadcast scalar advantage to all tokens)
            token_advantages = advantages.unsqueeze(1).expand(-1, seq_len)

            # Clipped objective
            surr1 = ratio * token_advantages
            surr2 = torch.clamp(ratio, 1 - cfg.clip_range,
                               1 + cfg.clip_range) * token_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL penalty to reference
            kl = (new_log_probs - ref_log_probs).mean()

            loss = policy_loss + cfg.kl_coef * kl

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / cfg.ppo_epochs
        return avg_loss, rewards.mean().item(), all_sequences


# ============================================================
# 5. Reward Model Training
# ============================================================

def create_preference_data(env, model, n_pairs=200, seq_len=32, device='cpu'):
    """Create preference pairs using the environment reward."""
    chosen_sequences = []
    rejected_sequences = []
    chosen_rewards = []
    rejected_rewards = []

    for _ in range(n_pairs):
        # Generate two sequences
        prompt = env.create_prompts(1)
        seq1 = model.generate(prompt, max_new_tokens=seq_len, temperature=1.0)
        seq2 = model.generate(prompt, max_new_tokens=seq_len, temperature=1.0)

        r1 = env.compute_reward(seq1).item()
        r2 = env.compute_reward(seq2).item()

        # Higher reward = chosen
        if r1 >= r2:
            chosen_sequences.append(seq1[0])
            rejected_sequences.append(seq2[0])
            chosen_rewards.append(r1)
            rejected_rewards.append(r2)
        else:
            chosen_sequences.append(seq2[0])
            rejected_sequences.append(seq1[0])
            chosen_rewards.append(r2)
            rejected_rewards.append(r1)

    chosen = torch.stack(chosen_sequences)
    rejected = torch.stack(rejected_sequences)
    return chosen, rejected, chosen_rewards, rejected_rewards


def train_reward_model(reward_model, chosen, rejected, n_epochs=10,
                       batch_size=32, lr=1e-3, device='cpu'):
    """Train reward model using Bradley-Terry loss."""
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)
    losses = []
    accuracies = []

    for epoch in range(n_epochs):
        total_loss = 0
        correct = 0
        total = 0

        indices = torch.randperm(len(chosen))
        for start in range(0, len(chosen), batch_size):
            end = min(start + batch_size, len(chosen))
            idx = indices[start:end]

            r_chosen = reward_model(chosen[idx])
            r_rejected = reward_model(rejected[idx])

            # Bradley-Terry loss: -log σ(r_chosen - r_rejected)
            loss = -F.logsigmoid(r_chosen - r_rejected).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (r_chosen > r_rejected).sum().item()
            total += len(idx)

        avg_loss = total_loss / (len(chosen) / batch_size)
        acc = correct / max(total, 1)
        losses.append(avg_loss)
        accuracies.append(acc)

    return losses, accuracies


# ============================================================
# 6. Experiments
# ============================================================

def experiment_ppo_vs_grpo(device='cpu'):
    """Compare PPO vs GRPO training dynamics."""
    print("\n  === Experiment: PPO vs GRPO ===")

    vocab_size = 128
    d_model = 64
    seq_len = 24
    n_steps = 30

    env = SimpleTextEnv(vocab_size, seq_len, device)

    # Shared initial policy (for fair comparison)
    torch.manual_seed(42)
    ref_policy = MiniGPT(vocab_size, d_model, n_head=4, n_layer=2, max_len=64).to(device)

    # --- PPO ---
    torch.manual_seed(42)
    ppo_policy = MiniGPT(vocab_size, d_model, n_head=4, n_layer=2, max_len=64).to(device)
    ppo_policy.load_state_dict(ref_policy.state_dict())
    ppo_ref = MiniGPT(vocab_size, d_model, n_head=4, n_layer=2, max_len=64).to(device)
    ppo_ref.load_state_dict(ref_policy.state_dict())
    ppo_ref.eval()

    ppo_model = PolicyWithValue(vocab_size, d_model, n_head=4, n_layer=2, max_len=64).to(device)
    ppo_model.policy.load_state_dict(ppo_policy.state_dict())

    ppo_config = PPOConfig(
        learning_rate=3e-4, ppo_epochs=3, clip_range=0.2,
        batch_size=16, mini_batch_size=4, max_new_tokens=seq_len,
        kl_coef=0.05, entropy_coef=0.01
    )
    ppo_trainer = PPOTrainer(ppo_model, ppo_ref, env.compute_reward, ppo_config, device)

    ppo_rewards = []
    ppo_losses = []
    for step in range(n_steps):
        prompts = env.create_prompts(ppo_config.batch_size)
        loss, avg_reward, _ = ppo_trainer.train_step(prompts)
        ppo_rewards.append(avg_reward)
        ppo_losses.append(loss)
        if (step + 1) % 10 == 0:
            print(f"    PPO step {step+1}: reward={avg_reward:.4f}, loss={loss:.4f}")

    # --- GRPO ---
    torch.manual_seed(42)
    grpo_policy = MiniGPT(vocab_size, d_model, n_head=4, n_layer=2, max_len=64).to(device)
    grpo_policy.load_state_dict(ref_policy.state_dict())
    grpo_ref = MiniGPT(vocab_size, d_model, n_head=4, n_layer=2, max_len=64).to(device)
    grpo_ref.load_state_dict(ref_policy.state_dict())
    grpo_ref.eval()

    grpo_config = GRPOConfig(
        learning_rate=3e-4, n_samples=4, clip_range=0.2,
        batch_size=8, ppo_epochs=2, max_new_tokens=seq_len, kl_coef=0.05
    )
    grpo_trainer = GRPOTrainer(grpo_policy, grpo_ref, env.compute_reward, grpo_config, device)

    grpo_rewards = []
    grpo_losses = []
    for step in range(n_steps):
        prompts = env.create_prompts(grpo_config.batch_size)
        loss, avg_reward, _ = grpo_trainer.train_step(prompts)
        grpo_rewards.append(avg_reward)
        grpo_losses.append(loss)
        if (step + 1) % 10 == 0:
            print(f"    GRPO step {step+1}: reward={avg_reward:.4f}, loss={loss:.4f}")

    # Summary
    print(f"\n    PPO:  initial={ppo_rewards[0]:.4f} → final={ppo_rewards[-1]:.4f} "
          f"(Δ={ppo_rewards[-1]-ppo_rewards[0]:+.4f})")
    print(f"    GRPO: initial={grpo_rewards[0]:.4f} → final={grpo_rewards[-1]:.4f} "
          f"(Δ={grpo_rewards[-1]-grpo_rewards[0]:+.4f})")
    print(f"    PPO model params: {sum(p.numel() for p in ppo_model.parameters()):,}")
    print(f"    GRPO model params: {sum(p.numel() for p in grpo_policy.parameters()):,}")
    print(f"    (PPO has {sum(p.numel() for p in ppo_model.value_head.parameters()):,} "
          f"extra value head params)")

    return {
        'ppo_rewards': ppo_rewards,
        'grpo_rewards': grpo_rewards,
        'ppo_losses': ppo_losses,
        'grpo_losses': grpo_losses,
        'ppo_params': sum(p.numel() for p in ppo_model.parameters()),
        'grpo_params': sum(p.numel() for p in grpo_policy.parameters()),
    }


def experiment_grpo_n_samples(device='cpu'):
    """Test how GRPO group size (n_samples) affects training."""
    print("\n  === Experiment: GRPO Group Size ===")

    vocab_size = 128
    d_model = 64
    seq_len = 24
    n_steps = 20

    env = SimpleTextEnv(vocab_size, seq_len, device)

    results = {}
    for n_samples in [2, 4, 8, 16]:
        torch.manual_seed(42)
        policy = MiniGPT(vocab_size, d_model, n_head=4, n_layer=2, max_len=64).to(device)
        ref = MiniGPT(vocab_size, d_model, n_head=4, n_layer=2, max_len=64).to(device)
        ref.load_state_dict(policy.state_dict())
        ref.eval()

        config = GRPOConfig(
            n_samples=n_samples, batch_size=8, max_new_tokens=seq_len,
            learning_rate=3e-4, kl_coef=0.05
        )
        trainer = GRPOTrainer(policy, ref, env.compute_reward, config, device)

        rewards = []
        for step in range(n_steps):
            prompts = env.create_prompts(config.batch_size)
            _, avg_reward, _ = trainer.train_step(prompts)
            rewards.append(avg_reward)

        improvement = rewards[-1] - rewards[0]
        results[n_samples] = {
            'rewards': rewards,
            'final_reward': rewards[-1],
            'improvement': improvement,
        }
        print(f"    n={n_samples:>2}: initial={rewards[0]:.4f} → final={rewards[-1]:.4f} "
              f"(Δ={improvement:+.4f})")

    return results


def experiment_kl_penalty(device='cpu'):
    """Test KL penalty coefficient effect on policy drift."""
    print("\n  === Experiment: KL Penalty Effect ===")

    vocab_size = 128
    d_model = 64
    seq_len = 24
    n_steps = 20

    env = SimpleTextEnv(vocab_size, seq_len, device)

    results = {}
    for kl_coef in [0.0, 0.01, 0.05, 0.1, 0.5]:
        torch.manual_seed(42)
        policy = MiniGPT(vocab_size, d_model, n_head=4, n_layer=2, max_len=64).to(device)
        ref = MiniGPT(vocab_size, d_model, n_head=4, n_layer=2, max_len=64).to(device)
        ref.load_state_dict(policy.state_dict())
        ref.eval()

        config = GRPOConfig(
            n_samples=4, batch_size=8, max_new_tokens=seq_len,
            learning_rate=3e-4, kl_coef=kl_coef
        )
        trainer = GRPOTrainer(policy, ref, env.compute_reward, config, device)

        rewards = []
        kl_divs = []

        for step in range(n_steps):
            prompts = env.create_prompts(config.batch_size)
            n_prompts = prompts.shape[0]
            n_s = config.n_samples

            # Generate
            policy.eval()
            all_seqs = []
            for i in range(n_prompts):
                p = prompts[i:i+1].expand(n_s, -1)
                s = policy.generate(p, max_new_tokens=seq_len, temperature=0.9)
                all_seqs.append(s)
            all_seqs = torch.cat(all_seqs, dim=0)
            gen = all_seqs[:, prompts.shape[1]:]

            # Compute KL divergence
            with torch.no_grad():
                p_log = policy.get_log_probs(all_seqs, gen)[:, -seq_len:]
                r_log = ref.get_log_probs(all_seqs, gen)[:, -seq_len:]
                kl = (p_log - r_log).mean().item()

            rewards.append(env.compute_reward(all_seqs).mean().item())
            kl_divs.append(kl)

            # Train step
            trainer.train_step(prompts)

        results[kl_coef] = {
            'rewards': rewards,
            'kl_divs': kl_divs,
            'final_reward': rewards[-1],
            'final_kl': kl_divs[-1],
        }
        print(f"    β_kl={kl_coef:.2f}: reward={rewards[-1]:.4f}, "
              f"KL={kl_divs[-1]:.4f}")

    return results


def experiment_reward_model(device='cpu'):
    """Train and evaluate a reward model."""
    print("\n  === Experiment: Reward Model Training ===")

    vocab_size = 128
    d_model = 64

    env = SimpleTextEnv(vocab_size, seq_len=24, device=device)

    # Create a policy for data generation
    torch.manual_seed(42)
    policy = MiniGPT(vocab_size, d_model, n_head=4, n_layer=2, max_len=64).to(device)

    # Generate preference data
    print("    Generating preference data...")
    chosen, rejected, c_rewards, r_rewards = create_preference_data(
        env, policy, n_pairs=200, seq_len=24, device=device
    )
    print(f"    Created {len(chosen)} preference pairs")
    print(f"    Avg chosen reward: {sum(c_rewards)/len(c_rewards):.4f}")
    print(f"    Avg rejected reward: {sum(r_rewards)/len(r_rewards):.4f}")

    # Train reward model
    reward_model = RewardModel(vocab_size, d_model, n_head=4, n_layer=2, max_len=64).to(device)
    losses, accuracies = train_reward_model(
        reward_model, chosen, rejected,
        n_epochs=15, batch_size=32, lr=1e-3, device=device
    )

    print(f"    Reward model loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"    Reward model accuracy: {accuracies[0]:.1%} → {accuracies[-1]:.1%}")

    # Use reward model for GRPO
    torch.manual_seed(42)
    grpo_policy = MiniGPT(vocab_size, d_model, n_head=4, n_layer=2, max_len=64).to(device)
    grpo_ref = MiniGPT(vocab_size, d_model, n_head=4, n_layer=2, max_len=64).to(device)
    grpo_ref.load_state_dict(grpo_policy.state_dict())
    grpo_ref.eval()

    config = GRPOConfig(n_samples=4, batch_size=8, max_new_tokens=24, kl_coef=0.05)
    trainer = GRPOTrainer(
        grpo_policy, grpo_ref,
        reward_model.forward,  # Use learned reward model!
        config, device
    )

    rewards = []
    for step in range(15):
        prompts = env.create_prompts(config.batch_size)
        _, avg_reward, _ = trainer.train_step(prompts)
        rewards.append(avg_reward)

    print(f"    GRPO with learned RM: {rewards[0]:.4f} → {rewards[-1]:.4f}")

    return {
        'rm_losses': losses,
        'rm_accuracies': accuracies,
        'grpo_rewards': rewards,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("RLHF / GRPO Training Simulator")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"\n  GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")
    else:
        print(f"\n  Device: CPU")

    results = {}

    # Exp 1: PPO vs GRPO
    comparison = experiment_ppo_vs_grpo(device)
    results['ppo_vs_grpo'] = {
        'ppo_final_reward': comparison['ppo_rewards'][-1],
        'grpo_final_reward': comparison['grpo_rewards'][-1],
        'ppo_params': comparison['ppo_params'],
        'grpo_params': comparison['grpo_params'],
    }

    # Exp 2: GRPO group size
    n_samples_results = experiment_grpo_n_samples(device)
    results['grpo_n_samples'] = {
        k: {'final': v['final_reward'], 'improvement': v['improvement']}
        for k, v in n_samples_results.items()
    }

    # Exp 3: KL penalty
    kl_results = experiment_kl_penalty(device)
    results['kl_penalty'] = {
        k: {'final_reward': v['final_reward'], 'final_kl': v['final_kl']}
        for k, v in kl_results.items()
    }

    # Exp 4: Reward model + GRPO
    rm_results = experiment_reward_model(device)
    results['reward_model'] = {
        'final_rm_acc': rm_results['rm_accuracies'][-1],
        'grpo_with_rm': rm_results['grpo_rewards'][-1],
    }

    # Summary
    print("\n" + "=" * 60)
    print("RLHF Algorithm Comparison Summary")
    print("=" * 60)
    print("""
    ┌──────────────────────────────────────────────────────────────┐
    │ Algorithm   │ Models Needed │ Value Func │ Reward Model │ Complexity   │
    │─────────────│───────────────│────────────│──────────────│──────────────│
    │ PPO (RLHF)  │ 4 (π, π_ref, │ ✓ (critic) │ ✓ (trained)  │ Highest     │
    │             │   V, RM)      │            │              │              │
    │ GRPO        │ 2 (π, π_ref)  │ ✗ (group)  │ ✓ or oracle  │ Medium      │
    │ DPO         │ 2 (π, π_ref)  │ ✗          │ ✗            │ Lowest      │
    └──────────────────────────────────────────────────────────────┘

    Key Insights:
    - PPO: Most flexible, but requires 4 models + reward model training
    - GRPO: Eliminates critic via group statistics, 2x fewer models
    - DPO: Simplest (no RL), but needs pre-collected preference data
    - GRPO n_samples: 4-8 is optimal (too few = noisy, too many = expensive)
    - KL penalty: 0.05-0.1 prevents reward hacking while allowing learning

    GRPO Advantage (from verl):
    - No value function → 50% less GPU memory for training
    - Group normalization replaces learned baseline
    - n=8 responses per prompt → efficient prefix sharing in rollout
    - Used in DeepSeek-R1 for reasoning training
    """)

    if device == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"  Peak GPU memory: {mem:.1f} MB")
        results['gpu_memory_mb'] = round(mem, 1)

    with open("rlhf_grpo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to rlhf_grpo_results.json")


if __name__ == "__main__":
    main()
