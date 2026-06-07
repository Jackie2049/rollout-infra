#!/usr/bin/env python3
"""Mini GRPO Training Pipeline — End-to-End on CPU/GPU

Complete GRPO training with real reward signals on a small GQA Transformer.
Validates the entire RL training loop: prompt → rollout → reward → advantage → actor update.

Uses a simple arithmetic task: given numbers, generate the correct sum.
Reward = 1 if correct, 0 otherwise.

Can run on CPU (small model) or GPU (RTX 4090).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import argparse
from collections import defaultdict

torch.manual_seed(42)

# ============================================================
# Model: Small GQA Transformer for token-level generation
# ============================================================

class MiniGQATransformer(nn.Module):
    """Small GQA Transformer for arithmetic generation."""
    def __init__(self, hidden_dim=64, num_layers=2, num_heads=4, num_kv_heads=2, vocab_size=20):
        super().__init__()
        self.head_dim = hidden_dim // num_heads
        self.g = num_heads // num_kv_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(hidden_dim),
                'q_proj': nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False),
                'k_proj': nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False),
                'v_proj': nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False),
                'o_proj': nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False),
                'ln2': nn.LayerNorm(hidden_dim),
                'gate_proj': nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
                'up_proj': nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
                'down_proj': nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            }))
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        """input_ids: [B, S] of token indices → logits [B, S, vocab_size]"""
        x = self.embed(input_ids)
        for layer in self.layers:
            residual = x
            x = layer['ln1'](x)
            B, S, H = x.shape
            Q = layer['q_proj'](x).view(B, S, -1, self.head_dim)
            K = layer['k_proj'](x).view(B, S, -1, self.head_dim)
            V = layer['v_proj'](x).view(B, S, -1, self.head_dim)
            if self.g > 1:
                K = K.repeat_interleave(self.g, dim=2)
                V = V.repeat_interleave(self.g, dim=2)
            attn_out = F.scaled_dot_product_attention(
                Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2), is_causal=True
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
            x = residual + layer['o_proj'](attn_out)
            residual = x
            x = layer['ln2'](x)
            gate = torch.sigmoid(layer['gate_proj'](x))
            x = residual + layer['down_proj'](gate * layer['up_proj'](x))
        x = self.final_ln(x)
        return self.lm_head(x)


# ============================================================
# Arithmetic Task: generate correct sums
# ============================================================

# Token vocabulary: digits 0-9, operators + =, and special tokens
TOKENS = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '+': 10, '=': 11,
    '<pad>': 12, '<eos>': 13, '<bos>': 14, '<unk>': 15,
    'a': 16, 'b': 17, 'r': 18, '<space>': 19,
}
VOCAB_SIZE = len(TOKENS)  # 20
IDX_TO_TOKEN = {v: k for k, v in TOKENS.items()}


def generate_arithmetic_prompt():
    """Generate a simple arithmetic prompt: 'a+b=' as token indices."""
    a = np.random.randint(0, 5)
    b = np.random.randint(0, 5)
    prompt_tokens = [TOKENS[str(a)], TOKENS['+'], TOKENS[str(b)], TOKENS['=']]
    correct_sum = a + b
    return prompt_tokens, correct_sum


def compute_reward(generated_tokens, correct_sum):
    """Reward = 1 if the generated number matches correct sum, 0 otherwise.

    We look at the first token after the prompt (= sign) and check if it's
    the correct digit. Since max sum is 4+4=8, single digit is sufficient.
    """
    if len(generated_tokens) == 0:
        return 0.0
    # The response starts after the prompt
    # Check if first response token is the correct digit
    first_token = generated_tokens[0]
    token_str = IDX_TO_TOKEN.get(first_token, '<unk>')
    if token_str == str(correct_sum):
        return 1.0
    # Partial credit: close numbers get 0.3
    try:
        val = int(token_str)
        if abs(val - correct_sum) == 1:
            return 0.3
        elif abs(val - correct_sum) <= 2:
            return 0.1
    except (ValueError, TypeError):
        pass
    return 0.0


def decode_tokens(token_ids):
    """Convert token IDs to string."""
    return ''.join(IDX_TO_TOKEN.get(t, '?') for t in token_ids)


# ============================================================
# GRPO Training Step
# ============================================================

def grpo_training_step(model, prompts, n_samples, max_response_len, optimizer, device):
    """One GRPO training step.

    For each prompt, rollout n_samples responses, compute reward,
    group-normalize advantage, and update policy.

    Returns: dict of metrics for this step.
    """
    model.train()
    all_rewards = []
    all_advantages = []
    total_loss = 0
    correct_count = 0
    total_count = 0
    group_count = len(prompts)

    # For each prompt, generate n responses
    group_data = []  # (prompt_tokens, correct_sum, [(response_tokens, reward)])

    for prompt_tokens, correct_sum in prompts:
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
        responses = []

        for _ in range(n_samples):
            # Rollout: autoregressive generation
            current_ids = prompt_tensor.clone()
            response_tokens = []

            for step in range(max_response_len):
                logits = model(current_ids)
                next_logits = logits[:, -1, :]  # [1, vocab]
                probs = F.softmax(next_logits, dim=-1)

                # Sample next token
                next_token = torch.multinomial(probs, 1).item()

                # Stop on EOS
                if next_token == TOKENS['<eos>']:
                    break

                response_tokens.append(next_token)
                current_ids = torch.cat([current_ids,
                    torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)

            reward = compute_reward(response_tokens, correct_sum)
            responses.append((response_tokens, reward))
            all_rewards.append(reward)

            if reward == 1.0:
                correct_count += 1
            total_count += 1

        group_data.append((prompt_tokens, correct_sum, responses))

    # Compute GRPO advantages per group
    for prompt_tokens, correct_sum, responses in group_data:
        group_rewards = [r for _, r in responses]
        mean_r = np.mean(group_rewards)
        std_r = np.std(group_rewards)
        if std_r < 1e-8:
            std_r = 1.0

        for resp_tokens, reward in responses:
            advantage = (reward - mean_r) / std_r
            all_advantages.append(advantage)

    # Now compute policy loss for all samples
    # We need log_probs for each (prompt, response) pair
    loss = 0
    num_valid = 0

    sample_idx = 0
    for prompt_tokens, correct_sum, responses in group_data:
        group_rewards = [r for _, r in responses]
        mean_r = np.mean(group_rewards)
        std_r = np.std(group_rewards)
        if std_r < 1e-8:
            std_r = 1.0

        for resp_tokens, reward in responses:
            advantage = (reward - mean_r) / std_r

            if len(resp_tokens) == 0:
                sample_idx += 1
                continue

            # Re-compute log_probs for this full sequence
            full_ids = torch.tensor(prompt_tokens + resp_tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(full_ids)

            # log_prob of each response token
            log_probs = F.log_softmax(logits, dim=-1)
            response_start = len(prompt_tokens)
            token_log_probs = []
            for t_idx, token in enumerate(resp_tokens):
                pos = response_start + t_idx - 1  # log_prob at position before token
                if pos < 0:
                    continue
                token_log_probs.append(log_probs[0, pos, token])

            if len(token_log_probs) > 0:
                total_log_prob = sum(token_log_probs)
                # GRPO loss: -advantage * log_prob
                loss += -advantage * total_log_prob
                num_valid += 1

            sample_idx += 1

    if num_valid > 0:
        loss = loss / num_valid

    # Backward + update
    optimizer.zero_grad()
    if num_valid > 0:
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    metrics = {
        'loss': loss.item() if num_valid > 0 else 0,
        'reward_mean': np.mean(all_rewards),
        'reward_std': np.std(all_rewards),
        'accuracy': correct_count / total_count if total_count > 0 else 0,
        'advantage_mean': np.mean(all_advantages),
        'advantage_std': np.std(all_advantages),
        'num_valid': num_valid,
        'num_groups': group_count,
    }

    return metrics


def generate_sft_dataset(num_examples=500):
    """Generate supervised training data: prompt + correct answer."""
    dataset = []
    for _ in range(num_examples):
        a = np.random.randint(0, 5)
        b = np.random.randint(0, 5)
        prompt_tokens = [TOKENS[str(a)], TOKENS['+'], TOKENS[str(b)], TOKENS['=']]
        correct_sum = a + b
        # Correct response: just the digit of the sum + EOS
        response_tokens = [TOKENS[str(correct_sum)], TOKENS['<eos>']]
        full_tokens = prompt_tokens + response_tokens
        dataset.append((full_tokens, len(prompt_tokens)))
    return dataset


def sft_training_step(model, dataset, optimizer, device, batch_size=32):
    """Supervised fine-tuning: cross-entropy loss on correct responses."""
    model.train()
    indices = np.random.choice(len(dataset), batch_size, replace=True)
    total_loss = 0

    for idx in indices:
        full_tokens, prompt_len = dataset[idx]
        full_ids = torch.tensor(full_tokens, dtype=torch.long, device=device).unsqueeze(0)

        logits = model(full_ids)
        # Only compute loss on response tokens (after prompt)
        response_logits = logits[:, prompt_len-1:-1, :]  # positions that predict response
        response_targets = torch.tensor(full_tokens[prompt_len:], dtype=torch.long, device=device).unsqueeze(0)

        # Cross-entropy loss
        if response_logits.size(1) == response_targets.size(1):
            loss = F.cross_entropy(response_logits.reshape(-1, VOCAB_SIZE),
                                   response_targets.reshape(-1))
            total_loss += loss.item()

    # Average and backward
    avg_loss = total_loss / batch_size
    optimizer.zero_grad()

    # Compute loss for backward (need to recompute since we accumulated .item())
    losses = []
    for idx in indices:
        full_tokens, prompt_len = dataset[idx]
        full_ids = torch.tensor(full_tokens, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(full_ids)
        response_logits = logits[:, prompt_len-1:-1, :]
        response_targets = torch.tensor(full_tokens[prompt_len:], dtype=torch.long, device=device).unsqueeze(0)
        if response_logits.size(1) == response_targets.size(1):
            loss = F.cross_entropy(response_logits.reshape(-1, VOCAB_SIZE),
                                   response_targets.reshape(-1))
            losses.append(loss)

    if len(losses) > 0:
        total_backward_loss = sum(losses) / len(losses)
        total_backward_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return avg_loss
    return avg_loss


# ============================================================
# PPO Training Step (with critic)
# ============================================================

class SimpleCritic(nn.Module):
    """Simple critic network for PPO: predicts V(x) from prompt."""
    def __init__(self, hidden_dim=32, vocab_size=20):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, prompt_ids):
        """prompt_ids: [B, 4] → value [B, 1]"""
        x = self.embed(prompt_ids)  # [B, 4, H]
        x = x.view(x.size(0), -1)  # [B, 4*H]
        return self.net(x)  # [B, 1]


def ppo_training_step(actor, critic, prompts, n_samples, max_response_len,
                       actor_optimizer, critic_optimizer, device, clip_epsilon=0.2):
    """One PPO training step with critic.

    Returns: dict of metrics for this step.
    """
    actor.train()
    critic.train()

    all_rewards = []
    total_loss = 0
    correct_count = 0
    total_count = 0

    for prompt_tokens, correct_sum in prompts:
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

        # Get V(x) from critic
        with torch.no_grad():
            V_x = critic(prompt_tensor).item()

        for _ in range(n_samples):
            # Rollout
            current_ids = prompt_tensor.clone()
            response_tokens = []

            for step in range(max_response_len):
                logits = actor(current_ids)
                next_logits = logits[:, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                if next_token == TOKENS['<eos>']:
                    break
                response_tokens.append(next_token)
                current_ids = torch.cat([current_ids,
                    torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)

            reward = compute_reward(response_tokens, correct_sum)
            all_rewards.append(reward)
            if reward == 1.0:
                correct_count += 1
            total_count += 1

            # PPO advantage: A = R - V(x)
            advantage = reward - V_x

            if len(response_tokens) == 0:
                continue

            # Compute actor loss: -advantage * log_prob (no clip for simplicity)
            full_ids = torch.tensor(prompt_tokens + response_tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = actor(full_ids)
            log_probs = F.log_softmax(logits, dim=-1)

            response_start = len(prompt_tokens)
            token_log_probs = []
            for t_idx, token in enumerate(response_tokens):
                pos = response_start + t_idx - 1
                if pos < 0:
                    continue
                token_log_probs.append(log_probs[0, pos, token])

            if len(token_log_probs) > 0:
                total_log_prob = sum(token_log_probs)
                loss = -advantage * total_log_prob
                total_loss += loss

        # Update critic: V(x) should predict mean reward
        target_value = np.mean([compute_reward(r, correct_sum) for r in [[] for _ in range(n_samples)]])
        # Simplified: just use actual group mean as target
        group_mean = np.mean(all_rewards[-n_samples:])

        critic_loss = (critic(prompt_tensor) - group_mean) ** 2
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

    actor_optimizer.zero_grad()
    if total_count > 0:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        actor_optimizer.step()

    metrics = {
        'loss': total_loss.item() / total_count if total_count > 0 else 0,
        'reward_mean': np.mean(all_rewards) if len(all_rewards) > 0 else 0,
        'accuracy': correct_count / total_count if total_count > 0 else 0,
    }

    return metrics


# ============================================================
# DPO Training Step (offline, no RL)
# ============================================================

def generate_dpo_preference_pairs(n_pairs=100):
    """Generate preference pairs: (prompt, chosen_response, rejected_response)."""
    pairs = []
    for _ in range(n_pairs):
        a = np.random.randint(0, 5)
        b = np.random.randint(0, 5)
        prompt_tokens = [TOKENS[str(a)], TOKENS['+'], TOKENS[str(b)], TOKENS['=']]
        correct_sum = a + b

        # Chosen: correct answer
        chosen_tokens = [TOKENS[str(correct_sum)], TOKENS['<eos>']]
        # Rejected: wrong answer
        wrong_digit = np.random.randint(0, 10)
        while str(wrong_digit) == str(correct_sum):
            wrong_digit = np.random.randint(0, 10)
        rejected_tokens = [TOKENS[str(wrong_digit)], TOKENS['<eos>']]

        pairs.append((prompt_tokens, chosen_tokens, rejected_tokens, correct_sum))
    return pairs


def dpo_training_step(model, pairs, optimizer, device, beta=0.3):
    """DPO training: offline preference learning.

    DPO loss: -log σ(β log(π(y_w)/π_ref(y_w)) - β log(π(y_l)/π_ref(y_l)))
    """
    model.train()
    ref_model = MiniGQATransformer(hidden_dim=64, num_layers=2, num_heads=4,
                                    num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    batch_pairs = np.random.choice(len(pairs), min(16, len(pairs)), replace=False)
    batch_size = len(batch_pairs)

    for idx in batch_pairs:
        prompt_tokens, chosen_tokens, rejected_tokens, correct_sum = pairs[idx]

        chosen_full = prompt_tokens + chosen_tokens
        rejected_full = prompt_tokens + rejected_tokens

        chosen_ids = torch.tensor(chosen_full, dtype=torch.long, device=device).unsqueeze(0)
        rejected_ids = torch.tensor(rejected_full, dtype=torch.long, device=device).unsqueeze(0)
        prompt_len = len(prompt_tokens)

        # Policy log probs (with gradient)
        chosen_logits = model(chosen_ids)
        rejected_logits = model(rejected_ids)

        chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)

        # Sum log probs of response tokens (keep as tensors)
        chosen_lp = torch.tensor(0.0, device=device)
        rejected_lp = torch.tensor(0.0, device=device)
        ref_chosen_lp = 0.0
        ref_rejected_lp = 0.0

        for t_idx, t in enumerate(chosen_tokens):
            pos = prompt_len + t_idx - 1
            if pos >= 0:
                chosen_lp = chosen_lp + chosen_log_probs[0, pos, t]

        for t_idx, t in enumerate(rejected_tokens):
            pos = prompt_len + t_idx - 1
            if pos >= 0:
                rejected_lp = rejected_lp + rejected_log_probs[0, pos, t]

        # Reference log probs (frozen, no gradient)
        with torch.no_grad():
            ref_chosen_logits = ref_model(chosen_ids)
            ref_rejected_logits = ref_model(rejected_ids)
            ref_chosen_log_probs = F.log_softmax(ref_chosen_logits, dim=-1)
            ref_rejected_log_probs = F.log_softmax(ref_rejected_logits, dim=-1)

            for t_idx, t in enumerate(chosen_tokens):
                pos = prompt_len + t_idx - 1
                if pos >= 0:
                    ref_chosen_lp += ref_chosen_log_probs[0, pos, t].item()

            for t_idx, t in enumerate(rejected_tokens):
                pos = prompt_len + t_idx - 1
                if pos >= 0:
                    ref_rejected_lp += ref_rejected_log_probs[0, pos, t].item()

        # DPO loss: -log σ(β × (log π(y_w) - log π_ref(y_w)) - β × (log π(y_l) - log π_ref(y_l)))
        chosen_ratio = chosen_lp - ref_chosen_lp  # policy - ref (has grad for policy)
        rejected_ratio = rejected_lp - ref_rejected_lp  # policy - ref (has grad for policy)
        margin = beta * (chosen_ratio - rejected_ratio)

        loss = -F.logsigmoid(margin)  # -log σ(margin)
        total_loss = total_loss + loss

    avg_loss = total_loss / batch_size
    optimizer.zero_grad()
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    metrics = {
        'loss': avg_loss.item(),
        'margin': margin.item() if hasattr(margin, 'item') else margin,
    }
    return metrics


# ============================================================
# Main Training Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--num_steps', type=int, default=200, help='Number of training steps')
    parser.add_argument('--n_samples', type=int, default=4, help='GRPO group size')
    parser.add_argument('--max_response_len', type=int, default=8, help='Max response tokens')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_prompts_per_step', type=int, default=8, help='Prompts per step')
    parser.add_argument('--mode', default='grpo', choices=['grpo', 'ppo', 'both', 'sft_grpo', 'dpo'], help='Training mode')
    parser.add_argument('--sft_steps', type=int, default=100, help='SFT warmup steps (for sft_grpo mode)')
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 70)
    print("Mini GRPO Training Pipeline — Arithmetic Task")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Task: a+b=correct_sum (a,b ∈ {0,1,2,3,4})")
    print(f"Reward: 1.0 if correct, 0.3 if ±1, 0.1 if ±2, 0 otherwise")
    print(f"Vocab size: {VOCAB_SIZE} (digits 0-9, +, =, special)")
    print(f"Mode: {args.mode}, n_samples: {args.n_samples}")
    if args.mode == 'sft_grpo':
        print(f"SFT warmup: {args.sft_steps} steps → then {args.num_steps} GRPO steps")
    print()

    # Initialize models
    actor = MiniGQATransformer(hidden_dim=64, num_layers=2, num_heads=4,
                               num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    param_count = sum(p.numel() for p in actor.parameters())
    print(f"Actor params: {param_count:,}")

    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=args.lr)

    # SFT warmup phase
    sft_metrics_history = []
    if args.mode == 'sft_grpo':
        print("\n--- Phase 1: SFT Warmup ---")
        sft_dataset = generate_sft_dataset(500)
        sft_lr = args.lr * 2  # Higher LR for SFT
        sft_optimizer = torch.optim.AdamW(actor.parameters(), lr=sft_lr)

        for step in range(args.sft_steps):
            loss = sft_training_step(actor, sft_dataset, sft_optimizer, device)
            sft_metrics_history.append({'step': step, 'loss': loss})

            if step % 20 == 0 or step == args.sft_steps - 1:
                # Quick eval during SFT
                eval_prompts = [generate_arithmetic_prompt() for _ in range(50)]
                correct = 0
                actor.eval()
                for prompt_tokens, correct_sum in eval_prompts:
                    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
                    current_ids = prompt_tensor.clone()
                    response_tokens = []
                    for _ in range(args.max_response_len):
                        logits = actor(current_ids)
                        next_token = logits[:, -1, :].argmax(dim=-1).item()
                        if next_token == TOKENS['<eos>']:
                            break
                        response_tokens.append(next_token)
                        current_ids = torch.cat([current_ids,
                            torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
                    reward = compute_reward(response_tokens, correct_sum)
                    if reward == 1.0:
                        correct += 1
                actor.train()
                print(f"  SFT step {step:3d}: loss={loss:.4f}, eval_acc={correct}%")

        # Switch optimizer for GRPO phase (lower LR)
        actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=args.lr)

    if args.mode in ['ppo', 'both']:
        critic = SimpleCritic(hidden_dim=32, vocab_size=VOCAB_SIZE).to(device)
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr)
        critic_params = sum(p.numel() for p in critic.parameters())
        print(f"Critic params: {critic_params:,}")

    # DPO setup
    dpo_metrics_history = []
    if args.mode == 'dpo':
        dpo_pairs = generate_dpo_preference_pairs(500)
        print(f"DPO preference pairs: {len(dpo_pairs)}")

    # Training loop
    grpo_metrics_history = []
    ppo_metrics_history = []

    print("\n--- Training ---")
    for step in range(args.num_steps):
        if args.mode == 'dpo':
            metrics = dpo_training_step(actor, dpo_pairs, actor_optimizer, device, beta=0.3)
            dpo_metrics_history.append(metrics)

            if step % 20 == 0 or step == args.num_steps - 1:
                # Evaluate DPO accuracy
                eval_prompts = [generate_arithmetic_prompt() for _ in range(50)]
                correct = 0
                actor.eval()
                for prompt_tokens, correct_sum in eval_prompts:
                    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
                    current_ids = prompt_tensor.clone()
                    response_tokens = []
                    for _ in range(args.max_response_len):
                        logits = actor(current_ids)
                        next_token = logits[:, -1, :].argmax(dim=-1).item()
                        if next_token == TOKENS['<eos>']:
                            break
                        response_tokens.append(next_token)
                        current_ids = torch.cat([current_ids,
                            torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
                    reward = compute_reward(response_tokens, correct_sum)
                    if reward == 1.0:
                        correct += 1
                actor.train()
                print(f"  Step {step:3d} | DPO: loss={metrics['loss']:.4f}, "
                      f"margin={metrics['margin']:.3f}, eval_acc={correct}%")
            continue

        # Generate prompts for this step
        prompts = [generate_arithmetic_prompt() for _ in range(args.num_prompts_per_step)]

        if args.mode == 'grpo' or args.mode == 'both':
            metrics = grpo_training_step(
                actor, prompts, args.n_samples, args.max_response_len,
                actor_optimizer, device
            )
            grpo_metrics_history.append(metrics)

            if step % 20 == 0 or step == args.num_steps - 1:
                print(f"  Step {step:3d} | GRPO: loss={metrics['loss']:.4f}, "
                      f"reward={metrics['reward_mean']:.3f}, "
                      f"acc={metrics['accuracy']:.1%}, "
                      f"adv_mean={metrics['advantage_mean']:.3f}")

        if args.mode == 'ppo' or args.mode == 'both':
            # For PPO, use a fresh actor
            if args.mode == 'both' and step == 0:
                ppo_actor = MiniGQATransformer(hidden_dim=64, num_layers=2, num_heads=4,
                                             num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
                ppo_optimizer = torch.optim.AdamW(ppo_actor.parameters(), lr=args.lr)

            if args.mode == 'both':
                metrics = ppo_training_step(
                    ppo_actor, critic, prompts, args.n_samples, args.max_response_len,
                    ppo_optimizer, critic_optimizer, device
                )
            else:
                metrics = ppo_training_step(
                    actor, critic, prompts, args.n_samples, args.max_response_len,
                    actor_optimizer, critic_optimizer, device
                )
            ppo_metrics_history.append(metrics)

            if step % 20 == 0 or step == args.num_steps - 1:
                print(f"  Step {step:3d} | PPO:  loss={metrics['loss']:.4f}, "
                      f"reward={metrics['reward_mean']:.3f}, "
                      f"acc={metrics['accuracy']:.1%}")

    # Final evaluation: sample 100 prompts and check accuracy
    print("\n--- Final Evaluation ---")

    for mode_name, model in [('GRPO', actor)] if args.mode == 'grpo' else \
                                     [('GRPO', actor), ('PPO', ppo_actor)] if args.mode == 'both' else \
                                     [('PPO', actor)] if args.mode == 'ppo' else \
                                     [('SFT+GRPO', actor)] if args.mode == 'sft_grpo' else \
                                     [('DPO', actor)]:
        model.eval()
        eval_prompts = [generate_arithmetic_prompt() for _ in range(100)]
        correct = 0

        for prompt_tokens, correct_sum in eval_prompts:
            prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
            current_ids = prompt_tensor.clone()
            response_tokens = []

            for step in range(args.max_response_len):
                logits = model(current_ids)
                next_logits = logits[:, -1, :]
                # Greedy decoding for evaluation
                next_token = next_logits.argmax(dim=-1).item()
                if next_token == TOKENS['<eos>']:
                    break
                response_tokens.append(next_token)
                current_ids = torch.cat([current_ids,
                    torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)

            reward = compute_reward(response_tokens, correct_sum)
            if reward == 1.0:
                correct += 1

        print(f"  {mode_name} final accuracy: {correct}% (greedy decoding)")

        # Show some examples
        print(f"  {mode_name} examples:")
        for i in range(5):
            prompt_tokens, correct_sum = eval_prompts[i]
            prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
            current_ids = prompt_tensor.clone()
            response_tokens = []

            for step in range(args.max_response_len):
                logits = model(current_ids)
                next_logits = logits[:, -1, :]
                next_token = next_logits.argmax(dim=-1).item()
                if next_token == TOKENS['<eos>']:
                    break
                response_tokens.append(next_token)
                current_ids = torch.cat([current_ids,
                    torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)

            prompt_str = decode_tokens(prompt_tokens)
            response_str = decode_tokens(response_tokens)
            print(f"    {prompt_str}{response_str} (correct: {correct_sum})")

    # Save results
    results = {
        'config': {
            'vocab_size': VOCAB_SIZE,
            'param_count': param_count,
            'n_samples': args.n_samples,
            'num_steps': args.num_steps,
            'lr': args.lr,
            'device': str(device),
            'mode': args.mode,
        },
        'grpo_metrics': grpo_metrics_history,
        'ppo_metrics': ppo_metrics_history,
        'dpo_metrics': dpo_metrics_history,
        'sft_metrics': sft_metrics_history,
    }

    output_file = "mini_grpo_training_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if grpo_metrics_history:
        initial = grpo_metrics_history[0]
        final = grpo_metrics_history[-1]
        print(f"GRPO: reward {initial['reward_mean']:.3f} → {final['reward_mean']:.3f}, "
              f"accuracy {initial['accuracy']:.1%} → {final['accuracy']:.1%}")

    if ppo_metrics_history:
        initial = ppo_metrics_history[0]
        final = ppo_metrics_history[-1]
        print(f"PPO:  reward {initial['reward_mean']:.3f} → {final['reward_mean']:.3f}, "
              f"accuracy {initial['accuracy']:.1%} → {final['accuracy']:.1%}")

    if dpo_metrics_history:
        print(f"DPO: {len(dpo_metrics_history)} steps completed")


if __name__ == "__main__":
    main()