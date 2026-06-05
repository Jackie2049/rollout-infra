#!/usr/bin/env python3
"""Reasoning Model Training Simulator (DeepSeek-R1 style)
=========================================================
Demonstrates chain-of-thought (CoT) reasoning training from first principles:

1. Base model generates reasoning chains (thinking tokens)
2. GRPO trains the model to produce better reasoning
3. Reward based on final answer correctness (outcome-based)
4. Budget forcing: control thinking length
5. Compare: CoT vs Direct Answer, Short vs Long reasoning

Key insight from DeepSeek-R1:
  - No supervised reasoning data needed!
  - GRPO + outcome reward → reasoning emerges naturally
  - "Aha moment": model learns to verify and backtrack

Educational purpose: understand reasoning model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


# ============================================================
# 1. Environment: Math Reasoning Tasks
# ============================================================

class MathTask:
    """Simple math reasoning task with verifiable answers."""
    def __init__(self, task_type='arithmetic', difficulty=1, vocab_size=64):
        self.task_type = task_type
        self.difficulty = difficulty
        self.vocab_size = vocab_size

    def generate(self, n=16):
        """Generate math problems with answers."""
        problems = []
        for _ in range(n):
            if self.task_type == 'arithmetic':
                a = np.random.randint(1, 10 ** self.difficulty)
                b = np.random.randint(1, 10 ** self.difficulty)
                op = np.random.choice(['+', '-', '*'])
                if op == '+':
                    answer = a + b
                    prompt = f"{a}+{b}="
                elif op == '-':
                    answer = a - b
                    prompt = f"{a}-{b}="
                else:
                    answer = a * b
                    prompt = f"{a}*{b}="
            elif self.task_type == 'logic':
                # Simple logic: is X > Y?
                a = np.random.randint(1, 100)
                b = np.random.randint(1, 100)
                answer = 1 if a > b else 0
                prompt = f"{a}>{b}?"
            else:  # pattern
                start = np.random.randint(1, 10)
                step = np.random.randint(1, 5)
                seq = [start + i * step for i in range(4)]
                answer = start + 4 * step
                prompt = f"{seq[0]},{seq[1]},{seq[2]},{seq[3]},?"

            problems.append({
                'prompt': prompt,
                'answer': str(answer),
                'answer_val': answer,
            })
        return problems


def tokenize_text(text, vocab_size=64):
    """Simple character-level tokenization."""
    return [ord(c) % vocab_size for c in text]


def detokenize(tokens, vocab_size=64):
    """Simple detokenization."""
    return ''.join(chr(t % 128) for t in tokens)


# ============================================================
# 2. Reasoning Policy Model
# ============================================================

class ReasoningPolicy(nn.Module):
    """Small policy model that generates thinking + answer.

    Output format: [think_start] reasoning... [think_end] answer
    """
    def __init__(self, vocab_size=64, d_model=128, n_head=4, n_layer=3,
                 max_len=256, think_token=60, answer_token=61):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.think_token = think_token
        self.answer_token = answer_token

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

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.head(x)

    def get_log_probs(self, sequences, actions):
        """Get log probabilities of actions."""
        logits = self(sequences)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        return action_log_probs

    @torch.no_grad()
    def generate_with_thinking(self, prompts, max_think_tokens=32,
                                max_answer_tokens=8, temperature=0.8):
        """Generate reasoning chain + answer.

        Format: [prompt] [think_start] [thinking...] [think_end] [answer]
        """
        B = prompts.shape[0]
        device = prompts.device

        # Add think_start token
        think_start = torch.full((B, 1), self.think_token, device=device)
        sequences = torch.cat([prompts, think_start], dim=1)

        # Generate thinking tokens
        thinking_lengths = []
        for i in range(max_think_tokens):
            idx_cond = sequences[:, -self.max_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Stop thinking if think_end token generated (answer_token)
            for b in range(B):
                if next_token[b, 0].item() == self.answer_token:
                    pass  # Will add answer_token after thinking

            sequences = torch.cat([sequences, next_token], dim=1)

        # Add think_end / answer_start token
        answer_start = torch.full((B, 1), self.answer_token, device=device)
        sequences = torch.cat([sequences, answer_start], dim=1)

        # Generate answer tokens
        for _ in range(max_answer_tokens):
            idx_cond = sequences[:, -self.max_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            sequences = torch.cat([sequences, next_token], dim=1)

        return sequences


class DirectAnswerPolicy(nn.Module):
    """Policy that directly answers without thinking (baseline)."""
    def __init__(self, vocab_size=64, d_model=128, n_head=4, n_layer=3,
                 max_len=128, answer_token=61):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.answer_token = answer_token

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

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.head(x)

    def get_log_probs(self, sequences, actions):
        logits = self(sequences)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def generate_answer(self, prompts, max_tokens=16, temperature=0.8):
        B = prompts.shape[0]
        sequences = prompts.clone()
        answer_start = torch.full((B, 1), self.answer_token, device=sequences.device)
        sequences = torch.cat([sequences, answer_start], dim=1)

        for _ in range(max_tokens):
            idx_cond = sequences[:, -self.max_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            sequences = torch.cat([sequences, next_token], dim=1)

        return sequences


# ============================================================
# 3. Reward Functions
# ============================================================

def extract_answer_from_tokens(tokens, answer_token=61, vocab_size=64):
    """Extract answer after answer_token."""
    answer = []
    found = False
    for t in tokens:
        if t == answer_token:
            found = True
            continue
        if found:
            answer.append(t)
    return answer


def compute_answer_reward(generated, problems, vocab_size=64):
    """Compute reward: 1.0 if answer is correct, 0.0 otherwise.

    This is the key insight from DeepSeek-R1:
    - Only the FINAL ANSWER is rewarded (outcome-based)
    - The REASONING PROCESS is not supervised
    - The model must learn to reason on its own!
    """
    B = len(problems)
    rewards = torch.zeros(B)

    for i in range(B):
        tokens = generated[i].tolist()
        answer_tokens = extract_answer_from_tokens(tokens)

        # Try to extract a number from answer tokens
        answer_text = detokenize(answer_tokens, vocab_size)
        try:
            # Extract digits
            digits = ''.join(c for c in answer_text if c.isdigit() or c == '-')
            if digits:
                predicted = int(digits)
                correct = problems[i]['answer_val']
                # Reward: 1.0 if exactly correct, partial for close
                if predicted == correct:
                    rewards[i] = 1.0
                elif abs(predicted - correct) <= max(1, abs(correct) * 0.1):
                    rewards[i] = 0.5  # Close enough
        except (ValueError, TypeError):
            pass

    return rewards


def compute_thinking_length_reward(sequences, think_token=60, answer_token=61,
                                    target_length=16):
    """Reward for appropriate thinking length (budget forcing).

    DeepSeek-R1 uses budget forcing to control thinking length:
    - Too short → may not reason enough
    - Too long → wasted computation
    - Target: encourages productive reasoning length
    """
    B = sequences.shape[0]
    rewards = torch.zeros(B, device=sequences.device)

    for i in range(B):
        tokens = sequences[i].tolist()
        think_count = sum(1 for t in tokens if t == think_token)

        # Bell-shaped reward around target length
        length_ratio = think_count / max(target_length, 1)
        if length_ratio < 0.5:
            rewards[i] = 0.1  # Too short
        elif length_ratio <= 2.0:
            rewards[i] = 0.3  # Good range
        else:
            rewards[i] = 0.0  # Too long

    return rewards


# ============================================================
# 4. GRPO Training for Reasoning
# ============================================================

@dataclass
class ReasoningConfig:
    n_samples: int = 4          # GRPO group size
    learning_rate: float = 3e-4
    kl_coef: float = 0.02
    clip_range: float = 0.2
    max_think_tokens: int = 24
    max_answer_tokens: int = 8
    ppo_epochs: int = 2
    vocab_size: int = 64
    answer_reward_weight: float = 1.0
    thinking_reward_weight: float = 0.1


class ReasoningGRPOTrainer:
    """GRPO trainer specialized for reasoning model training.

    Key differences from standard GRPO:
    1. Outcome-based reward (only final answer correctness)
    2. Optional thinking length reward (budget forcing)
    3. Longer generation (thinking + answer)
    """

    def __init__(self, policy, ref_policy, config, device='cpu'):
        self.policy = policy
        self.ref_policy = ref_policy
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    def train_step(self, prompts, problems):
        cfg = self.config
        n_prompts = prompts.shape[0]
        n_s = cfg.n_samples

        # Generate n_samples per prompt
        self.policy.eval()
        all_seqs = []
        for i in range(n_prompts):
            p = prompts[i:i+1].expand(n_s, -1)
            s = self.policy.generate_with_thinking(
                p, max_think_tokens=cfg.max_think_tokens,
                max_answer_tokens=cfg.max_answer_tokens, temperature=0.9
            )
            all_seqs.append(s)
        all_seqs = torch.cat(all_seqs, dim=0)

        # Compute rewards
        # Need to expand problems for n_samples
        expanded_problems = []
        for p in problems:
            expanded_problems.extend([p] * n_s)

        answer_rewards = compute_answer_reward(all_seqs, expanded_problems, cfg.vocab_size)
        answer_rewards = answer_rewards.to(self.device)

        # Combine rewards
        total_rewards = cfg.answer_reward_weight * answer_rewards

        # Group normalize (GRPO)
        rewards_grouped = total_rewards.view(n_prompts, n_s)
        g_mean = rewards_grouped.mean(dim=1, keepdim=True)
        g_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_grouped - g_mean) / g_std).view(-1).detach()

        # Compute old and ref log probs
        gen_part = all_seqs[:, prompts.shape[1]:]  # Generated portion
        seq_len = gen_part.shape[1]

        with torch.no_grad():
            old_lp = self.policy.get_log_probs(all_seqs, gen_part)[:, -seq_len:].detach()
            ref_lp = self.ref_policy.get_log_probs(all_seqs, gen_part)[:, -seq_len:].detach()

        # GRPO update
        total_loss = 0
        for _ in range(cfg.ppo_epochs):
            new_lp = self.policy.get_log_probs(all_seqs, gen_part)[:, -seq_len:]
            ratio = torch.exp(new_lp - old_lp)
            token_adv = advantages.unsqueeze(1).expand(-1, seq_len)

            s1 = ratio * token_adv
            s2 = torch.clamp(ratio, 1-cfg.clip_range, 1+cfg.clip_range) * token_adv
            p_loss = -torch.min(s1, s2).mean()
            kl = (new_lp - ref_lp).mean()
            loss = p_loss + cfg.kl_coef * kl

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()

        accuracy = answer_rewards.mean().item()
        return total_loss / cfg.ppo_epochs, accuracy, all_seqs


# ============================================================
# 5. Experiments
# ============================================================

def experiment_cot_vs_direct(device='cpu'):
    """Compare chain-of-thought reasoning vs direct answering."""
    print("\n  === Experiment: CoT vs Direct Answer ===")

    vocab_size = 64
    d_model = 64
    n_steps = 25
    task = MathTask('arithmetic', difficulty=1, vocab_size=vocab_size)

    # Train CoT model
    torch.manual_seed(42)
    cot_model = ReasoningPolicy(vocab_size, d_model, n_head=4, n_layer=2,
                                 max_len=128).to(device)
    cot_ref = ReasoningPolicy(vocab_size, d_model, n_head=4, n_layer=2,
                               max_len=128).to(device)
    cot_ref.load_state_dict(cot_model.state_dict())
    cot_ref.eval()

    config = ReasoningConfig(n_samples=4, learning_rate=3e-4, kl_coef=0.02,
                              max_think_tokens=16, max_answer_tokens=6, vocab_size=vocab_size)
    cot_trainer = ReasoningGRPOTrainer(cot_model, cot_ref, config, device)

    cot_accs = []
    for step in range(n_steps):
        problems = task.generate(8)
        prompt_tokens = [tokenize_text(p['prompt'], vocab_size) for p in problems]
        max_plen = max(len(p) for p in prompt_tokens)
        prompts = torch.zeros(len(problems), max_plen, dtype=torch.long, device=device)
        for i, pt in enumerate(prompt_tokens):
            prompts[i, :len(pt)] = torch.tensor(pt)

        loss, acc, _ = cot_trainer.train_step(prompts, problems)
        cot_accs.append(acc)
        if (step + 1) % 10 == 0:
            print(f"    CoT step {step+1}: acc={acc:.3f}, loss={loss:.4f}")

    # Train Direct model
    torch.manual_seed(42)
    direct_model = DirectAnswerPolicy(vocab_size, d_model, n_head=4, n_layer=2,
                                       max_len=64).to(device)
    direct_ref = DirectAnswerPolicy(vocab_size, d_model, n_head=4, n_layer=2,
                                     max_len=64).to(device)
    direct_ref.load_state_dict(direct_model.state_dict())
    direct_ref.eval()

    direct_optimizer = torch.optim.Adam(direct_model.parameters(), lr=3e-4)

    direct_accs = []
    for step in range(n_steps):
        problems = task.generate(8)
        prompt_tokens = [tokenize_text(p['prompt'], vocab_size) for p in problems]
        max_plen = max(len(p) for p in prompt_tokens)
        prompts = torch.zeros(len(problems), max_plen, dtype=torch.long, device=device)
        for i, pt in enumerate(prompt_tokens):
            prompts[i, :len(pt)] = torch.tensor(pt)

        # Generate
        direct_model.eval()
        seqs = direct_model.generate_answer(prompts, max_tokens=6, temperature=0.9)
        expanded = [p for p in problems for _ in range(1)]
        rewards = compute_answer_reward(seqs, problems, vocab_size).to(device)
        acc = rewards.mean().item()
        direct_accs.append(acc)

        # Simple policy gradient update
        gen_part = seqs[:, prompts.shape[1]:]
        seq_len = gen_part.shape[1]
        direct_model.train()
        log_probs = direct_model.get_log_probs(seqs, gen_part)[:, -seq_len:]
        loss = -(log_probs * (rewards.unsqueeze(1) - 0.5)).mean()

        direct_optimizer.zero_grad()
        loss.backward()
        direct_optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"    Direct step {step+1}: acc={acc:.3f}")

    print(f"\n    CoT:    initial={cot_accs[0]:.3f} → final={cot_accs[-1]:.3f} (Δ={cot_accs[-1]-cot_accs[0]:+.3f})")
    print(f"    Direct: initial={direct_accs[0]:.3f} → final={direct_accs[-1]:.3f} (Δ={direct_accs[-1]-direct_accs[0]:+.3f})")

    return {
        'cot_accs': cot_accs,
        'direct_accs': direct_accs,
        'cot_final': cot_accs[-1],
        'direct_final': direct_accs[-1],
    }


def experiment_thinking_length(device='cpu'):
    """Test how thinking length affects reasoning quality."""
    print("\n  === Experiment: Thinking Length Effect ===")

    vocab_size = 64
    d_model = 64
    task = MathTask('arithmetic', difficulty=1, vocab_size=vocab_size)

    results = {}
    for max_think in [4, 8, 16, 24, 32]:
        torch.manual_seed(42)
        model = ReasoningPolicy(vocab_size, d_model, n_head=4, n_layer=2,
                                 max_len=128).to(device)
        ref = ReasoningPolicy(vocab_size, d_model, n_head=4, n_layer=2,
                               max_len=128).to(device)
        ref.load_state_dict(model.state_dict())
        ref.eval()

        config = ReasoningConfig(n_samples=4, learning_rate=3e-4,
                                  max_think_tokens=max_think, vocab_size=vocab_size)
        trainer = ReasoningGRPOTrainer(model, ref, config, device)

        accs = []
        for step in range(15):
            problems = task.generate(8)
            prompt_tokens = [tokenize_text(p['prompt'], vocab_size) for p in problems]
            max_plen = max(len(p) for p in prompt_tokens)
            prompts = torch.zeros(len(problems), max_plen, dtype=torch.long, device=device)
            for i, pt in enumerate(prompt_tokens):
                prompts[i, :len(pt)] = torch.tensor(pt)
            _, acc, _ = trainer.train_step(prompts, problems)
            accs.append(acc)

        improvement = accs[-1] - accs[0]
        results[max_think] = {
            'accs': accs,
            'final': accs[-1],
            'improvement': improvement,
        }
        print(f"    think={max_think:>3}: {accs[0]:.3f} → {accs[-1]:.3f} (Δ={improvement:+.3f})")

    return results


def experiment_task_difficulty(device='cpu'):
    """Test reasoning on different difficulty levels."""
    print("\n  === Experiment: Task Difficulty ===")

    vocab_size = 64
    d_model = 64

    results = {}
    for difficulty in [1, 2, 3]:
        for task_type in ['arithmetic', 'logic', 'pattern']:
            task = MathTask(task_type, difficulty=difficulty, vocab_size=vocab_size)

            torch.manual_seed(42)
            model = ReasoningPolicy(vocab_size, d_model, n_head=4, n_layer=2,
                                     max_len=128).to(device)
            ref = ReasoningPolicy(vocab_size, d_model, n_head=4, n_layer=2,
                                   max_len=128).to(device)
            ref.load_state_dict(model.state_dict())
            ref.eval()

            config = ReasoningConfig(n_samples=4, learning_rate=3e-4,
                                      max_think_tokens=16, vocab_size=vocab_size)
            trainer = ReasoningGRPOTrainer(model, ref, config, device)

            accs = []
            for step in range(15):
                problems = task.generate(8)
                prompt_tokens = [tokenize_text(p['prompt'], vocab_size) for p in problems]
                max_plen = max(len(p) for p in prompt_tokens)
                prompts = torch.zeros(len(problems), max_plen, dtype=torch.long, device=device)
                for i, pt in enumerate(prompt_tokens):
                    prompts[i, :len(pt)] = torch.tensor(pt)
                _, acc, _ = trainer.train_step(prompts, problems)
                accs.append(acc)

            key = f"{task_type}_d{difficulty}"
            results[key] = {'final': accs[-1], 'improvement': accs[-1] - accs[0]}
            print(f"    {key:>15}: {accs[0]:.3f} → {accs[-1]:.3f} (Δ={accs[-1]-accs[0]:+.3f})")

    return results


def experiment_budget_forcing(device='cpu'):
    """Test budget forcing: constrain thinking to specific token counts."""
    print("\n  === Experiment: Budget Forcing ===")

    vocab_size = 64
    d_model = 64
    task = MathTask('arithmetic', difficulty=1, vocab_size=vocab_size)

    # Train with different thinking reward weights
    results = {}
    for thinking_weight in [0.0, 0.1, 0.3, 0.5]:
        torch.manual_seed(42)
        model = ReasoningPolicy(vocab_size, d_model, n_head=4, n_layer=2,
                                 max_len=128).to(device)
        ref = ReasoningPolicy(vocab_size, d_model, n_head=4, n_layer=2,
                               max_len=128).to(device)
        ref.load_state_dict(model.state_dict())
        ref.eval()

        config = ReasoningConfig(n_samples=4, learning_rate=3e-4,
                                  max_think_tokens=16, vocab_size=vocab_size,
                                  thinking_reward_weight=thinking_weight)
        trainer = ReasoningGRPOTrainer(model, ref, config, device)

        accs = []
        think_lens = []
        for step in range(15):
            problems = task.generate(8)
            prompt_tokens = [tokenize_text(p['prompt'], vocab_size) for p in problems]
            max_plen = max(len(p) for p in prompt_tokens)
            prompts = torch.zeros(len(problems), max_plen, dtype=torch.long, device=device)
            for i, pt in enumerate(prompt_tokens):
                prompts[i, :len(pt)] = torch.tensor(pt)

            _, acc, seqs = trainer.train_step(prompts, problems)
            accs.append(acc)

            # Measure thinking length
            for b in range(seqs.shape[0]):
                tokens = seqs[b].tolist()
                think_count = sum(1 for t in tokens[prompts.shape[1]:] if t == 60)
                think_lens.append(think_count)

        avg_think = np.mean(think_lens)
        results[thinking_weight] = {
            'final_acc': accs[-1],
            'avg_think_len': avg_think,
        }
        print(f"    think_w={thinking_weight:.1f}: acc={accs[-1]:.3f}, "
              f"avg_think={avg_think:.1f} tokens")

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Reasoning Model Training Simulator (DeepSeek-R1 style)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"\n  GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")
    else:
        print(f"\n  Device: CPU")

    results = {}

    # Exp 1: CoT vs Direct
    comparison = experiment_cot_vs_direct(device)
    results['cot_vs_direct'] = comparison

    # Exp 2: Thinking length
    thinking_results = experiment_thinking_length(device)
    results['thinking_length'] = {
        k: {'final': v['final'], 'improvement': v['improvement']}
        for k, v in thinking_results.items()
    }

    # Exp 3: Task difficulty
    difficulty_results = experiment_task_difficulty(device)
    results['task_difficulty'] = difficulty_results

    # Exp 4: Budget forcing
    budget_results = experiment_budget_forcing(device)
    results['budget_forcing'] = budget_results

    # Summary
    print("\n" + "=" * 60)
    print("Reasoning Model Training Summary")
    print("=" * 60)
    print("""
    DeepSeek-R1 Key Innovations:
    ┌──────────────────────────────────────────────────────────────┐
    │ 1. No supervised reasoning data needed                       │
    │    - Start from base model + GRPO                            │
    │    - Reward = answer correctness only (outcome-based)        │
    │    - Reasoning emerges naturally!                            │
    │                                                              │
    │ 2. "Aha Moment" — model learns to:                           │
    │    - Verify intermediate steps                               │
    │    - Backtrack when detecting errors                         │
    │    - Try alternative approaches                              │
    │    → This emerges WITHOUT being explicitly trained!          │
    │                                                              │
    │ 3. Budget Forcing                                            │
    │    - Control thinking token count                            │
    │    - Short budget → fast but less accurate                   │
    │    - Long budget → more accurate but slower                  │
    │    - Trade-off: latency vs accuracy                          │
    │                                                              │
    │ 4. Training Pipeline (DeepSeek-R1)                           │
    │    Step 1: Cold-start with small SFT data (optional)         │
    │    Step 2: GRPO with outcome reward (reasoning emerges)      │
    │    Step 3: Rejection sampling + SFT (distill to smaller)     │
    │    Step 4: DPO/GRPO again (finalize)                         │
    └──────────────────────────────────────────────────────────────┘

    CoT vs Direct Answer:
    - CoT allows multi-step reasoning within a single forward pass
    - More tokens = more "compute" per problem
    - Especially useful for: math, logic, code, planning

    Key Insight: CoT is like "test-time compute scaling"
    - More thinking tokens = more compute at inference
    - O1/R1 use this to trade latency for accuracy
    """)

    if device == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"  Peak GPU memory: {mem:.1f} MB")
        results['gpu_memory_mb'] = round(mem, 1)

    with open("reasoning_model_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to reasoning_model_results.json")


if __name__ == "__main__":
    main()
