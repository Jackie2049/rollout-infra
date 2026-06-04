#!/usr/bin/env python3
"""DPO (Direct Preference Optimization) Training Pipeline
==========================================================
Demonstrates the DPO algorithm from scratch:
1. Pretrain: Next-token prediction (reuse MiniGPT)
2. SFT: Supervised fine-tuning (same as mini_train_pipeline)
3. DPO: Direct Preference Optimization — align model to preferences
   without a separate reward model or RL loop

DPO Loss:
    L_DPO = -E[log σ(β * (log π_θ(y_w|x) / π_ref(y_w|x)
                           - log π_θ(y_l|x) / π_ref(y_l|x)))]

Key insight: DPO converts RLHF into a simple classification problem.
Instead of training a reward model and doing PPO, it directly optimizes
the policy using preference pairs (chosen vs rejected).

Runs on GPU (RTX 4090) with a ~838K parameter GPT model.
Uses synthetic data — no model download needed.

Educational purpose: understand alignment from first principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import os
import copy


# ============================================================
# Model: Reuse MiniGPT from mini_train_pipeline.py
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size=256, block_size=256, n_embd=128,
                 n_head=4, n_layer=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def get_log_probs(self, idx, targets):
        """Get per-token log probabilities for DPO training.

        Args:
            idx: [B, T] input token IDs (includes prompt + response)
            targets: [B, T] target token IDs (shifted by 1)

        Returns:
            log_probs: [B, T] log probabilities for each target token
            loss: scalar cross-entropy loss
        """
        logits, loss = self(idx, targets)
        # log_probs[i] = log P(target_i | idx_0..idx_i)
        # logits: [B, T, V], targets: [B, T]
        log_probs = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='none'
        ).reshape(idx.shape[0], idx.shape[1])
        return log_probs, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Data: Synthetic Pretraining + SFT + Preference Data
# ============================================================

def create_pretrain_data(vocab_size=256, seq_len=128, num_sequences=2000, device='cpu'):
    """Create synthetic pretraining data (character-level)."""
    data = []
    for _ in range(num_sequences):
        pattern_type = torch.randint(0, 4, (1,)).item()
        if pattern_type == 0:
            pattern_len = torch.randint(4, 16, (1,)).item()
            pattern = torch.randint(10, vocab_size // 4, (pattern_len,))
            repeats = seq_len // pattern_len + 1
            seq = pattern.repeat(repeats)[:seq_len]
        elif pattern_type == 1:
            start = torch.randint(0, vocab_size // 4, (1,)).item()
            step = torch.randint(1, 5, (1,)).item()
            seq = torch.tensor([(start + i * step) % (vocab_size // 2) for i in range(seq_len)])
        elif pattern_type == 2:
            a, b = torch.randint(1, 20, (2,)).tolist()
            fib = [a % (vocab_size // 2), b % (vocab_size // 2)]
            for i in range(seq_len - 2):
                fib.append((fib[-1] + fib[-2]) % (vocab_size // 2))
            seq = torch.tensor(fib)
        else:
            bias = torch.zeros(vocab_size)
            center = torch.randint(10, vocab_size // 2, (1,)).item()
            bias[max(0, center - 20):center + 20] = 1.0
            bias = bias / bias.sum()
            seq = torch.multinomial(bias, seq_len, replacement=True)
        data.append(seq)
    return torch.stack(data).to(device)


def create_sft_data(vocab_size=256, seq_len=64, num_pairs=500, device='cpu'):
    """Create synthetic SFT data (instruction → response)."""
    INST_START = vocab_size - 4
    INST_END = vocab_size - 3
    RESP_START = vocab_size - 2
    RESP_END = vocab_size - 1

    pairs = []
    for _ in range(num_pairs):
        inst_len = torch.randint(8, 24, (1,)).item()
        instruction = torch.randint(10, vocab_size // 2, (inst_len,))

        resp_type = torch.randint(0, 2, (1,)).item()
        if resp_type == 0:
            response = instruction.flip(0) % (vocab_size // 2)
        else:
            response = (instruction * 2) % (vocab_size // 2)

        full = torch.cat([
            torch.tensor([INST_START]),
            instruction,
            torch.tensor([INST_END, RESP_START]),
            response,
            torch.tensor([RESP_END]),
        ])
        if len(full) > seq_len:
            full = full[:seq_len]
        else:
            full = torch.cat([full, torch.zeros(seq_len - len(full), dtype=torch.long)])
        pairs.append(full)

    return torch.stack(pairs).to(device), INST_START, INST_END, RESP_START, RESP_END


def create_preference_data(vocab_size=256, seq_len=64, num_pairs=300, device='cpu'):
    """Create synthetic preference data for DPO.

    Each sample: (prompt, chosen_response, rejected_response)
    - Chosen: follows the correct pattern (reverse or double)
    - Rejected: random tokens (doesn't follow pattern)

    This simulates real preference data where:
    - chosen = human-preferred response
    - rejected = human-dispreferred response
    """
    INST_START = vocab_size - 4
    INST_END = vocab_size - 3
    RESP_START = vocab_size - 2
    RESP_END = vocab_size - 1

    preference_pairs = []
    labels = []  # For evaluation: 0=reverse, 1=double

    for _ in range(num_pairs):
        inst_len = torch.randint(6, 16, (1,)).item()
        instruction = torch.randint(10, vocab_size // 2, (inst_len,))
        resp_type = torch.randint(0, 2, (1,)).item()

        # Chosen: correct pattern
        if resp_type == 0:
            chosen_response = instruction.flip(0) % (vocab_size // 2)
        else:
            chosen_response = (instruction * 2) % (vocab_size // 2)

        # Rejected: random (no pattern)
        rejected_response = torch.randint(10, vocab_size // 2, (inst_len,))

        def make_full(resp):
            full = torch.cat([
                torch.tensor([INST_START]),
                instruction,
                torch.tensor([INST_END, RESP_START]),
                resp,
                torch.tensor([RESP_END]),
            ])
            if len(full) > seq_len:
                full = full[:seq_len]
            else:
                full = torch.cat([full, torch.zeros(seq_len - len(full), dtype=torch.long)])
            return full

        chosen_full = make_full(chosen_response).to(device)
        rejected_full = make_full(rejected_response).to(device)

        preference_pairs.append((chosen_full, rejected_full))
        labels.append(resp_type if isinstance(resp_type, int) else resp_type.item())

    return preference_pairs, labels


# ============================================================
# DPO Training
# ============================================================

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Compute DPO loss.

    L_DPO = -E[log σ(β * (log(π_θ/π_ref)(y_w) - log(π_θ/π_ref)(y_l)))]

    Args:
        policy_chosen_logps: [B] log P_θ(chosen|prompt)
        policy_rejected_logps: [B] log P_θ(rejected|prompt)
        ref_chosen_logps: [B] log P_ref(chosen|prompt)
        ref_rejected_logps: [B] log P_ref(rejected|prompt)
        beta: temperature controlling deviation from reference

    Returns:
        loss: scalar
        chosen_rewards: [B] implicit reward for chosen
        rejected_rewards: [B] implicit reward for rejected
    """
    # Log-ratio: how much policy deviates from reference
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    # Implicit rewards (for monitoring, not training)
    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios

    # DPO loss: -log σ(β * (log_ratio_chosen - log_ratio_rejected))
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()

    # Accuracy: how often chosen reward > rejected reward
    accuracy = (chosen_rewards > rejected_rewards).float().mean()

    return loss, chosen_rewards, rejected_rewards, accuracy


def train_dpo(policy_model, ref_model, preference_data, labels,
              epochs=10, batch_size=16, lr=5e-5, beta=0.1,
              max_grad_norm=1.0):
    """DPO training loop.

    Args:
        policy_model: The model being trained (starts from SFT checkpoint)
        ref_model: Frozen reference model (same as SFT checkpoint)
        preference_data: List of (chosen_seq, rejected_seq) tensors
        labels: Response type labels for evaluation
    """
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * (len(preference_data) // batch_size)
    )

    ref_model.eval()
    policy_model.train()

    metrics = {
        'losses': [], 'accuracies': [], 'chosen_rewards': [],
        'rejected_rewards': [], 'reward_margins': [], 'tokens_per_sec': []
    }

    for epoch in range(epochs):
        # Shuffle
        indices = torch.randperm(len(preference_data))
        epoch_loss = 0
        epoch_acc = 0
        epoch_chosen_r = 0
        epoch_rejected_r = 0
        n_batches = 0

        for i in range(0, len(preference_data) - batch_size, batch_size):
            batch_indices = indices[i:i + batch_size]

            # Build batch tensors
            chosen_batch = torch.stack([preference_data[j][0] for j in batch_indices])
            rejected_batch = torch.stack([preference_data[j][1] for j in batch_indices])

            start = time.time()

            # Get log probs from policy model
            chosen_x = chosen_batch[:, :-1]
            chosen_y = chosen_batch[:, 1:]
            rejected_x = rejected_batch[:, :-1]
            rejected_y = rejected_batch[:, 1:]

            policy_chosen_logps, _ = policy_model.get_log_probs(chosen_x, chosen_y)
            policy_rejected_logps, _ = policy_model.get_log_probs(rejected_x, rejected_y)

            # Get log probs from reference model (no grad)
            with torch.no_grad():
                ref_chosen_logps, _ = ref_model.get_log_probs(chosen_x, chosen_y)
                ref_rejected_logps, _ = ref_model.get_log_probs(rejected_x, rejected_y)

            # Sum log probs over sequence to get log P(y|x)
            policy_chosen_logps = policy_chosen_logps.sum(dim=-1)
            policy_rejected_logps = policy_rejected_logps.sum(dim=-1)
            ref_chosen_logps = ref_chosen_logps.sum(dim=-1)
            ref_rejected_logps = ref_rejected_logps.sum(dim=-1)

            # DPO loss
            loss, chosen_r, rejected_r, accuracy = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps, beta=beta
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            elapsed = time.time() - start
            n_tokens = chosen_x.numel() + rejected_x.numel()
            tps = n_tokens / elapsed

            epoch_loss += loss.item()
            epoch_acc += accuracy.item()
            epoch_chosen_r += chosen_r.mean().item()
            epoch_rejected_r += rejected_r.mean().item()
            n_batches += 1
            metrics['tokens_per_sec'].append(tps)

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(n_batches, 1)
        avg_chosen_r = epoch_chosen_r / max(n_batches, 1)
        avg_rejected_r = epoch_rejected_r / max(n_batches, 1)
        reward_margin = avg_chosen_r - avg_rejected_r

        metrics['losses'].append(avg_loss)
        metrics['accuracies'].append(avg_acc)
        metrics['chosen_rewards'].append(avg_chosen_r)
        metrics['rejected_rewards'].append(avg_rejected_r)
        metrics['reward_margins'].append(reward_margin)

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            avg_tps = sum(metrics['tokens_per_sec'][-n_batches:]) / max(n_batches, 1)
            lr_now = scheduler.get_last_lr()[0]
            print(f"  [dpo] Epoch {epoch:>3}/{epochs}: loss={avg_loss:.4f}, "
                  f"acc={avg_acc:.3f}, margin={reward_margin:.4f}, "
                  f"chosen_r={avg_chosen_r:.4f}, rejected_r={avg_rejected_r:.4f}, "
                  f"lr={lr_now:.2e}, tps={avg_tps:,.0f}")

    return metrics


def train_sft(model, data, epochs, batch_size, lr, max_grad_norm=1.0, label="sft"):
    """SFT training loop (same as mini_train_pipeline)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * (len(data) // batch_size)
    )
    model.train()
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(len(data))
        data_shuffled = data[perm]
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(data) - batch_size, batch_size):
            batch = data_shuffled[i:i + batch_size]
            x = batch[:, :-1]
            y = batch[:, 1:]

            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"  [{label}] Epoch {epoch:>3}/{epochs}: loss={avg_loss:.4f}")

    return losses


# ============================================================
# Evaluation
# ============================================================

def evaluate_preference(model, preference_data, labels, INST_START, INST_END,
                        RESP_START, n_eval=20, device='cpu'):
    """Evaluate if model generates correct (chosen-like) responses."""
    model.eval()
    correct = 0
    total = 0

    for i in range(min(n_eval, len(preference_data))):
        chosen = preference_data[i][0]
        # Extract instruction portion
        inst_start_idx = (chosen == INST_START).nonzero(as_tuple=True)[0]
        inst_end_idx = (chosen == INST_END).nonzero(as_tuple=True)[0]

        if len(inst_start_idx) == 0 or len(inst_end_idx) == 0:
            continue

        s = inst_start_idx[0].item() + 1
        e = inst_end_idx[0].item()
        instruction = chosen[s:e]

        if len(instruction) < 2:
            continue

        # Build prompt: [INST_START] instruction [INST_END] [RESP_START]
        prompt = torch.cat([
            torch.tensor([INST_START], device=device),
            instruction,
            torch.tensor([INST_END, RESP_START], device=device),
        ]).unsqueeze(0).to(device)

        # Generate
        response = model.generate(prompt, max_new_tokens=len(instruction) + 4,
                                  temperature=0.3, top_k=20)

        # Check if response matches either reverse or double
        inst_list = instruction.tolist()
        generated = response[0, prompt.shape[1]:prompt.shape[1] + len(inst_list)].tolist()

        reverse_expected = inst_list[::-1]
        double_expected = [(x * 2) % 128 for x in inst_list]

        match_reverse = sum(a == e for a, e in zip(generated, reverse_expected))
        match_double = sum(a == e for a, e in zip(generated, double_expected))

        # If either pattern matches >50%, count as correct
        if max(match_reverse, match_double) > len(inst_list) * 0.5:
            correct += 1
        total += 1

    return correct, total


# ============================================================
# Experiments
# ============================================================

def experiment_beta_sweep(policy_model, ref_model, preference_data, labels,
                          vocab_size, device):
    """Exp: Sweep β (DPO temperature) to find optimal value."""
    print("\n  --- Experiment: β Sweep ---")
    print(f"  {'β':>6} {'Loss':>8} {'Accuracy':>10} {'Margin':>10} {'Chosen_R':>10} {'Rejected_R':>12}")
    print("  " + "-" * 60)

    results = {}
    for beta in [0.05, 0.1, 0.3, 0.5, 1.0]:
        # Reset to SFT checkpoint
        policy_model.load_state_dict(ref_model.state_dict())
        metrics = train_dpo(policy_model, ref_model, preference_data, labels,
                           epochs=15, batch_size=16, lr=5e-5, beta=beta)

        final = {k: v[-1] for k, v in metrics.items() if isinstance(v, list) and v}
        results[beta] = final
        print(f"  {beta:>6.2f} {final['losses']:>8.4f} {final['accuracies']:>10.3f} "
              f"{final['reward_margins']:>10.4f} {final['chosen_rewards']:>10.4f} "
              f"{final['rejected_rewards']:>12.4f}")

    return results


def experiment_dpo_vs_sft_only(model_sft, model_dpo, preference_data, labels,
                                sft_data, INST_START, INST_END, RESP_START, device):
    """Exp: Compare DPO-aligned vs SFT-only model."""
    print("\n  --- Experiment: DPO vs SFT-Only ---")

    # Evaluate both on preference accuracy
    for name, model in [("SFT-only", model_sft), ("SFT+DPO", model_dpo)]:
        model.eval()
        correct, total = evaluate_preference(
            model, preference_data, labels, INST_START, INST_END, RESP_START, device=device
        )
        print(f"  {name}: {correct}/{total} correct ({correct/max(total,1)*100:.0f}%)")

    # Compare NLL on SFT data
    sft_x = sft_data[:50, :-1]
    sft_y = sft_data[:50, 1:]

    for name, model in [("SFT-only", model_sft), ("SFT+DPO", model_dpo)]:
        model.eval()
        with torch.no_grad():
            _, loss = model(sft_x, sft_y)
        print(f"  {name} SFT NLL: {loss.item():.4f}")


def experiment_dpo_convergence(policy_model, ref_model, preference_data, labels):
    """Exp: Track DPO training dynamics over epochs."""
    print("\n  --- Experiment: DPO Training Dynamics ---")

    metrics = train_dpo(policy_model, ref_model, preference_data, labels,
                       epochs=30, batch_size=16, lr=5e-5, beta=0.1)

    print("\n  Training trajectory:")
    print(f"  {'Epoch':>6} {'Loss':>8} {'Accuracy':>10} {'Margin':>10}")
    for i in [0, 4, 9, 14, 19, 24, 29]:
        if i < len(metrics['losses']):
            print(f"  {i:>6} {metrics['losses'][i]:>8.4f} "
                  f"{metrics['accuracies'][i]:>10.3f} "
                  f"{metrics['reward_margins'][i]:>10.4f}")

    return metrics


def experiment_length_normalized_dpo(policy_model, ref_model, preference_data, labels):
    """Exp: Length-normalized DPO (divide log probs by sequence length).

    Without normalization, longer sequences have more extreme log-ratios,
    which can bias the loss. Length normalization helps with varying-length responses.
    """
    print("\n  --- Experiment: Length-Normalized DPO ---")

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-5, weight_decay=0.01)
    ref_model.eval()
    policy_model.train()

    results = {'normalized': {}, 'unnormalized': {}}

    for use_norm in [False, True]:
        label = "normalized" if use_norm else "unnormalized"
        policy_model.load_state_dict(ref_model.state_dict())
        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-5, weight_decay=0.01)
        policy_model.train()

        for epoch in range(15):
            indices = torch.randperm(len(preference_data))
            epoch_loss = 0
            n_batches = 0

            for i in range(0, len(preference_data) - 16, 16):
                batch_idx = indices[i:i + 16]
                chosen = torch.stack([preference_data[j][0] for j in batch_idx])
                rejected = torch.stack([preference_data[j][1] for j in batch_idx])

                policy_chosen_logps, _ = policy_model.get_log_probs(chosen[:, :-1], chosen[:, 1:])
                policy_rejected_logps, _ = policy_model.get_log_probs(rejected[:, :-1], rejected[:, 1:])

                with torch.no_grad():
                    ref_chosen_logps, _ = ref_model.get_log_probs(chosen[:, :-1], chosen[:, 1:])
                    ref_rejected_logps, _ = ref_model.get_log_probs(rejected[:, :-1], rejected[:, 1:])

                if use_norm:
                    # Count non-padding tokens for normalization
                    # Padding = 0 tokens
                    chosen_len = (chosen[:, 1:] > 0).sum(dim=-1).float().clamp(min=1)
                    rejected_len = (rejected[:, 1:] > 0).sum(dim=-1).float().clamp(min=1)
                    policy_chosen_logps = (policy_chosen_logps.sum(dim=-1)) / chosen_len
                    policy_rejected_logps = (policy_rejected_logps.sum(dim=-1)) / rejected_len
                    ref_chosen_logps = (ref_chosen_logps.sum(dim=-1)) / chosen_len
                    ref_rejected_logps = (ref_rejected_logps.sum(dim=-1)) / rejected_len
                else:
                    policy_chosen_logps = policy_chosen_logps.sum(dim=-1)
                    policy_rejected_logps = policy_rejected_logps.sum(dim=-1)
                    ref_chosen_logps = ref_chosen_logps.sum(dim=-1)
                    ref_rejected_logps = ref_rejected_logps.sum(dim=-1)

                loss, _, _, accuracy = dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps, beta=0.1
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if epoch % 5 == 0 or epoch == 14:
                print(f"  [{label}] Epoch {epoch}: loss={avg_loss:.4f}")

        results[label] = {'final_loss': avg_loss}

    return results


# ============================================================
# Main: Full Pipeline
# ============================================================

def main():
    print("=" * 60)
    print("DPO Training Pipeline — Pretrain → SFT → DPO Alignment")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"\n  GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")
    else:
        print(f"\n  Device: CPU")

    results = {}

    # Model config
    vocab_size = 256
    block_size = 128
    n_embd = 128
    n_head = 4
    n_layer = 4

    # ===== Phase 1: Pretraining =====
    print("\n" + "=" * 60)
    print("Phase 1: Pretraining (Next-Token Prediction)")
    print("=" * 60)

    model = MiniGPT(vocab_size, block_size, n_embd, n_head, n_layer, dropout=0.1).to(device)
    n_params = model.count_parameters()
    print(f"\n  Model: {n_params:,} parameters")

    pt_data = create_pretrain_data(vocab_size, seq_len=block_size, num_sequences=2000, device=device)
    print(f"  Pretrain data: {pt_data.shape[0]} sequences")

    # Quick pretrain (fewer epochs — we did 30 last time, 15 is enough to learn patterns)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=15 * (len(pt_data) // 32)
    )
    model.train()

    for epoch in range(15):
        perm = torch.randperm(len(pt_data))
        data_shuffled = pt_data[perm]
        epoch_loss = 0
        n_batches = 0
        for i in range(0, len(pt_data) - 32, 32):
            batch = data_shuffled[i:i + 32]
            _, loss = model(batch[:, :-1], batch[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1
        if epoch % 5 == 0 or epoch == 14:
            print(f"  [pretrain] Epoch {epoch}: loss={epoch_loss/max(n_batches,1):.4f}")

    results["pretrain"] = {"params": n_params, "final_loss": round(epoch_loss / max(n_batches, 1), 4)}

    # ===== Phase 2: SFT =====
    print("\n" + "=" * 60)
    print("Phase 2: SFT (Supervised Fine-Tuning)")
    print("=" * 60)

    sft_data, INST_START, INST_END, RESP_START, RESP_END = \
        create_sft_data(vocab_size, seq_len=block_size, num_pairs=500, device=device)
    print(f"  SFT data: {sft_data.shape[0]} pairs")

    sft_losses = train_sft(model, sft_data, epochs=20, batch_size=16, lr=1e-4)
    results["sft"] = {"final_loss": round(sft_losses[-1], 4)}

    # Save SFT checkpoint as reference model
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"\n  Reference model frozen (SFT checkpoint)")

    # Evaluate SFT baseline
    print("\n  SFT baseline evaluation:")
    correct_sft, total_sft = evaluate_preference(
        model, [(sft_data[i], sft_data[i]) for i in range(min(50, len(sft_data)))],
        [0] * 50, INST_START, INST_END, RESP_START, n_eval=15, device=device
    )
    print(f"    SFT pattern match: {correct_sft}/{total_sft}")

    # ===== Phase 3: DPO Alignment =====
    print("\n" + "=" * 60)
    print("Phase 3: DPO (Direct Preference Optimization)")
    print("=" * 60)

    pref_data, pref_labels = create_preference_data(
        vocab_size, seq_len=block_size, num_pairs=300, device=device
    )
    print(f"  Preference data: {len(pref_data)} pairs (chosen vs rejected)")
    print(f"  β=0.1 (DPO temperature)")

    # Run main DPO training
    dpo_metrics = train_dpo(model, ref_model, pref_data, pref_labels,
                           epochs=20, batch_size=16, lr=5e-5, beta=0.1)

    results["dpo"] = {
        "final_loss": round(dpo_metrics['losses'][-1], 4),
        "final_accuracy": round(dpo_metrics['accuracies'][-1], 4),
        "final_margin": round(dpo_metrics['reward_margins'][-1], 4),
        "chosen_reward": round(dpo_metrics['chosen_rewards'][-1], 4),
        "rejected_reward": round(dpo_metrics['rejected_rewards'][-1], 4),
    }

    # Save DPO model for comparison
    model_dpo = copy.deepcopy(model)

    # ===== Phase 4: Experiments =====
    print("\n" + "=" * 60)
    print("Phase 4: DPO Experiments")
    print("=" * 60)

    # Exp 1: β sweep
    beta_results = experiment_beta_sweep(model, ref_model, pref_data, pref_labels,
                                         vocab_size, device)
    results["beta_sweep"] = {str(k): round(v.get('reward_margins', 0), 4)
                             for k, v in beta_results.items()}

    # Exp 2: DPO vs SFT-only comparison
    experiment_dpo_vs_sft_only(ref_model, model_dpo, pref_data, pref_labels,
                              sft_data, INST_START, INST_END, RESP_START, device)

    # Exp 3: Training dynamics
    model.load_state_dict(ref_model.state_dict())
    dynamics = experiment_dpo_convergence(model, ref_model, pref_data, pref_labels)

    # Exp 4: Length-normalized DPO
    model.load_state_dict(ref_model.state_dict())
    norm_results = experiment_length_normalized_dpo(model, ref_model, pref_data, pref_labels)
    results["length_norm_comparison"] = {
        k: round(v.get('final_loss', 0), 4) for k, v in norm_results.items()
    }

    # ===== Phase 5: Analysis =====
    print("\n" + "=" * 60)
    print("Phase 5: Training Analysis")
    print("=" * 60)

    # Memory analysis
    if device == 'cuda':
        alloc_mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"\n  Peak GPU memory: {alloc_mem:.1f} MB")
        print(f"  Note: DPO needs 2x forward pass (policy + reference)")
        print(f"  Reference model memory: ~{sum(p.numel() * p.element_size() for p in ref_model.parameters()) / 1e6:.1f} MB (frozen)")
        results["gpu_memory_mb"] = round(alloc_mem, 1)

    # DPO vs PPO comparison
    print("\n  DPO vs PPO/RLHF comparison:")
    print(f"    DPO models needed: 2 (policy + reference, no RM/Critic)")
    print(f"    PPO models needed: 4 (actor + critic + reference + reward)")
    print(f"    DPO memory overhead: ~2x (reference model)")
    print(f"    PPO memory overhead: ~4x (3 extra models)")
    print(f"    DPO training: simple supervised loss")
    print(f"    PPO training: rollout + reward + GAE + clip + value update")

    # Compute comparison
    n_tokens_total = (pt_data.numel() * 15 + sft_data.numel() * 20 +
                      len(pref_data) * block_size * 2 * 20)  # DPO: chosen+rejected
    total_flops = 6 * n_params * n_tokens_total  # Forward + backward
    dpo_flops = len(pref_data) * block_size * 2 * 20 * 6 * n_params * 2  # 2x forward per step
    print(f"\n  Compute breakdown:")
    print(f"    Total tokens: {n_tokens_total:,}")
    print(f"    Total FLOPS: {total_flops:.2e}")
    print(f"    DPO FLOPS: {dpo_flops:.2e} ({dpo_flops/total_flops*100:.1f}% of total)")

    with open("dpo_train_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to dpo_train_results.json")


if __name__ == "__main__":
    main()
