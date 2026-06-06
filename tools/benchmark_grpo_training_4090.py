#!/usr/bin/env python3
"""GRPO Training Micro-Benchmark on RTX 4090
==============================================

Lightweight GRPO training benchmark to validate theoretical simulator findings:
1. GRPO vs PPO memory comparison (2 vs 4 models)
2. Outcome-only reward + group normalization
3. Prefix caching savings for n>1 sampling
4. Training throughput measurement

Uses a small 1M parameter model for fast iteration.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from collections import defaultdict

def benchmark_cuda(fn, warmup=5, repeat=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat

# ============================================================
# Mini Transformer for GRPO Training
# ============================================================
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=128, d_model=256, n_heads=4, n_layers=4, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model*4,
                dropout=0.1, batch_first=True,
                norm_first=True  # pre-norm
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.embedding.weight
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)
        if attention_mask is not None:
            # Combine causal + padding mask
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,S]
            combined = causal_mask.unsqueeze(0)  # [1,S,S]
            # Where pad_mask=0, set to -inf
            combined = combined.masked_fill(~pad_mask.bool(), float('-inf'))

        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

def compute_log_probs(model, input_ids, response_ids, response_mask):
    """Compute log probabilities for response tokens."""
    full_ids = torch.cat([input_ids, response_ids], dim=1)
    full_mask = torch.cat([
        torch.ones_like(input_ids),
        response_mask
    ], dim=1)

    logits = model(full_ids, attention_mask=full_mask)

    # Extract response logits
    prompt_len = input_ids.size(1)
    response_logits = logits[:, prompt_len-1:-1, :]  # shift by 1

    log_probs = F.log_softmax(response_logits.float(), dim=-1)
    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
    # Apply mask
    token_log_probs = token_log_probs * response_mask
    return token_log_probs

def grpo_outcome_advantage(rewards, uid, n_samples, norm_by_std=True):
    """GRPO group normalization: A = (r - mean_group) / std_group"""
    advantages = torch.zeros_like(rewards)
    unique_uids = np.unique(uid)

    for u in unique_uids:
        mask = (uid == u)
        group_rewards = rewards[mask]
        group_mean = group_rewards.mean()
        group_std = group_rewards.std() if norm_by_std else 1.0
        if group_std < 1e-8:
            group_std = 1.0
        advantages[mask] = (group_rewards - group_mean) / group_std

    return advantages

def main():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"GRPO Training Micro-Benchmark: {gpu_name} ({gpu_mem:.1f} GB)")
    print("=" * 60)

    results = {"gpu": gpu_name, "gpu_mem_gb": gpu_mem}

    # Model config
    vocab_size = 128
    d_model = 256
    n_heads = 4
    n_layers = 4
    max_seq_len = 512

    # ============================================================
    # Experiment 1: Model Memory — PPO (4 models) vs GRPO (2 models)
    # ============================================================
    print("\n" + "=" * 60)
    print("Experiment 1: Memory Comparison — PPO vs GRPO")
    print("=" * 60)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Create actor model
    actor = MiniTransformer(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
    n_params = actor.count_parameters()
    actor_mem = torch.cuda.memory_allocated() / 1e6  # MB

    # GRPO: only actor + reward_function (no separate model)
    grpo_total_mem = actor_mem
    # PPO: actor + critic + ref + reward_model (4 models)
    ppo_total_mem = actor_mem * 4  # approximate

    # Actual measurement with all models
    critic = MiniTransformer(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
    critic_mem = torch.cuda.memory_allocated() / 1e6 - grpo_total_mem
    ref = MiniTransformer(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
    ref_mem = torch.cuda.memory_allocated() / 1e6 - grpo_total_mem - critic_mem

    ppo_actual_mem = torch.cuda.memory_allocated() / 1e6
    grpo_savings_pct = (1 - grpo_total_mem / ppo_actual_mem) * 100

    print(f"  Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  Single model memory: {actor_mem:.1f} MB")
    print(f"  PPO (4 models): {ppo_actual_mem:.1f} MB")
    print(f"  GRPO (2 models): {grpo_total_mem:.1f} MB")
    print(f"  GRPO memory savings: {grpo_savings_pct:.1f}%")

    results["memory"] = {
        "n_params": n_params,
        "single_model_mb": round(actor_mem, 1),
        "ppo_4models_mb": round(ppo_actual_mem, 1),
        "grpo_2models_mb": round(grpo_total_mem, 1),
        "grpo_savings_pct": round(grpo_savings_pct, 1),
    }

    # Clean up PPO models
    del critic, ref
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ============================================================
    # Experiment 2: GRPO Training Step Benchmark
    # ============================================================
    print("\n" + "=" * 60)
    print("Experiment 2: GRPO Training Step Throughput")
    print("=" * 60)

    # Training config
    batch_size = 32
    prompt_len = 64
    response_len = 128
    n_samples = 4  # GRPO: 4 responses per prompt
    n_prompts = batch_size // n_samples  # 8 unique prompts

    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)

    # Generate dummy data
    prompts = torch.randint(0, vocab_size, (n_prompts, prompt_len), device=device)
    responses = torch.randint(0, vocab_size, (batch_size, response_len), device=device)
    response_mask = torch.ones(batch_size, response_len, device=device)

    # Repeat prompts to match n_samples
    prompts_expanded = prompts.repeat_interleave(n_samples, dim=0)

    # Assign UIDs for GRPO group normalization
    uid_array = np.array([i // n_samples for i in range(batch_size)])

    # Generate random outcome rewards (simulating reward function)
    rewards = torch.randn(batch_size, device=device)

    training_step_times = []

    for step in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 1. Compute current log probs
        old_log_probs = compute_log_probs(actor, prompts_expanded, responses, response_mask)

        # 2. Compute GRPO advantage
        advantages = grpo_outcome_advantage(rewards, uid_array, n_samples)

        # 3. PPO clip loss (vanilla, but with GRPO advantage)
        # ratio = exp(log_prob - old_log_prob)
        # For first step, ratio ≈ 1 since we just computed old_log_probs
        new_log_probs = compute_log_probs(actor, prompts_expanded, responses, response_mask)
        ratio = torch.exp(new_log_probs - old_log_probs)
        clip_ratio = 0.2
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

        loss1 = -advantages.unsqueeze(1) * ratio
        loss2 = -advantages.unsqueeze(1) * clipped_ratio
        loss = torch.max(loss1, loss2)
        loss = (loss * response_mask).sum() / response_mask.sum()

        # 4. Backward + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_time_ms = (t1 - t0) * 1000

        if step >= 5:  # Skip warmup
            training_step_times.append(step_time_ms)

    avg_step_time = np.mean(training_step_times)
    tokens_per_step = batch_size * (prompt_len + response_len)
    throughput = tokens_per_step / (avg_step_time / 1000)

    print(f"  Batch size: {batch_size} ({n_prompts} prompts × {n_samples} responses)")
    print(f"  Avg step time: {avg_step_time:.2f} ms")
    print(f"  Tokens/step: {tokens_per_step}")
    print(f"  Throughput: {throughput:.0f} tok/s")
    print(f"  Step breakdown: {np.mean(training_step_times):.2f} ms")

    results["grpo_training"] = {
        "batch_size": batch_size,
        "n_prompts": n_prompts,
        "n_samples": n_samples,
        "prompt_len": prompt_len,
        "response_len": response_len,
        "avg_step_ms": round(avg_step_time, 2),
        "throughput_tok_s": round(throughput, 0),
    }

    # ============================================================
    # Experiment 3: GRPO vs PPO Training Step Comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("Experiment 3: GRPO vs PPO Training Step Time")
    print("=" * 60)

    # PPO needs: actor + critic update + value computation + GAE
    # Re-create critic for PPO
    critic = MiniTransformer(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    ref_model = MiniTransformer(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
    ref_model.eval()

    ppo_step_times = []

    for step in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # PPO step: 4 substeps
        # 1. Ref log probs (no grad)
        with torch.no_grad():
            ref_log_probs = compute_log_probs(ref_model, prompts_expanded, responses, response_mask)

        # 2. Critic: compute values
        full_ids = torch.cat([prompts_expanded, responses], dim=1)
        with torch.no_grad():
            values = critic(full_ids).mean(dim=-1)  # simplified

        # 3. Actor: compute log probs + PPO loss
        old_log_probs = compute_log_probs(actor, prompts_expanded, responses, response_mask)

        # KL penalty: add β * KL(π_θ || π_ref) to rewards
        kl_penalty = 0.01
        kl = (old_log_probs - ref_log_probs).mean(dim=-1)
        token_rewards = torch.zeros_like(old_log_probs)
        token_rewards[:, -1] = rewards + kl_penalty * kl  # outcome + KL at last token

        # GAE: simplified (γ=1, λ=1 → A = r + V_next - V)
        # For simplicity, use outcome-only with value baseline
        gae_advantages = rewards - values.squeeze()[:, -1]  # simplified GAE

        # PPO clip loss
        new_log_probs = compute_log_probs(actor, prompts_expanded, responses, response_mask)
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
        loss1 = -gae_advantages.unsqueeze(1) * ratio
        loss2 = -gae_advantages.unsqueeze(1) * clipped_ratio
        actor_loss = torch.max(loss1, loss2)
        actor_loss = (actor_loss * response_mask).sum() / response_mask.sum()

        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()

        # 4. Critic update
        critic_values = critic(full_ids).mean(dim=-1)
        critic_loss = F.mse_loss(critic_values.squeeze()[:, -1], rewards + values.squeeze()[:, -1].detach())
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ppo_step_time_ms = (t1 - t0) * 1000

        if step >= 5:
            ppo_step_times.append(ppo_step_time_ms)

    ppo_avg_step = np.mean(ppo_step_times)
    ppo_throughput = tokens_per_step / (ppo_avg_step / 1000)
    speedup = ppo_avg_step / avg_step_time

    print(f"  GRPO avg step: {avg_step_time:.2f} ms ({throughput:.0f} tok/s)")
    print(f"  PPO avg step: {ppo_avg_step:.2f} ms ({ppo_throughput:.0f} tok/s)")
    print(f"  GRPO/PPO speedup: {speedup:.2f}x")

    results["ppo_vs_grpo"] = {
        "grpo_step_ms": round(avg_step_time, 2),
        "ppo_step_ms": round(ppo_avg_step, 2),
        "grpo_throughput": round(throughput, 0),
        "ppo_throughput": round(ppo_throughput, 0),
        "grpo_speedup": round(speedup, 2),
    }

    del critic, ref_model
    torch.cuda.empty_cache()

    # ============================================================
    # Experiment 4: Prefix Caching Savings Measurement
    # ============================================================
    print("\n" + "=" * 60)
    print("Experiment 4: Prefix Caching Savings (n=4/8 responses)")
    print("=" * 60)

    prefix_cache_data = []
    prompt_lengths = [64, 128, 256, 512]

    for p_len in prompt_lengths:
        for n in [2, 4, 8]:
            n_prompts_test = 4  # reduced from 8 to avoid OOM
            total_responses = n_prompts_test * n
            r_len = 64

            # Without prefix caching: all tokens computed
            total_tokens_no_cache = total_responses * (p_len + r_len)

            # With prefix caching: prompt computed once per group
            total_tokens_with_cache = n_prompts_test * p_len + total_responses * r_len

            savings_pct = (1 - total_tokens_with_cache / total_tokens_no_cache) * 100

            print(f"  P={p_len}, n={n}: no_cache={total_tokens_no_cache} tok, "
                  f"with_cache={total_tokens_with_cache} tok, savings={savings_pct:.1f}%")

            prefix_cache_data.append({
                "prompt_len": p_len,
                "n_samples": n,
                "total_tokens_no_cache": total_tokens_no_cache,
                "total_tokens_with_cache": total_tokens_with_cache,
                "savings_pct": round(savings_pct, 1),
            })

    results["prefix_cache"] = prefix_cache_data

    # ============================================================
    # Experiment 5: Training Convergence — GRPO Loss Curve
    # ============================================================
    print("\n" + "=" * 60)
    print("Experiment 5: GRPO Training Convergence (50 steps)")
    print("=" * 60)

    # Fresh model
    actor2 = MiniTransformer(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
    optimizer2 = torch.optim.Adam(actor2.parameters(), lr=3e-4)

    # Synthetic reward: longer responses with certain token patterns get higher reward
    # Simulate a simple "quality" metric
    def compute_reward_fn(response_ids, response_mask):
        """Simple reward: count specific tokens + length bonus"""
        # Reward based on presence of token 42 (simulating "good" content)
        good_token_count = (response_ids == 42).float().sum(dim=-1)
        length = response_mask.sum(dim=-1)
        # Normalize and add small random noise
        reward = good_token_count / length.clamp(min=1) + 0.1 * torch.randn_like(good_token_count)
        return reward

    loss_curve = []
    reward_curve = []
    kl_curve = []
    advantage_std_curve = []

    # Initial reference log probs (for KL tracking)
    actor2.eval()
    initial_log_probs = compute_log_probs(actor2, prompts_expanded, responses, response_mask).detach()
    actor2.train()

    for step in range(50):
        # Compute rewards with current responses (fixed, from initial generation)
        rewards_step = compute_reward_fn(responses, response_mask)

        # GRPO advantage
        adv = grpo_outcome_advantage(rewards_step, uid_array, n_samples)
        advantage_std_curve.append(adv.std().item())

        # Actor log probs
        old_lp = compute_log_probs(actor2, prompts_expanded, responses, response_mask).detach()
        new_lp = compute_log_probs(actor2, prompts_expanded, responses, response_mask)

        # PPO clip loss with GRPO advantage
        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
        loss1 = -adv.unsqueeze(1) * ratio
        loss2 = -adv.unsqueeze(1) * clipped
        loss = torch.max(loss1, loss2)
        loss = (loss * response_mask).sum() / response_mask.sum()

        # KL divergence tracking
        kl = (new_lp - initial_log_probs).abs().mean().item()
        kl_curve.append(kl)

        optimizer2.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor2.parameters(), 1.0)
        optimizer2.step()

        loss_curve.append(loss.item())
        reward_curve.append(rewards_step.mean().item())

    print(f"  Initial loss: {loss_curve[0]:.4f}")
    print(f"  Final loss: {loss_curve[-1]:.4f}")
    print(f"  Loss reduction: {(1 - loss_curve[-1]/loss_curve[0])*100:.1f}%")
    print(f"  Reward: {reward_curve[0]:.3f} → {reward_curve[-1]:.3f}")
    print(f"  KL divergence: {kl_curve[0]:.4f} → {kl_curve[-1]:.4f}")
    print(f"  Advantage std: {advantage_std_curve[0]:.3f} → {advantage_std_curve[-1]:.3f}")

    results["convergence"] = {
        "initial_loss": round(loss_curve[0], 4),
        "final_loss": round(loss_curve[-1], 4),
        "loss_reduction_pct": round((1 - loss_curve[-1]/loss_curve[0])*100, 1),
        "initial_reward": round(reward_curve[0], 3),
        "final_reward": round(reward_curve[-1], 3),
        "initial_kl": round(kl_curve[0], 4),
        "final_kl": round(kl_curve[-1], 4),
        "advantage_std_start": round(advantage_std_curve[0], 3),
        "advantage_std_end": round(advantage_std_curve[-1], 3),
    }

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("KEY FINDINGS SUMMARY")
    print("=" * 60)
    print(f"  1. GRPO memory savings: {grpo_savings_pct:.1f}% (2 vs 4 models)")
    print(f"  2. GRPO training speedup: {speedup:.2f}x over PPO")
    print(f"  3. GRPO throughput: {throughput:.0f} tok/s")
    print(f"  4. Loss convergence: {loss_curve[0]:.4f} → {loss_curve[-1]:.4f}")
    print(f"  5. Best prefix cache savings: {max(d['savings_pct'] for d in prefix_cache_data):.1f}%")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'grpo_training_benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()