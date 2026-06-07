#!/usr/bin/env python3
"""torch.compile + GRPO Training Benchmark — RTX 4090

Key question: does torch.compile speed up GRPO training?

3 experiments:
1. Compile modes on GRPO training step (actor fwd+bwd)
2. Compile actor vs eager actor — full GRPO training loop
3. Compile actor + ref model vs eager — memory + throughput

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u tools/torch_compile_grpo_benchmark.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.mini_grpo_training import (
    MiniGQATransformer, VOCAB_SIZE, TOKENS,
    generate_arithmetic_prompt, generate_sft_dataset,
    compute_reward, decode_tokens,
)


def evaluate_model(model, device, num_samples=100):
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


def measure_time(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return np.median(times), np.mean(times), np.std(times)


def grpo_step(model, ref_model, prompts, n_samples, max_response_len,
              optimizer, device, normalize_sigma=False, grad_noise_sigma=0.0):
    """Single GRPO training step with optional ref model for KL."""
    model.train()
    all_rewards = []
    group_data = []
    zero_adv_groups = 0

    for prompt_tokens, correct_sum in prompts:
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
        responses = []
        for _ in range(n_samples):
            current_ids = prompt_tensor.clone()
            response_tokens = []
            for step in range(max_response_len):
                logits = model(current_ids)
                next_logits = logits[:, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                if next_token == TOKENS['<eos>']:
                    break
                response_tokens.append(next_token)
                current_ids = torch.cat([current_ids,
                    torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
            reward = compute_reward(response_tokens, correct_sum)
            responses.append((response_tokens, reward))
            all_rewards.append(reward)
        group_data.append((prompt_tokens, correct_sum, responses))

    loss = 0
    num_valid = 0
    kl_div_total = 0

    for prompt_tokens, correct_sum, responses in group_data:
        group_rewards = [r for _, r in responses]
        mean_r = np.mean(group_rewards)
        std_r = np.std(group_rewards)
        if std_r < 1e-8:
            std_r = 1.0
            zero_adv_groups += 1

        for resp_tokens, reward in responses:
            if normalize_sigma:
                advantage = (reward - mean_r) / std_r
            else:
                advantage = (reward - mean_r)
            if len(resp_tokens) == 0:
                continue

            full_ids = torch.tensor(prompt_tokens + resp_tokens,
                                    dtype=torch.long, device=device).unsqueeze(0)

            # Actor log probs
            logits = model(full_ids)
            log_probs = F.log_softmax(logits, dim=-1)
            response_start = len(prompt_tokens)
            token_log_probs = []
            for t_idx, token in enumerate(resp_tokens):
                pos = response_start + t_idx - 1
                if pos < 0:
                    continue
                token_log_probs.append(log_probs[0, pos, token])

            # Ref log probs (for KL penalty)
            if ref_model is not None:
                with torch.no_grad():
                    ref_logits = ref_model(full_ids)
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_token_log_probs = []
                    for t_idx, token in enumerate(resp_tokens):
                        pos = response_start + t_idx - 1
                        if pos < 0:
                            continue
                        ref_token_log_probs.append(ref_log_probs[0, pos, token])

                kl = sum(token_log_probs) - sum(ref_token_log_probs)
                kl_div_total += kl.item()

            if len(token_log_probs) > 0:
                total_log_prob = sum(token_log_probs)
                loss += -advantage * total_log_prob
                num_valid += 1

    if num_valid > 0:
        loss = loss / num_valid

    optimizer.zero_grad()
    if num_valid > 0 and abs(loss.item()) > 1e-8:
        loss.backward()
        if grad_noise_sigma > 0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * grad_noise_sigma
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return {
        'loss': loss.item() if num_valid > 0 else 0,
        'reward': np.mean(all_rewards) if all_rewards else 0,
        'zero_adv_pct': zero_adv_groups / len(group_data) * 100,
        'kl_div': kl_div_total / num_valid if num_valid > 0 and ref_model is not None else 0,
    }


def exp1_compile_modes_grpo(device):
    """Compare compile modes on GRPO training step."""
    print("\n" + "="*50)
    print("Exp 1: Compile Modes on GRPO Training Step")
    print("="*50)

    model = MiniGQATransformer(
        hidden_dim=256, num_layers=4, num_heads=8,
        num_kv_heads=4, vocab_size=VOCAB_SIZE).to(device)

    dataset = generate_sft_dataset(500)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    # SFT warm-start (500 steps for strong baseline)
    for step in range(500):
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

    sft_eval = evaluate_model(model, device, 100)
    print(f"  SFT eval: {sft_eval:.1%}")

    # GRPO step function
    prompts_fn = lambda: [generate_arithmetic_prompt() for _ in range(2)]

    # Measure eager baseline
    def grpo_step_fn():
        prompts = prompts_fn()
        return grpo_step(model, None, prompts, 4, 3, optimizer, device)

    for _ in range(10):
        grpo_step_fn()
    torch.cuda.synchronize()
    median_eager, _, _ = measure_time(grpo_step_fn, warmup=3, repeat=20)
    print(f"  Eager GRPO step: {median_eager:.2f}ms")

    results = {'eager_ms': median_eager, 'sft_eval': sft_eval}

    for mode in ['default', 'reduce-overhead']:
        torch._dynamo.reset()
        compiled_model = torch.compile(model, mode=mode)
        compiled_optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=2e-3)

        def compiled_grpo_step():
            prompts = prompts_fn()
            return grpo_step(compiled_model, None, prompts, 4, 3, compiled_optimizer, device)

        # Warmup (compilation happens here)
        for _ in range(10):
            compiled_grpo_step()
        torch.cuda.synchronize()

        median, _, _ = measure_time(compiled_grpo_step, warmup=3, repeat=20)
        speedup = median_eager / median
        results[mode] = {'median_ms': median, 'speedup': speedup}
        print(f"  {mode}: {median:.2f}ms ({speedup:.2f}x)")

    return results


def exp2_compile_grpo_training(device):
    """Full GRPO training loop with compile — convergence comparison."""
    print("\n" + "="*50)
    print("Exp 2: GRPO Training Loop — Compile vs Eager")
    print("="*50)

    dataset = generate_sft_dataset(500)
    results = {}

    for mode in ['eager', 'default', 'reduce-overhead']:
        torch.manual_seed(42)
        np.random.seed(42)
        torch._dynamo.reset()

        model = MiniGQATransformer(
            hidden_dim=256, num_layers=4, num_heads=8,
            num_kv_heads=4, vocab_size=VOCAB_SIZE).to(device)

        if mode == 'eager':
            actor = model
            optimizer = torch.optim.AdamW(actor.parameters(), lr=2e-3)
        else:
            actor = torch.compile(model, mode=mode)
            optimizer = torch.optim.AdamW(actor.parameters(), lr=2e-3)

        # SFT warm-start (200 steps)
        for step in range(200):
            optimizer.zero_grad()
            idx = np.random.randint(len(dataset))
            full_tokens, prompt_len = dataset[idx]
            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
            targets = input_ids.clone()
            logits = actor(input_ids)
            loss = F.cross_entropy(
                logits[:, prompt_len-1:-1, :].reshape(-1, VOCAB_SIZE),
                targets[:, prompt_len:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()

        sft_eval = evaluate_model(actor, device, 100)
        print(f"\n  {mode}: SFT eval = {sft_eval:.1%}")

        # GRPO training (300 steps)
        eval_trajectory = []
        step_times = []

        for step in range(300):
            prompts = [generate_arithmetic_prompt() for _ in range(2)]

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            metrics = grpo_step(actor, None, prompts, 4, 3, optimizer, device,
                                normalize_sigma=False, grad_noise_sigma=0.01)
            end.record()
            torch.cuda.synchronize()
            step_times.append(start.elapsed_time(end))

            if step % 50 == 0 or step == 299:
                eval_acc = evaluate_model(actor, device, 50)
                eval_trajectory.append({'step': step, 'eval': eval_acc, **metrics})

        final_eval = evaluate_model(actor, device, 100)
        peak_eval = max(e['eval'] for e in eval_trajectory)
        median_time = np.median(step_times[20:])  # skip warmup

        results[mode] = {
            'sft_eval': sft_eval,
            'final_eval': final_eval,
            'peak_eval': peak_eval,
            'median_step_ms': median_time,
            'trajectory': eval_trajectory,
        }
        print(f"  {mode}: final={final_eval:.1%}, peak={peak_eval:.1%}, "
              f"step_time={median_time:.2f}ms")

    # Speedup comparison
    eager_ms = results['eager']['median_step_ms']
    for mode in ['default', 'reduce-overhead']:
        if mode in results:
            speedup = eager_ms / results[mode]['median_step_ms']
            results[mode]['speedup'] = speedup
            print(f"  {mode} speedup: {speedup:.2f}x")

    return results


def exp3_compile_actor_ref(device):
    """Compile both actor + ref model — memory and throughput impact."""
    print("\n" + "="*50)
    print("Exp 3: Compile Actor + Ref Model (Memory + Throughput)")
    print("="*50)

    torch.manual_seed(42)
    np.random.seed(42)

    # Create actor + ref (same architecture, ref frozen)
    actor = MiniGQATransformer(
        hidden_dim=256, num_layers=4, num_heads=8,
        num_kv_heads=4, vocab_size=VOCAB_SIZE).to(device)

    ref = MiniGQATransformer(
        hidden_dim=256, num_layers=4, num_heads=8,
        num_kv_heads=4, vocab_size=VOCAB_SIZE).to(device)
    ref.load_state_dict(actor.state_dict())  # same init
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    # Memory baseline
    torch.cuda.reset_peak_memory_stats(device)
    actor_size = sum(p.numel() * p.element_size() for p in actor.parameters())
    ref_size = sum(p.numel() * p.element_size() for p in ref.parameters())
    total_model_mem = actor_size + ref_size
    print(f"  Actor params: {sum(p.numel() for p in actor.parameters())/1e6:.2f}M")
    print(f"  Ref params: {sum(p.numel() for p in ref.parameters())/1e6:.2f}M")
    print(f"  Total model memory: {total_model_mem/1e6:.2f}MB")

    # Eager GRPO step with KL
    optimizer = torch.optim.AdamW(actor.parameters(), lr=2e-3)

    def eager_grpo_kl_step():
        prompts = [generate_arithmetic_prompt() for _ in range(2)]
        return grpo_step(actor, ref, prompts, 4, 3, optimizer, device,
                         normalize_sigma=False)

    for _ in range(10):
        eager_grpo_kl_step()
    torch.cuda.synchronize()

    peak_mem_eager = torch.cuda.max_memory_allocated(device) / 1e6
    median_eager, _, _ = measure_time(eager_grpo_kl_step, warmup=3, repeat=20)

    print(f"  Eager (actor+ref): {median_eager:.2f}ms, peak_mem={peak_mem_eager:.2f}MB")

    # Compile actor + ref
    torch._dynamo.reset()
    torch.cuda.reset_peak_memory_stats(device)

    compiled_actor = torch.compile(actor, mode='default')
    compiled_ref = torch.compile(ref, mode='default')
    compiled_optimizer = torch.optim.AdamW(compiled_actor.parameters(), lr=2e-3)

    def compiled_grpo_kl_step():
        prompts = [generate_arithmetic_prompt() for _ in range(2)]
        return grpo_step(compiled_actor, compiled_ref, prompts, 4, 3,
                         compiled_optimizer, device, normalize_sigma=False)

    # Warmup (compilation)
    for _ in range(10):
        compiled_grpo_kl_step()
    torch.cuda.synchronize()

    peak_mem_compile = torch.cuda.max_memory_allocated(device) / 1e6
    median_compile, _, _ = measure_time(compiled_grpo_kl_step, warmup=3, repeat=20)
    speedup = median_eager / median_compile

    print(f"  Compile (actor+ref): {median_compile:.2f}ms, peak_mem={peak_mem_compile:.2f}MB")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Memory overhead: {(peak_mem_compile - peak_mem_eager)/1e6:.2f}MB (compile cache)")

    return {
        'eager_ms': median_eager, 'compile_ms': median_compile, 'speedup': speedup,
        'eager_peak_mem_mb': peak_mem_eager, 'compile_peak_mem_mb': peak_mem_compile,
        'mem_overhead_mb': peak_mem_compile - peak_mem_eager,
    }


def main():
    device = torch.device('cuda:0')
    print("=" * 70)
    print("torch.compile + GRPO Training Benchmark — RTX 4090")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    all_results = {}
    all_results['exp1_compile_modes_grpo'] = exp1_compile_modes_grpo(device)
    all_results['exp2_compile_grpo_training'] = exp2_compile_grpo_training(device)
    all_results['exp3_compile_actor_ref'] = exp3_compile_actor_ref(device)

    # Summary
    print("\n" + "=" * 70)
    print("Summary: torch.compile + GRPO Training on RTX 4090")
    print("=" * 70)

    e1 = all_results['exp1_compile_modes_grpo']
    print(f"  Exp1 — GRPO step compile modes:")
    for mode in ['default', 'reduce-overhead']:
        if mode in e1:
            print(f"    {mode}: {e1[mode]['speedup']:.2f}x")

    e2 = all_results['exp2_compile_grpo_training']
    print(f"  Exp2 — GRPO training convergence:")
    for mode in ['eager', 'default', 'reduce-overhead']:
        if mode in e2:
            print(f"    {mode}: eval={e2[mode]['final_eval']:.1%}, "
                  f"step={e2[mode]['median_step_ms']:.2f}ms")

    e3 = all_results['exp3_compile_actor_ref']
    print(f"  Exp3 — Actor+Ref compile:")
    print(f"    Speedup: {e3['speedup']:.2f}x")
    print(f"    Memory overhead: {e3['mem_overhead_mb']/1e6:.2f}MB")

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', 'torch_compile_grpo_benchmark.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()