#!/usr/bin/env python3
"""Double Normalization Verification — RTX 4090

Test the hypothesis: σ-normalization + clip_grad_norm = double normalization → hurts performance.
If correct, removing clip_grad_norm should make σ-normalization effective.

3 variants:
1. σ-norm + clip (baseline from previous experiment → 43%)
2. unnormalized + clip (baseline → 65%)
3. σ-norm NO clip (hypothesis → should be ≥65%)

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u tools/double_norm_verification.py
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
    MiniGQATransformer, VOCAB_SIZE, TOKENS,
    generate_arithmetic_prompt, generate_sft_dataset,
    IDX_TO_TOKEN, compute_reward, decode_tokens,
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


def train_sft(model, dataset, optimizer, device, num_steps, lr=2e-3):
    losses = []
    eval_trajectory = []
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
    return losses, eval_trajectory


def grpo_step(model, prompts, n_samples, max_response_len, optimizer,
              device, normalize_sigma=True, clip_grad=True):
    model.train()
    all_rewards = []
    group_data = []

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

    for prompt_tokens, correct_sum, responses in group_data:
        group_rewards = [r for _, r in responses]
        mean_r = np.mean(group_rewards)
        std_r = np.std(group_rewards)
        if std_r < 1e-8:
            std_r = 1.0

        for resp_tokens, reward in responses:
            if normalize_sigma:
                advantage = (reward - mean_r) / std_r
            else:
                advantage = (reward - mean_r)

            if len(resp_tokens) == 0:
                continue

            full_ids = torch.tensor(prompt_tokens + resp_tokens,
                                    dtype=torch.long, device=device).unsqueeze(0)
            logits = model(full_ids)
            log_probs = F.log_softmax(logits, dim=-1)
            response_start = len(prompt_tokens)
            token_log_probs = []
            for t_idx, token in enumerate(resp_tokens):
                pos = response_start + t_idx - 1
                if pos < 0:
                    continue
                token_log_probs.append(log_probs[0, pos, token])

            if len(token_log_probs) > 0:
                total_log_prob = sum(token_log_probs)
                loss += -advantage * total_log_prob
                num_valid += 1

    if num_valid > 0:
        loss = loss / num_valid

    optimizer.zero_grad()
    if num_valid > 0 and abs(loss.item()) > 1e-8:
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    avg_reward = np.mean(all_rewards) if all_rewards else 0
    return {'loss': loss.item() if num_valid > 0 else 0, 'reward': avg_reward, 'num_valid': num_valid}


def train_grpo(model, optimizer, device, num_steps, n_samples=8,
               max_response_len=3, normalize_sigma=True, clip_grad=True):
    eval_trajectory = []
    step_metrics = []
    for step in range(num_steps):
        prompts = [generate_arithmetic_prompt() for _ in range(2)]
        metrics = grpo_step(model, prompts, n_samples, max_response_len,
                            optimizer, device, normalize_sigma, clip_grad)
        step_metrics.append(metrics)
        if step % 25 == 0 or step == num_steps - 1:
            eval_acc = evaluate_model(model, device, 50)
            eval_trajectory.append({'step': step, 'eval': eval_acc, **metrics})
    return step_metrics, eval_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', default='2.28m', choices=['76k', '2.28m'])
    parser.add_argument('--sft_steps', type=int, default=200)
    parser.add_argument('--grpo_steps', type=int, default=300)
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-3)
    args = parser.parse_args()

    device = torch.device('cuda:0')
    print("=" * 70)
    print("Double Normalization Verification — RTX 4090")
    print("=" * 70)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_size}, SFT: {args.sft_steps}, GRPO: {args.grpo_steps}, n={args.n_samples}")

    hd = 64 if args.model_size == '76k' else 256
    nl = 2 if args.model_size == '76k' else 4
    nh = 4 if args.model_size == '76k' else 8
    nkv = 2 if args.model_size == '76k' else 4
    dataset = generate_sft_dataset(500)

    strategies = [
        ('norm_clip',     True,  True,  'A=(r-μ)/σ + clip'),
        ('unnorm_clip',   False, True,  'A=r-μ + clip'),
        ('norm_noclip',   True,  False, 'A=(r-μ)/σ NO clip'),
        ('unnorm_noclip', False, False, 'A=r-μ NO clip'),
    ]

    results = {}

    for key, normalize, clip, label in strategies:
        print(f"\n{'='*50}")
        print(f"Strategy: {label}")
        print(f"{'='*50}")

        torch.manual_seed(42)
        np.random.seed(42)
        torch.cuda.reset_peak_memory_stats()

        model = MiniGQATransformer(
            hidden_dim=hd, num_layers=nl, num_heads=nh,
            num_kv_heads=nkv, vocab_size=VOCAB_SIZE).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        print(f"[Phase 1] SFT warm-start ({args.sft_steps} steps)")
        sft_losses, sft_eval = train_sft(model, dataset, optimizer, device, args.sft_steps, args.lr)
        sft_final_eval = evaluate_model(model, device, 100)
        print(f"  SFT eval: {sft_final_eval:.1%}, final loss: {sft_losses[-1]:.3f}")

        print(f"[Phase 2] GRPO ({args.grpo_steps} steps, n={args.n_samples}, "
              f"normalize={normalize}, clip={clip})")
        grpo_metrics, grpo_eval = train_grpo(
            model, optimizer, device, args.grpo_steps, args.n_samples,
            normalize_sigma=normalize, clip_grad=clip)

        final_eval = evaluate_model(model, device, 100)
        peak_eval = max(e['eval'] for e in grpo_eval)
        avg_reward = np.mean([m['reward'] for m in grpo_metrics])
        zero_loss_count = sum(1 for m in grpo_metrics if m['loss'] == 0.0)
        zero_loss_pct = zero_loss_count / len(grpo_metrics) * 100

        result = {
            'strategy': key,
            'normalize_sigma': normalize,
            'clip_grad': clip,
            'model_size': args.model_size,
            'sft_steps': args.sft_steps,
            'grpo_steps': args.grpo_steps,
            'n_samples': args.n_samples,
            'sft_final_eval': sft_final_eval,
            'grpo_final_eval': final_eval,
            'grpo_peak_eval': peak_eval,
            'avg_reward': avg_reward,
            'zero_loss_pct': zero_loss_pct,
            'grpo_eval_trajectory': grpo_eval,
            'peak_gpu_mem_gb': torch.cuda.max_memory_allocated() / 1e9,
        }
        results[key] = result

        print(f"  GRPO final eval: {final_eval:.1%}, peak: {peak_eval:.1%}, "
              f"reward: {avg_reward:.3f}, loss=0: {zero_loss_pct:.0f}%")

    print("\n" + "=" * 70)
    print("Summary: Double Normalization Verification")
    print("=" * 70)

    for key, normalize, clip, label in strategies:
        r = results[key]
        print(f"  {label}: SFT→{r['sft_final_eval']:.1%}, "
              f"GRPO→{r['grpo_final_eval']:.1%}, peak={r['grpo_peak_eval']:.1%}, "
              f"loss=0: {r['zero_loss_pct']:.0f}%")

    norm_clip = results['norm_clip']['grpo_final_eval']
    unnorm_clip = results['unnorm_clip']['grpo_final_eval']
    norm_noclip = results['norm_noclip']['grpo_final_eval']
    unnorm_noclip = results['unnorm_noclip']['grpo_final_eval']

    print(f"\n  Hypothesis test:")
    print(f"    σ-norm+clip vs unnorm+clip: {norm_clip:.1%} vs {unnorm_clip:.1%} "
          f"(Δ={norm_clip-unnorm_clip:+.1%})")
    print(f"    σ-norm NO clip vs unnorm+clip: {norm_noclip:.1%} vs {unnorm_clip:.1%} "
          f"(Δ={norm_noclip-unnorm_clip:+.1%})")
    print(f"    σ-norm NO clip vs σ-norm+clip: {norm_noclip:.1%} vs {norm_clip:.1%} "
          f"(Δ={norm_noclip-norm_clip:+.1%})")

    if norm_noclip >= unnorm_clip:
        print(f"\n  ✓ Hypothesis CONFIRMED: Removing clip makes σ-norm effective!")
    else:
        print(f"\n  ✗ Hypothesis REJECTED: σ-norm still worse even without clip")

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', 'double_norm_verification.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()