#!/usr/bin/env python3
"""SFT Warm-start Degree Impact on GRPO — RTX 4090

Test hypothesis: σ-normalization effectiveness depends on SFT warm-start quality.
- Strong SFT → reward concentrated → σ-norm hurts
- Weak SFT → reward dispersed → σ-norm helps

5 SFT levels × 2 advantage types = 10 conditions:
  SFT: 0, 50, 100, 200, 500 steps → GRPO: 300 steps
  Each: σ-norm and unnorm

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u tools/sft_degree_impact.py
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
    return evaluate_model(model, device, 100), losses[-1]


def grpo_step(model, prompts, n_samples, max_response_len, optimizer,
              device, normalize_sigma=True):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return {
        'loss': loss.item() if num_valid > 0 else 0,
        'reward': np.mean(all_rewards) if all_rewards else 0,
        'zero_adv_pct': zero_adv_groups / len(group_data) * 100,
    }


def train_grpo(model, optimizer, device, num_steps, n_samples=8,
               max_response_len=3, normalize_sigma=True):
    eval_trajectory = []
    for step in range(num_steps):
        prompts = [generate_arithmetic_prompt() for _ in range(2)]
        metrics = grpo_step(model, prompts, n_samples, max_response_len,
                            optimizer, device, normalize_sigma)
        if step % 50 == 0 or step == num_steps - 1:
            eval_acc = evaluate_model(model, device, 50)
            eval_trajectory.append({'step': step, 'eval': eval_acc, **metrics})
    final = evaluate_model(model, device, 100)
    peak = max(e['eval'] for e in eval_trajectory)
    avg_reward = np.mean([e['reward'] for e in eval_trajectory])
    avg_zero_adv = np.mean([e['zero_adv_pct'] for e in eval_trajectory])
    return final, peak, avg_reward, avg_zero_adv, eval_trajectory


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--model_size', default='2.28m', choices=['76k', '2.28m'])
    args.add_argument('--grpo_steps', type=int, default=300)
    args.add_argument('--n_samples', type=int, default=8)
    args.add_argument('--lr', type=float, default=2e-3)
    args = args.parse_args()

    device = torch.device('cuda:0')
    print("=" * 70)
    print("SFT Warm-start Degree Impact on GRPO — RTX 4090")
    print("=" * 70)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    hd = 64 if args.model_size == '76k' else 256
    nl = 2 if args.model_size == '76k' else 4
    nh = 4 if args.model_size == '76k' else 8
    nkv = 2 if args.model_size == '76k' else 4
    dataset = generate_sft_dataset(500)

    sft_steps_list = [0, 50, 100, 200, 500]
    results = {}

    for sft_steps in sft_steps_list:
        for normalize in [True, False]:
            key = f"sft{sft_steps}_{('norm' if normalize else 'unnorm')}"
            label = f"SFT={sft_steps} + {('A=(r-μ)/σ' if normalize else 'A=r-μ')}"

            print(f"\n{'='*40}")
            print(f"{label}")
            print(f"{'='*40}")

            torch.manual_seed(42)
            np.random.seed(42)

            model = MiniGQATransformer(
                hidden_dim=hd, num_layers=nl, num_heads=nh,
                num_kv_heads=nkv, vocab_size=VOCAB_SIZE).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

            # SFT phase
            if sft_steps > 0:
                sft_eval, sft_loss = train_sft(model, dataset, optimizer, device, sft_steps, args.lr)
            else:
                sft_eval = evaluate_model(model, device, 100)
                sft_loss = 0

            print(f"  SFT eval: {sft_eval:.1%}")

            # GRPO phase
            final, peak, avg_reward, avg_zero_adv, traj = train_grpo(
                model, optimizer, device, args.grpo_steps, args.n_samples, 3, normalize)

            delta = final - sft_eval
            results[key] = {
                'sft_steps': sft_steps,
                'normalize_sigma': normalize,
                'sft_eval': sft_eval,
                'grpo_final_eval': final,
                'grpo_peak_eval': peak,
                'delta_grpo': delta,
                'avg_reward': avg_reward,
                'avg_zero_adv_pct': avg_zero_adv,
            }

            print(f"  GRPO: {final:.1%} (peak={peak:.1%}), Δ={delta:+.1%}, "
                  f"reward={avg_reward:.3f}, zero_adv={avg_zero_adv:.0f}%")

    # Summary table
    print("\n" + "=" * 70)
    print("Summary: SFT Quality × σ-norm Effectiveness")
    print("=" * 70)

    print(f"\n  {'SFT steps':<10} {'SFT eval':<10} {'σ-norm':<10} {'unnorm':<10} "
          f"{'Δ(σ-unnorm)':<12} {'zero_adv%':<10}")
    print(f"  {'-'*60}")

    for sft_steps in sft_steps_list:
        norm_key = f"sft{sft_steps}_norm"
        unnorm_key = f"sft{sft_steps}_unnorm"
        n = results[norm_key]
        u = results[unnorm_key]
        delta = n['grpo_final_eval'] - u['grpo_final_eval']
        print(f"  {sft_steps:<10} {n['sft_eval']:<10.1%} "
              f"{n['grpo_final_eval']:<10.1%} {u['grpo_final_eval']:<10.1%} "
              f"{delta:<+12.1%} {n['avg_zero_adv_pct']:<10.0f}")

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', 'sft_degree_impact.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()