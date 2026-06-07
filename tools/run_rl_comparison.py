#!/usr/bin/env python3
"""Unified RL Training Comparison — All 7 modes head-to-head

Run GRPO, RLOO, DAPO, PPO, SFT→GRPO, DPO, and both(GRPO+PPO)
with identical seeds and configurations for fair comparison.

Can run on GPU (RTX 4090) or CPU.
"""

import torch
import numpy as np
import json
import argparse
import time
from tools.mini_grpo_training import (
    MiniGQATransformer, VOCAB_SIZE, TOKENS,
    grpo_training_step, rloo_training_step,
    dapo_training_step, ppo_training_step,
    SimpleCritic, generate_arithmetic_prompt,
    compute_reward, generate_sft_dataset,
    generate_dpo_preference_pairs,
)

def run_single_mode(mode, num_steps, n_samples, num_prompts_per_step,
                    max_response_len, lr, device, hidden_dim=64, num_layers=2,
                    kl_coeff=0.01, clip_lower=0.3, clip_upper=0.2):
    """Run a single training mode for num_steps and return metrics history."""
    torch.manual_seed(42)  # Reset seed for each mode for fair comparison
    np.random.seed(42)

    actor = MiniGQATransformer(hidden_dim=hidden_dim, num_layers=num_layers,
                                num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=lr)
    param_count = sum(p.numel() for p in actor.parameters())

    metrics_history = []
    start_time = time.time()

    if mode == 'sft_grpo':
        # Phase 1: SFT warmup (100 steps)
        sft_dataset = generate_sft_dataset(500)
        sft_optimizer = torch.optim.AdamW(actor.parameters(), lr=lr * 2)
        sft_metrics = []
        for step in range(100):
            batch_idx = np.random.randint(0, len(sft_dataset), size=4)
            batch = [sft_dataset[i] for i in batch_idx]
            full_ids = torch.tensor([b['full_ids'] for b in batch], dtype=torch.long, device=device)
            targets = torch.tensor([b['target_ids'] for b in batch], dtype=torch.long, device=device)
            logits = actor(full_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            sft_optimizer.zero_grad()
            loss.backward()
            sft_optimizer.step()
            if step % 20 == 0:
                print(f"    SFT step {step}: loss={loss.item():.4f}")
        # Phase 2: GRPO RL
        mode_for_rl = 'grpo'
    else:
        mode_for_rl = mode

    # Main RL training loop
    critic = None
    critic_optimizer = None
    dapo_ref_model = None

    if mode_for_rl == 'ppo':
        critic = SimpleCritic(hidden_dim=32, vocab_size=VOCAB_SIZE).to(device)
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=lr)

    if mode_for_rl == 'dapo':
        dapo_ref_model = MiniGQATransformer(hidden_dim=hidden_dim, num_layers=num_layers,
                                              num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
        dapo_ref_model.load_state_dict(actor.state_dict().copy())
        dapo_ref_model.eval()

    for step in range(num_steps):
        prompts = [generate_arithmetic_prompt() for _ in range(num_prompts_per_step)]

        if mode_for_rl == 'grpo':
            metrics = grpo_training_step(actor, prompts, n_samples, max_response_len,
                                          actor_optimizer, device)
        elif mode_for_rl == 'rloo':
            metrics = rloo_training_step(actor, prompts, n_samples, max_response_len,
                                          actor_optimizer, device)
        elif mode_for_rl == 'ppo':
            metrics = ppo_training_step(actor, critic, prompts, n_samples, max_response_len,
                                        actor_optimizer, critic_optimizer, device)
        elif mode_for_rl == 'dapo':
            metrics = dapo_training_step(actor, dapo_ref_model, prompts, n_samples,
                                          max_response_len, actor_optimizer, device,
                                          clip_epsilon_lower=clip_lower,
                                          clip_epsilon_upper=clip_upper,
                                          kl_coeff=kl_coeff)
        elif mode_for_rl == 'dpo':
            # DPO mode: train on preference pairs
            pairs = generate_dpo_preference_pairs(8, actor, device)
            loss = 0
            num_pairs = 0
            for pair in pairs:
                prompt_tensor = torch.tensor(pair['prompt_ids'], dtype=torch.long, device=device).unsqueeze(0)
                chosen_tensor = torch.tensor(pair['chosen_ids'], dtype=torch.long, device=device).unsqueeze(0)
                rejected_tensor = torch.tensor(pair['rejected_ids'], dtype=torch.long, device=device).unsqueeze(0)
                ref_chosen_logp = pair['ref_chosen_logp']
                ref_rejected_logp = pair['ref_rejected_logp']

                chosen_logits = actor(chosen_tensor)
                chosen_logps = torch.nn.functional.log_softmax(chosen_logits, dim=-1)
                chosen_log_prob = sum(chosen_logps[0, pos, pair['chosen_ids'][pos+1]]
                                      for pos in range(len(pair['chosen_ids'])-1)
                                      if pos+1 < len(pair['chosen_ids']) and pair['chosen_ids'][pos+1] < VOCAB_SIZE)

                rejected_logits = actor(rejected_tensor)
                rejected_logps = torch.nn.functional.log_softmax(rejected_logits, dim=-1)
                rejected_log_prob = sum(rejected_logps[0, pos, pair['rejected_ids'][pos+1]]
                                       for pos in range(len(pair['rejected_ids'])-1)
                                       if pos+1 < len(pair['rejected_ids']) and pair['rejected_ids'][pos+1] < VOCAB_SIZE)

                beta = 0.3
                chosen_ratio = chosen_log_prob - ref_chosen_logp
                rejected_ratio = rejected_log_prob - ref_rejected_logp
                dpo_loss = -torch.nn.functional.logsigmoid(beta * (chosen_ratio - rejected_ratio))
                loss += dpo_loss
                num_pairs += 1

            if num_pairs > 0:
                loss = loss / num_pairs
                actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                actor_optimizer.step()

            # Quick eval for DPO
            correct = 0
            total = 0
            for prompt_tokens, correct_sum in prompts:
                prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
                current_ids = prompt_tensor.clone()
                resp = []
                for _ in range(max_response_len):
                    logits = actor(current_ids)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                    if next_token == TOKENS['<eos>']:
                        break
                    resp.append(next_token)
                    current_ids = torch.cat([current_ids, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
                reward = compute_reward(resp, correct_sum)
                if reward == 1.0:
                    correct += 1
                total += 1

            metrics = {
                'loss': loss.item() if num_pairs > 0 else 0,
                'reward_mean': correct / total if total > 0 else 0,
                'reward_std': 0,
                'accuracy': correct / total if total > 0 else 0,
                'advantage_mean': 0,
                'advantage_std': 0,
                'num_valid': num_pairs,
                'num_groups': 0,
            }

        metrics_history.append(metrics)
        if step % 50 == 0 or step == num_steps - 1:
            print(f"  [{mode}] Step {step}: acc={metrics['accuracy']:.1%}, "
                  f"reward={metrics['reward_mean']:.3f}, loss={metrics['loss']:.4f}")

    elapsed = time.time() - start_time

    # Final eval
    actor.eval()
    eval_prompts = [generate_arithmetic_prompt() for _ in range(100)]
    correct = 0
    for prompt_tokens, correct_sum in eval_prompts:
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
        current_ids = prompt_tensor.clone()
        resp = []
        for _ in range(max_response_len):
            logits = actor(current_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            if next_token == TOKENS['<eos>']:
                break
            resp.append(next_token)
            current_ids = torch.cat([current_ids, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
        if compute_reward(resp, correct_sum) == 1.0:
            correct += 1
    eval_accuracy = correct / 100

    # Summary
    accs = [m['accuracy'] for m in metrics_history]
    rewards = [m['reward_mean'] for m in metrics_history]
    peak_acc = max(accs)
    final_acc = accs[-1]
    ge50 = sum(1 for a in accs if a >= 0.5)

    result = {
        'mode': mode,
        'params': param_count,
        'peak_accuracy': peak_acc,
        'final_accuracy': final_acc,
        'eval_accuracy': eval_accuracy,
        'steps_ge_50pct': ge50,
        'avg_reward_last50': np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards),
        'reward_std_last50': np.std(rewards[-50:]) if len(rewards) >= 50 else np.std(rewards),
        'advantage_mean_avg': np.mean([m['advantage_mean'] for m in metrics_history]),
        'elapsed_seconds': elapsed,
        'metrics_history': metrics_history,
    }

    print(f"\n  [{mode}] SUMMARY: peak={peak_acc:.1%}, final={final_acc:.1%}, "
          f"eval={eval_accuracy:.1%}, steps≥50%={ge50}/{num_steps}, "
          f"adv_mean_avg={result['advantage_mean_avg']:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', default='grpo,rloo,dapo', help='Comma-separated modes to compare')
    parser.add_argument('--num_steps', type=int, default=300)
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--num_prompts_per_step', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--output', default='rl_comparison_results.json')
    args = parser.parse_args()

    device = torch.device(args.device)
    modes = args.modes.split(',')

    print("=" * 70)
    print("Unified RL Training Comparison — Head-to-Head")
    print("=" * 70)
    print(f"Modes: {modes}")
    print(f"Steps: {args.num_steps}, n={args.n_samples}, prompts/step={args.num_prompts_per_step}")
    print(f"Model: hidden={args.hidden_dim}, layers={args.num_layers}")
    print(f"Device: {device}")
    print()

    all_results = {}
    for mode in modes:
        print(f"\n--- Running {mode} ---")
        result = run_single_mode(
            mode, args.num_steps, args.n_samples, args.num_prompts_per_step,
            args.max_response_len if hasattr(args, 'max_response_len') else 3,
            args.lr, device, args.hidden_dim, args.num_layers,
            kl_coeff=0.01, clip_lower=0.3, clip_upper=0.2
        )
        all_results[mode] = result

    # Final comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Mode':<10} {'Peak':>8} {'Final':>8} {'Eval':>8} {'≥50%':>6} {'AdvMean':>10} {'Time(s)':>8}")
    print("-" * 60)
    for mode, r in all_results.items():
        print(f"{mode:<10} {r['peak_accuracy']:>8.1%} {r['final_accuracy']:>8.1%} "
              f"{r['eval_accuracy']:>8.1%} {r['steps_ge_50pct']:>6d} "
              f"{r['advantage_mean_avg']:>10.4f} {r['elapsed_seconds']:>8.1f}")

    # Save
    output = {
        'config': {
            'modes': modes,
            'num_steps': args.num_steps,
            'n_samples': args.n_samples,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'lr': args.lr,
            'device': str(device),
        },
        'results': all_results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()