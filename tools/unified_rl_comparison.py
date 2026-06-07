#!/usr/bin/env python3
"""Unified 7-Method RL Comparison — RTX 4090

Run all 7 RL methods simultaneously on 8 GPUs (one method per GPU).
All methods use identical seed=42, same architecture (76K params),
same training steps, for fair comparison.

Methods:
1. GRPO (GPU 0)
2. SFT→GRPO (GPU 1)
3. DAPO (GPU 2)
4. SFT→DAPO (GPU 3)
5. PPO (GPU 4)
6. DPO (GPU 5)
7. RLOO (GPU 6)

Usage: python tools/unified_rl_comparison.py --run
       (on GPU server with CUDA available)
"""

import subprocess
import json
import time
import sys
import os

METHODS = [
    ('grpo',       0, {'num_steps': 300, 'n_samples': 8}),
    ('sft_grpo',   1, {'num_steps': 300, 'n_samples': 8, 'sft_steps': 200}),
    ('dapo',       2, {'num_steps': 300, 'n_samples': 8, 'kl_coeff': 0.01, 'clip_lower': 0.3, 'clip_upper': 0.2}),
    ('sft_dapo',   3, {'num_steps': 300, 'n_samples': 8, 'sft_steps': 200, 'kl_coeff': 0.01, 'clip_lower': 0.3, 'clip_upper': 0.2}),
    ('ppo',        4, {'num_steps': 300, 'n_samples': 8}),
    ('dpo',        5, {'num_steps': 300}),
    ('rloo',       6, {'num_steps': 300, 'n_samples': 8}),
]

OUTPUT_FILES = {
    'grpo': 'results/grpo_unified.json',
    'sft_grpo': 'results/sft_grpo_unified.json',
    'dapo': 'results/dapo_unified.json',
    'sft_dapo': 'results/sft_dapo_unified.json',
    'ppo': 'results/ppo_unified.json',
    'dpo': 'results/dpo_unified.json',
    'rloo': 'results/rloo_unified.json',
}


def run_method(mode, gpu_id, params, timeout=600):
    """Run one method on a specific GPU."""
    cmd = [
        'python', 'tools/mini_grpo_training.py',
        '--mode', mode,
        '--device', f'cuda:{gpu_id}',
        '--hidden_dim', '64',
        '--num_layers', '2',
        '--lr', '1e-3',
        '--output', OUTPUT_FILES[mode],
    ]
    for k, v in params.items():
        cmd.extend([f'--{k}', str(v)])

    print(f"  Launching {mode} on GPU {gpu_id}...")
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['PYTHONPATH'] = ''

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, text=True
    )
    return proc


def analyze_results(results_dir='results'):
    """Analyze all results and create unified comparison."""
    comparison = {}

    for mode, output_file in OUTPUT_FILES.items():
        filepath = os.path.join(results_dir, output_file.split('/')[-1])
        try:
            with open(filepath) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"  {mode}: results not found")
            continue

        # Extract key metrics
        metrics_key = {
            'grpo': 'grpo_metrics', 'sft_grpo': 'grpo_metrics',
            'dapo': 'dapo_metrics', 'sft_dapo': 'dapo_metrics',
            'ppo': 'ppo_metrics', 'dpo': 'dpo_metrics',
            'rloo': 'grpo_metrics',  # RLOO stores in grpo_metrics
        }
        key = metrics_key[mode]
        m = data.get(key, [])

        if not m:
            print(f"  {mode}: no metrics")
            continue

        # Compute summary
        rewards = [x.get('reward_mean', 0) for x in m]
        accuracies = [x.get('accuracy', 0) for x in m]
        losses = [x.get('loss', 0) for x in m]

        comparison[mode] = {
            'initial_reward': rewards[0],
            'final_reward': rewards[-1],
            'peak_reward': max(rewards),
            'initial_accuracy': accuracies[0],
            'final_accuracy': accuracies[-1],
            'peak_accuracy': max(accuracies),
            'final_loss': losses[-1],
            'num_steps': len(m),
            'final_eval_accuracy': data.get('final_accuracy', None),
            'sft_steps': data.get('sft_metrics', [None])[0] if data.get('sft_metrics') else 0,
        }

        # For DAPO, also get zero_gradient_groups and dynamic_n
        if mode in ['dapo', 'sft_dapo']:
            zg = [x.get('zero_gradient_groups', 0) for x in m]
            dn = [x.get('dynamic_n_mean', 8) for x in m]
            comparison[mode]['zero_gradient_groups_mean'] = sum(zg) / len(zg)
            comparison[mode]['dynamic_n_mean'] = sum(dn) / len(dn)

        print(f"  {mode}: reward {comparison[mode]['initial_reward']:.3f} → {comparison[mode]['final_reward']:.3f}, "
              f"eval_acc={comparison[mode]['final_eval_accuracy']}")

    # Save comparison
    with open(os.path.join(results_dir, 'unified_rl_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 70)
    print("Unified 7-Method RL Comparison Summary")
    print("=" * 70)

    # Sort by eval accuracy
    sorted_modes = sorted(comparison.items(),
                          key=lambda x: (x[1].get('final_eval_accuracy') or 0),
                          reverse=True)

    for mode, data in sorted_modes:
        eval_acc = data.get('final_eval_accuracy', '?')
        peak_reward = data['peak_reward']
        final_reward = data['final_reward']
        print(f"  {mode:12s}: eval_acc={eval_acc}, peak_reward={peak_reward:.3f}, "
              f"final_reward={final_reward:.3f}")

    return comparison


if __name__ == '__main__':
    if '--run' in sys.argv:
        print("=" * 70)
        print("Unified 7-Method RL Comparison — RTX 4090")
        print("=" * 70)
        print("Running all methods in parallel on separate GPUs...")

        processes = {}
        for mode, gpu_id, params in METHODS:
            proc = run_method(mode, gpu_id, params)
            processes[mode] = proc

        # Wait for all to complete
        print("\nWaiting for all experiments to complete...")
        results = {}
        for mode, proc in processes.items():
            stdout, stderr = proc.communicate(timeout=600)
            results[mode] = {
                'returncode': proc.returncode,
                'stdout_tail': stdout[-500:] if stdout else '',
                'stderr_tail': stderr[-300:] if stderr else '',
            }
            status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
            print(f"  {mode}: {status}")

        # Analyze
        print("\n--- Analyzing Results ---")
        comparison = analyze_results('results')

    elif '--analyze' in sys.argv:
        print("--- Analyzing Existing Results ---")
        comparison = analyze_results('results')

    else:
        print("Usage: python tools/unified_rl_comparison.py --run    (run all experiments)")
        print("       python tools/unified_rl_comparison.py --analyze (analyze existing results)")