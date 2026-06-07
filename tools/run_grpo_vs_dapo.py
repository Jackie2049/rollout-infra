#!/usr/bin/env python3
"""Run GRPO vs DAPO comparison on GPU."""

import subprocess
import json
import time

# The comparison script will:
# 1. Run original GRPO for 300 steps
# 2. Run DAPO for 300 steps
# 3. Compare results

def run_experiment(mode, num_steps=300, n_samples=8):
    """Run mini_grpo_training.py with given mode on GPU."""
    cmd = [
        "python", "tools/mini_grpo_training.py",
        "--mode", mode,
        "--device", "cuda",
        "--num_steps", str(num_steps),
        "--n_samples", str(n_samples),
        "--num_prompts_per_step", "8",
        "--lr", "1e-3",
    ]
    if mode == "dapo":
        cmd.extend([
            "--kl_coeff", "0.01",
            "--clip_lower", "0.3",
            "--clip_upper", "0.2",
        ])

    print(f"\n{'='*60}")
    print(f"Running {mode} for {num_steps} steps, n={n_samples}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)

    # Parse results
    try:
        with open("mini_grpo_training_results.json") as f:
            data = json.load(f)
        return data
    except:
        return None

print("=" * 70)
print("GRPO vs DAPO Comparison Experiment")
print("RTX 4090, 300 steps, n=8")
print("=" * 70)

# Run GRPO
print("\n\n=== Experiment 1: Original GRPO ===")
grpo_data = run_experiment("grpo", num_steps=300, n_samples=8)

# Run DAPO
print("\n\n=== Experiment 2: DAPO (Improved GRPO) ===")
dapo_data = run_experiment("dapo", num_steps=300, n_samples=8)

# Compare
if grpo_data and dapo_data:
    print("\n\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    grpo_metrics = grpo_data.get('grpo_metrics', [])
    dapo_metrics = dapo_data.get('dapo_metrics', [])

    if grpo_metrics:
        print(f"\nGRPO: {len(grpo_metrics)} steps")
        print(f"  Initial: reward={grpo_metrics[0]['reward_mean']:.3f}, acc={grpo_metrics[0]['accuracy']:.1%}")
        print(f"  Peak:    reward={max(m['reward_mean'] for m in grpo_metrics):.3f}, "
              f"acc={max(m['accuracy'] for m in grpo_metrics):.1%}")
        print(f"  Final:   reward={grpo_metrics[-1]['reward_mean']:.3f}, acc={grpo_metrics[-1]['accuracy']:.1%}")

    if dapo_metrics:
        print(f"\nDAPO: {len(dapo_metrics)} steps")
        print(f"  Initial: reward={dapo_metrics[0]['reward_mean']:.3f}, acc={dapo_metrics[0]['accuracy']:.1%}")
        print(f"  Peak:    reward={max(m['reward_mean'] for m in dapo_metrics):.3f}, "
              f"acc={max(m['accuracy'] for m in dapo_metrics):.1%}")
        print(f"  Final:   reward={dapo_metrics[-1]['reward_mean']:.3f}, acc={dapo_metrics[-1]['accuracy']:.1%}")
        zero_grad_total = sum(m.get('zero_gradient_groups', 0) for m in dapo_metrics)
        print(f"  Zero-gradient groups total: {zero_grad_total} (DAPO avoids these)")
        dynamic_n_mean = np.mean([m.get('dynamic_n_mean', 0) for m in dapo_metrics]) if dapo_metrics else 0

        import numpy as np
        print(f"  Dynamic n mean: {dynamic_n_mean:.1f}")

    # Save comparison
    comparison = {
        'grpo_data': grpo_data,
        'dapo_data': dapo_data,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open("grpo_vs_dapo_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to grpo_vs_dapo_comparison.json")