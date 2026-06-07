#!/usr/bin/env python3
"""Loss Landscape Visualization — RTX 4090

Visualize 2D loss landscape slices around different trained models
to understand why SFT warmstart produces better optimization basins.

Key questions:
1. Does SFT→GRPO land in a wider basin (better generalization)?
2. Does GRPO-only land in a narrow basin (worse generalization)?
3. How does the landscape change with training method?

Method: Use filter normalization (Li et al., 2018) for 2D landscape slices.
For each model, compute loss along two random directions in parameter space,
normalized by filter norms for fair comparison.

Usage: On GPU server: python tools/loss_landscape_visualization.py
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
    MiniGQATransformer, VOCAB_SIZE, TOKENS, IDX_TO_TOKEN,
    generate_arithmetic_prompt, compute_reward,
    grpo_training_step, sft_training_step, generate_sft_dataset,
)


def eval_model(model, device, n_eval=200):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    for _ in range(n_eval):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]], dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(input_ids)[0, -2, :].argmax().item()
        if pred == correct_sum:
            correct += 1
        total += 1
    return correct / total


def compute_loss(model, device, n_samples=100):
    """Compute CE loss on random prompts."""
    model.eval()
    total_loss = 0
    count = 0
    for _ in range(n_samples):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        answer_str = str(correct_sum)
        answer_tokens = [TOKENS.get(d, TOKENS['<unk>']) for d in answer_str]
        full_ids = prompt_tokens + answer_tokens + [TOKENS['<eos>']]
        target_ids = full_ids[1:]
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        targets = torch.tensor([target_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids)
            # Only compute loss on answer tokens
            prompt_len = len(prompt_tokens)
            answer_logits = logits[:, prompt_len:-1, :]
            answer_targets = targets[:, prompt_len:]
            if answer_logits.size(1) == answer_targets.size(1):
                loss = F.cross_entropy(answer_logits.reshape(-1, VOCAB_SIZE),
                                       answer_targets.reshape(-1))
                total_loss += loss.item()
                count += 1
    return total_loss / max(count, 1)


def get_random_direction(model, device):
    """Generate a random direction with filter normalization."""
    direction = []
    for name, param in model.named_parameters():
        d = torch.randn_like(param.data)
        # Filter normalization: normalize each filter (row) independently
        if param.dim() >= 2:
            # For weight matrices, normalize each output dimension
            for i in range(d.size(0)):
                d[i].div_(d[i].norm() + 1e-8)
                d[i].mul_(param.data[i].norm() + 1e-8)
        else:
            d.div_(d.norm() + 1e-8)
            d.mul_(param.data.norm() + 1e-8)
        direction.append(d.to(device))
    return direction


def apply_direction(model, base_params, direction1, direction2, alpha, beta):
    """Create model with parameters shifted along two directions."""
    for (name, param), base, d1, d2 in zip(
        model.named_parameters(), base_params, direction1, direction2):
        param.data = base + alpha * d1 + beta * d2


def compute_landscape(model, base_params, d1, d2, device,
                      alpha_range=(-1.0, 1.0), beta_range=(-1.0, 1.0),
                      resolution=21, n_loss_samples=100):
    """Compute 2D loss landscape."""
    alphas = np.linspace(alpha_range[0], alpha_range[1], resolution)
    betas = np.linspace(beta_range[0], beta_range[1], resolution)

    landscape = np.zeros((resolution, resolution))
    for i, alpha in enumerate(alphas):
        for j, beta_val in enumerate(betas):
            apply_direction(model, base_params, d1, d2, alpha, beta_val)
            landscape[i, j] = compute_loss(model, device, n_samples=n_loss_samples)

    return landscape, alphas, betas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--resolution', type=int, default=21)
    parser.add_argument('--output', default='loss_landscape_results.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    results = {}

    # ============================================================
    # Phase 1: Train models (same seed for fair comparison)
    # ============================================================
    print("\n=== Phase 1: Training models ===")

    # Base model (random init)
    torch.manual_seed(42)
    np.random.seed(42)
    base_model = MiniGQATransformer(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    base_acc = eval_model(base_model, device)
    base_loss = compute_loss(base_model, device)
    base_params = [p.data.clone() for p in base_model.parameters()]
    print(f"Base model: acc={base_acc:.1%}, loss={base_loss:.4f}")
    results['base'] = {'acc': base_acc, 'loss': base_loss}

    # SFT-only model
    torch.manual_seed(42)
    np.random.seed(42)
    sft_model = MiniGQATransformer(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    sft_dataset = generate_sft_dataset(500)
    sft_optimizer = torch.optim.AdamW(sft_model.parameters(), lr=2e-3)
    for step in range(200):
        sft_training_step(sft_model, sft_dataset, sft_optimizer, device)
    sft_acc = eval_model(sft_model, device)
    sft_loss = compute_loss(sft_model, device)
    sft_params = [p.data.clone() for p in sft_model.parameters()]
    print(f"SFT model: acc={sft_acc:.1%}, loss={sft_loss:.4f}")
    results['sft'] = {'acc': sft_acc, 'loss': sft_loss}

    # GRPO-only model
    torch.manual_seed(42)
    np.random.seed(42)
    grpo_model = MiniGQATransformer(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    grpo_optimizer = torch.optim.AdamW(grpo_model.parameters(), lr=1e-3)
    for step in range(300):
        prompts = [generate_arithmetic_prompt() for _ in range(8)]
        grpo_training_step(grpo_model, prompts, 8, 3, grpo_optimizer, device)
    grpo_acc = eval_model(grpo_model, device)
    grpo_loss = compute_loss(grpo_model, device)
    grpo_params = [p.data.clone() for p in grpo_model.parameters()]
    print(f"GRPO model: acc={grpo_acc:.1%}, loss={grpo_loss:.4f}")
    results['grpo'] = {'acc': grpo_acc, 'loss': grpo_loss}

    # SFT→GRPO model
    torch.manual_seed(42)
    np.random.seed(42)
    sft_grpo_model = MiniGQATransformer(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    sft_optimizer2 = torch.optim.AdamW(sft_grpo_model.parameters(), lr=2e-3)
    for step in range(200):
        sft_training_step(sft_grpo_model, sft_dataset, sft_optimizer2, device)
    grpo_optimizer2 = torch.optim.AdamW(sft_grpo_model.parameters(), lr=1e-3)
    for step in range(300):
        prompts = [generate_arithmetic_prompt() for _ in range(8)]
        grpo_training_step(sft_grpo_model, prompts, 8, 3, grpo_optimizer2, device)
    sft_grpo_acc = eval_model(sft_grpo_model, device)
    sft_grpo_loss = compute_loss(sft_grpo_model, device)
    sft_grpo_params = [p.data.clone() for p in sft_grpo_model.parameters()]
    print(f"SFT→GRPO model: acc={sft_grpo_acc:.1%}, loss={sft_grpo_loss:.4f}")
    results['sft_grpo'] = {'acc': sft_grpo_acc, 'loss': sft_grpo_loss}

    # ============================================================
    # Phase 2: Compute landscapes
    # ============================================================
    print("\n=== Phase 2: Computing loss landscapes ===")
    res = args.resolution

    models_data = {
        'base': (base_model, base_params),
        'sft': (sft_model, sft_params),
        'grpo': (grpo_model, grpo_params),
        'sft_grpo': (sft_grpo_model, sft_grpo_params),
    }

    for name, (model, params) in models_data.items():
        print(f"\nComputing landscape for {name}...")
        torch.manual_seed(123)  # Fixed seed for directions
        d1 = get_random_direction(model, device)
        d2 = get_random_direction(model, device)

        landscape, alphas, betas = compute_landscape(
            model, params, d1, d2, device,
            alpha_range=(-1.5, 1.5), beta_range=(-1.5, 1.5),
            resolution=res, n_loss_samples=50
        )

        # Restore original params
        for (pname, param), base_p in zip(model.named_parameters(), params):
            param.data = base_p.clone()

        results[name]['landscape'] = landscape.tolist()
        results[name]['alphas'] = alphas.tolist()
        results[name]['betas'] = betas.tolist()

        # Compute basin metrics
        center_loss = landscape[res//2, res//2]
        min_loss = landscape.min()
        max_loss = landscape.max()
        loss_std = landscape.std()

        # Flatness: how much loss increases when moving away from center
        # (flat basin = small increase, sharp basin = large increase)
        radius_losses = []
        for i in range(res):
            for j in range(res):
                dist = np.sqrt((i - res//2)**2 + (j - res//2)**2)
                if dist > 0 and dist <= 3:  # Within 3 steps from center
                    radius_losses.append(landscape[i, j] - center_loss)
        avg_increase = np.mean(radius_losses) if radius_losses else 0

        print(f"  Center loss={center_loss:.4f}, min={min_loss:.4f}, max={max_loss:.4f}")
        print(f"  Std={loss_std:.4f}, avg_increase(flatness)={avg_increase:.4f}")

        results[name]['center_loss'] = center_loss
        results[name]['min_loss'] = min_loss
        results[name]['max_loss'] = max_loss
        results[name]['loss_std'] = loss_std
        results[name]['avg_increase'] = avg_increase

    # ============================================================
    # Phase 3: Basin comparison
    # ============================================================
    print("\n=== Phase 3: Basin comparison ===")
    print("Flatness (avg loss increase from center):")
    for name in ['base', 'sft', 'grpo', 'sft_grpo']:
        flat = results[name]['avg_increase']
        acc = results[name]['acc']
        print(f"  {name:12s}: flatness={flat:.4f}, acc={acc:.1%}")

    print("\nLoss std (basin width proxy):")
    for name in ['base', 'sft', 'grpo', 'sft_grpo']:
        std = results[name]['loss_std']
        print(f"  {name:12s}: std={std:.4f}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("Loss Landscape Visualization Summary")
    print("=" * 70)
    print("Flat basin = small avg_increase = better generalization")
    print("Sharp basin = large avg_increase = worse generalization")
    print()
    for name in ['base', 'sft', 'grpo', 'sft_grpo']:
        flat = results[name]['avg_increase']
        acc = results[name]['acc']
        center = results[name]['center_loss']
        basin_type = "flat" if flat < 0.5 else "sharp"
        print(f"  {name:12s}: {basin_type} basin (flatness={flat:.3f}), "
              f"center_loss={center:.3f}, eval_acc={acc:.1%}")

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()