#!/usr/bin/env python3
"""Model Merging Experiment — RTX 4090

Test Task Arithmetic, SLERP, Simple Average, and DARE-TIES merging
on MiniGQATransformer models trained with different methods.

Key questions:
1. Can we merge GRPO + SFT models → better than either alone?
2. Does Task Arithmetic work? (add task vectors)
3. SLERP vs linear average — which preserves more?
4. DARE — can we drop 90% of delta and still work?

Usage: On GPU server with CUDA_VISIBLE_DEVICES=X python tools/model_merging_experiment.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.mini_grpo_training import (
    MiniGQATransformer, VOCAB_SIZE, TOKENS, IDX_TO_TOKEN,
    generate_arithmetic_prompt, compute_reward,
)


def generate_sft_dataset(n_examples):
    """Generate SFT training dataset."""
    dataset = []
    for _ in range(n_examples):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        answer_str = str(correct_sum)
        answer_tokens = [TOKENS.get(d, TOKENS['<unk>']) for d in answer_str]
        full_ids = prompt_tokens + answer_tokens + [TOKENS['<eos>']]
        target_ids = full_ids[1:]
        dataset.append({'full_ids': full_ids, 'target_ids': target_ids,
                         'correct_sum': correct_sum})
    return dataset


def train_sft(model, device, n_steps=200, lr=2e-3):
    """SFT warmup training."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    sft_dataset = generate_sft_dataset(500)
    for step in range(n_steps):
        batch_idx = np.random.randint(0, len(sft_dataset), size=8)
        batch = [sft_dataset[i] for i in batch_idx]
        max_len = max(len(b['full_ids']) for b in batch)
        padded_full = [b['full_ids'] + [TOKENS['<pad>']] * (max_len - len(b['full_ids'])) for b in batch]
        padded_target = [b['target_ids'] + [TOKENS['<pad>']] * (max_len - len(b['target_ids'])) for b in batch]
        full_ids = torch.tensor(padded_full, dtype=torch.long, device=device)
        targets = torch.tensor(padded_target, dtype=torch.long, device=device)
        logits = model(full_ids)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


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


def get_task_vector(model_finetuned, model_base):
    """Compute task vector: delta = finetuned - base."""
    delta = {}
    for (name_ft, param_ft), (name_base, param_base) in zip(
        model_finetuned.named_parameters(), model_base.named_parameters()):
        assert name_ft == name_base
        delta[name_ft] = param_ft.data - param_base.data
    return delta


def apply_task_vector(model_base, delta, alpha=1.0):
    """Apply task vector: new = base + alpha * delta."""
    merged = MiniGQATransformer(
        hidden_dim=model_base.layers[0]['o_proj'].out_features,
        num_layers=len(model_base.layers),
        num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE
    ).to(model_base.layers[0]['o_proj'].weight.device)

    for (name_m, param_m), (name_b, param_b) in zip(
        merged.named_parameters(), model_base.named_parameters()):
        assert name_m == name_b
        if name_m in delta:
            param_m.data = param_b.data + alpha * delta[name_m]
        else:
            param_m.data = param_b.data.clone()

    return merged


def slerp_merge(model_a, model_b, t=0.5):
    """SLERP merge: spherical linear interpolation."""
    merged = MiniGQATransformer(
        hidden_dim=model_a.layers[0]['o_proj'].out_features,
        num_layers=len(model_a.layers),
        num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE
    ).to(next(model_a.parameters()).device)

    for (name_m, param_m), (name_a, param_a), (name_b, param_b) in zip(
        merged.named_parameters(), model_a.named_parameters(), model_b.named_parameters()):
        v0 = param_a.data.flatten().float()
        v1 = param_b.data.flatten().float()

        # Normalize
        v0_norm = v0 / (v0.norm() + 1e-8)
        v1_norm = v1 / (v1.norm() + 1e-8)

        # Compute angle
        dot = torch.dot(v0_norm, v1_norm).clamp(-1.0, 1.0)
        theta = torch.acos(dot)

        if theta < 1e-6:  # Nearly parallel → linear interpolation
            result = (1 - t) * v0 + t * v1
        else:
            # SLERP formula
            sin_theta = torch.sin(theta)
            a = torch.sin((1 - t) * theta) / sin_theta
            b = torch.sin(t * theta) / sin_theta
            result = a * v0 + b * v1

        param_m.data = result.reshape(param_m.shape).to(param_m.dtype)

    return merged


def dare_preprocess(delta, drop_ratio=0.9):
    """DARE: randomly drop delta parameters and rescale."""
    dare_delta = {}
    for name, d in delta.items():
        mask = torch.rand_like(d) > drop_ratio  # Keep ~10%
        rescale = 1.0 / (1.0 - drop_ratio)
        dare_delta[name] = d * mask * rescale
    return dare_delta


def simple_average(model_a, model_b):
    """Simple weight averaging."""
    merged = MiniGQATransformer(
        hidden_dim=model_a.layers[0]['o_proj'].out_features,
        num_layers=len(model_a.layers),
        num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE
    ).to(next(model_a.parameters()).device)

    for (name_m, param_m), (name_a, param_a), (name_b, param_b) in zip(
        merged.named_parameters(), model_a.named_parameters(), model_b.named_parameters()):
        param_m.data = (param_a.data + param_b.data) / 2.0

    return merged


def ties_merge(model_base, deltas, top_k_ratio=0.2):
    """TIES merging: Trim, Elect, Sign-based merge."""
    # Step 1: Trim each delta — keep top-k% most significant
    trimmed_deltas = []
    for delta in deltas:
        trimmed = {}
        for name, d in delta.items():
            abs_vals = d.abs()
            threshold = abs_vals.kthvalue(int(abs_vals.numel() * (1 - top_k_ratio))).values
            trimmed[name] = d * (abs_vals >= threshold).float()
        trimmed_deltas.append(trimmed)

    # Step 2: Elect — for each parameter, take sign majority
    merged_delta = {}
    for name in trimmed_deltas[0]:
        signs = torch.stack([td[name].sign() for td in trimmed_deltas], dim=0)
        elected_sign = signs.sum(dim=0).sign()  # Majority vote

        # Step 3: Only keep sign-consistent changes
        final_delta = torch.zeros_like(trimmed_deltas[0][name])
        for td in trimmed_deltas:
            mask = (td[name].sign() == elected_sign).float()
            final_delta += td[name] * mask

        merged_delta[name] = final_delta / len(trimmed_deltas)

    return apply_task_vector(model_base, merged_delta, alpha=1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--grpo_steps', type=int, default=300)
    parser.add_argument('--sft_steps', type=int, default=200)
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--output', default='model_merging_results.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = {}

    # ============================================================
    # Phase 1: Train base, SFT-only, GRPO-only, SFT→GRPO models
    # ============================================================
    print("\n=== Phase 1: Training individual models ===")

    # Base model (random init)
    torch.manual_seed(42)
    np.random.seed(42)
    base_model = MiniGQATransformer(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    base_acc = eval_model(base_model, device)
    print(f"Base model (random): {base_acc:.1%}")
    results['base_acc'] = base_acc

    # SFT-only model
    torch.manual_seed(42)
    np.random.seed(42)
    sft_model = MiniGQATransformer(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    sft_model = train_sft(sft_model, device, n_steps=args.sft_steps)
    sft_acc = eval_model(sft_model, device)
    print(f"SFT-only: {sft_acc:.1%}")
    results['sft_only_acc'] = sft_acc

    # GRPO-only model
    torch.manual_seed(42)
    np.random.seed(42)
    grpo_model = MiniGQATransformer(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    optimizer_grpo = torch.optim.AdamW(grpo_model.parameters(), lr=1e-3)
    from tools.mini_grpo_training import grpo_training_step
    for step in range(args.grpo_steps):
        prompts = [generate_arithmetic_prompt() for _ in range(8)]
        metrics = grpo_training_step(grpo_model, prompts, args.n_samples, 3, optimizer_grpo, device)
        if step % 100 == 0:
            print(f"  GRPO step {step}: acc={metrics['accuracy']:.1%}")
    grpo_acc = eval_model(grpo_model, device)
    print(f"GRPO-only: {grpo_acc:.1%}")
    results['grpo_only_acc'] = grpo_acc

    # SFT→GRPO model
    torch.manual_seed(42)
    np.random.seed(42)
    sft_grpo_model = MiniGQATransformer(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    sft_grpo_model = train_sft(sft_grpo_model, device, n_steps=args.sft_steps)
    optimizer_sg = torch.optim.AdamW(sft_grpo_model.parameters(), lr=1e-3)
    for step in range(args.grpo_steps):
        prompts = [generate_arithmetic_prompt() for _ in range(8)]
        metrics = grpo_training_step(sft_grpo_model, prompts, args.n_samples, 3, optimizer_sg, device)
        if step % 100 == 0:
            print(f"  SFT→GRPO step {step}: acc={metrics['accuracy']:.1%}")
    sft_grpo_acc = eval_model(sft_grpo_model, device)
    print(f"SFT→GRPO: {sft_grpo_acc:.1%}")
    results['sft_grpo_acc'] = sft_grpo_acc

    # ============================================================
    # Phase 2: Task Arithmetic experiments
    # ============================================================
    print("\n=== Phase 2: Task Arithmetic ===")

    # Task vectors
    delta_sft = get_task_vector(sft_model, base_model)
    delta_grpo = get_task_vector(grpo_model, base_model)
    delta_sft_grpo = get_task_vector(sft_grpo_model, base_model)

    # Experiment 1: Add SFT task vector to base (alpha sweep)
    print("\n--- SFT task vector alpha sweep ---")
    alpha_results = {}
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        merged = apply_task_vector(base_model, delta_sft, alpha=alpha)
        acc = eval_model(merged, device)
        print(f"  α={alpha:.2f}: {acc:.1%}")
        alpha_results[str(alpha)] = acc
    results['sft_task_vector_sweep'] = alpha_results

    # Experiment 2: Add GRPO task vector to base
    print("\n--- GRPO task vector alpha sweep ---")
    alpha_results_grpo = {}
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        merged = apply_task_vector(base_model, delta_grpo, alpha=alpha)
        acc = eval_model(merged, device)
        print(f"  α={alpha:.2f}: {acc:.1%}")
        alpha_results_grpo[str(alpha)] = acc
    results['grpo_task_vector_sweep'] = alpha_results_grpo

    # Experiment 3: Add SFT + GRPO vectors (combine two tasks)
    print("\n--- Combined SFT+GRPO task vectors ---")
    combined_results = {}
    for alpha_sft, alpha_grpo in [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)]:
        merged = apply_task_vector(base_model, delta_sft, alpha=alpha_sft)
        # Then add GRPO delta
        merged_delta = {}
        for (name_m, param_m), (name_b, param_b) in zip(
            merged.named_parameters(), base_model.named_parameters()):
            merged_delta[name_m] = param_m.data - param_b.data + alpha_grpo * delta_grpo[name_m]
        merged2 = apply_task_vector(base_model, merged_delta, alpha=1.0)
        acc = eval_model(merged2, device)
        print(f"  α_sft={alpha_sft:.1f}, α_grpo={alpha_grpo:.1f}: {acc:.1%}")
        combined_results[f"{alpha_sft}_{alpha_grpo}"] = acc
    results['combined_sft_grpo_task_vectors'] = combined_results

    # ============================================================
    # Phase 3: SLERP vs Linear Average
    # ============================================================
    print("\n=== Phase 3: SLERP vs Linear Average ===")

    # SLERP: SFT + GRPO
    slerp_model = slerp_merge(sft_model, grpo_model, t=0.5)
    slerp_acc = eval_model(slerp_model, device)
    print(f"SLERP(SFT, GRPO, t=0.5): {slerp_acc:.1%}")
    results['slerp_sft_grpo'] = slerp_acc

    # Linear average: SFT + GRPO
    avg_model = simple_average(sft_model, grpo_model)
    avg_acc = eval_model(avg_model, device)
    print(f"Linear avg(SFT, GRPO): {avg_acc:.1%}")
    results['linear_avg_sft_grpo'] = avg_acc

    # SLERP t-sweep
    print("\n--- SLERP t-sweep ---")
    slerp_sweep = {}
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        slerp_m = slerp_merge(sft_model, grpo_model, t=t)
        acc = eval_model(slerp_m, device)
        print(f"  t={t:.1f}: {acc:.1%}")
        slerp_sweep[str(t)] = acc
    results['slerp_sweep'] = slerp_sweep

    # SLERP: SFT + SFT→GRPO (should be close to both)
    slerp_model2 = slerp_merge(sft_model, sft_grpo_model, t=0.5)
    slerp_acc2 = eval_model(slerp_model2, device)
    print(f"SLERP(SFT, SFT→GRPO, t=0.5): {slerp_acc2:.1%}")
    results['slerp_sft_sftgrpo'] = slerp_acc2

    # ============================================================
    # Phase 4: DARE experiments
    # ============================================================
    print("\n=== Phase 4: DARE ===")

    # DARE: drop different ratios of SFT delta
    print("\n--- DARE drop ratio sweep (SFT task vector) ---")
    dare_results = {}
    for drop_ratio in [0.5, 0.7, 0.9, 0.95, 0.99]:
        torch.manual_seed(123)  # Fixed seed for reproducibility
        dare_delta = dare_preprocess(delta_sft, drop_ratio=drop_ratio)
        merged = apply_task_vector(base_model, dare_delta, alpha=1.0)
        acc = eval_model(merged, device)

        # Count how many params survived
        total_params = sum(d.numel() for d in delta_sft.values())
        kept_params = sum((d.abs() > 0).sum().item() for d in dare_delta.values())
        kept_pct = kept_params / total_params * 100
        print(f"  drop={drop_ratio:.0%}: {acc:.1%} (kept {kept_pct:.1f}% params)")
        dare_results[str(drop_ratio)] = {'acc': acc, 'kept_pct': kept_pct}
    results['dare_sft_sweep'] = dare_results

    # DARE: drop ratio for GRPO delta
    print("\n--- DARE drop ratio sweep (GRPO task vector) ---")
    dare_results_grpo = {}
    for drop_ratio in [0.5, 0.7, 0.9, 0.95, 0.99]:
        torch.manual_seed(123)
        dare_delta = dare_preprocess(delta_grpo, drop_ratio=drop_ratio)
        merged = apply_task_vector(base_model, dare_delta, alpha=1.0)
        acc = eval_model(merged, device)
        print(f"  drop={drop_ratio:.0%}: {acc:.1%}")
        dare_results_grpo[str(drop_ratio)] = acc
    results['dare_grpo_sweep'] = dare_results_grpo

    # DARE-TIES: combine DARE preprocessing + simple averaging
    print("\n--- DARE-TIES: merge SFT and GRPO ---")
    dare_ties_results = {}
    for drop_ratio in [0.9, 0.95]:
        torch.manual_seed(123)
        dare_delta_sft = dare_preprocess(delta_sft, drop_ratio=drop_ratio)
        dare_delta_grpo = dare_preprocess(delta_grpo, drop_ratio=drop_ratio)
        # Simple average of DARE-processed deltas
        combined_delta = {}
        for name in dare_delta_sft:
            combined_delta[name] = (dare_delta_sft[name] + dare_delta_grpo[name]) / 2.0
        merged = apply_task_vector(base_model, combined_delta, alpha=1.0)
        acc = eval_model(merged, device)
        print(f"  DARE-TIES drop={drop_ratio:.0%}: {acc:.1%}")
        dare_ties_results[str(drop_ratio)] = acc
    results['dare_ties_sft_grpo'] = dare_ties_results

    # ============================================================
    # Phase 5: Task negation (subtract undesired behavior)
    # ============================================================
    print("\n=== Phase 5: Task Negation ===")

    # Subtract GRPO from SFT→GRPO → should get closer to SFT-only
    delta_neg = {}
    for name in delta_sft_grpo:
        delta_neg[name] = delta_sft_grpo[name] - 0.5 * delta_grpo[name]
    neg_model = apply_task_vector(base_model, delta_neg, alpha=1.0)
    neg_acc = eval_model(neg_model, device)
    print(f"SFT→GRPO - 0.5×GRPO (negate GRPO influence): {neg_acc:.1%}")
    results['negation_grpo'] = neg_acc

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("Model Merging Experiment Summary — RTX 4090")
    print("=" * 70)
    print(f"  Base (random):      {results['base_acc']:.1%}")
    print(f"  SFT-only:           {results['sft_only_acc']:.1%}")
    print(f"  GRPO-only:          {results['grpo_only_acc']:.1%}")
    print(f"  SFT→GRPO:           {results['sft_grpo_acc']:.1%}")
    print(f"  Linear avg(SFT,GRPO): {results['linear_avg_sft_grpo']:.1%}")
    print(f"  SLERP(SFT,GRPO,t=0.5): {results['slerp_sft_grpo']:.1%}")
    print(f"  DARE(90% drop, SFT): {results['dare_sft_sweep']['0.9']['acc']:.1%}")
    print(f"  Task negation:      {results['negation_grpo']:.1%}")

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert all float values to plain floats for JSON
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (float, np.floating)):
            return float(obj)
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        return obj
    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()