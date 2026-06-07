#!/usr/bin/env python3
"""GRPO vs SFT→GRPO Circuit Comparison

Compare internal representations of two differently trained models:
1. GRPO-only: trained from random initialization with GRPO RL
2. SFT→GRPO: SFT warmup then GRPO RL

Key question: Does SFT create different "arithmetic circuits" than GRPO-only?
- If yes → explains why SFT→GRPO has 93% eval vs GRPO-only 81%
- Measures: cosine similarity, activation norm, patching sensitivity, feature overlap

Can run on CPU (76K model) or GPU (RTX 4090).
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
    generate_arithmetic_prompt, compute_reward, grpo_training_step,
)


def generate_sft_dataset(n_examples):
    dataset = []
    for _ in range(n_examples):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        answer_str = str(correct_sum)
        answer_tokens = [TOKENS.get(d, TOKENS['<unk>']) for d in answer_str]
        full_ids = prompt_tokens + answer_tokens + [TOKENS['<eos>']]
        target_ids = full_ids[1:]
        dataset.append({'full_ids': full_ids, 'target_ids': target_ids})
    return dataset


def train_sft(model, device, n_steps=200, lr=2e-3):
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
        if step % 50 == 0:
            correct = 0
            for _ in range(20):
                pt, cs = generate_arithmetic_prompt()
                fi = torch.tensor([pt + [TOKENS['<eos>']]], dtype=torch.long, device=device)
                with torch.no_grad():
                    pred = model(fi)[0, -2, :].argmax().item()
                if pred == cs:
                    correct += 1
            print(f"  SFT step {step}: loss={loss.item():.4f}, acc={correct/20:.1%}")
    return model


def collect_activations(model, input_ids):
    """Collect all layer activations for given input."""
    acts = {}
    hooks = []
    def make_hook(name):
        def hook_fn(module, input, output):
            acts[name] = output.detach().clone()
        return hook_fn
    for i, layer in enumerate(model.layers):
        hooks.append(layer['ln2'].register_forward_hook(make_hook(f'ln2_{i}')))
        hooks.append(layer['down_proj'].register_forward_hook(make_hook(f'mlp_{i}')))
        hooks.append(layer['o_proj'].register_forward_hook(make_hook(f'attn_{i}')))
    with torch.no_grad():
        logits = model(input_ids)
    for h in hooks:
        h.remove()
    return acts, logits


# ============================================================
# Circuit Comparison
# ============================================================

def compare_circuits(grpo_model, sft_model, device, num_prompts=200):
    """Compare internal representations of GRPO-only vs SFT→GRPO models.

    Measures per (layer, position):
    1. Cosine similarity: how similar are activations?
    2. Norm ratio: which model has larger activations?
    3. Answer probability: how confident is each model?
    """
    grpo_model.eval()
    sft_model.eval()

    results = {
        'cosine_sim': defaultdict(list),
        'norm_ratio': defaultdict(list),
        'prob_diff': defaultdict(list),
        'logit_diff': defaultdict(list),
    }

    for _ in range(num_prompts):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        input_ids = [prompt_tokens + [TOKENS['<eos>']]]
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)

        g_acts, g_logits = collect_activations(grpo_model, input_tensor)
        s_acts, s_logits = collect_activations(sft_model, input_tensor)

        # Compare per layer, per position
        for layer_name in g_acts:
            g_act = g_acts[layer_name]  # [1, S, H]
            s_act = s_acts[layer_name]  # [1, S, H]
            seq_len = min(g_act.shape[1], s_act.shape[1])

            for pos in range(seq_len):
                g_vec = g_act[0, pos, :]
                s_vec = s_act[0, pos, :]
                key = f'{layer_name}_pos{pos}'

                # Cosine similarity
                cos_sim = F.cosine_similarity(g_vec.unsqueeze(0), s_vec.unsqueeze(0)).item()
                results['cosine_sim'][key].append(cos_sim)

                # Norm ratio
                g_norm = g_vec.norm().item()
                s_norm = s_vec.norm().item()
                norm_ratio = g_norm / s_norm if s_norm > 0 else 0
                results['norm_ratio'][key].append(norm_ratio)

            # Compare logits at equals-sign position (pos 3, predicting next token)
            g_logits_at_eq = g_logits[0, 2, :]  # Position after '=' (pos=2 is '='... wait, '=' is pos 3)
            s_logits_at_eq = s_logits[0, 3, :]   # Actually '=' is pos 3

        # Compare final answer probabilities
        g_prob_correct = F.softmax(g_logits[0, -2, :], dim=-1)[correct_sum].item()
        s_prob_correct = F.softmax(s_logits[0, -2, :], dim=-1)[correct_sum].item()
        results['prob_diff']['answer'].append(s_prob_correct - g_prob_correct)

    # Aggregate
    print("\n  GRPO vs SFT→GRPO Circuit Comparison:")
    print("  " + "=" * 60)

    # Per-layer averages (collapse positions)
    layer_avgs = {}
    for layer_name in ['ln2_0', 'mlp_0', 'attn_0', 'ln2_1', 'mlp_1', 'attn_1']:
        keys = [k for k in results['cosine_sim'] if k.startswith(layer_name)]
        avg_cos = np.mean([np.mean(results['cosine_sim'][k]) for k in keys])
        avg_norm = np.mean([np.mean(results['norm_ratio'][k]) for k in keys])
        layer_avgs[layer_name] = {'cos_sim': avg_cos, 'norm_ratio': avg_norm}
        print(f"  {layer_name}: cos_sim={avg_cos:.4f}, norm_ratio(G/S)={avg_norm:.3f}")

    avg_prob_diff = np.mean(results['prob_diff']['answer'])
    print(f"\n  Answer prob diff (SFT-GRPO): {avg_prob_diff:+.3f}")

    # Position-level analysis (collapse layers)
    print("\n  Position-level cos_sim averages:")
    for pos in range(5):
        keys = [k for k in results['cosine_sim'] if k.endswith(f'pos{pos}')]
        avg_cos = np.mean([np.mean(results['cosine_sim'][k]) for k in keys])
        print(f"  Position {pos} (token={IDX_TO_TOKEN.get(input_ids[0][pos], '?')}): cos_sim={avg_cos:.4f}")

    return results, layer_avgs


# ============================================================
# Activation Patching Sensitivity Comparison
# ============================================================

def compare_patching_sensitivity(grpo_model, sft_model, device, num_prompts=30):
    """Compare how sensitive each model is to activation patching.

    Key question: Is SFT→GRPO model more robust (less sensitive to patching)?
    Or is GRPO-only model more fragile (more affected by patching)?
    """
    torch.manual_seed(42)
    np.random.seed(42)

    grpo_effects = defaultdict(float)
    sft_effects = defaultdict(float)

    for trial in range(num_prompts):
        a1, b1 = np.random.randint(0, 5), np.random.randint(0, 5)
        a2, b2 = np.random.randint(0, 5), np.random.randint(0, 5)
        while a1 == a2 and b1 == b2:
            a2 = np.random.randint(0, 5)
        correct_sum = a1 + b1

        clean_ids = [TOKENS[str(a1)], TOKENS['+'], TOKENS[str(b1)], TOKENS['='], TOKENS['<eos>']]
        corrupt_ids = [TOKENS[str(a2)], TOKENS['+'], TOKENS[str(b2)], TOKENS['='], TOKENS['<eos>']]
        clean_tensor = torch.tensor([clean_ids], dtype=torch.long, device=device)
        corrupt_tensor = torch.tensor([corrupt_ids], dtype=torch.long, device=device)

        for model, effects_dict in [(grpo_model, grpo_effects), (sft_model, sft_effects)]:
            model.eval()

            # Collect clean activations
            clean_acts, _ = collect_activations(model, clean_tensor)

            # Get corrupt baseline
            with torch.no_grad():
                corrupt_logits = model(corrupt_tensor)
            corrupt_prob = F.softmax(corrupt_logits[0, -2, :], dim=-1)[correct_sum].item()

            # Patch at equals-sign position (pos 3) for each layer
            for layer_name in ['attn_0', 'attn_1', 'ln2_1', 'mlp_1']:
                clean_act = clean_acts[layer_name]
                pos = 3  # Equals sign

                hook = None
                for i, layer in enumerate(model.layers):
                    if f'attn_{i}' == layer_name:
                        target_layer = layer['o_proj']
                    elif f'ln2_{i}' == layer_name:
                        target_layer = layer['ln2']
                    elif f'mlp_{i}' == layer_name:
                        target_layer = layer['down_proj']

                def make_patch(ca, p):
                    def hook_fn(module, input, output):
                        patched = output.clone()
                        if p < patched.shape[1]:
                            patched[0, p] = ca[0, p]
                        return patched
                    return hook_fn

                # Find the correct layer module
                patch_hooks = []
                for i, layer_m in enumerate(model.layers):
                    if f'attn_{i}' == layer_name:
                        patch_hooks.append(layer_m['o_proj'].register_forward_hook(make_patch(clean_act, pos)))
                    elif f'ln2_{i}' == layer_name:
                        patch_hooks.append(layer_m['ln2'].register_forward_hook(make_patch(clean_act, pos)))
                    elif f'mlp_{i}' == layer_name:
                        patch_hooks.append(layer_m['down_proj'].register_forward_hook(make_patch(clean_act, pos)))

                with torch.no_grad():
                    patched_logits = model(corrupt_tensor)

                for h in patch_hooks:
                    h.remove()

                patched_prob = F.softmax(patched_logits[0, -2, :], dim=-1)[correct_sum].item()
                effect = patched_prob - corrupt_prob
                effects_dict[layer_name + '_pos3'] += effect / num_prompts

    print("\n  Patching Sensitivity Comparison (at equals-sign pos 3):")
    print("  " + "=" * 50)
    print(f"  {'Layer':<10} {'GRPO-only':>12} {'SFT→GRPO':>12} {'Ratio':>8}")
    print("  " + "-" * 42)

    for layer_name in ['attn_0', 'attn_1', 'ln2_1', 'mlp_1']:
        g_eff = grpo_effects[layer_name + '_pos3']
        s_eff = sft_effects[layer_name + '_pos3']
        ratio = s_eff / g_eff if abs(g_eff) > 0.001 else 'N/A'
        print(f"  {layer_name:<10} {g_eff:>12.4f} {s_eff:>12.4f} {ratio:>8}")

    return grpo_effects, sft_effects


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--grpo_steps', type=int, default=300)
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--output', default='circuit_comparison.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("GRPO vs SFT→GRPO Circuit Comparison")
    print("=" * 70)

    # Train GRPO-only model
    print("\n--- Training GRPO-only model ---")
    torch.manual_seed(42)
    np.random.seed(42)
    grpo_model = MiniGQATransformer(hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                                      num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    optimizer = torch.optim.AdamW(grpo_model.parameters(), lr=1e-3)

    for step in range(args.grpo_steps):
        prompts = [generate_arithmetic_prompt() for _ in range(8)]
        metrics = grpo_training_step(grpo_model, prompts, args.n_samples, 3, optimizer, device)
        if step % 50 == 0:
            print(f"  GRPO step {step}: acc={metrics['accuracy']:.1%}")

    # Eval GRPO model
    grpo_model.eval()
    correct = 0
    for _ in range(100):
        pt, cs = generate_arithmetic_prompt()
        fi = torch.tensor([pt + [TOKENS['<eos>']]], dtype=torch.long, device=device)
        with torch.no_grad():
            pred = grpo_model(fi)[0, -2, :].argmax().item()
        if pred == cs:
            correct += 1
    grpo_acc = correct / 100
    print(f"  GRPO eval accuracy: {grpo_acc:.1%}")

    # Train SFT→GRPO model
    print("\n--- Training SFT→GRPO model ---")
    torch.manual_seed(42)
    np.random.seed(42)
    sft_model = MiniGQATransformer(hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                                     num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)

    # Phase 1: SFT warmup
    sft_model = train_sft(sft_model, device, n_steps=200, lr=2e-3)

    # Phase 2: GRPO RL
    optimizer2 = torch.optim.AdamW(sft_model.parameters(), lr=1e-3)
    for step in range(args.grpo_steps):
        prompts = [generate_arithmetic_prompt() for _ in range(8)]
        metrics = grpo_training_step(sft_model, prompts, args.n_samples, 3, optimizer2, device)
        if step % 50 == 0:
            print(f"  SFT→GRPO step {step}: acc={metrics['accuracy']:.1%}")

    sft_model.eval()
    correct = 0
    for _ in range(100):
        pt, cs = generate_arithmetic_prompt()
        fi = torch.tensor([pt + [TOKENS['<eos>']]], dtype=torch.long, device=device)
        with torch.no_grad():
            pred = sft_model(fi)[0, -2, :].argmax().item()
        if pred == cs:
            correct += 1
    sft_acc = correct / 100
    print(f"  SFT→GRPO eval accuracy: {sft_acc:.1%}")

    # Circuit comparison
    print("\n--- Circuit Comparison ---")
    circuit_results, layer_avgs = compare_circuits(grpo_model, sft_model, device)

    # Patching sensitivity comparison
    print("\n--- Patching Sensitivity Comparison ---")
    grpo_eff, sft_eff = compare_patching_sensitivity(grpo_model, sft_model, device)

    # Save results
    output = {
        'grpo_eval_accuracy': grpo_acc,
        'sft_eval_accuracy': sft_acc,
        'circuit_comparison': layer_avgs,
        'grpo_patching_effects': dict(grpo_eff),
        'sft_patching_effects': dict(sft_eff),
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()