#!/usr/bin/env python3
"""Mini Interpretability Pipeline — SAE + Activation Patching on MiniGQATransformer

Train a Sparse Autoencoder on MiniGQATransformer activations, then use activation patching
to identify which layers/positions carry "arithmetic knowledge".

Can run on CPU (76K model) or GPU.
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


def generate_sft_dataset_local(n_examples):
    """Generate SFT dataset of (prompt, correct_answer) pairs."""
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


# ============================================================
# Sparse Autoencoder (SAE)
# ============================================================

class SparseAutoencoder(nn.Module):
    """TopK Sparse Autoencoder for decomposing neural activations.

    Uses TopK activation (Anthropic recommended):
    - Keep exactly K features with highest pre-activation values
    - Zeroes all others → guarantees L0=K, no feature death!
    - More stable than ReLU+L1 which suffers from feature death
    """
    def __init__(self, input_dim, feature_dim, top_k=10):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.top_k = top_k

        # Encoder: larger init to avoid feature death
        self.W_enc = nn.Parameter(torch.randn(feature_dim, input_dim) * (1.0 / input_dim))
        self.b_enc = nn.Parameter(torch.zeros(feature_dim))
        # Decoder: unit-norm columns for better feature separation
        self.W_dec = nn.Parameter(torch.randn(input_dim, feature_dim) * (1.0 / feature_dim))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def forward(self, z):
        f_pre = F.linear(z, self.W_enc, self.b_enc)
        # TopK: keep exactly K features
        topk_vals, topk_idx = f_pre.topk(self.top_k, dim=-1)
        f = torch.zeros_like(f_pre)
        f.scatter_(-1, topk_idx, F.relu(topk_vals))
        z_recon = F.linear(f, self.W_dec, self.b_dec)
        return f, z_recon

    def loss(self, z, f, z_recon):
        recon_loss = F.mse_loss(z_recon, z)
        # No L1 needed — TopK guarantees sparsity
        return recon_loss, recon_loss.item(), 0.0


# ============================================================
# Hook-based activation collection
# ============================================================

class ActivationCollector:
    """Collect intermediate activations from MiniGQATransformer using forward hooks."""
    def __init__(self, model):
        self.model = model
        self.activations = defaultdict(list)
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for i, layer in enumerate(self.model.layers):
            self.hooks.append(layer['ln2'].register_forward_hook(
                lambda mod, inp, out, idx=i: self.activations[f'residual_attn_{idx}'].append(out.detach())
            ))
            self.hooks.append(layer['down_proj'].register_forward_hook(
                lambda mod, inp, out, idx=i: self.activations[f'mlp_out_{idx}'].append(out.detach())
            ))
            self.hooks.append(layer['o_proj'].register_forward_hook(
                lambda mod, inp, out, idx=i: self.activations[f'attn_out_{idx}'].append(out.detach())
            ))

    def collect(self, input_ids):
        self.activations.clear()
        with torch.no_grad():
            _ = self.model(input_ids)
        return dict(self.activations)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================
# SAE Training
# ============================================================

def train_sae(model, sae, device, num_prompts=1000, batch_size=32, num_epochs=50,
              target_layer='residual_attn_0', lr=1e-3):
    collector = ActivationCollector(model)
    all_activations = []
    for _ in range(num_prompts // batch_size + 1):
        batch_prompts = [generate_arithmetic_prompt() for _ in range(batch_size)]
        input_ids_list = []
        for prompt_tokens, correct_sum in batch_prompts:
            full_ids = prompt_tokens + [TOKENS['<eos>']]
            input_ids_list.append(full_ids)
        max_len = max(len(ids) for ids in input_ids_list)
        padded = [ids + [TOKENS['<pad>']] * (max_len - len(ids)) for ids in input_ids_list]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)

        acts = collector.collect(input_ids)
        if target_layer in acts:
            for act_tensor in acts[target_layer]:
                all_activations.append(act_tensor)

    collector.remove_hooks()

    if not all_activations:
        print(f"  WARNING: No activations collected for {target_layer}")
        return sae, []

    flat_acts = []
    for act_tensor in all_activations:
        flat_acts.append(act_tensor.reshape(-1, model.hidden_dim))
    all_acts = torch.cat(flat_acts, dim=0)
    all_acts = all_acts[all_acts.abs().sum(dim=1) > 0]

    print(f"  Collected {all_acts.shape[0]} activation vectors, dim={all_acts.shape[1]}")
    print(f"  Training SAE: input_dim={sae.input_dim}, feature_dim={sae.feature_dim}")

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    metrics_history = []

    for epoch in range(num_epochs):
        indices = torch.randperm(all_acts.shape[0])[:batch_size * 10]
        batch = all_acts[indices].to(device)

        f, z_recon = sae(batch)
        total_loss, recon_loss, sparsity_loss = sae.loss(batch, f, z_recon)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        active_features = (f > 0).float().mean()
        avg_l0 = (f > 0).float().sum(dim=1).mean()

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"    Epoch {epoch}: recon={recon_loss:.6f}, sparsity={sparsity_loss:.6f}, "
                  f"active={active_features:.1%}, L0={avg_l0:.1f}")

        metrics_history.append({
            'epoch': epoch, 'recon_loss': recon_loss, 'sparsity_loss': sparsity_loss,
            'active_ratio': active_features.item(), 'avg_l0': avg_l0.item(),
        })

    return sae, metrics_history


# ============================================================
# SAE Feature Analysis
# ============================================================

def analyze_sae_features(model, sae, device, num_prompts=200):
    collector = ActivationCollector(model)
    feature_activations = defaultdict(list)

    for _ in range(num_prompts):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        full_ids = prompt_tokens + [TOKENS['<eos>']]
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

        acts = collector.collect(input_ids)
        if 'residual_attn_0' in acts and len(acts['residual_attn_0']) > 0:
            z = acts['residual_attn_0'][0][0, -1, :]
            f, _ = sae(z.unsqueeze(0))
            f = f.squeeze(0)

            top_features = f.topk(10)
            for feat_idx, feat_val in zip(top_features.indices, top_features.values):
                if feat_val > 0:
                    feature_activations[feat_idx.item()].append({
                        'prompt': ''.join(IDX_TO_TOKEN[t] for t in prompt_tokens),
                        'correct_sum': correct_sum,
                        'activation': feat_val.item(),
                    })

    collector.remove_hooks()

    print("\n  SAE Feature Analysis:")
    print("  " + "=" * 50)

    top_global_features = []
    for feat_idx, activations in feature_activations.items():
        avg_activation = np.mean([a['activation'] for a in activations])
        count = len(activations)
        sums = [a['correct_sum'] for a in activations]
        avg_sum = np.mean(sums)
        top_global_features.append((feat_idx, avg_activation, count, avg_sum))

    top_global_features.sort(key=lambda x: x[1], reverse=True)

    for feat_idx, avg_act, count, avg_sum in top_global_features[:10]:
        examples = feature_activations[feat_idx][:5]
        example_strs = [f"{e['prompt']}={e['correct_sum']}(act={e['activation']:.2f})" for e in examples]
        print(f"  Feature {feat_idx}: avg_act={avg_act:.3f}, count={count}, avg_sum={avg_sum:.1f}")
        print(f"    Examples: {', '.join(example_strs)}")

    return feature_activations


# ============================================================
# Activation Patching (Causal Intervention)
# ============================================================

def activation_patching(model, device, num_prompts=50):
    """Causal tracing: find which layer/position carries arithmetic knowledge.

    Strategy: Run model on TWO different inputs with same format but different numbers.
    - Clean: "3+2=" → correct_sum=5
    - Corrupt: "1+0=" → wrong answer from the perspective of clean_sum=5

    Patch: inject clean activation at each (layer, position) into corrupt run.
    Measure: how much does patching change corrupt output toward clean_sum.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    results = []

    avg_effects = defaultdict(float)
    effect_counts = defaultdict(int)

    model.eval()

    for trial in range(num_prompts):
        # Generate two different prompts
        a_clean, b_clean = np.random.randint(0, 5), np.random.randint(0, 5)
        a_corrupt, b_corrupt = np.random.randint(0, 5), np.random.randint(0, 5)
        # Ensure they're different
        while a_clean == a_corrupt and b_clean == b_corrupt:
            a_corrupt = np.random.randint(0, 5)
        correct_sum = a_clean + b_clean

        # Both inputs same length: [a, +, b, =, eos] (5 tokens)
        clean_ids = [TOKENS[str(a_clean)], TOKENS['+'], TOKENS[str(b_clean)], TOKENS['='], TOKENS['<eos>']]
        corrupt_ids = [TOKENS[str(a_corrupt)], TOKENS['+'], TOKENS[str(b_corrupt)], TOKENS['='], TOKENS['<eos>']]

        clean_tensor = torch.tensor([clean_ids], dtype=torch.long, device=device)
        corrupt_tensor = torch.tensor([corrupt_ids], dtype=torch.long, device=device)

        # Step 1: Collect clean activations
        clean_acts = {}
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                clean_acts[name] = output.detach().clone()
            return hook_fn

        for i, layer in enumerate(model.layers):
            hooks.append(layer['ln2'].register_forward_hook(make_hook(f'ln2_{i}')))
            hooks.append(layer['down_proj'].register_forward_hook(make_hook(f'mlp_{i}')))
            hooks.append(layer['o_proj'].register_forward_hook(make_hook(f'attn_{i}')))

        with torch.no_grad():
            clean_logits = model(clean_tensor)
        clean_prob = F.softmax(clean_logits[0, -2, :], dim=-1)[correct_sum].item()
        for h in hooks:
            h.remove()

        # Step 2: Get corrupt baseline probability for correct_sum
        with torch.no_grad():
            corrupt_logits = model(corrupt_tensor)
        corrupt_prob = F.softmax(corrupt_logits[0, -2, :], dim=-1)[correct_sum].item()

        # Step 3: Patch each (layer, position) with clean activation
        layer_names = list(clean_acts.keys())

        for layer_name in layer_names:
            clean_act = clean_acts[layer_name]  # [1, 5, H]
            seq_len = clean_act.shape[1]

            for pos in range(seq_len):
                patch_hooks = []

                def make_patch_hook(ln, p, ca):
                    def hook_fn(module, input, output):
                        patched = output.clone()
                        if p < patched.shape[1]:
                            patched[0, p] = ca[0, p]
                        return patched
                    return hook_fn

                for i, layer in enumerate(model.layers):
                    if f'ln2_{i}' == layer_name:
                        patch_hooks.append(layer['ln2'].register_forward_hook(
                            make_patch_hook(layer_name, pos, clean_act)))
                    elif f'mlp_{i}' == layer_name:
                        patch_hooks.append(layer['down_proj'].register_forward_hook(
                            make_patch_hook(layer_name, pos, clean_act)))
                    elif f'attn_{i}' == layer_name:
                        patch_hooks.append(layer['o_proj'].register_forward_hook(
                            make_patch_hook(layer_name, pos, clean_act)))

                with torch.no_grad():
                    patched_logits = model(corrupt_tensor)

                patched_prob = F.softmax(patched_logits[0, -2, :], dim=-1)[correct_sum].item()

                for h in patch_hooks:
                    h.remove()

                effect = patched_prob - corrupt_prob
                key = f'{layer_name}_pos{pos}'
                avg_effects[key] += effect
                effect_counts[key] += 1

        results.append({
            'a_clean': a_clean, 'b_clean': b_clean, 'a_corrupt': a_corrupt, 'b_corrupt': b_corrupt,
            'correct_sum': correct_sum, 'clean_prob': clean_prob, 'corrupt_prob': corrupt_prob,
        })

    # Aggregate
    final_effects = {}
    for key in avg_effects:
        final_effects[key] = avg_effects[key] / effect_counts[key]

    print("\n  Activation Patching Results (effect = patched_prob - corrupt_baseline):")
    print("  " + "=" * 50)

    sorted_effects = sorted(final_effects.items(), key=lambda x: x[1], reverse=True)
    for key, val in sorted_effects[:15]:
        print(f"  {key}: {val:.4f}")

    avg_clean = np.mean([r['clean_prob'] for r in results])
    avg_corrupt = np.mean([r['corrupt_prob'] for r in results])
    print(f"\n  Avg clean baseline prob: {avg_clean:.3f}")
    print(f"  Avg corrupt baseline prob: {avg_corrupt:.3f}")

    return results, final_effects


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Mini Interpretability Pipeline')
    parser.add_argument('--mode', default='all',
                        choices=['sae', 'patching', 'compare', 'all'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--sae_features', type=int, default=128)
    parser.add_argument('--sae_topk', type=int, default=10, help='TopK for SAE (number of active features)')
    parser.add_argument('--sae_epochs', type=int, default=50)
    parser.add_argument('--output', default='interpretability_results.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("Mini Interpretability Pipeline — SAE + Activation Patching")
    print("=" * 70)
    print(f"Model: hidden={args.hidden_dim}, layers={args.num_layers}")
    print(f"SAE: features={args.sae_features}, top_k={args.sae_topk}")
    print(f"Device: {device}")

    all_results = {}

    # Create and SFT-train model
    torch.manual_seed(42)
    np.random.seed(42)
    model = MiniGQATransformer(hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                                num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count}")

    # SFT warmup
    print("\n--- SFT Warmup ---")
    sft_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sft_dataset = generate_sft_dataset_local(500)
    for step in range(200):
        batch_idx = np.random.randint(0, len(sft_dataset), size=8)
        batch = [sft_dataset[i] for i in batch_idx]
        max_len = max(len(b['full_ids']) for b in batch)
        padded_full = [b['full_ids'] + [TOKENS['<pad>']] * (max_len - len(b['full_ids'])) for b in batch]
        padded_target = [b['target_ids'] + [TOKENS['<pad>']] * (max_len - len(b['target_ids'])) for b in batch]
        full_ids = torch.tensor(padded_full, dtype=torch.long, device=device)
        targets = torch.tensor(padded_target, dtype=torch.long, device=device)
        logits = model(full_ids)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        sft_optimizer.zero_grad()
        loss.backward()
        sft_optimizer.step()
        if step % 50 == 0:
            correct = 0
            for _ in range(20):
                pt, cs = generate_arithmetic_prompt()
                fi = torch.tensor([pt + [TOKENS['<eos>']]], dtype=torch.long, device=device)
                with torch.no_grad():
                    pred = model(fi)[0, -2, :].argmax().item()
                if pred == cs:
                    correct += 1
            print(f"  SFT step {step}: loss={loss.item():.4f}, accuracy={correct/20:.1%}")

    model.eval()
    correct = 0
    for _ in range(100):
        pt, cs = generate_arithmetic_prompt()
        fi = torch.tensor([pt + [TOKENS['<eos>']]], dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(fi)[0, -2, :].argmax().item()
        if pred == cs:
            correct += 1
    sft_accuracy = correct / 100
    print(f"  SFT final accuracy: {sft_accuracy:.1%}")
    all_results['sft_accuracy'] = sft_accuracy

    if args.mode in ['sae', 'all']:
        print("\n--- SAE Training ---")
        sae = SparseAutoencoder(
            input_dim=args.hidden_dim,
            feature_dim=args.sae_features,
            top_k=args.sae_topk,
        ).to(device)
        sae, sae_metrics = train_sae(model, sae, device, num_epochs=args.sae_epochs)
        feature_acts = analyze_sae_features(model, sae, device)
        all_results['sae'] = {
            'feature_dim': args.sae_features, 'top_k': args.sae_topk,
            'metrics': sae_metrics, 'param_count': sum(p.numel() for p in sae.parameters()),
        }

    if args.mode in ['patching', 'all']:
        print("\n--- Activation Patching ---")
        patch_results, avg_effects = activation_patching(model, device)
        all_results['patching'] = {
            'avg_effects': avg_effects, 'num_prompts': len(patch_results),
        }

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()