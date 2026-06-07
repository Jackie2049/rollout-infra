#!/usr/bin/env python3
"""Mini Interpretability Pipeline — SAE + Activation Patching on MiniGQATransformer

Train a Sparse Autoencoder on MiniGQATransformer activations, then use activation patching
to identify which layers/positions carry "arithmetic knowledge".

Key experiments:
1. SAE feature discovery: find "digit features" and "operation features"
2. Activation patching: identify "arithmetic circuit" critical layers
3. Compare GRPO-only vs SFT→GRPO models' internal representations
4. Verify: SFT models have more robust arithmetic circuits

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
    """Sparse Autoencoder for decomposing neural activations into interpretable features.

    Architecture: z (d_dim) → encoder (d_dim → f_dim) → ReLU → decoder (f_dim → d_dim)
    L1 sparsity forces most features to be zero → each active feature is interpretable.
    """
    def __init__(self, input_dim, feature_dim, sparsity_weight=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.sparsity_weight = sparsity_weight

        # Encoder: z → f (with ReLU activation for sparsity)
        self.W_enc = nn.Parameter(torch.randn(feature_dim, input_dim) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(feature_dim))

        # Decoder: f → z' (reconstruct from sparse features)
        self.W_dec = nn.Parameter(torch.randn(input_dim, feature_dim) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

    def forward(self, z):
        """Encode to sparse features, then reconstruct."""
        # Pre-activation features
        f_pre = F.linear(z, self.W_enc, self.b_enc)
        # ReLU sparsity: most features are exactly 0
        f = F.relu(f_pre)
        # Reconstruct from sparse features
        z_recon = F.linear(f, self.W_dec, self.b_dec)
        return f, z_recon

    def loss(self, z, f, z_recon):
        """Reconstruction loss + L1 sparsity penalty."""
        recon_loss = F.mse_loss(z_recon, z)
        sparsity_loss = self.sparsity_weight * f.abs().mean()
        return recon_loss + sparsity_loss, recon_loss.item(), sparsity_loss.item()


# ============================================================
# Hook-based activation collection
# ============================================================

class ActivationCollector:
    """Collect intermediate activations from MiniGQATransformer using forward hooks.

    Saves: MLP activations, Residual stream activations, Attention output activations.
    """
    def __init__(self, model):
        self.model = model
        self.activations = defaultdict(list)
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on all layers to capture activations."""
        for i, layer in enumerate(self.model.layers):
            # Hook 1: Residual stream after attention (before MLP)
            self.hooks.append(layer['ln2'].register_forward_hook(
                lambda mod, inp, out, idx=i: self.activations[f'residual_attn_{idx}'].append(out.detach())
            ))
            # Hook 2: MLP output (after down_proj)
            self.hooks.append(layer['down_proj'].register_forward_hook(
                lambda mod, inp, out, idx=i: self.activations[f'mlp_out_{idx}'].append(out.detach())
            ))
            # Hook 3: Attention output (before o_proj reshape)
            self.hooks.append(layer['o_proj'].register_forward_hook(
                lambda mod, inp, out, idx=i: self.activations[f'attn_out_{idx}'].append(out.detach())
            ))

    def collect(self, input_ids):
        """Run forward pass and collect all activations."""
        self.activations.clear()
        with torch.no_grad():
            _ = self.model(input_ids)
        return dict(self.activations)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================
# SAE Training
# ============================================================

def train_sae(model, sae, device, num_prompts=1000, batch_size=32, num_epochs=50,
              target_layer='residual_attn_0', lr=1e-3):
    """Train SAE on collected activations from target_layer.

    Steps:
    1. Generate prompts and collect activations
    2. Train SAE to reconstruct activations with L1 sparsity
    """
    collector = ActivationCollector(model)

    # Collect activations
    print(f"  Collecting activations from {target_layer}...")
    all_activations = []
    for _ in range(num_prompts // batch_size + 1):
        batch_prompts = [generate_arithmetic_prompt() for _ in range(batch_size)]
        # Build input tensor: [prompt + <eos>] for each prompt
        input_ids_list = []
        for prompt_tokens, correct_sum in batch_prompts:
            full_ids = prompt_tokens + [TOKENS['<eos>']]
            input_ids_list.append(full_ids)
        # Pad to same length
        max_len = max(len(ids) for ids in input_ids_list)
        padded = [ids + [TOKENS['<pad>']] * (max_len - len(ids)) for ids in input_ids_list]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)

        acts = collector.collect(input_ids)
        if target_layer in acts:
            # acts[target_layer] is a list of tensors from hooks
            for act_tensor in acts[target_layer]:
                all_activations.append(act_tensor)

    collector.remove_hooks()

    if not all_activations:
        print(f"  WARNING: No activations collected for {target_layer}")
        return sae, []

    # Concatenate: each activation is [B, S, H] → flatten to [N, H]
    flat_acts = []
    for act_tensor in all_activations:
        flat_acts.append(act_tensor.reshape(-1, model.hidden_dim))
    all_acts = torch.cat(flat_acts, dim=0)
    # Remove padding positions (where all activations might be zero)
    all_acts = all_acts[all_acts.abs().sum(dim=1) > 0]

    print(f"  Collected {all_acts.shape[0]} activation vectors, dim={all_acts.shape[1]}")
    print(f"  Training SAE: input_dim={sae.input_dim}, feature_dim={sae.feature_dim}")

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    metrics_history = []

    for epoch in range(num_epochs):
        # Mini-batch training
        indices = torch.randperm(all_acts.shape[0])[:batch_size * 10]
        batch = all_acts[indices].to(device)

        f, z_recon = sae(batch)
        total_loss, recon_loss, sparsity_loss = sae.loss(batch, f, z_recon)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Compute sparsity metrics
        active_features = (f > 0).float().mean()
        avg_l0 = (f > 0).float().sum(dim=1).mean()

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"    Epoch {epoch}: recon={recon_loss:.6f}, sparsity={sparsity_loss:.6f}, "
                  f"active={active_features:.1%}, L0={avg_l0:.1f}")

        metrics_history.append({
            'epoch': epoch,
            'recon_loss': recon_loss,
            'sparsity_loss': sparsity_loss,
            'active_ratio': active_features.item(),
            'avg_l0': avg_l0.item(),
        })

    return sae, metrics_history


# ============================================================
# SAE Feature Analysis
# ============================================================

def analyze_sae_features(model, sae, device, num_prompts=200):
    """Analyze which SAE features activate on which inputs.

    For each feature, find the inputs that most activate it → interpret what it represents.
    """
    collector = ActivationCollector(model)

    # Collect activations with associated prompts
    feature_activations = defaultdict(list)  # feature_idx → list of (prompt, correct_sum, activation_strength)
    prompt_data = []

    for _ in range(num_prompts):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        full_ids = prompt_tokens + [TOKENS['<eos>']]
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

        acts = collector.collect(input_ids)
        # Use last position's residual_attn_0 activation (where the model "decides" the answer)
        if 'residual_attn_0' in acts and len(acts['residual_attn_0']) > 0:
            # acts dict values are lists of tensors from hooks; take first tensor [1, S, H]
            z = acts['residual_attn_0'][0][0, -1, :]  # [H] last position
            f, _ = sae(z.unsqueeze(0))
            f = f.squeeze(0)  # [feature_dim]

            # Find top activating features
            top_features = f.topk(10)
            for feat_idx, feat_val in zip(top_features.indices, top_features.values):
                if feat_val > 0:
                    feature_activations[feat_idx.item()].append({
                        'prompt': ''.join(IDX_TO_TOKEN[t] for t in prompt_tokens),
                        'correct_sum': correct_sum,
                        'activation': feat_val.item(),
                    })
            prompt_data.append({
                'prompt_tokens': prompt_tokens,
                'correct_sum': correct_sum,
                'features': {IDX_TO_TOKEN.get(t, '?') for t in prompt_tokens},
            })

    collector.remove_hooks()

    # Analyze top features: what inputs activate them most?
    print("\n  SAE Feature Analysis:")
    print("  " + "=" * 50)

    top_global_features = []
    for feat_idx, activations in feature_activations.items():
        avg_activation = np.mean([a['activation'] for a in activations])
        count = len(activations)
        # Find common patterns in prompts that activate this feature
        sums = [a['correct_sum'] for a in activations]
        avg_sum = np.mean(sums)
        top_global_features.append((feat_idx, avg_activation, count, avg_sum))

    # Sort by average activation
    top_global_features.sort(key=lambda x: x[1], reverse=True)

    # Print top 10 features
    for feat_idx, avg_act, count, avg_sum in top_global_features[:10]:
        examples = feature_activations[feat_idx][:5]
        example_strs = [f"{e['prompt']}={e['correct_sum']}(act={e['activation']:.2f})" for e in examples]
        print(f"  Feature {feat_idx}: avg_act={avg_act:.3f}, count={count}, avg_sum={avg_sum:.1f}")
        print(f"    Examples: {', '.join(example_strs)}")

    return feature_activations


# ============================================================
# Activation Patching (Causal Intervention)
# ============================================================

def activation_patching(model, device, num_prompts=50, max_response_len=3):
    """Causal tracing: find which layer/position carries arithmetic knowledge.

    Method:
    1. Run model on correct input → get clean activations
    2. Run model on corrupted input → get corrupted activations
    3. Patch: replace corrupted activations with clean ones at each (layer, position)
    4. Measure: how much does patching restore correct output?

    Corruption: replace the answer digit with a wrong digit.
    """
    torch.manual_seed(42)
    results = []

    for trial in range(num_prompts):
        a = np.random.randint(0, 5)
        b = np.random.randint(0, 5)
        correct_sum = a + b
        wrong_sum = (correct_sum + 3) % 10  # Corrupted answer

        # Clean input: a + b = <eos> (model predicts correct_sum at position after =)
        clean_ids = [TOKENS[str(a)], TOKENS['+'], TOKENS[str(b)], TOKENS['='], TOKENS['<eos>']]
        # Corrupted input: replace answer-related position
        corrupt_ids = clean_ids.copy()

        clean_tensor = torch.tensor([clean_ids], dtype=torch.long, device=device)
        corrupt_tensor = torch.tensor([corrupt_ids], dtype=torch.long, device=device)

        # Collect clean activations with hooks
        clean_acts = {}
        hooks = []

        def make_hook(layer_name):
            def hook_fn(module, input, output):
                clean_acts[layer_name] = output.detach().clone()
            return hook_fn

        for i, layer in enumerate(model.layers):
            hooks.append(layer['ln2'].register_forward_hook(make_hook(f'ln2_{i}')))
            hooks.append(layer['down_proj'].register_forward_hook(make_hook(f'mlp_{i}')))
            hooks.append(layer['o_proj'].register_forward_hook(make_hook(f'attn_{i}')))

        # Run clean
        model.eval()
        with torch.no_grad():
            clean_logits = model(clean_tensor)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Now patch each (layer, position) and measure effect
        patching_effects = {}
        for layer_name in clean_acts:
            clean_act = clean_acts[layer_name]  # [1, S, H]

            # Patch at each position
            for pos in range(clean_act.shape[1]):
                # Create patched input: run corrupt, but replace activation at (layer, pos) with clean
                patch_hooks = []
                patched_output = [None]

                def make_patch_hook(ln, p):
                    def hook_fn(module, input, output):
                        patched = output.clone()
                        patched[0, p] = clean_act[0, p]
                        return patched
                    return hook_fn

                # Only patch the specific layer and position
                target_hooks = []
                for i, layer in enumerate(model.layers):
                    if f'ln2_{i}' == layer_name:
                        target_hooks.append(layer['ln2'].register_forward_hook(make_patch_hook(layer_name, pos)))
                    elif f'mlp_{i}' == layer_name:
                        target_hooks.append(layer['down_proj'].register_forward_hook(make_patch_hook(layer_name, pos)))
                    elif f'attn_{i}' == layer_name:
                        target_hooks.append(layer['o_proj'].register_forward_hook(make_patch_hook(layer_name, pos)))

                with torch.no_grad():
                    patched_logits = model(corrupt_tensor)

                # Remove patch hooks
                for h in target_hooks:
                    h.remove()

                # Measure effect: probability of correct_sum vs wrong_sum
                last_logits = patched_logits[0, -2, :]  # Position after '=' sign
                correct_prob = F.softmax(last_logits, dim=-1)[correct_sum].item()

                patching_effects[f'{layer_name}_pos{pos}'] = correct_prob

        results.append({
            'a': a, 'b': b, 'correct_sum': correct_sum,
            'patching_effects': patching_effects,
        })

    # Aggregate results
    avg_effects = defaultdict(float)
    for r in results:
        for key, val in r['patching_effects'].items():
            avg_effects[key] += val / len(results)

    print("\n  Activation Patching Results (avg prob of correct answer):")
    print("  " + "=" * 50)

    # Sort by effect magnitude
    sorted_effects = sorted(avg_effects.items(), key=lambda x: x[1], reverse=True)
    for key, val in sorted_effects[:15]:
        print(f"  {key}: {val:.3f}")

    return results, avg_effects


# ============================================================
# Model Comparison: GRPO vs SFT→GRPO circuits
# ============================================================

def compare_model_circuits(grpo_model, sft_model, device, num_prompts=100):
    """Compare internal representations of GRPO-only vs SFT→GRPO models.

    Measures:
    1. Cosine similarity of activations at each layer
    2. Activation norm differences
    3. Feature overlap (which SAE features activate in each model)
    """
    results = {'cosine_sim': {}, 'norm_diff': {}, 'activation_overlap': {}}

    for _ in range(num_prompts):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        full_ids = prompt_tokens + [TOKENS['<eos>']]
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

        # Collect activations from both models
        grpo_acts = {}
        sft_acts = {}

        def make_hook(store, name):
            def hook_fn(module, input, output):
                store[name] = output.detach().clone()
            return hook_fn

        # GRPO model hooks
        grpo_hooks = []
        for i, layer in enumerate(grpo_model.layers):
            grpo_hooks.append(layer['ln2'].register_forward_hook(make_hook(grpo_acts, f'ln2_{i}')))

        with torch.no_grad():
            _ = grpo_model(input_ids)
        for h in grpo_hooks:
            h.remove()

        # SFT model hooks
        sft_hooks = []
        for i, layer in enumerate(sft_model.layers):
            sft_hooks.append(layer['ln2'].register_forward_hook(make_hook(sft_acts, f'ln2_{i}')))

        with torch.no_grad():
            _ = sft_model(input_ids)
        for h in sft_hooks:
            h.remove()

        # Compare activations at each layer
        for layer_name in grpo_acts:
            g_act = grpo_acts[layer_name].reshape(-1)
            s_act = sft_acts[layer_name].reshape(-1)

            # Cosine similarity
            cos_sim = F.cosine_similarity(g_act.unsqueeze(0), s_act.unsqueeze(0)).item()

            # Norm difference
            norm_diff = (g_act.norm().item() - s_act.norm().item())

            if layer_name not in results['cosine_sim']:
                results['cosine_sim'][layer_name] = []
                results['norm_diff'][layer_name] = []

            results['cosine_sim'][layer_name].append(cos_sim)
            results['norm_diff'][layer_name].append(norm_diff)

    # Aggregate
    for layer_name in results['cosine_sim']:
        results['cosine_sim'][layer_name] = np.mean(results['cosine_sim'][layer_name])
        results['norm_diff'][layer_name] = np.mean(results['norm_diff'][layer_name])

    print("\n  Model Circuit Comparison (GRPO vs SFT→GRPO):")
    print("  " + "=" * 50)
    for layer_name in sorted(results['cosine_sim'].keys()):
        print(f"  {layer_name}: cos_sim={results['cosine_sim'][layer_name]:.4f}, "
              f"norm_diff={results['norm_diff'][layer_name]:.2f}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Mini Interpretability Pipeline')
    parser.add_argument('--mode', default='all',
                        choices=['sae', 'patching', 'compare', 'all'],
                        help='Which experiment to run')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--sae_features', type=int, default=128,
                        help='Number of SAE features (should be > hidden_dim)')
    parser.add_argument('--sae_sparsity', type=float, default=0.01)
    parser.add_argument('--sae_epochs', type=int, default=50)
    parser.add_argument('--output', default='interpretability_results.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("Mini Interpretability Pipeline — SAE + Activation Patching")
    print("=" * 70)
    print(f"Model: hidden={args.hidden_dim}, layers={args.num_layers}")
    print(f"SAE: features={args.sae_features}, sparsity={args.sae_sparsity}")
    print(f"Device: {device}")
    print()

    all_results = {}

    # Create model (same seed as GRPO training for consistency)
    torch.manual_seed(42)
    np.random.seed(42)
    model = MiniGQATransformer(hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                                num_heads=4, num_kv_heads=2, vocab_size=VOCAB_SIZE).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count}")

    # Phase 1: SFT warmup — train model to learn arithmetic before interpretability
    # Without training, patching is meaningless (all positions have random effect)
    print("\n--- SFT Warmup (training model to learn arithmetic) ---")
    sft_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sft_dataset = generate_sft_dataset_local(500)
    for step in range(200):
        batch_idx = np.random.randint(0, len(sft_dataset), size=8)
        batch = [sft_dataset[i] for i in batch_idx]
        # Pad to same length
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
            # Quick eval
            correct = 0
            for _ in range(20):
                pt, cs = generate_arithmetic_prompt()
                fi = torch.tensor([pt + [TOKENS['<eos>']]], dtype=torch.long, device=device)
                with torch.no_grad():
                    pred = model(fi)[0, -2, :].argmax().item()
                if pred == cs:
                    correct += 1
            print(f"  SFT step {step}: loss={loss.item():.4f}, accuracy={correct/20:.1%}")

    # Final eval
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
            sparsity_weight=args.sae_sparsity,
        ).to(device)
        sae, sae_metrics = train_sae(model, sae, device, num_epochs=args.sae_epochs)
        feature_acts = analyze_sae_features(model, sae, device)
        all_results['sae'] = {
            'feature_dim': args.sae_features,
            'sparsity': args.sae_sparsity,
            'metrics': sae_metrics,
            'param_count': sum(p.numel() for p in sae.parameters()),
        }

    if args.mode in ['patching', 'all']:
        print("\n--- Activation Patching ---")
        patch_results, avg_effects = activation_patching(model, device)
        all_results['patching'] = {
            'avg_effects': avg_effects,
            'num_prompts': len(patch_results),
        }

    # Save results
    with open(args.output, 'w') as f:
        # Convert defaultdict to dict for JSON
        serializable = {}
        for key, val in all_results.items():
            if isinstance(val, dict):
                serializable[key] = {k: v if not isinstance(v, defaultdict) else dict(v)
                                     for k, v in val.items()}
            else:
                serializable[key] = val
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()