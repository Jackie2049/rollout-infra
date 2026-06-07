#!/usr/bin/env python3
"""Activation Checkpointing Experiment — RTX 4090

Compare different checkpointing strategies:
1. No checkpointing (baseline)
2. Every layer checkpoint (all activations recomputed)
3. Every 2nd layer checkpoint (selective)
4. Every 3rd layer checkpoint
5. Last N layers only checkpoint (tail strategy)

Key questions:
1. What memory savings does each strategy give?
2. What throughput cost (recompute overhead)?
3. Which strategy gives the best trade-off?
4. Does checkpointing affect convergence?

Usage: On GPU server:
  CUDA_VISIBLE_DEVICES=0 python -u tools/activation_checkpointing_experiment.py --model_size 76k
  CUDA_VISIBLE_DEVICES=0 python -u tools/activation_checkpointing_experiment.py --model_size 2.28m
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.mini_grpo_training import (
    MiniGQATransformer, VOCAB_SIZE, TOKENS,
    generate_arithmetic_prompt, generate_sft_dataset,
)


def apply_checkpointing(model, strategy, num_layers):
    """Apply checkpointing by modifying the forward pass to use checkpoint."""
    # MiniGQATransformer has inline forward — we modify it to checkpoint
    # individual layer blocks using a wrapper approach
    model._checkpoint_strategy = strategy
    model._num_layers = num_layers


class _CheckpointedLayer(nn.Module):
    """Wrapper that applies gradient checkpointing to a layer."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(
            self.layer, *args, use_reentrant=False, **kwargs)


def _run_layer(layer, x, g, head_dim):
    """Run a single transformer layer block (for checkpointing)."""
    residual = x
    x = layer['ln1'](x)
    B, S, H = x.shape
    Q = layer['q_proj'](x).view(B, S, -1, head_dim)
    K = layer['k_proj'](x).view(B, S, -1, head_dim)
    V = layer['v_proj'](x).view(B, S, -1, head_dim)
    if g > 1:
        K = K.repeat_interleave(g, dim=2)
        V = V.repeat_interleave(g, dim=2)
    attn_out = F.scaled_dot_product_attention(
        Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2), is_causal=True
    )
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
    x = residual + layer['o_proj'](attn_out)
    residual = x
    x = layer['ln2'](x)
    gate = torch.sigmoid(layer['gate_proj'](x))
    x = residual + layer['down_proj'](gate * layer['up_proj'](x))
    return x


def train_with_checkpointing(strategy, model_size, num_steps, grad_accum_steps=4,
                               lr=2e-3, dtype=torch.bfloat16):
    """Train with a specific checkpointing strategy."""
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    np.random.seed(42)

    hd = 64 if model_size == '76k' else (256 if model_size == '2.28m' else 512)
    nl = 2 if model_size == '76k' else (4 if model_size == '2.28m' else 8)
    nh = 4 if model_size == '76k' else (8 if model_size == '2.28m' else 16)
    nkv = 2 if model_size == '76k' else (4 if model_size == '2.28m' else 8)

    model = MiniGQATransformer(
        hidden_dim=hd, num_layers=nl, num_heads=nh,
        num_kv_heads=nkv, vocab_size=VOCAB_SIZE).to(device=device, dtype=dtype)

    # Determine which layers to checkpoint
    ckpt_indices = set()
    if strategy == 'every':
        ckpt_indices = set(range(nl))
    elif strategy == 'every_2nd':
        ckpt_indices = set(range(1, nl, 2))
    elif strategy == 'every_3rd':
        ckpt_indices = set(range(2, nl, 3))
    elif strategy == 'tail_half':
        ckpt_indices = set(range(nl // 2, nl))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset = generate_sft_dataset(500)

    # Memory measurement
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    losses = []
    start_time = time.time()

    for step in range(num_steps):
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro in range(grad_accum_steps):
            idx = np.random.randint(len(dataset))
            full_tokens, prompt_len = dataset[idx]
            input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
            targets = input_ids.clone()

            # Custom forward with checkpointing
            x = model.embed(input_ids)
            for i, layer in enumerate(model.layers):
                if i in ckpt_indices and strategy != 'none':
                    x = torch.utils.checkpoint.checkpoint(
                        _run_layer, layer, x, model.g, model.head_dim,
                        use_reentrant=False)
                else:
                    x = _run_layer(layer, x, model.g, model.head_dim)
            x = model.final_ln(x)
            logits = model.lm_head(x)

            loss = F.cross_entropy(
                logits[:, prompt_len-1:-1, :].reshape(-1, VOCAB_SIZE),
                targets[:, prompt_len:].reshape(-1))
            loss = loss / grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(accum_loss * grad_accum_steps)

    elapsed = time.time() - start_time
    throughput = num_steps / elapsed
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Eval
    model.eval()
    model_fp32 = model.float()
    correct = 0
    for _ in range(100):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]],
                                  dtype=torch.long, device=device)
        with torch.no_grad():
            # No checkpointing during eval
            pred = model_fp32(input_ids)[0, -2, :].argmax().item()
        if pred == correct_sum:
            correct += 1
    eval_acc = correct / 100

    # Count checkpointed layers
    checkpointed = len(ckpt_indices)

    result = {
        'strategy': strategy,
        'model_size': model_size,
        'num_layers': nl,
        'checkpointed_layers': checkpointed,
        'grad_accum_steps': grad_accum_steps,
        'num_steps': num_steps,
        'total_time_s': elapsed,
        'throughput_steps_s': throughput,
        'final_loss': losses[-1],
        'min_loss': min(losses),
        'eval_accuracy': eval_acc,
        'peak_gpu_mem_gb': peak_mem,
    }

    print(f"  {strategy}: {throughput:.1f} steps/s, "
          f"loss={losses[-1]:.3f}, eval={eval_acc:.1%}, "
          f"mem={peak_mem:.4f}GB, "
          f"ckpt_layers={checkpointed}/{nl}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--model_size', default='76k',
                        choices=['76k', '2.28m'])
    parser.add_argument('--output', default='activation_checkpointing_results.json')
    args = parser.parse_args()

    print("=" * 70)
    print("Activation Checkpointing Experiment — RTX 4090")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}
    strategies = ['none', 'every_2nd', 'every_3rd', 'tail_half', 'every']

    # Only run strategies that make sense for the model size
    # 76K has 2 layers → every_2nd = 1 layer, every_3rd = 0 layers (skip)
    # 2.28M has 4 layers → every_2nd = 2, every_3rd = 1, tail_half = 2, every = 4

    print(f"\n--- Model: {args.model_size}, GA=4 ---")
    for strategy in strategies:
        # Skip strategies that don't make sense
        nl = 2 if args.model_size == '76k' else 4
        if strategy == 'every_3rd' and nl < 3:
            continue
        if strategy == 'tail_half' and nl < 2:
            continue

        result = train_with_checkpointing(strategy, args.model_size, args.num_steps)
        key = f'ckpt_{strategy}_{args.model_size}'
        results[key] = result

    # Also test without GA (single micro-step) to see memory difference
    print(f"\n--- Memory comparison (GA=1, no gradient accumulation) ---")
    for strategy in ['none', 'every_2nd'] if args.model_size == '76k' else ['none', 'every_2nd', 'every', 'tail_half']:
        result = train_with_checkpointing(strategy, args.model_size, args.num_steps,
                                           grad_accum_steps=1)
        key = f'ckpt_{strategy}_ga1_{args.model_size}'
        results[key] = result

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    baseline_key = f'ckpt_none_{args.model_size}'
    baseline = results[baseline_key]
    baseline_mem = baseline['peak_gpu_mem_gb']
    baseline_throughput = baseline['throughput_steps_s']

    print(f"\nCheckpointing comparison (GA=4, {args.model_size}):")
    for strategy in strategies:
        key = f'ckpt_{strategy}_{args.model_size}'
        if key in results:
            r = results[key]
            mem_saved = (1 - r['peak_gpu_mem_gb'] / baseline_mem) * 100 if baseline_mem > 0 else 0
            throughput_ratio = r['throughput_steps_s'] / baseline_throughput
            print(f"  {strategy}: mem={r['peak_gpu_mem_gb']:.4f}GB "
                  f"(saved {mem_saved:.1f}%), "
                  f"throughput={throughput_ratio:.2f}x, "
                  f"eval={r['eval_accuracy']:.1%}, "
                  f"ckpt={r['checkpointed_layers']}/{r['num_layers']}")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()