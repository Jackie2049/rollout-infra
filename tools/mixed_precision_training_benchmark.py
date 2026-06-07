#!/usr/bin/env python3
"""Mixed Precision Training Benchmark — RTX 4090

Compare training quality and throughput across different precision modes:
1. FP32 (baseline)
2. FP16 + AMP (automatic mixed precision with GradScaler)
3. BF16 (native, no GradScaler needed)
4. FP8 (if supported on SM89 — experimental)

Key questions:
1. Does BF16 match FP32 quality without GradScaler?
2. What throughput gain does each precision offer?
3. Is FP16+AMP safe for small model training?
4. Can FP8 work for training on RTX 4090 (SM89)?

Usage: On GPU server:
  CUDA_VISIBLE_DEVICES=0 python -u tools/mixed_precision_training_benchmark.py
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


def train_single_precision(precision, model_size, num_steps, lr=2e-3):
    """Train with a specific precision mode."""
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    np.random.seed(42)

    # Model config
    hd = 64 if model_size == '76k' else (256 if model_size == '2.28m' else 512)
    nl = 2 if model_size == '76k' else (4 if model_size == '2.28m' else 8)
    nh = 4 if model_size == '76k' else (8 if model_size == '2.28m' else 16)
    nkv = 2 if model_size == '76k' else (4 if model_size == '2.28m' else 8)

    # Set autocast dtype and GradScaler based on precision
    use_amp = False
    amp_dtype = torch.float32
    use_scaler = False

    if precision == 'fp32':
        dtype = torch.float32
    elif precision == 'fp16_amp':
        dtype = torch.float32  # model created in FP32, AMP casts during forward
        use_amp = True
        amp_dtype = torch.float16
        use_scaler = True
    elif precision == 'bf16':
        dtype = torch.bfloat16
        # BF16: no GradScaler needed (wider dynamic range)
    elif precision == 'bf16_amp':
        dtype = torch.float32
        use_amp = True
        amp_dtype = torch.bfloat16
        # BF16 AMP: no GradScaler needed
    elif precision == 'fp8_e4m3':
        # FP8 training: experimental, use FP8 for weight storage + compute where possible
        dtype = torch.bfloat16  # fallback — RTX 4090 SM89 doesn't support FP8 GEMM training
        print(f"  WARNING: FP8 training not supported on SM89 (RTX 4090)")
        print(f"  Using BF16 as fallback — FP8 only works for inference via cuBLAS")
    else:
        raise ValueError(f"Unknown precision: {precision}")

    model = MiniGQATransformer(
        hidden_dim=hd, num_layers=nl, num_heads=nh,
        num_kv_heads=nkv, vocab_size=VOCAB_SIZE).to(device=device, dtype=dtype)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda') if use_scaler else None
    dataset = generate_sft_dataset(500)

    # Memory measurement
    torch.cuda.reset_peak_memory_stats()
    peak_mem_start = torch.cuda.max_memory_allocated() / 1e9

    losses = []
    grad_norms = []
    start_time = time.time()

    for step in range(num_steps):
        idx = np.random.randint(len(dataset))
        full_tokens, prompt_len = dataset[idx]
        input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
        targets = input_ids.clone()

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, prompt_len-1:-1, :].reshape(-1, VOCAB_SIZE),
                    targets[:, prompt_len:].reshape(-1))

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits[:, prompt_len-1:-1, :].reshape(-1, VOCAB_SIZE),
                targets[:, prompt_len:].reshape(-1))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        losses.append(loss.item())
        grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

    elapsed = time.time() - start_time
    throughput = num_steps / elapsed
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Eval (always in FP32 for fair comparison)
    model.eval()
    if dtype != torch.float32:
        model_fp32 = model.float()
    else:
        model_fp32 = model

    correct = 0
    for _ in range(100):
        prompt_tokens, correct_sum = generate_arithmetic_prompt()
        input_ids = torch.tensor([prompt_tokens + [TOKENS['<eos>']]],
                                  dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model_fp32(input_ids)[0, -2, :].argmax().item()
        if pred == correct_sum:
            correct += 1
    eval_acc = correct / 100

    # Check for NaN/Inf in losses (training instability signal)
    has_nan = any(np.isnan(l) or np.isinf(l) for l in losses)
    loss_explosion = max(losses) > 100.0

    # Model weight health check
    param_stats = {}
    total_params = sum(p.numel() for p in model.parameters())
    nan_params = sum(1 for p in model.parameters() if torch.isnan(p).any())
    inf_params = sum(1 for p in model.parameters() if torch.isinf(p).any())
    max_param = max(p.abs().max().item() for p in model.parameters())

    result = {
        'precision': precision,
        'model_size': model_size,
        'num_steps': num_steps,
        'total_time_s': elapsed,
        'throughput_steps_s': throughput,
        'final_loss': losses[-1],
        'min_loss': min(losses),
        'max_loss': max(losses),
        'mean_loss': np.mean(losses),
        'loss_std': np.std(losses),
        'eval_accuracy': eval_acc,
        'peak_gpu_mem_gb': peak_mem,
        'has_nan': has_nan,
        'loss_explosion': loss_explosion,
        'mean_grad_norm': np.mean(grad_norms),
        'final_grad_norm': grad_norms[-1],
        'nan_params': nan_params,
        'inf_params': inf_params,
        'max_param_value': max_param,
        'total_params': total_params,
        'used_amp': use_amp,
        'used_scaler': use_scaler,
        'model_dtype': str(dtype),
    }

    print(f"  {precision}: {throughput:.1f} steps/s, "
          f"loss={losses[-1]:.3f}, eval={eval_acc:.1%}, "
          f"mem={peak_mem:.3f}GB, "
          f"grad_norm={grad_norms[-1]:.3f}, "
          f"nan={has_nan}, explosion={loss_explosion}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--model_size', default='76k',
                        choices=['76k', '2.28m'])
    parser.add_argument('--output', default='mixed_precision_training_results.json')
    args = parser.parse_args()

    print("=" * 70)
    print("Mixed Precision Training Benchmark — RTX 4090")
    print("=" * 70)

    # Print GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}, {gpu_mem:.1f}GB")
        print(f"CUDA: {torch.version.cuda}")
        # Check SM version for FP8 support
        sm = torch.cuda.get_device_capability(0)
        print(f"SM: {sm[0]}.{sm[1]} (FP8 GEMM requires SM 9.0+)")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")

    results = {}
    precisions = ['fp32', 'fp16_amp', 'bf16', 'bf16_amp']
    # FP8 only on H100/SM90+, not RTX 4090 SM89

    print(f"\n--- Model: {args.model_size} ---")
    for prec in precisions:
        print(f"\n[{prec}]")
        result = train_single_precision(prec, args.model_size, args.num_steps)
        key = f'{prec}_{args.model_size}'
        results[key] = result

    # Summary comparison
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    fp32_result = results[f'fp32_{args.model_size}']
    fp32_throughput = fp32_result['throughput_steps_s']
    fp32_eval = fp32_result['eval_accuracy']

    for prec in precisions:
        r = results[f'{prec}_{args.model_size}']
        speedup = r['throughput_steps_s'] / fp32_throughput
        eval_diff = r['eval_accuracy'] - fp32_eval
        loss_diff = r['final_loss'] - fp32_result['final_loss']
        print(f"  {prec}: speedup={speedup:.2f}x, "
              f"eval_diff={eval_diff:+.1%}, "
              f"loss_diff={loss_diff:+.3f}, "
              f"mem={r['peak_gpu_mem_gb']:.3f}GB, "
              f"stable={'YES' if not r['has_nan'] and not r['loss_explosion'] else 'NO'}")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()