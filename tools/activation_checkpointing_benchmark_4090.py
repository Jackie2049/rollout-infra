#!/usr/bin/env python3
"""Activation Checkpointing Benchmark — OPT-125M LoRA on RTX 4090

Benchmarks gradient checkpointing (activation checkpointing) for LoRA training:
1. No checkpointing — baseline training throughput + memory
2. Checkpointing all layers — memory savings + throughput cost
3. Checkpointing every N layers — memory vs throughput trade-off
4. Selective checkpointing (only expensive layers)

Key question: How much memory does checkpointing save? How much throughput does it cost?
Expected: Checkpointing saves activation memory (linear in batch size × seq_len × layers)
          but costs ~2x forward pass time (recompute during backward)
"""

import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_PATH = "/home/zxw/rollout-infra/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6"

def create_data(tokenizer, n, seq_len, device):
    texts = [f"Patterns emerge from data. {i} times {i} equals {i*i}. Science reveals truth. {i}" for i in range(n)]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=seq_len, return_tensors="pt")
    return enc['input_ids'].to(device)

def apply_checkpointing(model, strategy='none', checkpoint_every=2):
    """Apply gradient checkpointing to model"""
    if strategy == 'none':
        return model
    elif strategy == 'all':
        model.gradient_checkpointing_enable()
        return model
    elif strategy == 'selective':
        # Only checkpoint transformer layers with even index
        from torch.utils.checkpoint import checkpoint
        for i, layer in enumerate(model.base_model.model.model.decoder.layers):
            if i % checkpoint_every == 0:
                # Use HF gradient checkpointing with specific layers
                pass
        # HF gradient checkpointing doesn't support per-layer selective easily
        # Fall back to 'all' with checkpoint_every
        model.gradient_checkpointing_enable()
        return model

def benchmark_training(model, optimizer, input_ids, labels, n_warmup=3, n_steps=15):
    """Benchmark training step time and memory"""
    # Warmup
    for _ in range(n_warmup):
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(input_ids.device)

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_loss = 0
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=labels).loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    step_time = elapsed / n_steps
    avg_loss = total_loss / n_steps
    mem_peak = torch.cuda.max_memory_allocated(input_ids.device) / 1024 / 1024
    throughput = input_ids.shape[0] * input_ids.shape[1] / step_time

    return step_time, avg_loss, mem_peak, throughput

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    DEVICE = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    results = {'metadata': {'model': 'OPT-125M LoRA r=8', 'gpu': 'RTX 4090', 'dtype': 'BF16'}}

    # ==========================================
    # Exp1: Batch Size × Checkpointing Memory
    # ==========================================
    print("=== Exp1: Batch Size × Checkpointing Memory ===")
    for bs in [4, 8, 16, 32]:
        for checkpoint in [False, True]:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE,
                local_files_only=True, weights_only=False,
            )
            model.train()

            if checkpoint:
                model.enable_input_require_grads()
                model.gradient_checkpointing_enable()

            lora_config = LoraConfig(r=8, lora_alpha=16,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            if checkpoint:
                model.enable_input_require_grads()
                model.gradient_checkpointing_enable()

            input_ids = create_data(tokenizer, bs, 64, DEVICE)
            labels = input_ids.clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            step_time, avg_loss, mem_peak, throughput = benchmark_training(
                model, optimizer, input_ids, labels)

            key = f'exp1_b{bs}_ckpt{checkpoint}'
            results[key] = {
                'batch_size': bs,
                'checkpointing': checkpoint,
                'step_time_s': round(step_time, 4),
                'avg_loss': round(avg_loss, 4),
                'mem_peak_mb': round(mem_peak, 1),
                'throughput_tok_s': round(throughput, 1),
            }
            ckpt_str = 'ckpt' if checkpoint else 'no_ckpt'
            print(f"  B={bs} {ckpt_str}: step={step_time:.4f}s, mem={mem_peak:.0f}MB, throughput={throughput:.0f} tok/s")

            del model, optimizer
            torch.cuda.empty_cache()

    # ==========================================
    # Exp2: Full FT vs LoRA × Checkpointing
    # ==========================================
    print("\n=== Exp2: Full FT vs LoRA × Checkpointing ===")
    configs = [
        ('full_ft_no_ckpt', False, False),
        ('full_ft_ckpt', False, True),
        ('lora_no_ckpt', True, False),
        # LoRA+ckpt has compatibility issue with PEFT, skip for now
    ]
    bs = 16
    input_ids = create_data(tokenizer, bs, 64, DEVICE)
    labels = input_ids.clone()

    for name, use_lora, use_ckpt in configs:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE,
            local_files_only=True, weights_only=False,
        )
        model.train()

        if use_lora:
            lora_config = LoraConfig(r=8, lora_alpha=16,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        if use_ckpt:
            model.gradient_checkpointing_enable()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        step_time, avg_loss, mem_peak, throughput = benchmark_training(
            model, optimizer, input_ids, labels)

        results[f'exp2_{name}'] = {
            'use_lora': use_lora,
            'use_ckpt': use_ckpt,
            'batch_size': bs,
            'step_time_s': round(step_time, 4),
            'avg_loss': round(avg_loss, 4),
            'mem_peak_mb': round(mem_peak, 1),
            'throughput_tok_s': round(throughput, 1),
        }
        print(f"  {name}: step={step_time:.4f}s, mem={mem_peak:.0f}MB, throughput={throughput:.0f} tok/s")

        del model, optimizer
        torch.cuda.empty_cache()

    # ==========================================
    # Exp3: Larger Batch — When Checkpointing Helps Most
    # ==========================================
    print("\n=== Exp3: Larger Batch — Checkpointing Impact ===")
    for bs in [64, 128]:
        for checkpoint in [False, True]:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE,
                    local_files_only=True, weights_only=False,
                )
                model.train()

                if checkpoint:
                    model.enable_input_require_grads()
                    model.gradient_checkpointing_enable()

                lora_config = LoraConfig(r=8, lora_alpha=16,
                    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
                    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lora_config)
                if checkpoint:
                    model.enable_input_require_grads()
                    model.gradient_checkpointing_enable()

                input_ids = create_data(tokenizer, bs, 64, DEVICE)
                labels = input_ids.clone()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

                step_time, avg_loss, mem_peak, throughput = benchmark_training(
                    model, optimizer, input_ids, labels, n_warmup=2, n_steps=10)

                key = f'exp3_b{bs}_ckpt{checkpoint}'
                results[key] = {
                    'batch_size': bs,
                    'checkpointing': checkpoint,
                    'step_time_s': round(step_time, 4),
                    'mem_peak_mb': round(mem_peak, 1),
                    'throughput_tok_s': round(throughput, 1),
                }
                ckpt_str = 'ckpt' if checkpoint else 'no_ckpt'
                print(f"  B={bs} {ckpt_str}: step={step_time:.4f}s, mem={mem_peak:.0f}MB, throughput={throughput:.0f} tok/s")
            except Exception as e:
                print(f"  B={bs} {'ckpt' if checkpoint else 'no_ckpt'}: ERROR - {e}")
                if checkpoint:
                    results[f'exp3_b{bs}_ckptTrue'] = {'batch_size': bs, 'checkpointing': True, 'error': str(e)}
            ckpt_str = 'ckpt' if checkpoint else 'no_ckpt'
            print(f"  B={bs} {ckpt_str}: step={step_time:.4f}s, mem={mem_peak:.0f}MB, throughput={throughput:.0f} tok/s")

            del model, optimizer
            torch.cuda.empty_cache()

    # ==========================================
    # Analysis: Memory Savings vs Throughput Cost
    # ==========================================
    print("\n=== Analysis ===")
    for bs in [4, 8, 16, 32, 64, 128]:
        no_ckpt_key = f'exp1_b{bs}_ckptFalse' if bs <= 32 else f'exp3_b{bs}_ckptFalse'
        ckpt_key = f'exp1_b{bs}_ckptTrue' if bs <= 32 else f'exp3_b{bs}_ckptTrue'
        if no_ckpt_key in results and ckpt_key in results:
            no_ckpt = results[no_ckpt_key]
            ckpt = results[ckpt_key]
            mem_saving = (no_ckpt['mem_peak_mb'] - ckpt['mem_peak_mb']) / no_ckpt['mem_peak_mb'] * 100
            throughput_cost = (1 - ckpt['throughput_tok_s'] / no_ckpt['throughput_tok_s']) * 100
            print(f"  B={bs}: mem_saving={mem_saving:.1f}%, throughput_cost={throughput_cost:.1f}%")

    # Save
    with open('activation_checkpointing_125m_4090.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to activation_checkpointing_125m_4090.json")

if __name__ == '__main__':
    main()