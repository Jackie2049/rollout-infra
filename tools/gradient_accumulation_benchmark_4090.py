#!/usr/bin/env python3
"""Gradient Accumulation Benchmark — OPT-125M on RTX 4090

Benchmarks gradient accumulation vs true large batch:
1. True batch sizes (B=4,8,16,32,64) — step time, memory, throughput
2. Gradient accumulation (B=4, accum=2,4,8,16) — simulates B=8,16,32,64
3. Comparison: real B vs accum-equivalent B — overhead analysis
4. Memory scaling: B vs accum — which saves more memory?

Key question: Is gradient accumulation free? Or does it add overhead?
Expected: Accumulation ≈ free (no extra forward/backward, just delayed optimizer)
"""

import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_PATH = "/home/zxw/rollout-infra/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6"

def create_data(tokenizer, n, seq_len, device):
    texts = [f"Finding patterns in nature reveals mathematical truths. {i} squared is {i*i}." for i in range(n)]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=seq_len, return_tensors="pt")
    return enc['input_ids'].to(device)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    DEVICE = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    seq_len = 64
    input_pool = create_data(tokenizer, 128, seq_len, DEVICE)

    results = {'metadata': {'model': 'OPT-125M', 'gpu': 'RTX 4090', 'dtype': 'BF16', 'seq_len': seq_len}}

    # ==========================================
    # Exp1: True Batch Size Sweep
    # ==========================================
    print("=== Exp1: True Batch Size Sweep ===")
    for bs in [4, 8, 16, 32, 64]:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE,
            local_files_only=True, weights_only=False,
        )
        model.train()
        lora_config = LoraConfig(r=8, lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        ids = input_pool[:bs]
        labels = ids.clone()

        # Warmup
        for _ in range(2):
            optimizer.zero_grad()
            model(input_ids=ids, labels=labels).loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)

        # Benchmark
        n_steps = 15
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_loss = 0
        for _ in range(n_steps):
            optimizer.zero_grad()
            loss = model(input_ids=ids, labels=labels).loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        step_time = elapsed / n_steps
        throughput = bs * seq_len / step_time
        mem = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024

        results[f'exp1_b{bs}'] = {
            'batch_size': bs, 'accum_steps': 1, 'effective_batch': bs,
            'step_time_s': round(step_time, 4),
            'throughput_tok_s': round(throughput, 1),
            'mem_peak_mb': round(mem, 1),
            'avg_loss': round(total_loss / n_steps, 4),
        }
        print(f"  True B={bs}: step={step_time:.4f}s, throughput={throughput:.0f} tok/s, mem={mem:.0f}MB")

        del model, optimizer
        torch.cuda.empty_cache()

    # ==========================================
    # Exp2: Gradient Accumulation (B=4, accum=N)
    # ==========================================
    print("\n=== Exp2: Gradient Accumulation (B=4 base) ===")
    base_bs = 4
    for accum in [2, 4, 8, 16]:
        effective_bs = base_bs * accum
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE,
            local_files_only=True, weights_only=False,
        )
        model.train()
        lora_config = LoraConfig(r=8, lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Warmup
        for _ in range(2):
            optimizer.zero_grad()
            for micro in range(accum):
                start = micro * base_bs
                ids = input_pool[start:start + base_bs]
                labels = ids.clone()
                loss = model(input_ids=ids, labels=labels).loss
                loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)

        # Benchmark — measure full optimizer step (accum micro-batches + 1 optimizer step)
        n_opt_steps = 10
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_loss = 0
        for _ in range(n_opt_steps):
            optimizer.zero_grad()
            for micro in range(accum):
                start = micro * base_bs
                ids = input_pool[start:start + base_bs]
                labels = ids.clone()
                loss = model(input_ids=ids, labels=labels).loss
                total_loss += loss.item()
                loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        opt_step_time = elapsed / n_opt_steps
        # Effective throughput = effective_batch * seq_len / opt_step_time
        throughput = effective_bs * seq_len / opt_step_time
        mem = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024

        results[f'exp2_accum{accum}'] = {
            'batch_size': base_bs, 'accum_steps': accum, 'effective_batch': effective_bs,
            'opt_step_time_s': round(opt_step_time, 4),
            'throughput_tok_s': round(throughput, 1),
            'mem_peak_mb': round(mem, 1),
            'avg_loss': round(total_loss / (n_opt_steps * accum), 4),
            'n_forward_backward': accum,
        }
        print(f"  Accum={accum} (eff B={effective_bs}): opt_step={opt_step_time:.4f}s, throughput={throughput:.0f} tok/s, mem={mem:.0f}MB")

        del model, optimizer
        torch.cuda.empty_cache()

    # ==========================================
    # Exp3: Compare Real B vs Accum-equivalent
    # ==========================================
    print("\n=== Exp3: Real vs Accum Comparison ===")
    comparisons = [(8, 8, 1), (8, 4, 2), (16, 4, 4), (32, 4, 8), (64, 4, 16)]
    for eff_bs, real_bs, accum in comparisons:
        real_key = f'exp1_b{eff_bs}'
        accum_key = f'exp2_accum{accum}'
        if real_key in results and accum_key in results:
            real = results[real_key]
            accum = results[accum_key]

            # Throughput comparison
            real_throughput = real['throughput_tok_s']
            accum_throughput = accum['throughput_tok_s']
            throughput_ratio = accum_throughput / real_throughput

            # Memory comparison
            real_mem = real['mem_peak_mb']
            accum_mem = accum['mem_peak_mb']
            mem_ratio = accum_mem / real_mem

            print(f"  Eff B={eff_bs}: Real throughput={real_throughput:.0f}, Accum={accum_throughput:.0f}, ratio={throughput_ratio:.2f}x; Mem: Real={real_mem:.0f}, Accum={accum_mem:.0f}, ratio={mem_ratio:.2f}x")

    # ==========================================
    # Exp4: Full vs LoRA Memory with Accumulation
    # ==========================================
    print("\n=== Exp4: Full FT vs LoRA Memory at Different Effective Batch ===")
    for eff_bs in [8, 16, 32]:
        # Full FT — real batch
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE,
            local_files_only=True, weights_only=False,
        )
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ids = input_pool[:eff_bs]
        labels = ids.clone()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        optimizer.zero_grad()
        model(input_ids=ids, labels=labels).loss.backward()
        optimizer.step()
        full_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024

        del model, optimizer
        torch.cuda.empty_cache()

        # LoRA — real batch
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE,
            local_files_only=True, weights_only=False,
        )
        model.train()
        lora_config = LoraConfig(r=8, lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        optimizer.zero_grad()
        model(input_ids=ids, labels=labels).loss.backward()
        optimizer.step()
        lora_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024

        # LoRA + accum (B=4, accum=eff_bs/4)
        base = 4
        accum_steps = eff_bs // base
        del model, optimizer
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE,
            local_files_only=True, weights_only=False,
        )
        model.train()
        lora_config = LoraConfig(r=8, lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        optimizer.zero_grad()
        for micro in range(accum_steps):
            ids = input_pool[micro*base:(micro+1)*base]
            labels = ids.clone()
            model(input_ids=ids, labels=labels).loss.backward()
        optimizer.step()
        lora_accum_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024

        results[f'exp4_eff{eff_bs}'] = {
            'effective_batch': eff_bs,
            'full_ft_mem_mb': round(full_mem, 1),
            'lora_real_mem_mb': round(lora_mem, 1),
            'lora_accum_mem_mb': round(lora_accum_mem, 1),
            'full_vs_lora_saving': round((full_mem - lora_mem) / full_mem * 100, 1),
            'lora_real_vs_accum_saving': round((lora_mem - lora_accum_mem) / lora_mem * 100, 1),
        }
        print(f"  Eff B={eff_bs}: Full={full_mem:.0f}MB, LoRA={lora_mem:.0f}MB(saving {(full_mem-lora_mem)/full_mem*100:.1f}%), LoRA+accum={lora_accum_mem:.0f}MB(saving {(lora_mem-lora_accum_mem)/lora_mem*100:.1f}%)")

        del model, optimizer
        torch.cuda.empty_cache()

    # Save
    with open('gradient_accumulation_125m_4090.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to gradient_accumulation_125m_4090.json")

if __name__ == '__main__':
    main()