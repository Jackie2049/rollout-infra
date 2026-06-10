#!/usr/bin/env python3
"""LoRA Fine-tuning Benchmark — OPT-125M on RTX 4090

Validates "RTX 4090最优=LoRA单GPU训练" recommendation:
1. Full fine-tuning vs LoRA (r=4,8,16) training throughput
2. LoRA memory savings and merge overhead
3. Batch size scaling for LoRA training
4. Target module comparison (attn-only vs all-linear)
5. LoRA + gradient accumulation for larger effective batch
"""

import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

MODEL_PATH = "/home/zxw/rollout-infra/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6"
DEVICE = 'cuda:0'

def create_dummy_data(tokenizer, n_samples=256, seq_len=128):
    """Create synthetic training data"""
    texts = [f"The meaning of life is about finding purpose and joy. Mathematics teaches us that {i} + {i} = {2*i}." for i in range(n_samples)]
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=seq_len, return_tensors="pt")
    return encodings

def benchmark_step_time(model, optimizer, input_ids, labels, n_warmup=3, n_steps=20):
    """Benchmark training step time"""
    # Warmup
    for _ in range(n_warmup):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Reset peak memory after warmup
    torch.cuda.reset_peak_memory_stats(DEVICE)

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_loss = 0
    for _ in range(n_steps):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    step_time = total_time / n_steps
    avg_loss = total_loss / n_steps
    mem_peak = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024

    return step_time, avg_loss, mem_peak

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    DEVICE = 'cuda:0'  # remapped by CUDA_VISIBLE_DEVICES

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    encodings = create_dummy_data(tokenizer)
    input_ids = encodings['input_ids'].to(DEVICE)
    labels = input_ids.clone()

    batch_size = input_ids.shape[0]  # 256
    seq_len = input_ids.shape[1]

    results = {
        'metadata': {
            'model': 'OPT-125M',
            'gpu': 'RTX 4090 24GB',
            'dtype': 'BF16',
            'batch_size': batch_size,
            'seq_len': seq_len,
        }
    }

    # ==========================================
    # Experiment 1: Full Fine-tuning Baseline
    # ==========================================
    print("=== Exp1: Full Fine-tuning Baseline ===")
    model_full = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE,
        local_files_only=True, weights_only=False,
    )
    model_full.train()
    n_params_full = sum(p.numel() for p in model_full.parameters())
    n_trainable_full = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    optimizer_full = torch.optim.AdamW(model_full.parameters(), lr=1e-4, weight_decay=0.01)

    step_time, avg_loss, mem_peak = benchmark_step_time(model_full, optimizer_full, input_ids, labels)
    throughput = batch_size * seq_len / step_time

    results['exp1_full'] = {
        'step_time_s': round(step_time, 4),
        'avg_loss': round(avg_loss, 4),
        'throughput_tok_s': round(throughput, 1),
        'mem_peak_mb': round(mem_peak, 1),
        'n_params': n_params_full,
        'n_trainable': n_trainable_full,
        'trainable_pct': round(n_trainable_full / n_params_full * 100, 2),
    }
    print(f"  Full FT: step={step_time:.4f}s, loss={avg_loss:.4f}, throughput={throughput:.0f} tok/s, peak={mem_peak:.0f}MB")

    del model_full, optimizer_full
    torch.cuda.empty_cache()

    # ==========================================
    # Experiment 2: LoRA Rank Sweep
    # ==========================================
    print("\n=== Exp2: LoRA Rank Sweep ===")
    for rank in [2, 4, 8, 16, 32]:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE,
            local_files_only=True, weights_only=False,
        )
        model.train()

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,  # common: alpha = 2*r
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],  # all-linear
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

        step_time, avg_loss, mem_peak = benchmark_step_time(model, optimizer, input_ids, labels)
        throughput = batch_size * seq_len / step_time

        results[f'exp2_r{rank}'] = {
            'rank': rank,
            'alpha': rank * 2,
            'step_time_s': round(step_time, 4),
            'avg_loss': round(avg_loss, 4),
            'throughput_tok_s': round(throughput, 1),
            'mem_peak_mb': round(mem_peak, 1),
            'n_trainable': n_trainable,
            'trainable_pct': round(n_trainable / n_params * 100, 2),
        }
        print(f"  LoRA r={rank}: step={step_time:.4f}s, loss={avg_loss:.4f}, throughput={throughput:.0f} tok/s, peak={mem_peak:.0f}MB, trainable={n_trainable}")

        del model, optimizer
        torch.cuda.empty_cache()

    # ==========================================
    # Experiment 3: Target Module Comparison
    # ==========================================
    print("\n=== Exp3: Target Module Comparison ===")
    target_configs = {
        'attn_only': ["q_proj", "v_proj", "k_proj", "out_proj"],
        'ffn_only': ["fc1", "fc2"],
        'all_linear': ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
    }
    for name, targets in target_configs.items():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE,
            local_files_only=True, weights_only=False,
        )
        model.train()

        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=targets,
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

        step_time, avg_loss, mem_peak = benchmark_step_time(model, optimizer, input_ids, labels)

        results[f'exp3_{name}'] = {
            'target_modules': targets,
            'step_time_s': round(step_time, 4),
            'avg_loss': round(avg_loss, 4),
            'mem_peak_mb': round(mem_peak, 1),
            'n_trainable': n_trainable,
            'trainable_pct': round(n_trainable / sum(p.numel() for p in model.parameters()) * 100, 2),
        }
        print(f"  {name}: step={step_time:.4f}s, loss={avg_loss:.4f}, peak={mem_peak:.0f}MB, trainable={n_trainable}")

        del model, optimizer
        torch.cuda.empty_cache()

    # ==========================================
    # Experiment 4: LoRA Merge Overhead
    # ==========================================
    print("\n=== Exp4: LoRA Merge Overhead ===")
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE,
        local_files_only=True, weights_only=False,
    )
    model_base.eval()

    # Benchmark base model inference
    test_ids = input_ids[:4]  # B=4 for inference
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            _ = model_base(test_ids)
    torch.cuda.synchronize()
    base_time = (time.perf_counter() - t0) / 20

    # Create LoRA model and benchmark unmerged inference
    model_base.train()
    lora_config = LoraConfig(r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model_lora = get_peft_model(model_base, lora_config)
    model_lora.eval()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            _ = model_lora(test_ids)
    torch.cuda.synchronize()
    unmerged_time = (time.perf_counter() - t0) / 20

    # Merge LoRA weights and benchmark merged inference
    model_merged = model_lora.merge_and_unload()
    model_merged.eval()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            _ = model_merged(test_ids)
    torch.cuda.synchronize()
    merged_time = (time.perf_counter() - t0) / 20

    results['exp4'] = {
        'base_ms': round(base_time * 1000, 2),
        'unmerged_ms': round(unmerged_time * 1000, 2),
        'merged_ms': round(merged_time * 1000, 2),
        'merge_speedup': round(unmerged_time / merged_time, 2),
        'lora_overhead_vs_base': round(unmerged_time / base_time, 2),
    }
    print(f"  Base: {base_time*1000:.2f}ms, Unmerged: {unmerged_time*1000:.2f}ms, Merged: {merged_time*1000:.2f}ms, Merge speedup: {unmerged_time/merged_time:.2f}x")

    del model_lora, model_merged
    torch.cuda.empty_cache()

    # ==========================================
    # Experiment 5: Batch Size Scaling for LoRA
    # ==========================================
    print("\n=== Exp5: LoRA Batch Size Scaling ===")
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

        ids = input_ids[:bs]
        lbl = ids.clone()
        torch.cuda.reset_peak_memory_stats(DEVICE)

        # Warmup
        for _ in range(2):
            optimizer.zero_grad()
            model(ids, labels=lbl).loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)

        # Benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            optimizer.zero_grad()
            model(ids, labels=lbl).loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        step_time = (time.perf_counter() - t0) / 10
        mem_peak = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024
        throughput = bs * seq_len / step_time

        results[f'exp5_b{bs}'] = {
            'batch_size': bs,
            'step_time_s': round(step_time, 4),
            'throughput_tok_s': round(throughput, 1),
            'mem_peak_mb': round(mem_peak, 1),
        }
        print(f"  B={bs}: step={step_time:.4f}s, throughput={throughput:.0f} tok/s, peak={mem_peak:.0f}MB")

        del model, optimizer
        torch.cuda.empty_cache()

    # ==========================================
    # Experiment 6: LoRA Memory Savings Detail
    # ==========================================
    print("\n=== Exp6: LoRA Memory Detail ===")
    # Full fine-tuning memory
    model_full = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE,
        local_files_only=True, weights_only=False,
    )
    model_full.train()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    optimizer_full = torch.optim.AdamW(model_full.parameters(), lr=1e-4, weight_decay=0.01)
    optimizer_full.zero_grad()
    model_full(input_ids=input_ids[:8], labels=input_ids[:8].clone()).loss.backward()
    mem_full = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024

    del model_full, optimizer_full
    torch.cuda.empty_cache()

    # LoRA memory
    model_lora = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map=DEVICE,
        local_files_only=True, weights_only=False,
    )
    model_lora.train()
    lora_config = LoraConfig(r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model_lora = get_peft_model(model_lora, lora_config)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    optimizer_lora = torch.optim.AdamW(model_lora.parameters(), lr=1e-4, weight_decay=0.01)
    optimizer_lora.zero_grad()
    model_lora(input_ids=input_ids[:8], labels=input_ids[:8].clone()).loss.backward()
    mem_lora = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024

    results['exp6'] = {
        'full_mem_mb': round(mem_full, 1),
        'lora_mem_mb': round(mem_lora, 1),
        'saving_pct': round((mem_full - mem_lora) / mem_full * 100, 2),
        'saving_mb': round(mem_full - mem_lora, 1),
    }
    print(f"  Full: {mem_full:.0f}MB, LoRA: {mem_lora:.0f}MB, Saving: {(mem_full-mem_lora)/mem_full*100:.1f}%")

    del model_lora, optimizer_lora
    torch.cuda.empty_cache()

    # Save results
    with open('lora_125m_benchmark_4090.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to lora_125m_benchmark_4090.json")

if __name__ == '__main__':
    main()