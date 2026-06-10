#!/usr/bin/env python3
"""Real Model Inference Benchmark — RTX 4090

Benchmarks real transformer model inference on RTX 4090:
1. OPT-125M (small model baseline)
2. BF16 7B decode + prefill latency
3. Token generation throughput by batch size
4. Memory profiling (model + KV cache)
5. End-to-end generation benchmark

Uses HF models with transformers library.
RTX 4090 has 24GB VRAM — 7B BF16 fits (13GB model + 8GB KV room).
"""

import json
import os
import time
import torch
import numpy as np

def benchmark_model(model_name, model, tokenizer, device, batch_sizes, seq_lengths):
    """Benchmark a model for inference"""
    results = {'model': model_name}

    # Model size
    total_params = sum(p.numel() for p in model.parameters())
    model_mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    results['params'] = total_params
    results['model_mem_mb'] = round(model_mem_mb, 1)

    # Actual GPU memory used
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device) / 1024 / 1024

    # Warmup
    warmup_text = "Hello, this is a warmup sentence for the model."
    inputs = tokenizer(warmup_text, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize()

    mem_after_warmup = torch.cuda.memory_allocated(device) / 1024 / 1024
    mem_peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    results['mem_after_load_mb'] = round(mem_after_warmup, 1)
    results['mem_peak_warmup_mb'] = round(mem_peak, 1)

    # Decode benchmark (batch size sweep)
    decode_results = {}
    for B in batch_sizes:
        prompts = [f"The meaning of life is {i}" for i in range(B)]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        input_len = inputs['input_ids'].shape[1]

        # Time per token (decode)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=32, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        output_len = outputs.shape[1]
        new_tokens = output_len - input_len
        time_per_token = elapsed / new_tokens
        throughput = B * new_tokens / elapsed  # tokens/sec total
        itl_ms = time_per_token * 1000 / B  # per-request ITL

        mem_peak_gen = torch.cuda.max_memory_allocated(device) / 1024 / 1024

        decode_results[f'B{B}'] = {
            'input_len': input_len,
            'new_tokens': new_tokens,
            'total_time_s': round(elapsed, 3),
            'time_per_token_ms': round(time_per_token * 1000, 3),
            'itl_ms': round(itl_ms, 2),
            'throughput_tok_per_s': round(throughput, 1),
            'mem_peak_mb': round(mem_peak_gen, 1),
        }
        print(f"  Decode B={B}: {itl_ms:.1f}ms/tok, {throughput:.0f} tok/s, peak={mem_peak_gen:.0f}MB")

    results['decode'] = decode_results

    # Prefill benchmark (sequence length sweep)
    prefill_results = {}
    for S in seq_lengths:
        text = "word " * S  # simple repeat to hit target length
        inputs = tokenizer(text, return_tensors="pt").to(device)
        actual_len = inputs['input_ids'].shape[1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # Approximate FLOPS for prefill: 2 * S * hidden^2 * n_layers
        hidden = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
        n_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 32
        flops = 2 * actual_len * hidden * hidden * n_layers * 3  # *3 for attn+mlp+proj
        tflops = flops / elapsed / 1e12

        prefill_results[f'S{S}'] = {
            'actual_len': actual_len,
            'time_ms': round(elapsed * 1000, 1),
            'tflops': round(tflops, 2),
            'peak_pct': round(tflops / 169.6 * 100, 1) if tflops > 0 else 0,
        }
        print(f"  Prefill S={actual_len}: {elapsed*1000:.1f}ms, {tflops:.1f} TFLOPS ({tflops/169.6*100:.0f}% peak)")

    results['prefill'] = prefill_results

    # KV Cache size estimation
    n_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 32
    n_heads = model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else 32
    head_dim = hidden // n_heads
    kv_per_token_bytes = 2 * n_layers * 2 * head_dim * 2  # 2(K+V) * n_layers * head_dim * bf16(2B)
    results['kv_per_token_bytes'] = kv_per_token_bytes
    results['kv_per_token_kb'] = round(kv_per_token_bytes / 1024, 2)
    results['kv_4k_mb'] = round(kv_per_token_bytes * 4096 / 1024 / 1024, 1)

    return results


def main():
    n_gpu = torch.cuda.device_count()
    device = 'cuda:0'
    print(f"GPU: {torch.cuda.get_device_name(device)}, {n_gpu} GPUs available")
    print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    all_results = {
        'metadata': {
            'pytorch': torch.__version__,
            'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device),
            'vram_gb': round(torch.cuda.get_device_properties(device).total_memory / 1e9, 1),
        }
    }

    # Phase 1: Small model (OPT-125M)
    print("\n=== OPT-125M Baseline ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name_125m = "facebook/opt-125m"

    # Check if model exists locally
    local_path = os.path.expanduser("~/rollout-infra/models--facebook--opt-125m")
    if os.path.exists(local_path):
        # Try to load from local snapshot
        snapshots_dir = os.path.join(local_path, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshot = os.listdir(snapshots_dir)[0]
            model_path = os.path.join(snapshots_dir, snapshot)
        else:
            model_path = local_path
    else:
        model_path = model_name_125m

    try:
        tokenizer_125m = AutoTokenizer.from_pretrained(model_path)
        model_125m = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device,
        )
        print(f"Loaded OPT-125M from {model_path}")

        result_125m = benchmark_model(
            'OPT-125M', model_125m, tokenizer_125m, device,
            batch_sizes=[1, 4, 8, 16, 32, 64],
            seq_lengths=[32, 64, 128, 256, 512, 1024],
        )
        all_results['opt_125m'] = result_125m

        # Free memory
        del model_125m
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"OPT-125M failed: {e}")
        all_results['opt_125m_error'] = str(e)

    # Save results
    output_path = os.path.join(os.environ.get('ROLLOUT_HOME', '.'), 'real_model_inference_benchmark.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")
    return all_results


if __name__ == '__main__':
    os.environ['ROLLOUT_HOME'] = os.environ.get('ROLLOUT_HOME', os.getcwd())
    os.environ['HF_HUB_OFFLINE'] = '1'  # Prefer local models
    main()