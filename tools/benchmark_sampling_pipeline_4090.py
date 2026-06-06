#!/usr/bin/env python3
"""Sampling Pipeline Benchmark on RTX 4090
============================================

Measures timing breakdown of each step in the LLM decode sampling pipeline.
Simulates the 9-step pipeline from logits to final token selection.

Usage:
  python benchmark_sampling_pipeline_4090.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import time
import json

def main():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print(f"Sampling Pipeline Benchmark: {torch.cuda.get_device_name(device)}")

    # Model configs to test
    configs = [
        ("7B vocab", 32000, 4096),   # LLaMA-7B vocab=32K, hidden=4096
        ("13B vocab", 32000, 5120),  # LLaMA-13B
        ("70B vocab", 32000, 8192),  # LLaMA-70B (hidden=8192)
    ]

    # Batch sizes
    batch_sizes = [1, 8, 32, 128, 256, 512]

    results = []

    for name, vocab_size, hidden_dim in configs:
        print(f"\n=== {name}: vocab={vocab_size}, hidden={hidden_dim} ===")

        for B in batch_sizes:
            # Simulate logits output from model (B, vocab_size)
            logits = torch.randn(B, vocab_size, device=device, dtype=torch.float16)

            # Step timings (all in ms)
            step_times = {}

            # Warmup
            for _ in range(10):
                _ = logits.float()
                _ = torch.argmax(logits, dim=-1)
            torch.cuda.synchronize()

            n_iters = 100
            torch.cuda.synchronize()

            # Step 1: FP16 → FP32 conversion
            t0 = time.perf_counter()
            for _ in range(n_iters):
                logits_fp32 = logits.float()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            step_times["fp16_to_fp32"] = (t1 - t0) * 1000 / n_iters

            # Step 2: Temperature scaling
            temperatures = [1.0, 0.5, 0.1, 0.01]  # different temperatures
            temp_results = {}
            for temp in temperatures:
                t0 = time.perf_counter()
                for _ in range(n_iters):
                    scaled = logits_fp32 / temp
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                temp_ms = (t1 - t0) * 1000 / n_iters
                temp_results[f"temp_{temp}"] = round(temp_ms, 3)
            step_times["temperature_scale"] = temp_results

            # Step 3: Top-k filtering
            top_k_values = [1, 10, 50, 100, 500, 1000]
            topk_results = {}
            for k in top_k_values:
                # Get top-k values and indices
                t0 = time.perf_counter()
                for _ in range(n_iters):
                    topk_vals, topk_idx = torch.topk(logits_fp32, k, dim=-1)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                topk_ms = (t1 - t0) * 1000 / n_iters
                topk_results[f"top_k_{k}"] = round(topk_ms, 3)
            step_times["top_k"] = topk_results

            # Step 4: Top-p (nucleus) filtering
            # Sort logits, compute cumulative prob, find cutoff
            t0 = time.perf_counter()
            for _ in range(n_iters):
                sorted_logits, sorted_indices = torch.sort(logits_fp32, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= 0.9
                # Shift to keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                logits_filtered = logits_fp32.masked_fill(indices_to_remove, float('-inf'))
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            step_times["top_p_0.9"] = round((t1 - t0) * 1000 / n_iters, 3)

            # Step 5: Softmax (for multinomial sampling)
            t0 = time.perf_counter()
            for _ in range(n_iters):
                probs = torch.softmax(logits_fp32, dim=-1)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            step_times["softmax"] = round((t1 - t0) * 1000 / n_iters, 3)

            # Step 6: Argmax sampling (greedy)
            t0 = time.perf_counter()
            for _ in range(n_iters):
                greedy_tokens = torch.argmax(logits_fp32, dim=-1)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            step_times["argmax"] = round((t1 - t0) * 1000 / n_iters, 3)

            # Step 7: Multinomial sampling
            t0 = time.perf_counter()
            for _ in range(n_iters):
                sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            step_times["multinomial"] = round((t1 - t0) * 1000 / n_iters, 3)

            # Step 8: Full pipeline: logits → fp32 → temperature → top-k → softmax → sample
            # (simulating the most common decode path)
            t0 = time.perf_counter()
            for _ in range(n_iters):
                lf = logits.float()
                lf = lf / 1.0  # temperature=1.0
                tv, ti = torch.topk(lf, 100, dim=-1)  # top-k=100
                p = torch.softmax(tv, dim=-1)
                tok = torch.multinomial(p, 1).squeeze(-1)
                final = ti.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            step_times["full_pipeline_top100"] = round((t1 - t0) * 1000 / n_iters, 3)

            # Step 9: Full pipeline with temperature=0 (greedy path)
            t0 = time.perf_counter()
            for _ in range(n_iters):
                lf = logits.float()
                final = torch.argmax(lf, dim=-1)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            step_times["greedy_pipeline"] = round((t1 - t0) * 1000 / n_iters, 3)

            if B == 1:
                print(f"  B={B}: fp32={step_times['fp16_to_fp32']:.3f}ms, "
                      f"softmax={step_times['softmax']:.3f}ms, "
                      f"argmax={step_times['argmax']:.3f}ms, "
                      f"multinomial={step_times['multinomial']:.3f}ms, "
                      f"full_top100={step_times['full_pipeline_top100']:.3f}ms, "
                      f"greedy={step_times['greedy_pipeline']:.3f}ms")
            elif B <= 32:
                print(f"  B={B}: full_top100={step_times['full_pipeline_top100']:.3f}ms, "
                      f"greedy={step_times['greedy_pipeline']:.3f}ms")
            else:
                print(f"  B={B}: full_top100={step_times['full_pipeline_top100']:.3f}ms, "
                      f"greedy={step_times['greedy_pipeline']:.3f}ms")

            results.append({
                "name": name,
                "vocab_size": vocab_size,
                "hidden_dim": hidden_dim,
                "batch_size": B,
                "step_times": step_times,
            })

    # Save results
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'sampling_pipeline_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n=== Summary: 7B model, most important batch sizes ===")
    for r in results:
        if r["name"] == "7B vocab" and r["batch_size"] in [1, 32, 128, 512]:
            st = r["step_times"]
            print(f"  B={r['batch_size']}: "
                  f"fp16→fp32={st['fp16_to_fp32']:.3f}ms, "
                  f"softmax={st['softmax']:.3f}ms, "
                  f"argmax={st['argmax']:.3f}ms, "
                  f"multinomial={st['multinomial']:.3f}ms, "
                  f"full_top100={st['full_pipeline_top100']:.3f}ms")

if __name__ == '__main__':
    main()