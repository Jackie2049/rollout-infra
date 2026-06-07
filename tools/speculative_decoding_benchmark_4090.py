#!/usr/bin/env python3
"""Speculative Decoding Simulation Benchmark — RTX 4090
========================================================
Validates speculative decoding theory on real hardware.
Measures:
1. Draft model latency vs target model latency
2. Acceptance rate at different draft sizes
3. Speedup = 1 / (1 + draft_time/target_time - accepted_tokens/target_time)
4. Optimal K (number of draft tokens) at different acceptance rates
5. Draft model size vs acceptance rate tradeoff
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import math


class TinyDraftModel(nn.Module):
    """Smaller draft model for speculative decoding."""
    def __init__(self, vocab_size=32000, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, 4*d_model, dropout=0.0, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)


class TargetModel(nn.Module):
    """Target model (larger)."""
    def __init__(self, vocab_size=32000, d_model=256, n_heads=8, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, 4*d_model, dropout=0.0, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)


def benchmark_decode_latency(model, input_ids, n_runs=50, warmup=10):
    """Measure single-token decode latency."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)

        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    return sorted(times)[len(times) // 2]


def simulate_acceptance_rate(p_draft, p_target, K):
    """Simulate acceptance rate for speculative decoding.

    For each of K draft tokens, accept if draft prob matches target prob.
    Acceptance probability per token ≈ min(p_draft/p_target, 1) averaged.
    """
    # Simplified: acceptance rate depends on KL divergence between draft and target
    # Higher KL → lower acceptance rate
    # For a well-trained draft model: acceptance ≈ 0.8-0.9
    # For a poor draft model: acceptance ≈ 0.3-0.5

    # We simulate with actual softmax distributions
    torch.manual_seed(42)
    vocab_size = p_draft.shape[-1]

    accepted = 0
    for k in range(K):
        # Rejection sampling: accept token k if p_draft[k] ≤ p_target[k]
        # More precisely: accept with probability min(p_target[token]/p_draft[token], 1)
        draft_token = torch.argmax(p_draft, dim=-1)  # Draft model's choice
        p_d = p_draft[draft_token].item()
        p_t = p_target[draft_token].item()

        # Acceptance probability = min(p_t / p_d, 1.0)
        accept_prob = min(p_t / max(p_d, 1e-10), 1.0)

        if torch.rand(1).item() < accept_prob:
            accepted += 1
        else:
            break  # Rejection: stop and resample from adjusted distribution

    return accepted


def run_experiment():
    device = "cuda"
    vocab_size = 32000
    S = 64  # Prompt length for context
    results = {}

    # Create models
    torch.manual_seed(42)

    # Draft models of different sizes
    draft_configs = {
        "tiny_0.5M": {"d": 64, "heads": 2, "layers": 2},     # ~0.5M params
        "small_2M": {"d": 128, "heads": 4, "layers": 4},      # ~2M params
        "medium_7M": {"d": 256, "heads": 8, "layers": 4},      # ~7M params
    }

    # Target model (25M)
    target = TargetModel(vocab_size=vocab_size, d_model=256, n_heads=8, n_layers=4).to(device).to(torch.float16)
    target_params = sum(p.numel() for p in target.parameters())

    input_ids = torch.randint(0, vocab_size, (1, S), device=device)

    # Measure target model decode latency
    target_latency = benchmark_decode_latency(target, input_ids)
    print(f"Target model ({target_params:,} params): decode latency = {target_latency:.2f}ms")

    results["target"] = {
        "params": target_params, "decode_latency_ms": target_latency,
        "d_model": 256, "n_heads": 8, "n_layers": 4
    }

    # For each draft model
    for draft_name, cfg in draft_configs.items():
        torch.manual_seed(42)
        draft = TinyDraftModel(
            vocab_size=vocab_size, d_model=cfg["d"],
            n_heads=cfg["heads"], n_layers=cfg["layers"]
        ).to(device).to(torch.float16)
        draft_params = sum(p.numel() for p in draft.parameters())

        draft_latency = benchmark_decode_latency(draft, input_ids)
        draft_ratio = draft_params / target_params
        latency_ratio = draft_latency / target_latency

        print(f"\nDraft model '{draft_name}' ({draft_params:,} params): "
              f"decode latency = {draft_latency:.2f}ms "
              f"(param ratio={draft_ratio:.3f}, latency ratio={latency_ratio:.2f})")

        # Simulate speculative decoding with different K values
        seq_len_for_sim = S  # Use S (64) tokens for simulation
        torch.manual_seed(42)
        prompt_ids_long = torch.randint(0, vocab_size, (1, seq_len_for_sim), device=device)

        with torch.no_grad():
            target_logits = target(prompt_ids_long)
            draft_logits = draft(prompt_ids_long)

            target_probs = F.softmax(target_logits.float(), dim=-1)
            draft_probs = F.softmax(draft_logits.float(), dim=-1)

        # Calculate KL divergence (measure of distribution similarity)
        kl_div = F.kl_div(
            F.log_softmax(draft_logits.float(), dim=-1),
            F.softmax(target_logits.float(), dim=-1),
            reduction='batchmean'
        ).item()

        # Cosine similarity of logits
        cos_sim = F.cosine_similarity(
            target_logits.flatten().float(),
            draft_logits.flatten().float(), dim=0
        ).item()

        # Top-1 agreement rate (how often draft and target agree on top token)
        target_top1 = target_logits.argmax(dim=-1)
        draft_top1 = draft_logits.argmax(dim=-1)
        agreement_rate = (target_top1 == draft_top1).float().mean().item()

        # Simulate speculative decoding with different K values
        K_values = [1, 2, 3, 4, 5, 8]
        acceptance_results = {}

        for K in K_values:
            # Average acceptance over 100 trials
            n_accepted = []
            for trial in range(100):
                # Random token position
                pos = torch.randint(0, S-1, (1,)).item()
                t_probs = target_probs[0, pos]
                d_probs = draft_probs[0, pos]

                accepted = 0
                for k in range(K):
                    draft_token = d_probs.argmax().item()
                    p_d = d_probs[draft_token].item()
                    p_t = t_probs[draft_token].item()
                    accept_prob = min(p_t / max(p_d, 1e-10), 1.0)

                    if torch.rand(1).item() < accept_prob:
                        accepted += 1
                        # Move to next position
                        pos = min(pos + 1, S - 1)
                        t_probs = target_probs[0, pos]
                        d_probs = draft_probs[0, pos]
                    else:
                        break

                n_accepted.append(accepted)

            avg_accepted = sum(n_accepted) / len(n_accepted)
            acceptance_rate = avg_accepted / K if K > 0 else 0

            # Speedup formula:
            # speculative_time = draft_time * K + target_time * 1
            # tokens_produced = accepted_tokens + 1 (first verified) + 1 (resampled if rejected)
            # For accepted = α*K:
            #   tokens_per_step = α*K + 1
            #   time_per_step = K * draft_latency + target_latency
            #   speedup = (α*K + 1) * target_latency / (K * draft_latency + target_latency)

            tokens_per_step = avg_accepted + 1
            time_per_step = K * draft_latency + target_latency
            naive_time = (avg_accepted + 1) * target_latency

            speedup = naive_time / time_per_step if time_per_step > 0 else 0

            acceptance_results[f"K{K}"] = {
                "K": K,
                "avg_accepted": avg_accepted,
                "acceptance_rate": acceptance_rate,
                "tokens_per_step": tokens_per_step,
                "time_per_step_ms": time_per_step,
                "naive_time_ms": naive_time,
                "speedup": speedup,
                "speedup_formula": (avg_accepted + 1) * target_latency / (K * draft_latency + target_latency) if (K * draft_latency + target_latency) > 0 else 0,
            }

            print(f"  K={K}: avg_accepted={avg_accepted:.1f}, "
                  f"acceptance_rate={acceptance_rate:.1%}, "
                  f"speedup={speedup:.2f}x")

        results[draft_name] = {
            "params": draft_params,
            "draft_latency_ms": draft_latency,
            "param_ratio": draft_ratio,
            "latency_ratio": latency_ratio,
            "kl_divergence": kl_div,
            "cos_sim_logits": cos_sim,
            "agreement_rate_top1": agreement_rate,
            "acceptance_by_K": acceptance_results,
        }

    # Theoretical speedup analysis
    print("\n=== Theoretical Speedup Analysis ===")
    print("Speedup = (α*K + 1) × T_target / (K × T_draft + T_target)")
    print("  α = acceptance rate")
    print("  K = number of draft tokens")
    print("  T_target = target model decode latency")
    print("  T_draft = draft model decode latency")

    # Optimal K for different acceptance rates and latency ratios
    for alpha in [0.5, 0.8, 0.9, 0.95]:
        for lr in [0.1, 0.2, 0.3, 0.5]:
            best_K = 1
            best_speedup = 0
            for K in range(1, 20):
                speedup = (alpha * K + 1) / (K * lr + 1)
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_K = K

            results[f"optimal_alpha{alpha}_lr{lr}"] = {
                "alpha": alpha, "latency_ratio": lr,
                "optimal_K": best_K, "max_speedup": best_speedup
            }
            print(f"  α={alpha}, latency_ratio={lr}: optimal K={best_K}, speedup={best_speedup:.2f}x")

    # Save
    with open("results/speculative_decoding_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/speculative_decoding_benchmark.json")

    return results


if __name__ == "__main__":
    run_experiment()