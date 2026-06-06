#!/usr/bin/env python3
"""Batched GEMM MoE Compute Benchmark on RTX 4090
=============================================

Benchmarks the core compute pattern in MoE inference:
1. Grouped/Batched GEMM — tokens grouped by expert, then batched matmul
2. Scatter-Gather overhead — Python-level vs optimized
3. MoE Layer end-to-end — full router→expert→combine pipeline
4. Expert load imbalance impact — skewed vs balanced routing
5. Shared Expert vs Routed Expert compute comparison

This connects FusedMoE source code analysis to real GPU performance.

Usage:
  python batched_gemm_moe_4090.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import json
import math
import time

def benchmark_cuda(fn, warmup=10, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat

def compute_tflops(M, N, K, time_ms, dtype_bytes=2):
    flops = 2 * M * N * K
    return flops / (time_ms * 1e-3) / 1e12

# ================================================================
# Experiment 1: Grouped/Batched GEMM (MoE Core Compute)
# ================================================================
def exp1_grouped_gemm():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 1: Grouped/Batched GEMM (MoE Core Compute)")
    print("=" * 60)

    # Simulate MoE expert computation
    # Each expert has: gate_up_proj (H → 4H) and down_proj (4H → H)
    # Tokens are routed to different experts → grouped by expert → batched GEMM

    H = 4096   # hidden_size (7B model)
    E = 8      # number of experts (Mixtral-8x7B style)
    intermediate = 4 * H

    # Test different batch sizes and expert assignments
    B_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512]
    results = []

    # Pre-create weight matrices for all experts
    weights_gate_up = [torch.randn(H, intermediate, device=device, dtype=torch.float16) for _ in range(E)]
    weights_down = [torch.randn(intermediate, H, device=device, dtype=torch.float16) for _ in range(E)]

    for B in B_sizes:
        # Method 1: Sequential per-expert GEMM (Python loop)
        # Each expert processes its assigned tokens
        tokens_per_expert_balanced = max(1, B // E)  # balanced assignment
        expert_assignments = [tokens_per_expert_balanced] * E
        # Handle remainder
        remainder = B - tokens_per_expert_balanced * E
        for i in range(remainder):
            expert_assignments[i] += 1

        # Create input tokens
        x = torch.randn(B, H, device=device, dtype=torch.float16)

        # Method 1a: Sequential (one expert at a time, no grouping)
        def sequential_moe_fn():
            outputs = []
            for e_idx in range(E):
                n_tokens = expert_assignments[e_idx]
                if n_tokens == 0:
                    continue
                # Simple: all tokens for this expert (balanced → each expert gets B/E tokens)
                start_idx = sum(expert_assignments[:e_idx])
                expert_input = x[start_idx:start_idx+n_tokens]
                gate_up = expert_input @ weights_gate_up[e_idx]
                # SwiGLU activation
                gate, up = gate_up.chunk(2, dim=-1)
                hidden = torch.nn.functional.silu(gate) * up
                output = hidden @ weights_down[e_idx]
                outputs.append(output)
            return torch.cat(outputs, dim=0)

        t_sequential = benchmark_cuda(sequential_moe_fn)

        # Method 1b: Grouped GEMM (batch all expert tokens together)
        # Use torch.bmm — batched matmul with [E, B/E, H] × [E, H, 4H]
        # This is the key optimization in FusedMoE

        def grouped_gemm_fn():
            # Reshape inputs: [E, tokens_per_expert, H]
            expert_inputs = x.reshape(E, tokens_per_expert_balanced, H)
            if remainder > 0:
                # Handle uneven: pad remainder tokens
                # For simplicity in benchmark, use balanced case
                pass

            # Batched GEMM: [E, B/E, H] × [E, H, 4H] → [E, B/E, 4H]
            w_gate_up_batched = torch.stack(weights_gate_up)  # [E, H, 4H]
            gate_up = torch.bmm(expert_inputs, w_gate_up_batched)

            # SwiGLU
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = torch.nn.functional.silu(gate) * up

            # Batched down proj: [E, B/E, 4H] × [E, 4H, H] → [E, B/E, H]
            w_down_batched = torch.stack(weights_down)  # [E, 4H, H]
            output = torch.bmm(hidden, w_down_batched)

            return output.reshape(-1, H)

        # Only run grouped if B is evenly divisible by E
        if B % E == 0 and B >= E:
            t_grouped = benchmark_cuda(grouped_gemm_fn)
            speedup = t_sequential / t_grouped
            print(f"  B={B}: sequential={t_sequential:.4f}ms, grouped={t_grouped:.4f}ms, "
                  f"speedup={speedup:.2f}x")
        else:
            t_grouped = None
            speedup = None
            print(f"  B={B}: sequential={t_sequential:.4f}ms, grouped=N/A (B not divisible by E)")

        # Compute TFLOPS for sequential approach
        # Each token: 2×H×4H (gate_up) + 2×4H×H (down) = 2×8×H² = 2×8×4096² = 268M FLOPS per token
        flops_per_token = 2 * H * intermediate + 2 * intermediate * H  # gate_up + down
        total_flops = B * flops_per_token
        tflops_sequential = total_flops / (t_sequential * 1e-3) / 1e12

        results.append({
            "B": B, "E": E, "H": H,
            "tokens_per_expert": tokens_per_expert_balanced,
            "sequential_ms": round(t_sequential, 4),
            "grouped_ms": round(t_grouped, 4) if t_grouped else None,
            "speedup": round(speedup, 2) if speedup else None,
            "tflops_sequential": round(tflops_sequential, 2),
        })

    return results

# ================================================================
# Experiment 2: Scatter-Gather Overhead (Python vs Optimized)
# ================================================================
def exp2_scatter_gather():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 2: Scatter-Gather Overhead (Python vs Optimized)")
    print("=" * 60)

    H = 4096
    E = 8
    B_sizes = [1, 8, 32, 64, 128, 256, 512]

    results = []

    for B in B_sizes:
        x = torch.randn(B, H, device=device, dtype=torch.float16)

        # Simulate routing: assign each token to an expert
        # Balanced: B//E tokens per expert
        expert_ids = torch.arange(E, device=device).repeat(B // E + 1)[:B]

        # Method 1: Python scatter (group tokens by expert using indexing)
        def scatter_fn():
            expert_inputs = []
            for e in range(E):
                mask = expert_ids == e
                expert_inputs.append(x[mask])
            return expert_inputs

        t_scatter = benchmark_cuda(scatter_fn, warmup=5, repeat=50)

        # Method 2: Python gather (reconstruct output by indexing)
        outputs_per_expert = [torch.randn(max(1, B // E), H, device=device, dtype=torch.float16) for _ in range(E)]

        def gather_fn():
            output = torch.zeros(B, H, device=device, dtype=torch.float16)
            for e in range(E):
                mask = expert_ids == e
                output[mask] = outputs_per_expert[e][:mask.sum()]
            return output

        t_gather = benchmark_cuda(gather_fn, warmup=5, repeat=50)

        # Method 3: Sort-based grouping (alternative to scatter)
        def sort_group_fn():
            sorted_ids, sorted_indices = torch.sort(expert_ids)
            sorted_x = x[sorted_indices]
            # Find boundaries
            boundaries = torch.searchsorted(sorted_ids, torch.arange(E, device=device))
            return sorted_x, boundaries

        t_sort_group = benchmark_cuda(sort_group_fn, warmup=5, repeat=50)

        # Baseline: just a simple matmul (no scatter/gather)
        w = torch.randn(H, 4*H, device=device, dtype=torch.float16)
        def baseline_fn():
            return x @ w

        t_baseline = benchmark_cuda(baseline_fn, warmup=5, repeat=50)

        scatter_pct = t_scatter / t_baseline * 100
        gather_pct = t_gather / t_baseline * 100
        sort_pct = t_sort_group / t_baseline * 100

        print(f"  B={B}: scatter={t_scatter:.4f}ms({scatter_pct:.1f}% baseline), "
              f"gather={t_gather:.4f}ms({gather_pct:.1f}%), "
              f"sort={t_sort_group:.4f}ms({sort_pct:.1f}%), "
              f"baseline GEMM={t_baseline:.4f}ms")

        results.append({
            "B": B,
            "scatter_ms": round(t_scatter, 4),
            "gather_ms": round(t_gather, 4),
            "sort_group_ms": round(t_sort_group, 4),
            "baseline_gemm_ms": round(t_baseline, 4),
            "scatter_overhead_pct": round(scatter_pct, 1),
            "gather_overhead_pct": round(gather_pct, 1),
            "sort_overhead_pct": round(sort_pct, 1),
        })

    return results

# ================================================================
# Experiment 3: MoE Layer End-to-End (Router → Expert → Combine)
# ================================================================
def exp3_moe_layer():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 3: MoE Layer End-to-End (Router → Expert → Combine)")
    print("=" * 60)

    # Simulate a full MoE layer like Mixtral-8x7B
    H = 4096      # hidden_size
    E = 8         # num_experts
    K = 2         # top_k (Mixtral uses top-2)
    intermediate = 4 * H  # 16384

    # Router weights
    router_weight = torch.randn(H, E, device=device, dtype=torch.float16)

    # Expert weights (gate_up + down for each)
    expert_gate_up = torch.randn(E, H, intermediate, device=device, dtype=torch.float16)
    expert_down = torch.randn(E, intermediate, H, device=device, dtype=torch.float16)

    B_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512]
    results = []

    for B in B_sizes:
        x = torch.randn(B, H, device=device, dtype=torch.float16)

        # Full MoE layer (Python implementation)
        def moe_layer_fn():
            # Step 1: Router
            router_logits = x @ router_weight  # [B, E]
            # Top-K selection
            top_k_logits, top_k_indices = torch.topk(router_logits, K, dim=-1)
            top_k_weights = torch.softmax(top_k_logits, dim=-1)  # [B, K]

            # Step 2: Expert computation (sequential per-expert)
            output = torch.zeros(B, H, device=device, dtype=torch.float16)

            for e in range(E):
                # Find tokens routed to this expert (top-K includes this expert)
                # For top-2: each token goes to 2 experts
                mask = (top_k_indices == e).any(dim=-1)  # [B]
                if not mask.any():
                    continue
                expert_input = x[mask]  # scatter
                n_tokens = expert_input.shape[0]

                # gate_up proj
                gate_up = expert_input @ expert_gate_up[e]
                gate, up = gate_up.chunk(2, dim=-1)
                hidden = torch.nn.functional.silu(gate) * up

                # down proj
                expert_output = hidden @ expert_down[e]  # [n_tokens, H]

                # Step 3: Combine with weights
                # Get the weight for this expert for each routed token
                expert_mask = top_k_indices == e  # [B, K]
                weight_idx = expert_mask.nonzero()  # which (token, k) pairs
                for idx in weight_idx:
                    token_idx = idx[0].item()
                    k_idx = idx[1].item()
                    output[token_idx] += top_k_weights[token_idx, k_idx] * expert_output[weight_idx[:, 0] == token_idx][0]

            return output

        t_moe = benchmark_cuda(moe_layer_fn, warmup=5, repeat=20 if B <= 64 else 10)

        # Baseline: dense MLP (single expert)
        def dense_mlp_fn():
            gate_up = x @ expert_gate_up[0]
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = torch.nn.functional.silu(gate) * up
            return hidden @ expert_down[0]

        t_dense = benchmark_cuda(dense_mlp_fn, warmup=5, repeat=20 if B <= 64 else 10)

        # Compute TFLOPS
        # MoE: each token goes through K experts → K×(2×H×4H + 2×4H×H) = K×16×H²
        # Dense: 1×(2×H×4H + 2×4H×H) = 16×H²
        flops_per_token_moe = K * (2 * H * intermediate + 2 * intermediate * H)
        flops_per_token_dense = 2 * H * intermediate + 2 * intermediate * H
        tflops_moe = B * flops_per_token_moe / (t_moe * 1e-3) / 1e12
        tflops_dense = B * flops_per_token_dense / (t_dense * 1e-3) / 1e12

        ratio = t_moe / t_dense
        print(f"  B={B}: MoE={t_moe:.4f}ms({tflops_moe:.2f} TFLOPS), "
              f"Dense={t_dense:.4f}ms({tflops_dense:.2f} TFLOPS), "
              f"MoE/Dense={ratio:.2f}x")

        results.append({
            "B": B, "E": E, "K": K, "H": H,
            "moe_ms": round(t_moe, 4),
            "dense_ms": round(t_dense, 4),
            "moe_dense_ratio": round(ratio, 2),
            "tflops_moe": round(tflops_moe, 2),
            "tflops_dense": round(tflops_dense, 2),
            "moe_active_params_ratio": K / E,  # top-K/E = fraction of active params
        })

    return results

# ================================================================
# Experiment 4: Expert Load Imbalance Impact
# ================================================================
def exp4_load_imbalance():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 4: Expert Load Imbalance Impact")
    print("=" * 60)

    H = 4096
    E = 8
    intermediate = 4 * H

    B = 128  # Fixed batch size

    # Different imbalance levels
    # Balanced: [16, 16, 16, 16, 16, 16, 16, 16] = B/E
    # Mild: [24, 20, 16, 12, 12, 16, 16, 12] = std=4.0
    # Severe: [48, 32, 16, 8, 8, 8, 8, 8] = std=14
    # Extreme: [96, 16, 4, 4, 2, 2, 2, 2] = std=30

    distributions = {
        "balanced": [16, 16, 16, 16, 16, 16, 16, 16],
        "mild":     [24, 20, 16, 12, 12, 16, 16, 12],
        "severe":   [48, 32, 16, 8, 8, 8, 8, 8],
        "extreme":  [96, 16, 4, 4, 2, 2, 2, 2],
    }

    # Pre-create all expert weights
    weights_gate_up = [torch.randn(H, intermediate, device=device, dtype=torch.float16) for _ in range(E)]
    weights_down = [torch.randn(intermediate, H, device=device, dtype=torch.float16) for _ in range(E)]

    results = []

    for name, tokens_per_expert in distributions.items():
        total = sum(tokens_per_expert)
        std = torch.tensor(tokens_per_expert).float().std().item()

        # Sequential per-expert computation
        def imbalanced_fn():
            outputs = []
            for e in range(E):
                n = tokens_per_expert[e]
                if n == 0:
                    continue
                expert_input = torch.randn(n, H, device=device, dtype=torch.float16)
                gate_up = expert_input @ weights_gate_up[e]
                gate, up = gate_up.chunk(2, dim=-1)
                hidden = torch.nn.functional.silu(gate) * up
                output = hidden @ weights_down[e]
                outputs.append(output)
            return outputs

        t_imbalanced = benchmark_cuda(imbalanced_fn, warmup=5, repeat=20)

        # Balanced baseline (same total tokens, evenly distributed)
        balanced_per = total // E
        def balanced_fn():
            expert_inputs = torch.randn(E, balanced_per, H, device=device, dtype=torch.float16)
            w_gate_up = torch.stack(weights_gate_up)
            gate_up = torch.bmm(expert_inputs, w_gate_up)
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = torch.nn.functional.silu(gate) * up
            w_down = torch.stack(weights_down)
            output = torch.bmm(hidden, w_down)
            return output

        t_balanced = benchmark_cuda(balanced_fn, warmup=5, repeat=20)

        # Max single-expert time (hot expert bottleneck)
        max_expert_tokens = max(tokens_per_expert)
        def single_expert_fn():
            input = torch.randn(max_expert_tokens, H, device=device, dtype=torch.float16)
            gate_up = input @ weights_gate_up[0]
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = torch.nn.functional.silu(gate) * up
            return hidden @ weights_down[0]

        t_single = benchmark_cuda(single_expert_fn, warmup=5, repeat=20)

        ratio = t_imbalanced / t_balanced
        print(f"  {name} (std={std:.1f}): imbalanced={t_imbalanced:.4f}ms, "
              f"balanced={t_balanced:.4f}ms, ratio={ratio:.2f}x, "
              f"hot_expert={max_expert_tokens}tok/{t_single:.4f}ms")

        results.append({
            "distribution": name,
            "tokens_per_expert": tokens_per_expert,
            "std": round(std, 2),
            "imbalanced_ms": round(t_imbalanced, 4),
            "balanced_ms": round(t_balanced, 4),
            "ratio": round(ratio, 2),
            "hot_expert_tokens": max_expert_tokens,
            "hot_expert_ms": round(t_single, 4),
        })

    return results

# ================================================================
# Experiment 5: Shared Expert vs Routed Expert
# ================================================================
def exp5_shared_expert():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 5: Shared Expert vs Routed Expert (DeepSeek-V3 style)")
    print("=" * 60)

    # DeepSeek-V3: 256 routed experts (Top-6) + 1 shared expert
    # Shared expert: ALL tokens pass through (like dense MLP)
    # Routed experts: only Top-K=6 tokens

    H = 4096
    intermediate = 4 * H  # intermediate_size

    B_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512]

    # Shared expert weight (single, all tokens use)
    w_shared_gate_up = torch.randn(H, intermediate, device=device, dtype=torch.float16)
    w_shared_down = torch.randn(intermediate, H, device=device, dtype=torch.float16)

    # Routed expert weight (8 experts, Top-2 for simplicity)
    E = 8
    K = 2
    w_routed_gate_up = [torch.randn(H, intermediate, device=device, dtype=torch.float16) for _ in range(E)]
    w_routed_down = [torch.randn(intermediate, H, device=device, dtype=torch.float16) for _ in range(E)]
    w_router = torch.randn(H, E, device=device, dtype=torch.float16)

    results = []

    for B in B_sizes:
        x = torch.randn(B, H, device=device, dtype=torch.float16)

        # Shared expert (dense MLP, ALL tokens)
        def shared_expert_fn():
            gate_up = x @ w_shared_gate_up
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = torch.nn.functional.silu(gate) * up
            return hidden @ w_shared_down

        t_shared = benchmark_cuda(shared_expert_fn, warmup=5, repeat=50 if B <= 64 else 20)

        # Routed experts (Top-K, subset of tokens per expert)
        def routed_experts_fn():
            router_logits = x @ w_router
            top_k_logits, top_k_indices = torch.topk(router_logits, K, dim=-1)
            top_k_weights = torch.softmax(top_k_logits, dim=-1)

            output = torch.zeros(B, H, device=device, dtype=torch.float16)
            for e in range(E):
                mask = (top_k_indices == e).any(dim=-1)
                if not mask.any():
                    continue
                expert_input = x[mask]
                gate_up = expert_input @ w_routed_gate_up[e]
                gate, up = gate_up.chunk(2, dim=-1)
                hidden = torch.nn.functional.silu(gate) * up
                expert_output = hidden @ w_routed_down[e]
                # Weighted combine
                expert_mask = top_k_indices == e
                for idx in expert_mask.nonzero():
                    token_idx = idx[0].item()
                    k_idx = idx[1].item()
                    output[token_idx] += top_k_weights[token_idx, k_idx] * expert_output[expert_mask[:, 0] == token_idx][0]
            return output

        t_routed = benchmark_cuda(routed_experts_fn, warmup=3, repeat=10 if B <= 64 else 5)

        # Combined: shared + routed (full DeepSeek-V3 MoE layer)
        def full_moe_fn():
            shared_out = shared_expert_fn()
            routed_out = routed_experts_fn()
            return shared_out + routed_out

        t_full = benchmark_cuda(full_moe_fn, warmup=3, repeat=10 if B <= 64 else 5)

        # Compute TFLOPS
        # Shared: B × (2×H×4H + 2×4H×H) = B × 16×H²
        # Routed: B × K × (2×H×4H + 2×4H×H) = B × K × 16×H²
        # Total: B × (1+K) × 16 × H²
        flops_shared = B * 2 * H * intermediate + B * 2 * intermediate * H
        flops_routed = B * K * (2 * H * intermediate + 2 * intermediate * H)
        tflops_shared = flops_shared / (t_shared * 1e-3) / 1e12
        tflops_routed = flops_routed / (t_routed * 1e-3) / 1e12

        # FLOPS ratio
        # DeepSeek-V3: shared(1) + routed(K=6) → K+1=7 experts per token
        # vs pure dense: 1 expert → (K+1)/1 = 7x more compute per token
        # But total params: shared(1) + routed(E=256) → much larger model
        active_ratio = (1 + K) / (1 + E)  # fraction of total expert params active

        print(f"  B={B}: shared={t_shared:.4f}ms({tflops_shared:.2f} TFLOPS), "
              f"routed={t_routed:.4f}ms({tflops_routed:.2f} TFLOPS), "
              f"full={t_full:.4f}ms, "
              f"shared_pct={t_shared/t_full*100:.1f}%")

        results.append({
            "B": B,
            "shared_ms": round(t_shared, 4),
            "routed_ms": round(t_routed, 4),
            "full_ms": round(t_full, 4),
            "tflops_shared": round(tflops_shared, 2),
            "tflops_routed": round(tflops_routed, 2),
            "shared_pct_of_full": round(t_shared/t_full*100, 1),
            "active_params_ratio": round(active_ratio, 4),
            "K": K, "E": E,
        })

    return results

# ================================================================
# Main
# ================================================================
def main():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
    gpu_props = torch.cuda.get_device_properties(device)

    print(f"Batched GEMM MoE Benchmark: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"  SM: {gpu_props.major}.{gpu_props.minor}")
    print(f"  CUDA cores: {gpu_props.multi_processor_count * 128}")
    print(f"  MPs: {gpu_props.multi_processor_count}")
    print("=" * 60)

    all_results = {
        "gpu": gpu_name, "gpu_mem_gb": round(gpu_mem, 1),
        "sm_version": f"{gpu_props.major}.{gpu_props.minor}",
        "cuda_cores": gpu_props.multi_processor_count * 128,
        "mps": gpu_props.multi_processor_count,
    }

    all_results["exp1_grouped_gemm"] = exp1_grouped_gemm()
    all_results["exp2_scatter_gather"] = exp2_scatter_gather()
    all_results["exp3_moe_layer"] = exp3_moe_layer()
    all_results["exp4_load_imbalance"] = exp4_load_imbalance()
    all_results["exp5_shared_expert"] = exp5_shared_expert()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Grouped GEMM speedup
    grouped = all_results["exp1_grouped_gemm"]
    grouped_speedups = [r["speedup"] for r in grouped if r["speedup"] is not None]
    if grouped_speedups:
        print(f"  Grouped GEMM speedup range: {min(grouped_speedups):.2f}x - {max(grouped_speedups):.2f}x")

    # Scatter/gather overhead
    sg = all_results["exp2_scatter_gather"]
    scatter_max = max(r["scatter_overhead_pct"] for r in sg)
    gather_max = max(r["gather_overhead_pct"] for r in sg)
    print(f"  Scatter overhead max: {scatter_max:.1f}% of baseline GEMM")
    print(f"  Gather overhead max: {gather_max:.1f}% of baseline GEMM")

    # MoE vs Dense
    moe = all_results["exp3_moe_layer"]
    moe_ratios = [r["moe_dense_ratio"] for r in moe]
    print(f"  MoE/Dense time ratio: {min(moe_ratios):.2f}x - {max(moe_ratios):.2f}x")
    print(f"  Expected: ~{2/8*100:.1f}% active params → compute ~{(2/8)*2:.1f}x dense")

    # Imbalance impact
    imbalance = all_results["exp4_load_imbalance"]
    for r in imbalance:
        print(f"  {r['distribution']}: {r['ratio']:.2f}x vs balanced (std={r['std']:.1f})")

    # Shared vs Routed
    shared_exp = all_results["exp5_shared_expert"]
    shared_pcts = [r["shared_pct_of_full"] for r in shared_exp]
    print(f"  Shared expert % of total: {shared_pcts[-1]:.1f}% (B=512)")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'batched_gemm_moe_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()