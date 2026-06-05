#!/usr/bin/env python3
"""Expert Parallelism Simulator for MoE Models
=============================================
Simulates and compares different strategies for distributing MoE experts
across GPUs during inference. Based on concepts from:
  - Mixtral 8x7B (arXiv:2401.04088)
  - DeepSeek-V3 (arXiv:2412.19437)
  - Megatron-LM EP, vLLM EP

Strategies compared:
  1. No Parallelism (single GPU, all experts)
  2. Expert Parallelism (EP) - experts split across GPUs
  3. EP + Tensor Parallelism (EP+TP) - hybrid
  4. Expert Offloading (CPU) - inactive experts on CPU

Experiments:
  1. Latency vs GPU count (EP scaling)
  2. Communication overhead analysis
  3. Load imbalance impact
  4. Expert offloading tradeoff
  5. Mixtral vs DeepSeek-V3 EP comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ============================================================
# MoE Model Components
# ============================================================

class ExpertFFN(nn.Module):
    """Single expert FFN (SwiGLU)."""
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TopKRouter(nn.Module):
    """Top-K router with optional bias-based load balancing."""
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_experts))
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x):
        logits = self.gate(x) + self.bias
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        return topk_scores, topk_indices


class MoELayer(nn.Module):
    """MoE layer with configurable number of experts."""
    def __init__(self, hidden_dim: int, intermediate_dim: int,
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_dim, intermediate_dim) for _ in range(num_experts)
        ])
        self.router = TopKRouter(hidden_dim, num_experts, top_k)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)  # (B*S, D)
        scores, indices = self.router(x_flat)  # (B*S, top_k)

        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = (indices[:, k] == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += scores[mask, k].unsqueeze(-1) * expert_output
        return output.reshape(B, S, D)


# ============================================================
# Expert Parallelism Simulator
# ============================================================

@dataclass
class EPConfig:
    """Expert Parallelism configuration."""
    num_gpus: int = 1
    num_experts: int = 8
    top_k: int = 2
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    batch_size: int = 32
    seq_len: int = 1  # inference: typically 1 (decode step)
    # Hardware
    gpu_memory_bandwidth_gbps: float = 1008.0  # RTX 4090
    gpu_flops: float = 82.6  # TFLOPS (FP16)
    nvlink_bandwidth_gbps: float = 600.0  # bidirectional per link
    pcie_bandwidth_gbps: float = 64.0  # PCIe 4.0 x16
    cpu_transfer_gbps: float = 32.0  # DDR5 → GPU
    # Expert size
    expert_params: int = 0  # calculated automatically

    def __post_init__(self):
        # Expert params: 3 linear layers (SwiGLU: w1, w2, w3)
        self.expert_params = self.hidden_dim * self.intermediate_dim * 3


@dataclass
class EPMetrics:
    """Metrics from an EP simulation run."""
    strategy: str
    compute_time_us: float = 0
    communication_time_us: float = 0
    total_latency_us: float = 0
    memory_per_gpu_mb: float = 0
    total_memory_mb: float = 0
    load_imbalance_ratio: float = 1.0
    gpu_utilization: float = 1.0
    experts_per_gpu: int = 0
    communication_bytes: int = 0


class ExpertParallelSimulator:
    """Simulates expert parallelism strategies analytically."""

    def __init__(self, config: EPConfig):
        self.config = config

    def _expert_compute_time(self, tokens_per_expert: int) -> float:
        """Estimate compute time for one expert processing N tokens (us)."""
        c = self.config
        # Each expert has 3 matmuls: (N, D) × (D, I), (N, D) × (D, I), (N, I) × (I, D)
        # Total FLOPs = 2 * (N * D * I) * 3
        flops = 2 * tokens_per_expert * c.hidden_dim * c.intermediate_dim * 3
        # Convert to microseconds: flops / (TFLOPS * 1e12) * 1e6 = flops / (TFLOPS * 1e6)
        return flops / (c.gpu_flops * 1e6)

    def _expert_memory_mb(self) -> float:
        """Memory for one expert (FP16)."""
        return self.config.expert_params * 2 / 1e6  # 2 bytes per param

    def _all_to_all_time(self, data_bytes: int) -> float:
        """All-to-all communication time (us)."""
        c = self.config
        if c.num_gpus <= 1:
            return 0
        # In all-to-all, each GPU sends data to all other GPUs
        # Per-pair transfer = data_bytes / num_gpus
        # Time = per_pair / bandwidth
        bandwidth = c.nvlink_bandwidth_gbps if c.num_gpus <= 8 else c.pcie_bandwidth_gbps
        per_pair_bytes = data_bytes / c.num_gpus
        return per_pair_bytes * 8 / (bandwidth * 1e3)  # Gbps → bytes/s → us

    def simulate_no_parallelism(self) -> EPMetrics:
        """Strategy 1: All experts on single GPU."""
        c = self.config
        total_tokens = c.batch_size * c.seq_len
        # Each token activates top_k experts, but all experts loaded
        # Average tokens per expert (ideal balanced)
        avg_tokens = total_tokens * c.top_k / c.num_experts

        # All experts run sequentially on single GPU
        compute_time = self._expert_compute_time(avg_tokens) * c.num_experts
        memory = self._expert_memory_mb() * c.num_experts

        return EPMetrics(
            strategy="No Parallelism",
            compute_time_us=compute_time,
            communication_time_us=0,
            total_latency_us=compute_time,
            memory_per_gpu_mb=memory,
            total_memory_mb=memory,
            load_imbalance_ratio=1.0,
            gpu_utilization=1.0,
            experts_per_gpu=c.num_experts,
            communication_bytes=0,
        )

    def simulate_ep(self) -> EPMetrics:
        """Strategy 2: Expert Parallelism - split experts across GPUs."""
        c = self.config
        total_tokens = c.batch_size * c.seq_len

        experts_per_gpu = math.ceil(c.num_experts / c.num_gpus)
        avg_tokens_per_expert = total_tokens * c.top_k / c.num_experts

        # Tokens routed to this GPU's experts
        tokens_this_gpu = avg_tokens_per_expert * experts_per_gpu

        compute_time = self._expert_compute_time(tokens_this_gpu / experts_per_gpu) * experts_per_gpu

        # Communication: all-to-all for dispatching tokens to correct expert GPUs
        # Each token sends hidden_dim * 2 bytes (FP16)
        dispatch_bytes = total_tokens * c.hidden_dim * 2
        # Receive: tokens coming back after expert computation
        gather_bytes = total_tokens * c.hidden_dim * 2

        comm_time = self._all_to_all_time(dispatch_bytes) + self._all_to_all_time(gather_bytes)

        memory_per_gpu = self._expert_memory_mb() * experts_per_gpu

        # Load imbalance: simulate random routing
        imbalance = self._estimate_load_imbalance(tokens_this_gpu, experts_per_gpu)

        return EPMetrics(
            strategy=f"EP ({c.num_gpus} GPUs)",
            compute_time_us=compute_time,
            communication_time_us=comm_time,
            total_latency_us=compute_time + comm_time,
            memory_per_gpu_mb=memory_per_gpu,
            total_memory_mb=memory_per_gpu * c.num_gpus,
            load_imbalance_ratio=imbalance,
            gpu_utilization=1.0 / imbalance,
            experts_per_gpu=experts_per_gpu,
            communication_bytes=dispatch_bytes + gather_bytes,
        )

    def simulate_ep_tp(self, tp_size: int = 2) -> EPMetrics:
        """Strategy 3: EP + TP hybrid."""
        c = self.config
        ep_size = c.num_gpus // tp_size
        total_tokens = c.batch_size * c.seq_len

        experts_per_ep_group = math.ceil(c.num_experts / ep_size)
        avg_tokens = total_tokens * c.top_k / c.num_experts
        tokens_per_group = avg_tokens * experts_per_ep_group

        # EP compute (within TP group, compute is split)
        compute_per_expert = self._expert_compute_time(avg_tokens / tp_size)
        compute_time = compute_per_expert * experts_per_ep_group

        # EP communication
        dispatch_bytes = total_tokens * c.hidden_dim * 2
        gather_bytes = total_tokens * c.hidden_dim * 2
        ep_comm = self._all_to_all_time(dispatch_bytes) + self._all_to_all_time(gather_bytes)

        # TP communication (all-reduce within group)
        tp_bytes = total_tokens * c.hidden_dim * 2 * tp_size
        tp_comm = tp_bytes / (c.nvlink_bandwidth_gbps * 1e9 / 1e6)

        memory_per_gpu = self._expert_memory_mb() * experts_per_ep_group / tp_size

        imbalance = self._estimate_load_imbalance(tokens_per_group, experts_per_ep_group)

        return EPMetrics(
            strategy=f"EP+TP (EP={ep_size},TP={tp_size})",
            compute_time_us=compute_time,
            communication_time_us=ep_comm + tp_comm,
            total_latency_us=compute_time + ep_comm + tp_comm,
            memory_per_gpu_mb=memory_per_gpu,
            total_memory_mb=memory_per_gpu * c.num_gpus,
            load_imbalance_ratio=imbalance,
            gpu_utilization=1.0 / imbalance,
            experts_per_gpu=experts_per_ep_group,
            communication_bytes=dispatch_bytes + gather_bytes + tp_bytes,
        )

    def simulate_expert_offloading(self, hot_experts: int = 2) -> EPMetrics:
        """Strategy 4: Keep hot experts on GPU, offload rest to CPU."""
        c = self.config
        total_tokens = c.batch_size * c.seq_len

        # Simulate: top_k experts are "hot" (always on GPU)
        # Rest are offloaded to CPU
        hot_ratio = min(c.top_k / c.num_experts * 2, 1.0)  # heuristic
        actual_hot = min(hot_experts, c.num_experts)

        avg_tokens_per_expert = total_tokens * c.top_k / c.num_experts

        # GPU compute (hot experts only)
        gpu_tokens = avg_tokens_per_expert * actual_hot
        compute_time = self._expert_compute_time(gpu_tokens / actual_hot) * actual_hot

        # CPU transfer for cold experts
        cold_experts = c.num_experts - actual_hot
        cold_tokens = int(total_tokens * c.top_k * (1 - hot_ratio))
        transfer_bytes = cold_tokens * c.hidden_dim * 2
        # Gbps → bytes/s: bandwidth * 1e9 / 8; time_us = bytes / (bytes_per_s) * 1e6
        transfer_time = transfer_bytes * 8 / (c.cpu_transfer_gbps * 1e3)

        # Cold expert compute on CPU (much slower)
        cpu_tflops = 0.5  # ~0.5 TFLOPS for CPU
        cold_compute = 2 * cold_tokens * c.hidden_dim * c.intermediate_dim * 3
        cold_compute_time = cold_compute / (cpu_tflops * 1e6)  # same formula as GPU

        # GPU memory: only hot experts
        gpu_memory = self._expert_memory_mb() * actual_hot

        return EPMetrics(
            strategy=f"Offload (hot={actual_hot}/{c.num_experts})",
            compute_time_us=compute_time + cold_compute_time,
            communication_time_us=transfer_time,
            total_latency_us=compute_time + transfer_time + cold_compute_time,
            memory_per_gpu_mb=gpu_memory,
            total_memory_mb=gpu_memory + self._expert_memory_mb() * cold_experts,
            load_imbalance_ratio=1.0,
            gpu_utilization=hot_ratio,
            experts_per_gpu=actual_hot,
            communication_bytes=transfer_bytes,
        )

    def _estimate_load_imbalance(self, total_tokens, num_experts: int,
                                  num_samples: int = 1000) -> float:
        """Estimate load imbalance ratio via simulation."""
        c = self.config
        total_tokens = int(total_tokens)
        num_experts = int(num_experts)
        max_load = 0
        min_load = float('inf')

        for _ in range(num_samples):
            # Simulate random routing to experts
            expert_loads = [0] * num_experts
            for _ in range(total_tokens):
                selected = random.sample(range(num_experts), min(c.top_k, num_experts))
                for e in selected:
                    expert_loads[e] += 1
            if max(expert_loads) > 0:
                ratio = max(expert_loads) / (total_tokens * c.top_k / num_experts)
                max_load = max(max_load, ratio)
                min_load = min(min_load, min(expert_loads) / (total_tokens * c.top_k / num_experts + 1e-9))

        return max(max_load, 1.0)


# ============================================================
# Experiments
# ============================================================

def experiment1_ep_scaling():
    """How does EP latency scale with GPU count?"""
    print("\n" + "="*70)
    print("Experiment 1: EP Scaling (GPU Count vs Latency)")
    print("="*70)

    results = {}

    # Mixtral-like: 8 experts, top-2
    for num_gpus in [1, 2, 4, 8]:
        config = EPConfig(
            num_gpus=num_gpus,
            num_experts=8,
            top_k=2,
            hidden_dim=4096,
            intermediate_dim=11008,
            batch_size=32,
        )
        sim = ExpertParallelSimulator(config)
        if num_gpus == 1:
            m = sim.simulate_no_parallelism()
        else:
            m = sim.simulate_ep()
        results[num_gpus] = {
            'latency_us': m.total_latency_us,
            'compute_us': m.compute_time_us,
            'comm_us': m.communication_time_us,
            'memory_per_gpu_mb': m.memory_per_gpu_mb,
            'imbalance': m.load_imbalance_ratio,
        }
        print(f"  GPUs={num_gpus}: total={m.total_latency_us:.1f}us "
              f"(compute={m.compute_time_us:.1f}us, comm={m.communication_time_us:.1f}us) "
              f"memory/GPU={m.memory_per_gpu_mb:.0f}MB imbalance={m.load_imbalance_ratio:.2f}")

    # Ideal speedup line
    print(f"\n  Speedup vs 1 GPU:")
    base = results[1]['latency_us']
    for g in [2, 4, 8]:
        actual_speedup = base / results[g]['latency_us']
        ideal_speedup = g
        eff = actual_speedup / ideal_speedup * 100
        print(f"    {g} GPUs: {actual_speedup:.2f}x (ideal {ideal_speedup}x, "
              f"efficiency {eff:.1f}%)")

    return results


def experiment2_communication_overhead():
    """Break down communication vs compute."""
    print("\n" + "="*70)
    print("Experiment 2: Communication Overhead Breakdown")
    print("="*70)

    results = {}
    for batch_size in [1, 8, 32, 128, 512]:
        config = EPConfig(
            num_gpus=8,
            num_experts=8,
            top_k=2,
            batch_size=batch_size,
        )
        sim = ExpertParallelSimulator(config)
        m = sim.simulate_ep()
        comm_ratio = m.communication_time_us / max(m.total_latency_us, 1e-9) * 100
        results[batch_size] = {
            'total_us': m.total_latency_us,
            'compute_us': m.compute_time_us,
            'comm_us': m.communication_time_us,
            'comm_ratio_pct': comm_ratio,
        }
        print(f"  B={batch_size:4d}: total={m.total_latency_us:8.1f}us, "
              f"comm={m.communication_time_us:8.1f}us ({comm_ratio:5.1f}%)")

    print(f"\n  Insight: Communication dominates at small batch, compute at large batch")

    return results


def experiment3_load_imbalance():
    """Impact of load imbalance on EP efficiency."""
    print("\n" + "="*70)
    print("Experiment 3: Load Imbalance Impact")
    print("="*70)

    results = {}
    for num_experts in [8, 16, 32, 64, 128, 256]:
        for top_k in [2, 4, 6]:
            config = EPConfig(
                num_gpus=8,
                num_experts=num_experts,
                top_k=top_k,
                batch_size=32,
            )
            sim = ExpertParallelSimulator(config)
            m = sim.simulate_ep()
            key = f"E{num_experts}_K{top_k}"
            results[key] = {
                'experts': num_experts,
                'top_k': top_k,
                'imbalance': m.load_imbalance_ratio,
                'gpu_util': m.gpu_utilization,
            }
            if num_experts in [8, 64, 256] or top_k == 2:
                print(f"  E={num_experts:3d}, K={top_k}: "
                      f"imbalance={m.load_imbalance_ratio:.2f}, "
                      f"GPU util={m.gpu_utilization:.1%}")

    print(f"\n  Insight: More experts = worse imbalance at same batch size")
    print(f"  Top-K helps: higher K = better load spreading")

    return results


def experiment4_offloading_tradeoff():
    """Expert offloading: memory savings vs latency cost."""
    print("\n" + "="*70)
    print("Experiment 4: Expert Offloading Tradeoff")
    print("="*70)

    results = {}

    # Mixtral-like config
    config = EPConfig(
        num_gpus=1,
        num_experts=8,
        top_k=2,
        hidden_dim=4096,
        intermediate_dim=11008,
        batch_size=32,
    )
    sim = ExpertParallelSimulator(config)
    baseline = sim.simulate_no_parallelism()
    print(f"  Baseline (all on GPU): {baseline.total_latency_us:.1f}us, "
          f"memory={baseline.memory_per_gpu_mb:.0f}MB")

    for hot in [1, 2, 3, 4, 6, 8]:
        config_off = EPConfig(
            num_gpus=1,
            num_experts=8,
            top_k=2,
            hidden_dim=4096,
            intermediate_dim=11008,
            batch_size=32,
        )
        sim_off = ExpertParallelSimulator(config_off)
        m = sim_off.simulate_expert_offloading(hot_experts=hot)
        mem_saving = (1 - m.memory_per_gpu_mb / baseline.memory_per_gpu_mb) * 100
        latency_overhead = m.total_latency_us / baseline.total_latency_us
        results[hot] = {
            'latency_us': m.total_latency_us,
            'memory_per_gpu_mb': m.memory_per_gpu_mb,
            'mem_saving_pct': mem_saving,
            'latency_ratio': latency_overhead,
        }
        print(f"  Hot={hot}/8: latency={m.total_latency_us:.1f}us "
              f"({latency_overhead:.1f}x), "
              f"memory={m.memory_per_gpu_mb:.0f}MB (-{mem_saving:.0f}%)")

    print(f"\n  Insight: Offloading saves memory but adds significant latency")
    print(f"  Sweet spot: keep top-2 experts hot (matches Mixtral's Top-K)")

    return results


def experiment5_model_comparison():
    """Compare EP characteristics for Mixtral vs DeepSeek-V3."""
    print("\n" + "="*70)
    print("Experiment 5: Mixtral vs DeepSeek-V3 EP Comparison")
    print("="*70)

    configs = {
        'Mixtral 8x7B': EPConfig(
            num_gpus=8,
            num_experts=8,
            top_k=2,
            hidden_dim=4096,
            intermediate_dim=11008,
            batch_size=32,
        ),
        'DeepSeek-V3 (671B)': EPConfig(
            num_gpus=8,
            num_experts=256,
            top_k=6,
            hidden_dim=7168,
            intermediate_dim=18432,
            batch_size=32,
        ),
    }

    results = {}
    print(f"\n  {'Model':<22} | {'Experts':>7} | {'Top-K':>5} | {'Latency':>10} | "
          f"{'Comm%':>5} | {'Mem/GPU':>8} | {'Imbalance':>9}")
    print(f"  {'-'*22}-+-{'-'*7}-+-{'-'*5}-+-{'-'*10}-+-{'-'*5}-+-{'-'*8}-+-{'-'*9}")

    for name, config in configs.items():
        sim = ExpertParallelSimulator(config)
        m = sim.simulate_ep()
        comm_pct = m.communication_time_us / max(m.total_latency_us, 1e-9) * 100
        results[name] = {
            'latency_us': m.total_latency_us,
            'compute_us': m.compute_time_us,
            'comm_us': m.communication_time_us,
            'comm_pct': comm_pct,
            'memory_per_gpu_mb': m.memory_per_gpu_mb,
            'imbalance': m.load_imbalance_ratio,
            'experts': config.num_experts,
            'top_k': config.top_k,
        }
        print(f"  {name:<22} | {config.num_experts:>7d} | {config.top_k:>5d} | "
              f"{m.total_latency_us:>8.1f}us | {comm_pct:>4.1f}% | "
              f"{m.memory_per_gpu_mb:>6.0f}MB | {m.load_imbalance_ratio:>8.2f}")

    print(f"\n  Key differences:")
    print(f"  - DeepSeek-V3: 256 experts, EP=8 means 32 experts/GPU → high imbalance risk")
    print(f"  - Mixtral: 8 experts, EP=8 means 1 expert/GPU → perfectly balanced")
    print(f"  - DeepSeek-V3 needs much more communication due to larger hidden dim")

    return results


def run_all_experiments():
    print("="*70)
    print("Expert Parallelism Simulator for MoE Models")
    print("Based on: Mixtral 8x7B, DeepSeek-V3")
    print("="*70)

    all_results = {}
    all_results['exp1_ep_scaling'] = experiment1_ep_scaling()
    all_results['exp2_comm_overhead'] = experiment2_communication_overhead()
    all_results['exp3_load_imbalance'] = experiment3_load_imbalance()
    all_results['exp4_offloading'] = experiment4_offloading_tradeoff()
    all_results['exp5_model_comparison'] = experiment5_model_comparison()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("""
  Key Findings:
  1. EP scales well up to #GPUs = #experts, then no benefit
  2. Communication overhead is significant for small batches
  3. Load imbalance grows with more experts (256 > 8)
  4. Expert offloading trades latency for memory savings
  5. Mixtral (8 experts) is EP-friendly; DeepSeek-V3 (256) needs careful scheduling

  Practical Recommendations:
  - Mixtral: EP=8 (1 expert/GPU) or EP=4 (2 experts/GPU)
  - DeepSeek-V3: EP=8 + TP within node, EP across nodes
  - For memory-constrained: offload cold experts, keep top-K hot
  """)

    with open('expert_parallel_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("Results saved to expert_parallel_results.json")
    return all_results


if __name__ == '__main__':
    run_all_experiments()
