#!/usr/bin/env python3
"""
MoE Router 模拟器

模拟 Mixture-of-Experts (MoE) 模型中 Token Routing 的行为，
包括负载均衡分析、Expert 容量、以及不同路由策略的对比。

用法:
    python moe_router_demo.py
    python moe_router_demo.py --num-experts 8 --top-k 2 --num-tokens 1024
"""

import argparse
import math
import numpy as np


def softmax(x, axis=-1):
    """数值稳定的 softmax"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


class TopKRouter:
    """Top-K 路由器：每个 token 选择 top-k 个 expert"""

    def __init__(self, num_experts, top_k, capacity_factor=1.0):
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

    def route(self, gate_logits):
        """
        gate_logits: [num_tokens, num_experts] — 每个 token 对每个 expert 的评分
        返回: routing_weights [num_tokens, top_k], expert_indices [num_tokens, top_k]
        """
        # 计算 softmax gate 概率
        gates = softmax(gate_logits, axis=-1)

        # 选择 top-k experts
        top_k_indices = np.argsort(-gates, axis=-1)[:, :self.top_k]
        top_k_gates = np.take_along_axis(gates, top_k_indices, axis=-1)

        # 归一化选中的 gate 权重
        top_k_gates = top_k_gates / top_k_gates.sum(axis=-1, keepdims=True)

        return top_k_gates, top_k_indices

    def analyze_load_balance(self, expert_indices, num_tokens):
        """分析负载均衡情况"""
        expert_counts = np.zeros(self.num_experts, dtype=int)
        for indices in expert_indices:
            for idx in indices:
                expert_counts[idx] += 1

        ideal = num_tokens * self.top_k / self.num_experts
        max_load = expert_counts.max()
        min_load = expert_counts.min()

        # 负载均衡损失 (越低越好)
        balance_loss = self.num_experts * np.sum((expert_counts / num_tokens) ** 2)

        return {
            "expert_counts": expert_counts,
            "ideal_per_expert": ideal,
            "max_load": max_load,
            "min_load": min_load,
            "load_ratio": max_load / max(min_load, 1),
            "balance_loss": balance_loss,
        }


class ExpertCapacityRouter(TopKRouter):
    """带 Expert Capacity 限制的路由器"""

    def route(self, gate_logits):
        gates = softmax(gate_logits, axis=-1)
        num_tokens = gates.shape[0]
        capacity = int(self.capacity_factor * num_tokens / self.num_experts)

        expert_counts = np.zeros(self.num_experts, dtype=int)
        routing_weights = []
        expert_indices = []

        for i in range(num_tokens):
            selected = []
            weights = []
            sorted_experts = np.argsort(-gates[i])

            for expert_idx in sorted_experts:
                if len(selected) >= self.top_k:
                    break
                if expert_counts[expert_idx] < capacity:
                    selected.append(expert_idx)
                    weights.append(gates[i, expert_idx])
                    expert_counts[expert_idx] += 1

            if len(selected) == 0:
                # Token dropped! 分配给负载最低的 expert
                min_expert = np.argmin(expert_counts)
                selected = [min_expert]
                weights = [gates[i, min_expert]]
                expert_counts[min_expert] += 1

            # Pad to top_k
            while len(selected) < self.top_k:
                selected.append(selected[0])
                weights.append(0.0)

            total_w = sum(weights)
            if total_w > 0:
                weights = [w / total_w for w in weights]

            routing_weights.append(weights)
            expert_indices.append(selected)

        return np.array(routing_weights), np.array(expert_indices), expert_counts


def simulate_routing(num_experts=8, top_k=2, num_tokens=1024, capacity_factor=1.25,
                     strategy="top_k", seed=42):
    """模拟路由过程并输出分析"""

    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  MoE Router Simulation")
    print(f"{'='*60}")
    print(f"  Experts: {num_experts}, Top-K: {top_k}, Tokens: {num_tokens}")
    print(f"  Strategy: {strategy}, Capacity Factor: {capacity_factor}")

    # 生成随机 gate logits (模拟 token-expert 亲和度)
    gate_logits = np.random.randn(num_tokens, num_experts)

    if strategy == "top_k":
        router = TopKRouter(num_experts, top_k)
        weights, indices = router.route(gate_logits)
        balance = router.analyze_load_balance(indices, num_tokens)
        dropped = 0
    elif strategy == "capacity":
        router = ExpertCapacityRouter(num_experts, top_k, capacity_factor)
        weights, indices, counts = router.route(gate_logits)
        balance = router.analyze_load_balance(indices, num_tokens)
        capacity = int(capacity_factor * num_tokens / num_experts)
        dropped = max(0, int(num_tokens * top_k - sum(min(c, capacity) for c in counts)))

    print(f"\n--- Load Balance Analysis ---")
    print(f"  Ideal tokens/expert: {balance['ideal_per_expert']:.1f}")
    print(f"  Actual distribution:")
    for i, count in enumerate(balance['expert_counts']):
        bar = '█' * int(count / balance['ideal_per_expert'] * 20)
        print(f"    Expert {i}: {count:4d} {bar}")
    print(f"  Max/Min load ratio: {balance['load_ratio']:.2f}")
    print(f"  Balance loss: {balance['balance_loss']:.4f}")
    if strategy == "capacity":
        print(f"  Dropped tokens: {dropped} ({100*dropped/(num_tokens*top_k):.1f}%)")

    # 模拟计算量对比
    print(f"\n--- Compute Comparison ---")
    # 假设每个 expert 是一个 FFN: [hidden, 4*hidden] + [4*hidden, hidden]
    hidden = 4096
    ffn_flops = 2 * hidden * (4 * hidden) * 2  # 两个线性层
    expert_params = hidden * 4 * hidden * 2  # 参数量

    # Dense model: 所有 token 走完整 FFN
    dense_flops = num_tokens * ffn_flops
    dense_params = expert_params  # 只有一组 FFN

    # MoE model: top_k 路由
    moe_flops = num_tokens * top_k * ffn_flops
    moe_params = num_experts * expert_params

    print(f"  Dense FFN params: {dense_params/1e6:.0f}M")
    print(f"  MoE total params: {moe_params/1e6:.0f}M ({num_experts}x)")
    print(f"  Dense FLOPs: {dense_flops/1e9:.2f}G")
    print(f"  MoE FLOPs:    {moe_flops/1e9:.2f}G ({top_k}x per token)")
    print(f"  Compute ratio: {moe_flops/dense_flops:.1f}x params, but only {top_k/num_experts:.0%} active")

    return balance


def compare_strategies(num_experts=8, num_tokens=1024):
    """对比不同路由策略"""
    print(f"\n{'='*60}")
    print(f"  Strategy Comparison (E={num_experts}, T={num_tokens})")
    print(f"{'='*60}")
    print(f"{'Strategy':<20} {'Top-K':<6} {'Balance Loss':<14} {'Load Ratio':<12} {'Dropped'}")
    print("-" * 60)

    for strategy in ["top_k"]:
        for top_k in [1, 2, 4]:
            balance = simulate_routing(num_experts, top_k, num_tokens,
                                       strategy=strategy, seed=42)
            print(f"{strategy:<20} {top_k:<6} {balance['balance_loss']:<14.4f} "
                  f"{balance['load_ratio']:<12.2f} {'N/A'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE Router 模拟器")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--strategy", choices=["top_k", "capacity"], default="top_k")
    args = parser.parse_args()

    simulate_routing(args.num_experts, args.top_k, args.num_tokens,
                     args.capacity_factor, args.strategy)
    print()
    compare_strategies(args.num_experts, args.num_tokens)
