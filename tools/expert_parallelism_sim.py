"""Expert Parallelism (EP) 模拟器 — Token Routing 与负载均衡分析

模拟 MoE 模型中的 Expert Parallelism:
1. Token routing: Top-K 路由 + All-to-All 通信模拟
2. 负载均衡分析: 不同 routing 策略的 expert 负载分布
3. EP vs DP/TP: 不同并行策略的通信量对比
4. DeepSeek-V3 模式: Shared + Routed experts

使用方法:
    python expert_parallelism_sim.py   # CPU 可运行
"""

import numpy as np
import time
from collections import defaultdict


class ExpertParallelismSimulator:
    """Expert Parallelism 模拟器。"""

    def __init__(self, num_experts=8, num_gpus=4, top_k=2, hidden=768,
                 ffn_dim=3072, num_tokens=1024):
        self.num_experts = num_experts
        self.num_gpus = num_gpus
        self.experts_per_gpu = num_experts // num_gpus
        self.top_k = top_k
        self.hidden = hidden
        self.ffn_dim = ffn_dim
        self.num_tokens = num_tokens

    def simulate_routing(self, strategy="top_k"):
        """模拟 token routing 过程。"""
        np.random.seed(42)

        # 生成 router logits (每个 token 对每个 expert 的分数)
        router_logits = np.random.randn(self.num_tokens, self.num_experts)

        # Softmax
        exp_logits = np.exp(router_logits - router_logits.max(axis=1, keepdims=True))
        router_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        if strategy == "top_k":
            # 标准 Top-K routing
            top_k_indices = np.argsort(router_probs, axis=1)[:, -self.top_k:]
            top_k_probs = np.take_along_axis(router_probs, top_k_indices, axis=1)
            top_k_probs = top_k_probs / top_k_probs.sum(axis=1, keepdims=True)
            return top_k_indices, top_k_probs

        elif strategy == "capacity":
            # Capacity Factor routing (限制每个 expert 的 token 数)
            capacity = int(self.num_tokens / self.num_experts * 1.25)  # 1.25 capacity factor
            expert_counts = np.zeros(self.num_experts, dtype=int)
            assignments = []

            for token_idx in range(self.num_tokens):
                probs = router_probs[token_idx].copy()
                assigned = []

                for k in range(self.top_k):
                    while True:
                        best = np.argmax(probs)
                        if expert_counts[best] < capacity:
                            assigned.append(best)
                            expert_counts[best] += 1
                            probs[best] = 0
                            break
                        else:
                            probs[best] = 0
                            if probs.max() == 0:
                                assigned.append(-1)  # Dropped
                                break

                assignments.append(assigned)

            return np.array(assignments), None

        elif strategy == "dropless":
            # Dropless routing (保证所有 token 都被处理)
            # 使用动态 batch allocation
            expert_assignments = defaultdict(list)
            token_expert_map = []

            for token_idx in range(self.num_tokens):
                top_k_idx = np.argsort(router_probs[token_idx])[-self.top_k:]
                token_expert_map.append(top_k_idx)
                for expert in top_k_idx:
                    expert_assignments[int(expert)].append(token_idx)

            return token_expert_map, expert_assignments

    def analyze_load_balance(self, assignments):
        """分析 expert 负载均衡情况。"""
        expert_counts = np.zeros(self.num_experts)

        for token_idx in range(self.num_tokens):
            for expert in assignments[token_idx]:
                if expert >= 0:
                    expert_counts[expert] += 1

        total = expert_counts.sum()
        ideal = total / self.num_experts
        max_load = expert_counts.max()
        min_load = expert_counts[expert_counts > 0].min()

        # Load balance metrics
        load_ratio = max_load / ideal if ideal > 0 else float('inf')
        balance_loss = np.sum((expert_counts / total - 1.0 / self.num_experts) ** 2)

        return {
            'expert_counts': expert_counts,
            'ideal': ideal,
            'max': max_load,
            'min': min_load,
            'load_ratio': load_ratio,
            'balance_loss': balance_loss,
        }

    def simulate_all_to_all(self, assignments):
        """模拟 All-to-All 通信。"""
        # 每个 GPU 发送到每个其他 GPU 的 token 数
        gpu_send = np.zeros((self.num_gpus, self.num_gpus), dtype=int)

        for token_idx in range(self.num_tokens):
            src_gpu = token_idx % self.num_gpus  # Token 初始分配的 GPU
            for expert in assignments[token_idx]:
                if expert >= 0:
                    dst_gpu = expert // self.experts_per_gpu
                    gpu_send[src_gpu][dst_gpu] += 1

        # All-to-All 通信量
        # 每对 GPU 之间传输的数据量 (tokens × hidden × bytes)
        bytes_per_token = self.hidden * 2  # FP16
        comm_matrix = gpu_send * bytes_per_token / 1e6  # MB

        return {
            'token_matrix': gpu_send,
            'comm_mb': comm_matrix,
            'total_mb': comm_matrix.sum(),
            'max_pair_mb': comm_matrix.max(),
        }

    def simulate_ep_compute(self):
        """模拟 EP 的计算过程。"""
        # 每个 GPU 处理的 expert 数
        experts_per_gpu = self.experts_per_gpu

        # 每个 expert 的 FFN 计算量
        # gate_proj: hidden × ffn_dim, up_proj: hidden × ffn_dim, down_proj: ffn_dim × hidden
        flops_per_token_per_expert = 2 * (self.hidden * self.ffn_dim * 3)  # 3 projections

        # 理想情况: 每个 expert 处理相同数量 tokens
        tokens_per_expert = self.num_tokens * self.top_k / self.num_experts
        total_flops = tokens_per_expert * flops_per_token_per_expert * experts_per_gpu

        return total_flops / 1e9  # GFLOPS

    def compare_parallel_strategies(self):
        """对比 EP vs TP vs DP 的通信量。"""
        token_bytes = self.hidden * 2  # FP16 per token

        results = {}

        # Data Parallelism (DP)
        # 每个 GPU 有完整模型，只需要梯度 AllReduce
        # 通信量: 模型参数量 × 2 (gradient + broadcast)
        model_params = self.num_experts * self.hidden * self.ffn_dim * 3
        dp_comm = model_params * 4 * 2 / self.num_gpus  # FP32 gradient, split across GPUs
        results['DP'] = {
            'comm_per_step_mb': dp_comm / 1e6,
            'description': f'AllReduce gradient ({model_params/1e6:.0f}M params ÷ {self.num_gpus} GPUs)',
        }

        # Tensor Parallelism (TP)
        # 每层需要 2x AllReduce (attention + FFN)
        # 对于 MoE FFN: 每个 expert 的 FFN 被切分
        # 通信量: 2 × tokens × hidden × 2 (send + recv) per layer
        tp_comm = 2 * self.num_tokens * token_bytes * 2  # per AllReduce
        results['TP'] = {
            'comm_per_step_mb': tp_comm / 1e6 * 2,  # 2 AllReduce per layer
            'description': f'2× AllReduce ({self.num_tokens} tokens × {self.hidden} hidden)',
        }

        # Expert Parallelism (EP)
        # All-to-All: 发送 tokens 到对应 expert 所在 GPU，处理完后收回
        # 通信量: tokens × hidden × 2 (dispatch + combine)
        ep_comm = self.num_tokens * self.top_k * token_bytes * 2  # dispatch + combine
        results['EP'] = {
            'comm_per_step_mb': ep_comm / 1e6,
            'description': f'All-to-All ({self.num_tokens} tokens × top-{self.top_k})',
        }

        return results

    def simulate_deepseek_v3(self):
        """模拟 DeepSeek-V3 的 EP 模式。"""
        # DeepSeek-V3 配置 (简化)
        num_shared_experts = 1  # 共享 expert (所有 token 都经过)
        num_routed_experts = 256  # 路由 expert
        top_k_routed = 8  # 每个 token 路由到 8 个 expert
        num_gpus = 64  # EP 用 64 GPU

        print(f"\n  DeepSeek-V3 EP 配置:")
        print(f"    Shared experts: {num_shared_experts}")
        print(f"    Routed experts: {num_routed_experts}")
        print(f"    Top-K (routed): {top_k_routed}")
        print(f"    EP GPUs: {num_gpus}")
        print(f"    Experts/GPU: {num_routed_experts // num_gpus}")

        # 每个 GPU 的 expert 分配
        experts_per_gpu = num_routed_experts // num_gpus

        # Token routing 模拟
        np.random.seed(42)
        num_tokens = 4096  # 一个 batch 的 token 数
        hidden = 7168  # DeepSeek-V3 hidden size

        # Router logits
        router_logits = np.random.randn(num_tokens, num_routed_experts)
        exp_logits = np.exp(router_logits - router_logits.max(axis=1, keepdims=True))
        router_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Top-K routing
        top_k_indices = np.argsort(router_probs, axis=1)[:, -top_k_routed:]

        # 分析每个 GPU 的负载
        gpu_counts = np.zeros(num_gpus, dtype=int)
        for token_idx in range(num_tokens):
            for expert in top_k_indices[token_idx]:
                gpu = expert // experts_per_gpu
                gpu_counts[gpu] += 1

        ideal = num_tokens * top_k_routed / num_gpus
        max_load = gpu_counts.max()
        min_load = gpu_counts.min()

        print(f"\n  Token routing 负载分析 (num_tokens={num_tokens}):")
        print(f"    每个 GPU 理想 token 数: {ideal:.0f}")
        print(f"    最大负载: {max_load} ({max_load/ideal:.2f}x)")
        print(f"    最小负载: {min_load} ({min_load/ideal:.2f}x)")
        print(f"    负载比: {max_load/min_load:.2f}x")

        # 通信量
        bytes_per_token = hidden * 2  # FP16
        dispatch_comm = num_tokens * top_k_routed * bytes_per_token / 1e9  # GB
        combine_comm = dispatch_comm  # Same size for combine
        total_comm = (dispatch_comm + combine_comm) * 2  # Send + recv

        print(f"\n  All-to-All 通信量:")
        print(f"    Dispatch: {dispatch_comm:.2f} GB")
        print(f"    Combine:  {combine_comm:.2f} GB")
        print(f"    总计:     {total_comm:.2f} GB")

        # Shared expert 的计算
        shared_flops = num_tokens * hidden * hidden * 4 * 3  # 3 projections, 4x expansion
        routed_flops_per_expert = (num_tokens * top_k_routed / num_routed_experts) * hidden * hidden * 4 * 3
        total_routed_flops = routed_flops_per_expert * num_routed_experts

        print(f"\n  计算量分析:")
        print(f"    Shared expert: {shared_flops/1e12:.2f} TFLOPS")
        print(f"    Routed experts (total): {total_routed_flops/1e12:.2f} TFLOPS")
        print(f"    比例: shared={shared_flops/(shared_flops+total_routed_flops)*100:.1f}%, routed={total_routed_flops/(shared_flops+total_routed_flops)*100:.1f}%")


def main():
    print("=" * 60)
    print("Expert Parallelism (EP) 模拟器")
    print("=" * 60)

    # ============================================================
    # 实验 1: Routing 策略对比
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 1: Routing 策略对比 (8 experts, Top-2)")
    print("=" * 60)

    sim = ExpertParallelismSimulator(
        num_experts=8, num_gpus=4, top_k=2,
        hidden=768, ffn_dim=3072, num_tokens=1024
    )

    strategies = ["top_k", "capacity", "dropless"]
    print(f"\n{'策略':<15} {'负载比':>8} {'均衡损失':>12} {'Max':>6} {'Min':>6} {'Ideal':>6}")
    print("-" * 55)

    for strategy in strategies:
        result = sim.simulate_routing(strategy)

        if strategy == "top_k":
            assignments = result[0]
        elif strategy == "capacity":
            assignments = result[0]
        else:
            token_expert_map = result[0]
            assignments = np.array(token_expert_map)

        balance = sim.analyze_load_balance(assignments)
        print(f"{strategy:<15} {balance['load_ratio']:>8.2f} {balance['balance_loss']:>12.6f} "
              f"{balance['max']:>6.0f} {balance['min']:>6.0f} {balance['ideal']:>6.0f}")

    # Top-K 详细负载分布
    print(f"\n  Top-K 各 Expert 负载:")
    top_k_indices, _ = sim.simulate_routing("top_k")
    balance = sim.analyze_load_balance(top_k_indices)
    for i, count in enumerate(balance['expert_counts']):
        gpu = i // (sim.num_experts // sim.num_gpus)
        bar = "█" * int(count / balance['max'] * 30)
        print(f"    Expert {i} (GPU {gpu}): {count:>5.0f} tokens {bar}")

    # ============================================================
    # 实验 2: All-to-All 通信分析
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 2: All-to-All 通信分析 (4 GPUs, 8 Experts)")
    print("=" * 60)

    top_k_indices, _ = sim.simulate_routing("top_k")
    a2a = sim.simulate_all_to_all(top_k_indices)

    print(f"\n  GPU→GPU Token 传输矩阵:")
    print(f"  {'':>6}", end="")
    for g in range(sim.num_gpus):
        print(f" {'GPU'+str(g):>6}", end="")
    print()
    for src in range(sim.num_gpus):
        print(f"  GPU{src}:", end="")
        for dst in range(sim.num_gpus):
            print(f" {a2a['token_matrix'][src][dst]:>6}", end="")
        print()

    print(f"\n  通信量统计:")
    print(f"    总传输: {a2a['total_mb']:.2f} MB")
    print(f"    最大 GPU 对: {a2a['max_pair_mb']:.2f} MB")
    print(f"    平均每 GPU: {a2a['total_mb']/sim.num_gpus:.2f} MB")

    # ============================================================
    # 实验 3: EP vs DP vs TP 通信量对比
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 3: EP vs TP vs DP 通信量对比")
    print("=" * 60)

    strategies_comm = sim.compare_parallel_strategies()
    print()
    for name, info in strategies_comm.items():
        print(f"  {name}: {info['comm_per_step_mb']:.2f} MB/layer — {info['description']}")

    # 不同规模的对比
    print(f"\n  {'Config':<30} {'DP (MB)':>10} {'TP (MB)':>10} {'EP (MB)':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    configs = [
        (8, 4, 2, 768, 3072, 1024, "Small (8E, 4GPU)"),
        (64, 8, 4, 4096, 11008, 2048, "Medium (64E, 8GPU)"),
        (256, 64, 8, 7168, 18432, 4096, "Large (256E, 64GPU)"),
    ]

    for ne, ng, tk, h, ff, nt, label in configs:
        s = ExpertParallelismSimulator(ne, ng, tk, h, ff, nt)
        r = s.compare_parallel_strategies()
        print(f"  {label:<30} {r['DP']['comm_per_step_mb']:>10.1f} "
              f"{r['TP']['comm_per_step_mb']:>10.1f} {r['EP']['comm_per_step_mb']:>10.1f}")

    print(f"""
关键洞察:
  - DP 通信量 ∝ 模型参数量 / GPU数 (与 expert 数线性相关)
  - TP 通信量 ∝ tokens × hidden (每层固定，与 expert 数无关)
  - EP 通信量 ∝ tokens × top_k × hidden (All-to-All)
  - 大 MoE (256 experts): DP 通信量远大于 EP，EP 更优
  - 小 MoE (8 experts): EP 通信量可能超过 TP
  → EP 适合 expert 数多、模型大的场景
    """)

    # ============================================================
    # 实验 4: DeepSeek-V3 EP 模式模拟
    # ============================================================
    print("=" * 60)
    print("实验 4: DeepSeek-V3 EP 模式模拟")
    print("=" * 60)

    sim_dsv3 = ExpertParallelismSimulator(
        num_experts=256, num_gpus=64, top_k=8,
        hidden=7168, ffn_dim=18432, num_tokens=4096
    )
    sim_dsv3.simulate_deepseek_v3()

    # ============================================================
    # 实验 5: EP 扩展效率
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 5: EP 扩展效率 (固定 64 experts, 变化 GPU 数)")
    print("=" * 60)

    num_experts = 64
    hidden = 4096
    ffn_dim = 11008
    num_tokens = 2048
    top_k = 4

    print(f"\n  {'GPU数':>6} {'Expert/GPU':>12} {'通信量(MB)':>12} {'计算(GFLOP)':>14} {'效率':>8}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*14} {'-'*8}")

    base_compute = None
    for num_gpus in [2, 4, 8, 16, 32, 64]:
        experts_per_gpu = num_experts // num_gpus
        if experts_per_gpu < 1:
            continue

        # 每个GPU的计算量
        tokens_per_expert = num_tokens * top_k / num_experts
        flops_per_expert = tokens_per_expert * hidden * ffn_dim * 3 * 2  # 3 projections
        compute_per_gpu = flops_per_expert * experts_per_gpu / 1e9  # GFLOPS

        # All-to-All 通信量
        token_bytes = hidden * 2
        comm = num_tokens * top_k * token_bytes * 2 / 1e6  # MB

        if base_compute is None:
            base_compute = compute_per_gpu * num_gpus

        total_compute = compute_per_gpu * num_gpus
        efficiency = total_compute / base_compute

        print(f"  {num_gpus:>6} {experts_per_gpu:>12} {comm:>12.2f} {compute_per_gpu:>14.2f} {efficiency:>8.2f}x")

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
Expert Parallelism 核心知识:

1. 基本原理:
   将不同的 expert 放在不同的 GPU 上
   通过 All-to-All 通信将 token 路由到对应 expert
   处理完后通过 All-to-All 收回结果

2. 通信模式:
   Dispatch All-to-All: tokens → expert GPUs (发送)
   Combine All-to-All: expert GPUs → tokens (收回)
   通信量 ∝ tokens × top_k × hidden × 2

3. 与 TP/DP 的选择:
   小模型 (8 experts): TP 可能更优 (通信量小)
   大模型 (64+ experts): EP 更优 (通信量可控，计算更均衡)
   超大模型 (256 experts): EP 必需 (DP 通信量太大)

4. DeepSeek-V3 创新:
   Shared expert (所有 token 都经过) + Routed experts
   256 routed experts + 1 shared expert
   64 GPU EP, 每个 GPU 4 routed experts
   Group-level routing 减少通信量

5. 负载均衡挑战:
   Top-K routing 可能导致某些 expert 过载
   解决: capacity factor, dropless routing, auxiliary loss
   实际生产中负载比通常 1.2-1.5x
    """)


if __name__ == "__main__":
    main()
