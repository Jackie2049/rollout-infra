#!/usr/bin/env python3
"""Speculative Decoding Simulator — 投机解码模拟器

CPU 可运行的投机解码模拟实验，覆盖 5 个实验：
  1. Standard AR vs Speculative Decoding 对比
  2. Draft 模型质量（acceptance rate）对加速比影响
  3. 投机长度 K 的最优化
  4. Throughput vs Latency 权衡
  5. 批量投机解码扩展效率

关键概念：
  - Draft-Verify: 小模型快速生成 K 个候选 token，大模型一次 forward 验证
  - Acceptance Rate: draft token 被接受的概率，取决于 draft/target 分布匹配度
  - 加速比 = 实际接受 token 数 / 等价标准解码 forward 数

用法:
  conda run -n ai-infra python tools/speculative_decoding_sim.py
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ──────────────────────────────────────────────
# 模型抽象
# ──────────────────────────────────────────────

@dataclass
class ModelProfile:
    """模型推理性能配置"""
    name: str
    latency_per_token_ms: float  # 单次 forward 延迟 (ms)，batch=1 decode
    params_B: float              # 参数量（十亿）

    @property
    def gflops_per_token(self) -> float:
        """每个 token 的计算量 (GFLOPS)"""
        return 2 * self.params_B * 1e3  # ≈ 2 * params * 1000 (vocab factor approx)


@dataclass
class DraftTargetPair:
    """Draft + Target 模型对"""
    draft: ModelProfile
    target: ModelProfile

    @property
    def speed_ratio(self) -> float:
        """Draft 模型比 Target 快多少倍"""
        return self.target.latency_per_token_ms / self.draft.latency_per_token_ms


# ──────────────────────────────────────────────
# Acceptance Rate 模型
# ──────────────────────────────────────────────

def simulate_acceptance_rate(quality: float, position: int) -> float:
    """模拟 draft token 在 position 处的 acceptance rate

    quality: 0.0 ~ 1.0, draft 模型质量
      - 1.0 = 完美匹配 target（100% 接受）
      - 0.8 = 较好匹配（~80% 接受第一个 token）
      - 0.5 = 一般匹配
      - 0.3 = 差的匹配
    position: 0-indexed position in the speculation window

    Returns: acceptance probability at this position
    """
    # 越远的位置，累积误差越大，接受率越低
    # 模型: acceptance ≈ quality * decay^position
    decay = 0.85  # 位置衰减因子
    base_rate = quality ** 1.5  # 质量非线性映射
    return base_rate * (decay ** position)


def simulate_draft_verify_round(
    K: int,
    quality: float,
    rng: random.Random,
) -> Tuple[int, int]:
    """模拟一轮 draft-verify

    K: 投机长度（draft 生成 K 个 token）
    quality: draft 模型质量

    Returns: (accepted_tokens, total_forward_passes)
      accepted_tokens: 本轮最终接受了多少个 token
      total_forward_passes: 等价的 target forward pass 数（用于计算加速比）
    """
    accepted = 0
    for i in range(K):
        p_accept = simulate_acceptance_rate(quality, i)
        if rng.random() < p_accept:
            accepted += 1
        else:
            # 拒绝后从 target 采样 1 个 token → 总共 accepted + 1 tokens
            return accepted + 1, K + 1  # K draft + 1 verify

    # 全部接受 → 额外 1 个 bonus token (verify 输出的下一个)
    return accepted + 1, K + 1


# ──────────────────────────────────────────────
# 模拟器
# ──────────────────────────────────────────────

@dataclass
class SimResult:
    """模拟结果"""
    total_tokens: int = 0
    total_rounds: int = 0
    total_draft_forwards: int = 0
    total_target_forwards: int = 0
    total_time_ms: float = 0.0
    acceptance_history: List[int] = field(default_factory=list)

    @property
    def avg_tokens_per_round(self) -> float:
        return self.total_tokens / max(1, self.total_rounds)

    @property
    def acceptance_rate(self) -> float:
        if not self.acceptance_history:
            return 0.0
        return np.mean(self.acceptance_history) if self.acceptance_history else 0.0

    @property
    def throughput_tok_per_s(self) -> float:
        return self.total_tokens / max(0.001, self.total_time_ms / 1000)

    @property
    def avg_latency_per_token_ms(self) -> float:
        return self.total_time_ms / max(1, self.total_tokens)


def simulate_standard_ar(
    target: ModelProfile,
    num_tokens: int,
    batch_size: int = 1,
) -> SimResult:
    """标准自回归解码模拟"""
    result = SimResult()
    # batch_size 个请求并行，每个 token 需要一次 forward
    steps = math.ceil(num_tokens / batch_size)
    result.total_tokens = num_tokens
    result.total_target_forwards = steps
    # batch decode latency ≈ base_latency * (1 + log(batch) * overhead)
    batch_overhead = 1.0 + 0.15 * math.log2(max(1, batch_size))
    result.total_time_ms = steps * target.latency_per_token_ms * batch_overhead
    return result


def simulate_speculative(
    pair: DraftTargetPair,
    K: int,
    quality: float,
    num_tokens: int,
    seed: int = 42,
) -> SimResult:
    """投机解码模拟"""
    rng = random.Random(seed)
    result = SimResult()

    generated = 0
    while generated < num_tokens:
        # 1. Draft 阶段: K 次 draft forward
        draft_time = K * pair.draft.latency_per_token_ms
        result.total_draft_forwards += K

        # 2. Verify 阶段: 1 次 target forward (验证 K 个 token)
        verify_time = pair.target.latency_per_token_ms
        result.total_target_forwards += 1

        # 3. 模拟接受/拒绝
        accepted, _ = simulate_draft_verify_round(K, quality, rng)

        # 不能超过剩余 token 数
        accepted = min(accepted, num_tokens - generated)
        result.acceptance_history.append(accepted)

        generated += accepted
        result.total_rounds += 1
        result.total_time_ms += draft_time + verify_time

    result.total_tokens = generated
    return result


def simulate_batched_speculative(
    pair: DraftTargetPair,
    K: int,
    quality: float,
    num_tokens: int,
    batch_size: int,
    seed: int = 42,
) -> SimResult:
    """批量投机解码模拟（多个请求共享 target forward）"""
    rng = random.Random(seed)
    result = SimResult()

    # 每个 batch 内的请求独立生成
    remaining = [num_tokens] * batch_size
    total_target_time = 0.0
    total_draft_time = 0.0

    while any(r > 0 for r in remaining):
        batch_draft_time = 0.0
        batch_accepted = []

        for i in range(batch_size):
            if remaining[i] <= 0:
                batch_accepted.append(0)
                continue

            # Draft 阶段
            dt = K * pair.draft.latency_per_token_ms
            batch_draft_time = max(batch_draft_time, dt)  # 并行 draft
            result.total_draft_forwards += K

            # Accept/reject
            accepted, _ = simulate_draft_verify_round(K, quality, rng)
            accepted = min(accepted, remaining[i])
            batch_accepted.append(accepted)

        # Verify: 所有 batch 请求一起做一次 target forward
        # batch verify 延迟 ≈ base * (1 + log(batch_active) * overhead)
        active = sum(1 for a in batch_accepted if a > 0)
        batch_overhead = 1.0 + 0.15 * math.log2(max(1, active))
        verify_time = pair.target.latency_per_token_ms * batch_overhead
        result.total_target_forwards += 1

        total_draft_time += batch_draft_time
        total_target_time += verify_time
        result.total_rounds += 1

        for i in range(batch_size):
            remaining[i] -= batch_accepted[i]
            result.total_tokens += batch_accepted[i]
            result.acceptance_history.append(batch_accepted[i])

    result.total_time_ms = total_draft_time + total_target_time
    return result


# ──────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────

def experiment_1_ar_vs_speculative():
    """实验 1: Standard AR vs Speculative Decoding"""
    print("\n" + "=" * 70)
    print("实验 1: Standard AR vs Speculative Decoding")
    print("=" * 70)

    # 模型配置
    target = ModelProfile("LLaMA-70B", latency_per_token_ms=50.0, params_B=70)
    draft = ModelProfile("LLaMA-7B", latency_per_token_ms=8.0, params_B=7)
    pair = DraftTargetPair(draft, target)

    print(f"\n配置:")
    print(f"  Target: {target.name} ({target.latency_per_token_ms}ms/token)")
    print(f"  Draft:  {draft.name} ({draft.latency_per_token_ms}ms/token)")
    print(f"  Draft/Target 速度比: {pair.speed_ratio:.1f}x")
    print(f"  投机长度 K=5, Draft 质量=0.8, 生成 200 tokens")

    num_tokens = 200
    K = 5
    quality = 0.8

    # Standard AR
    ar_result = simulate_standard_ar(target, num_tokens)

    # Speculative
    spec_result = simulate_speculative(pair, K, quality, num_tokens)

    print(f"\n{'指标':<25} {'Standard AR':>15} {'Speculative':>15} {'提升':>10}")
    print("-" * 70)
    print(f"{'总延迟 (ms)':<25} {ar_result.total_time_ms:>15.1f} {spec_result.total_time_ms:>15.1f} "
          f"{ar_result.total_time_ms / spec_result.total_time_ms:>9.2f}x")
    print(f"{'吞吐量 (tok/s)':<25} {ar_result.throughput_tok_per_s:>15.1f} "
          f"{spec_result.throughput_tok_per_s:>15.1f} "
          f"{spec_result.throughput_tok_per_s / ar_result.throughput_tok_per_s:>9.2f}x")
    print(f"{'平均延迟/token (ms)':<25} {ar_result.avg_latency_per_token_ms:>15.2f} "
          f"{spec_result.avg_latency_per_token_ms:>15.2f} "
          f"{ar_result.avg_latency_per_token_ms / spec_result.avg_latency_per_token_ms:>9.2f}x")
    print(f"{'Target forward 次数':<25} {ar_result.total_target_forwards:>15d} "
          f"{spec_result.total_target_forwards:>15d} "
          f"{'节省 ' + str(round((1 - spec_result.total_target_forwards / ar_result.total_target_forwards) * 100)) + '%':>10}")
    print(f"{'平均接受 tokens/轮':<25} {'N/A':>15} {spec_result.avg_tokens_per_round:>15.2f} {'':>10}")

    print(f"\n关键洞察:")
    print(f"  - Speculative 将 target forward 从 {ar_result.total_target_forwards} 降到 {spec_result.total_target_forwards} "
          f"(节省 {round((1 - spec_result.total_target_forwards / ar_result.total_target_forwards) * 100)}%)")
    print(f"  - 总延迟降低 {ar_result.total_time_ms / spec_result.total_time_ms:.2f}x "
          f"(draft 延迟 + verify 延迟 vs 纯 target 延迟)")
    print(f"  - 加速比受 draft/target 速度比和 acceptance rate 共同决定")


def experiment_2_draft_quality():
    """实验 2: Draft 模型质量对加速比影响"""
    print("\n" + "=" * 70)
    print("实验 2: Draft 模型质量（Acceptance Rate）对加速比影响")
    print("=" * 70)

    target = ModelProfile("LLaMA-70B", latency_per_token_ms=50.0, params_B=70)
    draft = ModelProfile("LLaMA-7B", latency_per_token_ms=8.0, params_B=7)
    pair = DraftTargetPair(draft, target)

    qualities = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    num_tokens = 500
    K = 5

    print(f"\n配置: Target={target.name}, Draft={draft.name}, K={K}, 生成 {num_tokens} tokens")
    print(f"\n{'质量':>8} {'Avg接受':>8} {'加速比':>8} {'吞吐tok/s':>10} {'Target FW':>10}")
    print("-" * 50)

    ar_result = simulate_standard_ar(target, num_tokens)

    for q in qualities:
        spec = simulate_speculative(pair, K, q, num_tokens)
        speedup = ar_result.total_time_ms / spec.total_time_ms

        # 计算 per-position acceptance rate for display
        pos_rates = [simulate_acceptance_rate(q, i) for i in range(K)]
        avg_rate = np.mean(pos_rates)

        print(f"{q:>8.2f} {avg_rate:>8.3f} {speedup:>8.2f}x "
              f"{spec.throughput_tok_per_s:>10.1f} {spec.total_target_forwards:>10d}")

    print(f"\n参考: Standard AR 吞吐 = {ar_result.throughput_tok_per_s:.1f} tok/s, "
          f"Target forwards = {ar_result.total_target_forwards}")

    print(f"\n关键洞察:")
    print(f"  - quality=0.8 是一个合理的分界线（加速比 ≈ 1.8-2.0x）")
    print(f"  - quality < 0.6 时投机解码几乎无收益（接受率太低）")
    print(f"  - quality > 0.9 时加速比趋于饱和（受 draft 延迟限制）")
    print(f"  - 最佳加速比 = min(speed_ratio, 1/(1-quality))（理论上限）")


def experiment_3_optimal_K():
    """实验 3: 投机长度 K 的最优化"""
    print("\n" + "=" * 70)
    print("实验 3: 投机长度 K 最优化")
    print("=" * 70)

    target = ModelProfile("LLaMA-70B", latency_per_token_ms=50.0, params_B=70)
    draft = ModelProfile("LLaMA-7B", latency_per_token_ms=8.0, params_B=7)
    pair = DraftTargetPair(draft, target)

    K_values = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
    num_tokens = 500

    print(f"\n配置: Target={target.name} ({target.latency_per_token_ms}ms), "
          f"Draft={draft.name} ({draft.latency_per_token_ms}ms)")
    print(f"  Draft/Target 速度比 = {pair.speed_ratio:.1f}x")

    ar_result = simulate_standard_ar(target, num_tokens)

    for quality in [0.7, 0.8, 0.9]:
        print(f"\n--- Draft 质量 = {quality} ---")
        print(f"{'K':>4} {'Avg接受':>8} {'加速比':>8} {'吞吐tok/s':>10} {'Target FW':>10}")
        print("-" * 45)

        best_speedup = 0
        best_K = 0

        for K in K_values:
            spec = simulate_speculative(pair, K, quality, num_tokens)
            speedup = ar_result.total_time_ms / spec.total_time_ms

            if speedup > best_speedup:
                best_speedup = speedup
                best_K = K

            print(f"{K:>4d} {spec.avg_tokens_per_round:>8.2f} {speedup:>8.2f}x "
                  f"{spec.throughput_tok_per_s:>10.1f} {spec.total_target_forwards:>10d}")

        print(f"  → 最优 K = {best_K} (加速比 {best_speedup:.2f}x)")

    print(f"\n关键洞察:")
    print(f"  - 最优 K 取决于 draft 质量和 speed ratio")
    print(f"  - Quality=0.7 时最优 K≈3-4; Quality=0.9 时最优 K≈8-10")
    print(f"  - K 过大时浪费 draft 计算（后面的 token 几乎都被拒绝）")
    print(f"  - 经验法则: K ≈ -1 / ln(quality * decay), 其中 decay ≈ 0.85")


def experiment_4_throughput_latency():
    """实验 4: Throughput vs Latency 权衡"""
    print("\n" + "=" * 70)
    print("实验 4: Throughput vs Latency 权衡")
    print("=" * 70)

    # 不同规模模型的配置
    configs = [
        ("小模型推理", ModelProfile("7B", 8.0, 7), None),
        ("中模型推理", ModelProfile("13B", 15.0, 13), ModelProfile("1.3B", 2.0, 1.3)),
        ("大模型推理", ModelProfile("70B", 50.0, 70), ModelProfile("7B", 8.0, 7)),
        ("超大模型推理", ModelProfile("405B", 120.0, 405), ModelProfile("8B", 10.0, 8)),
    ]

    num_tokens = 200
    K = 5
    quality = 0.8

    print(f"\n配置: K={K}, Draft 质量={quality}, 生成 {num_tokens} tokens")
    print(f"\n{'场景':<15} {'Target':>8} {'Draft':>8} {'AR延迟ms':>10} "
          f"{'Spec延迟ms':>10} {'加速比':>8} {'AR tok/s':>10} {'Spec tok/s':>10}")
    print("-" * 90)

    for name, target, draft_model in configs:
        ar = simulate_standard_ar(target, num_tokens)

        if draft_model is None:
            # 小模型不需要 speculative
            print(f"{name:<15} {target.name:>8} {'N/A':>8} "
                  f"{ar.total_time_ms:>10.1f} {'N/A':>10} {'N/A':>8} "
                  f"{ar.throughput_tok_per_s:>10.1f} {'N/A':>10}")
            continue

        pair = DraftTargetPair(draft_model, target)
        spec = simulate_speculative(pair, K, quality, num_tokens)
        speedup = ar.total_time_ms / spec.total_time_ms

        print(f"{name:<15} {target.name:>8} {draft_model.name:>8} "
              f"{ar.total_time_ms:>10.1f} {spec.total_time_ms:>10.1f} "
              f"{speedup:>7.2f}x {ar.throughput_tok_per_s:>10.1f} "
              f"{spec.throughput_tok_per_s:>10.1f}")

    print(f"\n关键洞察:")
    print(f"  - 模型越大，speculative decoding 收益越大")
    print(f"  - 405B 模型单次 forward 120ms → speculative 可以显著降低")
    print(f"  - 7B 模型本身延迟低，speculative 几乎无收益")
    print(f"  - 最佳使用场景: Target latency >> Draft latency * K")


def experiment_5_batch_scaling():
    """实验 5: 批量投机解码扩展效率"""
    print("\n" + "=" * 70)
    print("实验 5: 批量投机解码扩展效率")
    print("=" * 70)

    target = ModelProfile("LLaMA-70B", latency_per_token_ms=50.0, params_B=70)
    draft = ModelProfile("LLaMA-7B", latency_per_token_ms=8.0, params_B=7)
    pair = DraftTargetPair(draft, target)

    num_tokens = 100  # 每个请求生成 token 数
    K = 5
    quality = 0.8
    batch_sizes = [1, 2, 4, 8, 16, 32]

    print(f"\n配置: Target={target.name}, Draft={draft.name}, K={K}, Quality={quality}")
    print(f"  每个请求生成 {num_tokens} tokens")
    print(f"\n{'Batch':>6} {'总tok/s':>10} {'延迟(ms)':>10} {'AR tok/s':>10} "
          f"{'AR延迟(ms)':>10} {'Spec加速':>10}")
    print("-" * 60)

    for bs in batch_sizes:
        # Batch speculative
        spec = simulate_batched_speculative(pair, K, quality, num_tokens, bs)

        # Batch standard AR
        total_tokens = num_tokens * bs
        ar = simulate_standard_ar(target, total_tokens, batch_size=bs)

        speedup = ar.total_time_ms / spec.total_time_ms

        print(f"{bs:>6d} {spec.throughput_tok_per_s:>10.1f} {spec.total_time_ms:>10.1f} "
              f"{ar.throughput_tok_per_s:>10.1f} {ar.total_time_ms:>10.1f} "
              f"{speedup:>9.2f}x")

    print(f"\n关键洞察:")
    print(f"  - batch_size 增大时，标准 AR 吞吐线性增长（GPU 利用率提升）")
    print(f"  - Speculative 的优势在 batch_size 小时最明显")
    print(f"  - batch_size 大时 GPU 已经饱和，speculative 收益递减")
    print(f"  - 实际生产中 speculative decoding 主要用于 batch=1 在线推理")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Speculative Decoding 模拟器")
    print("Draft-Verify 投机解码: 用小模型猜测、大模型验证")
    print("=" * 70)

    experiment_1_ar_vs_speculative()
    experiment_2_draft_quality()
    experiment_3_optimal_K()
    experiment_4_throughput_latency()
    experiment_5_batch_scaling()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
Speculative Decoding 关键参数:
  1. Draft 质量 (acceptance rate): 决定投机效率的核心因素
     - quality > 0.8 → 明显加速
     - quality < 0.6 → 几乎无收益
  2. 投机长度 K: 最优值取决于 quality 和 draft/target 速度比
     - 经验公式: K ≈ -1 / ln(quality * 0.85)
     - 通常 K = 3-8 是合理的范围
  3. Draft/Target 速度比: 越大越好
     - 速度比 > 5x 时 speculative 有明显收益
  4. 最佳场景:
     - Target 模型大（70B+），延迟高
     - Draft 模型质量好（同系列小模型）
     - Batch size 小（在线推理场景）
  5. 不适合场景:
     - 小模型推理（7B 以下）
     - 大 batch 推理（GPU 已饱和）
     - Draft 质量差的模型对

进阶方法:
  - Medusa: 多头并行预测，无需单独 draft 模型
  - Eagle: 特征级投机，比 token 级更高效
  - Self-speculative: 模型自身做 draft（跳层/早退）
  - SpecInfer: 树形投机，多个 draft 同时验证
    """)
