#!/usr/bin/env python3
"""RLHF/GRPO 训练模拟器

模拟 RLHF 训练循环: Rollout → Reward → Advantage → Actor/Critic Update
覆盖 5 个实验:
  1. PPO vs GRPO: 两种 RL 对齐算法对比
  2. Rollout Batch Size: 推理 batch 对训练效率的影响
  3. KL 惩罚影响: KL 系数对 reward hacking 的抑制作用
  4. Prefix Caching 收益: GRPO 场景下的 KV cache 复用
  5. 分布式扩展: Actor/Critic/Reward 不同并行策略

基于 verl 架构理解:
  - RayPPOTrainer: 训练主循环
  - ActorRolloutRefWorker: 混合 Worker (训练+推理)
  - PrefixGrouper: RL prompt 分组复用

Usage:
    conda run -n ai-infra python tools/rlhf_training_simulator.py
    conda run -n ai-infra python tools/rlhf_training_simulator.py -e 3
"""

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Prompt:
    """训练 prompt"""
    prompt_id: int
    prompt_text: str
    prompt_tokens: int
    ground_truth: str = ""
    group_id: int = 0  # GRPO: 同一 prompt 的多个 response


@dataclass
class Response:
    """模型生成的 response"""
    prompt_id: int
    response_tokens: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    reward: float = 0.0
    value: float = 0.0
    advantage: float = 0.0
    kl_divergence: float = 0.0


@dataclass
class TrainingConfig:
    """训练配置"""
    # Model
    model_params_b: float = 7.0
    # PPO
    clip_range: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    # GRPO
    grpo_n: int = 8  # responses per prompt
    # Training
    learning_rate: float = 1e-6
    ppo_epochs: int = 1
    mini_batch_size: int = 64
    # RL
    kl_coeff: float = 0.02
    reward_scale: float = 1.0
    # Infra
    rollout_batch_size: int = 256
    max_response_tokens: int = 512
    use_prefix_caching: bool = False
    prompt_tokens_avg: int = 256


# ============================================================
# Simulated Models
# ============================================================

class SimRewardModel:
    """模拟 Reward Model"""

    def __init__(self, base_quality: float = 0.5, noise: float = 0.1):
        self.base_quality = base_quality
        self.noise = noise

    def compute_reward(self, prompt: Prompt, response_len: int) -> float:
        """计算 reward

        真实场景: RM 对 prompt+response 打分
        模拟: 基于长度和随机噪声的简化 reward
        """
        # Longer responses tend to have slightly higher reward (up to a point)
        length_bonus = min(response_len / 512, 1.0) * 0.2
        # Random quality component
        quality = self.base_quality + random.gauss(0, self.noise)
        reward = max(0, min(1, quality + length_bonus))
        return reward


class SimActor:
    """模拟 Actor (Policy Model)"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.quality = 0.3  # 初始质量较低
        self.steps = 0

    def generate(self, prompts: list[Prompt]) -> list[Response]:
        """生成 responses (rollout)"""
        responses = []
        for prompt in prompts:
            # Response length influenced by quality
            avg_len = int(self.config.max_response_tokens *
                          (0.3 + 0.7 * self.quality))
            length = max(10, min(self.config.max_response_tokens,
                                 int(random.gauss(avg_len, 50))))

            tokens = [random.randint(0, 31999) for _ in range(length)]
            log_probs = [random.gauss(-2, 1) for _ in range(length)]

            responses.append(Response(
                prompt_id=prompt.prompt_id,
                response_tokens=tokens,
                log_probs=log_probs,
            ))
        return responses

    def compute_log_prob(self, response: Response) -> list[float]:
        """重新计算 log prob (after update)"""
        quality_factor = 1.0 - self.quality * 0.3  # Better policy = less entropy
        return [lp * (1 + quality_factor) for lp in response.log_probs]

    def update(self, responses: list[Response], old_log_probs: list[list[float]]):
        """PPO/GRPO 策略更新"""
        # Simulate quality improvement
        avg_advantage = sum(r.advantage for r in responses) / len(responses)
        improvement = 0.01 * (1 + max(0, avg_advantage))
        self.quality = min(0.95, self.quality + improvement)
        self.steps += 1


class SimCritic:
    """模拟 Critic (Value Model)"""

    def __init__(self):
        self.quality = 0.3

    def compute_values(self, responses: list[Response]) -> list[float]:
        """估计 value function"""
        values = []
        for resp in responses:
            v = self.quality + random.gauss(0, 0.05)
            values.append(max(0, min(1, v)))
        return values

    def update(self, responses: list[Response]):
        """更新 value function"""
        if responses:
            avg_reward = sum(r.reward for r in responses) / len(responses)
            self.quality = min(0.95, self.quality + 0.01 * abs(avg_reward - self.quality))


class SimRefModel:
    """模拟 Reference Model (for KL penalty)"""

    def __init__(self):
        self.quality = 0.3  # Frozen at initial quality

    def compute_kl(self, response: Response, actor_quality: float) -> float:
        """计算 KL(π_θ || π_ref)"""
        # Simulate: as actor diverges from ref, KL increases
        divergence = abs(actor_quality - self.quality)
        return divergence * len(response.response_tokens) * 0.001


# ============================================================
# Training Loop
# ============================================================

def compute_gae(responses: list[Response], gamma: float = 1.0,
                lam: float = 0.95) -> list[Response]:
    """Generalized Advantage Estimation"""
    for resp in responses:
        # Simplified GAE: advantage = reward - value + future
        resp.advantage = resp.reward - resp.value + gamma * lam * max(0, resp.reward - resp.value)
    return responses


def compute_grpo_advantage(responses: list[Response], n: int) -> list[Response]:
    """GRPO: group relative advantage"""
    # Group responses by prompt_id
    by_prompt: dict[int, list[Response]] = {}
    for resp in responses:
        by_prompt.setdefault(resp.prompt_id, []).append(resp)

    for prompt_id, group in by_prompt.items():
        rewards = [r.reward - r.kl_divergence for r in group]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8

        for resp, r in zip(group, rewards):
            resp.advantage = (r - mean_r) / std_r

    return responses


def run_training(config: TrainingConfig, algorithm: str = "ppo",
                 num_iterations: int = 100) -> dict:
    """运行 RLHF 训练模拟"""
    random.seed(42)

    actor = SimActor(config)
    critic = SimCritic() if algorithm == "ppo" else None
    ref_model = SimRefModel()
    reward_model = SimRewardModel()

    metrics = {
        'rewards': [],
        'advantages': [],
        'kl_divs': [],
        'actor_quality': [],
        'values': [],
        'tokens_per_step': [],
        'steps': [],
        'rollout_time_pct': [],
        'training_time_pct': [],
    }

    for iteration in range(num_iterations):
        # 1. Sample prompts
        n_prompts = config.rollout_batch_size
        if algorithm == "grpo":
            n_prompts = config.rollout_batch_size // config.grpo_n

        prompts = []
        for i in range(n_prompts):
            p = Prompt(
                prompt_id=i,
                prompt_text=f"prompt_{i}",
                prompt_tokens=int(random.gauss(config.prompt_tokens_avg, 50)),
                group_id=i if algorithm == "grpo" else 0,
            )
            prompts.append(p)

        # GRPO: duplicate prompts for n responses
        if algorithm == "grpo":
            expanded_prompts = []
            for p in prompts:
                for j in range(config.grpo_n):
                    expanded_prompts.append(Prompt(
                        prompt_id=p.prompt_id,
                        prompt_text=p.prompt_text,
                        prompt_tokens=p.prompt_tokens,
                        group_id=p.prompt_id,
                    ))
            rollout_prompts = expanded_prompts
        else:
            rollout_prompts = prompts

        # 2. Rollout (generate responses)
        responses = actor.generate(rollout_prompts)

        # 3. Compute rewards
        for resp in responses:
            resp.reward = reward_model.compute_reward(
                prompts[resp.prompt_id], len(resp.response_tokens)
            )

        # 4. Compute KL divergence
        for resp in responses:
            resp.kl_divergence = ref_model.compute_kl(resp, actor.quality)
            resp.reward -= config.kl_coeff * resp.kl_divergence

        # 5. Compute advantages
        if algorithm == "ppo":
            values = critic.compute_values(responses)
            for resp, v in zip(responses, values):
                resp.value = v
            responses = compute_gae(responses, config.gamma, config.gae_lambda)
            critic.update(responses)
        else:  # GRPO
            responses = compute_grpo_advantage(responses, config.grpo_n)

        # 6. Update actor
        old_log_probs = [r.log_probs[:] for r in responses]
        actor.update(responses, old_log_probs)

        # Record metrics
        avg_reward = sum(r.reward for r in responses) / len(responses)
        avg_adv = sum(abs(r.advantage) for r in responses) / len(responses)
        avg_kl = sum(r.kl_divergence for r in responses) / len(responses)
        total_tokens = sum(len(r.response_tokens) for r in responses)

        # Time estimation
        rollout_time_pct = 0.4 + 0.1 * (total_tokens / (config.rollout_batch_size * config.max_response_tokens))
        rollout_time_pct = min(0.7, rollout_time_pct)

        metrics['rewards'].append(avg_reward)
        metrics['advantages'].append(avg_adv)
        metrics['kl_divs'].append(avg_kl)
        metrics['actor_quality'].append(actor.quality)
        metrics['tokens_per_step'].append(total_tokens)
        metrics['steps'].append(iteration)
        metrics['rollout_time_pct'].append(rollout_time_pct)
        metrics['training_time_pct'].append(1 - rollout_time_pct)

        if critic:
            avg_val = sum(r.value for r in responses) / len(responses)
            metrics['values'].append(avg_val)

    return metrics


# ============================================================
# Experiments
# ============================================================

def experiment_1_ppo_vs_grpo():
    """实验 1: PPO vs GRPO 对比"""
    print("\n" + "=" * 70)
    print("实验 1: PPO vs GRPO — 两种 RL 对齐算法对比")
    print("=" * 70)

    config = TrainingConfig()
    num_iters = 50

    ppo_metrics = run_training(config, "ppo", num_iters)
    grpo_metrics = run_training(config, "grpo", num_iters)

    print(f"\n{'Metric':<20} {'PPO':<15} {'GRPO':<15} {'Difference':<15}")
    print("-" * 65)
    print(f"{'Final Reward':<20} {ppo_metrics['rewards'][-1]:<15.4f} "
          f"{grpo_metrics['rewards'][-1]:<15.4f} "
          f"{grpo_metrics['rewards'][-1]-ppo_metrics['rewards'][-1]:+.4f}")
    print(f"{'Final Quality':<20} {ppo_metrics['actor_quality'][-1]:<15.4f} "
          f"{grpo_metrics['actor_quality'][-1]:<15.4f} "
          f"{grpo_metrics['actor_quality'][-1]-ppo_metrics['actor_quality'][-1]:+.4f}")
    print(f"{'Avg KL Div':<20} {sum(ppo_metrics['kl_divs'])/len(ppo_metrics['kl_divs']):<15.4f} "
          f"{sum(grpo_metrics['kl_divs'])/len(grpo_metrics['kl_divs']):<15.4f}")
    print(f"{'Avg Advantage':<20} {sum(ppo_metrics['advantages'])/len(ppo_metrics['advantages']):<15.4f} "
          f"{sum(grpo_metrics['advantages'])/len(grpo_metrics['advantages']):<15.4f}")

    # Key differences
    print(f"\n💡 PPO vs GRPO 关键区别:")
    print(f"  PPO:")
    print(f"    - 需要 Critic (Value Model) → 额外显存和计算")
    print(f"    - GAE advantage: A_t = Σ (γλ)^l δ_(t+l)")
    print(f"    - Clip: ratio ∈ [1-ε, 1+ε]")
    print(f"    - 模型: Actor + Critic + Ref + Reward = 4 个模型")
    print(f"  GRPO:")
    print(f"    - 无需 Critic → 节省 ~25% 显存")
    print(f"    - Group Relative: A = (R - mean(group)) / std(group)")
    print(f"    - 同一 prompt 生成 n 个 response, 组内比较")
    print(f"    - 模型: Actor + Ref + Reward = 3 个模型")
    print(f"    - verl 默认推荐 GRPO (更简单, 效果相当)")

    # Convergence curve
    print(f"\n📊 Reward 收敛曲线 (每 10 步):")
    print(f"  {'Step':<6} {'PPO':<10} {'GRPO':<10}")
    for i in range(0, num_iters, 10):
        print(f"  {i:<6} {ppo_metrics['rewards'][i]:<10.4f} {grpo_metrics['rewards'][i]:<10.4f}")


def experiment_2_rollout_batch_size():
    """实验 2: Rollout Batch Size 影响"""
    print("\n" + "=" * 70)
    print("实验 2: Rollout Batch Size — 推理 batch 对训练效率的影响")
    print("=" * 70)

    print(f"\n{'Batch':<8} {'Rollout%':<10} {'Train%':<10} {'Tokens/step':<15} "
          f"{'Final Rwd':<12} {'GPU Mem(GB)':<12}")
    print("-" * 70)

    for batch_size in [32, 64, 128, 256, 512, 1024]:
        config = TrainingConfig(rollout_batch_size=batch_size)
        metrics = run_training(config, "grpo", 30)

        # Memory estimation (7B model)
        model_mem = 14  # BF16 weights
        # Rollout: model weights + KV cache
        kv_per_seq = config.prompt_tokens_avg + config.max_response_tokens
        kv_bytes = kv_per_seq * 2 * 32 * 2  # 32 layers * BF16
        total_kv = batch_size * kv_bytes / 1e9
        # Training: model + optimizer (Adam = 2x model)
        train_mem = model_mem * 3  # weights + momentum + variance
        rollout_mem = model_mem + total_kv
        total_mem = max(rollout_mem, train_mem)

        print(f"{batch_size:<8} {metrics['rollout_time_pct'][-1]*100:<10.1f} "
              f"{metrics['training_time_pct'][-1]*100:<10.1f} "
              f"{metrics['tokens_per_step'][-1]:<15,} "
              f"{metrics['rewards'][-1]:<12.4f} "
              f"{total_mem:<12.1f}")

    print(f"\n💡 关键发现:")
    print(f"  - Rollout 占训练时间 40-70% (推理是瓶颈)")
    print(f"  - 更大 batch → 更稳定梯度, 但更多 GPU 显存")
    print(f"  - verl 使用异步 rollout 解耦训练和推理")
    print(f"  - vLLM 作为 rollout engine 提供 continuous batching")


def experiment_3_kl_penalty():
    """实验 3: KL 惩罚影响"""
    print("\n" + "=" * 70)
    print("实验 3: KL 惩罚 — KL 系数对 Reward Hacking 的抑制")
    print("=" * 70)

    kl_coeffs = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]

    print(f"\n{'KL Coeff':<10} {'Final Reward':<14} {'Final Quality':<14} "
          f"{'Avg KL Div':<14} {'Reward/Quality':<14}")
    print("-" * 70)

    for kl_coeff in kl_coeffs:
        config = TrainingConfig(kl_coeff=kl_coeff)
        metrics = run_training(config, "ppo", 50)

        final_reward = metrics['rewards'][-1]
        final_quality = metrics['actor_quality'][-1]
        avg_kl = sum(metrics['kl_divs']) / len(metrics['kl_divs'])
        efficiency = final_reward / final_quality if final_quality > 0 else 0

        print(f"{kl_coeff:<10.3f} {final_reward:<14.4f} {final_quality:<14.4f} "
              f"{avg_kl:<14.4f} {efficiency:<14.4f}")

    print(f"\n💡 KL 惩罚的作用:")
    print(f"  - KL=0: 无约束, policy 可能 reward hacking (高 reward 低质量)")
    print(f"  - KL 小 (0.01-0.05): 适度约束, 最佳平衡点")
    print(f"  - KL 大 (0.5-1.0): 过度约束, policy 几乎不更新")
    print(f"  - verl 使用 adaptive KL: 自动调整系数保持 target KL")
    print(f"  - Reward = task_reward - β × KL(π_θ || π_ref)")


def experiment_4_prefix_caching():
    """实验 4: Prefix Caching 收益 — GRPO 场景"""
    print("\n" + "=" * 70)
    print("实验 4: Prefix Caching 收益 — GRPO 场景下的 KV Cache 复用")
    print("=" * 70)

    print(f"\nGRPO: 同一 prompt 生成 n 个 response → prompt KV 可复用")

    n_responses = [2, 4, 8, 16, 32]
    prompt_tokens = 512
    response_tokens = 256
    bytes_per_token_layer = 2  # BF16
    n_layers = 32

    print(f"\n{'n':<6} {'Prompt KV':<12} {'Response KV':<14} {'Total(n×R+P)':<14} "
          f"{'With PC':<14} {'Savings':<10} {'Savings%':<10}")
    print("-" * 80)

    for n in n_responses:
        # Per sequence
        prompt_kv = prompt_tokens * n_layers * bytes_per_token_layer  # bytes per seq
        response_kv = response_tokens * n_layers * bytes_per_token_layer

        # Without prefix caching: n × (P + R)
        total_without = n * (prompt_kv + response_kv)

        # With prefix caching: P + n × R
        total_with = prompt_kv + n * response_kv

        savings = total_without - total_with
        savings_pct = savings / total_without * 100

        print(f"{n:<6} {prompt_kv/1024:<12.1f} {response_kv/1024:<14.1f} "
              f"{total_without/1024/1024:<14.2f} "
              f"{total_with/1024/1024:<14.2f} "
              f"{savings/1024/1024:<10.2f} {savings_pct:<10.1f}")

    # verl Prefix Grouper mechanism
    print(f"\n💡 verl Prefix Grouper 机制:")
    print(f"  1. build_pg_from_micro_batch(): 按 prompt 分组")
    print(f"  2. 只 forward 一次 prefix → 得到 prefix KV cache")
    print(f"  3. 每个 response 共享 prefix KV, 只需 forward suffix")
    print(f"  4. 在 verl 中: lora_shrink/lora_expand 支持 LoRA + prefix caching")
    print(f"  5. GRPO n=8 时节省 ~{(1 - 1/8)*100:.0f}% 的 prompt KV cache")

    # Impact on training throughput
    print(f"\n📊 对训练吞吐的影响 (7B model, GRPO n=8, 8 GPU):")
    for n in [4, 8, 16]:
        prompt_frac = prompt_tokens / (prompt_tokens + response_tokens)
        # Without PC: all prompts recomputed
        # With PC: prompts computed once
        compute_saving = (n - 1) * prompt_frac / (n * (prompt_frac + 1 - prompt_frac))
        throughput_gain = 1 / (1 - compute_saving * 0.6)  # 60% of saving translates to throughput
        print(f"  n={n:<3}: prefix占比={prompt_frac*100:.0f}%, "
              f"计算节省={compute_saving*100:.1f}%, "
              f"吞吐提升={throughput_gain:.2f}x")


def experiment_5_distributed_scaling():
    """实验 5: 分布式扩展 — Actor/Critic/Reward Worker 配置"""
    print("\n" + "=" * 70)
    print("实验 5: 分布式扩展 — verl Worker 配置策略")
    print("=" * 70)

    print(f"\nverl Ray Actor 架构:")
    print(f"  - ActorRolloutRefWorker: 训练+推理+参考 (混合 Worker)")
    print(f"  - CriticWorker: Value function (PPO only)")
    print(f"  - RewardWorker: Reward 计算")
    print(f"  - ResourcePool: Ray 管理 GPU 资源分配")

    configs = [
        {"name": "1-GPU PPO", "gpus": 1, "algo": "ppo",
         "actor": 1, "critic": 1, "reward": 1, "colocate": True},
        {"name": "4-GPU PPO TP=4", "gpus": 4, "algo": "ppo",
         "actor": 4, "critic": 4, "reward": 1, "colocate": True},
        {"name": "4-GPU PPO Colocate", "gpus": 4, "algo": "ppo",
         "actor": 4, "critic": 4, "reward": 4, "colocate": True},
        {"name": "8-GPU PPO TP=8", "gpus": 8, "algo": "ppo",
         "actor": 8, "critic": 8, "reward": 1, "colocate": True},
        {"name": "8-GPU GRPO TP=8", "gpus": 8, "algo": "grpo",
         "actor": 8, "critic": 0, "reward": 1, "colocate": True},
        {"name": "16-GPU GRPO 2×TP=8", "gpus": 16, "algo": "grpo",
         "actor": 8, "critic": 0, "reward": 1, "colocate": False},
        {"name": "16-GPU GRPO DP=2", "gpus": 16, "algo": "grpo",
         "actor": 16, "critic": 0, "reward": 2, "colocate": False},
    ]

    print(f"\n{'Config':<25} {'GPUs':<6} {'Algo':<6} {'Models':<8} "
          f"{'Mem/GB':<8} {'Thruput':<10} {'Notes':<20}")
    print("-" * 95)

    model_7b_mem = 14  # GB BF16
    adam_factor = 2  # Adam states = 2x params

    for cfg in configs:
        n_gpus = cfg['gpus']
        n_models = 4 if cfg['algo'] == 'ppo' else 3  # actor+ref+critic+reward vs actor+ref+reward

        # Memory per GPU
        if cfg['colocate']:
            # All models on same GPUs
            actor_mem = model_7b_mem * (1 + adam_factor)  # weights + optimizer
            ref_mem = model_7b_mem
            critic_mem = model_7b_mem * (1 + adam_factor) if cfg['algo'] == 'ppo' else 0
            reward_mem = model_7b_mem if cfg['reward'] > 0 else 0
            total_mem = actor_mem + ref_mem + critic_mem + reward_mem
            mem_per_gpu = total_mem / n_gpus
        else:
            # Separate workers
            actor_gpus = cfg['actor']
            critic_gpus = cfg['critic'] if cfg['critic'] > 0 else 0
            mem_per_gpu = (model_7b_mem * (1 + adam_factor)) if actor_gpus == n_gpus else model_7b_mem * 3

        # Throughput estimate (relative)
        tp_size = cfg['actor']
        dp_size = n_gpus // tp_size if cfg['colocate'] else max(1, n_gpus // tp_size)
        throughput = dp_size * 0.9  # ~90% scaling efficiency

        notes = ""
        if cfg['colocate'] and n_gpus > 1:
            notes = "Weights共享GPU"
        elif dp_size > 1:
            notes = f"DP={dp_size}线性扩展"
        elif tp_size > 1:
            notes = f"TP={tp_size}延迟降低"

        print(f"{cfg['name']:<25} {n_gpus:<6} {cfg['algo']:<6} {n_models:<8} "
              f"{mem_per_gpu:<8.1f} {throughput:<10.1f}x {notes:<20}")

    print(f"\n💡 verl 分布式策略:")
    print(f"  - Colocate: Actor+Ref+Critic+Reward 共享 GPU (省显存, verl 推荐)")
    print(f"  - TP: 单个大模型跨多 GPU (通信开销 ~10-15%)")
    print(f"  - DP: 多个独立实例 (线性吞吐扩展)")
    print(f"  - GRPO 无 Critic → 少一个模型 → 节省 ~25% 显存")
    print(f"  - 405B 需要 TP=8 + PP=2 ≈ 16 GPU per instance")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RLHF/GRPO 训练模拟器")
    parser.add_argument("--experiment", "-e", type=int, default=0,
                        help="运行特定实验 (1-5), 0=全部")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          RLHF/GRPO 训练模拟器                              ║")
    print("║   基于 verl 架构的 RL 对齐训练模拟                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    experiments = {
        1: experiment_1_ppo_vs_grpo,
        2: experiment_2_rollout_batch_size,
        3: experiment_3_kl_penalty,
        4: experiment_4_prefix_caching,
        5: experiment_5_distributed_scaling,
    }

    if args.experiment == 0:
        for exp_fn in experiments.values():
            exp_fn()
    elif args.experiment in experiments:
        experiments[args.experiment]()
    else:
        print(f"Unknown experiment: {args.experiment}")
        return

    print("\n" + "=" * 70)
    print("✅ 模拟完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
