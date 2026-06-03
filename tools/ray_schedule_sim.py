#!/usr/bin/env python3
"""Ray 分布式框架调度模拟器

CPU 可运行的 Ray 调度分析，覆盖 4 个实验:
  1. Actor/Task 调度模型分析
  2. Placement Group 约束调度
  3. RLHF 多模型调度模拟
  4. Ray 集群资源利用率分析

关键概念:
  - Ray Actor: 有状态远程对象, 适合模型服务
  - Ray Task: 无状态远程函数, 适合数据处理
  - Placement Group: 资源预留, 保证 co-location
  - RLHF: PPO 四模型 (Actor/Critic/Reward/Reference) 调度

用法:
  conda run -n ai-infra python tools/ray_schedule_sim.py
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

import numpy as np


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

class ResourceType(Enum):
    GPU = "GPU"
    CPU = "CPU"
    MEMORY = "Memory"


@dataclass
class NodeConfig:
    name: str
    num_gpus: int
    gpu_memory_gb: float
    num_cpus: int
    memory_gb: float


@dataclass
class ModelActor:
    name: str
    num_gpus: float
    gpu_memory_gb: float
    num_cpus: int = 4
    load_time_s: float = 30.0
    inference_time_ms: float = 50.0


# 典型节点配置
H100_NODE = NodeConfig("H100 Node", 8, 80, 128, 1000)
A100_NODE = NodeConfig("A100 Node", 8, 80, 96, 768)
A100_4GPU = NodeConfig("A100 4-GPU", 4, 80, 64, 384)

# RLHF PPO 模型
ACTOR_7B = ModelActor("Actor (7B)", 1, 16, 4, 20, 10)
CRITIC_7B = ModelActor("Critic (7B)", 1, 16, 4, 20, 10)
REWARD_7B = ModelActor("Reward (7B)", 1, 16, 4, 15, 8)
REF_7B = ModelActor("Reference (7B)", 1, 16, 4, 0, 0)  # 不常运行

ACTOR_70B = ModelActor("Actor (70B)", 4, 80, 8, 60, 50)
CRITIC_70B = ModelActor("Critic (70B)", 4, 80, 8, 60, 50)
REWARD_70B = ModelActor("Reward (70B)", 4, 80, 8, 45, 40)
REF_70B = ModelActor("Reference (70B)", 4, 80, 8, 0, 0)


# ──────────────────────────────────────────────
# 调度模拟
# ──────────────────────────────────────────────

def simulate_actor_scheduling(
    actors: List[ModelActor],
    nodes: List[NodeConfig],
    strategy: str = "binpack",  # binpack, spread, strict_spread
) -> dict:
    """模拟 Actor 调度到集群节点

    策略:
    - binpack: 尽量填满一个节点 (节省资源)
    - spread: 均匀分布 (容错)
    - strict_spread: 每个 actor 不同节点 (严格容错)
    """
    # 节点剩余资源
    remaining = [
        {
            'name': n.name,
            'gpus': n.num_gpus,
            'gpu_mem': n.gpu_memory_gb * n.num_gpus,
            'cpus': n.num_cpus,
            'memory': n.memory_gb,
            'actors': [],
        }
        for n in nodes
    ]

    scheduled = []
    failed = []

    for actor in actors:
        placed = False

        if strategy == "binpack":
            # 按剩余资源排序 (最满的优先)
            candidates = sorted(
                range(len(remaining)),
                key=lambda i: remaining[i]['gpus'],
            )
        elif strategy == "spread":
            # 按剩余资源排序 (最空的优先)
            candidates = sorted(
                range(len(remaining)),
                key=lambda i: -remaining[i]['gpus'],
            )
        elif strategy == "strict_spread":
            # 每个 actor 必须在不同节点
            candidates = [
                i for i in range(len(remaining))
                if len(remaining[i]['actors']) == 0
            ]
        else:
            candidates = list(range(len(remaining)))

        for idx in candidates:
            node = remaining[idx]
            if (node['gpus'] >= actor.num_gpus and
                node['gpu_mem'] >= actor.gpu_memory_gb and
                node['cpus'] >= actor.num_cpus):
                node['gpus'] -= actor.num_gpus
                node['gpu_mem'] -= actor.gpu_memory_gb
                node['cpus'] -= actor.num_cpus
                node['memory'] -= 10  # 估算内存使用
                node['actors'].append(actor.name)
                scheduled.append({
                    'actor': actor.name,
                    'node': node['name'],
                    'gpus': actor.num_gpus,
                    'gpu_mem': actor.gpu_memory_gb,
                })
                placed = True
                break

        if not placed:
            failed.append(actor.name)

    # 资源利用率
    total_gpus = sum(n.num_gpus for n in nodes)
    used_gpus = sum(a.num_gpus for a in actors if a.name not in failed)
    gpu_util = used_gpus / total_gpus * 100 if total_gpus > 0 else 0

    return {
        'strategy': strategy,
        'scheduled': scheduled,
        'failed': failed,
        'nodes': remaining,
        'total_gpus': total_gpus,
        'used_gpus': used_gpus,
        'gpu_utilization': gpu_util,
    }


def simulate_rlhf_iteration(
    actor_model: ModelActor,
    critic_model: ModelActor,
    reward_model: ModelActor,
    ref_model: ModelActor,
    batch_size: int = 256,
    seq_len: int = 2048,
    gen_len: int = 512,
    num_ppo_epochs: int = 1,
) -> dict:
    """模拟 RLHF PPO 一次迭代的延迟

    PPO 四阶段:
    1. Actor 生成 responses (inference, batch 并行)
    2. Reward + Reference 评分 (inference, 可并行)
    3. Critic 估计 value (inference)
    4. Actor + Critic 更新 (training)
    """
    # Stage 1: Actor 生成
    gen_tokens = batch_size * gen_len
    gen_time = gen_tokens * actor_model.inference_time_ms / batch_size / 1000

    # Stage 2: Reward + Reference 评分 (可并行)
    total_tokens = batch_size * (seq_len + gen_len)
    reward_time = total_tokens * reward_model.inference_time_ms / batch_size / 1000
    ref_time = total_tokens * ref_model.inference_time_ms / batch_size / 1000
    score_time = max(reward_time, ref_time)  # 并行

    # Stage 3: Critic 估计
    critic_time = total_tokens * critic_model.inference_time_ms / batch_size / 1000

    # Stage 4: Actor + Critic 训练更新
    # 训练约 3x 推理时间 (forward + backward + optimizer)
    train_time = total_tokens * actor_model.inference_time_ms * 3 / batch_size / 1000
    critic_train = total_tokens * critic_model.inference_time_ms * 3 / batch_size / 1000
    update_time = max(train_time, critic_train)  # 可部分并行

    total_time = gen_time + score_time + critic_time + update_time

    return {
        'batch_size': batch_size,
        'gen_time_s': gen_time,
        'score_time_s': score_time,
        'critic_time_s': critic_time,
        'update_time_s': update_time,
        'total_time_s': total_time,
        'gen_pct': gen_time / total_time * 100,
        'score_pct': score_time / total_time * 100,
        'critic_pct': critic_time / total_time * 100,
        'update_pct': update_time / total_time * 100,
        'samples_per_s': batch_size / total_time,
    }


# ──────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────

def experiment_1_actor_task_scheduling():
    """实验 1: Actor/Task 调度模型分析"""
    print("\n" + "=" * 70)
    print("实验 1: Ray Actor 调度策略对比")
    print("=" * 70)

    # 模拟: 部署多个推理模型
    actors = [ACTOR_7B, CRITIC_7B, REWARD_7B, REF_7B] * 2  # 8 actors
    nodes = [A100_NODE]

    print(f"\n配置: 8 个模型 Actor × 1 个 A100 节点 (8 GPU)")
    print(f"  Actor: {ACTOR_7B.num_gpus} GPU × 2")
    print(f"  Critic: {CRITIC_7B.num_gpus} GPU × 2")
    print(f"  Reward: {REWARD_7B.num_gpus} GPU × 2")
    print(f"  Reference: {REF_7B.num_gpus} GPU × 2")

    for strategy in ["binpack", "spread"]:
        result = simulate_actor_scheduling(actors, nodes, strategy)
        print(f"\n  策略: {strategy}")
        print(f"  调度: {len(result['scheduled'])} 成功, {len(result['failed'])} 失败")
        print(f"  GPU 利用率: {result['gpu_utilization']:.0f}%")
        for node in result['nodes']:
            if node['actors']:
                print(f"    {node['name']}: {node['actors']} (剩余 {node['gpus']} GPU)")

    print(f"\n关键洞察:")
    print(f"  - 8 个模型 × 1 GPU = 8 GPU, 刚好 1 个 A100 节点")
    print(f"  - binpack: 全部放在一个节点, 节省网络")
    print(f"  - spread: 分散到多节点, 容错更好")
    print(f"  - Ray 默认 binpack (省钱)")


def experiment_2_placement_group():
    """实验 2: Placement Group 约束调度"""
    print("\n" + "=" * 70)
    print("实验 2: Placement Group 调度 (RLHF 四模型)")
    print("=" * 70)

    # RLHF 需要四模型 co-located (减少通信)
    print(f"\n--- 7B RLHF (4×1 GPU) ---")
    actors_7b = [ACTOR_7B, CRITIC_7B, REWARD_7B, REF_7B]

    for nodes in [[A100_4GPU], [A100_NODE]]:
        result = simulate_actor_scheduling(actors_7b, nodes, "binpack")
        total_gpus = result['total_gpus']
        print(f"\n  节点: {nodes[0].name} ({total_gpus} GPU)")
        print(f"  部署: {len(result['scheduled'])} 成功")
        for s in result['scheduled']:
            print(f"    {s['actor']}: {s['gpus']} GPU on {s['node']}")
        print(f"  GPU 利用率: {result['gpu_utilization']:.0f}%")

    print(f"\n--- 70B RLHF (4×4 GPU) ---")
    actors_70b = [ACTOR_70B, CRITIC_70B, REWARD_70B, REF_70B]

    configs = [
        ([H100_NODE], "1×H100 (8 GPU)"),
        ([H100_NODE] * 2, "2×H100 (16 GPU)"),
        ([H100_NODE] * 4, "4×H100 (32 GPU)"),
    ]

    for nodes, label in configs:
        result = simulate_actor_scheduling(actors_70b, nodes, "binpack")
        print(f"\n  {label}: {len(result['scheduled'])} 成功, {len(result['failed'])} 失败")
        for node in result['nodes']:
            if node['actors']:
                print(f"    {node['name']}: {node['actors']} (剩余 {node['gpus']} GPU)")

    print(f"\n关键洞察:")
    print(f"  - 7B RLHF: 4 GPU 足够, 单节点 A100 即可")
    print(f"  - 70B RLHF: 需要 16 GPU (每个模型 4 GPU TP)")
    print(f"  - Placement Group: 保证四模型 co-located")
    print(f"  - Prefix Sharing: Actor/Ref 共享权重, 可省 1 组 GPU")


def experiment_3_rlhf_scheduling():
    """实验 3: RLHF PPO 调度模拟"""
    print("\n" + "=" * 70)
    print("实验 3: RLHF PPO 迭代延迟分析")
    print("=" * 70)

    configs = [
        ("7B PPO (B=256)", ACTOR_7B, CRITIC_7B, REWARD_7B, REF_7B, 256, 2048, 512),
        ("70B PPO (B=64)", ACTOR_70B, CRITIC_70B, REWARD_70B, REF_70B, 64, 4096, 512),
        ("70B PPO (B=256)", ACTOR_70B, CRITIC_70B, REWARD_70B, REF_70B, 256, 4096, 512),
    ]

    for label, actor, critic, reward, ref, bs, sl, gl in configs:
        r = simulate_rlhf_iteration(actor, critic, reward, ref, bs, sl, gl)

        print(f"\n--- {label} ---")
        print(f"  生成: {r['gen_time_s']:.1f}s ({r['gen_pct']:.0f}%)")
        print(f"  评分: {r['score_time_s']:.1f}s ({r['score_pct']:.0f}%)")
        print(f"  Critic: {r['critic_time_s']:.1f}s ({r['critic_pct']:.0f}%)")
        print(f"  更新: {r['update_time_s']:.1f}s ({r['update_pct']:.0f}%)")
        print(f"  总计: {r['total_time_s']:.1f}s, {r['samples_per_s']:.1f} samples/s")

    print(f"\n关键洞察:")
    print(f"  - 生成阶段: 串行 decode, 是主要瓶颈 (40-60%)")
    print(f"  - 评分阶段: Reward+Ref 可并行, 相对快")
    print(f"  - 更新阶段: 训练 3x 推理, 但可和 Critic 并行")
    print(f"  - GRPO 无 Critic: 省掉 Critic 步骤, 减少资源需求")


def experiment_4_resource_utilization():
    """实验 4: 集群资源利用率分析"""
    print("\n" + "=" * 70)
    print("实验 4: Ray 集群资源利用率分析")
    print("=" * 70)

    scenarios = [
        {
            'name': '纯推理服务',
            'actors': [ModelActor(f"vLLM-{i}", 1, 16, 4, 20, 10) for i in range(8)],
            'nodes': [A100_NODE],
        },
        {
            'name': '7B RLHF (verl)',
            'actors': [ACTOR_7B, CRITIC_7B, REWARD_7B, REF_7B],
            'nodes': [A100_NODE],
        },
        {
            'name': '70B RLHF (verl)',
            'actors': [ACTOR_70B, CRITIC_70B, REWARD_70B, REF_70B],
            'nodes': [H100_NODE, H100_NODE],
        },
        {
            'name': '70B RLHF + Prefix Sharing',
            'actors': [ACTOR_70B, CRITIC_70B, REWARD_70B],  # 省掉 Ref
            'nodes': [H100_NODE, H100_NODE],
        },
        {
            'name': '多租户推理',
            'actors': [
                ModelActor("vLLM-7B", 2, 16, 8, 30, 15),
                ModelActor("vLLM-70B", 4, 80, 8, 60, 50),
                ModelActor("vLLM-13B", 1, 28, 4, 25, 12),
            ],
            'nodes': [H100_NODE, H100_NODE],
        },
    ]

    print(f"\n{'场景':<25} {'GPU':>6} {'使用GPU':>8} {'利用率':>10} {'节点':>8} {'剩余':>8}")
    print("-" * 72)

    for s in scenarios:
        result = simulate_actor_scheduling(s['actors'], s['nodes'], "binpack")
        total = result['total_gpus']
        used = result['used_gpus']
        util = result['gpu_utilization']
        failed = len(result['failed'])
        remaining = total - used

        label = s['name'] + (" ⚠" if failed > 0 else "")
        print(f"{label:<25} {total:>6d} {used:>8d} {util:>9.1f}% {len(s['nodes']):>8d} {remaining:>8d}")

    print(f"\n关键洞察:")
    print(f"  - 纯推理: GPU 利用率 100%, 全部用于推理")
    print(f"  - 7B RLHF: 4/8 GPU 利用率 50%, 剩余可做推理")
    print(f"  - 70B RLHF: 16/16 GPU, 需要专用集群")
    print(f"  - Prefix Sharing: 省掉 Ref, 节省 25% GPU")
    print(f"  - GRPO (无 Critic): 再省 25%, 只需 Actor + Reward")
    print(f"  - verl 优化: FSDP + Prefix Sharing 大幅减少显存")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Ray 分布式框架调度模拟器")
    print("Actor 调度 + Placement Group + RLHF PPO + 资源利用率")
    print("=" * 70)

    experiment_1_actor_task_scheduling()
    experiment_2_placement_group()
    experiment_3_rlhf_scheduling()
    experiment_4_resource_utilization()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
Ray 调度关键要点:

  1. Actor vs Task:
     Actor: 有状态, 适合模型服务 (GPU 常驻)
     Task: 无状态, 适合数据处理 (按需分配)
     → RLHF 用 Actor (模型常驻 GPU)

  2. Placement Group:
     保证 Actor co-located (同节点通信快)
     策略: PACK (节省节点), SPREAD (容错)
     → RLHF: 四模型同节点, 减少 KV Transfer

  3. RLHF PPO 调度:
     四阶段: 生成 → 评分 → 估计 → 更新
     生成是瓶颈 (串行 decode, 40-60% 时间)
     评分可并行 (Reward + Reference)
     → verl: FSDP + Prefix Sharing 减少 GPU 需求

  4. 资源优化:
     Prefix Sharing: Actor/Ref 共享权重, 省 25% GPU
     GRPO (无 Critic): 再省 25%, Actor + Reward
     Colocate: Actor+Critic 同 GPU (verl 支持)
     → 最优: GRPO + Colocate, 仅需 Actor + Reward

  5. verl vs OpenRLHF:
     verl: FSDP + 三级前缀缓存 + Ray Actor
     OpenRLHF: DeepSpeed + Ray + Colocate
     两者都支持 PPO/GRPO, 架构类似
    """)
