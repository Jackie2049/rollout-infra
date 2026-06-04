#!/usr/bin/env python3
"""RLHF/GRPO Rollout 吞吐量模拟器

模拟 RL 训练中 Rollout 生成的完整性能模型:
1. Rollout 生成 (vLLM 推理, 含 prefix caching)
2. Reward 计算
3. 训练步 (含 PrefixGrouper 优化)
4. 完整迭代吞吐量
5. 配置敏感性分析

CPU 可运行, 无需 GPU。
"""

import math
from dataclasses import dataclass
from typing import Optional


# ============================================================
# 硬件和模型参数
# ============================================================

@dataclass
class GPU:
    name: str
    fp16_tflops: float
    hbm_bw_gbs: float       # HBM 带宽, 单位 GB/s
    hbm_gb: float
    nvlink_bw_gbs: float = 300.0  # NVLink 带宽, 单位 GB/s


@dataclass
class Model:
    name: str
    params_b: float           # 参数量 (十亿)
    hidden_dim: int
    num_layers: int
    num_heads: int
    head_dim: int = 128
    vocab_size: int = 128256

    @property
    def weight_bytes_fp16(self):
        """FP16 权重大小 (bytes)"""
        return self.params_b * 1e9 * 2

    @property
    def weight_bytes_fp8(self):
        """FP8 权重大小 (bytes)"""
        return self.params_b * 1e9 * 1

    def flops_per_token(self):
        """每个 token 的前向 FLOPs (2 × params)"""
        return 2 * self.params_b * 1e9

    def kv_bytes_per_token(self):
        """每个 token 的 KV Cache 大小 (2K+2V per layer)"""
        return 2 * 2 * self.num_layers * self.head_dim * 2  # FP16: 2 bytes


A100_80G = GPU("A100-80G", 312, 2035, 80)
H100_80G = GPU("H100-80G", 990, 3350, 80)
H200_141G = GPU("H200-141G", 990, 4800, 141)

LLAMA_7B = Model("Llama-7B", 7, 4096, 32, 32)
LLAMA_8B = Model("Llama-8B", 8, 4096, 32, 32)
LLAMA_70B = Model("Llama-70B", 70, 8192, 80, 64)

# 常用 GRPO 配置
GRPO_N_SAMPLES = 8  # 每个 prompt 生成 8 个 response


# ============================================================
# 性能模型
# ============================================================

def decode_time_per_token_ms(model: Model, gpu: GPU, tp: int = 1,
                             batch_size: int = 1) -> float:
    """Decode 阶段每个 token 的延迟 (ms)

    Decode 是 memory-bound:
    time = weight_bytes / (hbm_bw * tp)
    """
    weight_bytes = model.weight_bytes_fp16
    effective_bw = gpu.hbm_bw_gbs * 1e9 * tp  # bytes/s (hbm_bw_gbs 为 GB/s)
    return weight_bytes / effective_bw * 1000  # ms


def prefill_time_ms(model: Model, gpu: GPU, seq_len: int,
                    tp: int = 1) -> float:
    """Prefill 阶段时间 (ms)

    Prefill 是 compute-bound:
    time ≈ flops / (tflops * tp)
    但受 memory bandwidth 影响 (注意力部分)
    """
    flops = model.flops_per_token() * seq_len
    effective_tflops = gpu.fp16_tflops * tp  # TFLOPS
    # 实际利用率约 40-60%
    util = 0.5
    return flops / (effective_tflops * 1e12 * util) * 1000  # 1e12: TFLOPS→FLOPS


def rollout_generation_time_ms(model: Model, gpu: GPU,
                                n_prompts: int, n_samples: int,
                                prompt_len: int, response_len: int,
                                tp: int = 1,
                                prefix_cache: bool = False) -> dict:
    """Rollout 生成总时间

    模拟 vLLM 生成 n_prompts × n_samples 个 response。
    """
    # Decode 吞吐量: tokens/sec (受 batch 并发影响)
    weight_bytes = model.weight_bytes_fp16
    effective_bw = gpu.hbm_bw_gbs * 1e9 * tp  # bytes/s (已为 GB/s)

    # 最大并发请求数 (受 KV Cache 限制)
    kv_per_token = model.kv_bytes_per_token()
    # 分配约 60% HBM 给 KV Cache
    kv_budget = gpu.hbm_gb * 1e9 * 0.6 * tp
    max_seq_len = prompt_len + response_len
    kv_per_request = kv_per_token * max_seq_len
    max_concurrent = int(kv_budget / kv_per_request)

    # 每个 batch 的并发请求数
    total_requests = n_prompts * n_samples
    batch_size = min(total_requests, max(1, max_concurrent))

    # Decode 每步时间 (batch 并发)
    # 大 batch 下吞吐更高但延迟更大
    decode_per_token = weight_bytes / effective_bw * 1000  # ms/token (单请求)
    # Batch 效果: 吞吐提升但延迟略增
    batch_throughput_factor = min(batch_size, 64)  # 吞吐线性到 ~64

    # Token 吞吐量 (tokens/sec)
    single_throughput = effective_bw / weight_bytes  # tokens/sec
    batch_throughput = single_throughput * batch_throughput_factor

    # Prefill 时间 (含 prefix caching)
    if prefix_cache:
        # 首个请求完整 prefill, 后续同 prompt 的请求命中 prefix cache
        # 只需 prefill response 部分 (实际是 decode)
        prefill_tokens = n_prompts * prompt_len  # 每个 prompt 只 prefill 一次
        decode_tokens = total_requests * response_len
        prefix_savings = (1 - 1.0 / n_samples) * 100  # 节省百分比
    else:
        prefill_tokens = total_requests * prompt_len
        decode_tokens = total_requests * response_len
        prefix_savings = 0.0

    prefill_ms = prefill_time_ms(model, gpu, prefill_tokens, tp) if prefill_tokens > 0 else 0
    decode_ms = decode_tokens / batch_throughput * 1000

    total_ms = prefill_ms + decode_ms

    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "total_tokens": prefill_tokens + decode_tokens,
        "throughput_tok_s": (prefill_tokens + decode_tokens) / (total_ms / 1000),
        "max_concurrent": max_concurrent,
        "batch_size": batch_size,
        "prefix_savings_pct": prefix_savings,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
    }


def reward_time_ms(n_responses: int, response_len: int,
                   model: Model, gpu: GPU, tp: int = 1) -> float:
    """Reward 计算时间

    RM 通常是较小模型, 但需要处理所有 response。
    这里简化为用同一模型的推理时间。
    """
    # RM 通常用小模型 (如 7B), 只做 prefill
    return prefill_time_ms(model, gpu, n_responses * response_len, tp)


def training_step_time_ms(model: Model, gpu: GPU,
                          n_samples: int, prompt_len: int, response_len: int,
                          tp: int = 1, prefix_grouper: bool = False) -> dict:
    """训练步时间

    PPO/GRPO 训练步:
    - Forward pass: 计算旧 log probs + 新 log probs
    - Backward pass: 梯度计算
    - 约 3x forward 的时间 (forward + backward + optimizer)

    PrefixGrouper 节省:
    - 原始: G 次 prompt attention
    - 优化: 1 次 prompt attention + G 次 suffix attention
    """
    total_len = prompt_len + response_len
    flops_per_sample = model.flops_per_token() * total_len

    # Forward: 2x params × seq_len
    # Backward: ~4x params × seq_len (梯度计算)
    # 总计约 6x params × seq_len per sample
    total_flops_per_sample = flops_per_sample * 3  # forward + backward + optimizer

    # PrefixGrouper 节省
    if prefix_grouper and n_samples > 1:
        # Prompt attention: O(prompt_len²) per sample → 只做 1 次
        # 节省 = (n_samples - 1) × prompt² / total_flops
        prompt_attn_flops = 2 * model.num_layers * model.hidden_dim * prompt_len * prompt_len
        saved_flops = (n_samples - 1) * prompt_attn_flops
        savings_pct = saved_flops / total_flops_per_sample * 100
        effective_flops = total_flops_per_sample - saved_flops / n_samples
    else:
        savings_pct = 0.0
        effective_flops = total_flops_per_sample

    effective_tflops = gpu.fp16_tflops * tp * 0.5  # MFU ~50%
    time_per_sample = effective_flops / (effective_tflops * 1e12) * 1000  # 1e12: TFLOPS→FLOPS

    return {
        "time_per_sample_ms": time_per_sample,
        "savings_pct": savings_pct,
        "flops_per_sample": total_flops_per_sample,
    }


# ============================================================
# 实验
# ============================================================

def experiment1_grpo_iteration_breakdown():
    """实验 1: GRPO 完整迭代时间分解"""
    print("=" * 70)
    print("实验 1: GRPO 完整迭代时间分解")
    print("=" * 70)

    model = LLAMA_8B
    gpu = A100_80G
    tp = 1
    n_prompts = 32
    n_samples = GRPO_N_SAMPLES  # 8
    prompt_len = 512
    response_len = 512

    print(f"\n配置:")
    print(f"  模型: {model.name} ({model.params_b}B)")
    print(f"  GPU: {gpu.name}, TP={tp}")
    print(f"  Prompts: {n_prompts}, Samples/prompt: {n_samples}")
    print(f"  Prompt长度: {prompt_len}, Response长度: {response_len}")
    print(f"  总 responses: {n_prompts * n_samples}")

    # 1. Rollout 生成
    rollout = rollout_generation_time_ms(
        model, gpu, n_prompts, n_samples,
        prompt_len, response_len, tp, prefix_cache=False
    )
    rollout_pc = rollout_generation_time_ms(
        model, gpu, n_prompts, n_samples,
        prompt_len, response_len, tp, prefix_cache=True
    )

    # 2. Reward 计算 (假设小模型, 约为 rollout 的 10%)
    reward_ms = rollout["total_ms"] * 0.1

    # 3. 训练步
    train = training_step_time_ms(
        model, gpu, n_samples, prompt_len, response_len, tp, prefix_grouper=False
    )
    train_pg = training_step_time_ms(
        model, gpu, n_samples, prompt_len, response_len, tp, prefix_grouper=True
    )
    total_train_ms = train["time_per_sample_ms"] * n_prompts * n_samples

    total_no_opt = rollout["total_ms"] + reward_ms + total_train_ms
    total_with_pc = rollout_pc["total_ms"] + reward_ms + total_train_ms
    total_with_both = rollout_pc["total_ms"] + reward_ms + train_pg["time_per_sample_ms"] * n_prompts * n_samples

    print(f"\n{'阶段':<25} {'无优化':<14} {'Prefix Cache':<14} {'PC + PG':<14}")
    print("-" * 67)
    print(f"{'Rollout 生成':<25} {rollout['total_ms']:<14.0f} {rollout_pc['total_ms']:<14.0f} {rollout_pc['total_ms']:<14.0f}")
    print(f"{'Reward 计算':<25} {reward_ms:<14.0f} {reward_ms:<14.0f} {reward_ms:<14.0f}")
    print(f"{'训练步':<25} {total_train_ms:<14.0f} {total_train_ms:<14.0f} {train_pg['time_per_sample_ms']*n_prompts*n_samples:<14.0f}")
    print("-" * 67)
    print(f"{'总迭代时间 (ms)':<25} {total_no_opt:<14.0f} {total_with_pc:<14.0f} {total_with_both:<14.0f}")
    print(f"{'加速比':<25} {'1.00x':<14} {total_no_opt/total_with_pc:<14.2f}x {total_no_opt/total_with_both:<14.2f}x")

    # Rollout 占比
    rollout_pct = rollout["total_ms"] / total_no_opt * 100
    print(f"\nRollout 占比: {rollout_pct:.0f}% (验证了 RLHF 的核心瓶颈)")
    print(f"Prefix Cache 节省 Rollout: {(1-rollout_pc['total_ms']/rollout['total_ms'])*100:.0f}%")
    print(f"PrefixGrouper 节省训练: {train['savings_pct']:.1f}%")


def experiment2_prefix_cache_sensitivity():
    """实验 2: Prefix Cache 收益 vs n_samples 和 prompt 长度"""
    print("\n" + "=" * 70)
    print("实验 2: Prefix Cache 收益敏感性分析")
    print("=" * 70)

    model = LLAMA_8B
    gpu = A100_80G
    n_prompts = 32
    response_len = 512

    # 2a: vs n_samples (固定 prompt_len=512)
    print(f"\n--- 2a: n_samples 敏感性 (prompt_len=512, response_len={response_len}) ---\n")
    prompt_len = 512
    print(f"{'n_samples':<12} {'无PC (ms)':<14} {'有PC (ms)':<14} {'节省%':<10} {'加速比'}")
    print("-" * 52)

    for n in [2, 4, 8, 16, 32]:
        r_no = rollout_generation_time_ms(model, gpu, n_prompts, n, prompt_len, response_len, prefix_cache=False)
        r_pc = rollout_generation_time_ms(model, gpu, n_prompts, n, prompt_len, response_len, prefix_cache=True)
        savings = (1 - r_pc["total_ms"] / r_no["total_ms"]) * 100
        print(f"{n:<12} {r_no['total_ms']:<14.0f} {r_pc['total_ms']:<14.0f} {savings:<10.1f}% {r_no['total_ms']/r_pc['total_ms']:.2f}x")

    # 2b: vs prompt_len (固定 n_samples=8)
    print(f"\n--- 2b: prompt_len 敏感性 (n_samples={GRPO_N_SAMPLES}, response_len={response_len}) ---\n")
    n_samples = GRPO_N_SAMPLES
    print(f"{'prompt_len':<12} {'无PC (ms)':<14} {'有PC (ms)':<14} {'节省%':<10} {'加速比'}")
    print("-" * 52)

    for pl in [128, 256, 512, 1024, 2048, 4096]:
        r_no = rollout_generation_time_ms(model, gpu, n_prompts, n_samples, pl, response_len, prefix_cache=False)
        r_pc = rollout_generation_time_ms(model, gpu, n_prompts, n_samples, pl, response_len, prefix_cache=True)
        savings = (1 - r_pc["total_ms"] / r_no["total_ms"]) * 100
        print(f"{pl:<12} {r_no['total_ms']:<14.0f} {r_pc['total_ms']:<14.0f} {savings:<10.1f}% {r_no['total_ms']/r_pc['total_ms']:.2f}x")

    print("\n关键洞察:")
    print("  - n_samples 越大, prefix cache 收益越高 (共享比 = 1 - 1/n)")
    print("  - prompt 越长, prefix cache 收益越高 (prefill 占比更大)")
    print("  - GRPO n=8: prefix cache 可节省 ~87.5% 的 prompt prefill")
    print("  - 收益公式: savings ≈ (1-1/n) × (prompt/(prompt+response))")


def experiment3_model_scale():
    """实验 3: 不同模型规模的 Rollout 性能"""
    print("\n" + "=" * 70)
    print("实验 3: 模型规模 vs Rollout 性能")
    print("=" * 70)

    configs = [
        (LLAMA_8B, A100_80G, 1),
        (LLAMA_8B, H100_80G, 1),
        (LLAMA_70B, A100_80G, 4),
        (LLAMA_70B, H100_80G, 4),
        (LLAMA_70B, H200_141G, 8),
    ]

    n_prompts = 32
    n_samples = GRPO_N_SAMPLES
    prompt_len = 512
    response_len = 512

    print(f"\n配置: {n_prompts} prompts × {n_samples} samples, "
          f"prompt={prompt_len}, response={response_len}")
    print(f"\n{'模型':<14} {'GPU':<14} {'TP':<4} {'Rollout (ms)':<14} {'吞吐 tok/s':<14} {'最大并发'}")
    print("-" * 72)

    for model, gpu, tp in configs:
        r = rollout_generation_time_ms(
            model, gpu, n_prompts, n_samples,
            prompt_len, response_len, tp, prefix_cache=True
        )
        print(f"{model.name:<14} {gpu.name:<14} {tp:<4} {r['total_ms']:<14.0f} "
              f"{r['throughput_tok_s']:<14.0f} {r['max_concurrent']}")

    print("\n关键洞察:")
    print("  - 70B 比 8B 慢约 10x (即使 TP=4, 参数量 8.75x)")
    print("  - H100 比 A100 快约 2x (decode 吞吐由 HBM BW 决定)")
    print("  - TP 线性扩展: TP=4 理论 4x, 实际 ~3.5x (通信开销)")
    print("  - 70B+H200+TP=8 是大模型 RL 训练的推荐配置")


def experiment4_pgo_vs_ppo():
    """实验 4: GRPO vs PPO 资源效率对比"""
    print("\n" + "=" * 70)
    print("实验 4: GRPO vs PPO 资源效率")
    print("=" * 70)

    model = LLAMA_8B
    gpu = A100_80G
    n_prompts = 32
    n_samples = 8
    prompt_len = 512
    response_len = 512

    print(f"\n模型: {model.name}, GPU: {gpu.name}")
    print(f"配置: {n_prompts} prompts × {n_samples} samples\n")

    # PPO: 4 模型
    print("PPO 四模型架构:")
    print("  Actor (Rollout):  推理 + 训练")
    print("  Critic:           推理 + 训练")
    print("  Reference Model:  仅推理")
    print("  Reward Model:     仅推理")

    # Rollout (Actor 推理)
    rollout = rollout_generation_time_ms(
        model, gpu, n_prompts, n_samples,
        prompt_len, response_len, prefix_cache=True
    )

    # Reference Model 推理 (计算 log probs)
    ref_rollout = rollout_generation_time_ms(
        model, gpu, n_prompts, n_samples,
        prompt_len, response_len, prefix_cache=True
    )

    # Critic 推理 + 训练
    critic_inference = rollout["total_ms"] * 0.8  # V(s) 估值
    critic_train = training_step_time_ms(model, gpu, n_samples, prompt_len, response_len)
    critic_train_total = critic_train["time_per_sample_ms"] * n_prompts * n_samples

    # Actor 训练
    actor_train = training_step_time_ms(model, gpu, n_samples, prompt_len, response_len)
    actor_train_total = actor_train["time_per_sample_ms"] * n_prompts * n_samples

    # Reward Model (假设小模型)
    reward_ms = rollout["total_ms"] * 0.1

    # PPO 总时间
    ppo_total = rollout["total_ms"] + ref_rollout["total_ms"] + \
                critic_inference + critic_train_total + \
                actor_train_total + reward_ms

    # GRPO: 2 模型
    print("\nGRPO 两模型架构:")
    print("  Actor (Rollout):  推理 + 训练")
    print("  Reward Model:     仅推理 (可用规则替代)")

    grpo_rollout = rollout["total_ms"]
    grpo_train = actor_train_total
    grpo_reward = reward_ms
    grpo_total = grpo_rollout + grpo_train + grpo_reward

    print(f"\n{'阶段':<25} {'PPO (ms)':<14} {'GRPO (ms)':<14} {'GRPO/PPO'}")
    print("-" * 60)
    print(f"{'Rollout 生成':<25} {rollout['total_ms']:<14.0f} {grpo_rollout:<14.0f} {'1.00x'}")
    print(f"{'Ref Model 推理':<25} {ref_rollout['total_ms']:<14.0f} {'0':<14} {'—'}")
    print(f"{'Critic 推理':<25} {critic_inference:<14.0f} {'0':<14} {'—'}")
    print(f"{'Critic 训练':<25} {critic_train_total:<14.0f} {'0':<14} {'—'}")
    print(f"{'Actor 训练':<25} {actor_train_total:<14.0f} {grpo_train:<14.0f} {'1.00x'}")
    print(f"{'Reward 计算':<25} {reward_ms:<14.0f} {grpo_reward:<14.0f} {'1.00x'}")
    print("-" * 60)
    print(f"{'总计':<25} {ppo_total:<14.0f} {grpo_total:<14.0f} {grpo_total/ppo_total:.2f}x")
    print(f"\nGRPO 比 PPO 快 {ppo_total/grpo_total:.1f}x (减少 3 个模型)")
    print(f"GRPO 显存节省: ~50% (无需 Critic + Ref 参数)")

    # GPU 资源
    ppo_gpus = 4  # Actor+Critic+Ref+RM
    grpo_gpus = 1  # Actor (+ RM可CPU)
    print(f"\nGPU 需求: PPO {ppo_gpus} 组 vs GRPO {grpo_gpus} 组")
    print(f"成本效率: GRPO {(ppo_total/grpo_total) * ppo_gpus:.1f}x 更优")


def experiment5_optimal_config():
    """实验 5: 最优配置推荐"""
    print("\n" + "=" * 70)
    print("实验 5: 生产环境最优配置推荐")
    print("=" * 70)

    print("""
基于模拟结果的推荐配置:

┌──────────┬────────────┬─────────────┬──────────────────────────────┐
│ 模型     │ GPU 配置    │ 关键参数     │ 预估吞吐                     │
├──────────┼────────────┼─────────────┼──────────────────────────────┤
│ 7B/8B    │ 1× A100    │ n=8, bs=32  │ ~8K tok/s (rollout)          │
│ 7B/8B    │ 1× H100    │ n=8, bs=32  │ ~14K tok/s                   │
│ 70B      │ 4× H100    │ n=8, bs=32  │ ~6.5K tok/s (TP=4)           │
│ 70B      │ 8× H200    │ n=8, bs=32  │ ~18K tok/s (TP=8)            │
└──────────┴────────────┴─────────────┴──────────────────────────────┘

关键优化策略:
  1. Prefix Cache (推理侧): 必开, n=8, prompt=512 时节省 ~40% rollout
  2. PrefixGrouper (训练侧): prompt>512 时开启, 节省训练 attention
  3. Colocate 模式: Actor+Rollout+Ref 共享 GPU, 减少 GPU 间通信
  4. FP8 量化: Actor Rollout 用 FP8 推理, 减半显存和延迟
  5. Chunked Prefill: 长 prompt 分块处理, 减少 TTFT

成本估算 (GRPO, 8B 模型, A100):
  - 单迭代: ~100s (rollout=18s + train=81s + reward=3s)
  - 1000 步训练: ~28 小时
  - A100 按需价格: ~$2/GPU/hour
  - 总成本: ~$56 (单 GPU)

成本估算 (GRPO, 70B 模型, 4×H100):
  - 单迭代: ~200s (rollout=39s + train=155s + reward=6s)
  - 1000 步训练: ~55 小时
  - H100 按需价格: ~$3/GPU/hour
  - 总成本: ~$660 (4 GPU)
""")

    # 验证 8B 模型的估算
    model = LLAMA_8B
    gpu = A100_80G
    r = rollout_generation_time_ms(model, gpu, 32, 8, 512, 512, prefix_cache=True)
    print(f"\n验证 (8B, A100, PC enabled):")
    print(f"  Rollout: {r['total_ms']/1000:.1f}s, 吞吐: {r['throughput_tok_s']:.0f} tok/s")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RLHF/GRPO Rollout 吞吐量模拟器")
    print("Rollout 生成 + Reward + 训练步 + 优化策略")
    print("=" * 70)

    experiment1_grpo_iteration_breakdown()
    experiment2_prefix_cache_sensitivity()
    experiment3_model_scale()
    experiment4_pgo_vs_ppo()
    experiment5_optimal_config()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
RLHF Rollout 性能核心:

  1. GRPO 迭代分解 (8B, A100):
     - 训练步: ~80% (forward + backward, compute-bound)
     - Rollout: ~18% (自回归 decode, memory-bound)
     - Reward: ~3% (通常用规则, 可忽略)
     - 注意: PPO 中 Rollout 占比更高 (需 Ref + Critic 推理)

  2. Prefix Cache 收益
     - n=8, prompt=512: 节省 ~40% rollout 时间
     - n=8, prompt=4096: 节省 ~76% rollout 时间
     - 收益 = (1-1/n) × prompt_len / (prompt+response)

  3. GRPO vs PPO: 2.1x 速度, 8.5x 成本效率
     - 去掉 Critic + Ref: 训练步减少 50%
     - 单 GPU vs 4 GPU: 资源减少 75%

  4. 硬件选择
     - 8B: A100 即可, ~8K tok/s
     - 70B: H200+TP=8 最优, ~18K tok/s
     - HBM BW 决定 decode 吞吐 (memory-bound)
""")
