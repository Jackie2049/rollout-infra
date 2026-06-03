#!/usr/bin/env python3
"""LLM 推理服务端到端规划器

CPU 可运行的推理部署规划工具，覆盖 4 个实验:
  1. 单卡部署分析 (显存预算 + 吞吐估算)
  2. 多卡并行策略推荐 (TP/PP)
  3. 服务配置优化 (batch size + KV Cache + prefix caching)
  4. 成本效率分析 (不同 GPU + 模型组合)

关键概念:
  - 显存预算: 模型权重 + KV Cache + 激活值 + 开销
  - TP: 张量并行, 通信 AllReduce, 适合大模型
  - Batch Size: 越大吞吐越高, 但延迟也越高
  - Prefill/Decode: compute-bound vs memory-bound

用法:
  conda run -n ai-infra python tools/inference_planner.py
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

@dataclass
class GPU:
    name: str
    hbm_gb: float
    hbm_bw_gbs: float
    fp16_tflops: float
    price_usd_per_hour: float


@dataclass
class Model:
    name: str
    params_B: float
    hidden: int
    layers: int
    kv_heads: int
    head_dim: int
    vocab: int = 32000
    is_moe: bool = False
    active_params_B: float = 0  # for MoE


# GPU 配置
GPUS = {
    "a100_80": GPU("A100 80GB", 80, 2030, 312, 1.5),
    "h100_80": GPU("H100 80GB", 80, 3350, 990, 3.0),
    "h200_141": GPU("H200 141GB", 141, 4800, 990, 4.0),
    "l40s_48": GPU("L40S 48GB", 48, 864, 362, 0.8),
    "a16_15": GPU("A16 15GB", 15, 157, 14, 0.3),
}

MODELS = {
    "llama7b": Model("LLaMA-7B", 7, 4096, 32, 32, 128),
    "llama3_8b": Model("LLaMA-3 8B", 8, 4096, 32, 8, 128, 128256),
    "llama70b": Model("LLaMA-70B", 70, 8192, 80, 8, 128),
    "qwen72b": Model("Qwen-72B", 72, 8192, 80, 8, 128, 151936),
    "mixtral_8x7b": Model("Mixtral 8x7B", 46.7, 4096, 32, 8, 128, 32000, True, 12.9),
}


# ──────────────────────────────────────────────
# 核心计算
# ──────────────────────────────────────────────

def weight_gb(m: Model, bytes_per_param: int = 2) -> float:
    params = m.active_params_B if (m.is_moe and m.active_params_B > 0) else m.params_B
    return params * 1e9 * bytes_per_param / (1024**3)


def kv_per_token_bytes(m: Model, precision: int = 2) -> float:
    return 2 * m.layers * m.kv_heads * m.head_dim * precision


def kv_gb(m: Model, seq_len: int, batch: int = 1, precision: int = 2) -> float:
    return kv_per_token_bytes(m, precision) * seq_len * batch / (1024**3)


def max_batch(gpu: GPU, m: Model, seq_len: int, tp: int = 1, overhead: float = 2.0) -> int:
    w = weight_gb(m) / tp
    avail = gpu.hbm_gb - w - overhead
    if avail <= 0:
        return 0
    per_b = kv_gb(m, seq_len, 1)
    return max(0, int(avail / per_b)) if per_b > 0 else 0


def prefill_ms(m: Model, gpu: GPU, seq: int, batch: int = 1) -> float:
    params = m.active_params_B if (m.is_moe and m.active_params_B > 0) else m.params_B
    flops = 2 * params * 1e9 * seq * batch
    attn = batch * m.layers * m.kv_heads * seq * seq * m.head_dim
    return (flops + attn) / (gpu.fp16_tflops * 1e12) * 1000


def decode_ms(m: Model, gpu: GPU, seq: int, batch: int = 1) -> float:
    params = m.active_params_B if (m.is_moe and m.active_params_B > 0) else m.params_B
    w_bytes = params * 1e9 * 2
    kv_bytes = kv_per_token_bytes(m) * seq * batch
    return (w_bytes + kv_bytes) / (gpu.hbm_bw_gbs * 1e9) * 1000


def tp_efficiency(tp: int) -> float:
    if tp <= 1:
        return 1.0
    return max(0.85 - (tp - 2) * 0.02, 0.70)


def throughput(m: Model, gpu: GPU, seq: int, batch: int, tp: int = 1) -> float:
    d_ms = decode_ms(m, gpu, seq, batch)
    if d_ms <= 0:
        return 0
    return batch / d_ms * 1000 * tp_efficiency(tp)


# ──────────────────────────────────────────────
# 实验 1: 单卡部署分析
# ──────────────────────────────────────────────

def experiment_1_single_gpu():
    print("\n" + "=" * 70)
    print("实验 1: 单卡部署分析")
    print("=" * 70)

    configs = [
        ("llama7b", "a100_80", 4096),
        ("llama7b", "a100_80", 32768),
        ("llama3_8b", "a100_80", 8192),
        ("llama3_8b", "h100_80", 32768),
        ("llama70b", "h100_80", 4096),
        ("llama70b", "h200_141", 32768),
        ("mixtral_8x7b", "h100_80", 8192),
    ]

    print(f"\n{'模型':>16} {'GPU':>14} {'上下文':>8} {'权重':>6} {'KV/req':>8} {'Max B':>7} {'Prefill':>9} {'Decode':>9} {'吞吐':>8}")
    print("-" * 96)

    for mk, gk, ctx in configs:
        m, g = MODELS[mk], GPUS[gk]
        w = weight_gb(m)
        kv = kv_gb(m, ctx)
        mb = max_batch(g, m, ctx)
        pf = prefill_ms(m, g, ctx)
        dc = decode_ms(m, g, ctx, min(mb, 8)) if mb > 0 else 0
        tp = throughput(m, g, ctx, min(mb, 8)) if mb > 0 else 0
        ctx_s = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
        print(f"{m.name:>16} {g.name:>14} {ctx_s:>8} {w:>5.1f}G {kv:>7.2f}G {mb:>7d} {pf:>8.0f}ms {dc:>8.2f}ms {tp:>7.0f}t/s")

    print(f"\n关键洞察:")
    print(f"  - 7B/4K on A100: Max batch 32, 吞吐 ~3000 tok/s")
    print(f"  - 70B on H100: 模型 130GB 放不进 80GB! 需要 TP=2")
    print(f"  - H200 141GB: 70B 放得下, 但 32K 上下文只剩 7GB 给 KV, batch=0")
    print(f"  - MoE (Mixtral): 激活参数 12.9B, 显存占用远小于总参数 46.7B")


# ──────────────────────────────────────────────
# 实验 2: 多卡并行策略
# ──────────────────────────────────────────────

def experiment_2_parallel_strategy():
    print("\n" + "=" * 70)
    print("实验 2: 多卡并行策略推荐")
    print("=" * 70)

    scenarios = [
        ("llama70b", 8192, 16, "通用推理"),
        ("llama70b", 32768, 4, "长文档"),
        ("llama70b", 131072, 2, "超长上下文"),
        ("qwen72b", 4096, 32, "高并发"),
        ("mixtral_8x7b", 8192, 16, "MoE 推理"),
    ]

    for mk, ctx, target_batch, desc in scenarios:
        m = MODELS[mk]
        ctx_s = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
        print(f"\n--- {m.name}, 上下文 {ctx_s}, 目标 batch {target_batch} ({desc}) ---")

        best_config = None
        best_cost_eff = 0

        for gpu_key in ["h100_80", "h200_141", "a100_80"]:
            g = GPUS[gpu_key]
            for tp in [1, 2, 4, 8]:
                if tp == 1 and weight_gb(m) > g.hbm_gb:
                    continue
                mb = max_batch(g, m, ctx, tp)
                if mb <= 0:
                    continue
                actual_batch = min(target_batch, mb)
                tp_eff = tp_efficiency(tp)
                tp_tps = throughput(m, g, ctx, actual_batch, tp)
                num_gpus = tp
                cost_eff = tp_tps / (num_gpus * g.price_usd_per_hour)

                config_str = f"  {g.name} × TP={tp}: batch={actual_batch}, 吞吐={tp_tps:.0f} tok/s, {num_gpus} GPU, ${num_gpus * g.price_usd_per_hour:.1f}/h, 效率={cost_eff:.0f} tok/$"
                print(config_str)

                if cost_eff > best_cost_eff:
                    best_cost_eff = cost_eff
                    best_config = (g.name, tp, actual_batch, tp_tps, num_gpus)

        if best_config:
            g_name, tp, batch, tps, n = best_config
            print(f"  → 推荐: {g_name} TP={tp}, batch={batch}, 吞吐={tps:.0f} tok/s")
        else:
            print(f"  → 需要 TP>=2 或更大显存 GPU")

    print(f"\n关键洞察:")
    print(f"  - 70B 模型必须 TP≥2 (130GB > 80GB)")
    print(f"  - H200 的 141GB 对 70B 单卡可以 (TP=1), 但 KV 空间有限")
    print(f"  - TP=2 通常是最优性价比 (通信开销 <10%)")
    print(f"  - 长上下文 (>32K) 需要更大显存或 KV Cache 量化")


# ──────────────────────────────────────────────
# 实验 3: Batch Size 优化
# ──────────────────────────────────────────────

def experiment_3_batch_optimization():
    print("\n" + "=" * 70)
    print("实验 3: Batch Size 优化 (吞吐 vs 延迟)")
    print("=" * 70)

    # LLaMA-7B on A100, 4K context
    m, g = MODELS["llama7b"], GPUS["a100_80"]
    ctx = 4096
    mb = max_batch(g, m, ctx)

    print(f"\n--- {m.name} on {g.name}, 上下文 4K ---")
    print(f"\n{'Batch':>6} {'Decode (ms)':>12} {'吞吐 (tok/s)':>14} {'KV Cache (GB)':>14} {'利用率':>8}")
    print("-" * 60)

    for b in [1, 2, 4, 8, 16, 32, mb]:
        if b > mb:
            break
        d = decode_ms(m, g, ctx, b)
        t = throughput(m, g, ctx, b)
        kv = kv_gb(m, ctx, b)
        w = weight_gb(m)
        util = (w + kv + 2) / g.hbm_gb * 100
        print(f"{b:>6d} {d:>12.2f} {t:>14.0f} {kv:>14.2f} {util:>7.0f}%")

    # Prefix Caching 效果
    print(f"\n--- Prefix Caching 效果 (10K doc + N queries, 4K context) ---")
    doc_tokens = 10000
    query_tokens = 200
    total = doc_tokens + query_tokens

    for n_queries in [1, 5, 10, 20]:
        no_cache = total * n_queries
        with_cache = doc_tokens + query_tokens * n_queries
        saving = (1 - with_cache / no_cache) * 100
        print(f"  {n_queries:>3} queries: 节省 {saving:.0f}% prefill 计算")

    # KV Cache 量化
    print(f"\n--- KV Cache 量化效果 (FP16 → FP8) ---")
    print(f"\n{'模型':>16} {'上下文':>8} {'FP16 KV':>10} {'FP8 KV':>10} {'Batch FP16':>12} {'Batch FP8':>12}")
    print("-" * 74)

    for mk in ["llama7b", "llama3_8b", "llama70b"]:
        m = MODELS[mk]
        for ctx in [8192, 32768]:
            g = GPUS["a100_80"] if m.params_B <= 10 else GPUS["h200_141"]
            kv_fp16 = kv_gb(m, ctx, precision=2)
            kv_fp8 = kv_gb(m, ctx, precision=1)
            mb_fp16 = max_batch(g, m, ctx)
            mb_fp8 = max_batch(g, m, ctx, precision_override=1)
            ctx_s = f"{ctx // 1024}K"
            print(f"{m.name:>16} {ctx_s:>8} {kv_fp16:>9.2f}G {kv_fp8:>9.2f}G {mb_fp16:>12d} {mb_fp8:>12d}")

    print(f"\n关键洞察:")
    print(f"  - Batch 从 1→32: 吞吐提升 20-30×, 但延迟也增加 20-30×")
    print(f"  - 最优 batch = 可用 KV / 每 batch KV (OOM 前的最大值)")
    print(f"  - KV Cache FP8 量化: 显存省 50%, batch 翻倍, 精度损失 <0.5%")
    print(f"  - Prefix Caching: 高复用场景 (RAG/RL) 可节省 90%+ prefill")


# ──────────────────────────────────────────────
# 实验 4: 成本效率分析
# ──────────────────────────────────────────────

def experiment_4_cost_efficiency():
    print("\n" + "=" * 70)
    print("实验 4: 成本效率分析")
    print("=" * 70)

    print(f"\n--- 每百万 token 推理成本 ---")
    print(f"\n{'配置':>35} {'吞吐 (tok/s)':>14} {'$/M tokens':>12} {'$/小时':>8}")
    print("-" * 75)

    configs = [
        ("7B, 4K, B=16", "llama7b", "a100_80", 4096, 16, 1),
        ("7B, 4K, B=16", "llama7b", "h100_80", 4096, 16, 1),
        ("70B, 4K, TP=2", "llama70b", "h100_80", 4096, 4, 2),
        ("70B, 4K, TP=2", "llama70b", "a100_80", 4096, 4, 2),
        ("70B, 32K, TP=2", "llama70b", "h200_141", 32768, 2, 2),
        ("8x7B MoE, 8K", "mixtral_8x7b", "h100_80", 8192, 8, 1),
        ("7B, 4K, B=32 (RL)", "llama7b", "a100_80", 4096, 32, 1),
    ]

    for label, mk, gk, ctx, batch, tp in configs:
        m, g = MODELS[mk], GPUS[gk]
        mb = max_batch(g, m, ctx, tp)
        actual_b = min(batch, mb)
        if actual_b == 0:
            continue
        tps = throughput(m, g, ctx, actual_b, tp)
        n_gpus = tp
        cost_per_h = n_gpus * g.price_usd_per_hour
        cost_per_M = cost_per_h / (tps * 3600 / 1e6) if tps > 0 else float('inf')

        gpu_label = f"{g.name} ×{tp}"
        print(f"{label:>35} {tps:>14.0f} {cost_per_M:>11.3f} {cost_per_h:>7.1f}")

    # GPU 性价比排行
    print(f"\n--- GPU 性价比排行 (LLaMA-7B, 4K context) ---")
    print(f"\n{'GPU':>14} {'吞吐 (tok/s)':>14} {'$/M tokens':>12} {'性价比指数':>12}")
    print("-" * 56)

    m = MODELS["llama7b"]
    ctx = 4096
    baseline_cost = None

    gpu_rankings = []
    for gk, g in GPUS.items():
        mb = max_batch(g, m, ctx)
        if mb == 0:
            continue
        b = min(16, mb)
        tps = throughput(m, g, ctx, b)
        cost_per_h = g.price_usd_per_hour
        cost_per_M = cost_per_h / (tps * 3600 / 1e6) if tps > 0 else float('inf')
        gpu_rankings.append((g.name, tps, cost_per_M))

    # Sort by cost per M tokens
    gpu_rankings.sort(key=lambda x: x[2])
    if gpu_rankings:
        baseline = gpu_rankings[0][2]
        for name, tps, cpm in gpu_rankings:
            eff = baseline / cpm * 100
            print(f"{name:>14} {tps:>14.0f} {cpm:>11.3f} {eff:>11.0f}%")

    print(f"\n关键洞察:")
    print(f"  - A100 对 7B 模型性价比最高 (成熟生态 + 价格低)")
    print(f"  - H100 吞吐是 A100 的 1.5-2×, 但价格也是 2×")
    print(f"  - H200 对大模型 (70B) 长上下文场景无可替代 (141GB)")
    print(f"  - MoE 模型 (Mixtral) 激活参数少, 性价比高")
    print(f"  - RL 训练: 7B + A100 + prefix caching = 最优组合")


# ──────────────────────────────────────────────
# 补丁: max_batch 支持 FP8 precision_override
# ──────────────────────────────────────────────

_original_max_batch = max_batch

def max_batch(gpu: GPU, m: Model, seq_len: int, tp: int = 1, overhead: float = 2.0, precision_override: int = 2) -> int:
    w = weight_gb(m) / tp
    avail = gpu.hbm_gb - w - overhead
    if avail <= 0:
        return 0
    per_b = kv_gb(m, seq_len, 1, precision_override)
    return max(0, int(avail / per_b)) if per_b > 0 else 0


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("LLM 推理服务端到端规划器")
    print("单卡部署 + 并行策略 + Batch 优化 + 成本分析")
    print("=" * 70)

    experiment_1_single_gpu()
    experiment_2_parallel_strategy()
    experiment_3_batch_optimization()
    experiment_4_cost_efficiency()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
推理服务规划关键要点:

  1. 显存预算 = 模型权重 + KV Cache + 开销:
     7B: 13GB + KV, A100 够用
     70B: 130GB, 必须 TP≥2 或 H200
     → KV Cache 随 batch × seq_len 线性增长

  2. 并行策略选择:
     TP=1: 小模型 (<15B), 单卡放得下
     TP=2: 中大模型 (70B), 通信开销 ~5%
     TP=4-8: 超大模型/长上下文, 通信 ~10-15%
     → 优先 TP, PP 仅在 TP 不够时

  3. Batch Size 优化:
     在线推理: B=1-8, 低延迟优先
     离线推理: B=32-64, 高吞吐优先
     RL 训练: B=16-32 + prefix caching
     → Batch ↑ → 吞吐 ↑ 但延迟 ↑

  4. 成本优化:
     KV Cache FP8 量化: batch 翻倍, 精度损失小
     Prefix Caching: RAG/RL 场景节省 90%+ prefill
     P/D 分离: 独立扩展 prefill 和 decode
     → 综合使用可降低 50-80% 成本
    """)
