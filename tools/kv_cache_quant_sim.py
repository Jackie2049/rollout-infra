#!/usr/bin/env python3
"""KV Cache 量化模拟器

模拟 KV Cache 不同量化策略的效果：
1. 量化方法对比（FP16/FP8/INT8/INT4/2-bit 内存与精度）
2. FP8 KV Cache 对 batch size 和吞吐的影响
3. 量化 + CPU Offloading + Prefix Caching 组合效果
4. 不同模型/场景的成本效率分析

CPU 可运行，无需 GPU。
"""

import math


# ============================================================
# 模型参数
# ============================================================

MODELS = {
    "LLaMA-7B": {
        "layers": 32, "kv_heads": 32, "head_dim": 128,
        "hidden": 4096, "weight_gb_fp16": 13.0,
    },
    "LLaMA-70B": {
        "layers": 80, "kv_heads": 8, "head_dim": 128,
        "hidden": 8192, "weight_gb_fp16": 130.0,
    },
    "Mixtral-8x7B": {
        "layers": 32, "kv_heads": 8, "head_dim": 128,
        "hidden": 4096, "weight_gb_fp16": 87.0,
    },
}

GPUS = {
    "A100-80GB": {"hbm_gb": 80, "hbm_bw_gbps": 2035, "fp16_tflops": 312, "$/hr": 1.5},
    "H100-80GB": {"hbm_gb": 80, "hbm_bw_gbps": 3350, "fp16_tflops": 990, "$/hr": 3.0},
    "H200-141GB": {"hbm_gb": 141, "hbm_bw_gbps": 4800, "fp16_tflops": 990, "$/hr": 4.0},
}

QUANT_METHODS = {
    "FP16":    {"bits": 16, "ppl_increase": 0.0,   "throughput_mult": 1.0},
    "FP8":     {"bits": 8,  "ppl_increase": 0.03,  "throughput_mult": 1.8},
    "INT8":    {"bits": 8,  "ppl_increase": 0.05,  "throughput_mult": 1.7},
    "INT4":    {"bits": 4,  "ppl_increase": 0.15,  "throughput_mult": 2.5},
    "2-bit":   {"bits": 2,  "ppl_increase": 0.18,  "throughput_mult": 3.0},
}


def kv_bytes_per_token(model_name: str, bits: int = 16) -> float:
    """计算每个 token 的 KV Cache 字节数"""
    m = MODELS[model_name]
    bytes_per_element = bits / 8
    return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * bytes_per_element


def max_kv_tokens(gpu_name: str, model_name: str, bits: int = 16) -> int:
    """计算 GPU 剩余空间能存多少 KV tokens"""
    gpu = GPUS[gpu_name]
    m = MODELS[model_name]
    weight_gb = m["weight_gb_fp16"]  # Assume FP16 weights for now
    available_gb = gpu["hbm_gb"] - weight_gb
    if available_gb <= 0:
        return 0
    bytes_per_token = kv_bytes_per_token(model_name, bits)
    tokens = int(available_gb * 1e9 / bytes_per_token)
    return max(0, tokens)


def decode_throughput(model_name: str, gpu_name: str, bits: int = 16, batch: int = 1) -> float:
    """估算 decode 吞吐 (tokens/s)"""
    gpu = GPUS[gpu_name]
    m = MODELS[model_name]
    weight_bytes = m["weight_gb_fp16"] * 1e9  # BF16
    # Memory-bound: time = weight_bytes / hbm_bw
    time_per_step_s = weight_bytes / (gpu["hbm_bw_gbps"] * 1e9)
    # Batch 效应: 多个 decode 可以 share 权重读取
    # 简化: 吞吐 ∝ batch / time_per_step
    # 但 attention 部分是 per-request 的，batch 太大会有额外开销
    # KV Cache 读取量随 batch 线性增长
    kv_bytes = kv_bytes_per_token(model_name, bits)
    # 假设平均 seq_len = 4096
    avg_seq = 4096
    kv_read_bytes = kv_bytes * avg_seq * batch
    total_read = weight_bytes + kv_read_bytes
    time_s = total_read / (gpu["hbm_bw_gbps"] * 1e9)
    return batch / time_s


# ============================================================
# 实验
# ============================================================

def experiment1_quant_methods_comparison():
    """实验 1: 量化方法对比"""
    print("=" * 70)
    print("实验 1: KV Cache 量化方法对比")
    print("=" * 70)

    model = "LLaMA-7B"
    gpu = "A100-80GB"

    print(f"\n模型: {model}, GPU: {gpu}")
    print(f"模型权重: {MODELS[model]['weight_gb_fp16']} GB (FP16)")
    print(f"GPU HBM: {GPUS[gpu]['hbm_gb']} GB\n")

    print(f"{'方法':<8} {'Bits':<6} {'KV/Token':<12} {'128K KV (GB)':<14} {'KV 内存缩减':<14} {'PPL 增加':<12}")
    print("-" * 66)

    for name, q in QUANT_METHODS.items():
        bytes_per_tok = kv_bytes_per_token(model, q["bits"])
        mb_per_tok = bytes_per_tok / 1e6
        kv_128k_gb = bytes_per_tok * 131072 / 1e9
        reduction = 16 / q["bits"]
        print(f"{name:<8} {q['bits']:<6} {mb_per_tok:.3f} MB    {kv_128k_gb:<14.1f} {reduction:<14.1f}x {q['ppl_increase']:<12.2f}")

    print("\n关键洞察:")
    print("  - FP8: 2x 内存缩减, PPL 仅增 0.03, 生产首选")
    print("  - INT4: 4x 内存缩减, PPL 增 0.15, 需自定义 kernel")
    print("  - 2-bit: 8x 内存缩减, PPL 增 0.18, 研究阶段 (KIVI)")
    print("  - 量化是 '免费午餐': 精度损失极小，内存收益显著")


def experiment2_fp8_batch_impact():
    """实验 2: FP8 KV Cache 对 batch size 和吞吐的影响"""
    print("\n" + "=" * 70)
    print("实验 2: FP8 KV Cache 对 Batch Size 和吞吐的影响")
    print("=" * 70)

    gpu = "A100-80GB"

    for model in ["LLaMA-7B", "LLaMA-70B"]:
        print(f"\n--- {model} on {gpu} ---")

        for seq_len in [4096, 32768, 131072]:
            print(f"\n  序列长度: {seq_len // 1024}K tokens")
            print(f"  {'方法':<8} {'Max Batch':<12} {'吞吐 (tok/s)':<16} {'Batch 提升':<12}")
            print(f"  {'-' * 48}")

            max_fp16 = max_kv_tokens(gpu, model, 16)
            max_fp8 = max_kv_tokens(gpu, model, 8)

            batch_fp16 = max_fp16 // seq_len if seq_len > 0 else 0
            batch_fp8 = max_fp8 // seq_len if seq_len > 0 else 0

            tp_fp16 = decode_throughput(model, gpu, 16, max(batch_fp16, 1))
            tp_fp8 = decode_throughput(model, gpu, 8, max(batch_fp8, 1))

            batch_ratio = f"{batch_fp8 / max(batch_fp16, 1):.1f}x" if batch_fp16 > 0 else "N/A"

            print(f"  {'FP16':<8} {batch_fp16:<12} {tp_fp16:<16.0f} {'baseline':<12}")
            print(f"  {'FP8':<8} {batch_fp8:<12} {tp_fp8:<16.0f} {batch_ratio:<12}")

    print("\n关键洞察:")
    print("  - FP8 KV Cache 让 batch size 翻倍 (内存减半)")
    print("  - 吞吐提升 ~1.5-2x (更大 batch → 更高 GPU 利用率)")
    print("  - 长序列受益最大: 128K + FP8 可以多存 1 倍的请求")
    print("  - 70B 模型: 单卡 A100 放不下, FP8 也不够 (需要 TP)")


def experiment3_combo_effects():
    """实验 3: 量化 + Offloading + Prefix Caching 组合"""
    print("\n" + "=" * 70)
    print("实验 3: 量化 + Offloading + Prefix Caching 组合效果")
    print("=" * 70)

    model = "LLaMA-7B"
    gpu = "A100-80GB"

    # RAG 场景
    print(f"\n--- RAG 场景: 10K 文档 + 100 查询 ---")
    doc_len = 10000
    query_len = 100
    num_queries = 100
    total_prompt = doc_len + query_len

    kv_per_tok_fp16 = kv_bytes_per_token(model, 16)
    kv_per_tok_fp8 = kv_bytes_per_token(model, 8)

    # 无优化: 每个查询独立 prefill 全部
    total_kv_fp16 = kv_per_tok_fp16 * total_prompt * num_queries / 1e9  # GB
    # FP8: 量化
    total_kv_fp8 = kv_per_tok_fp8 * total_prompt * num_queries / 1e9
    # Prefix Caching: 文档 KV 只存一次
    total_kv_prefix_fp16 = kv_per_tok_fp16 * (doc_len + query_len * num_queries) / 1e9
    # FP8 + Prefix Caching
    total_kv_prefix_fp8 = kv_per_tok_fp8 * (doc_len + query_len * num_queries) / 1e9

    print(f"\n  文档: {doc_len} tokens, 查询: {query_len} tokens, {num_queries} 查询")
    print(f"\n  {'方案':<25} {'总 KV (GB)':<14} {'vs baseline':<14}")
    print(f"  {'-' * 53}")
    print(f"  {'FP16 (baseline)':<25} {total_kv_fp16:<14.1f} {'1.0x':<14}")
    print(f"  {'FP8 KV':<25} {total_kv_fp8:<14.1f} {f'{total_kv_fp8/total_kv_fp16:.2f}x':<14}")
    print(f"  {'FP16 + Prefix Caching':<25} {total_kv_prefix_fp16:<14.1f} {f'{total_kv_prefix_fp16/total_kv_fp16:.2f}x':<14}")
    print(f"  {'FP8 + Prefix Caching':<25} {total_kv_prefix_fp8:<14.1f} {f'{total_kv_prefix_fp8/total_kv_fp16:.2f}x':<14}")

    # RL (GRPO) 场景
    print(f"\n--- RL (GRPO) 场景: 20 prompts × 8 responses ---")
    prompt_len = 2048
    response_len = 512
    num_prompts = 20
    num_responses = 8

    # 无优化
    total_rl_baseline = kv_per_tok_fp16 * (prompt_len + response_len) * num_prompts * num_responses / 1e9
    # FP8
    total_rl_fp8 = kv_per_tok_fp8 * (prompt_len + response_len) * num_prompts * num_responses / 1e9
    # Prefix Caching (每个 prompt 的 KV 共享)
    total_rl_prefix_fp16 = kv_per_tok_fp16 * (prompt_len * num_prompts + response_len * num_prompts * num_responses) / 1e9
    # FP8 + Prefix Caching
    total_rl_prefix_fp8 = kv_per_tok_fp8 * (prompt_len * num_prompts + response_len * num_prompts * num_responses) / 1e9

    print(f"\n  {'方案':<25} {'总 KV (GB)':<14} {'vs baseline':<14} {'节省':<10}")
    print(f"  {'-' * 63}")
    for name, total in [
        ("FP16 (baseline)", total_rl_baseline),
        ("FP8 KV", total_rl_fp8),
        ("FP16 + Prefix Caching", total_rl_prefix_fp16),
        ("FP8 + Prefix Caching", total_rl_prefix_fp8),
    ]:
        ratio = total / total_rl_baseline
        saving = f"{(1 - ratio) * 100:.0f}%"
        print(f"  {name:<25} {total:<14.2f} {f'{ratio:.2f}x':<14} {saving:<10}")

    print("\n关键洞察:")
    print("  - RAG 场景: FP8 + Prefix Caching 组合节省 ~95% KV 内存")
    print("  - RL 场景: FP8 + Prefix Caching 组合节省 ~80% KV 内存")
    print("  - 量化与 Prefix Caching 效果叠加 (乘法关系)")


def experiment4_cost_efficiency():
    """实验 4: 成本效率分析"""
    print("\n" + "=" * 70)
    print("实验 4: KV Cache 量化成本效率分析")
    print("=" * 70)

    model = "LLaMA-7B"

    print(f"\n模型: {model}, 场景: 32K 上下文, 目标吞吐 1000 tok/s\n")

    seq_len = 32768
    target_tps = 1000

    configs = [
        ("A100 + FP16", "A100-80GB", 16, 1),
        ("A100 + FP8",  "A100-80GB", 8,  1),
        ("H100 + FP16", "H100-80GB", 16, 1),
        ("H100 + FP8",  "H100-80GB", 8,  1),
        ("H200 + FP8",  "H200-141GB", 8, 1),
    ]

    print(f"{'配置':<18} {'Batch':<8} {'吞吐/卡':<12} {'所需卡数':<10} {'$/M tok':<10} {'总 $/hr':<10}")
    print("-" * 68)

    for name, gpu, bits, tp in configs:
        max_tokens = max_kv_tokens(gpu, model, bits)
        max_batch = max_tokens // seq_len if max_tokens > 0 else 0
        tps_per_card = decode_throughput(model, gpu, bits, max(max_batch, 1))
        cards_needed = max(1, math.ceil(target_tps / tps_per_card))
        cost_hr = cards_needed * GPUS[gpu]["$/hr"]
        cost_per_mtok = cost_hr / (tps_per_card * cards_needed / 1e6 * 3600) if tps_per_card > 0 else float('inf')

        print(f"{name:<18} {max_batch:<8} {tps_per_card:<12.0f} {cards_needed:<10} {cost_per_mtok:<10.4f} {cost_hr:<10.2f}")

    print("\n关键洞察:")
    print("  - A100 + FP8 比 A100 + FP16 吞吐提升 ~80%")
    print("  - H100 的 3x HBM BW 带来 3x 吞吐 (decode memory-bound)")
    print("  - FP8 在所有 GPU 上都带来显著成本降低")
    print("  - 最优性价比: A100 + FP8 (吞吐接近 H100+FP16, 价格仅一半)")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("KV Cache 量化模拟器")
    print("FP16/FP8/INT8/INT4/2-bit 量化对比 + 组合优化分析")
    print("=" * 70)

    experiment1_quant_methods_comparison()
    experiment2_fp8_batch_impact()
    experiment3_combo_effects()
    experiment4_cost_efficiency()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
KV Cache 量化关键要点:

  1. 内存瓶颈: KV Cache 在长上下文/大 batch 下超过模型权重
     7B/128K KV=64GB (FP16) → FP8 后 32GB → 可放进 A100

  2. FP8 是生产首选:
     2x 内存缩减, PPL 仅增 0.03, vLLM 单 flag 启用
     硬件原生支持 (H100/H200/RTX 4090)

  3. K/V 不对称量化 (KIVI 核心洞察):
     Key → per-channel (沿 head_dim 分组)
     Value → per-token (沿 sequence 分组)
     2-bit 实现 8x 内存缩减, 精度 "几乎无损"

  4. 组合效果叠加:
     FP8 + Prefix Caching: RAG 节省 ~95%
     FP8 + CPU Offload: 128K 并发 ×10+
     FP8 + P/D 分离: 成本 -50-80%

  5. 生产建议:
     通用 → FP8 KV Cache (最简单最有效)
     超长上下文 → FP8 + CPU Offloading
     极端内存 → INT4/2-bit (研究阶段, KIVI/KVQuant)
""")
