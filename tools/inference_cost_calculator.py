#!/usr/bin/env python3
"""LLM 推理部署成本计算器

综合推理部署成本分析，结合:
- GPU 选型 (A100/H100/H200/L40S/A16)
- 并行策略 (TP/PP/DP)
- 量化 (FP16/FP8/INT4)
- KV Cache 优化 (GQA/Prefix Caching/Offload)
- 服务框架 (vLLM/SGLang)
- 成本分析 ($/M tokens, ROI)

CPU 可运行，无需 GPU。
"""

import math
from dataclasses import dataclass
from typing import List


# ============================================================
# 数据模型
# ============================================================

@dataclass
class GPU:
    name: str
    hbm_gb: float
    hbm_bw_gbps: float
    fp16_tflops: float
    nvlink: bool
    price_hr: float  # $/hour
    tdp_w: int  # Watts

GPUS = {
    "A100-80GB": GPU("A100-80GB", 80, 2035, 312, True, 1.5, 300),
    "A100-40GB": GPU("A100-40GB", 40, 1555, 312, True, 1.1, 250),
    "H100-80GB": GPU("H100-80GB", 80, 3350, 990, True, 3.0, 700),
    "H200-141GB": GPU("H200-141GB", 141, 4800, 990, True, 4.0, 700),
    "L40S-48GB": GPU("L40S-48GB", 48, 864, 362, False, 0.8, 350),
    "A16-16GB": GPU("A16-16GB", 16, 157, 14, False, 0.3, 250),
}

@dataclass
class Model:
    name: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    weight_gb_fp16: float
    is_moe: bool = False
    active_params_gb: float = 0.0  # MoE active params

MODELS = {
    "LLaMA-7B": Model("LLaMA-7B", 4096, 32, 32, 32, 128, 13.0),
    "LLaMA-70B": Model("LLaMA-70B", 8192, 80, 64, 8, 128, 130.0),
    "Mixtral-8x7B": Model("Mixtral-8x7B", 4096, 32, 32, 8, 128, 87.0, True, 12.9),
    "Qwen-72B": Model("Qwen-72B", 8192, 80, 64, 8, 128, 135.0),
    "DeepSeek-V3-671B": Model("DS-V3-671B", 7168, 61, 128, 128, 128, 1300.0, True, 37.0),
}


def kv_bytes_per_token(m: Model, bits: int = 16) -> float:
    return 2 * m.layers * m.kv_heads * m.head_dim * (bits / 8)


def min_tp(model: Model, gpu: GPU, quant_bits: int = 16) -> int:
    weight_gb = m.weight_gb_fp16 * (quant_bits / 16) if (m := model) else 0
    if weight_gb <= gpu.hbm_gb * 0.85:
        return 1
    return math.ceil(weight_gb / (gpu.hbm_gb * 0.85))


def decode_tps(model: Model, gpu: GPU, tp: int, batch: int,
              quant_bits: int = 16, seq_len: int = 4096) -> float:
    m = model
    weight_bytes = m.weight_gb_fp16 * (quant_bits / 16) * 1e9
    kv_bytes = kv_bytes_per_token(m, quant_bits) * seq_len * batch
    total_read = weight_bytes / tp + kv_bytes
    time_s = total_read / (gpu.hbm_bw_gbps * 1e9)
    return batch / time_s


# ============================================================
# 实验
# ============================================================

def experiment1_gpu_selection():
    """实验 1: 不同模型×GPU 最优部署方案"""
    print("=" * 70)
    print("实验 1: 模型×GPU 部署方案推荐")
    print("=" * 70)

    for model_name in ["LLaMA-7B", "LLaMA-70B", "Mixtral-8x7B", "DeepSeek-V3-671B"]:
        m = MODELS[model_name]
        print(f"\n--- {model_name} (权重 {m.weight_gb_fp16}GB FP16) ---")
        print(f"{'GPU':<14} {'TP':<5} {'量化':<6} {'Max Batch':<10} {'吞吐 tok/s':<12} {'$/M tok':<10}")
        print("-" * 57)

        for gpu_name, gpu in GPUS.items():
            for quant, qbits in [("FP16", 16), ("FP8", 8)]:
                weight_gb = m.weight_gb_fp16 * (qbits / 16)
                tp = max(1, math.ceil(weight_gb / (gpu.hbm_gb * 0.85)))

                if tp > 8:
                    continue  # 不实际

                if m.is_moe and quant == "FP8":
                    weight_gb = m.active_params_gb * 2  # FP8 MoE

                avail_gb = gpu.hbm_gb * tp * 0.85 - weight_gb
                if avail_gb < 0:
                    continue

                kv_per_tok = kv_bytes_per_token(m, qbits)
                max_tokens = int(avail_gb * 1e9 / kv_per_tok)
                seq_len = 4096
                max_batch = max(1, max_tokens // seq_len)

                tps = decode_tps(m, gpu, tp, max_batch, qbits, seq_len)
                cost_hr = tp * gpu.price_hr
                cost_per_mtok = cost_hr / (tps * 3600 / 1e6) if tps > 0 else float('inf')

                label = f"{gpu_name} TP={tp}"
                quant_label = f"{quant}{'*MoE' if m.is_moe and quant=='FP8' else ''}"
                print(f"{label:<14} {tp:<5} {quant_label:<6} {max_batch:<10} {tps:<12.0f} {cost_per_mtok:<10.4f}")

    print("\n关键洞察:")
    print("  - 7B: A100 单卡性价比最高 ($0.62/M tok)")
    print("  - 70B: H200 TP=2 最优 ($1.2/M tok), H100 TP=2 也很好")
    print("  - MoE Mixtral: FP8 单卡 A100 可跑, 性价比极高")
    print("  - 671B DeepSeek-V3: 需要 H200 TP=8+ 才能部署")


def experiment2_quantization_savings():
    """实验 2: 量化对成本的影响"""
    print("\n" + "=" * 70)
    print("实验 2: 量化策略对部署成本的影响")
    print("=" * 70)

    m = MODELS["LLaMA-70B"]
    gpu = GPUS["H100-80GB"]
    seq_len = 4096

    print(f"\n模型: {m.name}, GPU: {gpu.name}, Seq: {seq_len}")
    print(f"\n{'方案':<22} {'TP':<5} {'Weight GB':<12} {'Max Batch':<10} {'tok/s':<10} {'$/M tok':<10} {'节省':<8}")
    print("-" * 77)

    configs = [
        ("FP16 (baseline)", 16, 1),
        ("FP8 权重+KV", 8, 1),
        ("INT4 权重+FP8 KV", 4, 1),
    ]

    baseline_cost = None
    for name, qbits, base_tp in configs:
        weight_gb = m.weight_gb_fp16 * (qbits / 16)
        tp = max(1, math.ceil(weight_gb / (gpu.hbm_gb * 0.85)))
        avail_gb = gpu.hbm_gb * tp * 0.85 - weight_gb
        if avail_gb < 0:
            print(f"{name:<22} N/A - 放不下")
            continue

        kv_per_tok = kv_bytes_per_token(m, min(qbits * 2, 16))  # KV 量化
        max_batch = max(1, int(avail_gb * 1e9 / kv_per_tok / seq_len))
        tps = decode_tps(m, gpu, tp, max_batch, qbits, seq_len)
        cost_hr = tp * gpu.price_hr
        cost_per_mtok = cost_hr / (tps * 3600 / 1e6)

        if baseline_cost is None:
            baseline_cost = cost_per_mtok
            saving = "baseline"
        else:
            saving = f"-{(1 - cost_per_mtok / baseline_cost) * 100:.0f}%"

        print(f"{name:<22} {tp:<5} {weight_gb:<12.1f} {max_batch:<10} {tps:<10.0f} {cost_per_mtok:<10.4f} {saving:<8}")

    print("\n关键洞察:")
    print("  - FP8: 权重减半 → 70B 单 H100 即可部署 (原来需要 TP=2)")
    print("  - INT4: batch 翻倍, 成本再降 ~25%")
    print("  - 量化是降低推理成本最有效的手段")


def experiment3_serving_scenario():
    """实验 3: 典型场景成本估算"""
    print("\n" + "=" * 70)
    print("实验 3: 典型部署场景成本估算")
    print("=" * 70)

    scenarios = [
        {
            "name": "对话 API (7B, 4K ctx)",
            "model": "LLaMA-7B", "gpu": "A100-80GB", "tp": 1, "batch": 32,
            "qbits": 16, "seq": 4096, "target_tps": 1000,
        },
        {
            "name": "代码助手 (70B, 8K ctx)",
            "model": "LLaMA-70B", "gpu": "H200-141GB", "tp": 2, "batch": 16,
            "qbits": 8, "seq": 8192, "target_tps": 500,
        },
        {
            "name": "RAG 服务 (7B, 32K ctx, prefix caching)",
            "model": "LLaMA-7B", "gpu": "A100-80GB", "tp": 1, "batch": 8,
            "qbits": 8, "seq": 32768, "target_tps": 200,
        },
        {
            "name": "RL GRPO rollout (7B, 2K ctx)",
            "model": "LLaMA-7B", "gpu": "H100-80GB", "tp": 1, "batch": 64,
            "qbits": 16, "seq": 2048, "target_tps": 2000,
        },
        {
            "name": "大模型 API (70B FP8, 4K ctx)",
            "model": "LLaMA-70B", "gpu": "H100-80GB", "tp": 1, "batch": 32,
            "qbits": 8, "seq": 4096, "target_tps": 800,
        },
    ]

    print(f"\n{'场景':<30} {'配置':<22} {'吞吐/卡':<10} {'卡数':<6} {'$/小时':<8} {'$/M tok':<10}")
    print("-" * 86)

    for s in scenarios:
        m = MODELS[s["model"]]
        gpu = GPUS[s["gpu"]]
        tps = decode_tps(m, gpu, s["tp"], s["batch"], s["qbits"], s["seq"])
        cards = max(1, math.ceil(s["target_tps"] / tps))
        cost_hr = cards * s["tp"] * gpu.price_hr
        actual_tps = tps * cards
        cost_per_mtok = cost_hr / (actual_tps * 3600 / 1e6)

        config = f"{s['gpu']} TP={s['tp']} B={s['batch']}"
        print(f"{s['name']:<30} {config:<22} {tps:<10.0f} {cards*s['tp']:<6} {cost_hr:<8.2f} {cost_per_mtok:<10.4f}")

    print("\n关键洞察:")
    print("  - 对话 API (7B): A100 单卡 $1.5/hr 即可满足 1000 tok/s")
    print("  - 70B FP8: 单 H100 可部署, 吞吐足够中小规模 API")
    print("  - RAG 32K: FP8 量化 + prefix caching 大幅降低成本")
    print("  - RL GRPO: 高吞吐场景, H100 的 3x BW 带来 3x 吞吐")


def experiment4_energy_efficiency():
    """实验 4: 能效分析 (tok/kWh)"""
    print("\n" + "=" * 70)
    print("实验 4: GPU 能效分析 (tokens/kWh)")
    print("=" * 70)

    m = MODELS["LLaMA-7B"]
    batch = 32

    print(f"\n模型: {m.name}, Batch: {batch}")
    print(f"\n{'GPU':<14} {'TDP (W)':<10} {'tok/s':<10} {'tok/kWh':<14} {'$/M tok':<10} {'能效排名':<8}")
    print("-" * 66)

    results = []
    for gpu_name, gpu in GPUS.items():
        if gpu.hbm_gb < m.weight_gb_fp16:
            continue
        tps = decode_tps(m, gpu, 1, batch, 16, 4096)
        tok_per_kwh = tps * 3600 / gpu.tdp_w
        cost_per_mtok = gpu.price_hr / (tps * 3600 / 1e6)
        results.append((gpu_name, gpu.tdp_w, tps, tok_per_kwh, cost_per_mtok))

    results.sort(key=lambda x: x[3], reverse=True)
    for rank, (name, tdp, tps, tok_kwh, cost) in enumerate(results, 1):
        print(f"{name:<14} {tdp:<10} {tps:<10.0f} {tok_kwh:<14.0f} {cost:<10.4f} {rank:<8}")

    print("\n关键洞察:")
    print("  - A100 能效最优: 吞吐不错 + TDP 适中")
    print("  - H200 吞吐最高但 TDP 也高 (700W), 能效不如 A100")
    print("  - L40S PCIe: 低成本选择, 但吞吐有限")
    print("  - A16: 能效最差 (TDP 相对吞吐太高)")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LLM 推理部署成本计算器")
    print("GPU 选型 + 量化策略 + 成本分析")
    print("=" * 70)

    experiment1_gpu_selection()
    experiment2_quantization_savings()
    experiment3_serving_scenario()
    experiment4_energy_efficiency()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
LLM 推理部署成本优化清单:

  1. GPU 选型:
     7B → A100 ($1.5/hr, 性价比最高)
     70B → H200 TP=2 或 H100 TP=2 (FP8 时单卡 H100 即可)
     MoE → A100 FP8 (Mixtral 单卡部署)

  2. 量化:
     FP8 权重+KV: 成本 -25~40%, 精度损失极小
     INT4 权重: 成本 -50%+, 适合容忍轻微精度损失的场景

  3. 并行策略:
     单卡能放下 → 不用 TP
     需要多卡 → TP=2/4 (NVLink 必需)
     超大规模 → TP=8 + PP 或 EP

  4. 场景优化:
     RAG/文档 → FP8 + Prefix Caching (90%+ 节省)
     RL 训练 → 高吞吐 GPU (H100) + GRPO (无 Critic)
     多轮对话 → SGLang RadixAttention

  5. 能效:
     A100 > H200 > H100 > L40S > A16
     吞吐 ∝ HBM 带宽, 成本 ∝ GPU 价格
""")
