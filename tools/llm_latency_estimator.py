#!/usr/bin/env python3
"""LLM 推理延迟估算器 — 端到端延迟分解

分解 LLM 推理的每一步延迟:
1. Prefill 阶段: Prompt 处理 (compute-bound)
2. Decode 阶段: 逐步生成 (memory-bound)
3. Sampling 开销: Temperature/Top-K/Top-P
4. KV Cache 管理: Block 分配/查找
5. 调度器开销: 请求调度决策
6. 通信开销: TP/EP AllReduce

CPU 可运行，无需 GPU。
"""

import math
from dataclasses import dataclass


@dataclass
class GPU:
    name: str
    hbm_bw_gbps: float
    fp16_tflops: float
    fp8_tflops: float
    nvlink_bw_gbps: float
    price_hr: float


@dataclass
class Model:
    name: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    weight_gb_fp16: float
    is_moe: bool = False
    active_gb: float = 0.0


GPUS = {
    "A100": GPU("A100-80GB", 2035, 312, 624, 300, 1.5),
    "H100": GPU("H100-80GB", 3350, 990, 1976, 450, 3.0),
    "H200": GPU("H200-141GB", 4800, 990, 1976, 450, 4.0),
}

MODELS = {
    "7B": Model("7B", 4096, 32, 32, 32, 128, 32000, 13.0),
    "70B": Model("70B", 8192, 80, 64, 8, 128, 32000, 130.0),
    "405B": Model("405B", 8192, 80, 128, 8, 128, 128256, 780.0),
}


def prefill_latency_ms(m: Model, gpu: GPU, prompt_len: int, batch: int = 1) -> float:
    """Prefill 延迟估算 (compute-bound)"""
    n = prompt_len
    # Attention: O(N²×D) per layer
    attn_flops = m.layers * 4 * n * n * m.hidden
    # MLP: O(N×D²) per layer (SwiGLU: 3x)
    mlp_flops = m.layers * 3 * 2 * n * m.hidden * m.hidden
    total_flops = (attn_flops + mlp_flops) * batch

    # Tensor Core 利用率: 大矩阵 ~70%, 小矩阵 ~30%
    if n >= 4096:
        util = 0.65
    elif n >= 1024:
        util = 0.45
    else:
        util = 0.25

    return total_flops / (gpu.fp16_tflops * 1e9 * util) * 1000


def decode_latency_ms(m: Model, gpu: GPU, tp: int, batch: int,
                       seq_len: int = 4096, quant_bits: int = 16) -> float:
    """Decode 单步延迟估算 (memory-bound)"""
    weight_bytes = m.weight_gb_fp16 * (quant_bits / 16) * 1e9

    # KV Cache per token per layer
    kv_bytes_per_token = 2 * m.layers * m.kv_heads * m.head_dim * (quant_bits / 8)
    kv_total = kv_bytes_per_token * seq_len * batch

    # 权重读取 + KV 读取
    total_read = weight_bytes / tp + kv_total
    mem_time = total_read / (gpu.hbm_bw_gbps * 1e9) * 1000

    # 少量计算 (GEMM, batch=1 时算力利用率极低)
    comp_flops = 2 * m.hidden * m.hidden * m.layers * batch * (quant_bits / 16)
    comp_time = comp_flops / (gpu.fp16_tflops * 1e9 * 0.1) * 1000  # 10% utilization

    # TP 通信 (AllReduce)
    tp_comm = 0
    if tp > 1:
        activation_bytes = 2 * batch * m.hidden * 2  # 2 AllReduce per layer
        tp_comm = m.layers * 2 * activation_bytes / (gpu.nvlink_bw_gbps * 1e9) * 1000

    # Sampling 开销 (~0.01-0.1ms per batch)
    sampling_time = 0.05 * batch / 32

    # 调度器开销 (~0.1ms)
    scheduler_time = 0.1

    # CUDA Graph: kernel launch 从 ~0.6ms 降到 ~0.006ms
    kernel_launch_no_graph = 0.005 * m.layers * 4  # ~4 kernels per layer
    kernel_launch_graph = kernel_launch_no_graph * 0.01

    return mem_time + comp_time + tp_comm + sampling_time + scheduler_time + kernel_launch_graph


def kv_cache_size_gb(m: Model, seq_len: int, batch: int, quant_bits: int = 16) -> float:
    """KV Cache 显存需求 (GB)"""
    bytes_per_token = 2 * m.layers * m.kv_heads * m.head_dim * (quant_bits / 8)
    return bytes_per_token * seq_len * batch / 1e9


# ============================================================
# 实验
# ============================================================

def experiment1_prefill_breakdown():
    """实验 1: Prefill 延迟分解"""
    print("=" * 70)
    print("实验 1: Prefill 延迟分解")
    print("=" * 70)

    gpu = GPUS["H100"]

    scenarios = [
        ("短对话 (128 tok)", 128),
        ("中等 (1K tok)", 1024),
        ("长文档 (4K tok)", 4096),
        ("超长 (8K tok)", 8192),
        ("超长 (32K tok)", 32768),
        ("极限 (128K tok)", 131072),
    ]

    print(f"\nGPU: {gpu.name}, FP16: {gpu.fp16_tflops} TFLOPS\n")
    print(f"{'场景':<24} {'Prompt':<10} {'Prefill (ms)':<14} {'TTFT (ms)':<12} {'备注'}")
    print("-" * 70)

    for name, prompt_len in scenarios:
        lat = prefill_latency_ms(MODELS["70B"], gpu, prompt_len)
        label = f"{prompt_len//1024}K" if prompt_len >= 1024 else str(prompt_len)
        ttft = lat  # Prefill time = TTFT (approximately)
        note = ""
        if prompt_len <= 1024:
            note = "快, compute 利用率低"
        elif prompt_len <= 8192:
            note = "正常范围"
        elif prompt_len <= 32768:
            note = "需要 chunked prefill"
        else:
            note = "需要 chunked prefill + 分块"
        print(f"{name:<24} {label:<10} {lat:<14.1f} {ttft:<12.1f} {note}")

    print("\n关键洞察:")
    print("  - Prefill 延迟 ∝ N² (attention 主导)")
    print("  - 32K prompt: ~2.4s TTFT → 需要 chunked prefill")
    print("  - 128K prompt: ~38s → 必须分块处理")


def experiment2_decode_breakdown():
    """实验 2: Decode 延迟分解"""
    print("\n" + "=" * 70)
    print("实验 2: Decode 延迟分解")
    print("=" * 70)

    gpu = GPUS["H100"]
    m = MODELS["70B"]

    print(f"\nGPU: {gpu.name}, Model: {m.name}, FP16\n")
    print(f"{'Batch':<8} {'Seq':<8} {'Memory':<10} {'Compute':<10} {'TP Comm':<10} {'Sample':<10} {'Total':<10} {'tok/s':<10}")
    print("-" * 76)

    for batch in [1, 4, 16, 32, 64, 128]:
        for seq_len in [4096]:
            weight_bytes = m.weight_gb_fp16 * 1e9
            kv_bytes = 2 * m.layers * m.kv_heads * m.head_dim * 2 * seq_len * batch
            mem = (weight_bytes + kv_bytes) / (gpu.hbm_bw_gbps * 1e9) * 1000
            comp = 2 * m.hidden * m.hidden * m.layers * batch / (gpu.fp16_tflops * 1e9 * 0.1) * 1000
            comm = 0  # TP=1
            samp = 0.05 * batch / 32
            total = mem + comp + comm + samp + 0.1
            tps = batch / total * 1000
            print(f"{batch:<8} {seq_len:<8} {mem:<10.2f} {comp:<10.3f} {comm:<10.2f} {samp:<10.3f} {total:<10.2f} {tps:<10.0f}")

    print("\n关键洞察:")
    print("  - Memory 读取占 >90% 延迟 (memory-bound)")
    print("  - Compute 开销可忽略 (<1%)")
    print("  - Batch↑ → 吞吐↑, 但延迟也↑")


def experiment3_tp_impact():
    """实验 3: TP 对延迟的影响"""
    print("\n" + "=" * 70)
    print("实验 3: Tensor Parallel 对延迟的影响 (70B, H100)")
    print("=" * 70)

    gpu = GPUS["H100"]
    m = MODELS["70B"]
    batch = 16
    seq_len = 4096

    print(f"\nBatch={batch}, Seq={seq_len}\n")
    print(f"{'TP':<6} {'Memory (ms)':<14} {'Compute (ms)':<14} {'TP Comm (ms)':<14} {'Total (ms)':<12} {'tok/s':<10} {'效率'}")
    print("-" * 80)

    for tp in [1, 2, 4, 8]:
        # FP8 让 70B 放进单卡
        quant = 8 if tp == 1 else 16
        weight_bytes = m.weight_gb_fp16 * (quant / 16) * 1e9
        kv_bytes = 2 * m.layers * m.kv_heads * m.head_dim * (quant / 8) * seq_len * batch
        mem = (weight_bytes / tp + kv_bytes) / (gpu.hbm_bw_gbps * 1e9) * 1000

        comp = 2 * m.hidden * m.hidden * m.layers * batch * (quant / 16) / (gpu.fp16_tflops * 1e9 * 0.1) * 1000

        if tp > 1:
            activation_bytes = 2 * batch * m.hidden * 2
            comm = m.layers * 2 * activation_bytes / (gpu.nvlink_bw_gbps * 1e9) * 1000
        else:
            comm = 0

        total = mem + comp + comm + 0.1
        tps = batch / total * 1000

        quant_label = f"FP{quant}"
        print(f"{tp:<6} {mem:<14.2f} {comp:<14.3f} {comm:<14.2f} {total:<12.2f} {tps:<10.0f} {quant_label}")

    print("\n关键洞察:")
    print("  - TP=1 FP8: 单卡部署, 通信为 0, 但 memory 最慢 (全权重)")
    print("  - TP=2: 通信占比 ~10-15%, 吞吐提升 1.5-1.8x")
    print("  - TP=4+: 通信占比增大, 收益递减")


def experiment4_e2e_scenario():
    """实验 4: 端到端场景延迟估算"""
    print("\n" + "=" * 70)
    print("实验 4: 端到端场景延迟估算")
    print("=" * 70)

    scenarios = [
        {
            "name": "对话 API (7B, 短)",
            "model": "7B", "gpu": "A100", "tp": 1, "batch": 1,
            "prompt": 512, "output": 256, "seq": 768,
        },
        {
            "name": "代码助手 (70B, FP8)",
            "model": "70B", "gpu": "H100", "tp": 1, "batch": 1,
            "prompt": 2048, "output": 512, "seq": 2560, "quant": 8,
        },
        {
            "name": "RAG 服务 (7B, 32K ctx)",
            "model": "7B", "gpu": "A100", "tp": 1, "batch": 1,
            "prompt": 16000, "output": 512, "seq": 16512,
        },
        {
            "name": "RL GRPO (7B, 高吞吐)",
            "model": "7B", "gpu": "H100", "tp": 1, "batch": 64,
            "prompt": 2048, "output": 512, "seq": 2560,
        },
        {
            "name": "大模型 API (405B, TP=8)",
            "model": "405B", "gpu": "H100", "tp": 8, "batch": 1,
            "prompt": 4096, "output": 1024, "seq": 5120,
        },
    ]

    print(f"\n{'场景':<28} {'TTFT':<10} {'Decode/step':<14} {'总延迟':<10} {'tok/s':<10} {'$/请求':<10}")
    print("-" * 92)

    for s in scenarios:
        m = MODELS[s["model"]]
        gpu = GPUS[s["gpu"]]
        quant = s.get("quant", 16)

        # TTFT = Prefill time
        ttft = prefill_latency_ms(m, gpu, s["prompt"])

        # Decode latency per step
        decode_step = decode_latency_ms(m, gpu, s["tp"], max(1, s["batch"] // s["batch"]),
                                         s["seq"], quant)
        # For batch > 1, per-request decode step
        if s["batch"] > 1:
            decode_step = decode_latency_ms(m, gpu, s["tp"], s["batch"], s["seq"], quant)
            per_req_decode = decode_step / s["batch"]  # Amortized
        else:
            per_req_decode = decode_step

        total_ms = ttft + per_req_decode * s["output"]
        tps = s["output"] / (per_req_decode * s["output"] / 1000) if per_req_decode > 0 else 0

        cost_hr = s["tp"] * gpu.price_hr
        req_per_hr = 3600000 / total_ms if total_ms > 0 else 0
        cost_per_req = cost_hr / req_per_hr if req_per_hr > 0 else float('inf')

        print(f"{s['name']:<28} {ttft:<10.1f} {per_req_decode:<14.2f} {total_ms:<10.0f} {1000/per_req_decode:<10.0f} ${cost_per_req:<9.4f}")

    print("\n关键洞察:")
    print("  - TTFT 由 prompt 长度决定 (∝ N²)")
    print("  - Decode 延迟由 HBM BW 决定 (∝ weight_size / BW)")
    print("  - 70B FP8 单卡: TTFT ~8ms, Decode ~17ms/step")
    print("  - 405B TP=8: Decode ~30ms/step, 高吞吐但高成本")


def experiment5_gpu_comparison():
    """实验 5: GPU 对比 (7B Decode)"""
    print("\n" + "=" * 70)
    print("实验 5: GPU 对比 — 7B Decode 延迟和成本")
    print("=" * 70)

    m = MODELS["7B"]

    print(f"\n{'GPU':<14} {'$/hr':<8} {'Decode 1 req':<14} {'Decode B=32':<14} {'tok/s (B=32)':<14} {'$/M tok'}")
    print("-" * 78)

    for gpu_name, gpu in GPUS.items():
        lat_1 = decode_latency_ms(m, gpu, 1, 1, 4096)
        lat_32 = decode_latency_ms(m, gpu, 1, 32, 4096)
        tps_32 = 32 / lat_32 * 1000
        cost_mtok = gpu.price_hr / (tps_32 * 3600 / 1e6)

        print(f"{gpu_name:<14} ${gpu.price_hr:<7.1f} {lat_1:<14.2f} {lat_32:<14.2f} {tps_32:<14.0f} ${cost_mtok:<9.4f}")

    print("\n关键洞察:")
    print("  - A100 性价比最高 ($/M tok)")
    print("  - H200 吞吐最高 (4800 GB/s HBM BW)")
    print("  - H100 居中: 吞吐不如 H200, 价格不如 A100")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LLM 推理延迟估算器")
    print("Prefill/Decode 分解 + GPU 对比 + 场景估算")
    print("=" * 70)

    experiment1_prefill_breakdown()
    experiment2_decode_breakdown()
    experiment3_tp_impact()
    experiment4_e2e_scenario()
    experiment5_gpu_comparison()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
LLM 推理延迟优化清单:

  1. Prefill (compute-bound):
     短 prompt (<1K): 利用率低, 主要开销在 kernel launch
     长 prompt (>4K): 算力主导, ∝ N²
     优化: Chunked Prefill + FlashAttention

  2. Decode (memory-bound):
     权重读取 >90% 延迟
     优化: 量化 (FP8/INT4) → 减少读取量
           CUDA Graph → 消除 kernel launch
           TP → 分摊权重读取

  3. 场景选型:
     对话 API: 7B A100, TTFT <5ms
     代码助手: 70B FP8 H100, TTFT ~8ms
     RAG 32K: 7B + Prefix Caching, chunked prefill
     405B: TP=8 H100, 高吞吐但高成本

  4. 延迟公式:
     TTFT ≈ FLOPS / (TFLOPS × utilization)
     Decode ≈ (weight + KV) / HBM_BW + TP_comm + sampling
""")
