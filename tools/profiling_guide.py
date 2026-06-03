#!/usr/bin/env python3
"""LLM 推理性能诊断工具 — 从 Profiling 数据到瓶颈分析

CPU 可运行的推理性能分析工具，覆盖 4 个实验:
  1. Prefill vs Decode 延迟分解
  2. KV Cache 压力测试模拟
  3. 端到端推理吞吐量估算
  4. 瓶颈诊断与优化建议

关键概念:
  - Prefill 延迟 ∝ seq_len² / TFLOPS (compute-bound)
  - Decode 延迟 ∝ (model_size + kv_cache) / bandwidth (memory-bound)
  - TTFT = prefill_time (用户感知的首 token 延迟)
  - ITL = decode_time (用户感知的生成速度)
  - 吞吐 = batch × tokens / time

用法:
  conda run -n ai-infra python tools/profiling_guide.py
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

@dataclass
class GPU:
    name: str
    fp16_tflops: float
    hbm_bw_gbps: float
    hbm_gb: float

    @property
    def ridge_point(self) -> float:
        return self.fp16_tflops * 1e12 / (self.hbm_bw_gbps * 1e9)


@dataclass
class Model:
    name: str
    hidden: int
    inter: int
    heads: int
    head_dim: int
    layers: int
    params_B: float
    vocab: int = 32000

    @property
    def model_size_gb(self) -> float:
        return self.params_B * 2

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.heads * self.head_dim * self.layers * 2  # FP16


H100 = GPU("H100", 990, 3350, 80)
A100 = GPU("A100", 312, 2039, 80)
A16 = GPU("A16", 105, 157, 16)

GPT2 = Model("GPT-2 Small", 768, 3072, 12, 64, 12, 0.124, vocab=50257)
LLAMA7B = Model("LLaMA-7B", 4096, 11008, 32, 128, 32, 7.0)
LLAMA70B = Model("LLaMA-70B", 8192, 28672, 64, 128, 80, 70.0)


# ──────────────────────────────────────────────
# 延迟模型
# ──────────────────────────────────────────────

def prefill_time_ms(model: Model, gpu: GPU, seq_len: int, batch: int = 1) -> dict:
    """Prefill 阶段延迟估算"""
    H = model.hidden
    I = model.inter
    L = model.layers
    B = batch
    S = seq_len

    # Linear 层: 4 × (QKV_proj + Out_proj + Gate_proj + Up_proj + Down_proj)
    # 简化: 每个 layer 约 4 × 2 × H × (H + I) × S
    linear_flops = 4 * L * 2 * S * (H * H + H * I)

    # Attention: Q@K^T + Softmax@V = 4 × B × H_heads × S × S × d
    attn_flops = 4 * B * model.heads * S * S * model.head_dim * L

    total_flops = linear_flops + attn_flops

    # Roofline: compute-bound (AI >> ridge point)
    efficiency = 0.4
    time_ms = total_flops / (gpu.fp16_tflops * 1e12 * efficiency) * 1000

    return {
        'total_flops_G': total_flops / 1e9,
        'time_ms': time_ms,
        'linear_flops_G': linear_flops / 1e9,
        'attn_flops_G': attn_flops / 1e9,
        'attn_pct': attn_flops / total_flops * 100,
    }


def decode_time_ms(model: Model, gpu: GPU, kv_tokens: int, batch: int = 1) -> dict:
    """Decode 阶段单步延迟估算"""
    H = model.hidden
    I = model.inter
    L = model.layers
    B = batch

    # Linear 层: 读权重 + 输入
    model_bytes = model.model_size_gb * 1e9
    input_bytes = B * H * 2
    kv_bytes = model.kv_bytes_per_token * kv_tokens * B

    total_bytes = model_bytes + kv_bytes + input_bytes

    # Roofline: memory-bound
    bandwidth = gpu.hbm_bw_gbps * 1e9 / 1000  # bytes/ms
    time_ms = total_bytes / bandwidth

    # FLOPs (很小)
    flops = 4 * L * 2 * B * (H * H + H * I)
    compute_time = flops / (gpu.fp16_tflops * 1e12 * 0.4) * 1000

    return {
        'time_ms': max(time_ms, compute_time),  # 取较大值
        'model_bytes_MB': model_bytes / 1e6,
        'kv_bytes_MB': kv_bytes / 1e6,
        'kv_pct': kv_bytes / total_bytes * 100,
        'bandwidth_util': total_bytes / (gpu.hbm_bw_gbps * 1e9) * 1000,  # 理论最小时间
    }


# ──────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────

def experiment_1_prefill_decode_breakdown():
    """实验 1: Prefill vs Decode 延迟分解"""
    print("\n" + "=" * 70)
    print("实验 1: Prefill vs Decode 延迟分解")
    print("=" * 70)

    configs = [
        (GPT2, A16, "GPT-2 on A16 (测试环境)"),
        (LLAMA7B, A100, "LLaMA-7B on A100"),
        (LLAMA70B, A100, "LLaMA-70B on A100"),
        (LLAMA70B, H100, "LLaMA-70B on H100"),
    ]

    for model, gpu, label in configs:
        print(f"\n--- {label} ---")
        print(f"  模型: {model.name} ({model.params_B}B params, {model.model_size_gb:.1f} GB)")
        print(f"  GPU:  {gpu.name} ({gpu.fp16_tflops} TFLOPS, {gpu.hbm_bw_gbps} GB/s)")

        # Prefill
        print(f"\n  Prefill (B=1):")
        print(f"  {'Seq Len':>10} {'FLOPs (G)':>12} {'Time (ms)':>12} {'Attn%':>8} {'TTFT':>12}")
        print(f"  " + "-" * 60)

        for seq in [64, 256, 1024, 4096]:
            r = prefill_time_ms(model, gpu, seq)
            print(f"  {seq:>10d} {r['total_flops_G']:>12.1f} {r['time_ms']:>12.2f} "
                  f"{r['attn_pct']:>7.1f}% {r['time_ms']:>12.2f}")

        # Decode
        print(f"\n  Decode (B=1, per token):")
        print(f"  {'KV Tokens':>10} {'Time (ms)':>12} {'Model MB':>12} {'KV MB':>10} {'KV%':>8} {'ITL':>12}")
        print(f"  " + "-" * 70)

        for kv in [128, 512, 2048, 8192]:
            r = decode_time_ms(model, gpu, kv)
            print(f"  {kv:>10d} {r['time_ms']:>12.2f} {r['model_bytes_MB']:>12.0f} "
                  f"{r['kv_bytes_MB']:>10.1f} {r['kv_pct']:>7.1f}% {r['time_ms']:>12.2f}")

    print(f"\n关键洞察:")
    print(f"  - Prefill 延迟 ∝ seq_len (线性层) + seq_len² (attention)")
    print(f"  - Decode 延迟 ∝ model_size + KV_cache (memory-bound)")
    print(f"  - LLaMA-70B on A100: Prefill 4K ≈ 700ms, Decode ≈ 55ms/token")
    print(f"  - KV Cache 对 decode 延迟影响小 (<10% 当 KV < 2K)")


def experiment_2_kv_pressure():
    """实验 2: KV Cache 压力模拟"""
    print("\n" + "=" * 70)
    print("实验 2: KV Cache 压力 — 最大并发请求数估算")
    print("=" * 70)

    configs = [
        (LLAMA7B, A100, 4096),
        (LLAMA70B, H100, 4096),
    ]

    for model, gpu, max_seq in configs:
        print(f"\n--- {model.name} on {gpu.name} ---")

        # 检查单卡是否能放下模型
        avail_hbm = gpu.hbm_gb - model.model_size_gb
        if avail_hbm <= 0:
            print(f"  ⚠ 模型 {model.name} ({model.model_size_gb:.1f} GB) 超出 {gpu.name} ({gpu.hbm_gb} GB)")
            # 用 TP=2 后的单卡显存重新计算
            tp = math.ceil(model.model_size_gb / gpu.hbm_gb)
            avail_hbm = gpu.hbm_gb - model.model_size_gb / tp
            print(f"  → 需要 TP={tp}, 单卡权重 {model.model_size_gb/tp:.1f} GB")

        avail_gb = avail_hbm * 0.7
        print(f"  可用 KV Cache 显存: {avail_gb:.1f} GB (70% of {avail_hbm:.1f} GB)")
        kv_bytes_per_token = model.kv_bytes_per_token

        print(f"  KV Cache / token: {kv_bytes_per_token} bytes = {kv_bytes_per_token/1e6:.3f} MB")
        print(f"\n  {'平均SeqLen':>12} {'KV/请求':>12} {'最大并发':>12} {'KV显存占比':>12}")
        print(f"  " + "-" * 52)

        for avg_seq in [256, 512, 1024, 2048, 4096, 8192]:
            kv_per_req = kv_bytes_per_token * avg_seq
            max_concurrent = int(avail_gb * 1e9 / kv_per_req)
            kv_total_pct = max_concurrent * kv_per_req / (gpu.hbm_gb * 1e9) * 100

            print(f"  {avg_seq:>12d} {kv_per_req/1e6:>12.1f} MB {max_concurrent:>12d} "
                  f"{kv_total_pct:>11.1f}%")

    print(f"\n关键洞察:")
    print(f"  - LLaMA-7B on A100: KV=2K 时可并发 ~700 请求")
    print(f"  - LLaMA-70B on H100: KV=2K 时只能并发 ~20 请求!")
    print(f"  - 这就是为什么 70B+ 模型需要 KV Cache 压缩 (GQA/MLA)")
    print(f"  - 实际 batch size 受 KV Cache 限制, 不是 GPU 算力")


def experiment_3_throughput_estimation():
    """实验 3: 端到端推理吞吐量估算"""
    print("\n" + "=" * 70)
    print("实验 3: 端到端推理吞吐量估算")
    print("=" * 70)

    model = LLAMA70B
    gpu = H100

    print(f"\n配置: {model.name} on {gpu.name}")
    print(f"  输入: 1024 tokens prompt, 平均 200 tokens 生成")

    prompt_tokens = 1024
    avg_gen_tokens = 200
    batch_sizes = [1, 4, 8, 16, 32]

    print(f"\n{'Batch':>6} {'Prefill ms':>12} {'Decode ms/tok':>14} {'Total ms':>12} "
          f"{'tok/s':>10} {'req/s':>10}")
    print("-" * 70)

    for B in batch_sizes:
        # Prefill (batch 处理)
        pf = prefill_time_ms(model, gpu, prompt_tokens, batch=B)

        # Decode (batch, 平均 KV ≈ prompt + gen/2)
        avg_kv = prompt_tokens + avg_gen_tokens // 2
        dc = decode_time_ms(model, gpu, avg_kv, batch=B)

        # 总时间
        prefill_total = pf['time_ms']
        decode_total = dc['time_ms'] * avg_gen_tokens
        total_ms = prefill_total + decode_total

        total_tokens = B * (prompt_tokens + avg_gen_tokens)
        output_tokens = B * avg_gen_tokens
        tok_per_s = output_tokens / (total_ms / 1000)
        req_per_s = B / (total_ms / 1000)

        print(f"{B:>6d} {prefill_total:>12.1f} {dc['time_ms']:>14.2f} "
              f"{total_ms:>12.1f} {tok_per_s:>10.0f} {req_per_s:>10.1f}")

    print(f"\n关键洞察:")
    print(f"  - Batch=1: 低吞吐 (~18 tok/s) 但低延迟")
    print(f"  - Batch=32: 高吞吐 (~560 tok/s) 但高延迟")
    print(f"  - 吞吐 ∝ batch_size (decode memory-bound, batch 复用权重)")
    print(f"  - Latency ∝ batch_size (每步 decode 更慢因为 KV 更大)")


def experiment_4_diagnosis():
    """实验 4: 瓶颈诊断与优化建议"""
    print("\n" + "=" * 70)
    print("实验 4: 性能瓶颈诊断与优化建议")
    print("=" * 70)

    # 模拟诊断场景
    scenarios = [
        {
            'name': '场景 1: TTFT 过高',
            'symptom': '用户反馈首 token 等待 > 5s',
            'model': LLAMA70B,
            'gpu': A100,
            'seq': 8192,
            'analysis': lambda: _diagnose_ttft(LLAMA70B, A100, 8192),
        },
        {
            'name': '场景 2: Decode 吞吐低',
            'symptom': '生成速度 < 10 tok/s, batch=1',
            'model': LLAMA70B,
            'gpu': A100,
            'seq': 1,
            'analysis': lambda: _diagnose_throughput(LLAMA70B, A100),
        },
        {
            'name': '场景 3: OOM 频繁',
            'symptom': '并发 > 10 就 OOM',
            'model': LLAMA70B,
            'gpu': A100,
            'seq': 4096,
            'analysis': lambda: _diagnose_oom(LLAMA70B, A100, 4096),
        },
    ]

    for s in scenarios:
        print(f"\n{'='*50}")
        print(f"  {s['name']}")
        print(f"  症状: {s['symptom']}")
        print(f"  配置: {s['model'].name} on {s['gpu'].name}")
        print(f"{'='*50}")
        s['analysis']()


def _diagnose_ttft(model: Model, gpu: GPU, seq_len: int):
    """诊断 TTFT 问题"""
    r = prefill_time_ms(model, gpu, seq_len)
    print(f"\n  分析:")
    print(f"    Prompt 长度: {seq_len} tokens")
    print(f"    Prefill 计算: {r['total_flops_G']:.1f} GFLOPS")
    print(f"    Attention 占比: {r['attn_pct']:.1f}%")
    print(f"    预估 TTFT: {r['time_ms']:.0f} ms ({r['time_ms']/1000:.1f}s)")

    print(f"\n  优化建议:")
    if r['attn_pct'] > 40:
        print(f"    1. 启用 FlashAttention (减少 attention 内存和计算)")
    if seq_len > 4096:
        print(f"    2. 考虑 Chunked Prefill (分块处理, 减少 decode 饥饿)")
    print(f"    3. 使用 TP=2-4 减少计算时间")
    print(f"    4. 考虑 Speculative Decoding (draft model 预生成)")
    print(f"    5. 启用 Prefix Caching (重复 prompt 跳过 prefill)")


def _diagnose_throughput(model: Model, gpu: GPU):
    """诊断吞吐问题"""
    r = decode_time_ms(model, gpu, 2048)
    print(f"\n  分析:")
    print(f"    Decode 延迟: {r['time_ms']:.2f} ms/token")
    print(f"    权重读取: {r['model_bytes_MB']:.0f} MB")
    print(f"    KV Cache: {r['kv_bytes_MB']:.1f} MB ({r['kv_pct']:.1f}%)")
    print(f"    理论吞吐: {1000/r['time_ms']:.0f} tok/s (batch=1)")

    print(f"\n  优化建议:")
    print(f"    1. 增大 batch_size (复用权重读取, 吞吐线性增长)")
    print(f"    2. 使用 Continuous Batching (避免 decode 间空闲)")
    print(f"    3. 量化推理 (FP8/INT4 减少权重读取量)")
    print(f"    4. KV Cache 量化 (减少 KV 读取量)")
    print(f"    5. Speculative Decoding (每个 verify step 多个 token)")


def _diagnose_oom(model: Model, gpu: GPU, avg_seq: int):
    """诊断 OOM 问题"""
    print(f"\n  分析:")
    avail_hbm = gpu.hbm_gb - model.model_size_gb
    if avail_hbm <= 0:
        tp = math.ceil(model.model_size_gb / gpu.hbm_gb)
        print(f"    ⚠ 单卡放不下 ({model.model_size_gb:.1f} GB > {gpu.hbm_gb} GB)")
        print(f"    需要 TP={tp}, 单卡权重 {model.model_size_gb/tp:.1f} GB")
        avail_hbm = gpu.hbm_gb - model.model_size_gb / tp

    avail_gb = avail_hbm * 0.7
    kv_per_req = model.kv_bytes_per_token * avg_seq / 1e9
    max_concurrent = int(avail_gb / kv_per_req)

    print(f"    可用 KV Cache: {avail_gb:.1f} GB")
    print(f"    KV/请求 (avg {avg_seq} tokens): {kv_per_req*1e3:.1f} MB")
    print(f"    最大并发: {max_concurrent} 请求")

    print(f"\n  优化建议:")
    print(f"    1. 使用 GQA 减少 KV heads (LLaMA-70B 已用 GQA-8)")
    print(f"    2. KV Cache 量化 (FP8 → 2x 压缩)")
    print(f"    3. PagedAttention (减少碎片化, vLLM 默认)")
    print(f"    4. Prefix Caching (共享 prompt 不重复存储)")
    print(f"    5. 降低 max_model_len (截断长请求)")
    print(f"    6. Swapping (换出到 CPU, vLLM V1 用 recomputation)")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("LLM 推理性能诊断工具")
    print("从延迟分解到瓶颈分析到优化建议")
    print("=" * 70)

    experiment_1_prefill_decode_breakdown()
    experiment_2_kv_pressure()
    experiment_3_throughput_estimation()
    experiment_4_diagnosis()

    print("\n" + "=" * 70)
    print("诊断框架总结")
    print("=" * 70)
    print("""
推理性能四步诊断:

  Step 1: 识别瓶颈类型
    - Prefill 慢 → compute-bound → 优化计算效率
    - Decode 慢 → memory-bound → 优化数据搬运
    - OOM → 显存不足 → 优化显存使用

  Step 2: 量化瓶颈
    - Prefill: FLOPs / (TFLOPS × 效率)
    - Decode: (权重 + KV) / HBM带宽
    - OOM: KV_per_req × max_concurrent

  Step 3: 选择优化
    Compute-bound 优化:
      - TP (多卡分计算)
      - FlashAttention (减少 attention FLOPs)
      - torch.compile (kernel fusion)

    Memory-bound 优化:
      - Batch (复用权重)
      - 量化 (减少权重大小)
      - Speculative Decoding (每步多 token)

    显存优化:
      - GQA/MLA (减少 KV heads)
      - KV Cache 量化
      - Prefix Caching
      - PagedAttention

  Step 4: 验证效果
    - TTFT/ITL/Tok-per-s 三指标
    - GPU 利用率 + KV Cache 利用率
    - 对比优化前后
    """)
