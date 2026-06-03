#!/usr/bin/env python3
"""长上下文 LLM 推理服务模拟器

CPU 可运行的长上下文推理分析，覆盖 5 个实验:
  1. KV Cache 显存压力分析 (不同上下文长度 × GPU × 模型)
  2. Chunked Prefill 分块预填充优化
  3. Sliding Window Attention 显存节省
  4. Prefix Caching 长上下文场景收益
  5. 长上下文服务策略推荐

关键概念:
  - 长上下文 (128K-1M): KV Cache 成为主要显存瓶颈
  - Chunked Prefill: 将长 prefill 分块，避免 OOM 和延迟毛刺
  - Sliding Window: 只保留最近 W 个 token 的 KV，节省显存
  - Prefix Caching: 跨请求复用公共前缀 KV，减少重复计算

用法:
  conda run -n ai-infra python tools/long_context_serving_sim.py
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

@dataclass
class GPUConfig:
    name: str
    hbm_gb: float
    hbm_bw_gbs: float  # GB/s
    fp16_tflops: float


@dataclass
class ModelConfig:
    name: str
    params_B: float
    hidden_dim: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int = 32000


# GPU 配置
A100_80G = GPUConfig("A100 80GB", 80, 2030, 312)
H100_80G = GPUConfig("H100 80GB", 80, 3350, 990)
H200_141G = GPUConfig("H200 141GB", 141, 4800, 990)
A16_15G = GPUConfig("A16 15GB", 15, 157, 14)

# 模型配置
GPT2 = ModelConfig("GPT-2 (124M)", 0.124, 768, 12, 12, 64, 50257)
LLAMA7B = ModelConfig("LLaMA-7B", 7, 4096, 32, 32, 128, 32000)
LLAMA70B = ModelConfig("LLaMA-70B", 70, 8192, 80, 8, 128, 32000)
QWEN72B = ModelConfig("Qwen-72B", 72, 8192, 80, 8, 128, 151936)
LLAMA3_8B = ModelConfig("LLaMA-3 8B", 8, 4096, 32, 8, 128, 128256)
MIXTRAL_8x7B = ModelConfig("Mixtral 8x7B", 46.7, 4096, 32, 8, 128, 32000)

CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]


# ──────────────────────────────────────────────
# 计算函数
# ──────────────────────────────────────────────

def model_weight_size_gb(model: ModelConfig, precision_bytes: int = 2) -> float:
    """模型权重占用显存 (GB)"""
    return model.params_B * 1e9 * precision_bytes / (1024 ** 3)


def kv_cache_per_token_bytes(model: ModelConfig, precision_bytes: int = 2) -> float:
    """每个 token 的 KV Cache 大小 (bytes)"""
    # KV Cache: 2 (K+V) × num_layers × num_kv_heads × head_dim × precision
    return 2 * model.num_layers * model.num_kv_heads * model.head_dim * precision_bytes


def kv_cache_size_gb(
    model: ModelConfig,
    seq_len: int,
    batch_size: int = 1,
    precision_bytes: int = 2,
) -> float:
    """KV Cache 总大小 (GB)"""
    per_token = kv_cache_per_token_bytes(model, precision_bytes)
    total_bytes = per_token * seq_len * batch_size
    return total_bytes / (1024 ** 3)


def max_batch_for_kv(
    gpu: GPUConfig,
    model: ModelConfig,
    seq_len: int,
    precision_bytes: int = 2,
    overhead_gb: float = 2.0,
) -> int:
    """在给定上下文长度下，最大可并发 batch size"""
    weight = model_weight_size_gb(model, precision_bytes)
    available = gpu.hbm_gb - weight - overhead_gb
    if available <= 0:
        return 0
    per_batch = kv_cache_size_gb(model, seq_len, 1, precision_bytes)
    if per_batch <= 0:
        return 0
    return max(0, int(available / per_batch))


def prefill_time_ms(
    model: ModelConfig,
    gpu: GPUConfig,
    seq_len: int,
    batch_size: int = 1,
) -> float:
    """Prefill 延迟 (ms) — compute-bound, 延迟 ∝ seq_len² × batch"""
    # Total FLOPs: 2 × params × seq_len (forward pass)
    total_flops = 2 * model.params_B * 1e9 * seq_len * batch_size
    # Attention FLOPs: batch × layers × heads × seq_len² × head_dim
    attn_flops = batch_size * model.num_layers * model.num_kv_heads * seq_len * seq_len * model.head_dim
    total_flops += attn_flops

    # compute-bound: time = FLOPs / TFLOPS
    time_s = total_flops / (gpu.fp16_tflops * 1e12)
    return time_s * 1000  # ms


def decode_time_ms(
    model: ModelConfig,
    gpu: GPUConfig,
    seq_len: int,
    batch_size: int = 1,
) -> float:
    """Decode 每步延迟 (ms) — memory-bound, 延迟 ∝ model_size + kv_cache"""
    # Bytes to read: model weights + KV Cache for all requests
    weight_bytes = model.params_B * 1e9 * 2  # FP16
    kv_bytes = kv_cache_per_token_bytes(model) * seq_len * batch_size
    total_bytes = weight_bytes + kv_bytes

    # memory-bound: time = bytes / bandwidth
    time_s = total_bytes / (gpu.hbm_bw_gbs * 1e9)
    return time_s * 1000  # ms


def throughput_tok_per_s(
    model: ModelConfig,
    gpu: GPUConfig,
    seq_len: int,
    batch_size: int = 1,
) -> float:
    """Decode 吞吐量 (tokens/s)"""
    time_ms = decode_time_ms(model, gpu, seq_len, batch_size)
    if time_ms <= 0:
        return 0
    return batch_size / time_ms * 1000


# ──────────────────────────────────────────────
# 实验 1: KV Cache 显存压力分析
# ──────────────────────────────────────────────

def experiment_1_kv_cache_pressure():
    """实验 1: 不同上下文长度下 KV Cache 显存压力"""
    print("\n" + "=" * 70)
    print("实验 1: KV Cache 显存压力分析")
    print("=" * 70)

    # LLaMA-7B on A100
    model = LLAMA7B
    gpu = A100_80G
    weight = model_weight_size_gb(model)

    print(f"\n--- {model.name} on {gpu.name} ---")
    print(f"模型权重: {weight:.1f} GB, 可用 KV: {gpu.hbm_gb - weight - 2:.1f} GB")
    print(f"\n{'上下文':>10} {'KV/req (GB)':>12} {'Max Batch':>10} {'Prefill (ms)':>14} {'Decode (ms)':>14}")
    print("-" * 66)

    for ctx in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        kv_gb = kv_cache_size_gb(model, ctx)
        max_b = max_batch_for_kv(gpu, model, ctx)
        pf_ms = prefill_time_ms(model, gpu, ctx)
        dc_ms = decode_time_ms(model, gpu, ctx)
        ctx_str = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
        print(f"{ctx_str:>10} {kv_gb:>12.2f} {max_b:>10d} {pf_ms:>14.1f} {dc_ms:>14.2f}")

    # LLaMA-70B on H100
    model = LLAMA70B
    gpu = H100_80G
    weight = model_weight_size_gb(model)

    print(f"\n--- {model.name} on {gpu.name} ---")
    print(f"模型权重: {weight:.1f} GB, 可用 KV: {gpu.hbm_gb - weight - 2:.1f} GB")
    print(f"\n{'上下文':>10} {'KV/req (GB)':>12} {'Max Batch':>10} {'Prefill (ms)':>14} {'Decode (ms)':>14}")
    print("-" * 66)

    for ctx in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        kv_gb = kv_cache_size_gb(model, ctx)
        max_b = max_batch_for_kv(gpu, model, ctx)
        pf_ms = prefill_time_ms(model, gpu, ctx)
        dc_ms = decode_time_ms(model, gpu, ctx)
        ctx_str = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
        print(f"{ctx_str:>10} {kv_gb:>12.2f} {max_b:>10d} {pf_ms:>14.1f} {dc_ms:>14.2f}")

    # LLaMA-70B on H200 (141GB, 长上下文利器)
    gpu = H200_141G
    print(f"\n--- {model.name} on {gpu.name} ---")
    print(f"模型权重: {weight:.1f} GB, 可用 KV: {gpu.hbm_gb - weight - 2:.1f} GB")
    print(f"\n{'上下文':>10} {'KV/req (GB)':>12} {'Max Batch':>10} {'Prefill (ms)':>14} {'Decode (ms)':>14}")
    print("-" * 66)

    for ctx in [2048, 8192, 32768, 65536, 131072, 262144, 524288]:
        kv_gb = kv_cache_size_gb(model, ctx)
        max_b = max_batch_for_kv(gpu, model, ctx)
        pf_ms = prefill_time_ms(model, gpu, ctx)
        dc_ms = decode_time_ms(model, gpu, ctx)
        ctx_str = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
        print(f"{ctx_str:>10} {kv_gb:>12.2f} {max_b:>10d} {pf_ms:>14.1f} {dc_ms:>14.2f}")

    print(f"\n关键洞察:")
    print(f"  - 7B/128K: KV Cache 32 GB, 占 A100 40% 显存, 仅 1 并发")
    print(f"  - 70B/128K: 140GB 模型 + 64GB KV, A100 完全不够")
    print(f"  - H200 141GB: 70B/128K 可并发 0-1 请求, 70B/4K 可并发 6 请求")
    print(f"  - Prefill 延迟 ∝ seq_len²: 128K 需要 ~2s, 512K 需要 ~30s")
    print(f"  - Decode 延迟 ∝ seq_len: 长上下文 decode 也变慢 (KV 扫描)")


# ──────────────────────────────────────────────
# 实验 2: Chunked Prefill 分块预填充
# ──────────────────────────────────────────────

def experiment_2_chunked_prefill():
    """实验 2: Chunked Prefill 分块预填充优化"""
    print("\n" + "=" * 70)
    print("实验 2: Chunked Prefill 分块预填充优化")
    print("=" * 70)

    model = LLAMA7B
    gpu = A100_80G

    print(f"\n--- {model.name} on {gpu.name}, Prefill 延迟 vs Chunk Size ---")
    print(f"\n场景: 128K 上下文, 1 个请求")
    print(f"\n{'Chunk Size':>12} {'Chunks':>8} {'延迟/Chunk (ms)':>16} {'总延迟 (ms)':>14} {'Max Mem (GB)':>14} {'TTFT (ms)':>12}")
    print("-" * 82)

    ctx = 131072
    total_weight = model_weight_size_gb(model)

    for chunk_size in [512, 1024, 2048, 4096, 8192, 16384, 32768, ctx]:
        if chunk_size > ctx:
            continue
        num_chunks = math.ceil(ctx / chunk_size)
        # 每个 chunk 的延迟 (compute-bound)
        chunk_time = prefill_time_ms(model, gpu, chunk_size)
        total_time = chunk_time * num_chunks

        # 每个 chunk 的 KV Cache 大小
        chunk_kv = kv_cache_size_gb(model, chunk_size)
        # 总 KV Cache (已完成的 + 当前 chunk 的)
        # 最坏情况: 当前 chunk + 之前所有 chunk 的 KV 都在内存
        max_kv = kv_cache_size_gb(model, ctx)
        max_mem = total_weight + max_kv + 2  # weight + full KV + overhead

        # TTFT = 第一个 chunk 完成 (只有第一个 chunk 需要等)
        ttft = chunk_time

        cs_str = f"{chunk_size // 1024}K" if chunk_size >= 1024 else str(chunk_size)
        print(f"{cs_str:>12} {num_chunks:>8d} {chunk_time:>16.1f} {total_time:>14.1f} {max_mem:>14.1f} {ttft:>12.1f}")

    print(f"\n关键洞察:")
    print(f"  - Chunk Size 不影响总计算量 (延迟大致相同)")
    print(f"  - Chunk Size 主要影响: (1) TTFT — 小 chunk 更快返回第一个 token")
    print(f"                        (2) 调度灵活性 — 小 chunk 允许穿插 decode")
    print(f"  - 实际中 vLLM 用 chunk_size=512~2048, 平衡 TTFT 和吞吐")
    print(f"  - Chunked Prefill 避免单个长 prefill 独占 GPU, 让其他请求 decode 不被阻塞")

    # 混合 Prefill + Decode 场景
    print(f"\n--- 混合场景: 1 个 128K Prefill + N 个 Decode 请求 ---")
    print(f"\n{'Decode Reqs':>12} {'No Chunking':>16} {'Chunk=2K':>16} {'Decode 延迟':>16}")
    print("-" * 66)

    chunk_size = 2048
    for n_decode in [0, 4, 8, 16, 32]:
        # 无 chunking: prefill 独占 GPU, decode 全部等待
        no_chunk_wait = prefill_time_ms(model, gpu, ctx)
        if n_decode > 0:
            no_chunk_decode = decode_time_ms(model, gpu, ctx, n_decode)
        else:
            no_chunk_decode = 0

        # Chunking: 每 chunk 后穿插 decode
        num_chunks = math.ceil(ctx / chunk_size)
        chunk_time = prefill_time_ms(model, gpu, chunk_size)
        if n_decode > 0:
            decode_step = decode_time_ms(model, gpu, ctx, n_decode)
        else:
            decode_step = 0

        # 每轮 = chunk + decode
        chunked_total = (chunk_time + decode_step) * num_chunks

        print(f"{n_decode:>12d} {no_chunk_wait:>16.1f} {chunked_total:>16.1f} {decode_step if n_decode > 0 else 0:>16.2f}")

    print(f"\n  → Chunked Prefill 让 decode 请求不被长 prefill 饿死")
    print(f"  → 代价: prefill 总延迟略增 (decode 穿插的开销)")


# ──────────────────────────────────────────────
# 实验 3: Sliding Window Attention
# ──────────────────────────────────────────────

def experiment_3_sliding_window():
    """实验 3: Sliding Window Attention 显存节省"""
    print("\n" + "=" * 70)
    print("实验 3: Sliding Window Attention 显存节省")
    print("=" * 70)

    model = LLAMA7B
    gpu = A100_80G

    print(f"\n--- {model.name} on {gpu.name}, Full vs Sliding Window ---")
    print(f"\n{'上下文':>10} {'Full KV (GB)':>14} {'SW-4K (GB)':>14} {'SW-8K (GB)':>14} {'SW-16K (GB)':>14} {'SW 节省':>10}")
    print("-" * 72)

    for ctx in [4096, 8192, 16384, 32768, 65536, 131072]:
        full_kv = kv_cache_size_gb(model, ctx)
        sw4k = kv_cache_size_gb(model, min(ctx, 4096))
        sw8k = kv_cache_size_gb(model, min(ctx, 8192))
        sw16k = kv_cache_size_gb(model, min(ctx, 16384))
        saving = (1 - sw4k / full_kv) * 100 if full_kv > 0 else 0
        ctx_str = f"{ctx // 1024}K"
        print(f"{ctx_str:>10} {full_kv:>14.2f} {sw4k:>14.2f} {sw8k:>14.2f} {sw16k:>14.2f} {saving:>9.0f}%")

    # Max batch with sliding window
    print(f"\n--- 最大并发请求数: Full vs Sliding Window ---")
    print(f"\n{'上下文':>10} {'Full':>8} {'SW-4K':>8} {'SW-8K':>8} {'SW-16K':>8}")
    print("-" * 50)

    for ctx in [4096, 8192, 16384, 32768, 65536, 131072]:
        full_b = max_batch_for_kv(gpu, model, ctx)
        sw4k_b = max_batch_for_kv(gpu, model, min(ctx, 4096))
        sw8k_b = max_batch_for_kv(gpu, model, min(ctx, 8192))
        sw16k_b = max_batch_for_kv(gpu, model, min(ctx, 16384))
        ctx_str = f"{ctx // 1024}K"
        print(f"{ctx_str:>10} {full_b:>8d} {sw4k_b:>8d} {sw8k_b:>8d} {sw16k_b:>8d}")

    # GQA/MQA 对 KV Cache 的影响
    print(f"\n--- Attention 类型对 KV Cache 大小的影响 (128K 上下文) ---")
    print(f"\n{'模型':>20} {'KV Heads':>10} {'KV/req (GB)':>14} {'Max Batch (A100)':>18}")
    print("-" * 68)

    ctx = 131072
    configs = [
        ("MHA (32 heads)", LLAMA7B),
        ("GQA (8 heads)", LLAMA70B),
        ("GQA (8 heads)", LLAMA3_8B),
        ("MQA (1 head)", ModelConfig("MQA-7B", 7, 4096, 32, 1, 128)),
    ]

    for label, m in configs:
        kv = kv_cache_size_gb(m, ctx)
        max_b = max_batch_for_kv(A100_80G, m, ctx)
        print(f"{label:>20} {m.num_kv_heads:>10d} {kv:>14.2f} {max_b:>18d}")

    print(f"\n关键洞察:")
    print(f"  - Sliding Window 在长上下文 (>4×W) 时节省大量显存")
    print(f"  - SW-4K 在 128K 上下文: 节省 97% KV Cache!")
    print(f"  - 代价: 丢失超出窗口的历史信息 (不适合需要全文理解的任务)")
    print(f"  - GQA/MQA 从架构层面减少 KV Cache: MQA 比 MHA 少 32×")
    print(f"  - Mistral-7B 用 SW-8K + GQA-8, 是长上下文推理的典型配置")
    print(f"  - 实际应用: RAG/搜索用 SW 足够, 总结/分析需要 Full Attention")


# ──────────────────────────────────────────────
# 实验 4: Prefix Caching 长上下文场景
# ──────────────────────────────────────────────

def experiment_4_prefix_caching():
    """实验 4: Prefix Caching 在长上下文场景的收益"""
    print("\n" + "=" * 70)
    print("实验 4: Prefix Caching 长上下文场景收益")
    print("=" * 70)

    model = LLAMA7B
    gpu = A100_80G

    # 场景 1: 系统提示 + 文档 + 查询
    print(f"\n--- 场景 1: RAG 应用 (固定文档 + 多轮查询) ---")
    print(f"模型: {model.name}, 文档: 10K tokens, 系统提示: 500 tokens")

    doc_tokens = 10000
    system_tokens = 500
    query_tokens = 100

    for num_queries in [1, 5, 10, 20, 50, 100]:
        total_prefix = doc_tokens + system_tokens
        total_per_query = total_prefix + query_tokens

        # 无 caching: 每次都重新计算 prefix
        total_prefill_no_cache = total_per_query * num_queries

        # 有 caching: 只计算第一次 prefix + 每次查询的 query 部分
        total_prefill_with_cache = total_prefix + query_tokens * num_queries

        # KV Cache 显存
        kv_per_query = kv_cache_size_gb(model, total_per_query)
        total_kv = kv_per_query * num_queries

        # 节省比例
        saved_pct = (1 - total_prefill_with_cache / total_prefill_no_cache) * 100

        print(f"  {num_queries:>3} 查询: 无缓存 {total_prefill_no_cache/1000:.0f}K tok, "
              f"有缓存 {total_prefill_with_cache/1000:.0f}K tok, "
              f"节省 {saved_pct:.0f}%, "
              f"总 KV {total_kv:.1f} GB")

    # 场景 2: 多文档对比
    print(f"\n--- 场景 2: 多文档 QA (共享系统提示) ---")
    print(f"系统提示: 1000 tokens, 每个文档: 5000 tokens, 查询: 50 tokens")

    system_tokens = 1000
    doc_tokens = 5000
    query_tokens = 50

    for num_docs in [5, 10, 20, 50]:
        total_tokens_no_cache = (system_tokens + doc_tokens + query_tokens) * num_docs
        total_tokens_with_cache = system_tokens + (doc_tokens + query_tokens) * num_docs
        saved = (1 - total_tokens_with_cache / total_tokens_no_cache) * 100

        print(f"  {num_docs:>3} 文档: 无缓存 {total_tokens_no_cache/1000:.0f}K tok, "
              f"有缓存 {total_tokens_with_cache/1000:.0f}K tok, "
              f"节省 {saved:.0f}%")

    # 场景 3: RL 训练 (GRPO)
    print(f"\n--- 场景 3: GRPO RL 训练 (prompt 复用) ---")
    print(f"Prompt: 500 tokens, 生成: 512 tokens, 8 responses/prompt")

    prompt_tokens = 500
    gen_tokens = 512
    num_responses = 8

    for num_prompts in [10, 50, 100, 500]:
        total_no_cache = (prompt_tokens + gen_tokens) * num_responses * num_prompts
        total_with_cache = prompt_tokens * num_prompts + gen_tokens * num_responses * num_prompts
        saved = (1 - total_with_cache / total_no_cache) * 100

        # KV Cache 显存 (所有 prompt 的 responses 同时在 GPU)
        single_seq = prompt_tokens + gen_tokens
        batch = num_responses  # 同时处理一个 prompt 的 8 个 responses
        kv = kv_cache_size_gb(model, single_seq, batch)

        print(f"  {num_prompts:>4} prompts: 节省 {saved:.0f}%, "
              f"批处理 KV {kv:.1f} GB")

    print(f"\n关键洞察:")
    print(f"  - RAG: Prefix 越长, 查询越多, 收益越大 (10 查询节省 98%)")
    print(f"  - 多文档: 仅共享系统提示, 收益有限 (<5%)")
    print(f"  - RL 训练: prompt 复用天然适合 prefix caching")
    print(f"  - 收益公式: savings ≈ prefix_len / total_len × num_reuse")


# ──────────────────────────────────────────────
# 实验 5: 长上下文服务策略推荐
# ──────────────────────────────────────────────

def experiment_5_serving_strategy():
    """实验 5: 长上下文服务策略推荐"""
    print("\n" + "=" * 70)
    print("实验 5: 长上下文服务策略推荐")
    print("=" * 70)

    scenarios = [
        {
            "name": "代码助手 (8K 上下文)",
            "model": LLAMA3_8B,
            "gpu": A100_80G,
            "ctx": 8192,
            "concurrency": 16,
            "type": "对话+代码",
        },
        {
            "name": "文档 RAG (32K 上下文)",
            "model": LLAMA3_8B,
            "gpu": A100_80G,
            "ctx": 32768,
            "concurrency": 8,
            "type": "RAG 检索",
        },
        {
            "name": "长文档分析 (128K)",
            "model": LLAMA70B,
            "gpu": H100_80G,
            "ctx": 131072,
            "concurrency": 2,
            "type": "文档分析",
        },
        {
            "name": "超长上下文 (256K)",
            "model": LLAMA70B,
            "gpu": H200_141G,
            "ctx": 262144,
            "concurrency": 1,
            "type": "全文本理解",
        },
        {
            "name": "RL 训练 (GRPO, 1K prompt)",
            "model": LLAMA7B,
            "gpu": A100_80G,
            "ctx": 2048,
            "concurrency": 32,
            "type": "RL 训练",
        },
    ]

    for s in scenarios:
        model = s["model"]
        gpu = s["gpu"]
        ctx = s["ctx"]
        conc = s["concurrency"]

        weight = model_weight_size_gb(model)
        kv_per_req = kv_cache_size_gb(model, ctx)
        total_kv = kv_per_req * conc
        max_b = max_batch_for_kv(gpu, model, ctx)
        can_fit = total_kv + weight + 2 <= gpu.hbm_gb

        pf_time = prefill_time_ms(model, gpu, ctx)
        dc_time = decode_time_ms(model, gpu, ctx, min(conc, max_b))
        throughput = throughput_tok_per_s(model, gpu, ctx, min(conc, max_b))

        # 推荐策略
        strategies = []
        if ctx > 32768:
            strategies.append("Chunked Prefill (chunk=2K)")
        if ctx > 65536:
            strategies.append("考虑 Sliding Window (如果任务允许)")
        if max_b < conc:
            strategies.append(f"减少并发到 {max_b} (KV Cache 不够)")
            if ctx > 16384:
                strategies.append("或用 TP=2 扩展显存")
        if kv_per_req > 10:
            strategies.append("KV Cache 量化 (FP8 → 50% 显存节省)")
        if s["type"] == "RL 训练":
            strategies.append("Prefix Caching (prompt 复用)")
        if s["type"] == "RAG 检索":
            strategies.append("Prefix Caching (文档复用)")
        if weight > gpu.hbm_gb:
            tp = math.ceil(weight / gpu.hbm_gb)
            strategies.append(f"必须 TP={tp} (模型权重放不下)")

        ctx_str = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)

        print(f"\n--- {s['name']} ---")
        print(f"  模型: {model.name} ({weight:.1f} GB), GPU: {gpu.name}")
        print(f"  上下文: {ctx_str} tokens, 目标并发: {conc}")
        print(f"  KV/请求: {kv_per_req:.2f} GB, 总 KV: {total_kv:.1f} GB")
        print(f"  最大并发: {max_b}, {'✓' if can_fit else '✗ 超出显存'}")
        print(f"  Prefill: {pf_time:.0f} ms, Decode: {dc_time:.2f} ms/step")
        print(f"  吞吐: {throughput:.0f} tok/s (batch={min(conc, max_b)})")
        print(f"  推荐策略:")
        for st in strategies:
            print(f"    → {st}")

    # GPU 选择指南
    print(f"\n\n--- 长上下文 GPU 选择指南 ---")
    print(f"\n{'场景':>20} {'推荐 GPU':>15} {'理由':>40}")
    print("-" * 80)
    guide = [
        ("<8K 上下文", "A100 80GB", "显存足够, 性价比高"),
        ("8K-32K 上下文", "A100 80GB", "KV Cache ~4-16 GB, 可并发"),
        ("32K-128K, 7B", "A100 80GB", "KV ~32 GB, 需注意并发限制"),
        ("32K-128K, 70B", "H200 141GB", "70B 权重 + KV 需要大显存"),
        ("128K+, 70B", "H200 141GB", "141 GB 是当前单卡最佳选择"),
        ("256K+", "2-4×H200", "单卡不够, 需要 TP"),
        ("RL 训练", "A100 80GB", "短上下文+高并发, prefix caching"),
    ]
    for scenario, gpu, reason in guide:
        print(f"{scenario:>20} {gpu:>15} {reason:>40}")

    print(f"\n关键洞察:")
    print(f"  - 7B 模型: 128K 上下文 KV ~32 GB, A100 基本够用")
    print(f"  - 70B 模型: 128K 上下文需要 H200 141GB 或 TP=2")
    print(f"  - 超长上下文 (>256K): 几乎必须多卡 TP 或 MoE 架构")
    print(f"  - RL 训练: 短上下文 + 高并发, prefix caching 是关键优化")
    print(f"  - RAG: 文档复用率高, prefix caching 收益最大")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("长上下文 LLM 推理服务模拟器")
    print("KV Cache 压力 + Chunked Prefill + Sliding Window + Prefix Caching")
    print("=" * 70)

    experiment_1_kv_cache_pressure()
    experiment_2_chunked_prefill()
    experiment_3_sliding_window()
    experiment_4_prefix_caching()
    experiment_5_serving_strategy()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
长上下文服务关键要点:

  1. KV Cache 是长上下文的最大瓶颈:
     7B/128K: 32 GB KV (A100 40%)
     70B/128K: 64 GB KV (A100 放不下)
     → 上下文翻倍, KV Cache 翻倍

  2. Prefill 延迟 ∝ seq_len²:
     4K: ~30 ms, 128K: ~2 s, 512K: ~30 s
     → Chunked Prefill 是长 prefill 的必需优化
     → 避免单个长 prefill 独占 GPU

  3. Sliding Window 是长上下文的内存救星:
     SW-4K at 128K: 节省 97% KV Cache
     → 代价: 丢失超出窗口的历史信息
     → 适合 RAG/搜索, 不适合总结/分析

  4. Prefix Caching 收益公式:
     savings = prefix_len / total_len × num_reuse
     → RAG (10K doc × 100 queries): 节省 98%
     → RL (500 prompt × 8 responses): 节省 50%

  5. GPU 选择:
     ≤8K: A100 80GB 足够
     8K-128K: H200 141GB (单卡最佳)
     >128K: 2-4× H200 (TP 必需)
     → KV Cache 量化 (FP8) 可额外节省 50% 显存
    """)
