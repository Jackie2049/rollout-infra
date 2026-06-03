# 长上下文 LLM 推理服务

> 从 4K 到 1M：长上下文推理的挑战、优化策略和服务架构

## 1. 核心挑战

### 1.1 KV Cache 显存爆炸

长上下文推理的根本挑战：**KV Cache 线性增长，很快超过模型权重**。

```
KV Cache per token = 2 × num_layers × num_kv_heads × head_dim × precision_bytes

LLaMA-7B (FP16): 2 × 32 × 32 × 128 × 2 = 512 KB/token
LLaMA-70B GQA (FP16): 2 × 80 × 8 × 128 × 2 = 320 KB/token

上下文长度 vs KV Cache (LLaMA-7B):
  4K → 2 GB     (权重 13 GB 的 15%)
  32K → 16 GB   (权重 13 GB 的 123%! 超过权重)
  128K → 64 GB  (权重 13 GB 的 492%!)
```

**关键数据** (模拟器验证):
| 模型 | 上下文 | KV/请求 | A100 80GB 并发 |
|------|--------|---------|----------------|
| 7B MHA | 128K | 64 GB | 1 |
| 7B GQA | 128K | 16 GB | 3 |
| 70B GQA | 128K | 40 GB | 0 (需 TP) |
| 70B GQA | 32K | 10 GB | 0 (需 TP) |

### 1.2 Prefill 延迟二次增长

Prefill 的 attention 计算量 ∝ seq_len²：

```
FLOPs ≈ 2 × params × seq_len + layers × heads × seq_len² × head_dim
                                        ^^^^^^^^
                                        主导项

实测估算 (LLaMA-7B/A100):
  4K: ~30 ms
  32K: ~1.9 s
  128K: ~13 s
  512K: ~97 s (!!!)
```

### 1.3 Decode 延迟增长

Decode 虽然 memory-bound，但需要扫描全部 KV Cache：

```
Decode 延迟 ∝ (model_weights + kv_cache) / hbm_bandwidth

LLaMA-7B/A100:
  4K: 7.4 ms/step
  128K: 40.8 ms/step (5.5× 慢)
  → 上下文每翻倍, decode 延迟增加 ~15%
```

## 2. 优化策略

### 2.1 Chunked Prefill

**问题**: 长 prefill 独占 GPU，其他请求 decode 被饿死。

**方案**: 将长 prefill 分成小 chunk，每个 chunk 后穿插 decode。

```
无 Chunked Prefill:
  [Prefill 128K ——— 13s ———]  [Decode batch]

有 Chunked Prefill (chunk=2K):
  [PF2K][Decode][PF2K][Decode]...×64
  TTFT: 94 ms (vs 13 s)

关键参数:
  chunk_size: 512~2048 (vLLM 默认)
  trade-off: 小 chunk → 低 TTFT, 但总 prefill 时间略增
```

**收益**:
- TTFT 降低 100×+ (128K: 13s → 94ms)
- 其他请求 decode 不被饿死
- GPU 利用率更均匀

### 2.2 Sliding Window Attention

**原理**: 只保留最近 W 个 token 的 KV Cache，丢弃更早的。

```
Full Attention:   KV Cache = O(N)
Sliding Window:   KV Cache = O(min(N, W))

SW-4K at 128K: KV = 2 GB (vs 64 GB, 节省 97%)
SW-4K at 128K: 并发 = 32 (vs 1, 32× 提升)
```

**适用场景**:
- RAG/搜索: 只需要相关段落附近的上下文 ✓
- 对话: 最近几轮对话最重要 ✓
- 长文档总结: 需要全文理解 ✗
- 代码生成: 需要文件级上下文 ✗

**实际应用**: Mistral-7B 用 SW-8K + GQA-8，是长上下文推理的典型配置。

### 2.3 GQA / MQA

从架构层面减少 KV Cache 大小：

```
MHA: KV Heads = num_attention_heads (32)
GQA: KV Heads = num_kv_heads < num_attention_heads (8)
MQA: KV Heads = 1

KV Cache 大小 (128K):
  MHA 32 heads: 64 GB
  GQA 8 heads:  16 GB (4× 减少)
  MQA 1 head:    2 GB (32× 减少)
```

LLaMA-2/3, Mistral, Qwen 等现代模型都用 GQA。

### 2.4 Prefix Caching

**核心公式**: `savings = prefix_len / total_len × num_reuse`

**典型场景收益**:
- RAG (10K doc × 100 queries): 98% 节省
- RL GRPO (500 prompt × 8 responses): 43% 节省
- 多文档 QA (共享 1K 系统提示): 16% 节省

详见 [Prefix Caching 深度对比](prefix-caching.md)。

### 2.5 KV Cache 量化

将 KV Cache 从 FP16 (2 bytes/element) 量化到 FP8 (1 byte/element)：

```
显存节省: 50%

7B/128K: 64 GB → 32 GB (A100 可并发 2)
70B/128K: 40 GB → 20 GB (配合 TP=2 可并发 1)

精度影响: 通常 <0.5% perplexity 增加
```

vLLM 支持 `--kv-cache-dtype fp8` 启用 KV Cache 量化。

### 2.6 PagedAttention

vLLM 的核心创新——将 KV Cache 分成固定大小的 block，按需分配：

```
传统: 预分配最大上下文 × batch_size 的连续内存
  → 浪费 60-80% (请求上下文长度差异大)

PagedAttention: 按 block 分配，类似虚拟内存分页
  → 浪费 <4%
  → 块大小通常 16 tokens
```

详见 [KV Cache 深度解析](kv-cache.md), [GPU 内存分配器](gpu-memory-allocator.md)。

## 3. 服务架构

### 3.1 GPU 选择

| 场景 | 推荐 GPU | 理由 |
|------|----------|------|
| ≤8K 上下文 | A100 80GB | 显存足够, 性价比高 |
| 8K-32K | A100 80GB | KV ~4-16 GB, 可并发 |
| 32K-128K, 7B | A100 80GB | KV ~32 GB, 并发受限 |
| 32K-128K, 70B | H200 141GB | 权重 + KV 需要大显存 |
| 128K+, 70B | 2-4× H200 | 单卡不够, TP 必需 |
| RL 训练 | A100 80GB | 短上下文 + 高并发 |

### 3.2 显存预算公式

```
总显存 = 模型权重 + KV Cache + 激活值 + 框架开销

KV Cache = 2 × L × H_kv × D × S × B × P

其中:
  L = 层数, H_kv = KV heads, D = head dim
  S = 序列长度, B = batch size, P = precision bytes

可用 KV = GPU_HBM - 模型权重 - 2 GB (overhead)
Max Batch = 可用 KV / KV_per_request
```

### 3.3 70B 模型长上下文方案

70B 模型 (130GB FP16) 的长上下文服务需要：

```
方案 A: TP=2 + H200 (141GB × 2 = 282GB 可用)
  → 每卡 65 GB 权重 + KV Cache 空间
  → 128K 可并发 ~1 请求

方案 B: TP=4 + A100 (80GB × 4 = 320GB 可用)
  → 每卡 32.5 GB 权重 + KV Cache 空间
  → 128K 可并发 ~1 请求, 但 TP 通信开销更大

方案 C: KV Cache 量化 FP8 + TP=2 + H200
  → KV Cache 减半, 128K 可并发 ~2 请求

方案 D: MoE 架构 (如 Mixtral 8x7B)
  → 激活参数少, KV Cache 少, 但总参数多
```

## 4. vLLM 中的长上下文支持

### 4.1 V1 架构支持

```
Scheduler → SchedulerOutput
  → num_scheduled_tokens: per-request token 数 (支持不同长度的请求)
  → num_common_prefix_blocks: 公共前缀 (prefix caching)

GPUModelRunner
  → _determine_batch_execution_and_padding: 自动选择 eager/CUDA Graph
  → _build_attention_metadata: 构建 block table + slot mapping
  → 支持 ubatching: 大 batch 分成 micro-batch
```

### 4.2 Attention Backend 选择

```
Full Attention: 标准 MHA/GQA (vLLM 默认)
Sliding Window: Mistral 等模型自动检测
MLA: DeepSeek-V2/V3 的压缩 KV Cache (56.9× 压缩)
Mamba: SSM 无需 KV Cache
Chunked Local: 分块局部注意力
```

详见 [vLLM V1 Executor 源码阅读](../projects/vllm-v1-executor-reading.md)。

## 5. 前沿方向

### 5.1 稀疏注意力

- **MoA (Mixture of Attention)**: 不同层使用不同稀疏模式
- **Quest**: 查询感知的 KV Cache 压缩
- **ClusterKV**: 基于 token 聚类的 KV Cache 选择

### 5.2 KV Cache 压缩

- **KV Cache Eviction**: 丢弃不重要的 KV (如 H2O)
- **Cross-Layer KV Sharing**: 层间共享 KV (如 YOCO)
- **KV Cache Distillation**: 用小模型生成 KV

### 5.3 分布式长上下文

- **Context Parallelism**: 序列维度切分 (Ring Attention)
- **P/D 分离**: Prefill 和 Decode 用不同 GPU
- **Hierarchical KV**: GPU → CPU → SSD 多级 KV Cache

## 参考资料

- 模拟器: `tools/long_context_serving_sim.py`
- 相关笔记: [KV Cache](kv-cache.md), [Prefix Caching](prefix-caching.md), [FlashAttention](flash-attention.md), [Continuous Batching](continuous-batching.md)
- vLLM: [V1 Executor](../projects/vllm-v1-executor-reading.md), [KV Cache Manager](../projects/vllm-v1-kv-cache-manager-reading.md)
