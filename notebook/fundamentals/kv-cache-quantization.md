# KV Cache 量化深度解析

> 从 FP16 到 2-bit：KV Cache 量化方法、精度影响与工程实践

## 1. 为什么需要量化 KV Cache

### 1.1 内存瓶颈

KV Cache 是 LLM 推理中除了模型权重之外最大的内存消费者：

```
KV Cache (bytes) = 2 × num_layers × batch_size × seq_len × num_kv_heads × head_dim × sizeof(dtype)
```

| 模型 | KV/Token (FP16) | 128K 上下文 | 32K × B=32 |
|------|----------------|------------|-----------|
| LLaMA-7B | 0.5 MB | 64 GB | 512 GB |
| LLaMA-70B (GQA-8) | 0.31 MB | 40 GB | 312 GB |
| Mixtral-8x7B | 0.5 MB | 64 GB | 512 GB |

**关键洞察**: 长上下文 (128K) 或大 batch (32+) 场景下，KV Cache 内存超过模型权重。A100-80GB 上 7B 模型权重 ~14GB，剩余 ~66GB 给 KV Cache — FP16 仅支持 ~132K tokens。

### 1.2 Decode 阶段的带宽瓶颈

Decode 每个 step 只处理 1 个新 token，需要读取全部 KV Cache：

```
Decode 延迟 ∝ KV_Cache_Size / HBM_Bandwidth
```

减少 KV Cache 字节数直接减少 HBM 读取量 → 提升 decode 吞吐。

## 2. 量化方法对比

### 2.1 总览

| 方法 | Bits | KV 内存缩减 | Perplexity 增加 | 硬件要求 | vLLM 支持 |
|------|------|------------|----------------|---------|----------|
| FP16 (baseline) | 16 | 1x | 0 | 所有 GPU | 原生 |
| FP8 (E4M3) | 8 | 2x | ~0.02-0.05 | Ada/Hopper | `--kv-cache-dtype fp8` |
| INT8 | 8 | 2x | ~0.03-0.08 | 所有 GPU | 有限 |
| INT4 | 4 | 4x | ~0.1-0.3 | 自定义 kernel | 研究阶段 |
| 3-bit (KVQuant) | 3 | ~5.3x | <0.1 | 自定义 kernel | 研究阶段 |
| 2-bit (KIVI) | 2 | 8x | ~0.1-0.2 | 自定义 kernel | 研究阶段 |

### 2.2 FP8 KV Cache

**原理**: 最简单的量化方式，用 FP8 E4M3 格式存储 K 和 V：

```
K_fp8 = K_fp16 / k_scale    # per-tensor scale
V_fp8 = V_fp16 / v_scale    # per-tensor scale
# Decode 时反量化:
K_fp16 = K_fp8 × k_scale
V_fp16 = V_fp8 × v_scale
```

**优势**:
- 硬件原生支持 (H100/H200/RTX 4090)
- 2x 内存缩减，精度损失极小
- 部署简单 (vLLM 单个 flag)

**缩放策略**:
- **Per-tensor**: 单个 scale 因子，硬件友好，vLLM 当前默认
- **Per-token-head**: 更精细的 scale，TurboQuant backend 支持
- **Per-channel**: 精度最好但开销最大，研究阶段

### 2.3 INT8 KV Cache

- 8-bit 整数量化，对称或非对称
- 精度接近 FP8
- 适用于无 FP8 硬件的 GPU (Ampere, Turing)
- 部署复杂度中等

### 2.4 INT4 KV Cache

- 4-bit 量化需要 group quantization (每 32/128 元素一个 scale + zero-point)
- 4x 内存缩减
- 精度下降开始显著，需要 careful tuning
- 需要自定义 CUDA kernel

## 3. KIVI: 2-bit KV Cache (ICML 2024)

### 3.1 核心洞察：K/V 不对称分布

KIVI 通过分析 LLM 中 K 和 V 的分布，发现一个关键不对称性：

- **Key Cache**: 同一 channel 维度 (head_dim) 上的元素分布相似
  → **Per-channel 量化最优** (沿 head_dim 轴分组)

- **Value Cache**: 同一 token 内的元素分布相似
  → **Per-token 量化最优** (沿 sequence 轴分组)

使用错误的分组策略会导致显著精度损失。

### 3.2 算法

```
Key 量化 (per-channel, 2-bit):
  沿 head_dim 分组 → per-channel min/max → 非对称 2-bit 量化

Value 量化 (per-token, 2-bit):
  沿 sequence 分组 → per-token min/max → 非对称 2-bit 量化
```

### 3.3 性能

| 指标 | 数值 |
|------|------|
| KV Cache 内存缩减 | **8x** |
| 最大 batch 增加 | **4x** |
| 吞吐提升 | **2.35-3.47x** |
| 精度 | "几乎无损" |
| 调优需求 | 无 (tuning-free) |

## 4. KVQuant: 3-bit + 高级技术

### 4.1 四大创新

1. **Per-Channel Key 量化**: 与 KIVI 相同的洞察
2. **Pre-RoPE 量化**: 在 RoPE 之前量化 Key。RoPE 混入位置信息后扩大动态范围，降低量化精度
3. **非均匀量化**: 按 sensitivity 加权放置量化级别，密集分布区域更多级别
4. **Dense-and-Sparse 分离**: 离群值单独处理 (sparse)，主体用更紧凑的量化范围 (dense)

### 4.2 性能

| 指标 | 数值 |
|------|------|
| 3-bit perplexity 增加 | **<0.1** |
| A100 单卡最大上下文 (7B) | **1M tokens** |
| 8-GPU 系统最大上下文 | **10M tokens** |
| 推理加速 | **~1.7x** (自定义 CUDA kernel) |

## 5. vLLM KV Cache 量化实践

### 5.1 启用 FP8 KV Cache

```bash
# 方式 1: 动态 scale (推荐)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B \
    --kv-cache-dtype fp8 \
    --calculate-kv-scales

# 方式 2: 静态 scale (从 checkpoint 加载)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B \
    --kv-cache-dtype fp8
```

### 5.2 Attention Backend 支持

| Backend | FP8 KV Cache | 说明 |
|---------|-------------|------|
| FlashInfer | 支持 | 主要 backend，per-tensor scale |
| FlashAttention | 支持 | per-tensor scale |
| Triton | 支持 | Fallback |
| TurboQuant | 支持 | per-token-head scale |
| FlashMLA | 支持 | MLA 专用 FP8 处理 |

### 5.3 源码架构

核心文件: `vllm/model_executor/layers/quantization/kv_cache.py`

```python
class BaseKVCacheMethod:
    # q_scale, k_scale, v_scale, prob_scale 作为可学习参数
    # 初始化为 -1.0 (表示未校准)
    # 权重加载: 从 checkpoint 覆盖 scale
    # FNUZ 平台检测: AMD GPU 需要 2x scale 调整
```

## 6. 量化 vs 其他优化组合

### 6.1 FP8 KV Cache + Prefix Caching

```
RAG 场景: 10K 文档 × 100 查询
  FP16: KV Cache 100GB → 放不进 A100
  FP8:  KV Cache 50GB → 放进 A100 + Prefix Caching 复用文档 KV
  → 组合效果: 内存节省 2x (量化) + 额外 90% (prefix 复用)
```

### 6.2 FP8 KV Cache + CPU Offloading

```
7B/128K 场景:
  FP16: KV=64GB, GPU 只存 1 个请求
  FP8:  KV=32GB, GPU 可存 2 个请求
  FP8 + CPU Offload: GPU 2 活跃 + CPU 16 冷却 → 18 并发
```

### 6.3 KV Cache 量化 + 模型量化

```
70B 推理:
  FP16 模型 + FP16 KV: 需要 TP=2 (140GB 权重 + 大量 KV)
  FP8 模型 + FP8 KV:   单 H200 (70GB 权重 + 2x KV 容量)
  → 两者组合效果叠加
```

## 7. 生产建议 (2025-2026)

### 7.1 推荐方案

| 场景 | 推荐配置 | 理由 |
|------|---------|------|
| 通用 H100/H200 部署 | FP8 KV Cache | 硬件原生支持，精度损失极小 |
| A100 部署 | FP8 (软件) 或 INT8 | FP8 需软件模拟但可用 |
| 超长上下文 (>100K) | FP8 + CPU Offloading | 2x 量化 + 分层存储 |
| 极端内存限制 | INT4/2-bit (研究) | 4-8x 内存缩减，需自定义 kernel |
| 精度敏感应用 | INT8 或不量化 | 最小精度损失 |

### 7.2 关键结论

1. **FP8 是生产默认选择**: vLLM 单 flag 启用，精度损失 <0.05，2x 内存缩减
2. **量化 + Offload + Prefix Caching 是最佳组合**: 三者效果叠加，可降低 50-80% 推理成本
3. **2-bit/3-bit 是未来方向**: KIVI/KVQuant 证明了极低比特量化的可行性，但需要更多工程化
4. **K/V 需要不同量化策略**: K 用 per-channel，V 用 per-token (KIVI 核心洞察)

## 参考资料

- KIVI (ICML 2024): arXiv:2402.02750
- KVQuant: arXiv:2401.18079
- vLLM KV Cache 量化: `vllm/model_executor/layers/quantization/kv_cache.py`
- 相关笔记: [KV Cache](kv-cache.md), [FP8 量化](fp8-quantization.md), [Prefix Caching](prefix-caching.md)
- 相关工具: `tools/kv_cache_offload_sim.py`, `tools/fp8_simulation.py`
