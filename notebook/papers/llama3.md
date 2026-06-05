# Paper Reading: LLaMA 3 — Open Foundation Models

> Meta AI, 2024 | LLaMA 3 / LLaMA 3.1
> 精读日期: 2026-06-05
> 优先级: P1 (AI Expert Roadmap Phase 3)

## 1. 论文概要

**核心贡献**: 开源了一系列基础模型 (8B, 70B, 405B), 证明标准 Transformer 架构 + 大规模数据可以匹敌 GPT-4.

**关键数据**:
- 405B 模型: 16K GPUs H100, 30.84M GPU-hours, 15T+ tokens
- 8B 模型: 性能超越 Mistral 7B 和 Gemma 7B
- 70B 模型: 匹配 GPT-3.5 Turbo
- 405B 模型: 匹配 GPT-4 (在许多任务上)

## 2. 架构

### 2.1 设计选择: 为什么回归标准架构?

```
LLaMA 3 的设计哲学: "Keep it simple"

对比 LLaMA 2 的变化:
  几乎没有! 核心架构完全一样:
  - Decoder-only Transformer
  - Pre-norm with RMSNorm
  - SwiGLU activation
  - RoPE positional encoding
  - GQA (Grouped Query Attention)

唯一变化:
  8B:  GQA with 8 KV heads (vs MHA in LLaMA 2 7B)
  70B: GQA with 8 KV heads (same as LLaMA 2 70B)
  405B: GQA with 8 KV heads

为什么不用 MLA/MoE 等创新?
  1. 标准架构更容易 scale (工程简单)
  2. 推理效率好 (GQA 已经足够)
  3. 社区兼容性 (广泛支持的架构)
  4. 数据规模 > 架构创新 (LLaMA 3 的核心论点)
```

### 2.2 模型规格

| 规格 | 8B | 70B | 405B |
|------|-----|------|-------|
| d_model | 4096 | 8192 | 16384 |
| n_layers | 32 | 80 | 126 |
| n_heads | 32 | 64 | 128 |
| n_kv_heads | 8 | 8 | 8 |
| ffn_dim | 14336 | 28672 | 53248 |
| vocab_size | 128256 | 128256 | 128256 |
| 训练 tokens | 15T+ | 15T+ | 15T+ |
| 训练 GPU-hours | 1.3M | 7.0M | 30.84M |

### 2.3 Tokenizer

```
BPE tokenizer (tiktoken)
vocab_size = 128,256 (比 LLaMA 2 的 32K 大 4x)

为什么更大 vocab?
  1. 多语言支持更好 (非英语文本编码更高效)
  2. 代码处理更好 (常见代码模式成为单个 token)
  3. 压缩率: ~4 chars/token (英语)
  4. 推理效率: 更少的 tokens → 更短的序列

对比:
  GPT-4 (cl100k): 100K vocab
  LLaMA 2: 32K vocab
  LLaMA 3: 128K vocab → 目前最大
```

## 3. 训练基础设施

### 3.1 硬件

```
16,384 × H100 80GB GPU
互联: NVLink + InfiniBand (4x400 Gbps)

训练 FLOPS 估算:
  405B 模型, 15T tokens
  理论 FLOPS = 6 × 405B × 15T = 3.6 × 10^25
  实际 GPU-hours = 30.84M
  MFU = 3.6e25 / (30.84M × 3600 × 990e12) ≈ 33%
  (H100 FP16 峰值 ~990 TFLOPS)

对比:
  Chinchilla 最优: MFU ~50% (理论)
  LLaMA 3: MFU ~33% (实际, 包含 checkpointing/通信等开销)
```

### 3.2 并行策略

```
405B 模型 (16384 GPUs):
  TP = 8   (Tensor Parallelism, NVLink 节点内)
  PP = 16  (Pipeline Parallelism, 跨节点)
  DP = 128 (Data Parallelism, 梯度 AllReduce)

8B 模型:
  DP = 256 (纯数据并行, 模型 fit 单 GPU)

70B 模型:
  TP = 8
  PP = 1
  DP = 64

关键: 405B 需要 3D 并行, 70B 只需 TP+DP, 8B 纯 DP
```

### 3.3 训练稳定性

```
LLaMA 3 遇到的训练挑战:

1. Loss spikes:
   - 出现在特定数据批次
   - 解决: 降低学习率, 跳过问题数据

2. Gradient norm explosion:
   - 405B 模型更常见
   - 解决: gradient clipping (norm=1.0)

3. Hardware failures:
   - 16K GPU 集群每天 ~1-2 个 GPU 故障
   - 解决: checkpoint 频繁保存, 自动恢复

4. Checkpoint 大小:
   405B AdamW checkpoint = 405B × 16 bytes = 6.48 TB!
   → 分片保存 (每个 rank 保存自己的分片)
```

## 4. 数据

### 4.1 数据组成

```
15T+ tokens 训练数据:
  - 英语: ~70%
  - 非英语: ~20% (30+ 语言)
  - 代码: ~10%

关键数据质量管线:
  1. 去重 (MinHash + exact dedup)
  2. 质量过滤 (fasttext classifier)
  3. 安全过滤 (有害内容移除)
  4. 域名过滤 (高质量网站优先)

数据配比很重要:
  代码太少 → 推理能力下降
  非英语太少 → 多语言能力差
  → 需要精心平衡不同数据源
```

## 5. 后训练对齐

### 5.1 SFT + DPO + PPO

```
LLaMA 3 对齐流程:

Step 1: SFT (Supervised Fine-Tuning)
  - 高质量对话数据
  - 代码, 数学, 推理示例

Step 2: Rejection Sampling
  - 模型生成多个回答
  - 用 RM 选最好的
  - 再做 SFT

Step 3: DPO (Direct Preference Optimization)
  - 用 RM 生成偏好对
  - DPO 直接优化

Step 4: PPO (可选)
  - 在某些场景进一步优化
  - 但 DPO 是主要方法

关键: LLaMA 3 的对齐主要靠 DPO, 不是 PPO!
→ 与 InstructGPT 的 PPO 不同
→ 行业趋势: DPO 简单稳定, 逐渐取代 PPO
```

## 6. 与其他架构对比

| 特性 | LLaMA 3 | DeepSeek-V3 | Mixtral | GPT-4 |
|------|---------|-------------|---------|-------|
| 架构 | Dense | MoE | MoE | 未知(可能MoE) |
| 参数 | 8/70/405B | 671B/37B active | 46.7B/12.9B active | ~1.8T? |
| Attention | GQA | MLA | GQA | 未知 |
| 位置编码 | RoPE | RoPE (decoupled) | RoPE | 未知 |
| 激活 | SwiGLU | SwiGLU | SwiGLU | 未知 |
| 训练 tokens | 15T+ | 14.8T | 8T | ~13T? |
| 开源 | 完全 | 完全 | 完全 | 闭源 |
| MoE | 否 | 是 | 是 | 可能 |

**核心洞察**: LLaMA 3 用 Dense 架构 + 大数据匹敌 MoE 模型
→ **数据规模 > 架构创新**

## 7. 对 AI Infra 的启示

### 7.1 大规模训练系统设计

```
从 LLaMA 3 学到的 Infra 经验:

1. 数据管线 > 一切:
   - 数据质量决定模型质量
   - 去重/过滤/配比是最重要的工程决策

2. Checkpoint 管理:
   - 405B checkpoint = 6.48TB
   - 异步保存, 分片, 定期清理
   - 自动故障恢复

3. 通信优化:
   - NVLink 节点内 TP (8 卡)
   - IB 跨节点 PP (最小化通信量)
   - DP AllReduce (ZeRO 分片优化器)

4. 监控和调试:
   - 实时 loss 曲线监控
   - 自动异常检测 (loss spike)
   - 硬件故障自动切换
```

## 8. 核心学习

1. **简单架构 + 大数据 = 强模型**: LLaMA 3 没有架构创新, 但数据做到了极致
2. **GQA 是标配**: 8 KV heads 对推理效率至关重要 (KV Cache 大幅减少)
3. **128K vocab**: 大 vocab 提高编码效率, 是现代趋势
4. **3D 并行是 405B 的必需**: TP=8+PP=16+DP=128, 缺一不可
5. **DPO > PPO**: 简单稳定, 已成为对齐首选
6. **训练稳定性是工程问题**: loss spike, GPU 故障, checkpoint 管理都是 infra 核心挑战
7. **MFU ~33%**: 实际训练效率远低于理论峰值, 还有很大优化空间
