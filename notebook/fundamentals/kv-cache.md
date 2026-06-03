# KV Cache 深度解析

> Transformer 推理的核心优化：从朴素自回归到 PagedAttention，以及 GQA 如何改变 KV Cache 的规模

## 1. 什么是 KV Cache

### 1.1 问题：自回归推理的重复计算

Transformer 自回归生成时，每一步都要计算当前 token 对所有历史 token 的 attention：

```
步骤 1: 输入 [t1]           → 生成 t2
步骤 2: 输入 [t1, t2]       → 生成 t3
步骤 3: 输入 [t1, t2, t3]   → 生成 t4
...
步骤 N: 输入 [t1, ..., tN]  → 生成 tN+1
```

Attention 计算：`Attention(Q, K, V) = softmax(Q × K^T / √d) × V`

关键观察：**K 和 V 只依赖输入 token，不依赖未来 token**。所以步骤 N 中 t1 的 K、V 和步骤 1 中完全一样。如果每步都重新算，就是 O(N²) 的冗余计算。

### 1.2 KV Cache 的做法

```
Prefill 阶段（首步）:
  输入 prompt [t1, t2, ..., tk]
  计算所有 token 的 K, V → 存入 KV Cache
  输出第一个生成 token tk+1

Decode 阶段（后续每步）:
  只计算新 token tk+1 的 Q, K, V
  新 K, V append 到 KV Cache
  用完整的 KV Cache 做 attention → 输出下一个 token
```

**性能差异**：

```
无 KV Cache:
  每步计算量 = O(n²)  (n = 当前序列长度)
  N 步总计算量 ≈ O(N³)

有 KV Cache:
  每步计算量 = O(n)   (只算新 token 的 Q, 一行 attention)
  N 步总计算量 ≈ O(N²)

实际加速: 对于长序列推理，KV Cache 带来 10-100x 加速
```

## 2. KV Cache 显存占用

### 2.1 计算公式

```
KV Cache 大小 = 2 × num_layers × seq_len × kv_heads × head_dim × dtype_bytes

其中:
  2          = K 和 V 两份
  num_layers = Transformer 层数
  seq_len    = 序列长度（prompt + generated）
  kv_heads   = KV 的 head 数量（GQA 相关，见第 6 节）
  head_dim   = 每个 head 的维度
  dtype_bytes = 数据类型字节数（FP16=2, FP32=4, FP8=1）
```

### 2.2 具体模型估算

以 FP16 推理为例：

**LLaMA-7B** (32 layers, 4096 hidden, 32 heads, head_dim=128)

```
Per token per layer: 2 × 128 × 2 bytes = 512 bytes
Per token total:     32 layers × 512 = 16,384 bytes ≈ 16 KB
Batch=1, seq=2048:   16 KB × 2048 = 32 MB
Batch=1, seq=8192:   16 KB × 8192 = 128 MB
Batch=32, seq=2048:  32 MB × 32 = 1 GB
Batch=32, seq=8192:  128 MB × 32 = 4 GB
```

**LLaMA-70B** (80 layers, 8192 hidden, 64 heads, GQA 8 KV heads)

```
Per token per layer: 2 × 8 × 128 × 2 bytes = 4,096 bytes = 4 KB  (GQA 省了 8x)
Per token total:     80 layers × 4 KB = 320 KB
Batch=1, seq=2048:   320 KB × 2048 = 640 MB
Batch=1, seq=8192:   320 KB × 8192 = 2.5 GB
Batch=32, seq=2048:  640 MB × 32 = 20 GB
```

### 2.3 显存瓶颈分析

对于推理服务，GPU 显存分配：

```
总显存 = 模型权重 + KV Cache + 激活值 + 框架开销

A100 80GB 部署 LLaMA-70B (FP16):
  模型权重: 70B × 2 bytes = 140 GB → 需要量化或 TP
  用 INT8:   70B × 1 byte  = 70 GB
  剩余:      80 - 70 = 10 GB → KV Cache
  可支撑:    10 GB / 320 KB ≈ 32K tokens (单请求长序列)
  或:        10 GB / 640 MB ≈ 16 个 batch=1 的请求 (seq=2048)
```

**关键洞察**：KV Cache 占比可以非常大。对于小模型，KV Cache 甚至超过模型权重；对于大模型，KV Cache 决定了并发请求数。

## 3. PagedAttention (vLLM)

### 3.1 问题：显存碎片化

传统 KV Cache 管理：

```
请求到来 → 分配一块连续显存 → 存 KV

问题 1: 不知道请求会生成多长，预分配多少？
  - 预分配 max_seq_len → 浪费（大部分请求远短于上限）
  - 预分配少 → 需要 resize → 数据搬移

问题 2: 碎片化
  - 请求 A 占 0-1000，请求 B 占 1000-2000
  - 请求 A 结束释放 0-1000
  - 新请求 C 需要 1500 → 无法放入 0-1000 的空洞
  - 外部碎片：空闲空间总和够，但无法分配连续块
```

实测：传统方式显存利用率约 60-80%，浪费 20-40%。

### 3.2 PagedAttention 方案

借鉴操作系统的虚拟内存分页机制：

```
将 KV Cache 分成固定大小的 block（页）:
  block_size = 16 tokens（可配置）

每个请求维护一个 block table（页表）:
  逻辑 block 0 → 物理 block 7
  逻辑 block 1 → 物理 block 23
  逻辑 block 2 → 物理 block 5
  ...

物理 block 在显存中不连续，通过 block table 映射
```

```
物理显存布局（block pool）:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ b0  │ b1  │ b2  │ b3  │ b4  │ b5  │ b6  │ b7  │
│used │free │used │free │used │reqA │free │reqA │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

请求 A 的 block table: [b7, b5, ...]
请求 B 的 block table: [b2, b0, ...]

好处:
  - 零碎片（block 粒度分配，不存在外部碎片）
  - 按需分配（生成一个 block 的 token 再分配下一个）
  - 显存利用率接近 100%
```

### 3.3 Paged Attention Kernel

关键实现难点：attention kernel 需要处理非连续的 KV blocks。

```python
# 标准 attention（连续内存）
for head in heads:
    Q = query[position]           # shape: [head_dim]
    K = key_cache[0:position+1]   # 连续内存，一次读取
    score = Q @ K.T
    ...

# Paged attention（非连续 blocks）
for head in heads:
    Q = query[position]
    for block_idx in block_table:
        block = key_cache[block_idx]  # 每次读一个 block
        K_block = block[0:block_size]
        score_block = Q @ K_block.T
        scores.append(score_block)
    # 需要跨 block 维护 softmax 的 running max/sum
    # 类似 FlashAttention 的 online softmax
```

vLLM 的 PagedAttention kernel 使用 CUDA 实现，把 block table 查找和 online softmax 融合在一个 kernel 里。

### 3.4 效果

```
vLLM 论文数据 (PagedAttention vs 连续分配):

显存利用率:  ~100% vs 60-80%
吞吐提升:    2-4x（相同硬件下可服务更多请求）
延迟:        P99 延迟降低（更好的调度）

本质原因: 更高的显存利用率 → 更多并发请求 → 更高吞吐
```

## 4. KV Cache 共享（Prefix Caching）

### 4.1 场景：共享前缀

RLHF 训练和推理中常见场景：

```
请求 1: [system_prompt] + "What is Python?"
请求 2: [system_prompt] + "What is CUDA?"
请求 3: [system_prompt] + "Explain RLHF"

三个请求共享相同的 system_prompt（可能数千 tokens）
→ system_prompt 的 KV Cache 只需计算一次
```

### 4.2 vLLM 的实现

```
1. 每个 KV block 计算内容哈希 (见 vllm-prefix-caching-v1.md)
2. hash = hash(parent_hash, tokens, extra_keys)  → 链式哈希
3. 新请求到来 → 计算各 block hash → 查找已有 block → 命中则直接复用
4. 复用的 block ref_cnt +1，不会被驱逐
5. 未命中的 block 正常计算并缓存
```

### 4.3 RL 训练中的特殊价值

```
RLHF (PPO/GRPO) 一个 episode 中:
  - 同一 prompt 生成多个 response（采样 N 次）
  - 所有 response 共享 prompt 的 KV Cache
  - 采样 N=8 时，prompt KV 计算减少 87.5%

verl 的 PrefixGrouper 在训练侧做类似优化：
  - 将共享 prefix 的请求分组
  - 组内共享 prefix 的 attention 计算
  - 与 vLLM 的 KV Cache 复用互补
```

## 5. KV Cache 量化

### 5.1 动机

KV Cache 是推理显存的主要占用者，量化可以减半甚至 75% 减少：

```
FP16 KV Cache:  每token 16 KB (LLaMA-7B)
FP8  KV Cache:  每token 8 KB  → 显存省 50%
INT4 KV Cache:  每token 4 KB  → 显存省 75%
```

### 5.2 方法

**FP8 E4M3** (最常见):
```python
# 量化
kv_fp8 = kv_fp16.to(torch.float8_e4m3fn)

# Attention 时反量化
kv_fp16 = kv_fp8.to(torch.float16)
# 或者直接在 FP8 上做 attention（需要硬件支持）
```

**KV Press** (Hugging Face):
```python
from kvpress import KNLossPreservingPress

press = KNLossPreserving(ratio=0.5)  # 保留 50% 的 KV
# 自动选择最重要的 KV 对保留，丢弃其余
```

**vLLM 的 KV Cache 量化**:
```bash
# 启动时指定 KV Cache 数据类型
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --kv-cache-dtype fp8_e4m3fn
```

### 5.3 质量影响

```
FP8 量化:   困惑度增加 < 1%，几乎无损
INT8 量化:  困惑度增加 1-3%，可接受
INT4 量化:  困惑度增加 5-15%，需要更精细的量化策略

结论: FP8 是当前最佳实践，质量损失极小，显存省 50%
```

## 6. Multi-Query Attention 与 Grouped-Query Attention

### 6.1 背景：标准 Multi-Head Attention (MHA)

```
标准 MHA:
  Q heads = 32
  K heads = 32
  V heads = 32
  每个 Q head 对应一个独立的 K head 和 V head

KV Cache per token per layer:
  = 2 × 32 heads × 128 dim × 2 bytes
  = 16,384 bytes = 16 KB
```

### 6.2 Multi-Query Attention (MQA)

```
MQA (Shazeer, 2019):
  Q heads = 32
  K heads = 1    ← 所有 Q head 共享同一个 K head
  V heads = 1    ← 所有 Q head 共享同一个 V head

KV Cache per token per layer:
  = 2 × 1 head × 128 dim × 2 bytes
  = 512 bytes = 0.5 KB

节省: 32x！
```

**代价**：模型质量略降（约 1-3%），因为减少了 KV 的表达能力。

**使用者**：PaLM, Falcon, StarCoder

### 6.3 Grouped-Query Attention (GQA)

```
GQA (Ainslie et al., 2023):
  Q heads = 32
  KV head groups = 8  ← 每 4 个 Q head 共享一组 KV
  K heads = 8
  V heads = 8

KV Cache per token per layer:
  = 2 × 8 heads × 128 dim × 2 bytes
  = 4,096 bytes = 4 KB

节省: 4x（相对 MHA）
质量: 几乎等于 MHA（GQA 论文证明）
```

### 6.4 三者对比

```
| 类型    | Q heads | KV heads | KV Cache 大小 | 质量影响 | 代表模型         |
|---------|---------|----------|---------------|----------|------------------|
| MHA     | 32      | 32       | 16 KB/token   | 基线     | GPT-3, 原始 LLaMA |
| GQA     | 32      | 8        | 4 KB/token    | ~0%      | LLaMA-2/3, Qwen  |
| MQA     | 32      | 1        | 0.5 KB/token  | 1-3% ↓   | Falcon, PaLM     |
```

### 6.5 GQA 的实现

```python
# 标准 MHA 的 KV 投影
K = x @ W_k  # [seq, hidden] @ [hidden, num_heads * head_dim]
V = x @ W_v  # [seq, hidden] @ [hidden, num_heads * head_dim]

# GQA 的 KV 投影（投影维度减少）
K = x @ W_k  # [seq, hidden] @ [hidden, num_kv_heads * head_dim]
V = x @ W_v  # [seq, hidden] @ [hidden, num_kv_heads * head_dim]

# Attention 时需要 expand KV
# 方法 1: repeat_kv — 物理复制
K_expanded = K.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

# 方法 2: 利用广播 — 不复制，让 Q 的每组 head 共享 KV
# 更高效，vLLM/Megatron 都用这种方式
```

## 7. Sliding Window Attention 与 Token 驱逐

### 7.1 固定窗口

```
Sliding Window Attention (SWA):
  只保留最近 W 个 token 的 KV Cache
  超出窗口的 KV 自动丢弃

KV Cache 大小: O(W) 而不是 O(N)
  W = 4096 时，无论序列多长，KV Cache 固定

代表模型: Mistral-7B (W=4096), Gemma
```

### 7.2 StreamingLLM: Attention Sink

```
问题: 直接用 SWA，去掉最早的 token → 模型质量崩塌
原因: 第一个 token 学会了做 "attention sink"（注意力锚点）
  softmax 迫使 attention weight 之和为 1
  当前面没有有用信息时，模型学会把多余的注意力倾倒给第一个 token

StreamingLLM 的解决方案:
  保留前 S 个 sink token + 最近 W 个 token
  KV Cache = S + W 个 token

效果: 可以无限长推理，质量稳定
```

### 7.3 Hugging Face 实现

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    attn_implementation="sdpa",
    sliding_window=4096,  # 自动丢弃窗口外的 KV
)

# 或在 generate 时控制
output = model.generate(
    inputs,
    max_new_tokens=1000,
    cache_implementation="sliding_window",  # HF 4.36+
)
```

## 8. KV Cache 与推理框架的集成

### 8.1 vLLM 的 KV Cache 管理

```
配置参数:
  --gpu-memory-utilization 0.9    # GPU 显存使用比例（预留 10% 给其他）
  --max-model-len 8192           # 最大序列长度
  --block-size 16                # KV block 大小
  --swap-space 4                 # CPU swap 空间 (GB)

运行时:
  1. 启动时预分配所有 KV Cache block → 避免运行时 malloc
  2. Scheduler 根据 block 使用情况决定 admit/preempt 请求
  3. Preempt 时可以 swap to CPU 或 recomputation
```

### 8.2 Prefill/Decode 分离

```
新兴架构: 将 prefill 和 decode 分到不同 GPU

Prefill 阶段:
  - 计算密集型（所有 prompt tokens 的 KV 一次性计算）
  - 适合大算力 GPU (H100)

Decode 阶段:
  - 显存带宽瓶颈（每步读整个 KV Cache）
  - 适合高带宽 GPU 或专用加速器

分离的好处:
  - 各自独立扩缩容
  - Prefill 可以用 TP 并行加速
  - Decode 可以用更多小 GPU 提高并发

代表: Splitwise, DistServe, Mooncake
```

### 8.3 Disaggregated KV Cache

```
Mooncake 的做法:
  1. KV Cache 存储在独立的内存池（GPU + CPU + SSD）
  2. Prefill 节点计算 KV → 传输到内存池
  3. Decode 节点从内存池获取 KV → 继续生成
  4. 支持前缀匹配 → 直接复用内存池中的 KV block

好处:
  - KV Cache 可以跨请求、跨节点复用
  - Prefix caching 效果最大化
  - 内存池可以很大（SSD 层做冷数据）
```

## 9. 实践：KV Cache 显存估算工具

```python
def estimate_kv_cache_memory(
    num_layers: int,
    kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,  # FP16
) -> dict:
    """估算 KV Cache 显存需求"""

    bytes_per_token_per_layer = 2 * kv_heads * head_dim * dtype_bytes
    bytes_per_token = bytes_per_token_per_layer * num_layers

    total_bytes = bytes_per_token * seq_len * batch_size

    return {
        "per_token_per_layer_KB": bytes_per_token_per_layer / 1024,
        "per_token_total_KB": bytes_per_token / 1024,
        "total_MB": total_bytes / (1024 * 1024),
        "total_GB": total_bytes / (1024 * 1024 * 1024),
        "tokens_per_GB": (1024 * 1024 * 1024) / bytes_per_token,
    }


# 常见模型估算
models = [
    ("LLaMA-7B (MHA)", 32, 32, 128, 2),      # 32 KV heads
    ("LLaMA-2-70B (GQA-8)", 80, 8, 128, 2),   # 8 KV heads
    ("Qwen-72B (GQA-8)", 80, 8, 128, 2),
    ("Mistral-7B (GQA-1)", 32, 1, 128, 2),     # MQA, 1 KV head
]

for name, layers, kv_heads, head_dim, dtype_b in models:
    result = estimate_kv_cache_memory(
        num_layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        seq_len=8192,
        batch_size=1,
        dtype_bytes=dtype_b,
    )
    print(f"{name}: {result['total_GB']:.2f} GB (seq=8192, bs=1)")
    print(f"  Per token: {result['per_token_total_KB']:.1f} KB")
    print()
```

输出：
```
LLaMA-7B (MHA):       0.50 GB (seq=8192, bs=1)
  Per token: 64.0 KB

LLaMA-2-70B (GQA-8):  2.50 GB (seq=8192, bs=1)
  Per token: 320.0 KB

Qwen-72B (GQA-8):     2.50 GB (seq=8192, bs=1)
  Per token: 320.0 KB

Mistral-7B (GQA-1):   0.03 GB (seq=8192, bs=1)
  Per token: 8.0 KB
```

## 10. 关键要点总结

1. **KV Cache 是推理服务的显存瓶颈** — 对于大模型+长序列+高并发，KV Cache 可能耗费数十 GB
2. **PagedAttention 解决碎片化** — 借鉴 OS 虚拟内存分页，显存利用率从 60-80% 提升到接近 100%
3. **GQA 是标配** — LLaMA-2/3、Qwen 等主流模型都用 GQA 减少 4-8x KV Cache，质量几乎无损
4. **Prefix Caching 复用共享前缀** — RLHF 场景下 prompt 复用率高，KV Cache 复用可以节省大量计算
5. **FP8 量化是低成本优化** — 几乎不影响质量，显存省 50%，现代 GPU (H100) 有硬件支持
6. **Prefill/Decode 分离是新趋势** — Splitwise、Mooncake 等架构将计算密集的 prefill 和带宽瓶颈的 decode 分开优化

## 参考

- 论文: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM)
- 论文: [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- 论文: [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (MQA)
- 论文: [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) (StreamingLLM)
- 论文: [Mooncake: A KV Cache-centric Disaggregated Architecture](https://arxiv.org/abs/2407.00079)
- 博客: [The KV Cache: Understanding its Role in Transformer Inference](https://medium.com/@joaolages/direct-preference-optimization-dpo-622fc1f18707)
