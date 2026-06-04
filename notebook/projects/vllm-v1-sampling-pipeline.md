# vLLM V1 Sampling Pipeline 源码深度分析

> 源码路径: `vllm/v1/sample/` (14 个文件, ~1500 行)
> 核心文件: `sampler.py` (435 行), `rejection_sampler.py` (922 行), `ops/topk_topp_sampler.py` (491 行)

## 1. 架构总览

```
Logits (来自模型前向传播)
  │
  ▼
┌─────────────────────────────────────────┐
│           Sampler.forward()             │
│  1. compute raw logprobs (if needed)    │
│  2. convert to float32                  │
│  3. apply logits processors:            │
│     ├─ allowed_token_ids (whitelist)    │
│     ├─ bad_words exclusion              │
│     ├─ non-argmax-invariant processors  │
│     ├─ penalties (rep/freq/presence)    │
│     └─ thinking_budget state            │
│  4. sample (greedy or random)           │
│  5. gather logprobs                     │
│  6. return SamplerOutput                │
└─────────────────────────────────────────┘
  │
  ▼
SamplerOutput(sampled_token_ids, logprobs_tensors)
```

## 2. 核心流程: Sampler.forward()

**文件**: `sampler.py:60-144`

### 2.1 Logprobs 预计算 (3 种模式)

```python
# sampler.py:80-88
if logprobs_mode == "raw_logprobs":       # log_softmax(logits)
    raw_logprobs = self.compute_logprobs(logits)
elif logprobs_mode == "raw_logits":       # float32(logits) — 延迟到后处理
    raw_logprobs = logits.to(torch.float32)
# "processed_logits"/"processed_logprobs": 在采样后获取
```

**设计决策**: 区分 "raw" 和 "processed" logprobs — 前者在 logit processor 之前计算 (反映原始分布), 后者在之后计算 (反映采样约束后的分布)。

### 2.2 Logits Processor 管线 (5 步)

**文件**: `sampler.py:366-415`

```python
def apply_logits_processors(self, logits, sampling_metadata, predict_bonus_token):
    # Step 1: Allowed token whitelist (masked_fill -inf)
    if sampling_metadata.allowed_token_ids_mask is not None:
        logits.masked_fill_(mask, float("-inf"))

    # Step 2: Bad words exclusion
    if bad_words_token_ids:
        apply_bad_words(logits, bad_words_token_ids, output_token_ids)

    # Step 3: Non-argmax-invariant processors (min_tokens, logit_bias)
    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        logits = processor.apply(logits)

    # Step 4: Penalties (repetition, frequency, presence)
    logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)

    # Step 5: Thinking budget state (for reasoning models)
    if holder is not None and holder.has_tracked_requests():
        logits = holder.apply_to_logits(logits, ...)
```

**关键区分**:
- **Non-argmax-invariant**: 改变 argmax 结果 (如 min_tokens 强制不生成 EOS)
- **Argmax-invariant**: 不改变 argmax (如 temperature, top_k, top_p) — 在 sample() 中应用

### 2.3 Sample (2 路径)

**文件**: `sampler.py:238-297`

```python
def sample(self, logits, sampling_metadata):
    if not sampling_metadata.all_random:
        greedy_sampled = self.greedy_sample(logits)  # argmax
        if sampling_metadata.all_greedy:
            return greedy_sampled, None  # 全部 greedy, 直接返回

    # Random sampling 路径:
    logits = self.apply_temperature(logits, temp, all_random)
    # Argmax-invariant processors (top_k, top_p 等)
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)
    # top_k/top_p + 随机采样
    random_sampled, processed_logprobs = self.topk_topp_sampler(
        logits, generators, top_k, top_p)

    # 合并 greedy 和 random 结果
    sampled = torch.where(temperature < eps, greedy_sampled, random_sampled)
```

**优化**: 即使 batch 中混合 greedy/random 请求, 也只走一次 argmax 路径, 最后用 `torch.where` 合并。

## 3. TopK/TopP Sampler (5 个后端)

**文件**: `ops/topk_topp_sampler.py`

### 3.1 后端分发策略

```python
class TopKTopPSampler(nn.Module):
    def __init__(self, logprobs_mode):
        if current_platform.is_cuda():
            if can_use_flashinfer:
                self.forward = self.forward_cuda     # FlashInfer (最快)
            else:
                self.forward = self.forward_native   # PyTorch native
        elif current_platform.is_cpu():
            self.forward = self.forward_cpu           # torch.compile
        elif current_platform.is_xpu():
            self.forward = self.forward_xpu           # Intel GPU
        elif rocm_aiter:
            self.forward = self.forward_hip           # ROCm aiter
```

**5 个采样后端**:
| 后端 | 实现 | 特点 |
|------|------|------|
| FlashInfer (CUDA) | `flashinfer.sampling.top_k_top_p_sampling_from_logits` | Rejection sampling, 无需排序 |
| PyTorch Native | sort + cumsum + masked_fill | 通用但慢 (O(V log V) 排序) |
| CPU | `@torch.compile` + Gumbel-Max | 编译优化 |
| XPU | `torch.ops.vllm.xpu_topk_topp_sampler` | Intel GPU 自定义 kernel |
| ROCm aiter | `aiter.ops.sampling.top_k_top_p_sampling_from_probs` | AMD GPU 优化 |

### 3.2 Gumbel-Max Trick (核心采样算法)

```python
# topk_topp_sampler.py:422-443
def random_sample(probs, generators):
    q = torch.empty_like(probs)
    q.exponential_()  # Gumbel(0,1) 噪声
    for i, generator in generators.items():
        q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1)  # 等价于 multinomial 但无 CPU-GPU 同步
```

**为什么不用 torch.multinomial?**
- `torch.multinomial` 需要 CPU-GPU 同步 (在 GPU 上生成随机数后需要传回 CPU)
- Gumbel-Max trick: `probs / exp(random)`, 然后 argmax — 完全在 GPU 上, 零同步
- 数学证明: argmax(probs / Gumbel_noise) 等价于按 probs 概率采样

### 3.3 Top-K/Top-P 过滤算法

```python
# topk_topp_sampler.py:337-396
def apply_top_k_top_p(logits, k, p):
    # 优先使用 Triton kernel (batch >= 8)
    if HAS_TRITON and logits.shape[0] >= 8:
        return apply_top_k_top_p_triton(logits, k, p)
    # 小 batch 用 PyTorch sort
    return apply_top_k_top_p_pytorch(logits, k, p)
```

**PyTorch 实现**:
```python
# 1. 排序 (O(V log V), V=vocab_size)
logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

# 2. Top-K: 保留最大的 k 个, 其余 -inf
top_k_mask = logits_sort < k_threshold
logits_sort.masked_fill_(top_k_mask, -inf)

# 3. Top-P: cumsum 后保留累积概率 >= 1-p 的部分
probs_sort = logits_sort.softmax(dim=-1)
probs_sum = torch.cumsum(probs_sort, dim=-1)
top_p_mask = probs_sum <= 1 - p  # 概率累积不够的mask掉
logits_sort.masked_fill_(top_p_mask, -inf)

# 4. 恢复原始顺序
logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)
```

**性能问题**: sort 是 O(V log V), V=32000 时排序开销大 → FlashInfer 用 rejection sampling 避免排序。

### 3.4 CPU 优化: Top-K Only 快速路径

```python
# topk_topp_sampler.py:399-419 (仅 CPU)
def apply_top_k_only(logits, k):
    # 避免 sort 整个 vocab, 只需 topk (O(V log k))
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index)
    return logits.masked_fill_(logits < top_k_mask, -inf)
```

**权衡**: 涉及 GPU→CPU 同步 (获取 max_top_k), 仅在 CPU 上使用。

## 4. Rejection Sampler (Speculative Decoding)

**文件**: `rejection_sampler.py` (922 行)

### 4.1 算法概述

严格遵循 [Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192)。

```
Draft tokens:     [d1,   d2,   d3,   ...]
Target probs:     [p1,   p2,   p3,   ...]
Draft probs:      [q1,   q2,   q3,   ...]

For each position i:
  1. Accept with probability min(1, pi(di) / qi(di))
  2. If rejected: sample from normalized max(pi - qi, 0) (recovered token)
  3. If all accepted: append bonus token from target distribution
```

### 4.2 Triton Kernel: rejection_greedy_sample_kernel

**文件**: `rejection_sampler.py:708-757`

```python
@triton.jit
def rejection_greedy_sample_kernel(..., SYNTHETIC_MODE: tl.constexpr):
    req_idx = tl.program_id(0)  # 每个 request 一个 program

    for pos in range(num_draft_tokens):
        if not rejected:
            if SYNTHETIC_MODE:
                accepted = uniform_prob < conditional_rate  # 合成模式
            else:
                rejected = (draft_token_id != target_argmax_id)  # Greedy: 直接比较
                token_id = target_argmax_id

    if not rejected:
        # 全部接受 → 添加 bonus token
        tl.store(output + req_idx * (max_spec_len + 1) + num_draft_tokens, bonus_token)
```

**Greedy 特化**: 不需要计算概率比值, 直接比较 argmax — 节省 softmax 开销。

### 4.3 Triton Kernel: rejection_random_sample_kernel

**文件**: `rejection_sampler.py:762-826`

```python
@triton.jit
def rejection_random_sample_kernel(...):
    req_idx = tl.program_id(0)

    for pos in range(num_draft_tokens):
        if not rejected:
            # 接受条件: uniform < target_prob / draft_prob
            draft_prob = tl.load(draft_probs + (start_idx + pos) * vocab_size + draft_token_id)
            target_prob = tl.load(target_probs + (start_idx + pos) * vocab_size + draft_token_id)
            accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob

            if accepted:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = recovered_token_id  # 从 adjusted 分布采样
```

**N-gram 模式** (`NO_DRAFT_PROBS=True`): draft_prob 设为 1 → 接受条件退化为 `target_prob >= uniform` → 等价于以 target_prob 概率接受。

### 4.4 Recovered Token 采样

**文件**: `rejection_sampler.py:659-703, 853-921`

```python
def sample_recovered_tokens(...):
    # Gumbel-Max trick for recovered tokens
    q = torch.empty((batch_size, vocab_size)).exponential_()
    inv_q = q.reciprocal()

    # Triton kernel: 逐 request 逐 position
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](...)
```

**Kernel 核心**: 逐 vocab block 计算:
```python
# rejection_sampler.py:882-919
if NO_DRAFT_PROBS:
    prob = target_prob[vocab_offset]
    prob[draft_token_id] = 0  # 排除已拒绝的 draft token
else:
    prob = max(target_prob - draft_prob, 0)  # adjusted distribution
score = prob * inv_q  # Gumbel-Max
local_max, local_id = tl.max(score, axis=0)  # 块内最大
```

**BLOCK_SIZE=8192**: 分块扫描整个 vocab, 避免一次加载 32K 元素。

### 4.5 Synthetic Mode

```python
# 预配置的接受率 (不依赖 draft model 质量)
synthetic_conditional_rates = unconditional_to_conditional_rates(
    spec_config.synthetic_acceptance_rates
)
# 在 greedy kernel 中: accepted = uniform < rate
```

**用途**: 测试/基准测试, 可以精确控制接受率而不需要真实的 draft model。

## 5. Penalties 实现

**文件**: `ops/penalties.py` (58 行)

```python
def apply_all_penalties(logits, prompt_token_ids, presence, frequency, repetition, output_token_ids):
    # 1. 将 output_token_ids (变长 list) 转为 tensor
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, device)
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)  # 填充无效位置

    # 2. 委托给底层 apply_penalties (C++/CUDA 实现)
    return apply_penalties(logits, prompt_token_ids, output_tokens_t, ...)
```

**底层实现** (`vllm/model_executor/layers/utils.py`):
- **Repetition**: `logits[tokens] /= repetition_penalty` — 直接缩放
- **Frequency**: `logits[tokens] -= frequency_penalty * count(tokens)` — 按频率惩罚
- **Presence**: `logits[tokens] -= presence_penalty` — 出现就惩罚

**性能优化**: 使用 `pin_memory=True` + `non_blocking=True` 的异步 CPU→GPU 传输。

## 6. Logprobs 收集

**文件**: `sampler.py:303-351`

```python
def gather_logprobs(logprobs, num_logprobs, token_ids):
    # 1. Top-K logprobs (最高效)
    topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

    # 2. 采样 token 的 logprob
    token_logprobs = logprobs.gather(-1, token_ids.unsqueeze(-1))

    # 3. 计算 rank (避免重编译)
    torch._dynamo.decorators.mark_unbacked(logprobs, 0)
    token_ranks = batched_count_greater_than(logprobs, token_logprobs)

    # 4. 拼接
    indices = torch.cat((token_ids, topk_indices), dim=1)
    logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
```

**mark_unbacked**: 标记 batch 维度为符号化大小, 避免 torch.compile 在 batch=1 和 batch=2 之间重编译。

### 特定 Token Logprobs (generative_scoring API)

**文件**: `sampler.py:146-220`

```python
def gather_specific_token_logprobs(logprobs, logprob_token_ids, sampled):
    # 变长 token list → padded tensor
    token_ids_cpu = torch.zeros(batch_size, max_num_tokens + 1, pin_memory=True)
    valid_mask_cpu = torch.zeros(batch_size, max_num_tokens + 1, dtype=torch.bool)

    # GPU 上 gather
    gathered_logprobs = logprobs.gather(-1, token_ids_tensor)
    gathered_logprobs.masked_fill_(~valid_mask, float("-inf"))  # pad 位置填 -inf
```

**用途**: generative_scoring API 只需要特定 token 的 logprobs, 不需要完整 top-K。

## 7. Thinking Budget State

**文件**: `thinking_budget_state.py`

推理模型 (如 DeepSeek-R1) 的 thinking token 预算控制:
- 跟踪请求的 thinking 状态 (是否在 `<think/>` 块内)
- 当 thinking token 超过预算时, 强制结束 thinking 并生成 `</think/>`
- 在 spec decode 中正确处理 thinking token 的累积

## 8. 文件结构总结

```
vllm/v1/sample/
├── sampler.py                     # 主采样器 (9步管线)
├── metadata.py                    # SamplingMetadata 数据结构
├── rejection_sampler.py           # Spec decode rejection sampler (922行, 6个Triton kernel)
├── thinking_budget_state.py       # 推理模型 thinking 预算控制
├── logits_processor/
│   ├── interface.py               # LogitsProcessor 接口
│   ├── builtin.py                 # 内置 processor (min_tokens, logit_bias)
│   └── state.py                   # 有状态 processor
└── ops/
    ├── topk_topp_sampler.py       # Top-K/Top-P 采样器 (5后端)
    ├── topk_topp_triton.py        # Triton top-k/top-p kernel
    ├── bad_words.py               # Bad words 排除
    ├── penalties.py               # Rep/Freq/Presence 惩罚
    └── logprobs.py                # Logprobs 辅助
```

## 9. 关键性能优化

| 优化 | 方法 | 效果 |
|------|------|------|
| Gumbel-Max Trick | `probs/div(q).argmax()` 替代 multinomial | 消除 CPU-GPU 同步 |
| FlashInfer Sampling | Rejection sampling 避免排序 | 避免 O(V log V) sort |
| Greedy 快速路径 | 直接 argmax, 不做 softmax | 节省 softmax 开销 |
| mark_unbacked | 符号化 batch 维度 | 避免重编译 |
| pin_memory + non_blocking | 异步 CPU→GPU 传输 | 隐藏传输延迟 |
| In-place 操作 | `div_()`, `masked_fill_()` | 减少内存分配 |
| int32 压缩 | token_ids 用 int32 而非 int64 | 减少 50% 传输量 |
| Triton kernel | rejection sampling 在 GPU 端完成 | 避免 CPU 循环 |

## 10. 与 Speculative Decoding 的交互

1. **Bonus token**: 由 `Sampler` 正常采样 (支持 top_k/top_p)
2. **Draft tokens**: 由 Proposer 生成 (N-gram/Draft model/Eagle)
3. **Rejection**: Triton kernel 在 GPU 端完成, 无 CPU 介入
4. **Rollback**: 被拒绝的 token 用 `PLACEHOLDER_TOKEN_ID=-1` 标记, 后续过滤
5. **Penalties**: spec tokens 合并到 output_token_ids 后统一计算

## 11. 采样管线数据流

```
Model Output Logits [batch, vocab]
    │
    ├─ if logprobs needed: compute_logprobs(logits) → raw_logprobs
    │
    ▼ logits.to(float32)
    │
    ├─ allowed_token_ids: masked_fill_(-inf)
    ├─ bad_words: scatter_(-inf)
    ├─ non-argmax-invariant: processor.apply()
    ├─ penalties: scatter_(scaled)
    ├─ thinking_budget: state.apply_to_logits()
    │
    ▼
    ├─ greedy: argmax(logits) ──────────────┐
    │                                        │
    ├─ random:                               │
    │   ├─ temperature: logits.div_(temp)    │
    │   ├─ argmax-invariant: processor()    │
    │   ├─ top_k: sort + masked_fill_(-inf) │
    │   ├─ top_p: cumsum + masked_fill_     │
    │   └─ sample: probs/exp().argmax()     │
    │                                        │
    ├─ merge: torch.where(greedy, random) ◄─┘
    │
    ├─ gather logprobs: topk + gather + rank
    │
    ▼
    SamplerOutput(sampled_token_ids, logprobs_tensors)
```
