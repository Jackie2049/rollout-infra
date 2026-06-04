# vLLM V1 Sampling Pipeline 源码阅读

> 从 logits 到 token 输出的完整采样管线

## 1. 采样管线概览

```
Logits [batch, vocab]
    │
    ├── 1. 计算原始 logprobs (如果请求)
    │     └── log_softmax(logits)
    │
    ├── 2. 转 float32
    │     └── logits.to(torch.float32)
    │
    ├── 3. Allowed token ids 白名单
    │
    ├── 4. Bad words 排除
    │     └── apply_bad_words()
    │
    ├── 5. Logits Processors (非 argmax-invariant)
    │     ├── Min tokens processor
    │     └── Logit bias processor
    │
    ├── 6. Penalties (惩罚)
    │     ├── Repetition penalty
    │     ├── Frequency penalty
    │     └── Presence penalty
    │
    ├── 7. Sample
    │     ├── Greedy? → argmax
    │     ├── Apply temperature: logits /= temperature
    │     ├── Argmax-invariant processors (MinP)
    │     ├── Top-K / Top-P
    │     └── Random sample from distribution
    │
    └── 8. Gather logprobs + 返回 SamplerOutput
```

## 2. 核心类

### Sampler (`vllm/v1/sample/sampler.py`)

```python
class Sampler(nn.Module):
    def forward(self, logits, sampling_metadata, predict_bonus_token=False):
        # 1. 原始 logprobs (用于 top-k logprobs 返回)
        if num_logprobs or logprob_token_ids:
            raw_logprobs = self.compute_logprobs(logits)  # log_softmax

        # 2-6. Logits 处理
        logits = self.apply_logits_processors(logits, sampling_metadata)

        # 7. 采样
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)

        # 8. 返回
        return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=...)
```

### SamplingMetadata

每个请求的采样参数，批量化存储:

```python
@dataclass
class SamplingMetadata:
    temperature: torch.Tensor          # [batch] 温度
    top_k: torch.Tensor               # [batch] Top-K
    top_p: torch.Tensor               # [batch] Top-P
    all_greedy: bool                   # 是否全部 greedy
    all_random: bool                   # 是否全部 random
    generators: dict[int, torch.Generator]  # 随机数生成器
    max_num_logprobs: int | None
    logitsprocs: LogitsProcessors      # 自定义处理器
```

## 3. Logits 处理管线详解

### 3.1 Logits Processors (非 argmax-invariant)

这些处理器**影响 greedy 采样结果**:

| Processor | 作用 | 影响 greedy? |
|-----------|------|-------------|
| Min tokens | 强制生成至少 N 个 token | 是 |
| Logit bias | 对特定 token 加偏置 | 是 |

### 3.2 Penalties

```python
def apply_all_penalties(logits, prompt_token_ids, presence_pen, frequency_pen, repetition_pen, output_token_ids):
    # 结合 prompt token + 已输出 token 计算惩罚
    return apply_penalties(logits, prompt_token_ids, output_tokens_t, ...)
```

- **Repetition penalty**: `logit = logit > 0 ? logit/penalty : logit*penalty`
- **Frequency penalty**: `logit -= frequency * count(token)`
- **Presence penalty**: `logit -= presence * (count > 0)`

### 3.3 Argmax-invariant Processors

这些处理器**不影响 greedy 采样结果**:

| Processor | 作用 |
|-----------|------|
| MinP | 移除概率 < min_p × max_prob 的 token |

## 4. 采样方法

### 4.1 Greedy

```python
@staticmethod
def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1).view(-1)
```

- 最简单: 直接取 argmax
- 无随机性, 确定性输出
- 速度最快

### 4.2 Random (Temperature + Top-K/Top-P)

```python
def sample(self, logits, sampling_metadata):
    # 1. Greedy 采样 (备用)
    if not all_random:
        greedy_sampled = self.greedy_sample(logits)

    # 2. Temperature
    logits = logits.div_(temperature.unsqueeze(1))

    # 3. Argmax-invariant processors (MinP)
    for processor in logitsprocs.argmax_invariant:
        logits = processor.apply(logits)

    # 4. Top-K + Top-P
    random_sampled = self.topk_topp_sampler(logits, generators, top_k, top_p)

    # 5. 选择 greedy or random
    sampled = torch.where(temperature < eps, greedy_sampled, random_sampled)
    return sampled
```

### 4.3 Top-K/Top-P 实现

**文件**: `vllm/v1/sample/ops/topk_topp_sampler.py`

三种实现:
1. **FlashInfer Sampler**: GPU 原生采样, 最快
2. **Triton Kernel**: `apply_top_k_top_p_triton()` 自定义 kernel
3. **PyTorch Fallback**: `torch.topk()` + `torch.softmax()`

选择逻辑:
```python
if flashinfer_sampler_supported():
    # FlashInfer: GPU 原生, 支持 CUDA Graph
    return flashinfer.sampling.top_k_top_p_sampling_from_logits(...)
elif HAS_TRITON:
    # Triton kernel: 自定义, 灵活
    return apply_top_k_top_p_triton(...)
else:
    # PyTorch fallback
    logits = torch.topk(logits, top_k)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)
```

## 5. Top-K/Top-P Triton Kernel

**文件**: `vllm/v1/sample/ops/topk_topp_triton.py`

```python
@triton.jit
def top_k_top_p_kernel(
    logits_ptr, output_ptr, top_k_ptr, top_p_ptr,
    vocab_size, ...
):
    # 1. 读取 logits 到 SRAM
    # 2. Top-K: 保留最大的 K 个值
    # 3. Top-P: 在 Top-K 结果上计算累积概率, 截断
    # 4. 归一化并采样
```

特点:
- Triton 实现, 编译到 PTX
- 避免 PyTorch topk 的大矩阵分配
- 支持 CUDA Graph

## 6. 与 Speculative Decoding 集成

### RejectionSampler (`vllm/v1/sample/rejection_sampler.py`)

```python
# Spec decode 的采样路径:
# 1. 目标模型对 draft token 位置计算 logits
# 2. 拒绝采样: acceptance_prob = P(target)/P(draft)
# 3. 如果接受: 继续
# 4. 如果拒绝: 从修正分布采样 bonus token
```

### SpecDecodeMetadata

```python
@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor      # [num_tokens]
    num_draft_tokens: list[int]        # [batch_size]
    cu_num_draft_tokens: torch.Tensor  # 累积
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor
```

## 7. 性能优化

### CUDA Graph 兼容

- Greedy 采样: `argmax` 可用 CUDA Graph
- Top-K/Top-P: FlashInfer sampler 支持 CUDA Graph
- Triton kernel: 需要固定 shape, 用 `mark_unbacked` 处理动态 batch

### 批量采样

```python
# 所有请求的采样参数批量化:
temperature = torch.tensor([0.7, 1.0, 0.0, ...])  # [batch]
top_k = torch.tensor([50, 100, 1, ...])             # [batch]
top_p = torch.tensor([0.9, 0.95, 1.0, ...])         # [batch]

# 一个 kernel 处理所有请求
logits.div_(temperature.unsqueeze(1))  # 批量 temperature
```

### Logprobs 优化

- `raw_logprobs`: 用原始 logits 计算 (不受 penalty/temperature 影响)
- `processed_logprobs`: 用处理后的 logits 计算
- `logprob_token_ids`: 只收集特定 token 的 logprobs (避免全 vocab 计算)

## 8. 关键洞察

1. **处理顺序固定**: whitelist → bad_words → min_tokens → logit_bias → penalties → temperature → min_p → top_k/p → sample
2. **Greedy vs Random 分支**: 先算 greedy, 再算 random, 最后用 temperature 选择
3. **三种 Top-K/P 实现**: FlashInfer (最快) > Triton > PyTorch (fallback)
4. **Penalty 三种类型**: repetition (乘法), frequency (减法×次数), presence (减法×存在)
5. **CUDA Graph**: FlashInfer sampler 兼容, Triton 需 mark_unbacked
6. **Logprobs 两种模式**: raw (不受 penalty 影响) vs processed (受影响)

## 参考资料

- 源码: `vllm/v1/sample/` (sampler.py, ops/, logits_processor/)
- 相关: [vLLM V1 Executor](vllm-v1-executor-reading.md), [Spec Decoding](vllm-spec-decode-reading.md)
- `tools/sampling_simulator.py` — 采样方法模拟器
