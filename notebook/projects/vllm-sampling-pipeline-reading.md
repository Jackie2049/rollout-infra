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

## 9. GRPO 采样交互

### 9.1 GRPO rollout 采样需求

GRPO训练中的rollout阶段需要特殊采样行为:

| 需求 | 说明 | vLLM配置 |
|------|------|----------|
| 多response生成 | 同prompt生成N个response (rollout_n) | 重复发送N个请求, 不同seed |
| temperature>0 | 需要随机采样, 不能greedy | `temperature=0.7-1.0` |
| top_p限制 | 避免极端采样 | `top_p=0.9-0.95` |
| logprobs收集 | 需要每个token的logprob计算GRPO advantage | `logprobs=1` |
| 响应长度限制 | 防止生成过长 | `max_tokens=512-2048` |

### 9.2 GRPO bypass_mode与采样

★★★ bypass_mode的核心交互:

```
标准PPO路径:
  Actor forward → sample tokens → get log_probs
  Ref forward → get ref_log_probs (需要额外forward!)

bypass_mode (★★★ rLLM Tinker + verl V1):
  Actor forward → sample tokens → get log_probs = rollout_log_probs
  bypass: old_log_probs = rollout_log_probs (零额外forward!)
  → 省掉ref_in_actor的forward → ~3.5s/step on RTX 4090
```

**采样关键**: rollout阶段必须收集logprobs(`logprobs=1`), 否则bypass_mode无法工作。

### 9.3 outcome reward = last token

```
GRPO outcome reward:
  reward = reward_fn(response)  → scalar
  ★★★ outcome reward = last nonzero token位置

  token-level reward mask:
  [0, 0, 0, ..., 0, reward]  → 只有最后一个token获得reward
  → GRPO天然对齐: group-relative advantage = (reward - mean(group_rewards)) / std
```

**采样交互**: vLLM的`SamplingParams.max_tokens`控制response长度 → reward只在response结束时赋值。

## 10. Top-n-sigma Logits Processor (★★★ 我们的fork PR #7)

### 10.1 算法

```
Top-n-sigma = 动态top-k, 基于logits分布自适应选择候选token数

算法:
  1. 计算logits的mean和std
  2. threshold = mean + n * std  (n通常=2)
  3. 保留所有 logits > threshold 的token
  4. 从候选token中采样

优点:
  - 自适应: 窄分布→少候选(高效), 宽分布→多候选(多样)
  - 比fixed top-k更灵活
  - 比top-p更可控(不会collapse到1-2个token)
```

### 10.2 我们的实现

**Fork PR #7**: Top-n-sigma logits processor, vectorized实现:
- 原版: 逐token Python循环 → O(batch × vocab) CPU
- 我们的: Triton vectorized kernel → 10-66x speedup
- 状态: 已完成, 待upstream PR到vLLM

```python
# Vectorized top-n-sigma: batch处理
def top_n_sigma_logits_processor(token_ids, logits):
    # 1. 计算每个样本的mean+std → threshold
    thresholds = logits.mean(dim=-1) + n * logits.std(dim=-1)
    # 2. 批量mask: logits < threshold → -inf
    mask = logits < thresholds.unsqueeze(-1)
    logits[mask] = -float('inf')
    return logits
```

### 10.3 对GRPO的意义

```
★★★ GRPO需要多样化response → Top-n-sigma天然适配:
  - rollout_n=8 → 需要8个不同response
  - fixed top-k=50 → 可能过于确定 → response多样性不足
  - top_n_sigma(n=2) → 动态候选 → 保证多样性+避免极端token
  - 特别适合数学推理: 窄分布(确定步骤)→少候选, 宽分布(探索)→多候选
```

## 11. SM89 采样约束

### 11.1 RTX 4090 采样限制

| 特性 | SM89状态 | 影响 |
|------|---------|------|
| FlashInfer sampler | ✓ 支持 | Top-K/P采样正常, CUDA graph兼容 |
| Triton sampling kernel | ✓ 支持 | Fallback路径, SM89完全可用 |
| FP8 logits | ✗ 不支持 | logits必须BF16/FP16 → 采样输入正常 |
| CUDA graph + sampling | ✓ 支持 | FlashInfer sampler在SM89上graph-compatible |
| Speculative decoding sampling | ✓ 支持 | EAGLE/MTP rejection sampling正常 |

### 11.2 RTX 4090 采样配置

```python
# ★★★ RTX 4090最优GRPO采样配置
SamplingParams(
    temperature=0.7,           # 多样性+稳定性平衡
    top_p=0.95,               # 稍宽, 保证diversity
    max_tokens=1024,           # response长度限制
    logprobs=1,               # ★★★ 必须! bypass_mode需要
    seed=None,                # 每个rollout不同seed → 不同response
)
```

### 11.3 采样性能瓶颈分析

```
RTX 4090 decode阶段 (memory-bound):
  95.1% 时间读权重 → sampling几乎无开销(<1%)

  Sampling时间分解:
  - logits→float32: <0.1ms (batch小)
  - temperature: <0.1ms (in-place div)
  - Top-K/P FlashInfer: <0.5ms (GPU native)
  - argmax/gather: <0.1ms

  ★★★ 采样不是瓶颈 → 权重读取才是
  → INT4 = 权重4x小 → 采样不变 → 整体3.4x加速
```

## 12. 关键洞察 (更新)

1. **处理顺序固定**: whitelist → bad_words → min_tokens → logit_bias → penalties → temperature → min_p → top_k/p → sample
2. **Greedy vs Random 分支**: 先算 greedy, 再算 random, 最后用 temperature 选择
3. **三种 Top-K/P 实现**: FlashInfer (最快) > Triton > PyTorch (fallback)
4. **Penalty 三种类型**: repetition (乘法), frequency (减法×次数), presence (减法×存在)
5. **CUDA Graph**: FlashInfer sampler 兼容, Triton 需 mark_unbacked
6. **Logprobs 两种模式**: raw (不受 penalty 影响) vs processed (受影响)
7. ★★★ **GRPO采样**: rollout_n=8需要logprobs=1 → bypass_mode必需 → 3.5s/step
8. ★★★ **outcome reward**: last nonzero token → GRPO天然对齐
9. ★★★ **Top-n-sigma**: 动态top-k → vectorized 10-66x → 我们的PR #7
10. ★★★ **SM89采样**: 全部兼容 → 采样不是瓶颈 → INT4才是

## 参考资料

- 源码: `vllm/v1/sample/` (sampler.py, ops/, logits_processor/)
- 相关: [vLLM V1 Executor](vllm-v1-executor-reading.md), [Spec Decoding](vllm-spec-decode-reading.md)
- ★★★ [GRPO Training Loop](verl-grpo-training-loop-internals-reading.md) — bypass_mode+采样交互
- ★★★ [Top-n-sigma PR](https://github.com/Jackie2049/vllm/pull/7) — 我们的vectorized实现
- `tools/sampling_simulator.py` — 采样方法模拟器
- `tools/sm89_compatibility_checker.py` — SM89特性兼容矩阵
