# vLLM 投机解码源码深度阅读

> 基于代码仓库 `/Users/jackiemac/workspace/rollout-infra/vllm-latest/` 的分析
> 分析日期: 2026-06-04

---

## 1. 整体架构概览

vLLM 的投机解码(Speculative Decoding)分为 **V1 架构**(当前主力)和旧架构。
本次分析聚焦 V1 实现。核心代码分布在以下目录:

```
vllm/v1/
  spec_decode/          # 各类 draft proposer 实现
    metadata.py         # SpecDecodeMetadata 数据结构
    metrics.py          # 投机解码性能统计
    utils.py            # Triton kernel 工具函数
    llm_base_proposer.py  # SpecDecodeBaseProposer 基类 (核心!)
    eagle.py            # EAGLE proposer
    medusa.py           # Medusa proposer
    draft_model.py      # 独立小模型 proposer
    ngram_proposer.py   # CPU N-gram proposer
    ngram_proposer_gpu.py  # GPU N-gram proposer
    dflash.py           # DFlash 并行提议
    suffix_decoding.py  # Suffix Decoding
    step3p5.py          # Step3.5 MTP
    gemma4.py           # Gemma4 MTP
    extract_hidden_states.py  # 隐藏状态提取
    custom_class_proposer.py  # 用户自定义 proposer
  sample/
    rejection_sampler.py  # V1 rejection sampler (非 GPU native)
  worker/
    gpu_model_runner.py    # 模型执行主流程,投机解码集成入口
    gpu/spec_decode/
      rejection_sampler.py      # GPU-native rejection sampler
      rejection_sampler_utils.py  # Triton 拒绝采样 kernel
      eagle/
        speculator.py   # EAGLE speculator 相关
      utils.py          # DraftTokensHandler
  core/sched/
    scheduler.py         # 调度器中投机 token 的调度与接受统计
```

---

## 2. V1 投机解码端到端流程

### 2.1 完整执行流水线

```
Scheduler.schedule()
    |
    |  scheduler_output.scheduled_spec_decode_tokens = {req_id: [tok1, tok2, ...]}
    |  (从上一步 draft 中获取的 token)
    v
GPUModelRunner.execute_model()
    |
    +-- _prepare_inputs()          # 准备输入,计算 logits_indices 和 spec_decode_metadata
    |     |_ _calc_spec_decode_metadata()  # 计算 draft token 索引, bonus token 索引
    |
    +-- _build_attention_metadata()  # 构建 attention metadata (含 spec decode 扩展)
    |
    +-- model_forward()             # 目标模型(target model)前向传播
    |     |_ 目标模型处理所有 token (包括 draft tokens)
    |     |_ 返回 hidden_states + logits
    |
    v
GPUModelRunner.sample_tokens()
    |
    +-- _sample(logits, spec_decode_metadata)
    |     |
    |     +-- [无投机] Sampler(logits)  # 正常采样
    |     |
    |     +-- [有投机] RejectionSampler(logits, draft_probs, ...)
    |           |
    |           +-- bonus logits 采样  # bonus token = 所有 draft 被接受后的额外 token
    |           +-- target logits 处理 (penalties, top_k, top_p)
    |           +-- rejection_sample()  # 核心: 拒绝采样
    |           |     |_ 对每个 draft token: 比较 p(x) vs q(x)
    |           |     |_ 接受的 token 保留,拒绝的 token 从残差分布重采样
    |           |     |_ 全部接受则追加 bonus token
    |           |
    |           v
    |         output_token_ids: [batch_size, max_spec_len + 1]
    |
    +-- propose_draft_token_ids()   # 为下一步生成新的 draft tokens
    |     |
    |     +-- [ngram]    NgramProposer.propose()
    |     +-- [medusa]   MedusaProposer.propose()
    |     +-- [eagle]    EagleProposer.propose()
    |     +-- [draft]    DraftModelProposer.propose()
    |     +-- [dflash]   DFlashProposer.propose()
    |     +-- [suffix]   SuffixDecodingProposer.propose()
    |     +-- [custom]   CustomClass.propose()
    |
    v
Scheduler.update_from_output()
    |
    +-- 统计接受率: num_accepted = len(generated) - 1
    +-- 更新 num_computed_tokens (减去被拒绝的 token 数)
    +-- update_draft_token_ids() (为下一步设置新的 spec_token_ids)
```

### 2.2 关键数据结构: SpecDecodeMetadata

```python
@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor       # [num_tokens] 所有 draft token id
    num_draft_tokens: list[int]         # [batch_size] 每个 request 的 draft token 数
    cu_num_draft_tokens: torch.Tensor   # [batch_size] 累积 draft token 数
    cu_num_sampled_tokens: torch.Tensor # [batch_size] 累积采样 token 数 (draft+1)
    target_logits_indices: torch.Tensor # [num_tokens] 目标模型需要验证的位置
    bonus_logits_indices: torch.Tensor  # [batch_size] bonus token 位置
    logits_indices: torch.Tensor        # [num_tokens + batch_size] 所有需要 logits 的位置
```

索引计算示例(来自 `_calc_spec_decode_metadata`):
```
假设:
  cu_num_scheduled_tokens: [4, 104, 107, 207, 209]
  num_draft_tokens:        [3,   0,   2,   0,   1]

计算结果:
  cu_num_draft_tokens:     [3,   3,   5,   5,   6]
  logits_indices:          [0,1,2,3,  103,104,  105,106,  206,207,208]
  target_logits_indices:   [0,1,2,  5,6,  9]       # 验证 draft 的位置
  bonus_logits_indices:    [3,  4,  7,  8,  10]     # bonus token 位置
```

---

## 3. Draft-Verify 流程图

```
+---------------------------------------------------------------+
|                    调度器 (Scheduler)                          |
|                                                               |
|  request.spec_token_ids = [d0, d1, d2, d3, d4]               |
|  (上一轮 draft proposer 生成的候选 token)                      |
|                                                               |
|  schedule(): 将 draft tokens 与真实 token 一起送入目标模型     |
+-----------------------------+---------------------------------+
                              |
                              v
+---------------------------------------------------------------+
|               GPUModelRunner.execute_model()                  |
|                                                               |
|  目标模型输入 = [prefill_tokens | decode_token | draft_tokens]|
|                                                               |
|  目标模型前向传播 --> 对所有位置计算 logits                     |
|                                                               |
|  logits_indices 标记:                                         |
|    - target_logits_indices: 需要验证 draft 的位置              |
|    - bonus_logits_indices: 需要采样 bonus token 的位置          |
+-----------------------------+---------------------------------+
                              |
                              v
+---------------------------------------------------------------+
|                   RejectionSampler                             |
|                                                               |
|  输入:                                                         |
|    - target_logits (来自目标模型, 所有验证位置的 logits)        |
|    - draft_token_ids (候选 token)                              |
|    - draft_probs (可选, draft 模型的概率分布)                   |
|    - bonus_token_ids (bonus token)                             |
|                                                               |
|  对每个 request 的每个 draft token:                            |
|                                                               |
|    position 0:  draft=d0  target_argmax=t0                     |
|      greedy: d0 == t0?  接受/拒绝                              |
|      random:  p(d0)/q(d0) >= u?  接受/拒绝                     |
|              |                                                  |
|              v (接受)                                           |
|    position 1:  draft=d1  target_argmax=t1                     |
|      d1 == t1?  接受/拒绝                                      |
|              |                                                  |
|              v (接受)                                           |
|    position 2:  draft=d2  target_argmax=t2                     |
|      d2 != t2  ==> 拒绝!                                       |
|      输出: t2 (从残差分布采样)                                   |
|              |                                                  |
|              v (停止,后续 draft 全部丢弃)                        |
|    position 3:  d3, d4 被丢弃                                   |
|                                                               |
|  最终输出: [d0, d1, t2]  (accepted + recovered)               |
+-----------------------------+---------------------------------+
                              |
                              v
+---------------------------------------------------------------+
|              propose_draft_token_ids()                         |
|                                                               |
|  使用采样后的 token 作为输入, 生成下一轮 draft tokens          |
|                                                               |
|  EAGLE: hidden_states + sampled_token --> autoregressive loop |
|  Ngram: 已生成 token 中的模式匹配                               |
|  Medusa: hidden_states --> 多头并行预测                        |
|  DraftModel: 独立小模型前向传播                                 |
|                                                               |
|  输出: [batch_size, num_speculative_tokens]                   |
+-----------------------------+---------------------------------+
                              |
                              v
+---------------------------------------------------------------+
|                   Scheduler.update_from_output()              |
|                                                               |
|  num_accepted = len(generated_tokens) - 1                     |
|  num_rejected = num_draft - num_accepted                      |
|  num_computed_tokens -= num_rejected                          |
|  request.spec_token_ids = new_draft_tokens                    |
+---------------------------------------------------------------+
```

---

## 4. Proposer 实现详解

### 4.1 SpecDecodeBaseProposer (基类)

文件: `vllm/v1/spec_decode/llm_base_proposer.py`

这是所有基于 LLM 的 proposer 的基类(EAGLE, DraftModel, MTP 等)。
Ngram 和 Medusa 有独立的实现路径。

```
SpecDecodeBaseProposer
  |
  +-- EagleProposer       (EAGLE/EAGLE3)
  +-- DraftModelProposer  (独立小模型)
  +-- DFlashProposer      (并行提议)
  +-- Gemma4Proposer      (Gemma4 MTP, 常量位置)
  +-- Step3p5MTPProposer  (Step3.5 MTP)
```

核心属性:
```python
class SpecDecodeBaseProposer:
    pass_hidden_states_to_model: bool   # 是否将目标模型的 hidden_states 传入 draft 模型
    parallel_drafting: bool             # 是否并行生成所有 draft token
    num_speculative_tokens: int         # 生成多少个 draft token
    constant_draft_positions: bool      # 是否所有 draft step 使用相同位置(Gemma4)
```

核心方法 `propose()` 的流程:
```
1. set_inputs_first_pass()
   - EAGLE: 将 input_ids 左移, 最后位置填入 next_token_id
   - DraftModel: 需要额外的 input slots (rejected token padding)
   - DFlash: 分离 context/query, 跨注意力机制

2. build_per_group_and_layer_attn_metadata()
   - 为 draft 模型的 attention 层构建 metadata

3. 第一次前向传播 (first pass)
   - 运行 draft 模型
   - 采样得到第一批 draft tokens

4. [num_speculative_tokens > 1 且非 parallel_drafting]
   自回归循环:
   for i in range(num_speculative_tokens - 1):
     - 更新 input_ids, positions, slot_mapping
     - 运行 draft 模型
     - 采样得到下一个 draft token

5. 返回 draft_token_ids: [batch_size, num_speculative_tokens]
```

### 4.2 EAGLE Proposer

文件: `vllm/v1/spec_decode/eagle.py`

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) 是当前最主流的投机解码方法之一。

```python
class EagleProposer(SpecDecodeBaseProposer):
    # pass_hidden_states_to_model = True
    # EAGLE 接收目标模型的 hidden_states 作为输入
```

特点:
- **输入**: 目标模型的 hidden_states + token_ids + positions
- **原理**: 利用目标模型最后一层的 hidden_states, 通过一个轻量级"头"来预测后续 token
- **自回归**: EAGLE 模型循环调用自己, 每步生成一个 draft token
- **KV Cache 共享**: draft 模型的 attention 层可以共享目标模型的 KV cache

EAGLE 输入处理的关键步骤(`set_inputs_first_pass`):
```
目标模型输出:     [a1, b1, b2, c1, c2, c3]  (hidden_states)
input_ids 左移:   [b1, b2, c1, c2, c3, c3]  (去掉第一个, 复制最后一个)
替换最后位置:     [b1, b2, c1, c2, c3, next]  (next = 采样得到的 token)
hidden_states:    [h_a1, h_b1, h_b2, h_c1, h_c2, h_c3]  (直接复制)
```

### 4.3 EAGLE3

EAGLE3 通过 `use_aux_hidden_state` 利用目标模型中间层的 hidden_states:
```python
# 在 propose() 中:
if self.method in ("eagle3", "dflash"):
    target_hidden_states = self.model.combine_hidden_states(
        target_hidden_states
    )
```
EAGLE3 将多个中间层的 hidden_states 拼接/融合, 提供更丰富的特征给 draft 模型。

### 4.4 Medusa Proposer

文件: `vllm/v1/spec_decode/medusa.py`

```python
class MedusaProposer:
    def propose(self, target_hidden_states, sampling_metadata):
        blocks = self.model(target_hidden_states)
        logits = self.model.compute_logits(blocks)
        # 每个头独立预测, argmax 取最大概率 token
        draft_tokens = torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)
        return draft_tokens  # [batch_size, num_heads]
```

特点:
- **多头并行**: 多个 MLP head 同时预测不同位置的 token
- **单次前向**: 不需要自回归循环, 一次前向生成所有 draft token
- **不需要 KV Cache**: Medusa head 是纯 MLP, 无 attention
- **输入**: 只需目标模型最后一个 token 的 hidden_state

### 4.5 DraftModel Proposer

文件: `vllm/v1/spec_decode/draft_model.py`

```python
class DraftModelProposer(SpecDecodeBaseProposer):
    # pass_hidden_states_to_model = False
    # 独立的完整小模型, 不使用目标模型的 hidden_states
```

特点:
- 使用一个完全独立的小模型(如 Llama-68M 作为 Llama-70B 的 draft)
- 需要 vocab_size 匹配
- 需要 tensor_parallel_size 匹配
- 自回归循环生成 draft tokens
- 有自己的 embedding 和 lm_head (不与目标模型共享)

### 4.6 N-gram Proposer (CPU)

文件: `vllm/v1/spec_decode/ngram_proposer.py`

```python
class NgramProposer:
    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu):
        # 在已生成的 token 中寻找最长匹配的 n-gram
        # 使用 Numba JIT 加速
        # 基于 KMP 算法(LPS 数组)的变体
```

特点:
- **零模型开销**: 不需要任何模型, 纯基于 token 匹配
- **算法**: 在已生成序列中寻找最长 n-gram 匹配, 提取后续 k 个 token 作为 draft
- **CPU 实现**: 使用 Numba `@njit(parallel=True)` 加速
- **参数**: `prompt_lookup_min`(最小 n-gram 长度), `prompt_lookup_max`(最大), `k`(draft token 数)

匹配算法示意:
```
已生成序列: [A, B, C, D, E, B, C, D, F, G]
当前后缀:   [B, C, D]  (n=3)
匹配位置:   [B, C, D, E] 中的 [B, C, D]
Draft:      [E]  (匹配位置后面的 token)
```

### 4.7 N-gram GPU Proposer

文件: `vllm/v1/spec_decode/ngram_proposer_gpu.py`

```python
class NgramProposerGPU:
    # 使用 PyTorch tensor 操作在 GPU 上并行匹配 n-gram
    # 支持 torch.compile 优化
    # kernel: NgramGPUKernel (nn.Module)
```

特点:
- 全 GPU 实现, 避免 CPU-GPU 同步
- 使用 `unfold` + 向量化比较进行滑动窗口匹配
- 异步 D2H 复制 valid draft counts
- 可配合 CUDA graph 使用

### 4.8 DFlash Proposer (并行提议)

文件: `vllm/v1/spec_decode/dflash.py`

```python
class DFlashProposer(SpecDecodeBaseProposer):
    # parallel_drafting = True
    # 所有 draft token 在一次前向传播中并行生成
```

特点:
- **并行起草**: 所有 speculative token 一次生成, 不需要自回归循环
- **跨注意力**: 使用非因果(non-causal)注意力, context KV 来自目标模型
- **上下文分离**: context (来自目标模型) 和 query (draft tokens) 分别处理
- 先预计算并存储 context KV, 然后只对 query tokens 运行模型

### 4.9 Suffix Decoding Proposer

文件: `vllm/v1/spec_decode/suffix_decoding.py`

```python
class SuffixDecodingProposer:
    # 基于后缀树(Suffix Tree)的 token 匹配
    # 使用 Arctic Inference 的 SuffixDecodingCache
    # 每个请求的 draft 长度可动态变化
```

### 4.10 其他 Proposer

- **Gemma4Proposer**: Gemma4 的 MTP(Multi-Token Prediction), `constant_draft_positions=True`
  所有 draft step 使用相同位置(与目标模型共享 KV cache)
- **Step3p5MTPProposer**: Step3.5 的 MTP, 支持 per-layer draft-step selection
- **ExtractHiddenStatesProposer**: 从模型中间层提取 hidden states 用于外部验证
- **CustomClassProposer**: 用户自定义 proposer, 通过 `speculative_config.model` 指定类路径

---

## 5. 拒绝采样算法详解

vLLM 有两套拒绝采样实现:

### 5.1 标准 Rejection Sampler (vllm/v1/sample/rejection_sampler.py)

用于 Ngram, Medusa, Suffix Decoding 等方法。

**Greedy 场景**:
```
对每个 draft position i:
  if draft_token[i] == target_argmax[i]:
    接受 draft_token[i]
  else:
    拒绝, 输出 target_argmax[i]
    停止后续验证

如果全部接受: 追加 bonus_token
```

**Random 采样场景** (标准拒绝采样):
```
对每个 draft position i:
  u = uniform(0, 1)  # 随机数
  if p(draft_token) / q(draft_token) >= u:
    接受 draft_token
  else:
    从残差分布 max(p(x) - q(x), 0) / sum(...) 采样
    停止后续验证

其中:
  p(x) = 目标模型在 x 上的概率
  q(x) = draft 模型在 x 上的概率
  (无 draft_probs 时 q(x) = 1, 即 one-hot)
```

**残差采样细节** (`sample_recovered_tokens_kernel`):
```python
# 对每个 token position, 遍历 vocabulary:
if NO_DRAFT_PROBS:
    # Ngram 模式: 只排除 draft_token, 从 target 分布采样
    prob = target_prob[vocab_offset]
    prob[vocab_offset == draft_token_id] = 0
else:
    # 标准: 残差 = max(target_prob - draft_prob, 0)
    prob = max(target_prob - draft_prob, 0)

# Gumbel-like 采样: argmax(prob * inv_q) 其中 inv_q ~ Exponential
score = prob * inv_q
recovered_id = argmax(score)
```

### 5.2 GPU-native Rejection Sampler (vllm/v1/worker/gpu/spec_decode/)

用于 EAGLE, DraftModel 等需要与 GPU 前向传播紧密集成的场景。

这个实现使用 Triton kernel, 分为三个阶段:

**阶段 1: _compute_block_stats_kernel**
```
对每个 logit 位置的每个 vocab block:
  - 计算局部 max 和 softmax 分母
  - 计算局部 argmax (用于 greedy)
  - 同时处理 target 和 draft logits
```

**阶段 2: _rejection_kernel**
```
对每个 request:
  accepted = True
  for i in range(num_tokens - 1):
    if accepted:
      if temperature == 0:
        # Greedy: 比较 draft 与 target argmax
        target_argmax = global_argmax(local_max_blocks)
        accepted &= (target_argmax == draft_sampled)
      else:
        # Random: 对数概率比测试
        target_log_prob = target_logit[draft_token] - target_lse
        draft_log_prob = draft_logit[draft_token] - draft_lse
        u = rand64(seed, pos)
        accepted &= target_log_prob > log(u) + draft_log_prob
```

**阶段 3: _resample_kernel**
```
对被拒绝的位置 (或 bonus position):
  if bonus:
    residual = target_logits  # 全部可用
  elif HAS_DRAFT_LOGITS:
    residual = log(max(p(x) - q(x), 0))
  else:  # one-hot draft
    residual = target_logits (排除 draft token)

  使用 Gumbel block argmax 采样:
  value, idx = gumbel_block_argmax(residual_logits, ...)
```

### 5.3 Synthetic 模式

两种 sampler 都支持 `synthetic` 拒绝采样模式:
```python
# 不比较概率, 而是用预定义的接受率
synthetic_acceptance_rates: [r0, r1, r2, ...]  # 递减

# 转换为条件概率
conditional_rate[i] = rate[i] / rate[i-1]

# 验证时:
u = uniform(0, 1)
accepted = (u < conditional_rate[pos])
```

---

## 6. 调度器集成

### 6.1 Draft Token 调度

在 `Scheduler._schedule()` 中:

```python
# 每个 request 有 spec_token_ids 属性
if request.spec_token_ids:
    num_scheduled_spec_tokens = min(
        len(request.spec_token_ids),
        # 受限于 KV cache 容量等
    )
    if num_scheduled_spec_tokens > 0:
        scheduled_spec_decode_tokens[req_id] = spec_token_ids[:num_scheduled]
    # 清空, 下次 update_draft_token_ids 会设置新的
    request.spec_token_ids = []
```

### 6.2 接受率统计

在 `Scheduler._process_model_output()` 中:

```python
scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
if scheduled_spec_token_ids and generated_token_ids:
    num_draft_tokens = len(scheduled_spec_token_ids)
    num_accepted = len(generated_token_ids) - 1  # 第一个是非投机 token
    num_rejected = num_draft_tokens - num_accepted

    # 回退 num_computed_tokens
    request.num_computed_tokens -= num_rejected

    # 统计指标
    spec_decoding_stats.observe_draft(num_draft_tokens, num_accepted)
```

### 6.3 异步投机解码

vLLM 支持 `use_async_spec_decode` 模式:
- 目标模型和 draft 模型可以异步执行
- EAGLE/DraftModel 可以直接使用 GPU 上的 sampled token, 不需要等 CPU bookkeeping
- Ngram 等需要 CPU token, 所以必须等 bookkeeping 完成后再 propose

---

## 7. 性能特征分析

### 7.1 什么时候投机解码有帮助

**高接受率场景** (spec decode 有利):
- **Greedy decoding**: 接受率通常很高 (draft 和 target 的 top-1 经常一致)
- **确定性高的任务**: 代码生成、格式化输出等
- **draft 模型与 target 模型分布接近**: 如同系列的小模型
- **长序列 decode**: 前缀匹配概率高 (Ngram 特别有效)
- **batch size 小时**: GPU 利用率低, 投机可以填充算力

**理论加速比**:
```
设接受率 = alpha, draft token 数 = K
期望接受长度 = 1 + alpha * K (含 bonus token)
加速比 ≈ (1 + alpha * K) / (1 + C_draft)

其中 C_draft 是 draft 模型相对于 target 模型的成本比

例如: alpha=0.8, K=5, C_draft=0.1
加速比 ≈ (1 + 4) / (1 + 0.1) = 4.5x
```

### 7.2 什么时候投机解码有害

**低接受率场景** (spec decode 不利):
- **高温度采样**: 随机性大, draft 和 target 容易不一致
- **分布不匹配**: draft 模型质量差, 大部分 draft 被拒绝
- **大 batch size**: GPU 已经满载, draft 模型的额外计算成为瓶颈
- **Prefill 阶段**: 大量新 token, draft 猜测几乎无用
- **词汇表差异大**: 不同 tokenizer/不同模型架构

**额外开销来源**:
1. Draft 模型前向传播 (EAGLE: 每步一个 forward; 自回归 K 步)
2. Rejection sampling 本身的计算
3. KV cache 扩展 (需要为 draft token 分配空间)
4. CPU-GPU 同步 (某些方法需要)
5. 调度器需要处理 draft token 的回退

### 7.3 各方法性能对比

| 方法 | Draft 开销 | 接受率 | 适用场景 | 额外显存 |
|------|-----------|--------|---------|---------|
| EAGLE | 低(单层头) | 高 | 通用, greedy | 中(hidden_states) |
| EAGLE3 | 中(多层) | 很高 | 通用 | 高(多层 hidden) |
| Medusa | 最低(单次) | 中 | 长序列 | 低(MLP 头) |
| DraftModel | 高(完整模型) | 中 | 架构匹配 | 高(独立模型) |
| Ngram CPU | 零 | 低~中 | 重复性文本 | 零 |
| Ngram GPU | 零 | 低~中 | 重复性文本 | 极低 |
| DFlash | 中 | 高 | 支持 GPU | 中 |
| Suffix | 零 | 中 | 重复后缀 | 低 |
| MTP | 中 | 高 | 特定模型 | 中 |

### 7.4 关键优化技巧

1. **Padded Drafter Batch**: EAGLE/DraftModel 使用 GPU padded tensor 避免同步
2. **CUDA Graph**: Draft 模型可以捕获 CUDA graph 加速
3. **KV Cache 共享**: EAGLE 的 attention 层共享目标模型的 KV cache
4. **Embedding/LM Head 共享**: MTP 模型共享目标模型的 embedding 和 lm_head
5. **Local Argmax Reduction**: TP 场景下用 O(2*tp_size) 通信替代 O(vocab_size)
6. **Parallel Drafting**: DFlash 一次生成所有 token, 消除自回归开销
7. **异步 D2H 复制**: Ngram GPU 使用专用 stream 异步复制结果

---

## 8. 配置示例

```python
# EAGLE
--speculative-model "yuhuili/EAGLE-LLaMA3-8B" \
--num-speculative-tokens 5

# EAGLE3
--speculative-model "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B" \
--speculative-method eagle3 \
--num-speculative-tokens 5

# Medusa
--speculative-model "vllm/Medusa-FastChat-Llama3-8B" \
--speculative-method medusa \
--num-speculative-tokens 5

# Draft Model
--speculative-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
--speculative-method draft_model \
--num-speculative-tokens 5

# N-gram (CPU)
--speculative-method ngram \
--num-speculative-tokens 5 \
--prompt-lookup-min 3 \
--prompt-lookup-max 5

# N-gram (GPU)
--speculative-method ngram_gpu \
--num-speculative-tokens 5 \
--prompt-lookup-min 3 \
--prompt-lookup-max 5

# DFlash
--speculative-model "Qwen/Qwen3-DFlash-0.5B" \
--speculative-method dflash \
--num-speculative-tokens 5

# Suffix Decoding
--speculative-method suffix \
--num-speculative-tokens 5
```

---

## 9. 核心文件索引

| 文件路径 | 作用 |
|---------|------|
| `vllm/v1/spec_decode/llm_base_proposer.py` | LLM-based proposer 基类, `propose()` 核心循环 |
| `vllm/v1/spec_decode/eagle.py` | EAGLE proposer |
| `vllm/v1/spec_decode/medusa.py` | Medusa proposer |
| `vllm/v1/spec_decode/draft_model.py` | 独立小模型 proposer |
| `vllm/v1/spec_decode/ngram_proposer.py` | CPU N-gram (Numba) |
| `vllm/v1/spec_decode/ngram_proposer_gpu.py` | GPU N-gram (torch.compile) |
| `vllm/v1/spec_decode/dflash.py` | DFlash 并行提议 |
| `vllm/v1/spec_decode/metadata.py` | SpecDecodeMetadata 数据结构 |
| `vllm/v1/spec_decode/metrics.py` | 投机解码统计和 Prometheus 指标 |
| `vllm/v1/spec_decode/utils.py` | Triton kernels (slot mapping, input expansion) |
| `vllm/v1/sample/rejection_sampler.py` | 标准 rejection sampler + Triton kernels |
| `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` | GPU-native rejection sampler |
| `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` | GPU-native 拒绝采样 Triton kernels |
| `vllm/v1/worker/gpu_model_runner.py` | 模型执行主流程, spec decode 集成 |
| `vllm/v1/core/sched/scheduler.py` | 调度器, draft token 调度和接受统计 |
| `vllm/config/speculative.py` | SpeculativeConfig 配置类 |

---

## 10. 总结

vLLM V1 的投机解码实现是一个**高度模块化、GPU 友好**的系统:

1. **统一框架**: 所有 proposer 通过 `propose()` 接口接入, model_runner 统一调度
2. **两套 rejection sampler**: 标准版(用于 CPU-based proposer)和 GPU-native 版(用于 EAGLE/DraftModel)
3. **丰富的策略**: 从零开销的 Ngram 到高准确率的 EAGLE3, 覆盖不同场景
4. **性能优化**: CUDA graph, KV cache 共享, 异步 D2H, padded batch, local argmax reduction
5. **异步调度**: 支持目标模型和 draft 模型重叠执行, 最大化 GPU 利用率
