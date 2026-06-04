# vLLM V1 Sampling Pipeline 源码分析

> 文件路径: `vllm/v1/sample/`
> 分析日期: 2026-06-04

## 完整采样流程 (9 步)

```
Logits → Logprobs计算 → FP32转换 → Logits Processors → 采样
→ Token转换 → Logprob收集 → 输出封装
```

### Step 1: Logprobs 计算 (sampler.py L25-30)
- `logprobs_mode=raw_logprobs`: 用 `log_softmax` 计算
- `logprobs_mode=raw_logits`: 直接 clone logits

### Step 2: FP32 转换 (L91)
- 所有后续操作在 float32 上进行 (数值稳定性)

### Step 3: Logits Processors (L93-95, L391-415)
- 两个阶段:
  1. **Non-argmax invariant**: 可能改变 greedy 结果
     - Allowed token IDs 白名单
     - Bad words 排除
     - Min tokens 处理器
     - Logit bias
  2. **Argmax invariant**: 不影响 greedy 结果
     - Temperature
     - Min-P 过滤
     - Top-K / Top-P 过滤

### Step 4: 采样 (L97, L238-297)
- Greedy: `argmax` (temp < 1e-5)
- Random: multinomial sampling

### Step 5-6: Token 转换 + Logprob 收集 (L104-132)

### Step 7: 输出封装 (L137-144)
- 返回 `SamplerOutput` (sampled tokens + logprobs)

## 采样参数实现

### Temperature (L223-232)
```python
logits.div_(temp.unsqueeze(dim=1))  # in-place
```
- Greedy (temp < 1e-5) 特殊处理

### Top-K / Top-P (L280-286)
- `TopKTopPSampler` 类
- **多后端**: FlashInfer (最快) > Triton > PyTorch > ROCm
- FlashInfer 无需 per-request generator → 最快

### Min-P (builtin.py L101-115)
1. logits → probs
2. 找 max_prob
3. mask: `prob < max_prob * min_p`
4. Argmax invariant: 不影响 greedy

### Penalties (L418-434)
- **Repetition**: 惩罚已出现 token
- **Frequency**: 按频率惩罚
- **Presence**: 按是否出现惩罚 (二值)

## Batch 处理

### BatchUpdate (interface.py L37-58)
- `removed`: 移除的请求
- `added`: 新增的请求
- `moved`: 重排的请求

### Per-Request 状态
- 每个请求独立: temperature, top_k/p, penalties, logit_bias
- 单次 batched 操作处理所有请求

### 优化策略
- Pinned memory: CPU→GPU 高效传输
- Non-blocking transfers: 异步数据移动
- Sparse updates: 只更新需要的部分
- Compiled kernels: CPU 采样编译优化

## Speculative Decoding 支持
- Bonus tokens 特殊处理 (L380-388)
- Base outputs + speculative tokens 合并
- State 更新考虑 draft tokens

## 关键设计原则

1. **正确性优先**: Non-argmax invariant 先于 argmax invariant
2. **批量效率**: 异构参数在单次 batch 操作中处理
3. **多后端**: FlashInfer > Triton > PyTorch fallback
4. **Speculative 兼容**: 采样管线原生支持 speculative decoding
