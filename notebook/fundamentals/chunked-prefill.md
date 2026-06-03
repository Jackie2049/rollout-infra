# Chunked Prefill 深度解析

> 将长 prompt 的 Prefill 拆分为多个 chunk，与 Decode 交替执行，消除 Prefill 阻塞

## 1. 问题：长 Prefill 阻塞 Decode

传统 LLM serving 中，一个长 prompt 的 prefill 必须一口气完成。问题：

```
请求 A (128K prompt) 正在 prefill...
  → 需要计算 128K × hidden_dim 的 attention (O(N²) 复杂度)
  → 期间所有 decode 请求被阻塞
  → 等待 13+ 秒，decode 延迟暴涨
```

**核心矛盾**:
- Prefill: compute-bound (O(N²) attention)，单次计算量大
- Decode: memory-bound，单次极快 (~10ms)，但需要高频执行
- 两者共享同一个 GPU，长 prefill 会"饿死" decode

## 2. Chunked Prefill 方案

### 2.1 核心思想

将长 prompt 的 prefill 拆分为固定大小的 chunk，每个 scheduling step 处理一个 chunk：

```
Step 1: [Chunk 1 of Req-A (2K tokens)] + [Decode Req-B, C, D]
Step 2: [Chunk 2 of Req-A (2K tokens)] + [Decode Req-B, C, D]
...
Step N: [Last chunk of Req-A] + [Decode Req-B, C, D]
Step N+1: [Decode Req-A (now generating)] + [Decode Req-B, C, D]
```

**效果**:
- 每个 step 的 prefill 计算量受控 (chunk_size 固定)
- Decode 请求在每个 step 都能获得服务
- TTFT 略增 (总 prefill 时间不变，但穿插了 decode 开销)
- 但 **decode 延迟稳定**，不受长 prefill 影响

### 2.2 定量对比 (128K prompt, LLaMA-7B, A100)

| 方案 | TTFT | Decode 延迟 | 最大 batch |
|------|------|------------|-----------|
| 无 chunking | 13s (一次性 prefill) | 13s 内完全阻塞 | 1 |
| chunk=2K | 13.2s (+1.5%) | 每步 ~10ms | 32+ |
| chunk=4K | 13.1s (+0.8%) | 每步 ~10ms | 16+ |
| chunk=512 | 13.5s (+3.8%) | 每步 ~8ms | 64+ |

TTFT 略增因为 decode 请求消耗了部分 token budget。但 **decode 延迟从 13s 阻塞降到 ~10ms**。

## 3. vLLM V1 实现

### 3.1 统一调度模型

vLLM V1 的调度器**不区分 prefill 和 decode 阶段**。核心只有一个计数器：

```python
# 每个请求维护
num_computed_tokens  # 已处理的 token 数

# 每个 scheduling step
num_new_tokens = num_tokens_with_spec - num_computed_tokens
# 如果 > 0 → 需要处理更多 token (可能是 prefill chunk)
# 如果 = 0 → 当前 step 只需 decode 1 个 token
```

Prefill、Chunked Prefill、Decode 本质上都是 "分配 token 预算" 的过程。

### 3.2 Chunk Size 三层约束

```
                     原始需求
                        │
          num_tokens_with_spec - num_computed_tokens
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
     long_prefill    token     剩余
     _token_threshold budget   KV blocks
     (chunk 上限)     (全局)    (内存)
            │           │           │
            └───────────┼───────────┘
                        ▼
              min(三者) = 本 step 的 chunk size
```

**具体代码逻辑** (`scheduler.py`):

```python
# 约束 1: long_prefill_token_threshold 限制单个 chunk 大小
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold

# 约束 2: 全局 token budget (所有请求共享)
num_new_tokens = min(num_new_tokens, token_budget)

# 约束 3: 如果关闭 chunked prefill 且放不下，直接停止调度
if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > token_budget:
    break  # 不接受任何新请求
```

### 3.3 Scheduling 两轮处理

```
schedule() 每步:
  ├── 第一轮: RUNNING 请求 (正在进行的 decode + 正在进行的 chunked prefill)
  │   ├── decode 请求: num_new_tokens = 1
  │   ├── chunked prefill 继续: num_new_tokens = min(剩余, threshold, budget)
  │   └── KV block 不够 → 抢占 (preempt)
  │
  └── 第二轮: WAITING 请求 (新到达的请求)
      ├── 计算 prefix cache hits
      ├── 分配 chunk 大小
      └── 移入 RUNNING 队列
```

**关键**: Decode 请求**优先于新 prefill 请求**，因为 RUNNING 先于 WAITING 处理。

### 3.4 Forward Pass 处理

在 ModelRunner 中，partial prefill 的处理非常优雅：

```python
# 1. 判断哪些请求仍在 prefill
is_prefilling = num_computed_tokens < num_prompt_tokens

# 2. 标记 partial prefill (丢弃采样结果)
discard_request_mask = (num_computed_tokens + num_scheduled_tokens < num_total_tokens)

# 3. 计算 logits (包含 partial prefill 的 "无用" logits)
logits_indices = query_start_loc[1:] - 1  # 每个 request 最后一个 token

# 4. 丢弃 partial prefill 的采样结果
for i in discard_request_indices:
    gen.set_offset(gen.get_offset() - 4)  # 回退 RNG 状态 (保持确定性)
    valid_sampled_token_ids[i].clear()
```

**设计选择**: 即使 partial prefill 的 logits 会被丢弃，仍然计算它们。原因是简化实现——所有请求走同一条 forward path，不需要分支。

### 3.5 进度追踪

```python
# _update_after_schedule()
request.num_computed_tokens += num_scheduled_token
request.is_prefill_chunk = (request.num_computed_tokens < request.num_tokens)

# 当 num_computed_tokens >= num_prompt_tokens 时:
# → 自动转为 decode 阶段
# → 下一个 step 只 schedule 1 个 token
```

### 3.6 内存保护: full_sequence_must_fit

```python
# scheduler_reserve_full_isl (默认 True)
# 在分配 KV blocks 时，检查完整序列长度是否能放下
# 而不是只检查当前 chunk 能否放下
# 防止过度接纳导致后续 KV cache thrash
```

这个设计非常重要：即使一个 chunked prefill 的第一个 chunk 放得下，如果完整序列放不下，就不应该开始。

## 4. Chunked Prefill 与其他优化组合

### 4.1 + Prefix Caching

```
请求 A: prompt = [system_prompt | user_query | document]
请求 B: prompt = [system_prompt | user_query | document2]

Prefix Caching 命中: system_prompt + user_query 的 KV blocks
→ Chunked Prefill 只需处理 document 部分
→ num_computed_tokens 起始值 = prefix_length (非零)
```

### 4.2 + Speculative Decoding

```
Step 1: [Chunk of Req-A] + [Draft+Verify for Req-B]
→ token_budget 被 chunk 和 draft tokens 共享
```

### 4.3 + Sliding Window Attention

```
Sliding Window = 4K tokens
→ Chunked Prefill 的 chunk 可以被丢弃 (超过窗口)
→ 但需要注意: 已计算的 chunk 在窗口外时会被淘汰
```

## 5. 调优参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_chunked_prefill` | True | 是否启用 |
| `max_num_batched_tokens` | model_max_len | 每 step 最大 token 数 |
| `long_prefill_token_threshold` | max_model_len × 4% | 单 chunk 最大 token 数 |
| `max_num_partial_prefills` | 1 (model_len < 32K) / >1 | 允许同时进行的 partial prefill 数 |
| `scheduler_reserve_full_isl` | True | 是否预检查完整序列能否放下 |

## 6. 最佳实践

### 6.1 Chunk Size 选择

- **小 chunk (512-1K)**: 适合高并发 decode，延迟最平滑
- **中 chunk (2K-4K)**: 平衡 TTFT 和 decode 延迟，推荐默认
- **大 chunk (8K+)**: 接近无 chunking，适合低并发场景

### 6.2 与 P/D 分离对比

| 特性 | Chunked Prefill | P/D 分离 |
|------|----------------|---------|
| 硬件 | 单 GPU 即可 | 需要至少 2 GPU |
| TTFT | 略增 (+1-5%) | 最低 (专用 prefill GPU) |
| Decode 延迟 | 稳定 (~10ms) | 最低 (无 prefill 干扰) |
| 复杂度 | 低 (调度器内处理) | 高 (KV Transfer + 两套 GPU) |
| 成本 | 低 | 高 (需要更多 GPU) |

### 6.3 场景推荐

- **通用对话 (4K-8K prompt)**: Chunked Prefill 即可，无需 P/D 分离
- **RAG / 长文档 (>32K prompt)**: Chunked Prefill + Prefix Caching
- **大规模部署 (>1000 并发)**: P/D 分离 + Chunked Prefill on Prefill GPU

## 参考资料

- vLLM V1 Scheduler: `vllm/v1/core/sched/scheduler.py`
- GPU ModelRunner: `vllm/v1/worker/gpu_model_runner.py`
- Scheduler Config: `vllm/config.py` (SchedulerConfig)
- 相关笔记: [Continuous Batching](continuous-batching.md), [Long Context Serving](long-context-serving.md)
- 相关工具: `tools/long_context_serving_sim.py`
