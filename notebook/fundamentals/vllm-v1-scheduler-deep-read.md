# vLLM V1 Scheduler 深度阅读

> 2026-06-07 | vllm/v1/core/sched/scheduler.py (~2400 行)

## 核心调度算法

### schedule() — 无"prefill phase"或"decode phase"

vLLM V1 scheduler 没有传统的 "prefill phase" 和 "decode phase" 分离!
每个请求只有 `num_computed_tokens` 和 `num_tokens_with_spec`.
调度器每步尝试为所有请求分配 token, 使 `num_computed_tokens` → `num_tokens_with_spec`.

这统一了 chunked prefill, prefix caching, speculative decoding.

### schedule() 流程图

```
schedule()
├── 1. RUNNING requests (decode + chunked prefill continuation)
│   ├── 计算 num_new_tokens = num_tokens_with_spec - num_computed_tokens
│   ├── long_prefill_token_threshold 限制 chunk 大小
│   ├── allocate_slots() → 如无空间则 preempt
│   │   ├── PRIORITY policy: preempt 最低优先级 request
│   │   ├── FCFS policy: preempt 最新 request (pop())
│   │   └── 如果 preempted == current request → break (无法调度)
│   └── 加入 scheduled_running_reqs
│
├── 2. WAITING requests (新prefill) — 仅在无 preempt 时
│   ├── 检查 len(running) < max_num_running_reqs
│   ├── get_computed_blocks() → prefix cache hit
│   ├── connector.get_num_new_matched_tokens() → external KV hit
│   ├── 计算 num_new_tokens = total - computed (local + external)
│   ├── chunked_prefill 限制
│   ├── allocate_slots() → 如无空间则 break (不 preempt)
│   ├── LoRA max_loras 检查
│   ├── 加入 running, scheduled_new_reqs
│   └── KV Connector: load_kv_async → WAITING_FOR_REMOTE_KVS
│
├── 3. skipped_waiting 处理 (blocked requests)
│   ├── WAITING_FOR_REMOTE_KVS: 等 KV transfer 完成
│   ├── 其他 blocked status → try promote
│   └── prepend to self.skipped_waiting
│
├── 4. get_num_common_prefix_blocks() → cascade attention
│
├── 5. 构建 SchedulerOutput
│   ├── scheduled_new_reqs, scheduled_cached_reqs
│   ├── num_scheduled_tokens, total_num_scheduled_tokens
│   ├── num_common_prefix_blocks (cascade)
│   ├── preempted_req_ids, finished_req_ids
│   └── kv_connector_metadata (如果使用)
│
└── 6. _update_after_schedule()
    ├── request.num_computed_tokens += num_scheduled_tokens
    ├── request.is_prefill_chunk 标记
    ├── 清空 finished_req_ids
```

## 关键设计决策

### 1. RUNNING 优先, WAITING 次之

```python
# First, schedule the RUNNING requests.
while req_index < len(self.running) and token_budget > 0:
    ...

# Next, schedule the WAITING requests.
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
    while (self.waiting or self.skipped_waiting) and token_budget > 0:
        ...
```

**RUNNING 请求优先分配 token_budget** → decode 请求不被 prefill 阻塞!
这是 V1 对 V0 的关键改进 — V0 有 prefill phase, 会导致 decode 延迟飙升.

### 2. Preemption 策略: RUNNING → WAITING

当 KV cache 不足时:
- **PRIORITY policy**: preempt `max(running, key=priority)` → 最低优先级先被抢占
- **FCFS policy**: preempt `running.pop()` → 最新请求被抢占

被 preempt 的请求:
- `kv_cache_manager.free(request)` → 释放所有 KV blocks
- `request.num_computed_tokens = 0` → 需要从头 recompute
- 加入 `self.waiting.prepend_request()` → 重新排队

**关键**: Preemption 只发生在 RUNNING request 调度失败时!
WAITING request 调度失败 → 直接 break (不 preempt running requests).

这意味着 **prefill 请求不会抢占 decode 请求的 KV cache**.

### 3. Token Budget 管理

```python
token_budget = self.max_num_scheduled_tokens  # 通常 8192-32768

# RUNNING: 先分配
token_budget -= num_new_tokens

# WAITING: 剩余 budget 分配
num_new_tokens = min(num_new_tokens, token_budget)
```

**Budget 是全局共享的** — RUNNING + WAITING 共用同一 budget.
RUNNING 消耗完后 WAITING 无法分配 → 但 RUNNING 通常只消耗 1 token/request.

### 4. Chunked Prefill

```python
threshold = self.scheduler_config.long_prefill_token_threshold
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold
```

长 prompt (如 8K tokens) 被分成多个 chunk, 每个 chunk ≤ threshold.
这避免了单个长 prompt 占满 budget, 阻塞所有 decode 请求.

### 5. Prefix Caching (in scheduler)

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

新请求进入 WAITING 时, 检查本地 KV cache 有多少 prefix 已经缓存.
如果命中, `num_new_local_computed_tokens` > 0 → 减少 `num_new_tokens`.

### 6. KV Connector (P/D Disaggregation)

```python
if self.connector is not None:
    ext_tokens, load_kv_async = (
        self.connector.get_num_new_matched_tokens(request, ...)
    )
```

检查远程 KV cache (Prefill 节点) 是否有匹配的 prefix.
如果有:
- `load_kv_async=True` → 设置 `WAITING_FOR_REMOTE_KVS` 状态
- 不计算新 token, 只等 KV transfer 完成

### 7. Cascade Attention 触发

```python
num_common_prefix_blocks = (
    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
)
```

在 schedule 结束时计算所有 running 请求的最长公共 prefix blocks 数.
这个值传给 ModelRunner → 可能触发 cascade attention (如果 prefix≥256, requests≥8).

### 8. LoRA 限制

```python
if self.lora_config and request.lora_request:
    if len(scheduled_loras) == self.lora_config.max_loras:
        and request.lora_request.lora_int_id not in scheduled_loras:
        # Scheduling would exceed max_loras, skip.
```

LoRA 有 `max_loras` 限制 → 如果活跃 LoRA 数已达上限, 新请求必须等.

## update_from_output() — 处理模型输出

```python
def update_from_output(self, scheduler_output, model_runner_output):
    # 对每个 scheduled request:
    for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
        # 1. 获取 sampled token IDs
        generated_token_ids = sampled_token_ids[req_index]

        # 2. Spec decode: 调整 num_computed_tokens
        if scheduled_spec_token_ids and generated_token_ids:
            num_rejected = num_draft_tokens - num_accepted
            request.num_computed_tokens -= num_rejected

        # 3. 更新 request
        new_token_ids, stopped = _update_request_with_output(request, new_token_ids)

        # 4. Structured output: grammar.advance()
        if self.structured_output_manager.should_advance(request):
            struct_output_request.grammar.accept_tokens(req_id, new_token_ids)

        # 5. 如果 stopped → free request
        if stopped:
            finished = self._handle_stopped_request(request)
            if finished:
                kv_transfer_params = self._free_request(request)

        # 6. 构建 EngineCoreOutput
```

### Request 状态转换

```
WAITING → RUNNING (schedule)
RUNNING → PREEMPTED (preempt)
PREEMPTED → WAITING (prepend_request)
WAITING_FOR_REMOTE_KVS → WAITING (KV transfer complete)
RUNNING → FINISHED_STOPPED/FINISHED_LENGTH/etc
```

## 与 SGLang Scheduler 对比

| | vLLM V1 | SGLang |
|---|---------|--------|
| 算法 | unified token budget | separate prefill/decode |
| RUNNING优先 | YES (decode不被阻塞) | YES (但merge-based) |
| Preemption | preempt running→waiting | evict & recompute |
| Chunked prefill | yes (threshold) | yes (chunked) |
| Prefix cache | BlockHashToBlockMap | RadixTree |
| KV Connector | NIXL/FlexKV/etc | Mooncake |
| LoRA limit | max_loras check | Punica batched matmul |
| Cascade attn | num_common_prefix_blocks | no |
| Spec decode | integrated in schedule | separate proposer |
| Policy | FCFS or PRIORITY | FCFS |
| Budget | max_num_scheduled_tokens | max_running_req + tokens |

**vLLM unified approach 更灵活** — 同一步可同时处理 prefill 和 decode.
SGLang 的 RadixAttention 更灵活 (无 block 对齐限制), 但调度更简单.

## 关键代码路径总结

1. **新请求**: add_request() → WAITING → schedule() → get_computed_blocks → allocate_slots → RUNNING
2. **Decode**: RUNNING → schedule() → 1 token → execute → sample → update → RUNNING (循环)
3. **Chunked prefill**: WAITING → schedule(threshold tokens) → RUNNING → ... → 直到所有 prompt tokens 完成
4. **Preemption**: RUNNING → allocate_slots失败 → preempt → PREEMPTED → WAITING → recompute
5. **KV Transfer**: WAITING → schedule → WAITING_FOR_REMOTE_KVS → transfer完成 → WAITING → RUNNING
6. **Completion**: RUNNING → update_from_output → stopped → FINISHED → free