# vLLM V1 Scheduler 源码分析

> 文件: `vllm/v1/core/sched/scheduler.py` (~2374行)
> 分析日期: 2026-06-04

## 核心理念

V1 Scheduler 最大的创新：**取消 "prefill phase" 和 "decode phase" 的区分**。

```python
# 每个 request 只需维护:
request.num_computed_tokens  # 已经计算过的 token 数
request.num_tokens_with_spec # 目标 token 数 (prompt + output + spec)

# Scheduler 每步尝试分配 token，让 num_computed_tokens 追赶 num_tokens_with_spec
num_new_tokens = num_tokens_with_spec - num_computed_tokens
```

这一设计统一处理了：
- **Chunked Prefill**: 长 prompt 分多步处理 (`long_prefill_token_threshold`)
- **Prefix Caching**: `num_computed_tokens` 自动从缓存 tokens 开始
- **Speculative Decoding**: spec tokens 计入 `num_tokens_with_spec`
- **Jump Decoding**: 未来一次跳多个 decode step

## schedule() 算法

```
schedule():
  1. Running requests (FCFS, 先来先服务)
     while running 非空 and token_budget > 0:
       num_new_tokens = min(tokens_needed, token_budget,
                            long_prefill_threshold, max_model_len)
       尝试 allocate_slots(num_new_tokens)
       if 分配失败:
         preempt 最低优先级请求 → 释放 block → 重试
       if 分配成功:
         加入 scheduled_running_reqs, token_budget -= num_new_tokens

  2. Waiting requests (新请求)
     while waiting 非空 and token_budget > 0 and running < max:
       new_computed_blocks = get_computed_blocks()  # prefix caching
       num_computed_tokens = local_cached + external_cached  # KV transfer
       尝试 allocate_slots(remaining_tokens)
       if 分配失败:
         skip 或 preempt running
       if 分配成功:
         加入 scheduled_running_reqs, running.append(request)

  3. 构建 SchedulerOutput
     返回给 EngineCore → ModelRunner.execute_model()
```

## 抢占策略

| 策略 | 行为 | 使用场景 |
|------|------|---------|
| `FCFS` | preempt 最后加入的 running request | 默认，简单 |
| `PRIORITY` | preempt 优先级最高的 (priority, arrival_time) | 需要 QoS 区分 |

preempt 时: 释放 KV Cache blocks → 恢复 token budget → 请求回到 waiting 队列。

## Token 预算管理

```python
token_budget = self.max_num_scheduled_tokens  # 每步可分配的总 token 数
long_prefill_token_threshold                 # 单请求单步最大 token 数 (chunked prefill)
```

**Chunked Prefill 的实现**:
```python
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold  # 切分长 prompt
```

## Prefix Caching 集成

```python
# 新请求调度时:
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
# 自动跳过已缓存的 tokens，num_computed_tokens 从缓存位置开始
```

外部 KV（P/D 分离）:
```python
ext_tokens = self.connector.get_num_new_matched_tokens(request, local_tokens)
num_computed_tokens = local_cached + external_cached
```

## 约束条件

| 约束 | 说明 |
|------|------|
| `max_num_running_reqs` | 最大并发请求数 |
| `max_num_scheduled_tokens` | 每步最大 token 预算 |
| `max_model_len` | 序列最大长度 |
| `max_loras` | 最大并发 LoRA 适配器数 |
| `long_prefill_token_threshold` | Chunked prefill 单步限制 |
| `encoder_compute_budget` | 多模态编码器输入预算 |
| `next_decode_eligible_step` | PP>1 时的 decode 步数间隔 |

## SchedulerInterface 抽象

```python
class SchedulerInterface(ABC):
    def schedule() -> SchedulerOutput        # 核心调度
    def update_from_output()                  # 更新状态
    def add_request(request)                  # 添加新请求
    def finish_requests(req_ids)              # 标记完成
```

支持其他调度策略（如 RoundRobin、EDF 等）通过继承扩展。

## 与 ModelRunner 的交互

```
Scheduler.schedule()
  → SchedulerOutput {
      scheduled_new_reqs,
      scheduled_running_reqs,
      num_scheduled_tokens: dict[req_id, int],
      scheduled_spec_decode_tokens,
      scheduled_encoder_inputs,
      ...
    }
    → GPUModelRunner.execute_model(scheduler_output)
      → 准备 attn_metadata, CUDA Graph, UBatch
        → model.forward()
          → Scheduler.update_from_output(model_runner_output)
            → 更新 num_computed_tokens, 添加 output tokens
```

## 代码位置速查

| 行号 | 内容 |
|------|------|
| L65- | `Scheduler.__init__` |
| L339- | `schedule()` 核心调度算法 |
| L377-516 | Running requests 调度循环 |
| L562- | Waiting/new requests 调度 |
| L1310- | `update_from_output()` |
| L1782- | `add_request()` |
| L1806- | `finish_requests()` |
