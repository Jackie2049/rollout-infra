# vLLM V1 Scheduler 深度源码精读 (2026-06-05 更新)

> 源码路径: `vllm/v1/core/sched/scheduler.py` (~2375 行)
> 基于 vLLM latest main 分支
> 重点文件:
> - `scheduler.py` -- 核心 Scheduler 类 (~2375 行)
> - `async_scheduler.py` -- AsyncScheduler 子类 (~68 行)
> - `request_queue.py` -- FCFS/Priority 队列实现 (~209 行)
> - `output.py` -- SchedulerOutput 数据结构 (~264 行)
> - `interface.py` -- SchedulerInterface 抽象接口 (~245 行)
> - `utils.py` -- check_stop/remove_all 工具函数 (~131 行)
> - `request.py` -- Request/RequestStatus 数据模型 (~362 行)

---

## 目录

1. [类结构与初始化](#1-类结构与初始化)
2. [schedule() 核心方法逐行精读](#2-schedule-核心方法逐行精读)
3. [调度策略 FCFS vs PRIORITY](#3-调度策略-fcfs-vs-priority)
4. [Prefill 调度详解](#4-prefill-调度详解)
5. [Decode 调度详解](#5-decode-调度详解)
6. [Preemption 抢占机制](#6-preemption-抢占机制)
7. [Prefix Caching 集成](#7-prefix-caching-集成)
8. [Chunked Prefill 机制](#8-chunked-prefill-机制)
9. [Batch 管理与约束](#9-batch-管理与约束)
10. [update_from_output 输出处理](#10-update_from_output-输出处理)
11. [AsyncScheduler 异步调度扩展](#11-asyncscheduler-异步调度扩展)
12. [KV Connector 与 P/D 分离](#12-kv-connector-与-pd-分离)
13. [边缘场景与错误处理](#13-边缘场景与错误处理)
14. [数据流总图](#14-数据流总图)

---

## 1. 类结构与初始化

### 1.1 类继承关系

```
SchedulerInterface (ABC)
    |
    +-- Scheduler
         |
         +-- AsyncScheduler
```

`SchedulerInterface` 定义了调度器必须实现的抽象接口:
- `schedule()` -> SchedulerOutput
- `update_from_output()` -> dict[int, EngineCoreOutputs]
- `add_request()` / `finish_requests()`
- `get_num_unfinished_requests()` / `has_finished_requests()`
- `reset_prefix_cache()` / `reset_encoder_cache()`
- `update_draft_token_ids()` / `update_draft_token_ids_in_output()`

### 1.2 Scheduler.__init__ 关键属性

```
文件: scheduler.py:66-288
```

```python
class Scheduler(SchedulerInterface):
    def __init__(self, vllm_config, kv_cache_config, ...):
```

**调度约束** (scheduler.py:103-113):
```python
self.max_num_running_reqs = scheduler_config.max_num_seqs          # 默认 128
self.max_num_scheduled_tokens = scheduler_config.max_num_scheduled_tokens  # 默认等于 max_num_batched_tokens (2048)
self.max_model_len = model_config.max_model_len
```

**三个核心队列** (scheduler.py:157-168):
```python
self.waiting: RequestQueue        # 等待调度的新请求
self.skipped_waiting: RequestQueue # 因异步依赖被跳过的请求 (KV loading, grammar 等待等)
self.running: list[Request]       # 正在执行的请求 (注意: 是普通 list, 不是优先队列)
```

**关键子组件**:
| 组件 | 作用 | 位置 |
|------|------|------|
| `kv_cache_manager` | KV cache 块分配/释放/前缀缓存 | scheduler.py:231-244 |
| `encoder_cache_manager` | 多模态 encoder 输出缓存 | scheduler.py:206-210 |
| `connector` | KV Transfer (P/D 分离) | scheduler.py:119-136 |
| `ec_connector` | Encoder Cache Transfer | scheduler.py:142-146 |
| `kv_event_publisher` | KV cache 事件发布 | scheduler.py:138-141 |
| `structured_output_manager` | 结构化输出管理 | scheduler.py:91 |

**请求字典** (scheduler.py:156):
```python
self.requests: dict[str, Request] = {}  # 全局 req_id -> Request 映射
```

**调度策略** (scheduler.py:158-163):
```python
self.policy = SchedulingPolicy(scheduler_config.policy)  # "fcfs" 或 "priority"
```

**Speculative Decode 相关** (scheduler.py:212-226):
```python
self.num_spec_tokens = 0        # speculative token 数量
self.num_lookahead_tokens = 0   # lookahead tokens (EAGLE/DFlash 用)
```

**状态追踪**:
```python
self.finished_req_ids: set[str] = set()             # 本步完成的请求 ID
self.finished_recving_kv_req_ids: set[str] = set()  # KV 异步接收完成的请求
self.failed_recving_kv_req_ids: set[str] = set()    # KV 加载失败的请求
self.prev_step_scheduled_req_ids: set[str] = set()  # 上一步调度的请求 ID
self.current_step: int = 0                           # 步数计数器
self._pause_state: PauseState = PauseState.UNPAUSED  # 暂停状态
```

---

## 2. schedule() 核心方法逐行精读

### 2.1 总体流程图

```
文件: scheduler.py:339-952 (~613 行, 核心中的核心)

schedule()
    |
    v
[1] 初始化局部变量 (scheduler.py:352-371)
    |
    v
[2] kv_cache_manager.new_step_starts() (scheduler.py:373)
    |
    v
[3] 遍历 RUNNING 请求 -> decode 调度 (scheduler.py:376-550)
    |   |-- 计算 num_new_tokens
    |   |-- allocate_slots()
    |   |-- 若内存不足 -> preempt
    |   +-- 记录 spec_decode_tokens, encoder_inputs
    |
    v
[4] 遍历 WAITING 请求 -> prefill 调度 (scheduler.py:562-853)
    |   |-- get_computed_blocks() (prefix cache)
    |   |-- connector.get_num_new_matched_tokens() (remote KV)
    |   |-- 计算 num_new_tokens (考虑 chunked prefill)
    |   |-- allocate_slots()
    |   |-- 若内存不足 -> break (不抢占)
    |   +-- 加入 running 队列
    |
    v
[5] 后处理 (scheduler.py:855-952)
    |-- 计算公共前缀块
    |-- 构建 SchedulerOutput
    |-- KV connector metadata
    +-- _update_after_schedule()
```

### 2.2 步骤 1: 初始化

```python
# scheduler.py:339-371
def schedule(self) -> SchedulerOutput:
    self.current_step += 1

    scheduled_new_reqs: list[Request] = []
    scheduled_resumed_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    preempted_reqs: list[Request] = []

    req_to_new_blocks: dict[str, KVCacheBlocks] = {}
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens  # 初始 token 预算
    if self._pause_state == PauseState.PAUSED_ALL:
        token_budget = 0  # 全部暂停时不调度任何请求
```

**设计哲学注释** (scheduler.py:342-350, 原作者 woosuk):

> "没有'解码阶段'或'prefill阶段'的概念。每个请求只有 `num_computed_tokens` 和 `num_tokens_with_spec`。每步尝试分配 token 使 `num_computed_tokens` 追上 `num_tokens_with_spec`。这足够通用以覆盖 chunked prefill、prefix caching、speculative decoding。"

### 2.3 步骤 2: RUNNING 请求调度 (Decode)

```
文件: scheduler.py:375-550
```

这是调度器第一阶段的循环, 处理已经在 running 队列中的请求。

```
running: [Req_A, Req_B, Req_C, ...]
                |
                v
         遍历每个 request
                |
         +------+------+
         |             |
    跳过条件检查    计算 num_new_tokens
    (PP/async)      |
         |      allocate_slots()
         |          |
         |    +-----+-----+
         |    |           |
         |  成功        失败(OOM)
         |    |           |
         |  加入scheduled  preempt 最低优先级请求
         |  消耗budget      |
         |               重试 allocate_slots()
```

**跳过条件** (scheduler.py:380-400):

1. **异步调度跳过**: 当请求已到达 max_tokens 但可能有 draft tokens 待处理时:
   ```python
   if request.num_output_placeholders > 0 and \
      request.num_computed_tokens + 2 - request.num_output_placeholders \
      >= request.num_prompt_tokens + request.max_tokens:
       continue  # 确定已到 max_tokens, 不再调度
   ```

2. **PP 节奏控制**: Pipeline Parallelism 中强制间隔 `pp_size` 步:
   ```python
   if self.current_step < request.next_decode_eligible_step:
       continue
   ```

**计算 num_new_tokens** (scheduler.py:402-415):
```python
num_new_tokens = (
    request.num_tokens_with_spec    # prompt + output + spec tokens
    + request.num_output_placeholders  # 异步调度占位符
    - request.num_computed_tokens      # 已计算 tokens
)
# 长 prefill 截断
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold
# 不超过 token 预算
num_new_tokens = min(num_new_tokens, token_budget)
# 不超过 max_model_len
num_new_tokens = min(num_new_tokens, max_model_len - 1 - request.num_computed_tokens)
```

**内存分配与抢占** (scheduler.py:458-508):

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
    if new_blocks is not None:
        break  # 分配成功

    # 分配失败 -> 抢占最低优先级请求
    if self.policy == SchedulingPolicy.PRIORITY:
        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
        # 优先级最高(数值最大)或最晚到达的请求被抢占
    else:
        preempted_req = self.running.pop()  # FCFS: 弹出最后加入的

    self._preempt_request(preempted_req, scheduled_timestamp)
    preempted_reqs.append(preempted_req)
    if preempted_req == request:
        break  # 抢占到自己了, 无法继续
```

**关键细节**: PRIORITY 模式下抢占时会回滚已调度该请求的资源:
```python
# scheduler.py:479-496
if preempted_req in scheduled_running_reqs:
    # 回退 token_budget, blocks, spec_tokens, encoder_inputs
    token_budget += num_scheduled_tokens.pop(preempted_req_id)
    req_to_new_blocks.pop(preempted_req_id)
    # ... 恢复 encoder_compute_budget
    req_index -= 1  # 回退索引以重新检查
```

### 2.4 步骤 3: WAITING 请求调度 (Prefill)

```
文件: scheduler.py:562-853
```

只有**没有发生抢占**且 scheduler 未暂停时才进入此阶段:

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
    # 开始调度 waiting 请求
```

**为何抢占后不调度新请求?** 因为抢占意味着内存紧张, 新请求大概率也无法分配内存。

**主循环** (scheduler.py:565-853):
```python
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    if len(self.running) == self.max_num_running_reqs:
        break  # 已达最大并发请求数

    request_queue = self._select_waiting_queue_for_scheduling()
    request = request_queue.peek_request()
```

**调度新请求的完整流程**:

```
     WAITING Request
          |
          v
   [阻塞状态检查] -- WAITING_FOR_REMOTE_KVS / GRAMMAR / STREAMING
          |
          v (pass)
   [LoRA 约束检查] -- max_loras 限制
          |
          v (pass)
   [Prefix Cache 查找] -- get_computed_blocks()
          |
          v
   [KV Connector 查找] -- connector.get_num_new_matched_tokens()
          |
          v
   [计算 num_new_tokens]
          |
          v
   [allocate_slots()]
          |
     +----+----+
     |         |
   成功      失败
     |         |
   加入running  break (不抢占!)
```

**关键差异: Waiting 阶段不做抢占!** 只有 Running 阶段才会抢占。Waiting 阶段分配失败直接 `break` 退出循环。

### 2.5 步骤 4: 后处理

```
文件: scheduler.py:855-952
```

```python
# 断言约束满足
assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
assert token_budget >= 0
assert len(self.running) <= self.max_num_running_reqs

# 计算公共前缀块 (用于 cascade attention)
num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(...)

# 构建 SchedulerOutput
scheduler_output = SchedulerOutput(
    scheduled_new_reqs=new_reqs_data,
    scheduled_cached_reqs=cached_reqs_data,
    num_scheduled_tokens=num_scheduled_tokens,
    ...
)

# KV connector metadata 构建
if self.connector is not None:
    scheduler_output.kv_connector_metadata = self._build_kv_connector_meta(...)

# 更新内部状态 (推进 num_computed_tokens)
self._update_after_schedule(scheduler_output)
return scheduler_output
```

---

## 3. 调度策略 FCFS vs PRIORITY

### 3.1 策略枚举

```
文件: request_queue.py:13-17
```

```python
class SchedulingPolicy(Enum):
    FCFS = "fcfs"        # 先来先服务
    PRIORITY = "priority"  # 优先级调度
```

### 3.2 队列实现对比

```
文件: request_queue.py:75-129 (FCFS) / 131-198 (PRIORITY)
```

| 特性 | FCFSRequestQueue | PriorityRequestQueue |
|------|-----------------|---------------------|
| 底层数据结构 | `collections.deque` | `heapq` (最小堆) |
| add_request | `append()` O(1) | `heappush()` O(log n) |
| pop_request | `popleft()` O(1) | `heappop()` O(log n) |
| prepend_request | `appendleft()` O(1) | `heappush()` O(log n) -- 注意: 优先级队列没有"前插"语义 |
| remove_request | `deque.remove()` O(n) | `heap.remove() + heapify()` O(n) |

### 3.3 Request 的比较运算

```
文件: request.py:305-316
```

```python
def __lt__(self, other):
    if self.priority != other.priority:
        return self.priority < other.priority     # 1. 优先级数值小的先
    if self.arrival_time != other.arrival_time:
        return self.arrival_time < other.arrival_time  # 2. 到达时间早的先
    if self.request_id != other.request_id:
        return self.request_id < other.request_id      # 3. ID 字典序
    return id(self) < id(other)                         # 4. 对象地址
```

### 3.4 双队列机制: waiting vs skipped_waiting

调度器维护**两个**等待队列:
- `waiting`: 正常等待的请求 (status == WAITING)
- `skipped_waiting`: 被异步依赖阻塞的请求 (status == WAITING_FOR_REMOTE_KVS / WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR / WAITING_FOR_STREAMING_REQ)

**队列选择逻辑** (scheduler.py:1646-1656):
```python
def _select_waiting_queue_for_scheduling(self):
    if self.policy == SchedulingPolicy.FCFS:
        return self.skipped_waiting or self.waiting or None
        # FCFS: 优先检查 skipped (先进先出语义)

    # PRIORITY: 比较两个队列头部的优先级
    if self.waiting and self.skipped_waiting:
        waiting_req = self.waiting.peek_request()
        skipped_req = self.skipped_waiting.peek_request()
        return self.waiting if waiting_req < skipped_req else self.skipped_waiting

    return self.waiting or self.skipped_waiting or None
```

**入队逻辑** (scheduler.py:1640-1644):
```python
def _enqueue_waiting_request(self, request):
    if self._is_blocked_waiting_status(request.status):
        self.skipped_waiting.add_request(request)
    else:
        self.waiting.add_request(request)
```

### 3.5 Running 队列的排序

**重要**: `self.running` 是一个**普通 list**, 不是优先队列!

这意味着 running 队列中的请求按**加入的顺序**排列:
- FCFS: 按到达时间排列
- PRIORITY: 按调度优先级排列 (但一旦加入 running 就不再重排)

抢占时 PRIORITY 模式通过 `max(self.running, key=lambda r: (r.priority, r.arrival_time))` 找到最低优先级的请求。

---

## 4. Prefill 调度详解

### 4.1 Prefill Token 计算

```
文件: scheduler.py:606-698
```

**Step 1: 获取本地前缀缓存** (scheduler.py:610-612):
```python
new_computed_blocks, num_new_local_computed_tokens = \
    self.kv_cache_manager.get_computed_blocks(request)
```

**Step 2: 获取远程 KV 缓存** (scheduler.py:615-640):
```python
if self.connector is not None:
    ext_tokens, load_kv_async = \
        self.connector.get_num_new_matched_tokens(request, num_new_local_computed_tokens)
    if ext_tokens is None:
        # Connector 无法确定匹配 token 数 -> 跳过此请求
        request_queue.pop_request()
        step_skipped_waiting.prepend_request(request)
        continue
    num_external_computed_tokens = ext_tokens
```

**Step 3: 总计算 token 数** (scheduler.py:638-641):
```python
num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
```

**Step 4: 计算需要调度的 token 数** (scheduler.py:679-698):
```python
num_new_tokens = request.num_tokens - num_computed_tokens

# 长 prefill 截断
threshold = self.scheduler_config.long_prefill_token_threshold
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold

# 非 chunked prefill 模式: 如果预算不够, 直接 break
if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > token_budget:
    break

num_new_tokens = min(num_new_tokens, token_budget)
```

### 4.2 Prefill 状态转换

```
文件: scheduler.py:814-834
```

```python
self.running.append(request)
request.status = RequestStatus.RUNNING
request.num_computed_tokens = num_computed_tokens  # 设为已有缓存 token 数

# 分类记录
if request.status == RequestStatus.WAITING:
    scheduled_new_reqs.append(request)     # 全新请求
elif request.status == RequestStatus.PREEMPTED:
    scheduled_resumed_reqs.append(request) # 被抢占后恢复的请求
```

### 4.3 Prefill 统计追踪

```
文件: scheduler.py:656-662
```

```python
if request.prefill_stats is not None:
    request.prefill_stats.set(
        num_prompt_tokens=request.num_prompt_tokens,
        num_local_cached_tokens=num_new_local_computed_tokens,
        num_external_cached_tokens=num_external_computed_tokens,
    )
```

---

## 5. Decode 调度详解

### 5.1 Decode Token 计算

Decode 阶段对 running 队列中每个请求计算需要处理的 token 数:

```
文件: scheduler.py:402-415
```

```python
num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
```

正常 decode 时:
- `num_tokens_with_spec` = prompt_tokens + output_tokens + spec_tokens
- `num_output_placeholders` = 0 (同步调度时)
- `num_computed_tokens` = 上一步推进后的值
- 所以 `num_new_tokens` 通常 = 1 (每步生成一个新 token) + spec_tokens

### 5.2 Speculative Decode Token 处理

```
文件: scheduler.py:519-534
```

```python
if request.spec_token_ids:
    num_scheduled_spec_tokens = num_new_tokens + request.num_computed_tokens \
                                - request.num_tokens - request.num_output_placeholders
    if num_scheduled_spec_tokens > 0:
        spec_token_ids = request.spec_token_ids
        if len(spec_token_ids) > num_scheduled_spec_tokens:
            spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
        scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
    request.spec_token_ids = []  # 清空, 下一步会通过 update_draft_token_ids 重新设置
```

### 5.3 Decode 的 num_computed_tokens 推进

```
文件: scheduler.py:981-1000
```

在 `_update_after_schedule()` 中, **调度完成后立即推进** num_computed_tokens:

```python
for req_id, num_scheduled_token in num_scheduled_tokens.items():
    request = self.requests[req_id]
    request.num_computed_tokens += num_scheduled_token
    request.is_prefill_chunk = request.num_computed_tokens < \
        (request.num_tokens + request.num_output_placeholders)
```

**注意**: 如果 spec tokens 后续被拒绝, `update_from_output()` 会回退 num_computed_tokens。

---

## 6. Preemption 抢占机制

### 6.1 _preempt_request 流程

```
文件: scheduler.py:959-979
```

```python
def _preempt_request(self, request, timestamp):
    assert request.status == RequestStatus.RUNNING

    self.kv_cache_manager.free(request)       # 释放所有 KV cache blocks
    self.encoder_cache_manager.free(request)  # 释放 encoder cache

    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0           # 重置为 0 -> 下次需要重新 prefill
    if request.spec_token_ids:
        request.spec_token_ids = []           # 清空 spec tokens
    request.num_preemptions += 1

    self.waiting.prepend_request(request)     # 放回 waiting 队列头部
```

### 6.2 抢占策略选择

```
文件: scheduler.py:473-498
```

**FCFS 模式**:
```python
preempted_req = self.running.pop()  # 弹出 running 尾部 (最后加入的)
```

**PRIORITY 模式**:
```python
preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
# 找到优先级最低 (priority 值最大) 或最晚到达的请求
self.running.remove(preempted_req)  # 从 running 中移除
```

### 6.3 抢占时的资源回收

PRIORITY 模式下, 如果被抢占的请求**在当前步骤已被调度**, 需要回收其资源:

```python
# scheduler.py:479-496
if preempted_req in scheduled_running_reqs:
    scheduled_running_reqs.remove(preempted_req)
    token_budget += num_scheduled_tokens.pop(preempted_req_id)  # 归还 token budget
    req_to_new_blocks.pop(preempted_req_id)                      # 归还 blocks
    scheduled_spec_decode_tokens.pop(preempted_req_id, None)     # 归还 spec tokens

    # 归还 encoder 计算预算
    preempted_encoder_inputs = scheduled_encoder_inputs.pop(preempted_req_id, None)
    if preempted_encoder_inputs:
        num_embeds_to_restore = sum(
            preempted_req.get_num_encoder_embeds(i) for i in preempted_encoder_inputs
        )
        encoder_compute_budget += num_embeds_to_restore

    req_index -= 1  # 回退索引以重新检查当前位置
```

### 6.4 抢占的代价

**vLLM V1 使用 recompute 策略 (不使用 swap)**:

```
抢占前:
  Request: num_computed_tokens = 500
  KV Cache: 500 tokens 的 KV blocks 已缓存

抢占后:
  1. kv_cache_manager.free(request) -> 释放所有 blocks
  2. num_computed_tokens = 0        -> 需要重新计算所有 tokens

恢复时:
  -> 从 waiting 队列头部优先调度
  -> 重新 prefill 所有 tokens (如果有 prefix cache 则部分可复用)
```

---

## 7. Prefix Caching 集成

### 7.1 本地 Prefix Cache

```
文件: scheduler.py:608-612
```

```python
if request.num_computed_tokens == 0:
    new_computed_blocks, num_new_local_computed_tokens = \
        self.kv_cache_manager.get_computed_blocks(request)
```

`get_computed_blocks()` 基于 block hash 匹配已有 KV cache blocks, 返回:
- `new_computed_blocks`: 匹配到的 block 列表
- `num_new_local_computed_tokens`: 匹配到的 token 数量

### 7.2 远程 Prefix Cache (KV Connector)

```
文件: scheduler.py:615-636
```

```python
if self.connector is not None:
    ext_tokens, load_kv_async = \
        self.connector.get_num_new_matched_tokens(request, num_new_local_computed_tokens)
```

- `ext_tokens`: 远程已有的 token 数
- `load_kv_async`: 是否需要异步加载

### 7.3 Prefix Cache 对调度的实际影响

```
假设:
  prompt_tokens = 1024
  本地 prefix cache hit = 768 tokens
  num_computed_tokens = 768

  num_new_tokens = 1024 - 768 = 256 (只需 prefill 256 tokens)
```

前缀缓存命中减少了:
1. **需要 prefill 的 token 数**: 从 1024 降到 256
2. **需要分配的 KV blocks**: 只需为新增 token 分配
3. **TTFT**: 大幅降低

### 7.4 scheduler_reserve_full_isl

```
文件: scheduler.py:760
```

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request, num_new_tokens, ...,
    full_sequence_must_fit=self.scheduler_reserve_full_isl,  # 默认 True
)
```

当 `scheduler_reserve_full_isl=True` 时, allocate_slots 会检查**完整序列长度**是否能放入 KV cache, 而不只是第一个 chunk。这防止了过度接纳导致 KV cache 震荡。

---

## 8. Chunked Prefill 机制

### 8.1 长 Prefill 截断

```
文件: scheduler.py:407-408 (running) / 684-686 (waiting)
```

```python
threshold = self.scheduler_config.long_prefill_token_threshold
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold
```

当 prompt 长度超过 `long_prefill_token_threshold` 时, 每步最多只处理 `threshold` 个 token。

### 8.2 Chunked Prefill 禁用时的行为

```
文件: scheduler.py:690-696
```

```python
if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > token_budget:
    # 如果预算不够处理整个 prefill, 直接 break
    # 不允许分块处理
    break
```

**对比**: chunked prefill 启用时, 即使预算不够处理完整 prefill, 也处理预算允许的部分:
```python
num_new_tokens = min(num_new_tokens, token_budget)  # 截断到预算
```

### 8.3 Prefill Chunk 追踪

```
文件: scheduler.py:995-997
```

```python
request.is_prefill_chunk = request.num_computed_tokens < \
    (request.num_tokens + request.num_output_placeholders)
```

`is_prefill_chunk = True` 表示请求的 prefill 还没完成 (还有更多 token 待计算)。

### 8.4 Chunked Prefill 数据流图

```
假设: prompt = 4096 tokens, threshold = 1024, budget = 2048

Step 1:
  num_computed_tokens = 0
  num_new_tokens = min(4096, 1024, 2048) = 1024
  -> 处理 tokens [0, 1024)
  -> num_computed_tokens 推进到 1024

Step 2:
  num_computed_tokens = 1024
  num_new_tokens = min(4096-1024, 1024, 2048) = 1024
  -> 处理 tokens [1024, 2048)
  -> num_computed_tokens 推进到 2048

Step 3:
  num_computed_tokens = 2048
  num_new_tokens = min(4096-2048, 1024, 2048) = 1024
  -> 处理 tokens [2048, 3072)
  -> num_computed_tokens 推进到 3072

Step 4:
  num_computed_tokens = 3072
  num_new_tokens = min(4096-3072, 1024, 2048) = 1024
  -> 处理 tokens [3072, 4096)
  -> num_computed_tokens 推进到 4096
  -> is_prefill_chunk = False, 开始 decode
```

### 8.5 Mamba Block-Aligned Split

```
文件: scheduler.py:289-337
```

对于混合模型 (Mamba + Attention), chunk 大小必须对齐到 `block_size` 的倍数, 以确保 Mamba 状态能被正确缓存。

---

## 9. Batch 管理与约束

### 9.1 三个核心约束

```
文件: scheduler.py:103-113, 855-866
```

| 约束 | 变量 | 含义 |
|------|------|------|
| 最大并发请求数 | `max_num_running_reqs` (= max_num_seqs, 默认 128) | running 队列最大长度 |
| 最大调度 token 数 | `max_num_scheduled_tokens` (= max_num_batched_tokens, 默认 2048) | 每步总 token 数上限 |
| 最大模型长度 | `max_model_len` | 单个请求最大 token 数 |

### 9.2 约束检查点

```
token_budget 管理:
  初始: token_budget = max_num_scheduled_tokens

  Running 阶段:
    每调度一个 request: token_budget -= num_new_tokens

  Waiting 阶段:
    每调度一个 request: token_budget -= num_new_tokens
    循环条件: token_budget > 0

  抢占时:
    回收: token_budget += num_scheduled_tokens[preempted_req_id]

max_num_running_reqs 检查:
  scheduler.py:566-567:
    if len(self.running) == self.max_num_running_reqs:
        break  # 不再调度新请求

max_model_len 检查:
  scheduler.py:413-415:
    num_new_tokens = min(num_new_tokens, max_model_len - 1 - request.num_computed_tokens)
```

### 9.3 LoRA 约束

```
文件: scheduler.py:552-559, 590-601
```

```python
scheduled_loras: set[int] = set()
# Running 阶段记录所有 LoRA ID
# Waiting 阶段检查:
if self.lora_config and request.lora_request and \
   (len(scheduled_loras) == self.lora_config.max_loras and \
    request.lora_request.lora_int_id not in scheduled_loras):
    # 超过 max_loras, 跳过此请求
    step_skipped_waiting.prepend_request(request)
    continue
```

---

## 10. update_from_output 输出处理

### 10.1 流程概览

```
文件: scheduler.py:1310-1630 (~320 行)

update_from_output(scheduler_output, model_runner_output)
    |
    v
[1] 提取 model_runner_output 各字段
    |
    v
[2] 处理无效 KV blocks (KV load failure)
    |
    v
[3] 遍历 num_scheduled_tokens:
    |   |-- 获取 generated_token_ids
    |   |-- 处理 spec decode 拒绝 (回退 num_computed_tokens)
    |   |-- 释放 encoder inputs
    |   |-- _update_request_with_output() (追加 token, 检查 stop)
    |   |-- 结构化输出 grammar 验证
    |   +-- 构建 EngineCoreOutput
    |
    v
[4] 移除 stopped 请求
    |
    v
[5] KV connector 状态更新
    |
    v
[6] 发布 KV cache 事件
    |
    v
[7] 构建 EngineCoreOutputs 返回
```

### 10.2 Spec Decode 拒绝处理

```
文件: scheduler.py:1395-1419
```

```python
if scheduled_spec_token_ids and generated_token_ids:
    num_draft_tokens = len(scheduled_spec_token_ids)
    num_accepted = len(generated_token_ids) - 1  # 第一个是真实 token
    num_rejected = num_draft_tokens - num_accepted

    if request.num_computed_tokens > 0:
        request.num_computed_tokens -= num_rejected  # 回退
    if request.num_output_placeholders > 0:
        request.num_output_placeholders -= num_rejected
```

### 10.3 请求停止检查

```
文件: utils.py:94-130
```

`check_stop()` 检查以下停止条件:
1. `num_output_tokens < min_tokens` -> 不停止
2. `last_token_id == eos_token_id` -> FINISHED_STOPPED
3. `last_token_id in stop_token_ids` -> FINISHED_STOPPED
4. `num_tokens >= max_model_len` 或 `num_output_tokens >= max_tokens` -> FINISHED_LENGTH_CAPPED
5. 重复模式检测 -> FINISHED_REPETITION

### 10.4 Streaming 请求处理

```
文件: scheduler.py:1658-1674
```

```python
def _handle_stopped_request(self, request):
    if not request.resumable:
        return True  # 普通请求 -> 真正完成

    if request.streaming_queue:
        update = request.streaming_queue.popleft()
        if update is None:
            return True  # streaming 完成
        self._update_request_as_session(request, update)  # 继续下一轮
    else:
        request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
        self.num_waiting_for_streaming_input += 1

    self._enqueue_waiting_request(request)
    return False  # 未完成, 继续等待
```

---

## 11. AsyncScheduler 异步调度扩展

```
文件: async_scheduler.py:12-67
```

### 11.1 核心差异

AsyncScheduler 继承 Scheduler, 重写两个方法:

1. **`_update_after_schedule()`**: 预分配 output placeholders
2. **`_update_request_with_output()`**: 处理异步 token 丢弃

### 11.2 Output Placeholder 机制

```
文件: async_scheduler.py:19-41
```

```python
def _update_after_schedule(self, scheduler_output):
    super()._update_after_schedule(scheduler_output)
    for req_id in scheduler_output.num_scheduled_tokens:
        request = self.requests[req_id]
        if request.is_prefill_chunk:
            continue

        # 每步预分配: 1 (真实 token) + num_spec_tokens (draft tokens)
        request.num_output_placeholders += 1 + cur_num_spec_tokens
        request.spec_token_ids = self._spec_token_placeholders  # [-1] * num_spec_tokens

        # PP 节奏控制
        if self.use_v2_model_runner:
            request.next_decode_eligible_step = self.current_step + self.pp_size
```

### 11.3 异步 Token 丢弃

```
文件: async_scheduler.py:43-67
```

```python
def _update_request_with_output(self, request, new_token_ids):
    if request.async_tokens_to_discard > 0:
        # reset_prefix_cache 强制抢占时, 丢弃过期的异步输出帧
        request.async_tokens_to_discard -= 1
        return [], False

    # 正常处理
    new_token_ids, stopped = super()._update_request_with_output(request, new_token_ids)

    # 更新 placeholder 计数
    request.num_output_placeholders -= len(new_token_ids)

    # 缓存 blocks (异步模式下需要显式缓存)
    if status_before_update == RequestStatus.RUNNING:
        self.kv_cache_manager.cache_blocks(
            request, request.num_computed_tokens - request.num_output_placeholders
        )
```

---

## 12. KV Connector 与 P/D 分离

### 12.1 异步 KV 加载流程

```
文件: scheduler.py:674-812
```

```
Request 进入 waiting
    |
    v
connector.get_num_new_matched_tokens() -> ext_tokens, load_kv_async=True
    |
    v
allocate_slots() (delay_cache_blocks=True)
    |
    v
request.status = WAITING_FOR_REMOTE_KVS
request.num_computed_tokens = num_computed_tokens (预设置)
    |
    v (放入 skipped_waiting)
    |
    v (Worker 异步加载 KV)
    |
    v
kv_connector_output.finished_recving -> finished_recving_kv_req_ids
    |
    v
_update_waiting_for_remote_kv():
    - kv_cache_manager.cache_blocks()
    - 全命中时: num_computed_tokens = num_tokens - 1 (需要重新计算最后一 token)
    |
    v
request.status -> WAITING / PREEMPTED
    |
    v
下次 schedule() 时正常调度
```

### 12.2 KV 加载失败处理

```
文件: scheduler.py:2202-2374
```

两种失败策略 (由 `kv_load_failure_policy` 控制):
1. **recompute** (默认): 回退 num_computed_tokens 到第一个失败 block, 下一步重新计算
2. **fail**: 直接将请求标记为 FINISHED_ERROR

```
_update_requests_with_invalid_blocks():
    遍历每个请求的 blocks
    -> 发现无效 block -> 截断 num_computed_tokens
    -> 收集下游依赖 blocks 用于驱逐

_handle_invalid_blocks():
    1. 处理异步加载请求 (skipped_waiting)
    2. 处理同步加载请求 (running)
    3. 根据 policy 决定 recompute 或 fail
```

---

## 13. 边缘场景与错误处理

### 13.1 GPU 满载

当 `allocate_slots()` 返回 None 时:

**Running 阶段**:
- 抢占最低优先级请求, 释放内存后重试
- 如果抢占到当前请求自身, 放弃分配

**Waiting 阶段**:
- 直接 `break`, 不抢占任何请求
- 释放 encoder cache: `self.encoder_cache_manager.free(request)`

### 13.2 请求中止 (Abort)

```
文件: scheduler.py:1806-1867
```

```python
def finish_requests(self, request_ids, finished_status):
    # 第一遍: 从 running/waiting 队列中移除
    if running_requests_to_remove:
        self.running = remove_all(self.running, running_requests_to_remove)
    if waiting_requests_to_remove:
        self.waiting.remove_requests(waiting_requests_to_remove)
        self.skipped_waiting.remove_requests(waiting_requests_to_remove)

    # 第二遍: 设置状态并释放
    for request in valid_requests:
        delay_free_blocks = (request.status == WAITING_FOR_REMOTE_KVS and \
                            req_id not in finished_recving_kv_req_ids)
        request.status = finished_status
        self._free_request(request, delay_free_blocks=delay_free_blocks)
```

**注意**: WAITING_FOR_REMOTE_KVS 状态的请求可能需要延迟释放 blocks, 因为 KV transfer 可能仍在进行。

### 13.3 Prefix Cache 重置

```
文件: scheduler.py:1923-1971
```

```python
def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
    if reset_running_requests:
        # 抢占所有 running 请求
        while self.running:
            request = self.running.pop()
            self._preempt_request(request, timestamp)
            request.async_tokens_to_discard = request.num_output_placeholders
            request.num_output_placeholders = 0
        self.prev_step_scheduled_req_ids.clear()

    reset_successful = self.kv_cache_manager.reset_prefix_cache()
```

### 13.4 暂停状态

```
文件: interface.py:22-33
```

```python
class PauseState(IntEnum):
    UNPAUSED = 0     # 正常运行
    PAUSED_NEW = 1   # 只继续 running 请求, 不调度新请求
    PAUSED_ALL = 2   # 完全暂停
```

- `PAUSED_ALL`: `token_budget = 0`, 不执行任何调度
- `PAUSED_NEW`: 只执行 Running 阶段 (scheduler.py:562 检查 `self._pause_state == PauseState.UNPAUSED`)

### 13.5 Encoder 预算耗尽

```
文件: scheduler.py:1126-1284
```

当 encoder 计算预算或缓存不足时:
```python
if not self.encoder_cache_manager.can_allocate(request, i, encoder_compute_budget, ...):
    if num_computed_tokens + shift_computed_tokens < start_pos:
        num_new_tokens = start_pos - (num_computed_tokens + shift_computed_tokens)
    else:
        num_new_tokens = 0  # 因为 prefix caching, 无法跳过 encoder input
    break
```

### 13.6 NaN in Logits

```
文件: scheduler.py:1522-1523
```

```python
if num_nans_in_logits is not None and req_id in num_nans_in_logits:
    request.num_nans_in_logits = num_nans_in_logits[req_id]
```

记录每个请求的 logits NaN 数量, 但不直接影响调度决策。

---

## 14. 数据流总图

### 14.1 完整调度周期

```
                        +------------------+
                        |   Engine Core    |
                        |  (busy loop)     |
                        +--------+---------+
                                 |
              +------------------+-------------------+
              |                                      |
    schedule() [构建调度计划]              update_from_output() [处理结果]
              |                                      |
              v                                      v
    +---------+---------+                 +----------+----------+
    | 1. Running 阶段   |                 | 1. 提取输出         |
    |    (decode)       |                 | 2. Spec拒绝处理     |
    | 2. Waiting 阶段   |                 | 3. Stop检查         |
    |    (prefill)      |                 | 4. 状态更新         |
    | 3. 构建 Output    |                 | 5. KV事件发布       |
    +---------+---------+                 +----------+----------+
              |                                      |
              v                                      v
    +---------+---------+                 +----------+----------+
    | SchedulerOutput   |------forward---->| ModelRunnerOutput  |
    | - new_reqs_data   |                  | - sampled_tokens   |
    | - cached_reqs_data|                  | - logprobs         |
    | - num_scheduled   |                  | - spec_tokens      |
    |   _tokens         |                  | - kv_connector     |
    | - block_ids       |                  |   _output          |
    +-------------------+                  +---------------------+
```

### 14.2 请求状态机

```
    +----------+
    | WAITING  |<---- add_request()
    +----+-----+
         |
    [调度成功]
         |
    +----v-----+     [内存不足]     +----------+
    | RUNNING  +------抢占-------> | PREEMPTED|
    +----+-----+                   +-----+----+
         |                               |
    [正常完成]                      [重新调度]
         |                               |
    +----v-----+                    +----v-----+
    | FINISHED |                    | WAITING  |
    | _STOPPED |                    | (回到队列)|
    | _LENGTH  |                    +----------+
    | _ABORTED |
    | _ERROR   |
    +----------+

    特殊状态:
    WAITING_FOR_REMOTE_KVS        -- KV 异步加载中
    WAITING_FOR_STRUCTURED_OUTPUT -- Grammar 初始化中
    WAITING_FOR_STREAMING_REQ     -- 等待 streaming 输入
```

### 14.3 Token Budget 流转

```
初始 budget = max_num_scheduled_tokens (默认 2048)

Running 阶段:
  for each running request:
    num_new_tokens = min(needed, budget)
    budget -= num_new_tokens       [消耗]
    if preempted:
      budget += recovered_tokens   [回收]

Waiting 阶段:
  for each waiting request:
    num_new_tokens = min(needed, budget)
    budget -= num_new_tokens       [消耗]
    if allocate_slots fails:
      break                        [停止, 不回收]

最终:
  total_scheduled = max_num_scheduled_tokens - budget
```

### 14.4 关键方法调用频率

```
每步调用一次:
  schedule()                    ~613 行
  _update_after_schedule()      ~40 行
  update_from_output()          ~320 行

每次 schedule() 内:
  kv_cache_manager.new_step_starts()     -- 1 次
  kv_cache_manager.allocate_slots()      -- N 次 (N = 请求数)
  kv_cache_manager.get_computed_blocks() -- M 次 (M = waiting 请求数)
  kv_cache_manager.get_num_common_prefix_blocks() -- 1 次
  _make_cached_request_data()            -- 1 次

低频调用:
  _preempt_request()            -- 仅 OOM 时
  finish_requests()             -- 外部触发 (abort)
  reset_prefix_cache()          -- 模型权重更新时
```

---

## 附录: 源码文件索引

| 文件 | 行数 | 核心职责 |
|------|------|---------|
| `vllm/v1/core/sched/scheduler.py` | ~2375 | 核心调度逻辑 |
| `vllm/v1/core/sched/async_scheduler.py` | ~68 | 异步调度扩展 |
| `vllm/v1/core/sched/request_queue.py` | ~209 | 队列实现 |
| `vllm/v1/core/sched/output.py` | ~264 | 输出数据结构 |
| `vllm/v1/core/sched/interface.py` | ~245 | 抽象接口 |
| `vllm/v1/core/sched/utils.py` | ~131 | 工具函数 |
| `vllm/v1/request.py` | ~362 | Request 数据模型 |
| `vllm/v1/core/kv_cache_manager.py` | - | KV cache 管理 |
| `vllm/config/scheduler.py` | ~309 | SchedulerConfig |

### 关键行号速查

| 功能 | 文件:行 |
|------|---------|
| Scheduler.__init__ | scheduler.py:66-288 |
| schedule() 入口 | scheduler.py:339 |
| Running 阶段循环 | scheduler.py:376-550 |
| Waiting 阶段循环 | scheduler.py:562-853 |
| Preempt 逻辑 | scheduler.py:458-508 |
| _preempt_request() | scheduler.py:959-979 |
| _update_after_schedule() | scheduler.py:981-1021 |
| update_from_output() | scheduler.py:1310-1630 |
| Prefix cache 查找 | scheduler.py:608-668 |
| KV async loading | scheduler.py:674-812 |
| Queue 选择 | scheduler.py:1646-1656 |
| check_stop() | utils.py:94-130 |
| Request.__lt__ | request.py:305-316 |
| RequestStatus 枚举 | request.py:319-361 |
| AsyncScheduler._update_after_schedule | async_scheduler.py:19-41 |
| FCFSRequestQueue | request_queue.py:75-129 |
| PriorityRequestQueue | request_queue.py:131-198 |
| SchedulerOutput | output.py:181-256 |
| PauseState | interface.py:22-33 |
