# vLLM V1 Scheduler vs SGLang Overlap调度架构对比

> 2026-06-15 | 源码: vllm/v1/core/sched/scheduler.py + vllm/v1/engine/core.py + sglang/srt/managers/scheduler.py + overlap_utils.py
> 核心: vLLM=同步两阶段(RUNNING→WAITING)+抢占全重置; SGLang=Overlap CPU-GPU+radix tree保存KV+FutureMap relay+PrefillAdder精细预算

## 1. vLLM V1 Scheduler架构

### 三队列结构

```python
# scheduler.py:154-167
self.requests: dict[str, Request] = {}          # 所有活跃请求
self.waiting = create_request_queue(self.policy) # 主等待队列(FCFS/PRIORITY)
self.skipped_waiting = create_request_queue(self.policy) # 阻塞请求(异步依赖)
self.running: list[Request] = []                # 运行队列(按准入时间排序)
```

### 两阶段调度算法

```
Phase 1: RUNNING (行374-549)
  → 优先为正在运行的decode请求分配token预算
  → 每个请求: num_new_tokens = min(剩余tokens, token_budget, long_prefill_threshold)
  → KV cache分配失败 → 抢占! (见Section 3)
  → 关键: num_new_tokens==0时用continue而非break → 非严格FCFS!

Phase 2: WAITING (行560-852)
  → 只在Phase 1无抢占且UNPAUSED状态时执行!
  → 从waiting/skipped_waiting选择请求
  → 预算检查: token_budget > 0 且 running_count < max_num_running
  → KV cache必须能容纳完整输入序列长度 → scheduler_reserve_full_isl
  → 请求状态: WAITING → RUNNING

关键设计哲学 (行340-349注释):
  "没有'decode phase'或'prefill phase' → 每个请求只有num_computed_tokens vs num_tokens_with_spec
   → scheduler只是让computed追上specified → 不区分decode/prefill!"
```

### EngineCore事件循环

```python
# core.py:1212-1220
def run_busy_loop(self):
    while self._handle_shutdown():
        self._process_input_queue()   # 接收新请求/abort
        self._process_engine_step()   # schedule→forward→update

# core.py:439-468 step()
1. scheduler.has_requests() → 检查有工作
2. scheduler.schedule() → SchedulerOutput{req_id: num_tokens}
3. model_executor.execute_model() → GPU forward (async Future)
4. grammar_bitmask → 并行计算
5. future.result() → 等GPU完成 → 阻塞!
6. process_aborts → 处理abort
7. scheduler.update_from_output() → 处理生成的tokens → 释放完成请求

→ 关键: 每步是同步的! schedule→forward→process → 无overlap!
```

## 2. vLLM抢占策略

### FCFS抢占

```python
# scheduler.py:458-507
while True:
    new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
    if new_blocks is not None:
        break  # 成功
    # FCFS: evict最新准入的请求 → 最低优先级(最少进展)
    preempted_req = self.running.pop()  # 最后准入的!

def _preempt_request(self, request):
    kv_cache_manager.free(request)       # 释放所有KV blocks!
    request.num_computed_tokens = 0       # 完全重置 → 从头recompute!
    request.status = RequestStatus.PREEMPTED
    self.waiting.prepend_request(request) # 放回waiting队列前端
```

### PRIORITY抢占

```python
# evict最低priority值的请求
preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
```

### 关键缺陷: 完全重置!

```
vLLM抢占 = 完全驱逐 → 所有KV blocks释放 → num_computed_tokens=0 → 从头recompute!

问题:
  → 1000-token system prompt被抢占 → 1000 tokens KV cache全部丢失!
  → 下次调度 → 必须从头prefill → 完全浪费之前的计算!
  → 内存压力大时 → 反复抢占→反复recompute → KV cache thrashing!

vs SGLang:
  → SGLang抢占 → KV blocks进入radix tree cache → prefix可复用!
  → system prompt 1000 tokens → radix tree保存 → 下次只需prefill question部分!
  → 省system prompt KV计算 → 极大减少抢占后的recompute开销!
```

## 3. Token预算模型

### vLLM: 单一计数器

```python
# scheduler.py:358
token_budget = self.max_num_scheduled_tokens  # 默认=max_num_batched_tokens=2048

# 运行时:
token_budget -= num_new_tokens  # Phase 1每请求
token_budget -= num_new_tokens  # Phase 2每请求
# 抢占回收: token_budget += preempted_req.num_scheduled_tokens

# 附加约束:
# max_num_running_reqs = 128  → 限制运行请求数
# max_model_len → 单请求上限
# long_prefill_token_threshold → chunked prefill上限
```

### SGLang: PrefillAdder精细预算

```python
# schedule_policy.py PrefillAdder
rem_total_tokens  # 可用+可驱逐KV池 - 运行decode tokens预留
rem_input_tokens  # 剩余输入token预算
rem_chunk_tokens  # chunked prefill剩余
rem_swa_tokens    # sliding window attention剩余

# 更精细:
# → 考虑运行decode请求的KV空间需求 → 减去offset
# → 考虑可驱逐KV → eviction释放后可用于新prefill
# → 考虑chunked prefill → 分批处理长序列
```

## 4. SGLang Overlap调度架构

### event_loop_overlap()

```python
# sglang scheduler.py:1453-1510
def event_loop_overlap(self):
    while True:
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)

        batch = self.get_next_batch_to_run()
        if batch:
            batch_result = self.run_batch(batch)  # GPU执行当前batch
            self.result_queue.append((batch.copy(), batch_result))

        if self.last_batch:  # 上一个batch的结果在CPU处理
            pop_and_process()  # CPU处理上一batch → 与当前GPU overlap!

# 关键: CPU处理上一batch结果 + GPU执行当前batch → pipeline overlap!
# → 消除CPU→GPU stall → decode迭代间无CPU瓶颈!
```

### FutureMap relay机制

```python
# overlap_utils.py FutureMap
# pool-indexed GPU buffer → 存储上一forward pass的sampled tokens
# 下一迭代的decode input_ids直接从GPU buffer获取 → 无D2H/H2D round-trip!
# → decode token IDs不经过CPU → 纯GPU路径 → 消除CPU latency!

vs vLLM:
# vLLM: SchedulerOutput → CPU → ModelRunnerOutput → CPU传递token IDs
# → 每步都有D2H → CPU处理 → H2D → 增加latency!
# → 这是vLLM decode吞吐的关键瓶颈之一!
```

### SGLang调度分phase?

```
SGLang不是严格两阶段! 而是:
  → is_mixed_chunk: prefill和decode可以混合在同一batch!
  → PrefillAdder: 精细预算 → 考虑运行decode占用 → 计算可用prefill空间
  → chunked_prefill_size: 长序列分chunk → 不是一次全prefill
  → enable_dynamic_chunking: 根据history_len预测最优chunk size

vs vLLM:
  → 严格两阶段: Phase 1(RUNNING) → Phase 2(WAITING, 只有无抢占时)
  → Phase 2被抢占跳过 → 新请求必须等 → 增加TTFT延迟!
  → 无混合batch → decode和prefill不能在同一step → 吞吐降低!
```

## 5. 核心架构差异总结

| 维度 | vLLM V1 | SGLang |
|------|---------|--------|
| **调度抽象** | Token级(num_computed vs num_specified) | 显式EXTEND(prefill)/DECODE模式 |
| **Phase顺序** | 严格: RUNNING→WAITING(只有无抢占时) | 混合: prefill+decode同一batch |
| **CPU-GPU overlap** | ❌ 同步: schedule→forward→process | ✅ overlap: CPU处理上batch+GPU执行本batch |
| **FutureMap relay** | ❌ CPU传递token IDs(D2H+H2D) | ✅ GPU buffer relay(无CPU round-trip) |
| **抢占KV处理** | 释放→从头recompute→全重置 | radix tree保存→prefix可复用→部分重compute |
| **预算模型** | 单一token_budget计数器 | PrefillAdder 4维(total/input/chunk/swa) |
| **抢占策略** | FCFS=evict最新/PRIORITY=evict最低priority | priority差>threshold才抢占→radix tree缓存 |
| **chunked prefill** | token级自动→不区分prefill/decode | 显式chunked_req+动态chunk size预测 |
| **waiting queue** | 2个(waiting+skipped_waiting) | 1个+radix tree prefix cache awareness |

## 6. RTX 4090实战影响

```
RTX 4090 24GB (7B推理):

1. vLLM decode吞吐瓶颈:
   → 同步调度 → CPU处理D2H+H2D → 每步stall → decode慢
   → 抢占全重置 → 内存压力时反复recompute → KV cache thrashing
   → 单一token_budget → 不考虑运行decode占用 → 预算不精细
   → INT4+INT8KV+GQA → 4791 tok/s → 但CPU overhead限制进一步提升!

2. SGLang decode吞吐优势:
   → overlap → CPU处理+GPU执行并行 → decode迭代无stall
   → FutureMap → decode token IDs纯GPU路径 → 无D2H+H2D latency
   → radix tree保存被抢占KV → 重compute只prefill suffix → 系统prompt复用!
   → PrefillAdder精细预算 → 更好利用24GB有限空间
   → INT4+INT8KV+GQA+RadixAttention → 多轮对话5x加速!

3. GRPO rollout场景对比:
   → vLLM GRPO rollout_n=8 → 8×system prompt → 无prefix复用(block对齐浪费)
   → SGLang GRPO rollout_n=8 → system prompt KV只1次 → 7×省prefill!
   → SGLang更适合GRPO推理 → prefix缓存直接减少rollout计算量!

4. 实战建议:
   → 单轮推理: vLLM和SGLang差距小(无prefix复用场景)
   → 多轮对话/GRPO: SGLang显著更快(radix tree+overlap)
   → RTX 4090: SGLang更优 → 但vLLM生态更成熟 → 根据场景选择
   → verl+GRPO: 推荐SGLang作为rollout引擎(已有SGLangRollout支持)
```

## 7. 关键设计洞察

```
1. 同步vs overlap → vLLM同步每步 → SGLangoverlap pipeline → decode吞吐差异根源
   → vLLM: schedule(CPU) → forward(GPU) → process(CPU) → 线性无overlap
   → SGLang: 上batch处理(CPU) + 本batchforward(GPU) → 并行overlap!

2. 抢占KV处理 → 最大架构差异!
   → vLLM: 全释放+全重置 → 抢占后从头recompute → system prompt浪费
   → SGLang: radix tree保存 → 抢占后只recompute suffix → prefix复用!
   → 这是SGLang多轮对话5x加速的关键!

3. FutureMap relay → 消除CPU瓶颈
   → vLLM: D2H(token IDs) → CPU处理 → H2D(input_ids) → 每步2次数据传输
   → SGLang: GPU buffer relay → 无D2H/H2D → 纯GPU路径 → latency消除!

4. 两阶段vs混合 → vLLM严格分离 → SGLang灵活混合
   → vLLM Phase 2条件: 无抢占+UNPAUSED → 有抢占时新请求全被阻塞!
   → SGLang混合: prefill+decode同一batch → 更充分利用GPU

5. 预算精细度 → PrefillAdder vs 单计数器
   → SGLang考虑运行decode占用+可驱逐KV → 更精确budgeting
   → vLLM单一token_budget → 可能过度或不足估计可用空间

6. RTX 4090选择:
   → 单轮推理/简单serving → vLLM(成熟生态+工具链)
   → 多轮对话/GRPO rollout → SGLang(radix tree+overlap+FutureMap)
   → 最佳策略: 根据场景灵活选择 → verl已支持两种rollout引擎!
```

---

Sources:
- vllm/v1/core/sched/scheduler.py — V1 Scheduler (两阶段+抢占+预算)
- vllm/v1/engine/core.py — EngineCore事件循环
- vllm/v1/core/sched/async_scheduler.py — AsyncScheduler (PP+async)
- vllm/v1/core/sched/request_queue.py — FCFS/PRIORITY队列
- sglang/python/sglang/srt/managers/scheduler.py — SGLang overlap调度
- sglang/python/sglang/srt/managers/schedule_policy.py — PrefillAdder+LPM+DFS
- sglang/python/sglang/srt/managers/overlap_utils.py — FutureMap relay
- notebook/fundamentals/sglang-radix-attention.md
- memory/vllm-source-reading.md
