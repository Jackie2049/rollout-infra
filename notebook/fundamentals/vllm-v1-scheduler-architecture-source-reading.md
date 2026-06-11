# vLLM V1 Scheduler Architecture Source Reading — Unified Token Budget + FCFS/Priority + Preemption + PD Disaggregation

> 2026-06-11 | vLLM V1调度器全链路源码分析: Scheduler → SchedulingPolicy(FCFS/Priority) → RequestQueue → Preemption → KVConnector → SpecDecode → EncoderCache
> 源码: vllm/v1/core/sched/scheduler.py (~1000行), request_queue.py (209行), interface.py (245行), output.py (264行)
> 关联: vllm-v1-kv-cache-architecture-source-reading.md, vllm-v1-spec-decode-architecture-source-reading.md, scheduler-architecture-deep-dive.md

## 0. 核心定律: Unified Token Budget + Decode优先 + Prefill填充

```
vLLM V1调度算法核心:
  1. 无"prefill phase"/"decode phase"区分 → 每个request有num_computed_tokens
  2. 每step: 为每个request分配token使其num_computed_tokens追赶num_tokens_with_spec
  3. 通用框架覆盖: chunked prefill + prefix caching + speculative decoding + jump decoding
  4. RUNNING优先 → decode请求先分配token → WAITING(prefill)用剩余budget填充

调度流程:
  Phase 1: 遍历RUNNING → decode请求 → 分配1+spec tokens → KV block分配 → 无则抢占
  Phase 2: 遍历WAITING → prefill请求 → prefix cache hit → chunk → KV block分配
  Phase 3: 构造SchedulerOutput → new_reqs + cached_reqs + spec_decode_tokens + connector_meta

关键约束:
  → max_num_scheduled_tokens (token budget per step) → 限制总token数
  → max_num_running_reqs → 限制并发request数
  → KV block数量 → 限制KV cache容量 → 不够则抢占低优先级request
  → LoRA max_loras → 限制同时激活的LoRA数量
```

## 1. Scheduler类 — 调度器核心

```
文件: vllm/v1/core/sched/scheduler.py

class Scheduler(SchedulerInterface):
  → requests: dict[str, Request] → 所有活跃请求
  → waiting: RequestQueue → 等待调度(prefill)
  → skipped_waiting: RequestQueue → 被跳过的等待请求(async KV/LoRA限制)
  → running: list[Request] → 正在运行(decode/prefill-in-progress)
  → finished_req_ids: set[str] → 已完成请求(下step清理)

关键配置:
  → max_num_running_reqs = scheduler_config.max_num_seqs → 最大并发数
  → max_num_scheduled_tokens → 每step token budget → 默认max_num_batched_tokens
  → max_model_len → 模型最大长度 → spec decode需要检查
  → policy: SchedulingPolicy → FCFS或Priority
  → use_eagle: bool → EAGLE投机解码 → 影响KV block分配
  → num_spec_tokens / num_lookahead_tokens → 投机步数 → lookahead=spec tokens(额外KV预留)
  → connector: KVConnectorBase_V1 → P/D分离 KV传输
  → ec_connector: ECConnector → Encoder Cache传输

Mamba特殊处理:
  → has_mamba_layers → Mamba SSM state cache → block-aligned splitting
  → mamba_cache_mode="align" → 必须block_size对齐 → 否则state缓存miss
  → EAGLE prune → last_cache_position -= block_size → 防止Mamba cache miss!

Spec Decode调度:
  → num_output_placeholders → draft tokens占位 → 计算实际新token数
  → spec_token_ids → 当前draft tokens → scheduler_output携带
  → DFlash → num_lookahead_tokens = num_spec_tokens + 1 → 需额外slot

编码器缓存:
  → EncoderCacheManager / EncoderDecoderCacheManager → 多模态编码器输出缓存
  → MultiModalBudget → 编码器计算budget + cache size限制
  → _try_schedule_encoder_inputs() → 编码器输入调度
```

## 2. schedule() 方法 — 核心调度算法

```
schedule()流程(约300行):

Step 1: 初始化
  → token_budget = max_num_scheduled_tokens
  → paused_all → token_budget = 0 → 不调度任何请求
  → kv_cache_manager.new_step_starts() → KV coordinator新step开始

Step 2: RUNNING请求调度 (decode优先)
  → 遍历self.running:
    → 如果达到max_tokens → 跳过(async scheduling优化)
    → 如果next_decode_eligible_step > current_step → PP节奏控制 → 跳过
    → num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens
    → long_prefill_token_threshold → chunk限制
    → min(num_new_tokens, token_budget) → budget限制
    → min(num_new_tokens, max_model_len - 1 - num_computed_tokens) → 长度限制

    → KV block分配:
      → kv_cache_manager.allocate_slots(request, num_new_tokens, num_lookahead_tokens)
      → 成功 → 分配 → token_budget -= num_new_tokens
      → 失败 → 抢占最低优先级request → 释放KV blocks → 重试

    → Spec decode tokens:
      → 如果request.spec_token_ids存在 → 计算scheduled_spec_tokens
      → 添加到scheduled_spec_decode_tokens → 清空spec_token_ids

Step 3: WAITING请求调度 (prefill)
  → 只有在没有抢占且pause_state==UNPAUSED时才调度
  → 遍历waiting + skipped_waiting:
    → 检查max_num_running_reqs限制
    → LoRA限制检查 → max_loras约束
    → num_computed_tokens==0 → prefix cache hit:
      → kv_cache_manager.get_computed_blocks(request) → 本地prefix hit
      → connector.get_num_new_matched_tokens() → 远程prefix hit(P/D分离)
      → 总computed = local + external

    → num_new_tokens = request.num_tokens - num_computed_tokens
    → long_prefill_token_threshold → chunk限制
    → 如果!enable_chunked_prefill且budget不够 → break!

    → 编码器调度 → _try_schedule_encoder_inputs()
    → Mamba block-aligned splitting → 如果需要

    → KV block分配:
      → allocate_slots(request, num_new_tokens, ...)
      → 成功 → 加入running → WAITING→RUNNING
      → 失败 → break(不再调度新请求)

    → Async KV load → WAITING_FOR_REMOTE_KVS → 不立即运行 → 等KV传输完成

Step 4: 构造SchedulerOutput
  → new_reqs_data → NewRequestData.from_request()
  → cached_reqs_data → CachedRequestData(差量传输!)
  → num_scheduled_tokens → {req_id: num_tokens}
  → scheduled_spec_decode_tokens → {req_id: spec_token_ids}
  → num_common_prefix_blocks → cascade attention用
  → finished_req_ids → 已完成请求
  → kv_connector_metadata → P/D分离KV传输元数据
  → ec_connector_metadata → encoder cache传输元数据
  → new_block_ids_to_zero → 新分配的block需清零(防NaN!)
```

## 3. SchedulingPolicy — FCFS vs Priority

```
SchedulingPolicy(Enum):
  → FCFS = "fcfs" → 先来先服务(deque)
  → PRIORITY = "priority" → 优先级(heap)

FCFSRequestQueue(deque[Request]):
  → add_request → append → 尾部添加
  → pop_request → popleft → 头部取出(最早的)
  → prepend_request → appendleft → 抢占后重新排队放在最前!
  → remove_request → self.remove() → O(n)删除

PriorityRequestQueue(heap):
  → _heap: list[Request] → 最小堆
  → add_request → heappush → 按(priority, arrival_time)排序
  → pop_request → heappop → 最高优先级
  → remove_request → _heap.remove() + heapify → O(n)删除+重建

抢占策略:
  → FCFS: self.running.pop() → 移除最后一个(最晚来的) → 最低优先级=最新请求
  → Priority: max(running, key=lambda r: (r.priority, r.arrival_time)) → 移除优先级最低的
  → 抢占后: prepend_request → 放到waiting最前 → 下次优先调度!

RTX 4090影响:
  → FCFS最公平 → 所有请求平等 → 小模型推荐
  → Priority → 高优先级请求优先 → 大规模生产可能需要
  → 抢占 → 释放KV blocks → 给高优先级request腾空间 → RTX 4090 24GB → 抢占频繁!
```

## 4. Preemption — 释放KV给高优先级

```
_preempt_request(request, timestamp):
  → kv_cache_manager.free(request) → 释放所有KV blocks → 归还block_pool
  → encoder_cache_manager.free(request) → 释放编码器缓存
  → _inflight_prefills.discard(request) → 移除in-flight标记
  → request.status = PREEMPTED
  → request.num_computed_tokens = 0 → 重置 → 需要重新计算!
  → request.spec_token_ids = [] → 清空draft tokens
  → request.num_preemptions += 1 → 抢占计数
  → waiting.prepend_request(request) → 放到waiting最前 → 下次优先调度

关键设计:
  → num_computed_tokens = 0 → 需要recompute → 不是swap!
  → → vLLM V1默认recomputation → 不swap到CPU → RTX 4090 PCIe swap慢→recompute更快
  → → 但recompute有overhead → 长prompt重新计算很慢 → 抢占代价高!

  → 抢占策略选择:
    → FCFS → running.pop() → 最晚请求被抢占
    → Priority → max(priority) → 最低优先级被抢占
    → → 但! 如果被抢占的request是自己 → break → 无法调度

RTX 4090最优:
  → 抢占=recomputation → 短请求快速恢复 → 长请求慢
  → → 限制max_num_running_reqs → 减少抢占频率
  → → INT4+INT8KV → KV占用少 → 抢占少 → 更稳定
  → → S=4K → 每request KV固定 → 抢占代价可控
```

## 5. SchedulerOutput — 调度器输出数据结构

```
SchedulerOutput (dataclass):
  → scheduled_new_reqs: list[NewRequestData] → 新请求(第一次调度)
  → scheduled_cached_reqs: CachedRequestData → 已缓存请求(差量传输!)
  → num_scheduled_tokens: dict[str, int] → 每request的token数
  → total_num_scheduled_tokens: int → 总token数 = sum(dict)
  → scheduled_spec_decode_tokens: dict[str, list[int]] → spec decode tokens
  → scheduled_encoder_inputs: dict[str, list[int]] → 编码器输入索引
  → num_common_prefix_blocks: list[int] → cascade attention前缀blocks
  → finished_req_ids: set[str] → 已完成请求
  → free_encoder_mm_hashes: list[str] → 可释放的编码器缓存
  → preempted_req_ids: set[str] → 被抢占的请求(V2 model runner用)
  → has_structured_output_requests: bool → 结构化输出(async scheduling)
  → pending_structured_output_tokens: bool → grammar bitmask未计算
  → num_invalid_spec_tokens: dict[str, int] → 无效spec tokens
  → kv_connector_metadata → P/D KV传输元数据
  → ec_connector_metadata → encoder cache传输元数据
  → new_block_ids_to_zero → 新blocks需要GPU清零 → 防NaN!

NewRequestData:
  → req_id, prompt_token_ids, mm_features, sampling_params
  → block_ids: tuple[list[int], ...] → KV cache blocks(多组)
  → num_computed_tokens → prefix cache hit后已计算的token数
  → lora_request → LoRA请求
  → prompt_embeds → 自定义embedding
  → prefill_token_ids → V2 model runner用

CachedRequestData (差量传输):
  → req_ids, resumed_req_ids → 哪些是resume(抢占后恢复)
  → new_token_ids: list[list[int]] → PP用的新token ids
  → all_token_ids: dict[str, list[int]] → 所有token ids(KV connector用)
  → new_block_ids: list[tuple] → 新分配的blocks(追加或替换)
  → num_computed_tokens, num_output_tokens → 进度信息

差量传输设计:
  → 新请求: 发送完整NewRequestData → 包含prompt+sampling+blocks
  → 已缓存请求: 只发送CachedRequestData差量 → blocks+token_ids+进度
  → → 减少CPU→GPU通信 → worker缓存request数据 → 不每次重发!
```

## 6. KVConnector Integration — P/D分离调度

```
Scheduler与KVConnector交互:

1. 初始化:
  → KVConnectorFactory.create_connector(role=SCHEDULER)
  → → Scheduler角色 → 管理KV block分配+传输计划
  → → Worker角色 → 执行实际GPU操作

2. 新请求调度(WAITING):
  → connector.get_num_new_matched_tokens(request, num_local_cached)
  → → 查询远程KV cache匹配 → 返回num_external_computed_tokens
  → → None → 无法确定 → 跳过请求 → 等下次
  → → 0 → 无远程匹配 → 仅本地prefix
  → → >0 → 有远程匹配 → 需要async load → WAITING_FOR_REMOTE_KVS

3. Async KV load:
  → request.status = WAITING_FOR_REMOTE_KVS
  → num_computed_tokens = local + external → 预设值
  → _inflight_prefills.add(request) → 标记in-flight
  → 不加入running → 等KV传输完成 → _update_waiting_for_remote_kv()
  → 完成后 → 变回WAITING → 重新调度

4. PD分离特殊处理:
  → EAGLE + async load → limit_lookahead_tokens=0 → 不预留EAGLE slots
  → → 防止本地和远程blocks数量不匹配!
  → reserved_blocks = _inflight_prefill_reserved_blocks() → 预留空间防死锁

5. _build_kv_connector_meta():
  → connector.build_connector_meta(scheduler_output) → 构建KV传输元数据
  → → 包含: push/pull指令 + block映射 + src/dst信息

RTX 4090影响:
  → PD分离需要NVLink → RTX 4090不适合!
  → → connector=None → 无P/D → 单GPU推理最优
  → → NIXL connector → vLLM PR #45157 → 文档贡献
```

## 7. RTX 4090 Scheduler优化建议

```
RTX 4090调度器参数优化:

1. max_num_running_reqs:
  → INT4+INT8KV+GQA-8 → ~118并发 → max_num_running_reqs=118
  → → 但! 太高→抢占频繁 → 建议80-100 → 留headroom
  → BF16 → ~23并发 → max_num_running_reqs=23 → 很少抢占

2. max_num_scheduled_tokens:
  → 7B INT4 B=55 → max_num_scheduled_tokens=55+prefill_budget
  → → 建议=128 (7B decode 55 + prefill 73 tokens)
  → → 小模型可以更高 → 125M → 4096

3. long_prefill_token_threshold:
  → chunked prefill → 建议2048 → 分段prefill → 不阻塞decode
  → → 但chunk=2048慢1.4-3.56x → 小budget下chunk更好

4. enable_chunked_prefill:
  → 必须启用! → 否则prefill占满budget → decode被阻塞 → ITL不稳定
  → → chunked prefill → decode+prefill混合 → ITL更稳定

5. 抢占策略:
  → recomputation(default) → RTX 4090最优 → PCIe swap太慢
  → → 但长prompt → recomputation慢 → 限制prompt长度或budget

6. EAGLE spec decode:
  → num_lookahead_tokens = num_spec_tokens → 额外KV预留
  → → 7B INT4+INT8KV+EAGLE d=5 → 额外5 blocks × 16tok/block = 80tok KV
  → → → KV overhead negligible → 不影响并发

7. 生产最优配置:
  → max_num_running_reqs=80, max_num_scheduled_tokens=128
  → long_prefill_token_threshold=2048, enable_chunked_prefill=True
  → EAGLE d=5, INT4+INT8KV+GQA-8+FlashInfer
  → → 7B: 80并发 × 55 tok/s/request ≈ 4,400 tok/s aggregate
```

## 参考文献

```
1. vLLM V1 Scheduler源码:
   - vllm/v1/core/sched/scheduler.py (~1000行) — Scheduler核心
   - vllm/v1/core/sched/request_queue.py (209行) — FCFS/Priority队列
   - vllm/v1/core/sched/interface.py (245行) — SchedulerInterface ABC
   - vllm/v1/core/sched/output.py (264行) — SchedulerOutput/CachedRequestData

2. vLLM V0 vs V1: Orca/continuous batching → V1 unified token budget
3. SGLang RadixAttention — 对比见scheduler-architecture-deep-dive.md

我们的笔记:
- vllm-v1-kv-cache-architecture-source-reading.md — KV Cache架构
- vllm-v1-spec-decode-architecture-source-reading.md — Spec Decode架构
- scheduler-architecture-deep-dive.md — vLLM vs SGLang调度对比