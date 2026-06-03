# vLLM V1 Scheduler 深度源码阅读

> 基于 `/Users/jackiemac/workspace/rollout-infra/vllm-latest/` 最新源码
> 日期: 2026-06-04

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [调度循环详解](#2-调度循环详解)
3. [请求队列与调度策略](#3-请求队列与调度策略)
4. [Preemption 抢占策略](#4-preemption-抢占策略)
5. [KV Cache 管理](#5-kv-cache-管理)
6. [Prefix Caching 集成](#6-prefix-caching-集成)
7. [Chunked Prefill 机制](#7-chunked-prefill-机制)
8. [关键数据结构](#8-关键数据结构)
9. [源码文件索引](#9-源码文件索引)

---

## 1. 整体架构概览

vLLM V1 的调度器不再区分 "prefill phase" 和 "decode phase"。每个请求只有一个核心概念：
`num_computed_tokens`（已计算的 token 数）需要追赶 `num_tokens_with_spec`（含 spec token 的总 token 数）。
调度器在每个 step 为每个请求决定本次要处理多少 token（可以是整个 prompt、一个 chunk、或 1 个 decode token）。

### 核心组件关系

```
+---------------------------+
|        Scheduler          |  vllm/v1/core/sched/scheduler.py
|  (SchedulerInterface)     |
+---------+--------+--------+
          |        |
    +-----+        +--------+
    |                       |
    v                       v
+-----------+        +------------------+
| Request   |        | KVCacheManager   |  vllm/v1/core/kv_cache_manager.py
| Queues    |        |                  |
|           |        +-------+----------+
| .waiting  |                |
| .skipped  |                v
| .running  |        +------------------+
+-----------+        | KVCacheCoordinator |
     |               |  (kv_cache_       |
     |               |   coordinator.py) |
     |               +-------+-----------+
     |                       |
     |               +-------+-----------+
     |               |                   |
     v               v                   v
+-----------+  +-----------+    +------------------+
| BlockPool |  | SingleType|    | SingleType       |
|           |  | Manager   |    | Manager          |
| (block_   |  | (FullAttn)|    | (SlidingWindow)  |
|  pool.py) |  +-----------+    +------------------+
+-----------+
```

### 调度器核心约束

```python
# 调度器初始化时的关键约束
self.max_num_running_reqs = scheduler_config.max_num_seqs       # 最大并发请求数
self.max_num_scheduled_tokens = scheduler_config.max_num_scheduled_tokens  # 每 step 最大 token 数
self.max_model_len = model_config.max_model_len                 # 模型最大序列长度
```

---

## 2. 调度循环详解

`Scheduler.schedule()` 是调度的核心方法，每次调用对应一次 model forward pass。

### 调度循环 ASCII 图

```
schedule() called
     |
     v
[1] new_step_starts()  -- KVCacheManager 准备新 step
     |
     v
[2] ===== Schedule RUNNING requests =====
     |
     +---> for each request in self.running:
     |       |
     |       [2a] Skip if output_placeholders exhausted max_tokens
     |       [2b] Skip if PP async: current_step < next_decode_eligible_step
     |       [2c] Compute num_new_tokens = num_tokens_with_spec - num_computed_tokens
     |       [2d] Apply long_prefill_token_threshold cap
     |       [2e] Clip by token_budget
     |       [2f] Clip by max_model_len
     |       [2g] Schedule encoder inputs (if multimodal)
     |       [2h] Mamba block-aligned split (if hybrid model)
     |       |
     |       +---> allocate_slots(request, num_new_tokens)
     |       |       |
     |       |       [Success] --> schedule this request
     |       |       [Fail]    --> trigger PREEMPTION loop:
     |       |                     pick lowest-priority running request
     |       |                     _preempt_request() --> free blocks, reset state
     |       |                     retry allocate_slots
     |       |                     (if preempted == current request, give up)
     |       |
     |       +---> Record scheduled_running_reqs
     |       +---> Record spec_decode_tokens (if applicable)
     |       +---> Allocate encoder cache
     |
     v
[3] ===== Schedule WAITING requests =====
     |
     +---> Only if no preemptions happened AND not paused
     |
     +---> while (waiting or skipped_waiting) and token_budget > 0:
     |       |
     |       [3a] Check max_num_running_reqs limit
     |       [3b] _select_waiting_queue_for_scheduling()
     |       [3c] _try_promote_blocked_waiting_request() (for async KV loads)
     |       [3d] Check LoRA max_loras constraint
     |       |
     |       [3e] Get computed blocks (prefix cache hit):
     |            kv_cache_manager.get_computed_blocks(request)
     |            + connector.get_num_new_matched_tokens() (external cache)
     |       |
     |       [3f] Compute num_new_tokens = num_tokens - num_computed_tokens
     |       [3g] Apply long_prefill_token_threshold
     |       [3h] If chunked_prefill disabled AND num_new_tokens > budget: BREAK
     |       [3i] Clip by token_budget
     |       [3j] Schedule encoder inputs
     |       [3k] Mamba block-aligned split
     |       |
     |       +---> allocate_slots(request, num_new_tokens, ...)
     |       |       [Success] --> promote to RUNNING
     |       |       [Fail]    --> BREAK (stop admitting new requests)
     |       |
     |       +---> For async KV loads: set WAITING_FOR_REMOTE_KVS, skip
     |       +---> Otherwise: add to self.running, set status=RUNNING
     |
     v
[4] Compute common prefix blocks (cascade attention)
     |
     v
[5] Construct SchedulerOutput
     |
     v
[6] _update_after_schedule()  -- advance num_computed_tokens
     |
     v
     Return SchedulerOutput
```

### 调度核心代码逻辑 (schedule() 方法关键路径)

**Step 1: 调度 RUNNING 请求** (行 376-550)

```python
# 每个正在运行的请求, 计算本次需要处理的 token 数
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
# 如果超过 long_prefill_token_threshold, 截断
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
# 不超过剩余 token budget
num_new_tokens = min(num_new_tokens, token_budget)
```

**Step 2: 调度 WAITING 请求** (行 562-850)

```python
# 只有没有发生抢占时才调度等待中的请求
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
    while (self.waiting or self.skipped_waiting) and token_budget > 0:
        if len(self.running) == self.max_num_running_reqs:
            break
        # ... prefix cache hit 查询, block 分配等
```

**Step 3: 更新计算进度** (_update_after_schedule)

```python
# 在 schedule() 返回前, 前进 num_computed_tokens
# 这样下一个 step 可以立即开始调度 prefill 的下一个 chunk
for req_id, num_scheduled_token in num_scheduled_tokens.items():
    request.num_computed_tokens += num_scheduled_token
```

---

## 3. 请求队列与调度策略

### 两种调度策略

```python
class SchedulingPolicy(Enum):
    FCFS = "fcfs"        # 先来先服务 (默认)
    PRIORITY = "priority" # 优先级调度
```

### 三个请求队列

```
self.waiting        -- 主等待队列, 新请求入队
self.skipped_waiting -- 被跳过的请求 (异步 KV 加载中, 等待 grammar 等)
self.running        -- 正在运行的请求列表
```

**队列选择逻辑** (`_select_waiting_queue_for_scheduling`):

- **FCFS 模式**: 优先检查 `skipped_waiting`, 再检查 `waiting`
- **PRIORITY 模式**: 比较两个队列头部的请求优先级, 选择优先级更高 (值更小) 的

### 请求队列数据结构

| 队列类型 | FCFS 实现 | PRIORITY 实现 |
|----------|-----------|---------------|
| `waiting` | `FCFSRequestQueue(deque)` | `PriorityRequestQueue(heap)` |
| `skipped_waiting` | `FCFSRequestQueue(deque)` | `PriorityRequestQueue(heap)` |
| `running` | `list[Request]` | `list[Request]` |

FCFS 使用 Python `deque` (双端队列), PRIORITY 使用 `heapq` 最小堆。
优先级排序键为 `(priority, arrival_time)`, priority 值越小优先级越高。

### 请求状态流转

```
                    add_request()
                         |
                         v
            +--- WAITING / skipped_waiting ---+
            |                                 |
            |  (prefix cache hit,             |
            |   block allocation)             |
            |                                 |
            v                                 |
         RUNNING ---------------------------> WAITING
            ^                    preemption   ^
            |                                 |
            |  (async KV transfer done)       |
            +--- WAITING_FOR_REMOTE_KVS ------+
            |                                 |
            +--- WAITING_FOR_STREAMING_REQ ---+
            |                                 |
            +--- WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR --+

         RUNNING --------> FINISHED_STOPPED / FINISHED_LENGTH_CAPPED / ...
```

---

## 4. Preemption 抢占策略

### 什么时候触发抢占?

当 `allocate_slots()` 返回 `None` (没有足够的 free blocks) 时, 调度器开始抢占:

```python
# scheduler.py 行 460-504
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
    if new_blocks is not None:
        break  # 成功分配

    # 分配失败, 需要抢占
    if self.policy == SchedulingPolicy.PRIORITY:
        # 优先级模式: 抢占优先级最低的运行中请求
        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
    else:
        # FCFS 模式: 抢占最后加入的请求 (最新到达的)
        preempted_req = self.running.pop()

    self._preempt_request(preempted_req, scheduled_timestamp)
    preempted_reqs.append(preempted_req)
```

### 抢占策略对比

```
+-------------------+--------------------------------+-------------------------------+
|                   |           FCFS 模式            |         PRIORITY 模式         |
+-------------------+--------------------------------+-------------------------------+
| 抢占目标          | self.running 的最后一个         | 优先级最低且到达最晚的请求     |
| 实现              | self.running.pop()             | max(running, key=priority)    |
| 已调度请求处理    | 无需特殊处理                   | 需要回滚已调度的 token budget  |
|                   |                                | 和 block 分配                  |
+-------------------+--------------------------------+-------------------------------+
```

### 抢占的具体动作 (_preempt_request)

**关键: vLLM V1 只使用 recomputation (重计算) 策略, 不使用 swap!**

```python
def _preempt_request(self, request, timestamp):
    # 1. 释放所有 KV cache blocks
    self.kv_cache_manager.free(request)
    # 2. 释放 encoder cache
    self.encoder_cache_manager.free(request)
    # 3. 重置状态
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0       # 全部重置!
    request.spec_token_ids = []
    request.num_preemptions += 1
    # 4. 放回 waiting 队列头部 (下次优先调度)
    self.waiting.prepend_request(request)
```

### 为什么 V1 不用 Swap?

V0 使用 swap (将 KV cache 搬到 CPU), 但 V1 完全采用 recomputation:
- `num_computed_tokens = 0` 意味着下次调度时需要从头重新计算
- 但如果开启了 prefix caching, 已缓存的前缀 blocks 仍然可以通过 hash 查找复用
- 被抢占的请求放入 `waiting` 队列头部, 下次优先调度

### 抢占后的恢复

```
抢占发生时:
  1. preempted_req 的 blocks 被 free, ref_cnt 归零
  2. preempted_req.num_computed_tokens = 0
  3. preempted_req 进入 waiting 队列头部

下次调度时:
  1. 从 waiting 取出 preempted_req
  2. 重新查询 prefix cache: get_computed_blocks() -- 可命中之前缓存的 blocks
  3. 重新 allocate_slots()
  4. 状态从 PREEMPTED 变为 RUNNING
  5. 被记录为 scheduled_resumed_reqs
```

---

## 5. KV Cache 管理

### 分层管理架构

```
KVCacheManager (kv_cache_manager.py)
    |
    +-- coordinator: KVCacheCoordinator
    |       |
    |       +-- block_pool: BlockPool           -- 物理块池
    |       |       |
    |       |       +-- blocks: list[KVCacheBlock]     -- 所有物理块
    |       |       +-- free_block_queue: FreeKVCacheBlockQueue  -- 空闲块双链表
    |       |       +-- cached_block_hash_to_block: BlockHashToBlockMap  -- hash 索引
    |       |
    |       +-- single_type_managers: tuple[SingleTypeKVCacheManager, ...]
    |               |
    |               +-- FullAttentionManager      -- 全注意力
    |               +-- SlidingWindowManager       -- 滑动窗口注意力
    |               +-- ChunkedLocalAttentionManager  -- 分块局部注意力
    |               +-- MambaManager               -- Mamba 状态
    |               +-- CrossAttentionManager       -- 交叉注意力
```

### Block 分配流程 (allocate_slots)

```
allocate_slots(request, num_new_tokens, ...)
    |
    [Phase 1] 可选的 full_sequence_must_fit 检查
    |   - 计算整个序列需要的 block 数
    |   - 如果 free blocks 不够, 直接返回 None
    |
    [Phase 2] remove_skipped_blocks (释放滑动窗口外的块)
    |   - 对 sliding window / chunked local / mamba 有效
    |   - Full attention 不跳过任何 block
    |
    [Phase 3] get_num_blocks_to_allocate (计算需要新分配的 block 数)
    |   - num_required_blocks = ceil(num_tokens / block_size)
    |   - 减去已有 blocks
    |   - 减去 prefix cache 命中的 blocks
    |
    [Phase 4] 如果不够 free blocks, 返回 None
    |
    [Phase 5] allocate_new_computed_blocks (添加 prefix cache 命中的 blocks)
    |   - touch() 命中的 blocks (增加 ref_cnt)
    |   - 填充 null blocks (sliding window 跳过的部分)
    |
    [Phase 6] allocate_new_blocks (分配新 blocks)
    |   - block_pool.get_new_blocks(num_new_blocks)
    |   - 每个 block.ref_cnt += 1
    |   - 如果 block 有 cached hash: 先 evict 再分配
    |
    [Phase 7] cache_blocks (将已满的 blocks 加入 hash 缓存)
    |   - 仅缓存 finalized tokens (不超过 request.num_tokens)
    |
    v
  返回 KVCacheBlocks 或 None
```

### Block Pool 内存管理

```
+------------------------------------------------------------+
|                    BlockPool                                |
|                                                            |
|  blocks[0..N]:  [B0] [B1] [B2] ... [BN]    所有物理块     |
|                                                            |
|  free_block_queue (LRU 双链表):                            |
|    HEAD <-> [B3] <-> [B7] <-> [B1] <-> ... <-> TAIL       |
|    ^                                          ^            |
|    | 最早可驱逐的 (LRU)     最新释放的 |            |
|                                                            |
|  cached_block_hash_to_block (hash 索引):                   |
|    {hash_key: KVCacheBlock}   或   {hash_key: {id: Block}}|
|                                                            |
+------------------------------------------------------------+
```

### 引用计数 (ref_cnt) 机制

```
Block 的生命周期:

  分配: get_new_blocks()      --> block.ref_cnt = 1
  共享: touch()               --> block.ref_cnt += 1  (prefix cache hit)
  释放: free_blocks()         --> block.ref_cnt -= 1
  回收: ref_cnt == 0 时, block 进入 free_block_queue

关键规则:
  - ref_cnt > 0: block 正在被某个请求使用, 不可驱逐
  - ref_cnt == 0 且有 hash: 是 prefix cache 候选, 可被驱逐
  - ref_cnt == 0 且无 hash: 纯空闲块
```

### Block 释放流程

```python
def free(self, request):
    # 以逆序释放 (tail blocks 先释放, 利于 LRU 驱逐顺序)
    ordered_blocks = reversed(req_blocks)
    self.block_pool.free_blocks(ordered_blocks)
```

```
释放时的 free_block_queue 行为:

假设一个请求持有 blocks [B2, B5, B8]:

  free_blocks([B8, B5, B2])  # 逆序
    |
    v
  free_block_queue:
    ... <-> [B8] <-> [B5] <-> [B2] <-> TAIL

  驱逐优先级: B8 (tail) 先被驱逐, B2 (head) 最后驱逐
  这保证了 prefix 的 blocks 尽可能保留
```

---

## 6. Prefix Caching 集成

### Hash 计算方式

每个 block 的 hash 是链式计算的 (包含前一个 block 的 hash):

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys):
    if not parent_block_hash:
        parent_block_hash = NONE_HASH  # 随机种子
    return BlockHash(
        hash_function((parent_block_hash, tuple(curr_block_token_ids), extra_keys))
    )
```

这意味着 prefix hash 是递归的:
```
Block 0: hash(NONE_HASH, tokens[0:B], extra_keys_0)
Block 1: hash(hash_0, tokens[B:2B], extra_keys_1)
Block 2: hash(hash_1, tokens[2B:3B], extra_keys_2)
...
```

### Extra Keys (影响 hash 的额外因素)

```python
def generate_block_hash_extra_keys(request, start_token_idx, end_token_idx, start_mm_idx):
    extra_keys = (
        lora_extra_keys      # LoRA adapter name
        + mm_extra_keys      # (mm_hash, offset) 多模态特征
        + cache_salt_keys    # cache_salt (仅第一个 block)
        + prompt_embeds_keys  # prompt embeddings 的 sha256 hash
    )
```

### Prefix Cache Hit 查找流程

```
get_computed_blocks(request)
    |
    v
coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)
    |
    +-- UnitaryKVCacheCoordinator (单 KV cache group):
    |     直接遍历 block_hashes, 逐个在 block_pool 中查找
    |
    +-- HybridKVCacheCoordinator (多 KV cache group):
    |     使用不动点迭代算法:
    |     1. 每种 attention type 检查当前候选长度
    |     2. 如果某种 type 缩短了长度, 重新检查所有 type
    |     3. 收敛条件: 长度不再缩短
    |
    v
返回 (hit_blocks, num_hit_tokens)
```

### Full Attention 的 Cache Hit 查找

```python
# FullAttentionManager.find_longest_cache_hit
for block_hash in block_hashes[:max_num_blocks]:
    if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
        computed.append(cached_block)
    else:
        break  # 链式 hash, 一个 miss 后面全部 miss
```

从左到右扫描, 遇到第一个 miss 就停止 -- O(hit_length) 复杂度。

### Sliding Window 的 Cache Hit 查找

```python
# SlidingWindowManager.find_longest_cache_hit
# 从右向左扫描, 找到足够长的连续匹配
for i in range(max_num_blocks - 1, -1, -1):
    if cached_block := block_pool.get_cached_block(block_hashes[i], ...):
        num_contiguous_blocks += 1
        if num_contiguous_blocks >= sliding_window_contiguous_blocks:
            match_found = True
            break
    else:
        num_contiguous_blocks = 0
```

滑动窗口只需要窗口范围内的连续 blocks, 窗口外的用 null blocks 替代。

### Prefix Cache 与调度的集成

```
调度 WAITING 请求时:

1. get_computed_blocks(request)
   |
   v  返回 (new_computed_blocks, num_new_local_computed_tokens)

2. 计算剩余需要计算的 tokens:
   num_new_tokens = request.num_tokens - num_computed_tokens

3. allocate_slots(request, num_new_tokens, new_computed_blocks, ...)
   |
   +-- touch(new_computed_blocks)  # 增加 ref_cnt, 防止驱逐
   +-- 分配额外的新 blocks

4. 更新 num_computed_tokens:
   request.num_computed_tokens = num_computed_tokens  # 包含 cache hits
```

---

## 7. Chunked Prefill 机制

### 核心参数

```python
# 控制 chunked prefill 的关键配置
scheduler_config.long_prefill_token_threshold  # 单次 prefill 的最大 token 数
scheduler_config.enable_chunked_prefill        # 是否启用 chunked prefill
scheduler_config.max_num_scheduled_tokens      # 每 step 的 token budget
```

### Chunked Prefill 调度流程

当一个新请求的 prompt 很长 (超过 `long_prefill_token_threshold`) 时:

```
请求到达: prompt 有 4096 tokens, long_prefill_token_threshold = 1024

Step 1:
  num_computed_tokens = 0
  num_new_tokens = min(4096, 1024, token_budget) = 1024
  allocate_slots(request, 1024)
  --> 分配 ceil(1024/block_size) 个 blocks
  request.num_computed_tokens = 1024

Step 2:
  num_computed_tokens = 1024
  num_new_tokens = min(4096 - 1024, 1024, token_budget) = 1024
  allocate_slots(request, 1024)
  request.num_computed_tokens = 2048

Step 3:
  num_computed_tokens = 2048
  num_new_tokens = min(4096 - 2048, 1024, token_budget) = 1024
  ...

Step 4:
  num_computed_tokens = 3072
  num_new_tokens = min(4096 - 3072, 1024, token_budget) = 1024
  --> Prefill 完成, 开始 decode
```

### Chunked Prefill 的 admission 控制

```python
# 当 chunked_prefill 禁用时
if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > token_budget:
    break  # 直接停止调度, 等下一个 step

# 当 chunked_prefill 启用时
num_new_tokens = min(num_new_tokens, token_budget)  # 按 budget 截断
```

### 全序列准入检查 (full_sequence_must_fit)

```python
# scheduler_reserve_full_isl 配置项
# 防止长序列被分块调度后, 中途发现 KV cache 不够
if full_sequence_must_fit:
    full_num_tokens = min(request.num_tokens, self.max_model_len)
    num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(
        request_id, full_num_tokens, ...
    )
    if num_blocks_to_allocate > block_pool.get_num_free_blocks():
        return None  # 拒绝调度
```

这确保只在 KV cache 能容纳整个序列时才准入请求, 避免中途抢占。

### Chunked Prefill + Running Requests 混合调度

V1 的关键设计: RUNNING 和 WAITING 请求在同一个 step 中可以混合调度。

```
一个 step 的 token budget 分配:

  Budget = max_num_scheduled_tokens
    |
    +-- [先分给 RUNNING 请求]
    |     每个 decode request 只需要 1 token
    |     每个 prefill chunk 请求可能需要 N tokens
    |
    +-- [再分给 WAITING 请求]
          只有 RUNNING 调度后还有 budget 剩余时
          才从 WAITING 队列中取请求
```

**关键**: 如果 RUNNING 请求中有正在做 chunked prefill 的请求, 它会消耗大量 budget, 可能导致 WAITING 队列中的请求无法被调度。

---

## 8. 关键数据结构

### SchedulerOutput

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]       # 本 step 新调度的请求
    scheduled_cached_reqs: CachedRequestData        # 之前已调度的请求的增量更新
    num_scheduled_tokens: dict[str, int]            # {req_id: 本 step token 数}
    total_num_scheduled_tokens: int                  # 本 step 总 token 数
    scheduled_spec_decode_tokens: dict[str, list]   # 投机解码的 draft tokens
    scheduled_encoder_inputs: dict[str, list[int]]  # 需要编码的多模态输入
    num_common_prefix_blocks: list[int]             # cascade attention 的公共前缀
    finished_req_ids: set[str]                      # 上 step 完成的请求
    preempted_req_ids: set[str]                     # 本 step 被抢占的请求
    new_block_ids_to_zero: list[int]                # 需要清零的新 blocks
```

### KVCacheBlock

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                               # 物理 block ID
    ref_cnt: int = 0                            # 引用计数
    _block_hash: BlockHashWithGroupId | None    # hash key (满块才有)
    prev_free_block: KVCacheBlock | None        # 空闲链表前驱
    next_free_block: KVCacheBlock | None        # 空闲链表后继
    is_null: bool = False                       # null block 标记
```

### KVCacheBlocks (调度器与 KVCacheManager 的接口)

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
    # blocks[i][j] = 第 i 个 kv_cache_group 的第 j 个 block

    def get_block_ids(self) -> tuple[list[int], ...]:
        # 转换为 block ID 元组

    def get_unhashed_block_ids(self) -> list[int]:
        # 获取未被 hash 的 block IDs (新分配的 blocks)
```

### Request 状态 (与调度相关的字段)

```
request.num_computed_tokens    -- 已计算的 token 数
request.num_tokens             -- prompt + output token 总数
request.num_tokens_with_spec   -- num_tokens + spec_token_ids
request.num_output_placeholders -- 异步调度中的占位符
request.block_hashes           -- 每个 block 的 hash 值列表
request.status                 -- WAITING / RUNNING / PREEMPTED / FINISHED_*
request.num_preemptions        -- 被抢占的次数
request.priority               -- 优先级 (仅 PRIORITY 模式)
request.arrival_time           -- 到达时间 (FCFS 和 tiebreaker)
```

---

## 9. 源码文件索引

| 文件路径 | 职责 |
|----------|------|
| `vllm/v1/core/sched/scheduler.py` | 主调度器, `schedule()`, `_preempt_request()`, `update_from_output()` |
| `vllm/v1/core/sched/async_scheduler.py` | 异步调度器 (继承 Scheduler), PP + async scheduling |
| `vllm/v1/core/sched/interface.py` | `SchedulerInterface` 抽象基类, `PauseState` 枚举 |
| `vllm/v1/core/sched/output.py` | `SchedulerOutput`, `NewRequestData`, `CachedRequestData` |
| `vllm/v1/core/sched/request_queue.py` | `RequestQueue`, `FCFSRequestQueue`, `PriorityRequestQueue` |
| `vllm/v1/core/sched/utils.py` | `check_stop()`, `remove_all()` 辅助函数 |
| `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` -- 对外接口, `allocate_slots()`, `free()` |
| `vllm/v1/core/block_pool.py` | `BlockPool` -- 物理块池管理, `get_new_blocks()`, `touch()`, `free_blocks()` |
| `vllm/v1/core/kv_cache_coordinator.py` | `KVCacheCoordinator` -- 多 KV cache group 协调 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | 各种 `SingleTypeKVCacheManager` 子类 |
| `vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock`, `FreeKVCacheBlockQueue`, hash 工具函数 |

---

## 附录: 调度循环全景 ASCII 图

```
                    Engine Loop (busy loop)
                         |
            +------------+------------+
            |                         |
            v                         |
       schedule()                     |
            |                         |
            +-- [1] KVCacheManager.new_step_starts()
            |                         |
            +-- [2] Schedule RUNNING  |
            |       (decode + ongoing |
            |        prefill chunks)  |
            |       |                 |
            |       +--> allocate_slots() --+-- success --> record
            |       |                      |
            |       |                      +-- fail --> preempt loop:
            |       |                           pick victim, free blocks
            |       |                           retry allocation
            |       |                         |
            +-- [3] Schedule WAITING  |       |
            |       (new requests +   |       |
            |        preempted reqs)  |       |
            |       |                 |       |
            |       +--> prefix cache hit     |
            |       +--> allocate_slots()     |
            |       +--> promote to RUNNING   |
            |                         |       |
            +-- [4] Build SchedulerOutput     |
            |                         |       |
            +-- [5] _update_after_schedule()  |
            |       (advance computed tokens) |
            |                         |       |
            v                         |       |
       SchedulerOutput --------+     |       |
            |                   |     |       |
            v                   |     |       |
       ModelRunner.execute()    |     |       |
            |                   |     |       |
            v                   |     |       |
       update_from_output() <---+     |       |
            |                         |       |
            +-- Process generated tokens      |
            +-- Check stop conditions         |
            +-- Free finished requests        |
            +-- Return EngineCoreOutputs       |
                                        |     |
                                        v     v
                                   Back to schedule()
```

---

## 总结

1. **统一调度**: V1 不区分 prefill/decode phase, 每个请求通过 `num_computed_tokens` 追赶 `num_tokens_with_spec` 来驱动调度

2. **Recomputation-only preemption**: 不使用 swap, 抢占时直接释放所有 blocks 并重置 `num_computed_tokens=0`; 但 prefix caching 可以加速恢复

3. **Chunked prefill**: 通过 `long_prefill_token_threshold` 控制每次 prefill 的最大 token 数, 多 step 完成 prompt 计算

4. **Block 管理**: 引用计数 + LRU 空闲队列 + hash 索引三层管理; prefix caching 通过链式 hash 实现跨请求的 block 共享

5. **混合模型支持**: 通过 `KVCacheCoordinator` 协调多种 attention type (Full, SWA, Mamba, ChunkedLocal), 各自管理不同的 block 分配策略

6. **异步调度**: `AsyncScheduler` 支持投机解码和 Pipeline Parallel 的异步调度模式, 通过 `num_output_placeholders` 跟踪在途 token
