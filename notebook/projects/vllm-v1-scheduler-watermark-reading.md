# vLLM V1 Scheduler + Watermark Optimization 深度阅读

> 2026-06-16 (updated) | vLLM V1调度器核心机制 + PR #44594 watermark优化 + thrashing prevention 4层机制 + BudgetRefiner互补分析
> 源码: vllm/v1/core/sched/scheduler.py (~2400行) + kv_cache_manager.py (~430行) + block_pool.py
> 关联: vllm-v1-preemption-source-reading.md, vllm-prefix-caching-v1.md, vllm-prefix-cache-hash-collision-reading.md, budgetrefiner-vllm-pr-draft.md

---

## 0. 核心定律: Unified Token Budget + RUNNING优先 + Watermark Headroom

```
★★★★★ V1调度算法三大核心:
  1. Unified Token Budget → 无prefill/decode phase区分 → 每个request有num_computed_tokens
  2. RUNNING优先 → decode先分配token → WAITING(prefill)用剩余budget → decode不被阻塞!
  3. Watermark Headroom → WAITING/PREEMPTED请求admission时保留5% blocks → 减少thrashing

★★★★★ PR #44594核心成果 (2026-06-11 merged):
  watermark=0.05 → preemption -82% → ITL p99 -56% → throughput +5.1% → E2EL p50 -7%

调度流程:
  Phase 1: 遍历RUNNING → decode请求 → 分配1+spec tokens → KV block分配 → 无则preempt
  Phase 2: 遍历WAITING → prefill请求 → prefix cache hit → chunk → KV block分配 + watermark
  Phase 3: 构造SchedulerOutput → new_reqs + cached_reqs + spec_decode_tokens + connector_meta

关键约束:
  → max_num_scheduled_tokens (token budget per step)
  → max_num_running_reqs → 限制并发request数
  → watermark_blocks = int(watermark × num_total_blocks) → WAITING/PREEMPTED的额外headroom
  → LoRA max_loras → 限制同时激活的LoRA数量
```

---

## 1. schedule() 方法 — 两阶段调度核心

### 1.1 Phase 1: RUNNING请求调度 (lines 376-551)

```python
★★★★★ schedule() Phase 1 — RUNNING优先调度 (scheduler.py L376-551):

token_budget = self.max_num_scheduled_tokens

# First, schedule the RUNNING requests.
req_index = 0
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
    
    # 计算新token数
    num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
    if 0 < long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = long_prefill_token_threshold  # chunk限制
    num_new_tokens = min(num_new_tokens, token_budget, max_model_len - 1 - request.num_computed_tokens)
    
    # ★★★ 分配KV blocks — 如果失败则preempt!
    while True:
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens
        )
        if new_blocks is not None:
            break  # ★★★ 成功分配!
        
        # ★★★★★ 分配失败 → preempt最低优先级request → 释放blocks → retry
        if PRIORITY policy:
            preempted_req = max(running, key=lambda r: (r.priority, arrival_time))
        else:  # FCFS (默认)
            preempted_req = running.pop()  # 淘汰最近admitted
        
        _preempt_request(preempted_req)
        if preempted_req == request: break  # ★★★ 自我淘汰 → 无法调度
    
    # 成功 → 加入scheduled_running_reqs
    scheduled_running_reqs.append(request)
    token_budget -= num_new_tokens
```

★★★★★ 关键设计:
  → RUNNING请求优先分配token_budget → decode不被prefill阻塞
  → KV block分配失败 → preempt最低优先级RUNNING request → 释放blocks → 重试
  → 自我淘汰检测 → preempted_req == request → break → 防止死循环

### 1.2 Phase 2: WAITING请求调度 (lines 562-868)

```python
★★★★★ schedule() Phase 2 — WAITING请求调度 (scheduler.py L562-868):

# ★★★★★ 只在没有preemption且pause_state==UNPAUSED时才调度WAITING!
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
    while (self.waiting or self.skipped_waiting) and token_budget > 0:
        if len(self.running) == self.max_num_running_reqs:
            break
        
        request = request_queue.peek_request()
        
        # ★★★ prefix cache hit检测 (num_computed_tokens==0时)
        if request.num_computed_tokens == 0:
            new_computed_blocks, num_new_local_computed_tokens = (
                self.kv_cache_manager.get_computed_blocks(request)
            )
            # KV Connector: 远程prefix hit (P/D分离)
            if self.connector is not None:
                ext_tokens, load_kv_async = connector.get_num_new_matched_tokens(request, ...)
        
        # 计算新token数
        num_new_tokens = request.num_tokens - num_computed_tokens
        if 0 < threshold < num_new_tokens:
            num_new_tokens = threshold  # chunked prefill
        
        # ★★★★★ allocate_slots — WATERMARK在这里生效!
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens, ...,
            reserved_blocks=reserved_blocks,
            has_scheduled_reqs=bool(self.running),  # ★★★ PR #44594新增参数!
        )
        
        if new_blocks is None:
            break  # ★★★ 不preempt running! → 只停止admission!
        
        # 成功 → 加入running → WAITING→RUNNING
        self.running.append(request)
        token_budget -= num_new_tokens
```

★★★★★ Phase 2关键设计:
  → 只有Phase 1没有preemption时才调度 → 防止新请求抢占刚释放的空间
  → allocate_slots失败 → break → 不preempt running → WAITING不会抢占RUNNING
  → ★★★★★ has_scheduled_reqs=True → watermark_blocks生效 → 保留headroom!
  → LoRA限制 → max_loras → 超过则skip

### 1.3 Phase 3: 构造SchedulerOutput (lines 883-)

```python
★★★★ Phase 3 — 构造SchedulerOutput:

# ★★★ cascade attention — 计算所有running请求的最长公共prefix blocks数
num_common_prefix_blocks = (
    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
)

# ★★★ 差量传输设计:
# NewRequestData → 新请求完整数据 (prompt+sampling+blocks)
# CachedRequestData → 已缓存请求差量 (blocks+token_ids+进度)
# → 减少CPU→GPU通信 → worker缓存request数据 → 不每次重发!
```

---

## 2. PR #44594 — Watermark Optimization 源码级深度分析

### 2.1 PR基本信息

```
★★★★★ PR #44594:
  Title: [Core] Add kvcache watermark to reduce preemptions
  Author: njhill
  Created: 2026-06-05
  Merged: 2026-06-11 (merged_at: 2026-06-11T15:27:32Z)
  ★★★ Revert PR #45344 → NOT merged → 原PR保持merged!

  ★★★★★ 核心思想:
  KV cache watermark = 总blocks的固定比例 → admission时保留 → running请求有增长空间
  → 减少preemption触发 → 减少thrashing → 提升吞吐+降低延迟!
```

### 2.2 Benchmark结果 (★★★★★ 官方数据)

```
★★★★★ Benchmark Config:
  Model: Qwen/Qwen2.5-7B-Instruct, TP=1, 1× NVIDIA GB200
  KV cache: 16 GiB → 299,584 tokens (~1.5× mean demand → near-critical!)
  Engine: --max-model-len 8192, --max-num-seqs 256, chunked prefill on
  Workload: input ~1000 / output ~5000 (decode-heavy!), max-concurrency 128
  ★★★ Decode-heavy → 被preempted的请求resume时需re-prefill ~3500 tokens → 高recompute代价!

★★★★★ Results Table:

| watermark | preempt | out tok/s | req/s | TTFT p50 (s) | TTFT p99 (s) | ITL p99 (ms) | E2EL p50 (s) |
|----------:|--------:|----------:|------:|-------------:|-------------:|-------------:|-------------:|
| off (0)   |    1065 |     11582 | 2.306 |         6.56 |        25.51 |        40.62 |         51.6 |
| 0.02      |     400 |     12134 | 2.416 |        11.80 |        26.49 |        18.80 |         48.4 |
| 0.05      |     187 |     12167 | 2.423 |        11.20 |        28.94 |        17.70 |         47.9 |
| 0.10      |     168 |     11835 | 2.356 |         8.96 |        31.02 |        17.57 |         48.6 |
| 0.15      |     168 |     11635 | 2.317 |         9.54 |        30.05 |        10.41 |         49.1 |

★★★★★ Sweet spot = watermark=0.05:
  → Preemptions: 1065 → 187 → ★★★★★ -82.4%!
  → ITL p99: 40.62ms → 17.70ms → ★★★★★ -56.3%!
  → Throughput: 11582 → 12167 tok/s → ★★★★ +5.1%!
  → E2EL p50: 51.6s → 47.9s → ★★★ -7.2%

★★★★★ Trade-off分析:
  → TTFT p50: 6.56→11.20s → 增加71% → 因为admission更保守 → 请求等待更久才被admitted
  → TTFT p99: 25.51→28.94s → 增加14% → 类似原因
  → 但! ITL p99大幅降低 → decode阶段稳定 → 总体E2EL反而更快!

  ★★★★★ 结论: watermark=0.05 →牺牲TTFT换取ITL和E2EL → 对decode-heavy workload最优!
```

### 2.3 代码变更 (★★★★★ 精确diff分析)

```
★★★★★ PR #44594代码变更 — 7个文件:

1. vllm/config/scheduler.py:
   + watermark: float = Field(default=0.0, ge=0.0, lt=1.0)
   "Fraction of total KV cache blocks to keep free (the watermark) when
   admitting waiting or preempted requests into the running queue.
   This headroom helps avoid frequent KV cache eviction and the resulting
   repeated preemption of requests when GPU memory is scarce.
   Must be in the range [0.0, 1.0); 0.0 (the default) disables the watermark."

2. vllm/engine/arg_utils.py:
   + --watermark CLI参数 → 传给SchedulerConfig

3. vllm/v1/core/kv_cache_manager.py:
   + __init__: watermark: float = 0.0 → self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
   + allocate_slots: has_scheduled_reqs: bool = True → 新参数
   + ★★★★★ watermark计算逻辑:
     watermark_blocks = 0
     if has_scheduled_reqs and request.status in (WAITING, PREEMPTED):
         watermark_blocks = self.watermark_blocks
     → ★★★★★ 只对WAITING/PREEMPTED请求生效! RUNNING请求不受影响!

   + ★★★★★ admission检查:
     required_blocks = num_blocks_to_allocate + watermark_blocks
     if required_blocks > available_blocks: return None
     → ★★★★★ 增加watermark_blocks → admission更难 → 但running有更多headroom → 减少thrashing!

4. vllm/v1/core/sched/scheduler.py:
   + __init__: watermark=self.scheduler_config.watermark → 传给KVCacheManager
   + WAITING scheduling: has_scheduled_reqs=bool(self.running) → 新参数
   → ★★★ 第一个新请求(cache为空) → has_scheduled_reqs=True → 但watermark_blocks仍生效
   → ★★★★★ RUNNING scheduling → 无has_scheduled_reqs参数 → watermark不生效 → 优先保证decode!

5. benchmarks/kv_cache_watermark.sh → 新增benchmark脚本 (248行)
6. tests/v1/core/test_scheduler.py → 新增测试
7. tests/v1/core/utils.py → 测试工具更新
```

### 2.4 PR Evolution History (★★★★★ 从proportional到simple fraction)

```
★★★★★ PR #44594提交历史 (9个commits):

Commit timeline:
  2026-06-05 5deb9bdd — 初始版本: simple fraction (watermark_blocks = int(watermark * num_blocks))
  2026-06-07 0753b81d — ★★★ 改为proportional to running reqs, with 4% cap!
  2026-06-07 de744518 — Merge main
  2026-06-10 ca66c1de — ★★★★★ 回退到simple fraction, 加benchmark script!
  2026-06-10 f40035e1 — Merge main
  2026-06-10 baef4ce1 — disable by default (watermark=0.0)
  2026-06-10 d5156f5c — ★★★★★ disable when no requests scheduled (has_scheduled_reqs参数!)
  2026-06-10 b27bec03 — update benchmark
  2026-06-10 f1b1762f — Merge main
  2026-06-11 MERGED (merge commit: 4085ff7c)

★★★★★ Key evolution decisions:
  → 最初simple fraction → njhill改为proportional(to running reqs + 4% cap) → WoosukKwon审查
  → → njhill说: "I have updated to take into account number of running requests"
  → → 但最终回退到simple fraction → 原因: simple更直观 → 无需estimate per-request growth!
  → → ★★★★★ has_scheduled_reqs参数 → 最终版: bool(self.running) → 当running空时不apply!
  → → → ★★★ 首个请求(cache为空) → has_scheduled_reqs=False → watermark不生效 → 保证首个请求可admit!
  → → → ★★★★★ 有running请求时 → has_scheduled_reqs=True → watermark生效 → 保守admission!

★★★★ njhill PR comment原文:
  "@WoosukKwon I have updated to take into account number of running requests,
   as we discussed. The parameter is now a scaling factor for determining
   num tokens per request to include in the watermark. The total headroom
   is capped at 4% of total kv cache size."
  → 但最终版不是这个! → 回退到simple fraction → 比proportional更简洁!
```

### 2.5 Revert PR #45344 Status (★★★★)

```
★★★★ Revert PR #45344 — OPEN (NOT merged!):

  Title: Revert "[Core] Add kvcache watermark to reduce preemptions" (#44594)
  State: OPEN
  ★★★★★ 原PR #44594保持merged! → revert未成功!

  ★★★★★ CI failure reason (why revert was proposed):
  → has_scheduled_reqs parameter added to allocate_slots()
  → test mock fake_allocate_slots_fn in test_mamba_prefix_cache.py NOT updated!
  → TypeError: fake_allocate_slots_fn() got an unexpected keyword argument 'has_scheduled_reqs'
  → ★★★ 这只是mock问题 → 不是功能bug → fix应该是更新mock → 而不是revert整个PR!

  ★★★★★ 教训:
  → 任何allocate_slots参数变更 → 必须更新所有mock/测试 → 否则CI fail
  → revert PR存在但不merged → watermark功能仍在upstream → 可安全使用!
  → → RTX 4090部署: watermark=0.05有效 → 但需确认vllm版本包含此PR (v0.22.0+)
```

### 2.6 Watermark为什么有效 (★★★★★ 深度分析)

```
★★★★★ Thrashing根因分析:
  
  问题: Decode-heavy workload + 高并发 → over-admission
  → Requests被admitted时只占用少量blocks (刚prefill完)
  → 但随着decode → output tokens增长 → KV blocks需求持续增长
  → 多个请求同时增长 → blocks不够 → preemption → release blocks
  → 被preempted的请求 → prepend到waiting → 下step重新admitted
  → 重新admitted → 需re-prefill ~3500 tokens → 大量blocks → 又触发preemption!
  → ★★★★★ 循环 → thrashing → preemption 1065次!

★★★★★ Watermark如何打破循环:
  
  1. admission时保留watermark_blocks (5%总blocks) → running请求有增长空间
  2. decode增长 → blocks需求增加 → 但有5%headroom → 不会立即触发preemption
  3. requests自然完成 → 释放blocks → 给其他running请求更多空间
  4. ★★★★★ 结果: preemption从1065→187 → thrashing大幅减少!

★★★★★ 为什么watermark=0.05是sweet spot:
  
  → 0.02 → headroom太小 → 仍有400次preemption →不够!
  → 0.05 → headroom适中 → 187次preemption → 接近最优!
  → 0.10 → headroom太大 → 减少admission → throughput下降(11835→11635)
  → 0.15 → headroom过大 → admission太少 → throughput更低
  
  → ★★★★★ 5% = running请求decode增长所需blocks的估计 → 太少不够 → 太多浪费!
```

---

## 2.7 Exact Source-Level Code Trace: BEFORE vs AFTER PR #44594

### 2.7.1 Current checkout (BEFORE watermark) — scheduler.py

```python
★★★★★ Scheduler.__init__ — BEFORE watermark (scheduler.py L230-243):

self.kv_cache_manager = KVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=self.max_model_len,
    max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
    enable_caching=self.cache_config.enable_prefix_caching,
    use_eagle=self.use_eagle,
    log_stats=self.log_stats,
    enable_kv_cache_events=self.enable_kv_cache_events,
    dcp_world_size=self.dcp_world_size,
    pcp_world_size=self.pcp_world_size,
    scheduler_block_size=self.block_size,
    hash_block_size=hash_block_size,
    metrics_collector=self.kv_metrics_collector,
)
# ★★★ NOTE: watermark NOT passed → default=0.0 → no headroom!

★★★★★ AFTER watermark (PR diff, scheduler.py L241-243):

    ...
    metrics_collector=self.kv_metrics_collector,
    watermark=self.scheduler_config.watermark,  # ★★★★★ NEW LINE!
)
```

### 2.7.2 Current checkout (BEFORE) — kv_cache_manager.py

```python
★★★★★ KVCacheManager.__init__ — BEFORE watermark (kv_cache_manager.py L122-157):

def __init__(self, ..., metrics_collector=None):  # ★★★ No watermark param!
    self.max_model_len = max_model_len
    ...
    self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
    self.block_pool = self.coordinator.block_pool
    self.kv_cache_config = kv_cache_config
    # ★★★ NOTE: No watermark_blocks computed!

★★★★★ AFTER watermark (PR diff, kv_cache_manager.py L124-161):

def __init__(self, ..., metrics_collector=None, watermark: float = 0.0):  # ★★★ NEW param!
    ...
    self.kv_cache_config = kv_cache_config
    # ★★★★★ NEW: Watermark block computation!
    assert watermark >= 0.0, "watermark must be non-negative"
    self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
    # → ★★★★★ watermark=0.05 → watermark_blocks = int(0.05 * num_total_blocks)
    # → e.g., 299584 tokens / 16 block_size = ~18724 blocks → watermark_blocks ≈ 936 blocks
```

### 2.7.3 Current checkout (BEFORE) — allocate_slots

```python
★★★★★ allocate_slots — BEFORE watermark (kv_cache_manager.py L238-392):

def allocate_slots(self, request, num_new_tokens, ...,
                   full_sequence_must_fit=False, reserved_blocks=0):
    # ★★★ NOTE: No has_scheduled_reqs param!

    if full_sequence_must_fit:
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(...)
        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # ★★★ No watermark_blocks added to check!
            return None

    # Regular allocation check:
    if num_blocks_to_allocate > self.block_pool.get_num_free_blocks() - reserved_blocks:
        # ★★★ No watermark_blocks added to check!
        return None

★★★★★ AFTER watermark (PR diff, kv_cache_manager.py L253-414):

def allocate_slots(self, request, num_new_tokens, ...,
                   reserved_blocks=0, has_scheduled_reqs: bool = True):
    # ★★★★★ NEW param: has_scheduled_reqs

    # ★★★★★ NEW: Watermark computation!
    watermark_blocks = 0
    if has_scheduled_reqs and request.status in (
        RequestStatus.WAITING,
        RequestStatus.PREEMPTED,
    ):
        watermark_blocks = self.watermark_blocks

    if full_sequence_must_fit:
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(...)
        # ★★★★★ CHANGED: watermark_blocks added!
        required_blocks = num_blocks_to_allocate + watermark_blocks
        if required_blocks > self.block_pool.get_num_free_blocks():
            return None

    # ★★★★★ CHANGED: watermark_blocks added + reserved_blocks documented!
    available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
    required_blocks = num_blocks_to_allocate + watermark_blocks
    if required_blocks > available_blocks:
        return None
```

### 2.7.4 Current checkout (BEFORE) — scheduler.py WAITING scheduling

```python
★★★★★ WAITING scheduling — BEFORE watermark (scheduler.py L750-770):

new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens,
    new_computed_blocks=new_computed_blocks,
    num_lookahead_tokens=effective_lookahead_tokens,
    num_external_computed_tokens=num_external_computed_tokens,
    delay_cache_blocks=load_kv_async,
    num_encoder_tokens=num_encoder_tokens,
    full_sequence_must_fit=self.scheduler_reserve_full_isl,
    reserved_blocks=reserved_blocks,
    # ★★★ NOTE: No has_scheduled_reqs!
)

★★★★★ AFTER watermark (PR diff, scheduler.py L750-793):

new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    ...
    reserved_blocks=reserved_blocks,
    has_scheduled_reqs=bool(self.running),  # ★★★★★ NEW LINE!
)
# → ★★★★★ bool(self.running) → True when any running requests exist
# → → First request (empty cache) → self.running=[] → bool=False → watermark NOT applied!
# → → Subsequent requests → self.running=[req1,...] → bool=True → watermark applied!
# → → ★★★★★ Ensures first request always gets admitted → prevents cold-start deadlock!
```

### 2.7.5 Current checkout (BEFORE) — config/scheduler.py

```python
★★★★★ SchedulerConfig — BEFORE watermark (config/scheduler.py L143-150):

    scheduler_reserve_full_isl: bool = True
    """If True, the scheduler checks whether the full input sequence length fits..."""

    async_scheduling: bool | None = None
    # ★★★ NOTE: No watermark field!

★★★★★ AFTER watermark (PR diff, config/scheduler.py L143-150):

    scheduler_reserve_full_isl: bool = True
    """If True, the scheduler checks whether the full input sequence length fits..."""

    watermark: float = Field(default=0.0, ge=0.0, lt=1.0)
    """Fraction of total KV cache blocks to keep free (the watermark) when
    admitting waiting or preempted requests into the running queue. This headroom
    helps avoid frequent KV cache eviction and the resulting repeated preemption
    of requests when GPU memory is scarce. Must be in the range [0.0, 1.0); 0.0
    (the default) disables the watermark."""
    # ★★★★★ Default=0.0 → DISABLED by default → must explicitly set --watermark 0.05!
```

### 2.7.6 Current checkout (BEFORE) — _inflight_prefill_reserved_blocks

```python
★★★★★ _inflight_prefill_reserved_blocks — BEFORE watermark (scheduler.py L2162-2170):

    def _inflight_prefill_reserved_blocks(self) -> int:
        """Blocks in-flight prefills still need to finish (their reservation).

        Sums remaining full-ISL blocks over `self._inflight_prefills` (running
        prefills + in-progress async loads). The candidate async load isn't yet
        in the set, so it's naturally excluded.
        """
        return sum(...)

★★★★★ AFTER watermark (PR diff, simplified docstring):

    def _inflight_prefill_reserved_blocks(self) -> int:
        """Num blocks in-flight prefills still need to finish (their reservation)."""
        # ★★★ Docstring simplified → 5-line docstring → 1-line
        # → ★★★ The longer docstring was removed → more concise
```

---

## 3. Thrashing Prevention 4层机制 (★★★★★)

### 3.1 Layer 1: Admission Gate after Preemption

```
★★★★★ Layer 1 — Admission gate (scheduler.py L562-563):

if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
    # ★★★★ 如果Phase 1有preemption → 完全跳过WAITING scheduling!

★★★★★ 关键行为:
  → Phase 1 preempt了一个request → Phase 2不admit任何新请求
  → → 释放的blocks不被新请求立即填充 → preempted请求有"呼吸空间"
  → → 下一步preempted请求prepend到waiting前端 → 优先重新admit → 有blocks可用

★★★★★ 效果:
  → 防止"抢占→admit→又抢占"的快速thrashing循环
  → 但! 长output请求被preempted → 重计算需要大量blocks → 可能仍然再次触发!
```

### 3.2 Layer 2: Watermark for WAITING/PREEMPTED

```
★★★★★ Layer 2 — Watermark headroom (PR #44594):

watermark_blocks = self.watermark_blocks  # int(watermark × num_total_blocks)
if has_scheduled_reqs and request.status in (WAITING, PREEMPTED):
    watermark_blocks = self.watermark_blocks

required_blocks = num_blocks_to_allocate + watermark_blocks
if required_blocks > available_blocks: return None

★★★★★ 关键行为:
  → WAITING/PREEMPTED请求admission时需要额外5% blocks → admission更保守
  → RUNNING请求不受watermark限制 → decode优先保证!
  → ★★★★★ 保留headroom → running请求decode增长时不会立即触发preemption
  
★★★★★ 效果:
  → preemption减少82% → ITL p99减少56% → throughput增加5.1%
  → ★★★★★ 最有效的thrashing prevention层!
```

### 3.3 Layer 3: Prefix Cache Sharing

```
★★★★★ Layer 3 — Prefix cache sharing (ref_cnt保护):

★★★★ block_pool.free_blocks (block_pool.py L419-441):
def free_blocks(self, ordered_blocks, prepend=False):
    for block in blocks_list:
        block.ref_cnt -= 1  # ★★★ 减1 → 不直接释放!
    freed_blocks = [b for b in blocks_list if b.ref_cnt == 0 and not b.is_null]
    # ★★★★ 只有ref_cnt==0的block才回到free queue!

★★★★★ Preemption后的prefix cache行为:
  1. Shared prefix block (ref_cnt=2 → 两个请求共享):
     → free → ref_cnt减1→=1 → ★★★ 不回free queue → KV数据完好!
     → 重新admit → get_computed_blocks() → hash查找 → touch(ref_cnt+=1) → ★★★ 部分重计算!
  
  2. Non-shared prefix block (ref_cnt=1 → 只一个请求):
     → free → ref_cnt减1→=0 → ★★★ 回free queue → KV数据丢失!
     → 重新admit → 完全重新计算 → 包括system prompt → ★★★★ 2×慢!

★★★★ SlidingWindowManager.free优化:
  → cached blocks → free到queue尾部 → 保持prefix cache → 尽久存活
  → uncached blocks → free到queue前端 → 优先reuse → 不保prefix
  → ★★★ uncached优先reuse → cached保到最后 → prefix cache寿命最大化!

★★★★★ 效果:
  → 减少preemption后的重计算量 → shared prefix不需要重计算 → 加速恢复
  → → 减少再次触发preemption的几率 → 缓解thrashing → 但不能根治!
  → ★★★★★ RTX 4090: enable_prefix_caching=True → GRPO共享system prompt → MUST!
```

### 3.4 Layer 4: Self-Preemption Detection

```
★★★★★ Layer 4 — Self-detection (scheduler.py L503-509):

while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, ...)
    if new_blocks is not None: break  # 成功
    
    preempted_req = ...  # preempt最低优先级
    self._preempt_request(preempted_req)
    preempted_reqs.append(preempted_req)
    
    if preempted_req == request: break  # ★★★★★ 自我淘汰 → 停止!

★★★★★ 关键行为:
  → 如果淘汰的是自己 → break → 不再尝试分配
  → → 防止"淘汰自己给自己腾空间"的荒谬循环!
  → → request变为PREEMPTED → prepend到waiting → 下step重新调度

★★★★★ 效果:
  → 防止单个request的无意义preemption循环
  → ★★★ 最基本的thrashing prevention → 但无法防止跨request thrashing!
```

### 3.5 4层机制的局限与残余thrashing风险

```
★★★★★ 残余thrashing风险:
  
  → ★★★ 长output请求(3500 tokens)被preempted → 重计算3500 tokens → 大量blocks → 可能再次触发!
  → → RTX 4090(24GB): 在<1.5× mean demand时 → thrashing仍可能!
  → → ★★★★★ 解决: reduce max_num_seqs + watermark=0.05 + enable prefix caching!

★★★★ 4层机制的协同效果:
  Layer 1 (admission gate) → 防止preemption后立即admit → 给preempted请求"呼吸空间"
  Layer 2 (watermark) → admission时保留5% headroom → running请求有增长空间 → ★★★★★ 最关键!
  Layer 3 (prefix sharing) → preemption后shared prefix保留 → 减少重计算 → 加速恢复
  Layer 4 (self-detection) → 防止自己淘汰自己 → 最基本防护
  
  ★★★★★ 4层+max_num_seqs限制+INT8 KV → 综合防护 → RTX 4090可稳定serving!
```

---

## 4. Preemption = Retraction (纯重计算) 源码级分析

### 4.1 _preempt_request精确行为

```python
★★★★★ _preempt_request (scheduler.py L974-995):

def _preempt_request(self, request, timestamp):
    # ★★★ request必须已经从running queue移除!
    assert request.status == RequestStatus.RUNNING
    
    self.kv_cache_manager.free(request)         # ★★★ Free ALL KV blocks (ref_cnt -= 1)
    self.encoder_cache_manager.free(request)    # Free encoder cache (vision等)
    self._inflight_prefills.discard(request)    # Remove from prefill tracking
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0             # ★★★★★ 重置为0 → 从头来过!
    request.spec_token_ids = []                 # Clear speculative tokens
    request.num_preemptions += 1               # Increment counter
    self.waiting.prepend_request(request)       # ★★★ 前端插入 → 优先重新admit
```

★★★★★ 关键行为:
  → num_computed_tokens = 0 → 完全重新计算 → 无swap → 无CPU offload!
  → prepend_request → deque.appendleft → 优先重新admit → 但与新请求竞争!
  → ★★★★★ V1 = 纯retraction → KV完全丢弃 → 只靠prefix cache缓解!

### 4.2 V0 vs V1 Preemption策略对比

```
★★★★ V0 Preemption:
  → 支持 swap → KV blocks搬到CPU → 稍后swap back → 不重计算
  → → PCIe带宽瓶颈 → swap慢 → RTX 4090不适合!
  → → recomputation → 重计算 → 但比swap快 (短prompt)

★★★★★ V1 Preemption:
  → ★★★★★ 纯retraction → 不支持swap → KV完全丢弃 → 重计算!
  → → 设计原因: PCIe swap太慢 → recompute更快 (尤其有prefix cache时)
  → → 但! 长output request → 重计算慢 → preemption代价高 → ★★★ 需watermark减少!

★★★★★ RTX 4090最优:
  → retraction+prefix cache → 短prompt快恢复 → 长prompt+prefix → 部分恢复
  → INT8 KV → KV占用少 → 更多blocks → preemption少 → 更稳定
  → ★★★★★ 限制max_num_seqs → 减少并发 → 减少preemption频率
```

---

## 5. Prefix Caching Hash Collision (#44701) — 与Scheduler交互

### 5.1 Hash Collision根因

```
★★★★★ Issue #44701 — Domain collision between LoRA name and cache_salt:

★★★★★ Collision机制:
  extra_keys = (*lora_keys, *mm_extra_keys, *cache_salt_keys, *prompt_embeds_keys)
  LoRA name和cache_salt都是bare string → 放入同一个flat tuple → 无domain分离!

  ★★★★★ Collision示例 (confirmed on A100):
  Request A: LoRA name="COLLIDE_SALT", cache_salt=None
  → extra_keys = ("COLLIDE_SALT",)
  
  Request B: LoRA=None, cache_salt="COLLIDE_SALT"
  → extra_keys = ("COLLIDE_SALT",) ← ★★★★★ 完全相同! → hash collision!

  → block_hash=cae16dc07873c36e9b370e40 → shared between base model and LoRA!
  → ★★★★★ Base request consumed LoRA-produced KV block → silent correctness bug!

★★★★★ Chained hash amplification:
  → collision只在Block 0 (cache_salt只在start_token_idx==0时加入)
  → 但hash是chained → Block 0 collision → 所有后续block hash也相同!
  → → ★★★★★ 整个prefix被corrupt → 模型输出错误 → 无任何错误信号!

★★★★★ Collision影响路径 → scheduler → allocate_slots → prefix cache:
  1. Request B → scheduler Phase 2 → get_computed_blocks()
  2. → 查找BlockHashToBlockMap → hit Request A的block → prefix hit!
  3. → touch(ref_cnt+=1) → 共享A的KV blocks → num_new_local_computed_tokens>0
  4. → ★★★★★ num_new_tokens减少 → 分配更少blocks → admission更容易 → 但KV是错的!
```

### 5.2 与Watermark的交互

```
★★★★ Hash collision + watermark交互:
  
  → prefix cache hit → num_computed_tokens增加 → num_new_tokens减少
  → → num_blocks_to_allocate减少 → required_blocks(num_blocks + watermark)更少
  → → admission更容易 → 但如果hit是collision → KV是错的 → ★★★ 加剧影响!
  
  → ★★★★★ 但! collision概率极低 → 只在LoRA name恰好等于cache_salt时
  → → RTX 4090 GRPO → 同adapter → extra_keys相同 → prefix共享正确 → collision不影响!
  → → Multi-tenant → collision概率低但非零 → correctness bug!

★★★★★ PR #44706 fix方向 (domain-tag):
  → ("lora", adapter_name) + ("cache_salt", salt_value) → domain separation → 数学保证安全
  → ★★★★ ~10行代码 → 最小变更 → 最大安全 → 正确方向
```

---

## 6. Scheduler与KV Cache Manager交互链路

### 6.1 核心交互路径

```
★★★★★ Scheduler → KVCacheManager → BlockPool → FreeKVCacheBlockQueue 完整链路:

1. schedule() Phase 1 (RUNNING):
   → kv_cache_manager.allocate_slots(request, num_new_tokens, num_lookahead_tokens)
   → → KVCacheManager.calculate_num_blocks() → 计算需要多少blocks
   → → BlockPool.get_num_free_blocks() → 检查可用blocks
   → → required_blocks > available_blocks → return None → 触发preemption

2. schedule() Phase 2 (WAITING):
   → kv_cache_manager.get_computed_blocks(request) → prefix cache hit
   → → coordinator.find_longest_cache_hit() → hash chain查找 → 返回computed blocks
   → → touch(ref_cnt+=1) → 保护共享block → 防止eviction
   → kv_cache_manager.allocate_slots(request, num_new_tokens, ..., has_scheduled_reqs=True)
   → → ★★★★★ watermark_blocks生效 → required_blocks = num_blocks + watermark_blocks
   → → BlockPool.get_num_free_blocks() - reserved_blocks - watermark_blocks → admission检查
   → → 失败 → return None → break → 不preempt running!

3. _preempt_request():
   → kv_cache_manager.free(request) → 释放所有KV blocks
   → → BlockPool.free_blocks() → ref_cnt -= 1 → 只有ref_cnt==0才回到free queue
   → → ★★★★★ shared prefix blocks (ref_cnt>0) → 不回free queue → KV数据完好!

4. cache_full_blocks() (每step结束后):
   → block_pool.cache_full_blocks(request, blocks, ...)
   → → 计算满block的hash → 插入cached_block_hash_to_block
   → → ★★★ 供后续请求prefix cache查找

5. get_num_common_prefix_blocks() → cascade attention:
   → 计算所有running请求的最长公共prefix → 传给ModelRunner
   → → prefix≥256 + requests≥8 → 可能触发cascade attention
```

### 6.2 allocate_slots核心逻辑 (★★★★★ 含watermark)

```python
★★★★★ allocate_slots (kv_cache_manager.py L238-420) — 含PR #44594 watermark:

def allocate_slots(self, request, num_new_tokens, ..., 
                   reserved_blocks=0, has_scheduled_reqs=True):
    
    # ★★★★★ Watermark计算
    watermark_blocks = 0
    if has_scheduled_reqs and request.status in (WAITING, PREEMPTED):
        watermark_blocks = self.watermark_blocks  # int(0.05 × num_total_blocks)
    
    # full_sequence_must_fit检查
    if full_sequence_must_fit:
        required_blocks = num_blocks_to_allocate + watermark_blocks
        if required_blocks > block_pool.get_num_free_blocks():
            return None
    
    # 常规allocation检查
    available_blocks = block_pool.get_num_free_blocks() - reserved_blocks
    required_blocks = num_blocks_to_allocate + watermark_blocks
    if required_blocks > available_blocks:
        return None  # ★★★★★ admission拒绝 → 但不preempt!

★★★★★ 关键设计:
  → watermark只对WAITING/PREEMPTED生效 → RUNNING不受影响 → decode优先!
  → has_scheduled_reqs=True → 有running请求 → watermark生效 → 保守admission
  → ★★★★★ reserved_blocks → 为in-flight async KV loads保留 → 防止deadlock
  → ★★★★★ watermark_blocks + reserved_blocks → 双重headroom → 最强防护!
```

### 6.3 BlockPool与FreeKVCacheBlockQueue

```
★★★★★ BlockPool核心数据结构:

class BlockPool:
    blocks: list[KVCacheBlock]               # 所有物理block
    free_block_queue: FreeKVCacheBlockQueue   # O(1)双向链表 LRU
    cached_block_hash_to_block: BlockHashToBlockMap  # 1:N hash→blocks映射
    null_block: KVCacheBlock                  # 占位符block

★★★★★ FreeKVCacheBlockQueue (O(1)双向链表):
fake_head → [LRU block] → [block] → ... → [MRU block] → fake_tail
    
  → popleft(): 取最久未用block → LRU eviction → 给新请求
  → append(): 新释放的block → 放尾部 → 保持MRU
  → ★★★★★ cached blocks放尾部 → 尽久存活 → prefix cache寿命最大化!
  → ★★★★★ uncached blocks放前端 → 优先reuse → 不保prefix → 因为没hash value
```

---

## 7. RTX 4090部署实践建议 (★★★★★)

### 7.1 Watermark配置

```
★★★★★ RTX 4090 Watermark配置:

★★★★★ 必须设置:
  --watermark 0.05  # ★★★★★ PR #44594 → preemption减少82% → ITL p99减少56%!
  → 默认0.0 → 不保留headroom → thrashing风险高 → ★★★ 必须显式设置!

★★★★★ 配合设置:
  --enable-prefix-caching True  # ★★★★★ prefix共享 → ref_cnt保护 → preemption后部分恢复
  --gpu-memory-utilization 0.90  # 90% → watermark配合 → 总blocks充足
  --max-num-seqs 48             # ★★★ 限制并发 → 减少preemption触发频率

★★★★ 推荐设置:
  --kv-cache-dtype int8  # INT8 KV → 灁一半内存 → 更多blocks → 减少preemption
  --max-model-len 8192   # 限制model len → 减少单request KV占用

★★★★ 避免:
  ✗ watermark=0 → 默认 → 无headroom → thrashing风险 → ★★★★ 必须改!
  ✗ watermark>0.10 → 太保守 → admission太少 → throughput下降
  ✗ max_num_seqs过大 → preemption频繁 → thrashing → watermark无法完全消除!
  ✗ 无prefix caching → GRPO共享prompt被重计算8次 → 浪费 → 且无ref_cnt保护!
```

### 7.2 GRPO场景最优配置

```
★★★★★ RTX 4090 GRPO serving preemption完整配置:

vllm serve <model> \
  --enable-prefix-caching \          # ★★★★★ GRPO共享system prompt → MUST!
  --watermark 0.05 \                 # ★★★★★ PR #44594 → preemption -82%
  --gpu-memory-utilization 0.90 \    # 90% → watermark配合
  --max-num-seqs 48 \                # ★★★ 限制并发 → 减少preemption触发
  --kv-cache-dtype int8 \            # INT8 KV → 更多blocks → 减少preemption
  --max-model-len 8192 \             # 限制长度 → 控制单request KV占用
  --enable-chunked-prefill \         # ★★★★ 分段prefill → 不阻塞decode
  --long-prefill-token-threshold 2048  # chunk大小 → 平衡prefill+decode

★★★★★ 预期效果:
  → 7B INT4 + INT8 KV + GQA-8 + FlashInfer → 48并发 × ~55 tok/s ≈ 2640 tok/s aggregate
  → preemption < 5次/1000 prompts → ITL p99 < 20ms → 稳定serving!
```

### 7.3 Watermark与其他机制协同

```
★★★★★ RTX 4090上4层+配置的协同效果:

  Layer 1 (admission gate):
  → ★★★★ preemption后不admit新请求 → 给preempted请求空间 → 配合watermark更有效
  
  Layer 2 (watermark=0.05):
  → ★★★★★ admission保留5% blocks → running增长有headroom → 最关键层!
  → RTX 4090 24GB → 5% of total blocks ≈ ~150 blocks → ~2400 tokens headroom
  
  Layer 3 (prefix caching + ref_cnt):
  → ★★★★★ GRPO system prompt共享 → ref_cnt>0 → preemption后prefix存活
  → → 减少重计算量 → 加速恢复 → 减少再次preemption
  
  Layer 4 (self-detection):
  → ★★★★ 防止自己preempt自己 → 最基本防护 → 不需要额外配置
  
  + INT8 KV (额外5th层):
  → ★★★★★ KV占用减半 → blocks数量翻倍 → watermark_blocks更多 → admission更容易
  → → 更多blocks → preemption更少 → thrashing大幅减少
  
  + max_num_seqs限制 (额外6th层):
  → ★★★★ 限制并发48 → 每request KV占用可控 → 总需求<可用 → preemption极少
  
  ★★★★★ 6层防护 = watermark + admission gate + prefix caching + self-detect + INT8 KV + max_num_seqs → 稳定serving!
```

---

## 8. 关键洞察总结

```
★★★★★ 1. PR #44594 watermark=0.05:
  → preemption -82%, ITL p99 -56%, throughput +5.1% → ★★★★★ 最有效的thrashing prevention!
  → 只对WAITING/PREEMPTED生效 → RUNNING不受影响 → decode优先保证
  → sweet spot=0.05 → 太少不够 → 太多浪费 → ★★★ 必须显式设置(default=0.0无效!)

★★★★★ 2. V1 preemption = retraction:
  → num_computed_tokens=0 → 完全重计算 → 无swap → 无CPU offload
  → 只靠prefix cache缓解 → ★★★★★ 必须enable_prefix_caching!
  → 长output请求 → 重计算代价高 → ★★★ watermark减少preemption最关键!

★★★★★ 3. 4层thrashing prevention:
  → admission gate → watermark → prefix sharing → self-detection
  → ★★★★★ watermark是最关键层 → 其他层是辅助 → 但综合效果最强

★★★★★ 4. Hash collision (#44701):
  → LoRA name + cache_salt domain collision → silent correctness bug
  → chained hash → Block 0 collision → 整个prefix corrupt
  → ★★★ GRPO同adapter → 安全 → Multi-tenant → collision风险
  → ★★★★★ fix方向: domain-tag prefix ("lora:" + name, "salt:" + value)

★★★★★ 5. Scheduler-KV Cache Manager交互:
  → allocate_slots → watermark_blocks + reserved_blocks → 双重headroom
  → get_computed_blocks → prefix cache hit → ref_cnt保护 → preemption后部分恢复
  → free_blocks → ref_cnt-=1 → 只有ref_cnt==0才释放 → shared prefix存活

★★★★★ 6. RTX 4090最优:
  → watermark=0.05 + INT8 KV + prefix caching + max_num_seqs=48 → 6层防护 → 稳定serving
  → ★★★★★ 不设置watermark → thrashing风险 → ★★★★ 必须显式设置!
  → INT4 weights → 灁weight内存 → 更多blocks → watermark效果更大

★★★★★★★ 7. Watermark vs BudgetRefiner互补 (NEW):
  → Watermark = KV层(reactive-ish) → admission保留5% blocks → preemption -82%
  → BudgetRefiner = compute层(proactive) → 动态token_budget → decode-first → SLO-compliant
  → ★★★★★★★★ 两者互补 → Watermark管空间 → BudgetRefiner管时间 → 双层防护!
  → → BudgetRefiner使Watermark更容易通过 → Watermark使budget更保守 → 不是叠加而是互补
  → ★★★★★★★★ INT8 KV → watermark_blocks翻倍 → headroom 2× → 与watermark协同放大!

★★★★★★★ 8. RTX 4090 quantitative impact (NEW):
  → watermark=0.05 + INT8 KV → ~840K tokens headroom → 48请求 × ~17500 tokens/request增长空间
  → → vs FP16 → headroom仅~420K → ~8750 tokens/request → INT8使watermark效果翻倍!
  → → ★★★★★★★★ Watermark=0.05 + INT8 KV = RTX 4090最强KV层防护组合!
```

---

## 9. Watermark vs BudgetRefiner SLO 互补分析 (★★★★★★★★★★★★★)

### 9.1 互补关系: KV Cache压力 vs Compute Time压力

```
★★★★★★★★★★★★★ Watermark与BudgetRefiner处理不同维度的压力:

  Watermark (PR #44594):
  → 处理: KV CACHE PRESSURE → blocks不够 → preemption → recompute
  → 机制: admission时保留5% blocks → running有增长headroom
  → 触发条件: available_blocks < required_blocks + watermark_blocks
  → 反应型(reactive-ish): 在admission时保守 → 但运行时不对decode throttling
  → 效果: preemption -82%, ITL p99 -56%, throughput +5.1%

  BudgetRefiner SLO (vllm-ascend):
  → 处理: COMPUTE TIME PRESSURE → too many tokens → step时间超SLO → decode延迟
  → 机制: 动态调整token_budget → decode多时prefill budget减小 → 保护decode
  → 触发条件: (ctx_len, d_num) lookup → budget drops as decode load increases
  → 预防型(proactive): 在scheduling时预判 → 动态减少prefill tokens
  → 效果: decode-first + SLO-compliant scheduling → 防止over-commit

★★★★★★★★★★★★★ 互补而非重叠!

  维度1: 内存空间 (Watermark)
  → 问题: KV blocks不足 → preemption → recompute → thrashing
  → 解决: 保留headroom → running请求有增长空间 → 减少preemption
  → → ★★★★★ RTX 4090 24GB → 内存最受限 → Watermark最关键!

  维度2: 计算时间 (BudgetRefiner)
  → 问题: 每step计算太多tokens → decode被prefill挤 → ITL恶化
  → 解决: 动态budget → decode多时限制prefill → 保护decode SLO
  → → ★★★★ RTX 4090 → compute不太受限(单GPU) → BudgetRefiner辅助!

★★★★★★★★★★★★★ 两者同时使用 → 近乎零preemption + 稳定decode:

  Watermark → 防止admission过激进 → KV有headroom → preemption极少
  BudgetRefiner → 防止scheduling过激进 → compute有headroom → ITL稳定
  → ★★★★★★★ 两者互补 → Watermark=KV层 → BudgetRefiner=compute层 → 双层防护!
```

### 9.2 代码层面的互补交互

```
★★★★★★★ Watermark与BudgetRefiner在同一schedule()流程中的交互:

  schedule() Phase 1 (RUNNING → decode):
  → Watermark: 不影响 → RUNNING请求不受watermark限制 → decode优先!
  → BudgetRefiner: token_budget动态调整 → d_num多时budget更小 → 限制prefill chunk
  → → ★★★★★ Watermark不管decode → BudgetRefiner不管block → 各管各的维度!

  schedule() Phase 2 (WAITING → prefill):
  → Watermark: allocate_slots检查 → watermark_blocks增加required_blocks → 更保守admission
  → BudgetRefiner: token_budget已被缩减 → num_new_tokens = min(tokens, budget) → 更小chunk
  → → ★★★★★ Watermark限制block admission → BudgetRefiner限制token scheduling → 双重把关!

  ★★★★★★★★ 关键交互点:
  1. BudgetRefiner先调整token_budget → Watermark检查block availability
  2. → budget更小 → num_new_tokens更小 → num_blocks_to_allocate更小
  3. → → required_blocks(num_blocks + watermark)更小 → admission更容易!
  4. → ★★★★★ BudgetRefiner使Watermark更容易通过 → 反过来watermark使budget更保守
  5. → → ★★★★★★★★ 两者不是叠加难度 → 而是互补! BudgetRefiner减少token需求 → Watermark确保空间!
```

### 9.3 RTX 4090双层防护最优配置

```
★★★★★★★★★★★★★ RTX 4090双层防护推荐:

  Watermark配置 (KV层):
  --watermark 0.05              # ★★★★★ admission时保留5% blocks → preemption -82%

  BudgetRefiner配置 (compute层):
  # ★★★★★★ BudgetRefiner暂不在vLLM mainline → 只在vllm-ascend
  # → 需要upstream贡献 → profile_table.csv需要RTX 4090数据
  # → ★★★★★★★★ RTX 4090 profile data = NO OTHER vLLM CONTRIBUTOR HAS THIS!
  # → 当前替代: --max-num-seqs 48 → 限制并发 → 间接控制compute压力

  双层防护完整配置:
  vllm serve <model> \
    --watermark 0.05 \                # ★★★★★ KV层: preemption -82%
    --enable-prefix-caching \         # ★★★★★ prefix共享 → ref_cnt保护
    --max-num-seqs 48 \               # ★★★★ compute层: 限制并发 → 间接BudgetRefiner
    --kv-cache-dtype int8 \           # ★★★★★ KV占用减半 → 更多blocks
    --gpu-memory-utilization 0.90 \   # 90% → 总blocks充足
    --max-model-len 8192 \            # 限制长度 → 控制单request占用
    --enable-chunked-prefill \        # ★★★★ compute层: 不阻塞decode
    --long-prefill-token-threshold 2048  # chunk大小 → 平衡prefill+decode

  ★★★★★★★ 未来: BudgetRefiner SLO upstream → 双层自动 → 无需手动max_num_seqs!
```

---

## 10. RTX 4090 Quantitative Watermark Impact (★★★★★★★)

### 10.1 RTX 4090 Block计算

```
★★★★★★★ RTX 4090 24GB — watermark=0.05 quantitative impact:

  假设配置: Qwen2.5-7B-Instruct INT4, GQA-8, block_size=16

  Step 1: GPU memory allocation
  → gpu_memory_utilization=0.90 → 24GB × 0.90 = 21.6GB可用
  → INT4 weights ~3.5GB → activation ~1GB → 剩余 ~17.1GB for KV cache
  → INT8 KV → 每token KV = 2 × 8 × num_layers × d_head / block_size

  Step 2: Number of blocks (估算)
  → 7B model, 28 layers, GQA-8 → 每layer KV = 2 × 8 × 128 = 2048 bytes/token
  → → 17.1GB / 2048 bytes/token ≈ 8.4M tokens → 但需要block_size alignment
  → → 8.4M / 16 ≈ 525,000 blocks (粗估)
  → → ★★★★★ 实际取决于model架构 → 需用vllm profile确认

  Step 3: watermark_blocks impact
  → watermark=0.05 → watermark_blocks = int(0.05 × 525000) ≈ 26,250 blocks
  → → 26,250 × 16 tokens = ~420,000 tokens reserved → ~420K tokens headroom
  → → ★★★★★ 420K tokens ≈ 84个请求的完整output空间 (8192 tokens/request × 5% headroom)
  → → → running 48请求 → 每请求约420K/48 ≈ 8750 tokens headroom → 足够decode增长!

  Step 4: Without watermark
  → watermark=0 → watermark_blocks=0 → no headroom → admission激进
  → → 48请求全部admitted → blocks刚够 → decode增长 → blocks不够 → preemption!
  → → → ★★★★ 被preempted请求 → num_computed_tokens=0 → 重计算3500+ tokens → 又需大量blocks!

★★★★★★★ 结论:
  → RTX 4090 24GB → watermark=0.05 → ~420K tokens headroom → 48请求有增长空间
  → → preemption大幅减少 → ITL稳定 → throughput增加
  → → ★★★★★ 不设watermark → 纯靠运气 → decode-heavy时thrashing → 必须设置!
```

### 10.2 RTX 4090 Watermark与INT8 KV的协同放大效应

```
★★★★★★★ Watermark + INT8 KV synergistic amplification:

  INT8 KV → 每token KV占用减半 → blocks数量翻倍 → watermark_blocks也翻倍!
  → FP16 KV: 525,000 blocks → watermark_blocks = 26,250
  → INT8 KV: 1,050,000 blocks → watermark_blocks = 52,500 → ★★★★★ 2× more headroom!
  → → 52,500 × 16 = 840,000 tokens headroom → 更大增长空间!

  ★★★★★★★★ INT8 KV makes watermark MORE effective:
  → 更多blocks → watermark_blocks更大 → admission更精确 → running增长空间更大
  → → preemption几乎消除 → ITL近乎稳定 → throughput最大化
  → → ★★★★★★★★★ Watermark=0.05 + INT8 KV = RTX 4090最强KV层防护!

  ★★★★★★ FP16 vs INT8 watermark对比 (估算):
  → FP16 + watermark=0.05 → ~420K tokens headroom → 48请求 → ~8750 tokens/request
  → INT8 + watermark=0.05 → ~840K tokens headroom → 48请求 → ~17500 tokens/request
  → → ★★★★★ INT8让headroom从8750→17500 → 2× → decode增长空间2× → preemption风险近乎零!
```

---

## 11. 参考资料 (更新版)

```
★★★★★ 源码 (local checkout at /Users/jackiemac/workspace/rollout-infra/vllm):
  - vllm/v1/core/sched/scheduler.py (~2400行) — Scheduler核心 → schedule() + _preempt_request()
    → ★★★★★ 当前checkout不含watermark PR → 需upgrade到v0.22.0+才有!
    → Phase 1 RUNNING: L376-515 → Phase 2 WAITING: L561-830 → _preempt_request: L958-978
  - vllm/v1/core/kv_cache_manager.py (~430行) — KVCacheManager → allocate_slots() + watermark
    → allocate_slots: L238-429 → BEFORE watermark: 无has_scheduled_reqs参数
    → AFTER watermark: +watermark param +watermark_blocks +has_scheduled_reqs
  - vllm/v1/core/block_pool.py — BlockPool → free_blocks() L419-433 + ref_cnt机制
    → free_blocks: ref_cnt-=1 → only ref_cnt==0 goes back to free queue
  - vllm/v1/core/kv_cache_coordinator.py — KVCacheCoordinator → free() L220-228 + prefix hit
  - vllm/v1/core/kv_cache_utils.py — KVCacheBlock + FreeKVCacheBlockQueue + hash
  - vllm/v1/core/single_type_kv_cache_manager.py — free() L363-377 (reversed order for eviction)
  - vllm/v1/request.py — RequestStatus enum: WAITING/RUNNING/PREEMPTED L319-327
  - vllm/config/scheduler.py — SchedulerConfig → watermark字段 (PR diff: +7 lines)

★★★★★ PR与Issue:
  - ★★★★★ PR #44594: Add kvcache watermark to reduce preemptions (merged 2026-06-11)
    → Author: njhill (Nick Hill) → Reviewer: WoosukKwon (approved)
    → 7 files changed: +291 additions, -8 deletions
    → ★★★★★ Merge commit: 4085ff7cb43d03bbfd05707238ea58a1561f87c2
    → ★★★★★ 9 commits: simple→proportional→back to simple→add benchmark→disable→add has_scheduled_reqs
  - ★★★★ Revert PR #45344: OPEN → NOT merged → 原PR保持!
    → Reason: CI failure (fake_allocate_slots_fn missing has_scheduled_reqs param)
    → ★★★ 只是mock问题 → 不是功能bug → revert不应该成功!
  - ★★★★★ Issue #44701: V1 prefix-cache extra-key domain collision
  - ★★★ PR #44706: Domain-tag fix for #44701 → OPEN but STALLED

★★★★★ Benchmark script (PR #44594 additions):
  - benchmarks/kv_cache_watermark.sh (248行) — reproducible watermark benchmark
    → Default: Qwen2.5-7B, TP=1, KV=16GB, concurrency=128, input=1000/output=5000
    → Sweeps watermark values: off/0.02/0.05/0.10/0.15
    → Scrapes /metrics for preemption count + vllm bench serve for latency
    → Generates matplotlib plot → 2x2 grid (preemptions/throughput/ITL/latency)

★★★★★ 我们的关联笔记:
  - vllm-v1-preemption-source-reading.md — Preemption机制源码分析
  - vllm-prefix-caching-v1.md — Prefix caching源码分析
  - vllm-prefix-cache-hash-collision-reading.md — Hash collision深度分析
  - budgetrefiner-vllm-pr-draft.md — BudgetRefiner SLO upstream PR draft
  - tools/profile_vllm_budget.py — RTX 4090 budget profiler for BudgetRefiner
  - tools/sm89_kv_cache_cost_analyzer.py — INT8 vs FP8 KV对比工具
  - tools/sm89_compatibility_checker.py — SM89 feature matrix
  - fundamentals/vllm-v1-scheduler-architecture-source-reading.md — Scheduler架构
```
