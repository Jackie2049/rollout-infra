# vLLM V1 Preemption机制源码级深度分析 (2026-06-15补充)

> ★★★★★ V1 preemption = retraction(纯重计算) → ref_cnt保护prefix共享block → thrashing prevention 4层机制
> 源码: vllm/v1/core/sched/scheduler.py (lines 488-534, 834-855, 1056) + block_pool.py + kv_cache_manager.py
> ★★★★ RTX 4090关键: retraction+prefix cache → 无CPU swap → 必须enable_prefix_caching!

## 1. Preemption触发精确条件

```
★★★★★ Phase A: Running request scheduling (lines 488-534):

触发: allocate_slots() returns None → required_blocks > available_blocks
  → required_blocks = num_blocks_to_allocate + watermark_blocks
  → available_blocks = block_pool.get_num_free_blocks() - reserved_blocks

循环:
  while True:
    new_blocks = kv_cache_manager.allocate_slots(request, ...)
    if new_blocks is not None: break  # 成功分配!
    # ★★★ 分配失败 → preempt最低优先级request → 释放blocks → retry
    if PRIORITY policy:
      preempted_req = max(running, key=lambda r: (r.priority, arrival_time))
    else:  # FCFS (默认)
      preempted_req = running.pop()  # 淘汰最近admitted
    _preempt_request(preempted_req)
    if preempted_req == request: break  # ★★★ 自我淘汰 → 无法调度

★★★★ Phase B: Waiting request scheduling (lines 834-855):
  → allocate_slots() returns None → break → 不preempt running!
  → ★★★ 只在running scheduling phase才preempt → waiting phase不preempt!

★★★★ Watermark只对WAITING/PREEMPTED请求:
  watermark_blocks = 0
  if has_scheduled_reqs and request.status in (WAITING, PREEMPTED):
    watermark_blocks = self.watermark_blocks
  → ★★★ PR #44594: watermark=0.05 → preemption减少82% → ITL p99减少56%!
```

## 2. _preempt_request精确行为

```python
★★★★★ _preempt_request (scheduler.py line 1056):

def _preempt_request(self, request, timestamp):
    # ★★★ request必须已经从running queue移除!
    self.kv_cache_manager.free(request)         # ★★★ Free ALL KV blocks (ref_cnt -= 1)
    self.encoder_cache_manager.free(request)    # Free encoder cache (vision等)
    self._inflight_prefills.discard(request)    # Remove from prefill tracking
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0             # ★★★★★ 重置为0 → 从头来过!
    request.spec_token_ids = []                 # Clear speculative tokens
    request.num_preemptions += 1               # Increment counter
    self.waiting.prepend_request(request)       # ★★★ 前端插入 → 优先重新admit

★★★★★ 关键行为:
  → num_computed_tokens = 0 → 完全重新计算 → 无swap → 无CPU offload!
  → prepend_request → deque.appendleft → 优先重新admit → 但与新请求竞争!
  → ★★★★★ V1 = 纯retraction → KV完全丢弃 → 只靠prefix cache缓解!
```

## 3. ref_cnt保护的prefix共享blocks

```
★★★★★ BlockPool.free_blocks → ref_cnt机制:

def free_blocks(self, ordered_blocks, prepend=False):
    for block in blocks_list:
        block.ref_cnt -= 1  # ★★★ 减1 → 不直接释放!
    freed_blocks = [b for b in blocks_list if b.ref_cnt == 0 and not b.is_null]
    # ★★★★ 只有ref_cnt==0的block才回到free queue!

★★★★★ 关键行为:
  1. prefix-cached block shared (ref_cnt=2):
     → free → ref_cnt减1→=1 → ★★★ 不回free queue → 保持allocated → KV数据完好!
     → 重新admit → get_computed_blocks() → hash查找 → touch(ref_cnt+=1) → ★★★ 部分重计算!

  2. prefix-cached block NOT shared (ref_cnt=1):
     → free → ref_cnt减1→=0 → ★★★ 回free queue → 成为eviction候选 → KV数据丢失!
     → 重新admit → 完全重新计算 → 包括system prompt → ★★★★ 2×慢!

★★★★ SlidingWindowManager.free优化:
  → cached blocks → free到queue尾部 → 保持prefix cache → 尽久存活
  → uncached blocks → free到queue前端 → 优先reuse → 不保prefix
  → ★★★ uncached优先reuse → cached保到最后 → prefix cache寿命最大化!
```

## 4. Thrashing Prevention 4层机制

```
★★★★★ V1有4层防thrashing机制:

Layer 1: ★★★ Admission gate after preemption (line 591):
  if not preempted_reqs and pause_state == UNPAUSED:
    # ★★★★ 如果本step有preemption → 完全跳过waiting scheduling!
    # → 释放的blocks不被新请求立即填充 → preempted请求有"呼吸空间"!

Layer 2: ★★★ Watermark for WAITING/PREEMPTED:
  watermark_blocks = self.watermark_blocks (default=0.05=5%总blocks)
  → ★★★ 保留headroom → admitted请求不占block到边缘 → running请求有增长空间
  → PR #44594: watermark=0.05 → preemption减少82% → ITL p99减少56%!

Layer 3: ★★★ Prefix cache sharing:
  → shared blocks (ref_cnt>0) survive preemption → 部分重计算 → 少blocks需分配
  → → ★★★ 减少再次触发preemption的几率 → 缓解但不能根治!

Layer 4: ★★★ Self-preemption detection (line 531):
  if preempted_req == request: break
  → ★★★ 如果淘汰的是自己 → 停止 → 防止"淘汰自己给自己腾空间"的荒谬循环!

★★★★★ 剩余thrashing风险:
  → ★★★ 长输出请求(3500 tokens)被preempted → 重计算3500 tokens → 大量blocks → 可能再次触发!
  → → RTX 4090(24GB): 在<1.5x mean demand时 → thrashing仍可能!
  → → ★★★★ 解决: reduce max_num_seqs + lower watermark + enable prefix caching!
```

## 5. RTX 4090实用建议

```
★★★★★ RTX 4090 GRPO serving preemption config:

★★★★★ 必须设置:
  enable_prefix_caching: True  # ★★★★★ GRPO共享system prompt → MUST!
  gpu_memory_utilization: 0.90  # 90% → watermark配合
  max_num_seqs: 48  # ★★★ 限制并发 → 减少preemption触发!

★★★★ 推荐设置:
  kv_cache_dtype: int8  # INT8 KV → 灁一半内存 → 更多blocks → 减少preemption!
  watermark: 0.05  # PR #44594 → preemption减少82% → ★★★ 改善ITL!

★★★★ 避免:
  ✗ max_num_seqs过大 → preemption频繁 → thrashing!
  ✗ 无prefix caching → GRPO共享prompt被重计算8次 → 浪费!
  ✗ FP8 KV → SM89 crash → 无法运行 → ★★★ INT8唯一可行!
```

## 6. 关键洞察

1. ★★★★★ **V1 retraction = 无swap** → KV完全丢弃 → 只靠prefix cache缓解 → ★★★ 必须enable!
2. ★★★★★ **ref_cnt保护shared blocks** → 多请求共享prefix → preemption后prefix存活 → ★★★ 优化关键
3. ★★★★ **4层防thrashing** → admission gate + watermark + prefix sharing + self-detect → 但仍有风险
4. ★★★★ **watermark=0.05** → preemption减少82% → ★★★ PR #44594 → 应配置!
5. ★★★ **SlidingWindow优化** → cached blocks保尾部 → uncached优先reuse → prefix cache寿命最大化
6. ★★★★ **RTX 4090: 限制max_num_seqs** → 减少并发 → 减少preemption → 配合prefix caching → 稳定serving

---

Sources:
- vLLM scheduler.py: schedule(), _preempt_request() (lines 488-534, 834-855, 1056)
- vLLM block_pool.py: free_blocks() (ref_cnt mechanism)
- vLLM single_type_kv_cache_manager.py: free() methods (base, SWA, Mamba)
- vLLM kv_cache_utils.py: KVCacheBlock dataclass (ref_cnt field)
- ★★★ PR #44594: Add kvcache watermark to reduce preemptions
- ★★★ PR #45344: Revert watermark PR (CI fix → re-merged)
- ★★★ V1 Architecture Blog: blog.vllm.ai/2025/v1-architecture.html
- ★★★ GRPO rollout scheduler: notebook/projects/vllm-grpo-rollout-scheduler-reading.md
