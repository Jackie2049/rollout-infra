# vLLM V1 KV Cache Memory Management 源码级阅读

> 2026-06-15 | 源码: vllm/v1/core/block_pool.py + kv_cache_utils.py + kv_cache_manager.py + kv_cache_coordinator.py + single_type_kv_cache_manager.py + scheduler.py
> 核心: BlockPool(FreeKVCacheBlockQueue双向链表) → LRU eviction → prefix caching(hash_map+chained_hash) → V1无swap=full reset → KV connector(NIXL/Mooncake/P2P/HF3FS) → HybridKVCacheCoordinator(迭代fixed-point)

## 1. BlockPool: KV Block分配/释放核心

```python
# block_pool.py 核心数据结构:
BlockPool → FreeKVCacheBlockQueue (双向链表) → KVCacheBlock[]

KVCacheBlock (@dataclass(slots=True)):
  - block_id: int (0 ~ num_gpu_blocks-1)
  - ref_cnt: int = 0 → 引用计数 → prefix共享时>1
  - _block_hash: BlockHashWithGroupId | None → 前缀缓存hash
  - prev_free_block, next_free_block → 双向链表指针
  - is_null: bool → null_block占位符(block_id=0)

FreeKVCacheBlockQueue (自定义双向链表,非Python deque):
  - fake head/tail sentinel → 减少分支 → O(1)中间删除
  - 排序: LRU → 最久未用在前 → 新释放的在后
  - popleft_n(num_blocks) → 分配N个block → 从头部取 → LRU优先淘汰!
  - append_n(blocks) → 释放 → 逆序添加 → tail block先释放 → 但排在free queue尾部
```

**Block分配流程** (block_pool.py):
```
get_new_blocks(n):
  1. 从free_block_queue头部取n个block → LRU优先淘汰!
  2. 对每个block: _maybe_evict_cached_block → 如果在前缀缓存hash map → 移除hash
  3. 每个block: ref_cnt += 1 → 引用计数增加

free_blocks(blocks):
  1. 每个block: ref_cnt -= 1
  2. ref_cnt == 0 → append到free_block_queue尾部 → 可再分配
  3. 逆序释放 → tail block先入queue → 但排在尾部 → LRU时最后淘汰

touch(blocks):
  1. 每个block: ref_cnt += 1
  2. ref_cnt从0到1 → 从free_queue移除 → 不再可淘汰 → 前缀命中保护!
```

## 2. Block Size计算

```python
# kv_cache_utils.py: resolve_kv_cache_block_sizes()
scheduler_block_size:
  - 单KV cache group: cache_config.block_size * dcp * pcp (DCP=decode context parallel, PCP=prefill context parallel)
  - 多group(hybrid): math.lcm(*group_block_sizes) → LCM!

hash_block_size:
  - 单group: = scheduler_block_size
  - 多group: cache_config.hash_block_size 或 math.gcd(*group_block_sizes)

默认block_size=16 tokens (CacheConfig.block_size)

num_blocks计算:
  num_blocks = available_memory // page_size // num_layers
  page_size_bytes per block per layer = 2 * block_size * num_kv_heads * head_size * dtype_size
  → 所有worker取最小值 → 统一block数量!
```

## 3. Eviction策略: LRU + 前缀缓存淘汰

```
LRU eviction逻辑:

1. 正常流程: request完成 → free_blocks → block回到free_queue尾部
2. 前缀缓存block: 仍在hash map + free_queue → 双重存在!
   → 分配时: _maybe_evict_cached_block → 先从hash map移除 → 再分配
   → 这保证: 前缀缓存block被分配时,hash entry被清除!

3. Preemption (scheduler.py:958-978):
   → allocate_slots返回None → 不足free blocks
   → self.kv_cache_manager.free(request) → 释放所有block
   → request.num_computed_tokens = 0 → ★ FULL RESET!
   → 必须从头重新计算 → V1不保留任何partial KV!

4. vs SGLang: SGLang radix tree → eviction保留prefix → preemption后可reuse prefix
   → vLLM V1: eviction不保留 → preemption = 从头开始 → 更浪费!
```

## 4. V1没有CPU/GPU Swap!

```
★ 关键发现: V1完全移除了swap机制!

V0: BlockSpaceManager + SwapMap → CPU/GPU block swap → 内存不足时可swap out到CPU
V1: BlockPool只管GPU blocks → 无CPU pool → 无swap操作
   → 内存不足 → preemption(释放running request blocks) → full reset
   → 简化设计 → 但失去了swap带来的内存弹性!

替代方案: KV connector offloading → 但这是PD分离专用,不是通用swap
   → SimpleCPUOffloadConnector → CPU offload for PD
   → NIXL → RDMA KV transfer → 跨节点
   → Mooncake → RDMA + store/worker架构
   → HF3FS → HuggingFace 3FS存储
   → OffloadingConnector → CPU offload with scheduler
```

## 5. Scheduler与BlockManager交互

```
每步流程 (scheduler.py):

1. new_step_starts() → 清Mamba cached_blocks_this_step
2. RUNNING request → kv_cache_manager.allocate_slots(request, num_new_tokens)
   → None → preemption!
3. WAITING request → kv_cache_manager.get_computed_blocks(request) → prefix cache hit
   → connector.get_num_new_matched_tokens() → external cache (PD)
   → allocate_slots() → 分配新block

4. allocate_slots 5-stage layout:
   | < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
   → comp: 已计算的前缀block → touch(ref_cnt+1) → 不分配新
   → new_comp: 新计算的前缀 → allocate新block → cache
   → ext_comp: external(PD transfer)的block → 临时buffer → 异步transfer
   → new: 需要新计算的token → allocate新block
   → lookahead: speculative decoding多分配的block

5. request完成 → kv_cache_manager.free(request) → block释放
6. block ID tracking → take_new_block_ids() → 零化新block(清KV内容)
```

## 6. Prefix Caching实现

```
3层前缀缓存架构:

1. BlockHashToBlockMap (block_pool.py:34-128):
   → BlockHashWithGroupId → KVCacheBlock mapping
   → 支持多block相同hash → dict存储 → 减少GC开销

2. cache_full_blocks (block_pool.py:211-331):
   → request的block变满 → 计算hash → 存入hash map
   → hash = hash_function((parent_hash, token_ids_tuple, extra_keys))
   → chained hashing → 父hash+当前token → 保证前缀完整性!
   → extra keys: mm_hash(multimodal), lora_name, cache_salt, prompt_embeds

3. find_longest_cache_hit (single_type_kv_cache_manager.py):
   → FullAttentionManager: 从左到右扫描 → 第一miss就停
   → SlidingWindowManager: 从右到左 → 需连续sliding_window_contiguous_blocks个hit
   → HybridKVCacheCoordinator: ★ 迭代fixed-point算法!
     → 每个group缩小候选hit length → 单调收敛 → 直到稳定
     → 这是V1处理混合attention类型的核心创新!

Block hash计算细节:
   → hash_block_tokens (kv_cache_utils.py:541-568)
   → hash = hash_function((parent_block_hash, curr_block_token_ids, extra_keys))
   → NONE_HASH seed: os.urandom随机 或 PYTHONHASHSEED可复现
   → 增量计算: append → 只hash新block → 不重算整个prefix!
```

## 7. V0 vs V1关键差异

```
V0 → V1核心变化:

1. 无CPU/GPU swap → V1简化 → 但失去内存弹性
   → V0: BlockSpaceManager + SwapMap → 可swap out到CPU → 内存不足时弹性
   → V1: BlockPool只GPU → 内存不足 → preemption(full reset)
   → trade-off: 简化代码 → 但大batch时可能更频繁preemption

2. 单BlockPool → flat双向链表 → 无CPU/GPU分离
   → V0: Block + CPU block → 2个pool
   → V1: 只有GPU block → 单pool

3. 前缀缓存直接集成 → BlockPool内hash_map
   → V0: 前缀缓存是独立层 → separate manager
   → V1: BlockPool.cached_block_hash_to_block → 集成 → 更简洁

4. KVCacheCoordinator → 多KV cache group支持
   → hybrid模型: full attention + sliding window + mamba
   → 多group共享同一pool → fixed-point算法协调
   → V0: 无此概念 → 只支持单一attention类型

5. ref_cnt在block上 → 直接引用计数
   → prefix hit: touch() → ref_cnt+1 → 不被evict
   → request完成: free() → ref_cnt-1 → 可被evict
   → 多request共享同一prefix block → ref_cnt>1 → 安全!

6. Preemption = full reset → num_computed_tokens=0
   → ★ 这是V1 vs SGLang最大差异!
   → SGLang: radix tree保留evicted prefix → preemption后prefix reuse
   → vLLM V1: evict后完全重算 → 潜在性能损失!
```

## 8. KV Connector for PD分离

```
KVConnectorBase_V1 (base.py) → 双侧接口:

Scheduler-side:
  - get_num_new_matched_tokens() → 多少token已在远程KV → 无副作用
  - update_state_after_alloc() → buffer分配后更新state
  - request_finished() → request完成 → 是否异步释放block
  - take_events() → 新KV事件

Worker-side:
  - handle_preemptions() → 处理preempted block
  - start_load_kv() → 开始加载(可能异步)
  - wait_for_layer_load(i) → 阻塞等layer i加载完成
  - save_kv_layer(i) → 开始保存layer i
  - wait_for_save() → 阻塞等所有保存完成

6种connector实现:
  1. NIXL → RDMA+ZMQ side channel → lease-based KV管理 → ★ 最新!
  2. P2P NCCL → NCCL peer-to-peer → 简单但慢
  3. Mooncake → RDMA+store/worker → 高吞吐
  4. LMCache → LMCache library → 第三方
  5. HF3FS → HuggingFace 3FS → 新存储backend
  6. OffloadingConnector → CPU offload → 本地offload

ActiveKVConnector (worker/gpu/kv_connector.py):
  - pre_forward → start_load_kv()
  - post_forward → wait_for_save() + get_finished()
  - no_forward → 纯KV transfer步骤 → 无model forward

Uniform KV cache layout (kv_connector_model_runner_mixin.py:114-183):
  - 所有layer共享single contiguous tensor → 跨layer KV transfer高效!
  - 条件: 单group + 同page_size + connector需要 + backend支持
  - allocate_uniform_kv_caches() → cross_layers_kv_cache[num_layers] → permuted
```

## 9. RTX 4090影响

```
RTX 4090 KV cache管理:

1. BlockPool → 24GB GPU → 7B BF16:
   - num_blocks = 24GB / (2*16*num_kv_heads*head_size*2) / num_layers
   - 7B GQA-8: num_kv_heads=8, head_size=128, 32 layers → ~5K blocks
   - block_size=16 → ~80K tokens → 足够单请求+prefill!

2. INT8 KV cache → block数量翻倍 → ~10K blocks → 更大容量!

3. 前缀缓存 → RTX 4090适用 → prefix reuse → 省4-7x prefill
   → 但preemption=full reset → 无radix tree → 不如SGLang

4. PD分离 → RTX 4090不适合 → 需要多GPU → 不够
   → 但: SimpleCPUOffloadConnector → 单GPU也能offload到CPU → 可行!

5. ★ V1无swap → RTX 4090内存管理更严格 → preemption更频繁
   → 建议: INT4+INT8KV → 减少KV内存 → 减少preemption → 更稳定

→ RTX 4090最优: INT4 weights + INT8 KV + GQA-8 + prefix caching
→ SGLang优势: radix tree + overlap → 前缀缓存更高效 → 但vLLM也可用!
```

## 10. 关键设计洞察

```
1. FreeKVCacheBlockQueue = 自定义双向链表 → 为什么不用deque?
   → deque不支持中间删除 → prefix hit需要从free queue移除block
   → 自定义链表: O(1)中间删除 → 前缀缓存命中时更高效!
   → sentinel节点 → 减少空判断 → 优化CPU开销

2. LRU + 前缀缓存双重存在 → 设计巧思!
   → block在hash map + free queue → 两个引用
   → 分配时先从hash map移除 → 保证hash一致性
   → free时只回free queue → hash自动失效 → 无需显式清除

3. chained hash → 增量计算 → 性能关键!
   → parent_hash + curr_tokens → 只hash新block → 不重算prefix
   → 这使prefix caching开销接近O(1) → 不随prefix长度增长!

4. V1无swap = 设计哲学 → 简化优于弹性
   → CPU/GPU swap → 代码复杂 → swap latency → 占CPU带宽
   → V1选择: 简化代码 + 预留足够GPU内存 + preemption重算
   → trade-off: 代码简洁 → 但OOM时更激进preemption

5. HybridKVCacheCoordinator → fixed-point算法 → V1独特!
   → 多attention类型(Full+Sliding+Mamba) → 如何共享block pool?
   → fixed-point: 每个group缩小候选 → 单调收敛 → 保证正确
   → 这是V1支持混合模型的基石 → V0无此概念!

6. KV connector双侧接口 → scheduler side + worker side
   → scheduler: 只管metadata → 不管数据传输 → 轻量
   → worker: 实际load/save → RDMA/ZMQ → 重量
   → 分离: scheduler不等transfer → 不阻塞调度!

7. NIXL lease-based → KV有效期 → 防止stale cache
   → kv_lease_duration=30s → P实例发lease → V实例必须在30s内使用
   → heartbeat: lease_duration // 6 → 5s → 定期续lease
   → 过期 → block释放 → 不浪费GPU内存
```

---

Sources:
- vllm/v1/core/block_pool.py — BlockPool + FreeKVCacheBlockQueue + BlockHashToBlockMap
- vllm/v1/core/kv_cache_utils.py — KVCacheBlock + hash_block_tokens + resolve_kv_cache_block_sizes
- vllm/v1/core/kv_cache_manager.py — KVCacheManager + allocate_slots (5-stage layout)
- vllm/v1/core/kv_cache_coordinator.py — HybridKVCacheCoordinator (fixed-point)
- vllm/v1/core/single_type_kv_cache_manager.py — FullAttention/SlidingWindow/Mamba managers
- vllm/v1/core/sched/scheduler.py — scheduler-block interaction + preemption
- vllm/v1/core/kv_cache_metrics.py — Block lifecycle metrics
- vllm/v1/kv_cache_interface.py — KVCacheSpec + block_size definitions
- vllm/distributed/kv_transfer/kv_connector/v1/base.py — KVConnectorBase_V1 ABC
- vllm/distributed/kv_transfer/kv_connector/v1/nixl/ — NIXL connector (scheduler+worker)
- vllm/v1/worker/gpu/kv_connector.py — ActiveKVConnector worker-side
- notebook/fundamentals/vllm-v1-scheduler-vs-sglang-overlap-scheduling.md
