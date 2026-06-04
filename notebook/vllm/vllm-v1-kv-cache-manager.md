# vLLM V1 KV Cache Manager 源码分析

> 文件: 6 个核心文件 (~4600 行)
> 分析日期: 2026-06-04

## 架构总览

```
                       ┌── Scheduler ──┐
                       │ schedule()     │
                       │ allocate_slots │ ← 这里调用 KV Cache Manager
                       │ get_computed   │
                       └───────┬────────┘
                               │
                   ┌───────────▼───────────┐
                   │   KV Cache Manager    │  ~572 行
                   │   allocate_slots()    │  核心调度逻辑
                   │   get_computed_blocks()│  Prefix caching 入口
                   │   free()              │  释放 blocks
                   └───────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
    ┌─────────────┐  ┌──────────────┐  ┌────────────────┐
    │ Coordinator │  │  KV Utils    │  │   Interface     │
    │ 691 行      │  │  2153 行     │  │   877 行        │
    │ P/D 协调    │  │  BlockPool   │  │  KVCacheSpec    │
    │ 多 group    │  │  BlockHash   │  │  AttentionSpec  │
    └─────────────┘  │  FreeQueue   │  │  KVQuantMode    │
                     └──────────────┘  └────────────────┘
```

## 核心类型 (kv_cache_interface.py)

### KVQuantMode

```python
class KVQuantMode(IntEnum):
    NONE = 0                  # FP16/BF16
    FP8_PER_TENSOR = 1        # FP8 单 scale (当前主流)
    INT8_PER_TOKEN_HEAD = 2   # INT8 动态 per-token-head scale
    FP8_PER_TOKEN_HEAD = 3    # FP8 动态 per-token-head scale
    NVFP4 = 4                 # NVFP4 打包 (Blackwell)
```

### KVCacheSpec 体系

```
KVCacheSpec (抽象基类)
├── AttentionSpec      # 标准 Attention 的 KV spec
├── MambaSpec          # Mamba SSM 的 state spec
├── EncoderOnlyAttentionSpec  # Encoder-only attention
└── CrossAttentionSpec # Encoder-decoder cross attention
```

每个 spec 定义: `page_size`, `block_size`, `dtype`, `num_layers`, 支持的 `kv_cache_groups`

### KVCacheGroupSpec

```python
class KVCacheGroupSpec:
    kv_cache_spec: KVCacheSpec  # 引用具体的 spec
    num_layers: int             # 该 group 包含的层数
    attention_type: AttentionType # DECODER / ENCODER 等
```

一个模型可能有多个 group (如 Full + Sliding Window)

### KVCacheConfig

```python
class KVCacheConfig:
    kv_cache_groups: list[KVCacheGroupSpec]
    num_blocks: int              # 总 block 数
    block_size: int              # block 大小 (默认 16)
    page_size: int               # page 大小
    enable_prefix_caching: bool
```

---

## KV Cache Manager (kv_cache_manager.py)

### get_computed_blocks() → Prefix Caching 入口

```python
def get_computed_blocks(request) -> tuple[KVCacheBlocks, int]:
    if not enable_caching or skip_reading_prefix_cache:
        return empty_blocks, 0

    max_cache_hit_length = request.num_tokens - 1  # 最后一 token 必须重算
    computed_blocks, num_new_tokens = coordinator.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )
    return create_kv_cache_blocks(computed_blocks), num_new_tokens
```

**关键**: `max_cache_hit_length = num_tokens - 1` — 即使全部 prefix 命中，最后 1 token 也必须重算以获取 logits。

### allocate_slots() → 核心分配逻辑

Block 布局:
```
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
              |   < to be cached >  |      |  < to be computed >      |
----------------------------------------------------------------------
              |            < to be allocated >                        |
----------------------------------------------------------------------
```

- `comp`: 已有的已计算 blocks (ref_cnt 已增加)
- `new_comp`: 命中 prefix cache 的新 blocks
- `ext_comp`: 外部 KV connector 的 tokens (P/D 分离)
- `new`: 需要实际计算的新 tokens
- `lookahead`: Speculative decode 的预留 tokens

返回 `KVCacheBlocks | None` — None 表示分配失败 (memory 不足)。

### 分配流程

```
1. 计算需要的新 block 数量
   total_blocks = (comp + new_comp + ext + new + lookahead tokens) / block_size

2. 在每个 KV cache group 中:
   - 尝试从 FreeKVCacheBlockQueue 获取 free blocks
   - 如果不够 → 执行 eviction (LRU)
   - 如果还不够 → 返回 None (触发 scheduler preemption)

3. 将 new_comp blocks 加入 BlockHashToBlockMap (prefix caching)

4. 更新 block 的 ref_cnt

5. 返回 KVCacheBlocks (每个 group 的 block IDs)
```

### free() → 释放

```python
def free(request):
    # 1. 减少所有 block 的 ref_cnt
    # 2. ref_cnt == 0 时: 如果 enable_caching → 保留在 BlockHashToBlockMap
    #                      否则 → 归还 FreeKVCacheBlockQueue
    # 3. 清理 request 的 block_table
```

---

## KV Cache Utils (kv_cache_utils.py, 2153 行)

核心数据结构:

### BlockHashToBlockMap — Prefix Caching 索引

```python
class BlockHashToBlockMap:
    """hash_value → {block_id: KVCacheBlock} 的 1:N 映射"""

    def cache_full_block(block, block_hash):
        # 满 block 计算 hash → 插入 map
        # 1:N 设计: 多请求可能共享同一 prefix hash

    def get_cached_block(block_hash, kv_cache_group_ids):
        # 遍历所有 group → 找到匹配的 cached block
```

### FreeKVCacheBlockQueue — O(1) 空闲块管理

```python
class FreeKVCacheBlockQueue:
    """双向链表, fake_head/fake_tail 哨兵节点"""
    # 不分配额外 Python 对象 (每个 block 复用)
```

### KVCacheBlock — Block 数据结构

```python
class KVCacheBlock:
    block_id: int
    ref_cnt: int       # 引用计数 (>0 = 使用中)
    block_hash: int | None  # Content hash (用于 prefix caching)
    prev_free_block: KVCacheBlock  # 双向链表指针
    next_free_block: KVCacheBlock
```

---

## KV Cache Coordinator (kv_cache_coordinator.py)

协调多个 KV cache group (如 Full + Sliding Window):

```python
class KVCacheCoordinator:
    def find_longest_cache_hit(block_hashes, max_length):
        # 在所有 group 中找到最长 prefix match
        # 任一 group miss → 整体 miss (必须所有 group 都命中)

    def cache_full_blocks(kv_cache_blocks, block_hashes):
        # 满 block 后计算 hash → 缓存到所有 group
```

### Hybrid KV Cache (Full + SW)

```
Group 0: Full Attention  →  block_size=16, 全序列
Group 1: Sliding Window  →  block_size=16, 最近 4096 tokens

alloc: 两个 group 都要分配
cache: 两个 group 都要缓存
free:  两个 group 都要释放
```

---

## 数据流完整图

```
1. Scheduler.schedule()
   └─ kv_cache_manager.get_computed_blocks(request)
      └─ coordinator.find_longest_cache_hit()
         └─ BlockHashToBlockMap.get_cached_block() × N groups
      → 返回: (new_blocks, num_computed_tokens)

2. 计算 num_new_tokens (= 需要计算的新 token 数)
   └─ kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
      ├─ 计算 total_blocks_needed = ceil(tokens / block_size)
      ├─ FreeKVCacheBlockQueue.dequeue() → free blocks
      │   └─ 不够 → eviction (LRU) → ref_cnt=0 blocks → dequeue
      │       └─ 还不够 → return None → Scheduler preempts
      ├─ BlockHashToBlockMap.cache_full_block() (prefix caching)
      ├─ 更新 block_table[request_id] = [block_ids...]
      └─ 返回: KVCacheBlocks

3. GPUModelRunner.execute_model()
   └─ 使用 block_table 构建 AttentionMetadata.slot_mapping
      └─ Attention backend 写入 KV cache [block_id * block_size + offset]

4. 请求完成 → kv_cache_manager.free(request)
   ├─ ref_cnt-- for all blocks
   ├─ if ref_cnt == 0 and cache:
   │   └─ BlockHashToBlockMap 保留 (留着 prefix caching)
   └─ else:
       └─ FreeKVCacheBlockQueue.enqueue(block)
```

---

## 关键设计决策

1. **1:N hash→blocks 映射**: 允许不同请求共享相同 prefix (不去重)
2. **哨兵节点链表**: fake_head/fake_tail → O(1) alloc + O(1) free
3. **Flat ref_cnt 而非 block_table 反向索引**: 简单可靠
4. **多 group 同时分配**: 任一 group 失败 → 全部回退 (保证一致性)
5. **最后 1 token 不缓存**: 保证始终能计算 logits
6. **LRU 驱逐**: ref_cnt==0 的 block 按 LRU 回收给新请求

## 代码位置速查

| 文件 | 行数 | 内容 |
|------|:---:|------|
| `kv_cache_interface.py` | 877 | KVQuantMode, KVCacheSpec, KVCacheConfig |
| `kv_cache_manager.py` | 572 | allocate_slots, get_computed_blocks, free |
| `kv_cache_utils.py` | 2153 | BlockPool, BlockHashToBlockMap, FreeQueue |
| `kv_cache_coordinator.py` | 691 | 多 group 协调, find_longest_cache_hit |
| `kv_cache_spec_registry.py` | 209 | Spec 注册表 |
| `kv_cache_metrics.py` | 96 | Prefix cache hit rate 统计 |
