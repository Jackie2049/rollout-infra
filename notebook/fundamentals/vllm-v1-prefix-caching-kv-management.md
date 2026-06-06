# vLLM V1 KV Cache Management & Prefix Caching 深度分析

> 2026-06-07 | vLLM V1 源码深度阅读

## 架构总览

vLLM V1 的 KV cache 管理采用 **分层架构**:

```
KVCacheManager (接口层)
    ├── KVCacheCoordinator (协调层)
    │   ├── BlockPool (全局block池)
    │   │   ├── FreeKVCacheBlockQueue (O(1) 双向链表 LRU)
    │   │   ├── BlockHashToBlockMap (1:N hash→blocks 映射)
    │   │   └── null_block (占位符)
    │   └── SingleTypeKVCacheManager × N (每种attn类型一个)
    │       ├── FullAttentionManager (标准MHA/GQA)
    │       ├── SlidingWindowManager (滑动窗口)
    │       ├── ChunkedLocalAttentionManager (局部attention)
    │       ├── MambaManager (SSM模型)
    │       ├── MLAAttentionManager (DeepSeek-V2 MLA)
    │       └── CrossAttentionManager (编码器-解码器)
    └── KVCacheBlocks (返回值, 隔离scheduler与内部数据结构)
```

## 核心数据结构

### 1. KVCacheBlock (`kv_cache_utils.py:117-162`)

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int          # 物理block编号 (0 ~ num_gpu_blocks-1)
    ref_cnt: int = 0       # 引用计数 (有多少请求共享此block)
    _block_hash: BlockHashWithGroupId | None = None  # hash键 (仅满block才有)
    prev_free_block: KVCacheBlock | None = None  # 双向链表前指针
    next_free_block: KVCacheBlock | None = None  # 双向链表后指针
    is_null: bool = False  # null block标记 (SWA中占位)
```

**关键**: `block_hash` 只在 block **满** 且被缓存时才设置. 设置后不可更改 (assert 保护).

### 2. FreeKVCacheBlockQueue (`kv_cache_utils.py:165-299`)

**O(1) 双向链表实现** — 不用 Python deque (C++ deque更快但无法O(1)中间删除):

```
fake_head → [LRU block] → [block] → ... → [MRU block] → fake_tail
```

- `popleft()`: 从头部取最久未用block (LRU驱逐)
- `popleft_n(n)`: 批量取n个block
- `remove(block)`: O(1)删除中间block (从free list移到allocated)
- `append(block)`: 添加到尾部 (新释放的block)

**驱逐顺序**:
1. LRU优先 (最近最少使用)
2. 同一序列释放的block, 反转顺序 → 尾部block (更近的hash) 先被驱逐

### 3. BlockHashToBlockMap (`kv_cache_manager.py:34-127`)

**1:N 映射** — 一个 hash 可以映射到多个物理 block:

```python
class BlockHashToBlockMap:
    _cache: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]

    # 单block: hash → KVCacheBlock (直接)
    # 多block: hash → {block_id: KVCacheBlock} (dict)
```

**为什么需要 1:N?**:
- 同样的 token 序列可能在不同请求中产生相同 hash
- 不同请求分配不同物理 block, 但 hash 相同 (prefix sharing!)
- `get_one_block()`: 返回任意一个 (用于cache hit查找)
- `insert()`: 单block→直接存储, 多block→升级为dict
- `pop()`: 删除特定block_id, dict只剩1个→降级回单block

**NOTE #1**: 不做 de-duplicate! 不会检查已有相同hash的block. 原因: 保持block ID不变 → block_table append-only.
**NOTE #2**: Union类型减少GC开销 (避免对单个block创建内层dict).

### 4. BlockPool (`block_pool.py:130-`)

全局 block 资源池:

```python
class BlockPool:
    blocks: list[KVCacheBlock]            # 所有物理block
    free_block_queue: FreeKVCacheBlockQueue  # 空闲block LRU队列
    cached_block_hash_to_block: BlockHashToBlockMap  # hash查找表
    null_block: KVCacheBlock              # 占位符block (block_id=0)
```

**操作**:
- `get_cached_block(hash, group_ids)`: 查找所有group的cached block → 前缀命中
- `cache_full_blocks(request, blocks, ...)`:
  1. 遍历新满block (`blocks[num_cached:num_full]`)
  2. 设置 `block.block_hash = make_block_hash_with_group_id(hash, group_id)`
  3. 插入 `cached_block_hash_to_block`
  4. 跳过 null block 和 mask=False 的block (SWA不需要的)

## Prefix Caching 流程

### Step 1: 计算block hashes (Request创建时)

Request 对象在创建时立即计算所有 block hashes:
- 使用 **hash chain**: `hash_i = hash_fn(hash_{i-1}, tokens[block_i])`
- 支持 sha256_cbor / xxhash_cbor 两种hash函数
- `NONE_HASH`: chain的起点 (使用 PYTHONHASHSEED 或 os.urandom(32))
- 每个block hash 包含 **group_id**: `BlockHashWithGroupId = BlockHash + group_id.to_bytes(4, "big")`

### Step 2: 查找prefix cache hit (`get_computed_blocks`)

```python
def get_computed_blocks(request: Request) -> tuple[KVCacheBlocks, int]:
    if not enable_caching or request.skip_reading_prefix_cache:
        return empty_blocks, 0  # 跳过prefix cache

    max_cache_hit_length = request.num_tokens - 1  # 必须recompute最后一个token
    computed_blocks, num_new_tokens = coordinator.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )
```

**为什么 `max_cache_hit_length = num_tokens - 1`?**:
- 如果所有prompt tokens都命中cache → 仍然需要recompute最后一个token获取logits
- 这是vLLM的设计约束: logits需要最后token的hidden state

### Step 3: `find_longest_cache_hit` (FullAttentionManager)

```python
@classmethod
def find_longest_cache_hit(cls, block_hashes, max_length, ...):
    max_num_blocks = max_length // block_size
    for block_hash in itertools.islice(block_hashes, max_num_blocks):
        # 遍历block hash chain, 找最长prefix hit
        cached_block = block_pool.get_cached_block(
            block_hash, kv_cache_group_ids
        )
        if cached_block:
            computed_blocks.append(cached_block)
        else:
            break  # 一旦miss, 后续不可能hit (hash chain)
```

**关键**: Hash chain 的递推性质 → 一旦miss就停止! 不需要继续查找.
- `hash_i = hash(hash_{i-1}, tokens_i)` → 依赖前一个hash
- 如果 `hash_j` miss → `hash_{j+1}, hash_{j+2}...` 一定也miss

### Step 4: allocate_slots (分配新block)

Blocks 布局:
```
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
| Prefix-cached tokens (ref_cnt > 0) | not cached by vLLM but connector |
|     to be allocated     |     to be cached      |
```

三个阶段:
1. **释放不必要的 comp blocks** (sliding window外的) + 检查空间
2. **处理prefix tokens**: 释放过期blocks + 分配ext_comp blocks
3. **分配新blocks**: `popleft_n(num_new_blocks)`

### Step 5: cache_full_blocks (缓存满blocks)

每次step后, scheduler调用 `cache_full_blocks`:
- 将满block的hash写入 `cached_block_hash_to_block`
- 增加 `ref_cnt` (共享prefix的请求都引用同一block)
- **Sliding window mask**: SWA外的blocks不缓存 (mask=False)

## LRU Eviction 机制

### 何时驱逐?

当 `free_block_queue.num_free_blocks` 不足以分配新请求时:
- Scheduler 触发 preemption (抢占running requests)
- 释放被抢占请求的blocks → `free_request_blocks`

### 驱逐策略

1. **优先驱逐 ref_cnt=0 的cached blocks** (prefix cache中无人引用的)
2. **FreeKVCacheBlockQueue LRU顺序**: 最久未用block在队头
3. **Hash chain reversal**: 同一请求释放的blocks反转 → 尾部(tail hash)先被驱逐

### 为什么反转?

```
Request blocks: [block_0(hash_0)] → [block_1(hash_1)] → [block_2(hash_2)]

释放时反转: block_2 → block_1 → block_0 进入 free queue

效果: block_2 (tail hash) 最先被驱逐
      block_0 (root hash) 最后被驱逐
```

原因: root hash 更有价值 → 更多prefix可以匹配. Tail hash 只对完整序列有用.

## 与 SGLang RadixAttention 的对比

| 特性 | vLLM V1 | SGLang RadixAttention |
|------|---------|----------------------|
| 数据结构 | BlockHashToBlockMap (1:N dict) | RadixTree (节点分裂) |
| Block对齐 | 必须对齐block_size (16) | 无对齐限制 (任意长度) |
| Hash函数 | sha256_cbor / xxhash_cbor chain | 内部hash |
| 共享方式 | ref_cnt计数 | lock_ref保护链 |
| 驱逐策略 | LRU (双向链表) | 7种策略 (LRU/LFU/...) |
| 查找效率 | O(N_blocks) hash chain | O(log N) tree search |
| 批内前缀 | scheduler.find_prefix_cache_hit | cache_aware_schedule LPM |
| 灵活性 | block对齐限制 | 节点分裂无限制 |
| 实际效果 | 简单高效 | 更灵活但更复杂 |

**vLLM的优势**: 实现简单, O(1) eviction, block-based与Paged Attention天然对齐
**SGLang的优势**: 无block对齐限制, 树形结构更自然, 多种驱逐策略

## Hybrid KV Cache Group (混合KV)

vLLM V1 支持 **多种attention类型混合** (KVCacheGroupSpec):

```
KVCacheConfig:
    kv_cache_groups: [
        KVCacheGroupSpec(FullAttentionSpec, block_size=16),  # 标准attention
        KVCacheGroupSpec(SlidingWindowSpec, block_size=16),  # 滑动窗口
        KVCacheGroupSpec(MambaSpec, block_size=1),           # SSM模型
    ]
```

**协调器**: `KVCacheCoordinator` 确保所有group的prefix hit长度一致 → 取 **最短公共前缀**

## 实际性能数据 (来自模拟器)

| 场景 | Prefix复用率 | Cache节省 |
|------|-------------|----------|
| GRPO n=8, prefix=512 | 68% | 88% KV计算 |
| 5轮对话 (RadixAttention) | 66.7% | 75% 累积 |
| Multi-turn, prefix>128 | 正ROI | >50% |
| RTX 4090 prefix=1024 | 91.7% | 2.79x加速 |

## 关键源码文件

| 文件 | 行数 | 内容 |
|------|------|------|
| `kv_cache_manager.py` | ~450 | BlockHashToBlockMap + KVCacheManager接口 |
| `kv_cache_utils.py` | ~300+ | KVCacheBlock + FreeKVCacheBlockQueue + hash函数 |
| `block_pool.py` | ~300+ | BlockPool (全局资源池) |
| `kv_cache_coordinator.py` | ~200+ | 协调器 + group间prefix hit |
| `single_type_kv_cache_manager.py` | ~1200+ | 5种attn类型的Manager实现 |
| `scheduler.py` | ~2400 | 调度逻辑 (触发prefix cache查找) |

## 代码质量观察

1. **Union类型优化**: BlockHashToBlockMap 用 `KVCacheBlock | dict[int, KVCacheBlock]` → 减少GC
2. **Fake head/tail**: FreeKVCacheBlockQueue 用fake node → 减少分支判断
3. **Append-only block_table**: 不de-duplicate → block ID不变 → block_table简单
4. **Slots=True**: KVCacheBlock 用 `@dataclass(slots=True)` → 减少内存开销
5. **Hash chain**: 递推hash → 一旦miss就停 → 高效prefix查找