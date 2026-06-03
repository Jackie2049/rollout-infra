# vLLM V1 KV Cache Manager 源码阅读

> 深入理解 vLLM V1 的 KV Cache 管理架构：从 Block 分配到 Prefix Caching

## 1. 核心文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `kv_cache_interface.py` | 877 | KV cache 接口定义、AttentionSpec 层次结构 |
| `kv_cache_manager.py` | 572 | 主 KVCacheManager 入口，连接 Scheduler |
| `single_type_kv_cache_manager.py` | 1347 | 各 Attention 类型的专用 Manager |
| `block_pool.py` | 520 | BlockPool 块池管理、free queue |
| `kv_cache_coordinator.py` | 691 | 跨 KV cache group 协调器 |
| `kv_cache_utils.py` | 2153 | KVCacheBlock 数据结构、BlockHash、工具函数 |
| `kv_cache_spec_registry.py` | 3138 | 可插拔 KVCacheSpec 注册表 |
| `sched/scheduler.py` | 2374 | 主调度器，创建和管理 KVCacheManager |

## 2. 核心数据结构

### KVCacheBlock

```python
# kv_cache_utils.py:116-150
class KVCacheBlock:
    block_id: int           # Block ID (0 to num_gpu_blocks-1)
    ref_cnt: int            # 引用计数
    _block_hash: BlockHashType  # Prefix caching hash (满块时计算)
    prev_free_block: Optional[KVCacheBlock]  # 双向链表
    next_free_block: Optional[KVCacheBlock]
    is_null: bool           # null block 标记 (不缓存)
```

### KVCacheBlocks

```python
# kv_cache_manager.py:25-108
@dataclass
class KVCacheBlocks:
    blocks: Tuple[Sequence[KVCacheBlock], ...]
    # Scheduler 和 KVCacheManager 的接口层
    # 提供 block ID 提取和操作方法
```

### BlockPool

```python
# block_pool.py:130+
class BlockPool:
    free_block_queue: FreeBlockQueue           # 空闲块双向链表
    cached_block_hash_to_block: BlockHashToBlockMap  # Prefix cache hash map
    null_block: KVCacheBlock                   # 特殊占位块 (block_id=0)
```

## 3. Block 分配与释放流程

### 分配

1. Scheduler 调用 `KVCacheManager.get_computed_blocks()`
2. Coordinator 委派给对应的 `SingleTypeKVCacheManager`
3. Manager 从 `BlockPool.free_block_queue` 取空闲块
4. 块满时计算 hash 用于 prefix caching

### 释放

1. 引用计数 `ref_cnt` 追踪使用情况
2. `ref_cnt == 0` 时返回 free queue
3. Hash 被重置，块从缓存 map 移除
4. null block 永远不释放

## 4. Manager 类型层次

`SingleTypeKVCacheManager` 是抽象基类，有 6 个具体实现：

| Manager | 用途 |
|---------|------|
| `FullAttentionManager` | 标准 MHA/GQA |
| `SlidingWindowManager` | 滑动窗口注意力 |
| `MLAAttentionManager` | DeepSeek-V2 MLA |
| `MambaManager` | Mamba SSM |
| `ChunkedLocalManager` | 分块局部注意力 |
| `CrossAttentionManager` | 交叉注意力 (encoder-decoder) |

每种 Manager 处理不同的 block 分配/释放策略，例如 SlidingWindow 有回收机制。

## 5. Prefix Caching 实现

### BlockHashToBlockMap (block_pool.py:34-127)

核心数据结构，支持 hash→blocks 的 1:N 映射：

```python
class BlockHashToBlockMap:
    _cache: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]

    def get_one_block(key) -> KVCacheBlock | None  # 取任意一个
    def insert(key, block) -> None                  # 插入（1→dict 升级）
    def pop(key, block_id) -> KVCacheBlock | None   # 弹出指定 block_id
```

设计要点：
- **不去重**: 相同 hash 的多个 block 都保留，保证 block table append-only
- **GC 优化**: 单 block 直接存 KVCacheBlock，多 block 用 dict（避免嵌套 dict）
- **group_id 隔离**: 不同 KV cache group 的 block 独立管理

### cache_full_blocks() 流程 (block_pool.py:211-331)

1. 取 `[num_cached_blocks:num_full_blocks]` 的满块
2. 获取 request 的 precomputed block_hashes
3. 对每个满块（跳过 null/masked blocks）：
   - 计算 `block_hash_with_group_id`
   - 设置 `blk.block_hash`
   - 插入 `cached_block_hash_to_block`
4. 如果启用 kv_cache_events，生成 `BlockStored` 事件

### get_cached_block() (block_pool.py:184-209)

- 遍历所有 `kv_cache_group_ids`
- 对每个 group 查找 hash→block 映射
- 任一 group miss → 整体返回 None
- 支持多 block_size（通过 `BlockHashListWithBlockSize` 转换）

### block_mask 机制

SWA/Mamba 等稀疏注意力模式使用 `block_mask` 选择性缓存：
- mask=False 的 block 不参与 prefix caching
- 避免永远不会被 hit 的 block 占用缓存空间

### Hash 计算

- 块满时计算 `BlockHash`
- Hash 包含块内容 + group ID (唯一性)
- 存储在 `BlockHashToBlockMap`

### Cache 查找

```
请求到达 → 计算 prefix hash → 查找 cached_block_hash_to_block
  → 命中: 复用 cached blocks, 跳过分配
  → 未命中: 分配新 blocks
```

### 淘汰策略

- 空闲块按 LRU 顺序维护
- 内存压力时淘汰最近最少使用的块
- 指标追踪 cache hit/miss ratio

## 6. Scheduler 与 KVCacheManager 关系

```
Scheduler
  ├── 创建 KVCacheManager (配置驱动)
  │     ├── 创建 BlockPool
  │     └── 创建 KVCacheCoordinator
  │           └── 创建各类型 SingleTypeKVCacheManager
  ├── 调度请求时调用 get_computed_blocks()
  │     → 返回 KVCacheBlocks (已分配的 blocks)
  ├── 跟踪 block 使用和引用计数
  └── 收集 KV cache 指标 (使用率, cache hit rate)
```

## 7. 关键架构特点

1. **模块化设计**: 不同 attention 类型有独立 Manager
2. **可插拔注册表**: 自定义 KVCacheSpec 无需修改核心代码
3. **引用计数**: 安全的 block 管理，自动回收
4. **Hash 缓存**: 基于 content 的 prefix caching
5. **分布式支持**: 内置 KV Transfer Connector 支持 P/D 分离
6. **多 Attention 模式**: Full/SW/MLA/Mamba/Chunked/Cross

## 8. 与 V0 的关键区别

| 方面 | V0 | V1 |
|------|----|----|
| 调度 | prefill/decode 分离 | 统一调度 |
| KV Cache | BlockSpaceManager | KVCacheManager + BlockPool |
| Prefix Cache | hash chain (block_table) | BlockHashToBlockMap |
| Attention 类型 | 仅 FullAttention | Full/SW/MLA/Mamba/Chunked/Cross |
| MLA 支持 | 无 | 原生 MLA Manager |

## 参考资料

- 源码路径: `vllm-latest/vllm/v1/core/`
- 相关笔记: [KV Cache 深度解析](../fundamentals/kv-cache.md), [Prefix Caching 深度对比](../fundamentals/prefix-caching.md)
- 相关源码阅读: [vLLM V1 Scheduler](vllm-v1-scheduler-reading.md), [vLLM MLA Backend](vllm-mla-backend-reading.md)
