# vLLM V1 Prefix Caching 源码深度分析

> 基于源码阅读 — hash chain / block reuse / eviction / V0→V1 对比

## 1. Hash 计算 — Block 级哈希链

### 关键文件
`vllm/v1/core/kv_cache_utils.py` (L541-568)

### Hash Chain 机制
```
每个 block 的 hash 依赖于:
  hash = hash_function((parent_block_hash, curr_block_token_ids, extra_keys))

形成链式结构:
  Block 0: hash(NONE_HASH, tokens[0:B], extra_keys)
  Block 1: hash(Block0_hash, tokens[B:2B], extra_keys)
  Block 2: hash(Block1_hash, tokens[2B:3B], extra_keys)
  ...

NONE_HASH: 随机 32 字节值 (用于第一个 block)

extra_keys 包含:
  - LoRA adapter ID
  - 多模态特征
  - cache salt
  - prompt embeddings
```

### 配置
- `prefix_caching_hash_algo`: 默认 "sha256"
- `hash_block_size`: 可独立于物理 block size 的哈希粒度

## 2. Block 命中检测 — 最长前缀匹配

### 关键文件
`vllm/v1/core/kv_cache_coordinator.py`

### 不同 Attention 类型的匹配策略

| 类型 | 搜索方向 | 特点 |
|------|---------|------|
| FullAttention | 左→右 | 线性扫描, 最简单 |
| SlidingWindow | 右→左 | 需要连续 block |
| ChunkedLocal | 右→左 | 有 early stopping |
| Mamba | 右→左 | 有对齐约束 |

### FullAttention 匹配流程
```
1. 计算请求的 block hash 链
2. 从左到右逐个查找 hash_map
3. 找到第一个 miss → 返回已匹配的 block 数量
4. 匹配的 block 直接复用 (ref_cnt +1)
5. 未匹配的部分需要重新计算
```

## 3. Block Pool 与缓存管理

### BlockHashToBlockMap
`vllm/v1/core/block_pool.py` (L34-128)
```
数据结构: Dict[BlockHash, KVCacheBlock]
  - 同一个 hash 可以映射到多个 block (不同 KV cache group)
  - 提供 get(), insert(), remove(), evict() 操作
```

### FreeKVCacheBlockQueue
```
双向链表, LRU 顺序:
  - 头部: 最久未使用的 block → 优先分配
  - 尾部: 最近使用的 block
  - O(1) 的插入/删除操作
```

## 4. Block 生命周期

```
分配: get_new_blocks()
  → 从 free_queue 取 block (优先取 LRU 的)
  → 如果 free_queue 空 → 调用 _maybe_evict_cached_block()

哈希: cache_full_blocks()
  → block 填满后计算 hash
  → 插入 cached_block_hash_to_block map

复用: find_longest_cache_hit()
  → 查找 hash map 中已有的 block
  → touch() 增加 ref_cnt → 防止被驱逐

释放: free_blocks()
  → ref_cnt -1
  → 如果 ref_cnt == 0 → 放回 free_queue

驱逐: _maybe_evict_cached_block()
  → 从 cache hash map 移除
  → block 可重新分配给新请求
```

## 5. APC 启用逻辑

```
配置: cache_config.enable_prefix_caching (默认 True)

启用流程:
  1. Engine 初始化 → _initialize_kv_caches()
  2. resolve_kv_cache_block_sizes() 确定哈希 block 大小
  3. KVCacheManager(enable_caching=True)
  4. BlockPool 初始化缓存

禁用时:
  → 使用 KVCacheCoordinatorNoPrefixCache
  → 不做 hash, 不缓存 block
```

## 6. V0 vs V1 差异

### 架构变化
```
V0: 单一 BlockManager
V1: 分层设计
    KVCacheCoordinator → KVCacheManager → BlockPool
    支持 hybrid KV cache (多种 attention 类型)
```

### V1 新特性
1. **Hybrid KV Cache**: 支持多个 KV cache group (不同 attention 类型)
2. **Hash 粒度控制**: hash_block_size 可独立于物理 block size
3. **EAGLE/MTP 支持**: drop_eagle_block 参数 (speculative decoding)
4. **对齐约束**: cache hit 必须对齐 scheduler_block_size
5. **更好指标**: block residency 和 cache hit 的 metrics

### 性能提升
- LRU free block queue: O(1) 操作
- 专门的 hit 检测 (不同 attention 类型)
- 支持 chunked prefill + prefix caching
- 更好的引用计数内存管理

## 7. 与 verl PrefixGrouper 的关联

```
vLLM prefix caching (推理侧):
  - KV Cache 级别复用
  - 自动检测相同前缀
  - 跨请求共享 KV block

verl PrefixGrouper (训练侧):
  - Self-attention 级别复用
  - 分解共享 prefix 的冗余计算
  - 组内请求共享 prefix 计算

两者互补:
  训练用 PrefixGrouper → 减少训练计算
  推理用 prefix caching → 减少 KV 重计算
```

## 参考

- 源码: `vllm/v1/core/kv_cache_utils.py`
- 源码: `vllm/v1/core/kv_cache_coordinator.py`
- 源码: `vllm/v1/core/block_pool.py`
- 源码: `vllm/v1/core/kv_cache_manager.py`
- 论文: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
