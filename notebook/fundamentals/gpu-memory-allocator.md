# GPU 内存分配器

> 目标：理解 PyTorch CUDA Caching Allocator 和 vLLM Paged Memory 的内存管理策略

## 1. 问题：GPU 内存碎片化

GPU 内存 (HBM) 是有限资源，分配/释放不当会导致碎片化：

```
初始:  [================ 空闲 8000 MB ================]
分配:  [AA][BB][CCCCC][DD][EE][FF][GGGGG][空闲]
释放:  [  ][BB][     ][DD][EE][FF][     ][空闲]
碎片:  5 个空闲碎片 (总计可能够分配，但不连续 → OOM)
```

**碎片化 OOM**: 总空闲内存足够，但没有足够大的连续块。

## 2. Naive 分配器

直接 malloc/free，最常见的策略：

```
allocate(size):
  找到第一个 >= size 的空闲块
  切出所需大小
  剩余部分变成新的空闲块

free(block):
  标记为空闲
  尝试合并相邻空闲块
```

**问题**：锯齿形分配模式 (大→小→大→小) 导致严重碎片化。

## 3. PyTorch CUDA Caching Allocator

PyTorch 的核心内存管理策略：**不释放回 GPU，而是缓存到 pool**。

### 3.1 核心原理

```
allocate(size):
  1. 从 pool 找 matching size block → 直接复用 (cache hit)
  2. 从 pool 找更大的 block → split 后使用
  3. Pool 不够 → 从 CUDA 真正分配 (malloc)
  4. 都不够 → OOM

free(block):
  放回 pool (不调用 cudaFree)
  尝试 merge 相邻空闲块
```

### 3.2 两级分配结构

PyTorch Caching Allocator 使用 **Segment → Block** 两级结构：

```
Segment: 从 CUDA malloc 获取的大块内存 (cudaMalloc)
  ├── small_pool: 分配 < 1MB
  └── large_pool: 分配 >= 1MB

Block: Segment 内的分配单元
  ├── 在 Segment 内切分
  ├── 可以 split (大→小) 和 merge (相邻合并)
  └── 空闲时留在 pool 中，不释放给 CUDA
```

两个 pool 隔离了小型临时分配和大型张量分配，防止交叉碎片化。

### 3.3 关键指标

```python
torch.cuda.memory_allocated()    # 当前实际使用的 GPU 内存
torch.cuda.memory_reserved()     # 从 CUDA 预留的总内存 (>= allocated)
torch.cuda.max_memory_allocated() # 峰值 allocated
torch.cuda.max_memory_reserved()  # 峰值 reserved

# 内存层次关系:
# nvidia-smi used >= memory_reserved() >= memory_allocated()
#   ↑ 包括 CUDA context, NCCL    ↑ 包括 cached blocks    ↑ 只有活跃张量
```

`inactive_split_bytes` 指标特别重要——它表示已 split 但空闲的字节数，直接反映**碎片化程度**。

### 3.3 Split 和 Merge

**Split**: 大块切成小块
```
[空闲 256 MB] → 分配 64 MB → [已用 64 MB][空闲 192 MB]
```

**Merge**: 相邻空闲块合并
```
[空闲 64 MB][空闲 192 MB] → merge → [空闲 256 MB]
```

### 3.4 配置选项

```bash
# PYTORCH_CUDA_ALLOC_CONF
max_split_size_mb:128       # 超过此大小不 split
garbage_collection_threshold:0.6  # 当 allocated > 60% reserved 时触发 GC
expandable_segments:True    # 允许 segment 动态扩展 (PyTorch 2.x)
                           # vLLM 禁用此选项 (与其内存池不兼容)
backend:cudaMallocAsync     # 使用 CUDA 内置异步分配器
```

### 3.5 实际训练中的行为

```
Step 1: allocate forward activations  → cache miss → 从 CUDA malloc
Step 1: free activations              → 放回 pool
Step 2: allocate forward activations  → cache hit! → 从 pool 取 (快!)
```

训练中 activation 大小每步相同，cache 命中率极高。这就是为什么训练一旦开始，`nvidia-smi` 显示的显存使用量不会持续增长。

### 3.6 OOM 触发机制

当分配器需要新块时：
1. 先在 pool 中查找空闲块 → 有则复用
2. 没有 → 调用 `cudaMalloc()` 获取新 Segment
3. `cudaMalloc` 失败 → **同步所有 stream，释放所有 cached blocks，重试**
4. 重试仍失败 → `torch.cuda.OutOfMemoryError`

`num_alloc_retries` 和 `num_ooms` 在 `memory_stats()` 中可查，非零表示内存压力。

### 3.7 empty_cache()

```python
torch.cuda.empty_cache()  # 释放所有 cached blocks 回 CUDA
```

效果：`reserved` 下降到 `allocated`，但**不影响 `allocated`**。
用途：在需要精确测量内存用量或为其他进程释放 GPU 内存时使用。

## 4. vLLM Paged Memory Pool

vLLM 采用完全不同的策略：**预分配固定大小 block**，类似操作系统的虚拟内存分页。

### 4.1 核心设计

```
Block: 固定大小，存储 N 个 token 的 KV Cache
  - block_size = 16 tokens (默认)
  - 每个 block 大小 = 16 × 2 × num_kv_heads × head_dim × dtype_size

Block Table: 每个请求维护一个 block 列表 (类似页表)
  req_1: [block_0, block_5, block_12, ...]
  req_2: [block_1, block_8, block_3, ...]

Free Pool: 空闲 block 列表
```

### 4.2 为什么固定大小消除碎片化

```
可变大小 (传统):
  [64B][空闲][128B][空闲][256B][空闲] → 分配 200B → OOM (碎片化)

固定大小 (vLLM):
  [Block][Block][Block][Block][Block] → 分配 → [Block] (总是能分配)
```

所有 block 大小相同，任何空闲 block 都能满足请求。

### 4.3 Reference Counting

多个请求可以共享同一个 block（prefix caching）：

```
请求 A: [system_prompt_block] → [user_query_block_A]
请求 B: [system_prompt_block] → [user_query_block_B]
                 ↑ 共享，ref_count=2
```

释放规则：`ref_count -= 1`，当 `ref_count == 0` 时才真正回收。

### 4.4 内存预分配策略

vLLM 的内存预分配分三步（`gpu_worker.py:determine_available_memory()`）：

```python
# Step 1: 计算目标内存
requested_memory = ceil(total_memory * gpu_memory_utilization)  # 80GB × 0.9 = 72GB

# Step 2: 运行 profiling forward pass 测量峰值激活
# 测量: torch_peak_increase (激活值) + non_torch_increase (CUDA context 等)

# Step 3: 计算 KV Cache 可用内存
non_kv_memory = weights + torch_peak_increase + non_torch_increase + cudagraph_estimate
available_kv = requested_memory - non_kv_memory
num_blocks = available_kv // page_size // num_layers
```

**关键**: vLLM 运行一次虚拟 forward pass 来精确测量峰值激活内存，而不是估算。

### 4.5 物理内存分配

实际分配在 `gpu_model_runner.py:_allocate_kv_cache_tensors()` 中：

```python
# 分配一个扁平 int8 缓冲区
for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
    tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)
    # 多层共享同一物理内存（通过 reshape）

# Reshape 为每层的 KV Cache 形状
# [num_blocks, block_size, num_kv_heads, head_size]
```

### 4.6 BlockPool 源码结构

vLLM V1 的 `BlockPool`（`vllm/v1/core/block_pool.py`）：

```
BlockPool:
  ├── blocks: list[KVCacheBlock]           # 所有 block (按 block_id 索引)
  ├── free_block_queue: FreeKVCacheBlockQueue  # 空闲 block 双向链表 (LRU 顺序)
  └── cached_block_hash_to_block: BlockHashToBlockMap  # prefix cache 哈希表

分配流程:
  1. get_new_blocks(n) → 从 free_block_queue.pop(n)
  2. 如果空闲不足 → _maybe_evict_cached_block() (LRU 驱逐)
  3. 每个 block 的 ref_cnt += 1

释放流程:
  1. free(block) → ref_cnt -= 1
  2. ref_cnt == 0 → 放回 free_block_queue (尾部 = 低驱逐优先级)
  3. prefix-cached block → 放入 hash map 等待复用
```

```python
# vLLM 启动时的内存预分配
total_gpu_memory = get_gpu_memory()          # 80 GB (A100)
non_kv_memory = model_weights + misc          # ~15 GB
kv_cache_budget = total_gpu_memory * 0.9 - non_kv_memory  # ~57 GB
num_blocks = kv_cache_budget // block_size     # 计算 block 数量
preallocate(num_blocks)                        # 一次性分配所有 block
```

`gpu_memory_utilization` 参数 (默认 0.9)：预留 10% 给其他用途。

### 4.5 Swap to CPU

当 GPU block 不够时，可以将 block swap 到 CPU 内存：

```
GPU: [Block_1][Block_2][Block_3] → 满了!
Swap out Block_1 → CPU
GPU: [空闲][Block_2][Block_3]
分配新 block → GPU
需要 Block_1 时 → Swap in from CPU
```

vLLM V1 默认不使用 swap（使用 recomputation 代替）。

## 5. 训练 vs 推理的内存管理对比

| 特性 | 训练 (PyTorch) | 推理 (vLLM) |
|------|---------------|------------|
| 分配策略 | Caching Allocator | Paged Memory Pool |
| 块大小 | 可变 | 固定 |
| 碎片化 | 有 (但 cache 缓解) | 无 (固定大小) |
| 生命周期 | 一步内分配/释放 | 请求级别 |
| 共享 | 无 | Prefix Caching |
| 预分配 | 按需 | 启动时一次性 |
| CPU Swap | 少用 | 可选 |

## 6. 内存 Profiling 工具

### PyTorch

```python
# 内存快照
print(torch.cuda.memory_summary())

# 关键指标
allocated = torch.cuda.memory_allocated() / 1e9     # GB
reserved = torch.cuda.memory_reserved() / 1e9        # GB
fragmentation = (reserved - allocated) / reserved * 100  # %

# 内存历史 (PyTorch 2.x)
torch.cuda.memory._record_memory_history()
# → 生成 memory profile 文件，可用可视化工具分析
```

### vLLM

```python
# vLLM 内存统计
from vllm import LLM
llm = LLM(model="...")
# 启动日志中可以看到:
# "# GPU blocks: XXXX, # CPU blocks: XXXX"
# "KV cache size: XX.XX GiB"
```

## 实验验证

- `tools/memory_allocator_sim.py` — 内存分配器模拟器（4 个实验）
  - 实验 1: Naive vs Caching 碎片化对比
  - 实验 2: 训练场景内存分配模拟 (LLaMA-7B on A100)
  - 实验 3: vLLM 分页内存池模拟 (prefix caching, reference counting)
  - 实验 4: 锯齿形 OOM 抵抗力对比

关键发现：
- PyTorch Caching Allocator 通过 split/merge 减少碎片化
- `reserved >= allocated`（差值 = cached blocks）
- vLLM 固定大小 block 完全消除碎片化
- Prefix Caching 通过引用计数共享公共前缀 block

## 7. vLLM V1 源码深度解析

基于 vLLM V1 源码 (`vllm/v1/core/`) 的深度分析。

### 7.1 BlockPool 核心数据结构

```python
# block_pool.py — V1 BlockPool
class BlockPool:
    blocks: list[KVCacheBlock]                    # 所有 block (按 block_id 索引)
    free_block_queue: FreeKVCacheBlockQueue        # 空闲 block 双向链表 (LRU)
    cached_block_hash_to_block: BlockHashToBlockMap # prefix cache 哈希表 (1:N)
    null_block: KVCacheBlock                       # block_id=0 占位符
```

### 7.2 FreeKVCacheBlockQueue — O(1) 双向链表

核心设计：**不分配任何 Python 对象**，直接操作 `prev_free_block` / `next_free_block` 属性：

```
fake_head ←→ [Block_3] ←→ [Block_7] ←→ [Block_1] ←→ fake_tail
             ↑ LRU (最先驱逐)                    ↑ MRU (最后驱逐)

popleft()  : 从 fake_head 端弹出 (驱逐最久未用的)
append()   : 从 fake_tail 端插入 (刚释放的 block)
remove()   : O(1) 删除中间节点 (prefix cache hit 时)
```

设计要点：
- **fake_head / fake_tail**：哨兵节点，减少边界判断分支
- **LRU 顺序**：释放时 block 被反转后 append，保证同请求的尾部 block 先被驱逐
- **popleft_n()**：批量弹出，单次遍历，重置所有 prev/next 指针

### 7.3 KVCacheBlock 元数据

```python
# kv_cache_utils.py
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                           # 0 ~ num_gpu_blocks-1
    ref_cnt: int = 0                        # 引用计数
    _block_hash: BlockHashWithGroupId | None # 仅 full+cached 时设置
    prev_free_block: KVCacheBlock | None    # 双向链表
    next_free_block: KVCacheBlock | None
    is_null: bool = False                   # null_block 占位
```

**生命周期**：
1. 初始化：`ref_cnt=0`, 在 `free_block_queue` 中
2. 分配：`get_new_blocks()` → `ref_cnt += 1`, 从 `free_queue` 移除
3. Prefix cache hit：`touch()` → `ref_cnt += 1` (共享 block)
4. 释放：`free_blocks()` → `ref_cnt -= 1`, 若 ==0 则放回 `free_queue`
5. 驱逐：`_maybe_evict_cached_block()` → 清除 hash, 从 cache map 移除

### 7.4 BlockHashToBlockMap — 1:N Hash Map

```python
class BlockHashToBlockMap:
    _cache: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]

    # 1:1 常见情况 (单个 block):  hash → KVCacheBlock
    # 1:N 重复前缀:               hash → {block_id: KVCacheBlock, ...}
```

**为什么允许 1:N？** prefix caching 不做去重。两个请求生成相同前缀时，各自有自己的 block，但 hash 相同。这样 block table 保持 append-only，不需要修改已有 block_id。

### 7.5 Block Hash 计算链

```python
# 增量哈希链: 每个 block 的 hash = H(parent_hash + token_ids + extra_keys)
hash = hash_function((parent_block_hash, tuple(token_ids), extra_keys))
```

- `parent_block_hash`：前一个 block 的 hash (链式依赖)
- `extra_keys`：多模态 (mm_hash, offset)、LoRA name、cache_salt、prompt_embeds
- 第一个 block 的 parent 是 `NONE_HASH` (随机种子或 PYTHONHASHSEED)

### 7.6 Hybrid KV Cache Group

混合注意力模型 (如 Gemma3: 5 SW + 1 Full) 的 KV cache 管理：

```
模型层: [Full, SW, SW, SW, SW, SW] × N 组

KV Cache Group:
  Group 0 (Full): [full_0, full_1, ..., full_N-1] — 共享 block_table[0]
  Group 1 (SW):   [sw_0, sw_1, ..., sw_N-1]      — 共享 block_table[1]
  ...

内存约束: 所有 group 的 page_size_bytes 必须相同 (防碎片化)
```

分组算法 (`_get_kv_cache_groups_uniform_page_size`):
1. 按 KVCacheSpec 类型分组
2. 用最小层数的组作为 group_size
3. 按 PP stage 分配层 (跨 stage 均匀分布)

### 7.7 Auto-fit max_model_len

当 `max_model_len=-1` 时，二分搜索最大可用上下文长度：

```python
def estimate_max_model_len(vllm_config, kv_cache_spec, available_memory):
    # 二分搜索
    left, right = 1, original_max_model_len
    while left <= right:
        mid = (left + right) // 2
        vllm_config.model_config.max_model_len = mid
        memory_needed = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())
        if memory_needed <= available_memory:
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    return result
```

关键：跨所有 worker (PP) 取最小值，保证最受限的 worker 也能运行。

### 7.8 PyTorch ExpandableSegment (2.x)

PyTorch 2.x 引入 `ExpandableSegment` 解决碎片化：

```python
# 使用 CUDA 虚拟内存管理 API:
cuMemAddressReserve(256 * TiB)    # 预留虚拟地址空间
cuMemCreate(physical_page)         # 按需创建物理页
cuMemMap(vaddr, physical_page)     # 映射到虚拟地址
cuMemSetAccess(vaddr, READ_WRITE)  # 设置权限

# 增长: 新分配 → cuMemCreate + cuMemMap → append to segment
# 收缩: OOM 时 cuMemUnmap → 释放物理页给 CUDA
```

**优势**：N→N+1 batch size 变化时，在同一 segment 内增长，不产生碎片。
**限制**：不支持 IPC (多进程)，`cudaDeviceEnablePeerAccess` 不兼容。

### 7.9 两种分配器的根本区别

| 维度 | PyTorch CachingAllocator | vLLM Paged BlockPool |
|------|-------------------------|---------------------|
| 内存来源 | 按 segment 从 CUDA malloc | 启动时预分配扁平 buffer |
| 分配单元 | 可变大小 Block | 固定大小 KVCacheBlock |
| 查找结构 | std::set (best-fit) | 双向链表 FIFO (LRU) |
| 碎片化策略 | split/merge/expandable_segments | 固定大小完全避免 |
| 共享机制 | 无 | BlockHashToBlockMap (prefix) |
| 引用计数 | event_count (stream sync) | ref_cnt (跨请求共享) |
| 适用场景 | 通用 tensor 工作负载 | KV cache 专用 |

## 8. 关键源码文件

| 文件 | 行数 | 核心内容 |
|------|------|---------|
| `vllm/v1/core/kv_cache_utils.py` | ~2150 | KVCacheBlock, FreeKVCacheBlockQueue, 哈希计算, auto-fit |
| `vllm/v1/core/block_pool.py` | ~520 | BlockPool, BlockHashToBlockMap, LRU 驱逐 |
| `vllm/v1/core/kv_cache_manager.py` | ~600 | KVCacheManager 接口, 分配/释放/前缀查找 |
| `vllm/v1/worker/gpu_worker.py` | ~500 | 内存 profiling, available_memory 计算 |
| `vllm/v1/worker/gpu/model_runner.py` | ~1200 | KV cache tensor 分配, block table 初始化 |
| `c10/cuda/CUDACachingAllocator.cpp` | ~4800 | PyTorch CachingAllocator C++ 实现 |

## 参考资料

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [PYTORCH_CUDA_ALLOC_CONF](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- [vLLM PagedAttention Paper (SOSP 2023)](https://arxiv.org/abs/2309.06180)
- [Understanding PyTorch's Caching Allocator](https://zdevito.github.io/2022/08/04/memory-cache.html)
