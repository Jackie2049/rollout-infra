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

### 3.2 关键指标

```python
torch.cuda.memory_allocated()    # 当前实际使用的 GPU 内存
torch.cuda.memory_reserved()     # 从 CUDA 预留的总内存 (>= allocated)
torch.cuda.max_memory_allocated() # 峰值 allocated
torch.cuda.max_memory_reserved()  # 峰值 reserved

# 碎片化程度 ≈ reserved - allocated
```

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
```

### 3.5 实际训练中的行为

```
Step 1: allocate forward activations  → cache miss → 从 CUDA malloc
Step 1: free activations              → 放回 pool
Step 2: allocate forward activations  → cache hit! → 从 pool 取 (快!)
```

训练中 activation 大小每步相同，cache 命中率极高。这就是为什么训练一旦开始，`nvidia-smi` 显示的显存使用量不会持续增长。

### 3.6 empty_cache()

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

## 参考资料

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [PYTORCH_CUDA_ALLOC_CONF](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- [vLLM PagedAttention Paper (SOSP 2023)](https://arxiv.org/abs/2309.06180)
- [Understanding PyTorch's Caching Allocator](https://zdevito.github.io/2022/08/04/memory-cache.html)
