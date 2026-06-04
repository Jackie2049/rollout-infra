# SGLang RadixAttention 源码分析

> 目录: `sglang/python/sglang/srt/mem_cache/` + `sglang/srt/layers/radix_attention.py`
> 分析日期: 2026-06-04

## 核心创新: RadixTree-based Prefix Caching

vLLM 用 `BlockHashToBlockMap` (hash → block)，SGLang 用 **RadixTree** (前缀树)。

```
vLLM: hash("hello world") → block_id
       需要完整 block 对齐

SGLang:  root
         ├── "hello" → value
         │   ├── " world" → value
         │   └── " SGLang" → value
         └── "hi" → value
              └── " there" → value

        自动识别 "hello" 是 "hello world" 和 "hello SGLang" 的公共前缀
        不依赖 block_size 对齐! 任意 token 边界
```

---

## RadixCache 核心结构

### TreeNode

```python
class TreeNode:
    children: dict[str, TreeNode]     # 子节点 (RadixKey → Node)
    parent: TreeNode
    key: RadixKey                      # 该节点代表的 token 序列
    value: torch.Tensor | None         # KV cache indices (None = evicted)
    lock_ref: int                      # 引用计数 (>0 = 保护免驱逐)
    last_access_time: float            # LRU 驱逐用
    hash_value: list[str] | None       # 子序列 hash (用于 P/D 传输验证)
    host_value: torch.Tensor | None    # 备份到 host 的 KV (swap)
    priority: int                      # 优先级驱逐
```

### RadixKey

```python
class RadixKey:
    token_ids: array[int]     # token 序列
    extra_key: str | None     # 额外键 (lora_id, cache_salt)
    is_bigram: bool           # bigram 模式
```

**extra_key** 设计: 不同 LoRA adapter 或 cache_salt 的请求不共享 KV (即使 token_ids 相同)。

### match_prefix() — 最长前缀匹配

```python
def match_prefix(key) -> MatchResult:
    """
    在 RadixTree 中查找最长匹配前缀:

    1. 从 root 开始, 对比 key 与 child.keys
    2. 如果匹配部分: 分割节点 (node splitting)
    3. 继续匹配剩余部分
    4. 返回: device_indices (拼接的 KV indices) + last_node

    Node splitting:
    tree:  ["hello world" → value]
    查询:  "hello SGLang"
    →
    tree:  ["hello" → None]  ← 新分叉点
            ├── " world" → value
            └── " SGLang" → new_node  ← 新查询
    """
```

**关键优势**: 不要求 block_size 对齐的边界。vLLM 只能在 block 边界共享，SGLang 在任意 token 边界都能共享。

### insert() — 插入新 KV

```python
def insert(key, value, priority):
    """
    将新计算的 KV 插入 RadixTree:

    1. 查找最长已存在前缀
    2. 对未匹配部分创建新节点
    3. 存储 KV cache indices
    4. 更新 LRU 元数据
    """
```

### evict() — 驱逐策略

支持 7 种驱逐策略:
- **LRU**: 最近最少使用
- **LFU**: 最少使用频率
- **FIFO**: 先进先出
- **Priority**: 按优先级
- **Clock**: 时钟算法
- **Random**: 随机
- **None**: 不满不驱逐

---

## vs vLLM Prefix Caching

| 维度 | vLLM | SGLang RadixAttention |
|------|------|----------------------|
| **数据结构** | HashMap (BlockHashToBlockMap) | RadixTree (前缀树) |
| **对齐要求** | block_size 对齐 (默认 16) | **无对齐要求** |
| **节点分裂** | 不支持 | **自动分裂** |
| **LPM 匹配** | 精确 hash match | **最长前缀匹配 (LPM)** |
| **驱逐策略** | LRU only | LRU/LFU/FIFO/Priority/Clock/Random |
| **内存开销** | O(1) per block | O(nodes) per unique prefix |
| **多轮对话加速** | ~3x | **~5x** (无对齐浪费) |

---

## RadixAttention Kernel

```python
# layers/radix_attention.py
class RadixAttention:
    """
    vLLM-style PagedAttention, 但 block_table 由 RadixTree 管理。

    与 vLLM 的区别:
    - block_table 是 RadixTree match 的结果
    - 支持非 block_size 对齐的共享 (更精细)
    - 缓存感知调度: 优先调度可共享前缀的请求到同一 batch
    """
```

### 缓存感知调度 (Cache-Aware Scheduling)

SGLang Scheduler 利用 RadixTree 的 LPM 特性:
```python
# 先将 request 按 prefix 分组
# 同组内最大化 batch 内 prefix sharing
# → 减少 KV cache re-computation
```

---

## 关键设计决策

1. **RadixTree vs HashMap**: SGLang 选择了更复杂但更灵活的数据结构
2. **Node splitting**: 无需 block 对齐，任意 token 边界可共享
3. **7 种驱逐策略**: 比 vLLM 的 LRU 更灵活
4. **extra_key 隔离**: LoRA/multi-model 共享同一物理 KV cache 但逻辑隔离
5. **Host offload 支持**: `host_value` 允许 swap 到 CPU

## 代码位置

| 文件 | 内容 |
|------|------|
| `mem_cache/radix_cache.py` | RadixTree 实现 (~800行) |
| `layers/radix_attention.py` | RadixAttention kernel |
| `mem_cache/radix_cache_cpp.py` | C++ 加速版本 |
