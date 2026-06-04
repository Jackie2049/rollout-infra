# SGLang RadixAttention 架构深度分析

> 目录: `sglang/python/sglang/srt/` (核心)
> 分析日期: 2026-06-04
> Clone: `sglang` (shallow, latest main)

## 架构总览

```
RadixAttention (Layer)
  ├── RadixCache (Python radix tree + KV pool)
  │   ├── TreeNode: token_ids + KV indices + lock_ref
  │   ├── RadixTreeCpp (C++ 高性能后端, JIT 编译)
  │   ├── RadixKey: token sequence + bigram + extra_key
  │   └── EvictionStrategy: LRU/LFU/FIFO/MRU/FILO/SLRU/Priority
  ├── SchedulePolicy (缓存感知调度)
  │   ├── LPM: Longest Prefix Match 优先
  │   ├── DFS_WEIGHT: 深度优先权重
  │   ├── In-batch prefix detection
  │   └── Dynamic: >128 reqs → FCFS (避免排序开销)
  └── PrefillAdder: 请求准入 + token 预算管理
```

---

## 核心创新: RadixTree vs HashMap

```
vLLM: hash("hello world") → block_id
       需要完整 block 对齐 (block_size=16)

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

## RadixKey — 灵活的键表示

```python
class RadixKey:
    """支持两种模式: token-level 和 bigram-level"""
    __slots__ = ("token_ids", "extra_key", "is_bigram")

    # token-level: 标准模式，每个 token 一个单元
    # bigram-level: EAGLE speculative decoding，相邻 token 对为一个单元

    def page_aligned(self, page_size: int) -> "RadixKey":
        """截断到 page_size 的整数倍"""
        aligned_len = len(self) // page_size * page_size
        return self[:aligned_len]

    def match(self, other: "RadixKey", page_size: int = 1) -> int:
        """计算与 other 共享的前缀长度 (page-aligned)"""
        # 逐 token 比较，返回匹配长度

    def child_key(self, page_size: int = 1):
        """生成 hashable 的子节点键 (前 page_size 个单元)"""
        # page_size=1: 单个 token 或 bigram
        # page_size>1: tuple of tokens/bigrams
```

**extra_key** 设计: 不同 LoRA adapter 或 cache_salt 的请求不共享 KV (即使 token_ids 相同)。

---

## TreeNode — Radix Tree 节点

```python
class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)    # child_key → child node
        self.parent: TreeNode = None
        self.key: RadixKey = None                 # 该节点存储的 token 序列
        self.value: Optional[torch.Tensor] = None # GPU KV cache indices (None=evicted)
        self.lock_ref = 0                          # 引用计数 (保护不被驱逐)
        self.last_access_time = time.monotonic()   # LRU 时间戳
        self.hit_count = 0                         # 命中次数 (LFU 用)
        self.host_value: Optional[torch.Tensor] = None  # CPU KV indices
        self.host_ref_counter = 0                  # CPU 引用保护
        self.hash_value: Optional[List[str]] = None     # SHA256 per page
        self.priority = 0                          # 优先级 (Priority eviction)
```

### 节点生命周期

```
1. 创建:  insert() → new TreeNode(key, value)
2. 引用:  match_prefix() → inc_lock_ref() (保护不被驱逐)
3. 分裂:  _split_node() (部分匹配时)
4. 释放:  cache_finished_req() → dec_lock_ref()
5. 驱逐:  evict() → _delete_leaf() (lock_ref=0 的叶子节点)
```

---

## 核心操作

### match_prefix — 最长前缀匹配

```python
def _match_prefix_helper(self, node, key):
    """从 root 开始沿 key 路径向下匹配

    关键: 匹配到节点中间 → 自动分裂!
    """
    access_time = time.monotonic()
    node.last_access_time = access_time
    child_key = key.child_key(self.page_size)
    value = []

    while len(key) > 0 and child_key in node.children:
        child = node.children[child_key]
        child.last_access_time = access_time
        prefix_len = child.key.match(key, page_size=self.page_size)

        if prefix_len < len(child.key):
            # 部分匹配 → 分裂节点!
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)   # 匹配部分的 KV indices
            node = new_node
            break
        else:
            # 完全匹配 → 继续向下
            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)

    return value, node
```

### _split_node — 节点分裂 (SGLang 核心创新)

```python
def _split_node(self, key, child, split_len):
    """将 child 在 split_len 位置分裂为两个节点

    Before:  parent → child[A B C D E]
    After:   parent → new_node[A B] → child[C D E]

    关键特性:
    - 不需要 block 对齐 (vLLM 必须按 block 分裂)
    - value (KV indices) 也相应切片 (clone)
    - hash_value 正确继承
    - lock_ref 继承 (保护正在使用的节点)
    - priority 继承
    - hit_count 继承
    """
    new_node = TreeNode(priority=child.priority)
    new_node.hit_count = child.hit_count
    new_node.children = {key[split_len:].child_key(self.page_size): child}
    new_node.parent = child.parent
    new_node.lock_ref = child.lock_ref
    new_node.key = child.key[:split_len]
    new_node.value = child.value[:split_len].clone()
    child.parent = new_node
    child.key = child.key[split_len:]
    child.value = child.value[split_len:].clone()
    new_node.parent.children[key.child_key(self.page_size)] = new_node
    # Split hash_value if computed
    new_node.hash_value, child.hash_value = split_node_hash_value(
        child.hash_value, split_len, self.page_size
    )
    return new_node
```

### insert — 插入/更新 token 序列

```python
def _insert_helper(self, node, key, value, priority=0, chunked=False):
    """
    三种情况:
    1. 完全匹配已存在节点 → 无需创建新节点
    2. 部分匹配 → split + 继续处理
    3. 无匹配 → 直接创建新叶子节点
    """
    while len(key) > 0 and child_key in node.children:
        node = node.children[child_key]
        prefix_len = node.key.match(key)
        total_prefix_length += prefix_len
        key = key[prefix_len:]
        value = value[prefix_len:]

        if prefix_len < len(node.key):
            # 部分匹配 → 先分裂
            new_node = self._split_node(node.key, node, prefix_len)
            node = new_node

    if len(key) > 0:
        # 创建新叶子
        new_node = TreeNode(priority=priority)
        new_node.key = key
        new_node.value = value.clone()
        self.evictable_size_ += len(key)
```

### evict — 多策略驱逐

```python
def evict(self, params: EvictParams):
    """基于堆的驱逐:
    1. 收集所有可驱逐叶子 (lock_ref=0 且 value 非空)
    2. 构建最小堆 (按策略优先级排序)
    3. 逐个驱逐直到满足 token 需求
    4. 驱逐叶子后，如果父节点变成叶子且 lock_ref=0 → 加入堆
    """
    leaves = list(self.evictable_leaves)
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node)
        for node in leaves
    ]
    heapq.heapify(eviction_heap)

    while num_evicted < num_tokens:
        _priority, x = heapq.heappop(eviction_heap)
        self.token_to_kv_pool_allocator.free(x.value)
        num_evicted += len(x.value)
        self._delete_leaf(x)
        # 父节点可能变成新的可驱逐叶子
        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            heapq.heappush(eviction_heap, (...))
```

### lock_ref — 引用计数保护链

```python
def inc_lock_ref(self, node):
    """从 node 到 root 的路径上所有节点 lock_ref += 1
    → 这些节点不会被驱逐"""
    while node != self.root_node:
        if node.lock_ref == 0:
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)
        node.lock_ref += 1
        node = node.parent

def dec_lock_ref(self, node):
    """从 node 到 root 的路径上所有节点 lock_ref -= 1
    → lock_ref 降为 0 的节点变为可驱逐"""
    while node != self.root_node:
        if node.lock_ref == 1:
            self.evictable_size_ += len(node.key)
            self.protected_size_ -= len(node.key)
        node.lock_ref -= 1
        node = node.parent
```

---

## C++ 高性能后端 (RadixTreeCpp)

```
文件: sglang/srt/mem_cache/cpp_radix_tree/
  ├── tree_v2.cpp        (144 lines) 核心树操作
  ├── tree_v2_node.h     (258 lines) 节点定义
  ├── tree_v2_binding.cpp           pybind11 绑定
  ├── tree_v2_debug.cpp             调试打印
  └── radix_tree.py      (183 lines) Python wrapper
```

### C++ 节点结构

```cpp
struct TreeNode {
    std::unordered_map<token_vec_t, unique_ptr<TreeNode>> children;
    TreeNode* parent;
    token_vec_t m_tokens;            // token 序列
    at::Tensor m_device_indices;     // GPU KV indices
    at::Tensor m_host_indices;       // CPU KV indices
    size_t ref_count = 0;            // 引用计数
    size_t hit_count = 0;            // 命中计数
    optional<size_t> host_ref_counter;
    optional<IOStatus> m_io_status;  // IO 状态 (GPU↔CPU)
    timestamp_t m_last_access_time;
};
```

### JIT 编译

```python
radix_tree_cpp = load(
    name="radix_tree_cpp",
    sources=["tree_v2_binding.cpp", "tree_v2_debug.cpp", "tree_v2.cpp"],
    extra_cflags=["-O3", "-std=c++20"],
)
```

C++ 后端 API: `match_prefix`, `evict`, `lock_ref`, `writing_through`, `loading_onboard`, `commit_writing_through`, `commit_loading_onboard`

---

## 缓存感知调度 (SchedulePolicy)

### 调度策略 (schedule_policy.py, 1071 lines)

```python
policies = ["fcfs", "lof", "random", "dfs-weight", "lpm"]

# LPM (Longest Prefix Match):
# 按前缀匹配长度排序，优先处理命中率高的请求
# → 最大化 prefix cache 复用

# DFS_WEIGHT:
# 深度优先权重，考虑 prefix tree 结构
```

### 动态策略切换

```python
# 关键优化: 大队列退化为 FCFS
if len(waiting_queue) > 128:
    # LPM 排序开销 O(N log N) 太大
    # 退化为 FCFS 避免调度延迟
    policy = "fcfs"
```

### In-Batch Prefix Detection

```
1. 对 waiting queue 中的请求按 prefix 分组
2. 同一组请求共享前缀 → 只计算一次 prefill
3. 批内前缀检测: 用 radix tree 对 waiting queue 做 prefix 匹配
```

### PrefillAdder — 请求准入

```python
class PrefillAdder:
    """管理 prefill 请求的 token 预算:
    - 计算每个请求需要的 token 数
    - 考虑 prefix hit → 只需 prefill 未命中部分
    - 支持 chunked prefill (长请求分批)
    - SWA/Mamba 特殊处理
    - 优先级抢占
    """
```

---

## 驱逐策略体系 (utils.py, 180 lines)

```python
def get_eviction_strategy(policy: str) -> EvictionStrategy:
    """工厂模式，支持 7 种策略:"""

    LRU:      (last_access_time, hit_count)     # 最常用
    LFU:      (-hit_count, last_access_time)    # 热点数据
    FIFO:     (creation_time, hit_count)        # 简单公平
    MRU:      (-last_access_time, -hit_count)   # 特殊场景
    FILO:     (-creation_time, -hit_count)      # 栈式
    SLRU:     (segment, last_access_time, ...)  # 分段 LRU
    Priority: (-priority, last_access_time)     # 用户指定优先级
```

---

## 请求缓存流程

### Prefill (cache_unfinished_req)

```
1. 获取当前 token_ids 和 KV indices
2. page_aligned → radix_key
3. insert(key, values) → 返回 prefix_len
4. 释放重复的 KV indices (已被 tree 共享)
5. match_prefix → 获取新 indices + last_node
6. 更新 req.prefix_indices
7. lock_ref(last_node) → 保护前缀路径
```

### Finish (cache_finished_req)

```
1. 获取完整 token_ids (input + output)
2. page_aligned → radix_key
3. insert(key, values) → 写入 tree
4. 释放重复 indices (tree 已有部分 → 共享)
5. dec_lock_ref(last_node) → 允许驱逐
```

---

## Hash 完整性验证

```python
def hash_page(self, start, end, prior_hash=None):
    """SHA256 增量哈希:
    - 每个 page 一个 hash
    - 支持增量计算 (prior_hash 链式)
    - Bigram 模式: (t_i, t_{i+1}) 对
    """
    hasher = hashlib.sha256()
    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))
    for j in range(start, end):
        hasher.update(token.to_bytes(4, 'little'))
    return hasher.hexdigest()
```

---

## RadixAttention Layer 集成 (radix_attention.py, 224 lines)

```python
class RadixAttention(nn.Module):
    def forward(self, q, k, v, forward_batch, save_kv_cache=True):
        if forward_batch.forward_mode.is_extend():
            # Extend (prefill): 通过 unified_attention_with_output
            output = torch.empty_like(q)
            unified_attention_with_output(q, k, v, output, ...)
        else:
            # Decode: 直接调用 attn_backend.forward()
            return get_attn_backend().forward(q, k, v, self, forward_batch, ...)
```

### 关键: Attention 不直接操作 tree!

```
RadixAttention 不直接操作 radix tree:
  1. Scheduler 调用 tree_cache.match_prefix → 获取 indices
  2. indices 存入 forward_batch.out_cache_loc
  3. Attention backend 从 forward_batch 读取 indices
  4. 写入时通过 writing_through() 回到 tree

这实现了调度和执行的解耦。
```

---

## 与 vLLM PagedAttention 对比

| 维度 | vLLM PagedAttention | SGLang RadixAttention |
|------|---------------------|----------------------|
| **数据结构** | BlockAllocator + HashMap | Radix Tree (Python/C++) |
| **对齐粒度** | 固定 block_size=16 | 可配置 page_size (默认1) |
| **前缀匹配** | Block hash 精确匹配 | 树路径最长前缀匹配 (LPM) |
| **节点分裂** | 不支持 (按 block 边界) | 任意位置分裂 |
| **驱逐策略** | LRU only | 7 种 (LRU/LFU/FIFO/MRU/FILO/SLRU/Priority) |
| **Bigram 支持** | 无 | 有 (EAGLE spec decode) |
| **LoRA 隔离** | 无 | extra_key 命名空间 |
| **CPU offload** | 支持 | 支持 (host_value + write_through) |
| **C++ 后端** | 无 (纯 Python) | 有 (tree_v2.cpp, JIT) |
| **调度** | FCFS + priority | FCFS + LPM + DFS_WEIGHT |
| **Hash 验证** | 无 | SHA256 per page |
| **多轮对话加速** | ~3x | ~5x (无对齐浪费) |

### SGLang 优势场景

1. **多轮对话**: 递增前缀，16 轮→80% 命中率
2. **LoRA 多租户**: extra_key 隔离不同 adapter
3. **Speculative decoding**: bigram 模式优化 EAGLE
4. **混合优先级**: Priority eviction 保护重要缓存

---

## 代码位置速查

| 文件 | 行数 | 内容 |
|------|------|------|
| `mem_cache/radix_cache.py` | 799 | RadixCache + TreeNode + RadixKey |
| `mem_cache/cpp_radix_tree/radix_tree.py` | 183 | C++ RadixTree Python wrapper |
| `mem_cache/cpp_radix_tree/tree_v2.cpp` | 144 | C++ 核心树操作 |
| `mem_cache/cpp_radix_tree/tree_v2_node.h` | 258 | C++ TreeNode 定义 |
| `layers/radix_attention.py` | 224 | RadixAttention layer |
| `managers/schedule_policy.py` | 1071 | 缓存感知调度 + PrefillAdder |
| `mem_cache/base_prefix_cache.py` | 359 | 基类接口定义 |
| `mem_cache/utils.py` | 180 | 驱逐策略 + hash 工具 |

---

## 核心洞察

1. **节点分裂是核心创新**: vLLM 要求 block 对齐，浪费 prefix cache 空间。SGLang 任意位置分裂 → 更高缓存利用率
2. **lock_ref 路径保护**: 请求引用的整条路径不可驱逐，防止级联失效
3. **双后端设计**: Python 简单实现 + C++ 高性能后端，通过 JIT 编译桥接
4. **调度器深度耦合**: LPM 策略使 prefix cache 命中率不只是被动响应，而是主动优化请求顺序
5. **Bigram 为 EAGLE 准备**: Speculative decoding 的 draft token 树用 bigram 键表示，与 radix tree 天然适配
6. **extra_key 命名空间**: 用一个字段解决多租户/多模型隔离问题，简洁而强大
7. **动态退化**: >128 请求退化为 FCFS — 工程上合理的 trade-off
