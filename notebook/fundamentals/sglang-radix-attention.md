# SGLang RadixAttention 深度解析

> 目标：理解 SGLang 的 RadixAttention 前缀缓存机制，与 vLLM Hash-based 方案对比

## 1. 核心架构

```
SGLang 调度器 (CPU 单线程事件循环)
    │
    ├── PrefillAdder (准入控制)
    │     └── rem_total_tokens, rem_input_tokens
    │
    ├── RadixCache (基数树)
    │     ├── match_prefix() → 查找最长匹配前缀
    │     ├── insert() → 插入完成请求的 KV 缓存
    │     └── evict() → LRU/LFU/FIFO 驱逐
    │
    ├── req_to_token_pool [max_running, max_seq_len]
    │     └── 每个请求的 KV 索引映射
    │
    └── token_to_kv_pool_allocator
          └── GPU KV 缓存槽位池
```

## 2. Radix Tree 数据结构

### 与 Trie 的区别

Radix Tree 是 Trie 的空间优化变体，边由**可变长度 token 序列**标记（非单个 token）。

```
Trie (每个 token 一条边):
  root → s → y → s → t → e → m → [prompt_A]

Radix Tree (可变长度边):
  root → [system prompt] → [prompt_A]
                        → [prompt_B]
```

### 核心数据结构

```python
class RadixKey:
    token_ids: list[int]       # token 序列
    extra_key: Optional[str]   # LoRA/cache 隔离命名空间
    is_bigram: bool            # EAGLE 推测解码支持

class TreeNode:
    children: defaultdict       # 子节点
    parent: TreeNode
    key: RadixKey               # 边上的 token 序列
    value: torch.Tensor         # KV 池索引 (GPU slot IDs)
    lock_ref: int               # 保护计数器 (正在使用的引用)
    last_access_time: float     # LRU 驱逐用
    hit_count: int              # LFU 驱逐用
    host_value: Tensor          # CPU 卸载 (分层缓存)
    hash_value: bytes           # SHA256 链式哈希 (P/D 事件)
```

### 节点分裂机制 (_split_node)

这是 RadixAttention 的核心操作——**动态拆分现有缓存条目**以提取共享前缀：

```
初始状态:
  root → [system prompt + question A + answer A]

请求 B 到达: [system prompt + question B]

Step 1: match_prefix 找到 [system prompt] 是公共前缀
Step 2: _split_node 在边界处拆分:
  root → [system prompt] → [question A + answer A]
                          → [question B]

后续请求 C: [system prompt + question A + followup]

Step 3: match_prefix 匹配 [system prompt + question A]
Step 4: _split_node 再次拆分:
  root → [system prompt] → [question A] → [answer A]
                                                    → [followup]
                          → [question B]
```

```python
def _split_node(self, child, split_len):
    new_node = TreeNode()
    new_node.key = child.key[:split_len]
    new_node.value = child.value[:split_len].clone()

    child.key = child.key[split_len:]
    child.value = child.value[split_len:].clone()

    # 新节点成为父节点的子节点
    # 原子节点成为新节点的子节点
    new_node.children = {child.key: child}
    child.parent = new_node
```

## 3. 三大操作路径

### match_prefix: 查找最长匹配

```python
def match_prefix(self, token_ids, extra_key=None):
    # 从根节点开始遍历
    # 在每个节点检查子节点的 key 是否匹配
    # 如果匹配在子节点 key 的"中间"结束 → _split_node
    # 返回匹配的 KV 池索引 + 匹配长度

    # 同时: lock_ref += 1 (保护节点不被驱逐)
    # evictable_size → protected_size
```

### insert: 插入完成请求

```python
def insert(self, token_ids, kv_indices):
    # _insert_helper 遍历树，匹配现有前缀
    # 在分叉点: 现有节点被拆分
    # 不匹配的后缀创建新的叶节点
    # 重复的 KV 索引被释放 (已存在于树中的)
```

### evict: 驱逐策略

```python
def evict(self, num_tokens):
    # 基于 evictable_leaves 集合构建堆
    # 默认 LRU (也支持 LFU, FIFO, MRU, FILO, Priority, SLRU)
    # 叶节点从树中弹出，KV 池槽位释放
    # 递归: 如果父节点无子节点且 lock_ref==0 → 也可驱逐
```

## 4. vLLM Hash-based vs SGLang Radix Tree

| 维度 | vLLM (Hash-based) | SGLang (Radix Tree) |
|------|-------------------|---------------------|
| 数据结构 | BlockHashToBlockMap (哈希表) | RadixCache (基数树) |
| 匹配粒度 | Block 级 (16 tokens/block) | Token 级 (可变长度边) |
| 部分匹配 | 完全匹配或无匹配 | **可拆分现有节点** 精确匹配 |
| 冲突处理 | 哈希碰撞用链表解决 | 无冲突 (精确 token 比较) |
| 查找复杂度 | O(1) 每 block (hash lookup) | O(匹配深度) (树遍历) |
| 结构修改 | 否 (block 要么匹配要么不匹配) | **是** (节点拆分重构树) |
| 调度感知 | 调度器独立于前缀缓存 | **缓存感知调度** (LPM/DFS-weight) |
| 驱逐粒度 | Block 级, ref_cnt | 叶节点级, lock_ref 遍历到根 |
| LoRA 隔离 | hash 中包含 lora_name | extra_key 字段显式隔离 |

### 关键差异: 部分前缀重用

**场景**: 请求 A = `[system + question_A + answer]`, 请求 B = `[system + question_B]`

**vLLM 行为**:
1. A 完成 → blocks 缓存 (hash 链)
2. B 到达 → 逐 block hash 查找
3. `[system]` 匹配 (blocks 0-15) ✓
4. `[question_A 的第一个 block]` hash ≠ `[question_B 的第一个 block]` hash ✗
5. 结果: 只重用 `[system]` 的 blocks

**SGLang 行为**:
1. A 完成 → `[system+question_A+answer]` 存入树
2. B 到达 → match_prefix 遍历
3. 匹配 `[system]` 部分 → `_split_node` 拆分
4. 结果: 完全相同的 `[system]` 前缀 KV 被重用

看起来一样？不。关键区别在于 **block 边界对齐问题**:

**vLLM**: 如果 `system` 长度不是 block_size (16) 的整数倍，最后一个 block 包含部分 system + 部分 question_A。这个 block 的 hash 包含了 question_A 的 token，所以 B 无法命中。

**SGLang**: Radix tree 边是可变长度的，`_split_node` 在**任意 token 边界**拆分。不存在 block 对齐限制。

## 5. 缓存感知调度

### 调度策略

| 策略 | 描述 | 适用场景 |
|------|------|---------|
| LPM | 最长前缀匹配优先 | 前缀密集型工作负载 |
| DFS-weight | 基于树深度的 DFS 权重 | 结构化请求 |
| FCFS | 先到先服务 | 通用 (队列 >128 自动降级) |
| LOF | 最长输出优先 | 吞吐量优先 |
| Random | 随机选择 | 基准对比 |

### 批内前缀缓存

```
等待队列: [req_A: "system+Q1", req_B: "system+Q2", req_C: "system+Q3"]

1. 检测到 3 个请求共享 "system" 前缀 (< 32 tokens)
2. 调度 req_A (完全 prefill)
3. req_B, req_C 临时降低优先级
4. req_A 完成 → "system" KV 缓存到 Radix Tree
5. req_B, req_C 被调度 → 直接命中 "system" 缓存
```

这是 vLLM **没有**的优化。

## 6. 事件循环架构

```
事件循环 (重叠模式):
    ┌─────────────────────────────────────────────┐
    │ t=0  receive_batch() ← 从输入通道获取请求     │
    │ t=1  schedule() ← 缓存感知调度              │
    │ t=2  forward_batch_generation() ← GPU 计算   │
    │         │                                    │
    │         │ (GPU 计算)                         │
    │         │                                    │
    │ t=3  process_batch_result() ← CPU 处理结果   │
    │ t=4  → 下一次迭代的 receive_batch()           │
    └─────────────────────────────────────────────┘

    CUDA Stream 层次:
    ├── schedule_stream: CPU 驱动的操作
    ├── forward_stream: GPU 计算
    └── copy_stream: CPU-GPU 数据传输

    FutureMap: 存储 GPU 结果，下一次迭代的 CPU 处理读取
```

## 7. 性能数据

| 工作负载 | 加速倍数 | 原因 |
|---------|---------|------|
| 多轮对话 | ~5x | 对话历史自动缓存复用 |
| ReAct Agent | ~5x | 多轮推理路径缓存 |
| Tree of Thought | 显著 | 分支前缀共享 |
| Self-Consistency | 显著 | 相同 prompt 多次采样 |
| Few-shot Learning | 显著 | 相同示例跨请求共享 |
| 通用推理 | 最高 6.4x | LMSYS 基准测试 |

## 8. 关键洞察

1. **节点分裂是核心**: 使 SGLang 能从 vLLM 无法表示的序列中回收前缀匹配
2. **无 block 对齐限制**: Radix tree 边是可变长度，精确 token 级匹配
3. **缓存感知调度**: LPM 策略主动优先缓存命中率高的请求，vLLM 无此机制
4. **批内前缀检测**: 等待队列中识别共同前缀，串行化调度以最大化缓存收益
5. **lock_ref 保护**: 从叶节点到根节点的引用计数链，保证活跃请求的 KV 不被驱逐
6. **7 种驱逐策略**: 默认 LRU，可根据工作负载选择 LFU/FIFO/Priority 等
7. **分层缓存**: GPU → CPU → SSD 三级，host_value/host_ref_counter 追踪卸载状态

## 9. 源码级补充: C++ Radix Tree + HiCache + match_prefix算法详解

> 2026-06-15 源码深挖: radix_cache.py + cpp_radix_tree/tree_v2 + schedule_policy.py

### 9.1 _match_prefix_helper 算法精确步骤

```python
# radix_cache.py:619-643
def _match_prefix_helper(self, node, key, value):
    """精确匹配算法:
    1. 从root开始
    2. 计算 child_key = key.child_key(page_size) → 前 page_size 个tokens
    3. 如果 child_key 在 node.children 中:
       - child = node.children[child_key]
       - prefix_len = child.key.match(key, page_size)  → token级逐个比较!
       - 如果 prefix_len < len(child.key):
         → PARTIAL MATCH: _split_node(child, prefix_len)
         → value.append(new_node.value); node = new_node; break
       - 否则 (prefix_len == len(child.key)):
         → FULL NODE MATCH: value.append(child.value); node = child; key = key[prefix_len:]
    4. 如果 child_key 不在 children 中: break (无匹配)
    5. 返回: concat(value) → KV池索引 + last_node

    关键: match_prefix同时更新所有访问节点的 last_access_time → 驱逐策略数据!
    """
```

### 9.2 _split_node 源码级详解

```python
# radix_cache.py:645-665
def _split_node(self, child, split_len):
    """节点分裂 → Radix Tree核心操作(Patricia Trie标志!)

    1. 创建 new_node:
       - new_node.key = child.key[:split_len]  → 共享前缀部分
       - new_node.value = child.value[:split_len].clone()  → 对应KV索引(克隆!)
       - new_node继承: lock_ref, hit_count, priority → 父节点获得保护

    2. 修改 child (原节点保留不匹配后缀):
       - child.key = child.key[split_len:]
       - child.value = child.value[split_len:].clone()
       - 注意: value是clone不是slice! → KV索引独立管理

    3. 树结构更新:
       - new_node替换child在parent.children中
       - child成为new_node的子节点
       - child.parent = new_node

    4. 内存管理:
       - 被free的KV indices: token_to_kv_pool_allocator.free()
       - 已存在于树中的重复indices → 被释放(避免浪费!)

    → 关键: 这是Patricia Trie的标志性操作 → Trie中每边=1个token → 无需split!
    → Radix Tree中每边=可变长度 → 分叉点需要split → 这是空间效率的核心!
    """
```

### 9.3 C++ Radix Tree 实现 (tree_v2)

```cpp
// cpp_radix_tree/tree_v2_node.h
struct TreeNode {
    token_vec_t m_tokens;           // vector<int32> → token序列
    at::Tensor m_device_indices;    // GPU KV indices
    at::Tensor m_host_indices;      // CPU KV indices (HiCache)
    unordered_map<token_vec_t, unique_ptr<TreeNode>> m_children;
    TreeNode* m_parent;
    int ref_count, hit_count;
    float m_last_access_time;

    // HiCache IO状态:
    bool m_io_locked;               // 正在进行IO操作 → 保护
    IOStatus m_io_status;           // IDLE/WRITING_THROUGH/LOADING_BACK
    IOTicket m_io_ticket;           // async IO ticket → 等待完成
};

// tree_v2_impl.h:83-113 tree_walk
auto prefix_length = align(node->diff_key(key, page_size) + page_size);
// diff_key使用 std::ranges::mismatch → 线性扫描 → 非单token比较!
// → 确认: 这是真正的radix tree(Patricia Trie) → 不是simple trie!
```

### 9.4 HiCache 分层缓存机制

```
GPU → CPU → 三级缓存:

1. Device tier (GPU):
   - m_device_indices → GPU KV池slot IDs
   - 最快访问 → 解码必须

2. Host tier (CPU pinned memory):
   - m_host_indices → CPU KV indices
   - IO操作:
     - write_through: GPU→CPU异步写 → 不阻塞计算 → IOHandle
     - load_back: CPU→GPU异步读 → 需等待 → IOTicket→blocking!
   - m_io_status: IDLE→WRITING_THROUGH→LOADING_BACK→IDLE cycle
   - m_io_locked: IO进行中 → eviction被阻止 → 保护数据一致性

3. SSD tier (可选):
   - 通过外部存储系统 → SGLang核心不直接管理

关键: HiCache是C++版本独有的 → Python版本只有device tier
→ 生产环境应该用C++ radix tree → HiCache offload → 减少GPU内存压力!
```

### 9.5 Cache-aware调度详解

```python
# schedule_policy.py:85-126 match_prefix_for_req
def match_prefix_for_req(self, req, tree_cache):
    """为请求匹配前缀缓存

    1. tree_cache.match_prefix(req.token_ids)
    2. 返回: device_indices → 已缓存的KV池slot IDs
    3. req.prefix_indices = device_indices → prefill时直接用!
    4. req.num_matched_prefix_tokens = len(device_indices)
    5. → 未匹配的suffix tokens才需要计算KV → 省(prefill)!

    → lock_ref: inc_lock_ref(last_node) → 从叶到根 → 整条路径保护!
    → 任何active request的prefix路径节点 → 不可被eviction驱逐!
    → vs vLLM: vLLM只保护单个block的ref_cnt → 不保护parent blocks → 可能路径断裂!
    """

# schedule_policy.py LPM策略
# 等待队列按 num_matched_prefix_tokens 降序排列
# → 最长前缀匹配的请求 → 优先调度 → 最省GPU计算!

# schedule_policy.py DFS-weight策略
# 按tree子树权重DFS遍历 → 同子树请求连续调度 → maximize prefix reuse within batch!

# schedule_policy.py 批内前缀缓存 (行258-288)
# waiting_queue_radix_tree → 模拟的树(不占GPU内存)
# 如果多个请求共享前缀但缓存命中率低 → 只调度第一个 → 其余降优先级
# → 第一个完成后 → 前缀缓存到实际树 → 其余请求命中率高 → 再调度!
# → 这是vLLM完全没有的优化!
```

### 9.6 RadixKey 精确设计

```python
# radix_cache.py:56-196
class RadixKey:
    token_ids: array[int]         # 可变长度token序列
    extra_key: Optional[str]      # LoRA/cache隔离命名空间
    is_bigram: bool               # EAGLE推测解码 → 2 tokens=1逻辑单位

    def child_key(self, page_size):
        """返回children dict的key → 前page_size个tokens"""
        # page_size=1 → 第1个token → 类似Trie但边是可变长度!
        # page_size=16 → 前16个tokens → 类似block但可以split任意位置!

    def match(self, other, page_size):
        """逐token比较 → 返回共享前缀长度 → 向下对齐到page_size"""
        # 关键: 比较是token级精确 → 不是hash级近似 → 无碰撞!
```

### 9.7 Eviction 算法精确步骤

```python
# radix_cache.py:534-561
def evict(self, num_tokens):
    """驱逐算法:

    1. 从 evictable_leaves 构建heap → 按eviction策略排序
    2. While num_evicted < num_tokens and heap非空:
       a. Pop最低优先级叶节点
       b. Free其KV indices → token_to_kv_pool_allocator.free()
       c. _delete_leaf(x) → 从parent.children删除 → 从evictable_leaves删除
       d. 如果parent现在无children且lock_ref==0 → push parent到heap(cascade!)
          → 叶节点驱逐 → 空父节点也驱逐 → 递归释放!

    → vs vLLM: vLLM逐block驱逐 → 不cascade(parent block可能仍有其他children引用)
    → SGLang cascade更彻底 → 释放更多内存 → 但也更激进 → 需lock_ref保护!

    → 7种策略:
      LRU(last_access_time) / LFU(hit_count,last_access_time) / FIFO(creation_time)
      / MRU(-last_access_time) / FILO(-creation_time)
      / Priority(priority,last_access_time) / SLRU(is_protected,last_access_time)
    """
```

### 9.8 Patricia Trie确认: 可变长度边 + _split_node

```
问题: SGLang用的是true radix tree (Patricia Trie) 还是simple trie?

证据1: 可变长度边
  → TreeNode.key = RadixKey → array[int] → 可变长度token序列
  → Simple trie: 每边=1 token → children dict key = 单个整数
  → Radix tree: 每边=可变长度 → children dict key = child_key(page_size) = 前 page_size 个tokens
  → 结论: 是radix tree!

证据2: _split_node 操作
  → 在现有节点内部拆分 → new_node(前缀) + child(后缀)
  → Simple trie: 每节点只有1个token → 拆分无意义 → 不需要split!
  → Radix tree: 拆分是核心操作 → Patricia Trie标志性特征!
  → 结论: 是Patricia Trie!

证据3: C++ diff_key
  → 使用 std::ranges::mismatch → 线性扫描整个节点token序列
  → 不是单token比较 → 是多token比较 → radix tree语义!
  → 结论: 是radix tree!

空间复杂度:
  Trie: O(N × L) → N个字符串 × L平均长度 → 每token一个节点 → 穱碎
  Radix Tree: O(N) → N=所有字符串总token数 → 共享前缀合并 → 空省!
  → 1000-token system prompt × 100 requests: Trie=1000节点, Radix=1节点!

时间复杂度:
  Trie lookup: O(L) → L次节点访问
  Radix lookup: O(L) → worst case相同 → 但实践中更少节点访问
  → 共享前缀只需1次比较(边label全量) → 不需要逐token遍历!

结论: SGLang RadixAttention = **True Patricia Trie** → 不是simple trie!
```

## 10. RTX 4090 影响与实战

```
RTX 4090 24GB (SGLang推理):

1. RadixAttention对RTX 4090推理有显著帮助!
   → 多轮对话: KV cache重用 → 省GPU内存 → 更多并发请求
   → Self-Consistency/GRPO rollout_n: 相同prompt×N → 共享system prompt → KV复用!

2. HiCache → CPU pinned memory offload → 减少GPU内存压力
   → 7B BF16: 5500 tokens KV cache ≈ 44MB → CPU offload → GPU省44MB!
   → INT8 KV: 22MB → 更少 → 但INT8 KV cache和HiCache不兼容(INT8不能offload到CPU BF16)
   → 解决: INT8 KV + HiCache不做offload → INT8比HiCache更省空间!

3. RTX 4090最优SGLang配置:
   INT4权重 + INT8 KV + GQA-8 + FlashInfer + RadixAttention
   → INT4推理 → 4791 tok/s → 加RadixAttention → 多轮对话5x加速!
   → vs vLLM: INT4+INT8KV+APC → block对齐浪费 → 多轮对话场景不如SGLang!

4. verl GRPO + SGLang rollout:
   → GRPO rollout_n=8 → 相同prompt → RadixAttention自动复用system prompt KV!
   → rollout_n=8 → 8×system prompt KV → 只计算1次system + 8次question → 省7×system prefill!
   → SGLang GRPO rollout 比 vLLM rollout更快 → prefix缓存优势!

5. C++ radix tree → 生产环境必须:
   → HiCache write-through → GPU→CPU异步写 → 不阻塞计算
   → load_back → CPU→GPU → 需等待 → 但比recompute快10x+!
   → RTX 4090: HiCache利用24GB CPU内存 → offload冷KV → GPU腾空间给新请求
```

## 参考资料

- SGLang 论文: arXiv 2312.07104
- LMSYS Blog: SGLang RadixAttention (2024-01-17)
- 源码: `sglang/python/sglang/srt/mem_cache/radix_cache.py` (~800 行)
- 源码: `sglang/python/sglang/srt/mem_cache/cpp_radix_tree/tree_v2_node.h` (C++ TreeNode)
- 源码: `sglang/python/sglang/srt/mem_cache/cpp_radix_tree/tree_v2_impl.h` (C++ tree_walk/split_node)
- 源码: `sglang/python/sglang/srt/mem_cache/cpp_radix_tree/tree_v2.cpp` (C++ match_prefix/evict)
- 源码: `sglang/python/sglang/srt/mem_cache/evict_policy.py` (7种eviction策略)
- 源码: `sglang/python/sglang/srt/managers/schedule_policy.py` (LPM/DFS-weight/批内前缀缓存)
- 源码: `sglang/python/sglang/srt/managers/scheduler.py` (~1600 行)
