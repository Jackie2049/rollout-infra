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

## 参考资料

- SGLang 论文: arXiv 2312.07104
- LMSYS Blog: SGLang RadixAttention (2024-01-17)
- 源码: `sglang/python/sglang/srt/mem_cache/radix_cache.py` (~700 行)
- 源码: `sglang/python/sglang/srt/managers/scheduler.py` (~1600 行)
- 源码: `sglang/python/sglang/srt/managers/schedule_policy.py`
