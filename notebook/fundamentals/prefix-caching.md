# Prefix Caching 深度对比 — vLLM / SGLang / verl

> 目标：对比不同系统的 Prefix Caching 实现策略，理解各自的适用场景

## 1. 核心思想

当多个请求共享相同的 token 前缀时，其 KV Cache 可以复用，避免重复计算。

```
请求 A: [System Prompt][User Query A]  → 计算 KV for [System Prompt] (第一次)
请求 B: [System Prompt][User Query B]  → 复用 KV for [System Prompt] (命中!)
请求 C: [System Prompt][User Query C]  → 复用 KV for [System Prompt] (命中!)
```

收益取决于：**prompt 复用率 × prompt 占比**。

## 2. vLLM V1 — Hash-based (Block 级哈希链)

### 机制

```
Block Hash Chain:
  Block 0: hash = H(ROOT_HASH, tokens[0:B])
  Block 1: hash = H(Block0_hash, tokens[B:2B])
  Block 2: hash = H(Block1_hash, tokens[2B:3B])

相同 hash = 相同内容 (至 block 边界) → 可复用
```

### 特点

| 特性 | 说明 |
|------|------|
| 检测方式 | 自动 (基于 block hash) |
| 用户标注 | 不需要 |
| 匹配粒度 | Block 大小 (默认 16 tokens) |
| 数据结构 | Hash Map: hash → block_id |
| 复杂度 | O(1) 查找 |
| 局限 | 只能匹配到 block 边界，不满 block 不缓存 |

### Block 生命周期

```
1. 新请求到达 → 计算 block hash
2. Hash 命中 → 增加引用计数，不分配新 block
3. Hash 未命中 → 分配新 block，计算 KV，存入 cache
4. 请求完成 → 减少引用计数
5. 引用计数 = 0 → block 变为 free，可被 evict
```

## 3. SGLang — Trie-based (RadixAttention)

### Radix Tree 结构

```
                    Root
                   /
            [System Prompt tokens]
           /          \
   [Query A tokens]  [Query B tokens]
   Block: 0-15       Block: 16-20
```

### 机制

```
insert(tokens, block_ids):
  1. 从根节点开始，沿 token 序列匹配
  2. 匹配到分叉点 → split 节点
  3. 未匹配部分 → 创建新子节点
  4. 每个节点存储对应的 KV Cache block

match(prefix):
  1. 从根开始匹配 token
  2. 返回最长匹配路径
  3. 返回匹配到的 block 列表
```

### 特点

| 特性 | 说明 |
|------|------|
| 检测方式 | 自动 (Radix tree 匹配) |
| 匹配粒度 | Token 级别 (可变长) |
| 数据结构 | Radix Tree (前缀树变体) |
| 复杂度 | O(L) 查找 (L = 前缀长度) |
| 优势 | 支持任意长度前缀匹配 |

### vLLM Hash vs SGLang Radix

| 维度 | vLLM Hash | SGLang Radix |
|------|-----------|-------------|
| 匹配粒度 | 固定 block (16t) | 任意 token 位置 |
| 数据结构 | HashMap | Radix Tree |
| 查找复杂度 | O(1) | O(L) |
| 内存开销 | 低 (只存 hash) | 中 (存 tree nodes) |
| 不满 block | 不缓存 | 仍可部分匹配 |
| 适用 | 大批量、固定 prompt | 多轮对话、RAG |

## 4. verl — Group-based (三级前缀缓存)

### 三级缓存架构

```
Level 1: 系统级缓存 (跨 Worker)
  - 相同 prompt 在不同 DP worker 间共享
  - 通过 broadcast 或 shared memory 同步

Level 2: 进程级缓存 (Worker 内)
  - 同一 Worker 处理的相同 prompt 的 KV Cache
  - 在 PPO step 间保持

Level 3: 请求级缓存 (Batch 内)
  - 同一 batch 中相同 prompt 的请求
  - PrefixGrouper 负责分组
```

### PrefixGrouper

```python
# verl 的分组策略
class PrefixGrouper:
    def group_by_prefix(self, requests):
        # 1. 按 prompt hash 分组
        groups = defaultdict(list)
        for req in requests:
            groups[hash(req.prompt)].append(req)

        # 2. 每组第一个请求计算完整 KV
        # 3. 后续请求复用 prompt KV，只计算 response 部分
        return groups
```

### 特点

| 特性 | 说明 |
|------|------|
| 检测方式 | 显式分组 (知道哪些请求共享 prompt) |
| 适用场景 | RL 训练 (GRPO/PPO) |
| 精确度 | 最高 (分组信息已知) |
| 开销 | 最低 (无需 hash/tree 计算) |
| 局限 | 仅适用于已知分组信息的场景 |

## 5. 收益量化

### 模拟实验数据 (prefix_caching_sim.py)

**RL 训练场景 (GRPO: 20 prompts × 8 responses):**
- Prompt: 512 tokens, Response: 256 tokens
- No Cache: 7680 blocks
- With Cache: 3200 blocks
- **节省 58.3%**

**Prompt 复用率影响 (200 请求, prompt=256t):**

| Unique Prompts | 复用率 | 节省 |
|---------------|--------|------|
| 1 | 200x | 66.3% |
| 10 | 20x | 63.3% |
| 50 | 4x | 50.0% |
| 200 | 1x | 25.0% |

**Prompt 长度影响 (50 请求, 5 unique):**

| Prompt 长度 | 节省 |
|------------|------|
| 32 tokens | 18.0% |
| 256 tokens | 60.0% |
| 1024 tokens | 80.0% |
| 2048 tokens | 84.7% |

### 收益公式

```
savings ≈ prompt_reuse_ratio × (prompt_len / total_len)

其中:
- prompt_reuse_ratio = avg_requests_per_prompt
- prompt_len = 共享前缀 token 数
- total_len = prompt + input 总 token 数
```

## 6. 选择建议

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| RL 训练 | Group-based (verl) | 分组信息已知，最精确 |
| 通用推理 | Hash-based (vLLM) | 自动检测，无需标注 |
| 多轮对话 | Trie-based (SGLang) | 变长匹配更灵活 |
| RAG / 长文档 | Hash or Trie | prompt 长度高，收益大 |
| API 服务 | Hash-based | 用户独立，但可能共享 template |

## 实验验证

- `tools/prefix_caching_sim.py` — Prefix Caching 模拟器（4 个实验）
  - 实验 1: 四种策略对比 (60% 节省 for 10 unique prompts, 100 requests)
  - 实验 2: Prompt 复用率影响 (25%-66% 节省)
  - 实验 3: RL 训练场景 (58.3% 节省 for GRPO)
  - 实验 4: Prompt 长度影响 (18%-85% 节省)

## 参考资料

- [vLLM V1 Prefix Caching](https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html)
- [SGLang RadixAttention Paper](https://arxiv.org/abs/2312.07104)
- [verl PrefixGrouper](https://github.com/verl-project/verl)
- [PagedAttention Paper (SOSP 2023)](https://arxiv.org/abs/2309.06180)
- `notebook/projects/vllm-prefix-caching-v1.md` — vLLM V1 源码分析
- `notebook/projects/verl-prefix-grouper.md` — verl 分组策略
