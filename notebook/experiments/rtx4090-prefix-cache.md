# RTX 4090 Prefix Caching Benchmark 实测

> GPU: NVIDIA GeForce RTX 4090, 24GB
> 日期: 2026-06-05

## 1. Prefill Cost: Cached vs Uncached Prefix

| Prefix | Suffix | Full (ms) | Cached (ms) | Saved% | KV Saved (MB) |
|--------|--------|-----------|-------------|--------|---------------|
| 64     | 64     | 0.036     | 0.022       | 39.9%  | 33.6          |
| 128    | 64     | 0.057     | 0.022       | 62.2%  | 67.1          |
| 256    | 64     | 0.076     | 0.021       | 71.9%  | 134.2         |
| 512    | 64     | 0.134     | 0.021       | 84.0%  | 268.4         |
| 1024   | 64     | 0.257     | 0.021       | **91.7%** | 536.9      |
| 1024   | 256    | 0.279     | 0.057       | **79.6%** | 536.9      |

**Key Finding**: Prefix=1024, Suffix=64 时节省 **91.7%** prefill 时间。前缀越长、后缀越短，收益越大。

## 2. Batch Sharing with Common Prefix

| Batch | No Cache (ms) | Cache (ms) | Speedup | Mem Saved (MB) |
|-------|---------------|------------|---------|----------------|
| 1     | 0.091         | 0.093      | 0.99x   | 0.0            |
| 4     | 0.312         | 0.161      | 1.93x   | 402.7          |
| 8     | 0.604         | 0.265      | 2.28x   | 939.5          |
| 16    | 1.216         | 0.508      | 2.39x   | 2013.3         |
| 32    | 2.390         | 0.855      | **2.79x** | 4160.7       |

**Key Finding**: Batch=32 时 prefix cache 带来 **2.79x** 加速 + 4.1 GB KV 内存节省。Batch=1 时无收益 (overhead 抵消)。

## 3. Multi-turn Conversation Savings

5 轮对话累积缓存比例:
- Turn 2: 33.3% cached
- Turn 3: 50.0% cached
- Turn 5: **66.7% cached** (2/3 的 KV cache 可复用)

Turn=1024 时 5 轮后累积节省 **5368.7 MB** KV cache。

## 4. RadixAttention Tree Sharing

4 个请求共享 256 token 系统提示:
- 无共享: 1600 tokens
- RadixAttention: 832 tokens (prefix 计算一次)
- 节省: **48.0%** 计算 + **402.7 MB** KV cache

## 5. RLHF Rollout (GRPO n=8)

| Response Len | No Share (tok) | Share (tok) | Save% |
|-------------|----------------|-------------|-------|
| 128         | 5120           | 1536        | **70.0%** |
| 256         | 6144           | 2560        | 58.3% |
| 512         | 8192           | 4608        | 43.8% |
| 1024        | 12288          | 8704        | 29.2% |

**Key Finding**: GRPO n=8 短 response (128 tok) 时节省 **70%** 计算, 验证了 verl 的 PrefixGrouper 设计。

## 6. Summary

- **Prefix caching 最有效场景**: 长前缀 + 短后缀 + 多请求共享
- **Multi-turn**: 5 轮后 66.7% 可缓存, 对话越长收益越大
- **RLHF GRPO**: 短 response 时 70% 计算节省, 与 verl 源码分析一致
- **RadixAttention**: 树形共享比简单前缀共享更灵活 (无 block 对齐限制)
