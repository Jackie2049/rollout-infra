# SGLang Inference Engine Source Code Analysis

> 2026-06-07 | SGLang v0.5+ 源码阅读: Scheduler 4068行(7 mixin组合!), RadixCache 802行, UnifiedRadixCache 2846行

## 核心架构

```
SGLang推理引擎架构 (2026年版本):

┌──────────────────────────────────────────────────────────────────────┐
│ Scheduler (4068行, 7 mixin组合!)                                     │
│                                                                      │
│ class Scheduler(                                                     │
│   SchedulerDisaggregationDecodeMixin,  # 解耦推理: prefill/decode分离│
│   SchedulerDisaggregationPrefillMixin, # 解耦prefill               │
│   SchedulerMultiplexMixin,             # 多LoRA服务                │
│   SchedulerPPMixin,                    # Pipeline Parallelism      │
│   SchedulerDllmMixin,                  # DLLM(分布式LLM?)          │
│   SchedulerMlxOverlapMixin,            # MLX(Apple Silicon) overlap│
│ )                                                                     │
│                                                                      │
│ 关键数据结构:                                                        │
│   self.running_batch: ScheduleBatch  # 当前decode batch           │
│   self.last_batch: ScheduleBatch      # 上一批prefill batch       │
│   self.tree_cache: RadixCache         # KV prefix cache (RadixTree)│
│   self.chunked_req: Optional[Req]     # chunked prefill请求       │
│   self.waiting_queue: Deque[Req]       # 等待队列(FCFS)           │
│                                                                      │
│ 调度流程:                                                            │
│   get_next_batch_to_run() → merge prefill into running → schedule  │
│   → process_batch_result() → update tree_cache → repeat           │
│                                                                      │
│ Mixin设计: 每个mixin扩展Scheduler → 灵活但复杂                     │
│   → Disaggregation: prefill GPU ↔ decode GPU分离                │
│   → Multiplex: 多LoRA → batch内混合不同LoRA请求               │
│   → PP: Pipeline Parallel → 多层分布不同GPU                    │
│   → Overlap: compute-communication overlap (MLX/CUDA)            │
└──────────────────────────────────────────────────────────────────────┘
```

## RadixCache — KV Prefix Cache核心

```
RadixCache (802行) + UnifiedRadixCache (2846行):

核心数据结构: Radix Tree
  → 每个节点: RadixKey(token_ids) → KV cache blocks
  → 支持prefix matching → 重用历史KV → 减少重复计算

RadixKey设计:
  → token_ids: array[int] → 原始token序列
  → is_bigram: True → 大gram模式 → N+1 tokens for N bigrams
  → extra_key: lora_id, cache_salt → LoRA/加密区分

RadixTree操作:
  → match_prefix(): 查找最长匹配prefix → 返回matched tokens + cache blocks
  → insert(): 插入新token序列 → 分裂已有节点 → 创建新节点
  → evict(): 驱逐最少使用的节点 → LRU策略 → 释放KV cache blocks

与vLLM对比:
  → vLLM: BlockHashToBlockMap(1:N) → hash-based → 简单但不支持部分匹配
  → SGLang: RadixTree → 支持prefix matching → 更高效prefix reuse
  → → SGLang的RadixAttention在重复prompt时效率更高!

UnifiedRadixCache (2846行) — 更大更复杂:
  → 支持: HiSparse, hierarchical cache, disaggregation, distributed
  → 多层缓存: GPU(HBM) → CPU(DRAM) → disk → 分层驱逐
  → 事件系统: KVCacheEventMixin → 监控缓存命中率
  → EvictPolicy: LRU/LFU/custom → 可配置驱逐策略
```

## Scheduler调度策略

```
SGLang调度策略:

1. FCFS (默认):
   → waiting_queue = Deque[Req] → 先到先服务
   → 简单但可能不公平(长请求阻塞短请求)

2. Priority Scheduling:
   → enable_priority_scheduling → 请求优先级排序
   → priority_scheduling_preemption_threshold → 抢占阈值

3. Chunked Prefill:
   → 长prompt → 分chunk处理 → 不阻塞decode batch
   → chunked_req → 分块prefill → 每chunk一个forward pass
   → 与vLLM chunked prefill类似 → 但SGLang有自己的实现

4. Prefill-Decode Merge:
   → get_next_batch_to_run() → merge last prefill batch into running
   → running_batch.merge_batch(new_prefill) → 统一batch
   → 过滤已完成请求 → filter_batch() → 更新running_batch

5. Overlap Schedule:
   → enable_overlap → compute-communication overlap
   → 同时进行: forward + tokenization + cache update
   → CUDA stream并行 → 多stream overlap

关键差异 vs vLLM:
  → vLLM V1: FCFS + unified budget + single scheduler loop
  → SGLang: FCFS + merge-based + overlap + more mixins
  → SGLang更灵活(mixin)但更复杂(4068行 vs vLLM ~2000行)
```

## Disaggregation Mode

```
SGLang的prefill-decode disaggregation:

SchedulerDisaggregationPrefillMixin:
  → PrefillBootstrapQueue → prefill GPU的bootstrap队列
  → Prefill GPU只做prefill → 生成KV → 发送给decode GPU

SchedulerDisaggregationDecodeMixin:
  → DecodeTransferQueue → decode GPU接收KV
  → DecodePreallocQueue → 预分配decode KV cache
  → DecodeKVCacheOffloadManager → KV offload到CPU

KV Transfer:
  → prefill GPU → KV cache → 通过RDMA/NVLink → decode GPU
  → 不同backends: NCCL/RDMA/custom → 可配置

→ 这与vLLM的disaggregated prefill类似
→ 但SGLang的实现更精细(RDMA + KV offload + prealloc)
→ → 适合NVLink集群 → RTX 4090 PCIe不可行(无P2P)
```

## HiSparse Mode

```
SGLang的HiSparse(分层稀疏注意力):

hisparse_memory_pool.py + hisparse_coordinator:
  → 稀疏注意力 → 只计算部分KV → 减少内存和compute
  → 分层: GPU(HBM) → CPU(DRAM) → 不同层存储不同粒度KV
  → coordinator管理哪些KV在GPU哪些在CPU

→ 这是SGLang的特色功能 → vLLM没有类似实现
→ 适合长context推理(128K+) → 减少GPU内存压力
```

## 关键学习

```
SGLang vs vLLM架构对比:

| 特性 | vLLM V1 | SGLang |
|------|---------|---------|
| 核心调度 | FCFS unified budget | FCFS merge-based |
| KV Cache | BlockHashToBlockMap | RadixTree (prefix match) |
| Chunked Prefill | 内置 | 内置 |
| Disaggregation | 支持 | 支持(RDMA+offload+prealloc) |
| Overlap | 部分 | 全面(CUDA stream+MLX) |
| LoRA | LoRA serve | Multiplex(多LoRA混合batch) |
| Speculative | 内置 | 内置(ngram/Eagle) |
| MoE | DPEngineCoreProc | EPLB+elastic EP |
| 代码量 | ~2000行scheduler | 4068行scheduler |
| Mixin设计 | 否 | 7 mixin组合 → 更灵活 |

→ SGLang更灵活(mixin设计) → 但更复杂
→ vLLM更简洁 → 但功能不如SGLang丰富
→ RadixTree > BlockHash → prefix reuse效率更高
→ SGLang的HiSparse是特色 → vLLM没有

→ 对RTX 4090推理的建议:
  → 单GPU → vLLM或SGLang都可以(简单配置)
  → 多GPU → 不推荐PCIe TP → 单GPU最优
  → Prefix reuse → SGLang RadixTree效率更高
  → LoRA serve → SGLang Multiplex更好
```

## 工具

- SGLang源码: ~/rollout-infra/sglang/ (shallow clone)
- Skill: `.claude/skills/sglang-nav/SKILL.md`