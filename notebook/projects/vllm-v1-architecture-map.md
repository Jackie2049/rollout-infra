# vLLM V1 架构全景图

> 基于源码阅读的完整架构地图，串联所有已学模块

## 1. 请求生命周期

```
Client Request
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ API Server (async_engine.py)                            │
│   - HTTP/gRPC 接收请求                                    │
│   - 封装为 EngineRequest                                 │
│   - asyncio → EngineCore (进程间通信)                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ EngineCore (v1/engine/__init__.py)                      │
│   ├── Scheduler  ← 核心调度逻辑                          │
│   ├── ModelRunner ← 模型执行                             │
│   └── KV Cache Manager ← KV block 管理                  │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌─────────┐  ┌──────────┐  ┌──────────┐
    │ Prefill │  │ Decode   │  │ Spec     │
    │ 调度    │  │ 调度     │  │ Decode   │
    └────┬────┘  └────┬─────┘  └────┬─────┘
         │            │             │
         └─────────────┼─────────────┘
                       ▼
              ┌────────────────┐
              │ GPU Execution  │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │ Sampling       │ → 输出 token
              └────────────────┘
```

## 2. 四层执行架构

```
┌─ Executor ─────────────────────────────────────────────┐
│  管理多个 Worker, 处理分布式协调                          │
│  - GPUExecutor (单卡/多卡)                               │
│  - MPExecutor (多进程)                                   │
│  - RayExecutor (分布式)                                  │
│  - P/D 分离: PrefillExecutor + DecodeExecutor           │
├─ Worker ────────────────────────────────────────────────┤
│  每个 GPU 一个 Worker 进程                               │
│  - 模型加载与初始化                                      │
│  - CUDA Graph 管理                                      │
│  - KV Cache 分配                                        │
├─ ModelRunner ───────────────────────────────────────────┤
│  执行模型前向传播                                        │
│  - 准备输入 (token_ids, positions, attn_metadata)        │
│  - 调用模型 forward()                                   │
│  - 执行 Sampling                                        │
│  - Speculative Decoding (Proposer + Scorer)             │
├─ Model ─────────────────────────────────────────────────┤
│  实际的 Transformer 模型                                 │
│  - 注意力层 (20+ Backend 选择)                           │
│  - 量化层 (30+ 方法)                                    │
│  - MoE 层                                               │
│  - MLA 层                                               │
└─────────────────────────────────────────────────────────┘
```

## 3. 调度器核心 (Scheduler)

**文件**: `vllm/v1/core/sched/scheduler.py` (~2400 行)

```
Scheduler.schedule()
  │
  ├── 1. 处理新请求 (add_request)
  │     └── 检查 KV Cache 可用性
  │     └── Prefix Caching 查找
  │
  ├── 2. 调度 Running 请求
  │     ├── 计算 token budget
  │     ├── Chunked Prefill (长 prompt 分块)
  │     ├── Speculative Decoding tokens
  │     └── P/D 分离: 远程 KV 检查
  │
  ├── 3. 抢占 (Preemption)
  │     ├── 基于 KV Cache 压力
  │     ├── FCFS / PRIORITY 策略
  │     └── 释放 block 给高优先级请求
  │
  └── 4. 构建 SchedulerOutput
        ├── token_ids, positions
        ├── block_table (KV Cache 映射)
        ├── SpecDecodeMetadata
        └── KV Transfer 参数 (P/D 分离)
```

**关键特性**:
- **统一调度**: 无 prefill/decode 区分，只有 token budget 分配
- **Chunked Prefill**: 长 prompt 分块处理，3 层约束
- **Prefix Caching**: BlockHashToBlockMap, 1:N 映射
- **Speculative Decoding**: draft token 调度和验证

## 4. KV Cache 管理器

**核心文件** (~11,700 行):

```
kv_cache_manager.py     ← 主管理器: block 分配/释放
block_pool.py           ← Block 池: 空闲/使用中管理
block.py                ← Block 抽象: 引用计数, hash
prefix_caching_pool.py  ← Prefix Caching: hash→block 映射
```

**Block 生命周期**:
```
分配 → 写入 KV → 可能共享 (prefix hash) → 引用计数管理 → 释放回池
```

**Prefix Caching**:
- BlockHash = hash(token_ids[block_start:block_end])
- 同 hash 的 block 只存一份
- 引用计数管理生命周期

## 5. Attention Backend 矩阵

| Backend | CUDA Graph | Prefill | Decode | MLA | 适用 |
|---------|-----------|---------|--------|-----|------|
| FlashAttention-2 | UNIFORM_BATCH | ✓ | ✓ | ✗ | SM 80+ |
| FlashAttention-3 | ALWAYS | ✓ | ✓ | ✗ | SM 90 (H100) |
| FlashInfer | SINGLE_TOKEN | ✓ | ✓ | ✗ | 通用 serving |
| FlashMLA | - | ✓ | ✓ | ✓ | SM 90, MLA |
| CUTLASS MLA | - | ✓ | ✓ | ✓ | SM 100 |
| Triton | ✗ | ✓ | ✓ | ✓ | Fallback |
| Paged Attention | ✗ | ✓ | ✓ | ✗ | Legacy |

**选择逻辑**: `AttentionBackendRegistry.get()` → 按模型配置 + GPU 架构自动选择

## 6. 量化管线 (Quantization Pipeline)

**3 层抽象**:
```
QuantizationConfig     ← 配置 (方法名, 参数)
  └── QuantizeMethod   ← 策略 (kernel 选择逻辑)
       └── Kernel       ← 实际执行 (scaled_mm, marlin, etc.)
```

**QuantKey 驱动 dispatch**:
```
QuantKey = (method, weight_dtype, activation_dtype, ...)
→ choose_scaled_mm_linear_kernel(quant_key)
→ 匹配最优 kernel
```

**30+ 量化方法**: FP8, AWQ, GPTQ, Marlin, INT4, INT8, etc.

**在线 vs 离线**:
- 离线: checkpoint 已量化 → 直接加载
- 在线: meta device → `finalize_layerwise_processing()` → 即时量化

## 7. 权重加载管线 (5 阶段)

```
Stage A: 模型构建 (meta tensors)
  → quant_method.create_weights() → 预分配分片

Stage B: 文件发现
  → safetensors / pt / fastsafetensors

Stage C: 权重迭代
  → safe_open() → yield (name, tensor)
  → EP: should_skip_weight() 跳过非本地 expert

Stage D: 参数映射
  → weight_loader(param, loaded_weight)
  → narrow(tp_rank×shard, shard_size) → copy_()
  → MergedColumn: 先 sub-projection 再 TP 切片

Stage E: 后处理
  → Marlin 重打包 / FP8 scale 合并 / AWQ 转置
```

## 8. Speculative Decoding 架构

```
Proposer (提议者)          Scorer (评分者)
├── DraftModel            ├── RejectionSampler
├── Eagle (推荐)          │   └── Triton kernel
├── Medusa                │
├── N-gram (零开销)       │   acceptance_prob = P(target)/P(draft)
├── DFlash (并行)         │   accepted = rand < acceptance_prob
├── Suffix (后缀树)       │
└── MTP                   └── bonus token (额外采样)
```

**性能优化**:
- 本地 Argmax: O(vocab_size) → O(2×tp_size)
- 并行 Drafting: 一次生成所有 speculative token
- CUDA Graph: Proposer 部分可用 Graph 加速

## 9. P/D 分离架构

```
┌─────────────┐     KV Transfer     ┌─────────────┐
│ Prefill     │ ─────────────────→  │ Decode      │
│ Instance    │  NIXL / FlexKV /    │ Instance    │
│             │  LMCache / Mooncake │             │
└─────────────┘                     └─────────────┘
  TP=P          ← 异构 TP →           TP=D

Connector 抽象:
  KVConnectorBase_V1
    ├── NixlConnector (推荐)
    │   ├── Scheduler 端: 元数据 + 握手
    │   └── Worker 端: NIXL DMA 传输
    ├── FlexKVConnector
    ├── LMCacheConnector
    └── MooncakeConnector
```

**关键机制**:
- 异步 DMA + ZMQ side channel
- KV Lease 延迟释放
- 异构 TP 支持

## 10. CUDA Graph 集成

```
CudagraphDispatcher
  │
  ├── FULL 模式: 完整前向传播录制
  │   └── graph.replay() 直接回放
  │
  ├── PIECEWISE 模式: 分段录制
  │   └── torch.compile + Breakable Graph
  │
  └── 模式选择优先级: FULL > PIECEWISE > NONE
```

**Attention 兼容性**:
- FA3: ALWAYS → FULL 可用
- FA2: UNIFORM_BATCH → 仅均匀 batch
- FlashInfer: SINGLE_TOKEN → 仅 decode

## 11. MoE Serving

```
模型层: Router → Prepare → Experts → Finalize
  │
  ├── Expert Parallelism (EP)
  │   └── All-to-All 通信
  │       ├── DeepEP-HT (高吞吐)
  │       └── DeepEP-LL (低延迟)
  │
  └── 8+ All-to-All Backend
```

## 12. 模块依赖关系图

```
                    ┌──────────┐
                    │ Scheduler│
                    └─────┬────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌────────┐ ┌──────────┐
        │ KV Cache │ │  Spec  │ │    P/D   │
        │ Manager  │ │ Decode │ │ Connector│
        └────┬─────┘ └───┬────┘ └────┬─────┘
             │           │           │
             └───────────┼───────────┘
                         ▼
                  ┌──────────────┐
                  │  ModelRunner │
                  └──────┬───────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌───────────┐  ┌───────────┐  ┌───────────┐
   │ Attention │  │Quantize   │  │ Weight    │
   │ Backend   │  │ Pipeline  │  │ Loading   │
   └───────────┘  └───────────┘  └───────────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
                  ┌──────────────┐
                  │  CUDA Graph  │
                  │  + Streams   │
                  └──────────────┘
```

## 13. 已学源码阅读清单

| 模块 | 笔记 | 核心行数 |
|------|------|---------|
| V1 Scheduler | `vllm-v1-scheduler-reading.md` | ~2400 |
| V1 Executor | `vllm-v1-executor-reading.md` | 四层架构 |
| KV Cache Manager | `vllm-v1-kv-cache-manager-reading.md` | ~11,700 |
| MLA Backend | `vllm-mla-backend-reading.md` | 4 backend |
| MoE Serving | `vllm-moe-serving-reading.md` | 8+ backend |
| Quantization | `vllm-quantization-pipeline-reading.md` | 30+ methods |
| Weight Loading | `vllm-weight-loading-reading.md` | 5 阶段 |
| Speculative Decoding | `vllm-spec-decode-reading.md` | 8+ proposer |
| P/D Disaggregation | `vllm-pd-disaggregation-reading.md` | NIXL 等 |
| CUDA Graph | `vllm-cuda-graph-reading.md` | 5 模式 |
| Sampling Pipeline | `vllm-sampling-pipeline-reading.md` | 顺序管线, 3种Top-K/P实现 |
| Data Parallelism | `vllm-data-parallel-reading.md` | MoE EP, Wave协调, DBO |
| Observability | `vllm-observability-reading.md` | Stats/Logger/Prometheus 三层 |
| LoRA Serving | `vllm-lora-serving-reading.md` | Punica Multi-LoRA, Triton kernel |
| Multi-Modal | `vllm-multimodal-reading.md` | EncoderRunner, Hash Cache |
| Structured Output | `vllm-structured-output-reading.md` | Bitmask FSM, 4种后端 |
| Async Engine Frontend | `vllm-async-engine-frontend-reading.md` | 双进程 ZMQ, output_handler |
| Engine Core Detail | `vllm-engine-core-reading.md` | step() 循环, busy loop, ZMQ IO |
| 模型压缩 | — | 蒸馏/剪枝/量化/综合策略 |

## 14. 待探索模块

- [x] Sampling Pipeline (logits → penalties → sample → output)
- [x] vLLM V1 Multi-Modal 处理
- [x] vLLM V1 LoRA Serving
- [x] vLLM 日志和可观测性 (metrics/tracing)
- [x] vLLM 数据并行 (Data Parallelism)
- [x] Structured Output (JSON/Grammar/Regex Bitmask)
- [x] vLLM V1 Async Engine Frontend (AsyncLLM)
- [x] vLLM V1 Engine Core Detail (step loop, scheduling)
- [x] vLLM V1 Prompt Adapter (V1 中不存在独立 PromptAdapter，功能已合并到 LoRA)

## 参考资料

- 源码: `vllm/v1/` 目录
- 配套模拟器: `tools/` 目录下所有 `*_sim.py` 工具
