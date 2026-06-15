# MindIE生产部署架构深度分析

> 2026-06-16 | MindIE-Service | CANN 9.0 | HCCL | BudgetRefiner SLO
> ★★★★★ 完整生产部署pipeline → config.json → HCCL tuning → 多模型serving → SLO-aware调度
> ★★★★★ BudgetRefiner = Ascend独有 → vLLM V1无等效 → 可提议贡献!

## 1. ★★★★★ MindIE Serving配置架构

```
★★★★★★★ config.json两大主区:

serverConfig:
  → ipAddress/port → 服务绑定
  → maxBatchSize/maxSeqLen → 直接决定NPU HBM消耗! → 设太高→OOM!
  → maxConcurrentNum → 最大并发连接
  → preTimeOut/timeOut → prefill/total超时
  → authentication → 认证开关

modelConfig:
  → npuDeviceIds → ★★★★★ nested list → 外层=模型实例, 内层=NPU cards per instance
  → modelPath/modelName → 模型权重路径+标识
  → worldSize → 每实例设备数 → 必须match内层长度!
  → isOmOptimize → OM离线优化开关
  → maxPreFillBatchSize/maxDecodeBatchSize → 分阶段batch
  → maxPreFillSeqLen/maxDecodeSeqLen → 分阶段seq长度
  → supportPaPrefill/supportPaDecode → Parallel-and-Aggregate → 多卡prefill/decode
  → backendType → "atb" (推荐) 或 "acl"
  → cacheBlockSize → KV cache block大小 → 128 (default)

★★★★★★★ 2025新增配置区:
  → QuantConfig → INT4/INT8量化 (quant_method, quant_weight_path)
  → KVCacheConfig → KV cache优化 (cache_size, block_size, kv_cache_dtype)
  → tensor_parallel_size → 替代worldSize
  → HCCL参数 → hccl_timeout, hccl_debug_level

★★★★★★★ npuDeviceIds多模型serving:
  → [[0]] → 1实例, 1NPU
  → [[0,1]] → 1实例, 2NPU (TP=2)
  → [[0], [1]] → 2实例, 各1NPU (多模型)
  → [[0,1,2,3]] → 1实例, 4NPU (TP=4)
  → ★★★★★ 非重叠device IDs → 设备隔离 → 必须!
```

## 2. ★★★★★ MindIE推理Pipeline (HTTP→Model→Response)

```
★★★★★★★ 7阶段推理pipeline:

[1] MindIE-Service HTTP/gRPC API
  → OpenAI-compatible: /v1/chat/completions, /v1/completions, /v1/models
  → Request解析 + 验证 + tokenization

[2] Request Queue / Scheduler
  → Continuous batching → 请求动态加入/离开running batch
  → 优先级路由 (多模型)
  → ★★★★★ BudgetRefiner SLO-aware budget adjustment (Ascend独有!)

[3] MindIE-LLM Engine (或 vLLM-Ascend engine)
  → Prefill phase → 处理input tokens → build KV cache
  → Decode phase → 逐步生成output tokens
  → KV cache → PagedAttention-Ascend block管理

[4] ATB Operator Execution
  → FlashAttention-Ascend, ATB Linear, RMSNorm, RoPE, SwiGLU, MoE ops
  → ★★★★★ Compose-level fusion → 整Transformer层 = 1 composite op!
  → Quantize/dequantize inline

[5] CANN Runtime
  → Kernel dispatch → Ascend NPU AI Core
  → HCCL通信 → distributed inference
  → Memory管理 → CANN allocator

[6] Sampling & Output
  → ATB TopK/TopP fused sampling kernel
  → Token detokenization

[7] HTTP Response
  → SSE streaming 或 complete JSON response
  → OpenAI-compatible format
```

## 3. ★★★★★ NVIDIA vs Ascend栈映射

```
★★★★★★★ Ascend ↔ NVIDIA栈映射:

| Ascend Layer | NVIDIA Equivalent |
|-------------|-------------------|
| MindIE-Service | Triton Inference Server |
| MindIE-LLM | TensorRT-LLM |
| ATB | cuBLAS/cuDNN/CUTLASS |
| HCCL | NCCL |
| CANN Runtime | CUDA Runtime |
| CANN Driver | CUDA Driver |

★★★★★★★ 3种Ascend serving路径对比:

| 维度 | MindIE (官方) | vLLM-Ascend (社区) | SGLang-Ascend |
|------|--------------|--------------------|---------------|
| Patch粒度 | 全栈垂直集成 | op-level patch (5层) → 可替换! | graph-level → MindIE黑盒 |
| 灵活性 | ★★ 低 (闭源) | ★★★★★ 高 (每个op可替换) | ★★ 低 (graph-level不透明) |
| Debug能力 | ★★ 有限 (华为工具) | ★★★★★ 高 (标准vLLM debug) | ★★ 有限 (黑盒) |
| 生产推荐 | Ascend-only企业 | ★★★★★ 推荐 (灵活+可控) | 预生产/验证 |
| SLO支持 | MindIE内部 | ★★★★★ BudgetRefiner | 无builtin |

★★★★★★★ vLLM-Ascend推荐理由:
  → op-level patch → 每层可替换 → 灵活
  → 标准vLLM debugging → 熟悉工具链
  → BudgetRefiner SLO → 生产级SLO保证 → vLLM V1没有!
```

## 4. ★★★★★ BudgetRefiner SLO Pipeline集成

```
★★★★★★★ BudgetRefiner = vLLM-Ascend独有 → vLLM V1无等效!

Pipeline集成:
  Request → SchedulerDynamicBatch → BudgetRefiner.refine_budget(100)
  → Input: running requests + decode token count
  → Lookup: profile_table.csv (ctx_len, num_decode) → max_chunk_size
  → Output: dynamic token budget per iteration
  → Condition: slo_limit > 0 → 否则static max_num_batched_tokens

★★★★★★★ BudgetRefiner核心机制:
  → profile_table.csv → pre-profiled latency查找表
  → ProfilingChunkPredictor → 二次模型 f(l) = a*l^2 + b*l + c
  → 给target latency T → 求chunk size x: a*x^2 + (2aL+b)*x - T = 0
  → Decode-first sorting → decode优先 → 满足SLO
  → Prefill dynamically shrinks → 如果decode SLO at risk → throttle prefill

★★★★★★★ vs vLLM V1:

| 方面 | vLLM V1 | BudgetRefiner (Ascend) |
|------|---------|------------------------|
| Budget类型 | Fixed max_num_batched_tokens | Dynamic, per-iteration |
| SLO意识 | 无 | Full (TTFT/TPOT targets) |
| Decode优先 | FCFS/PRIORITY | Decode-first enforced |
| Prefill throttling | 无SLO机制 | Prefill动态缩小 |
| Preemption恢复 | Drop to waiting, 从头recompute | ★★★★★ KV transfer到decode实例! |
| 生产适用 | 通用, 无SLO保证 | ★★★★★ SLO-aware, SLA-guaranteed |

★★★★★★★ BudgetRefiner可提议贡献vLLM upstream!
  → vLLM V1 → 无SLO-aware scheduling → BudgetRefiner填补gap
  → profile_table.csv → portable → 可迁移到GPU
  → ★★★★★ 这是vLLM-Ascend最有价值的upstream贡献候选!
```

## 5. ★★★★★ CANN + HCCL生产部署

```
★★★★★★★ CANN版本兼容矩阵:

| CANN版本 | 硬件 | torch_npu | vLLM-Ascend |
|----------|------|-----------|-------------|
| 8.0.RC2 | 910B1 | 2.5.1 | v0.7.3 |
| 8.5.0/8.5.1 | 910B/910C | 2.9.0 | v0.18.0 |
| 8.5.2 | 910B/910C (Qwen3.5) | - | v0.18.x |
| ★★★★★ 9.0.0 | 910B/910C | 2.10.0 | v0.20.2rc1 |

★★★★★★★ HCCL关键环境变量:
  → HCCL_CONNECT_TIMEOUT=7200 → 连接超时
  → HCCL_EXEC_TIMEOUT=7200 → 执行超时
  → HCCL_WHITELIST_DISABLE=1 → 禁用白名单过滤
  → HCCL_IF_IP → 通信接口IP
  → HCCL_DETERMINISTIC=0 → 非确定性 (性能优先)
  → ★★★★★ HCCL_BUFFSIZE → 按phase不同!
    → Prefill: 800-1536 (大buffer → throughput)
    → Decode: 600-720 (小buffer → latency)

★★★★★★★ 性能tuning环境变量:
  → PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  → TASK_QUEUE_ENABLE=1/2 → 1=decode, 2=prefill
  → COMBINED_MODE_ENABLE=1
  → STREAMS_PER_DEVICE=32 → 多流overlap
  → SGLANG_NPU_USE_MLAPO=1 → MLA预处理优化
  → SGLANG_USE_FIA_NZ=1 → FlashInfer NZ格式
  → ★★★★★ HCCL_OP_EXPANSION_MODE=AIV → Ascend Vector Engine → 最接近NCCL Gin!

★★★★★★★ CPU/系统tuning:
  → echo performance → CPU governor
  → vm.swappiness=0 → 禁用swap
  → kernel.numa_balancing=0 → 禁用NUMA迁移
  → kernel.sched_migration_cost_ns=50000
  → SGLANG_SET_CPU_AFFINITY=1 → CPU亲和性绑定
```

## 6. ★★★★★ MindIE Turbo部署 (DeepSeek专用)

```
★★★★★★★ MindIE Turbo = DeepSeek专用 → 不开源!

3核心优化:
  → Speculative decoding → MTP/EAGLE-style draft → NEXTN → 1.5-2x
  → Compose-level fusion → 整Transformer层=1 op → 3-5x kernel launch reduction
  → MLA native kernels → MLA preprocess all-fused → 4 cache modes

★★★★★★★ DeepSeek-R1 PD disaggregation (SGLang-Ascend最佳实践):

Prefill配置:
  → HCCL_BUFFSIZE=800, TASK_QUEUE_ENABLE=2
  → TP=16, DP=4, enable-dp-attention
  → moe-a2a-backend=ascend_fuseep, deepep-mode=normal
  → NEXTN spec: steps=1, eagle-topk=1, draft-tokens=2
  → mem-fraction-static=0.778, max-prefill-tokens=60000

Decode配置:
  → HCCL_BUFFSIZE=600
  → TP=32, DP=32, enable-dp-attention
  → moe-a2a-backend=ascend_fuseep, deepep-mode=low_latency
  → SGLANG_ENABLE_SPEC_V2=1 + SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
  → mem-fraction-static=0.82, max-running-requests=1024
  → cuda-graph-bs: 2 4 6 8 ... 32 (16 batch sizes)

★★★★★★★ DeepSeek生产benchmark (A3 32-card):
  → DeepSeek-R1 PD: 20ms TPOT (W8A8 INT8)
  → DeepSeek-R1 PD: 19ms TPOT (3.5K+1.5K dataset)
  → DeepSeek-R1 8卡: 50ms TPOT (W4A8 INT8, PD Mixed)

★★★★★★★ MindIE Turbo → 不适用RTX 4090:
  → Ascend-only → SM89不可用
  → 不开源 → 无法学习具体优化
  → 但ATB compose-level理念 → 可启发NVIDIA stack改进!
```

## 7. ★★★★★ Observability Stack

```
★★★★★★★ MindIE完整observability stack:

[1] Prometheus + Grafana
  → MindIE-Service: /metrics endpoint
  → vLLM-Ascend: rich request latency, queue depth, NPU util, KV cache

[2] msmonitor-exporter → NPU硬件metrics
  → NPU utilization, memory, power, temperature
  → Huawei实时NPU监控daemon

[3] vLLM metrics → 应用级metrics
  → Request throughput, TTFT, TPOT, e2e latency
  → Queue depth, batch size distribution

[4] msprof → 深度profiling
  → Operator级profiling (类似NVIDIA Nsight)
  → ATB kernel级timing → 精确分析

[5] ★★★★★ BudgetRefiner SLO tracking
  → Per-iteration latency budget tracking
  → SLO target vs actual → SLA验证

★★★★★★★ Health check endpoints:
  → /health 或 /healthz → liveness probe
  → /ready 或 /readyz → readiness probe (model loaded + ready)
```

## 8. ★★★★★ 关键发现

```
★★★★★★ 5个关键发现:

1. ★★★★★ BudgetRefiner SLO = Ascend独有 → 可提议贡献vLLM upstream!
   → vLLM V1 → 无SLO → BudgetRefiner填补 → profile_table.csv portable → GPU也可用!
   → ★★★★★ 这是vLLM-Ascend最有价值的upstream贡献候选!

2. ★★★★★ npuDeviceIds nested list = 多模型serving核心
   → 外层=模型实例 → 内层=NPU cards → 设备隔离 → 路由by modelName
   → 类似vLLM multi-model API → 但Ascend硬件绑定更强

3. ★★★★★ HCCL BUFFSIZE按phase不同 → 生产tuning关键!
   → Prefill: 800-1536 (大buffer → throughput)
   → Decode: 600-720 (小buffer → latency)
   → ★★★★ 类似概念可应用于GPU NCCL tuning!

4. ★★★★★ MindIE vs vLLM-Ascend vs SGLang-Ascend:
   → vLLM-Ascend推荐 → op-level patch → 灵活 → debug → BudgetRefiner
   → MindIE → closed → 企业用 → 不透明
   → SGLang → graph-level → 快速验证 → 不深入tuning

5. ★★★★★ CANN 9.0 + HCCL AIV = 最接近NCCL Gin的Ascend等效
   → AIV mode → Ascend Vector Engine → GPU直接通信 → 类似NCCL Gin
   → 但HCCL ≠ NCCL → API不同 → 需要HCCL adapter for DeepEP
```

## 参考
- MindIE-Service config.json: 华为Ascend开发者文档
- vLLM-Ascend: https://github.com/vllm-project/vllm-ascend
- SGLang-Ascend: https://github.com/sgl-project/sglang
- BudgetRefiner: vllm-ascend/vllm_ascend/scheduler/scheduler_dynamic_batch.py
- CANN 9.0文档: https://www.hiascend.com/document
- 相关笔记: mindie-architecture-reading.md, mindie-atb-kernel-architecture-reading.md, vllm-ascend-serving-layer-reading.md, deepep-ascend-reading.md
