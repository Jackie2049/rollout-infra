# LLM Serving Framework 对比: vLLM vs SGLang vs TensorRT-LLM

> 2026-06-05 | 综合: 源码分析 + 论文 + 实测数据
> 参考: vLLM (arXiv:2309.06180), SGLang (arXiv:2312.07104), TensorRT-LLM (NVIDIA)

## 1. 框架定位

| 维度 | vLLM | SGLang | TensorRT-LLM |
|------|------|--------|--------------|
| 开发者 | UC Berkeley | LMSYS (UC Berkeley) | NVIDIA |
| Stars | ~55K | ~15K | ~3K |
| 设计哲学 | 通用、灵活 | 极致吞吐 | 极致性能 |
| 目标场景 | 通用推理/RLHF | 高吞吐/RL | 生产部署 |
| 核心语言 | Python | Python | C++/CUDA |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 |

## 2. 架构对比

### 2.1 vLLM V1 (2025+)

```
AsyncLLM (前端进程)
    │
    ├── EngineCoreProc (ZMQ IPC, DEALER/ROUTER)
    │     ├── input_thread → input_queue
    │     ├── output_thread ← output_queue
    │     └── main loop:
    │           schedule() → execute_model() → sample() → update()
    │
    ├── Scheduler (priority, preemption)
    │     ├── prefill queue
    │     └── running queue
    │
    ├── KV Cache (block-based, hash-indexed)
    │     ├── BlockSpaceManager
    │     └── PrefixCacher (hash-based prefix sharing)
    │
    └── Sampling Pipeline
          ├── logits → processor → sample
          └── Gumbel-Max trick (消除 CPU-GPU 同步)
```

**特点**:
- 双进程架构 (AsyncLLM + EngineCoreProc)
- Hash-based prefix caching (block 级, 16 tokens/block)
- Continuous batching + preemption (priority-based)
- torch.compile 优化 (V1 默认 level 3)
- 5 个采样后端 (FlashInfer/PyTorch/CPU/XPU/ROCm)

### 2.2 SGLang

```
TokenizerManager (前端)
    │
    ├── Scheduler (单线程事件循环)
    │     ├── receive_batch()
    │     ├── schedule() (缓存感知)
    │     ├── forward_batch_generation() (GPU)
    │     └── process_batch_result() (CPU)
    │
    ├── RadixAttention (基数树)
    │     ├── match_prefix() → 最长前缀匹配
    │     ├── insert() → 动态节点分裂
    │     └── evict() → 7 种策略 (LRU/LFU/FIFO/...)
    │
    ├── Cache-Aware Scheduling
    │     ├── LPM (最长前缀优先)
    │     ├── DFS-weight
    │     └── 批内前缀检测
    │
    └── 多后端
          ├── Triton (decode attention)
          ├── FlashInfer (prefill attention)
          └── CUDA Graph (decode 固定形状)
```

**特点**:
- 单进程事件循环 (比 vLLM 双进程更轻)
- Radix Tree prefix caching (token 级, 无 block 对齐限制)
- 缓存感知调度 (LPM 策略主动优化缓存命中)
- Compressed Finite State Machine (结构化输出零开销)
- Prefill-Decode 分离 (disaggregated serving)

### 2.3 TensorRT-LLM

```
TensorRT Engine (编译期)
    │
    ├── Model Builder
    │     ├── 权重预处理 (量化/融合)
    │     ├── 计算图优化 (kernel fusion)
    │     └── Shape 专化 (固定 batch/seq)
    │
    ├── Runtime
    │     ├── KV Cache Manager
    │     ├── Beam Search / Sampling
    │     └── Inflight Batching
    │
    └── Plugin System
          ├── Tensor Core 针对性优化
          ├── FP8/INT4/INT8 量化
          └── 自定义 CUDA kernel
```

**特点**:
- AOT (Ahead-of-Time) 编译 — 模型转换为优化的 TensorRT engine
- C++/CUDA 核心, 极低开销
- NVIDIA 硬件专化 (H100/H200/B200)
- 编译时间长 (分钟级), 但推理最快
- 量化集成最深 (FP8/INT4 weight-only)

## 3. 核心技术差异

### 3.1 KV Cache & Prefix Caching

| 特性 | vLLM | SGLang | TRT-LLM |
|------|------|--------|---------|
| 数据结构 | 哈希表 (block hash → block) | Radix Tree (可变长度边) | Paged KV (NVIDIA 专化) |
| 匹配粒度 | Block 级 (16 tok) | Token 级 | Block 级 |
| 部分匹配 | 不支持 (block 对齐限制) | 支持 (动态节点分裂) | 有限支持 |
| 驱逐策略 | LRU (block ref_cnt) | 7 种 (LRU/LFU/Priority/...) | NVIDIA 专化 |
| LoRA 隔离 | hash 包含 lora_name | extra_key 字段 | 不适用 |

**实测数据 (RTX 4090, Prefix Caching)**:
- vLLM OPT-125M: Prefix=1024, suffix=64 → 91.7% prefill 节省
- SGLang: RadixAttention 树形共享 4 请求 → 48% 计算 + 403MB KV 节省
- SGLang 批内前缀检测: GRPO n=8 → 70% 计算节省

### 3.2 调度策略

| 特性 | vLLM | SGLang | TRT-LLM |
|------|------|--------|---------|
| 调度器 | PriorityScheduler | Cache-Aware Scheduler | Inflight Batching |
| 前缀感知 | 否 (独立于 prefix cache) | 是 (LPM/DFS-weight) | 否 |
| Preemption | 优先级 + swap/recompute | 无 (等待 GPU 资源) | NVIDIA 专化 |
| 批内优化 | 无 | 前缀检测 + 串行化 | 无 |
| Chunked Prefill | 是 (V1) | 是 | 是 |

### 3.3 结构化输出

| 特性 | vLLM | SGLang | TRT-LLM |
|------|------|--------|---------|
| JSON Schema | 支持 (guided decoding) | Compressed FSM | 支持 |
| Regex | 支持 | Compressed FSM | 支持 |
| 开销 | 有 (logits processor) | **零开销** (FSM 编译) | 低 |
| 实现 | xgrammars library | 自研 compressed FSM | NVIDIA plugin |

### 3.4 并行策略

| 策略 | vLLM | SGLang | TRT-LLM |
|------|------|--------|---------|
| TP | 是 | 是 | 是 |
| PP | 是 | 是 | 是 |
| EP (MoE) | 是 | 是 | 是 |
| DP | 是 | 是 | 是 |
| PD 分离 | 实验中 | 是 (生产就绪) | 是 |
| Data Parallel | 是 | 是 | 是 |

## 4. 性能对比

### 4.1 吞吐量 (公开数据)

| 场景 | vLLM | SGLang | TRT-LLM |
|------|------|--------|---------|
| Llama-7B 单卡 | 基线 | 1.2-1.5x | 1.5-2x |
| Prefix Sharing | 1x | **3-6x** | 1x |
| 结构化输出 | 基线 | 1.1x (零开销) | 1.2x |
| Multi-turn Agent | 1x | **~5x** | 1.2x |
| RLHF (GRPO n=8) | 1x | **2-3x** | 1.5x |

### 4.2 延迟

| 指标 | vLLM | SGLang | TRT-LLM |
|------|------|--------|---------|
| TTFT (首 token) | 中 | 低 | 最低 |
| ITL (token 间) | 中 | 低 | 最低 |
| Overhead | 双进程 IPC | 单进程事件循环 | 编译期消除 |

### 4.3 我们的实测 (RTX 4090, OPT-125M)

| 指标 | vLLM V1 | SGLang |
|------|---------|--------|
| B=1 throughput | 659 tok/s | - |
| B=512 peak | 30,244 tok/s | - |
| Prompt 吞吐 | 77,501 tok/s | - |
| CUDA Graphs | 67+35 个 | 固定形状 |

## 5. RLHF/RL 场景

SGLang 在 RL 场景有显著优势:

| 特性 | vLLM | SGLang |
|------|------|--------|
| verl 集成 | 支持 | **主力后端** |
| AReaL 集成 | 否 | **是** |
| Miles 集成 | 否 | **是** |
| GRPO prefix sharing | 需额外实现 | **天然支持** (RadixAttention) |
| n 次采样 KV 复用 | 部分 | **完全** |
| LoRA serving | 支持 | 支持 |

**关键**: verl/AReaL/Miles 等 RL 框架选择 SGLang 作为主力后端, 因为 GRPO n=8 采样共享 prompt, RadixAttention 天然复用 prompt KV → 70% 计算节省。

## 6. 适用场景推荐

| 场景 | 推荐 | 原因 |
|------|------|------|
| 通用 API 服务 | vLLM | 社区最大, 生态最全 |
| RLHF/RL 训练 | SGLang | prefix sharing 天然优势 |
| 多轮 Agent | SGLang | RadixAttention 自动缓存 |
| NVIDIA 生产部署 | TRT-LLM | 极致性能, 硬件专化 |
| 研究/实验 | vLLM | 灵活, Python-native |
| 边缘/小团队 | vLLM | 易部署, 文档全 |
| 高吞吐离线推理 | SGLang | 6.4x 基线吞吐 |

## 7. 技术趋势

1. **vLLM 正在补齐**: V1 引入 chunked prefill + torch.compile + hash-based prefix caching
2. **SGLang 扩展通用性**: 从 RL 专用走向通用 serving
3. **TRT-LLM 编译期优化**: AOT 的极限性能 vs 灵活性权衡
4. **PD 分离成为标配**: vLLM 实验中, SGLang 已生产就绪
5. **MoE serving 成熟**: 三个框架都支持 EP, MoE 成为主流架构

## 8. FP8/Quantization对比 (TransformerEngine深读补充)

| 特性 | vLLM | SGLang | TRT-LLM |
|------|------|--------|---------|
| INT4 weight-only | ✅ (TurboQuant) | ✅ | ✅ (最佳) |
| INT8 KV cache | ✅ | ✅ | ✅ |
| FP8 training | TE集成(verl) | TE集成(verl) | ✅ (原生) |
| FP8 recipe选择 | Delayed/Current | Delayed/Current | 所有5种 |
| RTX 4090 FP8 | Delayed+Current | Delayed+Current | Delayed+Current |
| MXFP8/NVFP4 | ❌ (需Blackwell) | ❌ (需Blackwell) | ✅ (Blackwell) |

**关键**: RTX 4090(SM89)只能用DelayedScaling+Float8CurrentScaling → MXFP8/NVFP4需Blackwell(SM100+)

## 9. 核心学习

1. **数据结构选择很关键**: Radix Tree vs Hash Table 决定了 prefix caching 的效率上限
2. **缓存感知调度是差异化**: SGLang 的 LPM 策略是其 RL 场景优势的核心
3. **AOT vs JIT 权衡**: TRT-LLM 的编译期优化 vs vLLM/SGLang 的运行时灵活
4. **RL 是 serving 的新战场**: GRPO/RLHF 对 serving 提出了 prefix sharing 新需求
5. **Python 也能很快**: SGLang 单进程事件循环证明了 Python 调度器足够高效

## 相关笔记

- `notebook/fundamentals/sglang-radix-attention.md` — SGLang RadixAttention 深度源码分析
- `notebook/fundamentals/inference-serving.md` — LLM 推理服务基础
- `notebook/fundamentals/prefix-caching.md` — Prefix Caching 技术详解
- `notebook/fundamentals/disaggregated-serving.md` — Prefill-Decode 分离
- `notebook/experiments/rtx4090-vllm-benchmark.md` — vLLM 实测数据
- `notebook/experiments/rtx4090-prefix-cache.md` — Prefix Caching 实测
