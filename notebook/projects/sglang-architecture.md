# SGLang LLM Serving Framework 架构深度分析

> 作者: Claude + jackie2049 | 日期: 2026-06-04
> 论文: SGLang: Efficient Execution of Structured LLM Programs (arXiv:2312.07104)
> 作者: Lianmin Zheng, Liangsheng Yin 等 (LMSys, UC Berkeley, Stanford)
> 源码: https://github.com/sgl-project/sglang (28k stars)
> 生态: xAI, AMD, NVIDIA, LinkedIn, Cursor 等, 400,000+ GPUs 生产部署

---

## 目录

1. [SGLang vs vLLM: 架构对比](#1-sglang-vs-vllm-架构对比)
2. [RadixAttention: 基数树 KV Cache 共享](#2-radixattention-基数树-kv-cache-共享)
3. [调度系统: Continuous Batching + Chunked Prefill](#3-调度系统-continuous-batching--chunked-prefill)
4. [前端 DSL: 结构化生成控制](#4-前端-dsl-结构化生成控制)
5. [FlashInfer 集成与 Attention Backend](#5-flashinfer-集成与-attention-backend)
6. [性能基准测试](#6-性能基准测试)
7. [源码结构分析](#7-源码结构分析)

---

## 1. SGLang vs vLLM: 架构对比

### 1.1 设计哲学差异

| 维度 | SGLang | vLLM |
|------|--------|------|
| **KV Cache 共享** | RadixAttention (基数树) | Hash-based Prefix Caching (V1: BlockHash) |
| **调度器位置** | CPU 端 (零开销) | GPU 端 (V1 Scheduler) |
| **前端语言** | DSL (fork/gen/choices) | 无专用前端, OpenAI API |
| **结构化输出** | Compressed FSM (跳跃解码) | xGrammar / Outlines FSM |
| **Attention Backend** | FlashInfer 为主 | FlashInfer + 自研 V1 backend |
| **默认页大小** | 1 (FlashInfer) / 64 (MLA) | 16 (V1) |
| **CUDA Graph** | 全量 decode batch | 全量 decode batch |
| **并行策略** | TP/PP/EP/DP/DP-Attention | TP/PP/EP (V1) |

### 1.2 核心架构差异

**SGLang** 的核心创新在于将 LLM serving 视为一个 **结构化程序执行** 问题, 而非简单的请求-响应服务:

1. **前端 DSL**: 提供编程语言级别的生成控制 (fork, gen, choices, regex, json)
2. **RadixAttention**: 基于基数树的 KV cache 共享, 比 vLLM 的 hash prefix caching 更灵活
3. **Compressed FSM**: 压缩有限状态机实现结构化输出的跳跃解码
4. **零开销 CPU 调度器**: v0.4 引入, 调度开销从关键路径移除

**vLLM V1** 的对应设计:
1. **BlockHash Prefix Caching**: 基于 hash 的 KV block 复用 (V1 BlockHashToBlockMap 1:N 映射)
2. **xGrammar/Outlines**: 不同的结构化输出方案
3. **GPU 端调度**: Scheduler 在 GPU 上直接执行

### 1.3 性能差异来源

SGLang 相对 vLLM 的性能优势主要来自:
- **KV Cache 共享效率**: RadixAttention 的基数树比 hash prefix 匹配更高效, 支持更细粒度的共享
- **调度开销**: v0.4 的零开销 CPU 调度器消除了调度瓶颈
- **前端优化**: Compressed FSM 跳跃解码减少结构化输出的 decode 步数 (2x 延迟降低)
- **torch.compile 集成**: 小 batch 下 fuse kernel 带来 1.5x 延迟降低

---

## 2. RadixAttention: 基数树 KV Cache 共享

### 2.1 Radix Tree 基础

Radix Tree (基数树/压缩前缀树) 是 Trie 的空间优化变体:
- **Trie**: 每条边标记单个 token, 路径 = token 序列
- **Radix Tree**: 边可以标记为任意长度的 token 序列, 压缩单分支路径
- **空间效率**: 只有分叉节点才占用独立节点, 单分支路径合并为一条边

在 SGLang 中:
- **Key**: token 序列 (prompt + 已生成的 tokens)
- **Value**: 对应的 KV cache tensors (存储在 paged GPU layout 中)
- **操作**: prefix matching, insertion, eviction (LRU)

### 2.2 RadixAttention 工作流程

以 LMSys 博客的 9 步示例说明 (树进化过程):

```
初始状态: 空树, root node

Step 1: 请求 "A B C" prefill
  → 树: root → [A B C]
  → 分配 KV cache 给 A, B, C

Step 2: 请求 "A B C D E" (A B C 是前缀)
  → 匹配 [A B C], 复用其 KV cache
  → 只 prefill D, E
  → 树: root → [A B C] → [D E]

Step 3: 请求 "A B C F G" (A B C 是前缀)
  → 匹配 [A B C], 复用
  → prefill F, G
  → 树: root → [A B C] → [D E]
                       → [F G]

Step 4: 请求 "A B H I" (A B 是前缀)
  → 匹配 [A B], 分裂 [A B C] 为 [A B] + [C]
  → prefill H, I
  → 树: root → [A B] → [C] → [D E]
                       → [F G]
                       → [H I]
```

**关键操作**:
1. **Prefix Match**: 从 root 开始, 沿着 token 序列匹配已有的边
2. **Split**: 当匹配到一个边的部分前缀时, 将该边分裂
3. **Insert**: 将新 token 序列作为新的子树插入
4. **Evict**: 当 KV cache 容量不足时, 按 LRU 策略驱逐叶子节点

### 2.3 KV Cache 共享模式

RadixAttention 天然支持四种常见的 KV cache 复用场景:

1. **Multi-turn Chat**: 同一对话的前缀 (system prompt + 历史) 在每轮都复用
2. **Few-shot Learning**: 示例前缀在所有请求中共享
3. **Self-consistency**: 同一 prompt 多次采样, 只需复制 prompt KV cache
4. **Tree-of-thought**: 分叉点共享前缀 KV cache, 不同分支独立扩展

### 2.4 与 vLLM Prefix Caching 对比

| 特性 | SGLang RadixAttention | vLLM V1 Prefix Caching |
|------|----------------------|----------------------|
| **数据结构** | Radix Tree (内存) | Block Hash Table |
| **匹配粒度** | Token 序列任意长度 | Block 粒度 (默认 16 tokens) |
| **分裂操作** | 支持边分裂, 细粒度 | 不支持, block 为最小单位 |
| **驱逐策略** | LRU 叶子节点 | LRU block eviction |
| **树结构** | 自动维护前缀关系 | Hash table, 无显式树结构 |
| **共享模式** | 自动支持所有 4 种 | 主要支持 linear prefix 复用 |

**vLLM V1 的实现**:
- `BlockHashToBlockMap`: 1:N 映射 (同一 hash 可对应多个物理 block)
- `KVCacheManager`: 8 核心文件 ~11,700 行
- Hash 计算: 基于 block 内 token IDs 的 hash

**SGLang 的优势**:
- 基数树天然维护前缀关系, 查找 O(L) (L=token 序列长度)
- 边分裂支持细粒度共享, 不受 block 边界限制
- 树形结构使得 self-consistency / tree-of-thought 场景自动获得 KV cache 复用

### 2.5 源码位置

- RadixAttention 实现: `python/sglang/srt/mem_cache/` 目录
- Tree Cache: radix tree 数据结构
- KV Cache Builder: `python/sglang/srt/mem_cache/kv_cache_builder.py`
- 配置: `--disable-radix-cache` 可关闭 RadixAttention

---

## 3. 调度系统: Continuous Batching + Chunked Prefill

### 3.1 Scheduler 总体架构

SGLang 的 Scheduler 是整个运行时的核心, 位于 `python/sglang/srt/managers/scheduler.py` (~3800 行).

Scheduler 类使用 Mixin 模式组织:

```python
class Scheduler(
    SchedulerDisaggregationDecodeMixin,   # P/D 分离 decode 端
    SchedulerDisaggregationPrefillMixin,  # P/D 分离 prefill 端
    SchedulerMultiplexMixin,              # 多路复用
    SchedulerPPMixin,                     # Pipeline Parallel
    SchedulerDllmMixin,                   # Diffusion LLM
    SchedulerMlxOverlapMixin,             # MLX 重叠
):
```

### 3.2 调度循环核心

```
                    ┌──────────────┐
                    │  Waiting     │
                    │  Queue       │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Prefill     │ ← chunked prefill
                    │  Adder       │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Running     │ ← continuous batching
                    │  Batch       │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Forward     │ ← model worker 执行
                    │  Batch       │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Result      │ ← 处理输出, 淘汰完成的请求
                    │  Processor   │
                    └──────────────┘
```

### 3.3 关键调度组件

从 scheduler.py 的 import 可以看到完整的组件化架构:

| 组件 | 路径 | 职责 |
|------|------|------|
| `SchedulePolicy` | `schedule_policy.py` | FCFS / PRIORITY 调度策略 |
| `PrefillAdder` | `schedule_policy.py` | Prefill 请求准入控制 |
| `ScheduleBatch` | `schedule_batch.py` | 批次管理 (Req 列表) |
| `RequestReceiver` | `scheduler_components/request_receiver.py` | 接收新请求 |
| `BatchResultProcessor` | `scheduler_components/batch_result_processor.py` | 处理 forward 结果 |
| `SchedulerMetricsReporter` | `scheduler_components/metrics_reporter.py` | 性能指标上报 |
| `SchedulerOutputStreamer` | `scheduler_components/output_streamer.py` | 流式输出 |
| `SchedulerLoadInquirer` | `scheduler_components/load_inquirer.py` | 负载查询 |
| `SchedulerFlushWrapper` | `scheduler_components/flush_wrapper.py` | Cache 刷新 |
| `SchedulerInvariantChecker` | `scheduler_components/invariant_checker.py` | 不变量检查 (debug) |
| `SchedulerDPAttnAdapter` | `scheduler_components/dp_attn.py` | Data Parallel Attention |
| `IdleSleeper` | `scheduler_components/idle_sleeper.py` | 空闲休眠 |

### 3.4 Continuous Batching

SGLang 的 continuous batching 机制:

1. **Running Batch**: 维护当前正在 decode 的请求集合
2. **Prefill Admission**: 新请求从 waiting queue 进入, 经过 PrefillAdder 准入
3. **Decompletion**: 请求生成完成后从 running batch 移除, 释放 KV cache
4. **抢占**: 当 KV cache 不足时, 可抢占低优先级请求 (priority scheduling)

与 vLLM V1 类似:
- 都使用 iteration-level 调度
- 都支持 preemption (SGLang: retract, vLLM: preemption)
- 都支持 chunked prefill

### 3.5 Chunked Prefill

配置参数: `--chunked-prefill-size` (默认值较大, 可降至 4096/2048 防止 OOM)

**工作方式**:
- 长 prompt 被分成多个 chunk
- 每个 chunk 作为一次 forward pass 执行
- 与 decode batch 可交错执行 (overlap)
- 减少长 prompt 对短请求的延迟影响

**与 vLLM 的差异**:
- vLLM V1: `Scheduler.running_queue` + `Scheduler.waiting_queue`, chunked prefill 通过 `num_computed_tokens` 追踪
- SGLang: `Scheduler.waiting_queue` + `Scheduler.running_batch`, `chunked_req` 追踪进行中的 prefill

### 3.6 零开销 CPU 调度器 (v0.4)

v0.4 引入的关键优化:
- 将调度逻辑从 GPU 关键路径移到 CPU
- GPU 在执行当前 batch 时, CPU 并行准备下一个 batch
- 使用 CUDA stream 实现异步 overlap
- 调度开销从 ~5% 降低到接近 0%

### 3.7 分布式调度支持

Scheduler 通过 Mixin 支持多种分布式模式:

- **Tensor Parallel**: 通过 `ParallelState` 管理 tp_rank, tp_size
- **Pipeline Parallel**: `SchedulerPPMixin` 处理 micro-batch 调度
- **Expert Parallel**: MoE 模型的 expert 分布
- **Data Parallel**: `SchedulerDPAttnAdapter` 支持 DP Attention
- **P/D Disaggregation**: Prefill/Decode 分离部署
  - `SchedulerDisaggregationPrefillMixin`: prefill 节点调度
  - `SchedulerDisaggregationDecodeMixin`: decode 节点调度
  - `DecodeKVCacheOffloadManager`: KV cache offload 管理

---

## 4. 前端 DSL: 结构化生成控制

### 4.1 SGLang DSL 概述

SGLang 提供一个嵌入式 DSL (Python), 用于精确控制 LLM 生成过程. 这是 SGLang 区别于其他 serving 框架的独特特性.

两种执行模式:
1. **Interpreter 模式**: 逐条执行, 简单直接
2. **Compiler (Graph Executor) 模式**: 编译为执行图, 支持更激进的优化

### 4.2 核心原语

```python
import sglang as sgl

# gen: 非阻塞生成
@sgl.function
def example(s, question):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", max_tokens=256)

# fork: 并行分支 (用于 self-consistency)
@sgl.function
def self_consistency(s, question):
    s += "Q: " + question + "\n"
    forks = s.fork(5)  # 创建 5 个并行分支
    for f in forks:
        f += "A:" + sgl.gen("answer", max_tokens=100)
    s += sgl.join(forks)  # 合并结果
    s += "Best answer:" + sgl.gen("best", max_tokens=256)

# choices: 限定选择
@sgl.function
def classify(s, text):
    s += "Classify: " + text + "\n"
    s += "Category: " + sgl.gen("cat", choices=["positive", "negative", "neutral"])

# regex: 正则约束
@sgl.function
def extract_json(s, text):
    s += "Extract from: " + text + "\n"
    s += "Result: " + sgl.gen("result", regex=r'\{[a-z]+: [0-9]+\}')

# select: 从候选中选择
@sgl.function
def multi_choice(s, question, options):
    s += question + "\n"
    s += sgl.select("answer", choices=options)
```

### 4.3 Compressed FSM (压缩有限状态机)

这是 SGLang 在结构化输出上的核心优化.

**背景问题**: JSON/regex 约束解码需要逐 token 检查 FSM 状态, 每步只能生成 1 token.

**现有方案**:
1. **FSM-based (Outlines)**: 构建 regex 的 FSM, 逐 token 转移状态
2. **Interleaved-based (Guidance)**: 将 prompt 和约束交替执行

**SGLang 的创新**: Compressed FSM

```
原始 FSM:
  State 0 → 'a' → State 1 → 'p' → State 2 → 'p' → State 3 → 'l' → State 4 → 'e' → State 5

压缩后 (singular path 合并):
  State 0 → "apple" → State 5
```

**算法**:
1. 分析 regex 的 FSM
2. 识别所有 singular transition (只有一条出边的状态)
3. 压缩连续的 singular transitions 为单条 "jump-forward" 路径
4. 解码时, 遇到压缩路径直接生成多个 tokens, 作为一次 prefill 处理

**效果**:
- 延迟降低: 最多 2x
- 吞吐提升: 最多 2.5x
- Tokenization 边界处理: 约 4% 开销 (re-tokenization)

**与 RadixAttention 的协同**:
- 被压缩 FSM 终止的请求, 其 KV cache 自动被 RadixAttention 保留
- 后续类似请求可以直接复用这些 KV cache
- 实现了 "free" 的 FSM 上下文共享

---

## 5. FlashInfer 集成与 Attention Backend

### 5.1 Backend 矩阵

SGLang 支持丰富的 Attention Backend, 按硬件架构自动选择:

| Backend | 适用场景 | Page Size | 特性 |
|---------|----------|-----------|------|
| **FlashInfer** | 非Hopper GPU (A100, A40) | 1 | 默认, 最广泛支持 |
| **FlashAttention 3** | Hopper (H100, H200, H20) | 1 | Hopper 默认 |
| **FlashMLA** | DeepSeek MLA (SM90) | 64 | FP8 KV cache |
| **TRTLLM MLA** | DeepSeek MLA (Blackwell) | 32/64 | Blackwell 优化 |
| **Triton** | Fallback | 1 | 兼容性好 |
| **FlashAttention 4** | 最新架构 | 1 | 新一代 |
| **Cutlass MLA** | SM100 | 1 | Blackwell MLA |
| **dual_chunk_flash_attn** | 长上下文 | 1 | 双 chunk |
| **FlexAttention** | 实验性 | 1 | PyTorch native |
| **Ascend** | 华为 NPU | 1 | 国产芯片 |
| **Intel XPU** | Intel GPU | 1 | Intel 加速器 |

### 5.2 Backend 接口

所有 Attention Backend 实现统一接口:

**非 CUDA Graph 路径**:
- `forward_extend`: Prefill 阶段 (完整 attention)
- `forward_decode`: Decode 阶段 (增量 attention)
- `init_forward_metadata`: 初始化 forward 元数据

**CUDA Graph 路径** (decode 阶段性能优化):
- `init_cuda_graph_state`: 初始化 CUDA graph 状态
- `init_forward_metadata_capture_cuda_graph`: 捕获阶段元数据
- `init_forward_metadata_replay_cuda_graph`: 重放阶段元数据

### 5.3 Hybrid Attention

实验性特性, 允许 Prefill 和 Decode 使用不同的 Backend:

```python
# python/sglang/srt/layers/attention/hybrid_attn_backend.py
# 示例: Prefill 用 FA3, Decode 用 FlashInfer
```

### 5.4 MLA (Multi-head Latent Attention) 支持

SGLang 对 DeepSeek MLA 有专门的优化:

1. **FlashMLA Backend**: page_size=64, 专用 decode kernel
2. **Weight Absorption**: 将 MLA 的投影矩阵吸收到 attention 计算中
3. **FP8 Batched MatMul**: 利用 FP8 加速 MLA 矩阵运算
4. **FP8 KV Cache**: KV cache 使用 FP8 存储, 节省显存
5. **v0.3 优化**: 3x-7x 吞吐提升 (vs vLLM baseline)

### 5.5 与 vLLM Attention Backend 对比

| 维度 | SGLang | vLLM V1 |
|------|--------|---------|
| **Backend 数量** | 15+ | 20+ |
| **默认 Backend** | FlashInfer (A100) / FA3 (H100) | FlashInfer (A100) / FA3 (H100) |
| **MLA Backend** | FlashMLA, TRTLLM MLA, Cutlass MLA | FlashMLA, FlashInfer MLA, Triton MLA, Cutlass MLA |
| **Page Size** | 1 (FlashInfer) / 64 (MLA) | 16 (默认) / 64 (MLA) |

---

## 6. 性能基准测试

### 6.1 论文原始结果 (arXiv:2312.07104)

SGLang 论文报告了全面的基准测试:

**对比系统**: vLLM, Guidance, LMSys FastChat
**测试模型**: Llama-7B, Mixtral-8x7B
**测试任务**: MMLU, HellaSwag, ReAct Agent, Tree-of-Thought, JSON Decode, Chat, DSPy RAG, LLaVA Bench

**关键结果**:
- **最高 6.4x 吞吐提升** (vs Guidance + vLLM)
- **最高 5x 吞吐提升** (vs vLLM 在 prefix-heavy 任务上)
- RadixAttention 在 prefix 复用场景优势最明显
- Compressed FSM 在结构化输出任务上带来额外 2-2.5x 提升

### 6.2 v0.3 Benchmark (2024-09)

**DeepSeek MLA 吞吐**:
- DeepSeekCoder-V2-Lite (BF16, H100, TP=1): SGLang 3-7x vs vLLM
- DeepSeekCoder-V2 (BF16, H100, TP=8): SGLang 3-7x vs vLLM
- DeepSeekCoder-V2 (FP8, H100, TP=8): SGLang 3-7x vs vLLM
- 优化技术: weight absorption, grouped decode kernels, FP8 batched MatMul, FP8 KV cache

**torch.compile 延迟**:
- Meta-Llama-3-8B, batch=1: SGLang + torch.compile 快于 gpt-fast
- 小 batch (1-32): 最多 1.5x 延迟降低
- 技术: torch.compile fuse linear/norm/activation + FlashInfer attention

**LLaVA-OneVision**:
- 最多 4.5x 加速 vs HuggingFace/transformers

### 6.3 v0.4 Benchmark (2025-02)

v0.4 主要特性: **零开销 CPU 调度器**

- 调度开销从关键路径完全移除
- GPU 和 CPU 完全重叠执行
- 整体吞吐提升显著 (具体数字因 blog.sgl.ai 网络问题未能获取)

### 6.4 Nightly Benchmark 基础设施

SGLang 维护了 **Performance Dashboard** (docs/performance_dashboard/):
- 从 GitHub Actions 获取 nightly 测试结果
- 支持多模型对比 (DeepSeek-V3.1, Llama 等)
- 指标: throughput (input/output/overall), latency, TTFT
- 示例: DeepSeek-V3.1 on 8xH200, TP8+MTP, batch=1, output throughput ~232 tok/s

### 6.5 与 vLLM / TensorRT-LLM 对比总结

基于论文和各版本 blog 的综合数据:

| 场景 | SGLang 优势 | 倍率 |
|------|------------|------|
| Prefix-heavy 任务 (few-shot, multi-turn) | RadixAttention | 3-5x |
| 结构化输出 (JSON, regex) | Compressed FSM | 2-2.5x |
| DeepSeek MLA 模型 | 专用 MLA 优化 | 3-7x |
| 小 batch 延迟 | torch.compile | 1.5x |
| 综合吞吐 | 零开销调度 + 整体优化 | 1.2-2x |

注意: 具体数字高度依赖 workload 和硬件配置. TensorRT-LLM 在纯吞吐上可能仍有优势, 但灵活性不如 SGLang.

---

## 7. 源码结构分析

### 7.1 顶层目录

```
sglang/
├── python/sglang/
│   ├── srt/          # SGLang Runtime (核心)
│   ├── lang/         # 前端 DSL
│   ├── benchmark/    # 基准测试工具
│   ├── jit_kernel/   # JIT kernel 编译
│   └── test/         # 测试
├── docs/             # 文档
├── examples/         # 示例
└── scripts/          # 脚本
```

### 7.2 SRT Runtime 目录 (`python/sglang/srt/`)

这是 SGLang 的核心运行时, 包含 30+ 子目录:

```
srt/
├── managers/              # 调度和管理
│   ├── scheduler.py       # 核心调度器 (~3800 行)
│   ├── schedule_batch.py  # 批次管理
│   ├── schedule_policy.py # 调度策略 (FCFS/PRIORITY)
│   ├── tp_worker.py       # Tensor Parallel Worker
│   └── scheduler_components/  # 调度器子组件
│       ├── request_receiver.py
│       ├── batch_result_processor.py
│       ├── metrics_reporter.py
│       ├── output_streamer.py
│       ├── load_inquirer.py
│       ├── flush_wrapper.py
│       ├── invariant_checker.py
│       ├── dp_attn.py
│       ├── idle_sleeper.py
│       ├── profiler_manager.py
│       └── weight_updater.py
│
├── layers/                # 模型层实现
│   ├── attention/         # Attention 实现
│   │   ├── flashinfer_backend.py
│   │   ├── flashattention_backend.py
│   │   ├── triton_backend.py
│   │   ├── hybrid_attn_backend.py
│   │   └── ...            # 其他 backend
│   ├── moe/               # Mixture of Experts
│   ├── quantization/      # 量化 (FP8, FP4, AWQ, GPTQ)
│   └── radix_attention.py # RadixAttention 核心
│
├── mem_cache/             # KV Cache 管理
│   ├── kv_cache_builder.py
│   ├── radix_cache.py     # 基数树缓存
│   └── ...                # 其他缓存实现
│
├── model_executor/        # 模型执行器
│   └── forward_batch_info.py  # Forward mode 定义
│
├── models/                # 模型实现 (LLaMA, DeepSeek, etc.)
│
├── sampling/              # 采样策略
│   └── sampling_batch_info.py
│
├── constrained/           # 结构化输出约束
│   └── grammar_manager.py # Grammar/regex 管理
│
├── speculative/           # 推测解码
│   └── spec_info.py       # SpeculativeAlgorithm
│
├── disaggregation/        # P/D 分离
│   ├── prefill.py         # Prefill 端
│   ├── decode.py          # Decode 端
│   └── decode_kvcache_offload_manager.py
│
├── distributed/           # 分布式通信
│   └── parallel_state.py
│
├── lora/                  # LoRA 支持
│   ├── lora_drainer.py
│   └── lora_overlap_loader.py
│
├── configs/               # 模型配置
├── entrypoints/           # 入口点 (HTTP server, etc.)
├── platforms/             # 硬件平台适配
├── multimodal/            # 多模态支持
├── tokenizer/             # Tokenizer 管理
├── observability/         # 可观测性 (metrics, tracing)
├── session/               # 会话管理
├── compilation/           # torch.compile 集成
├── checkpoint_engine/     # 检查点管理
├── connector/             # 外部连接器
├── hardware_backend/      # 硬件抽象层
├── parser/                # 输入解析
├── plugins/               # 插件系统
├── utils/                 # 工具函数
└── weight_sync/           # 权重同步
```

### 7.3 前端 DSL 目录 (`python/sglang/lang/`)

```
lang/
├── interpreter.py    # DSL 解释器
├── compiler.py       # DSL 编译器 (graph executor)
└── ...
```

### 7.4 关键源码文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `scheduler.py` | ~3800 | 核心调度器, Mixin 模式 |
| `schedule_batch.py` | ~数百 | 批次管理 |
| `schedule_policy.py` | ~数百 | FCFS/PRIORITY 策略 |
| `kv_cache_builder.py` | ~数百 | KV cache 构建 |
| `flashinfer_backend.py` | ~数百 | FlashInfer 集成 |
| `grammar_manager.py` | ~数百 | 结构化输出约束 |
| `spec_info.py` | ~数十 | 推测解码算法 |
| `parallel_state.py` | ~数百 | 分布式并行状态 |

### 7.5 与 vLLM V1 源码对比

| 模块 | SGLang | vLLM V1 |
|------|--------|---------|
| **调度器** | `scheduler.py` (~3800 行) | `scheduler.py` (~2400 行) |
| **执行器** | `tp_worker.py` | Executor→Worker→ModelRunner→Model 四层 |
| **KV Cache** | `mem_cache/` + radix tree | `block_manager.py` + hash table |
| **Attention** | `layers/attention/` | `layers/attention/` (20+ backend) |
| **约束解码** | `constrained/` + compressed FSM | xGrammar / Outlines |
| **分布式** | `distributed/parallel_state.py` | `distributed/` |

---

## 关键参数速查

```bash
# 启动 SGLang server
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B \
    --tp 1 \                              # Tensor Parallel
    --mem-fraction-static 0.88 \          # KV cache 内存比例
    --chunked-prefill-size 4096 \         # Chunked prefill 大小
    --enable-torch-compile \              # 启用 torch.compile
    --disable-radix-cache \               # 关闭 RadixAttention
    --enable-mla \                        # 启用 MLA 优化
    --enable-deterministic-inference \    # 确定性推理
    --schedule-policy fcfs \              # 调度策略
    --enable-lora \                       # 启用 LoRA
    --speculative-algorithm EAGLE \       # 推测解码算法
```

---

## 学习要点总结

1. **RadixAttention 是 SGLang 最核心的创新**: 基数树实现细粒度 KV cache 共享, 比传统 hash prefix caching 更高效
2. **Compressed FSM 是结构化输出的关键优化**: 跳跃解码减少 decode 步数, 与 RadixAttention 协同工作
3. **Mixin 模式的调度器架构**: 功能解耦, 支持多种分布式和部署模式
4. **零开销 CPU 调度器 (v0.4)**: CPU/GPU 完全重叠, 消除调度瓶颈
5. **丰富的 Attention Backend**: 硬件自适应选择, MLA 有专用优化
6. **DSL 前端**: 嵌入式编程模型, 支持 fork/choices/regex 等高级原语

---

## 参考资料

- [SGLang 论文](https://arxiv.org/abs/2312.07104) - arXiv:2312.07104, Dec 2023
- [LMSYS Blog: RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/) - RadixAttention 详细解释
- [LMSYS Blog: Compressed FSM](https://lmsys.org/blog/2024-02-05-sglang/) - 结构化输出优化
- [LMSYS Blog: v0.3 Release](https://lmsys.org/blog/2024-09-04-sglang-v0-3/) - 7x MLA, 1.5x torch.compile
- [SGLang GitHub](https://github.com/sgl-project/sglang) - 源码仓库 (28k stars)
- [SGLang Attention Backend Docs](https://docs.sglang.ai/advanced_features/attention_backend.html) - Backend 文档
- [SGLang FAQ](https://docs.sglang.ai/faq.html) - 常见问题
