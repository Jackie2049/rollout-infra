# LLM Serving Framework 比较: vLLM vs SGLang vs TensorRT-LLM

> 2026-06-07 | 基于源码深度阅读(vLLM V1 + SGLang) + 公开benchmark数据

## 一、核心架构对比

| | **vLLM V1** | **SGLang** | **TensorRT-LLM** |
|---|------------|-----------|-----------------|
| **语言** | Python + C++/Triton kernels | Python + C++/Triton kernels | C++ runtime + Python builder |
| **核心引擎** | EngineCoreProc(ZMQ IPC双进程) | 事件循环(Normal/Overlap双模式) | C++ runtime(低延迟调度) |
| **调度** | unified token budget(无prefill/decode分离) | Mixin架构(6个Mixin) + merge-based批处理 | In-flight batching(迭代级注入) |
| **KV Cache** | PagedAttention(block对齐, BlockHashToBlockMap) | RadixAttention(无block对齐, RadixTree节点分裂) | Block manager(类似paging) + KV cache compression(2025) |
| **注意力后端** | 20+ backend(FlashInfer/FlashAttn/Triton/Cutlass/MLA等) | FlashInfer + Triton(fallback) | TensorRT plugins(高度优化的C++ kernels) |
| **部署模型** | 独立API server / Ray-based | 独立HTTP server | Triton Inference Server + C++ API |

### 架构关键差异

**vLLM V1**: 双进程架构(AsyncLLM API进程 + EngineCoreProc 核心进程) → API处理与推理分离, 三线程ZMQ IPC → 请求/输出/主循环并行. unified token budget让同一step可同时处理prefill和decode → RUNNING优先不被WAITING阻塞.

**SGLang**: Mixin架构(6个Mixin组合18个子组件) → 模块化可扩展. merge-based批处理不替换running_batch → 新请求merge入现有batch. Normal/Overlap双事件循环 → Overlap模式分离token处理与batch更新. RadixAttention用RadixTree节点分裂消除block对齐 → 更灵活的prefix caching.

**TensorRT-LLM**: 纯C++ runtime → 调度开销极低(微秒级). In-flight batching在每次迭代边界注入新请求 → 无需重启batch. 模型需要预编译(trtllm-build) → 编译时优化所有kernel → 运行时无Python开销. 紧密集成Triton Inference Server → 生产部署标准化.

## 二、调度策略深度对比

### vLLM V1 Scheduler

```
schedule() → RUNNING优先 → WAITING次之(仅在无preempt时)
→ preempt: FCFS(最新) / PRIORITY(最低优先级)
→ chunked prefill threshold限制每步新prefill token数
→ prefix cache: get_computed_blocks → 减少新token数
→ KV connector: WAITING_FOR_REMOTE_KVS → 等远程KV
→ cascade attention: get_num_common_prefix_blocks → prefix≥256+requests≥8触发
```

**关键**: RUNNING请求优先分配token_budget → decode不被prefill阻塞. WAITING调度失败不preempt → prefill不抢decode的KV.

### SGLang Scheduler

```
6个Mixin: ScheduleMixin + CacheMixin + RouterMixin + ...
→ merge-based: 新请求merge入running_batch(不替换)
→ 自适应new_token_ratio → 根据负载动态调整
→ batch_is_full → 跳过优化
→ warp barrier → 防竞态
→ RadixAttention: 缓存感知调度(LPM) → prefix复用优先调度
```

**关键**: merge-based更平滑(无preemption切换), RadixAttention缓存感知调度 → prefix复用请求优先分配到有缓存的路由.

### TensorRT-LLM In-flight Batching

```
每次迭代边界:
  1. 检查新请求队列
  2. 将新请求注入running batch (无需重新构建batch)
  3. 移除已完成请求
  4. 继续下一iteration
```

**关键**: C++调度开销极低(~微秒级 vs Python的毫秒级), 无Python GIL限制 → 高并发下吞吐更高.

## 三、KV Cache 管理

| | **vLLM V1** | **SGLang** | **TensorRT-LLM** |
|---|------------|-----------|-----------------|
| **管理方式** | BlockPool(固定block) + FreeKVCacheBlockQueue(O(1) LRU) | RadixTree(节点分裂, 无block对齐) | Block manager + KV compression |
| **驱逐策略** | LRU(tail hash先驱逐/root hash后驱逐) | 7种策略(evict/keep/recompute等) | LRU + compression |
| **Prefix caching** | BlockHashToBlockMap(1:N) → hash chain递推 | RadixTree节点分裂 → 无block对齐限制 | Basic(比vLLM/SGLang弱) |
| **P/D分离** | NIXL/FlexKV/LMCache/Mooncake connector | Mooncake connector | Disagg support(2025新增) |

**核心差异**: vLLM需要block对齐(如16 token/block) → prefix必须对齐到block边界才能缓存. SGLang RadixTree无此限制 → 更灵活的prefix共享(任何token位置都能缓存/分裂).

**实测对比**:
- vLLM prefix caching: 5轮对话 ~66-75%缓存节省
- SGLang RadixAttention: 5轮对话 ~75-85%缓存节省 (无block对齐浪费)
- SGLang在agentic/multi-turn场景3-6x加速 vs vLLM/TRT-LLM

## 四、量化支持

| | **vLLM V1** | **SGLang** | **TensorRT-LLM** |
|---|------------|-----------|-----------------|
| **FP8** | E4M3(Hopper), compressed-tensors | E4M3(Hopper), compressed-tensors | **E4M3+E5M2(Hopper原生优化)** |
| **INT8** | per-channel, per-token | per-channel, per-token | **W8A8(weight+activation)** |
| **INT4** | AWQ, GPTQ, Marlin kernel | AWQ, GPTQ | **AWQ, GPTQ + 自定义C++ kernel** |
| **KV Cache量化** | FP8 KV Cache(50%内存) | FP8 KV Cache | **FP8 + INT8 KV Cache + compression** |
| **量化方法总数** | 30+ methods | ~10 methods | ~15 methods (但每个高度优化) |
| **fused kernel** | Marlin/compressed-tensors/cuBLAS FP8 | FlashInfer量化 | **TRT plugin(C++高度融合)** |

**关键差异**: vLLM量化生态最全(30+), 但很多是Python-level实现 → 某些量化反比FP16慢(实测:INT8 0.17-0.37x). TRT-LLM每个量化方法都有高度优化的C++ kernel → 所有量化都能实际加速. SGLang量化方法少但每个都有fused kernel支持.

## 五、Speculative Decoding

| | **vLLM V1** | **SGLang** | **TensorRT-LLM** |
|---|------------|-----------|-----------------|
| **Proposer** | 8+ (Eagle推荐, N-gram零开销, MLP/Stable等) | Eagle + Medusa | Medusa + Eagle(2025) |
| **Scorer** | Rejection Sampler(6 Triton kernel) | Rejection Sampler | Native C++ scorer |
| **Draft model** | 支持外部draft model | 支持外部draft model | 支持外部draft model |
| **验证方式** | 拒绝采样(概率比) + Triton kernel | 拒绝采样 | 拒绝采样(C++实现) |

**实测**: Spec Decoding最优K=2-3(非5), 分布锐度是关键(sharp=10→5.1x). vLLM Triton rejection sampler最快(Eagle proposer + Triton scorer). TRT-LLM C++ scorer在极端高并发下可能更快.

## 六、性能实测对比

### Throughput (Llama-3.1-70B, 4×A100)

| 并发数 | vLLM | SGLang | TensorRT-LLM |
|--------|------|--------|--------------|
| 低(≤10) | 基准 | ~基准 | ~基准 |
| 中(50) | 基准 | ~基准 | **1.3x** |
| 高(200+) | 基准 | ~1.1x基准 | **1.8x** |

**关键**: TRT-LLM在高并发下优势显著(C++调度开销低). vLLM/SGLang在低中并发下接近.

### Prefix Caching (多轮对话)

| 场景 | vLLM | SGLang | TensorRT-LLM |
|------|------|--------|--------------|
| 5轮对话 | 3x | **6x** | 2.5x |
| Agentic循环 | 2x | **4-6x** | 2x |

**关键**: SGLang RadixAttention在prefix复用场景碾压.

### Latency (TTFT/ITL)

| Metric | vLLM | SGLang | TensorRT-LLM |
|--------|------|--------|--------------|
| TTFT | 好 | 好 | **最优** |
| ITL(单token) | 好 | 好 | **最优** |
| Jitter | 中(CUDA Graph减少) | 中 | **最低**(C++ runtime) |

### 实测经验总结 (RTX 4090)

基于我的大量RTX 4090实测:
- **vLLM 0.11.0**: OPT-125M 30K tok/s@B=512, V1引擎+torch.compile+chunked prefill
- **Decode瓶颈动态转移**: B=1权重87%, B≥32 KV+Attn86% → batch大小影响瓶颈
- **GQA Python-level比MHA更慢** (0.82-0.87x@B≥8) → 必须专用kernel
- **FlashAttn decode更慢** (0.67-0.84x) → 大GPU上FlashAttn主要价值是省内存不是加速

## 七、部署与易用性

| | **vLLM** | **SGLang** | **TensorRT-LLM** |
|---|---------|-----------|-----------------|
| **安装** | pip install vllm (一行) | pip install sglang (一行) | 需trtllm-build编译(复杂) |
| **模型加载** | 自动(HF格式) | 自动(HF格式) | 需预编译(build engine) |
| **API兼容** | OpenAI API server | OpenAI API server | Triton Inference Server |
| **配置复杂度** | 低(命令行参数) | 中(YAML+命令行) | 高(JSON engine config) |
| **调试** | Python → 容易 | Python → 容易 | C++ → 困难 |
| **自定义** | Python插件(logits processor等) | Python插件 | C++ plugin(高门槛) |
| **多模态** | 支持越来越多 | 支持 | 2025新增支持 |

## 八、MoE 模型支持

| | **vLLM V1** | **SGLang** | **TensorRT-LLM** |
|---|------------|-----------|-----------------|
| **MoE推理** | DPEngineCoreProc(Wave协调) | 基本支持 | **原生优化(Mixtral/DeepSeek-V3)** |
| **EP支持** | DBO microbatch + All-to-All重叠 | 基本EP | EP + All-to-All优化 |
| **负载均衡** | 依赖router aux_loss | 依赖router | **容量因子+专家并行优化** |

## 九、RL训练集成

| | **vLLM** | **SGLang** | **TensorRT-LLM** |
|---|---------|-----------|-----------------|
| **verl集成** | vLLM async rollout | SGLang async rollout | 无直接集成 |
| **权重同步** | sleep/wake + checkpoint engine | sleep/wake + checkpoint engine | 无原生支持 |
| **LoRA** | Punica Multi-LoRA | 支持 | 支持 |

**关键**: vLLM和SGLang都与verl深度集成 → RL训练rollout首选. TRT-LLM不适合RL训练场景.

## 十、适用场景总结

### 选择决策树

```
你的需求是什么?
│
├── 1. 通用LLM服务 (最低门槛, 最大灵活性)
│   → **vLLM** ✅ (pip install, OpenAI API, 30+量化, 最广硬件支持)
│
├── 2. 多轮对话/Agentic workflow (prefix复用关键)
│   → **SGLang** ✅ (RadixAttention 3-6x加速, 结构化输出最好)
│
├── 3. 延迟敏感的生产服务 (NVIDIA GPU, 高并发)
│   → **TensorRT-LLM** ✅ (最低TTFT/ITL, 最高吞吐, 最低jitter)
│
├── 4. RL训练rollout (verl集成)
│   → **vLLM/SGLang** ✅ (都支持, 看prefix复用需求选SGLang)
│
├── 5. 非NVIDIA硬件 (AMD/Intel)
│   → **vLLM** ✅ (唯一支持ROCm/XPU)
│
├── 6. MoE大规模推理 (DeepSeek-V3级)
│   → **TensorRT-LLM** ✅ (原生优化) 或 **vLLM V1** (DP支持)
│
└── 7. 快速原型/研究实验
    → **vLLM** ✅ (最易用, Python可扩展)
```

### 不推荐场景

- **TensorRT-LLM**: 不适合RL训练(无权重同步), 不适合非NVIDIA硬件, 不适合快速迭代研究
- **SGLang**: 量化方法少于vLLM, 硬件支持较窄
- **vLLM**: 高并发下吞吐不如TRT-LLM, prefix caching不如SGLang灵活

## 十一、趋势与展望

1. **性能差距缩小**: vLLM/SGLang在过去一年大幅缩小与TRT-LLM的吞吐差距
2. **vLLM生态领先**: 30+量化/20+注意力/8+Spec → 最大模型兼容性
3. **SGLang agentic场景领先**: RadixAttention + 结构化输出 → AI Agent时代首选
4. **TRT-LLM生产部署领先**: C++ runtime + Triton IS → NVIDIA生态最佳
5. **verl统一**: vLLM和SGLang都支持verl rollout → RL训练两个都能用
6. **未来方向**: P/D分离(vLLM NIXL), Disaggregated serving(SGLang), KV cache compression(TRT-LLM)