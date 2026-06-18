# vLLM V1 CUDA Graph 源码阅读

> CUDA Graph 消除重复 kernel launch 开销，是 vLLM Decode 加速的关键

## 1. 核心组件

```
CUDAGraphWrapper (vllm/compilation/cuda_graph.py)
  └── 包装函数，添加图捕获/回放能力

CudagraphDispatcher (vllm/v1/cudagraph_dispatcher.py)
  └── 中央控制器，维护 PIECEWISE 和 FULL 两种模式图
  └── 根据输入 batch 动态选择运行时模式

ModelCudaGraphManager (vllm/v1/worker/gpu/cudagraph_utils.py)
  └── 管理模型特定的图捕获和隐藏状态
```

## 2. CUDA Graph 模式

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| `NONE` | 关闭 | 调试、不稳定后端 |
| `PIECEWISE` | 分段图 | 最灵活，不兼容操作保持 eager |
| `FULL` | 完整图 | 全部操作在图中，适合小模型 |
| `FULL_DECODE_ONLY` | 仅 Decode 完整图 | 只优化 decode 阶段 |
| `FULL_AND_PIECEWISE` | 混合 (默认) | Decode 用完整图，其他用分段图 |

## 3. 捕获 (Capture) 流程

### 捕获时机

```python
for num_tokens in capture_sizes:
    if separate_decode_routine:
        desc = BatchExecutionDescriptor(
            cg_mode=decode_mode,
            num_tokens=num_tokens,
            num_reqs=num_tokens // decode_query_len,
            uniform_token_count=decode_query_len,
        )
```

### 捕获顺序

1. PIECEWISE 先捕获 (激活更大)
2. FULL 后捕获 (激活较小)
3. 按从大到小排序，确保内存复用

### 内存管理

- **图池 (Graph Pool)**: `self.pool = current_platform.get_global_graph_pool()` — 全局共享，避免频繁分配/释放
- **静态缓冲区**: 所有输入/输出在初始化时分配，确保捕获和重放时地址不变
- **弱引用**: 避免输出张量内存泄漏
- **同步 offloader**: 确保 H2D/D2H 操作完成后再捕获

## 4. 回放 (Replay) 流程

### 动态分发

```python
def dispatch(self, num_reqs, num_tokens, uniform_token_count):
    """查找匹配的图描述符，优先级: FULL > PIECEWISE > NONE"""
    for desc in self._candidates[num_tokens]:
        if _is_compatible(desc, num_reqs, num_tokens, uniform_token_count):
            return desc
    return BatchExecutionDescriptor(CUDAGraphMode.NONE, ...)
```

### FULL 模式回放

- 直接调用 `graph.replay()`
- 无需传递输入张量 (已在静态缓冲区)
- 确保内存地址与捕获时一致

### PIECEWISE 模式回放

- 使用 `torch.compile` 编译的图
- 或 Breakable CUDA Graph (打断→eager→继续)
- 部分操作保持 eager 执行

## 5. 与 Attention Backend 交互

```python
class AttentionCGSupport(enum.Enum):
    ALWAYS = 3        # 始终支持 (FA3)
    UNIFORM_BATCH = 2  # 仅均匀批次 (FA2)
    UNIFORM_SINGLE_TOKEN_DECODE = 1  # 仅单 token (FlashInfer)
    NEVER = 0         # 不支持
```

### 自动降级

- FlashInfer: 只能用 `FULL_DECODE_ONLY`
- Cascade Attention: 不支持，强制 PIECEWISE
- MLA: FlashMLA 支持，CUTLASS MLA 仅单 token
- Mamba: 仅支持单 token decode

## 6. Breakable CUDA Graph

替代传统分段编译的创新方案:

```python
@eager_break_during_capture
def some_operation():
    # 在图捕获期间打断，运行 eager 代码后继续
    ...
```

- 避免预编译，支持更多场景
- 标记需要 eager 执行的操作
- 打断→运行→继续捕获

## 7. 限制与注意事项

1. **动态形状**: 不同 batch size 需要不同图，序列长度变化需要适配
2. **内存开销**: 每个图实例需要额外静态内存，大模型占用显著增加
3. **功能限制**: KV offload 不完全兼容，编译优化受限，调试困难
4. **Attention 兼容**: 不是所有 backend 都支持 CUDA Graph

## 8. 典型配置

```python
compilation_config = {
    "mode": 3,  # VLLM_COMPILE
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
}
```

### 捕获大小选择

```bash
# 小模型 (7B): 捕获 [1, 2, 4, 8, 16, 32, 64, 128]
# 大模型 (70B): 捕获 [1, 2, 4, 8, 16, 32]
# 超大模型 (405B): 只捕获 [1, 2, 4, 8]
```

## 9. 性能数据

| 指标 | 无 CUDA Graph | 有 CUDA Graph |
|------|-------------|-------------|
| Kernel launch (128 kernels) | 0.64 ms | 0.006 ms (100x↓) |
| Decode step (7B, A100) | 7.03 ms | 6.40 ms (9%↓) |
| P90 延迟 | 基线 | -30~50% |
| 吞吐 | 基线 | +1.5~3x |

**注意**: 实际提升取决于 kernel launch 占比。Decode 场景 kernel 多但小，提升显著；Prefill 场景 kernel 少但大，提升有限。

## 10. 关键洞察

1. **FULL_AND_PIECEWISE 是默认模式**: Decode 用 FULL (高收益)，其他用 PIECEWISE (灵活)
2. **捕获大小需权衡**: 更多捕获点 → 更好性能，但更多内存
3. **Breakable Graph 是创新**: 不预编译也能混合 eager 和 graph 执行
4. **Attention 兼容性决定模式**: FlashInfer 限制更多，FA3 最灵活
5. **内存池是关键**: 全局图池避免 OOM 和碎片化
6. **Decode 是最大受益者**: 每步 shape 不变，kernel 多但小

## 11. PyTorch CUDA Graph 深层机制 (C++ Level)

```
★ ★ ★ CUDA Graph C++层5步生命周期:

1. capture_begin(pool, capture_mode)
   → cudaStreamBeginCapture → 进入stream capture
   → CUDA caching allocator → private memory pool(MempoolId_t)
   → RNG generators → checkpoint state

2. User runs model
   → All CUDA work recorded into graph → not executed
   → Memory allocations → private pool → zero-allocation replay!

3. capture_end()
   → cudaStreamEndCapture → cudaGraph_t
   → Allocator pool redirect ends
   → If keep_graph=False → instantiate immediately → destroy cudaGraph_t

4. instantiate()
   → cudaGraphInstantiateWithFlags(AutoFreeOnLaunch | UseNodePriority)
   → ★ AutoFreeOnLaunch → handle cudaMallocAsync allocator scenarios
   → ★ UseNodePriority → priority-aware scheduling within graph

5. replay()
   → cudaGraphLaunch(graph_exec_, getCurrentCUDAStream())
   → ★ Can replay on ANY stream (not just capture stream)
   → RNG generators → offset increment → prologue before launch

★ ★ ★ No cudaGraphExecUpdate in PyTorch!
  → Always destroy + recapture → no incremental update
  → ★ Makes recompilation expensive → vLLM captures multiple sizes at startup
```

### torch.cuda.make_graphed_callables()

```
★ ★ ★ PyTorch primary user-facing CUDA graph API:

1. Creates static_input_surface → flattened args + parameters → permanent addresses
2. Warmup: num_warmup_iters(=3) fwd+bwd on separate stream → cuDNN autotuning + lazy init
3. Forward capture: torch.cuda.graph(fwd_graph, pool=mempool)
4. Backward capture (in reverse order) → memory pool consistency
5. Creates Graphed autograd.Function:
   Forward: static_input_surface[i].copy_(inputs[i]) → fwd_graph.replay()
   Backward: static_grad_outputs.copy_(grads) → bwd_graph.replay()

★ ★ Key: static_input_surface includes parameters → assumed unchanged → not copied!
★ ★ Only user args copied via copy_() → minimize transfer overhead
```

### CUDAGraphTreeManager in Inductor (120KB!)

```
★ ★ ★ Most sophisticated CUDA graph system → torch.compile reduce-overhead!

Source: torch/_inductor/cudagraph_trees.py (120KB!)

Tree structure (not just sequences):
  → Previously: graphed callables replay in strict order A, B
  → Now: arbitrary trees → after A, can record/replay different B'
  → Memory sharing: max(mem(A,B), mem(A,B')) → pool reuse!

CUDAGraphNode:
  → parent + children → tree structure
  → first recording → reflect all live tensors in pool
  → checkpoint allocator state → subsequent recordings resume accurately

★ ★ Pool lifecycle:
  → TreeManagerContainer → tracks live functions + storages
  → All references die → tree manager + private pool deallocated
  → Prevents pool leaking memory indefinitely!

★ ★ Warmup in pool (not default pool):
  → Warmup runs happen inside cudagraph memory pool
  → ★ Critical: warmup in default pool + recording in private = double allocation!
  → clear_cublass_cache() → clear workspaces before/after warmup

★ ★ static_input_idxs:
  → Some inputs (parameters) = "static" → same address → not copied
  → Other inputs = "dynamic" → static_input() allocation + copy_()
  → cudagraphify tells which inputs are static
```

### torch.cond + CUDA Graphs (PyTorch 2.12)

```
★ ★ ★ Landmark feature → CUDA 12.4+ → data-dependent control flow IN graph!

1. ControlFlowOpWarmupDispatchMode → warmup both branches
2. CUDAGraphCaptureControlFlowOpDispatchMode → during capture → dispatch to if_else_node
3. if_else_node(pred, true_fn, false_fn, operands):
   → ★ Two conditional if-nodes (CUDA 12.8 lacks native if-else!)
   → First evaluates pred → second evaluates !pred
   → Else branch output → copy_() into if branch output buffer → fixed addresses!

C++ begin_capture_to_if_node:
   → cudaGraphConditionalHandleCreate → handle
   → cudaGraphNodeTypeConditional node → child stream capture inside body

★ RTX 4090 (SM 8.9): cudaGraphConditionalHandle needs CUDA driver 12.4+
   → SM无关 → 运行时/驱动层feature → SM 8.9应该支持!
   → ★ 但目前PyTorch实验性 → 需验证!
```

### GRPO cudagraph memory regression (Megatron)

```
★ ★ Megatron PR #5280 → GRPO CUDA graph memory regression:

Regression: exponential distribution + mixed-prefill grid → +13.6% peak memory!
  → Exponential sizing: [1,2,4,8,...] vs linear [1,2,3,...]
  → More graphs at low end → each allocates pool memory → compounding overhead
  → Mixed-prefill grid: 16 extra batch size graphs → each pool allocation!

★ Fix: --inference-dynamic-batching-cuda-graph-sizing-distribution: linear
  → ★ ★ GRPO → linear sizing → 省内存!
  → ★ ★ ★ RTX 4090: capture sizes少 → [1,2,4,8] → 内存可控!

★ ★ vLLM sleep/wake + CUDA graphs:
  → Sleep: offload weights → pool memory stays on GPU!
  → Wake: restore weights → same addresses → graph replay still works
  → ★ Graph pool persists across sleep/wake → replay valid after wake!
```

## 12. RTX 4090 CUDA Graph Summary

```
★ ★ ★ RTX 4090 CUDA Graph capability:

Works (SM 8.9 Ada):
  ✓ Basic CUDA graphs → cudaStreamCaptureBegin/End/Launch → 全支持
  ✓ Private memory pools → MempoolId_t → 不依赖SM
  ✓ torch.cuda.CUDAGraph → full support
  ✓ torch.cuda.make_graphed_callables() → full support
  ✓ torch.accelerator.Graph (v2.12) → CUDA backend✓
  ✓ vLLM FULL CUDA graphs → FA2(UNIFORM_BATCH) on SM 8.9✓
  ✓ vLLM PIECEWISE → FlashInfer(UNIFORM_SINGLE_TOKEN_DECODE)✓
  ✓ Breakable CUDA graphs → SM无关 → stream capture technique✓
  ✓ torch.cond + CUDA graphs → CUDA driver 12.4+ → RTX 4090✓(需验证)

Not available (SM 8.9 vs SM 9.0):
  ✗ NVLS → SM 9.0 (Hopper) → multimem_all_gatherv_3tensor ✗
  ✗ FlashAttention v3 → SM 9.0+ → FA2 only on RTX 4090
  ✗ TMA → SM 9.0+ → DeepEP hybrid ✗
  ✗ FP8 E5M2 → SM 8.9不支持 → E4M3 only for inference

Memory estimation (7B INT4 on RTX 4090 24GB):
  INT4 weights ~3.5GB + KV cache ~5GB + graph pool ~2GB + buffers ~10MB + overhead ~0.5GB
  = ~11GB total → ~13GB headroom → ✓ plenty of room!

★ ★ Performance gains on RTX 4090:
  Decode: ~10-15% throughput improvement (eliminate CPU launch overhead)
  Training: ~5-10% speedup with reduce-overhead mode (LoRA+compile)
  Warmup: ~30-60s for all batch sizes → amortized over session
```

## 参考资料

- 源码: `vllm/v1/cudagraph_dispatcher.py`, `vllm/compilation/cuda_graph.py`, `vllm/compilation/breakable_cudagraph.py`
- PyTorch: `aten/src/ATen/cuda/CUDAGraph.cpp`, `torch/cuda/graphs.py`, `torch/_inductor/cudagraph_trees.py`
- PyTorch conditional: `torch/_higher_order_ops/cudagraph_conditional_nodes.py`
- vLLM: `vllm/v1/worker/gpu/cudagraph_utils.py`, `vllm/v1/utils.py` (CpuGpuBuffer)
- Megatron: PR #5242 + PR #5280 (GRPO cudagraph regression)
- 相关: [CUDA Graphs 基础](cuda-graphs.md), [V1 Executor](vllm-v1-executor-reading.md)
- `tools/cuda_graph_demo.py` — CUDA Graph 实测数据

---

## 13. CUDA Graph Systematic Fragility Pattern (June 2026 Update)

```
★★★★★★★★★ CUDA graph replay bugs across ALL frameworks → SYSTEMATIC pattern:

1. vLLM #45972 MERGED June 18: REVERT of DSV4 cudagraph optimization
   → #45309 cudagraph + DeepSeek-V4 → GARBAGE OUTPUT → correctness regression!
   → REVERTED within days → same pattern: graph replay + complex model = incorrect results

2. SGLang #27097: multi-LoRA determinism bug → 4 factors
   → SGMV dynamic routing + CUDA graph replay + KV stale state + flashinfer batch-dep
   → #28499 csgmv MERGED June 17 → partial fix (Factor 2 only)

3. SGLang #28569: EAGLE3 CUDA graph replay illegal memory access on gpt-oss-120b
   → --disable-cuda-graph avoids crash → confirms systematic fragility

4. SGLang #28582: LoRA endpoint RCE (CVSS 9.8) → not graph-related but security fragility

5. SGLang #28588: image decompression bomb → 2nd security issue same week

6. vLLM #39096: SM89 batch invariance → Inductor fuses under cudagraph → batch-dependent
   → enforce_eager=True skips cudagraph → safe on RTX 4090

★★★★★★★★★ Root cause: CUDA graph replay assumes STATIC execution path
  → Any DYNAMIC routing (MoE expert selection, LoRA adapter selection, spec decode draft)
  → Under graph replay → PRE-CAPTURED path replayed → NOT current dynamic decision
  → → WRONG expert/adapter/draft → incorrect results, memory corruption, NaN

★★★★★★★★★ vLLM #45972 DSV4 REVERT — source-level mechanism:
  → #45309 removed @eager_break_during_capture from attention_impl
  → Used runtime BreakableCUDAGraphCapture.is_active() check instead
  → During CAPTURE: ran wq_b_kv_insert + compressor in 2-way parallel, indexer sequentially
  → BUT all inside stream capture context → BECOMES PART OF recorded CUDA graph!
  → During REPLAY: entire captured graph replayed with STATIC data from capture-time buffers
  → → NOT with live per-request metadata → garbage output like "the the the the..."
  → ★★★★★★★★ @eager_break_during_capture is the CORRECT separation boundary:
    → Static GEMMs (weight matmuls, norms) → CAN be captured → speed benefit
    → Dynamic routing (attention metadata, MoE expert selection, indexer) → MUST run eagerly
  → ★★★★★★★★ Universal rule: ANY operation whose behavior depends on per-request metadata
    → MUST run eagerly, NEVER inside captured CUDA graph

★★★★★★★★★ RTX 4090 recommendation: enforce_eager=True for training + inference
  → 10-15% throughput sacrifice → but CORRECTNESS guaranteed
  → BudgetRefiner SLO compensates throughput loss with better scheduling
```
