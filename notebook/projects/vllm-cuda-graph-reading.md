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

## 参考资料

- 源码: `vllm/v1/cudagraph_dispatcher.py`, `vllm/compilation/cuda_graph.py`
- 相关: [CUDA Graphs 基础](cuda-graphs.md), [V1 Executor](vllm-v1-executor-reading.md)
- `tools/cuda_graph_demo.py` — CUDA Graph 实测数据
