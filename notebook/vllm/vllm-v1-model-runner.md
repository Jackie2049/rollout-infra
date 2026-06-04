# vLLM V1 GPUModelRunner 源码分析

> 文件: `vllm/v1/worker/gpu_model_runner.py` (~7550行)
> 分析日期: 2026-06-04

## 定位

`GPUModelRunner` 是 vLLM V1 推理引擎的**执行核心**，位于 Worker → Model 之间，负责:

```
EngineCoreProc.step()
  └─> Executor.execute_model()
       └─> Worker.execute_model()
            └─> GPUModelRunner.execute_model()  ← 这里
                 ├─ _prepare_inputs()          # 准备 attn_metadata
                 ├─ _determine_batch_execution() # CUDA Graph 决策
                 ├─ UBatchWrapper / model.forward() # 实际执行
                 └─ _sample_tokens()           # 后处理
```

## 类架构

```python
class GPUModelRunner(
    LoRAModelRunnerMixin,           # LoRA 适配器管理
    KVConnectorModelRunnerMixin,    # K/V Transfer (P/D 分离)
    ECConnectorModelRunnerMixin,    # Encoder Cache 连接器
):
```

三个 Mixin 提供可插拔功能模块。

## execute_model() 流水线

```
1. Clear routed experts buffer
2. Handle KV preemptions (P/D disaggregation)
3. _update_states(scheduler_output)          # 更新持久 batch 状态
4. Handle EC transfer (encoder cache producer)
5. Early return if 0 tokens scheduled
6. _prepare_inputs()                         # 构建 attn_metadata, logits_indices
7. _determine_batch_execution_and_padding()  # CUDA Graph 模式选择
8. maybe_create_ubatch_slices()              # UBatch 切分决策
9. Prepare attention metadata (prefill/decode)
10. Model forward (via UBatchWrapper or direct):
    - UBatch: UBatchWrapper(model)(...)      # 通信-计算重叠
    - Non-ubatch: model.forward(...)
11. _sample_tokens()                         # 后处理 (spec decode 等)
12. Return ModelRunnerOutput
```

## CUDA Graph 模式决策

`_determine_batch_execution_and_padding()` 返回:

| 模式 | 条件 | 行为 |
|------|------|------|
| `FULL` | 已捕获 graph | 直接 replay cudagraph |
| `FULL` | 未捕获 | 捕获新 graph |
| `PIECEWISE` | 编译配置 | 逐段执行 (breakable) |
| `NONE` | fallback | 常规 eager 执行 |

批处理 pad 到 cudagraph 所需大小后执行。

## 与 UBatch 的集成

```python
ubatch_slices = maybe_create_ubatch_slices(
    should_ubatch, num_scheduled_tokens_np,
    num_tokens_padded, num_reqs_padded,
    self.parallel_config.num_ubatches,  # 默认 2
)
```

当 DP/EP 需要通信时，将 batch 沿 token dim 切分为 2 个 microbatch，通过 `UBatchWrapper` 实现通信-计算重叠。

## 关键职责

1. **状态管理**: 维护 `InputBatch` 持久状态 (req_ids, block_table, computed_tokens)
2. **Attention Metadata**: 为每个 Attention Backend 准备正确的 metadata
3. **CUDA Graph**: 管理 graph capture/replay 生命周期
4. **UBatch 协调**: 通过 `should_ubatch` 决定是否启用微批处理
5. **Memory**: 管理 GPU cache (KV Cache / Mamba state / Encoder Cache)
6. **Multi-Modal**: 处理图像编码器输入和缓存
7. **Speculative Decode**: 管理 draft tokens 和 target model

## 关键配置

| 配置 | 作用 |
|------|------|
| `max_num_batched_tokens` | 单次 forward 最大 token 数 |
| `max_num_seqs` | 最大并发请求数 |
| `num_ubatches` | UBatch 微批数量 (默认2) |
| `use_ubatching` | 是否启用 DBO |

## 与 Scheduler 的关系

```
Scheduler.schedule()
  → SchedulerOutput (token/request 分配)
    → GPUModelRunner.execute_model(scheduler_output)
      → ModelRunnerOutput (hidden_states, logits)
        → Scheduler.update_from_output()
```

GPUModelRunner 是 Scheduler 和 Model 之间的**执行桥接**，负责将调度决策转化为实际的 GPU 计算。

## 代码位置速查

| 行号 | 内容 |
|------|------|
| L418- | `GPUModelRunner.__init__` |
| L3998- | `execute_model()` 主入口 |
| L5080- | `load_model()` |
| L6157- | `profile_run()` |
| ~L4100 | `_determine_batch_execution_and_padding()` |
| ~L4125 | `maybe_create_ubatch_slices()` |
