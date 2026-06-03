# vLLM V1 Executor 架构源码阅读

> 从 Scheduler 到 GPU 的完整推理执行路径：Executor → Worker → ModelRunner → Model

## 1. 核心文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `v1/executor/abstract.py` | 381 | Executor 抽象基类，定义 `collective_rpc` / `execute_model` 接口 |
| `v1/executor/uniproc_executor.py` | 195 | 单进程 Executor（最简单路径） |
| `v1/executor/multiproc_executor.py` | ~200 | 多进程 Executor（TP>1） |
| `v1/executor/ray_executor.py` | ~300 | Ray 分布式 Executor（多节点） |
| `v1/worker/worker_base.py` | 352 | Worker 抽象基类 + WorkerWrapperBase 生命周期管理 |
| `v1/worker/gpu_worker.py` | ~900 | GPU Worker：加载模型、PP 通信、显存管理 |
| `v1/worker/gpu_model_runner.py` | ~7500 | **核心**: GPU 模型执行、输入准备、KV Cache 管理 |
| `v1/attention/backend.py` | ~430 | AttentionBackend 接口 + CommonAttentionMetadata |
| `v1/attention/backends/registry.py` | ~150 | Attention Backend 注册表（20+ backends） |
| `v1/core/sched/output.py` | ~260 | SchedulerOutput 数据结构 |

## 2. 架构层次

```
Engine Core (API 层)
    │
    ▼
Scheduler → SchedulerOutput
    │
    ▼
Executor (abstract.py:37)
    ├── UniProcExecutor     — 单 GPU，单进程
    ├── MultiprocExecutor   — 单节点多 GPU (TP)
    ├── RayExecutor         — 多节点 (Ray Actor)
    ├── RayExecutorV2       — 新版 Ray Executor
    └── ExternalLauncher    — torchrun 兼容
    │
    ▼ collective_rpc("execute_model")
Worker (worker_base.py:39)
    ├── GPUWorker           — CUDA 执行
    ├── CPUWorker           — CPU 执行
    └── TPUWorker           — TPU 执行
    │
    ▼ model_runner.execute_model()
GPUModelRunner (gpu_model_runner.py:418)
    │
    ▼ _model_forward()
Model (nn.Module)
    │
    ▼
AttentionLayer → AttentionBackend → GPU Kernel
```

## 3. Executor 选择与初始化

### 3.1 Executor 工厂 (abstract.py:48-92)

```python
@staticmethod
def get_class(vllm_config: VllmConfig) -> type["Executor"]:
    distributed_executor_backend = parallel_config.distributed_executor_backend

    if isinstance(distributed_executor_backend, type):
        # 直接传入自定义 Executor 类
        return distributed_executor_backend
    elif distributed_executor_backend == "ray":
        return RayDistributedExecutor  # or RayExecutorV2
    elif distributed_executor_backend == "mp":
        return MultiprocExecutor
    elif distributed_executor_backend == "uni":
        return UniProcExecutor
    elif distributed_executor_backend == "external_launcher":
        return ExecutorWithExternalLauncher
```

**选择逻辑**:
- 单 GPU: `UniProcExecutor`（最简单，无进程间通信）
- 单节点 TP: `MultiprocExecutor`（多进程，通过 pipe 共享内存通信）
- 多节点: `RayExecutor`（Ray Actor 远程调用）
- torchrun: `ExecutorWithExternalLauncher`（外部启动，用 `env://` 初始化）

### 3.2 UniProcExecutor 初始化 (uniproc_executor.py:46-69)

```python
class UniProcExecutor(Executor):
    def _init_executor(self):
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=0,
            rank=0,
            is_driver_worker=True,
            shared_worker_lock=Lock(),
        )
        self.driver_worker.init_worker(all_kwargs=[kwargs])  # 创建实际 Worker
        self.driver_worker.init_device()                      # 初始化 GPU
        self.driver_worker.load_model()                       # 加载模型权重
```

**关键**: `WorkerWrapperBase` 是延迟初始化包装器，先记住配置，再在 `init_worker` 时动态创建实际 Worker 类（通过 `parallel_config.worker_cls` 反射）。

### 3.3 通信接口: collective_rpc (abstract.py:153-202)

所有 Executor 操作都通过 `collective_rpc` 统一接口：

```python
def execute_model(self, scheduler_output, non_block=False):
    output = self.collective_rpc(
        "execute_model",
        args=(scheduler_output,),
        non_block=non_block,
    )
    return output[0]  # 取第一个 worker 的结果
```

- `UniProcExecutor`: 直接调用 `run_method(self.driver_worker, ...)`
- `MultiprocExecutor`: 通过 pipe 发送给子进程
- `RayExecutor`: 通过 Ray Actor 远程调用

## 4. Worker 层

### 4.1 WorkerBase (worker_base.py:39)

抽象接口，定义核心生命周期方法：

```python
class WorkerBase:
    def init_device(self) -> None: ...        # 初始化设备 + 加载模型
    def load_model(self) -> None: ...          # 加载模型权重
    def get_kv_cache_spec(self) -> dict: ...   # 获取 KV Cache 规格
    def compile_or_warm_up_model(self) -> CompilationTimes: ...  # 编译/预热
    def execute_model(self, scheduler_output) -> ModelRunnerOutput: ...
    def determine_available_memory(self) -> int: ...  # 可用显存
```

### 4.2 WorkerWrapperBase (worker_base.py:187)

延迟初始化 + 动态 Worker 类加载：

```python
class WorkerWrapperBase:
    def init_worker(self, all_kwargs):
        # 通过 parallel_config.worker_cls 反射加载 Worker 类
        worker_class = resolve_obj_by_qualname(parallel_config.worker_cls)

        # 支持 worker_extension_cls 动态注入
        if worker_extension_cls:
            worker_class.__bases__ += (worker_extension_cls,)

        self.worker = worker_class(**kwargs)
```

**设计要点**: Worker 类通过配置字符串指定（如 `"vllm.v1.worker.gpu_worker.GPUWorker"`），运行时动态加载。还支持 `worker_extension_cls` 动态注入额外方法（通过修改 `__bases__`）。

### 4.3 GPUWorker (gpu_worker.py)

GPU Worker 负责:
1. **设备初始化**: `init_device()` → 设置 CUDA、初始化分布式后端
2. **模型加载**: `load_model()` → `model_runner.load_model()` → `get_model_loader().load_model()`
3. **KV Cache 规格获取**: `get_kv_cache_spec()` → `model_runner.get_kv_cache_spec()`
4. **显存探测**: `determine_available_memory()` → 可用显存（减去模型权重）
5. **模型执行**: `execute_model()` → 处理 PP 通信 → `model_runner.execute_model()`

**Pipeline Parallelism 通信** (gpu_worker.py:784-872):
```python
def execute_model(self, scheduler_output):
    # 非第一个 PP rank: 接收上一阶段的中间 tensor
    if not get_pp_group().is_first_rank:
        tensor_dict = get_pp_group().irecv_tensor_dict(...)
        intermediate_tensors = AsyncIntermediateTensors(tensor_dict)

    output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)

    # 非最后一个 PP rank: 非阻塞发送到下一阶段
    if not get_pp_group().is_last_rank:
        self._pp_send_work = get_pp_group().isend_tensor_dict(output.tensors)
        return None
```

## 5. GPUModelRunner — 执行核心

### 5.1 类定义 (gpu_model_runner.py:418)

```python
class GPUModelRunner(
    LoRAModelRunnerMixin,           # LoRA 支持
    KVConnectorModelRunnerMixin,    # KV Transfer (P/D 分离)
    ECConnectorModelRunnerMixin,    # Encoder-Decoder Connector
):
```

**继承 Mixin**: 通过 Mixin 模式组合不同功能，保持核心 ModelRunner 代码清晰。

### 5.2 execute_model 主流程 (gpu_model_runner.py:3998-4346)

```
execute_model(scheduler_output)
    │
    ├── 1. _update_states() — 更新持久化 batch 状态
    │
    ├── 2. _execute_mm_encoder() — 多模态 encoder 处理（可选）
    │
    ├── 3. _prepare_inputs() — 准备 input_ids, positions
    │      └── 更新 block_table, seq_lens 等元数据
    │
    ├── 4. _determine_batch_execution_and_padding()
    │      └── 决定 CUDA Graph 模式、padding 大小、是否 ubatching
    │
    ├── 5. _get_slot_mappings() — 计算 KV Cache slot 映射
    │
    ├── 6. _build_attention_metadata() — 构建 Attention 元数据
    │
    ├── 7. _preprocess() — 最终输入预处理
    │
    ├── 8. set_forward_context() — 设置全局 forward 上下文
    │
    ├── 9. _model_forward() — 模型前向传播
    │      └── self.model(input_ids, positions, ...)
    │
    └── 10. compute_logits() → sample_tokens()
           └── 返回 ModelRunnerOutput
```

### 5.3 SchedulerOutput 结构 (output.py:181)

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]      # 新请求（含 token IDs）
    scheduled_cached_reqs: CachedRequestData       # 已缓存请求的更新
    num_scheduled_tokens: dict[str, int]           # req_id → 调度的 token 数
    total_num_scheduled_tokens: int                # 总 token 数
    finished_req_ids: set[str]                     # 已完成请求
    num_common_prefix_blocks: int                  # 公共前缀 block 数
    kv_connector_metadata: KVConnectorMetadata     # KV Transfer 元数据
    scheduled_spec_decode_tokens: dict             # 投机解码 token
```

### 5.4 CUDA Graph 策略 (gpu_model_runner.py:3764)

`_determine_batch_execution_and_padding` 决定执行模式：

| 模式 | 条件 | 效果 |
|------|------|------|
| `FULL` | 均匀 decode (每请求相同 token 数) | 录制 CUDA Graph，固定 batch size |
| `NONE` | Prefill 或混合 batch | 每次重新执行，无录制 |
| `UBATCHING` | 大 batch + 不均匀 | 分成多个 micro-batch 分别执行 |

**Padding**: CUDA Graph 需要固定输入大小。当实际 token 数小于录制大小时，用 padding 补齐。

### 5.5 模型前向传播 (gpu_model_runner.py:3711)

```python
def _model_forward(self, input_ids, positions, intermediate_tensors, **kwargs):
    return self.model(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        **kwargs,
    )
```

极简设计——实际复杂性在 `_prepare_inputs` 和 `_build_attention_metadata` 中。

### 5.6 后处理 (gpu_model_runner.py:4282-4346)

```python
# 取 logits（只取最后一个 token 的 logits）
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)

# Pooling 模型直接返回 embedding
# PP 中间 rank 返回 IntermediateTensors
# 最后一个 PP rank 返回 logits
```

## 6. Attention Backend 系统

### 6.1 Backend 注册表 (registry.py:34)

```python
class AttentionBackendEnum(Enum):
    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    FLASHINFER = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
    TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    # MLA backends
    FLASHMLA = "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend"
    FLASHINFER_MLA = "vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend"
    TRITON_MLA = "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"
    CUTLASS_MLA = "vllm.v1.attention.backends.mla.cutlass_mla.CutlassMLABackend"
    # ROCm backends
    ROCM_ATTN = "vllm.v1.attention.backends.rocm_attn.RocmAttentionBackend"
    # ...
```

**20+ 后端**: FlashAttention、FlashInfer、Triton、多种 MLA、ROCm、CPU 等。

### 6.2 Backend 选择流程

```
Platform.get_attn_backend_cls() (platforms/cuda.py)
    │
    ├── 检查 config 中指定了特定 backend
    │
    ├── 遍历优先级列表，检查每个 backend 的支持条件:
    │   - head_size 是否支持
    │   - dtype 是否支持
    │   - kv_cache_dtype 是否兼容
    │   - block_size 是否满足
    │   - device_capability (SM 版本) 是否满足
    │   - use_mla 是否需要 MLA
    │
    └── 返回第一个满足条件的 backend
```

### 6.3 CommonAttentionMetadata (backend.py:359)

```python
@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor     # (batch_size+1,) 每个请求的起始位置
    seq_lens: torch.Tensor            # (batch_size,) 已计算的 token 数
    num_reqs: int                     # 请求数
    num_actual_tokens: int            # 批次总 token 数
    max_query_len: int                # 最长查询长度
    max_seq_len: int                  # 最长上下文长度
    block_table_tensor: torch.Tensor  # Block table (PagedAttention)
    slot_mapping: torch.Tensor        # KV Cache slot 映射
    causal: bool = True               # 是否因果注意力
    is_prefilling: torch.Tensor       # (batch_size,) 是否在 prefill
```

每个 Backend 的 `AttentionMetadataBuilder` 将 `CommonAttentionMetadata` 转换为 backend 特定的元数据。

### 6.4 Backend 接口

```python
class AttentionBackend(ABC):
    @staticmethod
    def get_name() -> str: ...

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]: ...

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]: ...

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]: ...

    @staticmethod
    def get_kv_cache_shape(...) -> tuple: ...  # KV Cache tensor 形状

    # 关键属性
    forward_includes_kv_cache_update: bool  # 是否在 forward 中更新 KV Cache
```

**`forward_includes_kv_cache_update`**: 有些 backend（如 FlashInfer）在 attention forward 时同时更新 KV Cache；其他的（如 FlashAttention）需要在 forward 之后单独更新。这影响 slot_mapping 的 padding 策略。

## 7. 数据流总结

```
用户请求
    │
    ▼
Scheduler.schedule()
    │ 输出: SchedulerOutput
    │   - scheduled_new_reqs: token IDs + block table
    │   - scheduled_cached_reqs: 增量更新
    │   - num_scheduled_tokens: {req_id: num_tokens}
    │
    ▼
Executor.execute_model(scheduler_output)
    │ collective_rpc → Worker.execute_model()
    │
    ▼
GPUWorker.execute_model()
    │ PP: 接收 intermediate_tensors (非首个 rank)
    │
    ▼
GPUModelRunner.execute_model()
    │
    ├── _update_states()         # 持久化 batch 状态更新
    ├── _prepare_inputs()        # input_ids, positions, block_table
    ├── _build_attention_metadata()  # per-layer attention 元数据
    │
    ▼
    │ set_forward_context()      # 设置全局上下文（attn_metadata, kv_cache_config）
    │
    ▼
_model_forward()                 # self.model(input_ids, positions, ...)
    │
    ▼
Model.forward()
    │
    ├── Embedding(input_ids)
    │
    ├── for layer in transformer_layers:
    │   ├── Attention(query, key, value, kv_cache, attn_metadata)
    │   │   └── AttentionImpl.forward()  # FlashAttn/FlashInfer/Triton kernel
    │   └── MLP(hidden_states)
    │
    ├── compute_logits(hidden_states)  # LM Head
    │
    └── 返回 logits
    │
    ▼
GPUModelRunner 后处理
    │ logits[logits_indices]  # 只取需要采样的位置
    │
    ▼
sample_tokens() → Sampler (Triton kernel)
    │
    ▼
ModelRunnerOutput
    ├── sampled_token_ids: list[list[int]]  # 每个请求生成的 token
    ├── logits: torch.Tensor
    └── spec_decode_related...
```

## 8. 关键架构特点

### 8.1 Scheduler-Executor 分离

- **Scheduler** 在 CPU 上运行，决定"做什么"（分配 block、选择请求）
- **Executor** 在 GPU 上运行，决定"怎么做"（执行 forward、更新 KV Cache）
- 通过 `SchedulerOutput` 序列化传递决策

### 8.2 统一通信接口

`collective_rpc` 是 Executor 的核心抽象：
- 单进程：直接函数调用
- 多进程：pipe 通信
- Ray：远程 Actor 调用
- 对上层完全透明

### 8.3 Persistent Batch

`GPUInputBatch` 维护跨迭代的持久化状态：
- `req_ids`, `req_index`, `num_computed_tokens_cpu`
- 避免每步重建所有状态，只增量更新
- 配合 CUDA Graph 实现固定 buffer 大小

### 8.4 Multi-Backend Attention

20+ attention backend 通过统一接口接入：
- 标准注意力: FlashAttn, FlashInfer, Triton, CPU
- MLA: FlashMLA, FlashInferMLA, TritonMLA, CutlassMLA
- 特殊: FlexAttention (PyTorch 2.5+), NoAttention (Mamba)
- 平台: ROCm (Aiter), XPU

### 8.5 CUDA Graph 优化

- 仅在均匀 decode（每请求 1 token）时录制 CUDA Graph
- Prefill 和混合 batch 使用 eager 模式
- 支持 ubatching（将大 batch 分成多个 micro-batch）

## 9. 与 V0 的关键区别

| 方面 | V0 | V1 |
|------|----|----|
| Executor | `executor_base.py` | `abstract.py` + 4 种实现 |
| Worker | `worker/worker.py` | `worker_base.py` + GPU/CPU/TPU |
| Model Runner | `worker/model_runner.py` | `gpu_model_runner.py` (~7500 行) |
| Attention | 单一 backend | 20+ 可插拔 backend |
| 执行模式 | Prefill/Decode 分离 | 统一 execute_model() |
| CUDA Graph | 手动管理 | 自动检测 + padding |
| KV Cache 更新 | 集中在 ModelRunner | 由 Attention Backend 决定 |
| PP 通信 | 同步 | 异步 (`isend/irecv`) |

## 10. gpu_model_runner.py 行数分布

| 区域 | 行数范围 | 功能 |
|------|----------|------|
| 1-417 | imports + 配置 | 导入、常量、辅助类 |
| 418-1868 | GPUModelRunner.__init__ | 初始化、模型加载、KV Cache 分配 |
| 1869-3710 | 输入准备 | _prepare_inputs、_build_attention_metadata |
| 3711-3997 | 执行辅助 | _model_forward、batch 决策、padding |
| 3998-4346 | execute_model | **主执行流程** |
| 4347-7500 | 辅助功能 | CUDA Graph、sampling、LoRA、profiling |

## 参考资料

- 源码路径: `vllm-latest/vllm/v1/executor/`, `vllm-latest/vllm/v1/worker/`
- 相关笔记: [vLLM V1 架构](vllm-architecture.md), [vLLM V1 Scheduler](vllm-v1-scheduler-reading.md), [vLLM KV Cache Manager](vllm-v1-kv-cache-manager-reading.md), [vLLM MLA Backend](vllm-mla-backend-reading.md)
