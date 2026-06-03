# PyTorch FSDP (Fully Sharded Data Parallel)

> 目标：理解 FSDP 的分片机制、通信模式、与 ZeRO 的关系，以及在实际框架（verl）中的应用

## 1. 为什么需要 FSDP

### 问题：大模型训练的显存瓶颈

DDP (DistributedDataParallel) 在每张 GPU 上保存完整模型副本：

| 组件 | LLaMA-7B 显存 | LLaMA-70B 显存 |
|------|-------------|-------------|
| FP16 参数 | 13.5 GB | 157 GB |
| 梯度 | 13.5 GB | 157 GB |
| 优化器 (AdamW, FP32) | 53.9 GB | 628 GB |
| 激活值 | ~6 GB | ~25 GB |
| **总计/GPU** | **~87 GB** | **~967 GB** |

LLaMA-7B 单卡就需要 87GB，而 A100 只有 80GB。大模型训练必须在多卡间分片。

### FSDP 的核心思想

将模型参数、梯度、优化器状态均匀分片到 N 张 GPU，每张 GPU 只存 1/N：

```
DDP (4 GPUs):
  GPU 0: [全部参数] [全部梯度] [全部优化器]
  GPU 1: [全部参数] [全部梯度] [全部优化器]
  GPU 2: [全部参数] [全部梯度] [全部优化器]
  GPU 3: [全部参数] [全部梯度] [全部优化器]

FSDP FULL_SHARD (4 GPUs):
  GPU 0: [参数1/4] [梯度1/4] [优化器1/4]
  GPU 1: [参数2/4] [梯度2/4] [优化器2/4]
  GPU 2: [参数3/4] [梯度3/4] [优化器3/4]
  GPU 3: [参数4/4] [梯度4/4] [优化器4/4]
```

## 2. 与 DeepSpeed ZeRO 的对应关系

| FSDP 策略 | ZeRO 等价 | 分片内容 | 显存节省 |
|-----------|---------|---------|---------|
| `NO_SHARD` | ZeRO-0 | 无 (同 DDP) | 0% |
| `SHARD_GRAD_OP` | ZeRO-2 | 梯度 + 优化器 | ~67% |
| `FULL_SHARD` | ZeRO-3 | 参数 + 梯度 + 优化器 | ~76% |
| `HYBRID_SHARD` | ZeRO-3 + DP | 节点内全分片 + 节点间复制 | 取决于拓扑 |

### 实测数据 (fsdp_memory_sim.py)

LLaMA-7B, 4x A100:
- DDP: 114.26 GB/GPU (无法运行)
- FSDP-full: 27.06 GB/GPU (节省 76.3%)
- FSDP-grad: 36.77 GB/GPU (节省 67.8%)

LLaMA-70B 需要 **16x A100** 用 FSDP-full 才能训练 (76.8GB < 80GB)。

## 3. FlattenParameter 与分片机制

### 3.1 FlattenParameter

FSDP 将一个 module 组内的所有参数拼接成一个连续的 `FlatParameter`：

```python
# 原始参数
self.q_proj.weight  # (4096, 4096)
self.k_proj.weight  # (4096, 4096)
self.v_proj.weight  # (4096, 4096)
self.o_proj.weight  # (4096, 4096)

# FSDP flatten 后
self._flat_param    # (4 * 4096 * 4096,) = (67108864,)
```

好处：
- 单次 AllGather/ReduceScatter 通信（而非每个参数一次）
- 减少通信开销和 kernel launch
- 连续内存布局更高效

### 3.2 Forward/Backward 生命周期

```
Forward (per FSDP unit):
  1. AllGather: 从各 GPU 收集完整参数
  2. 计算: 用完整参数做 forward
  3. Free: 丢弃非本地分片 (reshard)

Backward (per FSDP unit):
  1. AllGather: 从各 GPU 收集完整参数
  2. 计算梯度: 用完整参数做 backward
  3. ReduceScatter: 将梯度按分片聚合到各 GPU
  4. Free: 丢弃非本地参数和梯度
```

关键：每层只在需要时临时收集完整参数，用完立即释放。

## 4. Sharding 策略详解

### 4.1 FULL_SHARD (ZeRO-3)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=transformer_auto_wrap_policy,
)
```

- 参数、梯度、优化器状态都分片
- Forward/Backward 需要 AllGather
- 显存节省最大，但通信量也最大

### 4.2 SHARD_GRAD_OP (ZeRO-2)

```python
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
)
```

- 参数完整复制，梯度和优化器分片
- Forward 不需要 AllGather（参数已在本地）
- Backward 只需要 ReduceScatter 梯度
- 适合中小模型，通信开销更小

### 4.3 HYBRID_SHARD

```python
# 2D Device Mesh: [DP_dim, FSDP_dim]
device_mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=["ddp", "fsdp"])

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    device_mesh=device_mesh,
)
```

- 节点内 (FSDP group): FULL_SHARD
- 节点间 (DP group): 复制模型
- 适合多节点训练：节点内 NVLink 全分片高效，节点间只做梯度 AllReduce

verl 的实现：
```python
# verl/workers/engine/fsdp/utils.py
if device_mesh.ndim == 1:
    sharding_strategy = fsdp_strategy  # FULL_SHARD
elif device_mesh.ndim == 2:
    sharding_strategy = hsdp_strategy  # HYBRID_SHARD
```

## 5. Wrapping 策略

Wrapping 决定了分片的粒度——哪些参数被组合成一个 FSDP unit。

### 5.1 常用策略

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy

# 策略 1: 按 Transformer 层包装（推荐）
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},  # 指定 Transformer 层类
)

# 策略 2: 按参数量包装
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=1e6,  # 超过 1M 参数的 module 单独包装
)
```

### 5.2 粒度影响 (LLaMA-7B, 4 GPUs)

| 策略 | Units | AllGather | ReduceScat | 峰值缓冲 |
|------|-------|-----------|------------|---------|
| per_layer | 32 | 25905 MB | 12953 MB | 405 MB |
| per_attn_ffn | 64 | 34628 MB | 17314 MB | 271 MB |
| entire_model | 1 | 26954 MB | 13477 MB | 13477 MB |

- **per_layer**: 平衡选择，峰值缓冲适中
- **per_attn_ffn**: 更细粒度，利于通信计算重叠
- **entire_model**: 通信量最少但峰值缓冲最大

### 5.3 verl 的 wrapping

```python
# verl/utils/fsdp_utils.py
def get_fsdp_wrap_policy(module, config, is_lora=False):
    if config is None:
        # 默认: transformer_auto_wrap_policy
        ...
```

## 6. Mixed Precision 训练

FSDP 原生支持混合精度：

```python
from torch.distributed.fsdp import MixedPrecision

mixed_precision = MixedPrecision(
    param_dtype=torch.bfloat16,    # 参数存储类型
    reduce_dtype=torch.float32,     # 梯度归约类型
    buffer_dtype=torch.float32,     # Buffer 类型
)

model = FSDP(model, mixed_precision=mixed_precision)
```

verl 的配置：
```python
# verl/workers/engine/fsdp/transformer_impl.py
mixed_precision = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)
```

BF16 参数 + FP32 归约：利用 BF16 的动态范围优势，同时保证梯度聚合精度。

## 7. Prefetching 通信计算重叠

### 原理

在计算第 N 层时，预先 AllGather 第 N+1 层的参数：

```
无 Prefetch:  [AG L1] [Compute L1] [AG L2] [Compute L2] [AG L3] [Compute L3]
有 Prefetch:  [AG L1] [Compute L1 + AG L2] [Compute L2 + AG L3] [Compute L3]
```

### 前向 Prefetching

```python
model = FSDP(model, forward_prefetch=True)
```

- **隐式预取**（始终开启）：使用独立 CUDA stream 执行 AllGather，与计算并行
- **显式预取**（`forward_prefetch=True`）：改变 CPU 侧的 AllGather 发起顺序

### 反向 Prefetching

```python
from torch.distributed.fsdp import BackwardPrefetch

model = FSDP(model, backward_prefetch=BackwardPrefetch.BACKWARD_PRE)
```

- `BACKWARD_PRE`：在当前 unit 反向**前**就预取下一个 unit（激进，内存高但重叠好）
- `BACKWARD_POST`：在当前 unit 反向**后**才预取（保守，内存友好但重叠少）

**关键限制**：反向预取必须显式配置，因为 NCCL 使用单个 process group 对应单个内部 stream。

### 效果分析

通信/计算比决定 Prefetch 收益：

| 互联 | 有效带宽 | AllGather (7B/层) | 计算 (7B/层) | 比值 |
|------|---------|-------------------|-------------|------|
| NVLink (A100) | 300 GB/s | 0.001 ms | 33 ms | 0.003% |
| RoCE/IB | 12 GB/s | 0.032 ms | 33 ms | 0.1% |
| PCIe (A16) | 1 GB/s | 0.4 ms | 33 ms | 1.2% |
| Ethernet | 0.5 GB/s | 0.65 ms | 33 ms | 2.0% |

NVLink 下通信开销极小，Prefetch 收益有限。低速互联下收益更明显。

## 8. FSDP2 (torch.distributed.tensor API)

PyTorch 2.x 引入了基于 DTensor 的 FSDP2：

```python
from torch.distributed._composable.fsdp import fully_shard

# FSDP2: 逐层分片
for layer in model.layers:
    fully_shard(layer, mesh=device_mesh, reshard_after_forward=True)
fully_shard(model, mesh=device_mesh)
```

### FSDP1 vs FSDP2

| 特性 | FSDP1 | FSDP2 |
|------|-------|-------|
| API | `FullyShardedDataParallel()` | `fully_shard()` |
| 参数表示 | FlatParameter | DTensor (per-parameter) |
| 灵活性 | 全局 wrap 策略 | 逐模块精细控制 |
| 混合并行 | 需手动管理 | 原生支持 2D/3D mesh |
| 状态管理 | state_dict 类型切换 | DTensor 直接序列化 |
| 内存管理 | recordStream (非确定性) | 显式管理 (确定性) |
| torch.compile | 需要 `use_orig_params=True` | 原生支持 |

### FSDP1 的 recordStream 问题

FSDP1 使用 `torch.Tensor.record_stream` 管理 buffer 生命周期：
- buffer 何时释放取决于 GC 和 CUDA event 完成时间
- 导致**非确定性内存使用**：同样的代码可能 OOM 也可能不 OOM
- 这是 FSDP2 的主要改进动机之一

### FSDP2 的显式内存管理

FSDP2 使用显式的 unshard/reshard 控制参数生命周期：
```python
# FSDP2 手动控制 API
module.unshard(async_op=False)   # 收集完整参数
module.reshard()                  # 释放回分片状态
module.set_modules_to_backward_prefetch(modules)  # 配置预取
```
- 内存使用可预测、可复现
- 适合需要精细控制的场景（如推理与训练混合）

verl 同时支持两种：
```python
if self.engine_config.strategy == "fsdp":
    module = FSDP(module, ...)           # FSDP1
elif self.engine_config.strategy == "fsdp2":
    apply_fsdp2(module, fsdp_kwargs, ...)  # FSDP2
```

## 9. verl 中的 FSDP 实践

### 9.1 整体架构

```
FSDPEngine
├── _init_device_mesh()          # 创建 1D/2D device mesh
├── _build_module()              # 加载 HF 模型
├── _build_lora_module()         # LoRA 适配器
├── _build_fsdp_module()         # FSDP 包装 (FSDP1/FSDP2)
├── _build_optimizer()           # 优化器
├── forward_backward_batch()     # 训练主循环
│   └── micro_batches           # 梯度累积
├── optimizer_step()             # 梯度裁剪 + 更新
└── save/load_checkpoint()       # 分片 checkpoint
```

### 9.2 关键配置

```python
# verl FSDP 配置
FSDPEngineConfig:
  strategy: "fsdp" | "fsdp2"     # FSDP 版本
  fsdp_size: -1                   # FSDP 组大小 (-1 = 全部 GPU)
  reshard_after_forward: True     # = FULL_SHARD, False = SHARD_GRAD_OP
  forward_prefetch: True          # 启用 prefetch
  wrap_policy: None               # wrapping 策略
  use_orig_params: False          # 保留原始参数名
  param_offload: False            # CPU offload 参数
  optimizer_offload: False        # CPU offload 优化器
  ulysses_sequence_parallel_size: 1  # Ulysses SP
```

### 9.3 RL 训练中的 FSDP 应用

在 RLHF/GRPO 中，通常有 4 个模型：Actor, Critic, Ref Policy, Reward Model。

verl 的 CPU offload 策略：
- **Actor**: FSDP + GPU 常驻（需要训练）
- **Ref Policy**: FSDP + CPUOffload（只做 forward，不训练）
- **Critic**: FSDP + GPU 常驻（需要训练）

```python
# verl: Reference policy 自动 CPU offload
if self.engine_config.forward_only:
    cpu_offload = CPUOffload(offload_params=True)  # FSDP1
    offload_policy = CPUOffloadPolicy(pin_memory=True)  # FSDP2
```

### 9.4 Ulysses Sequence Parallel 集成

verl 将 FSDP 与 Ulysses SP 结合：

```python
# 2D Mesh: [dp, sp]
device_mesh = init_device_mesh("cuda", (dp_size, sp_size), ["dp", "sp"])
```

DP 维度做 FSDP 分片，SP 维度做序列并行。

## 10. Checkpoint 管理

FSDP 的 checkpoint 分两种模式：

```python
# 完整 checkpoint (rank 0 保存全部参数)
FSDP.set_state_dict_type(
    model,
    state_dict_type=StateDictType.FULL_STATE_DICT,
    state_dict_config=FullStateDictConfig(),
)

# 分片 checkpoint (每个 rank 只保存自己的分片)
FSDP.set_state_dict_type(
    model,
    state_dict_type=StateDictType.SHARDED_STATE_DICT,
    state_dict_config=ShardedStateDictConfig(),
)
```

- **FULL_STATE_DICT**: 单文件，rank 0 保存，兼容性好
- **SHARDED_STATE_DICT**: 每个 rank 一个文件，保存/加载快

## 11. 常见问题与调试

### 11.1 OOM

```
RuntimeError: CUDA out of memory
```

解决：
1. 增加 GPU 数量
2. 使用 `FULL_SHARD` (如果当前是 `SHARD_GRAD_OP`)
3. 启用 gradient checkpointing: `module.gradient_checkpointing_enable()`
4. 启用 activation offloading
5. 减少 batch size，增加 gradient accumulation

### 11.2 通信瓶颈

症状：GPU utilization 低，通信时间长。

诊断：
```bash
# NCCL 调试
NCCL_DEBUG=INFO python train.py
# 查看通信时间和带宽
```

解决：
1. 确保 NVLink 可用: `nvidia-smi topo -m`
2. 启用 `forward_prefetch=True`
3. 调整 wrapping 策略（更细粒度）
4. 使用 `HYBRID_SHARD` 减少跨节点通信

### 11.3 FSDP + LoRA

```python
# verl: LoRA 参数在 FSDP 包装前添加
module = _build_lora_module(module)  # 先加 LoRA
module = _build_fsdp_module(module)  # 再 FSDP 包装
```

注意：FSDP 包装后不能动态添加参数。LoRA adapter 必须在 FSDP 包装前创建。

## 实验验证

- `tools/fsdp_memory_sim.py` — FSDP 内存模拟器（5 个实验）
  - 实验 1: DDP vs FSDP 显存对比
  - 实验 2: 不同模型规模
  - 实验 3: GPU 数量扩展
  - 实验 4: Wrapping 策略
  - 实验 5: Prefetching 分析

关键发现：
- FSDP-full 比 DDP 节省 **76-80%** 显存
- LLaMA-70B 需要 **16x A100** (FSDP-full) 才能训练
- NVLink 下 AllGather 开销仅占计算的 **0.003%**，通信几乎免费
- Prefetching 在 NVLink 下收益有限，在 Ethernet 下更有价值

## 参考资料

- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch FSDP Notes](https://pytorch.org/docs/stable/notes/fsdp.html)
- [Getting Started with FSDP](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html)
- [ZeRO Paper: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- verl FSDP engine: `verl/workers/engine/fsdp/transformer_impl.py`
