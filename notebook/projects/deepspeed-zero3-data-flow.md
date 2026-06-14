# DeepSpeed ZeRO-3 数据流序列图 (源码级)

> 2026-06-15 | 源码: deepspeed/runtime/zero/partition_parameters.py (2205+行), stage3.py (3579行)
> 核心: Init metaclass注入 → 参数自动分片 → forward时gather → backward时partition

## 1. Init Metaclass注入机制

### 1.1 工作原理

```
with deepspeed.zero.Init():
    model = MyLargeModel()  ← 每个Module.__init__结束时自动分片参数

内部流程:
    __enter__():
        self.patch_init_and_builtins()
        → 替换 torch.nn.Module.__init__ 的 post-init hook
        → 每个 Module 创建完 → 调用 _post_init_method(module)
        → 自动将参数转为 DeepSpeed param → broadcast → partition

    __exit__():
        self.unpatch_init_and_builtins()
        → 恢复原始 Module.__init__
        → 所有参数已分片完毕
```

### 1.2 `_post_init_method(module)` 流程

```python
for name, param in module.named_parameters(recurse=False):
    if not is_zero_param(param):
        param.data = param.data.to(local_device)  # 移到正确设备
        self._zero_init_param(param)               # 分片!
```

### 1.3 `_zero_init_param(param)` 流程

```
_zero_init_param(param):
    1. _convert_to_deepspeed_param(param)  # 添加ds_*属性
    2. dist.broadcast(param.data, rank=0)   # rank0广播完整参数
    3. param.partition()                    # 每个rank只保留自己的分片

→ 结果: param.data被释放, param.ds_tensor存1/N分片
```

## 2. DeepSpeed Param 属性 (ds_*)

每个参数在ZeRO-3下变成一个"DeepSpeed param",附加以下属性:

| 属性 | 类型 | 含义 |
|------|------|------|
| `ds_param_type` | ZeroParamType | PARTITIONED(分片)/NORMAL(完整)/REMOTE(NVMe) |
| `ds_status` | ZeroParamStatus | AVAILABLE(可用)/NOT_AVAILABLE(已释放)/INFLIGHT(正在gather) |
| `ds_shape` | tuple | 原始参数shape |
| `ds_numel` | int | 原始参数元素数(不含padding) |
| `ds_tensor` | Tensor | 分片后的1/N参数 (存储在GPU/CPU/NVMe) |
| `ds_id` | int | 全局唯一ID |
| `ds_process_group` | ProcessGroup | 分片所在的DP进程组 |
| `ds_persist` | bool | 是否持久化(小参数常驻GPU) |
| `ds_active_sub_modules` | set | 当前需要此参数的sub-module集合 |
| `ds_secondary_tensor` | Tensor | 双分片的第二份(用于零优化) |
| `nvme_swapper` | Swapper | NVMe offload的异步swap器 |

## 3. Forward时的参数Gather

### 3.1 pre-forward hook

```
module.forward()触发前:
    PartitionedParameterCoordinator:
        1. 检查 module需要的参数 → 计算需要gather的参数列表
        2. AllGatherCoalescedHandle:
           - 将多个参数的gather合并为一次AllGather → 减少通信次数!
           - 使用 prefetch (提前gather下一层参数)
        3. gather完成后:
           - param.data = gathered_full_param (恢复完整参数)
           - param.ds_status = AVAILABLE
        4. 不需要的参数:
           - param.partition() → 释放完整参数 → 只保留ds_tensor分片
```

### 3.2 AllGather Coalescing (关键优化!)

```
传统: 每个参数单独AllGather → N次通信 → 开销大
优化: 多个参数合并AllGather → 1次通信 → 开销小!

AllGatherCoalescedHandle:
    1. 收集本层所有需要gather的参数
    2. 将所有参数的分片拼接成一个大的buffer
    3. 一次AllGather → 所有参数同时恢复
    4. 从buffer中切出各个参数 → 恢复param.data

→ 通信次数: 从O(num_params)降为O(1)!
```

### 3.3 Prefetch (预取优化)

```
PartitionedParameterCoordinator:
    1. 当前层forward → 同时prefetch下一层需要的参数
    2. AllGather当前层 + Prefetch下一层 → overlap通信和计算!
    3. 类似CPU cache prefetch → 减少forward等待时间

实现:
    trace() → 记录参数使用顺序 → 计算prefetch计划
    execute_coalesced_prefetch() → 提前gather下一层参数
```

## 4. Backward时的参数Partition

```
module.backward()触发后:
    1. 计算梯度 → param.grad = computed_gradient
    2. partition_gradients():
       - reduce_scatter_gradients → 每个rank只保留1/N梯度
       - param.grad.data = partitioned_grad
    3. partition():
       - 释放完整param.data → 只保留ds_tensor分片
       - param.ds_status = NOT_AVAILABLE

→ 结果: backward结束后, GPU内存只持有1/N参数+1/N梯度+1/N优化器状态
```

## 5. 完整训练Step数据流

```
ZeRO-3 单个训练step:

    1. Forward Phase:
       for each layer in model:
           a) pre-forward hook:
              - AllGather coalesced → gather本层所有参数
              - prefetch下一层参数 (overlap!)
           b) forward computation:
              - 使用完整参数 → 计算output
           c) post-forward hook:
              - partition(释放)不需要的参数 → 释放内存

    2. Backward Phase:
       for each layer in reverse:
           a) pre-backward hook:
              - AllGather → gather本层参数(如果已释放)
           b) backward computation:
              - 计算梯度 param.grad
           c) post-backward hook:
              - reduce_scatter_gradients → 每rank保留1/N梯度
              - partition → 释放完整参数

    3. Optimizer Step:
       - 每个rank只更新自己持有的1/N优化器状态
       - (CPU/NVMe offload时: 从CPU/NVMe取optimizer states → 更新 → 写回)

    4. Parameter Update → Next Forward:
       - updated ds_tensor → 下次forward AllGather时自动包含更新
```

## 6. 通信量分析

```
单层forward:
    - AllGather: Ψ/N * N → Ψ (gather完整参数)
    - coalesced: 1次AllGather包含所有参数 → 通信次数O(1)

单层backward:
    - AllGather: Ψ (gather完整参数, 如果已释放)
    - ReduceScatter: Ψ → Ψ/N (梯度分片)

单层总通信:
    - Forward: 1x AllGather(Ψ)
    - Backward: 1x AllGather(Ψ) + 1x ReduceScatter(Ψ)
    - 总量: 3Ψ per layer (vs DDP=2Ψ per layer → ZeRO-3多50%)

整个模型:
    - L层 → 总通信 = 3LΨ
    - 但coalescing → 每层1次 → 实际更优
    - Prefetch → overlap → 实际wall time更低
```

## 7. 内存公式验证

```
ZeRO-3 单GPU内存:
    参数分片: Ψ/N * 2 bytes (fp16/bf16)
    梯度分片: Ψ/N * 2 bytes (fp16/bf16)
    优化器状态分片: Ψ/N * 12 bytes (fp32 master + fp32 m + fp32 v)
    总计: (2 + 2 + 12) * Ψ/N = 16Ψ/N

加上临时gather:
    forward gather: Ψ * 2 bytes (完整参数临时占用)
    backward gather: Ψ * 2 bytes (完整参数临时占用)
    总计峰值: 16Ψ/N + 4Ψ (forward+backward gather)

→ 公式: 16Ψ/N + 4Ψ (peak) = (16/N + 4) * Ψ

7B模型 N=8: (16/8 + 4) * 14GB = (2+4)*14 = 84GB peak → 不适合RTX 4090!
7B模型 N=1: (16/1 + 4) * 14 = 280GB → 完全不可能!
→ ZeRO-3不适合RTX 4090 → verl GRPO更适合
```

## 8. ZeRO-Offload 数据流

```
ZeRO-Offload (ZeRO-2 + CPU optimizer):

    Forward: GPU正常计算 (参数全在GPU)
    Backward: GPU正常计算 → 梯度reduce-scatter → 1/N梯度到GPU
    Optimizer: CPU Adam → 从GPU拷贝1/N梯度到CPU → 更新 → 拷回GPU

    通信:
    - GPU→CPU: Ψ/N (梯度拷贝)
    - CPU→GPU: Ψ/N (更新后参数拷贝)
    - 比ZeRO-3少2次AllGather → 总量Ψ/N vs 3Ψ

    内存:
    - GPU: 4Ψ + 2Ψ/N (参数+梯度) → N=1时 = 6Ψ = 84GB → 24GB RTX 4090不行
    - 实际: LoRA + CPU Adam → 20.04GB (verl方案)
```

## 9. 与其他框架对比

| 框架 | Forward Gather | Backward Partition | 通信量/层 | 内存 |
|------|---------------|-------------------|-----------|------|
| DDP | 无 | 无 | 2Ψ (AllReduce) | 16Ψ |
| ZeRO-1 | 无 | 无 | 2Ψ (AllReduce) | 4Ψ+12Ψ/N |
| ZeRO-2 | 无 | ReduceScatter | Ψ (ReduceScatter) | 4Ψ+14Ψ/N |
| ZeRO-3 | AllGather+partition | ReduceScatter+partition | 3Ψ | 16Ψ/N+4Ψ |
| FSDP2 | AllGather+reshard | ReduceScatter+reshard | 2Ψ (AllGather+RS=AllReduce) | 16Ψ/N |
| Megatron TP | 无 | 无(切分维度) | Ψ/TP_size | Ψ/TP_size |
| Megatron PP | P2P | P2P | 0 (layer间) | Ψ/stages |

→ ZeRO-3通信量最大(3Ψ), 但内存最省(16Ψ/N); FSDP2更优(2Ψ)
→ RTX 4090: verl GRPO+LoRA+CPU Adam = 最实用方案

## 10. 下一步

- [ ] 研究 PartitionedParameterCoordinator prefetch 机制
- [ ] 研究 ZeRO-Infinity NVMe swap 数据流
- [ ] 对比 DeepSpeed Init vs FSDP1 FlatParameter vs FSDP2 DTensor
- [ ] 在GPU可用时实测 ZeRO-2+CPU Adam 7B 训练
