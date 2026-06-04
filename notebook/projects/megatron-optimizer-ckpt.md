# Megatron-LM 分布式优化器 + Checkpoint 分析

> 文件: `megatron/core/optimizer/distrib_optimizer.py` (~3000+ 行)
> 分析日期: 2026-06-04

## DP Optimizer 设计 (ZeRO-1 风格)

**核心思想**: 将优化器状态 (Adam 的 m, v) 沿 DP 维度切分，每个 rank 只维护自己那部分。

```
ZeRO-1:  shard optimizer states (12 bytes/param savings)
ZeRO-2:  shard gradients + optimizer states
ZeRO-3:  shard params + gradients + optimizer states

Megatron-LM DistributedOptimizer = ZeRO-1
```

## ParamAndGradBuffer 架构

```
Rank 0:  [param_a] [param_b] [param_c] [padding]  ← 连续内存 buffer
Rank 1:  [param_a] [param_b] [param_c] [padding]  ← 每个 rank 都有完整 model params

Bucket: 将 buffer 按 DP size 切分:
  Rank 0 "owns" shard [0:N/4], 负责 reduce+gather its portion
  Rank 1 "owns" shard [N/4:N/2], etc.

Gradient AllReduce:
  每个 rank 对全局 grads 做 AllReduce → 完整 grads
  但只有自己 "own" 的 shard 用于 optimizer.step()
```

## 优化器 Step 流程

```
step_with_ready_grads():
  1. For each bucket in grad buffers:
     a. AllReduce grads (bucket-level, overlap with next bucket)
     b. Unscale grads (if AMP)
     c. Clip grads (global norm)
     d. optimizer.step() on local shard only

  2. AllGather updated params from each rank's shard
     → 每个 rank 获得完整的、更新后的 params

  3. Copy updated params back to model
```

## Memory 分析

```
基础: params (2B) + grads (2B) + FP32 param copy (4B) = 8B/param

Adam 优化器状态 (ZeRO-1 前):
  - momentum (m): 4B/param (FP32)
  - variance (v): 4B/param (FP32)
  - master weights: 4B/param (FP32)
  = 12B/param for Adam

总内存 (无 ZeRO):
  model (2B) + grad (2B) + optimizer (12B) + activation (variable)
  = 16B/param + activations

ZeRO-1 (DP=8):
  optimizer states per rank: 12B/8 = 1.5B/param
  Total: 2B + 2B + 1.5B + activations = 5.5B/param + activations
  Savings: ~65% of optimizer memory

ZeRO-2 (DP=8):
  optimizer + grads: (12+2)/8 = 1.75B/param
  Total: 2B + 1.75B + activations = 3.75B/param + activations
  Savings: ~76%

ZeRO-3 (DP=8):
  params + optimizer + grads: (2+12+2)/8 = 2B/param
  Total: 2B + activations
  Savings: ~87%
```

## Checkpoint 机制

### 分布式 Checkpoint

```python
sharded_state_dict():
  # 每个参数的状态分片存储 (不聚合)
  state_dict[param_name] = ShardedTensor(
      local_shard=optimizer_state[param].data,
      global_shape=[total_params],
      # track which DP rank owns which shard
  )
```

### Checkpoint 格式

| 格式 | 描述 |
|------|------|
| `fully_reshardable` | 默认: 状态与 DP rank 无关 |
| `fully_sharded_model_space` | 映射到 model param 维度 |
| `fsdp_dtensor` | FSDP DTensor 格式 |
| `dp_zero` | 依赖内部 buffer 结构 (最快) |
| `dp_reshardable` | Reshardable DP 格式 |

### 加载/恢复

```
load_checkpoint():
  1. 读取所有 shard files
  2. 根据当前 DP size reshape → 重新分配
  3. 如果 DP size 变化: 自动 re-shard
  4. 恢复 optimizer states (m, v) + lr_scheduler state
```

## 优化器类型

| Class | 基类 | 特点 |
|-------|------|------|
| `MixedPrecisionOptimizer` | - | 混合精度基础: FP16 fwd/bwd + FP32 optimizer |
| `DistributedOptimizer` | MixedPrecisionOptimizer | ZeRO-1 分片 |
| `HybridDeviceOptimizer` | - | CPU offloading |
| `LayerWiseOptimizer` | - | 逐层optimizer step |

## GradScaler (AMP)

```python
class MegatronGradScaler:
    # FP16 训练用, BF16 不需要
    loss_scale: float  # 动态调整
    def scale(loss): return loss * loss_scale
    def unscale(grads): grads /= loss_scale
    def update(overflow):
        if overflow: loss_scale /= 2
        else: loss_scale *= growth_factor
```

**BF16 native**: 不需要 GradScaler (指数范围与 FP32 相同)

## 通信-计算 Overlap

```
Bucket-level overlapping:
  ┌─────────┬─────────┬─────────┐
  │ AR bkt0 │ AR bkt1 │ AR bkt2 │  ← AllReduce 流水线
  │ step 0  │ step 1  │ step 2  │  ← optimizer.step()
  └─────────┴─────────┴─────────┘

GradReduce → Step 可以 overlap:
  在等待下一 bucket AllReduce 时对当前 bucket 做 step
```

## 代码位置速查

| 文件 | 内容 |
|------|------|
| `distrib_optimizer.py` | ZeRO-1 分布式优化器 (3000+ 行) |
| `optimizer.py` | MixedPrecisionOptimizer 基类 |
| `grad_scaler.py` | FP16 AMP GradScaler |
| `clip_grads.py` | 梯度裁剪 (global norm) |
| `optimizer_config.py` | 优化器配置 |
| `param_layout.py` | Parameter buffer 布局管理 |
| `cpu_offloading/` | CPU offloading 优化器 |
