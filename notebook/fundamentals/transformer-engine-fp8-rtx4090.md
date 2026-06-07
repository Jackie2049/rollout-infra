# TransformerEngine FP8 Deep Read — RTX 4090 (SM89)

```
┌─────────────────────────────────────────────────────────┐
│  TransformerEngine (NVIDIA) Source Deep Read            │
│                                                         │
│  5 Scaling Recipes: Delayed/Current/MXFP8/Block/NVFP4  │
│  RTX 4090 (SM89): FP8 ✅ / MXFP8 ❌ / NVFP4 ❌        │
│  FP8 Linear: quantize→GEMM→dequantize (fused kernel)   │
│  DelayedScaling: scale=FP8_MAX/amax/(2^margin)         │
│  Float8CurrentScaling: per-tensor current amax→scale    │
│  autocast: context manager → FP8GlobalStateManager      │
│  Quantizer: rowwise+fcolumnwise dual量化(for bwd)      │
│  Userbuffers: comm+GEMM overlap (AG/RS)                │
│  Key: TE用fused C++ kernel避免Python dequant overhead │
│  RTX 4090限制: 无MXFP8/NVFP4/BlockScaling             │
│  → 只能用DelayedScaling + Float8CurrentScaling         │
└─────────────────────────────────────────────────────────┘
```

## 1. TransformerEngine Architecture

### 1.1 整体设计

TransformerEngine是NVIDIA官方FP8训练库, 核心思想:
- **FP8量化**: 将BF16/FP32权重和激活量化为FP8 → GEMM计算用FP8 → 结果反量化回高精度
- **Fused Kernel**: 量化+GEMM+反量化在一个CUDA kernel完成 → 避免Python-level dequant overhead
- **Scaling Recipe**: 5种scaling策略控制量化精度 → 选择适合硬件的recipe

### 1.2 Module层次

```
TransformerEngineBaseModule (base.py)
  ├── Linear (linear.py) — FP8 Linear层
  ├── LayerNormLinear (layernorm_linear.py)
  ├── LayerNormMLP (layernorm_mlp.py)
  ├── RMSNorm (rmsnorm.py) — delegate to ops.basic.rmsnorm
  └── GroupedLinear (grouped_linear.py)

Float8Quantizer (float8_tensor.py) → tex.quantize() → C++ kernel
Float8CurrentScalingQuantizer → tex.quantize() → C++ kernel
MXFP8Quantizer (mxfp8_tensor.py) → block-wise scaling
NVFP4Quantizer (nvfp4_tensor.py) → 2-level quantization

FP8GlobalStateManager (quantization.py) → 全局FP8状态管理
autocast (quantization.py) → context manager开启FP8
```

## 2. FP8 Scaling Recipes详解

### 2.1 FP8 Format (recipe/__init__.py)

```python
class Format:
    E4M3  = auto()  # max=448, forward专用, 精度高
    E5M2  = auto()  # max=57344, backward专用, 范围大
    HYBRID = auto()  # forward=E4M3 + backward=E5M2
    E2M1  = auto()  # max=6, NVFP4专用(4bit)
```

**关键**: E4M3有5bit mantissa→精度好但范围小; E5M2有2bit mantissa→范围大但精度差→HYBRID策略最优!

### 2.2 DelayedScaling (最经典, RTX 4090可用)

```python
class DelayedScaling:
    margin: int = 0
    fp8_format: Format = Format.HYBRID
    amax_history_len: int = 1024  # amax历史窗口长度
    amax_compute_algo: str = "max"  # 从历史取max

    # 核心公式:
    # new_scaling_factor = (FP8_MAX / amax) / (2 ^ margin)
    # margin=0 → scale精确适配amax → 最大化FP8范围利用
    # margin>0 → scale更保守 → 减少溢出风险但损失精度
```

**工作流程**:
1. 前一步: 计算amax (tensor最大绝对值)
2. 从amax_history窗口取max → 当前amax
3. 计算new_scale = FP8_MAX / amax / (2^margin)
4. 当前步: 用new_scale量化输入 → FP8 GEMM
5. 量化过程中更新amax → 下一步循环

**为什么是"Delayed"?** 因为当前步用的是**上一步**的scale factor → scale更新有1步延迟。这是合理的: 量化需要scale→但scale依赖amax→amax在量化中计算→循环依赖→用上一步的scale打破循环!

### 2.3 Float8CurrentScaling (RTX 4090可用)

```python
class Float8CurrentScaling:
    fp8_format: Format = Format.E4M3
    use_split_accumulator: dict  # 控制GEMM累加器精度

    # 核心区别: 不用历史窗口!
    # 直接扫描当前tensor的amax → 计算scale → 量化
    # scale = FP8_MAX / current_amax
```

**优势**: 无延迟→scale更精确→精度更好
**劣势**: 每步需额外扫描amax→轻微开销→但比DelayedScaling更准确

### 2.4 MXFP8BlockScaling (RTX 4090 ❌, Blackwell+)

```python
class MXFP8BlockScaling:
    # 32-value blocks, E8M0 power-of-2 scales
    # scale = 2^exp (8bit exponent, 0 mantissa)
    # → 精确对齐, hardware-native on Blackwell
```

RTX 4090 (SM89)不支持MXFP8→需SM100+(Blackwell)

### 2.5 Float8BlockScaling (RTX 4090 ❌, Hopper+需CUDA 12.9)

```python
class Float8BlockScaling:
    # 1D/2D block scaling with FP32 scales
    # 比MXFP8更灵活(FP32 scales vs E8M0)
    # 需: compute_capability >= 9.0 AND CUDA >= 12.9
```

RTX 4090是SM89(>=9.0)但我们的CUDA是12.8(< 12.9)→不可用!

### 2.6 NVFP4BlockScaling (RTX 4090 ❌, Blackwell+)

```python
class NVFP4BlockScaling:
    # 2-level quantization:
    # Level 1: E4M3 (16-value blocks) + global FP32 super-scale
    # Level 2: random Hadamard transforms + stochastic rounding
    # 4over6 adaptive quantization
    # → 4bit权重! 2x compression over FP8
```

NVFP4需要SM100+(Blackwell)→RTX 4090完全不可用

### 2.7 RTX 4090 Recipe可用性总结

| Recipe | SM89可用 | CUDA 12.8可用 | 结论 |
|--------|----------|--------------|------|
| DelayedScaling | ✅ | ✅ | **可用** |
| Float8CurrentScaling | ✅ | ✅ | **可用** |
| MXFP8BlockScaling | ❌ (需SM100+) | - | 不可用 |
| Float8BlockScaling | ✅ (SM89≥9.0) | ❌ (需12.9) | 不可用 |
| NVFP4BlockScaling | ❌ (需SM100+) | - | 不可用 |

**结论**: RTX 4090只能用DelayedScaling和Float8CurrentScaling两种recipe!

## 3. FP8 Linear Layer详解 (linear.py)

### 3.1 Forward流程

```python
def _linear_forward_impl(args):
    # 1. Quantize input (if FP8)
    if fp8:
        input_quantizer.set_usage(rowwise=True, columnwise=bwd_needs_input)
        inputmat = input_quantizer(inputmat)  # tex.quantize() → C++ kernel

    # 2. Quantize weight (if FP8)
    if fp8:
        weight_quantizer.set_usage(rowwise=True, columnwise=dgrad_needs_weight)
        weightmat, new_ws = quantize_weight(weight, weight_quantizer, workspace)

    # 3. FP8 GEMM: y = x * w^T (fused quantize+GEMM+dequantize)
    gemm_out = general_gemm(
        weightmat,         # FP8 weight
        inputmat_total,    # FP8 input
        quantization_params=output_quantizer,  # optional output quantizer
        out_dtype=activation_dtype,  # dequantize to BF16/FP32
        use_split_accumulator=...,
    )

    # 4. Output → high precision (BF16/FP32)
    # general_gemm内部: FP8输入→FP8 GEMM→高精度输出
```

**关键**: `general_gemm`是一个C++ extension kernel, 它:
1. 接收FP8输入和权重
2. 在FP8精度下做矩阵乘法
3. 结果累加器可以选择: FP32(高精度) 或 split-accumulator模式
4. 输出dequantize到`activation_dtype`(BF16/FP32)

### 3.2 Backward流程

```python
def _linear_backward(args):
    # 1. Quantize grad_output (if FP8)
    if grad_output_quantizer:
        grad_output_quantizer.set_usage(rowwise=True, columnwise=True)
        # rowwise: for dgrad GEMM (dx = dy * w)
        # columnwise: for wgrad GEMM (dw = dy^T * x)

    # 2. dgrad GEMM: dx = dy * w (FP8 grad_output × FP8 weight)
    dgrad = general_gemm(
        weight_fp8,        # FP8 weight (columnwise data)
        grad_output,       # FP8 grad_output (rowwise data)
        layout="NN", grad=True,
        quantization_params=grad_input_quantizer,
    )

    # 3. wgrad GEMM: dw = dy^T * x (FP8 grad_output × FP8 input)
    wgrad = general_gemm(
        inputmat_total,    # FP8 input (columnwise data)
        grad_output,       # FP8 grad_output (columnwise data)
        layout="NT", grad=True,
        quantization_params=grad_weight_quantizer,
    )
```

**rowwise vs columnwise**: 这是TE的关键创新!

```
Forward:
  input: rowwise量化 → x[i,:] 按|row_max|缩放
  weight: rowwise量化 → w[j,:] 按|row_max|缩放
  → y = x * w^T 需要 rowwise输入 × rowwise权重

Backward:
  dgrad(dx = dy * w): 需要 rowwise dy × columnwise w
    columnwise w: w[:,j] 按|column_max|缩放 → 转置友好
  wgrad(dw = dy^T * x): 需要 columnwise dy × columnwise x
    columnwise数据用于layout="NT"的GEMM
```

TE的QuantizedTensorStorage同时存储rowwise和columnwise两种量化数据 → backward无需重新量化!

### 3.3 Quantizer角色

```python
class QuantizerRole:
    module_type: str  # "linear", "grouped_linear", "dpa"
    tensor_type: str  # "input", "weight", "grad_output"
    name: str         # "qkv", "proj", "fc1", "fc2"

# Linear module的6个quantizer:
# 1. input_quantizer: 量化输入(for forward GEMM)
# 2. weight_quantizer: 量化权重(for forward/dgrad GEMM)
# 3. output_quantizer: 量化输出(可选, for fp8_output)
# 4. grad_input_quantizer: 量化dgrad(可选, for fp8_grad)
# 5. grad_weight_quantizer: 量化wgrad(可选)
# 6. grad_output_quantizer: 量化grad_output(for backward GEMM)
```

## 4. FP8 Quantizer实现 (float8_tensor.py)

### 4.1 Float8Quantizer (DelayedScaling)

```python
class Float8Quantizer(Quantizer):
    scale: torch.Tensor  # scaling factor
    amax: torch.Tensor   # max absolute value

    def quantize_impl(self, tensor):
        return tex.quantize(tensor, self)  # C++ kernel: quantize + compute amax

    def update_quantized(self, src, dst, noop_flag=None):
        tex.quantize(src, self, dst, noop_flag)  # 增量更新已有FP8 tensor
```

**tex.quantize()** 是C++ extension kernel:
- 输入: 高精度tensor(BF16/FP32) + quantizer(scale, amax)
- 输出: FP8 tensor + 更新amax
- 实现: `value_fp8 = round(value * scale)` → 存为uint8/float8_e4m3

### 4.2 Float8CurrentScalingQuantizer

```python
class Float8CurrentScalingQuantizer(Quantizer):
    # 不需要scale和amax初始化!
    # 每次量化时直接计算当前tensor的amax
    # force_pow_2_scales: 是否强制2^n scale
    # amax_epsilon: amax下限(防止scale过大)
    # with_amax_reduction: TP时是否跨卡reduce amax
```

**优势**: 无delay→scale更精确→FP8精度更高
**TP支持**: with_amax_reduction=True → 跨TP组AllReduce amax → 所有卡用相同scale

## 5. autocast机制 (quantization.py)

```python
@contextmanager
def autocast(
    enabled: bool = True,
    recipe: Recipe = None,
    fp8_group: dist_group_type = None,
):
    # 1. 进入: 设置FP8GlobalStateManager
    FP8GlobalStateManager.set_fp8_enabled(enabled)
    FP8GlobalStateManager.set_fp8_recipe(recipe or get_default_recipe())

    # 2. 包裹forward: 所有TE模块自动启用FP8
    # TE模块在forward时检查FP8GlobalStateManager.is_fp8_enabled()
    # → 如果True, 使用quantizer量化输入/权重 → FP8 GEMM

    # 3. 退出: 清理FP8状态
    FP8GlobalStateManager.set_fp8_enabled(False)
```

**核心**: autocast是context manager, 所有在autocast范围内的TE模块自动使用FP8。类似PyTorch的`torch.autocast("cuda", dtype=torch.bfloat16)`。

### 5.1 FP8GlobalStateManager

```python
class FP8GlobalStateManager:
    quantization_state = FP8GlobalState()  # 全局单例

    # 关键功能:
    # - is_fp8_enabled(): 是否开启FP8
    # - get_fp8_recipe(): 当前recipe
    # - is_first_fp8_module(): 是否是第一个FP8模块(for amax reduce)
    # - reduce_and_update_fp8_tensors(): 全局amax reduce+scale update
    # - with_fp8_parameters(): 是否权重本身就是FP8
```

**DelayedScaling的scale更新流程**:
1. 每个FP8 module forward→量化→计算局部amax
2. backward结束时→`FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)`
3. 全局amax AllReduce → 取max → 计算new_scale → 更新所有module的scale

## 6. RTX 4090实测数据连接

### 6.1 我们的FP8实验结论 vs TE实现

| 我们的发现 | TE的实现 | 连接 |
|-----------|---------|------|
| FP8 dequant Python慢1.5-2.5x | TE用fused C++ kernel | TE避免了我们的Python overhead! |
| INT4 weight-only 0.87-1.08x | TE的FP8 weight quantize | FP8比INT4更适合GPU(tensor core) |
| INT8 KV 1.00x+50%内存省 | TE的FP8 activation | FP8 activation同样省内存 |

**关键差异**: 我们之前的FP8 benchmark用**Python-level dequantize** → 慢。TE用**fused C++ kernel** (quantize+GEMM+dequantize一体) → 无Python overhead → 实际提速!

### 6.2 FP8 GEMM on RTX 4090

RTX 4090 (SM89=Ada)的FP8支持:
- cuBLASLt支持FP8 GEMM (需≥12.1.3)
- HMMA tensor core支持FP8输入(E4M3/E5M2)
- 但不支持: TMA/WGMMA/Cluster(这些是SM90 Hopper专属)

**FP8 GEMM流程** (RTX 4090):
```
FP8 input (E4M3) → HMMA (FP8→FP32 accumulator) → dequantize output to BF16
FP8 weight (E4M3) → HMMA
```

**split_accumulator**: TE提供选项控制GEMM累加器精度
- 默认: FP32累加器 → 高精度但稍慢
- split_accumulator=True: 更大累加器 → 减少精度损失

## 7. Userbuffers (comm+GEMM overlap)

### 7.1 通信+计算重叠

TE支持Userbuffers机制重叠tensor-parallel通信和GEMM计算:

```python
# Column parallel (fprop):
ub_overlap_ag: AllGather input + fprop GEMM overlap
ub_overlap_rs: ReduceScatter output + fprop GEMM overlap

# Row parallel (fprop):
ub_overlap_rs: ReduceScatter output + fprop GEMM overlap

# Backward overlaps:
ub_overlap_ag_dgrad: AllGather grad_output + dgrad GEMM overlap
ub_overlap_rs_dgrad: ReduceScatter dgrad + dgrad GEMM overlap
ub_bulk_dgrad: AllGather input + dgrad GEMM overlap
ub_bulk_wgrad: ReduceScatter dgrad + wgrad GEMM overlap
```

**RTX 4090限制**: Userbuffers需要NVLink→RTX 4090只有PCIe→重叠可能无收益或负收益(类似我们之前Ring Attention PCIe的结论: comm开销>compute重叠收益)

## 8. 与CUTLASS/FlashAttention的连接

### 8.1 TE的GEMM backend选择

TE的`general_gemm`内部使用多种backend:
- **cuBLASLt**: FP8 GEMM on Ada/Hopper/Blackwell
- **cuBLASMp**: multi-process GEMM (for comm overlap)
- **CUTLASS**: custom GEMM kernel (for specific layouts/recipes)

### 8.2 与FlashAttention-3的关联

TE的FP8 DPA (Dot Product Attention, beta) 使用FP8 attention:
- FP8 QKV input → FP8 QK^T → FP8 softmax → FP8 PV
- 类似FlashAttention-3的FP8 attention (Hopper TMA+WGMMA)
- RTX 4090不支持FA-3→也不支持TE的FP8 DPA高级特性

## 9. Production决策 (RTX 4090)

| 场景 | 最佳配置 | 原因 |
|------|---------|------|
| Training 7B | DelayedScaling FP8 | 经典recipe, RTX 4090完全支持 |
| Training 7B (精度优先) | Float8CurrentScaling | 无delay→更精确 |
| Training 25M | 不用FP8 | 小模型量化收益不显著(1.02x) |
| Inference 7B | INT4 weight-only + INT8 KV | 推理不训练→TE不是最优选择 |
| FP8 communication | DelayedScaling FP8 | 通信量化→带宽减半 |

### 9.1 TE vs vLLM/SGLang

- **TE**: 训练场景 → FP8训练加速 → fused quantize+GEMM → autocast context manager
- **vLLM/SGLang**: 推理场景 → PagedAttention + continuous batching → INT8 KV cache
- **重叠**: TE可用于推理但不是最优(推理更关注延迟而非训练吞吐)

---

**源码**: `transformer-engine/` (NVIDIA, shallow clone)
**关键文件**:
- `transformer_engine/common/recipe/__init__.py` — 5种recipe定义
- `transformer_engine/pytorch/module/linear.py` — FP8 Linear层(2170行!)
- `transformer_engine/pytorch/module/base.py` — 基类+quantize_weight+FP8 meta管理
- `transformer_engine/pytorch/quantization.py` — autocast+FP8GlobalStateManager
- `transformer_engine/pytorch/tensor/float8_tensor.py` — Float8Quantizer+Float8CurrentScalingQuantizer
- `transformer_engine/pytorch/tensor/mxfp8_tensor.py` — MXFP8Quantizer(RTX4090不可用)
- `transformer_engine/pytorch/tensor/nvfp4_tensor.py` — NVFP4Quantizer(RTX4090不可用)

**相关笔记**: `quantization-inference-rtx4090.md`, `flashattention2-kernel-internals-vs-triton.md`, `cutlass-gemm-rtx4090.md`