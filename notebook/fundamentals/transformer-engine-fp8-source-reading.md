# Transformer Engine (TE) FP8 Training Source Code Deep Dive — FP8 Recipes (DelayedScaling+CurrentScaling+MXFP8+NVFP4) + Float8Quantizer + FP8GlobalStateManager + autocast + quantized_model_init + TE.Linear FP8 GEMM + Userbuffers comm+GEMM overlap + TE TransformerLayer + RTX 4090

> 2026-06-13 | NVIDIA Transformer Engine源码全链路分析: FP8 Recipe体系(DelayedScaling: amax_history+margin+scale计算+all-reduce amax+HYBRID E4M3fwd/E5M2bwd; Float8CurrentScaling: per-tensor即时计算; MXFP8BlockScaling: Blackwell专用; NVFP4BlockScaling: Blackwell FP4)+Float8Quantizer(scale+amax+FP8 dtype+tex.quantize kernel)+Float8CurrentScalingQuantizer(即时amax+无history)+FP8GlobalStateManager(全局状态+amax buffer+scale buffer+reduce+update+fused kernel)+autocast上下文管理器(enter→enable FP8+exit→reduce_and_update)+quantized_model_init(参数FP8初始化+preserve_high_precision)+TE.Linear(LinearFwdArgs+input_quantizer+weight_quantizer+general_gemm+FP8 GEMM流程+TP/SP通信+Userbuffers overlap)+RTX 4090 FP8训练
> 源码: github.com/NVIDIA/TransformerEngine — transformer_engine/pytorch/quantization.py, module/linear.py, module/base.py, common/recipe/, tensor/float8_tensor.py
> 关联: cuda-cutlass.md, fp8-quantization.md, distributed-nav SKILL

## 0. 核心定律: FP8量化=scale×cast + delayed scaling=历史窗口 + 自适应recipe

```
TE FP8训练架构层次:

Recipe层 (策略选择):
  → DelayedScaling → Ada/Hopper → amax_history窗口(1024步) → scale=(FP8_MAX/amax)/2^margin
  → Float8CurrentScaling → Hopper+ → 即时扫描input→amax→scale → 无历史 → 更精确!
  → MXFP8BlockScaling → Blackwell(B100/H200) → per-block MX格式 → 精度更高!
  → NVFP4BlockScaling → Blackwell → FP4(2bit!) → 极端压缩 → 4x比FP8!
  → → → RTX 4090: DelayedScaling唯一可用! → SM 8.9(Ada) → FP8支持!

Quantizer层 (量化执行):
  → Float8Quantizer → per-tensor delayed scaling → tex.quantize(CUDA kernel) → scale×cast
  → Float8CurrentScalingQuantizer → per-tensor current scaling → 即时amax → 无需history!
  → MXFP8Quantizer → per-block MX → block-level scale → 更细粒度!
  → → → 每Quantizer: scale+amax+dtype → quantize_impl → tex.quantize → GPU kernel!

StateManager层 (全局协调):
  → FP8GlobalStateManager → 全局FP8状态 → fp8_enabled + fp8_recipe + fp8_group
  → → global_amax_buffer + global_scale_buffer → 所有模块的amax/scale注册!
  → → reduce_and_update_fp8_tensors → exit时批量reduce+update → fused kernel!
  → → autocast_enter/exit → 上下文管理 → 嵌套支持(autocast_depth计数)

Module层 (FP8训练):
  → TE.Linear → FP8 GEMM → input_quantizer→weight_quantizer→general_gemm→output_quantizer
  → → LinearFwdArgs → dataclass → 所有forward参数打包 → 清晰!
  → → Userbuffers → comm+GEMM overlap → overlap all-gather/reduce-scatter with GEMM!
  → → TP/SP/FSDP2 → 分布式训练兼容 → gather+scatter+quantize!

数据流:
  BF16 input → input_quantizer(BF16→FP8) → FP8 GEMM(cublasLt) → FP8 output → dequantize→BF16
  → → backward: BF16 grad → grad_quantizer → FP8 GEMM → dequantize → BF16 grad
  → → → amax: 每步计算最大绝对值 → 更新history → 选择amax → 计算scale → 下步用!
```

## 1. FP8 Recipe体系 — common/recipe/__init__.py

```
### 1.1 Format枚举 — FP8数据格式

class Format(Enum):
  → E4M3 → max=448 → forward专用 → 高精度 → 3bit指数+4bit尾数
  → E5M2 → max=57344 → backward专用 → 高动态范围 → 5bit指数+2bit尾数
  → HYBRID → fwd=E4M3(448), bwd=E5M2(57344) → 默认! → 最佳训练精度!
  → E2M1 → max=6 → FP4格式 → NVFP4专用 → 2bit指数+1bit尾数 → 极端压缩!

关键设计:
  → HYBRID → forward用E4M3(精度好) → backward用E5M2(动态范围大) → 分工!
  → → E4M3: 4bit尾数→精度≈2^-4≈0.06 → 接近FP16 → forward够用!
  → → E5M2: 2bit尾数→精度≈2^-2≈0.25 → 粗糙 → 但backward梯度范围大→需要!
  → → → 纯E5M2训练不支持! → forward精度不够 → 灾难!

### 1.2 DelayedScaling — Ada/Hopper默认recipe

@dataclass DelayedScaling(Recipe):
  → margin: int = 0 → scale计算margin → 防溢出 → 2^margin因子
  → fp8_format: Format = Format.HYBRID → 默认HYBRID → fwd E4M3 + bwd E5M2
  → amax_history_len: int = 1024 → amax历史窗口 → 默认1024步 → 选最大!
  → amax_compute_algo: str|Callable = "max" → "max"选history最大 → "most_recent"选最新
  → scaling_factor_compute_algo: Callable = None → 自定义scale计算 → 默认=None用标准公式
  → reduce_amax: bool = True → 跨rank all-reduce amax → 保证scale一致!
  → fp8_dpa: bool = False → FP8 dot product attention → Beta功能
  → fp8_mha: bool = False → FP8 multi-head attention → 完整FP8注意力!
  → backward_override: str = None → backward精度控制 → DelayedScaling只支持None

DelayedScaling核心公式:
  → scale = (FP8_MAX / amax) / (2 ^ margin)
  → → FP8_MAX = 448(E4M3) 或 57344(E5M2) → 取目标format的max
  → → amax = max(history_window) → 1024步中最大的amax → 防近期溢出!
  → → → margin=0 → scale=FP8_MAX/amax → 精确利用FP8范围
  → → → margin=1 → scale=FP8_MAX/(2×amax) → 留2x安全余量 → 防溢出!

DelayedScaling工作流:
  1. 前步: 用旧scale量化 → BF16→FP8 → GEMM → 结果→BF16
  2. 本步: 计算新amax → 记录到amax_history → shift窗口 → 新amax入队!
  3. 选择: amax_compute_algo → "max"选1024步最大 → "most_recent"选当前
  4. 计算: scale=(FP8_MAX/amax)/2^margin → 新scale → 下步用!
  5. reduce: reduce_amax=True → all-reduce amax across ranks → scale一致!

### 1.3 Float8CurrentScaling — Hopper+即时recipe

@dataclass Float8CurrentScaling(Recipe):
  → fp8_format: Format = Format.HYBRID → 同DelayedScaling
  → backward_override: str|None → "high_precision"/"dequantized" → backward精度控制!

CurrentScaling核心区别:
  → 不用amax_history → 即时扫描input → 计算当前amax → 直接scale!
  → → 优势: scale更精确 → 精确利用FP8范围 → 无历史窗口开销!
  → → → 劣势: 每步都计算amax → 额外GPU开销 → 但小(一次aminmax)!
  → → → → 适用: Hopper+ → SM 90+ → cublasLt版本够 → Ada也支持!

### 1.4 MXFP8BlockScaling — Blackwell专用

MXFP8BlockScaling(Recipe):
  → per-block MX格式 → 32元素一个block → 每block独立scale → 精度更高!
  → → 需要compute_capability ≥ 10.0 → Blackwell(B100/H200) → RTX 4090不支持!

### 1.5 NVFP4BlockScaling — Blackwell FP4

NVFP4BlockScaling(Recipe):
  → FP4格式 → E2M1 → 2bit数据 → 极端压缩 → 4x比FP8 → 带superblock scale
  → → 需要compute_capability ≥ 10.0 → Blackwell → RTX 4090不支持!

### 1.6 Recipe选择逻辑 (RTX 4090关键!)

get_default_fp8_recipe():
  → Blackwell(B100+): MXFP8BlockScaling → block-level精度最优!
  → Blackwell架构限制(12.0+): Float8CurrentScaling → MXFP8部分GEMM不支持!
  → 其他(SM≥8.9): DelayedScaling → Ada/Hopper → 默认!
  → → RTX 4090(SM 8.9): DelayedScaling → 唯一可用recipe!
  → → → 需要: CUDA≥12.1 + cublasLt≥12.1.3 → 满足条件!

FP8支持检查:
  → SM < 8.9(pre-Ada): 不支持 → 需要Ada+架构!
  → SM ≥ 9.0(Hopper+): 支持 → 无额外要求!
  → SM = 8.9(Ada/RTX 4090): 需要 CUDA≥12.1 + cublasLt≥12.1.3!
  → → RTX 4090: 满足条件 → FP8可用! → 但只DelayedScaling!
```

## 2. Float8Quantizer + Float8CurrentScalingQuantizer — tensor/float8_tensor.py

```
### 2.1 Float8Quantizer — DelayedScaling量化器

class Float8Quantizer(Quantizer):
  → scale: torch.Tensor → 量化缩放因子 → per-tensor → 1/tensor一个scale!
  → amax: torch.Tensor → 最大绝对值 → per-tensor → 量化时记录!
  → dtype: DType → FP8数据类型 → kFloat8E4M3或kFloat8E5M2
  → rowwise: bool → 行方向量化 → forward用!
  → columnwise: bool → 列方向量化 → backward用! → weight需要双向!

核心方法:
  → quantize_impl(tensor) → tex.quantize(tensor, self) → CUDA kernel → scale×cast→FP8!
  → update_quantized(src, dst) → 更新已有FP8 tensor → tex.quantize → 不创建新tensor!
  → calibrate(tensor) → amin,amax = tensor.aminmax() → 记录amax → 更新!
  → create_tensor_from_data(data) → 从FP8 uint8数据创建Float8Tensor → scale_inv=1/scale!

量化流程:
  → BF16 tensor → scale × tensor → cast to FP8(E4M3/E5M2) → Float8Tensor!
  → → scale = (FP8_MAX / amax) / 2^margin → DelayedScaling公式!
  → → → tex.quantize → CUDA kernel → 一个kernel完成scale+cast → 高效!

### 2.2 Float8CurrentScalingQuantizer — CurrentScaling量化器

class Float8CurrentScalingQuantizer(Quantizer):
  → dtype: DType → FP8数据类型
  → with_amax_reduction: bool → 跨rank reduce amax → 保证一致!
  → amax_reduction_group: ProcessGroup → reduce的group
  → force_pow_2_scales: bool → scale强制2的幂 → 更快dequantize!
  → amax_epsilon: float = 0.0 → amax最小值 → 防scale过大!

关键区别(vs Float8Quantizer):
  → 不需要scale和amax初始化 → 即时计算 → GPU buffer填充!
  → → quantize_impl → tex.quantize → kernel内部计算amax+scale → 一步!
  → → → 无amax_history → 无延迟 → scale更精确 → 但每步额外开销!
  → calibrate → 不需要 → return → current scaling不预校准!

### 2.3 Float8Tensor — FP8数据容器

class Float8Tensor(QuantizedTensor):
  → _data: torch.Tensor → FP8 uint8数据 → 实际FP8存储!
  → _scale_inv: torch.Tensor → 1/scale → dequantize用! → scale_inv × FP8 → BF16
  → _fp8_dtype: DType → E4M3或E5M2 → 知道如何解释数据!
  → _quantizer: Quantizer → 量化器引用 → 可重新量化!
  → dtype → fake_dtype → 逻辑dtype → BF16/FP32 → 不实际存储!
  → shape → 原始shape → dequantize后恢复!

关键设计:
  → _data存FP8 → _scale_inv存反scale → dequantize=_data×_scale_inv → BF16!
  → → 注意: scale_inv=1/scale → 不是scale! → 因为dequantize=FP8×scale_inv → 方便!
  → → → Float8TensorStorage → 内部存储类 → 分离数据和元数据!
  → → → → FSDP2兼容: _ops_to_preserve_subclass_in_fsdp2 → torch.compile友好!

Float8Tensor dequantize:
  → dequantize() → _data.float() × _scale_inv → BF16 → 高精度恢复!
  → → 精度损失: BF16→FP8→BF16 → E4M3精度≈2^-4→约6%误差 → 可接受训练!
  → → → 关键: scale_inv精确 → FP8×scale_inv≈原始值 → 量化误差≈1/2^尾数位!
```

## 3. FP8GlobalStateManager — quantization.py

```
### 3.1 FP8GlobalState — 全局状态

@dataclass FP8GlobalState:
  → fp8_enabled: bool → 是否在FP8 autocast区域!
  → fp8_calibration: bool → 校准模式 → 记录amax但不量化!
  → fp8_recipe: Recipe → 当前recipe → DelayedScaling/CurrentScaling/etc!
  → fp8_distributed_group: ProcessGroup → amax reduction的分布式组!
  → fp8_parameters: bool → 参数是否FP8存储 → quantized_model_init设!
  → is_first_fp8_module: bool → 第一个FP8模块 → 控制reduce时机!
  → fp8_graph_capturing: bool → CUDA graph模式 → 特殊处理!
  → autocast_depth: int → 嵌套深度 → 支持嵌套autocast!
  → global_amax_buffer: Dict → 所有模块的amax → 批量reduce!
  → global_amax_history_buffer: Dict → 所有模块的amax_history!
  → global_scale_buffer: Dict → 所有模块的scale!

关键设计:
  → 全局单例 → 所有TE模块共享状态 → 统一管理FP8!
  → → global_amax_buffer → 每模块注册 → autocast_exit时批量reduce!
  → → → key = "forward_recipe=...,group=..." → 按recipe+group分区!
  → → → → 嵌套autocast → autocast_depth计数 → 嵌套不重复reduce!

### 3.2 FP8GlobalStateManager方法

add_fp8_tensors_to_global_buffer(fp8_meta):
  → 每模块调用一次 → 注册amax+scale到全局buffer!
  → → 按forward/backward区分 → fwd和bwd分别buffer!
  → → → DelayedScaling专用 → CurrentScaling不需要(无history)!
  → → → → 返回buffer_index → 模块知道自己在buffer的位置!

is_first_fp8_module():
  → 返回True只一次 → 第一次调用 → 用于触发初始化!
  → → 逻辑: tmp=is_first → is_first=False → return tmp → 只True一次!

reduce_and_update_fp8_tensors(forward=True):
  → autocast_exit时调用 → 批量处理所有模块的amax+scale!
  → → 1. 拼接所有amax → torch.cat(amax_buffer) → contiguous → 单tensor!
  → → 2. all-reduce → ReduceOp.MAX → 跨rank最大amax → scale一致!
  → → 3. split → split_and_copy → 分回各模块 → 各模块更新自己amax!
  → → 4. fused update → tex.fused_amax_and_scale_update_after_reduction → fused kernel!
  → → → → 或 unfused update → _amax_and_scale_update → Python计算 → 慢但灵活!
  → → → → → 默认: fused → 一个CUDA kernel → 拼接+reduce+split+update → 极快!

### 3.3 autocast上下文管理器

class autocast:
  → __enter__ → 保存旧状态 → FP8GlobalStateManager.autocast_enter → 设置新状态!
  → → autocast_enter: fp8_enabled=True → fp8_recipe=recipe → is_first_fp8_module=True
  → → → check_recipe_support → 检查硬件支持 → RTX 4090→DelayedScaling→OK!
  → → → → autocast_depth += 1 → 支持嵌套!

  → __exit__ → FP8GlobalStateManager.set_autocast_state(旧状态) → autocast_exit!
  → → autocast_exit: autocast_depth -= 1 → depth==0 → reduce_and_update_fp8_tensors!
  → → → → 关键: exit时才reduce → 不是每模块 → 集中处理 → 效率!
  → → → → → reduce_amax=True → all-reduce → 所有rank的scale一致!

### 3.4 quantized_model_init上下文

quantized_model_init(enabled=True, recipe=None, preserve_high_precision_init_val=False):
  → 模型初始化时进入 → 参数只存FP8副本 → 不存高精度 → 省内存!
  → → fp8_parameters=True → TE.Linear初始化时 → weight只存FP8 → 不存BF16!
  → → → 省内存: 7B BF16=14GB → 7B FP8=1.75GB → 省8x → 但精度有损!
  → → → → preserve_high_precision_init_val=True → 保存高精度初始值在CPU → 用于optimizer!

LoRA场景:
  → quantized_model_init(enabled=True) → 主权重FP8 → LoRA BF16 → 精度够!
  → → 省内存: 7B FP8权重=1.75GB → LoRA=0.5GB → 总≈2.25GB → 极省!
  → → → 但! RTX 4090推理主要省内存 → 训练LoRA本来就小 → 收益有限!

### 3.5 Activation Recompute (FP8专用)

FP8 activation recompute:
  → Phase 1 forward → 记录FP8 activation → stash amax+scale
  → Phase 2 forward(recompute) → 用stashed amax+scale → 数值一致!
  → → copy_forward_fp8_meta_tensors_for_recompute → stash到buffer!
  → → get_old_fp8_meta_tensors_for_recompute → restore stashed → Phase 2用!
  → → → restore_fp8_meta_tensors → Phase 2后恢复最新 → 不丢失!

  → → → 关键: Phase 1和Phase 2必须数值一致 → 否则梯度错误!
  → → → → 方法: stash Phase 1的amax+scale → Phase 2用 → 保证一致!
```

## 4. TE.Linear — FP8 GEMM执行 (module/linear.py)

```
### 4.1 LinearFwdArgs — Forward参数打包

@dataclass LinearFwdArgs:
  → weight/bias/inp → 基础tensor
  → input_quantizer/weight_quantizer/output_quantizer → FP8量化器!
  → grad_input_quantizer/grad_weight_quantizer/grad_output_quantizer → backward量化器!
  → fp8/fp8_calibration/fp8_output → FP8配置
  → is_first_microbatch/cache_weight/skip_fp8_weight_update → weight缓存!
  → parallel_mode/tp_group/tp_size → Tensor Parallel!
  → ub_overlap_ag/rs_fprop → Userbuffers overlap! → comm+GEMM重叠!
  → fsdp_group/is_fsdp2 → FSDP2兼容!

关键: Quantizer角色(QuantizerRole) → module_type+tensor_type+name → 精细控制!
  → module_type="linear" → tensor_type="input"/"weight"/"grad_output" → 量化什么!
  → → name="qkv"/"proj"/"fc1"/"fc2" → 模块名 → 按名定制recipe!

### 4.2 FP8 Forward流程

_linear_forward_impl(args):
  1. Input准备:
     → FP8: input_quantizer(inputmat) → BF16→FP8 → quantize → Float8Tensor!
     → → rowwise=True, columnwise=(backward需要) → 双向量化!
     → → TP column parallel: all-gather input → gather_along_first_dim → 通信!
     → → → UB overlap: fill_userbuffers_buffer_for_all_gather → overlap AG+GEMM!

  2. Weight准备:
     → FP8: weight_quantizer(weight) → BF16→FP8 → quantize → Float8Tensor!
     → → quantize_weight → workspace缓存 → is_first_microbatch时quantize → 后续缓存!
     → → → FSDP2: _fsdp_gather_tensors → 聚合分片权重 → 再quantize!

  3. GEMM:
     → general_gemm(weightmat, inputmat_total, ...) → FP8 GEMM → cublasLt!
     → → use_split_accumulator → FP8快速累加 → Hopper/Ada支持!
     → → → output: activation_dtype(BF16) → FP8 GEMM→BF16输出 → dequantize内置!
     → → → → UB overlap: reduce_scatter_out → overlap RS+GEMM → 通信计算重叠!

  4. Output:
     → gemm_out → BF16 → 返回 → 下层用!
     → → FP8 output: output_quantizer → FP8输出 → 下一层直接FP8 → 端到端FP8!

### 4.3 FP8 Backward流程

_linear_backward_impl(grad_output, ...):
  1. grad_output量化:
     → grad_output_quantizer(grad_output) → BF16→FP8 → E5M2!

  2. Input梯度(dgrad):
     → general_gemm(weightmat_T, grad_output_FP8) → FP8 GEMM → dgrad!
     → → weight columnwise数据 → Transpose → FP8×FP8 → BF16输出!
     → → → UB overlap: overlap RS dgrad → reduce-scatter + dgrad GEMM → 重叠!

  3. Weight梯度(wgrad):
     → general_gemm(grad_output_T, inputmat_FP8) → FP8 GEMM → wgrad!
     → → → fuse_wgrad_accumulation → fused wgrad → 不单独step → 累加!
     → → → → 2X_ACC_WGRAD=True → FP8 wgrad用split accumulator → 更精确!

### 4.4 Userbuffers — comm+GEMM重叠

initialize_ub(shape, tp_size, ...):
  → 初始化Userbuffers → 分配通信buffer → overlap通信与计算!
  → → CommOverlapType.RS → reduce-scatter overlap → 发送同时计算!
  → → CommOverlapType.AG → all-gather overlap → 接收同时计算!
  → → → pipeline方法 → split→send chunk→compute→send next→overlap!
  → → → → ring_exchange方法 → 逐ring发送→逐ring计算→极细粒度重叠!

RTX 4090 Userbuffers:
  → 无NVLink → 无multicast → UB_SKIPMC=1 → 用CUDA IPC!
  → → PCIe带宽有限 → overlap收益小 → 但TP=2时可能有帮助!
  → → → 建议: RTX 4090不用Userbuffers → TP=1最优 → 无通信!

### 4.5 TE.Linear vs PyTorch Linear对比

```
特性         PyTorch nn.Linear      TE.Linear
权重精度      FP32/BF16             BF16+FP8(scale×cast)
前向GEMM      cuBLAS BF16           cublasLt FP8
输入          BF16                   BF16→FP8(quantize)
输出          BF16                   BF16(dequantize)
backward      BF16 GEMM              FP8 GEMM(grad_quantize)
TP兼容        无                     ColumnParallel/RowParallel
FSDP兼容      无                     FSDP2(_fsdp_gather/scatter)
通信重叠      无                     Userbuffers(AG/RS overlap)
scale更新     无                     DelayedScaling(amax→scale)
内存          BF16×2(params)         FP8×1(params)+scale
训练速度      BF16 baseline          1.48-1.59x(RTX 4090实测!)
```
```

## 5. TE TransformerLayer — transformer.py

```
class TransformerLayer(torch.nn.Module):
  → self_attn → MultiheadAttention → QKV+DPA+Proj → 注意力!
  → → LayerNorm(QKV) → te.LayerNormLinear → FP8 Linear+LN融合!
  → → DPA → dot_product_attention → FlashAttention/FusedAttention → FP8可选!
  → → Proj → te.Linear → FP8 Linear → 输出投影!
  → → → fp8_dpa=True → FP8注意力 → QKV→FP8→attention→FP8→Proj → 端到端FP8!
  → → → fp8_mha=True → FP8 MHA → LN→FP8→DPA→FP8→Proj → 无BF16中间!

  → mlp → LayerNormMLP → FC1+GeLU+FC2 → 前馈网络!
  → → te.LayerNormMLP → FP8 FC1+FC2 → GeLU → 两层FP8 GEMM!

  → → drop_path → Stochastic Depth → 随机跳过层 → 正则化!

标准TransformerLayer FP8流程:
  → input(BF16) → LayerNorm → Linear(QKV, FP8) → DPA → Linear(Proj, FP8) → MLP(FC1+FC2, FP8) → output(BF16)
  → → 每层4次FP8 GEMM → QKV+Proj+FC1+FC2 → 全FP8训练!
  → → → 实测: 7B 4层GEMM → TE FP8 = 1.48-1.59x → 4次FP8 GEMM加速!
```

## 6. RTX 4090 TE FP8分析

```
1. RTX 4090 FP8支持:
  → SM 8.9(Ada) → FP8 E4M3/E5M2支持 → cublasLt FP8 GEMM!
  → → 需要: CUDA≥12.1 + cublasLt≥12.1.3 → 满足条件!
  → → → DelayedScaling → 唯一可用recipe → amax_history=1024 → margin=0!
  → → → → HYBRID format → fwd E4M3 + bwd E5M2 → 默认最优!

2. RTX 4090 TE FP8训练实测(7B BF16 baseline):
  → B=1: 不加速 → kernel太小 → FP8无优势 → BF16更好!
  → B≥4: 1.48-1.59x加速 → FP8 GEMM比BF16快 → 带宽减少8x→计算密度提高!
  → → 原因: BF16→FP8 → 数据量减8x → 同带宽下计算密度8x → cublasLt FP8更快!
  → → → 但! BF16→FP8→BF16 → 精度损失 → E4M3≈6%误差 → 训练可接受!
  → → → → 关键: B≥4时GPU利用率高 → FP8 GEMM优势显现 → 1.48x!

3. RTX 4090 TE FP8 vs BF16内存:
  → BF16: 参数14GB + optimizer 56GB = 70GB → OOM!
  → FP8参数(quantized_model_init): 14/8=1.75GB → 省内存!
  → → 但! FP8参数精度损失 → 训练可能不稳定 → 需验证!
  → → → LoRA场景: 主权重FP8=1.75GB + LoRA=0.5GB → 总≈2.25GB → 极省!
  → → → → 全参数微调: FP8参数+BF16 optimizer → 不推荐 → 精度不够!

4. RTX 4090 TE推荐:
  → LoRA微调 → BF16训练 → TE FP8=可选加速(B≥4) → 1.48x!
  → → → 不用quantized_model_init → LoRA参数BF16 → 主权重BF16 → 简单!
  → → → → 如需加速: with te.autocast(enabled=True) → FP8 GEMM → 1.48x!
  → → → → → 但: TE FP8需要BF16→FP8→BF16 → 增加量化开销 → 小B可能不划算!

  → 全参数微调 → ZeRO-2+CPU offload → 不推荐TE FP8 → 精度风险!
  → → → 单GPU BF16 → optimizer在CPU → 14.6GB fits → 稳定!
  → → → → TE FP8全参数 → FP8参数精度不够 → 训练可能不稳定!

5. RTX 4090 TE限制:
  → 无NVLink → 无Userbuffers multicast → UB_SKIPMC=1 → IPC!
  → → 无Blackwell → MXFP8/NVFP4不支持 → 只有DelayedScaling!
  → → → FP8 GEMM小batch不加速 → B=1不如BF16 → 需B≥4!
  → → → → SM 8.9 → 不支持SM90 TMA+WGMMA → 用SM80 HMMA路径 → 比Hopper慢!
```

## 7. TE源码关键发现

```
1. autocast_exit集中reduce → 效率!
  → 不是每模块reduce → 而是exit时集中 → global_amax_buffer → 拼接+reduce+split!
  → → tex.fused_amax_and_scale_update_after_reduction → fused kernel → 极快!
  → → → vs 逐模块: N次all-reduce → 慢 → 集中: 1次all-reduce → 快N倍!

2. QuantizerRole → 精细量化控制!
  → module_type+tensor_type+name → 按角色定制 → qkv用不同recipe!
  → → CustomRecipe → factory函数 → 按QuantizerRole返回不同Quantizer → 精细!

3. Float8Tensor → FSDP2兼容!
  → _ops_to_preserve_subclass_in_fsdp2 → torch.compile+FSDP2友好!
  → → QuantizedTensorStorage → 内部存储 → 分离数据和元数据 → clean!

4. Userbuffers → comm+GEMM overlap → 最激进优化!
  → pipeline/ring_exchange → 逐chunk overlap → 极细粒度 → TP加速!
  → → 但! 需NVLink/multicast → RTX 4090不支持 → PCIe下收益有限!

5. FP8 activation recompute → 数值一致!
  → Phase 1 stash amax+scale → Phase 2 restore → 保证一致 → 关键!
  → → → 否则: Phase 1用scale_A → Phase 2用scale_B → 梯度错误!

6. quantized_model_init → 省内存!
  → FP8参数=1.75GB(7B) → vs BF16=14GB → 省8x → 推理场景极省!
  → → preserve_high_precision_init_val → CPU存初始值 → optimizer初始化用!
  → → → LoRA: 主权重FP8+LoRA BF16 → 1.75+0.5=2.25GB → 端到端FP8训练!

7. split_accumulator → FP8 GEMM精度控制!
  → use_split_accumulator → cublasLt FP8快速累加 → 分块累加 → 精度更好!
  → → _2X_ACC_DGRAD=True, _2X_ACC_WGRAD=True → backward用split → 精度优先!
  → → → _2X_ACC_FPROP=False → forward不用 → 速度优先 → 权衡!
```

## 参考文献

```
1. Transformer Engine源码:
   - github.com/NVIDIA/TransformerEngine
   - transformer_engine/pytorch/quantization.py → FP8GlobalStateManager+autocast
   - transformer_engine/pytorch/module/linear.py → TE.Linear FP8 GEMM
   - transformer_engine/pytorch/module/base.py → TransformerEngineBaseModule+Userbuffers
   - transformer_engine/pytorch/tensor/float8_tensor.py → Float8Quantizer+Float8CurrentScalingQuantizer
   - transformer_engine/common/recipe/__init__.py → Recipe体系(DelayedScaling+CurrentScaling+MXFP8+NVFP4)

2. TE论文/文档:
   - NVIDIA, "Transformer Engine: FP8 Framework for H100", 2023
   - Micikevicius et al., "FP8 Formats for Deep Learning", 2022

3. 我们的笔记:
   - cuda-cutlass.md → CUTLASS SM80 FP8路径(RTX 4090)
   - fp8-quantization.md → FP8量化基础
   - distributed-nav SKILL → torch.distributed+NCCL导航
