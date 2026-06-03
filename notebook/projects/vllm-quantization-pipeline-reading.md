# vLLM 量化推理管线源码阅读

> 从 safetensors 到 kernel dispatch：理解 vLLM 的量化推理完整流程

## 1. 量化方法注册表

**文件**: `vllm/model_executor/layers/quantization/__init__.py`

vLLM 支持 ~30 种量化方法，通过注册表管理：

| 类别 | 方法 |
|------|------|
| W4A16 (权重4bit) | awq, auto_gptq, awq_marlin, gguf, quark, moe_wna16 |
| W8A8 (权重+激活8bit) | fp8, fbgemm_fp8, modelopt, compressed-tensors |
| W4A8 | experts_int8, humming |
| 在线量化 | fp8_per_tensor, fp8_per_block, mxfp8, int8_per_channel |
| 低比特 | mxfp4, gpt_oss_mxfp4, deepseek_v4_fp8 |

外部方法通过 `@register_quantization_config("name")` 装饰器注册。

## 2. 类层次结构

```
QuantizeMethodBase (ABC)
  ├── create_weights()          # 创建量化参数
  ├── apply()                   # 前向计算 (dispatch 到 kernel)
  ├── process_weights_after_loading()  # 权重后处理
  └── uses_meta_device: bool    # 在线量化用 meta device 延迟物化

QuantizationConfig (ABC)
  ├── get_name()                # 量化方法名
  ├── get_supported_act_dtypes() # 支持的激活 dtype
  ├── get_min_capability()      # 最低 GPU 算力
  ├── get_config_filenames()    # HF config 文件名
  ├── from_config()             # 从 HF config 构建
  └── get_quant_method(layer, prefix)  # 返回层的量化方法

LinearMethodBase extends QuantizeMethodBase
  └── 线性层特定基类
```

## 3. 量化检测流程

```
ModelConfig._verify_quantization()
  │
  ├── 1. 用户指定 --quantization? → 使用指定方法
  │
  └── 2. 自动检测 config.json 的 quantization_config
        │
        └── 按优先级遍历注册方法:
            │
            ├── override_quantization_method() 匹配
            │   (e.g., quant_method=="gptq" → AutoGPTQConfig)
            │
            └── 首个匹配 → 确定量化方法
```

## 4. 权重加载流程 (safetensors → kernel)

### 4.1 完整管线

```
1. 模型初始化
   ├── _verify_quantization() → 确定量化方法
   ├── quant_cls.from_config(hf_config) → QuantizationConfig
   └── model.__init__()
       └── 每个 LinearBase 层:
           ├── quant_config.get_quant_method(layer, prefix)
           │   └── 返回 QuantizeMethodBase (e.g., Fp8LinearMethod)
           └── quant_method.create_weights(layer, ...)
               └── init_fp8_linear_kernel()
                   └── choose_scaled_mm_linear_kernel()
                       └── 按优先级选择最佳 kernel

2. 权重加载
   ├── model.load_weights(get_all_weights())
   │   └── 从 safetensors 加载张量到参数
   │       └── custom weight_loader 处理分片/重打包
   │
   └── process_weights_after_loading(model)
       └── 每个 module 的 quant_method:
           ├── 转置权重适配 kernel 布局
           ├── 重量化 (per-shard scale → per-tensor scale)
           ├── 打包为 Marlin/CUTLASS 格式
           └── 填充对齐 (CUTLASS)

3. 前向推理
   └── quant_method.apply(layer, x, bias)
       └── dispatch 到选定的 kernel
```

### 4.2 离线 vs 在线量化

**离线** (FP8 checkpoint):
- 权重以 `float8_e4m3fn` 格式加载
- `Fp8LinearMethod.create_weights()` 创建 scale 参数
- `process_weights_after_loading()` 合并 per-shard scale 为 per-tensor scale

**在线** (BF16 checkpoint, `--quantization fp8`):
- `Fp8OnlineLinearMethod.create_weights()` 在 meta device 上创建 (延迟物化)
- `process_weights_after_loading()` 调用 `ops.scaled_fp8_quant()` 即时量化
- 然后选择并初始化 kernel

## 5. Kernel 选择逻辑

**文件**: `vllm/model_executor/kernels/linear/__init__.py`

核心函数 `choose_scaled_mm_linear_kernel()`:
```
对于 platform_kernels 中的每个 kernel:
  1. kernel.is_supported(compute_capability)  # 硬件检查
  2. kernel.can_implement(config)              # 配置兼容性
  3. 首个匹配 → 返回
```

### 5.1 FP8 (W8A8) CUDA Kernel 优先级

| 优先级 | Kernel | 条件 |
|--------|--------|------|
| 1 | MarlinFP8ScaledMM | SM < 89 (无 FP8 硬件) |
| 2 | FlashInferFP8ScaledMM | SM >= 100 (Blackwell) |
| 3 | CutlassFP8ScaledMM | 任何 CUDA (fallback) |
| 4 | PerTensorTorchFP8ScaledMM | SM >= 89, per-tensor scale |
| 5 | ChannelWiseTorchFP8ScaledMM | SM >= 89, per-channel |

### 5.2 FP8 Block 量化 (128×128) CUDA

| 优先级 | Kernel | 条件 |
|--------|--------|------|
| 1 | FlashInferFp8DeepGEMM | SM >= 100, FlashInfer 可用 |
| 2 | DeepGemmFp8BlockScaledMM | DeepSeek 风格 |
| 3 | CutlassFp8BlockScaledMM | CUTLASS 块支持 |
| 4 | MarlinFP8ScaledMM | SM < 89 fallback |
| 5 | TritonFp8BlockScaledMM | Triton fallback |

### 5.3 W4A16 (GPTQ/AWQ) CUDA

| 优先级 | Kernel |
|--------|--------|
| 1 | CutlassW4A8LinearKernel |
| 2 | MacheteLinearKernel |
| 3 | AllSparkLinearKernel |
| 4 | **MarlinLinearKernel** (主力) |
| 5 | ConchLinearKernel |
| 6 | ExllamaLinearKernel |
| 7 | TritonW4A16LinearKernel |

### 5.4 覆盖机制

```bash
--linear-backend cutlass    # 强制使用 CUTLASS kernel
VLLM_DISABLED_KERNELS=marlin  # 禁用 Marlin
VLLM_BATCH_INVARIANT=1      # 确定性执行 (优先 CUTLASS)
```

## 6. QuantKey 系统

**文件**: `vllm/model_executor/layers/quantization/quant_utils.py`

`QuantKey` 是结构化标识符，指定确切的量化格式：

```python
@dataclass
class QuantKey:
    dtype: ScalarType      # float8_e4m3fn, uint4, int8...
    scale: ScaleDesc       # 缩放因子描述 (dtype, static/dynamic, shape)
    scale2: ScaleDesc      # 第二级缩放 (NVFP4)
    symmetric: bool        # 对称 vs 非对称
```

常用 Key:
- `kFp8StaticTensorSym` → per-tensor FP8 → 映射到 `_POSSIBLE_FP8_KERNELS`
- `kFp8DynamicTokenSym` → dynamic per-token FP8
- `kFp8Static128BlockSym` → block 128×128 → 映射到 `_POSSIBLE_FP8_BLOCK_KERNELS`
- `kInt4StaticGroupScale` → INT4 group quantization

## 7. 在线量化分发表

```python
_ONLINE_LINEAR_METHODS = {
    kFp8StaticTensorSym:  Fp8PerTensorOnlineLinearMethod,
    kFp8Static128BlockSym: Fp8PerBlockOnlineLinearMethod,
    kMxfp8Dynamic:        Mxfp8OnlineLinearMethod,
}
_ONLINE_MOE_METHODS = {
    kFp8StaticTensorSym:  Fp8PerTensorOnlineMoEMethod,
    kFp8Static128BlockSym: Fp8PerBlockOnlineMoEMethod,
    kMxfp8Dynamic:        Mxfp8OnlineMoEMethod,
    kInt8StaticChannelSym: Int8OnlineMoEMethod,
}
```

启用方式: `--quantization fp8_per_tensor` 或 `--quantization fp8_per_block`

## 8. KV Cache 量化

**文件**: `vllm/model_executor/layers/quantization/kv_cache.py`

`BaseKVCacheMethod` 在 Attention 层上创建 `q_scale`, `k_scale`, `v_scale`, `prob_scale` 参数。

- 离线: 从 checkpoint 加载 scale
- 动态: `--calculate-kv-scales` 运行时计算
- Per-token-head: `kv_cache_uses_per_token_head_scales()` 为 True 时动态计算
- FNUZ: AMD GPU (MI300x) 需要 2× scale 调整

## 9. 关键架构洞察

1. **三层抽象**: QuantizationConfig (模型级) → QuantizeMethod (层级) → Kernel (硬件级)
2. **优先级调度**: 每种量化格式有预定义的 kernel 优先级列表，按 GPU capability 自动选择
3. **离线 vs 在线**: 离线 (checkpoint 已量化) 和在线 (BF16→FP8 即时量化) 走不同路径
4. **QuantKey 驱动**: 结构化的量化格式标识符统一了不同方法的 kernel 选择
5. **可扩展**: 新量化方法通过 `@register_quantization_config` 注册，新 kernel 通过 `choose_scaled_mm_linear_kernel` 优先级列表加入

## 参考资料

- 源码路径: `vllm/model_executor/layers/quantization/`
- Kernel 选择: `vllm/model_executor/kernels/linear/`
- 相关笔记: [FP8 量化](../fundamentals/fp8-quantization.md), [KV Cache 量化](../fundamentals/kv-cache-quantization.md)
