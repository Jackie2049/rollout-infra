# vLLM 权重加载管线源码阅读

> 从 safetensors 文件到 GPU 张量：理解 vLLM 的完整权重加载流程

## 1. 总体流程 (5 个阶段)

```
Stage A: 模型构建 (meta tensors, 参数创建有 shape 无数据)
Stage B: 权重文件发现 (定位/下载 safetensors 文件)
Stage C: 权重迭代 (逐 tensor 从磁盘流式读取)
Stage D: 权重→参数映射 (名称匹配 + 分片解析)
Stage E: 后处理 (量化、重打包、设备传输)
```

## 2. Stage A: 模型构建

**文件**: `model_loader/base_loader.py`

```
BaseModelLoader.load_model()
  → initialize_model()
    → get_model_architecture(model_config)  # 解析模型类
    → model_class(vllm_config, prefix)      # 在 meta/device 上实例化
      → 每个 LinearBase.__init__():
        → quant_method.create_weights(..., weight_loader=self.weight_loader)
          → torch.empty(..., shape=shard_shape)  # 预分配本 rank 的分片
          → set_weight_attrs(param, {"weight_loader": ..., "output_dim": ...})
```

**关键**: 模型在初始化时就预分配了每个 rank 的分片形状 (e.g., `[output_size/tp, input_size]`)。

## 3. Stage B: 权重文件发现

**文件**: `model_loader/default_loader.py`

`_prepare_weights()`:
1. 检测本地路径 vs HF Hub 下载
2. 按 `load_format` 选择文件模式:
   - `auto`: 先检查 Mistral 格式, 回退到 `*.safetensors` + `*.bin`
   - `safetensors`/`fastsafetensors`: `*.safetensors`
   - `pt`: `*.pt`
3. `filter_duplicate_safetensors_files()`: 用 `index.json` 排除重复
4. 返回 `(hf_folder, hf_weights_files, use_safetensors)`

## 4. Stage C: Safetensors 迭代

**文件**: `model_loader/weight_utils.py`

`safetensors_weights_iterator()` (~130 行):

```python
with safe_open(st_file, framework="pt") as f:
    for name in f.keys():
        if should_skip_weight(name, local_expert_ids):  # EP 跳过
            continue
        param = f.get_tensor(name)  # 单个 tensor, CPU
        yield name, param
```

**优化**:
- **Lazy loading**: 默认一次只读一个 tensor 到内存 (内存高效)
- **Auto-prefetch**: NFS/Lustre 上后台预读 checkpoint 文件
- **EP 过滤**: MoE 模型跳过非本地 expert 权重, 节省 ~85-90% I/O

**迭代器选择**:

| 格式 | 迭代器 | 特点 |
|------|--------|------|
| safetensors (默认) | `safetensors_weights_iterator` | `safe_open()` 逐 tensor |
| 多线程 | `multi_thread_safetensors_weights_iterator` | `load_file()` 线程池 |
| fastsafetensors | `fastsafetensors_weights_iterator` | GDS/direct IO |
| pt | `pt_weights_iterator` | `torch.load()` |

## 5. Stage D: 权重→参数映射

### 5.1 两种加载路径

**路径 1: 自定义 load_weights()** (e.g., LlamaModel)
```
model.load_weights(weights_iterator)
  → 创建 stacked_params_mapping (q_proj+k_proj+v_proj → qkv_proj)
  → 迭代 (name, loaded_weight):
    → 匹配 stacked params 或 params_dict
    → weight_loader(param, loaded_weight, [shard_id])
```

**路径 2: AutoWeightsLoader** (通用)
```
AutoWeightsLoader.load_weights()
  → _load_module("", self.module, weights)
    → _groupby_prefix(): 按前缀分组
    → 递归下降子模块
    → 如果子模块有 load_weights(), 委托
    → 否则 _load_param():
      → weight_loader = getattr(param, "weight_loader", default_weight_loader)
      → weight_loader(param, weight_data)
```

### 5.2 TP 分片: weight_loader 机制

**ColumnParallelLinear.weight_loader()**:
```python
def weight_loader(self, param, loaded_weight):
    output_dim = param.output_dim
    shard_size = param.shape[output_dim]
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
    param.data.copy_(loaded_weight)
```

- `loaded_weight` 是完整 checkpoint tensor
- `narrow()` 提取本 rank 的列切片: `start = tp_rank × shard_size`
- 每个 rank 读所有文件, 但只复制自己的分片

**RowParallelLinear.weight_loader()**: 同理但沿 input_dim 分片

**MergedColumnParallelLinear (qkv/gate_up)**:
- 额外接受 `shard_id` 参数 (q/k/v 或 gate/up)
- 计算 per-sub-projection offset
- 先按 sub-projection 切片, 再按 TP 切片

### 5.3 PP 处理

- `StageMissingLayer` / `PPMissingLayer` 替换不属于本 stage 的层
- `is_pp_missing_parameter(name, model)` 跳过这些层的权重
- 只加载本 PP stage 需要的层

### 5.4 EP 处理

- `_init_ep_weight_filter()` 计算 `local_expert_ids`
- `should_skip_weight()` 在**磁盘读取前**跳过非本地 expert
- 只过滤重的 `.weight` tensor, 保留 scale/metadata

## 6. Stage E: 后处理

**文件**: `model_loader/utils.py`

`process_weights_after_loading()`:
1. 迭代所有模块, 调用 `quant_method.process_weights_after_loading(module)`
   - Marlin: 重打包为 Marlin 格式
   - FP8: 合并 per-shard scale 为 per-tensor scale
   - AWQ: 转置 + 重打包
2. 处理 Attention 层的后处理 (MLA 解压等)
3. `device_loading_context()`: CPU offload 时临时移到 GPU 处理后移回

### 在线量化

**路径**: `Fp8OnlineLinearMethod` 在 meta device 创建权重 → `finalize_layerwise_processing()` → `ops.scaled_fp8_quant()` 即时量化

## 7. 完整流程图

```
get_model()
  ├── DefaultModelLoader.load_model()
  │   ├── initialize_model()
  │   │   └── model_class.__init__()
  │   │       └── 每层: quant_method.create_weights() → 预分配分片
  │   │
  │   ├── model.load_weights(get_all_weights())
  │   │   └── safetensors_weights_iterator()
  │   │       └── safe_open() → f.get_tensor() → yield (name, tensor)
  │   │           └── weight_loader(param, loaded_weight)
  │   │               └── narrow(tp_rank×shard) → param.copy_()
  │   │
  │   └── process_weights_after_loading()
  │       └── quant_method.process_weights_after_loading()
  │           └── 重打包/重量化/填充对齐
  │
  └── model.eval()
```

## 8. 关键洞察

1. **分片在加载时**: 每个 rank 读所有文件但只复制自己的分片 (`narrow + copy_`)
2. **ShardedStateLoader 优化**: 预分片 checkpoint 每个 rank 只读自己的文件
3. **EP 在磁盘前过滤**: MoE 模型跳过 85-90% 权重读取
4. **weight_loader 可扩展**: 自定义层通过 `set_weight_attrs` 附加 loader
5. **在线 vs 离线**: 离线 (checkpoint 已量化) 直接加载; 在线 (meta→quantize) 走后处理路径
6. **PP 跳层**: 用 `PPMissingLayer` 占位, 只加载本 stage 的层

## 参考资料

- 源码: `vllm/model_executor/model_loader/`
- 相关笔记: [量化管线](vllm-quantization-pipeline-reading.md), [Executor](vllm-v1-executor-reading.md)
