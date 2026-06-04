# vLLM V1 Weight Loading 管线分析

> 目录: `vllm/model_executor/model_loader/` (21 文件)
> 分析日期: 2026-06-04

## 5 阶段管线

```
load_model():
  1. Build         initialize_model()         → 空模型结构 (meta device)
  2. Discover      _prepare_weights()         → 权重文件列表 + EP 过滤
  3. Iterate       _get_weights_iterator()    → 流式读取 safetensors
  4. Map           weight_loader per module   → TP narrow/copy_, EP filter
  5. Postprocess   process_weights_after_loading() → 量化/repack/attention init
```

---

## Stage 1: Build — 模型初始化

```python
model = initialize_model(vllm_config, model_config)
```

- 根据 `model_config.model_type` 选择架构类
- 在 `target_device` 上创建空模型 (meta device 用于大模型)
- 设置 `torch.set_default_dtype` 匹配模型精度

---

## Stage 2: Discover — 权重文件发现 + EP 预过滤

```python
# DefaultModelLoader._prepare_weights()
weights_paths = download_weights_from_hf(model_config)  # 或本地路径
# EP 过滤: 提前排除非本地 expert 权重 (减少 I/O 85-90%)
if ep_size > 1:
    local_expert_ids = compute_local_expert_ids(num_experts, ep_size, ep_rank)
```

**EP 预过滤 (`ep_weight_filter.py`)**:
```python
# 解析权重名: ".experts.42.gate_proj.weight"
expert_id = parse_expert_id(weight_name)
if expert_id is not None and expert_id not in local_expert_ids:
    skip  # 不加载非本地 expert

# 两种 placement:
# "linear":    rank 0→experts 0-3, rank 1→experts 4-7
# "round_robin": rank 0→experts 0,2,4,6, rank 1→experts 1,3,5,7
```

**不影响**: FusedMoE 3D 融合张量 (无数字 id) 不预过滤，加载后由 `FusedMoE.weight_loader` 切片。

---

## Stage 3: Iterate — 流式权重迭代

```python
# 支持多种迭代器:
- safetensors_weights_iterator()           # 单线程
- multi_thread_safetensors_weights_iterator() # 多线程 (默认 8)
- fastsafetensors_weights_iterator()        # 零拷贝 fastsafetensors
- pt_weights_iterator()                    # PyTorch 格式
- np_cache_weights_iterator()              # numpy 缓存加速
- instanttensor_weights_iterator()         # 延迟加载
- runai_safetensors_weights_iterator()     # Run:ai streamer
```

流式设计: 不一次性加载所有权重到内存，边读边分发。

---

## Stage 4: Map — 权重分发 (TP/EP)

### TP Sharding

每个线性层有自定义 `weight_loader`:
```python
class ColumnParallelLinear:
    def weight_loader(self, param, loaded_weight):
        # TP: 沿 output dim 切分
        # param.shape = [output_size_per_partition, input_size]
        # loaded_weight.shape = [output_size, input_size]
        param.data.copy_(loaded_weight.narrow(
            0, tp_rank * output_size_per_partition, output_size_per_partition
        ))

class RowParallelLinear:
    def weight_loader(self, param, loaded_weight):
        # TP: 沿 input dim 切分
        param.data.copy_(loaded_weight.narrow(
            1, tp_rank * input_size_per_partition, input_size_per_partition
        ))
```

### EP Sharding (MoE)
```python
# 每个 rank 只加载本地 expert 的权重切片
if expert_id in local_expert_ids:
    local_idx = remap_expert(expert_id)
    param.data[local_idx].copy_(loaded_weight)
```

### ShardedStateLoader
```python
# 预切分格式: "model-rank-{rank}-part-{part}.safetensors"
# 每个 rank 只读自己的文件，零跨 rank 复制
pattern = "model-rank-{rank}-part-{part}.safetensors"
```

---

## Stage 5: Postprocess — 后处理

```python
process_weights_after_loading(model, model_config):
    # 1. Online quantization: FP16→INT8/FP8 (按需)
    for module in model.modules():
        quant_method.process_weights_after_loading(module)

    # 2. Attention 权重初始化
    for module in model.modules():
        if isinstance(module, Attention):
            module.process_weights_after_loading()

    # 3. Layer-wise processing (torchao, etc.)
    finalize_layerwise_processing(model)
```

---

## Loader 类型

| Loader | 用途 |
|--------|------|
| `DefaultModelLoader` | 标准 HuggingFace 格式 |
| `ShardedStateLoader` | 预切分格式 (每 rank 独立文件) |
| `BitsAndBytesLoader` | 4bit 量化 |
| `GGUFLoader` | GGUF 格式 (llama.cpp) |
| `TensorizerLoader` | CoreWeave tensorizer |
| `DummyModelLoader` | 测试用随机权重 |
| `RunaiStreamerLoader` | Run:ai S3 流式加载 |

---

## 关键优化

1. **EP pre-disk filter**: 85-90% MoE 权重跳过，大幅减少 I/O
2. **Multi-thread load**: 默认 8 线程并行解析 safetensors
3. **Zero-copy**: fastsafetensors 直接 mmap，避免拷贝
4. **TP narrow+copy_**: 不 slice 全量权重，只 `narrow` 需要的切片
5. **Lazy layer-wise**: 在线量化时逐层加载+量化，节省 peak memory

---

## 代码位置速查

| 文件 | 内容 |
|------|------|
| `base_loader.py` | 5-stage load_model() 主入口 |
| `default_loader.py` | HF 格式加载, _prepare_weights, _get_weights_iterator |
| `ep_weight_filter.py` | Expert 预过滤 (I/O 节省 85-90%) |
| `sharded_state_loader.py` | TP 预切分格式加载 |
| `utils.py` | initialize_model, process_weights_after_loading |
| `weight_utils.py` | 权重下载/迭代工具 |
