# vLLM V1 LoRA Serving 源码阅读

> Multi-LoRA 服务: Punica 架构, 批量 LoRA 推理, Multi-tenant LoRA

## 1. 核心概念

### 1.1 Multi-LoRA 服务

Multi-LoRA 服务允许在单个 GPU 上同时服务多个 LoRA adapter:
- 基础模型权重共享（只加载一次）
- 每个 LoRA adapter 有独立的 A/B 矩阵
- 一个 batch 内的不同请求可以使用不同的 LoRA

### 1.2 Punica 论文

vLLM 的 LoRA 实现基于 Punica (Chen et al., 2023):
- **核心思想**: 将 LoRA 计算批量化，不同请求使用不同 LoRA
- **Segmented Matrix Multiplication**: 按请求分组计算 LoRA A×B
- **CUDA Graph 兼容**: 通过 specialization 支持固定 LoRA 数量

## 2. 架构

```
┌──────────────────────────────────────────────┐
│                 API Server                    │
│   POST /v1/completions (lora_name=adapter_1)  │
└──────────────┬───────────────────────────────┘
               │
    ┌──────────┴──────────┐
    │  LoRA Resolver       │ ← 根据 lora_name 找到 LoRA 适配器
    │  (resolver.py)       │
    └──────────┬──────────┘
               │
    ┌──────────┴──────────┐
    │  LoRA Worker Manager│ ← 管理 LoRA 加载/卸载
    │  (worker_manager.py)│
    └──────────┬──────────┘
               │
    ┌──────────┴──────────┐
    │  LoRA Model Manager  │ ← 管理 LoRA 模型实例
    │  (model_manager.py)  │    维护 adapter 注册表
    └──────────┬──────────┘
               │
    ┌──────────┴──────────┐
    │  Punica Wrapper      │ ← 批量 LoRA kernel
    │  (punica_gpu.py)     │    lora_shrink + lora_expand
    └──────────┬──────────┘
               │
    ┌──────────┴──────────┐
    │  Triton/CUDA Kernels│ ← 实际计算
    │  (ops/triton_ops/)  │    lora_shrink, lora_expand, fused_moe_lora
    └─────────────────────┘
```

## 3. 关键组件

### 3.1 LoRARequest (`vllm/lora/request.py`)

```python
class LoRARequest(msgspec.Struct):
    lora_name: str           # 适配器名称
    lora_int_id: int         # 全局唯一 ID
    lora_path: str           # 适配器路径
    base_model_name: str     # 基础模型名
    load_inplace: bool       # 是否原地替换
    is_3d_lora_weight: bool  # MoE 3D 融合格式
```

### 3.2 LoRAModelManager (`vllm/lora/model_manager.py`)

```python
class LoRAModelManager:
    """管理多个 LoRA adapter 的核心类"""

    # 适配器存储
    _registered_adapters: dict[int, LoRAModel]  # ID → LoRA 模型
    _active_adapters: dict[int, None]            # 活跃适配器集合
    lora_index_to_id: list[int | None]           # slot → adapter ID 映射

    # 配置
    max_num_seqs: int            # 最大序列数
    max_num_batched_tokens: int  # 最大批量 token 数
    lora_slots: int              # LoRA slot 数 (= max_loras)

    # Punica 内核
    punica_wrapper_mapping: dict[str, PunicaWrapperBase]

    # 模块映射
    modules: dict[str, BaseLayerWithLoRA]       # 层名 → LoRA 层
    packed_modules: dict[str, list[str]]        # 打包模块映射
```

关键初始化流程:
1. 获取模型支持的 LoRA 模块类型
2. 创建 Punica wrapper (GPU/CPU/XPU)
3. 为每个 LoRA 层创建 `BaseLayerWithLoRA` 实例
4. 预分配 `lora_a_stacked` / `lora_b_stacked` 张量

### 3.3 LoRA 层 (`vllm/lora/layers/`)

```
BaseLayerWithLoRA (base.py)
└── BaseLinearLayerWithLoRA (base_linear.py)
    ├── VocabParallelEmbeddingWithLoRA (vocal_parallel_embedding.py)
    ├── ColumnParallelLinearWithLoRA (column_parallel_linear.py)
    ├── RowParallelLinearWithLoRA (row_parallel_linear.py)
    ├── ReplicatedLinearWithLoRA (replicated_linear.py)
    ├── FusedMoEWithLoRA (fused_moe.py)
    └── LogitsProcessorWithLoRA (logits_processor.py)
```

**核心计算流程** (`BaseLinearLayerWithLoRA.apply`):
```python
def apply(self, x, bias=None):
    # 1. 基础层前向
    output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

    # 2. LoRA 增量
    lora_output = self.punica_wrapper.add_lora_linear(
        output, x,
        self.lora_a_stacked,  # [max_loras, 1, r, input_dim]
        self.lora_b_stacked,  # [max_loras, 1, output_dim, r]
        1.0,                   # scaling factor
        self.output_slices
    )

    # 3. 合并
    return output + lora_output
```

### 3.4 Punica Wrapper (`vllm/lora/punica_wrapper/`)

```python
class PunicaWrapperGPU(PunicaWrapperBase):
    """GPU Punica wrapper — 管理 Multi-LoRA 元数据"""

    def add_lora_linear(self, output, x, lora_a, lora_b, scale, ...):
        """添加 LoRA 增量到输出"""
        # 1. lora_shrink: x @ lora_A → [tokens, r]
        # 2. lora_expand: intermediate @ lora_B → [tokens, output_dim]
        # 批量化: 不同 token 使用不同 LoRA
```

三种 Punica 实现:
- `PunicaWrapperGPU` — GPU (Triton kernels)
- `PunicaWrapperCPU` — CPU fallback
- `PunicaWrapperXPU` — Intel XPU

### 3.5 Triton Kernels (`vllm/lora/ops/triton_ops/`)

核心 kernel:
1. **lora_shrink**: `x @ A → [tokens, rank]` (降维)
2. **lora_expand**: `intermediate @ B → [tokens, output_dim]` (升维)
3. **fused_moe_lora**: MoE + LoRA 融合计算

```python
# lora_shrink: 输入 [tokens, input_dim] × A [rank, input_dim] → [tokens, rank]
@triton.jit
def lora_shrink_kernel(...)

# lora_expand: 输入 [tokens, rank] × B [output_dim, rank] → [tokens, output_dim]
@triton.jit
def lora_expand_kernel(...)
```

## 4. Multi-LoRA 批处理

### 4.1 LoRAMapping

每个请求关联一个 LoRA adapter:
```python
class LoRAMapping:
    # token → adapter 的映射
    index_mapping: list[int]  # [token_0_adapter, token_1_adapter, ...]
    prompt_mapping: list[int] # prompt token → adapter
    lora_index_to_id: list[int | None]  # slot → adapter_id
```

### 4.2 批量 LoRA 计算

```
Token 0 (adapter A): ─── lora_shrink(A_A) ─── lora_expand(A_B) ───
Token 1 (adapter B): ─── lora_shrink(B_A) ─── lora_expand(B_B) ───
Token 2 (no LoRA):   ─── (skip) ──────────────────────────────────
Token 3 (adapter A): ─── lora_shrink(A_A) ─── lora_expand(A_B) ───

所有 token 在同一个 Triton kernel 中并行处理
```

## 5. LoRA 权重管理

### 5.1 权重格式

LoRA A/B 矩阵预分配为固定大小张量:
```python
lora_a_stacked: torch.Tensor  # [max_loras, 1, max_rank, input_dim]
lora_b_stacked: torch.Tensor  # [max_loras, 1, output_dim, max_rank]
```

- `max_loras`: 最大同时活跃 LoRA 数
- `max_rank`: LoRA rank 上限
- 不足的 rank 自动 zero-padding

### 5.2 权重加载流程

1. 用户发送 LoRA 请求（包含 `lora_path`）
2. Worker Manager 加载 LoRA 权重文件
3. LoRAModel 解析 safetensors 中的 A/B 矩阵
4. 将 A/B 矩阵拷贝到 `lora_a_stacked` / `lora_b_stacked` 对应 slot
5. 如果超出 `max_loras`，LRU 驱逐最久未用的 adapter

## 6. CUDA Graph 兼容

LoRA 与 CUDA Graph 的交互:

```python
# 通过 LoRAKernelMeta 管理 CUDA Graph 特化
token_mapping_meta = LoRAKernelMeta.make(
    self.max_loras,
    max_num_batched_tokens,
    device=device,
    captured_lora_counts=captured_lora_counts,  # CUDA Graph capture 的 LoRA 数量
)
```

**specialize_active_lora**: CUDA Graph 需要固定 LoRA 数量
- 如果启用，为每种活跃 LoRA 数量捕获不同的 Graph
- 如果禁用，使用 max_loras 作为上限，填充 dummy LoRA

## 7. MoE + LoRA

`FusedMoEWithLoRA` (layers/fused_moe.py):
- MoE 层也支持 LoRA
- 融合计算: expert forward + LoRA 增量
- 支持 3D 融合格式 (gate_up_proj/down_proj)
- `enable_mixed_moe_lora_format`: 允许 2D/3D LoRA 共存

## 8. 配置参数

```bash
vllm serve meta-llama/Llama-3.1-70B \
    --enable-lora \
    --max-loras 4 \              # 最大同时活跃 LoRA 数
    --max-lora-rank 16 \         # 最大 LoRA rank
    --lora-extra-vocab-size 256 \# 额外词表大小
    --lora-dtype float16 \       # LoRA 权重精度
    --max-cpu-loras 8            # CPU 缓存的 LoRA 数
```

API 请求:
```python
# 加载 LoRA
POST /v1/load_lora_adapter
{
    "lora_name": "my_adapter",
    "lora_path": "/path/to/lora"
}

# 使用 LoRA
POST /v1/completions
{
    "model": "my_adapter",  # 使用 lora_name 作为 model
    "prompt": "Hello"
}

# 卸载 LoRA
POST /v1/unload_lora_adapter
{
    "lora_name": "my_adapter"
}
```

## 9. 关键洞察

1. **基础模型权重共享**: 所有 LoRA 共享基础模型，显存开销仅为 LoRA A/B 矩阵
2. **Punica 批量计算**: 一个 Triton kernel 处理不同 LoRA 的所有 token
3. **预分配 + slot 管理**: `lora_a_stacked` / `lora_b_stacked` 预分配，零碎片
4. **LRU 驱逐**: 超出 `max_loras` 时自动驱逐最久未用的 adapter
5. **CUDA Graph 特化**: 需要 `specialize_active_lora` 管理固定 LoRA 数量
6. **双流异步**: `VLLM_LORA_ENABLE_DUAL_STREAM` 启用辅助 CUDA stream
7. **MoE + LoRA**: 融合计算，支持 2D/3D 格式混合
8. **LoRA 指标**: Prometheus 追踪 per-LoRA waiting/running 请求数

## 10. 性能分析

| 组件 | 开销 | 优化 |
|------|------|------|
| LoRA A 矩阵 (shrink) | O(tokens × rank × input) | Triton kernel 批量化 |
| LoRA B 矩阵 (expand) | O(tokens × rank × output) | Triton kernel 批量化 |
| 权重加载 | safetensors IO | CPU cache + 异步加载 |
| LRU 驱逐 | 权重拷贝 | 异步卸载 |
| CUDA Graph | capture 开销 | 特化活跃 LoRA 数 |

LoRA 推理开销: ~2-5% (rank=16, 相比基础模型)

## 参考资料

- `vllm/lora/request.py` — LoRA 请求定义
- `vllm/lora/model_manager.py` — LoRA 模型管理器
- `vllm/lora/worker_manager.py` — Worker 端 LoRA 管理
- `vllm/lora/layers/` — LoRA 层实现 (base/column/row/moe)
- `vllm/lora/punica_wrapper/` — Punica wrapper (GPU/CPU/XPU)
- `vllm/lora/ops/triton_ops/` — Triton LoRA kernels
- `vllm/lora/resolver.py` — LoRA adapter 解析器
- Punica 论文: Chen et al., "Punica: Multi-Tenant LoRA Serving" (2023)
- 相关: [LoRA 基础知识](../fundamentals/lora-peft.md)
