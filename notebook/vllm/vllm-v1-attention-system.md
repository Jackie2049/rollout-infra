# vLLM V1 Attention 后端系统

> 目录: `vllm/v1/attention/` (60+ 文件)
> 分析日期: 2026-06-04

## 架构概览

```
vllm/v1/attention/
├── backend.py          # AttentionBackend 抽象基类
├── selector.py         # 后端选择器 (platform + hardware aware)
├── backends/
│   ├── registry.py     # 后端注册表 (插拔系统)
│   ├── flash_attn.py   # FlashAttention (Ampere+)
│   ├── flashinfer.py   # FlashInfer (多平台优化)
│   ├── triton_attn.py  # Triton decode attention
│   ├── flex_attention.py # PyTorch FlexAttention
│   ├── mla/            # 多头潜在注意力 (DeepSeek-V2/V3)
│   │   ├── flashmla.py    # NVIDIA FlashMLA (SM90+)
│   │   ├── cutlass_mla.py # CUTLASS MLA (SM100)
│   │   ├── triton_mla.py  # Triton fallback MLA
│   │   └── prefill/       # Prefill 专用 MLA 后端
│   ├── mamba_attn.py   # Mamba SSM
│   └── cpu_attn.py     # CPU fallback
└── ops/                # 底层 kernel 实现
    ├── triton_decode_attention.py
    ├── triton_prefill_attention.py
    ├── paged_attn.py
    └── prefix_prefill.py
```

## AttentionBackend 抽象接口

```python
class AttentionBackend(ABC):
    supported_dtypes: ClassVar[list[torch.dtype]]         # 支持的 KV 类型
    supported_kv_cache_dtypes: ClassVar[list]              # 支持的量化格式
    supported_attention_types: ClassVar[list[AttentionType]]# DECODER/ENCODER...
    forward_includes_kv_cache_update: ClassVar[bool]       # forward() 是否写 KV

    @staticmethod
    def get_name() -> str                                 # "FLASH_ATTN", etc.
    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]   # 返回 metadata 类型
    def get_pad_sizes(...) -> tuple[int, int]             # CUDA Graph padding
    def forward(metadata, query, key, value, ...) -> Tensor # 核心接口
```

## AttentionType

```python
class AttentionType(str, Enum):
    DECODER = "decoder"            # Self-attention (常规 decode)
    ENCODER = "encoder"            # Encoder self-attention
    ENCODER_ONLY = "encoder_only"  # Encoder-only 模型
    ENCODER_DECODER = "encoder_decoder" # Cross-attention (dec→enc)
```

## 后端选择器 (selector.py)

选择逻辑:
```python
1. 平台检测: NVIDIA/ROCm/CPU
2. 计算能力: SM 8.0 (A100), 8.9 (H100), 9.0 (H100), 10.0 (B200)
3. 模型类型: Dense/MLA/Mamba/Encoder-decoder
4. 量化模式: FP8/INT8/FP4
5. 序列长度: Prefill vs Decode (不同 kernel)

优先级:
- Ampere (SM 8.0-8.9): FlashInfer > FlashAttn > Triton
- Hopper (SM 9.0+):   FlashAttn > FlashInfer > CUTLASS > Triton
- Blackwell (SM 10.0): FlashAttn > FlashInfer > CUTLASS > Triton
```

## 后端清单

| 后端 | 计算能力 | 特点 |
|------|:---:|------|
| **FlashAttention** | SM 8.0+ | 最通用, tiling + IO-aware |
| **FlashInfer** | SM 8.0+ | 多平台优化, PagedKV native |
| **Triton Attn** | 全平台 | Python 编写, 易扩展 |
| **FlexAttention** | SM 9.0+ | PyTorch 原生, JIT 编译 |
| **FlashMLA** | SM 9.0+ | DeepSeek-V2/V3 专用 |
| **CUTLASS MLA** | SM 10.0+ | Blackwell MLA 加速 |
| **Mamba/Linear** | 全平台 | SSM 模型专用 |
| **CPU** | CPU | 调试/fallback |

## Metadata 系统

每个后端定义自己的 `AttentionMetadata` 子类:
- 包含: block_table, slot_mapping, query_start_loc, kv_cache layout
- GPUModelRunner._prepare_inputs() 填充 metadata
- Backend.forward() 消费 metadata

## Prefill vs Decode 分离

```python
# 大 sequence 用 prefill kernel (compute-bound, tiled)
# 单个 token 用 decode kernel (memory-bound, paged)
# Cascade attention: 自动检测 common prefix 切换路径
```

## 与 Scheduler 的交互

```
Scheduler.schedule()
  → SchedulerOutput (new/revised tokens per request)
    → GPUModelRunner._prepare_inputs()
      → 构建 AttentionMetadata (block_table, slot_mapping, ...)
        → AttentionBackend.forward(metadata=..., ...)
```

## 代码位置速查

| 文件 | 内容 |
|------|------|
| `backend.py` | AttentionBackend 抽象类, AttentionType 枚举 |
| `selector.py` | 平台感知后端选择 |
| `backends/registry.py` | 后端注册表 |
| `backends/flash_attn.py` | FlashAttention 实现 |
| `backends/flashinfer.py` | FlashInfer 实现 |
| `backends/mla/` | MLA 后端子目录 (DeepSeek-V2/V3) |
| `ops/triton_decode_attention.py` | Triton decode kernel |
| `ops/triton_prefill_attention.py` | Triton prefill kernel |
