# Attention Kernel 优化：FlashAttention-3 / FlashInfer / FlashMLA

> 深入理解 LLM 推理中 attention kernel 的选型和优化策略

## 1. Attention Kernel 为什么重要

Attention 是 Transformer 的核心操作，也是计算瓶颈：

| 阶段 | 瓶颈 | Attention 特征 |
|------|------|---------------|
| Prefill | compute-bound | O(N²) attention, Q 序列长, 利用率高 |
| Decode | memory-bound | Q=1, 读全部 KV Cache, 利用率极低 (<1%) |

不同场景需要不同的 kernel 优化策略，这就是为什么有这么多 attention kernel 变体。

## 2. FlashAttention 演进

### 2.1 FlashAttention-1/2 (回顾)

核心优化：**Tiling + Online Softmax**，将 HBM IO 从 O(N²d) 降到 O(N²d²/M)。

- FA1: 分块计算，online softmax，首次实现 IO-aware
- FA2: 更好的 work partitioning，减少 non-matmul FLOPs，Ampere (A100) 上 ~50% 利用率

### 2.2 FlashAttention-3: Hopper 专属优化

**问题**: FA2 在 H100 上仅 ~35% 利用率，因为 FP16 matmul (989 TFLOPS) vs 特殊函数 (3.9 TFLOPS) 256:1 的不对称。

**三大硬件特性利用**:

1. **WGMMA (Warpgroup MMA)**: 4 warp 组成 warpgroup，H100 新指令，吞吐远超 FA2 的 mma.sync
2. **TMA (Tensor Memory Accelerator)**: 硬件级异步数据搬运，解放线程做计算
3. **FP8 低精度**: H100 原生 FP8，~1.2 PFLOPS 吞吐

**三层调度优化**:

```
Pingpong (乒乓调度):
  WG0 做 GEMM ↔ WG1 做 softmax/rescale
  → GEMM 和 softmax 宏观并行
  → ~570 → ~620 TFLOPS

Intra-warpgroup Pipeline (组内流水线):
  单 warpgroup 内 matmul 拆为 micro-op 流水线
  → 等当前 micro-op 结果时，加载下一个
  → ~620 → ~650 TFLOPS

Incoherent Processing (非相干处理):
  Q/K 施加随机 Hadamard 变换 → 元素分布更均匀
  → FP8 量化误差降低 2.6x
```

**性能**:

| 配置 | FP16 Forward | FP16 Backward | FP8 Forward |
|------|-------------|---------------|-------------|
| FA2 on H100 | ~350 TFLOPS (35%) | ~350 TFLOPS | N/A |
| FA3 on H100 | **~740 TFLOPS (75%)** | ~680 TFLOPS | **~1.2 PFLOPS** |

### 2.3 FlashAttention-4 (CuTeDSL)

FA4 使用 NVIDIA CUTLASS CuTe DSL 重写，同时支持 Hopper (SM90) 和 Blackwell (SM100)：

```python
from flash_attn.cute import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)
```

### 2.4 版本选择

| GPU 架构 | 推荐 FA | 利用率 |
|----------|---------|--------|
| Ampere (A100) | FA2 | ~50% |
| Hopper (H100) | FA3/FA4 | ~75% |
| Blackwell (B200) | FA4 | TBD |

## 3. FlashInfer: 面向 Serving 的全能 Kernel

### 3.1 定位

FlashInfer 不只是 attention kernel，而是 **LLM serving 全场景 kernel 库**：

- **Attention**: paged/ragged KV-cache, decode/prefill/append, MLA, cascade, sparse, POD
- **GEMM**: BF16/FP8/FP4, grouped GEMM
- **MoE**: fused MoE, 多种 routing
- **Sampling**: sorting-free 采样
- **Communication**: AllReduce, MNNVL, NVSHMEM

GPU 支持: SM75 (Turing) 到 SM12.1 (RTX 50 系列)。

### 3.2 核心创新

**Paged KV-Cache 原生支持**:
- KV-cache 按 block 组织，非连续内存
- 与 vLLM 的 PagedAttention 一致，无需额外转换
- 支持 ragged tensor (batch 内不同序列长度)

**JIT 编译**:
- 运行时根据 head_dim/num_heads/dtype/causal 模式编译最优 kernel
- 避免预编译所有组合的巨大二进制
- 编译后缓存复用

**Load-Balanced Scheduling**:
- Attention 任务按计算量拆分为细粒度 work tiles
- 动态分配给 SM，避免长序列垄断
- 与 CUDA Graph 兼容

### 3.3 Prefill vs Decode

**Prefill Kernel**: GEMM-based (类似 FA2)，处理长 Q 序列
**Decode Kernel**: 高度优化的单 query attention，memory-bound
**POD-Attention**: Prefill-Or-Decode，融合到同一个 kernel launch，减少 launch overhead

### 3.4 Cascade Attention

专为**共享前缀**场景设计：
- 共享前缀的 KV 只计算一次，batch 内所有序列复用
- 与 prefix caching 互补
- 适合 RAG、few-shot 等场景

### 3.5 性能

| 指标 | 提升 |
|------|------|
| inter-token latency | -29~69% (vs compiler backends) |
| 长上下文延迟 | -28~30% |
| 并行生成 | +13~17% |

### 3.6 采用

SGLang, vLLM, TensorRT-LLM, TGI (HuggingFace), MLC-LLM 等均已集成。

## 4. FlashMLA: DeepSeek MLA 专用

### 4.1 背景

DeepSeek-V2/V3 的 Multi-head Latent Attention 通过低秩压缩将 KV Cache 压缩 56.9x。FlashMLA 是专门为 MLA 优化的 kernel。

### 4.2 Kernel 矩阵

| Kernel | GPU | 注意力类型 | 精度 |
|--------|-----|-----------|------|
| Dense Decoding | SM90 | MQA | BF16 |
| Sparse Decoding | SM90/SM100 | MQA | FP8 KV + BF16 |
| Dense Prefill | SM100 | MHA | BF16 |
| Sparse Prefill | SM90/SM100 | MQA | BF16 |

### 4.3 性能数据

| 配置 | 性能 |
|------|------|
| Dense Decode (H800) | 3000 GB/s / 660 TFLOPS |
| Sparse Decode FP8 (H800) | 410 TFLOPS |
| Dense MHA Prefill (B200) | 1460 TFLOPS (forward) |
| Sparse MLA Prefill (H800) | 640 TFLOPS |
| Sparse MLA Prefill (B200) | 1450 TFLOPS |

### 4.4 FP8 KV Cache

每个 token 的 KV Cache 仅 **656 bytes**:
- 512 bytes: 量化 NoPE 部分
- 16 bytes: scale factors
- 128 bytes: RoPE (BF16)

## 5. 选型指南

### 5.1 横向对比

| 特性 | FA2 | FA3 | FA4 | FlashInfer | FlashMLA | SDPA |
|------|-----|-----|-----|-----------|----------|------|
| 训练 | 优 | 优 | 优 | - | - | 良 |
| Serving | 一般 | 一般 | 一般 | **优** | MLA 专用 | 一般 |
| Paged KV | 否 | 否 | 否 | **原生** | **原生** | 否 |
| MLA | 否 | 否 | 否 | 支持 | **专门优化** | 否 |
| Sparse | 否 | 否 | 否 | 支持 | DSA | 否 |
| FP8 | 否 | 是 | 是 | 是 | 是 | 有限 |
| GPU 范围 | Ampere+ | Hopper | Hopper+ | Turing+ | Hopper/BW | 全部 |

### 5.2 场景推荐

**训练**: A100→FA2, H100→FA3/FA4, B200→FA4
**推理 Serving**: 通用→FlashInfer, MLA→FlashMLA, 快速原型→SDPA
**框架**: vLLM→FlashInfer, SGLang→FlashInfer

### 5.3 性能层级 (H100 FP16)

```
FA3:  ~740 TFLOPS (75%)  ← 训练最优
FA2:  ~350 TFLOPS (35%)
FlashInfer: ~FA2 水平, 但 serving 整体更优
SDPA: 取决于 backend, 通常等同 FA2
FlashMLA: MLA 专用, 不同 attention 变体
```

## 6. 趋势 (2025-2026)

1. **Blackwell 适配**: FA4 + FlashInfer 都在适配 SM100/SM120
2. **FP8 默认**: 所有 kernel 全线支持 FP8 推理
3. **稀疏注意力兴起**: FlashMLA DSA, FlashInfer sparse attention
4. **MLA 标准化**: 更多媒体可能采用 MLA 架构
5. **Kernel 融合**: POD-Attention (prefill+decode 融合), fused MoE+Attention
6. **CuTeDSL**: FA4 和 FlashInfer Blackwell 版本都采用 CuTe DSL

## 参考资料

- FlashAttention: https://github.com/Dao-AILab/flash-attention
- FlashAttention-3 Blog: https://tridao.me/blog/2024/flash3/
- FlashInfer: https://github.com/flashinfer-ai/flashinfer (MLSys 2025)
- FlashMLA: https://github.com/deepseek-ai/FlashMLA
- 相关笔记: [FlashAttention](flash-attention.md), [MLA](mla.md), [vLLM MLA Backend](../projects/vllm-mla-backend-reading.md)
