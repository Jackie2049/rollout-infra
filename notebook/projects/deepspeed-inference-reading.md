# DeepSpeed 推理优化架构深度分析

> 作者: Claude + jackie2049 | 日期: 2026-06-05
> 论文: DeepSpeed-FastGen (2023), FP6-LLM (arXiv:2401.14112), ZeroQuant 系列
> 源码: https://github.com/microsoft/DeepSpeed | https://github.com/microsoft/DeepSpeed-MII
> 版本: DeepSpeed-MII v0.3.3 (2025-03), DeepSpeed-Kernels (预编译 wheel)
> 用途: AI Infra 工程师学习笔记, 覆盖 FastGen/Dynamic SplitFuse/量化/Kernel/TP/对比/最新特性

---

## 目录

1. [DeepSpeed-FastGen 推理服务系统与 MII](#1-deepspeed-fastgen-推理服务系统与-mii)
2. [Dynamic SplitFuse: 动态分词融合调度](#2-dynamic-splitfuse-动态分词融合调度)
3. [量化支持: FP6/FP8/INT4](#3-量化支持-fp6fp8int4)
4. [Kernel 优化: Fused Attention 与自定义算子](#4-kernel-优化-fused-attention-与自定义算子)
5. [推理 Tensor Parallelism 与训练 TP 的差异](#5-推理-tensor-parallelism-与训练-tp-的差异)
6. [与 vLLM 的对比分析](#6-与-vllm-的对比分析)
7. [2025-2026 最新特性与现状](#7-2025-2026-最新特性与现状)

---

## 1. DeepSpeed-FastGen 推理服务系统与 MII

### 1.1 系统定位

DeepSpeed-FastGen 是 Microsoft DeepSpeed 团队推出的 LLM 推理服务系统, 通过 **DeepSpeed-MII** (Model Implementations for Inference) 和 **DeepSpeed-Inference** 的协同组合实现高吞吐、低延迟文本生成:

```
┌──────────────────────────────────────────────────────┐
│                 DeepSpeed-FastGen                     │
│            (MII + DeepSpeed-Inference)                │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │   DeepSpeed-MII │    │  DeepSpeed-Inference     │  │
│  │   (前端服务层)   │    │  (后端推理引擎)          │  │
│  │                 │    │                         │  │
│  │ - pipeline()    │    │ - Kernel 优化           │  │
│  │ - serve()       │───>│ - TP 通信               │  │
│  │ - client()      │    │ - KV Cache 管理         │  │
│  │ - Load Balancer │    │ - 模型执行              │  │
│  │ - RESTful API   │    │ - 量化支持              │  │
│  └─────────────────┘    └─────────────────────────┘  │
│                                                      │
│  ┌─────────────────────────────────────────────────┐  │
│  │           Dynamic SplitFuse Scheduler            │  │
│  │   (统一 prefill + decode 的 token 预算调度)      │  │
│  └─────────────────────────────────────────────────┘  │
│                                                      │
│  ┌─────────────────────────────────────────────────┐  │
│  │            Blocked KV Cache (非连续内存)          │  │
│  │       Continuous Batching (迭代级调度)            │  │
│  └─────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 1.2 MII (Model Implementations for Inference)

MII 是开源 Python 库, 提供两层 API:

**非持久化 Pipeline** (适合交互式/测试):

```python
from mii import pipeline

pipe = pipeline("mistralai/Mistral-7B-v0.1")
output = pipe(["Hello, my name is", "DeepSpeed is"], max_new_tokens=128)
print(output)
```

**持久化部署** (适合生产环境):

```python
import mii

# 服务端: 启动 GRPC 服务
mii.serve("mistralai/Mistral-7B-v0.1")

# 客户端: 连接并生成
client = mii.client("mistralai/Mistral-7B-v0.1")
output = client.generate("DeepSpeed is", max_new_tokens=128)
```

**高级部署选项**:

```python
# Tensor Parallelism
client = mii.serve("meta-llama/Llama-2-70b", tensor_parallel=4)

# 多副本 + 负载均衡
client = mii.serve("meta-llama/Llama-2-70b",
                   tensor_parallel=4,
                   replica_num=4)

# RESTful API (v0.2.1+)
client = mii.serve("mistralai/Mistral-7B-v0.1",
                   enable_restful_api=True,
                   restful_api_port=28080)

# 流式输出 (v0.2.1+)
for token in client.generate_stream("DeepSpeed is", max_new_tokens=128):
    print(token, end="", flush=True)
```

### 1.3 支持模型

MII 支持通过 HuggingFace 加载的 **37,000+** 模型, 核心架构包括:

| 架构 | 代表模型 | 特点 |
|------|---------|------|
| LLaMA 系列 | Llama-1/2/3, Code Llama | 最广泛支持 |
| Mistral | Mistral-7B, Mixtral-8x7B | 含 MoE 变体 |
| OPT | OPT-125M ~ OPT-66B | Meta 开源基座 |
| Falcon | Falcon-7B/40B/180B | TII 研究院 |
| Phi | Phi-2, Phi-3 | Microsoft 小模型 |
| Qwen | Qwen, Qwen2, Qwen2-MoE | 阿里云系列 |

### 1.4 DeepSpeed-Kernels

为避免用户漫长的编译等待, DeepSpeed 团队将自定义 CUDA kernel 预编译为 Python wheel:

- **硬件要求**: NVIDIA GPU, Compute Capability 8.0+ (Ampere 及以上: A100, A6000, H100 等)
- **CUDA 版本**: 11.6+
- **操作系统**: Ubuntu 20+
- **安装**: `pip install deepspeed-mii` 时自动安装为依赖

> 注意: 比 Ampere 更老的 GPU (如 V100/SM 7.0, A16/SM 8.6) **可能**需要手动编译 kernel, 部分优化无法使用。

---

## 2. Dynamic SplitFuse: 动态分词融合调度

### 2.1 问题背景: 现有系统的调度缺陷

LLM 推理有两个阶段:

1. **Prefill (提示处理)**: 处理完整 prompt, 构建 KV cache, 单次处理大量 token, compute-bound
2. **Decode (生成)**: 每步生成 1 个 token, memory-bound, 延迟敏感

现有系统的调度策略各有缺陷:

```
=== vLLM: Preemption 策略 ===

时间 →   T1        T2        T3        T4        T5
        ┌─────────────────────────┐┌──────┐┌─────────┐
  请求A │  Gen Gen Gen Gen       ││(暂停) ││ Gen Gen │
        └─────────────────────────┘│      │└─────────┘
              ┌────────────────────────────────────────┐
  请求B       │ Prefill (长 prompt 阻塞所有 decode)     │
              └────────────────────────────────────────┘

问题: 长 prompt 处理时, 所有 decode 被暂停 → 生成延迟飙升

=== Orca: 混合策略 ===

时间 →   T1        T2        T3        T4
  请求A │ Gen      │ Gen      │ Gen      │ Gen
  请求B │ Prefill(全部) │ Gen  │ Gen      │ Gen

问题: 完整 prompt 加入 batch → forward pass 大小不均 → 性能波动
```

### 2.2 三个关键性能洞察

Dynamic SplitFuse 的设计基于三个实验观察:

**洞察 1: Token 数量是性能的决定因素, 而非 batch size (序列数)**

```
性能影响因素:

  Token 数量 (每 forward pass)  ████████████████████  决定性影响
  序列数 (batch size)           ██                    可忽略
  组成比例 (prefill/decode)     █                    可忽略

结论: 调度器只需控制一个信号 → forward pass 中的总 token 数
```

**洞察 2: 吞吐量曲线呈凹函数, 存在饱和点**

```
吞吐量 (tokens/s)
  │
  │              ╭─────────────────────  计算饱和区
  │           ╭──╯
  │        ╭──╯    ← 陡峭过渡区
  │      ╭─╯
  │    ╭─╯
  │  ╭─╯           ← 内存瓶颈区 (小 batch)
  │╭─╯
  ╰─────────────────────────────────────── Token 数量
       ↑                 ↑
    少量 token       大量 token
   (memory-bound)  (compute-bound)

关键: 让所有 forward pass 都在饱和区运行 → 最高效
```

**洞察 3: 凹函数的等分最优性**

对于凹函数 f(x), 恒有: `2f(x) >= f(x+h) + f(x-h)`

即: **将 P 个 token 均分到 F 个 forward pass 中, 吞吐量最大**。

### 2.3 Dynamic SplitFuse 策略

Dynamic SplitFuse 的核心思想: **固定每个 forward pass 的 token 预算, 动态混合 prefill 和 decode token**:

```
=== Dynamic SplitFuse 策略 ===

目标: 每个 forward pass 固定 Token Budget = B

时间 →     T1          T2          T3          T4          T5
         ┌───────────┬───────────┬───────────┬───────────┬──────┐
  请求A  │ 64 decode │ 64 decode │ 64 decode │ 64 decode │ 64 d │
         │  tokens   │  tokens   │  tokens   │  tokens   │      │
  请求B  │ 128 prefill│ 128 prefill│ 128 prefill│ 64 prefill│      │
  (长)   │  chunk 1  │  chunk 2  │  chunk 3  │  chunk 4  │ (完) │
  请求C  │ 64 prefill│           │           │           │ 64 d │
  (短)   │ (完整)    │           │           │           │      │
         ├───────────┼───────────┼───────────┼───────────┼──────┤
  总计   │ =256 tok  │ =192 tok  │ =192 tok  │ =128 tok  │=128  │
         │ (≈B)      │ (≈B)      │ (≈B)      │ (≈B)      │(≈B)  │

关键:
1. 长 prompt (512 tok) → 拆成 4 个 128-token chunk, 分 4 个 forward pass 处理
2. 短 prompt → 直接填入当前 pass 剩余空间
3. Decode token 持续进行, 不会因 prefill 被暂停
```

### 2.4 两个核心行为

**行为 1: 长 prompt 分块处理**

- 将长 prompt 分解为多个小 chunk (通常 64-256 tokens/chunk)
- 每个 forward pass 只处理一个 chunk, 与 decode 混合
- 最后一个 chunk 处理完后才开始该请求的 decode
- 好处: 不阻塞其他请求的 decode, 延迟更可预测

**行为 2: 短 prompt 拼接填充**

- 多个短 prompt 拼接, 填满 token budget
- 如果某个短 prompt 无法完全填入, 也会被拆分
- 确保每个 forward pass 大小一致
- 好处: 模型始终在高吞吐区域运行

### 2.5 性能收益

| 指标 | Dynamic SplitFuse | vLLM (preemption) | 提升 |
|------|-------------------|-------------------|------|
| **有效吞吐** | 1.42 qps (70B/A100x4) | 0.63 qps | **2.3x** |
| **P95 延迟** | 低 | 高 3.7x | **3.7x 降低** |
| **SLA 违规率** | <1% | 28% | 显著改善 |
| **生成延迟方差** | 低 (稳定) | 高 (毛刺) | 显著改善 |

有效吞吐的定义: 满足 SLA 的请求数/秒
- Prompt 延迟 SLA: |prompt tokens| / 512 秒 (= 512 tok/s)
- 生成 EMA 延迟 SLA: 2/4/6 tok/s

---

## 3. 量化支持: FP6/FP8/INT4

### 3.1 DeepSpeed 量化研究谱系

```
量化研究时间线:

2022 ─── ZeroQuant ──────────────── 权重 INT8/FP8 + 激活 INT8
  │                                  (Post-Training Quantization)
  │
2023 ─── ZeroQuant-V2 ───────────── 改进精度, 分组量化
  │                                  (Groupwise Quantization)
  │
2023 ─── ZeroQuant-FP ───────────── FP8 权重量化
  │                                  (FP8 E4M3/E5M2)
  │
2024 ─── FP6-LLM / TC-FPx ──────── FP6 量化 + Tensor Core kernel
  │                                  (首个 FP6 GPU kernel)
  │
2024 ─── ZeroQuant-HERO ────────── 硬件感知量化
                                         (Hardware-Efficient)
```

### 3.2 FP6-LLM: 核心技术突破

**FP6-LLM** (arXiv:2401.14112) 是 DeepSpeed 团队在 FP6 量化方面的核心工作:

**挑战**:
1. FP6 是非标准位宽 (不是 8 的倍数), GPU 内存访问不友好
2. 权重反量化开销高, 可能抵消量化收益
3. 现有 Tensor Core 不原生支持 FP6

**TC-FPx Kernel 设计**:

```
┌─────────────────────────────────────────────────────┐
│               TC-FPx Kernel 架构                     │
│                                                      │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │ 权重存储      │    │ 运行时反量化 (Runtime)    │   │
│  │ (FP6/FP8/等) │───>│                          │   │
│  │              │    │  1. 从全局内存读取 FPx 权重│   │
│  │ Bit-level    │    │  2. 在 Shared Memory 中   │   │
│  │ Packing      │    │     反量化为 FP16/BF16    │   │
│  │ (减少内存占用)│    │  3. 写入 Fragment 寄存器  │   │
│  └──────────────┘    │  4. Tensor Core 执行 GEMM │   │
│                      └──────────────────────────┘   │
│                                                      │
│  关键优化:                                           │
│  - 统一接口: 支持 FP4/FP5/FP6/FP8 多种位宽          │
│  - Bit-level Packing: 减少非对齐位宽的内存浪费       │
│  - 在线反量化: 读取时即时转换, 无需预存储 FP16       │
│  - Tensor Core 兼容: 输出格式匹配 MMA 指令要求      │
└─────────────────────────────────────────────────────┘
```

**性能数据** (LLaMA-70B, 单 GPU):

| 配置 | 归一化吞吐 | 加速比 | 说明 |
|------|-----------|--------|------|
| FP16 (2xA100-80GB) | 基线 | 1.0x | 基准 |
| FP6 (1xA100-80GB) | 1.69-2.65x | 1.69-2.65x | 单 GPU 即可运行 |

FP6 的关键优势: **70B 模型单 GPU 部署** (FP16 需 2 张 A100-80GB, FP6 只需 1 张)

### 3.3 量化精度对比

```
量化精度 (以 LLaMA 系列为代表):

  FP16 (baseline)  │████████████████████████│  基线精度
  FP8 (E4M3)       │███████████████████████│  <0.5% 损失
  FP6              │███████████████████████│  <1% 损失 (取决于校准集)
  INT8 (W8A8)      │██████████████████████ │  ~1-2% 损失
  INT4 (W4A16)     │████████████████████   │  ~2-5% 损失
  INT4 (GPTQ/AWQ)  │█████████████████████  │  ~2-4% 损失 (per-group)

  注意: 精度损失高度依赖模型和任务, 以上为典型范围
```

### 3.4 MII 中的量化配置

从 v0.2.3 开始, MII 支持量化配置选项:

```python
import mii

# 使用量化模型
client = mii.serve("TheBloke/Llama-2-7B-Chat-GPTQ",
                   quant_config={"quant_method": "gptq"})
```

支持的量化方法:
- **GPTQ**: Post-training 量化, 4-bit 权重量化
- **AWQ**: Activation-aware 权重量化
- **FP6**: 通过 FP6-LLM kernel 支持 (需要 DeepSpeed-Kernels)
- **FP8/INT8**: 通过 ZeroQuant 系列技术

---

## 4. Kernel 优化: Fused Attention 与自定义算子

### 4.1 DeepSpeed 推理 Kernel 栈

```
┌─────────────────────────────────────────────────────┐
│              DeepSpeed 推理 Kernel 栈                │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │          DeepSpeed-Kernels (预编译 wheel)       │  │
│  │                                                │  │
│  │  ┌──────────────┐  ┌──────────────────────┐   │  │
│  │  │ Flash        │  │ MoE Fused Kernel      │   │  │
│  │  │ Attention    │  │ (基于 FasterTransformer)│  │  │
│  │  │ (修改版)     │  │                        │   │  │
│  │  └──────────────┘  └──────────────────────┘   │  │
│  │  ┌──────────────┐  ┌──────────────────────┐   │  │
│  │  │ QKV Fusion   │  │ TC-FPx 量化 Kernel    │   │  │
│  │  │ ( fused QKV │  │ (FP6/FP8 反量化)      │   │  │
│  │  │   投影)      │  │                        │   │  │
│  │  └──────────────┘  └──────────────────────┘   │  │
│  │  ┌──────────────┐  ┌──────────────────────┐   │  │
│  │  │ GeLU/MLP     │  │ Residual Add +       │   │  │
│  │  │ Fusion       │  │ LayerNorm Fusion     │   │  │
│  │  └──────────────┘  └──────────────────────┘   │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │         依赖的外部库                            │  │
│  │  - FlashAttention (修改版, 文件头注明来源)      │  │
│  │  - FasterTransformer (MoE kernel 部分)         │  │
│  │  - CUTLASS (GEMM 模板)                         │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 4.2 Fused Attention Kernel

DeepSpeed-FastGen 使用修改版 FlashAttention kernel:

**与标准 FlashAttention 的区别**:

| 特性 | 标准 FlashAttention | DeepSpeed 修改版 |
|------|-------------------|-----------------|
| **KV Cache 格式** | 连续内存 | Blocked (分块/分页) |
| **Batch 组成** | 均匀序列 | 混合 prefill + decode |
| **序列长度** | 对齐 | 动态可变 (SplitFuse) |
| **Block 管理** | 无 | 间接寻址 (block table) |

**Blocked KV Cache 的 Attention 实现**:

```
标准 Attention:
  Q × K^T → S → softmax → × V → O
  K, V 在连续内存中

Blocked KV Cache Attention:
  ┌───────────────────────────────────┐
  │  Block Table (间接寻址表)         │
  │  ┌───┬───┬───┬───┬───┬───┐      │
  │  │ 3 │ 7 │ 1 │ 5 │ 2 │...│  ← 逻辑 → 物理映射
  │  └───┴───┴───┴───┴───┴───┘      │
  │                                   │
  │  Physical KV Cache Pool:          │
  │  Block 0: [KV data...]           │
  │  Block 1: [KV data...] ← 使用中  │
  │  Block 2: [KV data...] ← 使用中  │
  │  Block 3: [KV data...] ← 使用中  │
  │  Block 4: [free...]              │
  │  ...                             │
  └───────────────────────────────────┘

  Attention 过程:
  1. 查 Block Table 获取物理 block 地址
  2. 按序加载 KV block
  3. 执行分块 attention (类似 FlashAttention 的 tiling)
  4. 在线 softmax (避免 O(N^2) 内存)
```

### 4.3 QKV Fusion Kernel

将 QKV 三个线性投影融合为单个 GEMM:

```
=== 非融合 ===
Q = x @ W_Q    ← GEMM 1
K = x @ W_K    ← GEMM 2  (3次 kernel launch + 2次额外内存读写)
V = x @ W_V    ← GEMM 3

=== 融合 ===
              ┌─────┐
              │ W_Q │
x ──GEMM──>  │ W_K │  ← 单个 GEMM (1次 kernel launch + 1次内存读写)
              │ W_V │
              └─────┘
QKV = x @ [W_Q | W_K | W_V]  → 然后切分为 Q, K, V

收益:
- Kernel launch 开销: 3 → 1 (减少 ~2 * 10us = 20us/layer)
- 内存读写: 3 次输入读取 → 1 次 (节省 2x 输入内存带宽)
- 对 decode (memory-bound) 尤其重要: 内存带宽节省 ~15-20%
```

### 4.4 MLP + GeLU Fusion

```
=== 标准 MLP ===
gate = x @ W_gate          ← GEMM 1
gate = GeLU(gate)          ← Element-wise kernel
up   = x @ W_up            ← GEMM 2
out  = gate * up            ← Element-wise kernel
out  = out @ W_down         ← GEMM 3

=== 融合后 ===
gate = x @ W_gate
gate = GeLU(gate)           ← fused into GEMM epilogue
up   = x @ W_up
out  = gate * up            ← fused into next GEMM prologue
out  = out @ W_down

收益:
- 减少 intermediate tensor 的 global memory 读写
- Element-wise 操作融入 GEMM 的 epilogue/prologue
- Prefill (compute-bound) 下收益有限, Decode (memory-bound) 下收益显著
```

### 4.5 MoE 专用 Kernel

DeepSpeed 的 MoE 推理 kernel 基于 FasterTransformer, 针对 Mixtral 等模型优化:

```
MoE 推理流程:

Input Tokens
     │
     ▼
  Router (GEMM + Top-K)
     │
     ├──> Expert 1 (GEMM) ──┐
     ├──> Expert 2 (GEMM) ──┤
     ├──> ...               ├──> Weighted Sum → Output
     └──> Expert N (GEMM) ──┘

优化点:
1. Expert GEMM: 小 batch size, 利用 grouped GEMM
2. Token 排序: 按分配的 expert 重排 token, 减少零填充
3. 融合 Router: GEMM + Top-K + Softmax 融合
```

---

## 5. 推理 Tensor Parallelism 与训练 TP 的差异

### 5.1 DeepSpeed 推理 TP

DeepSpeed-MII 通过 `tensor_parallel` 参数或 `--num_gpus` 启用 TP:

```python
# 4-way TP for 70B model
mii.serve("meta-llama/Llama-2-70b-hf", tensor_parallel=4)

# 命令行
mii serve meta-llama/Llama-2-70b-hf --num_gpus 4
```

### 5.2 推理 TP vs 训练 TP 对比

```
┌─────────────────────────────────────────────────────────────┐
│                    训练 TP vs 推理 TP                         │
├─────────────────┬───────────────────┬───────────────────────┤
│     维度        │     训练 TP       │     推理 TP           │
├─────────────────┼───────────────────┼───────────────────────┤
│ 通信原语        │ AllReduce         │ AllReduce             │
│                 │ (每层 2 次)       │ (每层 2 次)           │
├─────────────────┼───────────────────┼───────────────────────┤
│ 通信频率        │ 前向+反向 = 每层  │ 仅前向 = 每层 2 次    │
│                 │ 4 次 AllReduce    │ AllReduce             │
├─────────────────┼───────────────────┼───────────────────────┤
│ 通信量/层       │ 4 × 4B × H       │ 2 × 4B × H           │
│ (MLP+Attn)      │ (fwd+bwd 各2次)  │ (fwd 2次)             │
├─────────────────┼───────────────────┼───────────────────────┤
│ 通信占比        │ ~10-20%           │ ~5-15%                │
│                 │ (compute-bound)   │ (decode memory-bound  │
│                 │                   │  但通信仍重要)         │
├─────────────────┼───────────────────┼───────────────────────┤
│ 是否需要异步    │ 必须              │ 可选但推荐            │
│ 通信隐藏        │ (训练关键路径)    │ (decode 通信占比更小) │
├─────────────────┼───────────────────┼───────────────────────┤
│ Batch size      │ 大 (micro batch   │ Prefill: 大           │
│                 │  1-32)            │ Decode: 1 token/req   │
├─────────────────┼───────────────────┼───────────────────────┤
│ 权重分布        │ 每张卡存完整权重  │ 每张卡只存 1/TP 的权重│
│                 │ (ZeRO 才分片)     │ (天然分片)            │
├─────────────────┼───────────────────┼───────────────────────┤
│ KV Cache        │ 不持久化          │ 持久化, 按 TP 分片    │
│                 │                   │ (每张卡 1/TP 的 KV)   │
├─────────────────┼───────────────────┼───────────────────────┤
│ Pipeline        │ GPipe/1F1B/       │ 一般不用 PP           │
│ Parallelism     │ Interleaved       │ (延迟敏感,气泡太大)   │
├─────────────────┼───────────────────┼───────────────────────┤
│ 容错            │ checkpoint 恢复   │ 不需要 (无状态推理)   │
└─────────────────┴───────────────────┴───────────────────────┘
```

### 5.3 推理 TP 的关键差异

**1. 通信量减半**

训练中每层需要 4 次 AllReduce (前向 2 次 + 反向 2 次), 推理只需 2 次 (仅前向)。但 decode 阶段每步只处理 1 token/request, GEMM 极小, AllReduce 的固定延迟可能成为瓶颈。

**2. KV Cache 分片**

```
单 GPU:          TP=2:
┌──────────┐    ┌──────────┐ ┌──────────┐
│ 全部 KV  │    │ KV 前半  │ │ KV 后半  │
│ (H dim)  │    │ (H/2)    │ │ (H/2)    │
└──────────┘    └──────────┘ └──────────┘
   GPU 0           GPU 0        GPU 1

每张卡 KV Cache 内存减少 TP 倍, 可以服务更多并发请求
```

**3. 不使用 Pipeline Parallelism**

推理场景中, PP 的气泡延迟 (`(P-1)/(M+P-1)`) 不可接受, 因为:
- 推理要求低延迟, 气泡直接增加 TTFT 和 TPOT
- 训练中可以通过增大 micro-batch 数量 (M>>P) 减小气泡
- 推理中 batch 可能很小 (如 1 token/step), 气泡占比极大

DeepSpeed 推理优先使用 TP, 对于超大模型 (>70B) 也只用 TP, 不用 PP。

**4. Batch Size 动态变化**

```
训练: batch_size 固定, GEMM shape 一致 → 可以优化通信 schedule
推理: batch_size 动态 (SplitFuse), 混合 prefill + decode → 通信模式不一致

SplitFuse 下的 TP 通信:
  Prefill token: 大 GEMM → AllReduce 占比小 (~5%)
  Decode token:  小 GEMM → AllReduce 占比大 (~15-20%)
  混合 batch:   介于两者之间
```

---

## 6. 与 vLLM 的对比分析

### 6.1 架构级对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    DeepSpeed-FastGen vs vLLM                     │
├──────────────────┬──────────────────────┬───────────────────────┤
│    维度          │  DeepSpeed-FastGen   │  vLLM (V1)            │
├──────────────────┼──────────────────────┼───────────────────────┤
│ 调度策略         │ Dynamic SplitFuse    │ Chunked Prefill       │
│                  │ (固定 token budget,  │ (动态 chunk 大小,     │
│                  │  混合 prefill+decode)│  preemption 策略)     │
├──────────────────┼──────────────────────┼───────────────────────┤
│ KV Cache         │ Blocked KV Cache     │ Paged Attention       │
│                  │ (类似 Paged Attn)    │ (block_size=16)       │
├──────────────────┼──────────────────────┼───────────────────────┤
│ Continuous       │ 有 (SplitFuse 内置)  │ 有 (iteration-level)  │
│ Batching         │                      │                       │
├──────────────────┼──────────────────────┼───────────────────────┤
│ 前端 API         │ MII Python/GRPC/REST │ OpenAI-compatible     │
│                  │                      │ HTTP API              │
├──────────────────┼──────────────────────┼───────────────────────┤
│ Kernel 来源      │ DeepSpeed-Kernels    │ Triton + FlashInfer   │
│                  │ (预编译 CUDA/CUTLASS)│ + 自研 V1 backend     │
├──────────────────┼──────────────────────┼───────────────────────┤
│ 模型支持         │ ~37K (HF hub)        │ 200+ 架构 (开箱即用)  │
│                  │ 10 核心架构          │                       │
├──────────────────┼──────────────────────┼───────────────────────┤
│ 量化方法         │ FP6/FP8/INT8/GPTQ   │ 30+ 量化方法          │
│                  │ (ZeroQuant 系列)     │ (AWQ/GPTQ/FP8/INT4..)│
├──────────────────┼──────────────────────┼───────────────────────┤
│ Prefix Caching   │ 无专用机制           │ BlockHash Prefix      │
│                  │                      │ Caching (V1)          │
├──────────────────┼──────────────────────┼───────────────────────┤
│ Speculative      │ 无                   │ 8+ Proposer           │
│ Decoding         │                      │ (Eagle/N-gram等)      │
├──────────────────┼──────────────────────┼───────────────────────┤
│ 结构化输出       │ 无                   │ xGrammar/Guidance/    │
│                  │                      │ Outlines/LMFE         │
├──────────────────┼──────────────────────┼───────────────────────┤
│ P/D 分离         │ 无                   │ NIXL/FlexKV connector │
├──────────────────┼──────────────────────┼───────────────────────┤
│ 硬件支持         │ NVIDIA only          │ NVIDIA/AMD/TPU/       │
│                  │ (Ampere+)            │ Intel/Ascend          │
├──────────────────┼──────────────────────┼───────────────────────┤
│ LoRA Serving     │ 无                   │ Punica Multi-LoRA     │
├──────────────────┼──────────────────────┼───────────────────────┤
│ 负载均衡         │ 内置 Replica LB      │ 无内置 (外部 LB)      │
├──────────────────┼──────────────────────┼───────────────────────┤
│ 可观测性         │ 基础日志             │ Prometheus 30+ 指标   │
├──────────────────┼──────────────────────┼───────────────────────┤
│ 社区活跃度       │ 低 (2024后放缓)      │ 极高 (持续更新)       │
│ (2024-2026)      │                      │                       │
└──────────────────┴──────────────────────┴───────────────────────┘
```

### 6.2 调度策略深度对比

```
场景: 长prompt (2600 tok) + 短生成 (60 tok), 16 并发客户端

=== vLLM (Preemption) ===

时间 ───────────────────────────────────────────────>
     │<── 长 prefill ──>│<── decode ──>│<── 长 prefill ──>│
     │    (阻塞 decode)  │   (恢复)    │   (又阻塞)      │

  Decode 延迟: ░░░░░░░░████████████░░░░░░░░████████
                                ↑                ↑
                              毛刺!            毛刺!
  P50: ~50ms  P95: ~400ms

=== DeepSpeed-FastGen (Dynamic SplitFuse) ===

时间 ───────────────────────────────────────────────>
     │pf│pf│pf│pf│d│pf│pf│pf│pf│d│d│d│pf│pf│d│d│
     │  │  │  │  │ │  │  │  │  │ │ │ │  │  │ │ │
     (所有 forward pass 大小一致, decode 从不暂停)

  Decode 延迟: ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
  P50: ~50ms  P95: ~110ms (稳定, 无毛刺)

关键差异:
- vLLM P95 延迟是 FastGen 的 3.7 倍
- vLLM 28% 请求违反 4 tok/s SLA, FastGen 仅 <1%
- FastGen 有效吞吐 2.3x
```

### 6.3 选型建议

```
┌──────────────────────────────────────────────────────────┐
│                      选型决策树                           │
│                                                          │
│  你的场景是什么?                                          │
│       │                                                  │
│       ├──> 需要最新特性 (Spec Dec/LoRA/结构化输出)?       │
│       │         └──> vLLM (生态更全)                     │
│       │                                                  │
│       ├──> 需要多硬件 (AMD/TPU/Intel)?                   │
│       │         └──> vLLM 或 SGLang                     │
│       │                                                  │
│       ├──> 生产环境, 需要低 P95 延迟 + 长 prompt?        │
│       │         └──> 考虑 FastGen (SplitFuse 优势)       │
│       │              (但需评估维护风险)                   │
│       │                                                  │
│       ├──> 已在使用 DeepSpeed 训练?                      │
│       │         └──> FastGen (统一技术栈)                │
│       │                                                  │
│       ├──> 需要 FP6 量化部署 70B 单卡?                   │
│       │         └──> FastGen + FP6-LLM                  │
│       │                                                  │
│       └──> 其他场景?                                     │
│                 └──> vLLM (更活跃的社区, 更多特性)        │
└──────────────────────────────────────────────────────────┘
```

### 6.4 性能对比数据

**Llama-2 70B, 4xA100-80GB** (DeepSpeed-FastGen 博客数据, 2023):

| 指标 | DeepSpeed-FastGen | vLLM | 说明 |
|------|-------------------|------|------|
| 等延迟吞吐 | 1.36 rps vs 0.67 rps | **2.0x** | 相同延迟下吞吐翻倍 |
| 等吞吐延迟 | 7s vs 14s | **2.0x** | 相同吞吐下延迟减半 |
| 有效吞吐 | 1.42 qps vs 0.63 qps | **2.3x** | SLA 满足率差异大 |
| P95 生成延迟 | ~110ms vs ~400ms | **3.7x** | SplitFuse 消除毛刺 |
| 16副本扩展 | 23.7 qps (线性) | N/A | 内置负载均衡 |

> 注意: 以上数据来自 2023 年 FastGen 发布时, vLLM 此后已大幅改进 (V1 架构, Chunked Prefill 等), 差距可能缩小。

---

## 7. 2025-2026 最新特性与现状

### 7.1 项目活跃度

DeepSpeed-MII 和 DeepSpeed-FastGen 在 2024 年后更新放缓:

| 版本 | 日期 | 主要变更 |
|------|------|---------|
| v0.3.3 | 2025-03 | 构建系统更新 (pyproject.toml) |
| v0.3.2 | 2025-03 | DCO 替代 CLA, pydantic v2 支持 |
| v0.3.1 | 2024-10 | Streaming 支持, 模型列表更新 |
| v0.3.0 | 2024-08 | Pydantic v2 迁移, scheduling 修复 |
| v0.2.4 | 2024-07 | Llama-3 支持, KV cache starvation 修复 |
| v0.2.3 | 2024-03 | 量化配置选项 (quant_config) |
| v0.2.2 | 2024-02 | EOS token 修复, stop 功能 |
| v0.2.1 | 2024-02 | OpenAI RESTful API, Token streaming |
| v0.2.0 | 2024-01 | **Mixtral/Phi-2/Falcon 支持** |

### 7.2 当前项目状态评估

```
DeepSpeed 推理项目状态 (2025-2026):

活跃度:    ██░░░░░░░░░░░░░░░░░░  低
           (v0.3.x 主要是维护性更新, 无重大新特性)

社区参与:  ████░░░░░░░░░░░░░░░░  中等偏低
           (Issues 响应较慢, 核心贡献者集中于 Microsoft)

特性丰富度:████████████░░░░░░░░░  中等
           (基础推理功能完善, 缺少 Spec Dec/LoRA 等)

生态集成:  ██████░░░░░░░░░░░░░░  中等
           (HF 模型支持好, 但缺少 OpenAI 兼容性之外的集成)

对比 vLLM:
vLLM 在 2024-2026 期间快速发展:
- V1 架构重写 (Scheduler/Executor/BlockPool)
- 30+ 量化方法
- 8+ Speculative Decoding 方案
- P/D 分离 (NIXL/FlexKV connector)
- Multi-LoRA (Punica)
- 结构化输出 (xGrammar)
- 200+ 模型架构支持
```

### 7.3 DeepSpeed 生态整体方向

虽然推理部分更新放缓, DeepSpeed 整体仍在以下方向推进:

**1. 训练优化 (核心优势)**

DeepSpeed 的核心价值仍在训练侧:
- ZeRO 优化器 (ZeRO-1/2/3/3+, ZeRO-Infinity)
- 3D 并行 (TP/PP/DP)
- DeepSpeed-Chat (RLHF 训练框架)
- Long context 训练 (DeepSpeed-Ulysses)

**2. FP6 量化研究延续**

FP6-LLM/TC-FPx 的研究成果可能被集成到未来版本:
- 更广泛的 FP6 kernel 支持 (Blackwell 架构)
- 自动量化流水线
- 端到端量化感知训练

**3. 与其他框架的对比趋势**

```
2024-2026 推理框架格局:

vLLM:      ████████████████████  生态最全, 社区最活跃
SGLang:    ████████████████     RadixAttention + 零开销调度
TRT-LLM:   ████████████████     NVIDIA 官方, 编译优化
DeepSpeed: ████████             SplitFuse 优势, 但更新放缓
LMDeploy:  ██████████           商汤出品, 量化部署
```

### 7.4 关键技术遗产

尽管更新放缓, DeepSpeed-FastGen 的技术贡献仍然重要:

1. **Dynamic SplitFuse 思想**: 固定 token budget + prefill/decode 混合调度的理念被其他框架借鉴
2. **FP6 量化**: TC-FPx kernel 设计为非标准位宽量化提供了完整的 GPU 实现参考
3. **Blocked KV Cache**: 与 vLLM PagedAttention 类似的非连续 KV cache 设计
4. **Effective Throughput SLA**: 提出 prompt 延迟 + 生成 EMA 延迟的综合评估框架

### 7.5 实际部署建议 (2025-2026)

```
=== 选择 DeepSpeed-FastGen 的场景 ===

1. 已在 DeepSpeed 训练生态中, 需要统一技术栈
2. 长 prompt 工作负载 (SplitFuse 的核心优势)
3. FP6 量化需求 (70B 单卡部署)
4. Microsoft Azure 环境 (可能有更好的官方支持)

=== 不推荐 DeepSpeed-FastGen 的场景 ===

1. 需要 Speculative Decoding
2. 需要 Multi-LoRA serving
3. 需要 AMD/Intel GPU 支持
4. 需要最新模型架构的快速支持
5. 需要活跃社区和快速 issue 解决
6. 需要 P/D 分离 (Prefill/Decode disaggregation)
7. 需要结构化输出 (JSON/Regex 约束)

=== 替代方案 ===

- vLLM: 通用 LLM serving, 特性最全
- SGLang: 多轮对话/prefix-sharing 密集场景
- TRT-LLM: NVIDIA GPU 极致性能, 编译期优化
```

---

## 总结

| 主题 | 核心要点 |
|------|---------|
| **FastGen + MII** | MII (前端) + DeepSpeed-Inference (后端) 的协同架构, 支持 37K+ 模型 |
| **Dynamic SplitFuse** | 固定 token budget, 长prompt 分块 + 短 prompt 拼接, 消除 decode 毛刺, 2.3x 有效吞吐 |
| **量化** | ZeroQuant 系列研究 (INT8→FP8→FP6), FP6-LLM/TC-FPx 实现首个 FP6 Tensor Core kernel, 70B 单卡部署 |
| **Kernel** | FlashAttention (修改版), QKV Fusion, MLP+GeLU Fusion, MoE kernel (FasterTransformer), 预编译 wheel |
| **推理 TP** | 与训练 TP 核心差异: 仅前向通信量减半, 不用 PP, KV Cache 按 TP 分片, 动态 batch |
| **vs vLLM** | FastGen 优势在 SplitFuse 调度和长 prompt P95 延迟; vLLM 优势在生态/特性/社区/硬件支持 |
| **2025-2026 现状** | 维护模式更新, 无重大新特性, 核心价值在 SplitFuse 思想和 FP6 量化研究遗产 |

---

## 参考资料

1. DeepSpeed-FastGen Blog: https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen
2. DeepSpeed-MII: https://github.com/microsoft/DeepSpeed-MII
3. FP6-LLM Paper: arXiv:2401.14112 - "FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design"
4. ZeroQuant: https://arxiv.org/abs/2206.01861
5. DeepSpeed Main Repo: https://github.com/microsoft/DeepSpeed
6. DeepSpeed-Kernels: 随 deepspeed-mii 自动安装
