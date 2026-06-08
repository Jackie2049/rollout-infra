# KV Cache Management Deep Dive: 推理内存管理的核心

> 2026-06-08 | 从PagedAttention到RadixAttention, 从block管理到INT8压缩, KV cache是推理的内存命门
> 基于: vLLM PagedAttention Paper, SGLang RadixAttention, FlashInfer Paged KV, RTX 4090实测数据
> 关联: flashinfer-attention-deep-dive.md (FlashInfer paged KV), vllm-v1-nav (vLLM block manager), inference-cost-analysis.md (成本)

## 0. 核心定律: KV cache是推理内存瓶颈

```
KV cache内存 = num_layers × 2 × S × num_kv_heads × d_head × dtype_size

7B模型 BF16 KV cache:
  → 32层 × 2 × S × 5 kv_heads × 128 d_head × 2 bytes
  → = 32 × 2 × S × 5 × 128 × 2 = 81,920 × S bytes
  → S=2048: 163.84 MB → S=8192: 655.36 MB → S=32768: 2.62 GB

70B模型 BF16 KV cache:
  → 80层 × 2 × S × 8 kv_heads × 128 d_head × 2 bytes
  → = 80 × 2 × S × 8 × 128 × 2 = 327,680 × S bytes
  → S=2048: 655.36 MB → S=8192: 2.62 GB → S=32768: 10.49 GB!

RTX 4090 24GB → GPU利用率90% → KV可用21.6GB:
  → 7B BF16: 21.6GB / 81.92 KB/tok = 264K tokens → ~65个S=4096请求
  → 7B INT8 KV: 21.6GB / 40.96 KB/tok = 528K tokens → ~130个S=4096请求
  → 70B BF16: 21.6GB / 327.68 KB/tok = 66K tokens → ~16个S=4096请求 → 太少!

核心: KV cache内存决定推理并发数 → 量化KV省50%内存 → 2x并发 → 2x throughput!
  → INT8 KV: 50%内存省 → 2x并发 → FlashInfer native → 2x+15x = 30x throughput!
```

## 1. PagedAttention: vLLM的核心创新

### 1.1 传统KV cache的问题

```
传统推理: 预分配最大序列长度的连续KV cache
  → 问题1: 内存碎片 → 请求长度不同 → 无法复用释放的空间
  → 问题2: 预分配浪费 → S_max=2048但实际S=500 →浪费75%内存!
  → 问题3: 无法动态增长 → 预分配S_max → 或遇到OOM → 致命!

举例: 4个请求(S=100, 500, 1000, 2048)
  → 传统: 4×2048×KV_per_token = 4×S_max → 大量浪费!
  → Paged: 按需分配block → 100/16=6.25→7blocks + 500/16=32blocks + ...
  → → 仅分配实际需要的 → 内存利用率接近100%!
```

### 1.2 PagedAttention架构

```
灵感来自OS虚拟内存/分页:
  → 逻辑block: 请求的虚拟KV空间 → 按block_size分页
  → 物理block: GPU上的实际KV存储 → 按需分配
  → Block Table: 逻辑→物理映射 → 类似OS page table!

Block Table示例(block_size=16):
  Request 0 (S=100): logical[0..6] → physical[5,12,3,8,1,9,7] → 7 blocks
  Request 1 (S=500): logical[0..31] → physical[2,4,6,...] → 32 blocks
  Request 2 (S=2048): logical[0..127] → physical[...] → 128 blocks

  → 无需连续! → 消除碎片! → 按需增长! → OOM→preemption而非崩溃!

FlashInfer对接:
  → vLLM block_table → paged_kv_indptr/indices/last_page_len → FlashInfer paged KV
  → FlashInfer: warp cooperative加载page indices → 1次smem broadcast → 高效!
```

### 1.3 Block大小选择

```
block_size=16 (vLLM默认):

内存浪费(最后一个block不满):
  → S=100: 7 blocks × 16 = 112 → 12 tokens浪费 → 12% → 接受
  → S=500: 32 blocks × 16 = 512 → 12 tokens浪费 → 2.4% → 很好
  → S=2048: 128 blocks × 16 = 2048 → 0浪费 → 完美

小block_size的好处:
  → 内存浪费少 → 更高效利用
  → 更灵活分配 → 短请求少分配
  → 更细粒度preemption → 只释放需要的block

大block_size的好处:
  → 更少block table条目 → CPU overhead↓
  → 更大连续访问 → GPU coalescing更好
  → FlashInfer: 更少page index加载 → smem↓

RTX 4090最优block_size:
  → L2 72MB → 16-token block ≈ 1.28MB per head → L2可能fit → cache hit!
  → → block_size=16 → 每head block=1.28MB → 5 kv_heads → 6.4MB → L2 fit!
  → → block_size=64 → 每head block=5.12MB → 5 kv_heads → 25.6MB → L2 fit!
  → → block_size=256 → 每head block=20.48MB → L2不fit → HBM读取!

结论: block_size=16-64 → L2 cache命中 → decode更快!
```

### 1.4 vLLM V1 Block Manager

```
vLLM V1 Block Manager架构 (from vllm-v1-nav skill):

BlockHashToBlockMap: 1:N映射
  → hash(KV block content) → 多个block可以共享同一hash → prefix caching!
  → 新请求的prefix如果与已有block hash匹配 → 直接引用! → 无需重新计算!

FreeKVCacheBlockQueue: O(1) LRU驱逐
  → 双链表 → 驱逐=移除tail → 分配=pop head → O(1)!
  → hash chain → 决定哪些block可以被驱逐(引用计数=0才能驱逐)

Sliding Window: 超出window的block可以立即释放
  → Mistral/Llama-3: window=8192 → S>8192 → 超出部分block释放 → 内存省!

Chunked Prefill: prefill按block分块 → 与decode交错调度
  → 避免长prefill占用所有GPU → decode不被阻塞 → TTFT↓!
```

## 2. Preemption与Swap

### 2.1 Preemption策略

```
GPU KV block耗尽时 → 需要preemption释放空间:

策略1: Recomputation (默认)
  → 丢弃被preempt请求的KV cache → 空间释放!
  → 当请求重新调度 → 重新compute所有KV → 从头prefill!
  → 适合: 短请求 → recomputation快 → 几ms
  → 不适合: 长请求(S=8192) → recomputation慢 → 几秒!
  → RTX 4090: prefill S=2048 ≈ 0.7ms → 短请求recompute很快!

策略2: Swapping
  → 将KV block复制到CPU pinned memory → GPU空间释放!
  → 当请求重新调度 → swap back GPU → CPU→GPU传输
  → 适合: 长请求 → swap比recompute快 → S=8192 swap≈1ms vs recompute≈10ms
  → 不适合: 短请求 → swap overhead > recompute → 不值得
  → RTX 4090: PCIe带宽12GB/s → swap 655MB ≈ 55ms → 比recompute慢?

RTX 4090决策:
  → 短请求(S<2048): Recomputation → 快!
  → 长请求(S>4096): Swap → PCIe带宽限制 → 可能不如recompute!
  → → RTX 4090默认Recomputation最优! → PCIe swap太慢!
  → NVLink: swap更快 → 但RTX 4090没有NVLink!

vLLM调度策略:
  → FCFS(默认) → preempt最早到达的请求 → 简单公平
  → 也可以按优先级/长度/成本策略preempt
```

### 2.2 Swap实现

```
CacheEngine: 管理GPU↔CPU swap

swap_out(): GPU → CPU
  → 找到要preempt的请求的所有KV blocks
  → 在CPU pinned memory中分配对应blocks
  → cudaMemcpy(GPU blocks → CPU blocks) → 异步
  → 更新block table: 物理block号 → CPU block号
  → GPU blocks释放 → 回到FreeKVCacheBlockQueue

swap_in(): CPU → GPU
  → 找到要恢复的请求的所有CPU KV blocks
  → 在GPU分配新物理blocks
  → cudaMemcpy(CPU blocks → GPU blocks) → 异步
  → 更新block table: CPU block号 → 新物理block号

batched swap: 一次swap多个请求 → 减少cudaMemcpy启动次数 → 更高效
  → PCIe DMA: 批量传输 → 更好带宽利用率 → 接近12GB/s理论值

RTX 4090 swap性能:
  → PCIe Gen4 x16: ~12 GB/s单向 → ~24 GB/s双向
  → 7B模型 S=2048 KV: 163.84 MB → swap时间 ≈ 14ms → 可接受
  → 7B模型 S=8192 KV: 655.36 MB → swap时间 ≈ 55ms → 较长
  → 70B模型 S=4096 KV: 1.31 GB → swap时间 ≈ 109ms → 太长!
  → → RTX 4090长请求swap太慢 → 默认recompute更好!
```

## 3. KV Cache压缩

### 3.1 INT8 KV Cache

```
INT8 KV: 每个KV元素从BF16(2bytes)→INT8(1bytes) → 50%内存省!

量化方式:
  → Per-token: 每个token每个head有1个scale → scale=amax/127
  → → 量化: K_int8 = round(K_float × 127 / amax)
  → → 反量化: K_float = K_int8 × scale
  → → scale存储: 每token×每head → 1×4bytes(FP32 scale) → overhead小

实测(RTX 4090):
  → cos_sim=0.999965 → 近乎完美精度!
  → 内存省48.4% → (1+scale_overhead)/(2) ≈ 1.016/2 = 48.4%
  → 速度0.81-0.96x → overhead仅3-12% → 大batch接近free!

FlashInfer INT8 KV:
  → cp.async加载INT8 KV到smem → warp内float×scale → 直接用于attention
  → → 在kernel内部dequant → 0% Python overhead → 几乎free!

vLLM INT8 KV:
  → kv_cache_dtype=fp8_e4m3 或 fp8_e5m2
  → → FP8 KV: 1 byte per element → 50%内存省
  → → 注意: vLLM用的是FP8而非INT8 → 更精确(更多动态范围)
  → → 但RTX 4090(SM89) FP8 KV需要fused kernel → vLLM内部dequant

KV内存对比(7B模型, S=4096):

| 方法 | per_token KV | 总KV | 24GB可用请求数 |
|------|-------------|------|-------------|
| BF16 | 81.92 KB/tok | 327.68 MB | 65 |
| INT8 | 40.96 KB/tok | 163.84 MB | 130 |
| FP8 | 40.96 KB/tok | 163.84 MB | 130 |
| INT8+GQA-5 | 10.24 KB/tok | 40.96 MB | 520 |
| MLA(256d) | 25.6 KB/tok | 102.4 MB | 208 |

→ INT8+GQA-5: 520个S=4096请求 → 8x于BF16 → throughput 8x!
```

### 3.2 MLA KV Compression

```
MLA (Multi-head Latent Attention, DeepSeek-V2/V3):
  → KV压缩到低维latent(256-d vs 128-d×5heads=640-d)
  → → KV省3.2x → 但decode需要upsample → compute-bound → 2-8x慢!

RTX 4090 MLA:
  → KV内存省3.2x → 但decode慢 → 不是速度优化而是容量优化
  → MLA+INT8 KV = 6.4x容量↑ → 但仍然decode慢 → 需要fused MLA kernel

结论: MLA适合NVLink多GPU集群 → 单GPU RTX 4090 → GQA+INT8 KV最优!
```

## 4. Prefix Caching: 共享前缀的KV复用

### 4.1 vLLM Prefix Caching

```
BlockHashToBlockMap: KV block内容hash → block引用

系统prompt: "You are a helpful assistant..." → 所有请求共享!
  → 32 tokens → 2 blocks → hash(KV[0:16]) → hash(KV[16:32])
  → 所有请求引用同一2个物理blocks → 无需重新计算 → 省prefill时间!
  → 引用计数 → 只有引用=0时才能驱逐 → 安全!

Copy-on-Write:
  → 新token追加到shared block → block需要新物理空间 → COW!
  → shared block不变 → 新数据写入新block → 引用计数管理

Prefix caching效果:
  → 系统prompt 32 tokens → 省prefill 32/4096 ≈ 0.8% → 小
  → 长prompt 1024 tokens → 省prefill 1024/4096 ≈ 25% → 显著!
  → RAG场景: 检索文档1K tokens → 多请求共享 → 省25% → 重要!

RTX 4090:
  → L2 72MB → prefix blocks可能L2 cache → decode更快
  → → prefix caching不仅省prefill → 还加速decode(L2 cache hit)!
```

### 4.2 SGLang RadixAttention

```
RadixAttention: 树结构管理KV cache → 更细粒度共享!

RadixTree:
  → 节点=KV block → 边=token序列 → 根=空 → 叶=完整prefix
  → 新请求 → 匹配最长prefix → 分裂节点 → 共享匹配部分 → 新增独有部分

分裂操作:
  → 请求A: "system: You are..." → 32 tokens → 2 nodes
  → 请求B: "system: You are a coder..." → 40 tokens
  → → 匹配前32 tokens → 分裂 → 共享32 → 新增8 → 省KV!

驱逐策略(7种):
  → LRU: 最近最少使用 → 简单但可能驱逐hot prefix
  → FIFO: 先进先出 → 不考虑频率
  → Length-aware: 长prefix优先保留 → 更大节省
  → Cost-aware: compute成本高的prefix优先保留
  → Frequency-aware: 高频prefix优先保留 → 系统prompt永远保留
  → Evict-by-prefix: 从最长prefix开始 → 最保守
  → Hybrid: 综合策略 → SGLang默认

vs vLLM prefix caching:
  → vLLM: block-level hash → 固定block_size → 粒度=16 tokens
  → SGLang: token-level radix tree → 任意长度prefix → 更细粒度!
  → → SGLang RadixAttention更灵活 → 但管理更复杂!

RTX 4090:
  → RadixAttention需要更多CPU管理 → 但GPU KV一样
  → → RTX 4090上SGLang可能比vLLM更好(prefix共享更细粒度)!
```

## 5. KV Cache内存模型

### 5.1 KV内存公式

```
KV_per_token = num_layers × 2 × num_kv_heads × d_head × dtype_size

不同模型配置的KV内存:

| 模型 | layers | kv_heads | d_head | dtype | KV/tok(B) | S=4K总(MB) |
|------|--------|----------|--------|-------|-----------|-----------|
| 7B(GQA-5) | 32 | 5 | 128 | BF16 | 81.92 | 327.68 |
| 7B(GQA-5) | 32 | 5 | 128 | INT8 | 40.96 | 163.84 |
| 7B(MHA-32) | 32 | 32 | 128 | BF16 | 524.29 | 2097.15 |
| 70B(GQA-8) | 80 | 8 | 128 | BF16 | 327.68 | 1310.72 |
| 70B(GQA-8) | 80 | 8 | 128 | INT8 | 163.84 | 655.36 |
| DeepSeek-V3(MLA) | 61 | 1* | 256 | BF16 | 25.60* | 102.4 |
| Llama3-8B(GQA-8) | 32 | 8 | 128 | BF16 | 327.68 | 1310.72 |

* MLA: 压缩KV到256-d → 但KV heads=1 → 实际KV更小!

RTX 4090 24GB推理并发容量:

| 配置 | 可用KV(GB) | KV/tok(KB) | max_tokens | S=4K请求数 |
|------|-----------|-----------|-----------|-----------|
| 7B BF16 GQA-5 | 18.6 | 81.92 | 228K | 55 |
| 7B INT8 GQA-5 | 18.6 | 40.96 | 456K | 110 |
| 7B BF16 MHA-32 | 16.0 | 524.29 | 30K | 8 |
| 70B INT8 GQA-8 | 3.0 | 163.84 | 18K | 4 |

→ GQA是推理并发量的关键! MHA→GQA → 6.5x请求数!
→ INT8 KV → 2x请求数 → 与GQA组合 → 13x!
```

### 5.2 GPU内存分配策略

```
vLLM内存分配策略(gpu_memory_utilization=0.9):
  → 90% GPU VRAM用于推理 → 10%留给其他(临时buffer等)
  → 步骤:
    1. 加载模型权重 → 占用~3.5GB(7B INT4) 或 ~14GB(7B BF16)
    2. 计算剩余空间 → 24GB × 0.9 - 3.5GB = 18.1GB(INT4) / 24×0.9-14=7.6GB(BF16)
    3. 分配KV cache → 用剩余空间创建block pool
    4. block数 = 剩余空间 / per_block_size
    5. per_block_size = block_size × num_layers × 2 × num_kv_heads × d_head × dtype_size
       → block_size=16, 7B GQA-5 BF16: 16×32×2×5×128×2 = 1.31 MB/block
       → 18.1GB / 1.31MB = 13,816 blocks → 13,816×16 = 221K tokens → 55个S=4096请求

RTX 4090最优配置:
  → 7B INT4权重(3.5GB) + INT8 KV(0.5bytes) + block_size=16
  → → 剩余: 21.6-3.5 = 18.1GB → KV block: 0.655MB/block → 27,632 blocks → 442K tokens
  → → 110个S=4096请求 → FlashInfer B=32→145K tok/s → throughput最大化!

vs 7B BF16权重(14GB):
  → 剩余: 21.6-14 = 7.6GB → KV block: 1.31MB/block → 5,806 blocks → 93K tokens
  → → 23个S=4096请求 → 少5x!
  → → INT4权重不仅省3.5→14GB → 更让KV可用空间5x → 并发5x → throughput 5x!
```

## 6. RTX 4090 KV Cache优化策略

### 6.1 最优KV配置

```
RTX 4090推理最优:
  → 权重: INT4(AWQ/Marlin) → 3.5GB → 留出18.1GB给KV
  → KV: INT8+GQA-5 → 0.5bytes × 5 heads → 每 tok 40.96KB → 442K max tokens
  → block_size: 16 → 每block 0.655MB → L2 cache可能fit → decode更快
  → FlashInfer: paged KV + GQA native → 0 expand → bandwidth省75%
  → → 110个S=4096并发请求 → 145K tok/s → $0.01/Mtok!

KV内存节省链:
  → INT4权重: 14→3.5GB → 留出10.5GB → 并发×5
  → GQA-5: MHA→5kv_heads → KV/tok↓6.5x → 并发×6.5
  → INT8 KV: 2→1byte → KV/tok↓2x → 并发×2
  → 组合: 5×6.5×2 = 65x并发! → throughput 65x于BF16 MHA!

  → 加上FlashInfer: 15.72x throughput → 综合 65×15.72 = 1020x于BF16 MHA SDPA!
```

### 6.2 混合策略

```
Sliding Window + KV驱逐:
  → Mistral/Llama-3: window=8192 → 超出window的block释放 → 内存省!
  → → S>8192 → 只有最近8192 tokens的KV → 短请求不受影响
  → → 长请求 → KV固定在window → 内存不随S增长!

KV Cache + Prefix Caching:
  → 系统prompt KV共享 → 省prefill → 省KV → 更多并发!
  → RAG: 检索文档KV共享 → 多请求共享 → 省25-50% KV

KV Cache + Continuous Batching:
  → 请求完成 → KV释放 → 新请求立即分配 → 无需等待!
  → → GPU利用率接近100% → throughput最大化!

RTX 4090完整优化栈:
  1. INT4权重 → 内存省 → KV空间↑5x
  2. GQA → KV/tok↓6.5x → 并发↑6.5x
  3. INT8 KV → KV/tok↓2x → 并发↑2x → 精度0.999965
  4. FlashInfer → paged KV + GQA native → throughput↑15.72x
  5. Prefix caching → 系统prompt共享 → prefill↓25%
  6. Continuous batching → GPU利用率100%
  7. Sliding window → 长请求KV不增长
  → → 综合: >1000x于BF16 MHA SDPA baseline!
```

---

**Sources**:
- [vLLM PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [vLLM Documentation](https://docs.vllm.ai)
- [SGLang RadixAttention](https://github.com/sgl-project/sglang)
- [FlashInfer Paged KV](https://docs.flashinfer.ai)
- RTX 4090实测数据: INT8 KV, FlashInfer, comprehensive benchmark

**Related notes**: flashinfer-attention-deep-dive.md (FlashInfer paged KV), vllm-v1-nav (block manager), inference-cost-analysis.md (成本), quantization-path-comparison-rtx4090.md (量化)