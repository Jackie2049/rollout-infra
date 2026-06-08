# LLM Serving Architecture Deep Dive

> 2026-06-08 | 整合20+benchmark数据的LLM serving系统架构全景
> 从硬件到软件到生产部署的完整知识框架

## 1. Serving Stack全景

```
用户请求 → API Gateway → Scheduler → GPU Executor → Model Runner
                                    ↓
                          KV Cache Manager
                          Preemption Policy
                          Token Budget Allocation

硬件层:
  GPU (RTX 4090/H100/A100) → HBM → PCIe/NVLink → CPU
软件层:
  Scheduler → KV Cache → Attn Backend → Quantization → CUDA Graph
```

## 2. 核心矛盾: Compute vs Bandwidth

**LLM推理的根本矛盾**: Prefill需要Compute, Decode需要Bandwidth

| Phase | Resource | Utilization | Bottleneck |
|-------|---------|------------|-----------|
| Prefill (S≥256) | GPU Compute | **73.5% peak** | compute-bound |
| Decode (B=1) | HBM Bandwidth | **0.4% peak** | memory-bound |
| Decode (B=32) | HBM Bandwidth | **12.8% peak** | memory-bound |
| Decode (B=64) | HBM Bandwidth | **25.1% peak** | memory-bound |

**175x TFLOPS差距** → GPU在decode时compute资源闲置87-99.6%

**解决方案**: 不同策略解决不同问题
- Prefill: 不需要优化 → compute-bound → GPU自然高效
- Decode: 需要优化 → memory-bound → 三条路线:
  1. **量化** (减少bytes) → INT4 3.7x加速
  2. **并发** (增加B) → 线性吞吐增长
  3. **PD分离** (专用decode GPU) → 消除ITL stall

## 3. Decode瓶颈分解 (7B, GQA-5, BF16, S=4096)

```
Decode per token HBM traffic:
  Weight reads:   15,610 MB = 95.1%  ← 这是瓶颈!
  KV reads:          326 MB =  3.3%  ← 不是瓶颈(量化省容量不省带宽)
  lm_head reads:     251 MB =  1.6%  ← 不是瓶颈(vocab=32K)
  Activation:        ~100 MB =  0.6%
  Sampling:            2 MB =  0.0%

  Total: ~16,289 MB → 890 GB/s → 18.3ms per token (B=1)
  → 54 tok/s (实测59 → 接近!)

INT4 weight (75%省):
  Weight reads:   3,905 MB → Total ≈ 4,229 MB → 4.76ms → 210 tok/s
  INT4+INT8KV:    3,905 + 163 + 251 = 4,319 MB → 4.86ms → 206 tok/s
  → 3.70x roofline speedup ✓
```

**核心规律**: decode瓶颈=weight reads → 量化是唯一出路!

## 4. Continuous Batching Architecture

### vLLM V1 Scheduler (实测源码分析)

```
schedule():
  token_budget = max_num_scheduled_tokens

  1. RUNNING requests (decode优先):
     for each running request:
       num_new_tokens = min(remaining_tokens, budget)
       allocate KV slots → if insufficient → preempt

  2. WAITING requests (prefill填充剩余budget):
     while budget > 0 and running < max_concurrent:
       pop from waiting queue
       allocate KV for entire sequence (or chunk)
       consume budget

  Preemption (实测: recomputation only!):
    _preempt_request():
      kv_cache_manager.free(request)  ← 释放KV blocks
      request.num_computed_tokens = 0 ← 重置! → 全部recompute
      request.status = PREEMPTED
      waiting.prepend_request(request) ← 重新排队
```

**关键设计决策**:
- **Token budget**: unified → decode和prefill共享同一个budget → decode优先
- **Chunked prefill**: `long_prefill_token_threshold`限制每步prefill tokens → 减少ITL stall
- **Preemption**: recomputation only → `num_computed_tokens = 0` → 全部重算
- **No swap**: vLLM V1 scheduler没有swap选项 → 但实测swap比recompute快4-44x!

**修正**: vLLM应提供swap选项给PCIe GPU用户(pinned memory 24GB/s → swap胜recompute)

### SGLang RadixAttention Scheduler

```
RadixAttention:
  RadixTree → 节点分裂 → 任意粒度prefix共享
  7种驱逐策略 → 更灵活的KV管理
  token-level pool → 更细粒度并发控制
  Merge-based scheduler → 保守调度 → ITL更稳定
```

## 5. KV Cache Management

### 内存预算 (7B, RTX 4090 24GB)

| Config | Weight(GB) | KV/req(MB) | Max Concurrent | Available KV(GB) |
|--------|-----------|-----------|---------------|-----------------|
| BF16 GQA-5 | 15.6 | 326(S=4096) | 2-3 | ~7.4 |
| BF16+INT8KV | 15.6 | 163 | 4-5 | ~7.4 |
| INT4+INT8KV | 3.9 | 163 | 40+ | ~19.1 |
| INT4+INT8KV+GQA8 | 3.9 | 262 | 25+ | ~19.1 |
| StreamingLLM+INT8KV | 15.6 | fixed 168 | **57** | ~7.4 |

**关键**: 量化=并发倍增器 → INT4+INT8KV → 40并发 vs BF16 → 2并发 → 20x!

### Preemption: Recompute vs Swap (实测)

| S(tokens) | Recompute(ms) | Swap(ms) | Ratio | Winner |
|-----------|-------------|---------|-------|--------|
| 16 | 17.1 | 0.38 | 0.02x | **SWAP** |
| 128 | 19.3 | 3.04 | 0.16x | **SWAP** |
| 256 | 27.0 | 6.09 | 0.23x | **SWAP** |
| 2048 | 189.7 | 48.7 | 0.26x | **SWAP** |

**实测结论**: SWAP比recompute快4-44x → vLLM默认recompute在RTX 4090上可能需要修正
- 原因: recompute读取全部weights(15.6GB) → swap只读KV(1.25MB/block)
- PCIe pinned memory 24GB/s → KV传输比weight重算快得多

## 6. Prefill-Decode Interaction

### Mixed Workload ITL Impact (实测)

| S_prefill | ITL Change% | Why |
|-----------|-----------|-----|
| ≤128 | **-27%** | 资源互补(prefill=compute, decode=bandwidth → overlap有效) |
| 512 | +32% | Prefill开始阻塞decode |
| 2048 | **+326%** | 长prefill严重阻塞 → ITL翻3-4倍! |

**Chunked prefill**: vLLM限制每步S≤512 → ITL+32% → 可接受但吞吐下降

### PD Disaggregation

```
Without PD:
  GPU handles prefill AND decode → 资源争抢 → ITL stall

With PD (2 GPU):
  GPU1: Prefill专用 → compute-bound → 73.5% peak → 高效
  GPU2: Decode专用 → memory-bound → 但无prefill干扰 → ITL稳定

  KV Transfer:
    PCIe: 24 GB/s → 3% TTFT overhead → 可接受(修正之前"PCIe PD不可行"!)
    NVLink: 300 GB/s → 0.2% TTFT → 几乎免费 → 生产标配

  PD比例: 1:4-1:8 (1 prefill GPU : 4-8 decode GPU)
  原因: decode吞吐远低于prefill → 需要更多decode GPU平衡
```

## 7. RTX 4090 Serving Decision Tree

```
1. 单GPU推理 (RTX 4090):
   模型选择:
     7B INT4 AWQ + INT8 KV + FlashInfer(GQA-8) → 最优
     → 4,791 tok/s(B=118), ITL=4.5ms
   Context策略:
     S≤4K → 默认 → 2,468 tok/s(B=32)
     StreamingLLM → 无限对话 → B=57 → 2,311 tok/s
     NTK 4x → S=16K → 572 tok/s → 可用

2. 2GPU训练 (RTX 4090):
     25M以下 → FSDP1 2GPU → 1.12x (勉强可行)
     >25M → FSDP1/2 单GPU → 或用H100

3. 8GPU训练 (RTX 4090 PCIe):
     完全不可行! FSDP 8GPU = 0.50x → 比单GPU慢2x!

4. PD分离 (2×RTX 4090):
     PCIe可行但成本2x → TTFT+3%, ITL消除stall
     NVLink(H100)→生产标配

5. Preemption策略:
     Swap(pinned) → 比recompute快4-44x → 推荐
     StreamingLLM → 固定KV → 零preemption → 最优

6. 量化组合:
     INT4 AWQ(Marlin) + INT8 KV + FlashInfer → B=118 → 4,791 tok/s
     FP8 KV(cos_sim=0.999996) → 50%省+更高精度 → 推荐
     INT8 W8A8(SmoothQuant) → 2x计算吞吐 → 训练用

7. GPU利用率:
     Decode B=1 → 0.4% peak → 99.6% idle → 量化=唯一出路
     Prefill S≥256 → 73.5% peak → GPU高效 → 不需优化
```

## 8. 生产Serving Checklist

```
配置检查:
  ✓ block_size=16 → PagedAttention → 3.9x faster分配
  ✓ pin_memory() → 24GB/s vs pageable 6-13GB/s → 2x faster
  ✓ non_blocking=True → async transfer → 1.6x faster
  ✓ CUDA Graph → 消除jitter → 不加速(7B仅1.05x) → 稳定ITL
  ✓ FlashInfer → GQA-8 → 15.72x attention-only → 1.06-3.20x整体
  ✓ INT8 KV → 50%省 → FP8更准确(cos_sim=0.999996)
  ✓ AWQ INT4 → 75%省 → fused kernel必需(Marlin/TE)
  ✓ StreamingLLM → sink(4)+window → 固定KV → 无限对话

避免:
  ✗ Swap+compute overlap → -47.7% → RTX 4090多stream负优化
  ✗ torch.compile推理 → B≥4反而慢(0.80-0.97x)
  ✗ Triton GEMM → cuBLAS更快1.5x → Triton仅用于reduction
  ✗ Multi-GPU FSDP(>2 GPU) → 0.46-0.67x → PCIe灾难
  ✗ Ring Attention → RTX 4090慢7-67x → NVLink技术
  ✗ Python dequant → 20x慢 → fused kernel必需
```

## 9. Benchmark Cross-Validation Summary

| Finding | Verified By | Status |
|---------|-----------|--------|
| Decode memory-bound | GEMM shape, decode breakdown, PD separation | ✓ 3x验证 |
| Weight reads 95.1% | Decode breakdown, roofline, swap vs recompute | ✓ 3x验证 |
| INT4 3.7x roofline | Roofline, inference benchmark, PD separation | ✓ 3x验证 |
| lm_head 1-2% (vocab=32K) | Decode breakdown, roofline | ✓ 修正之前20-30% |
| SWAP > recompute | KV offloading benchmark | ✓ 新发现! |
| PCIe PD viable | PD separation benchmark | ✓ 修正之前不可行 |
| Multi-stream negative | Stream concurrency, KV offloading overlap | ✓ 2x验证 |
| FSDP PCIe disaster | FSDP scaling, NCCL benchmark | ✓ 2x验证 |