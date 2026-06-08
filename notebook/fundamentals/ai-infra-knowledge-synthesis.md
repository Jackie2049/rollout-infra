# AI Infra Knowledge Synthesis: 关键洞察网络

> 2026-06-08 | 跨领域知识综合: 连接量化、并行、通信、推理的底层规律
> 基于: 160+笔记 + 447 commits + RTX 4090实测数据

## 0. 核心洞察: 一切都围绕带宽与计算比

```
AI Infra的底层规律:
  → 计算瓶颈 vs 内存瓶颈 → 由FLOPS/byte比值决定
  → Ridge point ≈ 182 FLOPS/byte(RTX 4090 FP16)
  → 如果每个数据byte需要<182 FLOPS → memory-bound
  → 如果每个数据byte需要>182 FLOPS → compute-bound

这个单一规律解释了几乎所有AI Infra现象:
  1. Decode慢 → 每byte只5 FLOPS → memory-bound → 量化省带宽=省延迟
  2. Prefill快 → 每byte约182 FLOPS → 接近ridge → compute+memory平衡
  3. 训练backward快 → 每byte>200 FLOPS → compute-bound → 通信比重小
  4. EP All-to-All → 数据量∝GPU数 → 带宽瓶颈 → NVLink
  5. FlashInfer比SDPA快 → 省KV带宽(GQA native) → 带宽利用率高!
```

## 1. 量化 → 带宽 → 推理 → 成本 链条

```
量化 → 降低bytes_per_token → 降低memory-bound延迟 → 提高throughput → 降低成本

INT4: bytes_per_token ↓75% → 权重带宽 ↓75% → 内存省 → 更大batch → throughput↑
INT8 KV: bytes_per_token ↓50%(KV部分) → KV带宽↓50% → 内存省 → 更长context
FP8(TE): bytes_per_token ↓50% → GEMM带宽↓50% → cuBLASLt fused → 计算也加速!

但量化路径选择决定效果:
  → Python dequant: 加慢(多一步CPU操作) → 0.4-0.67x ❌
  → Fused kernel(TE/cuBLASLt): dequant在GEMM内部 → 1.48-1.59x ✅
  → INT4 weight-only: 只省权重带宽 → decode仍memory-bound → 0.87-1.08x → "免费内存省"

核心: 量化要省带宽 AND 避免额外计算开销 → fused kernel是关键!
```

## 2. GQA → 带宽 → FlashInfer → 成本 链条

```
GQA → KV heads ↓ → KV带宽↓ → FlashInfer native处理 → 带宽利用率↑ → throughput↑ → cost↓

链条展开:
  GQA-5 vs MHA-20: KV带宽 ↓75%
    → SDPA: expand KV → KV带宽 ×4 → 回到MHA带宽 → 没有省!
    → FlashInfer: native GQA → KV带宽 真实 ↓75% → 带宽利用率87%!

  FlashInfer constant time decode: ~0.22ms不随S/B增长
    → 原因: GQA native → KV读取量不随batch增长(每请求独立KV)
    → vs SDPA: expand KV → KV读取量 ∝ B → linear增长 → throughput plateaued

  经济影响: B=32 → FlashInfer 145K tok/s vs SDPA 9K → 15.72x → cost从$0.55→$0.01/Mtok!

核心: GQA省带宽, 但SDPA浪费了GQA的优势 → FlashInfer释放了GQA的全部潜力!
```

## 3. 通信 → 带宽 → 并行 → 扩展 链条

```
分布式并行 → 通信量 → 带宽需求 → 网络类型决定可行性

通信量 vs 带宽:
  → NVLink 726 GB/s(SM100) → 11.5%通信占比 → 可行!
  → PCIe ~12 GB/s → 39-83%通信占比 → 不可行(TP/PP)!
  → RDMA 90 GB/s(CX7) → EP可行!

选择策略:
  → NVLink可用 → TP(最小通信量) + FSDP2(ZeRO-3式分片)
  → PCIe only → 只FSDP2/ZeRO(通信量∝参数量而非∝batch×参数)
  → NVLink+RDMA → TP+PP+DP(Megatron) + EP(DeepEP)

RTX 4090特例:
  → 无NVLink → TP/PP不可行 → 只FSDP2
  → 无RDMA → EP不可行 → 单GPU推理
  → 但HBM 890 GB/s → 单GPU性能好 → 单GPU+量化+FlashInfer最优!
```

## 4. DeepEP → 通信优化 → MoE → 扩展 链条

```
DeepEP核心: 4-6 SMs做All-to-All → 98 SMs做GEMM → 完美重叠!

V2优化链:
  1. NCCL Gin → GPU-initiated → 对称内存 → 远程直写 → 无CPU介入
  2. Warp分工 → Notify(统计)+Dispatch(发送) → 并行处理
  3. SM理论计算 → 4-6 SMs足够(HBM读写跟上) → 不是通信瓶颈而是HBM瓶颈
  4. FP8 dispatch → 数据量↓50% → RDMA/NVLink带宽更充分利用
  5. EventOverlap → async_with_compute → comm_stream独立 → 计算不阻塞

但RTX 4090完全无法用:
  → SM89不支持TMA/mbarrier/elect → 需SM90(Hopper/Blackwell)
  → 无NVLink → intra-node EP不可行
  → 无RDMA → inter-node EP不可行
  → PCIe EP → 带宽太低 → latency>10ms → 完全不可行

核心: DeepEP是NVLink+RDMA集群的专用优化 → PCIe GPU只能用TP或单GPU!
```

## 5. 训练 vs 推理: 不同的瓶颈, 不同的策略

```
训练瓶颈:
  → Compute-bound(backward每byte>200 FLOPS)
  → 通信占比小(NVLink下11.5%)
  → 策略: 多GPU并行(TP+FSDP) + FP8加速计算(1.48-1.59x)

推理瓶颈:
  → Memory-bound(decode每byte≈5 FLOPS)
  → 通信占比: 单GPU内无通信, 多GPU有KV传输
  → 策略: 量化省带宽(INT4+INT8KV) + FlashInfer提高利用率(GQA native)

为什么不同?
  → 训练: backward计算量大 → compute-bound → 加速方向=加速计算
  → 推理: decode只读权重+KV → memory-bound → 加速方向=省带宽

统一公式:
  → cost ∝ GPU_price × bytes_per_token / (FLOPS_per_byte × peak_FLOPS)
  → compute-bound: cost ∝ GPU_price × FLOPS_needed / peak_FLOPS
  → memory-bound: cost ∝ GPU_price × bytes_needed / peak_bandwidth
```

## 6. 关键技术依赖图

```
                    ┌──────────────────┐
                    │  Bandwidth Law    │
                    │  (FLOPS/byte)     │
                    └──────────┬───────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────┴──────┐  ┌─────┴──────┐  ┌──────┴───────┐
    │ Compute-Bound  │  │  Balanced  │  │ Memory-Bound │
    │ (Training)     │  │ (Prefill)  │  │ (Decode)     │
    └─────────┬──────┘  └─────┬──────┘  └──────┬───────┘
              │                │                │
    ┌─────────┴──────┐         │    ┌───────────┴──────┐
    │ 加速计算方向    │         │    │ 省带宽方向        │
    │ FP8 TE(1.59x) │         │    │ INT4(75%省)       │
    │ TP(AllReduce) │         │    │ INT8KV(50%省)     │
    │ torch.compile │         │    │ GQA(75%省)        │
    │              │         │    │ FlashInfer(native) │
    └──────────────┘         │    │ DeepEP FP8 dispatch│
                             │    └───────────────────┘
                             │
                    ┌────────┴────────┐
                    │  通信扩展方向    │
                    │ NVLink(TP/PP)   │
                    │ RDMA(EP)        │
                    │ FSDP2(ZeRO)     │
                    └─────────────────┘
```

## 7. 跨领域洞察汇总

| 洞察 | 领域A | 领域B | 连接 |
|------|-------|-------|------|
| bandwidth是瓶颈 | 量化(INT4省带宽) | 推理(decode memory-bound) | 量化省带宽=省延迟=省成本 |
| GQA native释放GQA潜力 | GQA(KV省75%) | FlashInfer(native处理) | SDPA expand浪费GQA优势 |
| fused kernel避免额外开销 | FP8(cuBLASLt内部dequant) | TE(1.48-1.59x加速) | Python dequant慢→fused快 |
| SM分配可以优化 | DeepEP(4-6 SMs通信) | DualPipe(通信-计算重叠) | 少SM通信+多SM计算→完美重叠 |
| 网络类型决定并行策略 | PCIe(39-83%通信) | NVLink(11.5%通信) | RTX 4090→FSDP2而非TP |
| cost∝bytes/bandwidth | 量化(INT4省bytes) | 推理(bandwidth瓶颈) | 省bytes=省bandwidth=省cost |

---

**Related notes**: 所有160+笔记都是这个知识网络的节点!