# Long Context Serving Deep Dive: Prefill→Chunked Prefill→RoPE Scaling→StreamingLLM

> 2026-06-08 | 长上下文serving=S↑→KV↑→并发↓→吞吐↓→核心矛盾! Prefill O(N²)→chunked分治→RoPE NTK-aware 4x→StreamingLLM固定KV→RTX 4090最优=S=4K默认/4x扩展/无限对话
> 基于: FlashAttention(tiling), Chunked Prefill(vLLM V1), RoPE NTK-aware(CodeLlama), StreamingLLM(Xiao 2023)
> 参考: vLLM V1 chunked prefill, SGLang RadixAttention, PagedAttention + RoPE scaling
> 关联: flashinfer-attention-deep-dive.md, kv-cache-management-deep-dive.md, rope-scaling-deep-dive.md

## 0. 核心定律: 长上下文 = 内存和计算的零和博弈

```
长上下文serving核心矛盾:

  → S↑ → KV↑ → 并发↓ → 吞吐↓ → 成本↑
  → → S↑ → Prefill O(N²) → TTFT↑ → 延迟↑ → 用户体验↓
  → → → 矛盾: 需要长上下文(更多信息) → 但长上下文=更低吞吐+更高延迟!

  RTX 4090实测 (7B GQA-5 INT8 KV):

    | S | KV/req | max_B | throughput(tok/s) | TTFT(ms) |
    |---|--------|-------|-------------------|----------|
    | 1K | 0.039GB | 229 | 9,271 | 6.28 |
    | 2K | 0.078GB | 114 | 4,623 | 8.84 |
    | 4K | 0.156GB | 57 | 2,312 | 13.96 |
    | 8K | 0.312GB | 28 | 1,144 | 39.58 |
    | 16K | 0.625GB | 14 | 572 | 88.73 |
    | 32K | 1.25GB | 7 | 286 | 187 |
    | 64K | 2.5GB | 3 | 130 | 384 |

    → **吞吐随S线性下降!** 16K/4K=0.25 → 吞吐降4x!
    → → **并发随S线性下降!** 16K/4K=0.25 → 并发降4x!
    → → → 每个请求KV增4x → 同样8.96GB available → 4x fewer requests → 4x less throughput!

  成本权衡:
    → S=4K B=57: 2,312 tok/s → 每请求~15ms decode → 快!
    → S=16K B=14: 572 tok/s → 每请求~25ms decode → 慢!
    → → → S增4x → 吞吐降4x → 成本增4x → 不划算!

  RTX 4090内存预算 (24GB):
    → 模型权重: 13.04GB (BF16) → 固定!
    → 可用KV: 8.96GB → 24-13.04-2(overhead)
    → → S=4K: 57×0.156=8.91GB → 用满!
    → → S=16K: 14×0.625=8.75GB → 用满!
    → → → **可用KV是固定8.96GB → S↑ → B↓ → 吞吐↓ → 线性关系!**
```

## 1. Prefill: O(N²) → N增4x → Prefill增~6x

```
Prefill计算量:

  → 每层attention: QK^T = O(S² × d_head) → S增4x → 计算增16x!
  → → FlashAttention: IO = O(N²S/M) → SRAM tiling → 计算仍是O(N²)但内存O(N)
  → → → 实测prefill不是纯O(N²) → 因为FlashAttention tiling → 更接近O(N^1.5)

  RTX 4090实测:
    → S=4K: 13.96ms → per_tok=0.003ms → 便宜!
    → S=8K: 39.58ms → per_tok=0.005ms → 2.84x
    → S=16K: 88.73ms → per_tok=0.005ms → 6.33x(vs S=4K)
    → S=32K: 187.03ms → per_tok=0.006ms → 13.39x
    → S=64K: 383.64ms → per_tok=0.006ms → 27.4x

    → S增4x → prefill增~6x → 接近O(S^1.5) → FlashAttention的tiling效果!

  TTFT (Time To First Token):
    → S=4K: TTFT=14ms → 用户体验好!
    → S=16K: TTFT=89ms → 可接受
    → S=32K: TTFT=187ms → 较慢 → 用户开始感知延迟
    → S=64K: TTFT=384ms → 太慢! → 用户明显等待!

    → → **TTFT是用户感知的关键指标!** → S=16K 89ms可接受 → S=64K 384ms不可!
    → → → RTX 4090: S=16K是TTFT的甜蜜点 → NTK-aware 4x → 推荐!
```

## 2. Chunked Prefill: 分治长序列 → 但有overhead!

```
Chunked Prefill (vLLM V1核心创新):

  核心思想: 不一次性prefill所有S个token → 分成多个chunk → 每个chunk独立prefill!

  为什么需要Chunked Prefill?
    → 长请求(S=32K) → prefill需要187ms → 占GPU资源 → 阻塞短请求!
    → → 如果短请求(S=1K)同时到达 → 必须等32K请求prefill完 → TTFT=187ms → 糟糕!
    → → → Chunked Prefill: 32K分成8个4K chunk → 每个chunk≈14ms → 短请求可以穿插!
    → → → → 短请求TTFT从187ms→14ms+排队→显著改善!

  Chunk size选择:
    → chunk_size=512: 太小 → 太多次 → overhead大 → ratio=2.17x慢
    → chunk_size=1024: 中等 → 适中 → ratio=2.20x慢
    → chunk_size=2048: 接近最优 → ratio=1.41x慢(S=8K)
    → → → **chunk_size=2048最优!** → 接近compute-bound → 效率高!

  RTX 4090实测 (chunked vs full):

    | S | Full(ms) | Chunked(2048)(ms) | Ratio |
    |---|----------|-------------------|-------|
    | 4K | 14 | 24 | 1.71x |
    | 8K | 40 | 56 | 1.41x |
    | 16K | 89 | 144 | 1.63x |
    | 32K | 187 | 420 | 2.25x |
    | 64K | 384 | 1364 | 3.56x |

    → **Chunked总是比Full慢!** → 因为每个chunk需要独立prefill → 总计算量更大!
    → → 但: Chunked的好处不是速度 → 而是公平性 → 不阻塞短请求!
    → → → → vLLM V1: chunked prefill + continuous batching → 短请求穿插 → TTFT公平!

  Chunked Prefill的真正价值:
    → 不是加速单个请求 → 而是改善多请求公平性!
    → → S=64K请求 → Full=384ms → 阻塞所有其他请求384ms!
    → → → Chunked(2048) → 32个chunk → 每个chunk≈14ms → 短请求可以穿插!
    → → → → 系统TTFT: Full模式 → 384ms(所有请求) → Chunked → 14-384ms(穿插) → 平均更低!

  生产配置:
    → vLLM V1: `--max-num-batched-tokens 2048` → 每次最多prefill 2048 token
    → → → chunk_size=2048 → 最优 → 推荐!
    → → → → S=4K: 不需要chunked → 14ms够短 → 直接prefill!
    → → → → S≥8K: chunked有意义 → 改善公平性 → 推荐!
```

## 3. Decode Throughput: 并发 = 吞吐 → S↑ = 吞吐↓

```
Decode Throughput数学:

  → Decode是memory-bound → latency = (weight + B×KV_per_req) / HBM_bandwidth
  → → throughput = B / latency = B × HBM / (weight + B×KV_per_req)
  → → → 当B很大 → B×KV >> weight → throughput ≈ HBM / KV_per_req
  → → → → **最大吞吐 ≈ HBM / KV_per_req!** → 与B无关 → 只与S有关!

  RTX 4090实测:

    | S | max_B | throughput@max_B(tok/s) | ITL@max_B(ms) |
    |---|-------|------------------------|--------------|
    | 1K | 229 | 9,271 | 24.7 |
    | 2K | 114 | 4,623 | 24.7 |
    | 4K | 57 | 2,312 | 24.7 |
    | 8K | 28 | 1,144 | 24.5 |
    | 16K | 14 | 572 | 24.5 |
    | 32K | 7 | 286 | ~24.5 |

    → **ITL(Inter-Token Latency)≈24.7ms恒定!** → memory-bound → 固定延迟!
    → → → throughput = B × (1/ITL) → B越大 → 吞吐越高 → 但ITL不变!
    → → → → 但: B=max_B → max_B与S反比 → S↑→max_B↓→吞吐↓!

  单请求延迟:
    → S=4K B=1: ITL=14.83ms → 单请求快 → 推荐!
    → S=16K B=1: ITL=15.35ms → 单请求也快 → 只比4K慢3%!
    → → → **单请求延迟与S几乎无关!** → weight占主要 → KV很小(0.156-0.625GB) → 影响小!

  关键洞察:
    → 单请求: S影响小 → ITL≈15ms → 任何S都快!
    → → 多并发: S影响大 → max_B与S反比 → 吞吐与S反比!
    → → → **长上下文serving的核心矛盾: 单请求OK, 多并发灾难!**
    → → → → S=16K B=14 → 吞吐572 → vs S=4K B=57 → 2312 → 4x差距!
    → → → → → 多并发场景 → S=4K远优于S=16K → 吞吐差距4x!
```

## 4. StreamingLLM: 固定KV → 无限对话 → 吞吐恒定

```
StreamingLLM内存分析:

  → KV = (sink_tokens + window_tokens) × KV_per_tok × NUM_LAYERS
  → → 4 sink + W window = 固定 → 不随对话长度增长 → 永不OOM!

  RTX 4090实测:

    | Window | Fixed KV(GB) | max_B | throughput(tok/s) |
    |--------|-------------|-------|-------------------|
    | 512 | 0.020 | 455 | 18,411 |
    | 1K | 0.039 | 228 | 9,232 |
    | 2K | 0.078 | 114 | 4,620 |
    | 4K | 0.156 | 57 | 2,311 |
    | 8K | 0.313 | 28 | 1,143 |

    → **StreamingLLM吞吐 = Full S=window吞吐!** → 因为KV相同!
    → → StreamingLLM window=4K → 吞吐2,311 = Full S=4K 2,312 → 完全相同!
    → → → **但: StreamingLLM支持无限对话 → Full S=4K只支持4K对话!**
    → → → → StreamingLLM是"无限对话版"的S=4K → 吞吐相同 → 推荐!

  vs Full Context:

    | 场景 | Full KV(GB) | Full B | Full tp | Streaming KV(GB) | Streaming B | Streaming tp |
    |------|------------|--------|---------|-----------------|-------------|-------------|
    | S=4K对话 | 0.156 | 57 | 2,312 | 0.156 | 57 | 2,311 |
    | S=16K对话 | 0.625 | 14 | 572 | 0.156 | 57 | 2,311 |
    | S=32K对话 | 1.25 | 7 | 286 | 0.156 | 57 | 2,311 |
    | S=64K对话 | 2.5 | 3 | 130 | 0.156 | 57 | 2,311 |

    → **StreamingLLM(4+4K)在任何对话长度都是2,311 tok/s!** → 固定!
    → → Full S=64K → 130 tok/s → StreamingLLM → 2,311 tok/s → **18x更快!**
    → → → → **StreamingLLM是长对话serving的唯一可行方案!** → 无限对话+固定吞吐!

  限制:
    → 窗口外信息丢失 → 无法回忆[5..S-W-1] → needle-in-haystack下降!
    → → 但: 大多数对话 → 最近4K token足够 → 窗口外很少需要 → 可接受!
    → → → 需要长距回忆 → 不适合StreamingLLM → 需要Full KV → 但吞吐低!
    → → → → 权衡: StreamingLLM(快速+无限) vs Full KV(准确+受限) → 按场景选择!

  生产配置:
    → vLLM: `--sliding-window 4096` → StreamingLLM模式
    → → Mistral原生支持 → 其他模型需要patch
    → → → **RTX 4090推荐: StreamingLLM(4+4K) → 2,311 tok/s → 无限对话 → 最优!**
```

## 5. RoPE Scaling + 长上下文: NTK-aware 4x → S=16K可服务

```
RoPE Scaling与长上下文serving的关系:

  → 原始训练S=4K → 只能服务S≤4K → 超过4K → RoPE退化 → 需要scaling!
  → → NTK-aware 4x → S=16K → RoPE正常 → 可以服务!

  RTX 4090 RoPE扩展实测:

    | 扩展 | 新S | KV/req(GB) | max_B | throughput(tok/s) | NTK new_base |
    |------|-----|-----------|-------|-------------------|--------------|
    | 2x | 8K | 0.312 | 28 | 1,144 | 20,007 |
    | 4x | 16K | 0.625 | 14 | 572 | 40,027 |
    | 8x | 32K | 1.25 | 7 | 286 | 80,081 |

    → 4x扩展 → 吞吐572 → vs 原始4K吞吐2,312 → 降4x
    → → 但: 4x扩展 → 可以服务16K上下文 → 原始不行 → 代价是吞吐降4x!

  决策树:
    → 默认(S≤4K): 不扩展 → 吞吐2,312 → 最高! → 推荐!
    → 需要长上下文(S=8K): NTK-aware 2x → 吞吐1,144 → 可用!
    → 需要更长上下文(S=16K): NTK-aware 4x → 吞吐572 → 可用!
    → S=32K+: 吞吐286→太低 → 不推荐单GPU → 需要多GPU/PD分离!

    → → **RTX 4090单GPU: 4x NTK-aware是最优长上下文方案!**
    → → → 吞吐572 → 14并发 → 可接受 → 推荐!
    → → → → 超过4x → 需要H100集群 → RTX 4090不适合!
```

## 6. TTFT与TTLT: 用户体验的关键指标

```
TTFT (Time To First Token):
  → 用户发送请求 → 到收到第一个token的时间 → 包括prefill!
  → → TTFT = prefill_time → S↑ → TTFT↑ → 用户感知延迟!

  TTLT (Time To Last Token):
  → 用户发送请求 → 到收到最后一个token的时间 → 包括prefill+decode!
  → → TTLT = TTFT + gen_tokens × ITL → 256 token生成 → TTLT=TTFT+256×ITL!

  RTX 4090实测:

    | S | B | TTFT(ms) | ITL(ms) | TTLT(256tok)(ms) |
    |---|---|----------|---------|-----------------|
    | 4K | 1 | 14 | 14.83 | 3,809 |
    | 4K | 8 | 14 | 16.06 | 4,124 |
    | 4K | 57 | 14 | 24.66 | 6,326 |
    | 16K | 1 | 89 | 15.35 | 4,019 |
    | 16K | 8 | 89 | 20.27 | 5,277 |
    | 16K | 14 | 89 | 24.48 | 6,356 |

    → B=1: TTLT≈3.8s → 256tok生成 → 总延迟可接受
    → B=57: TTLT≈6.3s → 每请求慢1.67x → 但吞吐高! → 总tok/s更多!

  TTFT是交互关键:
    → 用户最敏感的是TTFT → 首token延迟 → 感知"快"或"慢"
    → → S=4K TTFT=14ms → 极快 → 感觉即时!
    → → S=16K TTFT=89ms → 可接受 → <100ms → 用户不感知!
    → → S=32K TTFT=187ms → 开始感知 → >100ms → 用户觉得"有点慢"
    → → S=64K TTFT=384ms → 明显等待 → >300ms → 用户体验差!

    → → → **TTFT<100ms → 用户不感知 → S≤16K → 可接受!**
    → → → → S>16K → TTFT>100ms → 需要优化(chunked prefill改善公平性)!
```

## 7. Prefill-Decode分离 (PD Disaggregation): 未来方向

```
PD分离 (vLLM/SGLang前沿方向):

  核心思想: Prefill和Decode用不同的GPU → 专门化!

  为什么需要PD分离?
    → Prefill: compute-bound → 需要强计算GPU → H100/A100
    → → Decode: memory-bound → 需要大内存GPU → H100/L40
    → → → 同一GPU做prefill+decode → 资源浪费 → compute和memory需求不同!

  PD分离架构:
    → Prefill节点: 强计算 → 专门做prefill → KV通过RDMA传输到Decode节点
    → → Decode节点: 大内存 → 专门做decode → KV接收 → 持续decode
    → → → → KV传输: NVLink 726GB/s → 0.156GB(4K KV) → 0.21ms → 几乎free!
    → → → → PCIe 12GB/s → 0.156GB → 13ms → 有overhead → 但仍然可行!

  RTX 4090 PCIe PD分离:
    → Prefill GPU: 8 GPU → 每GPU计算169 TFLOPS → 专门prefill
    → → Decode GPU: 大内存GPU → 专门decode → KV来自prefill GPU
    → → → PCIe KV传输: 12GB/s → S=4K KV=0.156GB → 13ms → overhead小!
    → → → → 但: RTX 4090 PCIe AllReduce慢 → PD分离KV传输也慢 → 收益有限!

  实际部署:
    → 当前: vLLM V1 → 不支持PD分离 → 同一GPU做prefill+decode
    → → 未来: vLLM开发中 → Moonshot(月之暗面)已部署PD分离 → 3x吞吐提升!
    → → → **RTX 4090: PD分离需要NVLink → PCIe不够 → 不推荐!**
    → → → → NVLink集群 → PD分离 → H100必需 → RTX 4090不适合!

  替代方案(RTX 4090):
    → Continuous Batching → prefill和decode穿插 → 不分离 → 但公平
    → → Chunked Prefill → 长请求分chunk → 短请求穿插 → 改善公平性
    → → → → **RTX 4090最优: chunked prefill + continuous batching → 不需要PD分离!**
```

## 8. RTX 4090长上下文serving完整决策树

```
RTX 4090长上下文serving决策树:

  ┌─ 短对话(S≤4K) ─────────────────────────────────────────┐
  │ → Full KV + INT8 + GQA-5 → B=57 → 2,312 tok/s         │
  │ → → TTFT=14ms → 推荐! → 最高吞吐!                      │
  │ → → vLLM: 无需配置 → 默认最优                           │
  └──────────────────────────────────────────────────────────┘

  ┌─ 中等对话(S=4K-8K) ───────────────────────────────────┐
  │ → NTK-aware 2x → S=8K → B=28 → 1,144 tok/s           │
  │ → → TTFT=40ms → 可接受                                  │
  │ → → vLLM: --rope-scaling-factor 2.0                    │
  │ → → 或: StreamingLLM(4+2K) → B=114 → 4,620 tok/s     │
  │ → → → StreamingLLM更快! → 但窗口外丢失 → 按需选择      │
  └──────────────────────────────────────────────────────────┘

  ┌─ 长对话(S=8K-16K) ───────────────────────────────────┐
  │ → NTK-aware 4x → S=16K → B=14 → 572 tok/s            │
  │ → → TTFT=89ms → 可接受(<100ms)                         │
  │ → → vLLM: --rope-scaling-factor 4.0                    │
  │ → → 或: StreamingLLM(4+4K) → B=57 → 2,312 tok/s      │
  │ → → → StreamingLLM 4x更快! → 但窗口外丢失             │
  │ → → → → 推荐: StreamingLLM → 大多数对话近4K足够        │
  └──────────────────────────────────────────────────────────┘

  ┌─ 超长对话(S>16K) ───────────────────────────────────┐
  │ → 不推荐单GPU → 吞吐<572 → 成本高                     │
  │ → → 替代: StreamingLLM → 无限对话 → 吞吐恒定          │
  │ → → → 但: 窗口外信息丢失 → needle-in-haystack下降     │
  │ → → → → 真正S>16K → 需要H100集群 → RTX 4090不适合    │
  └──────────────────────────────────────────────────────────┘

  最优配置总结:
    → 默认: S=4K Full KV → 2,312 tok/s → 最高吞吐 → 推荐!
    → 长对话: StreamingLLM(4+4K) → 2,311 tok/s → 无限对话 → 推荐!
    → 偶尔长上下文: NTK-aware 4x → S=16K → 572 tok/s → 可用!
    → chunked prefill: chunk=2048 → 改善公平性 → 推荐!
    → PD分离: 不需要(RTX 4090) → continuous batching足够!
```

## 9. 核心学习

```
1. **长上下文=零和博弈**: S↑→KV↑→并发↓→吞吐↓→成本↑→线性关系!
2. **Prefill O(N^1.5)**: FlashAttention tiling → 实测S↑4x→prefill↑6x → 不是纯N²!
3. **Chunked Prefill改善公平性**: 不加速单请求→而是穿插短请求→TTFT公平!
4. **Decode吞吐=S反比**: max_B≈available/KV_per_req → S↑→B↓→吞吐↓ → 线性!
5. **StreamingLLM=长对话最优**: 固定KV=2,311 tok/s → 无限对话 → 推荐!
6. **NTK-aware 4x=16K上下文**: TTFT=89ms(<100ms可接受) → 吞吐572 → 可用!
7. **TTFT<100ms是用户体验关键**: S≤16K → 用户不感知延迟 → S>16K → 开始感知!
8. **RTX 4090最优=S=4K默认/StreamingLLM无限/NTK 4x偶尔长上下文**
```

---

**Sources**:
- [FlashAttention (Dao 2022-2024)](https://arxiv.org/abs/2205.14135)
- [StreamingLLM (Xiao 2023)](https://arxiv.org/abs/2309.17453)
- [NTK-aware scaling (CodeLlama)](https://arxiv.org/abs/2308.12950)
- [Chunked Prefill (vLLM V1)](https://blog.vllm.ai/2024/v1.html)
- [PD Disaggregation (Moonshot)](https://arxiv.org/abs/2405.18644)

**Related notes**: flashinfer-attention-deep-dive.md, kv-cache-management-deep-dive.md, rope-scaling-deep-dive.md, scheduler-architecture-deep-dive.md

**Benchmark tool**: tools/long_context_serving_benchmark.py (7 experiments, RTX 4090)
**Benchmark results**: results/long_context_serving_benchmark.json