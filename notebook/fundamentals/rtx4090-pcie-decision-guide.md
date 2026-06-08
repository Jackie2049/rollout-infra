# RTX 4090 PCIe 统一决策指南

> 2026-06-07/08 | 综合30+benchmark实验的最终决策指南 (含20+推理benchmark更新)
> **核心结论: RTX 4090 PCIe是消费级GPU, 无NVLink → 多GPU并行策略受限 → 大部分场景单GPU最优**

## 决策树

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ RTX 4090 PCIe 决策树 (综合10+benchmark实验):                               │
│                                                                      │
│ === 训练 ===                                                         │
│                                                                      │
│ <10M模型                                                             │
│   → DDP(最快) + 无checkpointing(内存够用)                           │
│   → DDP效率: 2GPU 94%/4GPU 86%/8GPU 84% (76K模型)                │
│   → GA=4: eval 98% > DDP eval 84% (2.28M模型)                     │
│                                                                      │
│ 10M-100M模型                                                          │
│   → FSDP1(2-4GPU) + 无checkpointing(最省内存)                       │
│   → FSDP1比DDP快2x! (25M 2GPU: 125%效率 vs DDP 65%)               │
│   → FSDP1原因: ReduceScatter 4x更小 + prefetching重叠              │
│   → Scaling警告: FSDP1 2GPU=125% → 4GPU=70% → 8GPU=61%           │
│   → 4GPU以上FSDP1优势消失 → PCIe瓶颈                               │
│                                                                      │
│ >100M模型                                                             │
│   → FSDP2(ZeRO-3) + 更多GPU分片 + 无checkpointing                  │
│   → FSDP2内存最优: 0.329 vs FSDP1 0.478 vs DDP 0.639GB           │
│   → FSDP2比FSDP1慢2x(composable API开销) → 但内存省31%            │
│   → FSDP2+compile: 14%改善 → 但仍不如FSDP1                       │
│                                                                      │
│ 7B+模型                                                              │
│   → FSDP2(ZeRO-3) DP=8 + 无checkpointing(内存决定性)                │
│   → 单GPU24GB不够 → 需DP=8才能fit                                   │
│   → checkpointing仅在7B+大batch+128K context时有价值               │
│                                                                      │
│ === 序列并行(Ring Attention/CP) ===                                  │
│                                                                      │
│   → **RTX 4090 PCIe: 不可行!** 比单GPU慢7-67x                        │
│   → 原因: PCIe带宽仅5-6 GB/s, 无法overlap通信与计算              │
│   → 通信占比: P=2 25-55% / P=4 30-80% / P=8 20-83%              │
│   → NVLink(A100/H100): 仅1.16-1.94x开销 → Ring可行                │
│   → 长序列解决方案: 单GPU + FlashAttention(省85-97%内存)          │
│                                                                      │
│ === 推理 (综合20+benchmark实测) ===                                  │
│                                                                      │
│ 单GPU推理                                                             │
│   → 最优方案! 7B INT4+INT8 KV fits 24GB(B=118)                    │
│   → 7B BF16+INT8 KV+FlashInfer → B=35 → 3,726 tok/s               │
│   → 7B INT4+INT8 KV+FlashInfer → B=74 → ~5,100 tok/s              │
│   → 7B BF16+INT8 KV+Eagle d5 → B=35 → ~15,649 tok/s              │
│                                                                      │
│ FlashInfer: 推理attention生产唯一答案!                               │
│   → 整体加速 1.06-3.20x(实测,随B增长)                              │
│   → GQA-8 attention-only 54.03x(消除KV expansion!)                 │
│   → GQA-5不支持(group_size=6.4非整数) → 生产必须GQA-8!            │
│   → B=1: 1.06x → B=32: 2.63x → B=55: 3.20x                      │
│                                                                      │
│ Decode GEMM: 0.6% peak → 98.2% TC闲置!                              │
│   → 极端memory-bound → 量化是唯一出路                                │
│   → INT4 AWQ+Marlin → 2-3x decode加速                               │
│   → INT8 weight → 2x理论加速(权重占70%→半权重=半时间)              │
│   → Ridge M=256(AI≈228) → decode<256 memory-bound/prefill≥256 compute-bound│
│                                                                      │
│ Triton kernel决策:                                                   │
│   → RMSNorm: Triton 2.75-3.23x vs PyTorch(cos_sim=1.000000)       │
│   → SiLU: Triton 0.42x → PyTorch 2.3x更快!(launch overhead)       │
│   → Softmax+Temp: Triton≈PyTorch(0.94-1.02x) → 无优势             │
│   → SwiGLU MLP: torch.compile 0.85x → cuBLAS最快 → compile负优化  │
│   → Element-wise: all <0.03ms → PyTorch native够快                  │
│   → 规律: Triton(reduction) + PyTorch(elem) + cuBLAS(GEMM) + FlashInfer(attn)│
│                                                                      │
│ CUDA Stream:                                                         │
│   → Multi-stream = 负优化(0.69x) → stream切换开销 > compute      │
│   → Stream priority = 1.5x慢 → default stream最优                 │
│   → Prefill+1decode = 0.92x负优化 → sequential更快                │
│   → Prefill+16decode = 1.20x → B≥16开始overlap                    │
│   → vLLM V1 single stream+token budget = 最优方案                  │
│                                                                      │
│ Batched GEMM (MoE):                                                  │
│   → torch.bmm 0.38-0.77x vs sequential → 大多数配置bmm更慢!       │
│   → MoE GEMM 3-17% peak → 极低利用率                                │
│   → A2A PCIe≈2.8ms >> GEMM 0.1-1ms → 通信是瓶颈                   │
│   → Dense model(GQA-8+FlashInfer) > MoE on RTX 4090               │
│   → FusedMoE(segmented matmul)是必需品 → torch.bmm不适合           │
│                                                                      │
│ KV Cache + Prefill:                                                  │
│   → INT8 KV: cos_sim=0.99996 → 接近free → 推荐                    │
│   → FP8 KV: cos_sim=0.999996 → 更准确! → 推荐(需vLLM per-tensor scaling)│
│   → StreamingLLM(4+4K) → 固定168MB → 无限对话! → 推荐              │
│   → Prefill O(N^1.5) → S↑4x→prefill↑6x → TTFT≤100ms→S≤16K       │
│   → Chunked Prefill慢1.4x但改善公平 → chunk=2048最优               │
│                                                                      │
│ 多GPU推理(TP)                                                          │
│   → **不可行!** TP=4/8 PCIe AllReduce仅5-6 GB/s                   │
│   → 通信占比96% → 消费级GPU多卡只适合training不适合推理            │
│   → NVLink(A100): TP=8 通信<5% → 有效!                              │
│                                                                      │
│ === RL训练(GRPO/PPO) ===                                             │
│                                                                      │
│ <10M模型                                                              │
│   → DDP + GRPO(outcome-only) + SFT暖启动                              │
│   → SFT→GRPO 93% eval(决定性! vs 纯GRPO 81%)                        │
│   → GRPO σ-norm: HURTS强SFT, HELPS弱SFT                              │
│   → 噪声σ=0.01 Goldilocks Zone → 100% eval                          │
│   → Reward: graded 32.5% > binary 24% > shaped 10%(hacking!)       │
│                                                                      │
│ 10M+模型                                                              │
│   → FSDP1 + GRPO + vLLM async rollout                                │
│   → GRPO 2模型省50%GPU vs PPO 4模型                                 │
│   → Prefix Sharing: attn-only 0.99x vs full-model 2.46x(MLP82%!)  │
│   → Rollout占74%时间 → 最大杠杆=vLLM/SGLang 2-5x                  │
│                                                                      │
│ === 优化策略 ===                                                      │
│                                                                      │
│ BF16 native: 最佳选择! 无GradScaler, 1.23x加速, 39%eval            │
│   → FP16+AMP: 无加速+eval更差 → AMP cast overhead                   │
│   → FP8训练: SM89不支持(仅推理)                                     │
│                                                                      │
│ AdamW: lr=0.001, wd=0.1最优(loss↓6.5%)                              │
│   → SGD: lr=0.1↓3.3%但收敛差                                       │
│   → L2正则在Adam灾难(λθ被√v缩放→反直觉!)                            │
│                                                                      │
│ LoRA: merged零推理开销, r=4最优, B=0初始化                           │
│   → 小模型仅3%内存收益, 7B+预计20-30%                               │
│                                                                      │
│ torch.compile: forward B=1 4.09x, B≥4 0.80-0.97x                   │
│   → training B=4 1.96x, B≥16 1.01-1.05x → 大batch几乎无收益        │
│   → 推理用FlashInfer不是compile; 训练B≤4用compile(default)          │
│                                                                      │
│ CUDA Graph: launch 8us, OPT-125M 2.43x, 7B仅1.05x                  │
│   → 收益∝launch占比, 大GPU收益小但消除jitter有价值                  │
│                                                                      │
│ Multi-stream: **负优化!** 小kernel 0.69x, stream priority 1.5x慢   │
│   → vLLM V1: single stream+token budget → RTX 4090最优             │
│                                                                      │
│ Triton kernel: reduction ops Triton胜, simple ops PyTorch胜         │
│   → RMSNorm Triton 2.75-3.23x → SiLU Triton 0.42x → 不要盲目写Triton│
│                                                                      │
│ Batch Scaling: GA=8 100%eval > GA=1 87%(大batch=更准确梯度)        │
│   → AdamW自适应lr≈自动sqrt scaling → SGD+GA完全失败              │
│                                                                      │
│ LR Schedule: constant最优(48%eval) > cosine(24%)                   │
│   → 小模型sqrt scaling, 大模型no-scaling                            │
│                                                                      │
│ ZeRO-3: Adam optimizer 12B/param占78% → 分片→8x降                  │
│   → ZeRO-3 DP=8: 7B训练从112.5→14.5GB → fits单GPU!              │
└──────────────────────────────────────────────────────────────────────┘
```

## 关键数据汇总

```
训练通信带宽 (RTX 4090 PCIe):
  AllReduce: 2GPU 7.59, 4GPU 3.31, 8GPU 3.01 GB/s
  all_gather: P=2 5.94, P=4 4.30, P=8 4.67 GB/s
  NVLink对比: A100 300 GB/s → 50-100x差距!

训练效率:
  DDP: 76K 84-94%/2.28M 71-90% (4-8GPU)
  FSDP1: 2GPU 125%→4GPU 70%→8GPU 61% (25M)
  FSDP2: 2GPU 61%→比DDP慢(composable API开销)
  Ring Attention: 7-67x慢(PCIe瓶颈, 不可overlap)

内存策略:
  Adam optimizer: 12 bytes/param → 占训练内存78%
  ZeRO-3: 分片optimizer→8x内存降(DP=8)
  Checkpointing+FSDP: 反增内存! FSDP1 every +11%/every2 +21%/every3 +32%
  Best: FSDP + no checkpointing = 最省内存
  Checkpointing有价值: 仅7B+大batch+128K context (activation>5%)

推理 (20+benchmark实测):
  7B BF16+INT8 KV+FlashInfer: B=35 → 3,726 tok/s → 推荐!
  7B INT4+INT8 KV+FlashInfer: B=74 → ~5,100 tok/s → 并发优先
  7B BF16+INT8 KV+Eagle d5: B=35 → ~15,649 tok/s → 最快!

  FlashInfer整体: 1.06-3.20x(实测, 随B增长)
  FlashInfer GQA-8 attn-only: 54.03x → GQA-5不支持 → 生产必须GQA-8!

  Decode GEMM: 0.6% peak(98.2% TC idle!) → memory-bound → quant化是唯一出路
  INT4 AWQ+Marlin: 2-3x decode加速 → fused kernel必需
  INT8 weight: 2x理论加速 → 权重读占70% → 半权重=半时间

  Triton kernel决策:
    RMSNorm: Triton 2.75-3.23x vs PyTorch → Triton(reduction)胜!
    SiLU: Triton 0.42x vs PyTorch → PyTorch(simple)胜! → launch overhead主导
    Softmax+Temp: Triton≈PyTorch → 无优势 → PyTorch够快
    SwiGLU MLP: torch.compile 0.85x → cuBLAS最快 → compile负优化
    Element-wise: all <0.03ms → launch overhead主导 → PyTorch native够快
    生产决策: Triton(reduction) + PyTorch(elem) + cuBLAS(GEMM) + FlashInfer(attn)

  CUDA Stream:
    Multi-stream = 负优化(0.69x)! → stream切换开销 > compute
    Stream priority = 1.5x慢 → default stream最优
    Prefill+1decode = 0.92x负优化 → sequential更快
    Prefill+16decode = 1.20x → B≥16开始overlap → 但收益有限
    vLLM V1 single stream+token budget = 最优方案

  CUDA Memory Allocator:
    碎片化=0-1%(极低!) → alloc/free不累积 → 不是瓶颈
    Pool slice=3.9x faster(3us vs 12us dynamic) → vLLM PagedAttention验证
    Pool冷启动231ms for 10GB → 之后近零 → 生产可接受
    7B BF16 13GB → OOM at 20GB → 量化是必需品

  Batched GEMM MoE:
    torch.bmm 0.38-0.77x vs sequential → bmm更慢!
    MoE GEMM 3-17% peak → 极低利用率 → dense model更优
    A2A PCIe≈2.8ms >> GEMM 0.1-1ms → 通信是真正瓶颈
    FusedMoE(segmented matmul)必需品 → torch.bmm不适合

  KV Cache量化:
    INT8 KV: cos_sim=0.99996 → 推荐
    FP8 KV: cos_sim=0.999996 → 更准确! → 推荐(需per-tensor scaling)
    StreamingLLM: 固定168MB → 无限对话!

  Prefill:
    S=4096: compute-bound 92.4% peak → cuBLAS near optimal
    Prefill O(N^1.5) → S↑4x→prefill↑6x → TTFT≤100ms→S≤16K

RL训练:
  SFT→GRPO: 93%eval(决定性!) vs 纯GRPO 81%/PPO 62%/DPO 40%
  σ-norm: HURTS强SFT(-22%), HELPS弱SFT(+5%)
  噪声σ=0.01: Goldilocks Zone → 100% eval
  Reward: graded 32.5% > binary 24% > shaped 10%
  DAPO vs GRPO: Capacity matching(76K→GRPO胜/449K→DAPO胜)
  Prefix Sharing: 2.46x(full-model), 0.99x(attn-only)

优化:
  BF16 native: 最安全最简单, 1.23x加速+39%eval
  AdamW: lr=0.001 wd=0.1最优, L2灾难(√v缩放)
  LoRA: merged零开销, r=4最优, 小模型3%/7B+20-30%
  compile: forward B=1 4.09x/B≥4 0.80x, training B=4 1.96x/B≥16 ~1.0x
  CUDA Graph: launch 8us, 125M 2.43x, 7B 1.05x
  Multi-stream: 负优化! 0.69x小kernel/1.5x stream priority
  Triton: reduction→Triton胜/simple→PyTorch胜/GEMM→cuBLAS胜
```

## 与NVLink GPU对比

```
RTX 4090 PCIe vs A100 NVLink:

| 特性 | RTX 4090 PCIe | A100 NVLink |
|------|---------------|-------------|
| GPU间带宽 | 5-6 GB/s | 300 GB/s (50-60x!) |
| P2P access | 禁用 | 全启用 |
| DDP效率 | 84-90% (4GPU) | ~95% |
| FSDP1 | 125%(2GPU)→61%(8GPU) | ~95% 恒定 |
| TP推理 | 不可行(96%通信) | <5%通信 |
| Ring Attention | 7-67x慢 | 1.16-1.94x |
| 7B训练 | ZeRO-3 DP=8勉强 | ZeRO-3 DP=2轻松 |
| 128K推理 | 不可能(OOM) | CP+TP可行 |

→ RTX 4090是单GPU王者, 多GPU受限
→ A100/H100是多GPU王者, NVLink解锁全并行
→ 消费级GPU: 单GPU+FlashAttention+ZeRO-3
→ 生产级GPU: 全并行(TP+PP+CP+ZeRO-3+DP)
```

## 工具索引

```
训练benchmark:
  tools/benchmark_ddp_scaling_4090.py — DDP vs GA scaling
  tools/fsdp2_benchmark_4090.py — FSDP1/FSDP2/FSDP2+compile/DDP
  tools/checkpointing_fsdp_benchmark_4090.py — Checkpointing+FSDP/DDP
  tools/benchmark_grpo_training_4090.py — GRPO training
  tools/benchmark_nccl_allreduce_4090.py — NCCL AllReduce bandwidth
  tools/benchmark_gpu_interconnect_4090.py — GPU互连(P2P/PCIe/HBM)

推理benchmark (20+):
  tools/flashinfer_real_decode_benchmark.py — FlashInfer整体加速实测(1.06-3.20x)
  tools/e2e_inference_pipeline_benchmark.py — E2E推理管道(prefill+decode)
  tools/inference_calculator_4090.py — 综合推理计算器(整合20+benchmark)
  tools/triton_kernel_workshop_benchmark.py — Triton kernel workshop(RMSNorm/SiLU/SwiGLU)
  tools/gemm_shape_analysis.py — GEMM形状分析(decode 0.6% peak/ridge M=256)
  tools/cuda_stream_concurrency.py — CUDA stream并发(multi-stream负优化)
  tools/batched_gemm_analysis.py — Batched GEMM MoE分析(bmm 0.38x)
  tools/comprehensive_inference_benchmark.py — 量化+KV+HBM带宽
  tools/fp8_kv_cache_benchmark.py — FP8 vs INT8 KV量化
  tools/flashinfer_fp8_kv_benchmark.py — FlashInfer FP8 KV
  tools/benchmark_flash_attention_4090.py — FlashAttention vs SDPA
  tools/benchmark_decode_gemm_4090.py — Decode GEMM latency
  tools/benchmark_kv_cache_bandwidth_4090.py — KV cache bandwidth
  tools/benchmark_quantization_4090.py — Quantization FP16/INT8/FP8

RL训练benchmark:
  tools/mini_grpo_training.py — Mini GRPO/PPO/DPO/DAPO/RLOO
  tools/unified_rl_comparison.py — 7方法统一对比
  tools/batch_size_scaling.py — Batch size scaling
  tools/lr_schedule_batch_scaling.py — LR schedule+batch scaling

序列并行:
  tools/ring_attention_benchmark_4090.py — Ring Attention P=2/4/8
  tools/sequence_parallel.py — Ring Attention simulation (Python)
  tools/sequence_parallel_sim.py — SP模拟器

优化benchmark:
  tools/benchmark_fused_rms_norm.py — CUDA/Triton/PyTorch RMSNorm
  tools/benchmark_cuda_graph_4090.py — CUDA Graph
  tools/benchmark_decode_roofline_4090.py — Decode Roofline
```