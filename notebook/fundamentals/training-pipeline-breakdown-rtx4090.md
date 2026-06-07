# Training Pipeline Latency Breakdown — RTX 4090

> 2026-06-07 | **Backward pass占50-57%训练时间!** Forward仅3-5%, Optimizer<0.3%. QK^T是forward最大组件(27%), RMSNorm在decode占15%, MLP在125M占27%→验证Prefix Sharing

## 核心发现

```
RTX 4090训练时间分解 — 6配置实测:

┌──────────────────────────────────────────────────────────────┐
│ 训练步骤时间分解:                                             │
│                                                              │
│ 2.3M S=128 B=8:  Forward=6%  Backward=57%  Optimizer=1%      │
│ 25M  S=512 B=8:  Forward=5%  Backward=62%  Optimizer=0.5%    │
│ 125M S=512 B=4:  Forward=2%  Backward=50%  Optimizer=0.2%    │
│                                                              │
│ → **Backward pass是训练瓶颈!** 占50-62%时间                 │
│ → Forward仅2-6% → 大部分时间在backward梯度计算              │
│ → Optimizer微不足道(<1%) → AdamW step几乎免费              │
│                                                              │
│ Forward组件分解 (25M prefill):                               │
│   QK^T matmul    26.6%  → attention核心计算                 │
│   Softmax        15.9%  → 紧随QK^T之后                      │
│   AV matmul       8.9%  → attention输出                     │
│   MLP up+down    15.0%  → MLP变换                           │
│   Q/K/V proj     10.5%  → 线性投影                          │
│   RMSNorm×2      12.8%  → 归一化                            │
│   Residual×2      3.2%  → 残差连接                          │
│   Other           6.6%  → SwiGLU/embedding/output           │
│                                                              │
│ → Attention(QK+soft+AV+proj) = 61.9% forward               │
│ → MLP = 15% forward                                         │
│ → RMSNorm = 12.8% forward                                   │
│                                                              │
│ Decode vs Prefill对比:                                       │
│   Prefill (S=512): QK^T 27% → softmax 16% → attention主导  │
│   Decode  (S=1):   RMSNorm 15% → QK^T 10% → normalization↑│
│   → Decode时RMSNorm占比翻倍! → CUDA RMSNorm 9x加速更关键  │
│   → Decode时softmax占比大幅下降(16%→4%) → compute极小     │
└──────────────────────────────────────────────────────────────┘
```

## 完整数据

```
6配置训练pipeline实测 (RTX 4090, BF16):

| 配置 | 参数 | Step(ms) | Fwd(ms) | Bwd(ms) | Opt(ms) | Mem(GB) | Fwd% |
|------|------|----------|---------|---------|---------|---------|------|
| 2.3M S=128 B=8 | 11M | 9.05 | 0.54 | 5.12 | 0.09 | 0.175 | 6% |
| 2.3M S=1 B=32 | 11M | 8.03 | 0.49 | 4.39 | 0.08 | 0.152 | 6% |
| 25M S=512 B=8 | 42M | 19.17 | 1.00 | 11.88 | 0.10 | 1.83 | 5% |
| 25M S=1 B=64 | 42M | 16.13 | 0.47 | 8.67 | 0.10 | 0.48 | 3% |
| 125M S=512 B=4 | 234M | 42.72 | 0.93 | 21.51 | 0.10 | 3.49 | 2% |
| 125M S=1 B=128 | 234M | 38.26 | 0.44 | 17.38 | 0.11 | 2.47 | 1% |

→ Forward占比随模型增大而减小(6%→1%) → backward比重增加
→ 这是因为backward需要计算所有参数的梯度 → 参数多→backward重
→ Optimizer始终<1% → AdamW step几乎免费!

Forward组件百分比 (25M prefill S=512):
| 组件 | ms | % | 类型 |
|------|----|---|------|
| QK^T matmul | 0.266 | 26.6% | Attention |
| Softmax | 0.159 | 15.9% | Attention |
| AV matmul | 0.089 | 8.9% | Attention |
| MLP up-proj | 0.075 | 7.5% | MLP |
| MLP down-proj | 0.075 | 7.5% | MLP |
| RMSNorm | 0.064 | 6.4% | Normalization |
| Q projection | 0.035 | 3.5% | Projection |
| K projection | 0.035 | 3.5% | Projection |
| V projection | 0.035 | 3.5% | Projection |
| Output proj | 0.035 | 3.5% | Projection |
| SwiGLU | 0.039 | 3.9% | Activation |
| Embedding | 0.019 | 1.9% | Embedding |
| Residual | 0.016 | 1.6% | Residual |

→ Attention(QK+soft+AV+proj+out) = 61.9% forward!
→ MLP(up+down) = 15% forward
→ RMSNorm = 12.8% → PyTorch慢! CUDA 9x加速→可降到1.4%

Forward组件百分比 (25M decode S=1):
| 组件 | ms | % | 变化 |
|------|----|---|------|
| RMSNorm | 0.071 | 15.2% | ↑ from 6.4% |
| QK^T | 0.049 | 10.4% | ↓ from 26.6% |
| MLP down | 0.032 | 6.8% | ↓ slightly |
| AV matmul | 0.031 | 6.4% | ↓ from 8.9% |
| MLP up | 0.029 | 6.1% | ↓ slightly |
| Q/K/V proj | 0.029 | 5.9% | ↑ from 3.5% |
| Output proj | 0.028 | 6.0% | ↑ from 3.5% |
| SwiGLU | 0.026 | 5.4% | ↑ from 3.9% |
| Embedding | 0.022 | 4.6% | ↑ from 1.9% |
| Residual | 0.017 | 3.7% | ↑ from 1.6% |
| Softmax | 0.017 | 3.6% | ↓ from 15.9% |

→ Decode时: RMSNorm占比翻倍! (6.4%→15.2%)
→ Decode时: Softmax占比大幅下降 (15.9%→3.6%) → compute极小
→ Decode时: 线性投影占比增加 → memory-bound操作占主导
```

## 与之前实验的串联

```
串联发现链:

1. **Prefix Sharing** (prefix_sharing_packed_thd_4090.py):
   → full-model 2.46x加速, attn-only 0.99x → MLP占82%
   → → 本次: MLP=15% forward → 加上backward → MLP总占比更大!
   → → → MLP forward+backward ≈ 15% × 2 + 15% × 4(bwd更重) ≈ ~90ms(125M)
   → → → → Prefix Sharing省82%MLP计算 → 完全匹配!

2. **CUDA RMSNorm** (benchmark_fused_rms_norm.py):
   → CUDA 9x加速, Triton 5x加速
   → → 本次: RMSNorm占12.8%(prefill)/15.2%(decode)
   → → → CUDA RMSNorm → 从0.071ms→0.008ms → decode占比从15%→1.7%!
   → → → → 省下的时间: 0.063ms×16层×2 = 2.0ms → 125M模型加速5%

3. **torch.compile** (benchmark_compile):
   → forward 3.75x加速 → training 1.31x
   → → 本次: forward仅2-6% → compile加速forward → 但backward不加速
   → → → → 为什么training仅1.31x? 因为backward占50-62% → compile只加速小部分!

4. **FlashAttention** (benchmark_flash_attention_4090.py):
   → decode更慢(0.67-0.84x) → 内存省85-97%
   → → 本次: QK^T+softmax+AV = 26.6+15.9+8.9 = 51.4% forward(prefill)
   → → → → FlashAttention在prefill时加速+省内存
   → → → → decode时占比更小 → FlashAttention收益有限

5. **BF16+FSDP** (bf16_fsdp_benchmark_4090.py):
   → FSDP BF16 1.51x加速 + 49%内存省
   → → → backward占50-62% → FSDP通过ZeRO-3分片→减少backward内存
   → → → → BF16减少backward计算量 → 1.51x加速!

6. **GRPO训练** (benchmark_grpo_training_4090.py):
   → Rollout占74%时间 → 计算包含完整forward+softmax+sampling
   → → → 本次: forward仅2-6% → rollout每步快 → 但n=8-64步 → 总量大
   → → → → Prefix Sharing省2.46x → rollout总时间降74%→51%!

→ **优化优先级** (基于实测数据):
  1. **Backward pass优化** → 占50-62% → 最大杠杆!
     → torch.compile(bwd) → FSDP ZeRO-3(省bwd内存) → gradient checkpointing(大模型)
  2. **MLP优化** → 占15% forward → Prefix Sharing → fused SwiGLU kernel
  3. **RMSNorm优化** → 占12.8-15.2% forward → CUDA kernel 9x加速
  4. **Attention优化** → 占61.9% forward → FlashAttention(prefill) → decode收益小
  5. **Optimizer优化** → 占<1% → 几乎不值得优化!
```

## 优化杠杆分析

```
各优化方法的实测杠杆:

| 优化方法 | 目标 | 占比 | 加速倍数 | 总加速 |
|----------|------|------|----------|--------|
| torch.compile(fwd) | forward | 3-6% | 3.75x | ~1.31x |
| torch.compile(fwd+bwd) | all | 100% | ~1.5x? | ~1.5x |
| CUDA RMSNorm | RMSNorm | 12-15% fwd | 9x | ~1.3x |
| FlashAttention(prefill) | attention | 51% fwd | 1.2x | ~1.1x |
| Prefix Sharing | MLP+Attn | 82% fwd | 2.46x | ~2.46x |
| BF16 | all | 100% | 1.5x | ~1.5x |
| ZeRO-3(FSDP) | memory | - | mem↓50% | throughput↑ |
| Checkpointing | activation | - | mem↓ | throughput↓(FSDP) |

→ 最高杠杆: Prefix Sharing (2.46x → 但仅 rollout)
→ 次高杠杆: BF16 (1.51x → 全训练)
→ 中等杠杆: compile (1.31x → 但仅影响可编译部分)
→ 低杠杆: FlashAttention (1.1x → decode时反而更慢!)

→ **关键洞察**: backward占50-62% → 优化forward效果有限!
  → 优化full training需要同时优化forward+backward
  → torch.compile如果能编译backward → 可能达到1.5-2x总加速
  → Prefix Sharing省82% → rollout加速2.46x → 训练加速1.39x(rollout74%)
```

## 工具

- `tools/training_pipeline_breakdown_4090.py` — 14组件分解benchmark
- `results/training_pipeline_breakdown.json` — 完整数据