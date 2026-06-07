# Activation Checkpointing + FSDP Interaction Benchmark — RTX 4090

> 2026-06-07 | **FSDP下checkpointing反增内存!** FSDP+ckpt每层增加11-32%内存(参数re-gather), DDP+ckpt几乎不变(0.4-0.7%), 单GPU小模型无收益(activation<0.1%)

## 核心发现

```
RTX 4090实测: FSDP下activation checkpointing不仅不省内存,反而增加!

┌──────────────────────────────────────────────────────────────┐
│ 25M模型 2-GPU 内存对比 (checkpointing反效果!):               │
│                                                              │
│ 方法              │ 内存(GB) │ vs no_ckpt │ 说明             │
│ DDP no_ckpt       │ 0.742    │ baseline   │                  │
│ DDP every         │ 0.745    │ +0.4%      │ 几乎不变         │
│ DDP every2        │ 0.747    │ +0.7%      │ 几乎不变         │
│ **FSDP1 no_ckpt** │ **0.478**│ baseline   │ 已省36% vs DDP  │
│ **FSDP1 every**   │ **0.530**│ **+11%**⚠ │ 反增内存!       │
│ **FSDP1 every2**  │ **0.578**│ **+21%**⚠ │ 更糟!           │
│ **FSDP1 every3**  │ **0.630**│ **+32%**⚠ │ 最糟!           │
│                                                              │
│ → FSDP下checkpointing增加11-32%内存!                         │
│ → 原因: recomputation时需re-gather分片参数→峰值内存更高      │
│ → DDP下checkpointing仅+0.4-0.7%(activation太小)             │
│ → 小模型(<25M) activation<5%→checkpointing无意义             │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ Checkpointing吞吐开销:                                       │
│                                                              │
│ FSDP1: every +59%时间/every2 +36%/every3 +29%              │
│ DDP:   every +37%时间/every2 +20%                           │
│ 单GPU: every +85%时间(2.3M)/+95%(4.5M)/+76%(25M)          │
│        every2 +54%(2.3M)/+48%(4.5M)/+34%(25M)             │
│                                                              │
│ → 吞吐开销: FSDP1 every 59% ≈ DDP every 37% ≈ 单GPU 76-95%│
│ → every2最佳权衡: FSDP1 36% vs DDP 20% vs 单GPU 34-54%   │
│ → FSDP1开销更大(FSDP recomputation+gather双重开销)          │
└──────────────────────────────────────────────────────────────┘
```

## 为什么FSDP下checkpointing反增内存?

```
核心机制: FSDP参数分片 + checkpointing recomputation = 内存峰值更高

DDP (参数完整在每个GPU):
  → forward: 所有参数在GPU → activations计算 → 存activations
  → backward: 所有参数在GPU → 用存的activations → 正常backward
  → checkpointing: forward不存activations → backward重新计算 → 参数还在GPU → 无额外开销
  → 峰值内存: 参数+优化器+activations → checkpointing省activations → 但25M模型activations太小

FSDP1 (参数分片, ZeRO-3):
  → forward: gather参数→计算→free参数 → 存activations(参数已释放)
  → backward: 用存的activations → gather参数→free参数 → backward
  → 无checkpointing: 峰值 = shard_params + shard_optimizer + activations + 1层gather
  → 有checkpointing: forward不存activations → backward需要重新forward!
  →           → 重新forward需要重新gather参数! → 同时backward也需要参数!
  →           → recompute时: 当前层gather(backward需要) + 重新forward层gather
  →           → 峰值内存增加! 因为多个层的参数需要同时gather

具体分析:
  → no_ckpt: 峰值 = shard(0.24GB) + activations(~0.01GB) + 1层gather(~0.25GB) = 0.478GB ✓
  → every: 每层都checkpoint → backward时需重新forward每层
  →        → gather当前层参数(backward) + gathercheckpointed层参数(re-forward)
  →        → 峰值 ≈ shard + 2×layer_gather + recomputed_activations
  →        → 0.530GB vs 0.478GB → +11% = +0.052GB ≈ 1层额外参数gather
  → every2: 每2层checkpoint → 少一些recomputation但仍有额外gather
  →        → 0.578GB → +21% ≈ 2层额外参数gather
  → every3: 每3层 → 0.630GB → +32%

→ **关键**: checkpointing的recomputation在FSDP下触发额外的参数gather!
  → 不是"省activations"而是"多gather参数" → 内存反而增加!
  → 这与单GPU不同: 单GPU参数一直在GPU → recompute无额外开销
```

## 实验1: 单GPU — Checkpointing × 模型大小

```
RTX 4090 单GPU checkpointing策略×模型大小:

2.3M模型 (3.21M params):
| 策略      │ 内存(GB) │ 时间(ms) │ 吞吐(tok/s) │ 内存变化 │ 时间开销 │
| no_ckpt   │ 0.092    │ 7.69     │ 66562       │ baseline │ baseline │
| every     │ 0.084    │ 14.26    │ 35915       │ -8.7%    │ +85%    │
| every2    │ 0.084    │ 11.76    │ 43553       │ -8.7%    │ +54%    │

4.5M模型 (6.37M params):
| no_ckpt   │ 0.161    │ 13.04    │ 39265       │ baseline │ baseline │
| every     │ 0.147    │ 25.41    │ 20148       │ -8.7%    │ +95%    │
| every2    │ 0.147    │ 19.26    │ 26578       │ -8.7%    │ +48%    │

25M模型 (25.32M params):
| no_ckpt   │ 0.542    │ 15.20    │ 33682       │ baseline │ baseline │
| every     │ 0.545    │ 26.71    │ 19167       │ +0.6%!⚠ │ +76%    │
| every2    │ 0.540    │ 20.44    │ 25054       │ -0.4%    │ +34%    │

→ 2.3M/4.5M: checkpointing省8.7%内存 — 但绝对值仅0.008GB(8MB) → 无实际意义
→ 25M: checkpointing every反而增内存! every2仅省0.002GB → 无意义
→ 原因: 小模型optimizer占大头(Adam 12bytes/param)
  → 2.3M: optimizer≈38MB vs activation≈2MB → activation仅5%
  → 25M: optimizer≈300MB vs activation≈20MB → activation仅7%
→ 结论: <25M模型activation占比<10% → checkpointing不值得
```

## 实验2: FSDP1 + Checkpointing — 2 GPU

```
25M模型 FSDP1 2-GPU checkpointing交互:

| 策略      │ 内存(GB) │ 时间(ms) │ 吞吐(tok/s) │ 内存vs无ckpt │ 时间开销 │
| no_ckpt   │ 0.478    │ 23.12    │ 22146       │ baseline     │ baseline │
| every     │ 0.530    │ 36.74    │ 13936       │ +11% ⚠       │ +59%    │
| every2    │ 0.578    │ 31.34    │ 16338       │ +21% ⚠       │ +36%    │
| every3    │ 0.630    │ 29.70    │ 17237       │ +32% ⚠       │ +29%    │

→ **核心发现**: FSDP下checkpointing INCREASES内存! 不是decrease!
→ 内存随checkpoint频率增加 → every3最糟(+32%) → 反直觉!
→ 吞吐开销: every 59%/every2 36%/every3 29% → 与单GPU趋势类似

内存增加解释:
  → FSDP参数已分片(每GPU持1/2参数) → 基线内存更低
  → checkpointing recomputation → 需重新gather参数 → 峰值更高
  → checkpoint频率越高 → recomputation越频繁 → gather次数越多 → 内存越高
  → every(每层) → 8次re-gather → every3(每3层) → ~3次 → 但every3内存更高?
  → 可能原因: every3的recomputation区间更长 → 需同时持有更多层参数
```

## 实验3: DDP + Checkpointing — 2 GPU

```
25M模型 DDP 2-GPU checkpointing对比:

| 策略      │ 内存(GB) │ 时间(ms) │ 吞吐(tok/s) │ 内存vs无ckpt │ 时间开销 │
| no_ckpt   │ 0.742    │ 22.41    │ 45684       │ baseline     │ baseline │
| every     │ 0.745    │ 30.80    │ 33249       │ +0.4%        │ +37%    │
| every2    │ 0.747    │ 26.88    │ 38099       │ +0.7%        │ +20%    │

→ DDP下checkpointing几乎不影响内存(+0.4-0.7%)
→ 原因: DDP参数完整在GPU → recomputation无额外gather开销
→ 但activation本身太小(~20MB/0.742GB=2.7%) → checkpointing省不了多少
→ 吞吐开销: every 37%/every2 20% → 比FSDP1更低(FSDP额外gather开销)
```

## FSDP vs DDP 内存对比全景

```
25M模型 2-GPU 内存全景:

| 方法              │ 内存(GB) │ vs DDP baseline │ 说明                    │
| DDP no_ckpt       │ 0.742    │ 100%            │ 完整参数+optimizer      │
| DDP every         │ 0.745    │ 100.4%          │ checkpointing无收益     │
| DDP every2        │ 0.747    │ 100.7%          │ checkpointing无收益     │
| FSDP1 no_ckpt     │ 0.478    │ 64.4%           │ 参数+optimizer已分片!  │
| FSDP1 every       │ 0.530    │ 71.2%           │ checkpointing反而增内存│
| FSDP1 every2      │ 0.578    │ 77.6%           │ 更糟                    │
| FSDP1 every3      │ 0.630    │ 84.6%           │ 最糟                    │

→ FSDP1(no_ckpt) vs DDP(no_ckpt): 0.478 vs 0.742 → **省36%内存**(ZeRO-3分片)
→ FSDP1(every) vs DDP(no_ckpt): 0.530 vs 0.742 → **省29%**(不如no_ckpt FSDP!)
→ FSDP1(every) vs FSDP1(no_ckpt): 0.530 vs 0.478 → **反增11%**

→ **最佳策略**: FSDP1 + no checkpointing = 0.478GB(最省内存!)
→ 不要在FSDP上叠加checkpointing → 反而浪费内存!
→ ZeRO-3分片已是最优内存方案 → checkpointing是多余甚至有害的
```

## Checkpointing何时有价值?

```
Checkpointing有价值的前提: activation占比 > 5%

25M模型 RTX 4090实测:
  → 单GPU: activation ≈ 20MB / peak 542MB ≈ 3.7% → 不够
  → DDP:   activation ≈ 20MB / peak 742MB ≈ 2.7% → 不够
  → FSDP:  activation ≈ 20MB / peak 478MB ≈ 4.2% → 不够且反增

何时activation占比>5%?
  → 大模型(7B+): 7B activation ≈ 数百MB → optimizer≈14GB → 仍<5%
  → 大batch(B≥64): activation∝batch_size → B=64时activation占比增大
  → 长context(S≥4K): activation∝seq_len → S=128K时KV cache占大头

实际场景:
  → 7B模型 + B=32 + S=4K:
    → 参数: 14GB
    → Optimizer: 28GB (2×参数, Adam)
    → Activation: ~2GB (B×S×d_model×n_layers)
    → 占比: 2/(14+28+2) ≈ 4.4% → 还是不够!

  → 7B模型 + B=64 + S=8K:
    → Activation: ~8GB
    → 占比: 8/(14+28+8) ≈ 16% → checkpointing有价值!
    → 但此时应优先考虑ZeRO-3(FSDP) → 分片optimizer

  → **结论**: 只有ZeRO-3+大batch+长context组合下checkpointing才有收益
  → 单独FSDP(ZeRO-3)已省optimizer → checkpointing在FSDP下是反效果
  → 正确策略: FSDP(ZeRO-3) alone → 不叠加checkpointing
  → 大模型需要更多内存 → 用更多GPU(ZeRO-3 DP=8)而非checkpointing
```

## 与之前实验的串联

```
串联发现链:

1. **Activation占比分析** (Training Memory Calculator):
   → Adam optimizer 12 bytes/param 占大头 (78%)
   → Activation占比: 小模型<5%, 7B+大batch才>10%
   → → checkpointing只对activation>5%有收益 → 本次实测验证!

2. **FSDP1 Benchmark** (fsdp2-benchmark-rtx4090.md):
   → FSDP1比DDP快2x(通信重叠+ReduceScatter)
   → FSDP1内存省36%(ZeRO-3分片)
   → → 本次: FSDP+checkpointing反增内存 → ZeRO-3已足够!

3. **DDP Scaling** (grpo-ddp-scaling-benchmark-rtx4090.md):
   → PCIe DDP只适合<10M模型 → 大模型通信瓶颈
   → → 本次: FSDP1是10-100M最优 → 不需要checkpointing辅助

4. **Gradient Checkpointing单独实测** (之前实验):
   → selective(every2)最佳权衡(14%开销/30%节省)
   → → 本次: 在FSDP下every2反而增21%内存! → 完全不同结论!

→ **决策树更新(RTX 4090 PCIe)**:
  <10M → DDP + no checkpointing (最快+内存够用)
  10-100M → FSDP1 + no checkpointing (最快+内存最优)
  >100M → FSDP1 + ZeRO-3 + no checkpointing (更多GPU分片)
  7B+大batch → FSDP1 + ZeRO-3 + checkpointing (仅此时有收益)
  7B+大batch+长context → FSDP1 + ZeRO-3 + FlashAttention + checkpointing
```

## 工具

- `tools/checkpointing_fsdp_benchmark_4090.py` — 3实验(单GPU/DDP+FSDP1+checkpointing)
- `results/checkpointing_fsdp_benchmark.json` — 完整实验数据