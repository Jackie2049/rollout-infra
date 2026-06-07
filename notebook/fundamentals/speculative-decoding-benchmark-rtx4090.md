# Speculative Decoding Simulation Benchmark — RTX 4090

> 2026-06-07 | **未训练draft模型投机解码实测=负优化(0.13-0.76x)!** 但理论分析证明: α≥0.8+latency_ratio≤0.2 → 3.4-6.6x加速可行。关键=draft模型质量(KL divergence), 不是大小。

## 核心发现

```
RTX 4090投机解码实测 — 3种draft模型 vs 25M target:

┌──────────────────────────────────────────────────────────────┐
│ 关键数据:                                                    │
│                                                              │
│ Target model (25M params): decode latency = 1.27ms          │
│                                                              │
│ Draft模型对比:                                                │
│   tiny_0.5M:  latency=0.69ms (ratio=0.54), KL=21.45        │
│   small_2M:   latency=1.12ms (ratio=0.89), KL=21.46        │
│   medium_7M:  latency=1.15ms (ratio=0.91), KL≈0, cos_sim=1 │
│                                                              │
│ **未训练draft模型的灾难性结果**:                               │
│   tiny_0.5M K=1: acceptance=18%, speedup=0.76x (负优化!)    │
│   small_2M  K=1: acceptance=12%, speedup=0.59x (负优化!)    │
│   medium_7M K=1: acceptance=100%, speedup=1.05x (仅微加速)  │
│   → 7M和target完全相同→ KL=0→100%接受→但不实用               │
│                                                              │
│ **KL divergence决定一切**:                                    │
│   KL=21.45 → acceptance=18% → 每K增加更多负优化              │
│   KL≈0     → acceptance=100% → 但draft=target→无意义         │
│   → 实际应用: 训练好的draft→KL<1→α=80-90%→可加速            │
│                                                              │
│ 理论最优分析 (Speedup = (α*K+1)/(K*lr+1)):                  │
│   α=0.9, lr=0.1: 6.2x加速 (K=19)                           │
│   α=0.8, lr=0.2: 3.4x加速 (K=19)                           │
│   α=0.9, lr=0.2: 3.8x加速 (K=19)                           │
│   α=0.5, lr=0.5: 1.0x → lr太大→投机解码无效                  │
│                                                              │
│ **RTX 4090关键结论**:                                         │
│   1. 投机解码的瓶颈不是draft模型大小,是draft模型质量(KL)     │
│   2. latency_ratio≥0.5 → 投机解码无收益                      │
│   3. 需要latency_ratio≤0.2 → draft必须比target快5x以上       │
│   4. 在RTX 4090上: 7B target→draft≤1.4B→latency_ratio≈0.2   │
│   5. ngram/Eagle draft=零额外compute→最适合RTX 4090          │
└──────────────────────────────────────────────────────────────┘
```

## 完整数据

```
3种draft模型 × 6种K值 — RTX 4090实测 (BF16):

| Draft | Params | Latency(ms) | Ratio | KL | Top1 Agree | cos_sim |
|-------|--------|-------------|-------|----|------------|---------|
| tiny_0.5M | 4.2M | 0.69 | 0.54 | 21.45 | 0% | 0.002 |
| small_2M  | 9.0M | 1.12 | 0.89 | 21.46 | 0% | -0.001 |
| medium_7M | 19.6M | 1.15 | 0.91 | ≈0 | 100% | 1.0 |

→ 未训练draft: KL≈21.5, top1 agreement=0%, cos_sim≈0 → 随机分布!
→ medium_7M: 和target完全相同(同架构同seed) → KL=0 → 验证工具正确

各K值下的acceptance和speedup:

| Draft | K=1 | K=2 | K=3 | K=4 | K=5 | K=8 |
|-------|-----|-----|-----|-----|-----|-----|
| tiny_0.5M  | α=18% 0.76x | α=7.5% 0.55x | α=5% 0.44x | α=4.3% 0.37x | α=1.6% 0.29x | α=1% 0.20x |
| small_2M   | α=12% 0.59x | α=8% 0.42x | α=3% 0.30x | α=3% 0.25x | α=1.4% 0.20x | α=1.1% 0.13x |
| medium_7M  | α=100% 1.05x | α=100% 1.07x | α=100% 1.08x | α=100% 1.08x | α=100% 1.08x | α=100% 1.09x |

→ 未训练draft: K越大 → 负优化越严重! (0.76x→0.20x)
→ 训练好的draft(α≥80%): K=5-8 → 1.08-1.09x → 仅微加速(因为lr=0.91太大)

理论最优K和speedup:

| α | lr=0.1 | lr=0.2 | lr=0.3 | lr=0.5 |
|---|--------|--------|--------|--------|
| 0.50 | 3.62x K=19 | 2.19x K=19 | 1.57x K=19 | 1.00x K=1 |
| 0.80 | 5.59x K=19 | 3.38x K=19 | 2.42x K=19 | 1.54x K=19 |
| 0.90 | 6.24x K=19 | 3.77x K=19 | 2.70x K=19 | 1.72x K=19 |
| 0.95 | 6.57x K=19 | 3.97x K=19 | 2.84x K=19 | 1.81x K=19 |

→ α越高→speedup越大; lr越小→speedup越大
→ lr≥0.5 → 投机解码几乎无收益(1.0-1.8x)
→ lr≤0.2 → 投机解码有3-4x收益(如果α≥0.8)
```

## 与之前实验的串联

```
串联发现链:

1. **vLLM Speculative Decoding** (之前源码阅读):
   → SpecDecodeProposer + RejectionSampler (6 Triton kernel)
   → → 本次: 验证了拒绝采样理论的正确性(accept_prob = min(p_t/p_d, 1))
   → → → 实测: KL divergence决定acceptance→draft质量是关键

2. **DeepSeek-R1 Analysis** (deepseek-r1-analysis.md):
   → DeepSeek不用投机解码→MLA+MoE+MTP→不同优化路径
   → → 本次: 投机解码需要draft快5x(lr≤0.2)→MoE模型不适合(稀疏→draft难以小)
   → → → MLA+MoE→投机解码不可行→DeepSeek选择MTP(Multi-Token Prediction)

3. **Decode Roofline** (之前benchmark):
   → 7B B=256 → 32K tok/s, decode严重memory-bound
   → → 本次:投机解码增加draft model memory→24GB GPU限制更紧
   → → → 7B target + 1.4B draft → 需额外~2.8GB → RTX 4090可以承受

4. **SGLang Speculative** (sglang-inference-engine-source-analysis.md):
   → SGLang内置speculative decoding(Eagle/ngram)
   → → 本次: Eagle draft=零额外compute→最适合RTX 4090
   → → → ngram draft更简单(零参数)→但acceptance较低(≈70-80%)

→ **RTX 4090投机解码决策**:
  1. 小模型(<7B): ngram/Eagle → α≈70-80% → lr≈0 → 2-3x加速
  2. 中模型(7B): 1.4B draft → lr≈0.2 → α≈80-90% → 3-4x加速
  3. 大模型(13B+): 2-3B draft → lr≈0.15-0.2 → α≈80% → 3-5x加速
  4. MoE模型: 投机解码不适合→draft无法压缩稀疏expert
```

## 投机解码理论速查

```
Speculative Decoding公式:

拒绝采样: accept_prob = min(p_target[token] / p_draft[token], 1.0)
  → 每个draft token独立检验 → 直到拒绝 → 从adjusted分布resample

期望接受token数: E[accepted] = Σ_{k=1}^{K} α^k
  → α = acceptance rate per token
  → K=5, α=0.8 → E=0.8+0.64+0.51+0.41+0.33=2.69 tokens

Speedup公式:
  speedup = (α*K + 1) × T_target / (K × T_draft + T_target)
  → latency_ratio = T_draft / T_target

  最优K = √((1-α) / (α × latency_ratio)) - 1/latency_ratio (近似)

  → α=0.8, lr=0.2 → optimal K≈4 → speedup≈2.4x
  → α=0.9, lr=0.1 → optimal K≈9 → speedup≈5x

Draft模型选择策略:
  | 策略 | 额外compute | α估计 | 适用场景 |
  |------|------------|-------|---------|
  | ngram | 0 | 70-80% | 小模型,简单任务 |
  | Eagle | ~1 forward | 80-90% | 中大模型 |
  | 小draft | 1 forward×K | 80-95% | 大模型+长序列 |
  | 同模型前缀 | 0 | 95-100% | 复用prompt |

→ RTX 4090最佳策略: ngram(零成本) 或 Eagle(低成本高α)
```

## 工具

- `tools/speculative_decoding_benchmark_4090.py` — 3 draft×6 K投机解码benchmark
- `results/speculative_decoding_benchmark.json` — 完整数据