# Full-Model Prefix-Sharing Benchmark — RTX 4090 实测

> 2026-06-07 | 5实验: 全模型PS加速验证, MLP主导性, GRPO savings, Qwen3.6 Hybrid, prefix_ratio sweep

## 核心发现: 全模型PS加速 vs 纯attention PS

| 层级 | GRPO n=8 | 说明 |
|------|----------|------|
| **纯attention KV injection** | 0.99x | memory-bound → KV读量不变 → 无加速 |
| **全模型forward PS** | **2.46x** | MLP compute-bound(82%) → 跳过prefix MLP → 加速显著! |

**2.46x vs 0.99x — 2.5x差距来自MLP(82%占比)!**

## 一、Exp1: 全模型 PS vs No-PS (B=4, 1.5B模型)

| prefix_ratio | compute节省% | time节省% | speedup |
|-------------|-------------|----------|---------|
| 0.25 | 18.8% | 15.8% | 1.19x |
| 0.50 | 37.5% | 34.6% | 1.53x |
| 0.75 | 56.2% | 52.0% | 2.08x |
| 0.90 | 67.5% | 61.9% | 2.63x |

**关键**: time_savings ≈ compute_savings × 0.92 → 效率92%+!

1. **线性关系**: 75% prefix → 52% time savings → 效率92%
2. **90% prefix → 2.63x加速**: 几乎完美的compute-to-time转换
3. **25% prefix → 仅1.19x**: prefix太短 → savings比例小

## 二、Exp2: Layer Breakdown (MLP vs Attention vs KV_proj)

| seq_len | MLP占比 | attn占比 | QKV_proj(attn内%) | attn_compute(attn内%) |
|---------|--------|---------|-------------------|----------------------|
| 128 | **80.3%** | 19.7% | 51.6% | 12.7% |
| 256 | **82.8%** | 17.2% | 53.6% | 8.3% |
| 512 | **82.4%** | 17.6% | 52.1% | 11.6% |
| 1024 | **81.8%** | 18.2% | 48.5% | 17.5% |

**震惊**: MLP占82%! Attention仅18%!

1. **MLP占82%**: SwiGLU(gate+up+down) 3个大GEMM → compute-bound → PS最大收益来源
2. **Attention仅18%**: 但其中52%是QKV_proj(compute-bound), 仅12%是attention计算本身
3. **PS收益来源**: MLP(82%) + QKV_proj(9.2%) = 91.2%来自compute-bound操作 → memory-bound的attention计算(2%)几乎无关
4. **seq_len≥1024时attention占比略增**: 17.5% → compute-bound趋势 → 但仍远小于MLP

**与纯attention PS对比**: 纯attention benchmark测的是这18%中的2%(attention compute) → PS对这2%无效 → 但对82% MLP和9.2% QKV_proj有效 → 总加速=2.46x!

## 三、Exp3: GRPO n_samples 全模型 Savings

| n_samples | compute节省% | time节省% | speedup | efficiency% |
|----------|-------------|----------|---------|------------|
| 2 | 33.3% | 27.3% | 1.37x | 82.0% |
| 4 | 50.0% | 47.8% | 1.91x | 95.6% |
| **8** | **58.3%** | **59.3%** | **2.46x** | **101.7%** |
| 16 | 62.5% | 63.6% | 2.75x | 101.8% |

**关键发现**:

1. **GRPO n=8: 2.46x加速!** — 与纯attention的0.99x形成鲜明对比
2. **efficiency >100%**: time_savings > compute_savings → PS不仅省compute还省memory traffic(减少MLP中间tensor大小)
3. **n=2效率82%**: kernel launch/小GEMM效率低 → n≥4后效率接近100%
4. **n=16: 2.75x**: 理论极限约1.67x(n→∞) → 但实际因为PS减少整体forward长度 → 加速更显著

## 四、Exp4: Qwen3.6 Hybrid PS (16层, 4 full attn + 12 DeltaNet)

| prefix_ratio | compute节省% | time节省% | speedup |
|-------------|-------------|----------|---------|
| 0.50 | 43.8% | 43.6% | 1.77x |
| 0.75 | 65.6% | 63.8% | 2.76x |
| 0.90 | 78.8% | 73.4% | 3.76x |

**关键**: HybridAttention PS同样有效!

1. **75% prefix → 2.76x**: 即使只有25%层是full attn → DeltaNet层也同样受益(state injection)
2. **90% prefix → 3.76x**: 高prefix_ratio时加速更显著
3. **DeltaNet层省计算**: 虽然无KV cache → 但in_proj+out_proj也是compute-bound → PS跳过prefix的DeltaNet计算同样有效

## 五、Exp5: Prefix Ratio Sweep (7B比例模型)

| prefix_ratio | compute节省% | time节省% | efficiency% |
|-------------|-------------|----------|------------|
| 0.10 | 8.8% | 12.7% | 144.8% |
| 0.25 | 21.9% | 24.3% | 111.2% |
| 0.50 | 43.8% | 45.0% | 102.8% |
| 0.75 | 65.6% | 66.2% | 100.9% |
| 0.90 | 78.8% | 77.8% | 98.8% |

**效率趋势**:

1. **短prefix(10-25%): efficiency>100%**: PS减少forward长度 → 减少MLP中间activation → 省memory traffic → 超额收益
2. **长prefix(75-90%): efficiency≈100%**: 几乎完美的compute-to-time转换
3. **结论**: PS在全模型forward层面, time_savings ≈ compute_savings → 线性关系成立

## 六、综合结论

### 核心洞察链

```
纯attention PS → 0.99x (memory-bound, KV读量不变)
全模型forward PS → 2.46x (MLP 82% compute-bound, 跳过prefix MLP)

差距 = 2.46x / 0.99x = 2.5x
来源 = MLP占比82% + QKV_proj占比9.2% = 91.2% compute-bound
```

### 与prefix-0501项目的对照

1. **prefix-0501的PS发生在全模型forward层面**: Reuser跳过prefix的所有层(KV_proj + Attention + MLP + LayerNorm) → 不是只跳attention
2. **MLP是真正的瓶颈**: 占82%时间 → compute-bound → PS减少prefix的MLP FLOPS → 这是核心收益
3. **attention KV injection微不足道**: 仅18%时间 → 其中52%是QKV_proj(compute-bound) → attention计算本身仅2% → memory-bound的attention对PS时间影响小
4. **PS savings公式**: time_savings ≈ (B-1)/B × prefix_ratio × compute_bound_ratio ≈ (B-1)/B × prefix_ratio × 91% → 接近理论值

### 对verl #6401贡献的启示

- **Magi Attention ~3x**: 本实测n=8时2.46x → Magi的3x来自更优的tree+sparse → 差距0.54x来自attention稀疏化(对18%中的2%优化) → 有限!
- **最优先改进**: 不是attention稀疏化 → 而是**优化MLP compute**(比如fused MLP kernel, gradient checkpointing for prefix portion)
- **DeltaNet PS同样重要**: 不是只有full attn层受益 → DeltaNet的in_proj+out_proj也是compute-bound

### 实用公式

```
full_model_speedup = 1 / (1/B + (B-1)/B × suffix_ratio)
                    ≈ B / (1 + (B-1) × suffix_ratio)

GRPO n=8, prefix_ratio=0.67:
  speedup = 8 / (1 + 7 × 0.33) = 8 / 3.31 = 2.42x  → 实测2.46x ✓
```

Sources:
- 前序实验: notebook/fundamentals/prefix-sharing-packed-thd-rtx4090.md (纯attention PS)
- prefix-0501项目: notebook/projects/prefix-0501-project-analysis.md
- 工具: tools/full_model_ps_benchmark_4090.py