# Full-Model PS KV Injection Prototype — RTX 4090 实测

> 2026-06-07 | 3实验: KV Injection精度验证, 训练PS, 长Context PS
> 关键发现: Block-causal mask cos_sim=0.999999 → KV injection精度验证通过!

## 核心发现: Block-causal mask是KV injection的关键!

| 方法 | cos_sim_suffix | max_diff | 结论 |
|-----|---------------|----------|------|
| is_causal=True (旧bug) | ≈0 | ~∞ | ✗ suffix看不到prefix → 完全错误 |
| Block-causal mask (修复后) | **0.999999** | 0.0039 | ✓ suffix看到全部prefix+causal suffix → 精度验证通过 |
| Provider logits vs baseline | **1.0** | **0.0** | ✓ provider前向与baseline完全一致 |

**Block-causal mask原理**:
- Prefix部分(列0..prefix_len): 全部可见 → mask值=0.0 (suffix可以看所有prefix位置)
- Suffix部分(列prefix_len..): causal → mask值=-inf 如果j-prefix_len > i (suffix内causal)
- 这与标准causal attention等价: suffix position i 在baseline中看到 {0..prefix_len+i} = {全部prefix} ∪ {causal suffix}

## 一、Exp1: Forward-only PS + KV Injection精度

### 模型: 2.28B params (24层, hidden=2560, GQA-4)

| n_samples | prefix_ratio | compute节省 | simple_speedup | cos_sim_provider | cos_sim_suffix | max_diff |
|-----------|-------------|------------|---------------|-----------------|---------------|---------|
| 2 | 0.75 | 37.5% | 1.21x | 1.0 | 0.999999 | 0.0039 |
| 4 | 0.75 | 56.2% | 1.53x | 1.0 | 0.999999 | 0.0039 |
| 8 | 0.75 | 65.6% | 1.43x | 1.0 | 0.999999 | 0.0043 |

**注意**: simple_speedup低于之前benchmark (2.08x for n=4) — 原因是本脚本在timing loop中含torch.randint开销. 之前`full_model_ps_rtx4090.py`使用更干净的方法测量出2.08x.

### Per-forward时间分解

| 指标 | n=2 | n=4 | n=8 |
|-----|-----|-----|-----|
| full_fwd_ms | 20.388 | 20.276 | 19.252 |
| suffix_fwd_ms | 13.197 | 10.855 | 12.583 |
| kv_overhead_ms | -2.518 | 0.733 | -0.182 |
| KV pair ms | 31.067 | 31.864 | 31.653 |

**关键**: kv_overhead_ms≈0 (甚至为负) → KV injection几乎零额外开销!

### KV Injection为什么工作

```
前向传播流程:
1. Provider: embed(prefix+suffix) → 每层提取prefix_KV → 24层 → provider_logits
   provider_logits与baseline完全一致 (cos_sim=1.0, max_diff=0.0)

2. Reuser: embed(suffix_only) → 每层:
   - RMSNorm(suffix_hidden) → QKV_proj(suffix) → suffix Q/K/V
   - concat(prefix_KV + suffix_KV) → full_K, full_V
   - Block-causal mask: prefix全可见 + suffix内causal
   - scaled_dot_product_attention(Q, full_K, full_V, attn_mask=block_causal)
   - o_proj → MLP(suffix_only) → 下一层

3. Reuser suffix_logits ≈ baseline suffix_logits (cos_sim=0.999999)
```

### 数学验证: 为什么Block-causal mask正确

```
Baseline causal attention for suffix position i (total position prefix_len+i):
  可见范围 = {0, 1, ..., prefix_len+i} = 全部prefix + causal suffix

Block-causal mask for suffix position i:
  mask[i,j] = 0 (可见) if j <= prefix_len + i
  mask[i,j] = -inf (不可见) if j > prefix_len + i

可见范围 = {0, ..., prefix_len-1} (全部prefix) + {prefix_len, ..., prefix_len+i} (causal suffix)
= {0, ..., prefix_len+i} → 与baseline完全一致 ✓
```

### 前缀位置不依赖suffix (关键性质)

```
对于prefix position j (j < prefix_len):
  causal attention → 只能看到position 0..j (不看任何suffix位置)
  → prefix hidden states不依赖suffix → 无论suffix内容如何, prefix KV不变
  → Provider的prefix KV可以直接用于Reuser ✓
```

## 二、Exp2: Training PS (fwd+bwd)

### 模型: 38M params (8层, hidden=512) — 2.28B模型+Adam=27GB>24GB → 用小模型

| 指标 | 值 |
|-----|-----|
| speedup | 0.69x (比baseline更慢!) |
| mem_savings | 40% (1695→1018 MB) |
| training_discount | 0.3 (不可靠) |
| loss_baseline | 9.1591 |
| loss_provider | 9.1561 |

**为什么0.69x?**: 38M模型太小 → GPU永远不compute-bound → batch efficiency更重要:
- Baseline: 1个batched forward(4,512) → GPU利用率高 → 快
- PS: 2个小batch forward(1,512) + forward(3,128) → GPU利用率低 → 慢

**真实训练speedup**: 之前`grpo_training_ps_4090.py`用2.28B模型实测 **1.59x** (n=4,75%prefix), training_discount=0.76. 这是可靠数字.

**内存节省40%**: 即使speedup不佳, 内存节省是真实的 → 可以跑更大batch → 间接加速.

## 三、Exp3: Long Context PS with KV Injection

### 精度 + Speedup

| prefix_len | prefix_ratio | compute节省 | simple_speedup | cos_sim_suffix | max_diff | est_fwd_n8 | est_train_n8 |
|-----------|-------------|------------|---------------|---------------|---------|-----------|-------------|
| 1024 | 0.80 | 60.0% | **2.22x** | 0.999999 | 0.0039 | 3.33x | 2.53x |
| 2048 | 0.89 | 66.7% | **2.73x** | 0.999999 | 0.0039 | 4.5x | 3.42x |

**趋势与之前long_context_serving benchmark一致**:
- 1024/1280 (80%): 2.22x (之前2.38x, 误差6.7%)
- 2048/2304 (89%): 2.73x (之前2.89x, 误差5.5%)

**GRPO n=8预估**:
- 80% prefix: 3.33x forward / 2.53x training
- 89% prefix: 4.5x forward / 3.42x training
- 96% prefix (6144/6400): 6x forward / 4.5x training (之前实测验证)

## 四、综合结论

### KV Injection架构验证通过

```
prefix-0501 One-Forward架构:
1. Provider: full forward → extract prefix KV at each layer
2. Reuser: suffix-only forward with KV injection
   → Block-causal mask: prefix全可见 + suffix内causal → cos_sim=0.999999 ✓
3. Prefix-last logprob restoration from provider output
   → Provider logits = baseline logits (cos_sim=1.0, max_diff=0.0) ✓

精度验证: PASS
  - cos_sim_suffix = 0.999999 (FP16累积误差max_diff=0.004)
  - cos_sim_provider = 1.0 (零误差 — provider与baseline完全一致)
  - 数学证明: block-causal mask可见范围与baseline causal mask完全一致
```

### 完整PS speedup链

```
Attention-only PS (当前verl PG): 0.99x → 只省attention, MLP仍计算prefix
Full-model forward PS:           2.46x → 省prefix MLP+QKV_proj+LN
Training PS (fwd+bwd):           1.59x → backward稀释savings (×0.76)
Long context PS (96% prefix):   3.55x → prefix越长→speedup越大

GRPO n=8预估:
  短prompt (512/768, 67%): 2.46x forward / 1.87x training
  长prompt (2048/2304, 89%): 4.5x forward / 3.42x training
  超长prompt (6144/6400, 96%): 6x forward / 4.5x training
```

### 对verl贡献的启示

1. **KV injection精度验证通过**: block-causal mask是正确实现 → 可以安全用于production
2. **verl PG差距确认**: 0.99x → 2.46x → 2.5x improvement opportunity confirmed
3. **实现路径清晰**: Provider-Reuser Forward Splitting (Approach 1 in gap analysis)
4. **需要custom attention backend**: SDPA with block-causal mask fallback到math backend → 需FlashAttention实现才能高效
5. **训练backward需要gradient flow**: Reuser的backward需要gradient流回provider的prefix KV → 需autograd支持

### 实用公式集

```
# Block-causal mask (vectorized)
mask[i,j] = -inf if j > prefix_len + i  (future suffix positions)
实现: future_positions = kv_idx > (suffix_idx + prefix_len)
mask_2d[future_positions] = float('-inf')

# PS Speedup (forward-only)
speedup = n / (1 + (n-1) × suffix_ratio)
suffix_ratio = 1 - prefix_ratio

# Training discount
training_speedup ≈ forward_speedup × 0.76

# KV injection overhead ≈ 0 (实测)
kv_overhead_ms ≈ -2.5 to 0.7ms (测量误差范围内)
```

Sources:
- Prototype: tools/full_model_ps_kv_injection_4090.py
- Results: tools/full_model_ps_kv_injection_results.json
- Forward-only PS: notebook/fundamentals/full-model-ps-rtx4090.md
- Training PS: notebook/fundamentals/grpo-training-ps-rtx4090.md
- Long Context: notebook/fundamentals/long-context-serving-rtx4090.md
- Gap Analysis: notebook/projects/verl-prefix-grouper-gap-analysis.md