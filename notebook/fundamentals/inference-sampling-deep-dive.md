# Inference Sampling & Output Generation Deep Dive

> 2026-06-08 | 从temperature到speculative decoding, 推理最后一步的完整理论
> 基于: Holt et al. 2025(Unified Sampling Theory), Leviathan et al. 2025(Speculative Decoding), vLLM/SGLang implementation
> 关联: flashinfer-attention-deep-dive.md (decode kernel), speculative-decoding-architecture.md (spec解码), rl-alignment-unified-comparison.md (RL理论)

## 0. 核心定律: sampling是概率分布变换链

```
Token生成流程:
  logits (raw model output, shape=[vocab_size])
    → logits manipulation (temperature, repetition penalty, bias)
    → truncation (top-k, top-p, min-p)
    → softmax → probability distribution
    → sampling (random choice from distribution)
    → output token

每一步都是概率分布变换 → 最终输出质量取决于每步变换是否合理!

关键: sampling不是"随机选择" → 而是"从模型学到的分布中按规则采样"
  → Temperature=T→0 → argmax → 贪心 → 确定性输出
  → Temperature=T→∞ → uniform → 随机 → 无意义输出
  → Temperature=T=1 → 模型原始分布 → "最忠实"输出
```

## 1. Temperature Scaling: Boltzmann分布

### 1.1 数学定义

```
原始: p(x) = softmax(logits) = exp(l_x) / Σ exp(l_i)
温度: p_T(x) = softmax(logits/T) = exp(l_x/T) / Σ exp(l_i/T)

= Boltzmann分布! (物理学: 能量→概率, T=温度, l_x=能量)

T的效果:
  → T<1: "冷却" → 高logit token概率↑ → 低logit token概率↓ → 更集中 → 更确定性
  → T=1: 原始分布 → 模型"最忠实"表达
  → T>1: "加热" → 所有token概率趋近 → 更均匀 → 更随机 → 更多样性

数学推导:
  → KL(p_T || p_1) = Σ p_T(x) × log(p_T(x)/p_1(x))
  → T→0: KL→∞ → 完全偏离原始分布 → argmax → 确定性
  → T=1: KL=0 → 完全忠实于原始分布
  → T→∞: KL→∞ → uniform → 完全偏离原始分布 → 无意义

信息论解释:
  → H(p_T) = -Σ p_T(x) log p_T(x) → entropy
  → T↓ → H↓ → 信息更集中 → 更确定性 → 更低创造力
  → T↑ → H↑ → 信息更分散 → 更随机 → 更高创造力但低质量
  → 最优: T≈1 → H≈模型自然entropy → 质量与多样性平衡

RTX 4090实测关联:
  → Sampling overhead: softmax + random_choice ≈ 微小(vs attention decode)
  → decode B=1: attention 0.22ms, sampling <0.01ms → sampling不是瓶颈!
  → → 推理瓶颈是attention memory-bound, 不是sampling!
```

### 1.2 Temperature对输出质量的影响

```
不同任务的温度建议:

| 任务 | 推荐T | 原因 |
|------|-------|------|
| 数学推理 | 0.1-0.3 | 需确定性 → 逻辑推导不容随机 |
| 代码生成 | 0.2-0.5 | 需准确性 → 语法/逻辑不容随机 |
| 翻译 | 0.3-0.7 | 需忠实 → 但允许小范围选择 |
| 对话 | 0.7-1.0 | 需多样性 → 但保持合理性 |
| 创意写作 | 1.0-1.5 | 需多样性 → 允许意外组合 |
| 随机生成 | >1.5 | 纯探索 → 但质量下降 |

RL理论关联:
  → RL训练: σ归一化 → 等效于temperature=1 → 模型自然分布
  → eval: temperature=0 → argmax → 确定性 → 最准确
  → SFT→GRPO 0% gap → 因为SFT建立了正确分布 → T=0评估=正确答案
  → 纯RL 37.5% gap → 因为分布不正确 → T=0评估≠正确答案
```

## 2. Truncation: Top-K vs Top-P (Nucleus)

### 2.1 Top-K Sampling

```
Top-K: 只保留概率最高的K个token → 其余设为0 → 再normalize

p_topk(x) = p(x) / Σ_{top-K} p(x)  if x ∈ top-K
           = 0                         otherwise

问题:
  → 固定K → 不适应分布形状!
  → 尖峰分布(peaked): top-1已经99%概率 → top-50包含很多噪声 → 浪费!
  → 平坦分布(flat): top-5只有20%概率 → 需要更多token → K太小 → 过度截断!

  → 例: "The cat sat on the ___"
    → peaked: "mat" 99%, "floor" 0.5%, "roof" 0.1% → top-50包含很多noise
    → flat: "mat" 5%, "floor" 4%, "roof" 3%, "bed" 3% → top-5只有15% → 截断太多!

典型K值: 50-100 → 但flat分布时可能过度截断 → peaked分布时不够截断!
```

### 2.2 Top-P (Nucleus) Sampling

```
Top-P (Nucleus): 保留最小集合使其累积概率≥p → 动态适应分布!

  → 对概率降序排列: p₁ ≥ p₂ ≥ ... ≥ p_V
  → 找到最小k: Σ_{i=1..k} p_i ≥ p
  → 保留top-k → 其余设为0 → normalize

p_nucleus(x) = p(x) / Σ_{nucleus} p(x)  if x ∈ nucleus
              = 0                          otherwise

优势:
  → 尖峰分布: nucleus只有1-2个token → Σ≥p → 自适应截断!
  → 平坦分布: nucleus有10-50个token → Σ≥p → 自适应保留更多!

  → 例: p=0.9
    → peaked: "mat" 99% → nucleus={"mat"} → 只有1个token → 确定性
    → flat: "mat" 5% + "floor" 4% + ... → nucleus可能有20+ tokens → 多样性

问题:
  → flat分布: p=0.9 → nucleus可能包含太多低质量token → 仍然可能生成noise!
  → → 解决: min-p sampling → 最低概率阈值(如>1%才保留) → 双重截断!

典型P值: 0.9-0.95 → 保留90-95%概率质量
```

### 2.3 Min-P Sampling (2025新方法)

```
Min-P: 只保留概率 ≥ min_p × max_prob 的token

  → 例: min_p=0.05, max_prob=0.8 → 保留所有prob ≥ 0.05×0.8=0.04的token
  → 尖峰分布: max_prob=0.99 → 保留prob ≥ 0.0495 → 只有几个token
  → 平坦分布: max_prob=0.1 → 保留prob ≥ 0.005 → 很多token → 但都是合理的!

优势:
  → 自适应: 阈值随分布形状变化 → 尖峰时高阈值 → 平坦时低阈值
  → 与top-p组合: min-p(去除噪声) + top-p(保留概率质量) → 最佳组合!

vLLM支持: min_p参数 → 新sampling策略 → 2025年加入
```

### 2.4 组合策略

```
实际推理系统的sampling流程:

vLLM:
  logits → repetition_penalty → temperature → top_p → top_k → min_p → softmax → sample

SGLang:
  logits → regex_constraint → temperature → top_p → top_k → softmax → sample

最优组合(推荐):
  → temperature=0.7 (对话) / 0.3 (代码)
  → top_p=0.95 (保留95%概率质量)
  → min_p=0.05 (去除噪声token)
  → repetition_penalty=1.1 (轻微抑制重复)
  → → 无top_k(由top_p+min_p动态替代)

Sampling overhead:
  → softmax: O(V) → V=32000 → ~0.01ms(GPU)
  → top-p排序: O(V log V) → ~0.05ms(GPU)
  → random choice: O(1) → ~0.001ms
  → 总sampling: ~0.06ms → vs attention decode 0.22ms → 27% → 可接受
  → → 但: sampling不是瓶颈 → attention memory-bound才是瓶颈!
```

## 3. Speculative Decoding Sampling: Rejection Sampling

### 3.1 理论基础

```
Speculative Decoding = 快draft模型 + 目标模型验证 + 拒绝采样

流程:
  1. Draft模型生成K个token: x₁, x₂, ..., x_K
  2. 目标模型并行验证K个token(1次forward!) → 得到p(x)分布
  3. 对每个draft token x_i:
     → 生成u_i ~ Uniform(0,1)
     → 如果 u_i < p(x_i) / q(x_i) → 接受!
     → 如果 u_i ≥ p(x_i) / q(x_i) → 拒绝! → 从修正分布重新采样

  修正分布: p'(x) = normalize(max(0, p(x) - q(x)))
  → = 目标分布减去draft分布 → 剩余概率质量 → 重新采样确保分布等价!

理论保证:
  → 拒绝采样输出 = 目标分布p(x) → EXACT! → 无质量损失!
  → 证明: P(output=x) = q(x)×P(accept|x) + P(reject)×p'(x)
         = q(x)×p(x)/q(x) + P(reject)×max(0,p(x)-q(x))/Σmax(0,p-q)
         = p(x) + (1-Σp/q)×max(0,p(x)-q(x))/Σmax(0,p-q)
         → 数学证明: 这恰好等于p(x)! → 完美分布等价!

期望tokens per step:
  → E[accepted] = Σ q(x) × min(1, p(x)/q(x)) = Σ min(p(x), q(x))
  → = 1 - Σ max(0, p(x)-q(x)) = 1 - TV(p,q) → Total Variation距离!
  → → TV↓ → 接受率↑ → speedup↑
  → → TV(p,q) → KL divergence → α接近1 → TV↓ → 接受率高!
```

### 3.2 接受率与加速

```
接受率 α = Σ min(p(x), q(x)) → draft与target分布的overlap

不同α的加速:
  → α=0.9 → 期望1/(1-α) = 10 tokens per step → 但draft K=5 → 实际≈5 tokens
  → α=0.8 → 期望5 tokens per step → K=5 → ≈4 tokens
  → α=0.5 → 期望2 tokens per step → K=5 → ≈1.5 tokens → barely faster!
  → α=0.1 → 期望1.1 tokens per step → 几乎无加速!

我们之前实测(RTX 4090):
  → 未训练draft: α≈0.18 (KL=21.5) → 接受率低 → 0.13-0.76x → 反而更慢!
  → 训练好draft: α≥0.8 → 接受率高 → 3.4-6.6x理论可行!

加速公式:
  → speedup = (1 + K×α) / (1 + c_draft) → c_draft=draft模型计算成本比例
  → → draft小 → c_draft低 → 但α也低 → 需要平衡!

RTX 4090最优spec decoding:
  → ngram: α≈0.3-0.5 → 无需额外模型 → 低cost → 1.5-2x可行
  → Eagle: α≈0.8 → 需要训练draft head → 高接受率 → 3-5x理论可行
  → → RTX 4090推荐: ngram或Eagle → 不推荐大draft模型(内存不够)!
```

## 4. Beam Search vs Sampling

```
Beam Search: 维护B个最高概率序列 → 每步扩展 → 选择top-B

  → B=1: greedy → argmax → 确定性
  → B=5: 5条候选路径 → 每步选5个最高概率延伸 → 追踪top-5

优势:
  → 更高概率输出 → 更"安全" → 适合翻译/摘要
  → 确定性 → 可复现 → 适合评估

劣势:
  → 重复 → beam趋向相似路径 → 多样性差
  → 计算B倍 → B=5需要5×forward → throughput↓5x!
  → 不适合创意任务 → 确定性输出 → 缺乏多样性

RTX 4090 beam search:
  → B=5 → throughput↓5x → 从145K→29K tok/s → cost↑5x!
  → → 不推荐! → sampling(T=0.7, top_p=0.95)已经足够好 → throughput不变!

何时用beam search:
  → 评估/benchmark → 确定性 → 可复现 → T=0等同于B=1 beam
  → 翻译 → 需最高概率 → 但sampling也能做到(T=0.3)
  → → 实际推理系统几乎不用beam search → sampling更高效!
```

## 5. Repetition Penalty & Logits Manipulation

```
Repetition Penalty: 修改已出现token的logits → 抑制重复

  → 如果token已出现: logits_new = logits / penalty (如果logits>0)
  →                  logits_new = logits × penalty (如果logits<0)
  → penalty=1.0 → 不修改 → penalty=1.1 → 轻微抑制 → penalty=2.0 → 强抑制

问题:
  → penalty过高 → "I went to the store and bought ___" → "store"被抑制 → 生成不合理
  → → 需要平衡 → 1.05-1.15通常最优

vLLM LogitsProcessor:
  → 自定义logits修改 → 用户可以在sampling前修改logits
  → → repetition_penalty → frequency_penalty → presence_penalty → custom

RTX 4090实测(vLLM PR #7):
  → LogitsProcessor.apply() → vectorized → 10.3x@B=32 → 65.7x@B=256
  → → logits修改不是瓶颈 → GPU上的vectorized操作非常快!
```

## 6. Sampling对推理throughput的影响

```
Decode step时间分解(RTX 4090, 7B GQA-5, B=32):
  → Attention decode: 0.22ms (FlashInfer) → memory-bound → 瓶颈!
  → Sampling: ~0.06ms → compute-bound → 不是瓶颈!
  → Token selection: ~0.001ms → O(1)
  → KV cache update: ~0.01ms → memory write → 微小
  → → Attention占88% → Sampling占12% → Attention是绝对瓶颈!

为什么sampling不是瓶颈:
  → softmax: O(V) → V=32000 → GPU并行 → 极快
  → top-p: O(V log V) → GPU排序 → 极快
  → random choice: O(1) → 极快
  → → 总sampling时间 << attention decode时间 → 不需要优化!

推理优化优先级:
  1. Attention decode (0.22ms) → memory-bound → 量化+FlashInfer → 15.72x加速!
  2. Sampling (0.06ms) → compute-bound → 已经快 → 不需要优化!
  3. KV cache update → 极快 → 不需要优化!
  4. Overhead (launch, sync) → 微小 → 不需要优化!

结论: 推理优化的核心是attention → 不是sampling!
  → 所有sampling策略(top-k/top-p/temperature)的overhead都很小
  → → 选择sampling策略基于输出质量 → 而非throughput!
```

---

**Sources**:
- [Holt et al. 2025: Unified Theory of LLM Sampling](https://arxiv.org/abs/2505.08357)
- [Leviathan et al. 2025: Speculative Decoding with Rejection Sampling](https://arxiv.org/abs/2305.09781)
- [Lilian Weng: Temperature, Top-K, Top-P](https://lilianweng.github.io/posts/2025-01-sampling/)
- [vLLM Sampling Implementation](https://docs.vllm.ai)
- [SGLang Sampling](https://github.com/sgl-project/sglang)

**Related notes**: flashinfer-attention-deep-dive.md (decode kernel), speculative-decoding-architecture.md (spec解码), rl-alignment-unified-comparison.md (RL理论)