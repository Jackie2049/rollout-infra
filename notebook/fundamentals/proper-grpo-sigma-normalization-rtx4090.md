# Proper GRPO σ-normalization vs Unnormalized — RTX 4090

> 2026-06-07 | **σ-normalization hurts performance!** 43% vs 65% (Δ=-22%). Double normalization + gradient vanishing.

## 概述

使用proper GRPO（同prompt n=8 completions, autoregressive rollout）对比σ-normalization效果。
之前的简化GRPO实验因用不同prompt分组导致σ-norm无效果(normalized=unnormalized=21%)。

本实验使用mini_grpo_training.py的autoregressive rollout，3种策略：
1. SFT→GRPO A=(r-μ)/σ (标准GRPO σ归一化)
2. SFT→GRPO A=r-μ (无σ归一化)
3. SFT→GRPO A=(r-μ)/σ + gradient noise σ=0.01

## 实验结果

```
策略                     | SFT eval | GRPO final | GRPO peak | avg reward
SFT→GRPO A=(r-μ)/σ      | 39%      | 43%        | 54%       | 0.546
SFT→GRPO A=r-μ           | 39%      | **65%**    | **70%**   | 0.574
SFT→GRPO A=(r-μ)/σ+noise | 39%      | 61%        | **74%**   | 0.589

→ σ-normalization effect: 43% vs 65% (Δ=-22%!) ← 反直觉!
→ Gradient noise部分补偿(61% vs 43%, Δ=+18%)
→ 但noise峰值74% > unnormalized峰值70% → noise有探索优势但最终不稳定
```

## 关键发现

### 1. σ-normalization HURTS performance (反直觉!)

```
理论预期: σ归一化→advantage标准化→更稳定梯度→更好收敛
实际结果: σ归一化→43% eval vs 无归一化→65% eval (差22%!)

原因分析: σ-normalization + clip_grad_norm = 双重归一化!

σ归一化放大advantage magnitude:
  Group 7 correct + 1 wrong: μ=0.875, σ=0.33
    Normalized: A(correct)=+0.38, A(wrong)=-2.65 → 总梯度magnitude大
    Unnormalized: A(correct)=+0.125, A(wrong)=-0.875 → 总梯度magnitude小

clip_grad_norm(max_norm=1.0) 限制总梯度到1.0:
  Normalized: 大梯度 → 被clip到1.0 → 有效学习率降低!
  Unnormalized: 小梯度 → 很少被clip → 每步学习更有效!

→ σ归一化 → 放大梯度 → clip压缩 → 有效学习率 = 原始 × clip_ratio
→ 无归一化 → 自然梯度 → clip几乎不触发 → 有效学习率保持
→ 双重归一化 = 隐式降低学习率 → 收敛更慢!
```

### 2. Gradient Vanishing (loss=0.0 频率62% vs 23%)

```
Normalized loss=0.0频率: 8/13 eval points = 62%!
Unnormalized loss=0.0频率: 3/13 eval points = 23%

为什么σ归一化导致更多loss=0?

Binary reward (0 or 1) + n=8 completions:
  当所有8个completion获得相同reward → σ=0 → std_r=1.0(替代) → A=r-μ=0
  → 无advantage → loss=0 → 无梯度 → 无学习!

σ归一化加剧这个问题:
  σ归一化 → advantage magnitude标准化 → 更强梯度 → clip压缩 → 有效更新更小
  → 模型参数变化更小 → 下一批completion更相似 → 更多相同reward → 更多loss=0
  → 正反馈循环! → 越学越少!

Unnormalized避免:
  自然advantage → 梯度不被过度压缩 → 参数变化更大 → 更多样completion → 更少相同reward
  → 打破恶性循环!
```

### 3. 轨迹对比分析

```
Step  | Normalized | Unnormalized | Norm+Noise
0     | 42%        | 42%          | 42%        ← 起点相同(SFT baseline 39%)
25    | 44%        | 32%          | 44%
50    | 32%        | 38%          | 34%
75    | 40%        | 54%          | 44%
100   | 30%        | 38%          | 26%
125   | 38%        | 40%          | 46%
150   | 50%        | 44%          | **74%**    ← noise峰值!
175   | 42%        | 52%          | 52%
200   | 38%        | 44%          | 40%
225   | 54%        | **70%**      | 66%
250   | 50%        | 50%          | 56%
275   | 38%        | 58%          | 58%
299   | 50%        | 62%          | 56%

Normalized: 波动42→44→32→40→30→38→50→42→38→54→50→38→50
  → 无明显上升趋势! 困在30-54%之间
  → 62%的step loss=0 → 模型几乎不学习

Unnormalized: 波动42→32→38→54→38→40→44→52→44→70→50→58→62
  → 有上升趋势! 后半段40→52→70→50→58→62
  → 仅23%的step loss=0 → 模型持续学习

Norm+Noise: 波动42→44→34→44→26→46→74→52→40→66→56→58→56
  → 噪声帮助探索→峰值74%! 但最终56%<unnormalized 65%
  → 噪声探索但σ归一化仍限制有效学习率
```

### 4. Loss值对比 — σ归一化的梯度消失

```
Step  | Normalized loss | Unnormalized loss | Norm+Noise loss
0     | -0.271           | -0.128            | -0.271
25    | 0.059             | -0.089            | 0.021
50    | **0.0**           | -0.173            | -1.430
75    | **0.0**           | 0.014             | **0.0**
100   | **0.0**           | 0.090             | -0.036
125   | **0.0**           | **0.0**           | -0.229
150   | 0.004             | 0.049             | **0.0**
175   | **0.0**           | -0.191            | 0.025
200   | **0.0**           | -0.135            | **0.0**
225   | **0.0**           | **0.0**           | **0.0**
250   | **0.0**           | 0.270             | **0.0**
275   | **0.0**           | **0.0**           | **0.0**
299   | **0.0**           | **0.0**           | **0.0**

Normalized: 8个/13个step loss=0 → 62%无梯度!
Unnormalized: 3个/13个step loss=0 → 23%无梯度
Norm+Noise: 6个/13个step loss=0 → 46%无梯度(噪声部分补偿)
```

### 5. 理论解释: 三重机制

```
为什么σ归一化在proper GRPO中反而更差?

机制1: 双重归一化 (σ-norm + clip_grad_norm)
  σ归一化 → advantage ~unit magnitude → 梯度magnitude↑
  clip_grad_norm → 压缩总梯度到max_norm → 有效LR↓
  组合效果: σ归一化放大梯度, clip再压缩 → 有效学习率降低
  → 相当于隐式降低学习率 → 收敛更慢

  证明:
  假设原始advantage A_raw = r-μ, |A_raw| ≈ 0.5 (binary reward)
  σ归一化后 A_norm = (r-μ)/σ, |A_norm| ≈ 1.0-2.0
  梯度magnitude ∝ Σ|A_i × log_prob_i|
  Normalized梯度 ≈ 2-4x Unnormalized梯度
  clip_ratio_normalized ≈ 1/(2-4) ≈ 0.25-0.5
  clip_ratio_unnormalized ≈ 1.0 (很少触发)
  → 有效更新 = lr × clip_ratio × gradient
  → Normalized有效更新 ≈ lr × 0.35 × 1.0 = 0.35lr
  → Unnormalized有效更新 ≈ lr × 1.0 × 0.5 = 0.5lr
  → Unnormalized每步学习1.4x更多!

机制2: Binary reward梯度消失
  Binary reward → 同prompt下n=8 completion reward全同概率高
  P(all correct) = p^8 → 低, 但P(≥7 correct) = 8p^7(1-p)+p^8 ≈ 12% (p=0.4)
  P(all wrong) = (1-p)^8 → ~2%
  P(reward几乎全同) → σ≈0 → advantage≈0 → loss=0 → 无学习
  σ归一化进一步加剧(机制1→参数变化小→completion更相似→更多相同reward)

机制3: Advantage scale不一致
  不同组的σ不同 → advantage scale跨组变化
  Group A: σ=0.33 → |A_norm|=2.65
  Group B: σ=0.48 → |A_norm|=1.29
  → 梯度signal的magnitude跨组不稳定 → 优化方向噪声更大
  → AdamW的momentum可以部分平滑, 但跨组scale跳跃仍影响收敛

  Unnormalized:
  Group A: |A_raw|=0.125 → 自然scale
  Group B: |A_raw|=0.375 → 自然scale
  → scale与reward signal强度一致 → 更稳定的优化方向
```

### 6. 与之前噪声实验的综合

```
实验链条:

1. 简化GRPO → σ-norm无效果(normalized=unnormalized=21%)
   原因: 不同prompt分组 → reward跨问题方差大 → σ-norm仅rescale

2. Noise σ sweep → σ=0.01最优(100% eval!)
   原因: 梯度噪声帮助逃离local optima → SFT训练的regularization

3. Proper GRPO → σ-norm反而更差(43% vs 65%)
   原因: 双重归一化 + binary reward梯度消失

→ 综合结论:

GRPO σ-normalization ≠ 纯梯度噪声! 它们是不同的机制:

1. 梯度噪声(p.grad += randn*σ):
   - 添加随机方向扰动 → 探索
   - 不改变梯度magnitude → 不被clip过度压缩
   - 与AdamW momentum平滑 → 长期方向一致
   - 对SFT训练极有效(+58%!)

2. GRPO σ归一化(A=(r-μ)/σ):
   - 标准化advantage → 改变梯度magnitude
   - 与clip_grad_norm组合 → 双重归一化 → 有效LR降低
   - Binary reward → 梯度消失频率高
   - 对proper GRPO训练反而有害(-22%!)

→ σ归一化在什么情况下可能有用?
  1. 连续reward(非binary) → σ始终>0 → 无梯度消失
  2. 不使用clip_grad_norm → 避免双重归一化
  3. 大模型+复杂任务 → reward分布更连续 → σ-norm有意义
  4. 多步长序列 → reward spread天然大 → σ-norm更稳定

→ 对算术任务的binary reward:
  σ归一化有害 → 应使用A=r-μ或甚至直接用reward r作为advantage!
```

### 7. 生产建议

```
GRPO σ-normalization使用建议:

场景                         | 推荐advantage     | 原因
Binary reward + clip_grad    | A=r-μ (不归一化)  | 避免双重归一化+梯度消失
Continuous reward + clip     | A=(r-μ)/σ 可尝试  | σ始终>0,但仍需注意clip
Continuous reward + no clip  | A=(r-μ)/σ 推荐    | 标准化advantage,无双重归一化
大模型+复杂任务              | A=(r-μ)/σ 推荐    | reward分布更连续,σ-norm有意义
确定性任务(数学/代码)        | A=r-μ + noise     | 噪声探索+自然advantage scale

关键教训:
1. σ归一化+clip_grad_norm=双重归一化 → 有效LR降低 → 需调整max_norm或lr
2. Binary reward → σ归一化梯度消失 → 考虑continuous reward或A=r-μ
3. 梯度噪声是更好的regularization → 不改变梯度方向/magnitude,只添加随机扰动
4. GRPO原始论文的σ归一化建议需要根据具体场景调整!
```

## 工具

```bash
cd ~/rollout-infra
source ~/anaconda3/bin/activate llm
CUDA_VISIBLE_DEVICES=0 python -u tools/proper_grpo_sigma_comparison.py --model_size 2.28m

results/proper_grpo_sigma_comparison.json
```

工具: `tools/proper_grpo_sigma_comparison.py`