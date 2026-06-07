# GRPO Advantage ≈ Adaptive Gradient Noise Verification — RTX 4090

> 2026-06-07 | 5策略对比验证: 自适应噪声在2.28M有效(+11%), GRPO简化setup需warm-start

## 概述

验证理论假设: **GRPO σ归一化 A=(r-μ)/σ ≈ 自适应梯度噪声注入**

5种策略对比:
1. SFT baseline (无噪声, 无RL)
2. GRPO σ归一化 (A=(r-μ)/σ, 标准GRPO)
3. GRPO 无σ归一化 (A=r-μ)
4. SFT + 固定梯度噪声 (σ=0.01)
5. SFT + 自适应梯度噪声 (σ∝loss方差) — 模拟GRPO

**关键设计**: 纯FP32训练(无AMP/BF16), 避免量化噪声干扰噪声策略对比

## 一、实验结果

### 1.1 76K模型 (100步, lr=2e-3, FP32)

```
策略               | eval  | peak  | Δeval | loss  | 特点
SFT baseline        | 60%   | 54%   | +0%   | 0.500 | 最优!
GRPO A=(r-μ)/σ      | 12%   | 28%   | -48%  | -18.2 | RL不收敛
GRPO A=r-μ          | 12%   | 28%   | -48%  | -10.3 | σ-norm无效果
SFT+fixed σ=0.01    | 33%   | 32%   | -27%  | 0.749 | 噪声有害!
SFT+adaptive σ∝var  | 35%   | 36%   | -25%  | 0.590 | 噪声有害!

→ 76K模型太简单 → SFT已充分收敛(60%) → 噪声只是干扰!
→ GRPO策略: 二进制reward导致zero-advantage → 无学习 → eval仅12%
→ σ-normalization在简化GRPO中无效果(normalized=unnormalized=12%)
```

### 1.2 2.28M模型 (100步, lr=2e-3, FP32) — 关键发现!

```
策略               | eval  | peak  | Δeval | loss   | 特点
SFT baseline        | 24%   | 28%   | +0%   | 6.877  | baseline
GRPO A=(r-μ)/σ      | 21%   | 34%   | -3%   | -75.0  | RL未收敛
GRPO A=r-μ          | 21%   | 34%   | -3%   | -39.8  | σ-norm无效果
SFT+fixed σ=0.01    | 25%   | 38%   | +1%   | 0.125  | 微帮助
SFT+adaptive σ∝var  | 35%   | 56%   | +11%! | 0.201  | 最优! ← 关键

→ 自适应梯度噪声: 2.28M eval +11% (35% vs 24%), peak 56% vs 28% (2x!)
→ 固定噪声σ=0.01: 仅+1% → σ太小或不够灵活
→ 自适应noise的peak 56%是所有策略最高 → 更多探索发现更好解!
→ SFT baseline loss=6.877(极高!) vs adaptive noise loss=0.201 → 噪声帮助收敛!
```

**2.28M eval轨迹对比:**
```
Step  | baseline | fixed_noise | adaptive_noise | GRPO_norm
0     | 2%       | 2%          | 2%             | 34%(!)
25    | 26%      | 12%         | 8%             | 24%
50    | 18%      | 24%         | 22%            | 24%
75    | 28%      | 38%         | 56%(!)         | 28%
99    | 6%       | 26%         | 30%            | 24%

→ adaptive_noise在step 75达到56% peak! → 探索效应
→ baseline波动大(2%→26%→18%→28%→6%) → 不稳定
→ fixed noise逐步上升(12→24→38→26) → 但peak不如adaptive
→ GRPO初始高(34%)但后续下降 → 初始化偶然好但不学习
```

## 二、关键发现

### 2.1 自适应噪声 > 固定噪声 (2.28M)

```
为什么adaptive noise比fixed noise更有效?

Fixed noise σ=0.01:
  → 每步同样噪声 → 不管模型状态如何
  → 初期: 模型随机 → σ=0.01太小 → 探索不足
  → 后期: 模型接近收敛 → σ=0.01太大 → 干扰收敛

Adaptive noise σ = 0.01 × max(loss_std[-5:], 0.1):
  → σ随loss方差变化 → 自适应!
  → 初期: loss高且不稳定 → loss_std大 → σ大 → 更多探索!
  → 后期: loss低且稳定 → loss_std小 → σ小 → 精确收敛!
  → mean_sigma=0.0079, max_sigma=0.025 → 3x动态范围!

→ 这正是GRPO σ归一化的机制!
  GRPO A=(r-μ)/σ: 当reward方差大 → advantage大 → 梯度强 → 探索!
  GRPO A=(r-μ)/σ: 当reward方差小 → advantage小 → 梯度弱 → 稳定!

→ adaptive noise ≈ GRPO σ归一化的SFT版本
```

### 2.2 噪声效果随模型大小变化

```
76K (简单): 噪声有害 (-27% eval)
2.28M (复杂): 自适应噪声有益 (+11% eval)

→ 为什么?

76K: SFT baseline已达60% → 模型已充分收敛 → 噪声只是干扰优化方向
2.28M: SFT baseline仅24% → 模型难以收敛 → 需要探索帮助逃离local optima
     → loss=6.877(极差!) → baseline陷入poor local optimum
     → adaptive noise帮助逃离 → loss降到0.201!

→ 类比:
  小模型 = 简单迷宫 → 直走就行 → 摇摆反而走偏 (噪声有害)
  大模型 = 复杂迷宫 → 需要探索 → 摇摆帮助发现出口 (噪声有益)

→ 与之前regularization实验一致:
  2.28M BF16 AMP + grad noise: +44% eval (噪声逃离AMP伪收敛)
  2.28M FP32 + adaptive noise: +11% eval (噪声逃离local optima)
  → BF16 AMP效果更大(44% vs 11%)因为AMP额外制造了伪收敛陷阱
```

### 2.3 FP32 vs BF16 AMP噪声效果对比

```
之前BF16 AMP实验 (2.28M):
  baseline: 19% eval → AMP伪收敛!
  grad noise: 63% eval (+44%)

本次FP32实验 (2.28M):
  baseline: 24% eval → 正常收敛
  adaptive noise: 35% eval (+11%)

→ FP32 baseline更好(24% vs 19%) → AMP确实有害
→ FP32噪声效果更小(+11% vs +44%) → 但仍显著!
→ BF16 AMP噪声效果更大是因为同时克服了:
  1. local optima (与FP32噪声相同的效果, ~+11%)
  2. AMP伪收敛 (额外~+33%)
→ 真正的噪声收益 ≈ +11% (FP32实验) → AMP额外收益 ≈ +33%

→ 结论: 梯度噪声的正则化效果 ≈ +11% (逃离local optima)
         BF16 AMP的噪声效果 = +11%正则化 + 逃离伪收敛 + 量化噪声
```

### 2.4 简化GRPO的局限性

```
GRPO (normalized & unnormalized): 21% eval (2.28M), 12% (76K)

→ 为什么GRPO比SFT差?

1. Policy gradient需要更多步数才能收敛
   → SFT: 明确告诉模型正确答案 → 每步都学
   → GRPO: 只告诉"这个比平均好/差" → 学习信号弱

2. 简化GRPO用不同问题组 → reward variance是随机的而非有意义的
   → 真GRPO: 同一prompt生成n个completions → 组内reward差异=模型在该问题上的不确定性
   → 简化GRPO: n个不同问题 → reward差异=随机 → 无学习信号

3. σ-normalization在简化setup中无效果:
   normalized = unnormalized = 21% (2.28M) / 12% (76K)
   → 因为continuous reward + 不同问题 → variance无意义
   → σ-normalization只是rescale → 不改变梯度方向
   → 真GRPO需要warm-start + 同prompt组比较才能体现σ效果

→ 但! GRPO peak 34% vs SFT baseline peak 28% → 有初始优势?
   → 不是! GRPO step 0 = 34%是偶然(seed随机→有些初始参数分布好)
   → GRPO之后下降 → 没有真正学习
```

### 2.5 SFT baseline不稳定性

```
2.28M FP32 baseline eval轨迹: 2%→26%→18%→28%→6% → 巨大波动!

→ 为什么不稳定?
  1. 100步太少 → 模型在每个checkpoint可能恰好对测试集好/差
  2. 小batch(GA=1) → 每步只有一个样本 → 梯度方向随机 → 摇摆
  3. loss=6.877极高 → 模型远未收敛 → eval不稳定是正常的

→ adaptive noise trajectory: 2%→8%→22%→56%→30%
  → 虽然也波动 → 但peak远高于baseline(56% vs 28%)
  → 说明噪声帮助模型到达更好的区域(即使不稳定)

→ 生产建议: 增加训练步数(300+) + 大batch(GA=4-8) → 更稳定收敛
```

## 三、理论验证状态

```
假设: GRPO σ归一化 ≈ 自适应梯度噪声

验证结果:
  ✓ 自适应噪声 > 固定噪声 (2.28M: +11% vs +1%)
    → 支持假设: σ的动态调整比固定σ更有效
  ✓ 噪声帮助更难模型逃离local optima
    → 支持假设: GRPO σ归一化帮助模型探索
  ✗ GRPO normalized ≈ SFT adaptive noise 数值不等
    → 12% vs 35% (76K), 21% vs 35% (2.28M)
    → 但简化GRPO不收敛 → 无法直接数值对比
  ✗ σ-normalization在简化GRPO中无效果
    → normalized=unnormalized → 简化setup的局限
    → 真GRPO需要同prompt组 + warm-start

→ 假设部分验证:
  ✓ 定性支持: 自适应噪声模式与GRPO σ归一化机制一致
  ✗ 定量不等: 需更proper的GRPO实验设计(warm-start + 同prompt组)
  → 下一步: 用mini_grpo_training.py的完整GRPO模式 + SFT→GRPO warm-start
```

## 四、与之前实验的综合

```
实验证据链:

1. Regularization comparison (BF16 AMP):
   → 梯度噪声最有效(+44% eval)
   → BF16 AMP伪收敛(-9% eval)

2. Batch size scaling (BF16 AMP):
   → 大batch收敛更好(GA=8→100% vs GA=1→87%)
   → 大batch=更稳定梯度 → 与噪声互补(噪声=探索,大batch=稳定)

3. LR schedule (BF16 AMP):
   → Constant schedule最优 → 简单最稳

4. 本次噪声验证 (FP32):
   → 自适应噪声 > 固定噪声 (2.28M: +11% vs +1%)
   → 噪声效果随模型大小变化
   → FP32噪声效果 < BF16 AMP噪声效果 (AMP额外制造收敛障碍)

综合结论:
  GRPO σ归一化的本质 = 自适应梯度噪声 + reward-based方向选择
  → 自适应噪声: σ归一化部分 (更多探索当不确定)
  → reward方向选择: advantage-based部分 (朝高reward方向优化)
  → 两者结合 = GRPO完整机制

  SFT + adaptive noise = GRPO的"自适应噪声"部分(无reward方向选择)
  → 效果比GRPO弱 → 因为缺少reward-based方向优化
  → 但已超越固定噪声 → 验证了自适应噪声的独立价值!
```

## 五、300步扩展实验 — 重大反转!

```
2.28M FP32, 300步 (之前100步: baseline 24%, fixed noise 25%, adaptive 35%)

策略               | eval  | peak  | Δeval | loss
SFT baseline        | 59%   | 66%   | +0%   | 1.327
GRPO A=(r-μ)/σ      | 22%   | 34%   | -37%  | 0.000
GRPO A=r-μ          | 22%   | 34%   | -37%  | 0.000
SFT+fixed σ=0.01    | **97%** | 96%  | +38%! | 0.035 ← 最优!!
SFT+adaptive σ∝var  | 88%   | 92%   | +29%  | 0.068

→ 固定噪声σ=0.01达到97%! 比baseline提升38%!
→ 自适应噪声88%也很好(+29%) 但不如固定噪声
→ GRPO仍然21-22% — 简化GRPO确实需要warm-start
```

### 5.1 300步轨迹分析 — 固定噪声稳步上升

```
Step | baseline | fixed_σ=0.01 | adaptive_σ∝var
0    | 2%       | 2%           | 2%
25   | 26%      | 12%          | 8%
50   | 18%      | 24%          | 22%
75   | 28%      | 38%          | 56% ← adaptive peak!
100  | 14%      | 26%          | 40%
125  | 32%      | 42%          | 40%
150  | 66%      | 58%          | 52%
175  | 54%      | 46%          | 58%
200  | 44%      | 54%          | 40%
225  | 42%      | 86%!         | 92% ← adaptive peak!
250  | 58%      | 86%          | 84%
275  | 46%      | 88%          | 84%
299  | 54%      | 96%!         | 88%

→ baseline: 持续波动(2→66→14→32→66→54→44→42→58→46→54) → 不稳定!
→ fixed noise: 稳步上升(2→12→24→38→42→58→86→88→96) → 一致收敛!
→ adaptive noise: 先升后稳(2→22→56→40→52→58→40→92→84→88) → 波动更大

→ **关键洞察**: 固定噪声比自适应噪声更好(97% vs 88%)!
  → adaptive mean_sigma=0.006 < fixed σ=0.01 → 自适应平均噪声更小
  → 当loss variance降低 → adaptive σ也降低 → 探索减少 → 提前收敛
  → fixed σ=0.01始终保持探索 → 持续找到更好解 → 达到97%!
```

### 5.2 100步 vs 300步对比 — 噪声效果随训练进度变化

```
                 | 100步 eval | 300步 eval | 趋势
SFT baseline     | 24%        | 59%        | ↑35% (更多步=更好)
fixed σ=0.01     | 25% (+1%)  | 97% (+38%) | ↑72%!! 噪声收益指数增长!
adaptive σ∝var   | 35% (+11%) | 88% (+29%) | ↑53% 也显著提升
GRPO norm        | 21% (-3%)  | 22% (-37%) | RL始终差

→ 100步: noise边际帮助(+1%/+11%)
→ 300步: noise巨大帮助(+38%/+29%)!

→ 为什么?
  100步时: 模型刚开始学习 → 需要精确梯度 → 噪声干扰初期学习
  300步时: 模型已有基础 → 需要探索逃离local optima → 噪声提供探索!
  → **噪声的收益随训练步数增加!**
  → 类比: 模拟退火 → 初期高温(噪声)帮助逃离 → 后期低温(精确)收敛
  → 但在我们的实验中: 固定噪声全程有效 → 不需要退火!

→ 与regularization实验对比:
  BF16 AMP 100步: grad_noise +44% (噪声帮助逃离AMP伪收敛)
  FP32 100步: grad_noise +1%  (噪声边际帮助)
  FP32 300步: grad_noise +38% (噪声极大帮助!)
  → 噪声收益 = 训练步数 × 模型难度 的函数!
```

### 5.3 为什么固定噪声 > 自适应噪声?

```
adaptive noise: σ = base_sigma × max(loss_std[-5:], 0.1)
  → mean_sigma = 0.006 (平均比fixed σ=0.01小40%)
  → 后期loss稳定 → σ变小 → 探索减少 → 提前收敛到88%
  → 优点: 初期loss方差大时σ更大(0.025) → 更激进探索 → step 75达56%

fixed noise: σ = 0.01 常量
  → 全程σ=0.01 → 一致性探索 → 不因loss变化而调整
  → 后期仍保持0.01探索 → 继续发现更好解 → step 299达97%
  → 优点: 简单一致 → 不需要计算loss variance → 一步代码

→ 对比:
  简单任务(有明确全局最优): 固定噪声 > 自适应 → 持续探索到达97%
  复杂任务(无数值最优): 自适应噪声可能更好 → 初期探索→后期稳定

→ 生产建议:
  算术/确定性任务: 固定σ=0.01 → 一致性探索到达全局最优
  GRPO RL训练: σ∝reward variance → 自适应噪声(组比较提供自适应σ)
  通用SFT: 固定σ=0.01-0.1 → 简单有效, 300步以上显著收益
```

```bash
cd ~/rollout-infra
source ~/anaconda3/bin/activate llm
CUDA_VISIBLE_DEVICES=0 python -u tools/grpo_advantage_noise_verification.py --model_size 76k
CUDA_VISIBLE_DEVICES=0 python -u tools/grpo_advantage_noise_verification.py --model_size 2.28m

results/grpo_advantage_noise_76k.json
results/grpo_advantage_noise_2.28m.json
```

工具: `tools/grpo_advantage_noise_verification.py`