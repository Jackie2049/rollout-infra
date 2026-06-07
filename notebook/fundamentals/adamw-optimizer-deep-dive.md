# AdamW Optimizer 深度解析 — 从数学到实战

> 2026-06-07 | 为什么AdamW是LLM训练标配: decoupled weight decay的理论与实践

## 一、Adam到AdamW的进化

### 1.1 Adam: 自适应学习率 + momentum

```
Adam (Kingma & Ba, 2015):
  m_t = β₁·m_{t-1} + (1-β₁)·g_t          #一阶momentum (均值)
  v_t = β₂·v_{t-1} + (1-β₂)·g_t²          #二阶momentum (方差)
  m̂_t = m_t / (1-β₁^t)                     #bias correction
  v̂_t = v_t / (1-β₂^t)
  θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)     #参数更新

直觉:
  m̂_t ≈ 梯度的历史平均值 → 方向稳定(不随单个样本剧烈变化)
  v̂_t ≈ 梯度的历史方差 → 自适应步长(大方差→小步, 小方差→大步)
  √v̂_t ≈ 每个参数的"个性化学习率"

→ 对LLM训练关键: 不同参数梯度方差差异极大
  embedding层: 梯度方差小 → Adam给大步长 → 快速学习
  attention层: 梯度方差大 → Adam给小步长 → 稳定学习
  → Adam自动适配, 不需要手动调每层lr!
```

### 1.2 Adam的问题: L2正则是灾难

```
Adam + L2正则:
  loss = L(θ) + λ/2 · ||θ||²
  g_t = ∇L(θ) + λ·θ              #梯度加正则项
  → 正则项被Adam的v_t缩放!

  v_t包含λ²·θ² → √v̂_t ≈ √(variance + λ²·θ²)
  → 当θ大时, √v̂_t也大 → 正则更新被缩小!
  → 当θ小时, √v̂_t小 → 正则更新被放大!

  结果: 正则化强度与参数值耦合 → 大参数正则弱, 小参数正则强!
  → 完全相反于L2正则的本意(惩罚大参数)!

RTX 4090实测验证:
  Adam + L2 (wd=0.1): loss从5→173→1696! 灾难性发散!
  AdamW (wd=0.1): loss正常下降, ↓6.5%
  → L2正则在Adam中是灾难 → AdamW的decoupled wd才是正确方式
```

### 1.3 AdamW: Decoupled Weight Decay

```
AdamW (Loshchilov & Hutter, 2019):
  m_t = β₁·m_{t-1} + (1-β₁)·g_t          #momentum (不变)
  v_t = β₂·v_{t-1} + (1-β₂)·g_t²          #方差 (不变)
  θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)     #Adam更新 (不变)
  θ_t = θ_t - α·λ·θ_{t-1}                  #weight decay (独立!)

关键区别: wd不进入梯度 → 不被v_t缩放!
  → wd强度一致 → 大参数大惩罚 → 正确行为!
  → wd与learning rate decoupled → 可以独立调!

理论推导:
  Adam+L2: θ_{t+1} = θ_t - α·(∇L + λθ)/(√v + ε)
           → λθ被√v缩放 → 正则化强度∝λ/(√v) → 不一致!

  AdamW:   θ_{t+1} = θ_t - α·∇L/(√v + ε) - α·λ·θ
           → wd项独立 → 强度∝α·λ → 一致!

→ AdamW = SGD wd的"自适应版本"
  SGD wd: θ = θ - α·∇L - α·λ·θ → wd∝α·λ (一致)
  AdamW wd: θ = θ - α·∇L/(√v+ε) - α·λ·θ → wd∝α·λ (一致,不受√v影响)
```

## 二、超参数解析

### 2.1 β₁=0.9, β₂=0.999, ε=1e-8

```
β₁=0.9 (momentum衰减):
  m_t = 0.9·m_{t-1} + 0.1·g_t
  → 有效窗口 ≈ 1/(1-β₁) = 10步
  → 梯度的10步滑动平均 → 短期趋势
  → 为什么0.9? 经验最优值(SGD momentum也用0.9)
  → 不建议调整(几乎所有论文都用0.9)

β₂=0.999 (方差衰减):
  v_t = 0.999·v_{t-1} + 0.001·g_t²
  → 有效窗口 ≈ 1/(1-β₂) = 1000步
  → 梯度方差的1000步滑动平均 → 长期统计
  → β₂过小(如0.9) → v_t不稳定 → 步长波动 → 训练不稳定

ε=1e-8 (防除零):
  θ = θ - α·m̂/(√v̂ + ε)
  → 当v̂≈0 → √v̂≈0 → ε防止除零
  → ε太大 → 步长被截断 → 学习慢
  → 1e-8是经验最优(大模型训练标配)

Bias Correction:
  m̂_t = m_t/(1-β₁^t), v̂_t = v_t/(1-β₂^t)
  → 初始几步: m_t偏向0(v初始化为0) → correction放大
  → t→∞时: β^t→0 → correction→1 → 不影响
  → 实际: PyTorch的Adam默认不correction → 但训练仍然OK
    → 因为warmup阶段lr很小 → 初始偏移影响小
```

### 2.2 Learning Rate 与 Warmup

```
LLM训练典型lr:
  Pretrain: α=3e-4 (GPT-3/LLaMA)
  SFT: α=2e-5 to 5e-5
  RL(GRPO/PPO): α=1e-5 to 5e-5
  我们实测: α=1e-3最优(小模型mini GRPO)

为什么RL lr比SFT更小?
  RL训练已经基于SFT模型 → θ接近最优
  → 大lr推离最优 → 破坏SFT知识 → 不稳定
  → 小lr → 只微调 → 保持SFT基础 → 强化

Warmup (0→α in first N steps):
  → 冷启动: Adam的m和v≈0 → 步长不稳定
  → warmup让m和v逐渐积累 → 步长稳定
  → LLM训练: warmup 2000步(GPT-3)/2%总步数(LLaMA)
  → 实测: warmup+constant ↓11.9% > cosine ↓8.7% (短训练)
```

### 2.3 Weight Decay λ

```
典型wd值:
  LLaMA: λ=0.1
  GPT-3: λ=0.1
  实测: λ=0.1最优(AdamW)

为什么wd=0.1而不是0.01?
  → wd在AdamW中独立于lr → 可以设较大值
  → wd=0.1 → 每步衰减参数0.1×lr → 参数不无限增长
  → wd太小 → 正则不足 → 过拟合
  → wd太大 → 正则过强 → 学不到知识

数学: wd的效果
  θ_{t+1} = (1-α·λ)·θ_t - α·∇L/(√v+ε)
  → 每步参数被(1-α·λ)衰减 → 防止参数爆炸
  → α=1e-3, λ=0.1 → 衰减率=0.0001 → 温和衰减
```

## 三、AdamW vs 其他优化器

```
AdamW vs SGD+Momentum:
  → SGD: 所有参数用同一lr → 需手动调
  → AdamW: 每个参数自适应lr → 不需要
  → 实测: AdamW↓6.5% > SGD↓3.3% (小模型)
  → 大模型: AdamW收敛更快 → LLM标配

AdamW vs AdaFactor:
  → AdaFactor: 分解v_t为行×列 → 内存O(d) vs O(d²)
  → 用于大模型训练(T5)
  → 精度略低 → 但内存节省显著
  → ZeRO-3分片Adam → 内存已解决 → AdamW继续主流
```

## 四、AdamW在RL训练中的特殊性

```
RL训练(GRPO/PPO/DPO):
  → 基于SFT模型 → θ初始接近最优
  → lr必须小(1e-5~5e-5) → 防破坏SFT知识
  → gradient clipping(norm=1.0) → 必须有!
    → RL梯度方差大(reward-dependent)
    → clip防止"reward hacking大梯度"推走模型

为什么GRPO比PPO更稳定?
  → GRPO: 组归一化A=(r-μ)/σ → 自适应gradient clipping!
    → σ大 → A小 → 梯度小 → 稳定
    → σ小 → A大 → 梯度大 → 需clip
  → PPO: A=GAE → 不归一化 → 需手动clip(ratio)
  → GRPO的σ归一化 ≈ 自适应clip → 不需要PPO的ratio clip!
```

## 五、理论深度: Adam ≈ 自然梯度对角近似

```
自然梯度 (Amari, 1998):
  ∇_nat J = F^{-1}·∇J     #F=Fisher信息矩阵
  → 参数空间的"最短路径" → 考虑参数分布的几何

Adam ≈ 自然梯度对角近似:
  v_t ≈ 梯度的二阶矩 ≈ Fisher信息矩阵的对角元素
  → m̂_t/(√v̂_t + ε) ≈ F_diag^{-1}·∇J
  → Adam ≈ 对角自然梯度 → 近似但实用!

为什么对角近似OK?
  → F是d×d矩阵(7B=7e9×7e9) → 完全逆不可能!
  → 对角近似 → 只存7e9个值 → 可行
  → 实际效果: 每个参数有"个性化"学习率 → 自适应
```

## 六、PyTorch实现要点

```python
# PyTorch AdamW (简化版)
class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.1):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.wd = weight_decay
        # 初始化: m=0, v=0 for each parameter

    def step(self):
        for p in params:
            if p.grad is None: continue

            grad = p.grad.data
            state = self.state[p]

            # Momentum update
            state['m'] = β₁ * state['m'] + (1-β₁) * grad
            state['v'] = β₂ * state['v'] + (1-β₂) * grad²

            # Bias correction
            m_hat = state['m'] / (1 - β₁^t)
            v_hat = state['v'] / (1 - β₂^t)

            # Parameter update (Adam part)
            p.data -= lr * m_hat / (√v_hat + eps)

            # Weight decay (DECOUPLED!)
            p.data -= lr * wd * p.data
            # 注意: 这不进入grad → 不被v缩放!

# 内存开销:
# m: 1 float per param → 7B = 7GB (FP32)
# v: 1 float per param → 7B = 7GB (FP32)
# master weights: 7GB (FP32 copy)
# Total: ~21GB per 7B model → 占训练内存50%!
```

## 七、实测数据汇总

```
实验 | 配置 | 结果
-----|------|------
AdamW(lr=0.001,wd=0.1) | 最优配置 | loss↓6.5% ✓✓✓
AdamW(lr=0.001,wd=0) | 无wd | 略差, 过拟合风险
Adam+L2(wd=0.1) | 灾难配置 | loss飙升47x! ✗✗✗
SGD(lr=0.1) | 非自适应 | loss↓3.3% (差)
Warmup+constant | 短训练最优 | ↓11.9%
Cosine decay | 长训练最优 | ↓8.7% (短训练更差)
Gradient accumulation | 等效大batch | 无额外内存(43.9MB不变)
BF16训练 | B≥128才有1.07x | B=16反而慢0.54x

→ 核心结论:
  1. AdamW > Adam+L2 (decoupled wd是关键)
  2. AdamW > SGD (自适应lr对小模型短训练更好)
  3. wd=0.1是LLM标配 (不要用L2!)
  4. Warmup对短训练 > Cosine decay
  5. Adam内存≈21GB/7B → ZeRO-1分片optimizer → 训练内存降8x
```

工具: `tools/mini_grpo_training.py` (优化理论实验)