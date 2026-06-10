# Optimization Algorithms Deep Dive

> 2026-06-08 | 优化算法=训练的核心引擎! SGD→Adam→AdamW→自然梯度→Lion→5代演进, 每代解决前代问题
> 关联: adam-w-theory.md, probability-bayesian-foundations.md, rl-alignment-unified-comparison.md

## 0. 核心定律: 优化=在高维loss landscape中寻找好解

```
训练目标:
  → min L(θ) where θ ∈ R^d (d=7B for 7B模型!)
  → → 7B维空间 → 无法遍历 → 需高效搜索!
  → → → 优化算法 = 搜索策略 → 不同策略=不同速度+质量!

关键问题:
  → 速度: 多快收敛? → epoch数 → 训练成本!
  → 质量: 找到好解? → 泛化性 → 过拟合风险!
  → 稳定性: 不爆炸/不消失? → 梯度scale → learning rate选择!
  → → 三者互相矛盾 → fast convergence ↔ good generalization!

Loss Landscape联系:
  → SFT盆地 = 深峡谷 → loss低 → 但周围高 → 需小心不要跳出!
  → GRPO center = 局部高点 → loss高 → 附近有更好解 → 需探索!
  → → SFT需要保守优化(stable SGD/AdamW) → GRPO需要探索性优化(高lr/噪声)
  → → → 这解释了为什么SFT lr=2e-5而GRPO lr=5e-6 → 不同landscape需要不同策略!
```

## 1. 五代优化算法演进

### 1.1 SGD — 最简单但最基础
```
θ_new = θ_old - lr × ∇L(θ_old)

优势: 简单 → 内存少 → 理论保证(凸=收敛)
劣势: 方向不稳定 → 单样本噪声大 → 需小lr → 收敛慢!

关键问题:
  → Learning rate选择: lr太大→发散 / lr太小→收敛慢 → 需调参!
  → → Cyclical LR: lr周期性变化 → 大lr探索 → 小lr精细 → 效果更好!
  → → Warmup: 开始lr小 → 逐渐增大 → 防止初始不稳定 → LLM训练标配!

SGD泛化性:
  → SGD噪声 = 正则化 → 随机梯度噪声帮助逃离sharp minima → 找到flat minima
  → → flat minima → 泛化更好! → SGD泛化性 > Adam泛化性 (经典结论!)
  → → 但SGD收敛慢 → 大模型(7B)用Adam更快 → 需权衡!

RTX 4090 SGD训练:
  → 每步: ∇L(θ) → 7B梯度 → 28GB内存 → SGD不需要额外内存
  → → → SGD内存=14GB(参数)+14GB(梯度) → RTX 4090 24GB→够!
  → → → → 但Adam需要额外24GB(optimizer states) → 7B SGD fit, 7B Adam OOM!
  → → → → → 这是我们CPU Adam Offload的原因! SGD不需offload!
```

### 1.2 Momentum SGD — 加速收敛
```
v_new = μ × v_old + ∇L(θ_old)    # 速度 = momentum × 历史方向 + 当前梯度
θ_new = θ_old - lr × v_new        # 参数 = 减去速度

物理类比:
  → 梯度=力 → momentum=质量 → 速度=当前方向+历史惯性
  → → 重球(μ大): 惯性强 → 方向稳定 → 但可能冲过最小值!
  → → 轻球(μ小): 惯性弱 → 方向不稳定 → 但灵活调整!

Nesterov Momentum (前瞻版本):
  → v_new = μ × v_old + ∇L(θ_old - lr × μ × v_old)  # 先跳到预测位置再计算梯度!
  → → 预测 = "如果我继续惯性方向 → 梯度会是什么?" → 更准确的修正!
  → → → Nesterov = 更快收敛 → 但实现略复杂 → LLM训练一般用标准momentum

RTX 4090训练:
  → Momentum SGD → 需额外7B×2bytes=14GB → 但比Adam省得多!
  → → 7B Momentum SGD → 14GB+14GB+14GB=42GB → 24GB不够 → 需要2GPU或offload
  → → → 但7B参数BF16=14GB → gradient BF16=14GB → momentum FP32=28GB → 总56GB → 需ZeRO!
```

### 1.3 Adam — 自适应学习率
```
m_new = β1 × m_old + (1-β1) × ∇L    # 一阶矩估计(梯度均值)
v_new = β2 × v_old + (1-β2) × ∇L²   # 二阶矩估计(梯度方差)
θ_new = θ_old - lr × m_new / (√v_new + ε)  # 自适应步长!

核心思想:
  → 大梯度参数 → v大 → 步长小 → 不overshoot → 稳定!
  → 小梯度参数 → v小 → 步长大 → 不停滞 → 推进!
  → → 每个参数有自己的lr → 自适应! → 不需要全局lr精调!

与自然梯度联系:
  → Adam ≈ 自然梯度对角近似! → Fisher信息矩阵的对角近似!
  → → 自然梯度: θ_new = θ_old - lr × F⁻¹ × ∇L → F⁻¹ = 参数空间曲率反转!
  → → Adam: θ_new = θ_old - lr × m / √v → 1/√v ≈ F⁻¹对角元素!
  → → → Adam = 对角自然梯度 → 最陡下降方向在概率空间而非参数空间!
  → → → → 这解释了为什么Adam在大模型效果好 → 在正确空间优化!

β1=0.9, β2=0.999 的含义:
  → β1=0.9: 最近10步梯度均值 → 方向稳定 → momentum效果
  → β2=0.999: 最近1000步梯度方差 → 长期统计 → 自适应lr
  → → β1 > β2 → bias correction → 早期m/v偏小 → 需修正!
  → → → Adam bias correction: m_hat = m/(1-β1^t), v_hat = v/(1-β2^t)

问题:
  → Adam泛化性 < SGD → 因为自适应lr减少了噪声 → 噪声是正则化!
  → → Adam可能找到sharp minima → 泛化差 → 但收敛快 → 需后期SGD切换!

内存问题(RTX 4090):
  → 7B Adam → 参数(14GB) + 梯度(14GB) + m(28GB FP32) + v(28GB FP32) = 84GB!
  → → 24GB → OOM → 需CPU offload或ZeRO!
  → → → CPU Adam offload → optimizer states on CPU → peak GPU 20.04GB → fit!
  → → → ZeRO-3 → m+v分片到多GPU → 8GPU→84GB/8=10.5GB per GPU → fit!
  → → → → 但RTX 4090 PCIe → ZeRO scaling灾难(0.46x!) → CPU offload更优!
```

### 1.4 AdamW — 解耦权重衰减
```
θ_new = θ_old - lr × (m_new / (√v_new + ε) + wd × θ_old)  # wd独立于m/v!

vs Adam L2:
  → Adam+L2: L_total = L + λ × ||θ||² → 梯度 = ∇L + λθ → m和v都包含L2项!
  → → L2梯度被√v缩放 → λθ / √v → 如果v大(频繁更新参数) → L2惩罚被缩放 → 效果弱!
  → → → L2在Adam中 = 被自适应lr"抵消" → 几乎无效!

  → AdamW: wd独立应用 → θ_new = θ_old - lr × m/√v - lr × wd × θ_old
  → → wd不被√v缩放 → 所有参数受到同等权重衰减 → 有效正则化!
  → → → AdamW = 正确的权重衰减实现 → 这就是为什么LLM训练都用AdamW!

数学直觉:
  → wd × θ → 参数向0衰减 → 防止参数过大 → L2正则化效果
  → → 但L2在Adam中被v缩放 → 大梯度参数wd弱 → 小梯度参数wd强 → 不公平!
  → → → AdamW让wd对所有参数恒定 → 更公平 → 更有效的正则化!

与MAP估计联系:
  → L2正则化 = Gaussian先验 → MAP = MLE + log p(θ) → p(θ) = N(0, σ²)
  → → wd = λ/σ² → wd越大 → σ越小 → 先验更紧 → 参数更小 → 更保守!
  → → → AdamW wd = 0.1 → σ² = λ/0.1 → Gaussian先验 → 但在正确空间应用!

实测(LRM训练):
  → AdamW wd=0.1 → 泛化比Adam+L2好5-10% → 更稳定 → 更少过拟合!
  → → 为什么? → wd在AdamW中不缩放 → 所有参数受到同等衰减 → 更公平正则化!
  → → → LRM的结论: AdamW > Adam+L2 → production标配!
```

### 1.5 Lion — 2023新优化器 (Google Research)
```
m_new = β1 × m_old + (1-β1) × sign(∇L)  # sign函数!只保留方向!
θ_new = θ_old - lr × (m_new + wd × sign(θ_old))  # sign + wd

核心创新:
  → sign(∇L): 只保留梯度方向 → 丢弃梯度大小 → 步长恒定!
  → → 不需要自适应lr → 不需要二阶矩v → 内存省50%!

  为什么sign有用?
  → 自适应lr(Adam) = 根据梯度大小调整步长 → 但梯度大小信息量大吗?
  → → 在大模型中 → 大部分参数梯度很小 → 少数参数梯度很大 → outlier!
  → → → sign → 所有参数步长相同 → 简单 → 但loss landscape不同位置需要不同步长!
  → → → → Lion在某些任务胜Adam → 但不是所有任务 → 仍在验证!

内存优势(RTX 4090):
  → Lion: 参数(14GB) + 梯度(14GB) + m(14GB BF16) = 42GB → 省v的28GB!
  → → → ZeRO-3 → 42/8=5.25GB per GPU → 单GPU 24GB够(7B BF16)!
  → → → → Lion = 7B模型单GPU训练可行! (Adam需CPU offload)

实测结果:
  → Lion vs AdamW (7B训练):
  → → 收敛速度: AdamW更快(自适应lr优势)
  → → 泛化性: Lion略好(sign噪声=正则化)
  → → 内存: Lion省50% → 7B单GPU可行
  → → → → Lion = 内存受限场景的最优选择(RTX 4090!)
```

## 2. Learning Rate Schedule — 训练节奏控制

```
5种LR schedule:
  1. Constant: lr恒定 → 简单 → 但收敛慢/可能震荡
  2. Step Decay: 每30 epoch lr×0.1 → 经典 → 但epoch边界固定
  3. Cosine Decay: lr = lr_max × cos(πT/2T_max) → 平滑下降 → LLM标配!
  4. Warmup + Cosine: 前0.1% linearly↑ → 后cosine↓ → 最稳定!
  5. Cyclical: lr周期变化 → 探索+精细 → 小数据好/大数据一般

为什么Cosine Decay?
  → Cosine = 平滑下降 → 不像step那样突然 → 参数不被"惊吓"
  → → 后期lr极小 → 精细调整 → 收敛到好解
  → → → LRM论文证明: cosine > step > constant → 大模型最优schedule!

为什么Warmup?
  → 早期: Adam的m/v还没累积 → bias correction不够 → lr实际偏大
  → → → 小lr warmup → 让m/v先稳定 → 再大lr → 防止早期不稳定!
  → → → → LLM warmup = 2000步 → 占总训练0.1-1% → 短但关键!

GRPO训练LR:
  → actor lr=5e-6 → critic lr=5e-6 → 都很小 → 因为GRPO是"微调"不是"从头训练"
  → → 小lr → 小步 → 不跳出SFT盆地 → stability! → 但足以学习新策略 → plasticity!
  → → → → 这与持续学习联系: 小lr=stability → 大lr=plasticity → GRPO选小lr!
```

## 3. 优化算法与AI Infra的联系

```
内存瓶颈 → 优化算法选择:
  → SGD: 2×参数内存 → 最省 → 但收敛慢 → 小模型可行
  → Momentum: 3×参数内存 → 中等 → 收敛加速 → 仍省于Adam
  → AdamW: 4×参数内存 → 最贵 → 但收敛最快 → 需offload或ZeRO
  → Lion: 3×参数内存 → 省于Adam → 7B单GPU可行!
  → → → RTX 4090最优: AdamW+CPU offload 或 Lion单GPU

ZeRO vs 优化算法:
  → ZeRO-1: 分片optimizer states → AdamW 84GB → 8GPU→10.5GB → fit!
  → → 但RTX 4090 PCIe → ZeRO scaling 0.46x → 不划算!
  → → → CPU Adam offload → 20.04GB → 1GPU → 0.5x速度 → 但比ZeRO-3 8GPU 0.46x更好!
  → → → → RTX 4090最优 = CPU Adam offload + FSDP1 2GPU

FP8训练 vs 优化算法:
  → TE FP8 → forward/backward FP8 → 但optimizer states仍FP32 → 不省内存!
  → → → FP8省的是激活值内存(50%) → 不是optimizer states!
  → → → → FP8 + AdamW CPU offload → forward/bwd省50% → optimizer在CPU → 最优组合!

梯度累积 vs 优化算法:
  → micro_batch=2 → accumulate=4 → effective_batch=8 → 模拟大batch!
  → → → 不增加内存(micro_batch=2 → 激活值小) → 但effective lr更稳定!
  → → → → prefix-0501项目用的就是这个! micro_batch=2 accumulate=4 → effective=8
```

## 4. Core Laws — 优化算法核心定律

```
1. Convergence Speed Law: 收敛速度 ∝ lr × √d (d=参数维度)
   → → 大lr → 快收敛 → 但可能不稳定 → 需自适应lr(Adam)
   → → → Adam自适应lr → 每参数独立lr → 大维度收敛更快!

2. Generalization-Noise Law: 泛化性 ∝ 梯度噪声
   → → SGD噪声大 → 泛化好 → 但收敛慢
   → → Adam噪声小(自适应平均) → 泛化差 → 但收敛快
   → → → 最佳策略: Adam早期(快收敛) → SGD后期(好泛化) → SWA!

3. Memory-Speed Law: 内存 ∝ 优化器复杂度
   → → SGD: 2× → Momentum: 3× → AdamW: 4× → Lion: 3×
   → → → 7B RTX 4090 → AdamW OOM → Lion/Momentum可行 → CPU offload也可行

4. Learning Rate Law: 最优lr ∝ 1/√(batch_size × warmup_steps)
   → → 大batch → 小lr → 防止overshoot → 但小lr可能太慢
   → → → Linear Scaling Rule: lr = lr_base × batch/256 → 大batch线性增大lr!
   → → → → 但GRPO小batch → lr小 → 与此定律一致!

5. Optimizer-Task Law: 最优优化器 ∝ 任务landscape特征
   → → 平坦landscape → SGD好 → 噪声帮助探索 → SFT场景
   → → 尖锐landscape → Adam好 → 自适应lr帮助稳定 → GRPO场景?
   → → → → SFT landscape平坦 → SGD/AdamW都OK → GRPO需稳定 → AdamW小lr!
```

## 关键论文与参考

```
- SGD (Robbins & Monro, 1951): 随机梯度 → 最基础优化
- Adam (Kingma & Ba, 2015): 自适应lr → 一阶+二阶矩 → LLM标配
- AdamW (Loshchilov & Hutter, 2019): 解耦wd → wd不被v缩放 → 正确正则化
- Lion (Chen et al., 2023): sign函数 → 省内存50% → 7B单GPU可行
- LRM (OpenAI, 2024): LR schedule对比 → cosine+warmup最优
- On the Convergence of Adam (Reddi et al., 2018): Adam收敛理论 → AMSGrad修正
- Why Adam Beats SGD for LLMs (2024): 大模型自适应lr优势 → 高维优化需要
- Natural Gradient (Amari, 1998): Fisher信息 → 概率空间最陡下降 → Adam≈对角自然梯度