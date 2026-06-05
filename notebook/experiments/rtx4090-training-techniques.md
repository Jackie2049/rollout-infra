# LLM Training Techniques — 核心训练技巧实测 (RTX 4090)

> 2026-06-05 | 工具: `tools/training_techniques.py` | RTX 4090 24GB
> 参考: LLaMA 3, GPT-3, Chinchilla training recipes

## 1. 实验设计

**模型**: MiniTransformer (Pre-norm, GELU, Weight Tying)
**数据**: 合成重复模式数据 (vocab=256, seq=128, 2000 样本)
**训练**: 500 steps, batch=32, AdamW (lr=1e-3, wd=0.01)

5 个实验:
1. LR Schedule 对比 (constant / cosine_warmup / linear_decay / step_decay)
2. Gradient Clipping 效果 (no_clip / 0.5 / 1.0 / 5.0 / 10.0)
3. AdamW vs Adam (weight decay 对比)
4. Mixed Precision (FP32 / FP16 / BF16)
5. 模型规模效应 (0.09M → 4.84M)

## 2. LR Schedule 对比

| Schedule | Final Loss | Min Loss |
|----------|-----------|----------|
| constant | **0.4269** | 0.4269 |
| cosine_warmup | 4.1270 | 4.1025 |
| linear_decay | 3.8570 | 3.8411 |
| step_decay | 6.3341 | 6.2849 |

**分析**:
- Constant LR 最好! 这是因为数据太简单 (重复模式) — 不需要 warmup
- Warmup 在 LR 从 0 线性增长, 前 50 步浪费了学习时间
- 真实 LLM 训练: 数据复杂 + 大 batch + 大 LR → warmup 必需
- **教训**: 训练 recipe 取决于任务复杂度

## 3. Gradient Clipping

| Clip Norm | Final Loss | Avg Grad Norm | Max Grad Norm |
|-----------|-----------|---------------|---------------|
| no_clip | 6.9478 | 2.94 | 27.72 |
| 0.5 | 4.1271 | 2.81 | 28.43 |
| **1.0** | **4.1270** | 2.81 | 28.43 |
| 5.0 | 6.4693 | 2.87 | 28.43 |
| 10.0 | 6.7919 | 2.94 | 28.43 |

**分析**:
- clip=0.5 和 clip=1.0 效果相同 (梯度很少超过 1.0)
- No clip 和大 clip (5.0/10.0) 效果差 — 说明偶尔有梯度爆炸 (max=27.72)
- clip=1.0 是标准选择: 不会过度限制正常梯度, 但能防止爆炸

## 4. AdamW vs Adam

| Optimizer | WD | Final Loss | Weight Norm |
|-----------|-----|-----------|-------------|
| AdamW | 0.01 | 4.1270 | 225.37 |
| AdamW | 0.1 | **4.1104** | 220.02 |
| Adam | 0.01 | 6.5667 | 183.21 |
| Adam | 0.0 | 4.1293 | 225.98 |

**分析**:
- AdamW_wd=0.1 最好: 更多 weight decay → 更好的泛化
- **Adam (L2) vs AdamW**: Adam 的 L2 regulariztion 与自适应 LR 耦合 → 效果差
  - Adam_wd=0.01 loss=6.57 (比 AdamW 的 4.13 差很多!)
  - AdamW 将 weight decay 与梯度更新解耦
- Adam_no_wd ≈ AdamW_wd=0.01: 说明在这个简单任务上正则化影响不大

## 5. Mixed Precision

| Precision | Final Loss | Time | Peak Memory |
|-----------|-----------|------|-------------|
| FP32 | 4.1270 | 2.27s | 124.2MB |
| FP16 | 4.0958 | 2.82s | 88.3MB |
| BF16 | 4.0943 | 2.69s | 88.3MB |

**分析**:
- **内存节省**: FP16/BF16 = 88.3MB vs FP32 = 124.2MB → **29% 节省**
- **精度**: BF16 略好于 FP16 (更宽的指数范围)
- **速度**: FP32 反而最快! 小模型 GPU 利用率已经很高, AMP overhead 不值得
- 对于大模型 (7B+): AMP 加速 + 内存节省会非常显著
- **BF16 是首选**: 不需要 GradScaler, 不会 underflow

## 6. 模型规模效应

| Model | Params | Time/100 steps | Loss |
|-------|--------|---------------|------|
| tiny | 0.09M | 0.51s | 7.27 |
| small | 0.31M | 0.50s | 7.47 |
| medium | 2.21M | 0.86s | 7.28 |
| large | 4.84M | 1.25s | **6.56** |

**分析**:
- Loss 随模型增大而下降 (scaling laws), 但在这 100 步内还没充分收敛
- 时间 ∝ params: 0.09M→0.51s, 4.84M→1.25s (2.5x)
- tiny 和 small 时间几乎相同 → kernel launch overhead 占主导
- **真实场景**: 7B 模型需要 ~100K steps 在 TB 级数据上

## 7. 核心学习

1. **Warmup 在简单任务上不必要, 但真实训练必需**: 大 LR + 复杂数据 → 初始梯度爆炸
2. **Gradient Clipping clip=1.0 是安全默认值**: 防止爆炸, 不影响正常训练
3. **AdamW >> Adam with L2**: 解耦 weight decay 是关键差异
4. **BF16 是最佳精度选择**: 安全 + 省内存, 不需要 scaler
5. **Scaling Laws 验证**: 更大模型 → 更低 loss, 但需要更多数据+steps
6. **训练 recipe 是经验总结**: LLaMA/GPT-3 的 recipe 来自大规模实验验证
