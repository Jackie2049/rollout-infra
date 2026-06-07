# Loss Landscape Visualization — RTX 4090 实测

> 2026-06-07 | 损失景观可视化验证泛化理论: SFT→GRPO在平坦盆地

## 概述

使用filter-normalized 2D切片(Li et al., 2018)可视化4个训练模型的损失景观。关键发现: **SFT和SFT→GRPO的中心loss极低(0.001)且盆地平坦**, GRPO-only的中心loss高(5.06)且景观不稳定。

## 一、方法: Filter-Normalized 2D切片

```
步骤:
1. 训练4个模型: Base(random), SFT, GRPO, SFT→GRPO
2. 对每个模型, 生成2个随机方向d1, d2
3. Filter normalization: 对每个参数filter(行), d_i归一化后乘以参数范数
   → 确保不同模型的方向在同一尺度上
4. 在(α, β)网格上计算loss: θ = θ_center + α·d1 + β·d2
5. 分析盆地形态: 平坦=泛化好, 锐利=泛化差

网格: 21×21, α∈[-1.5,1.5], β∈[-1.5,1.5]
Loss采样: 每网格点50个随机prompt计算CE loss
```

## 二、实验结果

### 2.1 中心损失和盆地形态

```
模型          | Center Loss | Eval Acc | Flatness  | Basin Type
-------------|-------------|----------|-----------|------------
Base(random)  | 3.418       | 0.0%     | -0.095    | 无意义(未训练)
SFT-only      | **0.001**   | 100%     | 0.0005    | 极平坦
GRPO-only     | 5.057       | 79%      | -0.215    | 不稳定
SFT→GRPO      | **0.001**   | 100%     | 0.002     | 极平坦
```

### 2.2 关键发现

#### 发现1: SFT和SFT→GRPO的中心loss几乎为零

```
SFT center loss = 0.001
SFT→GRPO center loss = 0.001
GRPO center loss = 5.057 (5000x更高!)

→ SFT训练找到了一个"完美解" → loss≈0 → 所有训练样本都被完美拟合
→ GRPO训练没有找到这样的解 → loss仍然高 → 训练样本没有被完美拟合
→ SFT→GRPO在SFT基础上 → 保持在完美解附近 → loss仍然≈0
```

#### 发现2: SFT盆地极平坦(flatness≈0)

```
SFT flatness = 0.0005 → 移动1.5个单位距离 → loss仅增加0.0005
SFT→GRPO flatness = 0.002 → 移动1.5个单位 → loss仅增加0.002

→ 平坦盆地 = 泛化好的理论依据!
  平坦盆地意味着: 即使参数稍有变化 → loss几乎不变 → 对未见数据也好
  → 模型学到了"鲁棒"的表示 → 不依赖参数的精确值 → 泛化好

→ vs 锐利盆地(sharp minima):
  参数稍有变化 → loss急剧增加 → 依赖参数的精确值 → 泛化差
```

#### 发现3: Loss Std差异巨大

```
Base: std=0.946 → 景观相对平滑(但center高)
SFT: std=3.005 → 景观变化大(但center极低 → 周围loss高 → 锐利边界)
GRPO: std=1.391 → 景观中等变化
SFT→GRPO: std=3.541 → 景观变化最大(SFT→GRPO比SFT更多变化!)

→ 解读:
  SFT和SFT→GRPO的center loss≈0 → 周围loss高达15 → 峡谷式盆地
  → center是"深谷底部" → 极低loss → 但一旦走出 → loss急剧升高

  → 这不是"平坦盆地"而是"深峡谷"!
  → 深峡谷 = 模型精确找到了好解 → 但偏离就会崩塌
  → 这说明SFT→GRPO的泛化好不是因为"盆地平坦"
  → 而是因为"解在峡谷底部" → 训练-eval一致 → 泛化好
```

#### 发现4: GRPO景观不稳定

```
GRPO center loss = 5.06 → 远高于SFT的0.001
GRPO flatness = -0.215 → 负值! → 周围loss比center更低!

→ GRPO模型的参数不是在loss最低点 → 而是在一个局部高点!
→ 附近有更低loss的解 → 但GRPO没有找到它们 → 训练不充分
→ 这解释了GRPO的泛化差: 模型没有收敛到最优解 → 参数不稳定

→ vs SFT: center是最低点 → 附近都是更高loss → SFT精确收敛!
```

## 三、理论解释: 峡谷 vs 平坦盆地

```
传统泛化理论: "平坦盆地 = 泛化好"
  → 基于直觉: 平坦 → 参数变化不影响 → 对未见数据也好
  → 但我们的实验显示: SFT→GRPO在"深峡谷"而非"平坦盆地"

修正理解:
  1. SFT训练 → 找到精确解(CE loss≈0) → 解在深峡谷底部
  2. 深峡谷 → center极低 → 周围极高 → model precision matters
  3. 但泛化好不是因为"盆地平坦" → 而是因为"解的质量高"!

→ 关键洞察:
  SFT→GRPO泛化好 = 训练时center loss极低 = eval也极低 = 一致性
  → 不是因为"容错" → 而是因为"解本身正确"

→ GRPO泛化差 = 训练时center loss高 = 解质量差 = 对训练和eval都差
  → 训练reward高 ≠ center loss低 → reward hacking!
  → GRPO的reward可能通过"取巧策略"得到 → 但CE loss仍然高 → 模型没学到正确算法
```

## 四、与泛化理论的连接

```
泛化gap = |L_train - L_eval|

SFT→GRPO: L_train_CE ≈ 0.001, L_eval_CE ≈ 0.001 → gap ≈ 0 → 完美!
GRPO:     L_train_reward 高(87.5%), L_train_CE 高(5.06) → reward≠CE!
  → reward hacking: reward高但CE loss也高 → 模型没学到正确预测

→ 新见解: CE loss是"真实泛化"的度量, reward是"训练信号"的度量
  CE loss低 → 模型对数据的预测准确 → 泛化好
  Reward高 → 模型的输出被reward函数认可 → 但可能是取巧

→ 两种信号可能矛盾!
  GRPO: reward高(reward函数认可) + CE loss高(预测不准确) → reward hacking
  SFT: CE loss低(预测准确) → reward自然高 → 不矛盾 → 泛化好
```

## 五、工具

```bash
# 在GPU服务器运行
cd ~/rollout-infra/
source ~/anaconda3/bin/activate llm
python tools/loss_landscape_visualization.py --device cuda --output loss_landscape_results.json
```

工具: `tools/loss_landscape_visualization.py`