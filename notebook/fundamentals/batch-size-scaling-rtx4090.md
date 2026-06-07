# Batch Size Scaling — RTX 4090实测

> 2026-06-07 | 固定计算预算下batch size对收敛的影响: 大batch反而更好!

## 概述

在RTX 4090上测试梯度累积(GA=1/2/4/8)对训练收敛的影响。
**关键设计**: 固定总计算预算(1000 micro-steps) → 大GA = 更少优化器更新 = 同等总计算量

## 一、实验结果

### 1.1 2.28M模型 (1000 micro-steps, lr=2e-3, BF16 AMP)

```
GA | opt_steps | time(s) | final_eval | peak_eval | loss  | micro/s
1  | 1000      | 9.6     | 87%        | 76%       | 0.379 | 104
2  | 500       | 8.5     | 85%        | 80%       | 2.436 | 118
4  | 250       | 8.4     | 89%        | 84%       | 0.037 | 118
8  | 125       | 8.4     | 100%       | 100%      | 0.031 | 119

→ 反直觉发现: GA=8 (大batch) eval最高! 100% vs GA=1 87%
→ 大batch收敛更好 → 13% improvement!
→ 更少优化器更新(125 vs 1000) → 但每次更新更准确!
```

### 1.2 76K模型 (1000 micro-steps, lr=2e-3, BF16 AMP)

```
GA | opt_steps | time(s) | final_eval | peak_eval | loss  | micro/s
1  | 1000      | 5.9     | 100%       | 100%      | 0.000 | 169
2  | 500       | 4.9     | 100%       | 100%      | 0.003 | 205
4  | 250       | 4.7     | 100%       | 100%      | 0.027 | 213
8  | 125       | 4.5     | 100%       | 100%      | 0.073 | 221

→ 76K太简单 → 所有GA都100% eval → batch size不重要
→ 但micro/s: GA=8=221 > GA=1=169 → 1.31x吞吐提升!
  → 大batch让GPU利用率更高(减少optimizer.zero_grad/step频率)
```

## 二、关键发现

### 2.1 大batch收敛更好!

```
为什么GA=8比GA=1更好? (2.28M)

理论: 小batch → 更多噪声 → 更多探索 → 更快收敛(直觉)
但实测: 大batch → 更准确梯度 → 更稳定收敛 → 更高eval!

→ 原因:
  1. 算术任务梯度非常噪声 → 单样本的梯度方向不稳定
  2. GA=1: 每步只有1个样本 → 梯度方向随机 → 优化器摇摆
  3. GA=8: 8个样本平均 → 梯度方向一致 → 优化器稳步前进
  4. GA=8虽然只有125步更新 → 但每步都方向正确 → 更高效!

→ 与之前优化理论实验一致:
  AdamW lr=2e-3 → 最佳eval → GA=8用相同lr → 更好的梯度 → 更稳定!

→ 类比:
  小batch = 随机散步(很多步但方向随机)
  大batch = 精准行走(少步但方向正确) → 虽然步数少但更快到达!
```

### 2.2 吞吐提升: 大batch更高效

```
micro/s随GA增大:
  76K: GA=1=169 → GA=8=221 → 1.31x
  2.28M: GA=1=104 → GA=8=119 → 1.15x

→ 为什么大batch吞吐更高?
  optimizer.zero_grad() + optimizer.step() 每GA步只1次 → GA=8减少8x!
  → 小模型optimizer overhead占比高 → 减少更新频率 → 提升吞吐!

→ 2.28M提升小(1.15x) → 因为optimizer overhead占比低(GEMM主导)
→ 76K提升大(1.31x) → 因为optimizer overhead占比高(小GEMM)
```

### 2.3 "最优batch size"取决于任务

```
算术任务(确定性答案):
  → 大batch最优(GA=8→100% eval)
  → 因为正确答案唯一 → 梯度方向明确 → 大batch不丢失方向

RL训练(GRPO, 需要探索):
  → 小batch可能更好 → 因为探索需要噪声
  → 但之前GRPO实验: GA=4 (n=4)→81% vs GA=8→100%
  → → RL也需要合理的batch size → 太小噪声太大 → 太大方向太稳

→ 生产建议:
  SFT/对齐训练: GA=4-8 → 稳定收敛+高效吞吐
  RL训练(GRPO): n=4-8组 → 足够的组内统计量
  短训练(<100步): GA=4 → 最快到100% eval
  长训练(>1K步): GA=1-2 → 更多更新频率 → 但需要降低lr
```

## 三、工具

```bash
cd ~/rollout-infra
source ~/anaconda3/bin/activate llm
CUDA_VISIBLE_DEVICES=0 python -u tools/batch_size_scaling.py --model_size 76k --num_micro_steps 1000
CUDA_VISIBLE_DEVICES=0 python -u tools/batch_size_scaling.py --model_size 2.28m --num_micro_steps 1000

results/batch_size_scaling_results.json (76K)
results/batch_size_scaling_2.28m.json (2.28M)
```

工具: `tools/batch_size_scaling.py`