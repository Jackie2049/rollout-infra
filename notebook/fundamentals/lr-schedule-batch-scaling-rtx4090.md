# LR Schedule & Batch Scaling — RTX 4090实测

> 2026-06-07 | 学习率调度策略+梯度累积batch缩放对比: constant/cosine/warmup × linear/sqrt/no-scaling × AdamW/SGD

## 概述

系统对比3类训练超参数组合对收敛质量和吞吐的影响:
1. **LR Schedule**: constant / cosine / warmup+constant / warmup+cosine
2. **Batch Scaling**: 不缩放(no) / 线性(linear) / 根号(sqrt)
3. **Optimizer**: AdamW vs SGD

共44组实验(2模型 × 22配置), 100-200步训练.

## 一、实验结果

### 1.1 LR Schedule对比 (GA=1, AdamW, lr=2e-3)

```
76K模型:
Schedule          | eval  | final_loss | last10_loss | 步数=100
constant          | 48%   | 0.028      | 0.408
cosine            | 24%   | 0.424      | 0.915       ← 最差!
warmup+constant   | 45%   | 0.028      | 0.402
warmup+cosine     | 31%   | 0.334      | 0.818

→ constant最好(48%), cosine最差(24%) — 短训练cosine没机会衰减到低lr!
→ warmup微提升(+3%), 但不是决定性因素

2.28M模型:
Schedule          | eval  | final_loss | last10_loss | 步数=100
constant          | 39%   | 0.005      | 0.893       ← lowest loss!
cosine            | 30%   | 0.024      | 1.392       ← higher loss
warmup+constant   | 30%   | 0.340      | 1.309
warmup+cosine     | 30%   | 0.037      | 1.214

→ constant依然最好(39%), 但差距缩小
→ 2.28M需要更多步→ schedule差异被收敛速度掩盖
→ 注意: last10_loss比final_loss高! → 训练仍在收敛中→ 需更多步
```

### 1.2 Batch Scaling对比 (warmup+constant, AdamW)

```
76K模型:
Scaling   | GA | LR     | eval  | loss   | last10
no        | 1  | 2e-3   | 45%   | 0.028  | 0.402
no        | 2  | 2e-3   | 76%   | 1.617  | 0.893
no        | 4  | 2e-3   | 91%   | 0.489  | 0.911
no        | 8  | 2e-3   | 96%   | 0.269  | 0.552
linear    | 2  | 4e-3   | 64%   | 1.196  | 0.868
linear    | 4  | 8e-3   | 81%   | 0.704  | 0.691
linear    | 8  | 16e-3  | 88%   | 0.897  | 0.643
sqrt      | 2  | 2.8e-3 | 58%   | 1.533  | 0.882
sqrt      | 4  | 4e-3   | 98%   | 0.187  | 0.444  ← 最优!
sqrt      | 8  | 5.7e-3 | 100%  | 0.056  | 0.098  ← 完美!

→ sqrt scaling GA=4→98%, GA=8→100%!
→ no scaling GA=8→96%, linear GA=8→88%
→ sqrt scaling是短训练最优策略!

2.28M模型:
Scaling   | GA | LR     | eval  | loss   | last10
no        | 1  | 2e-3   | 30%   | 0.340  | 1.309
no        | 2  | 2e-3   | 49%   | 1.188  | 1.247
no        | 4  | 2e-3   | 92%   | 0.365  | 0.499
no        | 8  | 2e-3   | 100%  | 0.019  | 0.047  ← 完美!
linear    | 2  | 4e-3   | 33%   | 1.526  | 1.780
linear    | 4  | 8e-3   | 45%   | 2.676  | 3.800  ← 灾难!
linear    | 8  | 16e-3  | 35%   | 5.236  | 8.744  ← loss爆炸!
sqrt      | 2  | 2.8e-3 | 32%   | 2.262  | 1.543
sqrt      | 4  | 4e-3   | 78%   | 0.652  | 2.198
sqrt      | 8  | 5.7e-3 | 81%   | 0.137  | 1.362

→ 2.28M: no scaling GA=8→100%! (最简单策略竟然最优!)
→ linear scaling灾难(loss爆炸!) → lr太大→梯度不稳定
→ sqrt scaling GA=8→81% (低于no scaling 100%!)
```

### 1.3 AdamW vs SGD

```
76K (sqrt scaling, warmup+constant):
AdamW GA=1 lr=2e-3: eval 45%, loss 0.028
AdamW GA=4 lr=4e-3: eval 98%, loss 0.187
AdamW GA=8 lr=5.7e-3: eval 100%, loss 0.056

SGD GA=1 lr=0.1:   eval 18%, loss 1.438
SGD GA=4 lr=0.2:   eval 7%,  loss 4.605 ← 灾难!
SGD GA=8 lr=0.283: eval 30%, loss 8.676 ← loss爆炸!

→ SGD在大batch时完全失败! (eval 7%→30%, loss 4.6→8.7)
→ AdamW在大batch时反而更好! (eval 98%→100%, loss 0.19→0.06)
→ SGD不适合梯度累积 → lr scaling让梯度不稳定

2.28M (sqrt scaling, warmup+constant):
AdamW GA=4 lr=4e-3: eval 78%, loss 0.652
AdamW GA=8 lr=5.7e-3: eval 81%, loss 0.137

SGD GA=1 lr=0.1:  eval 11%, loss 1.438
SGD GA=4 lr=0.2:  eval 17%, loss 5.082 ← 灾难
SGD GA=8 lr=0.283: eval 16%, loss 11.941 ← 灾难!

→ SGD同样失败 → AdamW是梯度累积训练的唯一选择!
```

## 二、关键发现

### 2.1 LR Schedule: 短训练constant最优

```
为什么constant比cosine好?

cosine schedule:
  lr(t) = lr_max × 0.5 × (1 + cos(πt/T))
  → 100步时: lr = lr_max × 0.5 × (1 + cos(π×100/100))
  → lr = lr_max × 0.5 × (1 + cos(π))
  → lr = lr_max × 0.5 × 0 = 0! ← 最后lr=0!

  → 问题: 短训练(<200步), cosine太快衰减lr → 学习停止太早!
  → 需要更多步(>500)才能让cosine有足够高lr阶段

constant schedule:
  lr(t) = lr_max → 全程高lr → 更快收敛 → 短训练最优!
  → 长训练(>1K步) → constant可能不稳定 → cosine更好

→ 生产建议:
  短训练(<200步): constant/warmup+constant
  中等训练(200-1K步): warmup+cosine
  长训练(>1K步): cosine衰减
```

### 2.2 Batch Scaling: 小模型sqrt最优, 大模型不缩放最优!

```
为什么sqrt scaling在小模型好, 大模型反而差?

理论: lr应该随batch缩放以保持梯度统计量不变
  linear scaling: lr ∝ batch → Goyal et al. (2017)
  sqrt scaling: lr ∝ sqrt(batch) → Krizhevsky (2014)

实测:
  76K (小模型): sqrt GA=8→100% > linear GA=8→88% > no GA=8→96%
  2.28M (大模型): no GA=8→100% > sqrt GA=8→81% > linear GA=8→35%

→ 为什么翻转?
  小模型(GA=1): 模型在compute-bound → lr更大 → 更快收敛 → sqrt缩放OK
  大模型(GA=1): 模型更大 → 初始loss高 → lr太大 → 梯度不稳定
  → 2.28M linear lr=16e-3 → loss爆炸到5.24!
  → 2.28M sqrt lr=5.7e-3 → 不够大 → eval只有81%
  → 2.28M no scaling lr=2e-3 → 足够稳定 → 100% eval!

→ 关键: 大模型需要更保守的lr策略!
  → 2.28M的"最优lr"在2e-3附近 → 任何缩放都偏离最优!
  → 小模型的"最优lr"在5e-3附近 → sqrt缩放正好命中!

→ 生产建议:
  小模型(<1M): sqrt scaling → GA=8时lr=5.7e-3
  大模型(1M+): 不缩放 → GA=8时仍用lr=2e-3 (或更低)
  超大模型(7B+): warmup是关键 → lr从1e-5线性升至2e-3 → 再cosine衰减
```

### 2.3 SGD vs AdamW: 梯度累积下SGD完全失败

```
为什么SGD在梯度累积下失败?

SGD: g_t = ∇L(θ_t) → θ_{t+1} = θ_t - lr × Σg_micro
  → 累积N步梯度 → lr × Σg ≈ lr × N × g_avg
  → 如果lr随N缩放 → lr × N × g_avg → 有效lr=lr×N → 灾难!

  实测: SGD GA=8 lr=0.283 → eval 30% (76K) / 16% (2.28M)
  → loss爆炸! → lr×N太大 → 梯度不稳定

AdamW: g_t = ∇L(θ_t) → m_t = β₁m + (1-β₁)g → v_t = β₂v + (1-β₂)g²
  → θ_{t+1} = θ_t - lr × m/(√v+ε) - lr × wd × θ
  → 自适应lr: lr × m/(√v+ε) → v累积梯度方差 → 自动缩放!
  → lr × N × g / (√(N×g²)+ε) ≈ lr × √N × sign(g) → 有效缩放∝√N!

  实测: AdamW GA=8 → eval 100% (76K) / 81% (2.28M)
  → AdamW自适应lr自动处理batch缩放 → 不需要手动缩放!

→ 关键: AdamW的自适应lr ≈ 自动sqrt scaling!
  → v_t ≈ Σg² → √v ≈ √(Σg²) ≈ ||Σg|| → lr×Σg/||Σg|| ≈ lr×√N
  → 这就是为什么AdamW + no scaling在大模型也能达到100%!

→ 生产建议:
  梯度累积训练: 必须用AdamW! SGD完全不适合!
  AdamW + no scaling: 大模型最稳定(100% eval)
  AdamW + sqrt scaling: 小模型最优(100% eval)
```

## 三、结论

```
最优策略组合:

训练场景          | Schedule | Scaling | Optimizer | LR
短训练+小模型(<1M) | constant | sqrt     | AdamW     | 2e-3×√GA
短训练+大模型(1M+) | constant | none     | AdamW     | 2e-3
长训练+大模型      | cosine   | none     | AdamW     | warmup→cosine
RL训练(GRPO/PPO)  | constant | none     | AdamW     | 1e-5→1e-3 warmup

禁忌:
  × SGD + 梯度累积 → loss爆炸!
  × Linear scaling + 大模型 → loss爆炸!
  × Cosine + 短训练 → lr衰减太快 → 学习停止
  × FP16+AMP + 大模型 → GradScaler溢出(见mixed-precision笔记)
```

## 四、工具

```bash
# GPU服务器运行
cd ~/rollout-infra
source ~/anaconda3/bin/activate llm

# LR Schedule experiments
CUDA_VISIBLE_DEVICES=0 python -u tools/lr_schedule_experiment.py --model_size 76k --num_steps 100
CUDA_VISIBLE_DEVICES=0 python -u tools/lr_schedule_experiment.py --model_size 2.28m --num_steps 100

# Mixed Precision experiments
CUDA_VISIBLE_DEVICES=0 python -u tools/mixed_precision_training_benchmark.py --model_size 76k --num_steps 200
CUDA_VISIBLE_DEVICES=0 python -u tools/mixed_precision_training_benchmark.py --model_size 2.28m --num_steps 200

# Results
results/lr_schedule_results.json
results/lr_schedule_results_2.28m.json
results/mixed_precision_training_results.json
results/mixed_precision_training_2.28m.json
```

工具: `tools/lr_schedule_experiment.py`, `tools/mixed_precision_training_benchmark.py`