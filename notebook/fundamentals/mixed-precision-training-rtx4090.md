# Mixed Precision Training — RTX 4090实测

> 2026-06-07 | FP32/FP16+AMP/BF16/BF16+AMP训练精度对比: BF16是RTX 4090最佳选择!

## 概述

在RTX 4090 (SM 8.9)上对比4种训练精度模式:
1. **FP32**: 标准基线
2. **FP16+AMP**: 混合精度+GradScaler(传统方案)
3. **BF16 native**: 模型权重直接用BF16, 无AMP, 无GradScaler
4. **BF16+AMP**: BF16混合精度模式

**FP8无法用于训练**: SM89(RTX 4090)不支持FP8 GEMM训练, FP8仅用于推理(cuBLAS FP8)

## 一、实验结果

### 1.1 76K模型 (极小模型, RTX 4090)

```
精度         | steps/s | 加速  | final loss | eval  | mem     | 稳定 | grad_norm
FP32         | 118.4   | 1.00x | 0.024      | 48%   | 0.019GB | YES  | 0.198
FP16+AMP     | 142.6   | 1.20x | 0.023      | 46%   | 0.019GB | YES  | 0.183
BF16 native  | 176.0   | 1.49x | 0.028      | 48%   | 0.018GB | YES  | 0.219
BF16+AMP     | 199.4   | 1.68x | 0.024      | 48%   | 0.019GB | YES  | 0.196

→ 76K太小 → GPU未被充分利用 → 加速主要来自减少memory traffic
→ 所有精度都稳定 → 小模型对精度不敏感
→ BF16+AMP最快(1.68x) → 但eval无差异(48%)
→ BF16 native 1.49x → 也足够快+更简单(无AMP overhead)
```

### 1.2 2.28M模型 (关键发现!)

```
精度         | steps/s | 加速  | final loss | eval  | mem     | 稳定 | grad_norm
FP32         | 83.7    | 1.00x | 0.224      | 30%   | 0.065GB | YES  | 3.460
FP16+AMP     | 83.3    | 1.00x | 0.198      | 19%   | 0.065GB | YES  | 2.649
BF16 native  | 103.3   | 1.23x | 0.005      | 39%   | 0.041GB | YES  | 0.058
BF16+AMP     | 102.3   | 1.22x | 0.035      | 10%   | 0.065GB | YES  | 0.547

→ 关键发现:
  1. **BF16 native是最佳选择**: 1.23x加速+39% eval(比FP32的30%更好!)
     → loss 0.005 vs FP32 0.224 → BF16收敛更快!
     → 内存节省37% (0.041 vs 0.065GB)
     → grad_norm极小(0.058 vs FP32 3.46) → 训练更稳定!

  2. **FP16+AMP无加速(1.00x)**: eval反而更差(19% vs 30%)
     → 小模型 → AMP的cast overhead抵消了计算收益
     → GradScaler可能抑制学习 → loss更低但eval更差 → 过拟合!

  3. **BF16+AMP eval最差(10%)**: 加速1.22x但eval仅10%!
     → AMP+BF16的组合 → autocast引入精度损失 → 严重伤害收敛
     → loss看起来低(0.035)但eval极差 → **伪收敛!**

  4. **内存**: BF16 native仅0.041GB → 模型权重+优化器全BF16 → 节省37%
     → FP16+AMP仍0.065GB → AMP只在forward cast, master weights仍FP32!
```

## 二、理论解释

### 2.1 为什么BF16 native胜出?

```
BF16 vs FP16:
  BF16: 8bit exponent + 7bit mantissa → 动态范围同FP32!
  FP16: 5bit exponent + 10bit mantissa → 动态范围仅±65504 → 需GradScaler!

→ BF16不需要GradScaler → 不损失梯度信号 → 训练更稳定
→ BF16精度略低(7bit vs FP16 10bit) → 但训练不需要高精度!
→ 训练关键: 梯度方向(粗略) > 梯度精度(精确)

→ 实测验证:
  BF16 native grad_norm=0.058 → 非常小 → 优化器步长小 → 精确收敛
  FP32 grad_norm=3.460 → 很大 → 优化器步长大 → 收敛慢(100步loss仍0.224)
  → BF16反而收敛更好 → 因为模型+优化器全BF16 → 参数更新更一致!

→ 推理: BF16模型权重+Adam BF16 state → 参数更新全在BF16 → 无FP32↔BF16转换开销
  → FP32: 模型FP32+Adam FP32 → 参数更新FP32 → 但FP32计算慢(83.7 steps/s)
  → FP16+AMP: 模型FP32+forward cast→FP16→backward FP32 → 额外cast开销!
```

### 2.2 为什么AMP反而伤害2.28M模型?

```
AMP的隐含问题:
  1. autocast只在forward cast → backward仍FP32 → 混合精度导致不一致
  2. GradScaler放大loss → 防止FP16梯度underflow → 但也放大了噪声!
  3. master weights FP32 → 参数更新FP32 → 但forward FP16 → 不一致!

→ 小模型(76K): AMP伤害小 → 因为模型太小 → GPU未充分利用 → cast overhead占比小
→ 中模型(2.28M): AMP伤害大 → 因为:
  - cast overhead占比高 → 步骤/秒下降
  - GradScaler抑制了梯度信号 → eval更差
  - FP32 master weights → 与forward BF16不一致 → 伪收敛

→ BF16 native: 模型+优化器全BF16 → 无cast → 无不一致 → 最干净!
```

### 2.3 为什么"伪收敛"(loss低但eval差)?

```
AMP训练的loss低但eval差 → 这是什么?

→ loss = CE(model_output, target) → 训练时AMP cast → loss在低精度下计算
→ eval在FP32下计算 → 精度不同 → 模型在低精度"看起来好" → 但高精度"实际差"
→ 类似"考试作弊" → 训练精度低 → loss低 → 但真实评估(FP32) → eval差!

→ BF16 native无此问题 → 因为训练和eval都在BF16 → 无精度不一致!

→ 教训: **不要用AMP+FP16评估BF16训练的模型** → 会得到虚假的loss!
→ 正确做法: 评估时cast到训练精度 → 或始终用同一精度训练+评估
```

## 三、RTX 4090训练精度推荐

```
RTX 4090 (SM 8.9)训练精度选择:

1. **推荐: BF16 native**
   → 1.23x加速 + 37%内存节省 + 更好收敛 + 更稳定
   → 无AMP → 无GradScaler → 代码更简单
   → PyTorch: model.to(device, dtype=torch.bfloat16)

2. **不推荐: FP16+AMP** (大模型)
   → 无加速 + eval更差 + 伪收敛风险
   → GradScaler抑制梯度 + cast overhead

3. **FP32**: 仅用于debug/验证
   → 最慢但最准确 → 作为基准

4. **FP8训练**: RTX 4090不支持!
   → SM89无FP8 GEMM → 仅H100(SM90+)支持
   → FP8仅用于推理(cuBLAS addmm_cuda支持FP8)

5. **最佳实践**:
   → 生产训练: BF16 native
   → Debug: FP32
   → 超大模型(需ZeRO): BF16 + ZeRO → 内存最低
   → 推理: FP16/BF16 (无精度问题) + FP8量化(推理加速)

→ 与之前AMP实验一致:
  之前76K模型AMP测试: FP16+AMP 2.08x加速 + 24%内存节省
  → 但那是不同设置(lr/batch/steps不同)
  → 本次更严格: 相同seed/lr/steps → BF16 native胜出!
```

## 四、工具

```bash
# GPU服务器运行
cd ~/rollout-infra
source ~/anaconda3/bin/activate llm

# 76K模型
CUDA_VISIBLE_DEVICES=0 python -u tools/mixed_precision_training_benchmark.py --model_size 76k

# 2.28M模型
CUDA_VISIBLE_DEVICES=0 python -u tools/mixed_precision_training_benchmark.py --model_size 2.28m

# 结果
results/mixed_precision_training_results.json (76K)
results/mixed_precision_training_2.28m.json (2.28M)
```

工具: `tools/mixed_precision_training_benchmark.py`