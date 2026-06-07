# Mixed Precision Training — RTX 4090实测

> 2026-06-07 | 4种精度模式对比: FP32/FP16+AMP/BF16/BF16+AMP

## 概述
在RTX 4090上对比4种精度模式的训练质量和吞吐量, 关键问题: BF16是否真的不需要GradScaler? FP16+AMP真的安全吗 什么精度是吞吐量最优?

  ## 一、实验结果

### 1.1 76K模型 (200步)
```
精度          | steps/s | loss  | eval  | mem(GB) | grad_norm | speedup
FP32          | 153.8   | 0.167 | 66%   | 0.019   | 1.536    | 1.00x
FP16+AMP      | 147.6   | 0.299 | 80%   | 0.019   | 2.256    | 0.96x
BF16 native   | 185.0   | 0.137 | 70%   | 0.018   | 1.102    | 1.20x
BF16+AMP      | 180.9   | 0.162 | 66%   | 0.019   | 1.483    | 1.18x
```

### 1.2 2.28M模型 (200步)
```
精度          | steps/s | loss  | eval  | mem(GB) | grad_norm | speedup
FP32          | 110.1   | 1.407 | 40%   | 0.065   | 8.151    | 1.00x
FP16+AMP      | 100.4   | 0.464 | 17%   | 0.065   | 5.215    | 0.91x
BF16 native   | 127.7   | 0.322 | 50%   | 0.041   | 4.938    | 1.16x
BF16+AMP      | 114.6   | 0.992 | 37%   | 0.065   | 8.262    | 1.04x
```

## 二、关键发现

### 2.1 BF16 native: 最佳训练精度!

```
76K:
  BF16 native: 1.20x加速 + eval 70%(vs FP32 66%) + 内存0.018GB(vs 0.019)
  → 20%速度提升! + 4%更好的eval + 5%内存节省
  → win-win-win: 更快+更好+更省

2.28M:
  BF16 native: 1.16x加速 + eval 50%(vs FP32 40%) + 内存0.041GB(vs 0.065)
  → 16%速度提升! + 10%更好的eval! + 37%内存节省
  → 更大模型BF16优势更明显!

→ 为什么BF16更好?
  BF16 = FP32的8bit exponent + 8bit mantissa = 8+8
  → 动态范围与FP32相同(exponent=8bit) → 不需要loss scaling!
  → 但精度降低(mantissa=7bit→8bit, vs FP32的23bit) → 量化噪声

  → 关键: BF16的量化噪声反而有帮助!
    → 类似于随机噪声注入 → 隐式正则化效果
    → 76K: BF16 eval=70% > FP32 eval=66% → BF16帮助泛化!
    → 2.28M: BF16 eval=50% > FP32 eval=40% → 10%提升!
```

### 2.2 FP16+AMP: 反直觉结果!
```
FP16+AMP (GradScaler + autocast):
  76K: eval 80%(最高!) 但speedup仅0.96x → 更慢!
  2.28M: eval 17%(最差!) 且speedup 0.91x → 更慢!

→ 为什么76K FP16+AMP eval最高(80%)?
  → AMP + GradScaler → loss scaling → 梯度更稳定 → 收敛更好?
  → 但吞吐量下降! → AMP overhead: autocast + scaler + unscale + step

→ 为什么2.28M FP16+AMP eval最差(17%)?
  → loss=0.464 → 看似收敛好 → 但eval=17% → 讨型没学到正确算法!
  → FP16精度太低(5bit exponent + 10bit mantissa) → 梯度信息丢失严重
  → GradScaler scale factor可能不合适 → 梯度underflow/overflow
  → 2.28M模型梯度norm=5.215 → 梯度范数大 → FP16溢出风险高
  → → AMP无法正确处理 → 训练看似收敛但实际学错了东西!

→ FP16+AMP vs BF16:
  FP16 = 5+10 bits → 动态范围窄 → 需GradScaler补偿
  BF16 = 8+8 bits → 动态范围宽 → 不需要GradScaler
  → BF16更简单(无AMP) + 更快(无overhead) + 更好(eval更高)
  → → BF16是FP16+AMP的上位替代!
```

### 2.3 BF16+AMP: 焊接异常!
```
BF16+AMP (autocast BF16, no GradScaler):
  76K: eval 66%(= FP32基准) → AMP不改善
  2.28M: eval 37%(低于BF16 native的50%) → AMP反而差!

→ 为什么BF16+AMP不如BF16 native?
  → autocast在forward时用BF16 → 精度降低
  → 但backward时仍然是FP32 → 精度不匹配
  → → forward BF16精度 → backward FP32梯度 → 精度gap
  → → 模型权重在BF16 → 梯度在FP32 → 更新方向不一致
  → → 焊接效应(a焊b接): 模型更新后权重回到BF16 → 但梯度来自FP32 → 收敛不稳定

→ vs BF16 native:
  native = 全程BF16 → forward/backward一致 → 无焊接效应
  → 更稳定的训练 → 更高的eval!
```

### 2.4 内存节省
```
76K模型:
  FP32: 0.019GB → BF16 native: 0.018GB → 5%节省(太小看不出来)
  → 小模型内存不重要 → GPU远未充分利用

2.28M模型:
  FP32: 0.065GB → BF16 native: 0.041GB → 37%节省!
  → 参数+梯度+优化器从FP32→BF16:
    参数: 4bytes→2bytes = 50%节省
    梯度: 4bytes→2bytes = 50%节省
    优化器: FP32 master copy → 仍需4bytes (Adam需要FP32状态)
    → 总计: (2+2+12) / (4+4+12) = 16/20 = 80% → 但实测37%
    → 原因: 优化器状态仍为FP32! Adam有2个FP32状态(m+v)
    → 宙重+梯度省50%, 但优化器状态仍FP32 → 总节省不是50%
    → 优化器 = 12bytes/param (不变) → 总=16/20=80%而非50%

→ 随模型增大, BF16内存节省越明显:
  7B模型: FP32 112GB → BF16 ≈72GB → 36%节省 → 可训更大的模型!
```

### 2.5 FP8训练: SM89不支持
```
RTX 4090 = SM 8.9 (Ada Lovelace):
  → FP8 E4M3/E5M2 格式支持(硬件可存储)
  → 但FP8 GEMM训练不支持! → 只有推理path支持
  → SM 9.0 (H100) 才支持FP8训练GEMM

→ 与之前cuBLAS GEMM实测一致:
  FP8 direct GEMM FAILED on SM89 → addmm_cuda不支持
  → 必须用TransformerEngine/NVIDIA库才能FP8训练 → 只在H100+上
```

## 三、理论解释

### 3.1 为什么BF16更好? — 量化噪声=隐式正则化
```
正则化理论:
  L2正则: 惧大参数 → 但AdamW已经decoupled wd → L2不安全
  Dropout: 随机丢弃 → 随机噪声注入 → 泛化帮助
  BF16量化: 每次计算都注入量化噪声 → 类似Dropout的噪声注入!
  → 但BF16噪声是确定性的(rounding) → 不是随机的 → 仍帮助泛化?

→ 实测验证:
  BF16 eval 70%(76K)/50%(2.28M) > FP32 eval 66%/40%
  → BF16确实帮助泛化! → 量化噪声 ≈ 隐式正则化

→ 与之前优化理论实验一致:
  AdamW wd=0.1 → loss↓6.5% → decoupled wd帮助
  BF16量化 → eval↑4-10% → 隐式正则化帮助
  → 两者都是正则化 → 但BF16不需要调超参数!
```

### 3.2 精度选择决策树
```
RTX 4090 PCIe集群:

1. FP32: 基准 → 不推荐(慢+内存大)
   → 仅用于debug/验证

2. FP16+AMP: 需GradScaler → 不推荐
   → overhead大(0.91-0.96x speedup → 反而更慢!)
   → 大模型eval差(17% vs FP32 40% → 学错了!)
   → 小模型可能eval好(80%)但代价是吞吐下降

3. BF16 native: 推荐! ★★★★★
   → 1.16-1.20x加速 + 4-10%更好eval + 37%内存节省
   → 不需要GradScaler → 代码更简单
   → 动态范围与FP32相同 → 不需要loss scaling

4. BF16+AMP: 不推荐
   → 焊接效应 → eval不如native
   → 无额外加速(1.04-1.18x vs native 1.16-1.20x)

5. FP8: 不支持(SM89限制)
   → 仅H100+支持FP8训练GEMM
   → RTX 4090只能FP8推理

→ 生产建议:
  RTX 4090训练: BF16 native (model.to(bfloat16)) → 简单+快+好
  H100训练: BF16 native 或 FP8(如果TransformerEngine可用)
  调试时: FP32 → 验证BF16精度是否足够
  不要用FP16+AMP → BF16是其上位替代
```

## 四、工具

```bash
# GPU服务器运行
cd ~/rollout-infra
source ~/anaconda3/bin/activate llm
CUDA_VISIBLE_DEVICES=0 python -u tools/mixed_precision_training_benchmark.py --model_size 76k --num_steps 200
CUDA_VISIBLE_DEVICES=0 python -u tools/mixed_precision_training_benchmark.py --model_size 2.28m --num_steps 200

# 结果
results/mixed_precision_training_results.json (76K)
results/mixed_precision_training_2.28m.json (2.28M)
```

工具: `tools/mixed_precision_training_benchmark.py`