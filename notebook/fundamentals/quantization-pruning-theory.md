# Quantization & Pruning Theory: LLM压缩与加速全景

> 2026-06-08 | 从INT4到FP8, 从magnitude到Wanda, LLM压缩理论+RTX 4090实测
> 基于: 量化文献(AWQ/GPTQ/SmoothQuant/FP8), 剪枝文献(SparseGPT/Wanda), RTX 4090实测数据
> 关联: fp8-quantization.md (FP8格式), fp8-training-convergence-theory.md (FP8训练收敛), transformer-engine-fp8-rtx4090.md (TE FP8)

## 0. 为什么LLM需要压缩?

**LLM推理瓶颈**: Decode是memory-bound → KV cache + 权重带宽是瓶颈!

```
7B模型 Decode (B=32):
  权重读取: 14GB × 8bit/byte = 112GB (每次decode全部读取!)
  KV读取: B×S×H×2×2bytes = 32×512×2560×4 = 66.5GB (GQA expand后)

量化节省:
  INT4 weight-only: 权重112GB→28GB → 75%带宽省 → 0.87-1.08x速度(几乎免费!)
  INT8 KV cache: KV 66.5GB→33.2GB → 50%带宽省 → 1.00x速度+cos_sim=1.0(完美!)
```

## 1. 量化基础理论

### 1.1 量化数学定义

```
量化: x_float → x_quant = clamp(round(x_float / scale), -Qmax, Qmax)
反量化: x_dequant = x_quant × scale

均匀量化: scale = (xmax - xmin) / (Qmax - Qmin)
非均匀量化: scale = 不同区间不同 → 更精确但硬件不友好

对称量化: xmin = -xmax → scale = xmax / Qmax → 简单但浪费(负区间可能空)
非对称量化: xmin ≠ xmax → scale + zero_point → 更精确但多一个参数
```

### 1.2 量化误差分析

```
量化噪声: q(x) = x + ε, ε ∈ [-scale/2, scale/2] (均匀分布)
相对误差: |ε/x| ≈ scale/2x → 小值误差大!

关键洞察: activation分布是幂律分布 → 少数outlier值很大 → 大scale → 小值误差大!
→ 这是LLM量化困难的核心原因!

解决: AWQ(保护大activation对应的weight) / SmoothQuant(把activation outlier迁移到weight)
```

### 1.3 RTX 4090实测对比

| 方法 | 类型 | 加速 | 精度(cos_sim) | 内存省 | 适用场景 |
|------|------|------|-------------|--------|---------|
| INT4 weight-only | 推理量化 | 0.87-1.08x | 好 | 75%权重内存 | 推理部署 |
| INT8 KV cache | KV量化 | **1.00x** | **1.0** | 50% KV内存 | 推理decode |
| FP8 Python dequant | 训练量化 | 0.4-0.67x ❌ | 好 | — | 仅适合通信量化 |
| TE FP8 DelayedScaling | 训练量化 | **1.48-1.59x** ✅ | 0.996 | — | 训练B≥4 |
| TE FP8 CurrentScaling | 训练量化 | **1.44-1.55x** ✅ | 0.996 | — | 训练B≥4 |

**关键发现**: FP8加速取决于路径!
- Python dequant: 慢1.5-2.5x → 不适合推理/训练
- TE fused kernel: 快1.48-1.59x → cuBLASLt内部dequant → 零Python开销!

## 2. Weight-only量化: AWQ vs GPTQ

### 2.1 GPTQ: 基于Hessian的二次近似

```
核心思想: 量化权重w_i时, 补偿量化误差对后续权重的影响

数学推导:
  Loss ≈ Σ_i (w_i - q(w_i))² / H_ii + Σ_{i>j} 2(w_i-q(w_i))(w_j-q(w_j)) / H_ij
  → 量化w_i → 误差δ_i → 对后续w_j补偿: w_j -= δ_i × H_ij / H_ii

算法流程:
  1. 计算Hessian逆(用少量校准数据)
  2. 按列顺序量化 → 每量化一列 → 补偿剩余列
  3. 分组量化(group_size=128) → 每组独立scale

优点: 快速(175B模型3-4小时量化), 精度好
缺点: 量化顺序依赖 → 不同顺序不同结果, activation仍FP16
```

### 2.2 AWQ: Activation-aware权重保护

```
核心思想: 不是所有权重同等重要 → 大activation对应的权重更重要!

观察: activation有outlier → 对应channel的权重对输出影响大
  → 保护这些"salient weight" → 用更小的量化group/mix-precision

数学: 输出 = W × X → |output_j| = Σ_i |w_ij × x_i|
  → 如果x_i是outlier → w_ij即使小也对输出贡献大
  → 但如果w_ij本身也大 → 保护它!

AWQ策略:
  1. 识别salient channel: activation magnitude大的channel
  2. 对salient channel用per-channel scale放大 → 量化误差相对更小
  3. 其他channel正常量化 → 整体perplexity更好

优点: 比GPTQ perplexity更好(W4A16), 无需Hessian计算
缺点: 需校准数据识别salient channel, activation仍FP16
```

### 2.3 实际部署对比

| 方法 | 4-bit Perplexity(LLaMA-7B) | 量化速度 | 推理框架支持 |
|------|---------------------------|---------|------------|
| RTN(round-to-nearest) | 7.48 | 最快 | 基线 |
| GPTQ | 6.93 | 快(3-4h/175B) | vLLM/AutoGPTQ |
| AWQ | 6.72 | 中等 | vLLM/LMDeploy |
| SqueezeLLM | 6.68 |慢 | 专用 |
| AQLM | 6.41 |慢 | 专用 |

**2025趋势**: Marlin kernel使AWQ和GPTQ推理速度基本持平 → 选质量(AWQ)而非速度

## 3. Activation量化: SmoothQuant

### 3.1 为什么activation量化困难?

```
观察: LLM activation有outlier → 少数channel值很大(>100x平均)
  → 大scale → 小值量化误差大 → 精度灾难!

传统INT8(W8A8): activation outlier → 量化后小值被抹平 → perplexity爆炸
GPTQ/AWQ: 只量化weight → activation仍FP16 → 不是真正的INT8推理!
```

### 3.2 SmoothQuant核心: 数学等价的迁移

```
核心: Y = X × W → 可以等价变换:
  Y = (X × s^-1) × (s × W) = X_smooth × W_smooth

  其中s是per-channel平滑因子:
  s_j = max(|x_j|)^α / max(|w_j|)^(1-α)  (α通常=0.5)

效果:
  X_smooth: outlier被s^-1缩小 → 分布更均匀 → INT8量化精度好!
  W_smooth: 权重被s放大 → 但权重本身分布窄 → INT8量化仍然OK!

→ 数学等价变换 → 精度不变 → 但activation可以INT8量化了!
```

### 3.3 SmoothQuant推理优势

```
W8A8 = INT8 weight + INT8 activation:
  → cuBLASLt INT8 GEMM → HMMA.INT8 → 2x计算吞吐 vs FP16 HMMA
  → 真正硬件加速! 不是weight-only那种"带宽省但计算不变"

RTX 4090: HMMA.16816支持INT8 → W8A8理论上2x计算加速
但实际: activation量化overhead + scale开销 → 实测1.3-1.5x
```

## 4. KV Cache量化

### 4.1 INT8 KV量化实测(RTX 4090)

```
实测数据(7B模型, GQA H=5, d=128):
  INT8 KV: cos_sim=1.0(完美!), 速度=1.00x(零开销!), KV内存省50%

为什么INT8 KV这么好?
  1. KV per-token量化 → scale很小(每token独立) → 精度好
  2. Decode时KV只读取 → 量化在写入时做 → 读取时只需×scale → 几乎免费
  3. KV cache是memory-bound → 量化减半带宽需求 → 但我们实测1.00x而非1.5x
     → 原因: RTX 4090 KV读取量还不够大(7B小模型) → 带宽瓶颈不够严重

更大模型(70B): KV量化收益更大 → decode完全memory-bound → 50%带宽省→接近1.5x!
```

### 4.2 FP8 KV量化

```
FP8 KV: 用E4M3或E5M2存储KV → 1 byte vs BF16 2 bytes → 50%省

vLLM/FlashInfer支持FP8 KV:
  → KV cache存储FP8 → decode时反量化到BF16再计算
  → 但attention计算仍FP32 → 精度影响小

RTX 4090实测FP8 dequant慢1.5-2.5x → Python overhead →
  但FlashInfer fused kernel可能消除这个问题 → 需进一步验证
```

## 5. FP8训练量化: 从Python到Fused Kernel

### 5.1 FP8训练量化流程

```
BF16 → FP8(quantize) → FP8 GEMM(cuBLASLt) → BF16(dequantize) → 输出

关键: dequantize在哪里做?
  Python dequant: FP8 GEMM → FP8 output → Python × scale → BF16 → 慢!
  Fused dequant: cuBLASLt内部 × scale → 直接BF16输出 → 快!

TE fused kernel:
  quantize: tex.quantize() → C++ CastVectorizedKernelLauncher → ~0.32ms恒定
  GEMM: cuBLASLt FP8 GEMM → HMMA FP8 E4M3 → SCALAR_32F per-tensor dequant
  output: 直接BF16 → 无Python开销 → 1.48-1.59x加速!
```

### 5.2 量化开销crossover分析

```
Per-layer timing(RTX 4090, 7B模型):
  Quantize overhead: 恒定0.32-0.40ms (不随batch变化!)
    → B=1: 占FP8时间77% → 灾难性慢(0.23-0.47x)
    → B=16: 仅20% → 可接受
    → B=32: 仅13% → 加速1.27-1.71x

Crossover(量化开销=GEMM加速收益):
  gate/up_proj(N=10240): B=4 → 开始加速
  qkv_proj(N=5120): B=8 → 开始加速
  out_proj(N=2560): B=16 → 开始加速

→ 小维度layer需要更大batch才能加速!
→ 规律: crossover batch ≈ N/256 (粗估)
```

## 6. 剪枝理论

### 6.1 剪枝分类

```
非结构化剪枝: 删除单个weight → 稠密矩阵变稀疏 → 需稀疏硬件支持
结构化剪枝: 删除整行/列/head/layer → 矩阵变小 → 标准硬件直接加速
半结构化(N:M)剪枝: N个中M个为零 → 2:4 = 50%稀疏 → GPU sparse tensor core!

N:M稀疏在SM89(RTX 4090):
  2:4 sparsity → HMMA.SPARSE.16832 → 理论2x计算吞吐!
  但实际: 精度损失+稀疏模式约束 → 1.3-1.5x实际加速
```

### 6.2 经典方法对比

| 方法 | 类型 | 50%稀疏度精度 | 是否需要重训练 | 硬件加速 |
|------|------|-------------|-------------|---------|
| Magnitude | 非结构化 | 差(低4-8pp) | 否 | 需稀疏kernel |
| SparseGPT | 非结构化 | 好(97%保留) | 否(one-shot) | 需稀疏kernel |
| Wanda | 非结构化 | 好(96%保留) | 否(one-shot) | 需稀疏kernel |
| 2:4半结构化 | 半结构化 | 好(97-99%) | 否或LoRA | Sparse Tensor Core |
| LLM-Pruner | 结构化 | 中(需LoRA恢复) | LoRA微调 | 直接加速 |
| ShortGPT | 层级结构化 | 好(冗余层检测) | 可选 | 直接加速 |

### 6.3 SparseGPT: Hessian-based one-shot剪枝

```
核心: 与GPTQ类似的二次近似, 但用于剪枝而非量化

算法:
  1. 计算Hessian: 用校准数据算Fisher信息矩阵
  2. 逐行剪枝: 选择最小|w_i|×|H_ii|的权重 → 删除
  3. 补偿: 保留权重调整以弥补剪枝误差 → Hessian-guided更新
  4. Iterative: 多轮剪枝+补偿 → 精度逐步恢复

175B模型: 50%稀疏 → 97%零-shot精度保留 → 3-4小时完成!
```

### 6.4 Wanda: Weight×Activation pruning

```
核心: 剪枝标准 = |w_ij| × ||x_j||_2 (weight大小 × activation norm)

直觉: 小weight不重要, 但如果它对应大activation → 还可能重要!
  → |w| × ||x|| → 综合考虑weight和activation的影响

算法:
  1. 校准数据 → 计算||x_j||_2 per input channel
  2. 剪枝标准: score_ij = |w_ij| × ||x_j||_2
  3. 每行保留top-k score → 删除其余 → 无需Hessian计算!

优点: 简单快速, 无需Hessian, 精度好
缺点: 仍是非结构化 → 需稀疏kernel才能加速
```

### 6.5 RTX 4090剪枝可行性

```
RTX 4090(SM89)稀疏支持:
  2:4 sparse tensor core → HMMA.SPARSE.16832 → FP16/BF16/INT8
  → 50%稀疏 → 理论2x计算加速 → 实际约1.3-1.5x

但LLM推理是memory-bound → 剪枝减半权重带宽 → 但KV cache仍是瓶颈
  → 2:4剪枝+INT4量化 → 75%权重带宽省+50%计算省 → 理论最优组合!

实际部署:
  vLLM支持2:4稀疏INT4量化(Marlin kernel) → 但生态不成熟
  → 目前INT4 weight-only更实用(成熟+简单)
```

## 7. 量化+剪枝组合策略

### 7.1 组合优化层级

```
Level 1: INT4 weight-only → 75%权重内存省 → 0.87-1.08x → 最成熟
Level 2: INT8 KV cache → 50% KV内存省 → 1.00x → 简单实用
Level 3: SmoothQuant W8A8 → 2x计算加速 → 需INT8 kernel → 中等成熟
Level 4: TE FP8训练 → 1.48-1.59x → B≥4加速 → 训练场景
Level 5: 2:4剪枝+INT4 → 87.5%权重省+2x计算 → 生态不成熟 → 未来方向
```

### 7.2 RTX 4090推理部署决策

```
7B推理(RTX 4090单卡):
  → INT4 weight-only + INT8 KV cache → 组合最优!
  → 权重: 14GB→3.5GB(75%省), KV: 省50% → 总内存大幅省
  → 速度: INT4 0.87-1.08x + INT8 KV 1.00x → 接近无损!

70B推理(RTX 4090需要2卡+TP):
  → INT4 weight-only → 140GB→35GB → 2卡可能fit!
  → INT8 KV → 进一步省50% KV → 加长context!

训练(RTX 4090):
  → BF16 FSDP → 基线
  → BF16 + TE FP8 → B≥4时1.48-1.59x加速 → 已验证!
  → INT4训练? → 不推荐(精度差+梯度量化困难)
```

## 8. 关键理论洞察

### 8.1 量化为什么有效?

```
1. LLM权重/activation分布集中在窄范围 → 量化误差相对小
2. 大多数weight接近0 → 量化噪声对输出贡献小
3. 量化噪声 ≈ 均匀分布 → 类似SGD噪声 → 模型能容忍!
4. FP8噪声<12.5%相对误差 → 训练时被SGD噪声淹没 → 几乎无影响

但: outlier是量化灾难 → SmoothQuant迁移是关键突破!
```

### 8.2 剪枝为什么有效?

```
1. 过参数化: 大模型70%+权重接近0 → 删除不影响输出
2. 冗余表示: 多头/多层 redundancy → 结构化剪枝删除冗余
3. Lottery Ticket假设: 子网络就能达好性能 → 剪枝找到子网络

但: 高稀疏度(>70%)需要重训练 → one-shot方法最多50-60%安全剪枝
```

### 8.3 量化 vs 剪枝选择

```
推理加速:
  → 量化优先! 简单+成熟+硬件友好+精度好
  → INT4/INT8量化 → 几乎免费+内存大省
  → 剪枝 → 需稀疏kernel+生态不成熟 → 2:4半结构化最有前景

训练加速:
  → FP8量化优先! TE fused kernel → 1.48-1.59x → 已验证
  → 剪枝训练 → 目前不实用 → 主要是训练后剪枝
  → 混合精度 → BF16权重+FP8 GEMM → 速度+精度兼顾
```

---

**Sources**:
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [GPTQ Paper](https://arxiv.org/abs/2208.00938)
- [SmoothQuant Paper](https://arxiv.org/abs/2211.10238)
- [SparseGPT Paper](https://arxiv.org/abs/2301.00935)
- [Wanda Paper](https://arxiv.org/abs/2402.01873)
- [FP8 Specification](https://arxiv.org/abs/2209.05433)
- [TransformerEngine](https://github.com/NVIDIA/TransformerEngine)
- [vLLM Quantization](https://docs.vllm.ai/en/latest/features/quantization.html)

**Related notes**: fp8-quantization.md (FP8格式), fp8-training-convergence-theory.md (FP8收敛), transformer-engine-fp8-rtx4090.md (TE), fp8-gemm-algorithm-analysis-rtx4090.md (FP8 GEMM)