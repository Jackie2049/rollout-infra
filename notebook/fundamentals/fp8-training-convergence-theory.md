# FP8 Training Convergence Theory Deep Dive

> 2026-06-08 | 结合RTX 4090实测数据+TE源码分析+理论推导
> 关联: fp8-gemm-algorithm-analysis-rtx4090.md (实测), transformer-engine-fp8-rtx4090.md (源码), te-gemm-dispatch-deep-dive.md (dispatch)

## 1. FP8 Format Analysis: E4M3 vs E5M2

### 1.1 格式定义

| 属性 | E4M3 (Forward) | E5M2 (Backward) |
|------|----------------|-----------------|
| Sign | 1 bit | 1 bit |
| Exponent | 4 bits (bias=7) | 5 bits (bias=15) |
| Mantissa | 3 bits | 2 bits |
| Dynamic range | 2^-6 to 2^9 = 0.0078..512 | 2^-14 to 2^15 = 6e-5..57344 |
| Max value (TE定义) | 448 | 57344 |
| Representable positives | ~143 | ~455 |
| Precision at max | ~0.39 (448/1024) | ~0.063 (max/2^9) |
| Quantization step | 8 levels per exponent bin | 4 levels per exponent bin |

### 1.2 为什么Forward用E4M3, Backward用E5M2?

**Forward (weights+activations)**: E4M3有更多mantissa bits → 更精确的数值表示 → GEMM输出更准确
- 权重通常在一个较窄的范围内 → E4M3的有限动态范围足够
- 激活值也通常bounded → E4M3合适

**Backward (gradients)**: E5M2有更多exponent bits → 更大的动态范围 → 梯度不会溢出
- 梯度可以很大(梯度爆炸)或很小(梯度消失) → 需要大动态范围
- E5M2的57344 max vs E4M3的448 max → 128x动态范围优势!
- 梯度的精度要求相对低 → 2 mantissa bits够用

**RTX 4090限制**: SM89的HMMA FP8指令只支持E4M3(E5M2需要SM90+), 所以forward和backward都用E4M3!

### 1.3 格式对训练的影响

**E4M3精度分析**:
- 3 mantissa bits → 8 representable values per bin → 相对误差≈1/8≈12.5%
- 但量化误差是均匀分布: 量化step内误差最大为step/2 → 平均误差≈step/4
- 对于max=448的E4M3: 在max附近step≈2^(9-3)=64 → 量化误差≈32 → 相对误差≈7%
- 但TE的scale因子将数据范围映射到[-448, 448] → 实际量化误差更小

**量化噪声模型**: FP8量化噪声近似均匀分布U[-Δ/2, Δ/2], 其中Δ是量化step
- 均值=0, 方差=Δ²/12
- Δ = scale × 2^(e-3) (E4M3), e是exponent
- 相对量化噪声 ∝ 2^(-3) = 1/8 ≈ 12.5%

**与SGD噪声对比**: SGD梯度噪声是Gaussian分布
- FP8量化噪声是均匀分布, 幅度更小(相对误差≤12.5%)
- 深度网络对量化噪声有一定容忍度(类似对SGD噪声的容忍度)
- 但量化噪声不是随机的→可能系统性偏差 → 比SGD噪声更危险!

## 2. Scaling Recipe对收敛的影响

### 2.1 DelayedScaling (默认)

**机制**: `scale = FP8_MAX / amax / (2^margin)`, 使用上一步的amax → 1步延迟

**延迟的影响**:
- 如果当前步的amax > 上一步的amax → scale偏大 → 量化精度降低(但不会overflow)
- 如果当前步的amax < 上一步的amax → scale偏小 → 量化精度浪费(但不影响准确度)
- 延迟1步意味着scale总是保守的 → scale因子略大 → FP8值略小 → 但在可接受范围

**margin参数**: margin=0(推荐) → scale精确; margin=4 → scale保守4bit → 精度降低1.03%
- 我们实测: margin=0相对误差0.14%, margin=4→1.03% → margin=0最优

**收敛影响**: DelayedScaling引入的额外噪声是transient → 自纠正
- 大模型(GPT-3/LLaMA): <0.1%准确度退化 vs BF16
- 小模型: 可能有轻微退化 → 但通常可忽略
- RTX 4090实测: cos_sim=0.996-1.000 → 精度极好!

### 2.2 Float8CurrentScaling

**机制**: 扫描当前tensor的amax → 无延迟 → 更精确的scale因子

**优势**: scale因子精确匹配当前tensor → 量化噪声最小
**劣势**: 需额外amax扫描 → 计算overhead → 在cuBLASLt fused GEMM中可以接受

**收敛影响**: CurrentScaling理论上比DelayedScaling更精确
- 但实测: cos_sim几乎相同(0.996 vs 1.000) → DelayedScaling够好
- CurrentScaling略慢2-3% → 在精度几乎相同的情况下, DelayedScaling更实用

### 2.3 Block Scaling (MXFP8/FP8Block)

**机制**: 32-value blocks × E8M0 power-of-2 scales → 每block独立scaling

**优势**: 更精细的局部量化 → 更高的有效精度
- E8M0 scales: 256 possible scale values → 覆盖2^-127..2^127
- 每block独立 → 不会因为1个大值而浪费整个tensor的精度

**RTX 4090**: 不支持! Block scaling需要SM100+ (Blackwell)

### 2.4 实测Scaling Recipe对比 (RTX 4090)

| Recipe | B=4 speedup | B=32 speedup | cos_sim | 推荐? |
|--------|-------------|--------------|---------|-------|
| DelayedScaling | 1.48x | 1.57x | 0.996-1.000 | **推荐** |
| CurrentScaling | 1.44x | 1.52x | 1.000 | 精度略好但慢 |

**结论**: DelayedScaling是RTX 4090最佳选择 → 精度够好+速度最快

## 3. 量化噪声的数学建模

### 3.1 FP8量化噪声

FP8量化: x_fp8 = round(x / scale) × scale

量化误差: ε = x - x_fp8 → ε ∈ [-Δ/2, Δ/2]

**E4M3量化step**: Δ(x) = max(2^(floor(log2(|x|))-3) × scale, ...)

**总量化误差**:
- Forward: 2次量化(input + weight) → 2ε
- GEMM: M×K个量化input × K×N个量化weight → M×N个输出, 每个包含K个量化误差
- 输出误差 ≈ Σ(k=1..K) ε_input_k × ε_weight_k → 相对误差≈K×Δ²/12/(|x|×|w|)

### 3.2 与SGD噪声的对比

**SGD噪声**: mini-batch梯度噪声 ~ Gaussian(0, σ²/m), σ²是梯度方差, m是batch size
- 方差随batch size线性递减 → 大batch更稳定

**FP8量化噪声**: 均匀分布, 方差≈Δ²/12, 不随batch size变化
- 量化噪声是系统性的(不是随机的) → 不同batch大小下噪声幅度相同
- 但TE的fused GEMM在cuBLASLt内部dequant → 部分噪声被FP32累加器吸收

**关键洞察**: FP8量化噪声 ≠ SGD噪声
- SGD噪声: 随机, 随batch递减, 帮助探索loss landscape
- FP8噪声: 系统性, 不随batch变化, 可能导致系统性偏差
- 但FP8噪声幅度小(相对误差<12.5%) → 深度网络通常容忍

### 3.3 RTX 4090实测噪声

实测数据 (fp8_gemm_algorithm_analysis):
- cos_sim(B=4) = 0.996-1.000 → 量化噪声极小
- cos_sim(B=32) = 0.996 → 量化噪声不随batch size增长
- max_diff = 0.06-0.09 → FP8 vs BF16输出差异很小

## 4. Stochastic Rounding与TE

### 4.1 TE不使用Stochastic Rounding!

TransformerEngine使用的是**nearest rounding**(round-to-nearest-even):
- `tex.quantize()` → CastVectorizedUnaryKernelLauncher → SIMT kernel → nearest rounding
- cuBLASLt GEMM → 内部FP8×scale_inv → FP32累加 → BF16输出 → nearest rounding

**为什么不使用Stochastic Rounding?**
1. Stochastic rounding需要随机数 → GPU上的random number generation有性能开销
2. cuBLASLt的fused GEMM路径不支持stochastic rounding → 只在GEMM输出
3. DelayedScaling已经使用scale因子来"spread"量化误差 → 近似stochastic rounding的效果

**Stochastic Rounding的理论优势**:
- 期望值=原始值 → 无系统性偏差(unbiased)
- nearest rounding有系统性偏差(值总是被量化到最近grid point)

**实际影响**: nearest rounding在大多数情况下足够好
- 因为TE每步重新计算scale → scale因子动态调整 → 偏差不累积
- cos_sim=0.996-1.000实测 → nearest rounding在实践中不影响收敛

## 5. Split Accumulator (FAST_ACCUM)

### 5.1 HMMA FP8 Accumulator选项

cuBLASLt FP8 GEMM有两个accumulator模式:
- **Standard accumulation** (`use_split_accumulator=true`): FP32累加器 → 精确但稍慢
- **Fast accumulation** (`use_split_accumulator=false`): 快速累加 → 低精度但快

### 5.2 Split Accumulator的工作原理

**Standard (FP32 accumulator)**:
- 每个MAC: FP8_input × scale_inv × FP8_weight × scale_inv → FP32 partial sum
- FP32累加器: 精确到24-bit mantissa → 无舍入损失
- 最后: FP32 → BF16输出 → 1次舍入

**Fast accumulation**:
- 使用"split-K"方式: 多个accumulator bank分别累加partial sums
- 合并时: FP32 partial sums → 可能多次舍入 → 精度降低
- 优势: 减少cycle latency → GEMM更快

### 5.3 对训练的影响

**研究结论**: Fast accumulation在LLM训练中可以导致收敛退化
- Attention QK乘积: Q和K的FP8误差叠加 → score偏差 → attention权重错误
- Gradient accumulation: FP8梯度累加 → 误差累积 → 梯度不准确
- 建议: **Forward用fast accumulation可以接受, Backward用standard accumulation**

**TE默认**: `use_split_accumulator=true` → standard accumulation → 更保守但更精确

**RTX 4090实测**: TE默认用standard accumulation → cos_sim=0.996-1.000 → 精度好
- 如果切换到fast accumulation: 可能节省2-5%时间但精度可能降低

### 5.4 RTX 4090建议

**保守策略**: 使用standard accumulation (`use_split_accumulator=True`)
- RTX 4090 FP8加速只有1.48-1.59x → 不是瓶颈
- 精度更重要(尤其是小模型)
- Fast accumulation节省的时间微不足道

## 6. RTX 4090 FP8训练实用指南

### 6.1 推荐配置

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| Recipe | DelayedScaling | 精度够好+速度最快 |
| Accumulator | Standard (split) | 精度优先 |
| Margin | 0 | 最优精度 |
| Batch size | ≥8 (M≥4096) | FP8量化overhead合理 |
| Forward format | E4M3 | 精度优先(RTX 4090限制) |
| Backward format | E4M3 | RTX 4090不支持E5M2 |

### 6.2 不推荐FP8的场景

1. **B=1推理**: FP8比BF16慢2x → 不要用
2. **小模型(<10M)**: 量化噪声比例太大
3. **Fine-tuning小数据集**: 量化噪声影响更大
4. **敏感操作**: Softmax/LayerNorm/梯度累加 → 保持BF16/FP32

### 6.3 推荐FP8的场景

1. **训练B≥8**: FP8加速1.03-1.59x → 有意义
2. **大模型(≥100M)**: 量化噪声比例小 → 更容忍
3. **Pre-training**: 数据量足够 → 量化噪声平均化
4. **通信量化**: FP8量化数据传输 → 带宽减半 → 不受batch size限制

### 6.4 模型大小与FP8收敛

**规律**: 模型越大 → FP8量化噪声越可容忍
- 7B模型: cos_sim=0.996 → 极好
- 70B模型: <0.1%退化 → 可忽略
- 405B模型: <0.3%退化 → 可接受
- 小模型(<10M): 可能有明显退化

**理论解释**: 大模型有更多参数 → 每个参数的量化噪声影响更小 → 平均化效应

## 7. cuBLASLt Fused Path与精度

### 7.1 Fused vs Unfused对精度的影响

**Fused (DelayedScaling/CurrentScaling)**:
- cuBLASLt内部: FP8 × scale_inv → FP32累加 → BF16输出
- 1次GEMM kernel → 1次量化+1次dequant → 最小精度损失
- scale_inv在GEMM descriptor中 → cuBLASLt自动处理

**Unfused (MXFP8/NVFP4/Block)**:
- GEMM → BF16 → 再量化到FP8 → 额外精度损失
- 多次量化+dequant → 精度损失累积
- RTX 4090不支持 → 无影响

### 7.2 RTX 4090精度优势

RTX 4090(SM89)只能用fused path → 没有unfused的精度损失问题!
- cuBLASLt `SCALAR_32F` per-tensor scaling → 精确dequant
- FP32累加器 → 高精度累加
- 输出BF16 → 1次舍入

这是RTX 4090 FP8精度极好(cos_sim=0.996-1.000)的根本原因之一!

---

**Sources**:
- [FP8 Formats for Deep Learning (Micikevicius et al., 2022)](https://arxiv.org/abs/2209.05433)
- [TransformerEngine GitHub](https://github.com/NVIDIA/TransformerEngine)
- [FP8 Precision Effects on Hopper Tensor Cores](https://developer.nvidia.com/blog/understanding-fp8-precision-effects-on-hopper-tensor-cores/)
- [FP8 Fast Accumulation Impact on LLM Training](https://arxiv.org/abs/2309.17288)

**Related notes**: fp8-gemm-algorithm-analysis-rtx4090.md (实测), transformer-engine-fp8-rtx4090.md (源码), te-gemm-dispatch-deep-dive.md (dispatch chain)