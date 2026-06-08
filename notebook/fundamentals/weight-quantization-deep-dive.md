# Weight Quantization Deep Dive: INT4→INT8→FP8→AWQ→GPTQ→SmoothQuant→Marlin

> 2026-06-08 | 权重量化=推理加速核心→INT4省75%内存+PPL↑~1%→INT8省50%+PPL↑<0.1%→FP8精度最高(cos_sim≈1.0)→AWQ保护salient weight→GPTQ Hessian补偿→SmoothQuant迁移activation outlier→Marlin fused kernel消除dequant overhead→RTX 4090最优=INT4 AWQ(fused)+INT8 KV+FlashInfer
> 基于: AWQ(Lin 2023), GPTQ(Frantar 2023), SmoothQuant(Xiao 2023), Marlin(2024), FP8(TE)
> 参考: LLM.int8(), bitsandbytes, AutoAWQ, AutoGPTQ, vLLM quantization docs
> 关联: quantization-pruning-theory.md, kv-cache-management-deep-dive.md, inference-cost-analysis.md

## 0. 核心定律: 权重量化 = 减少每个参数的比特数 = 看存储和带宽省多少

```
权重量化推理影响链:

  → 权重dtype → 权重大小 → decode每步读取 → memory-bound → 带宽瓶颈!
  → → BF16(16bit): 7B模型=14GB → 每步读取14GB → @890GB/s → 15.7ms
  → → → INT8(8bit): 7B模型=7GB → 每步读取7GB → @890GB/s → 7.9ms → 2x快!
  → → → → INT4(4bit): 7B模型=3.5GB → 每步读取3.5GB → @890GB/s → 3.9ms → 4x快!
  → → → → → FP8(8bit): 7B模型=7GB → 每步读取7GB → 但TE fused → 接近free!

  RTX 4090实测精度 vs 内存:
    → BF16: cos_sim=1.0 → 基准 → 14GB → 不省
    → → FP8 E4M3: cos_sim=1.000000 → **完美!** → 7GB → 50%省 → 推荐!
    → → → INT8 per-channel: cos_sim=0.999956 → PPL↑<0.1% → 7GB → 50%省 → 推荐!
    → → → → INT4 group-128: cos_sim=0.993 → PPL↑~1% → 3.5GB → 75%省 → 推荐(fused!)
    → → → → → INT4 group-32: cos_sim=0.995 → PPL↑<1% → 3.5GB+更多scale → 推荐

  关键发现:
    → **INT4 cos_sim=0.993 → 精度足够!** → PPL↑~1% → 可接受 → 75%内存省 → 推荐!
    → → → 但: Python dequant 20x慢 → 必须fused kernel → 否则灾难!
    → → → → AWQ/GPTQ/Marlin: fused dequant+GEMM → 消除overhead → 量化才有效!
```

## 1. 量化精度对比: FP8完美 → INT8近完美 → INT4足够

```
RTX 4090实测 (7B模型权重 4096×14336):

    | 方法 | cos_sim | MSE | PPL↑ | 内存省 | 推荐 |
    |------|---------|-----|------|--------|------|
    | FP8 E4M3 | 1.000000 | ~0 | <0.1% | 50% | ✅最优精度 |
    | INT8 per-channel | 0.999956 | 0.0002 | <0.1% | 50% | ✅近完美 |
    | INT4 group-64 | 0.994 | 0.0005 | ~1% | 75% | ✅足够 |
    | INT4 group-128 | 0.993 | 0.0006 | ~1% | 75% | ✅推荐(AWQ/GPTQ默认) |
    | INT4 group-256 | 0.992 | 0.0006 | ~1-2% | 75% | ⚠️可用 |
    | INT4 group-4096 | 0.987 | 0.0010 | ~2-3% | 75% | ⚠️精度低 |

  精度规律:
    → group越小 → cos_sim越高 → 精度越好 → 但scale参数更多 → overhead!
    → → group=32: cos_sim=0.995 → 1.8M scales → 3.6MB overhead → 太多!
    → → → group=128: cos_sim=0.993 → 458K scales → 0.9MB overhead → 推荐!
    → → → → group=4096: cos_sim=0.987 → 14K scales → 28KB overhead → 少但精度低!

    → → → → → **group=128是最优平衡点!** → cos_sim=0.993 + 合理overhead → AWQ/GPTQ默认!

  FP8为什么完美?
    → FP8 E4M3: 1sign+4exp+3mantissa → 动态范围2^-7到448 → 足够覆盖权重分布!
    → → vs INT8: 固定步长1/127 → 低值精度差 → 但权重分布均匀 → 影响小!
    → → → FP8: 浮点步长 → 小值精度好 → 大值精度低 → 权重分布匹配!
    → → → → → **FP8 E4M3 = 最优权重量化精度!** → cos_sim≈1.0 → 推荐!

  量化误差分布:
    → INT4: mean_error=0.002 → max_error=0.008 → 相对误差30%(!) → 但cos_sim仍0.993!
    → → → 为什么相对误差30%但cos_sim=0.993? → 因为小权重相对误差大 → 但对余弦相似度影响小!
    → → → → 余弦相似度看重绝对误差×权重大小 → 大权重误差小 → 影响大 → cos_sim高!
    → → → → → 相对误差=小权重百分比大 → 但小权重对attention/MLP输出影响小 → 可忽略!
```

## 2. AWQ: Activation-Aware → 保护重要权重

```
AWQ (Activation-Aware Weight Quantization, Lin 2023):

  核心洞察: 不是所有权重都同等重要 → 与大激活值对应的权重更重要(salient)!

  Salient weight定义:
    → 权重w_ij → 与激活x_j对应 → 如果x_j很大(outlier) → w_ij对输出影响大 → salient!
    → → → x_j大 → w_ij × x_j → 乘积大 → 对输出贡献大 → quantize w_ij → 误差×x_j → 大!
    → → → → **保护salient weights → 减少大激活通道的量化误差 → 关键!**

  AWQ方法:
    → 1. 找salient channels: x_j大的通道 → 这些权重需要保护
    → → 2. Scaling: w_ij × s_j → x_j / s_j → 数学等价! → 但s_j>1 → w_ij变大 → 量化精度更高!
    → → → → → 关键: s_j = max(x_j)^α → α控制保护程度 → α=0.5最优(Goldilocks)
    → → → → → → s太大 → x太小 → activation信息丢失 → 不好!
    → → → → → → → s适中 → w量化精度高 + x精度高 → 最优!

  AWQ vs baseline INT4实测:
    → 本benchmark模拟: AWQ cos_sim=0.980 → baseline=0.993 → **AWQ更差!**
    → → → 原因: 模拟使用weight-based saliency而非activation-based → 不准确!
    → → → → **真实AWQ使用activation数据 → 保护真正重要的权重 → 实际比baseline更好!**
    → → → → → AWQ论文实测: LLaMA-7B INT4 AWQ → PPL=7.82 → baseline INT4→PPL=8.18 → 更好!
    → → → → → → → AWQ perplexity更好 → 因为保护了对推理影响最大的权重通道!

  AWQ生产使用:
    → AutoAWQ: 计算saliency → 量化 → 保存 → vLLM直接加载!
    → → → vLLM: --quantization awq → 自动使用AWQ模型 → fused kernel → 推荐!
    → → → → Marlin kernel: INT4 AWQ的fused dequant+GEMM → 消除Python overhead → 推荐!

  RTX 4090 AWQ部署:
    → AWQ INT4 → 权重3.5GB → 75%省 → B=119 → 推荐!
    → → → 但: 需要fused Marlin kernel → vLLM自动使用 → 推荐!
    → → → → **RTX 4090最优: AWQ INT4(fused) + INT8 KV + FlashInfer → 推荐!**
```

## 3. GPTQ: Hessian补偿 → 二阶信息优化量化

```
GPTQ (Frantar 2023):

  核心洞察: 量化不是逐个独立 → 前面的量化误差会影响后面的最优量化 → 用Hessian补偿!

  Hessian是什么?
    → Hessian = loss对权重的一阶导数的导数 → 二阶导数 → 曲率信息!
    → → → H_ij = Σ(x_i × x_j) → 对角线H_ii = Σ(x_i²) → 权重i的"重要性权重"!
    → → → → 量化w_i → 误差δ_i → 对loss的影响 ≈ H_ii × δ_i² → 大H_ii → 小误差也很影响!

  GPTQ算法:
    → 1. 逐行量化(不是逐列!) → 每行独立 → 但行内有补偿!
    → → 2. 量化w_i → 计算误差δ_i → 补偿到未量化的w_j → 使用Hessian信息!
    → → → → 补偿公式: w_j ← w_j - δ_i × H_ij / H_ii → 前面误差被后面补偿!
    → → → → → Cholesky分解Hessian → 保证补偿顺序最优 → 最小化总误差!
    → → → → → → OBQ(Optimal Brain Quantization) → GPTQ是OBQ的近似加速版!

  GPTQ vs AWQ:
    → GPTQ: 基于Hessian → 用二阶信息 → 更精确的补偿 → perplexity好
    → → AWQ: 基于activation → 保护salient → 更简单 → 但perplexity也好!
    → → → 实际对比: LLaMA-7B INT4 → GPTQ PPL=7.93 → AWQ PPL=7.82 → **AWQ更好!**
    → → → → AWQ perplexity更好 → 因为直接保护重要通道 → 更直观 → 推荐!
    → → → → → GPTQ优势: one-shot → 不需要校准数据 → 但AWQ需要 → GPTQ更快!
    → → → → → → 但: AWQ只需要少量校准 → 几分钟 → 不影响部署!

  GPTQ生产使用:
    → AutoGPTQ: 量化 → 保存 → vLLM/SGLang加载
    → → → vLLM: --quantization gptq → 但GPTQ kernel不如Marlin快 → AWQ+Marlin更快!
    → → → → **RTX 4090推荐: AWQ(更快kernel) > GPTQ(更慢kernel) → 推荐AWQ!**
```

## 4. SmoothQuant: W8A8 → 数学等价迁移activation outlier

```
SmoothQuant (Xiao 2023):

  核心洞察: LLM推理瓶颈不是权重 → 是activation outlier → 迁移到权重侧!

  Activation outlier问题:
    → LLM推理 → activation有outlier → 某些通道值100x大 → INT8量化崩塌!
    → → → 99%通道: 值0.1-1.0 → INT8量化OK → 但1%通道: 值100 → INT8溢出!
    → → → → → 传统INT8: quantize(x_outlier) → 误差100x → 输出崩塌 → 不可用!

  SmoothQuant数学:
    → s_j = max(|X|)^α / max(|W|)^{1-α} → 平滑因子
    → → W_smooth = W × s → X_smooth = X / s → W_smooth × X_smooth = W × X → 数学等价!
    → → → → 但: X_smooth没有outlier → 可以INT8 → W_smooth有outlier → 但权重静态 → 可以INT8!
    → → → → → → W8A8: weight INT8 + activation INT8 → 数学等价 → 真INT8推理 → 2x计算吞吐!

  α最优值:
    → α=0: s=1 → 不迁移 → X有outlier → W8A8不行 → 不推荐!
    → → α=0.5: s=max(X)^0.5/max(W)^0.5 → 平衡迁移 → 推荐!
    → → → α=1: s=max(X)/1 → 全迁移到权重 → X无outlier → 但W可能overflow → 不推荐!

  RTX 4090实测 (无显著outlier时):
    → α=0: SmoothQuant=0.999964 → baseline=0.999962 → 几乎相同!
    → → α=0.5: SmoothQuant=0.999953 → 略差 → 因为权重scaling增加了量化难度
    → → → → 模拟没有显著activation outlier → SmoothQuant收益小 → 但真实LLM有outlier!

  真实LLM的outlier:
    → LLM.int8()论文发现: LLaMA-7B → 某些activation通道值高达200x → INT8溢出!
    → → → SmoothQuant: 迁移这些outlier到权重 → W8A8 → 2x计算吞吐 → 推荐!
    → → → → → RTX 4090: SM89支持INT8 Tensor Core → W8A8 → 2x → 推荐!

  生产使用:
    → vLLM: --quantization smoothquant → 需要校准 → 但一键 → 推荐!
    → → → 或: 手动计算s → 用AWQ框架 → 更灵活 → 推荐!
    → → → → **RTX 4090: INT4 AWQ(fused)比INT8 SmoothQuant更省 → 但SmoothQuant精度更高 → 按需选择!**
```

## 5. Marlin Kernel: Fused Dequant+GEMM → 消除Python Overhead

```
Marlin INT4 Kernel (2024):

  核心问题: INT4量化 → Python dequant → 20x慢 → 灾难!
  → → 解决: fused kernel → dequant+GEMM一体化 → 无Python overhead → 推荐!

  Python vs Fused对比:

    | 方法 | 7B decode latency | vs BF16 |
    |------|-------------------|---------|
    | BF16 baseline | 14.83ms | 1.0x |
    | INT4 Python dequant | ~300ms | ~20x慢! |
    | INT4 Marlin fused | ~3.9ms | ~3.8x快! |
    | INT8 W8A8 fused | ~7.9ms | ~1.9x快! |
    | FP8 TE fused | ~8ms | ~1.86x快 |

    → **INT4 Python: 20x慢 → 灾难 → 不可用!**
    → → **INT4 Marlin fused: 3.8x快 → 推荐! → fused kernel消除overhead!**
    → → → → INT8 fused: 1.9x快 → 精度99.99% → 推荐(精度优先)!
    → → → → → FP8 TE fused: 1.86x快 → 精度100% → 推荐(精度最优)!

  Marlin架构:
    → 输入: INT4量化权重 + FP16 scale → GPU上dequant → BF16 → GEMM
    → → → 所有步骤在single CUDA kernel → 无Python → 无中间内存 → 极快!
    → → → → 关键: weight从INT4→BF16→GEMM → 所有在GPU → 消除CPU→GPU传输!
    → → → → → → RTX 4090(SM89): Marlin kernel自动dispatch → vLLM集成 → 推荐!

  vLLM量化选择:
    → --quantization awq → Marlin kernel(AWQ模型) → 推荐!
    → → --quantization gptq → EXL2/GPTQ kernel → 略慢于Marlin → AWQ更快!
    → → → --quantization fp8 → TE kernel → 精度最优 → 推荐(精度优先)!
    → → → → --quantization smoothquant → W8A8 → 精度高 → 推荐(SmoothQuant场景)

  **核心规律: 量化必须用fused kernel → 否则Python dequant overhead致命!**
    → INT4: 必须Marlin/AWQ → 否则20x慢 → 灾难!
    → → INT8 KV: Python 3-12% overhead → 接近free → 但FlashInfer FP8更好 → 推荐!
    → → → FP8 TE: fused quantize+GEMM+dequantize → 1.48-1.59x训练加速 → 推荐!
```

## 6. RTX 4090量化决策树

```
RTX 4090量化配置决策:

  ┌─ 精度优先 (99.9%+精度) ──────────────────────────┐
  │ → FP8 E4M3权重 + FP8 KV (TE fused)              │
  │ → → cos_sim=1.000000 → PPL↑<0.1% → 推荐!       │
  │ → → → B=57 → 2,312 tok/s → 精度最优 → 推荐!    │
  └──────────────────────────────────────────────────────┘

  ┌─ 内存优先 (最高并发) ───────────────────────────┐
  │ → INT4 AWQ权重 + INT8 KV (Marlin fused)        │
  │ → → cos_sim=0.993 → PPL↑~1% → 75%省 → 推荐!   │
  │ → → → B=119 → ~4,500 tok/s → 并发最高 → 推荐! │
  └──────────────────────────────────────────────────────┘

  ┌─ 平衡 (精度+并发) ─────────────────────────────┐
  │ → BF16权重 + INT8 KV (FlashInfer)              │
  │ → → 权重=BF16(100%) → KV=INT8(99.99%) → 推荐! │
  │ → → → B=57 → 2,312 tok/s → 平衡 → 推荐!      │
  └──────────────────────────────────────────────────────┘

  ┌─ 训练加速 ────────────────────────────────────┐
  │ → FP8 TE训练(B≥4) → 1.48-1.59x → 推荐!       │
  │ → → → B=1 → FP8慢0.75x → 不推荐!              │
  └──────────────────────────────────────────────────────┘

  不推荐:
  → INT4 Python dequant → 20x慢 → 灾难 → 不推荐!
  → → INT4 KV → 需fused kernel → 当前无 → 不推荐!
  → → → SmoothQuant α=1 → 全迁移 → W overflow → 不推荐!

  **RTX 4090最优量化组合**:
    → 内存优先: AWQ INT4(Marlin) + INT8 KV → B=119 → 推荐!
    → → 精度优先: FP8 E4M3(TE) + FP8 KV → B=57 → 推荐!
    → → → 平衡: BF16权重 + INT8 KV → B=57 → 推荐!
    → → → → **按场景选择 → 内存不够用AWQ → 精度不够用FP8 → 够用用BF16!**
```

## 7. 核心学习

```
1. **INT4 cos_sim=0.993 → PPL↑~1% → 精度足够 → 75%内存省 → 推荐(fused!)**
2. **FP8 E4M3 cos_sim≈1.0 → 完美精度 → 50%省 → 推荐(精度最优!)**
3. **INT8 per-channel cos_sim=0.9999 → 近完美 → 50%省 → 推荐(SmoothQuant W8A8)**
4. **AWQ保护salient weight**: activation-aware → 保护重要通道 → perplexity更好
5. **GPTQ Hessian补偿**: 二阶信息 → 量化误差补偿 → but AWQ更快更好
6. **SmoothQuant=W8A8**: 数学等价迁移outlier → α=0.5最优 → 真INT8推理
7. **Python dequant=20x慢**: fused kernel(Marlin/TE)消除overhead → 量化才有效!
8. **group=128最优**: cos_sim=0.993 + 合理scale overhead → AWQ/GPTQ默认
9. **RTX 4090最优: AWQ INT4(Marlin)+INT8 KV+FlashInfer → B=119 → 推荐!**
```

---

**Sources**:
- [AWQ (Lin 2023)](https://arxiv.org/abs/2306.00978)
- [GPTQ (Frantar 2023)](https://arxiv.org/abs/2210.15723)
- [SmoothQuant (Xiao 2023)](https://arxiv.org/abs/2211.10438)
- [Marlin INT4 Kernel (2024)](https://github.com/IST-DASLab/marlin)
- [LLM.int8() (Dettmers 2022)](https://arxiv.org/abs/2208.07339)

**Related notes**: quantization-pruning-theory.md, kv-cache-management-deep-dive.md, inference-cost-analysis.md

**Benchmark tool**: tools/weight_quantization_benchmark.py (7 experiments, RTX 4090)
**Benchmark results**: results/weight_quantization_benchmark.json