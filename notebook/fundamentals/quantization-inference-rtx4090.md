# Quantization Inference: RTX 4090 实测 — 为什么 Python-level Dequant 更慢

> 2026-06-07 | RTX 4090 (SM 8.9, 128 MPs, 24GB HBM)

## 核心发现

**Python-level 量化(dequant+matmul)全部比纯 FP16 更慢!**

| 方法 | Speedup vs FP16 | 内存节省 | Cosine Similarity | Max Relative Diff |
|------|----------------|---------|-------------------|-------------------|
| INT8 per-channel | **0.17-0.37x** (更慢!) | 50.0% | 0.953-0.994 | 0.4% |
| FP8 E4M3 | **0.31-0.56x** (更慢!) | 50.0% | 0.958-0.994 | 4.8% |
| INT4 AWQ-style | **0.24-0.48x** (更慢!) | 74.2% | 0.964-0.996 | 7.1% |

## 为什么量化反而更慢?

### Dequantize Overhead 分析

| 方法 | Dequant Time | FP16 Matmul | Total (dequant+matmul) |
|------|-------------|------------|------------------------|
| INT8 (H=4K) | **0.100ms** | 0.025ms | 0.125ms |
| FP8 (H=4K) | **0.039ms** | 0.025ms | 0.064ms |
| INT4 (H=4K) | ~0.075ms | 0.025ms | 0.100ms |

**关键**: Dequantize 耗时 > Matmul 耗时!

INT8 dequantize = 0.100ms vs FP16 matmul = 0.025ms → dequant是**4倍**于matmul!

### 为什么 dequantize 这么慢?

1. **类型转换**: `int8 → float16` 需要逐元素类型转换 → GPU kernel必须处理每个元素
2. **逐行缩放**: `w_int8.to(fp16) * scale.unsqueeze(1)` → 两个操作: cast + broadcast multiply
3. **reshape**: INT4 需要 reshape + dequant + reshape → 3步操作
4. **中间tensor**: dequant 产生完整 FP16 weight → 与原始 FP16 weight 内存一样大!
   - INT8: 34MB int8 → 34MB fp16 (临时!) + 2KB scale → 内存峰值更高
   - FP8: 同上

**矛盾**: 量化声称节省50%内存, 但 dequantize 时需要完整 FP16 中间tensor → 实际峰值内存不减!

## 生产环境的解决方案

Python-level dequantize 不行 → 需要 **fused quantization kernel**:

| 方法 | 生产实现 | 原理 |
|------|---------|------|
| INT8 weight-only | **vLLM compressed-tensors** | 内核里即时 dequant: `w_fp16 = w_int8 * scale` 融合在 GEMM |
| FP8 E4M3 | **cuBLAS FP8 GEMM** | 直接 FP8 输入到 Tensor Core, 无需 dequant |
| INT4 AWQ/Marlin | **Marlin kernel** | 专用 INT4→FP16 fused GEMM, 2x 吞吐 |
| GPTQ | **exllama_v2 kernel** | 按组 dequant+GEMM 融合 |

### 为什么 fused kernel 可以加速?

```
Python approach (3x更慢):
  w_fp16 = w_int8.to(fp16) * scale  → 0.100ms (临时tensor分配+类型转换+乘法)
  output = x @ w_fp16               → 0.025ms (标准GEMM)
  Total: 0.125ms

Fused kernel approach (目标加速):
  output = x @ dequant_inline(w_int8, scale)  → 融合在GEMM内部
  - 无临时tensor分配
  - 无完整FP16 weight重建
  - 按需dequant: 每行计算时才转换 → 省内存+省时间
  Total: ~0.030ms (预计2-3x加速 vs FP16 baseline)
```

## 精度影响分析

| 方法 | H=2048 cos_sim | H=4096 cos_sim | H=4096 rel_diff |
|------|---------------|---------------|-----------------|
| INT8 per-channel | 0.994 | 0.953 | 0.4% |
| FP8 E4M3 | 0.994 | 0.958 | 4.8% |
| INT4 AWQ (group=128) | 0.996 | 0.964 | 7.1% |

**注意**:
- H=2048 精度很好 (cos>0.99) → 小模型量化影响小
- H=4096 精度差 (cos<0.96) → 大矩阵累积误差更大
- FP8 rel_diff=4.8% → E4M3 只有 4 bit exponent + 3 bit mantissa → 范围限制
- INT8 per-channel rel_diff=0.4% → 精度最好
- INT4 groupwise rel_diff=7.1% → 压缩比大但精度损失也大

### 为什么 H=2048 和 H=4096 cos_sim 不同?

- cos_sim = Σ(a·b) / (||a|| × ||b||)
- H=4096: 矩阵更大 → 更多元素参与 → 量化误差累积更多
- H=2048: 矩阵更小 → 误差累积较少 → cos_sim 更高

## 实际应用建议

| 场景 | 推荐量化 | 原因 |
|------|---------|------|
| 7B FP16 单GPU推理 | **FP16 原生** | 14GB fits RTX 4090, 量化反而慢 |
| 13B FP16 单GPU推理 | **INT8 compressed-tensors** | 26GB→13GB fits, fused kernel有加速 |
| 70B 多GPU推理 | **FP8 (cuBLAS)** | 必须量化才fit, FP8 GEMM硬件支持 |
| 模型压缩部署 | **INT4 AWQ/Marlin** | 最大压缩比, Marlin kernel 2x吞吐 |
| KV cache量化 | **INT8 KV cache** | 省内存不影响compute, cos>0.999 |

**关键**: Python-level dequant 必须用 fused kernel 替代! vLLM/Marlin/exllama 都提供了.

## 与之前A16量化实测对比

| 指标 | A16 (10 SM) | RTX 4090 (128 SM) |
|------|------------|-------------------|
| FP16 vs FP32 matmul | 4-5x加速 | 2.76x |
| INT8 dequant overhead | 3.4x慢 | 4x慢 (dequant比matmul慢) |
| INT8 mem节省 | 50% | 50% |
| FP8 E4M3精度 | err=0.47 | rel_diff=4.8% |

**结论**: 两张卡上量化 overhead 都很大 → 问题本质是 Python-level dequant, 不是硬件特定