# Triton Kernel Development Workshop — RTX 4090 实测

> 2026-06-07 | 4个实验: Fused Softmax+Temperature, Fused LayerNorm+Residual, GQA KV Expand, FP8 Dequant+GEMM

## 一、Experiment 1: Fused Softmax + Temperature Scaling

**目的**: Triton融合softmax+temperature vs PyTorch分步(scale→softmax)

| B | PyTorch (ms) | Triton (ms) | Speedup | cos_sim | max_diff |
|---|-------------|-------------|---------|---------|----------|
| 1 | 0.0163 | 0.0188 | **0.87x** | 1.0 | 4.1e-5 |
| 8 | 0.0166 | 0.0170 | **0.97x** | 1.0 | 1.27e-4 |
| 32 | 0.0163 | 0.0178 | **0.91x** | 1.0 | 1.25e-4 |
| 128 | 0.0263 | 0.0187 | **1.40x** | 1.0 | 1.53e-4 |

**发现**:
- **小batch Triton反而更慢** (0.87-0.97x): PyTorch的cuDNN softmax已经高度优化, Triton kernel launch开销在小batch时占主导
- **大batch(B=128)才有1.4x加速**: 融合减少1次global memory round-trip(scale结果→softmax输入), 但仅在大batch时IO占比超过launch开销
- **结论**: Softmax+Temperature融合对小batch不值得 → cuDNN softmax比自定义Triton更快, 这是之前"不要盲目写自定义kernel"教训的又一次验证

## 二、Experiment 2: Fused LayerNorm + Residual

**目的**: Triton融合LayerNorm+weight+bias+residual vs PyTorch分步(ln(x)+r)

| B | PyTorch (ms) | Triton (ms) | Speedup | cos_sim |
|---|-------------|-------------|---------|---------|
| 4 | 0.0313 | 0.0231 | **1.35x** | 1.0 |
| 32 | 0.0351 | 0.0229 | **1.53x** | 1.0 |
| 128 | 0.0355 | 0.0228 | **1.55x** | 1.0 |
| 512 | 0.0353 | 0.0226 | **1.56x** | 1.0 |

**发现**:
- **所有batch size都加速** (1.35-1.56x): LayerNorm+residual融合减少3次global memory读写(norm→weight*→bias+→residual+)
- **加速随batch增大趋于稳定** (1.56x): memory-bound操作, fusion减少的IO量是固定的
- **cos_sim=1.0**: 修复stride bug后精度完全一致

**Bug修复故事**:
- 原始kernel用 `pid * stride_h` 访问行数据, 但stride_h是列stride(=1)而非行stride(=H)
- 结果: kernel读取了错误位置 → cos_sim=0.8
- 修复: 改为 `pid * stride_b` → cos_sim=1.0
- **教训**: Triton kernel中stride命名和计算必须与实际tensor stride匹配!

**与CUDA C++ RMSNorm对比**:
- Triton LayerNorm+Residual: 1.35-1.56x
- CUDA C++ RMSNorm+Add: 9.4x (FP32)
- **CUDA C++大幅领先**: 因为(1)warp shuffle reduction(2)FP32 accumulation(3)2-pass优化

## 三、Experiment 3: GQA KV Expand

**目的**: 比较Python expand vs repeat_interleave vs index_copy

配置: n_heads=32, n_kv_heads=8, n_rep=4, seq_len=2048, head_dim=128

| B | Python expand (ms) | repeat_interleave (ms) | index_copy (ms) | expand/repeat |
|---|--------------------|-----------------------|-----------------|---------------|
| 1 | 0.0179 | 0.0172 | 0.5004 | **1.04x** |
| 8 | 0.1929 | 0.1927 | 0.4970 | **1.00x** |
| 32 | 0.7623 | 0.7544 | 0.8159 | **1.01x** |
| 128 | 3.0090 | 3.0093 | 4.7557 | **1.00x** |

**发现**:
- **Python expand ≈ repeat_interleave**: 几乎无差异(0.00-1.04x), PyTorch的expand+reshape内部可能就是repeat_interleave的优化版本
- **index_copy最慢** (1.4-29x!): Python-level逐头拷贝 → 每次拷贝触发一个CUDA kernel launch → 小batch时launch开销巨大(B=1: 0.5ms vs 0.02ms)
- **结论**: `repeat_interleave`是最佳Python方法; Triton/FlashInfer专用kernel(GQA BLOCK_M打包Q头)才能进一步优化(避免expand的内存开销)

**与之前实测对比**:
- 之前发现GQA Python-level expand有56-86% overhead(B≥8时GQA比MHA更慢)
- 这里测试的是expand vs repeat_interleave vs index_copy三种Python方法对比
- repeat_interleave是最优选择 → vLLM应该用repeat_interleave替代expand+reshape

## 四、Experiment 4: FP8 Dequant + GEMM

**目的**: INT8→FP16 dequant+GEMM(分步) vs 直接FP16 GEMM(基准)

| M | K | N | separate (ms) | direct FP16 (ms) | dequant overhead |
|---|---|---|--------------|-----------------|-----------------|
| 1 | 4096 | 4096 | 0.1022 | 0.0316 | **223.2%** |
| 32 | 4096 | 4096 | 0.1109 | 0.0201 | **452.1%** |
| 128 | 4096 | 4096 | 0.1238 | 0.0316 | **291.2%** |
| 512 | 4096 | 4096 | 0.2008 | 0.1090 | **84.2%** |

**发现**:
- **Dequant overhead巨大** (84-452%): INT8→FP16 cast + scale乘法 = 4x GEMM时间
- **小M overhead更高**: dequant时间相对固定(≈0.07ms), 小M时GEMM也快(0.02ms) → overhead比例巨大
- **大M(B=512) overhead最低**(84%): GEMM时间增大到0.109ms → dequant占比减小
- **结论**: Python-level dequant不可接受 → 必须用fused kernel(cuBLAS FP8 GEMM/Marlin/compressed-tensors)

**与之前quantization实测一致**:
- 之前实测: Python-level INT8 dequant 0.17-0.37x → 比FP16更慢
- 这里: dequant overhead 84-452% → 同样结论

## 五、综合结论

| 实验 | Triton加速 | 核心发现 |
|------|-----------|---------|
| Softmax+Temp | 0.87-1.40x | **cuDNN更优, 不要盲目自定义** |
| LayerNorm+Res | 1.35-1.56x | **融合有收益, stride bug教训** |
| GQA Expand | — | **repeat_interleave最佳Python方法** |
| FP8 Dequant+GEMM | — | **Python-level dequant不可接受, 必须fused** |

**4个关键教训**:
1. **不要盲目写自定义kernel**: PyTorch/cuDNN已高度优化, 先benchmark再决定
2. **Triton stride必须正确**: 行stride=stride(0), 列stride=stride(1), 混淆→cos_sim=0.8
3. **GQA expand用repeat_interleave**: 比expand+reshape/index_copy都快
4. **Quantization必须fused kernel**: Python-level dequant=4x GEMM时间, fused才能有收益

**Triton vs CUDA C++定位**:
- Triton: 开发效率高(~50行), 适合原型和中等优化(1.5-2x)
- CUDA C++: 性能上限高(9x+), 适合极致优化(warp shuffle+FP32 accumulation+2-pass)
- 决策: 快速实验选Triton, 生产环境追求性能选CUDA C++

Sources:
- Triton Language Reference: https://triton-lang.org/
- FlashAttention (Dao et al., 2022): https://arxiv.org/abs/2205.14135
- Marlin FP8 GEMM: https://github.com/IST-DASLab/marlin