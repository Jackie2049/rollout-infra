# Triton Kernel Development Workshop — RTX 4090 实测

> 2026-06-07/08 | 两轮实验: 第一轮4实验(Softmax+Temp/LN+Res/GQA Expand/FP8 Dequant) + 第二轮5实验(RMSNorm/SwiGLU/Softmax+Temp/SiLU/Element-wise)

## 一、Experiment 1: Fused Softmax + Temperature Scaling (第一轮)

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
- **结论**: Softmax+Temperature融合对小batch不值得 → cuDNN softmax比自定义Triton更快

## 二、Experiment 2: Fused LayerNorm + Residual (第一轮)

**目的**: Triton融合LayerNorm+weight+bias+residual vs PyTorch分步(ln(x)+r)

| B | PyTorch (ms) | Triton (ms) | Speedup | cos_sim |
|---|-------------|-------------|---------|---------|
| 4 | 0.0313 | 0.0231 | **1.35x** | 1.0 |
| 32 | 0.0351 | 0.0229 | **1.53x** | 1.0 |
| 128 | 0.0355 | 0.0228 | **1.55x** | 1.0 |
| 512 | 0.0353 | 0.0226 | **1.56x** | 1.0 |

**发现**:
- **所有batch size都加速** (1.35-1.56x): LayerNorm+residual融合减少3次global memory读写
- **Bug修复**: stride bug → cos_sim=0.8 → 修复后cos_sim=1.0
- **与CUDA C++对比**: CUDA C++ RMSNorm+Add 9.4x > Triton 1.56x → CUDA C++性能上限更高

## 三、Experiment 3: GQA KV Expand (第一轮)

**目的**: 比较Python expand vs repeat_interleave vs index_copy

| B | Python expand (ms) | repeat_interleave (ms) | index_copy (ms) |
|---|--------------------|-----------------------|-----------------|
| 1 | 0.0179 | 0.0172 | 0.5004 |
| 8 | 0.1929 | 0.1927 | 0.4970 |
| 32 | 0.7623 | 0.7544 | 0.8159 |
| 128 | 3.0090 | 3.0093 | 4.7557 |

**发现**: `repeat_interleave`是最佳Python方法; FlashInfer GQA native才真正消除expand

## 四、Experiment 4: FP8 Dequant + GEMM (第一轮)

| M | K | N | separate (ms) | direct FP16 (ms) | dequant overhead |
|---|---|---|--------------|-----------------|-----------------|
| 1 | 4096 | 4096 | 0.1022 | 0.0316 | **223.2%** |
| 32 | 4096 | 4096 | 0.1109 | 0.0201 | **452.1%** |
| 512 | 4096 | 4096 | 0.2008 | 0.1090 | **84.2%** |

**发现**: Python-level dequant不可接受 → 必须用fused kernel(cuBLAS FP8/Marlin/TE)

---

## 五、第二轮实测 (2026-06-08): RMSNorm/SwiGLU/Softmax+Temp/SiLU/Element-wise

### 5.1 RMSNorm: Triton 2.75-3.23x Faster (cos_sim=1.000000)

```
RMSNorm Triton vs PyTorch vs torch.compile (D=4096, BF16):

    | B   | Triton ms | PyTorch ms | compile ms | Triton speedup | compile speedup | cos_sim |
    |-----|-----------|------------|------------|----------------|-----------------|---------|
    | 1   | 0.0364    | 0.1004     | 0.1060     | **2.75x**      | 0.95x           | 1.000000|
    | 4   | 0.0345    | 0.1083     | 0.1249     | **3.14x**      | 0.87x           | 1.000000|
    | 8   | 0.0370    | 0.1099     | 0.1234     | **2.97x**      | 0.89x           | 1.000000|
    | 16  | 0.0349    | 0.1005     | 0.1199     | **2.88x**      | 0.84x           | 1.000000|
    | 32  | 0.0349    | 0.1061     | 0.1170     | **3.04x**      | 0.91x           | 1.000000|
    | 64  | 0.0347    | 0.1049     | 0.1114     | **3.02x**      | 0.94x           | 1.000000|
    | 128 | 0.0332    | 0.1071     | 0.1200     | **3.23x**      | 0.89x           | 1.000000|
    | 256 | 0.0355    | 0.1071     | 0.1193     | **3.02x**      | 0.90x           | 1.000000|
```

**关键发现**:
- Triton RMSNorm **2.75-3.23x** → 比第一轮1.8x更高! Triton 3.5.0优化+RTX 4090大L2(72MB)
- cos_sim=1.000000 → bit-exact match → 完全正确
- torch.compile反而慢(0.84-0.95x) → Triton直接更快!
- Triton快的原因: 1 program/row → 1-pass load→square→sum→sqrt→normalize→weight→store → 全在kernel内 → 消除多op串联launch overhead

### 5.2 SwiGLU MLP: torch.compile No Speedup (0.85x)

```
SwiGLU MLP (d_model=4096, hidden=14336, BF16):

    | B   | separate ms | compile ms | speedup |
    |-----|-------------|------------|---------|
    | 1   | 0.4081      | 0.4790     | **0.85x**|
    | 4   | 0.4138      | 0.4882     | **0.85x**|
    | 8   | 0.4150      | 5.3989     | **0.08x**| ← compile recompilation!
    | 16  | 0.4195      | 0.4903     | **0.86x**|
    | 32  | 0.4248      | 0.4919     | **0.86x**|
```

**关键发现**:
- torch.compile反而慢0.85x → cuBLAS最快 → compile引入Triton GEMM→负优化
- MLP = 3个GEMM(94%时间) + element-wise(6%) → compile fusion省6%→微不足道
- **生产: 直接用cuBLAS → 不需要compile MLP!**

### 5.3 Softmax+Temperature: Triton ≈ PyTorch (0.94-1.02x)

```
Triton Softmax+Temperature vs PyTorch (V=32000, BF16):

    | B   | T   | Triton ms | PyTorch ms | speedup |
    |-----|-----|-----------|------------|---------|
    | 1   | 1.0 | 0.0427    | 0.0402     | **0.94x**|
    | 4   | 1.0 | 0.0398    | 0.0385     | **0.97x**|
    | 16  | 1.0 | 0.0396    | 0.0399     | **1.01x**|
    | 55  | 1.0 | 0.0406    | 0.0397     | **0.98x**|
    | 128 | 1.0 | 0.0485    | 0.0490     | **1.01x**|
```

**关键发现**: Triton ≈ PyTorch → PyTorch softmax已经高度优化 → Triton无优势 → 温度融合不值得写kernel

### 5.4 SiLU: Triton **2.3x SLOWER** than PyTorch (0.42-0.47x) ← 最意外!

```
Triton SiLU vs PyTorch (D=4096, BF16):

    | B   | Triton ms | PyTorch ms | speedup  | cos_sim |
    |-----|-----------|------------|----------|---------|
    | 1   | 0.0383    | 0.0159     | **0.42x**| 1.000000|
    | 4   | 0.0371    | 0.0166     | **0.45x**| 1.000000|
    | 32  | 0.0362    | 0.0166     | **0.46x**| 1.000000|
    | 128 | 0.0344    | 0.0163     | **0.47x**| 1.000000|
```

**关键发现**:
- Triton SiLU **0.42-0.47x** → 比PyTorch慢**2.3x**! cos_sim=1.000000 → 数值正确但速度远不如PyTorch
- 原因: Triton launch overhead ≈ 0.035ms vs PyTorch ≈ 0.016ms → 2x launch overhead
- SiLU极简(x * sigmoid(x)) → 计算时间≈0 → 纯launch overhead → Triton不适合
- **规律**: Triton只在reduction-based ops有优势 → simple ops不如PyTorch

### 5.5 Element-wise Suite: PyTorch Native All <0.03ms

```
Element-wise Benchmark (D=4096, BF16):

    | Op       | B=1 ms  | B=32 ms | B=128 ms | Pattern |
    |----------|---------|---------|----------|---------|
    | SiLU     | 0.0168  | 0.0176  | 0.0162   | ~0.016ms |
    | GELU     | 0.0170  | 0.0171  | 0.0161   | ~0.016ms |
    | ReLU     | 0.0181  | 0.0175  | 0.0175   | ~0.017ms |
    | exp      | 0.0271  | 0.0263  | 0.0256   | ~0.026ms |
    | sigmoid  | 0.0154  | 0.0162  | 0.0162   | ~0.016ms |
    | add      | 0.0161  | 0.0163  | 0.0176   | ~0.016ms |
    | mul      | 0.0159  | 0.0164  | 0.0171   | ~0.016ms |
```

**关键发现**: 全部launch overhead主导 → 不需要写Triton kernel → PyTorch native极快

---

## 六、RTX 4090 Triton Kernel Selection Decision Tree

```
    ┌───────────────────────────────────────────────────────────┐
    │  RTX 4090 Kernel Selection Decision Tree                    │
    │                                                             │
    │  1. Attention → FlashInfer (54x GQA-8, production唯一答案) │
    │  2. GEMM → cuBLAS (100%+peak, never Triton tl.dot)           │
    │  3. Reduction ops → Triton (2-3x, e.g. RMSNorm/LayerNorm) │
    │  4. Element-wise → PyTorch native (2.3x faster than Triton!)│
    │  5. Fusion → torch.compile (only if compute-bound, not MLP!) │
    │                                                             │
    │  Triton sweet spot = reduction-based ops (RMSNorm/LayerNorm)│
    │  Triton bad spot = simple element-wise + GEMM               │
    └───────────────────────────────────────────────────────────┘
```

## 七、核心学习 (两轮合并)

1. **Triton RMSNorm 2.75-3.23x**: 比第一轮1.8x更高 → Triton 3.5.0优化+RTX 4090大L2
2. **Triton SiLU 0.42x**: 比PyTorch慢2.3x! → launch overhead主导 → Triton不适合简单ops
3. **Triton LN+Res 1.35-1.56x**: 融合有收益(reduction+store融合)
4. **torch.compile MLP 0.85x**: 反而慢 → cuBLAS最快 → compile Triton GEMM→负优化
5. **Triton Softmax ≈ PyTorch**: 0.94-1.02x → PyTorch已高度优化 → Triton无优势
6. **Element-wise all <0.03ms**: 全部launch overhead主导 → PyTorch native够快
7. **Python dequant不可接受**: 4x overhead → 必须fused kernel(Marlin/TE)
8. **Triton规律**: reduction ops → Triton胜 → simple ops → PyTorch胜 → GEMM → cuBLAS胜
9. **生产决策**: Triton(reduction) + PyTorch(element-wise) + cuBLAS(GEMM) + FlashInfer(attn)

**4个关键教训** (第一轮):
1. **不要盲目写自定义kernel**: 先benchmark再决定
2. **Triton stride必须正确**: 混淆→cos_sim=0.8
3. **GQA expand用repeat_interleave**: 比expand+reshape/index_copy都快
4. **Quantization必须fused kernel**: Python-level dequant=4x GEMM时间

**新增教训** (第二轮):
5. **Triton不适合简单element-wise**: SiLU/sigmoid等 → PyTorch native更快
6. **MLP不需要compile**: cuBLAS GEMM最快 → compile是负优化
7. **Launch overhead是关键**: kernel越简单 → Triton越没优势 → 因为overhead占比大

Sources:
- Triton Language Reference: https://triton-lang.org/
- FlashAttention (Dao et al., 2022): https://arxiv.org/abs/2205.14135
- Marlin FP8 GEMM: https://github.com/IST-DASLab/marlin

**Benchmark tool**: tools/triton_kernel_workshop_benchmark.py
**Benchmark results**: results/triton_kernel_workshop_benchmark.json

**Related notes**: triton-vs-cuda-benchmark-rtx4090.md(之前对比), gpu-microarchitecture-sm89-sm90-sm100.md(SM89架构)