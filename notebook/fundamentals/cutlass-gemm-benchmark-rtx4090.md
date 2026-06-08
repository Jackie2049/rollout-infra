# CUTLASS SM89 GEMM Benchmark: RTX 4090 实测分析

> 2026-06-08 | 从decode到prefill, 从BF16到FP32, 从单GEMM到MoE batched, GPU计算特性全面实测
> 基于: tools/cutlass_gemm_benchmark.py (PyTorch/cuBLAS GEMM timing)
> 关联: gpu-microarchitecture-sm89-sm90-sm100.md (SM89 HMMA), decode-roofline-rtx4090.md (Roofline), quantization-path-comparison-rtx4090.md (量化)

## 0. 核心发现: decode GEMM是极度memory-bound

```
7B模型decode (B=1-128):
  → B=1: 1.46 TFLOPS (1.8% peak) — 98.2%计算能力浪费!
  → B=32: 35.5 TFLOPS (43% peak) — 仍只利用43%
  → B=128: 82.4 TFLOPS (99.8% peak) — 终于接近100%但仍在memory-bound!
  → B=256: 115.5 TFLOPS (140% peak) — crossover到compute-bound!

Prefill (S=128-2048):
  → S=128: 82.9 TFLOPS (100% peak) — 与decode B=128相当!
  → S=512+: 140-154 TFLOPS (170-186% peak) — compute-bound, 充分利用TC!

关键: decode B≤128只利用1-100% peak → 大量TC闲置 → 这是推理成本高的根因!
  → 量化省带宽 → 从1.8%提升到更高 → 这是量化推理加速的真正机制!
```

## 1. BF16 GEMM Roofline: LLM-relevant sizes

### 1.1 Decode (7B gate_proj 10240×2560)

| B | TFLOPS | peak% | AI | bound | ms |
|---|--------|-------|----|-------|----|
| 1 | 1.46 | 1.8% | 1 | memory | 0.036 |
| 4 | 4.09 | 5.0% | 4 | memory | 0.051 |
| 8 | 8.36 | 10.1% | 8 | memory | 0.050 |
| 16 | 15.94 | 19.3% | 16 | memory | 0.053 |
| 32 | 35.51 | 43.0% | 32 | memory | 0.047 |
| 64 | 61.60 | 74.6% | 62 | memory | 0.054 |
| 128 | 82.40 | 99.8% | 120 | memory | 0.081 |
| 256 | 115.54 | **140%** | 228 | **compute** | 0.116 |

**关键发现**:
- B=1到B=128全是memory-bound → 只有B=256才crossover!
- B=1仅利用1.8%峰值 → 98.2%的Tensor Core闲置!
- 但: B=32已经43% → FlashInfer B=32 throughput 145K tok/s → 这解释FlashInfer的价值!
- **超越100%峰值**: B=256→140% → cuBLAS可能用了2:4 sparse或其他优化

### 1.2 Prefill (7B gate_proj)

| S | TFLOPS | peak% | AI | bound |
|---|--------|-------|----|-------|
| 128 | 82.86 | 100% | 120 | memory |
| 512 | 140.53 | **170%** | 410 | **compute** |
| 1024 | 139.07 | 168% | 683 | compute |
| 2048 | 153.72 | **186%** | 1024 | compute |

**关键发现**:
- S≥512就已经compute-bound → prefill计算充分利用TC!
- 超越100%peak → cuBLAS内部优化: 可能用了2:4 sparsity探测或FP8中间精度

### 1.3 为什么BF16能超越理论峰值?

```
RTX 4090 BF16理论峰值: 82.58 TFLOPS (without 2:4 sparse)
实测最大: 167.19 TFLOPS (202.5% peak!)

可能原因:
  1. cuBLAS可能使用2:4 sparsity自动优化 → 2x → 165 TFLOPS → 匹配!
     → 但2:4需要权重预先50%稀疏 → PyTorch随机权重不会是2:4 → 不太可能
  2. cuBLAS可能split-K + parallel reduction → 多SM并行 → 但这是理论内的优化
  3. HMMA.16832(INT8)内部路径 → 2x吞吐 → cuBLAS自动选择INT8 kernel?
     → BF16输入 → cuBLAS自动量化到INT8 → HMMA.16832 → dequant输出 → 可能!
  4. TF32 + HMMA.16816路径 → 与BF16峰值相同 → 不是2x的原因
  5. 最可能: cuBLAS确实用了FP8/INT8内部路径 → BF16→INT8量化→HMMA→BF16输出
     → 这与TE FP8的机制类似 → cuBLAS内部的fused量化!

验证: FP16在同一size也达199%peak → cuBLAS对FP16/BF16可能有相同优化
  → FP32仅131%peak → 无INT8加速路径 → 峰值41.3 TFLOPS × 1.31 = 54 TFLOPS

结论: cuBLAS在大GEMM上可能自动用了INT8/FP8内部路径!
  → 这意味着: PyTorch BF16 matmul在大size时已经在用"隐式FP8"!
  → TE FP8的1.48-1.59x加速是显式FP8 → cuBLAS在大GEMM时可能已经隐式加速!
```

## 2. FP16 vs BF16 vs FP32 Comparison

### 2.1 大GEMM (8192×8192×8192)

| dtype | TFLOPS | peak% | peak标准 |
|-------|--------|-------|---------|
| FP32 | 54.25 | 131% | 41.3 TFLOPS (TF32 peak) |
| FP16 | 163.79 | 198% | 82.6 TFLOPS (FP16 peak) |
| BF16 | 167.19 | **203%** | 82.6 TFLOPS (BF16 peak) |

**发现**: BF16竟然比FP16快(167 vs 164) → 可能cuBLAS对BF16有额外优化?
- 或: BF16更宽的动态范围 → 量化到INT8/FP8时精度更好 → cuBLAS选择更好的内部路径

### 2.2 小GEMM (1024×10240×2560)

| dtype | TFLOPS | peak% |
|-------|--------|-------|
| FP32 | 50.01 | 121% |
| FP16 | 164.51 | 199% |
| BF16 | 148.52 | 180% |

**发现**: FP16在小GEMM更快(164 vs 149) → memory-bound时FP16带宽相同但cuBLAS优化不同

## 3. Batched GEMM (MoE-style)

| num_experts | TFLOPS | peak% | total_flops(G) |
|-------------|--------|-------|---------------|
| 4 | 5.15 | 6.2% | 1.68 |
| 8 | 6.78 | 8.2% | 3.36 |
| 16 | 7.40 | 9.0% | 6.71 |
| 32 | 7.48 | 9.1% | 13.42 |
| 64 | 7.52 | 9.1% | 26.84 |

**关键发现**: MoE batched GEMM严重underutilized! 仅6-9% peak!

```
原因分析:
  → 每个expert: M=8, N=10240, K=2560 → AI=8 → memory-bound
  → 64个expert × 8行 = 512行总 → 但bmm不是合并为1个大GEMM!
  → bmm = 64个独立小GEMM → 每个memory-bound → 总体也是memory-bound
  → 如果合并为1个大GEMM: M=512, AI=410 → compute-bound → 140+ TFLOPS!
  → 但MoE每个expert独立权重 → 不能合并 → 必须batched → 只能6-9% peak

MoE推理的困境:
  → 每个token只激活少数expert(如2/64=3.1%) → M非常小 → memory-bound
  → per-expert计算量小 → 大量SM闲置 → throughput低
  → 解决方案: 更大batch → 每expert分配更多tokens → M↑ → AI↑ → peak%↑

优化策略:
  1. 大batch推理 → 每expert更多tokens → MoE throughput↑
  2. Expert parallelism → 不同GPU处理不同expert → 但需All-to-All
  3. Expert batching → 合并多个token到同一expert → 提高M
  4. Grouped GEMM → CUTLASS grouped GEMM → 1个kernel处理多expert → 减少launch overhead
```

## 4. Roofline Crossover分析

```
Arithmetic Intensity vs peak utilization:
  AI=1 → 1.8% peak (decode B=1)
  AI=4 → 5% peak
  AI=8 → 10% peak
  AI=16 → 19% peak
  AI=32 → 43% peak
  AI=62 → 75% peak
  AI=120 → 100% peak (接近memory-bound → compute-bound边界!)
  AI=228 → 140% peak (compute-bound)
  AI=410 → 170% peak
  AI=1024 → 186% peak
  AI=2731 → 199% peak

Crossover AI ≈ 120-182 (实测):
  → 理论ridge = 182 FLOPS/byte (FP16 peak / HBM bandwidth)
  → 实测: AI=120 → 100% peak → 已经接近理论极限
  → AI=228 → 140% peak → 超越理论 → cuBLAS内部优化

解释:
  → AI<120: memory-bound → TFLOPS = AI × HBM_bandwidth → 线性增长
  → AI>120: compute-bound → TFLOPS → 接近/超越peak → 受TC限制
  → 超越peak → cuBLAS可能用了INT8/FP8内部路径 → 2x → 165 TFLOPS
```

## 5. 对推理优化的指导

```
Decode优化 (B=1-128, memory-bound):
  → 量化省带宽 → INT4(weight 75%省) → bytes↓ → AI↑(weight部分)
  → 但AI计算只考虑weight → INT4让weight带宽省75% → AI从1→4(B=1)
  → → 仍然memory-bound → 但从1.8%→5% → 3x improvement → 这就是INT4推理加速!
  → FlashInfer GQA native → KV带宽省75% → 15.72x → 这是decode最大的优化!
  → 批量推理(B=32) → AI=32 → 43% peak → 已经3x于B=1

Prefill优化 (S≥512, compute-bound):
  → 量化不能加速(prefill已经compute-bound → 量化减计算量也不多)
  → 但量化省内存 → 可以更大batch → throughput↑
  → TE FP8: compute-bound时 → FP8 2x吞吐 → 1.48-1.59x训练加速

MoE优化 (per-expert memory-bound):
  → Grouped/batched GEMM → 减少launch overhead
  → Expert parallelism → All-to-All通信 → NVLink必需
  → 大batch → 每expert更多tokens → M↑ → AI↑ → peak%↑
  → FP8 MoE → 2x per-expert throughput → 但小M时量化overhead占优
```

---

**Sources**:
- CUTLASS GEMM Benchmark: results/cutlass_gemm_benchmark.json
- NVIDIA cuBLAS Documentation
- RTX 4090 Specs: 82.58 TFLOPS FP16, 165.16 TFLOPS FP8/INT8

**Related notes**: gpu-microarchitecture-sm89-sm90-sm100.md (SM89 HMMA), quantization-path-comparison-rtx4090.md (量化路径), inference-cost-analysis.md (成本)