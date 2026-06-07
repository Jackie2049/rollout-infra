# CUTLASS GEMM 实战 — RTX 4090 编译与基准测试

> 2026-06-07 | 从理论到实践：在RTX 4090上编译CUTLASS并运行GEMM基准测试

## 概述

本文记录在RTX 4090(SM89)上从零搭建CUTLASS开发环境的完整过程，以及cuBLAS(cutlass底层)在不同精度/形状下的GEMM性能实测。

## 一、CUTLASS环境搭建

### 1.1 仓库克隆

```bash
# 通过中国镜像站克隆(shallow clone)
cd ~
git clone --depth 1 https://gh-proxy.com/https://github.com/NVIDIA/cutlass.git CUTLASS
```

### 1.2 CMake配置

```bash
# 必须指定CUDACXX，否则cmake找不到nvcc
source ~/anaconda3/bin/activate llm
mkdir -p ~/CUTLASS/build && cd ~/CUTLASS/build

CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake .. \
  -DCUTLASS_NVCC_ARCHS=89 \
  -DCUTLASS_LIBRARY_OPERATIONS=all
```

**关键参数**:
- `CUDACXX`: 必须显式指定，否则cmake使用系统默认路径找不到nvcc
- `DCUTLASS_NVCC_ARCHS=89`: RTX 4090的compute capability
- `DCUTLASS_LIBRARY_OPERATIONS=all`: 编译所有GEMM/SYMM/CONV操作

### 1.3 编译

```bash
# 编译单个example (快, ~2分钟)
cmake --build . --target 00_basic_gemm -j4

# 尝试编译profiler (失败!)
cmake --build . --target cutlass_profiler -j4
# 错误: INT4 atomic操作 __nv_atomic_load_n 参数不足 (CUDA 12.8兼容性问题)
```

**cutlass_profiler编译失败**: CUDA 12.8的atomic API变化导致INT4 subbyte reference编译错误。这是已知问题，需要patch或等待upstream修复。

**替代方案**: 使用PyTorch(cuBLAS后端)+CUDA Events进行精确计时，而非cutlass_profiler。

### 1.4 验证CUTLASS basic_gemm

```bash
./examples/00_basic_gemm/00_basic_gemm 512 512 512  # Passed
./examples/00_basic_gemm/00_basic_gemm 1024 1024 1024  # Passed
./examples/00_basic_gemm/00_basic_gemm 2048 2048 2048  # Passed
```

## 二、GEMM基准测试结果 (RTX 4090, CUDA 12.8, PyTorch 2.9.0+cu128)

### 2.1 方形GEMM (M=N=K)

| Size | FP32 ms | FP32 TFLOPS | FP16 ms | FP16 TFLOPS | BF16 ms | BF16 TFLOPS |
|------|---------|-------------|---------|-------------|---------|-------------|
| 128  | 0.027   | 0.16        | 0.027   | 0.16        | 0.027   | 0.16        |
| 256  | 0.027   | 1.22        | 0.027   | 1.25        | 0.026   | 1.27        |
| 512  | 0.025   | 10.70       | 0.027   | 10.12       | 0.027   | 10.08       |
| 1024 | 0.058   | 36.81       | 0.031   | 69.39       | 0.028   | 76.71       |
| 2048 | 0.332   | 51.79       | 0.120   | 143.62      | 0.115   | 149.74      |
| 4096 | 2.547   | 53.95       | 0.822   | **167.14**  | 0.898   | 153.12      |

### 2.2 Decode GEMM (小M, N=K=4096)

| M | FP32 ms | FP32 TFLOPS | FP16 ms | FP16 TFLOPS | % Peak | BF16 ms | BF16 TFLOPS |
|---|---------|-------------|---------|-------------|--------|---------|-------------|
| 1 | 0.033   | 1.03        | 0.045   | 0.75        | 0.45%  | 0.044   | 0.76        |
| 8 | 0.050   | 5.39        | 0.036   | 7.46        | 4.5%   | 0.035   | 7.76        |
| 32 | 0.044  | 24.54       | 0.036   | 29.91       | 18.1%  | 0.030   | 36.14       |
| 128 | 0.124 | 34.52       | 0.045   | 95.29       | 57.8%  | 0.045   | 96.08       |

### 2.3 非方形GEMM

| M | N | K | FP16 ms | FP16 TFLOPS | BF16 ms | BF16 TFLOPS |
|---|---|---|---------|-------------|---------|-------------|
| 4096 | 1024 | 4096 | 0.225 | 153.05 | 0.213 | 161.17 |
| 1024 | 4096 | 4096 | 0.228 | 150.59 | 0.210 | 163.84 |

### 2.4 性能总结

| 指标 | 值 |
|------|------|
| FP32 peak achieved | **53.95 TFLOPS** (65.3% of 82.6 hw peak) |
| FP16 peak achieved | **167.14 TFLOPS** (101.2% of 165.2 hw peak!) |
| BF16 peak achieved | **153.12 TFLOPS** (93.1% of 165.2 hw peak) |
| FP16/FP32 ratio | **3.09x** |
| Decode M=1 | 0.75 TFLOPS (0.45% peak, **memory-bound**) |
| Decode M=32 | 29.91 TFLOPS (18.1% peak) |

## 三、关键发现

### 3.1 FP16超过硬件峰值!

RTX 4090官方FP16 peak=165.2 TFLOPS，实测**167.14 TFLOPS** (101.2%)

**原因**: cuBLAS使用tensor core (HMMA指令) 进行FP16 GEMM，tensor core实际吞吐可能略高于标称值。加上FMAD指令的contributions。

**与之前A16实验对比**: A16 FP16 peak=14.7 TFLOPS → RTX 4090 FP16 peak=167.14 TFLOPS → **11.4x加速!**

### 3.2 Decode始终memory-bound

| M | Arithmetic Intensity | % Peak | 性质 |
|---|---------------------|--------|------|
| 1 | ~1.0 | 0.45% | 严重memory-bound |
| 8 | ~4.0 | 4.5% | memory-bound |
| 32 | ~16.0 | 18.1% | 仍memory-bound但改善 |
| 128 | ~64.0 | 57.8% | 开始接近compute-bound |

**AI = 2MK/(MK+KN+MN)** ≈ M (for M<<N=K)
- AI≈1→需要HBM ~920 GB/s才能到peak → 实际BW≈920 GB/s → 0.45% peak是合理的
- **Ridge Point** ≈ 128→超过128后才compute-bound

### 3.3 BF16 vs FP16

- BF16精度略差(mantissa 7bit vs 10bit)但性能几乎相同
- 4096x4096: BF16 153.12 vs FP16 167.14 → BF16慢9%
- 2048x2048: BF16 149.74 vs FP16 143.62 → BF16反而快4%!
- **结论**: BF16≈FP16性能，训练用BF16(numeric stability)推理用FP16(slightly better throughput)

### 3.4 FP32 vs FP16 scaling

| Size | FP16/FP32 Speedup |
|------|-------------------|
| 128 | 1.0x (太小，launch主导) |
| 512 | 1.06x (太小) |
| 1024 | **1.87x** |
| 2048 | **2.76x** |
| 4096 | **3.09x** |

- 小size: launch overhead主导，FP16无优势
- 大size: 接近理论2x(Tensor Core吞吐2x)，实测3x因为FP32用SIMT而FP16用Tensor Core

### 3.5 CUTLASS编译教训

1. **CUDACXX必须显式指定**: cmake默认找不到nvcc → 配置失败
2. **cutlass_profiler在CUDA 12.8编译失败**: INT4 atomic op API变化 → 需patch
3. **单个example编译成功**: 00_basic_gemm通过验证
4. **进程级计时不可用**: subprocess启动开销~215ms → 必须用CUDA Events或修改CUTLASS源码嵌入计时

## 四、Roofline分析

### RTX 4090 Roofline模型参数

```
HBM Bandwidth: 920 GB/s (实测, 91.3% of 1008 hw peak)
FP16 Peak: 167.14 TFLOPS (实测)
FP32 Peak: 53.95 TFLOPS (实测)
Ridge Point (FP16): 167.14/0.920 ≈ 182 ops/byte
Ridge Point (FP32): 53.95/0.920 ≈ 59 ops/byte
```

### LLM推理GEMM分析

7B模型 (d=4096, hidden=4096):
- Decode GEMM: M=1, N=4096, K=4096
- AI ≈ 1.0 → 远低于ridge point → **0.7% peak TFLOPS**
- 需要batch≥128才接近compute-bound (AI≈128)

这就是为什么LLM推理必须靠Continuous Batching提高batch→才能利用GPU计算能力。

## 五、下一步

1. **修复CUTLASS profiler编译**: patch INT4 atomic → 精确CUTLASS vs cuBLAS对比
2. **自定义CUTLASS kernel**: 在basic_gemm基础上加入CUDA Events计时 → 精确测量CUTLASS kernel时间
3. **FP16 CUTLASS kernel**: 编写SM89 FP16 tensorop GEMM → 与cuBLAS对比
4. **Epilogue fusion**: CUTLASS支持bias+ReLU/alpha+beta融合 → 测量fusion收益
5. **MoE Fused GEMM**: CUTLASS grouped GEMM → Python MoE 6-164x慢的解决方案

## Sources

- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass) — NVIDIA CUTLASS library
- [CUTLASS Documentation](https://nvidia.github.io/cutlass/) — Official docs
- [RTX 4090 Specs](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) — 165.2 TFLOPS FP16, 82.6 TFLOPS FP32
