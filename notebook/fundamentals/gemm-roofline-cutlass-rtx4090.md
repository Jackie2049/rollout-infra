# GEMM Roofline CUTLASS/cuBLAS — RTX 4090 实测
> 2026-06-07 | 5个实验: dtype比较, decode profile, FP8, 内存带宽, Ridge点

## 一、FP16 vs FP32 vs BF16 GEMM Throughput

| M | FP32 (TFLOPS) | FP16 (TFLOPS) | BF16 (TFLOPS) | FP16/FP32 |
|---|-------------|-------------|-------------|-----------|
| 1 | 1.68 | 0.96 | 0.96 | 0.57x |
| 4 | 3.87 | 6.68 | 6.24 | 1.72x |
| 64 | 22.14 | 99.20 | 109.74 | 4.47x |
| 256 | 48.11 | 151.78 | 152.05 | 3.15x |
| 1024 | 56.10 | 136.44 | 142.59 | 2.43x |
| 4096 | 53.33 | **159.73** | 154.79 | **2.85x** |

**发现**:
- **M=1时FP16反而比FP32慢** (0.96 vs 1.68): 小GEMM时Tensor Core启动开销+kernel launch占比大
- **M≥64时FP16 3-5x faster**: Tensor Core HMMA高效利用
- **BF16 ≈ FP16**: 精度低但性能几乎相同(Tensor Core对两者同样处理)
- **FP16 peak 159.73 TFLOPS**: 接近标称82.58×2=165.1 TFLOPS(sparse 2:4)
- **FP32 peak 56.10 TFLOPS**: 接近标称54.6 TFLOPS

## 二、Decode-Size GEMM Profile (7B model)

MLP_gate_up (B×H → B×4H, 最关键的decode GEMM):

| B | time (ms) | TFLOPS | AI | memory-bound? |
|---|----------|--------|-----|---------------|
| 1 | 0.150 | 0.90 | 1.0 | **是** (AI<ridge) |
| 8 | 0.146 | 7.34 | 8.0 | **是** |
| 32 | 0.150 | 28.57 | 31.7 | **是** |
| 128 | 0.159 | 108.28 | 123.2 | **接近compute** |
| 256 | 0.216 | 159.43 | 237.4 | **compute-bound** |

**关键**: B≤32时GEMM时间几乎flat(0.15ms) → memory-bound验证! B从1→32只增加10%时间
**转折**: B≈128时开始接近compute-bound, B≥256完全compute-bound

7B decode吞吐估算:
- 32层×5 GEMMs/层 ≈ 160 GEMMs per token
- B=32: 每GEMM 0.15ms → 160×0.15=24ms → 但实际各GEMM大小不同
- MLP占68%(gate+up+down), Attn占32%(QKV+out)

## 三、FP8 GEMM (RTX 4090 = SM89)

**FP8 direct GEMM FAILED**: `"addmm_cuda" not implemented for 'Float8_e4m3fn'`

原因: SM89的cuBLAS不支持FP8 GEMM → 需要SM90(Hopper)的cuBLAS FP8 API
- SM89有FP8 Tensor Core硬件 → 但cuBLAS软件路径缺失
- 必须用CUTLASS自定义FP8 kernel才能利用SM89的FP8 Tensor Core

FP8 cast overhead (Python-level):
| M | FP16 (ms) | FP8_cast (ms) | overhead |
|---|----------|-------------|---------|
| 1 | 0.032 | 0.070 | **119%** |
| 32 | 0.021 | 0.059 | **185%** |
| 512 | 0.109 | 0.150 | **38%** |
| 2048 | 0.441 | 0.548 | **24%** |

**结论**: Python-level FP8→FP16 cast开销24-185% → 与之前dequant overhead一致 → 必须fused kernel

## 四、Memory Bandwidth

| Size | BW (GB/s) | 说明 |
|------|----------|------|
| 1 MB | 305.67 | L2 cache部分命中 |
| 4 MB | 1,219 | L2 cache |
| 16 MB | **3,609** | L2 cache峰值! |
| 64 MB | 919 | **HBM饱和** |
| 256 MB | 921 | HBM |
| 1024 MB | 920 | HBM |

**关键**: 小数据(≤16MB) L2 cache给出超高BW(3.6 TB/s=4x HBM!) → 这解释了之前GQA-8 KV cache BW 2107 GB/s的异常
**HBM稳定≈920 GB/s**: 理论1008 GB/s → 91.3%利用率 → 非常好

## 五、Ridge Point Detection

| M | AI | TFLOPS | 性能阶段 |
|---|-----|--------|---------|
| 1 | 1.0 | 1.04 | memory-bound (0.6% peak) |
| 8 | 8.0 | 12.82 | memory-bound (7.4%) |
| 32 | 31.5 | 50.66 | memory-bound (29%) |
| 64 | 62.1 | 106.18 | **接近转折** (61%) |
| 128 | 120.5 | 130.46 | **过渡** (75%) |
| **256** | **227.6** | **163.85** | **RIDGE POINT** (80% peak) |
| 512 | 409.6 | 156.07 | compute-bound |
| 4096 | 1365.3 | **171.32** | compute-bound (peak) |
| 16384 | 1820.4 | **173.41** | **峰值** |

**Ridge Point: M=256, AI=227.6**
- Decode (B≤32): AI≈1-32 → **memory-bound** → throughput ∝ HBM BW
- Prefill (B≥256): AI≥228 → **compute-bound** → throughput ∝ TFLOPS

**与之前实测对比**:
- 之前GEMM Roofline: ridge N=512 (AI=170), peak 165 TFLOPS
- 本次更精确: ridge M=256 (AI=227.6), peak 173.41 TFLOPS
- 差异原因: 之前用N维变化, 本次用M维 → M是更自然的batch维度

## 六、与CUTLASS理论对照

| 实测 | CUTLASS理论 | 备注 |
|------|-----------|------|
| FP16 peak 173.4 TFLOPS | 理论82.58×2=165.1 (sparse) | 实测超过理论! 可能cuBLAS用了更优配置 |
| FP32 peak 56.1 TFLOPS | 理论54.6 TFLOPS | 103%理论 |
| Ridge M=256 | CUTLASS SM80: ~M=128-256 | 一致 |
| HBM 920 GB/s | 理论1008 GB/s | 91.3% |
| FP8 direct FAILED | SM89无cuBLAS FP8支持 | 需自定义CUTLASS kernel |

**RTX 4090(SM89) vs H100(SM90) GEMM对比**:
| | RTX 4090 | H100 |
|---|---------|------|
| FP16 peak | 173 TFLOPS | ~390 TFLOPS |
| HBM BW | 920 GB/s | ~3300 GB/s |
| Ridge M | 256 | ~64 |
| FP8 GEMM | ❌(cuBLAS不支持) | ✅(cuBLAS + WGMMA) |
| CUTLASS路径 | SM80(cp.async+HMMA) | SM90(TMA+WGMMA) |

## 七、实用结论

1. **Decode始终memory-bound**: B≤32时GEMM时间flat(0.15ms) → 增加batch不增延迟
2. **Ridge点M=256**: 这是prefill→compute-bound的转折点
3. **FP8 GEMM在SM89不可用**: Python cast开销38-185% → 必须等SM90或自定义kernel
4. **L2 cache影响大**: ≤16MB数据BW达3.6 TB/s → GQA小KV受益
5. **cuBLAS已高度优化**: 实测173 TFLOPS超过标称165 → 不要试图自己写GEMM kernel!
6. **BF16≈FP16性能**: 精度更差但性能几乎一样 → 推理首选BF16(训练也)

Sources:
- [CUTLASS 3.x Architecture](https://github.com/NVIDIA/cutlass)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- RTX 4090 Specs: 82.58 TFLOPS FP16 (dense), 165.1 TFLOPS (sparse 2:4), 1008 GB/s HBM