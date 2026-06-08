# Triton vs CUDA vs cuBLAS: RTX 4090 Kernel 实测对比

> 2026-06-08 | 4类kernel × 3种实现方式 × 系统化对比
> 基于: tools/triton_vs_cuda_benchmark.py (RTX 4090, Triton 3.5.0, FlashInfer 0.6.12, CUDA C++ RMSNorm)
> 关联: flashinfer-attention-deep-dive.md (FlashInfer), cuda-kernel-optimization-sm89.md (CUDA C++)

## 0. 核心发现: Triton在element-wise ops上比CUDA C++更快!

```
RTX 4090实测Kernel对比:

| Operation | Triton vs PyTorch | Triton vs CUDA C++ | Triton vs cuBLAS |
|-----------|-------------------|--------------------|-----------------|
| RMSNorm   | 1.33x faster ✅   | **1.8x faster** ✅ | N/A             |
| GEMM      | N/A               | N/A                | **1.5x slower** ❌ |
| QKV (3matmul) | N/A        | N/A                | stacked=1.42x ✅ |
| Attention | N/A               | N/A                | FlashInfer>>all |

结论:
  → Triton比CUDA C++更快 for element-wise ops! (出乎意料!)
  → cuBLAS always wins for GEMM/matmul (1.5x faster)
  → Triton best use: element-wise fusion + prototyping
  → CUDA C++ best use: when Triton不可用(老GPU/driver)
  → cuBLAS best use: matmul (永远用cuBLAS, 不用Triton tl.dot())
  → FlashInfer best use: decode attention (15.72x vs SDPA)
```

## 1. RMSNorm + Residual Add: Triton 1.8x faster than CUDA C++!

```
Benchmark配置: RMSNorm(x, w) + residual_add, bf16

| B | H    | PyTorch(ms) | Triton(ms) | CUDA C++(ms) | Tri/Py | Tri/CUDA |
|---|------|------------|-----------|-------------|--------|----------|
| 1 | 2560 | 0.090      | 0.068     | 0.103       | 1.32x  | 0.65     |
| 4 | 2560 | 0.079      | 0.063     | 0.121       | 1.25x  | 0.53     |
| 8 | 2560 | 0.080      | 0.063     | 0.121       | 1.27x  | 0.53     |
|32 | 2560 | 0.086      | 0.067     | 0.122       | 1.28x  | 0.54     |
| 8 | 4096 | 0.080      | 0.059     | 0.121       | 1.36x  | 0.49     |
| 8 | 8192 | 0.088      | 0.066     | 0.114       | 1.33x  | 0.58     |

平均: Triton vs PyTorch=1.33x, Triton vs CUDA C++=0.56 → Triton快1.8x!

为什么Triton更快:
  → Triton launch overhead更低! Triton~0.06ms vs CUDA C++~0.11ms固定开销
  → CUDA C++的ATen dispatch overhead → 更长launch路径
  → Triton JIT编译 → 直接PTX → 更短dispatch
  → 对element-wise ops(0.06-0.08ms), launch overhead占比大 → Triton优势明显

精度对比:
  → Triton vs PyTorch: cos_sim=0.999995-0.999997 → FP32 accumulation精度好
  → CUDA C++ vs PyTorch: cos_sim=1.000000 → bit-for-bit identical! → CUDA C++最精确

关键洞察:
  → 对small kernels(≤0.1ms), launch overhead主导 → Triton有优势
  → 对large kernels(>0.5ms), compute主导 → CUDA C++可能有优势(手动优化)
  → Triton FP32 accumulation→精度好但非bit-exact; CUDA C++→bit-exact
  → 生产场景: Triton已经够好(cos_sim>0.99999), 不需要bit-exact
```

## 2. QKV Projection: Stacked cuBLAS 1.42x faster than sequential

```
7B model config: H=2560, num_heads=20, num_kv_heads=5, d=128

| B | Sequential(3×matmul)(ms) | Stacked(1×matmul)(ms) | Speedup |
|---|--------------------------|----------------------|---------|
| 1 | 0.0717                   | 0.0552               | 1.30x   |
| 4 | 0.0840                   | 0.0573               | 1.47x   |
| 8 | 0.0840                   | 0.0573               | 1.46x   |
|16 | 0.0829                   | 0.0573               | 1.45x   |
|32 | 0.0870                   | 0.0614               | 1.42x   |

Stacked = 1.42x faster!

原因:
  → 3次kernel launch → 1次kernel launch → 减少launch overhead
  → Q/K/V权重stacked → 单次cuBLAS → 自动dispatch到最优TC layout
  → GQA(K/V更小) → stacked后Q仍主导 → 收益略低于MHA

实际推理:
  → vLLM/SGLang都用stacked QKV projection → 生产验证
  → Triton fused QKV不太可能超越stacked cuBLAS → cuBLAS TC路径最优
  → → 结论: QKV用stacked cuBLAS → 不需要Triton/CUDA C++ kernel
```

## 3. Attention Decode: FlashInfer dominates (已有数据)

```
已有RTX 4090实测(见flashinfer-attention-deep-dive.md):

| B  | SDPA(tok/s) | FlashInfer(tok/s) | Speedup |
|----|------------|-------------------|---------|
| 1  | 3,662      | 4,494             | 1.23x   |
| 32 | 9,278      | 145,827           | 15.72x  |

Triton decode vs FlashInfer:
  → Triton decode: 2-3x slower than SDPA(我们的实测)
  → FlashInfer: 15.72x faster than SDPA @B=32
  → → Triton比FlashInfer慢约45x! → FlashInfer是生产答案

结论:
  → Triton decode kernel: 教育价值(g完整实现online softmax + GQA + padding mask)
  → Triton decode: 不适合生产(2-3x慢于SDPA, 45x慢于FlashInfer)
  → → decode attention = FlashInfer, Triton = 教育工具
```

## 4. GEMM: Triton tl.dot() 1.5x slower than cuBLAS

```
| M    | N    | K    | cuBLAS(ms) | Triton(ms) | Tri/cuBLAS | cuBLAS(TF) | Triton(TF) | AI   |
|------|------|------|-----------|-----------|-----------|-----------|-----------|------|
| 1    | 2560 | 2560 | 0.0348    | 0.0666    | 1.91      | 0.38      | 0.20      | 1.0  |
| 4    | 2560 | 2560 | 0.041     | 0.0686    | 1.68      | 1.28      | 0.76      | 4.0  |
| 32   | 2560 | 2560 | 0.0389    | 0.0645    | 1.66      | 10.78     | 6.50      | 31.2 |
| 256  | 2560 | 2560 | 0.0543    | 0.0809    | 1.49      | 61.83     | 41.48     | 213  |
| 512  | 2560 | 2560 | 0.0768    | 0.0994    | 1.29      | 87.38     | 67.54     | 365  |
| 2048 | 2560 | 2560 | 0.1843    | 0.2217    | 1.20      | 145.64    | 121.08    | 787  |
| 4096 | 4096 | 4096 | 0.8888    | 0.9523    | 1.07      | 154.63    | 144.32    | 1365 |
| 8192 | 8192 | 8192 | 6.5987    | 8.5679    | 1.30      | 166.63    | 128.33    | 2730 |

关键趋势:
  → Triton始终比cuBLAS慢! 小GEMM慢1.91x, 大GEMM慢1.07-1.30x
  → 小GEMM差距更大 → Triton的tl.dot()使用FP32 accumulation → cuBLAS用TF32
  → 大GEMM差距缩小 → compute主导 → Triton接近cuBLAS性能
  → Triton不能超越cuBLAS → cuBLAS用了最优TC layout + 底层优化

为什么Triton matmul更慢:
  1. cuBLAS使用TF32(19-bit mantissa) → Triton tl.dot(allow_tf32=False)用FP32 → 更精确但更慢
  2. cuBLAS有最优shared memory tiling → Triton的BLOCK=64/64/32不如cuBLAS调优
  3. cuBLAS自动选择最优算法(heuristic) → Triton只有一个固定算法
  4. cuBLAS可能使用FP8/INT8内部路径 → BF16超出203%理论峰值!

结论:
  → 生产场景: 永远用cuBLAS(torch.nn.functional.linear) → 不用Triton tl.dot()
  → Triton matmul仅用于教育目的 → 理解GEMM原理
  → → Triton的优势在element-wise fusion, NOT matmul!
```

## 5. 为什么Triton Element-wise比CUDA C++更快?

```
原因分析:

Triton kernel launch路径:
  → Python调用 → Triton JIT编译 → 直接PTX codegen → GPU launch
  → overhead: ~0.06ms (从Python到GPU kernel开始执行)

CUDA C++ (PyTorch C++ Extension) kernel launch路径:
  → Python调用 → ATen dispatch → PyTorch C++ binding → CUDA kernel launch
  → overhead: ~0.11ms (ATen dispatch多一步 → 更长路径)

关键: RMSNorm total时间0.06-0.09ms → launch overhead占比70-80%!
  → Triton: compute=0.03ms + launch=0.03ms → total 0.06ms
  → CUDA C++: compute=0.03ms + launch=0.08ms → total 0.11ms
  → → Triton快不是因为kernel更快 → 而是因为dispatch更快!

大kernel情况下:
  → 如果kernel compute=0.5ms → Triton: 0.03+0.5=0.53, CUDA: 0.08+0.5=0.58
  → → Triton/CUDA=0.91 → 仅10%优势 → compute主导时差异缩小
  → → Triton优势在small kernels (launch overhead占比大时)

验证:
  → CUDA C++ cos_sim=1.000000(bit-for-bit) → 最精确
  → Triton cos_sim=0.999995-0.999997 → FP32 accumulation精度好
  → → Triton已经够好, 不需要bit-exact

RTX 4090具体:
  → 大L2 cache(72MB) → RMSNorm数据可能L2 hit → memory-bound不严重
  → → launch overhead占比更高 → Triton优势更明显
  → 小GPU(L2小): data在HBM → compute占比更高 → Triton优势可能缩小
```

## 6. Kernel实现决策树

```
RTX 4090 Kernel决策树:

Element-wise ops (RMSNorm/LayerNorm/softmax/activation):
  → Triton ✅ → 比PyTorch快1.3-1.4x, 比CUDA C++快1.8x!
  → → 推荐Triton for element-wise fusion
  → → 如果需要bit-exact → CUDA C++ (cos_sim=1.0)

Matmul (linear projection/GEMM):
  → cuBLAS ✅ → 比Triton快1.5x → 永远用cuBLAS!
  → → torch.nn.functional.linear / torch.nn.Linear
  → → Triton tl.dot()仅用于教育/理解GEMM原理
  → → Stacked QKV = 1.42x faster → 生产用stacked cuBLAS

Decode Attention:
  → FlashInfer ✅ → 15.72x vs SDPA, 45x vs Triton
  → → 生产必须用FlashInfer
  → → Triton decode: 教育价值(完整实现online softmax)
  → → SDPA: 通用baseline, 但GQA需expand → 慢

Prefill Attention:
  → FlashAttention-2(SDPA backend) ✅ → compute-bound → 正确且快
  → → Triton: 不推荐(prefill compute-bound → Triton更慢)

MoE Expert GEMM:
  → Grouped/cuBLAS ✅ → batched GEMM 6-9% peak → 需大batch
  → → Triton grouped GEMM: 不推荐 → cuBLAS优化更好

FP8 GEMM:
  → TransformerEngine ✅ → 1.48-1.59x faster(B≥4) → fused quantize+GEMM
  → → Triton FP8 GEMM: 不推荐 → TE cuBLASLt路径最优

总结:
  Triton最适用: element-wise fusion (比CUDA C++更快!)
  Triton不适用: matmul/GEMM (cuBLAS永远更快)
  Triton教育用途: 所有kernel (理解原理)
  生产最优: Triton(element-wise) + cuBLAS(matmul) + FlashInfer(attention) + TE(FP8)
```

---

**Sources**:
- RTX 4090 Triton vs CUDA Benchmark: results/triton_vs_cuda_benchmark.json
- FlashInfer Benchmark: results/flashinfer_decode_extended_benchmark.json
- CUTLASS GEMM: results/cutlass_gemm_benchmark.json
- Triton 3.5.0, FlashInfer 0.6.12, CUDA C++ RMSNorm (custom kernel)

**Related notes**: flashinfer-attention-deep-dive.md, cuda-kernel-optimization-sm89.md, cutlass-gemm-benchmark-rtx4090.md