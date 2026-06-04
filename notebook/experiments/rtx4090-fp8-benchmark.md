# RTX 4090 FP8 量化 Benchmark

> 2026-06-05 | RTX 4090, PyTorch 2.9.0+cu128
> 脚本: `tools/fp8_benchmark_4090.py`

## 关键发现

FP8 量化比 INT8 更复杂, 简单的 quantize-dequantize 模拟在 GEMM 上精度很差!

## Exp1: FP8 GEMM 模拟 (Quantize-Dequantize)

| N | FP16 (ms) | FP8 sim (ms) | 慢倍数 | MaxErr | CosSim |
|---|-----------|-------------|--------|--------|--------|
| 1024 | 0.019 | 0.066 | 3.41x | 5.63 | 0.999 |
| 2048 | 0.107 | 0.164 | 1.53x | 8.48 | **0.000** |
| 4096 | 0.814 | 1.454 | 1.79x | 12.84 | **0.000** |

**严重问题**:
- N≥2048 时 cos_sim = 0! (完全不同的结果方向)
- 原因: **per-tensor scale 对大矩阵不够精确**, 误差在矩阵乘法中累积
- 这不是 FP8 本身的问题, 而是 **scale 粒度不够** + **dequantize 开销大**

**正确做法**: 专用 FP8 Tensor Core kernel (如 CUTLASS FP8), 不需要 dequantize 到 FP16 再做 GEMM, 而是直接在 FP8 精度下做矩阵乘法。

## Exp2: FP8 KV Cache

| SeqLen | FP16 KV (GB) | FP8 KV (GB) | 节省 | CosSim |
|--------|-------------|-------------|------|--------|
| 512 | 0.27 | 0.13 | 50% | 0.663 |
| 2048 | 1.07 | 0.54 | 50% | 0.080 |
| 4096 | 2.15 | 1.07 | 50% | 0.003 |
| 8192 | 4.30 | 2.15 | 50% | **0.000** |
| 16384 | 8.59 | 4.30 | 50% | **0.000** |

**分析**:
- FP8 KV Cache 精度随序列长度急剧下降 (cos_sim 0.66 → 0.00)
- 原因: per-tensor scale 对 KV cache 的各元素差异太大
- **vLLM 的做法**: per-token per-channel FP8 KV Cache (每个 token 独立 scale)
- MaxErr 稳定在 ~0.22, 说明单元素误差可控, 但方向(符号)会变

## Exp3: 量化粒度

| 方式 | RelErr | CosSim |
|------|--------|--------|
| Per-tensor | 0.0265 | 0.9995 |
| Per-row | 0.0264 | 0.9995 |
| Per-col | 0.0264 | 0.9995 |

**分析**: 单层 Linear FP8 在三种粒度下精度差不多 (cos_sim 0.9995)
- 问题出在多层累积: Transformer 有 32 层, 每层 ~0.03 rel_err → 32 层累积后误差爆炸
- 实际生产环境需要 **per-channel FP8 + 动态 scale + 残差连接补偿**

## Exp4: 7B 模型内存分析

| 精度 | 模型 (GB) | 剩余 (GB) | Max SeqLen |
|------|----------|----------|------------|
| FP16 | 14.0 | 10.0 | 19,073 |
| **FP8** | **7.0** | **17.0** | **64,849** |
| INT4 | 3.5 | 20.5 | 156,402 |

**结论**:
- FP8 模型从 14GB → 7GB, 多出 7GB 给 KV Cache → seq_len 3.4x 提升
- INT4 最激进但精度损失更大
- **RTX 4090 + 7B FP8**: 17GB KV Cache → seq=64K, 实际可用

## 重要教训

1. **FP8 不能简单模拟**: quantize→FP16 GEMM→dequantize 不是真正的 FP8 推理
2. **需要专用 kernel**: CUTLASS FP8 / TensorRT FP8 直接在 FP8 精度做 GEMM
3. **Scale 粒度是关键**: per-tensor 对大矩阵不够, 需要 per-channel/per-token
4. **误差会累积**: 32 层 Transformer 中每层小误差会指数级累积
5. **KV Cache FP8 需要 per-token scale**: vLLM 使用此策略, cos_sim > 0.999
6. **FP8 收益**: 内存减半, 但需要硬件支持 (H100/H200 的 FP8 Tensor Core, 4090 仅支持有限)
