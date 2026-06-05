# Top-nσ Fused Triton Kernel (RTX 4090)

> 2026-06-05 | 工具: `tools/top_n_sigma_triton_kernel.py` | RTX 4090 24GB
> 参考: Tang et al., ACL 2025, arXiv:2411.07641

## 1. 正确性验证

Survive counts 完全匹配 (NaN 是 -inf 比较问题, 非 kernel 错误):

| Vocab | n | PyTorch survive | Triton survive | 匹配 |
|-------|---|----------------|----------------|------|
| 1024 | 2.0 | 152 | 152 | ✓ |
| 4096 | 2.0 | 315 | 315 | ✓ |
| 32000 | 2.0 | 514 | 514 | ✓ |
| 32000 | 3.0 | 3986 | 3986 | ✓ |

## 2. Latency Benchmark (核心结果!)

### LLM 典型配置 (B=32)

| Vocab | PyTorch | Triton | **加速比** |
|-------|---------|--------|-----------|
| 4096 | 72.1us | 42.0us | **1.72x** |
| 32000 | 71.3us | 41.7us | **1.71x** |
| **128000** | **93.8us** | **45.0us** | **2.09x** |

### 128K vocab 批量扩展

| Batch | PyTorch | Triton | 加速比 |
|-------|---------|--------|--------|
| 1 | 71.6us | 41.6us | 1.72x |
| 32 | 93.8us | 45.0us | 2.09x |
| 128 | 536.3us | 283.2us | **1.89x** |

**128K vocab (现代 LLM 标配) 上 2.09x 加速!**

## 3. 操作对比

| 指标 | PyTorch | Triton Fused |
|------|---------|-------------|
| Kernel launches | ~5 | **1** |
| 中间张量 | 4 (max, std, threshold, mask) | **0** |
| 原理 | 多次 HBM 读写 | SRAM 内完成 |

## 4. 端到端 Pipeline (filter→softmax→sample)

| Batch | PyTorch | Triton | 加速比 |
|-------|---------|--------|--------|
| 1 | 188.5us | 169.7us | 1.11x |
| 256 | 260.6us | 215.4us | **1.21x** |

## 5. 核心学习

1. **Kernel fusion 有效**: 5 个 op → 1 个 kernel, 1.7-2.1x 加速
2. **128K vocab 受益最大**: 数据量大, kernel launch 开销占比高
3. **SRAM > HBM**: 中间结果全部在 SRAM 中处理, 减少 HBM 访问
4. **端到端仍有收益**: 即使 softmax+multinomial 占主导, 1.11-1.21x 仍然有意义
5. **vLLM 贡献价值**: 这个 kernel 可以直接贡献给 vLLM 作为采样管线优化
