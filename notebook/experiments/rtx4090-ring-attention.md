# Ring Attention / Sequence Parallelism 实测 (RTX 4090)

> 2026-06-05 | 工具: `tools/sequence_parallel.py` | RTX 4090 24GB
> 参考: Liu et al., 2023 (Ring Attention), Li et al., 2023 (DeepSpeed Ulysses)

## 1. Ring Attention 精度

| N | P=2 | P=4 | P=8 |
|---|-----|-----|-----|
| 256 | 4.17e-7 | 4.17e-7 | 5.36e-7 |
| 512 | 4.77e-7 | 5.07e-7 | 3.58e-7 |
| 1024 | 5.07e-7 | 5.66e-7 | 5.07e-7 |

**所有配置 cos_sim = 1.0!** Ring Attention 数学上精确 (online softmax).

## 2. 通信量分析 (LLM 级别: B=1, H=32, D=128)

| N | P | Ring (MB) | Ulysses (MB) | Ratio |
|---|---|-----------|-------------|-------|
| 2048 | 2 | 33.6 | 16.8 | 2.0x |
| 2048 | 4 | 33.6 | 25.2 | 1.3x |
| 2048 | 8 | 33.6 | 29.4 | 1.1x |
| 8192 | 4 | 134.2 | 100.7 | 1.3x |
| 16384 | 8 | 268.4 | 234.9 | 1.1x |

**关键发现**:
- **Ring 通信量与 P 无关**: O(B*H*N*D) — 不管用多少设备
- **Ulysses 通信量随 P 增加**: all-gather 需要 (P-1)/P 比例
- P 越大, Ring 和 Ulysses 通信量越接近
- Ring 的优势: 不受 H 限制, 可以并行任意长序列

## 3. 时间模拟

| N | P | Naive | Ring Sim | Ratio |
|---|---|-------|----------|-------|
| 512 | 2 | 0.09ms | 0.93ms | 10.6x |
| 512 | 4 | 0.09ms | 3.43ms | 39.2x |
| 2048 | 2 | 1.89ms | 3.97ms | 2.1x |
| 2048 | 4 | 1.89ms | 3.46ms | 1.8x |

**Python 模拟开销**: Ring 比 naive 慢 (Python 循环), 但:
- N 越大, 开销越小 (2048 时仅 1.8x)
- 实际 CUDA kernel: 通信-计算重叠 → 接近免费
- 关键: 需要 NVLink (300 GB/s), PCIe 太慢

## 4. 核心学习

1. **Ring Attention 精确**: online softmax + 逐步更新 = 数学等价
2. **通信量与 P 无关**: 总量 = B*H*N*D, 不随设备数增加
3. **内存 ∝ N/P**: 每个设备只存 N/P 个 KV token → 长序列关键
4. **NVLink 是前提**: 通信-计算重叠需要高带宽互联
5. **Ulysses 更简单**: 但受 H 限制, 且不支持超长序列
6. **真实场景**: 需要自定义 CUDA kernel + 通信库 (NCCL)
