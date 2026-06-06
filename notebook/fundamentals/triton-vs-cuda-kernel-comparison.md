# Triton vs CUDA C++ Kernel 实测对比

> 2026-06-07 | RTX 4090 (SM 8.9, 24GB, CUDA 12.8) | Triton 3.5.0

## 背景

在实现了 Fused RMSNorm+Add 的 CUDA C++ 和 Triton 两种版本后，进行实测对比。

## 测试 Kernel: Fused RMSNorm + Residual Add

Forward: `y = (x / sqrt(mean(x²) + eps)) * w + r`
Backward: `dx = inv_rms * (dy*w - x_norm*mean(dy*w*x_norm))`, `dr=dy`, `dw=Σ(dy*x_norm)`

## 性能结果 (fwd+bwd combined, RTX 4090)

| Config | PyTorch (ms) | Triton (ms) | CUDA C++ (ms) | Triton Speedup | CUDA Speedup | CUDA/Triton |
|--------|-------------|-------------|---------------|---------------|-------------|------------|
| B=32,H=2048 | 0.371 | 0.252 | 0.155 | 1.47x | 2.38x | 1.62x |
| B=128,H=2048 | 0.347 | 0.248 | 0.153 | 1.40x | 2.28x | 1.61x |
| B=32,H=8192 | 0.336 | 0.244 | 0.164 | 1.38x | 2.04x | 0.67x* |
| B=128,H=8192 | 0.358 | 0.250 | 0.166 | 1.43x | 2.16x | 0.66x* |

*H=8192时差距缩小，可能因为 Triton 的 vectorized load 在大 hidden_size 时更有效

## 为什么 CUDA C++ 更快?

### CUDA C++ 优势
1. **Warp-level butterfly reduction**: 5步 XOR `__shfl_xor_sync` 完成行内求和，零共享内存开销，零 bank conflict
2. **Fused 3-pass per row**: 同一个 kernel 内 3 次遍历行数据 (inv_rms→dot/dw/dr→dx)，避免中间结果写回显存
3. **精细控制**: 1 warp (32 threads) per row，block 尺寸精确控制

### Triton 限制
1. **Program-level reduction**: Triton 的 `tl.sum()` 生成更通用的 reduction 代码，不如 warp-level butterfly 专用高效
2. **无法控制 thread-to-row 映射**: Triton 每个程序实例处理一行，但无法精细控制 32 threads/warp 的映射
3. **atomic_add overhead**: Triton 的 `tl.atomic_add` 可能比 CUDA `atomicAdd` 生成更多冗余代码

### Triton 优势 (开发体验)
1. **代码量**: Triton ~50行 (1个kernel) vs CUDA C++ ~200行 (6个kernel × 3 dtype)
2. **无需 dtype dispatch**: Triton 自动处理数据类型转换，CUDA 需要 6 个独立 kernel
3. **自动内存访问优化**: Triton 自动生成 vectorized load/store，CUDA 需手动设计
4. **开发迭代速度**: Triton 修改→测试 5分钟，CUDA C++ 修改→编译→测试 10-15分钟

## 结论

| 场景 | 推荐 | 原因 |
|------|------|------|
| 生产推理极致性能 | CUDA C++ | 1.6-1.7x faster, 精细 GPU 控制 |
| 快速实验/原型 | Triton | 3x faster development, 1.4x speedup already good |
| 大模型训练 (重复调用) | CUDA C++ | 性能差距在长时间训练中累积 |
| 研究/学术验证 | Triton | 易于理解和修改，适合论文实验 |
| FlashAttention 等复杂 kernel | Triton | Triton 生态更成熟 (vLLM/DeepSeek 用 Triton) |

## 学习要点

1. **Triton 不总比 CUDA C++ 快**: 对于简单的 reduction + elementwise fusion，CUDA warp-level 优化更高效
2. **开发效率 vs 运行性能**: Triton 3x 开发效率 vs CUDA 1.7x 运行性能 — 取决于场景
3. **最优选择**: vLLM 等生产系统混合使用 (Triton 用于注意力kernel, CUDA C++ 用于简单fusion)
4. **`tl.atomic_add` 是 Triton 跨行累加的关键**: `tl.store` 只写不累加，多行并发写同一地址会覆盖而非累加