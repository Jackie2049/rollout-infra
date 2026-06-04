# vLLM Benchmark on RTX 4090 (OPT-125M)

> 2026-06-05 | vLLM 0.11.0, RTX 4090, OPT-125M FP16
> 模型通过 hf-mirror.com 下载到本地, SCP 传到服务器 (HF 在服务器上无法访问)

## vLLM 配置

- **Engine**: V1, chunked prefill enabled, max_num_batched_tokens=8192
- **KV Cache**: 594,592 tokens, 20.41 GiB available
- **CUDA Graph**: PIECEWISE (67 个 graph) + FULL decode (35 个 graph)
- **torch.compile**: 启用 (level 3), 编译耗时 6.24s
- **Prefix Caching**: 默认开启
- **注意**: FlashInfer 未安装, 使用 PyTorch 原生采样

## Batch Size Scaling

| Batch | 延迟 (s) | Output tok/s | Prompt tok/s |
|-------|----------|-------------|-------------|
| 1 | 0.05 | 659 | 1,690 |
| 4 | 0.04 | 3,010 | 7,714 |
| 8 | 0.04 | 5,704 | 14,616 |
| 16 | 0.05 | 9,497 | 24,336 |
| 32 | 0.07 | 14,669 | 37,588 |
| 64 | 0.10 | 20,712 | 53,074 |
| 128 | 0.16 | 25,514 | 65,380 |
| 256 | 0.28 | 29,285 | 75,043 |
| 512 | 0.54 | 30,244 | 77,501 |

**关键发现**:
- 峰值 30,244 tok/s output @ B=512
- 延迟从 B=1 的 50ms 降到 B=4 的 40ms (batch 帮助摊薄调度开销)
- B=32 之后延迟开始增长 (GPU 饱和)
- Prompt 吞吐峰值 77,501 tok/s (chunked prefill)

## Output Length Scaling (B=32)

| Max Tokens | 延迟 (s) | tok/s |
|------------|----------|-------|
| 16 | 0.04 | 12,679 |
| 32 | 0.07 | 15,019 |
| 64 | 0.13 | 16,332 |
| 128 | 0.24 | 16,995 |
| 256 | 0.48 | 16,936 |
| 512 | 1.04 | 15,759 |

**分析**: Output 越长吞吐越高 (prefill 摊薄), 但 maxtok=256 之后持平 (decode 比例增大)

## 对比 A16 (vLLM 0.6.4)

| 指标 | RTX 4090 (vLLM 0.11) | A16 (vLLM 0.6.4) | 倍数 |
|------|----------------------|-------------------|------|
| B=1 output tok/s | 659 | 163 | 4.0x |
| B=64 output tok/s | 20,712 | 3,729 | 5.6x |
| Peak output tok/s | 30,244 | ~3,729 | 8.1x |
| vLLM 版本 | 0.11.0 | 0.6.4 | 新版本优化 |

**注意**: A16 使用 vLLM 0.6.4, 4090 使用 0.11.0, 版本差异也贡献了性能差距

## vLLM V1 引擎观察

1. **Chunked Prefill**: 默认启用, max_num_batched_tokens=8192
2. **CUDA Graph PIECEWISE**: 67 个 mixed prefill-decode graph + 35 个 decode-only graph
3. **torch.compile**: Level 3, Inductor backend, 编译 6.24s
4. **KV Cache**: 594K tokens ≈ 290 并发 (S=2048), 20.41 GiB
5. **Flash Attention**: V1 默认 backend (无 FlashInfer 时)
6. **模型加载**: 0.25s (OPT-125M 很小), 0.24 GiB

## 外推到 7B 模型

OPT-125M 有 ~125M params, 7B 有 ~7B = 56x 参数
- 模型大小: 125M FP16 ~250MB, 7B FP16 ~14GB (56x)
- 估算 7B decode 吞吐: 30,244 / 56 ≈ 540 tok/s @ B=512
- 实际会更低 (memory-bound, HBM BW 是瓶颈, 不完全是线性缩放)
- 更准确估算: 7B 每个token 读 ~14GB, 890 GB/s → ~63 tok/s @ B=1, 连续 batching 后 ~8K tok/s @ B=512
