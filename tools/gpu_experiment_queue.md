# GPU Experiment Queue
> 等待 GPU 可用时执行的实验列表。按优先级排序。
> GPU 登录: `sshpass -p 'TUR]Nr3fyxM%7)iD' ssh -p 28959 root@hz-t3.matpool.com`
> 推荐 GPU: RTX 4090 @ ¥2.20/GPU-hr (CUDA 12.x, 40 TFLOPS, 24GB)

## Priority 1: Critical (需要 CUDA 12+)

### 1.1 PR #7 Top-n-sigma Logits Processor 测试
- **目的**: 验证 vLLM PR #7 的 Top-n-sigma logits processor
- **需求**: CUDA 12+, 最新 vLLM (开发版)
- **GPU**: RTX 4090 (推荐) 或 RTX 3090
- **脚本**: 需要编写测试脚本 (使用 V1 LogitsProcessor/BatchUpdate API)
- **状态**: 代码已写好, 等待 GPU 测试

### 1.2 Triton Kernel 实测
- **目的**: 在真实 GPU 上运行 5 个 Triton kernel 实验
- **脚本**: `tools/triton_kernel_practice.py` (已有 CPU 版, 需加 GPU 路径)
- **需求**: CUDA 12+ (Triton 3.x)
- **预期**: Vector Add ~170GB/s, Fused Bias+ReLU 2-3x 加速, GEMM 接近 cuBLAS

## Priority 2: Performance Benchmark

### 2.1 vLLM Benchmark (新版)
- **目的**: 在 CUDA 12+ GPU 上测试最新 vLLM (v0.22.0) 性能
- **模型**: OPT-125M, OPT-350M, 可能试 Llama-3.2-1B
- **指标**: 吞吐 (tok/s), 延迟 (TTFT/TPOT), 并发能力
- **对比**: 与之前 A16 (CUDA 11.7) 的结果对比

### 2.2 FlashAttention Benchmark
- **目的**: FlashAttention-2 vs PyTorch SDPA vs naive attention
- **参数**: seq_len=512→16384, batch=1→32
- **预期**: FlashAttention 5-10x 加速 @长序列, O(N) 内存 vs O(N²)

### 2.3 CUDA Graph Benchmark
- **目的**: CUDA Graph vs 非 Graph 延迟对比
- **测试**: 不同 op 数量 (10→200), 不同 batch size
- **预期**: 5-10x kernel launch 开销消除

### 2.4 Speculative Decoding (EAGLE) Benchmark
- **目的**: 测试 EAGLE speculative decoding 实际加速比
- **需求**: 需要 EAGLE draft model
- **预期**: 接受率 0.7-0.9, 加速 2-3x

## Priority 3: Advanced Experiments

### 3.1 Multi-Stream Overlap (需 A100/H100)
- **目的**: 验证通信-计算重叠在多 SM GPU 上的效果
- **需求**: 108+ SM + NVLink (A100/H100)
- **预期**: 100% 通信隐藏

### 3.2 Tensor Parallelism Benchmark
- **目的**: 测试 TP=2/4 的扩展效率
- **需求**: 多 GPU 或大 GPU
- **预期**: TP=2 ~95% 效率, TP=4 ~90%

### 3.3 Triton vs CUDA C++ Extension 性能对比
- **目的**: 同一个 kernel 分别用 Triton 和 CUDA C++ 实现, 对比性能
- **Kernel**: Vector Add, Fused LayerNorm, Softmax
- **预期**: Triton 达到 CUDA 的 80-95%

### 3.4 MoE Serving Benchmark
- **目的**: 测试 MoE 模型 (Mixtral-8x7B) 的推理性能
- **需求**: 24GB+ VRAM
- **指标**: 吞吐, 延迟, EP 通信开销

## Completed Experiments (A16/CUDA 11.7)
- Vector Add (自定义 CUDA): 170 GB/s
- Fused Bias+ReLU: 2.32x 加速
- GEMM Benchmark: 15.1 TFLOPS @2048
- HBM Bandwidth: ~170 GB/s
- CUDA Streams: 双 stream 重叠在 A16 上无收益 (10 SM)
- LayerNorm Fusion: 3.2x 加速
- FP16 GEMM: 5.7x over FP32
- vLLM OPT-125M: 163→3729 tok/s (batch 1→64)
- Triton 不兼容 (CUDA 11.7 + PTX 加载失败)
