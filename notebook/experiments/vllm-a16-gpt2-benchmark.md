# vLLM 0.6.4 推理 Benchmark — A16 GPU

> vLLM 0.6.4+cu118 在 NVIDIA A16 15GB 上跑通，使用 Flash Attention 后端

## 环境配置

```
GPU:        NVIDIA A16 15GB (15.6GB)
CUDA:       11.8 (driver 510.54, CUDA 11.7 driver)
torch:      2.5.1+cu118
vLLM:       0.6.4+cu118 (官方 GitHub Releases cu118 wheel)
模型:       GPT-2 (124M params, 248.9 MB FP16)
Attention:  Flash Attention 后端
KV Cache:   12.29 GB (90% GPU utilization), 22382 GPU blocks
最大并发:    349.72x (1024 tokens/request)
```

## Batch Scaling (max_tokens=32)

| Batch | Time (s) | Total Tokens | tok/s |
|-------|----------|-------------|-------|
| 1     | 0.202    | 32          | 158   |
| 2     | 0.219    | 64          | 292   |
| 4     | 0.223    | 128         | 574   |
| 8     | 0.240    | 256         | 1068  |
| 16    | 0.273    | 512         | 1875  |

**扩展效率**: Batch=8 达到 6.75x (线性 8x), Batch=16 达到 11.87x (线性 16x)

## Decode Length Scaling (batch=4)

| max_tokens | Time (s) | tok/s |
|------------|----------|-------|
| 16         | 0.112    | 570   |
| 32         | 0.221    | 578   |
| 64         | 0.443    | 578   |
| 128        | 0.891    | 575   |

**Decode 吞吐稳定**: 无论生成长度，吞吐保持在 ~575 tok/s (batch=4)

## PyTorch 原生 vs vLLM 对比

| 方法 | Batch=1 tok/s | Batch=8 tok/s | KV Cache |
|------|---------------|---------------|----------|
| PyTorch 原生 | 162 | 1065 | 3.95x 加速 |
| vLLM 0.6.4 | 150-158 | 1068 | 自动管理 |

**结论**: vLLM 的 KV Cache 管理和 Continuous Batching 在小模型 (GPT-2) 上优势不大，
但在大模型和更复杂的请求模式下会有显著提升。

## vLLM 安装要点 (CUDA 11.7 环境)

```
# 必须使用 GitHub Releases 的 cu118 wheel，不能用 PyPI
pip install https://github.com/vllm-project/vllm/releases/download/v0.6.4/vllm-0.6.4%2Bcu118-cp38-abi3-manylinux1_x86_64.whl \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 运行时需要设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/nvidia/cuda_runtime/lib:/path/to/nvidia/cublas/lib:/path/to/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# 注意: v0.6.5 开始停止发布 cu118 wheel, v0.6.4 是最后支持版本
```

## 参考

- vLLM 0.6.4 Release: https://github.com/vllm-project/vllm/releases/tag/v0.6.4
- GPU benchmark 脚本: `tools/pytorch_inference_bench.py`
- vLLM 实验脚本: `tools/vllm_experiment.py`
