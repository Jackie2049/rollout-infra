# GPU 实验计划

> GPU 到位后的实操实验清单（预计 2026-06-05）

## 实验环境

- GPU: 待定（推荐 A16 15GB VRAM）
- conda env: `ai-infra` (Python 3.11 + PyTorch)
- 工作目录: `~/workspace/rollout-infra/`

## 实验 1：通信 Benchmark

**目标**：实测 AllReduce / ReduceScatter / AllGather 在不同数据规模下的延迟和带宽

```bash
# 单机多 GPU（如果有 2+ GPU）
torchrun --nproc_per_node=2 tools/comm_benchmark.py
```

**测量指标**：
- 各操作的 latency vs data size
- 算出实际带宽利用率
- 对比理论峰值

## 实验 2：vLLM 部署与 Benchmark

**目标**：部署 vLLM 推理服务，实测吞吐量

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-1.5B \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
```

**测量**：
- Single request latency
- Throughput (requests/sec) under different batch sizes
- Prefix caching 加速效果
- KV Cache 显存占用

## 实验 3：小模型分布式训练

**目标**：用 PyTorch DDP 训练小模型，理解分布式训练全流程

```bash
torchrun --nproc_per_node=2 tools/ddp_train_demo.py
```

**内容**：
- 训练一个小 GPT/CNN 模型
- 监控 GPU 利用率、显存占用
- 对比单卡 vs 多卡的 scaling efficiency

## 实验 4：混合精度训练对比

**目标**：对比 FP32 / FP16 / BF16 的训练速度和精度

```bash
python tools/amp_benchmark.py
```

**测量**：
- Training throughput (tokens/sec)
- GPU 显存占用
- 训练 loss 曲线对比

## 实验 5：FlashAttention 效果验证

**目标**：对比标准 Attention vs FlashAttention 的速度和显存

```bash
python tools/flash_attn_benchmark.py
```

**前提**：安装 flash-attn 包

## 实验优先级

1. **实验 2 (vLLM)** — 最直接，最快有产出
2. **实验 1 (通信 benchmark)** — 理解底层
3. **实验 3 (DDP 训练)** — 完整流程体验
4. **实验 4 (混合精度)** — 补充理解
5. **实验 5 (FlashAttention)** — 需要额外安装

## 注意事项

- 15GB VRAM 限制：模型大小 ≤ 7B (BF16)
- 先用小模型验证流程，再尝试大模型
- 记录所有实验数据和观察
- 注意显存监控，避免 OOM
