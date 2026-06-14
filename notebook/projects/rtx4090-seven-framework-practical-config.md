# 7框架 RTX 4090 实战配置清单

> 2026-06-15 | RTX 4090: 24GB VRAM, SM 8.9 (Ada), FP16 82.6 TFLOPS, BF16 82.6 TFLOPS
> 单GPU最优(PCIe scaling灾难: 8GPU=0.46x), BF16唯一正确训练精度

## 0. RTX 4090 硬件约束

| 参数 | 值 | 影响 |
|------|-----|------|
| VRAM | 24GB | 单GPU最大内存 → 决定所有配置上限 |
| SM | 8.9 (Ada) | 支持FP8 E4M3/E5M2, 不支持Hopper(TMA/WGMMA/Cluster) |
| FP16/BF16 | 82.6 TFLOPS | 训练主精度 |
| INT8 | 169.6 TFLOPS (GEMM) | 量化推理倍速 |
| HBM带宽 | 890.8 GB/s | decode=memory-bound → 带宽决定推理上限 |
| PCIe | Gen4 x16 | NVLink=否, 多GPU scaling=灾难 |
| CUDA | 12.8 | CUTLASS SM80可用, SM90不可用 |

## 1. DeepSpeed

### 1.1 ZeRO-2 + CPU Adam (推荐, 7B模型)

```yaml
# deepspeed_config.yaml
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "allgather_partitions": true,
    "gradient_accumulation_steps": 4
  },
  "optimizer": {
    "type": "AdamW",
    "params": {"lr": 1e-5, "betas": [0.9, 0.999], "weight_decay": 0.01}
  },
  "gradient_clipping": 1.0,
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 4,
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": true
  }
}
```

**内存估算 (7B)**:
- 参数(BF16): 14GB → 太大! 必须用LoRA
- LoRA(BF16): ~0.2GB → 可行
- 优化器(CPU): ~0 (offload到CPU)
- 梯度: ~0.4GB (LoRA梯度)
- Activation checkpointing: ~1-2GB
- **总计: ~5-8GB → fits RTX 4090**

### 1.2 ZeRO-3 (❌ 不推荐)

```
7B ZeRO-3 N=1: peak=(16/1+4)*14=280GB → 完全不可能
7B ZeRO-3 N=8: peak=(2+4)*14=84GB → 仍超出
ZeRO-3通信量3Ψ/层 → 单GPU无意义(无DP分片收益)
→ RTX 4090: ZeRO-3完全不适合, 用verl代替
```

### 1.3 ZeRO-Offload (⚠️ 可行但慢)

```
7B ZeRO-2+CPU Adam+LoRA → 有效但CPU-GPU来回慢
LoRA rank=16 → 参数量大幅减少 → optimizer offload开销小
→ 实际: verl方案更快(混合引擎)
```

## 2. Megatron-LM

### 2.1 单GPU训练 (受限但可行)

```bash
# Megatron-LM 单GPU配置
python pretrain_gpt.py \
  --num-layers 32 \
  --hidden-size 4096 \
  --num-attention-heads 32 \
  --seq-length 2048 \
  --micro-batch-size 1 \
  --global-batch-size 16 \
  --bf16 \
  --use-flash-attn \
  --recompute-granularity selective \
  --recompute-method block \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --no-async-p2p-comm
```

**内存估算 (7B)**:
- 参数(BF16): 14GB → 24GB勉强, 但activation+梯度不够
- selective recompute → 省activation内存 → ~4-6GB
- micro-batch=1 → 最小内存占用
- **总计: ~18-20GB → 勉强fits RTX 4090**

### 2.2 TP > 1 (❌ 不推荐)

```
RTX 4090无NVLink → PCIe带宽瓶颈
TP=2: AllReduce跨PCIe → 通信>>计算 → 灾难
→ 单GPU最优, TP>1无意义
```

### 2.3 MoE训练 (⚠️ EP=1)

```bash
# MoE 单GPU配置 (DeepSeek-V2风格)
--expert-model-parallel-size 1  # EP=1 → 所有expert在同一GPU
--num-experts 8  # 少量expert → 7B MoE每expert ~1.75B
--moe-grouped-gemm  # 合并expert GEMM → 更高效
--moe-per-layer-fft  # 或使用MoE层
```

**注意**: EP=1时所有expert在同一GPU → 内存压力大 → 需LoRA或更小模型

## 3. vLLM

### 3.1 7B 推理最优配置 (单GPU)

```bash
# vLLM 7B INT4 + INT8 KV 最佳吞吐
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --quantization awq_marlin \
  --kv-cache-dtype fp8_e5m2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --dtype float16 \
  --tensor-parallel-size 1
```

**吞吐估算 (GQA-8 + FlashInfer)**:
- INT4 weight + INT8 KV + GQA-8 → **4,791 tok/s** (decode)
- 启用prefix caching → 多对话30-50%节省

### 3.2 Speculative Decoding 配置

```bash
# EAGLE speculative decoding
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --speculative-config model=eagle-model,draft=5 \
  --quantization awq_marlin \
  --kv-cache-dtype fp8_e5m2 \
  --gpu-memory-utilization 0.85
```

**吞吐**: ~9,088 tok/s (EAGLE d=5)

### 3.3 LoRA Serving

```bash
# vLLM LoRA serving
python -m vllm.entrypoints.openai.api_server \
  --model base-model \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 16 \
  --lora-modules lora1=/path/to/lora1 lora2=/path/to/lora2
```

### 3.4 SGLang替代方案

```bash
# SGLang 推理 (Overlap调度优势)
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --quantization awq_marlin \
  --kv-cache-dtype fp8_e5m2 \
  --mem-fraction-static 0.85
```

**优势**: Overlap调度 → decode吞吐+20-30%

## 4. verl

### 4.1 GRPO训练 (推荐, 7B模型)

```yaml
# verl GRPO config (OmegaConf)
algorithm:
  adv_estimator: grpo
  use_kl_in_reward: false

actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-7B-Instruct
    dtype: bfloat16
    lora:
      rank: 16
      lora_alpha: 32
  rollout:
    name: vllm
    mode: async
    n: 8  # GRPO每个prompt 8个采样
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.85
  actor:
    strategy: fsdp2
    ppo_epochs: 1
    clip_range: 0.2
    use_kl_loss: true
    kl_loss_coef: 0.05
    use_prefix_grouper: true
    optimizer:
      type: cpu_adam  # CPU Adam → 省GPU内存
    grad_clip: 1.0
    micro_batch_size_per_gpu: 4

reward_manager:
  type: function  # 自定义reward函数
```

**内存估算 (7B GRPO)**:
- 参数(BF16): 14GB → LoRA后 ~0.2GB
- LoRA梯度: ~0.4GB
- Optimizer(CPU offload): ~0GPU
- Activation: ~2-4GB (micro_bs=4, checkpointing)
- vLLM rollout: ~8-12GB (共享GPU → 需切换)
- **总计: ~20.04GB → fits RTX 4090**

### 4.2 PPO训练 (⚠️ 需Critic, 内存更大)

```yaml
algorithm:
  adv_estimator: gae

critic:
  model:
    path: Qwen/Qwen2.5-7B-Instruct
    lora:
      rank: 16
  optimizer:
    type: cpu_adam
```

**内存**: Critic额外~2-4GB → 总计22-24GB → 勉强fit → 建议GRPO代替

### 4.3 异步Rollout配置

```yaml
actor_rollout_ref:
  rollout:
    mode: async  # 异步rollout → actor和rollout交替使用GPU
    name: vllm
```

**关键**: async mode → actor训练完→释放GPU→rollout生成→释放GPU→循环 → 24GB够用

## 5. MindIE

### 5.1 RTX 4090 (❌ 不支持)

```
MindIE = 华为昇腾NPU专用 → 不支持NVIDIA GPU
→ RTX 4090无法使用MindIE
→ 替代方案: vLLM (推理) + verl (训练)
```

### 1.2 昇腾910B配置 (如使用高校昇腾资源)

```yaml
# MindIE 昇腾910B配置 (64GB HBM)
mindie-service:
  model_name: qwen3-8b
  world_size: 1
  quantize: w8a8  # INT8量化
  max_batch_size: 32
  max_seq_len: 4096
```

**吞吐**: FP16 320 TFLOPS → 比4090快3.9x → 但需昇腾NPU环境

## 6. rLLM

### 6.1 rLLM + verl backend (推荐)

```bash
# rLLM GRPO训练 (tinker backend = 单GPU)
rllm train gsm8k \
  --algorithm estimator=grpo \
  --rollout.n=8 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --backend verl \
  --sandbox local \
  --max-response-length 1024
```

**配置**: 与verl GRPO相同 → rLLM只是上层编排

### 6.2 rLLM + tinker backend (单GPU调试)

```bash
# rLLM tinker backend (更轻量, 适合调试)
rllm train gsm8k \
  --algorithm estimator=grpo \
  --rollout.n=4 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --backend tinker \
  --sandbox local
```

**优势**: tinker更轻量 → 不需要vLLM → 单GPU更方便

### 6.3 rLLM eval (不训练, 只评估)

```bash
# rLLM 评估 (60+内置benchmark)
rllm eval gsm8k \
  --model Qwen/Qwen2.5-7B-Instruct \
  --sandbox local
```

## 7. PyTorch

### 7.1 FSDP2训练 (7B)

```python
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.fsdp import FSDP, MixedPrecisionPolicy

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# BF16 mixed precision
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
)

# FSDP2 wrapping
model = FSDP(
    model,
    device_mesh=device_mesh,
    shard_params=[Shard(0)],  # 沿dim=0分片
    mp_policy=mp_policy,
)

# CPU Adam optimizer (offload optimizer states)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
)
```

**内存估算 (7B, 单GPU)**:
- FSDP2单GPU: 不分片(N=1) → 全参数14GB → 需LoRA
- LoRA+FSDP2: ~0.2GB参数 + 梯度0.4GB + CPU optimizer → 可行

### 7.2 torch.compile训练加速

```python
# Compile model for faster training
model = torch.compile(model, mode="reduce-overhead")

# 或编译后FSDP2
model = FSDP(torch.compile(model), ...)
```

**注意**: compile + FSDP2 → 需确保没有graph break → LoRA layer需custom_op注册

### 7.3 custom_op注册 (融合kernel)

```python
from torch._custom_op import custom_op

@custom_op("my::fused_rms_norm")
def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    ...

@fused_rms_norm.register_kernel("CUDA")
def fused_rms_norm_cuda(x, weight, eps):
    # Triton kernel实现
    ...

@fused_rms_norm.register_fake
def fused_rms_norm_fake(x, weight, eps):
    return torch.empty_like(x)

# → compile兼容! 不产生graph break
```

## 8. 框架组合实战方案

### 方案A: GRPO训练 (最实用)

```
rLLM编排 → verl backend → vLLM rollout → FSDP2训练 → CPU Adam optimizer
→ 7B + LoRA + BF16 → 20.04GB → RTX 4090可行
→ 一行命令: rllm train gsm8k --backend verl --rollout.n=8
```

### 方案B: 纯推理服务

```
vLLM/SGLang → INT4 weight + INT8 KV + prefix caching
→ 7B → 4,791 tok/s (vLLM) / ~5,800 tok/s (SGLang overlap)
→ 一行命令: vllm serve Qwen/Qwen2.5-7B-Instruct --quantization awq_marlin
```

### 方案C: 推理+SpecDec

```
vLLM → INT4 + INT8KV + EAGLE speculative decoding
→ 7B → 9,088 tok/s → RTX 4090推理极致
→ 一行命令: vllm serve model --speculative-config model=eagle
```

### 方案D: SFT预训练 → GRPO微调

```
Step1: DeepSpeed ZeRO-2+CPU Adam+LoRA → SFT暖启动
Step2: verl/rLLM → GRPO → RL微调
→ SFT→GRPO 93% eval/0% gap vs 纯RL 40-52%
→ 必须先SFT! GRPO暖启动=2x差距
```

### 方案E: 多模态推理

```
vLLM → Qwen2VL/Qwen3VL → 多模态推理
→ RTX 4090: 7B多模态可用, 但图像KV内存大 → 需INT8 KV
→ 一行命令: vllm serve Qwen/Qwen2-VL-7B-Instruct --kv-cache-dtype fp8
```

## 9. 通用环境配置

### 9.1 conda环境

```bash
# 本地环境 (Mac, 无GPU)
conda create -n ai-infra python=3.12
conda activate ai-infra
pip install -i https://mirrors.aliyun.com/pypi/simple/ numpy torch transformers

# GPU环境 (RTX 4090)
conda create -n gpu-infra python=3.12
conda activate gpu-infra
pip install -i https://mirrors.aliyun.com/pypi/simple/ \
  torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -i https://mirrors.aliyun.com/pypi/simple/ \
  vllm verl transformers accelerate
```

### 9.2 Git配置

```bash
# GitHub镜像 (中国网络)
git config --global url."https://gh-proxy.com/https://github.com/".insteadOf "https://github.com/"

# Commit签名
git config user.name "jackie2049"
git config user.email "your-email"
# 所有commit: Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

### 9.3 GPU环境检查脚本

```bash
#!/bin/bash
# GPU健康检查
nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## 10. 决策树

```
RTX 4090 24GB + 单GPU (PCIe scaling灾难 → 不多GPU)

训练场景:
  SFT? → DeepSpeed ZeRO-2 + CPU Adam + LoRA (方案D Step1)
  GRPO RL? → verl/rLLM + LoRA + CPU Adam (方案A)
  PPO RL? → verl + Critic + LoRA → 勉强可行, 建议GRPO代替
  全参数训练? → ❌ 不可能 (7B BF16=14GB, 加optimizer >24GB)

推理场景:
  单模型服务? → vLLM INT4 + INT8KV (方案B)
  最高吞吐? → vLLM INT4 + INT8KV + EAGLE (方案C)
  多对话/prefix共享? → SGLang (Overlap调度+RadixAttention)
  LoRA动态服务? → vLLM --enable-lora
  多模态? → vLLM Qwen2VL + FP8 KV (方案E)

研究场景:
  ZeRO-3实验? → ❌ 单GPU不适用
  FSDP2实验? → ✅ 可行(compile+custom_op)
  MoE实验? → EP=1, 少expert
  PD分离? → PCIe KV transfer=3%TTFT → 可行但收益小
```

## 11. 内存预算表 (7B模型, BF16)

| 方案 | 参数 | 梯度 | Optimizer | Activation | 总计 | 可行? |
|------|------|------|-----------|------------|------|-------|
| 全参数 | 14GB | 14GB | 12GB(CPU) | 2GB | 28-30GB | ❌ |
| LoRA r=16 | 0.2GB | 0.4GB | 0(CPU) | 2GB | 2.6GB | ✅ |
| LoRA+GRPO | 0.2GB | 0.4GB | 0(CPU) | 4GB | 4.6GB | ✅ |
| LoRA+GRPO+vLLM | 8-12GB(vLLM) | 0.4GB | 0(CPU) | 4GB | 12-16GB | ✅ (交替) |
| INT4推理 | ~3.5GB | 0 | 0 | KV~2GB | 5.5GB | ✅ |
| INT4+INT8KV推理 | ~3.5GB | 0 | 0 | KV~1GB | 4.5GB | ✅ |

**关键**: 全参数训练不可能 → LoRA是RTX 4090唯一出路
