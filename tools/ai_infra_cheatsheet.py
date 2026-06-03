#!/usr/bin/env python3
"""AI Infra 常用命令速查表 — 打印或查阅

用法:
  python ai_infra_cheatsheet.py           # 打印全部
  python ai_infra_cheatsheet.py gpu       # 只看 GPU 相关
  python ai_infra_cheatsheet.py nccl      # 只看 NCCL 相关
  python ai_infra_cheatsheet.py training  # 只看训练相关
  python ai_infra_cheatsong.py checkpoint # 只看 Checkpoint 相关
"""

import sys

SECTIONS = {
    "gpu": """
═══════════════════════════════════════════════════════════════
  GPU 监控与调试
═══════════════════════════════════════════════════════════════

# 基础监控
nvidia-smi                              # GPU 状态一览
watch -n 1 nvidia-smi                   # 每秒刷新
nvidia-smi -q -d MEMORY                 # 显存详情
nvidia-smi -q -d TEMPERATURE            # 温度

# 进程级监控
nvidia-smi pmon -c 5                    # 进程级 GPU 使用 (5次)
fuser -v /dev/nvidia*                   # 占用 GPU 的进程

# 显存分析 (Python)
python -c "
import torch
print(f'Allocated: {torch.cuda.memory_allocated()/1e9:.1f}GB')
print(f'Reserved:   {torch.cuda.memory_reserved()/1e9:.1f}GB')
print(f'Max Alloc:  {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
"

# NVLink 状态
nvidia-smi nvlink --status              # NVLink 连接状态
nvidia-smi nvlink --capabilities        # NVLink 能力

# 强制清理
sudo fuser -k /dev/nvidia*              # 杀死占用 GPU 的进程
nvidia-smi --gpu-reset -i 0             # 重置 GPU (需无进程)

# Profile
ncu --set full -o profile.ncu-rep python train.py    # Nsight Compute
nsys profile -o trace python train.py                 # Nsight Systems
""",

    "nccl": """
═══════════════════════════════════════════════════════════════
  NCCL 通信调优
═══════════════════════════════════════════════════════════════

# 调试环境变量
export NCCL_DEBUG=INFO                  # 打印 NCCL 日志
export NCCL_DEBUG_SUBSYS=ALL            # 所有子系统
export NCCL_DEBUG_FILE=/tmp/nccl.log    # 输出到文件

# 网络配置
export NCCL_SOCKET_IFNAME=eth0          # 指定以太网卡
export NCCL_IB_DISABLE=0                # 启用 InfiniBand
export NCCL_IB_HCA=mlx5_0,mlx5_1       # 指定 IB 卡
export NCCL_NET_GDR_LEVEL=5             # GPUDirect RDMA

# 性能调优
export NCCL_MIN_NRINGS=4                # 最小通道数
export NCCL_BUFFSIZE=4194304            # Buffer 大小 (4MB)
export CUDA_DEVICE_MAX_CONNECTIONS=1    # Megatron 推荐设置

# 稳定性
export NCCL_TIMEOUT=1800                # 超时时间 (秒)
export NCCL_COMM_BLOCKING=1             # 阻塞初始化
export NCCL_SHM_DISABLE=1               # 禁用共享内存 (调试用)

# 检查 NCCL 版本
python -c "import torch; print(torch.cuda.nccl.version())"

# IB 状态
ibstat                                  # IB 卡状态
ibv_devinfo                             # IB 设备信息
ibping -S                               # IB ping 服务端
ibping -L <lid>                         # IB ping 客户端
""",

    "training": """
═══════════════════════════════════════════════════════════════
  分布式训练启动
═══════════════════════════════════════════════════════════════

# PyTorch DDP (单机多卡)
torchrun --nproc_per_node=4 train.py

# PyTorch DDP (多机多卡)
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_id=job1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=MASTER_IP:29500 \
  train.py

# Megatron-LM
torchrun --nproc_per_node=8 pretrain_gpt.py \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 2 \
  --num-layers 32 \
  --hidden-size 4096 \
  --num-attention-heads 32 \
  --micro-batch-size 1 \
  --global-batch-size 32 \
  --bf16

# DeepSpeed
deepspeed --num_gpus=8 train.py \
  --deepspeed ds_config.json

# 常见 ds_config.json 关键配置
{
  "bf16": {"enabled": true},
  "zero_optimization": {"stage": 2},
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 1
}

# Slurm 提交
sbatch --gres=gpu:4 --nodes=2 --ntasks-per-node=4 train.sh
""",

    "checkpoint": """
═══════════════════════════════════════════════════════════════
  Checkpoint 管理
═══════════════════════════════════════════════════════════════

# PyTorch 基础保存/加载
torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'ckpt.pt')
ckpt = torch.load('ckpt.pt', map_location='cpu')
model.load_state_dict(ckpt['model'])

# safetensors (更安全快速)
from safetensors.torch import save_file, load_file
save_file(model.state_dict(), 'model.safetensors')
state = load_file('model.safetensors')

# 分片保存 (大模型)
from torch.distributed.checkpoint import save, load
save(state_dict, checkpoint_path, storage_writer=...)
load(state_dict, checkpoint_path, storage_reader=...)

# 检查 checkpoint 文件
python -c "
import torch
ckpt = torch.load('model.pt', map_location='cpu')
for k, v in ckpt.items():
    if hasattr(v, 'shape'):
        print(f'{k}: {v.shape} {v.dtype}')
"

# 查看文件大小
ls -lh checkpoints/
du -sh checkpoints/

# 磁盘空间检查
df -h /path/to/checkpoints
""",

    "inference": """
═══════════════════════════════════════════════════════════════
  推理服务 (vLLM)
═══════════════════════════════════════════════════════════════

# 基础启动
python -m vllm.entrypoints.openai.api_server \\
  --model meta-llama/Llama-3-8B \\
  --tensor-parallel-size 2 \\
  --gpu-memory-utilization 0.9 \\
  --max-model-len 4096

# 测试请求
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "meta-llama/Llama-3-8B", "messages": [{"role": "user", "content": "Hello!"}]}'

# 性能测试
python -m vllm.entrypoints.openai.api_server --model ... &
python benchmarks/benchmark_serving.py \\
  --model meta-llama/Llama-3-8B \\
  --dataset-name random \\
  --random-input-len 1024 \\
  --random-output-len 256 \\
  --num-prompts 100

# 关键参数
--enforce-eager               # 禁用 CUDA graph (省显存)
--enable-prefix-caching       # 启用前缀缓存
--kv-cache-dtype fp8          # FP8 KV Cache
--quantization awq            # AWQ 量化
--max-num-seqs 256            # 最大并发序列数
""",

    "debug": """
═══════════════════════════════════════════════════════════════
  常见问题调试
═══════════════════════════════════════════════════════════════

# CUDA OOM
1. 减小 batch_size
2. model.gradient_checkpointing_enable()
3. 使用 ZeRO-2 或 ZeRO-3
4. 减小 seq_length

# Loss NaN
1. 检查学习率 (太大?)
2. 使用 BF16 替代 FP16
3. gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
4. torch.autograd.set_detect_anomaly(True)

# NCCL Timeout
1. 检查节点网络: ping, ibstat
2. NCCL_DEBUG=INFO 看日志
3. 增大 NCCL_TIMEOUT
4. 检查防火墙和 /etc/hosts

# GPU 利用率低
1. 增大 batch_size 或 gradient_accumulation
2. DataLoader: num_workers=8, pin_memory=True, prefetch_factor=2
3. 使用 torch.compile() 融合操作
4. 检查数据预处理是否是瓶颈

# 梯度异常
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"{name}: grad_norm={p.grad.norm():.4f}, has_nan={p.grad.isnan().any()}")
""",

    "conda": """
═══════════════════════════════════════════════════════════════
  Conda 环境管理 (中国大陆镜像)
═══════════════════════════════════════════════════════════════

# 创建环境
conda create -n myenv python=3.11

# 安装包 (阿里云镜像)
pip install torch -i https://mirrors.aliyun.com/pypi/simple/
pip install package -i https://pypi.tuna.tsinghua.edu.cn/simple/

# Conda 配置清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes

# 常用 AI Infra 包
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install triton vllm transformers accelerate deepspeed
pip install nvidia-nccl-cu12 nvidia-cublas-cu12
""",
}

def main():
    if len(sys.argv) > 1:
        section = sys.argv[1].lower()
        if section in SECTIONS:
            print(SECTIONS[section])
        else:
            print(f"Unknown section: {section}")
            print(f"Available: {', '.join(SECTIONS.keys())}")
    else:
        for section in SECTIONS.values():
            print(section)

if __name__ == "__main__":
    main()
