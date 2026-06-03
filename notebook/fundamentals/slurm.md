# Slurm 集群管理基础

> HPC 集群的事实标准 — 大规模训练的调度器

## 1. Slurm 概述

```
Slurm = Simple Linux Utility for Resource Management

组件:
  slurmctld  — 控制守护进程 (中央调度器)
  slurmd     — 计算节点守护进程 (每个节点一个)
  slurmdbd   — 数据库守护进程 (记账和统计)

命令:
  sinfo      — 查看集群和分区状态
  squeue     — 查看作业队列
  sbatch     — 提交批处理作业
  srun       — 提交交互式作业
  scancel    — 取消作业
  scontrol   — 管理和查询
```

## 2. GPU 训练作业提交

### 2.1 基础 sbatch 脚本

```bash
#!/bin/bash
#SBATCH --job-name=megatron-training
#SBATCH --partition=gpu          # 分区
#SBATCH --nodes=4                # 4 个节点
#SBATCH --ntasks-per-node=8      # 每节点 8 个任务 (8 GPU)
#SBATCH --gpus-per-node=8        # 每节点 8 个 GPU
#SBATCH --time=24:00:00          # 最长 24 小时
#SBATCH --output=logs/%j.out     # 标准输出 (%j = job ID)

# 加载模块
module load cuda/12.1
module load python/3.11

# 获取 master 地址 (Slurm 自动设置)
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# 关键: 限制 CUDA 连接数
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 启动训练
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --micro-batch-size 4 \
    --global-batch-size 256
```

### 2.2 Megatron-LM 官方 sbatch 模板

```bash
#!/bin/bash
#SBATCH --account=my-account
#SBATCH --constraint=a100  # 指定 GPU 型号
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00

# NCCL 优化
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=^docker0,lo

# 关键: CUDA_DEVICE_MAX_CONNECTIONS=1
# 让 NCCL 通信优先于计算，实现通信-计算重叠
export CUDA_DEVICE_MAX_CONNECTIONS=1

srun torchrun --nproc_per_node=8 ...
```

## 3. 关键环境变量

### 3.1 Slurm 自动设置

```bash
SLURM_JOB_ID          # 作业 ID
SLURM_JOB_NUM_NODES   # 节点数
SLURM_NTASKS          # 总任务数
SLURM_NTASKS_PER_NODE # 每节点任务数
SLURM_JOB_NODELIST    # 节点列表
SLURM_PROCID          # 当前任务的 rank
SLURM_LOCALID         # 节点内 local rank
```

### 3.2 torch.distributed 相关

```bash
# torchrun + Slurm 的推荐配置
MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=29500

# 或使用 rdzv ( rendezvous) 模式
torchrun --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID
```

## 4. 常见问题与排查

### 4.1 NCCL 超时

```bash
# 增加 NCCL 超时
export NCCL_MIN_TIMEOUT=1800  # 30 分钟

# 调试 NCCL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### 4.2 作业被抢占

```bash
# 检查作业状态
scontrol show job $SLURM_JOB_ID

# 处理抢占: 保存 checkpoint 后退出
trap 'echo "Preempted at $(date)"; save_checkpoint; exit' SIGTERM
```

### 4.3 GPU 不可见

```bash
# 确保 Slurm 分配了 GPU
echo $CUDA_VISIBLE_DEVICES  # 应该是 0,1,2,...,7

# 检查 GPU 状态
nvidia-smi
```

## 5. 学习要点

1. **sbatch 是核心命令** — 提交分布式训练作业
2. **CUDA_DEVICE_MAX_CONNECTIONS=1** — Megatron 的关键优化
3. **NCCL 环境变量** — 跨节点通信优化的关键
4. **srun 替代 torchrun 的 launcher** — Slurm 管理 process group
5. **checkpoint 是故障恢复的基础** — 作业可能被抢占或超时

## 参考

- [Slurm Documentation](https://slurm.schedmd.com/documentation.html)
- [Megatron-LM Slurm Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html)
