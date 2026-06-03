"""
PyTorch Distributed Data Parallel (DDP) CPU Demo

演示核心概念：
1. Process Group 初始化
2. Rank / World Size / Local Rank
3. 分布式梯度同步 (AllReduce)
4. DDP 包装模型

运行方式：
  torchrun --nproc_per_node=4 ddp_cpu_demo.py

注意：此 demo 在 CPU 上运行，不需要 GPU。
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    """初始化进程组"""
    # 使用 gloo 后端（CPU 通信）
    # GPU 上通常用 nccl 后端
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size,
    )
    print(f"[Rank {rank}] Process group initialized. Backend: {dist.get_backend()}")


def cleanup():
    """清理进程组"""
    dist.destroy_process_group()


def demo_basic_operations(rank, world_size):
    """演示基本集合通信操作"""
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Basic Collective Operations")
    print(f"{'='*60}")

    # --- AllReduce: 所有 rank 求和后广播 ---
    tensor = torch.tensor([rank + 1.0])
    print(f"[Rank {rank}] Before AllReduce: {tensor.item()}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] After AllReduce (SUM): {tensor.item()}")
    # 期望值: 1 + 2 + 3 + 4 = 10

    # --- AllReduce with AVG ---
    tensor = torch.tensor([rank + 1.0])
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    print(f"[Rank {rank}] After AllReduce (AVG): {tensor.item()}")
    # 期望值: (1+2+3+4)/4 = 2.5

    # --- Broadcast: rank 0 广播给所有 rank ---
    tensor = torch.tensor([rank * 10.0])
    print(f"[Rank {rank}] Before Broadcast: {tensor.item()}")
    dist.broadcast(tensor, src=0)
    print(f"[Rank {rank}] After Broadcast (from rank 0): {tensor.item()}")
    # 期望值: 所有 rank 都变成 0.0

    # --- AllGather: 收集所有 rank 的数据 ---
    tensor = torch.tensor([rank * 100.0])
    gathered = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    print(f"[Rank {rank}] After AllGather: {[t.item() for t in gathered]}")
    # 期望值: [0.0, 100.0, 200.0, 300.0]

    # --- ReduceScatter: 求和后分片 ---
    tensor = torch.tensor([rank + 1.0, rank + 10.0, rank + 100.0, rank + 1000.0])
    output = torch.zeros(world_size)
    dist.reduce_scatter(output, [tensor.clone() for _ in range(world_size)])
    print(f"[Rank {rank}] After ReduceScatter: {output.tolist()}")


def demo_ddp_training(rank, world_size):
    """演示 DDP 分布式训练"""
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] DDP Training Demo")
    print(f"{'='*60}")

    # 简单模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )

    # 用 DDP 包装
    # DDP 会自动：
    # 1. 在前向时记录哪些参数需要梯度同步
    # 2. 在反向时自动 AllReduce 梯度
    # 3. 用 bucket 优化通信（将小梯度合并发送）
    ddp_model = DDP(model)

    # 损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # 每个 rank 使用不同的数据（模拟数据并行）
    torch.manual_seed(rank)  # 不同 rank 不同数据
    inputs = torch.randn(32, 10)
    labels = torch.randn(32, 5)

    # 训练循环
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # 检查梯度是否已同步（DDP 自动完成）
        # 所有 rank 应该有相同的梯度
        grad_sum = 0.0
        for p in ddp_model.parameters():
            if p.grad is not None:
                grad_sum += p.grad.sum().item()

        optimizer.step()

        if rank == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}, grad_sum={grad_sum:.4f}")

    if rank == 0:
        print("  DDP 训练完成！所有 rank 的梯度已自动同步。")


def demo_gradient_sync_manual(rank, world_size):
    """手动演示梯度同步过程（理解 DDP 内部原理）"""
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Manual Gradient Sync (Understanding DDP Internals)")
    print(f"{'='*60}")

    # 每个 rank 有独立的模型副本
    torch.manual_seed(42)  # 相同种子 → 相同初始参数
    model = nn.Linear(4, 2)

    # 每个 rank 用不同数据计算梯度
    torch.manual_seed(rank)
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)

    loss = ((model(x) - y) ** 2).mean()
    loss.backward()

    print(f"[Rank {rank}] 梯度 (同步前): {model.weight.grad[0, :4].tolist()}")

    # 手动 AllReduce 梯度（这就是 DDP 自动做的事）
    dist.all_reduce(model.weight.grad.data, op=dist.ReduceOp.AVG)

    print(f"[Rank {rank}] 梯度 (同步后): {model.weight.grad[0, :4].tolist()}")
    print(f"[Rank {rank}] 同步后梯度在所有 rank 上相同 (DDP 保证)")


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"Starting: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    if world_size > 1:
        setup(rank, world_size)

        # 演示集合通信
        demo_basic_operations(rank, world_size)

        # 手动梯度同步
        demo_gradient_sync_manual(rank, world_size)

        # DDP 训练
        demo_ddp_training(rank, world_size)

        cleanup()
    else:
        print("请使用 torchrun --nproc_per_node=4 运行此脚本")


if __name__ == "__main__":
    main()
