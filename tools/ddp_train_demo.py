"""
PyTorch DDP 训练完整 Demo — CPU 版本

演示分布式训练的完整流程：
  1. 进程组初始化
  2. 数据并行 + DDP 包装
  3. 梯度累积
  4. 混合精度 (CPU BF16)
  5. 训练循环 + 性能统计
  6. 模型保存/加载

用法:
    torchrun --nproc_per_node=4 tools/ddp_train_demo.py
    torchrun --nproc_per_node=2 tools/ddp_train_demo.py --epochs 5
    python tools/ddp_train_demo.py  # 单进程模式也支持
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


class SmallTransformer(nn.Module):
    """一个小型 Transformer 模型，用于演示 DDP 训练"""

    def __init__(self, vocab_size=1000, d_model=128, nhead=4, num_layers=2, max_seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer(x)
        logits = self.output_head(x)
        return logits


def generate_synthetic_data(vocab_size=1000, seq_len=32, num_samples=1000, seed=42):
    """生成合成语言建模数据"""
    torch.manual_seed(seed)
    # 输入是 token IDs，目标是 shift 一位
    tokens = torch.randint(0, vocab_size, (num_samples, seq_len + 1))
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    return TensorDataset(inputs, targets)


def setup_distributed():
    """初始化分布式环境"""
    if "RANK" in os.environ:
        # torchrun 启动
        dist.init_process_group(backend="gloo")  # CPU 用 gloo
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="DDP Training Demo")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="每 GPU batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--bf16", action="store_true", help="使用 BF16 混合精度")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_distributed = world_size > 1

    if rank == 0:
        print("=" * 60)
        print("PyTorch DDP 训练 Demo — CPU 版本")
        print("=" * 60)
        print(f"  World size: {world_size}")
        print(f"  Device: CPU")
        print(f"  BF16: {args.bf16}")
        print(f"  Gradient accumulation: {args.grad_accum_steps}")
        print(f"  Effective batch size: {args.batch_size * world_size * args.grad_accum_steps}")
        print()

    # 创建模型
    model = SmallTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=4,
        num_layers=2,
        max_seq_len=args.seq_len + 1,
    )

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"  模型参数量: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB FP32)")

    # DDP 包装
    if is_distributed:
        model = DDP(model)
        if rank == 0:
            print("  使用 DistributedDataParallel")

    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # BF16 autocast
    scaler = None

    # 数据
    dataset = generate_synthetic_data(args.vocab_size, args.seq_len, num_samples=1000)
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 训练
    if rank == 0:
        print()
        print("开始训练...")

    global_step = 0
    total_tokens = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        if is_distributed:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # 前向（可选 BF16）
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=args.bf16):
                logits = model(inputs)
                loss = criterion(logits.view(-1, args.vocab_size), targets.view(-1))
                # 梯度累积：损失除以累积步数
                loss = loss / args.grad_accum_steps

            # 反向
            loss.backward()

            epoch_loss += loss.item() * args.grad_accum_steps
            epoch_steps += 1
            total_tokens += inputs.numel()

            # 梯度累积：每 N 步更新一次
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0 and global_step % 10 == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                    print(
                        f"  Epoch {epoch+1}/{args.epochs} | "
                        f"Step {global_step} | "
                        f"Loss: {loss.item() * args.grad_accum_steps:.4f} | "
                        f"Grad norm: {grad_norm:.4f} | "
                        f"Tokens/s: {tokens_per_sec:.0f}"
                    )

        avg_loss = epoch_loss / epoch_steps
        if rank == 0:
            print(f"  → Epoch {epoch+1} 完成, 平均 Loss: {avg_loss:.4f}")

    # 训练结束统计
    elapsed = time.time() - start_time
    if rank == 0:
        print()
        print("=" * 60)
        print("训练完成!")
        print(f"  总步数: {global_step}")
        print(f"  总 tokens: {total_tokens:,}")
        print(f"  总时间: {elapsed:.2f}s")
        print(f"  吞吐量: {total_tokens / elapsed:.0f} tokens/s")
        print(f"  步均时间: {elapsed / max(global_step, 1) * 1000:.1f} ms/step")
        print("=" * 60)

    # 保存模型 (rank 0)
    if rank == 0:
        save_path = os.path.join(os.path.dirname(__file__), "checkpoint_demo.pt")
        model_state = model.module.state_dict() if is_distributed else model.state_dict()
        torch.save(
            {
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "total_params": total_params,
            },
            save_path,
        )
        print(f"  模型已保存到: {save_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
