"""
集合通信原语可视化演示

用纯 Python + NumPy 模拟 AllReduce / ReduceScatter / AllGather / Broadcast
的数据流动过程，帮助理解分布式训练中的通信模式。

不需要 GPU，直接运行即可。

用法:
    python collective_ops_viz.py          # 运行所有演示
    python collective_ops_viz.py allreduce # 只运行 AllReduce
"""

import numpy as np


def visualize_tensor(rank, tensors, label=""):
    """可视化所有 rank 上的张量"""
    print(f"  [{label}]")
    for r, t in enumerate(tensors):
        print(f"    Rank {r}: {t}")


def demo_allreduce():
    """AllReduce: 所有 rank 的数据求和后广播给所有 rank

    在 DDP 中用于梯度同步，在 TP 中用于聚合部分结果。

    通信量: O(N * data_size)，其中 N = world_size
    """
    print("\n" + "=" * 60)
    print("AllReduce (SUM)")
    print("=" * 60)

    world_size = 4
    np.random.seed(42)

    # 每个 rank 有自己的数据
    inputs = [np.random.randint(1, 10, size=4) for _ in range(world_size)]
    print("\n初始状态 (每个 rank 有不同的数据):")
    visualize_tensor("Before", inputs, "每个 rank 的数据")

    # AllReduce: 求和后所有 rank 得到相同结果
    total = sum(inputs)
    outputs = [total.copy() for _ in range(world_size)]

    print(f"\n  计算: AllReduce(SUM) → 所有 rank 得到逐元素求和")
    print(f"  数学: output[i] = Σ(rank_j 的 data[i]) for all j")
    print(f"\nAllReduce 后 (所有 rank 相同):")
    visualize_tensor("After", outputs, "所有 rank 的结果")

    # 模拟 Ring AllReduce 的过程
    print(f"\n  --- Ring AllReduce 算法模拟 ---")
    print(f"  数据总量: {world_size} × {len(inputs[0])} elements = {world_size * len(inputs[0])}")
    chunk_size = len(inputs[0]) // world_size
    print(f"  每个 chunk 大小: {chunk_size} elements")

    # Phase 1: Reduce-Scatter (N-1 步)
    print(f"\n  Phase 1: Reduce-Scatter ({world_size - 1} 步)")
    buffers = [inp.copy().astype(float) for inp in inputs]
    for step in range(world_size - 1):
        new_buffers = [buf.copy() for buf in buffers]
        for rank in range(world_size):
            send_chunk = (rank - step) % world_size
            recv_chunk = (rank - step - 1) % world_size
            sender = (rank + 1) % world_size
            # 模拟: rank 接收 sender 的 send_chunk，加到自己的 recv_chunk
            new_buffers[rank][recv_chunk * chunk_size:(recv_chunk + 1) * chunk_size] += \
                buffers[sender][recv_chunk * chunk_size:(recv_chunk + 1) * chunk_size]
        buffers = new_buffers
        print(f"    Step {step}: 数据在 ring 上流动")

    # Phase 2: AllGather (N-1 步)
    print(f"\n  Phase 2: AllGather ({world_size - 1} 步)")
    for step in range(world_size - 1):
        new_buffers = [buf.copy() for buf in buffers]
        for rank in range(world_size):
            send_chunk = (rank + 1 - step) % world_size
            recv_chunk = (rank - step) % world_size
            sender = (rank + 1) % world_size
            new_buffers[rank][recv_chunk * chunk_size:(recv_chunk + 1) * chunk_size] = \
                buffers[sender][recv_chunk * chunk_size:(recv_chunk + 1) * chunk_size]
        buffers = new_buffers
        print(f"    Step {step}: 聚合后的 chunk 在 ring 上传播")

    print(f"\n  Ring AllReduce 后:")
    for r, buf in enumerate(buffers):
        print(f"    Rank {r}: {buf.astype(int)}")

    return buffers


def demo_reduce_scatter():
    """ReduceScatter: 求和后每个 rank 只保留自己负责的分片

    Megatron-LM 中替代 AllReduce 的关键操作。
    AllReduce = ReduceScatter + AllGather
    """
    print("\n" + "=" * 60)
    print("ReduceScatter")
    print("=" * 60)

    world_size = 4
    np.random.seed(123)

    inputs = [np.random.randint(1, 10, size=8) for _ in range(world_size)]
    print("\n初始状态:")
    visualize_tensor("Before", inputs, "每个 rank 的完整数据")

    # ReduceScatter: 求和后按 chunk 分给各 rank
    total = sum(inputs)
    chunk_size = len(inputs[0]) // world_size
    outputs = []
    for rank in range(world_size):
        chunk = total[rank * chunk_size:(rank + 1) * chunk_size]
        outputs.append(chunk.copy())

    print(f"\n  计算: 先 AllReduce(SUM)，然后每个 rank 取自己的分片")
    print(f"  数学: output_rank_i = Σ(rank_j 的 chunk_i) for all j")
    print(f"  总和: {total}")
    print(f"\nReduceScatter 后 (每个 rank 只有一部分):")
    for r, out in enumerate(outputs):
        print(f"    Rank {r}: {out}")

    print(f"\n  用途: 在 Megatron-LM 的 RowParallelLinear 中，")
    print(f"  替代 AllReduce，让每个 rank 只持有后续计算需要的部分数据")


def demo_allgather():
    """AllGather: 收集所有 rank 的分片，拼接成完整数据

    在 ZeRO-3 中收集参数，在 SP 中收集序列。
    """
    print("\n" + "=" * 60)
    print("AllGather")
    print("=" * 60)

    world_size = 4
    np.random.seed(456)

    inputs = [np.array([rank * 10 + i for i in range(2)]) for rank in range(world_size)]
    print("\n初始状态 (每个 rank 有自己的分片):")
    for r, inp in enumerate(inputs):
        print(f"    Rank {r}: {inp} (分片 {r})")

    # AllGather: 收集所有分片
    output = np.concatenate(inputs)

    print(f"\n  计算: 收集所有 rank 的分片，拼接成完整数据")
    print(f"  数学: output = [chunk_0, chunk_1, ..., chunk_{world_size - 1}]")
    print(f"\nAllGather 后 (所有 rank 相同):")
    print(f"    All ranks: {output}")

    print(f"\n  用途:")
    print(f"  - ZeRO-3: 前向时 AllGather 收集参数分片")
    print(f"  - 序列并行: AllGather 收集切分后的序列")
    print(f"  - Megatron ColumnParallelLinear: gather_output=True 时")


def demo_broadcast():
    """Broadcast: rank 0 的数据广播给所有 rank

    用于参数初始化、配置同步等。
    """
    print("\n" + "=" * 60)
    print("Broadcast")
    print("=" * 60)

    world_size = 4
    np.random.seed(789)

    data = [np.zeros(4) for _ in range(world_size)]
    data[0] = np.array([1, 2, 3, 4])

    print("\n初始状态 (只有 rank 0 有数据):")
    for r, d in enumerate(data):
        print(f"    Rank {r}: {d}")

    # Broadcast from rank 0
    output = [data[0].copy() for _ in range(world_size)]

    print(f"\n  计算: 将 rank 0 的数据复制给所有 rank")
    print(f"Broadcast 后:")
    for r, out in enumerate(output):
        print(f"    Rank {r}: {out}")


def demo_alltoall():
    """AllToAll: 每个 rank 给每个其他 rank 发送不同的数据

    用于 MoE 的 expert 通信、SP/HP 转换。
    """
    print("\n" + "=" * 60)
    print("AllToAll")
    print("=" * 60)

    world_size = 4

    # 每个 rank 有给所有 rank 的数据块
    print("\n初始状态 (每个 rank 有 world_size 个 chunk):")
    inputs = []
    for rank in range(world_size):
        chunks = []
        for dst in range(world_size):
            chunks.append(np.array([rank * 10 + dst]))
        inputs.append(chunks)
        print(f"    Rank {rank}: {chunks}  → chunks 分别发给 rank 0,1,2,3")

    # AllToAll: rank i 的 chunk j 发给 rank j
    outputs = [[] for _ in range(world_size)]
    for src in range(world_size):
        for dst in range(world_size):
            outputs[dst].append(inputs[src][dst])

    print(f"\n  计算: rank_i 的 chunk_j 发送给 rank_j")
    print(f"AllToAll 后:")
    for r, out in enumerate(outputs):
        received = np.concatenate(out) if out else np.array([])
        print(f"    Rank {r}: 收到 {out}")

    print(f"\n  用途:")
    print(f"  - MoE: token 路由到不同 expert")
    print(f"  - Megatron all_to_all_sp2hp: [tokens/TP, H] → [tokens, H/TP]")


def demo_allreduce_vs_rs_ag():
    """对比 AllReduce 和 ReduceScatter + AllGather 的通信模式"""
    print("\n" + "=" * 60)
    print("AllReduce vs ReduceScatter + AllGather")
    print("=" * 60)

    world_size = 4
    data_size = 8
    np.random.seed(42)

    inputs = [np.random.randint(1, 10, size=data_size) for _ in range(world_size)]

    print(f"\n场景: {world_size} 个 GPU，每个有 {data_size} 个元素")
    print(f"数据类型: float32 (4 bytes)")

    # Method 1: AllReduce
    total_allreduce = sum(inputs)
    allreduce_data_moved = (world_size - 1) * data_size * 4 * 2  # Ring: (N-1)*size*2
    print(f"\n  方法 1: AllReduce (Ring)")
    print(f"    通信量: 2 × (N-1)/N × data_size × sizeof(dtype)")
    print(f"         = 2 × {world_size - 1}/{world_size} × {data_size} × 4 bytes")
    print(f"         = {allreduce_data_moved} bytes")
    print(f"    结果: 每个 rank 都有完整的 {data_size} 元素")

    # Method 2: ReduceScatter + AllGather
    chunk_size = data_size // world_size
    rs_data = (world_size - 1) * chunk_size * 4  # RS
    ag_data = (world_size - 1) * chunk_size * 4  # AG
    rs_ag_total = rs_data + ag_data
    print(f"\n  方法 2: ReduceScatter + AllGather")
    print(f"    ReduceScatter: (N-1)/N × data_size/N × sizeof(dtype)")
    print(f"                = {world_size - 1}/{world_size} × {chunk_size} × 4 = {rs_data} bytes")
    print(f"    AllGather:    (N-1)/N × data_size/N × sizeof(dtype)")
    print(f"                = {world_size - 1}/{world_size} × {chunk_size} × 4 = {ag_data} bytes")
    print(f"    总计: {rs_ag_total} bytes")
    print(f"    注意: 总通信量相同！但可以与计算重叠")

    print(f"\n  关键区别:")
    print(f"    AllReduce:        通信完成后，每个 rank 有完整数据 → 才能开始计算")
    print(f"    ReduceScatter:    通信完成后，每个 rank 只有分片 → 可以立即开始分片的计算")
    print(f"    AllGather:        计算完成后，收集所有分片 → 实现通信与计算重叠！")
    print(f"\n  这就是为什么 Megatron-LM 用 RS+AG 替代 AllReduce：")
    print(f"  ReduceScatter 完成后，不用等 AllGather 就能开始计算！")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       集合通信原语可视化 — AI Infra 学习工具            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_allreduce()
    demo_reduce_scatter()
    demo_allgather()
    demo_broadcast()
    demo_alltoall()
    demo_allreduce_vs_rs_ag()

    print("\n" + "=" * 60)
    print("总结: 分布式训练中的通信原语速查")
    print("=" * 60)
    print("""
    AllReduce:       全部求和 → 全部广播    → DDP梯度同步, TP聚合
    ReduceScatter:   全部求和 → 分片保留    → ZeRO梯度分片, TP优化
    AllGather:       收集分片 → 全部拼接    → ZeRO参数收集, SP序列收集
    Broadcast:       一个rank → 全部广播    → 参数初始化
    AllToAll:        互相交换 → 各取所需    → MoE路由, SP↔HP转换
    Send/Recv:       点对点通信              → 流水线并行
    """)


if __name__ == "__main__":
    main()
