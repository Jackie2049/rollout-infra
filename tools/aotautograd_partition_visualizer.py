#!/usr/bin/env python3
"""
AOTAutograd Min-Cut Partition 可视化工具

模拟AOTAutograd的min-cut partition算法:
1. 构建joint fwd+bwd DAG (简单示例)
2. 应用min-cut partition → 分离fwd/bwd节点
3. 计算checkpoint节点 → 自动gradient checkpointing
4. 可视化partition结果

基于: torch/_functorch/partitioners.py (min_cut algorithm)
      torch/_functorch/aot_autograd.py (AOTAutograd trace)

用法:
  python tools/aotautograd_partition_visualizer.py --model simple
  python tools/aotautograd_partition_visualizer.py --model transformer --num-layers 4
  python tools/aotautograd_partition_visualizer.py --model custom --nodes 8
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ============================================================
# Min-Cut Partition Algorithm 模拟
# ============================================================

class DAGNode:
    """FX IR Node 的简化表示"""
    def __init__(self, name, op, target, size_mb=0, is_fwd=True):
        self.name = name
        self.op = op  # placeholder/call_function/output
        self.target = target
        self.size_mb = size_mb  # 输出tensor大小(MB) — 内存成本
        self.is_fwd = is_fwd  # True=forward节点, False=backward节点
        self.inputs = []  # 依赖的节点名
        self.outputs = []  # 被哪些节点依赖
        self.partition = None  # "fwd" or "bwd" or "checkpoint"

    def __repr__(self):
        return f"{self.name}({self.op}:{self.target})"


def build_simple_model_dag():
    """构建简单线性模型 joint fwd+bwd DAG"""
    nodes = {}

    # Forward nodes
    nodes["x"] = DAGNode("x", "placeholder", "input", size_mb=4)
    nodes["w1"] = DAGNode("w1", "placeholder", "weight1", size_mb=8)
    nodes["w2"] = DAGNode("w2", "placeholder", "weight2", size_mb=8)

    nodes["linear1"] = DAGNode("linear1", "call_function", "torch.mm", size_mb=4, is_fwd=True)
    nodes["relu"] = DAGNode("relu", "call_function", "torch.relu", size_mb=4, is_fwd=True)
    nodes["linear2"] = DAGNode("linear2", "call_function", "torch.mm", size_mb=4, is_fwd=True)
    nodes["loss"] = DAGNode("loss", "call_function", "torch.sum", size_mb=0.1, is_fwd=True)

    # Backward nodes
    nodes["loss_bwd"] = DAGNode("loss_bwd", "call_function", "torch.sum.backward", size_mb=4, is_fwd=False)
    nodes["linear2_bwd"] = DAGNode("linear2_bwd", "call_function", "torch.mm.backward", size_mb=4, is_fwd=False)
    nodes["relu_bwd"] = DAGNode("relu_bwd", "call_function", "torch.relu.backward", size_mb=4, is_fwd=False)
    nodes["linear1_bwd"] = DAGNode("linear1_bwd", "call_function", "torch.mm.backward", size_mb=4, is_fwd=False)

    nodes["grad_w2"] = DAGNode("grad_w2", "output", "grad_weight2", size_mb=8, is_fwd=False)
    nodes["grad_w1"] = DAGNode("grad_w1", "output", "grad_weight1", size_mb=8, is_fwd=False)

    # Forward edges (data flow)
    edges_fwd = [
        ("x", "linear1"), ("w1", "linear1"),
        ("linear1", "relu"),
        ("relu", "linear2"), ("w2", "linear2"),
        ("linear2", "loss"),
    ]

    # Backward edges (gradient flow)
    edges_bwd = [
        ("loss", "loss_bwd"),
        ("loss_bwd", "linear2_bwd"), ("linear2", "linear2_bwd"), ("w2", "linear2_bwd"),
        ("linear2_bwd", "relu_bwd"), ("relu", "relu_bwd"),
        ("relu_bwd", "linear1_bwd"), ("linear1", "linear1_bwd"), ("w1", "linear1_bwd"),
        ("linear2_bwd", "grad_w2"),
        ("linear1_bwd", "grad_w1"),
    ]

    # Set inputs/outputs
    for src, dst in edges_fwd + edges_bwd:
        nodes[src].outputs.append(dst)
        nodes[dst].inputs.append(src)

    return nodes


def build_transformer_dag(num_layers):
    """构建Transformer模型 joint fwd+bwd DAG"""
    nodes = {}

    # Input
    nodes["x"] = DAGNode("x", "placeholder", "input", size_mb=8)

    # Forward: each layer = linear + relu + residual
    for i in range(num_layers):
        nodes[f"w{i}"] = DAGNode(f"w{i}", "placeholder", f"weight{i}", size_mb=16)
        nodes[f"linear{i}"] = DAGNode(f"linear{i}", "call_function", "torch.mm", size_mb=8, is_fwd=True)
        nodes[f"relu{i}"] = DAGNode(f"relu{i}", "call_function", "torch.relu", size_mb=8, is_fwd=True)

    nodes["loss"] = DAGNode("loss", "call_function", "torch.sum", size_mb=0.1, is_fwd=True)

    # Backward
    nodes["loss_bwd"] = DAGNode("loss_bwd", "call_function", "torch.sum.backward", size_mb=8, is_fwd=False)
    for i in range(num_layers):
        nodes[f"linear{i}_bwd"] = DAGNode(f"linear{i}_bwd", "call_function", "torch.mm.backward", size_mb=8, is_fwd=False)
        nodes[f"relu{i}_bwd"] = DAGNode(f"relu{i}_bwd", "call_function", "torch.relu.backward", size_mb=8, is_fwd=False)
        nodes[f"grad_w{i}"] = DAGNode(f"grad_w{i}", "output", f"grad_weight{i}", size_mb=16, is_fwd=False)

    # Forward edges
    nodes["x"].outputs.append("linear0")
    nodes["linear0"].inputs.append("x")
    nodes["w0"].outputs.append("linear0")
    nodes["linear0"].inputs.append("w0")

    for i in range(num_layers):
        nodes[f"w{i}"].outputs.append(f"linear{i}")
        nodes[f"linear{i}"].inputs.append(f"w{i}")

        prev_relu = "relu" + str(i-1) if i > 0 else "x"
        nodes[prev_relu].outputs.append(f"linear{i}")
        nodes[f"linear{i}"].inputs.append(prev_relu)

        nodes[f"linear{i}"].outputs.append(f"relu{i}")
        nodes[f"relu{i}"].inputs.append(f"linear{i}")

        if i < num_layers - 1:
            nodes[f"relu{i}"].outputs.append(f"linear{i+1}")

    last_relu = f"relu{num_layers-1}"
    nodes[last_relu].outputs.append("loss")
    nodes["loss"].inputs.append(last_relu)

    # Backward edges
    nodes["loss"].outputs.append("loss_bwd")
    nodes["loss_bwd"].inputs.append("loss")
    nodes["loss_bwd"].outputs.append(f"linear{num_layers-1}_bwd")
    nodes[f"linear{num_layers-1}_bwd"].inputs.append("loss_bwd")

    for i in range(num_layers-1, -1, -1):
        nodes[f"linear{i}"].outputs.append(f"linear{i}_bwd")
        nodes[f"linear{i}_bwd"].inputs.append(f"linear{i}")
        nodes[f"w{i}"].outputs.append(f"linear{i}_bwd")
        nodes[f"linear{i}_bwd"].inputs.append(f"w{i}")

        nodes[f"relu{i}"].outputs.append(f"relu{i}_bwd")
        nodes[f"relu{i}_bwd"].inputs.append(f"relu{i}")
        nodes[f"linear{i}_bwd"].outputs.append(f"relu{i}_bwd")

        if i > 0:
            nodes[f"relu{i}_bwd"].outputs.append(f"linear{i-1}_bwd")
            nodes[f"linear{i-1}_bwd"].inputs.append(f"relu{i}_bwd")

        nodes[f"linear{i}_bwd"].outputs.append(f"grad_w{i}")
        nodes[f"grad_w{i}"].inputs.append(f"linear{i}_bwd")

    return nodes


def min_cut_partition(nodes, strategy="min_cut"):
    """
    模拟AOTAutograd min-cut partition算法:
    1. fwd节点放在forward partition
    2. bwd节点放在backward partition
    3. fwd→bwd的边 = "cut" → 这些fwd节点的输出需要保存 → checkpoint!
    4. 最小化checkpoint总大小 → min-cut

    Real algorithm: Uses network flow / max-flow min-cut theorem
    Simplified here: greedily minimize saved tensor sizes
    """
    # Phase 1: Assign fwd nodes to forward partition, bwd to backward
    for name, node in nodes.items():
        if node.is_fwd:
            node.partition = "fwd"
        else:
            node.partition = "bwd"

    # Phase 2: Find cut edges (fwd→bwd) → these need checkpointing
    checkpoints = []
    total_checkpoint_size = 0

    for name, node in nodes.items():
        if node.partition == "fwd":
            for output_name in node.outputs:
                if output_name in nodes and nodes[output_name].partition == "bwd":
                    # This is a cut edge → node needs to be checkpointed
                    if node.op != "placeholder":  # Don't checkpoint inputs (they're already saved)
                        node.partition = "checkpoint"
                        checkpoints.append(name)
                        total_checkpoint_size += node.size_mb

    # Phase 3: Gradient checkpointing optimization
    # Instead of saving ALL intermediate activations, only save subset
    # and recompute others during backward
    if strategy == "gradient_checkpointing":
        # Select checkpoints to minimize memory while keeping recomputation bounded
        # Simple strategy: checkpoint every sqrt(n) layer (standard approach)
        fwd_compute_nodes = [name for name, node in nodes.items()
                           if node.is_fwd and node.op == "call_function"]
        n = len(fwd_compute_nodes)
        checkpoint_interval = max(1, int(n**0.5))

        optimized_checkpoints = []
        optimized_size = 0
        for i, name in enumerate(fwd_compute_nodes):
            if i % checkpoint_interval == 0:
                nodes[name].partition = "checkpoint"
                optimized_checkpoints.append(name)
                optimized_size += nodes[name].size_mb
            else:
                # This node will be recomputed in backward → no checkpoint needed
                nodes[name].partition = "fwd_recompute"

        return {
            "strategy": "gradient_checkpointing",
            "full_checkpoint_size": total_checkpoint_size,
            "optimized_checkpoint_size": optimized_size,
            "full_checkpoints": checkpoints,
            "optimized_checkpoints": optimized_checkpoints,
            "savings_ratio": (1 - optimized_size / total_checkpoint_size) if total_checkpoint_size > 0 else 0,
            "checkpoint_interval": checkpoint_interval,
        }

    return {
        "strategy": "min_cut",
        "checkpoint_size": total_checkpoint_size,
        "checkpoints": checkpoints,
    }


def visualize_partition(nodes, result):
    """可视化partition结果"""
    print(f"\n{'='*60}")
    print(f"  AOTAutograd Min-Cut Partition 可视化")
    print(f"{'='*60}")

    # Print nodes by partition
    fwd_nodes = [n for n in nodes.values() if n.partition == "fwd"]
    bwd_nodes = [n for n in nodes.values() if n.partition == "bwd"]
    checkpoint_nodes = [n for n in nodes.values() if n.partition == "checkpoint"]
    recompute_nodes = [n for n in nodes.values() if n.partition == "fwd_recompute"]

    print(f"\n  ★ Forward Partition ({len(fwd_nodes)} nodes):")
    for node in fwd_nodes:
        print(f"    {node.name}: {node.op}:{node.target} ({node.size_mb}MB)")

    print(f"\n  ★ Checkpoint Nodes ({len(checkpoint_nodes)} nodes):")
    for node in checkpoint_nodes:
        print(f"    {node.name}: {node.op}:{node.target} ({node.size_mb}MB) → saved for backward")

    if recompute_nodes:
        print(f"\n  ★ Recompute Nodes ({len(recompute_nodes)} nodes):")
        for node in recompute_nodes:
            print(f"    {node.name}: {node.op}:{node.target} ({node.size_mb}MB) → recomputed in backward")

    print(f"\n  ★ Backward Partition ({len(bwd_nodes)} nodes):")
    for node in bwd_nodes:
        print(f"    {node.name}: {node.op}:{node.target} ({node.size_mb}MB)")

    # Print cut edges
    print(f"\n  ★ Cut Edges (fwd→bwd / checkpoint→bwd):")
    for name, node in nodes.items():
        for output_name in node.outputs:
            if output_name in nodes and nodes[output_name].partition == "bwd":
                if node.partition in ("checkpoint", "fwd_recompute"):
                    print(f"    {node.name} → {output_name} [SAVED/RECOMPUTED]")
                elif node.partition == "fwd":
                    print(f"    {node.name} → {output_name} [CUT - needs checkpoint]")

    # Print memory analysis
    if "checkpoint_size" in result:
        print(f"\n  ★ Memory Analysis:")
        print(f"    Total checkpoint size: {result['checkpoint_size']}MB")
        print(f"    Checkpoint nodes: {result['checkpoints']}")

    if "optimized_checkpoint_size" in result:
        print(f"\n  ★ Gradient Checkpointing Optimization:")
        print(f"    Full checkpoint: {result['full_checkpoint_size']}MB")
        print(f"    Optimized checkpoint: {result['optimized_checkpoint_size']}MB")
        print(f"    Savings: {result['savings_ratio']*100:.1f}%")
        print(f"    Checkpoint interval: every {result['checkpoint_interval']} compute nodes")
        print(f"    Full checkpoints: {result['full_checkpoints']}")
        print(f"    Optimized checkpoints: {result['optimized_checkpoints']}")

    # Print execution flow
    print(f"\n  ★ Execution Flow (AOTAutograd):")
    print(f"    1. Forward: execute fwd nodes → save checkpoint outputs")
    print(f"    2. Backward: load checkpoints → recompute non-checkpoint fwd nodes → compute bwd nodes")
    print(f"    3. ★ This is how torch.compile enables automatic gradient checkpointing!")


def explain_aotautograd():
    """解释AOTAutograd核心机制"""
    print(f"\n{'='*60}")
    print(f"  AOTAutograd 核心机制解释")
    print(f"{'='*60}")

    print("""
  ★ ★ ★ AOTAutograd = Ahead-Of-Time Autograd → torch.compile的核心组件!

  1. Joint Graph Tracing:
     → make_fx(fn) → trace forward + backward simultaneously
     → 得到 joint fwd+bwd FX Graph → 包含所有计算节点
     → 不是分开trace → 而是joint → 保证正确性!

  2. Min-Cut Partition:
     → 将joint graph分成 fwd_partition 和 bwd_partition
     → cut edges = 需要保存的中间值 → checkpoint!
     → 最小化保存的值 → 最小化内存 → 这是自动gradient checkpointing的原理!
     → ★ 算法: max-flow min-cut theorem → 网络流问题 → 最优解!

  3. Functionalization:
     → in-place ops (torch.add_, torch.relu_) → 替换为 out-of-place (torch.add)
     → mutation removal → FX graph是pure functional → 无side effects
     → ★ 这是compile兼容的前提 → mutation → functional → 确定性执行!

  4. CompiledFunction/CompiledBackward:
     → fwd_partition → CompiledFunction (编译后的forward函数)
     → bwd_partition → CompiledBackward (编译后的backward函数)
     → ★ fwd+bwd分开编译 → 可以分别用不同compile策略!

  5. vs PyTorch eager autograd:
     → eager: forward → autograd引擎 → backward → 逐op动态生成backward graph
     → AOTAutograd: forward → pre-computed backward → 执行预先编译的backward
     → ★ ★ AOT = Ahead-Of-Time → backward在compile时就确定了 → 不需要运行时生成!

  6. FSDP2 + AOTAutograd:
     → FSDP2 per-param DTensor → AOTAutograd trace → joint graph包含DTensor ops
     → FSDP2 intentional graph breaks → AOTAutograd只在compute段运行 → hooks在eager
     → ★ FSDP2 + compile = hooks(eager) + compute(AOT+Inductor) → 正确+快!

  7. ZeRO-3 + AOTAutograd:
     → ZeRO-3 dynamic AllGather → every layer calls allgather → graph breaks everywhere
     → ZeRO-3 + compile → constant graph breaks → AOTAutograd无法trace完整joint graph
     → ★ ★ ZeRO-3 + compile = 不兼容 → 这是根本原因 → vs FSDP2兼容!
  """)


def main():
    parser = argparse.ArgumentParser(
        description="AOTAutograd Min-Cut Partition 可视化工具"
    )
    parser.add_argument("--model", choices=["simple", "transformer", "explain"],
                        default="simple")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--strategy", choices=["min_cut", "gradient_checkpointing"],
                        default="gradient_checkpointing")

    args = parser.parse_args()

    if args.model == "explain":
        explain_aotautograd()
        return

    if args.model == "simple":
        nodes = build_simple_model_dag()
    elif args.model == "transformer":
        nodes = build_transformer_dag(args.num_layers)

    result = min_cut_partition(nodes, strategy=args.strategy)
    visualize_partition(nodes, result)

    # Save result
    Path("results").mkdir(exist_ok=True)
    with open(f"results/aotautograd_partition_{args.model}_{args.strategy}.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
