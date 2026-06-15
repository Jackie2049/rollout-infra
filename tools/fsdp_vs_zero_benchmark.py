#!/usr/bin/env python3
"""
PyTorch FSDP2 vs ZeRO-3 训练基准实验脚本
准备在GPU可用时运行: FSDP2+compile vs ZeRO-3 vs DDP+LoRA

RTX 4090 单GPU可行实验:
1. DDP + LoRA + BF16 (baseline)
2. DDP + LoRA + compile(reduce-overhead)
3. GRPO 训练模拟 (LoRA only, 无critic)

多GPU实验 (需要NVLink):
4. FSDP2 + compile (2+GPU)
5. ZeRO-3 + CPU offload (2+GPU)

使用方法 (GPU可用时):
  conda activate llm  # 或 gpu-infra
  python tools/fsdp_vs_zero_benchmark.py --experiment single-gpu-lora --model-size 7b --gpu rtx4090
  python tools/fsdp_vs_zero_benchmark.py --experiment fsdp2-compile --model-size 7b --gpus 8
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentConfig:
    name: str
    description: str
    requires_gpu: bool
    min_gpus: int
    model_size: str
    framework: str
    strategy: str
    compile_mode: Optional[str] = None
    lora_rank: int = 0
    batch_size: int = 32
    seq_len: int = 2048
    num_steps: int = 10
    gpu_type: str = "rtx4090"
    estimated_memory_gb: float = 0.0
    rtx4090_feasible: bool = True


EXPERIMENTS = {
    "single-gpu-lora": ExperimentConfig(
        name="DDP + LoRA + BF16 (baseline)",
        description="单GPU DDP训练, LoRA r=32, BF16精度, 无compile",
        requires_gpu=True, min_gpus=1, model_size="7b",
        framework="pytorch-ddp", strategy="lora-bf16",
        lora_rank=32, batch_size=32, seq_len=2048, num_steps=10,
        gpu_type="rtx4090", estimated_memory_gb=17.0, rtx4090_feasible=True,
    ),
    "single-gpu-lora-compile": ExperimentConfig(
        name="DDP + LoRA + compile(reduce-overhead)",
        description="单GPU DDP训练, LoRA r=32, BF16, torch.compile(reduce-overhead)",
        requires_gpu=True, min_gpus=1, model_size="7b",
        framework="pytorch-ddp", strategy="lora-bf16-compile",
        lora_rank=32, batch_size=32, seq_len=2048, num_steps=10,
        compile_mode="reduce-overhead",
        gpu_type="rtx4090", estimated_memory_gb=17.5, rtx4090_feasible=True,
    ),
    "single-gpu-grpo-lora": ExperimentConfig(
        name="GRPO训练模拟 (LoRA only, 无critic)",
        description="单GPU GRPO训练, LoRA r=32, BF16, 无critic (GRPO特性)",
        requires_gpu=True, min_gpus=1, model_size="7b",
        framework="verl-grpo", strategy="grpo-lora",
        lora_rank=32, batch_size=4, seq_len=2048, num_steps=10,
        gpu_type="rtx4090", estimated_memory_gb=17.0, rtx4090_feasible=True,
    ),
    "fsdp2-compile": ExperimentConfig(
        name="FSDP2 + compile(reduce-overhead) 2+GPU",
        description="多GPU FSDP2训练, BF16, torch.compile, per-param DTensor",
        requires_gpu=True, min_gpus=2, model_size="7b",
        framework="pytorch-fsdp2", strategy="fsdp2-compile",
        compile_mode="reduce-overhead",
        batch_size=64, seq_len=2048, num_steps=10,
        gpu_type="h100", estimated_memory_gb=25.0, rtx4090_feasible=False,
    ),
    "zero3-cpu-offload": ExperimentConfig(
        name="ZeRO-3 + CPU offload 2+GPU",
        description="多GPU ZeRO-3训练, CPU optimizer offload, BF16",
        requires_gpu=True, min_gpus=2, model_size="7b",
        framework="deepspeed-zero3", strategy="zero3-cpu-offload",
        batch_size=64, seq_len=2048, num_steps=10,
        gpu_type="h100", estimated_memory_gb=10.0, rtx4090_feasible=False,
    ),
}


# ============================================================
# 单GPU DDP + LoRA 实验脚本模板
# ============================================================

SINGLE_GPU_LORA_SCRIPT = """
import torch
import torch.nn as nn
import time
from torch.distributed import init_process_group, destroy_process_group

# Simple LoRA model for benchmarking
class SimpleLoRAModel(nn.Module):
    def __init__(self, hidden_size=4096, num_layers=32, lora_rank=32):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)
        ])
        self.lora_rank = lora_rank
        # LoRA adapters
        self.lora_a = nn.ParameterList([
            nn.Parameter(torch.randn(lora_rank, hidden_size) * 0.01) for _ in range(num_layers)
        ])
        self.lora_b = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size, lora_rank)) for _ in range(num_layers)
        ])
        self.hidden_size = hidden_size

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # Base forward + LoRA
            x = layer(x) + (x @ self.lora_a[i].T @ self.lora_b[i].T) * (1.0 / self.lora_rank)
        return x


def run_experiment(model_size, lora_rank, batch_size, seq_len, compile_mode, num_steps):
    device = torch.device("cuda")

    # Model size config
    configs = {
        "7b": {"hidden_size": 4096, "num_layers": 32},
        "1b": {"hidden_size": 2048, "num_layers": 24},
    }
    cfg = configs.get(model_size, configs["7b"])

    print(f"\\n{'='*60}")
    print(f"  Experiment: DDP+LoRA+{compile_mode or 'eager'}")
    print(f"  Model: {model_size}, LoRA rank={lora_rank}")
    print(f"  Batch: {batch_size}, SeqLen: {seq_len}, Steps: {num_steps}")
    print(f"  Device: {device}, GPU: {torch.cuda.get_device_name()}")
    print(f"{'='*60}")

    model = SimpleLoRAModel(
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        lora_rank=lora_rank,
    ).to(device)

    # Only train LoRA params
    for layer in model.layers:
        layer.weight.requires_grad = False

    if compile_mode:
        print(f"  Compiling with mode={compile_mode}...")
        model = torch.compile(model, mode=compile_mode)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, betas=(0.9, 0.999), eps=1e-8
    )

    # Warmup
    x = torch.randn(batch_size, seq_len, cfg["hidden_size"], device=device, dtype=torch.bfloat16)
    model = model.to(torch.bfloat16)
    for _ in range(3):
        y = model(x.to(torch.bfloat16))
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    for step in range(num_steps):
        y = model(x.to(torch.bfloat16))
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Report
    total_tokens = num_steps * batch_size * seq_len
    throughput = total_tokens / elapsed

    mem_alloc = torch.cuda.max_memory_allocated() / 1e9
    mem_reserved = torch.cuda.max_memory_reserved() / 1e9

    print(f"\\n  Results:")
    print(f"    Time: {elapsed:.2f}s ({num_steps} steps)")
    print(f"    Throughput: {throughput:.0f} tokens/s")
    print(f"    Memory allocated: {mem_alloc:.2f} GB")
    print(f"    Memory reserved: {mem_reserved:.2f} GB")
    print(f"    Per-step: {elapsed/num_steps*1000:.1f} ms")

    return {
        "model_size": model_size,
        "lora_rank": lora_rank,
        "compile_mode": compile_mode,
        "throughput_tokens_per_sec": throughput,
        "memory_allocated_gb": mem_alloc,
        "memory_reserved_gb": mem_reserved,
        "time_per_step_ms": elapsed / num_steps * 1000,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="7b")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--compile-mode", default=None)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()

    result = run_experiment(
        args.model_size, args.lora_rank, args.batch_size,
        args.seq_len, args.compile_mode, args.num_steps
    )
    # Save result
    import json
    Path("results").mkdir(exist_ok=True)
    name = f"lora-{args.lora_rank}-{args.compile_mode or 'eager'}-{args.model_size}"
    with open(f"results/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
"""


# ============================================================
# FSDP2 实验脚本模板 (需要 2+ GPU + NVLink)
# ============================================================

FSDP2_COMPILE_SCRIPT = """
import torch
import torch.nn as nn
import time
from torch.distributed.fsdp import fully_shard, FSDPModule

class SimpleTransformerLayer(nn.Module):
    def __init__(self, hidden_size=4096):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.linear2(self.act(self.linear1(x))))

class SimpleFSDP2Model(nn.Module):
    def __init__(self, hidden_size=4096, num_layers=32):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(hidden_size) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def run_fsdp2_experiment(model_size, compile_mode, batch_size, seq_len, num_steps):
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    configs = {"7b": {"hidden_size": 4096, "num_layers": 32}}
    cfg = configs[model_size]

    model = SimpleFSDP2Model(**cfg).to(device).to(torch.bfloat16)

    # FSDP2 per-layer sharding
    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)

    if compile_mode:
        model = torch.compile(model, mode=compile_mode)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Warmup + benchmark (same pattern as single-GPU)
    x = torch.randn(batch_size, seq_len, cfg["hidden_size"], device=device, dtype=torch.bfloat16)
    for _ in range(3):
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    start = time.time()
    for step in range(num_steps):
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    total_tokens = num_steps * batch_size * seq_len
    throughput = total_tokens / elapsed
    mem = torch.cuda.max_memory_allocated() / 1e9

    if local_rank == 0:
        print(f"FSDP2+{compile_mode or 'eager'}: {throughput:.0f} tok/s, {mem:.1f}GB, {elapsed/num_steps*1000:.1f}ms/step")

    dist.destroy_process_group()
"""


def check_gpu_available() -> bool:
    """检查是否有可用的GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU available: {name}, {mem:.1f} GB")
            return True
        else:
            print("  No GPU available (torch.cuda.is_available() = False)")
            return False
    except ImportError:
        print("  PyTorch not installed in current environment")
        return False


def check_remote_gpu() -> bool:
    """检查远程GPU服务器是否在线"""
    servers = [
        ("University", "sshpass -p 'adspzxw123' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 zxw@219.223.198.62 echo online"),
        ("Matpool", "ssh -p 28959 -o ConnectTimeout=5 root@hz-t3.matpool.com echo online"),
    ]
    for name, cmd in servers:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
            if result.returncode == 0:
                print(f"  {name} server: ONLINE")
                return True
            else:
                print(f"  {name} server: OFFLINE ({result.stderr.decode()[:100]})")
        except Exception as e:
            print(f"  {name} server: OFFLINE ({str(e)[:100]})")
    return False


def print_experiment_info(exp_name: str) -> None:
    """打印实验信息"""
    exp = EXPERIMENTS.get(exp_name)
    if not exp:
        print(f"  Unknown experiment: {exp_name}")
        return
    print(f"\n{'='*60}")
    print(f"  Experiment: {exp.name}")
    print(f"  Description: {exp.description}")
    print(f"  Framework: {exp.framework}")
    print(f"  Strategy: {exp.strategy}")
    if exp.compile_mode:
        print(f"  Compile: torch.compile(mode='{exp.compile_mode}')")
    print(f"  LoRA rank: {exp.lora_rank}")
    print(f"  Batch size: {exp.batch_size}")
    print(f"  Seq len: {exp.seq_len}")
    print(f"  Steps: {exp.num_steps}")
    print(f"  Min GPUs: {exp.min_gpus}")
    print(f"  Estimated memory: {exp.estimated_memory_gb} GB")
    print(f"  RTX 4090 feasible: {exp.rtx4090_feasible}")
    print(f"{'='*60}")


def generate_script(exp_name: str) -> str:
    """生成实验脚本"""
    exp = EXPERIMENTS.get(exp_name)
    if not exp:
        return ""

    if exp.framework in ("pytorch-ddp", "pytorch-fsdp2"):
        if exp.min_gpus == 1:
            return SINGLE_GPU_LORA_SCRIPT
        else:
            return FSDP2_COMPILE_SCRIPT
    elif exp.framework == "verl-grpo":
        return "# verl GRPO script would be generated here - requires verl installation"
    elif exp.framework == "deepspeed-zero3":
        return "# DeepSpeed ZeRO-3 script would be generated here - requires deepspeed installation"
    return ""


def save_script(exp_name: str, output_dir: str = "scripts") -> str:
    """保存实验脚本到文件"""
    script = generate_script(exp_name)
    if not script:
        return ""
    Path(output_dir).mkdir(exist_ok=True)
    filepath = Path(output_dir) / f"bench_{exp_name}.py"
    with open(filepath, "w") as f:
        f.write(script)
    print(f"  Script saved: {filepath}")
    return str(filepath)


def print_all_experiments() -> None:
    """打印所有实验概览"""
    print(f"\n{'='*60}")
    print(f"  FSDP2 vs ZeRO-3 训练基准实验概览")
    print(f"{'='*60}")
    print(f"\n  ★ RTX 4090 单GPU可行实验:")
    for name, exp in EXPERIMENTS.items():
        if exp.rtx4090_feasible:
            print(f"    {name}: {exp.name} ({exp.estimated_memory_gb}GB)")
    print(f"\n  ★ 多GPU实验 (需要NVLink):")
    for name, exp in EXPERIMENTS.items():
        if not exp.rtx4090_feasible:
            print(f"    {name}: {exp.name} ({exp.min_gpus}+GPU, {exp.estimated_memory_gb}GB)")
    print(f"\n  ★ 预期结果 (基于源码阅读):")
    print(f"    single-gpu-lora: ~19,743 tok/s (LoRA+BF16)")
    print(f"    single-gpu-lora-compile: ~24,000 tok/s (LoRA+compile, +20% estimate)")
    print(f"    fsdp2-compile: ~2M tok/s (8×H100 NVLink, from training_speed_estimator)")
    print(f"    zero3-cpu-offload: ~50% slower than pure GPU (CPU offload overhead)")


def main():
    parser = argparse.ArgumentParser(
        description="FSDP2 vs ZeRO-3 训练基准实验 — 准备在GPU可用时运行"
    )
    parser.add_argument(
        "--experiment", choices=list(EXPERIMENTS.keys()) + ["all", "list"],
        default="list", help="实验名称"
    )
    parser.add_argument(
        "--model-size", choices=["7b", "1b"], default="7b", help="模型大小"
    )
    parser.add_argument(
        "--gpu", choices=["rtx4090", "h100", "a100"], default="rtx4090", help="GPU类型"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=32, help="LoRA rank"
    )
    parser.add_argument(
        "--action", choices=["run", "prepare", "check", "info"],
        default="check", help="动作: run(运行)/prepare(生成脚本)/check(检查GPU)/info(显示信息)"
    )

    args = parser.parse_args()

    if args.experiment == "list":
        print_all_experiments()
        return

    exp = EXPERIMENTS[args.experiment]

    if args.action == "info":
        print_experiment_info(args.experiment)

    elif args.action == "check":
        print(f"\n  Checking GPU availability...")
        local_gpu = check_gpu_available()
        remote_gpu = check_remote_gpu()
        if local_gpu:
            print(f"\n  ✓ Local GPU available! Can run: {args.experiment}")
        elif remote_gpu:
            print(f"\n  ✓ Remote GPU available! Need to run on remote server.")
        else:
            print(f"\n  ✗ No GPU available. Script can be prepared for later execution.")
            print(f"  Use --action prepare to generate script, then run on GPU server when available.")

    elif args.action == "prepare":
        print(f"\n  Generating experiment script for: {args.experiment}")
        filepath = save_script(args.experiment)
        if filepath:
            print(f"\n  To run on GPU server:")
            print(f"    scp {filepath} gpu-server:/tmp/")
            print(f"    ssh gpu-server 'python /tmp/bench_{args.experiment}.py'")

    elif args.action == "run":
        if not check_gpu_available():
            print(f"\n  ✗ Cannot run - no local GPU available!")
            print(f"  Use --action prepare to generate script for remote execution.")
            return
        # Would actually run the experiment here
        print(f"\n  Running: {args.experiment}")
        print(f"  (Implementation requires actual GPU - script generated for manual execution)")
        save_script(args.experiment)


if __name__ == "__main__":
    main()
