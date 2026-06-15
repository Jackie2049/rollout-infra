
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

    print(f"\n{'='*60}")
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

    print(f"\n  Results:")
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
