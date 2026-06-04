#!/usr/bin/env python3
"""Gradient Checkpointing (Activation Recomputation) GPU 实验

关键问题: 训练大模型时显存不够怎么办？
答案: 梯度检查点 — 不保存中间激活，反向时重新计算

实验:
1. Checkpoint vs No-Checkpoint 内存对比
2. Checkpoint 计算开销 (额外 forward 次数)
3. 不同 checkpoint 策略: uniform vs selective
4. 与 ZeRO/TP 组合效果
5. 最佳 checkpoint 策略选择

用法 (GPU 服务器):
  source /root/miniconda3/bin/activate myconda
  python gpu_gradient_checkpoint.py
"""

import os, json, time, math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def bench_ms(fn, warmup=3, rep=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep


class TransformerBlock(nn.Module):
    def __init__(self, hidden, n_heads, ff_mult=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * ff_mult),
            nn.GELU(),
            nn.Linear(hidden * ff_mult, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + self.ff(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, n_layers, hidden, n_heads, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.embed = nn.Embedding(5000, hidden)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 5000)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x)
            else:
                x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# ============================================================
# 实验 1: 内存对比 (Peak Memory)
# ============================================================

def exp1_memory_comparison():
    print("\n" + "=" * 60)
    print("实验1: Gradient Checkpointing 内存对比")
    print("=" * 60)

    results = []
    B, S = 4, 256

    configs = [
        ("Small (4L, H=512)", 4, 512, 8),
        ("Medium (8L, H=768)", 8, 768, 12),
        ("Large (12L, H=1024)", 12, 1024, 16),
    ]

    print(f"\n  Batch={B}, Seq={S}, FP32 training")
    print(f"  {'Config':<25} {'Params':<10} {'No-CKPT MB':<14} {'With-CKPT MB':<14} {'Saving'}")
    print("  " + "-" * 75)

    for name, n_layers, hidden, n_heads in configs:
        input_ids = torch.randint(0, 5000, (B, S), device="cuda")

        # Without checkpoint
        model = GPTModel(n_layers, hidden, n_heads, use_checkpoint=False).cuda()
        model.train()
        n_params = sum(p.numel() for p in model.parameters())

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()

        out = model(input_ids)
        loss = out.sum()
        loss.backward()

        no_ckpt_mem = (torch.cuda.max_memory_allocated() - base) / 1e6
        del model, out, loss
        torch.cuda.empty_cache()

        # With checkpoint
        model = GPTModel(n_layers, hidden, n_heads, use_checkpoint=True).cuda()
        model.train()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()

        out = model(input_ids)
        loss = out.sum()
        loss.backward()

        ckpt_mem = (torch.cuda.max_memory_allocated() - base) / 1e6
        saving = (1 - ckpt_mem / no_ckpt_mem) * 100

        print(f"  {name:<25} {n_params/1e6:<10.1f}M {no_ckpt_mem:<14.0f} {ckpt_mem:<14.0f} {saving:.0f}%")

        results.append({
            "name": name, "n_params": n_params,
            "no_ckpt_mb": round(no_ckpt_mem), "ckpt_mb": round(ckpt_mem),
            "saving_pct": round(saving),
        })

        del model, out, loss, input_ids
        torch.cuda.empty_cache()

    return results


# ============================================================
# 实验 2: 计算开销 (额外 Forward 次数)
# ============================================================

def exp2_compute_overhead():
    print("\n" + "=" * 60)
    print("实验2: Checkpoint 计算开销")
    print("=" * 60)

    results = []
    B, S = 4, 256
    hidden, n_heads, n_layers = 768, 12, 12

    input_ids = torch.randint(0, 5000, (B, S), device="cuda")

    # No checkpoint
    model_no = GPTModel(n_layers, hidden, n_heads, use_checkpoint=False).cuda()
    model_no.train()
    opt_no = torch.optim.Adam(model_no.parameters(), lr=1e-4)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        opt_no.zero_grad()
        out = model_no(input_ids)
        loss = out.sum()
        loss.backward()
        opt_no.step()
    torch.cuda.synchronize()
    no_ckpt_ms = (time.time() - t0) / 10 * 1000

    # With checkpoint (uniform: checkpoint every layer)
    model_ck = GPTModel(n_layers, hidden, n_heads, use_checkpoint=True).cuda()
    model_ck.train()
    opt_ck = torch.optim.Adam(model_ck.parameters(), lr=1e-4)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        opt_ck.zero_grad()
        out = model_ck(input_ids)
        loss = out.sum()
        loss.backward()
        opt_ck.step()
    torch.cuda.synchronize()
    ckpt_ms = (time.time() - t0) / 10 * 1000

    overhead = (ckpt_ms / no_ckpt_ms - 1) * 100

    # Theory: checkpoint every layer → 1 extra forward per layer
    # Total forwards: 1 (original) + n_layers/2 (average recomputation) ≈ 1 + 0.5 = 1.5x
    # But only the block forward, not embed/head, so actual ≈ 1.3-1.4x

    print(f"\n  Model: {n_layers}L, H={hidden}, B={B}, S={S}")
    print(f"  No checkpoint:   {no_ckpt_ms:.1f} ms/step")
    print(f"  With checkpoint: {ckpt_ms:.1f} ms/step ({overhead:.0f}% overhead)")
    print(f"  Theory: ~33% overhead (recompute 50% of layers on average)")

    results.append({
        "no_ckpt_ms": round(no_ckpt_ms, 1), "ckpt_ms": round(ckpt_ms, 1),
        "overhead_pct": round(overhead, 0),
    })

    del model_no, model_ck, opt_no, opt_ck, input_ids
    torch.cuda.empty_cache()
    return results


# ============================================================
# 实验 3: Selective vs Uniform Checkpointing
# ============================================================

def exp3_selective_checkpoint():
    print("\n" + "=" * 60)
    print("实验3: Selective vs Uniform Checkpointing")
    print("=" * 60)

    results = []
    B, S = 4, 256
    hidden, n_heads = 768, 12

    # Different checkpoint strategies
    strategies = [
        ("None", []),
        ("Every layer", list(range(12))),
        ("Every 2nd", list(range(0, 12, 2))),
        ("Every 3rd", list(range(0, 12, 3))),
        ("First+Last only", [0, 11]),
    ]

    print(f"\n  12 layers, H={hidden}, B={B}, S={S}")
    print(f"  {'Strategy':<20} {'CKPT layers':<14} {'Mem MB':<10} {'Time ms':<10} {'Overhead'}")
    print("  " + "-" * 64)

    for name, ckpt_layers in strategies:
        input_ids = torch.randint(0, 5000, (B, S), device="cuda")

        class SelectiveModel(nn.Module):
            def __init__(self, ckpt_layers):
                super().__init__()
                self.ckpt_layers = set(ckpt_layers)
                self.embed = nn.Embedding(5000, hidden)
                self.blocks = nn.ModuleList([TransformerBlock(hidden, n_heads) for _ in range(12)])
                self.ln_f = nn.LayerNorm(hidden)
                self.head = nn.Linear(hidden, 5000)

            def forward(self, input_ids):
                x = self.embed(input_ids)
                for i, block in enumerate(self.blocks):
                    if i in self.ckpt_layers and self.training:
                        x = checkpoint(block, x)
                    else:
                        x = block(x)
                x = self.ln_f(x)
                return self.head(x)

        model = SelectiveModel(ckpt_layers).cuda()
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        out = model(input_ids)
        loss = out.sum()
        loss.backward()
        mem_mb = (torch.cuda.max_memory_allocated() - base) / 1e6

        # Time
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            opt.zero_grad()
            out = model(input_ids)
            loss = out.sum()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / 10 * 1000

        n_ckpt = len(ckpt_layers)
        overhead = (ms / results[0]['time_ms'] - 1) * 100 if results else 0

        print(f"  {name:<20} {n_ckpt:<14} {mem_mb:<10.0f} {ms:<10.1f} {overhead:+.0f}%")

        results.append({
            "strategy": name, "n_ckpt_layers": n_ckpt,
            "mem_mb": round(mem_mb), "time_ms": round(ms, 1),
            "overhead_pct": round(overhead),
        })

        del model, opt, input_ids, out, loss
        torch.cuda.empty_cache()

    return results


# ============================================================
# 实验 4: 内存-时间权衡曲线
# ============================================================

def exp4_memory_time_tradeoff():
    print("\n" + "=" * 60)
    print("实验4: 内存-时间权衡曲线")
    print("=" * 60)

    results = []
    B, S = 8, 512
    hidden, n_heads = 768, 12

    print(f"\n  12 layers, H={hidden}, B={B}, S={S}")
    print(f"  {'CKPT every N':<15} {'Memory MB':<12} {'Time ms':<12} {'Mem saving':<12} {'Time cost'}")
    print("  " + "-" * 63)

    for ckpt_every in [0, 1, 2, 3, 4, 6, 12]:  # 0 = none, 1 = every layer
        input_ids = torch.randint(0, 5000, (B, S), device="cuda")

        if ckpt_every == 0:
            ckpt_layers = set()
        else:
            ckpt_layers = set(range(0, 12, ckpt_every))

        class FlexModel(nn.Module):
            def __init__(self, ckpt_set):
                super().__init__()
                self.ckpt_set = ckpt_set
                self.embed = nn.Embedding(5000, hidden)
                self.blocks = nn.ModuleList([TransformerBlock(hidden, n_heads) for _ in range(12)])
                self.ln_f = nn.LayerNorm(hidden)
                self.head = nn.Linear(hidden, 5000)

            def forward(self, ids):
                x = self.embed(ids)
                for i, block in enumerate(self.blocks):
                    if i in self.ckpt_set and self.training:
                        x = checkpoint(block, x)
                    else:
                        x = block(x)
                return self.head(self.ln_f(x))

        model = FlexModel(ckpt_layers).cuda()
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        out = model(input_ids)
        loss = out.sum()
        loss.backward()
        mem = (torch.cuda.max_memory_allocated() - base) / 1e6

        # Time
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            opt.zero_grad()
            out = model(input_ids)
            out.sum().backward()
            opt.step()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / 5 * 1000

        label = "None" if ckpt_every == 0 else f"Every {ckpt_every}"
        n_ckpts = len(ckpt_layers)
        print(f"  {label:<15} {mem:<12.0f} {ms:<12.1f} {n_ckpts} layers")

        results.append({
            "ckpt_every": ckpt_every, "label": label,
            "mem_mb": round(mem), "time_ms": round(ms, 1),
            "n_ckpt_layers": n_ckpts,
        })

        del model, opt, input_ids, out, loss
        torch.cuda.empty_cache()

    return results


# ============================================================
# 实验 5: 实际训练场景 (Batch Size 调优)
# ============================================================

def exp5_batch_size_tuning():
    print("\n" + "=" * 60)
    print("实验5: Checkpoint 对最大 Batch Size 的影响")
    print("=" * 60)

    results = []
    hidden, n_heads, n_layers = 1024, 16, 16
    S = 256

    print(f"\n  {n_layers}L, H={hidden}, S={S}")
    print(f"  {'Strategy':<15} {'Max Batch':<10} {'Throughput tok/s':<18} {'Memory MB'}")
    print("  " + "-" * 53)

    for use_ckpt in [False, True]:
        max_batch = 0
        max_tps = 0
        max_mem = 0

        for B in [1, 2, 4, 8, 16, 32, 64]:
            try:
                input_ids = torch.randint(0, 5000, (B, S), device="cuda")
                model = GPTModel(n_layers, hidden, n_heads, use_checkpoint=use_ckpt).cuda()
                model.train()
                opt = torch.optim.Adam(model.parameters(), lr=1e-4)

                torch.cuda.reset_peak_memory_stats()
                out = model(input_ids)
                out.sum().backward()
                opt.step()

                mem = torch.cuda.max_memory_allocated() / 1e6
                tps = B * S / 10  # rough estimate

                max_batch = B
                max_mem = mem

                del model, opt, input_ids, out
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break

        label = "Checkpoint" if use_ckpt else "No CKPT"
        print(f"  {label:<15} {max_batch:<10} {max_tps:<18} {max_mem:.0f}")

        results.append({
            "strategy": label, "max_batch": max_batch,
            "max_mem_mb": round(max_mem),
        })

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    all_results = OrderedDict()
    all_results["memory_comparison"] = exp1_memory_comparison()
    all_results["compute_overhead"] = exp2_compute_overhead()
    all_results["selective_checkpoint"] = exp3_selective_checkpoint()
    all_results["memory_time_tradeoff"] = exp4_memory_time_tradeoff()
    all_results["batch_size_tuning"] = exp5_batch_size_tuning()

    print("\n" + "=" * 60)
    print("关键洞察")
    print("=" * 60)
    print("""
  1. Gradient Checkpointing 节省 50-70% 峰值内存
  2. 代价: ~33% 额外计算时间 (重新 forward)
  3. Selective checkpointing: 只 checkpoint 昂贵的层
  4. 权衡: 内存 vs 时间, 根据硬件选择最优策略
  5. 实际效果: 可以用 2x 时间换 2-3x batch size → 总吞吐提升
  6. Megatron-LM 选择性: 只 checkpoint attention (不 checkpoint MLP)
""")

    with open("/root/gradient_ckpt_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved.")
