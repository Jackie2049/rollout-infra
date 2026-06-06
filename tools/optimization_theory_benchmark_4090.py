#!/usr/bin/env python3
"""Optimization Theory Benchmark on RTX 4090
============================================

Benchmarks training optimization algorithms and their properties:
1. SGD vs Adam vs AdamW convergence on MiniGPT
2. Learning Rate Schedules (Cosine/Linear/Warmup) comparison
3. Weight Decay: AdamW vs L2 Regularization
4. Gradient Accumulation efficiency
5. BF16 vs FP32 training: convergence + speed

This bridges theory (★2 gap) with practice (RTX 4090 GPU benchmark).

Usage:
  python optimization_theory_benchmark_4090.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn as nn
import json
import math
import time

def benchmark_cuda(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat

# Simple Transformer model for training benchmark
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=40, d_model=256, n_heads=4, n_layers=2, max_seq=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=4*d_model,
                dropout=0.1, activation='gelu',
                batch_first=True, norm_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_seq = max_seq
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embed(x) * math.sqrt(self.d_model) + self.pos_embed(positions)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)

# ================================================================
# Experiment 1: SGD vs Adam vs AdamW Convergence
# ================================================================
def exp1_optimizer_comparison():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 1: SGD vs Adam vs AdamW Convergence")
    print("=" * 60)

    vocab_size = 40
    max_seq = 32
    n_steps = 100
    batch_size = 32

    # Generate synthetic training data (character-level, simple patterns)
    torch.manual_seed(42)
    data = torch.randint(0, vocab_size, (5000, max_seq + 1), device='cpu')

    configs = [
        ("SGD_lr0.01", lambda p: torch.optim.SGD(p, lr=0.01)),
        ("SGD_lr0.1", lambda p: torch.optim.SGD(p, lr=0.1)),
        ("SGD_lr1.0", lambda p: torch.optim.SGD(p, lr=1.0)),
        ("Adam_lr0.001", lambda p: torch.optim.Adam(p, lr=0.001)),
        ("Adam_lr0.01", lambda p: torch.optim.Adam(p, lr=0.01)),
        ("Adam_lr0.1", lambda p: torch.optim.Adam(p, lr=0.1)),
        ("AdamW_lr0.001_wd0.01", lambda p: torch.optim.AdamW(p, lr=0.001, weight_decay=0.01)),
        ("AdamW_lr0.001_wd0.1", lambda p: torch.optim.AdamW(p, lr=0.001, weight_decay=0.1)),
    ]

    results = []

    for name, opt_fn in configs:
        torch.manual_seed(42)
        model = MiniTransformer(vocab_size=vocab_size, max_seq=max_seq).to(device)
        optimizer = opt_fn(model.parameters())
        criterion = nn.CrossEntropyLoss()

        losses = []
        step_times = []

        for step in range(n_steps):
            # Sample batch
            idx = torch.randint(0, data.size(0) - 1, (batch_size,))
            inputs = data[idx, :max_seq].to(device)
            targets = data[idx, 1:max_seq+1].to(device)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            end.record()
            torch.cuda.synchronize()
            step_time = start.elapsed_time(end)

            losses.append(loss.item())
            step_times.append(step_time)

        # Compute metrics
        initial_loss = losses[0]
        final_loss = losses[-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        avg_step_time = sum(step_times) / len(step_times)

        # Find best loss
        best_loss = min(losses)
        best_step = losses.index(best_loss)

        # Compute throughput
        total_tokens = batch_size * max_seq * n_steps
        total_time = sum(step_times) / 1000  # convert to seconds
        throughput = total_tokens / total_time

        print(f"  {name}: loss {initial_loss:.3f} → {final_loss:.3f} "
              f"(↓{loss_reduction:.1f}%), best={best_loss:.3f}@step{best_step}, "
              f"avg_step={avg_step_time:.2f}ms, {throughput:.0f} tok/s")

        results.append({
            "optimizer": name,
            "initial_loss": round(initial_loss, 3),
            "final_loss": round(final_loss, 3),
            "loss_reduction_pct": round(loss_reduction, 1),
            "best_loss": round(best_loss, 3),
            "best_step": best_step,
            "avg_step_ms": round(avg_step_time, 2),
            "throughput_tok_s": round(throughput, 0),
            "loss_curve": [round(l, 3) for l in losses[::10]],  # every 10 steps
        })

    return results

# ================================================================
# Experiment 2: Learning Rate Schedule Comparison
# ================================================================
def exp2_lr_schedule():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 2: Learning Rate Schedule Comparison")
    print("=" * 60)

    vocab_size = 40
    max_seq = 32
    n_steps = 200
    batch_size = 32
    base_lr = 0.001

    torch.manual_seed(42)
    data = torch.randint(0, vocab_size, (5000, max_seq + 1), device='cpu')

    def cosine_lr(step, total_steps, warmup_steps=20):
        if step < warmup_steps:
            return base_lr * step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    def linear_lr(step, total_steps, warmup_steps=20):
        if step < warmup_steps:
            return base_lr * step / warmup_steps
        return base_lr * (1 - (step - warmup_steps) / (total_steps - warmup_steps))

    def constant_lr(step, total_steps, warmup_steps=0):
        return base_lr

    def warmup_constant_lr(step, total_steps, warmup_steps=20):
        if step < warmup_steps:
            return base_lr * step / warmup_steps
        return base_lr

    schedules = [
        ("cosine_warmup20", cosine_lr),
        ("linear_decay", linear_lr),
        ("constant", constant_lr),
        ("warmup_constant20", warmup_constant_lr),
    ]

    results = []

    for name, lr_fn in schedules:
        torch.manual_seed(42)
        model = MiniTransformer(vocab_size=vocab_size, max_seq=max_seq).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        losses = []
        lrs = []

        for step in range(n_steps):
            idx = torch.randint(0, data.size(0) - 1, (batch_size,))
            inputs = data[idx, :max_seq].to(device)
            targets = data[idx, 1:max_seq+1].to(device)

            # Update learning rate
            current_lr = lr_fn(step, n_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            lrs.append(current_lr)

        initial_loss = losses[0]
        final_loss = losses[-1]
        best_loss = min(losses)
        best_step = losses.index(best_loss)
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100

        print(f"  {name}: loss {initial_loss:.3f} → {final_loss:.3f} "
              f"(↓{loss_reduction:.1f}%), best={best_loss:.3f}@step{best_step}, "
              f"lr_range=[{min(lrs):.6f}, {max(lrs):.6f}]")

        results.append({
            "schedule": name,
            "initial_loss": round(initial_loss, 3),
            "final_loss": round(final_loss, 3),
            "loss_reduction_pct": round(loss_reduction, 1),
            "best_loss": round(best_loss, 3),
            "best_step": best_step,
            "lr_min": round(min(lrs), 6),
            "lr_max": round(max(lrs), 6),
            "loss_curve": [round(l, 3) for l in losses[::20]],
            "lr_curve": [round(l, 6) for l in lrs[::20]],
        })

    return results

# ================================================================
# Experiment 3: Weight Decay (AdamW vs L2 Regularization)
# ================================================================
def exp3_weight_decay():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 3: Weight Decay (AdamW vs L2 Regularization)")
    print("=" * 60)

    # Theory:
    # L2 regularization: gradient += weight_decay * weight → added BEFORE adaptive scaling
    # AdamW: weight decay applied AFTER adaptive scaling → decoupled
    # Key difference: L2 in Adam scales wd by 1/(sqrt(v)+eps) → wd is no longer constant!
    # AdamW: wd is truly constant, independent of gradient history

    vocab_size = 40
    max_seq = 32
    n_steps = 100
    batch_size = 32
    wd_values = [0.0, 0.01, 0.1, 0.5]

    torch.manual_seed(42)
    data = torch.randint(0, vocab_size, (5000, max_seq + 1), device='cpu')

    configs = [
        ("AdamW_wd0.0", 0.0),
        ("AdamW_wd0.01", 0.01),
        ("AdamW_wd0.1", 0.1),
        ("AdamW_wd0.5", 0.5),
        ("L2_wd0.01", 0.01),  # Adam with L2 (add wd to loss)
        ("L2_wd0.1", 0.1),
    ]

    results = []

    for name, wd in configs:
        torch.manual_seed(42)
        model = MiniTransformer(vocab_size=vocab_size, max_seq=max_seq).to(device)
        criterion = nn.CrossEntropyLoss()

        if name.startswith("AdamW"):
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=wd)
        else:
            # Adam with L2: add wd * ||w||^2 to loss
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        losses = []
        weight_norms = []

        for step in range(n_steps):
            idx = torch.randint(0, data.size(0) - 1, (batch_size,))
            inputs = data[idx, :max_seq].to(device)
            targets = data[idx, 1:max_seq+1].to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.reshape(-1, vocab_size), targets.reshape(-1))

            if name.startswith("L2"):
                # Add L2 regularization to loss
                l2_reg = 0
                for param in model.parameters():
                    l2_reg += torch.norm(param, 2) ** 2
                loss = loss + wd * l2_reg

            loss.backward()
            optimizer.step()

            # Track weight norm
            total_norm = 0
            for param in model.parameters():
                total_norm += torch.norm(param, 2).item() ** 2
            weight_norms.append(math.sqrt(total_norm))

            losses.append(loss.item())

        final_loss = losses[-1]
        final_weight_norm = weight_norms[-1]
        initial_norm = weight_norms[0]

        print(f"  {name}: loss={final_loss:.3f}, weight_norm "
              f"{initial_norm:.2f} → {final_weight_norm:.2f} "
              f"(change {(final_weight_norm-initial_norm)/initial_norm*100:.1f}%)")

        results.append({
            "config": name,
            "weight_decay": wd,
            "final_loss": round(final_loss, 3),
            "initial_weight_norm": round(initial_norm, 2),
            "final_weight_norm": round(final_weight_norm, 2),
            "weight_change_pct": round((final_weight_norm-initial_norm)/initial_norm*100, 1),
        })

    return results

# ================================================================
# Experiment 4: Gradient Accumulation
# ================================================================
def exp4_gradient_accumulation():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 4: Gradient Accumulation Efficiency")
    print("=" * 60)

    # Theory: gradient accumulation = simulate larger batch by accumulating gradients
    # effective_batch = micro_batch × accumulation_steps
    # Math: loss should be identical (gradients are averaged, not summed)
    # Speed: more steps = slower, but allows larger effective batch on limited memory

    vocab_size = 40
    max_seq = 32
    n_epochs = 5  # fixed total tokens processed
    micro_batch = 8
    accum_steps_options = [1, 2, 4, 8, 16]

    torch.manual_seed(42)
    data = torch.randint(0, vocab_size, (2000, max_seq + 1), device='cpu')

    results = []

    for accum_steps in accum_steps_options:
        effective_batch = micro_batch * accum_steps

        torch.manual_seed(42)
        model = MiniTransformer(vocab_size=vocab_size, max_seq=max_seq).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        total_steps = 0
        total_time_ms = 0
        losses = []

        # Fixed number of gradient updates (= fixed total tokens)
        n_updates = 50
        n_data_points_needed = n_updates * effective_batch

        for update_step in range(n_updates):
            torch.cuda.synchronize()
            t_start = time.time()

            optimizer.zero_grad()

            for micro_step in range(accum_steps):
                idx = torch.randint(0, data.size(0) - 1, (micro_batch,))
                inputs = data[idx, :max_seq].to(device)
                targets = data[idx, 1:max_seq+1].to(device)

                output = model(inputs)
                loss = criterion(output.reshape(-1, vocab_size), targets.reshape(-1))
                # Scale loss by accumulation steps (average, not sum)
                loss = loss / accum_steps
                loss.backward()

            # Single optimizer step after accumulation
            optimizer.step()

            torch.cuda.synchronize()
            t_end = time.time()
            step_time_ms = (t_end - t_start) * 1000

            # Record the unscaled loss for comparison
            losses.append(loss.item() * accum_steps)
            total_time_ms += step_time_ms
            total_steps += 1

        # Compute memory usage
        mem_allocated = torch.cuda.memory_allocated() / 1e6  # MB
        mem_reserved = torch.cuda.memory_reserved() / 1e6

        avg_loss = sum(losses[-10:]) / 10  # average of last 10
        avg_step_time = total_time_ms / total_steps

        # Throughput: tokens per second
        total_tokens = n_updates * effective_batch * max_seq
        throughput = total_tokens / (total_time_ms / 1000)

        print(f"  accum={accum_steps}: effective_batch={effective_batch}, "
              f"avg_loss={avg_loss:.3f}, step_time={avg_step_time:.2f}ms, "
              f"throughput={throughput:.0f} tok/s, "
              f"mem={mem_allocated:.1f}MB")

        results.append({
            "accum_steps": accum_steps,
            "effective_batch": effective_batch,
            "avg_loss": round(avg_loss, 3),
            "avg_step_time_ms": round(avg_step_time, 2),
            "throughput_tok_s": round(throughput, 0),
            "mem_allocated_mb": round(mem_allocated, 1),
            "mem_reserved_mb": round(mem_reserved, 1),
        })

    return results

# ================================================================
# Experiment 5: BF16 vs FP32 Training
# ================================================================
def exp5_bf16_fp32_training():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print("\n" + "=" * 60)
    print("Experiment 5: BF16 vs FP32 Training (Speed + Convergence)")
    print("=" * 60)

    vocab_size = 40
    max_seq = 32
    n_steps = 50
    batch_sizes = [16, 32, 64, 128]

    torch.manual_seed(42)
    data = torch.randint(0, vocab_size, (5000, max_seq + 1), device='cpu')

    dtypes = [("FP32", torch.float32), ("BF16", torch.bfloat16)]

    results = []

    for dtype_name, dtype in dtypes:
        for B in batch_sizes:
            torch.manual_seed(42)

            # Create model in target dtype
            model = MiniTransformer(vocab_size=vocab_size, max_seq=max_seq).to(device)
            if dtype == torch.bfloat16:
                model = model.to(torch.bfloat16)

            # AMP for BF16: use GradScaler for FP16, NOT for BF16
            # BF16 has the same dynamic range as FP32 → no overflow risk
            scaler = None  # BF16 doesn need GradScaler

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            step_times = []
            losses = []

            for step in range(n_steps):
                idx = torch.randint(0, data.size(0) - 1, (B,))
                inputs = data[idx, :max_seq].to(device)
                targets = data[idx, 1:max_seq+1].to(device)

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                optimizer.zero_grad()
                output = model(inputs)
                # CrossEntropyLoss needs float32 logits for numerical stability
                if dtype == torch.bfloat16:
                    loss = criterion(output.float().reshape(-1, vocab_size),
                                     targets.reshape(-1))
                else:
                    loss = criterion(output.reshape(-1, vocab_size),
                                     targets.reshape(-1))
                loss.backward()
                optimizer.step()

                end_event.record()
                torch.cuda.synchronize()
                step_time = start_event.elapsed_time(end_event)

                losses.append(loss.item())
                step_times.append(step_time)

            avg_loss = sum(losses[-10:]) / 10
            avg_step_time = sum(step_times) / len(step_times)

            # Throughput
            total_tokens = B * max_seq * n_steps
            total_time_s = sum(step_times) / 1000
            throughput = total_tokens / total_time_s

            # Memory
            mem = torch.cuda.max_memory_allocated() / 1e6  # MB

            print(f"  {dtype_name} B={B}: loss={avg_loss:.3f}, "
                  f"step={avg_step_time:.2f}ms, "
                  f"throughput={throughput:.0f} tok/s, mem={mem:.1f}MB")

            results.append({
                "dtype": dtype_name,
                "batch": B,
                "avg_loss": round(avg_loss, 3),
                "avg_step_ms": round(avg_step_time, 2),
                "throughput_tok_s": round(throughput, 0),
                "mem_mb": round(mem, 1),
            })

    # Compute speedup
    for B in batch_sizes:
        fp32_entry = next(r for r in results if r["dtype"] == "FP32" and r["batch"] == B)
        bf16_entry = next(r for r in results if r["dtype"] == "BF16" and r["batch"] == B)
        speedup = fp32_entry["avg_step_ms"] / bf16_entry["avg_step_ms"]
        print(f"  BF16 speedup at B={B}: {speedup:.2f}x")

    return results

# ================================================================
# Main
# ================================================================
def main():
    device = 'cuda:0'
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9

    print(f"Optimization Theory Benchmark: {gpu_name} ({gpu_mem:.1f} GB)")
    print("=" * 60)

    all_results = {
        "gpu": gpu_name, "gpu_mem_gb": round(gpu_mem, 1),
    }

    all_results["exp1_optimizer_comparison"] = exp1_optimizer_comparison()
    all_results["exp2_lr_schedule"] = exp2_lr_schedule()
    all_results["exp3_weight_decay"] = exp3_weight_decay()
    all_results["exp4_gradient_accumulation"] = exp4_gradient_accumulation()
    all_results["exp5_bf16_fp32_training"] = exp5_bf16_fp32_training()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    opt = all_results["exp1_optimizer_comparison"]
    best_optimizer = min(opt, key=lambda r: r["best_loss"])
    print(f"  Best optimizer: {best_optimizer['optimizer']} (loss={best_optimizer['best_loss']:.3f})")

    lr = all_results["exp2_lr_schedule"]
    best_schedule = min(lr, key=lambda r: r["best_loss"])
    print(f"  Best LR schedule: {best_schedule['schedule']} (loss={best_schedule['best_loss']:.3f})")

    wd = all_results["exp3_weight_decay"]
    for r in wd:
        print(f"  {r['config']}: loss={r['final_loss']:.3f}, weight_norm_change={r['weight_change_pct']:.1f}%")

    accum = all_results["exp4_gradient_accumulation"]
    best_accum = max(accum, key=lambda r: r["throughput_tok_s"])
    print(f"  Best throughput: accum={best_accum['accum_steps']} ({best_accum['throughput_tok_s']} tok/s)")

    bf16 = all_results["exp5_bf16_fp32_training"]
    for B in [32, 128]:
        fp32 = next(r for r in bf16 if r["dtype"] == "FP32" and r["batch"] == B)
        bf16e = next(r for r in bf16 if r["dtype"] == "BF16" and r["batch"] == B)
        print(f"  BF16/FP32 B={B}: {fp32['avg_step_ms']/bf16e['avg_step_ms']:.2f}x speedup, "
              f"loss diff={abs(fp32['avg_loss']-bf16e['avg_loss']):.3f}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'optimization_theory_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()