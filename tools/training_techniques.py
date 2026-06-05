#!/usr/bin/env python3
"""LLM Training Techniques — Key Recipes from Scratch
=====================================================
Demonstrates essential LLM training techniques:
1. Learning rate warmup + cosine decay
2. Gradient clipping (norm-based)
3. Mixed precision (FP16/BF16) with GradScaler
4. Weight decay (AdamW vs Adam)
5. Model scale effect

Reference: LLaMA 3, GPT-3, Chinchilla training recipes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json


class MiniTransformer(nn.Module):
    """Small transformer for training technique experiments."""
    def __init__(self, vocab_size=256, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            self._make_layer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying

    def _make_layer(self, d_model, n_heads, d_ff, dropout):
        return nn.ModuleDict({
            'ln1': nn.LayerNorm(d_model),
            'attn': nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True),
            'ln2': nn.LayerNorm(d_model),
            'ffn': nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            ),
            'dropout': nn.Dropout(dropout),
        })

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)

        for layer in self.layers:
            h_norm = layer['ln1'](h)
            attn_out, _ = layer['attn'](h_norm, h_norm, h_norm,
                                         need_weights=False)
            h = h + layer['dropout'](attn_out)
            h = h + layer['ffn'](layer['ln2'](h))

        return self.head(self.ln_f(h))


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps,
                                     min_lr_ratio=0.1):
    """Cosine annealing with linear warmup (LLaMA/GPT-3 style)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def generate_synthetic_data(vocab_size=256, seq_len=128, n_samples=2000,
                            device='cuda'):
    """Generate synthetic data with repeating patterns."""
    data = []
    for _ in range(n_samples):
        pattern_len = torch.randint(4, 16, (1,)).item()
        pattern = torch.randint(0, vocab_size, (pattern_len,))
        seq = pattern.repeat(seq_len // pattern_len + 1)[:seq_len]
        data.append(seq)
    return torch.stack(data).to(device)


def run_experiments(device='cuda'):
    print("=" * 70)
    print("LLM Training Techniques — Key Recipes from Scratch")
    print(f"Device: {device}")
    print("=" * 70)

    results = {}

    vocab_size = 256
    n_steps = 500
    warmup_steps = 50
    data = generate_synthetic_data(vocab_size, 128, 2000, device)

    # ----------------------------------------------------------
    # Experiment 1: LR Schedule Comparison
    # ----------------------------------------------------------
    print("\n--- Experiment 1: LR Schedule Comparison ---")

    d_model, n_heads, n_layers, d_ff, max_seq_len = 128, 4, 2, 512, 128

    schedules = {
        'constant': lambda opt, ws, ts: torch.optim.lr_scheduler.LambdaLR(
            opt, lambda step: 1.0),
        'cosine_warmup': lambda opt, ws, ts: get_cosine_schedule_with_warmup(
            opt, ws, ts),
        'linear_decay': lambda opt, ws, ts: torch.optim.lr_scheduler.LambdaLR(
            opt, lambda step: max(0.1, 1.0 - step / ts)),
        'step_decay': lambda opt, ws, ts: torch.optim.lr_scheduler.StepLR(
            opt, step_size=ts // 3, gamma=0.1),
    }

    for sched_name, sched_fn in schedules.items():
        torch.manual_seed(42)
        model = MiniTransformer(vocab_size, d_model, n_heads, n_layers,
                                d_ff, max_seq_len).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = sched_fn(optimizer, warmup_steps, n_steps)

        losses = []
        for step in range(n_steps):
            idx = torch.randint(0, data.shape[0], (32,))
            batch = data[idx]
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        print(f"  {sched_name:16s}: final={losses[-1]:.4f}, min={min(losses):.4f}")
        results[f'lr_{sched_name}'] = {'final': losses[-1], 'min': min(losses)}

    # ----------------------------------------------------------
    # Experiment 2: Gradient Clipping Effect
    # ----------------------------------------------------------
    print("\n--- Experiment 2: Gradient Clipping Effect ---")

    for max_norm in [0.0, 0.5, 1.0, 5.0, 10.0]:
        torch.manual_seed(42)
        model = MiniTransformer(vocab_size, d_model, n_heads, n_layers,
                                d_ff, max_seq_len).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, n_steps)

        losses = []
        grad_norms = []
        for step in range(n_steps):
            idx = torch.randint(0, data.shape[0], (32,))
            batch = data[idx]
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm if max_norm > 0 else float('inf'))
            grad_norms.append(norm.item())
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        label = f"clip={max_norm}" if max_norm > 0 else "no_clip"
        print(f"  {label:12s}: final={losses[-1]:.4f}, "
              f"avg_grad={sum(grad_norms)/len(grad_norms):.2f}, "
              f"max_grad={max(grad_norms):.2f}")
        results[f'grad_{label}'] = {
            'final': losses[-1], 'avg_grad': sum(grad_norms)/len(grad_norms),
        }

    # ----------------------------------------------------------
    # Experiment 3: AdamW vs Adam
    # ----------------------------------------------------------
    print("\n--- Experiment 3: AdamW vs Adam Weight Decay ---")

    for opt_name, opt_cls, wd in [
        ('AdamW_0.01', torch.optim.AdamW, 0.01),
        ('AdamW_0.1', torch.optim.AdamW, 0.1),
        ('Adam_0.01', torch.optim.Adam, 0.01),
        ('Adam_no_wd', torch.optim.Adam, 0.0),
    ]:
        torch.manual_seed(42)
        model = MiniTransformer(vocab_size, d_model, n_heads, n_layers,
                                d_ff, max_seq_len).to(device)
        optimizer = opt_cls(model.parameters(), lr=1e-3, weight_decay=wd)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, n_steps)

        losses = []
        wnorms = []
        for step in range(n_steps):
            idx = torch.randint(0, data.shape[0], (32,))
            batch = data[idx]
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if step % 50 == 0:
                wnorms.append(sum(p.norm().item()**2 for p in model.parameters())**0.5)

        print(f"  {opt_name:14s}: final={losses[-1]:.4f}, wnorm={wnorms[-1]:.2f}")
        results[f'opt_{opt_name}'] = {'final': losses[-1], 'wnorm': wnorms[-1]}

    # ----------------------------------------------------------
    # Experiment 4: Mixed Precision
    # ----------------------------------------------------------
    print("\n--- Experiment 4: Mixed Precision (FP16/BF16/FP32) ---")

    for dtype_name, dtype in [('FP32', torch.float32),
                               ('FP16', torch.float16),
                               ('BF16', torch.bfloat16)]:
        torch.manual_seed(42)
        model = MiniTransformer(vocab_size, d_model, n_heads, n_layers,
                                d_ff, max_seq_len).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, n_steps)
        use_scaler = (dtype == torch.float16)
        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        losses = []

        for step in range(n_steps):
            idx = torch.randint(0, data.shape[0], (32,))
            batch = data[idx]
            x, y = batch[:, :-1], batch[:, 1:]
            with torch.amp.autocast('cuda', dtype=dtype, enabled=(dtype != torch.float32)):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())

        elapsed = time.time() - t0
        mem_peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"  {dtype_name}: final={losses[-1]:.4f}, "
              f"time={elapsed:.2f}s, peak_mem={mem_peak:.1f}MB")
        results[f'dtype_{dtype_name}'] = {
            'final': losses[-1], 'time': elapsed, 'mem_mb': mem_peak,
        }

    # ----------------------------------------------------------
    # Experiment 5: Model Scale Effect
    # ----------------------------------------------------------
    print("\n--- Experiment 5: Model Scale Effect ---")

    for name, dm, nh, nl, dff in [
        ('tiny', 64, 2, 2, 128),
        ('small', 128, 4, 2, 256),
        ('medium', 256, 4, 4, 512),
        ('large', 256, 8, 6, 1024),
    ]:
        torch.manual_seed(42)
        model = MiniTransformer(vocab_size, dm, nh, nl, dff, 128).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        # Warmup
        for _ in range(5):
            idx = torch.randint(0, data.shape[0], (32,))
            batch = data[idx]
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                model(batch[:, :-1])
        torch.cuda.synchronize()

        t0 = time.time()
        for step in range(100):
            idx = torch.randint(0, data.shape[0], (32,))
            batch = data[idx]
            x, y = batch[:, :-1], batch[:, 1:]
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        print(f"  {name:8s}: {n_params/1e6:.2f}M params, {elapsed:.2f}s/100steps, loss={loss.item():.4f}")
        results[f'scale_{name}'] = {
            'params_M': n_params/1e6, 'time_100': elapsed,
            'loss': loss.item(),
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: LLM Training Recipe Key Findings")
    print("=" * 70)
    print("""
1. LR Schedule: Cosine warmup is gold standard (LLaMA, GPT-3)
2. Gradient Clipping: max_norm=1.0 prevents explosions
3. AdamW: Decoupled weight decay > Adam's L2 regularization
4. Mixed Precision: BF16 safest (no scaler), FP16 needs scaler
5. Model Scale: Larger → lower loss, throughput ∝ 1/params
    """)

    with open('training_techniques_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to training_techniques_results.json")
    return results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    run_experiments(device=device)
