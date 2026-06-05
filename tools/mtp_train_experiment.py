#!/usr/bin/env python3
"""Multi-Token Prediction (MTP) Training + Speculative Decoding
===============================================================
Implements DeepSeek-V3's MTP approach:
1. Main model predicts t+1 (standard)
2. MTP module predicts t+2 using main model's hidden state
3. Compare training convergence: with vs without MTP
4. Test speculative decoding speedup using MTP module

Reference: DeepSeek-AI, 2024, arXiv:2412.19437 (Section 2.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        S = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
        P = F.softmax(S, dim=-1)
        out = (P @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.ln2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """Mini GPT model with LLaMA-like architecture."""
    def __init__(self, vocab_size=200, dim=128, n_heads=4, n_layers=4, max_seq=256):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.ln_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.embed.weight
        self.max_seq = max_seq

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h), h  # Return logits AND hidden state (for MTP)


class MTPModule(nn.Module):
    """DeepSeek-V3 style Multi-Token Prediction module.

    Predicts token at position t+2 using:
    1. Main model's hidden state at position t (frozen)
    2. Embedding of the token at position t+1
    3. A projection + transformer layer
    """
    def __init__(self, dim, n_heads=4, vocab_size=200):
        super().__init__()
        self.embed_proj = nn.Linear(dim * 2, dim, bias=False)
        self.ln = RMSNorm(dim)
        self.block = TransformerBlock(dim, n_heads)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, main_hidden, target_embed):
        """
        main_hidden: (B, T, D) — hidden state from main model
        target_embed: (B, T, D) — embedding of the target token (t+1)
        Returns: logits for t+2
        """
        combined = torch.cat([main_hidden, target_embed], dim=-1)
        h = self.embed_proj(combined)
        h = self.ln(h)
        h = self.block(h)
        return self.head(h)


def generate_data(vocab_size, seq_len, n_samples, device='cpu'):
    """Generate synthetic data with learnable patterns."""
    torch.manual_seed(42)
    # Create data with repeating patterns
    pattern_len = min(16, seq_len // 2)
    patterns = torch.randint(0, vocab_size, (n_samples, pattern_len))
    data = patterns.repeat(1, seq_len // pattern_len + 1)[:, :seq_len + 2]
    return data.to(device)


def experiment1_mtp_training(device='cuda'):
    """Exp1: Compare training convergence with and without MTP."""
    print("\n" + "="*70)
    print("Experiment 1: MTP Training Convergence")
    print("="*70)

    vocab_size = 200
    seq_len = 64
    dim = 128
    n_heads = 4
    n_layers = 4
    batch_size = 32
    n_steps = 500

    results = {}

    for method in ['standard', 'mtp_depth1', 'mtp_depth2']:
        torch.manual_seed(42)
        data = generate_data(vocab_size, seq_len, 512, device=device)

        model = GPTModel(vocab_size, dim, n_heads, n_layers, max_seq=seq_len+4).to(device)

        # Create MTP modules
        n_mtp = {'standard': 0, 'mtp_depth1': 1, 'mtp_depth2': 2}[method]
        mtp_modules = nn.ModuleList([
            MTPModule(dim, n_heads, vocab_size).to(device) for _ in range(n_mtp)
        ])

        params = list(model.parameters()) + list(mtp_modules.parameters())
        optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

        loss_history = []
        main_loss_history = []
        start_time = time.time()

        for step in range(n_steps):
            # Sample batch
            indices = torch.randint(0, data.size(0) - batch_size, (1,)).item()
            batch = data[indices:indices + batch_size]

            x = batch[:, :seq_len]       # Input
            y1 = batch[:, 1:seq_len + 1] # Target t+1 (standard)
            y2 = batch[:, 2:seq_len + 2] # Target t+2 (MTP)

            # Forward pass
            logits, hidden = model(x)

            # Standard loss
            main_loss = F.cross_entropy(logits.reshape(-1, vocab_size), y1.reshape(-1))
            total_loss = main_loss

            # MTP losses
            for d in range(n_mtp):
                with torch.no_grad():
                    # Target embedding: embed the actual next token
                    target_embed = model.embed(y1 if d == 0 else y2)
                    main_h = hidden.detach()  # Stop gradient

                mtp_logits = mtp_modules[d](main_h, target_embed)
                target = y2  # Predict t+2
                mtp_loss = F.cross_entropy(mtp_logits.reshape(-1, vocab_size), target.reshape(-1))
                total_loss = total_loss + 0.3 * mtp_loss  # Lower weight

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            main_loss_history.append(main_loss.item())
            loss_history.append(total_loss.item())

            if (step + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  [{method:15s}] step {step+1:4d}: "
                      f"main_loss={main_loss.item():.4f}, "
                      f"total_loss={total_loss.item():.4f}, "
                      f"time={elapsed:.1f}s")

        final_main = sum(main_loss_history[-20:]) / 20
        final_total = sum(loss_history[-20:]) / 20
        total_time = time.time() - start_time

        # Generate text quality test
        model.eval()
        with torch.no_grad():
            prompt = data[:1, :8]  # 8 token prompt
            generated = prompt.clone()
            for _ in range(20):
                logits_g, _ = model(generated[:, -seq_len:])
                next_tok = logits_g[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tok], dim=1)

            # Check if generation follows the pattern
            pattern_match = sum(1 for i in range(8, min(28, generated.size(1)))
                              if generated[0, i].item() == data[0, i].item())

        results[method] = {
            'final_main_loss': final_main,
            'final_total_loss': final_total,
            'training_time': total_time,
            'pattern_match_pct': 100 * pattern_match / 20,
            'loss_history_last100': main_loss_history[-100:],
        }

        n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in mtp_modules.parameters())
        print(f"  [{method:15s}] final_main={final_main:.4f}, params={n_params/1e6:.2f}M, "
              f"pattern_match={pattern_match}/20 ({100*pattern_match/20:.0f}%)")

    print(f"\n  Convergence comparison (final main loss, lower is better):")
    for m in ['standard', 'mtp_depth1', 'mtp_depth2']:
        improvement = (results['standard']['final_main_loss'] - results[m]['final_main_loss']) / results['standard']['final_main_loss'] * 100
        print(f"    {m:15s}: {results[m]['final_main_loss']:.4f} ({improvement:+.2f}% vs standard)")

    return results


def experiment2_speculative_decoding(device='cuda'):
    """Exp2: Test speculative decoding speedup using MTP module."""
    print("\n" + "="*70)
    print("Experiment 2: Speculative Decoding Speedup with MTP")
    print("="*70)

    vocab_size = 200
    seq_len = 64
    dim = 128

    torch.manual_seed(42)

    # Create and train models
    data = generate_data(vocab_size, seq_len, 512, device=device)

    model = GPTModel(vocab_size, dim, n_heads=4, n_layers=4, max_seq=seq_len+4).to(device)
    mtp = MTPModule(dim, n_heads=4, vocab_size=vocab_size).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(mtp.parameters()),
        lr=3e-4
    )

    # Quick training
    model.train()
    for step in range(200):
        indices = torch.randint(0, data.size(0) - 16, (1,)).item()
        batch = data[indices:indices + 16]
        x, y1, y2 = batch[:, :seq_len], batch[:, 1:seq_len+1], batch[:, 2:seq_len+2]

        logits, hidden = model(x)
        main_loss = F.cross_entropy(logits.reshape(-1, vocab_size), y1.reshape(-1))

        with torch.no_grad():
            target_embed = model.embed(y1)
            main_h = hidden.detach()
        mtp_logits = mtp(main_h, target_embed)
        mtp_loss = F.cross_entropy(mtp_logits.reshape(-1, vocab_size), y2.reshape(-1))

        total_loss = main_loss + 0.3 * mtp_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    model.eval()
    mtp.eval()

    # Test speculative decoding
    results = {}
    prompt = data[:1, :16]  # 16 token prompt

    for decode_len in [20, 40, 60]:
        # Standard autoregressive decoding
        torch.cuda.synchronize()
        t0 = time.time()
        for trial in range(10):
            generated = prompt.clone()
            with torch.no_grad():
                for _ in range(decode_len):
                    logits, _ = model(generated[:, -seq_len:])
                    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_tok], dim=1)
        torch.cuda.synchronize()
        t_autoreg = (time.time() - t0) / 10 * 1000

        # Speculative decoding with MTP (verify 2 tokens at once)
        torch.cuda.synchronize()
        t0 = time.time()
        n_accepted = 0
        n_total_spec = 0
        for trial in range(10):
            generated = prompt.clone()
            with torch.no_grad():
                steps = 0
                while generated.size(1) - prompt.size(1) < decode_len:
                    steps += 1
                    logits, hidden = model(generated[:, -seq_len:])

                    # Main model's prediction for t+1
                    next_tok_main = logits[:, -1, :].argmax(dim=-1, keepdim=True)

                    # MTP prediction for t+2
                    next_embed = model.embed(next_tok_main)
                    mtp_logits = mtp(hidden[:, -1:, :], next_embed[:, -1:, :])
                    next_tok_mtp = mtp_logits[:, -1, :].argmax(dim=-1, keepdim=True)

                    # Verify: run main model on t+1 to check t+2
                    temp = torch.cat([generated, next_tok_main], dim=1)
                    verify_logits, _ = model(temp[:, -seq_len:])
                    verified_tok = verify_logits[:, -1, :].argmax(dim=-1, keepdim=True)

                    n_total_spec += 1
                    if verified_tok.item() == next_tok_mtp.item():
                        # Accept both tokens
                        generated = torch.cat([generated, next_tok_main, next_tok_mtp], dim=1)
                        n_accepted += 1
                    else:
                        # Accept only main model's token
                        generated = torch.cat([generated, next_tok_main, verified_tok], dim=1)

        torch.cuda.synchronize()
        t_spec = (time.time() - t0) / 10 * 1000

        accept_rate = n_accepted / max(n_total_spec, 1) * 100
        speedup = t_autoreg / t_spec

        results[f'decode_{decode_len}'] = {
            'autoreg_ms': t_autoreg,
            'spec_ms': t_spec,
            'speedup': speedup,
            'accept_rate': accept_rate,
        }

        print(f"  decode_len={decode_len}: autoreg={t_autoreg:.1f}ms, spec={t_spec:.1f}ms, "
              f"speedup={speedup:.2f}x, accept_rate={accept_rate:.1f}%")

    return results


def experiment3_mtp_depth_sweep(device='cuda'):
    """Exp3: Sweep MTP depth (0-4) and measure impact."""
    print("\n" + "="*70)
    print("Experiment 3: MTP Depth Sweep")
    print("="*70)

    vocab_size = 200
    seq_len = 64
    dim = 128
    batch_size = 16
    n_steps = 300

    results = {}

    for depth in [0, 1, 2, 3]:
        torch.manual_seed(42)
        data = generate_data(vocab_size, seq_len, 256, device=device)

        model = GPTModel(vocab_size, dim, n_heads=4, n_layers=4, max_seq=seq_len+6).to(device)
        mtp_modules = nn.ModuleList([
            MTPModule(dim, n_heads=4, vocab_size=vocab_size).to(device) for _ in range(depth)
        ])

        params = list(model.parameters()) + list(mtp_modules.parameters())
        optimizer = torch.optim.AdamW(params, lr=3e-4)
        n_params = sum(p.numel() for p in params)

        loss_hist = []
        for step in range(n_steps):
            idx = torch.randint(0, data.size(0) - batch_size, (1,)).item()
            batch = data[idx:idx + batch_size]
            x = batch[:, :seq_len]
            y1 = batch[:, 1:seq_len + 1]

            logits, hidden = model(x)
            total_loss = F.cross_entropy(logits.reshape(-1, vocab_size), y1.reshape(-1))

            for d in range(depth):
                with torch.no_grad():
                    # Target offset: depth d predicts t+d+2
                    te = model.embed(batch[:, 1+d:seq_len + 1+d])
                    mh = hidden[:, :seq_len - d - 1, :].detach() if d > 0 else hidden.detach()
                    te = te[:, :mh.size(1), :]
                ml = mtp_modules[d](mh, te)
                target = batch[:, 2+d:2+d+mh.size(1)]
                total_loss = total_loss + 0.3 * F.cross_entropy(ml.reshape(-1, vocab_size), target.reshape(-1))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_hist.append(total_loss.item())

        final = sum(loss_hist[-20:]) / 20
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()

        results[f'depth_{depth}'] = {
            'final_loss': final,
            'params_M': n_params / 1e6,
            'peak_mem_MB': peak_mem,
            'overhead_pct': (n_params / sum(p.numel() for p in model.parameters()) - 1) * 100,
        }

        print(f"  depth={depth}: loss={final:.4f}, params={n_params/1e6:.2f}M "
              f"(+{results[f'depth_{depth}']['overhead_pct']:.1f}%), "
              f"peak_mem={peak_mem:.0f}MB")

    return results


def experiment4_temperature_robustness(device='cuda'):
    """Exp4: Does MTP-trained model generalize better at different temperatures?"""
    print("\n" + "="*70)
    print("Experiment 4: Temperature Robustness")
    print("="*70)

    vocab_size = 200
    seq_len = 64
    dim = 128

    results = {}

    for method in ['standard', 'mtp']:
        torch.manual_seed(42)
        data = generate_data(vocab_size, seq_len, 256, device=device)

        model = GPTModel(vocab_size, dim, n_heads=4, n_layers=4, max_seq=seq_len+4).to(device)
        n_mtp = 1 if method == 'mtp' else 0
        mtp_modules = nn.ModuleList([
            MTPModule(dim, n_heads=4, vocab_size=vocab_size).to(device) for _ in range(n_mtp)
        ])

        params = list(model.parameters()) + list(mtp_modules.parameters())
        optimizer = torch.optim.AdamW(params, lr=3e-4)

        # Train
        for step in range(300):
            idx = torch.randint(0, data.size(0) - 16, (1,)).item()
            batch = data[idx:idx + 16]
            x, y1, y2 = batch[:, :seq_len], batch[:, 1:seq_len+1], batch[:, 2:seq_len+2]
            logits, hidden = model(x)
            total_loss = F.cross_entropy(logits.reshape(-1, vocab_size), y1.reshape(-1))

            for d in range(n_mtp):
                with torch.no_grad():
                    te = model.embed(y1)
                    mh = hidden.detach()
                ml = mtp_modules[d](mh, te)
                # Ensure matching sizes
                min_len = min(ml.size(1), y2.size(1))
                total_loss = total_loss + 0.3 * F.cross_entropy(
                    ml[:, :min_len, :].reshape(-1, vocab_size),
                    y2[:, :min_len].reshape(-1)
                )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        model.eval()

        # Test at different temperatures
        for temp in [0.1, 0.5, 1.0, 1.5, 2.0]:
            correct = 0
            total = 0
            with torch.no_grad():
                for i in range(50):
                    prompt = data[i:i+1, :16]
                    target = data[i, 16:36]  # Next 20 tokens

                    generated = prompt.clone()
                    for _ in range(20):
                        logits, _ = model(generated[:, -seq_len:])
                        probs = F.softmax(logits[:, -1, :] / temp, dim=-1)
                        next_tok = torch.multinomial(probs, 1)
                        generated = torch.cat([generated, next_tok], dim=1)

                    pred = generated[0, 16:36]
                    correct += (pred == target).sum().item()
                    total += 20

            accuracy = correct / total * 100
            results[f'{method}_T{temp}'] = accuracy
            print(f"  {method:8s} T={temp:.1f}: accuracy={accuracy:.1f}%")

    # Comparison table
    print(f"\n  Temperature robustness comparison:")
    print(f"  {'Temp':>5} | {'Standard':>10} | {'MTP':>10} | {'Diff':>10}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for temp in [0.1, 0.5, 1.0, 1.5, 2.0]:
        s = results[f'standard_T{temp}']
        m = results[f'mtp_T{temp}']
        print(f"  {temp:5.1f} | {s:9.1f}% | {m:9.1f}% | {m-s:+9.1f}%")

    return results


def run_all_experiments(device='cuda'):
    print("="*70)
    print("Multi-Token Prediction (MTP) Training + Speculative Decoding")
    print("Based on: DeepSeek-V3 (arXiv:2412.19437, Section 2.5)")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)

    all_results = {}
    all_results['exp1_convergence'] = experiment1_mtp_training(device)
    all_results['exp2_spec_decoding'] = experiment2_speculative_decoding(device)
    all_results['exp3_depth_sweep'] = experiment3_mtp_depth_sweep(device)
    all_results['exp4_temperature'] = experiment4_temperature_robustness(device)

    print("\n" + "="*70)
    print("SUMMARY: MTP Key Findings")
    print("="*70)

    exp1 = all_results['exp1_convergence']
    for m in ['standard', 'mtp_depth1', 'mtp_depth2']:
        if m in exp1:
            print(f"  {m:15s}: main_loss={exp1[m]['final_main_loss']:.4f}, "
                  f"pattern_match={exp1[m]['pattern_match_pct']:.0f}%")

    exp2 = all_results['exp2_spec_decoding']
    for k in sorted(exp2.keys()):
        print(f"  {k}: speedup={exp2[k]['speedup']:.2f}x, accept={exp2[k]['accept_rate']:.1f}%")

    def convert(obj):
        if isinstance(obj, torch.Tensor): return obj.item()
        elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    with open('mtp_results.json', 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print("Results saved to mtp_results.json")
    return all_results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    run_all_experiments(device=device)
