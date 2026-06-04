#!/usr/bin/env python3
"""Positional Encoding Comparison Experiment
==============================================
Compare different positional encoding schemes:
1. Learned positional embeddings (GPT-2 style)
2. Sinusoidal (original Transformer)
3. RoPE (Rotary Position Embedding, LLaMA style)
4. No positional encoding (baseline)

Tests on: length generalization, extrapolation, perplexity
Runs on GPU (RTX 4090) with synthetic data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ============================================================
# Positional Encoding Implementations
# ============================================================

class LearnedPositionalEncoding(nn.Module):
    """GPT-2 style: learnable position embeddings."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, T, C = x.shape
        return x + self.pe(torch.arange(T, device=x.device))


class SinusoidalPositionalEncoding(nn.Module):
    """Original Transformer: fixed sinusoidal."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class NoPositionalEncoding(nn.Module):
    """Baseline: no positional information."""
    def __init__(self, d_model, max_len=512):
        super().__init__()

    def forward(self, x):
        return x


class RoPEPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) — used in LLaMA, Mistral, etc.

    Key idea: encode position via rotation in 2D subspaces.
    Instead of adding position to embeddings, apply rotation to Q and K.
    """
    def __init__(self, d_model, max_len=512, base=10000.0):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        """Apply RoPE to input. x: [B, T, d_model]"""
        B, T, C = x.shape
        # Generate frequencies
        t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [T, d/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, d]
        cos = emb.cos().unsqueeze(0)  # [1, T, d]
        sin = emb.sin().unsqueeze(0)

        # Apply rotation
        x1, x2 = x[..., :C//2], x[..., C//2:]
        rotated = torch.cat([
            x1 * cos[..., :C//2] - x2 * sin[..., :C//2],
            x1 * sin[..., :C//2] + x2 * cos[..., :C//2],
        ], dim=-1)
        return rotated


# ============================================================
# Simple Transformer with configurable positional encoding
# ============================================================

class SimpleAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len=512, use_rope=False):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPEPositionalEncoding(self.head_dim, max_len)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len))
                             .view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = self.rope(q.reshape(B * self.n_head, T, self.head_dim)).reshape(B, self.n_head, T, self.head_dim)
            k = self.rope(k.reshape(B * self.n_head, T, self.head_dim)).reshape(B, self.n_head, T, self.head_dim)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, max_len=512, use_rope=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SimpleAttention(d_model, n_head, max_len, use_rope)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size=256, d_model=128, n_head=4, n_layer=4,
                 max_len=512, pos_type='learned'):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.max_len = max_len

        # Positional encoding
        if pos_type == 'rope':
            self.pos_enc = None  # RoPE applied in attention
            use_rope = True
        elif pos_type == 'learned':
            self.pos_enc = LearnedPositionalEncoding(d_model, max_len)
            use_rope = False
        elif pos_type == 'sinusoidal':
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
            use_rope = False
        else:  # none
            self.pos_enc = NoPositionalEncoding(d_model, max_len)
            use_rope = False

        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_head, max_len, use_rope) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Data: with positional dependencies
# ============================================================

def create_positional_data(vocab_size=256, seq_len=128, num_sequences=2000, device='cpu'):
    """Create data where position matters: token value depends on position."""
    data = []
    for _ in range(num_sequences):
        pattern_type = torch.randint(0, 4, (1,)).item()
        if pattern_type == 0:
            # Token = position % vocab_size / 2 (position-dependent)
            seq = torch.tensor([(i % (vocab_size // 2)) for i in range(seq_len)])
        elif pattern_type == 1:
            # Alternating pattern based on position parity
            a, b = torch.randint(10, vocab_size // 4, (2,)).tolist()
            seq = torch.tensor([a if i % 2 == 0 else b for i in range(seq_len)])
        elif pattern_type == 2:
            # Sliding window: token = sum of last 3 positions
            seq = []
            for i in range(seq_len):
                val = (seq[-1] + seq[-2] + i) % (vocab_size // 2) if len(seq) >= 2 else i % (vocab_size // 2)
                seq.append(val)
            seq = torch.tensor(seq)
        else:
            # Repeating with shift
            period = torch.randint(4, 16, (1,)).item()
            base = torch.randint(10, vocab_size // 4, (period,))
            seq = torch.tensor([base[i % period] + (i // period) % 10 for i in range(seq_len)])
        data.append(seq)
    return torch.stack(data).to(device)


# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model, data, epochs=20, batch_size=32, lr=3e-4, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * (len(data) // batch_size))
    model.train()
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(len(data))
        data_shuffled = data[perm]
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(data) - batch_size, batch_size):
            batch = data_shuffled[i:i + batch_size]
            _, loss = model(batch[:, :-1], batch[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

    return losses


def evaluate(model, data, device='cpu'):
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for i in range(0, len(data), 32):
            batch = data[i:i + 32]
            _, loss = model(batch[:, :-1], batch[:, 1:])
            total_loss += loss.item() * len(batch)
            n += len(batch)
    return total_loss / max(n, 1)


def test_extrapolation(model, max_train_len, vocab_size, device='cpu', n_samples=50):
    """Test how well model handles sequences longer than training length."""
    model.eval()
    results = {}
    for test_len in [max_train_len, max_train_len * 2, max_train_len * 4]:
        data = create_positional_data(vocab_size, seq_len=test_len,
                                       num_sequences=n_samples, device=device)
        loss = evaluate(model, data, device)
        ppl = math.exp(min(loss, 20))
        results[test_len] = {'loss': round(loss, 4), 'ppl': round(ppl, 2)}
    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Positional Encoding Comparison")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"\n  GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")

    vocab_size = 256
    train_len = 64
    d_model = 128
    n_head = 4
    n_layer = 4

    results = {}

    # Create train data
    train_data = create_positional_data(vocab_size, seq_len=train_len,
                                         num_sequences=2000, device=device)
    print(f"  Training: seq_len={train_len}, {len(train_data)} sequences")
    print(f"  Model: {n_layer} layers, {n_head} heads, d={d_model}")

    # Compare all 4 positional encodings
    pos_types = ['learned', 'sinusoidal', 'rope', 'none']
    all_results = {}

    for pos_type in pos_types:
        print(f"\n{'=' * 40}")
        print(f"  {pos_type.upper()} positional encoding")
        print(f"{'=' * 40}")

        model = TransformerModel(vocab_size, d_model, n_head, n_layer,
                                max_len=512, pos_type=pos_type).to(device)
        params = model.count_parameters()
        print(f"  Parameters: {params:,}")

        losses = train_model(model, train_data, epochs=25, batch_size=32, lr=3e-4, device=device)
        test_loss = evaluate(model, train_data, device)

        # Extrapolation test
        extrap = test_extrapolation(model, train_len, vocab_size, device)

        all_results[pos_type] = {
            'params': params,
            'train_loss': round(losses[-1], 4),
            'test_loss': round(test_loss, 4),
            'extrapolation': extrap,
        }

        print(f"  Final train loss: {losses[-1]:.4f}")
        print(f"  Test loss: {test_loss:.4f}")
        print(f"  Extrapolation:")
        for length, r in extrap.items():
            marker = " (train)" if length == train_len else ""
            print(f"    len={length}{marker}: loss={r['loss']:.4f}, ppl={r['ppl']:.2f}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary: Positional Encoding Comparison")
    print(f"{'=' * 60}")

    print(f"\n  {'Type':<15} {'Params':>10} {'Train Loss':>12} {'Extrap 2x':>12} {'Extrap 4x':>12}")
    print("  " + "-" * 65)
    for pos_type, r in all_results.items():
        extrap_2x = r['extrapolation'].get(train_len * 2, {}).get('loss', 'N/A')
        extrap_4x = r['extrapolation'].get(train_len * 4, {}).get('loss', 'N/A')
        print(f"  {pos_type:<15} {r['params']:>10,} {r['train_loss']:>12.4f} "
              f"{extrap_2x if isinstance(extrap_2x, str) else f'{extrap_2x:>12.4f}'} "
              f"{extrap_4x if isinstance(extrap_4x, str) else f'{extrap_4x:>12.4f}'}")

    # Degradation analysis
    print(f"\n  Extrapolation degradation (relative to train length):")
    print(f"  {'Type':<15} {'2x degrad.':>12} {'4x degrad.':>12}")
    print("  " + "-" * 42)
    for pos_type, r in all_results.items():
        train_loss = r['extrapolation'].get(train_len, {}).get('loss', r['train_loss'])
        loss_2x = r['extrapolation'].get(train_len * 2, {}).get('loss', 0)
        loss_4x = r['extrapolation'].get(train_len * 4, {}).get('loss', 0)
        deg_2x = ((loss_2x - train_loss) / train_loss * 100) if train_loss > 0 else 0
        deg_4x = ((loss_4x - train_loss) / train_loss * 100) if train_loss > 0 else 0
        print(f"  {pos_type:<15} {deg_2x:>+11.1f}% {deg_4x:>+11.1f}%")

    if device == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"\n  Peak GPU memory: {mem:.1f} MB")

    with open("positional_encoding_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to positional_encoding_results.json")


if __name__ == "__main__":
    import json
    main()
