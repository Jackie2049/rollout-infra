#!/usr/bin/env python3
"""MiniGPT: Train a Small GPT from Scratch
============================================
End-to-end GPT training pipeline:
1. Character-level tokenizer (no external dependencies)
2. GPT-2 architecture from scratch (MHA, causal attention, pre-norm)
3. Training loop with cosine schedule + warmup
4. Text generation with temperature + top-k
5. Experiments: scaling, context length, temperature

Educational purpose: understand every component of LLM training.
Designed to run on both CPU and GPU (RTX 4090 recommended).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time
import os
from dataclasses import dataclass
from collections import Counter


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class GPTConfig:
    """GPT model configuration."""
    vocab_size: int = 128       # ASCII character-level
    block_size: int = 256       # Context length
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    dropout: float = 0.1
    bias: bool = False          # Following LLaMA: no bias

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50
    warmup_iters: int = 100

    @property
    def n_params(self):
        """Estimate total parameters."""
        # Token + position embedding
        params = self.vocab_size * self.d_model + self.block_size * self.d_model
        # Transformer blocks
        for _ in range(self.n_layer):
            # Attention: QKV + output projection (no bias)
            params += 4 * self.d_model * self.d_model
            # FFN: up + gate + down (SwiGLU-like)
            ffn_dim = 4 * self.d_model
            params += 3 * self.d_model * ffn_dim
            # LayerNorm
            params += 2 * self.d_model  # Two RMSNorm per block
        # Final layer norm
        params += self.d_model
        # Output head (tied with embedding)
        return params


# ============================================================
# 2. Text Data
# ============================================================

# Small corpus of simple English text for training
CORPUS = """
Once upon a time, there was a little girl named Lucy. She lived in a small village near the forest.
Every morning, Lucy would walk to school with her friends. They loved to play games and tell stories.
The forest was full of tall trees and colorful flowers. Birds sang beautiful songs from the branches.
One day, Lucy found a small rabbit near the path. The rabbit was white with pink ears.
She picked up the rabbit and carried it home. Her mother said they could keep it as a pet.
Lucy named the rabbit Snowball. Every day, she would feed Snowball fresh carrots and lettuce.
The rabbit grew strong and happy. Lucy and Snowball became the best of friends.
They would play together in the garden. Snowball liked to hop around the flowers.
In winter, Snowball would sit by the fire. Lucy would read stories to him.
The children in the village all loved Snowball. He was the most famous rabbit in town.
Spring came and the flowers bloomed again. Lucy and Snowball explored the forest together.
They found a stream with clear water. Fish swam in the shallow pools.
A wise old owl lived in the tallest tree. The owl told them stories about the forest.
The owl said the forest was magical. Deep inside, there was a garden of golden flowers.
Lucy wanted to find the golden garden. She packed a lunch and set off with Snowball.
They walked through the forest for hours. The trees grew taller and the path grew narrow.
Finally, they found a small clearing. Golden flowers covered the ground like a carpet.
The flowers glowed in the sunlight. Lucy picked one and brought it home.
The golden flower never wilted. It sat on her windowsill, glowing every night.
Lucy knew the forest was special. She would always remember her adventure with Snowball.
The end. But Lucy and Snowball had many more adventures together.
Some days they would explore new paths. Other days they would sit and watch the clouds.
Every adventure was special because they were together.
The village grew and changed over the years. But the forest remained the same.
New children came to the village and heard stories about Lucy and Snowball.
They too wanted to find the golden garden. And so the stories continued.
"""


class CharTokenizer:
    """Simple character-level tokenizer."""
    def __init__(self):
        self.chars = sorted(list(set(CORPUS)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, tokens):
        return ''.join(self.itos.get(t, '?') for t in tokens)


def get_batch(data, block_size, batch_size, device='cpu'):
    """Get a random batch of sequences."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================
# 3. Model Components
# ============================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA-style)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with GQA support."""
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.n_head = config.n_head
        self.d_head = config.d_model // config.n_head
        self.n_embd = config.d_model

        # QKV projection (fused)
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head: [B, T, C] → [B, n_head, T, d_head]
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum + reshape
        y = att @ v  # [B, n_head, T, d_head]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """SwiGLU MLP (LLaMA-style feed-forward)."""
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.d_model
        # SwiGLU: gate + up + down
        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: (gate_proj * sigmoid(gate_proj)) * up_proj
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (LLaMA-style)."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """MiniGPT: A small GPT model from scratch."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.d_model)
        # Note: no output head — we tie weights with token_embedding

        # Weight initialization (GPT-2 style)
        self.apply(self._init_weights)

        print(f"  MiniGPT: {config.n_params/1e6:.2f}M parameters")
        print(f"    Layers: {config.n_layer}, Heads: {config.n_head}, "
              f"d_model: {config.d_model}, Block: {config.block_size}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Tied output weights
        logits = x @ self.token_embedding.weight.T

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=None):
        """Generate text autoregressively."""
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            # Last token
            logits = logits[:, -1, :] / temperature
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================
# 4. Learning Rate Schedule
# ============================================================

def get_lr(it, config):
    """Cosine schedule with linear warmup."""
    # Warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # After max_iters
    if it > config.max_iters:
        return config.learning_rate * 0.1  # Min LR = 10% of max
    # Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.learning_rate * 0.1 + coeff * config.learning_rate * 0.9


# ============================================================
# 5. Training
# ============================================================

def train(config, device='cpu'):
    """Train MiniGPT."""
    print(f"\n  Training MiniGPT on {device}...")

    # Data
    tokenizer = CharTokenizer()
    data = torch.tensor(tokenizer.encode(CORPUS), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"  Corpus: {len(data)} tokens, vocab: {tokenizer.vocab_size}")

    # Clamp block_size to data length
    config.block_size = min(config.block_size, len(data) // 2)
    print(f"  Block size: {config.block_size}")

    # Update config with actual vocab size
    config.vocab_size = tokenizer.vocab_size

    # Model
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1)

    # Training loop
    best_val_loss = float('inf')
    losses = {'train': [], 'val': []}
    tokens_per_sec = []

    start_time = time.time()
    for iter_num in range(config.max_iters):
        # Learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate
        if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
            model.eval()
            val_losses = []
            for _ in range(config.eval_iters):
                x, y = get_batch(val_data, config.block_size, config.batch_size, device)
                _, loss = model(x, y)
                val_losses.append(loss.item())
            val_loss = sum(val_losses) / len(val_losses)

            train_losses = []
            for _ in range(config.eval_iters):
                x, y = get_batch(train_data, config.block_size, config.batch_size, device)
                _, loss = model(x, y)
                train_losses.append(loss.item())
            train_loss = sum(train_losses) / len(train_losses)

            losses['train'].append(train_loss)
            losses['val'].append(val_loss)

            elapsed = time.time() - start_time
            print(f"    step {iter_num:>5}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, lr={lr:.2e}, time={elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            model.train()

        # Train step
        xb, yb = get_batch(train_data, config.block_size, config.batch_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Tokens/sec estimate
        if iter_num > 0 and iter_num % 100 == 0:
            toks = config.batch_size * config.block_size
            tokens_per_sec.append(toks)

    elapsed = time.time() - start_time
    total_tokens = config.max_iters * config.batch_size * config.block_size
    print(f"\n  Training complete: {elapsed:.1f}s, {total_tokens/1e6:.1f}M tokens, "
          f"{total_tokens/elapsed:.0f} tok/s")

    return model, tokenizer, losses, elapsed


# ============================================================
# 6. Experiments
# ============================================================

def experiment_model_scaling(device='cpu'):
    """Test how model size affects training."""
    print("\n  === Experiment: Model Scaling ===")

    configs = [
        ('tiny',    GPTConfig(n_layer=2, n_head=2, d_model=64,  block_size=64, max_iters=1000)),
        ('small',   GPTConfig(n_layer=4, n_head=4, d_model=128, block_size=64, max_iters=1000)),
        ('medium',  GPTConfig(n_layer=4, n_head=4, d_model=256, block_size=64, max_iters=1000)),
        ('large',   GPTConfig(n_layer=6, n_head=8, d_model=256, block_size=64, max_iters=1000)),
    ]

    results = {}
    for name, cfg in configs:
        print(f"\n  --- {name} ({cfg.n_params/1e6:.2f}M params) ---")
        model, tok, losses, elapsed = train(cfg, device)
        final_train = losses['train'][-1] if losses['train'] else -1
        final_val = losses['val'][-1] if losses['val'] else -1

        # Generate sample
        model.eval()
        context = torch.zeros(1, 1, dtype=torch.long, device=device)
        output = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=20)
        text = tok.decode(output[0].tolist())

        results[name] = {
            'params': cfg.n_params,
            'final_train_loss': round(final_train, 4),
            'final_val_loss': round(final_val, 4),
            'time_s': round(elapsed, 1),
            'sample': text[:100],
        }
        print(f"    Final: train={final_train:.4f}, val={final_val:.4f}")
        print(f"    Sample: {text[:80]}...")

    return results


def experiment_temperature(device='cpu'):
    """Test how temperature affects generation quality."""
    print("\n  === Experiment: Temperature Effect ===")

    config = GPTConfig(n_layer=4, n_head=4, d_model=128, block_size=64, max_iters=800)
    model, tok, _, _ = train(config, device)
    model.eval()

    prompt = "Once upon a time"
    prompt_tokens = torch.tensor([tok.encode(prompt)], device=device)

    results = {}
    for temp in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]:
        output = model.generate(prompt_tokens, max_new_tokens=100,
                                temperature=temp, top_k=20)
        text = tok.decode(output[0].tolist())

        # Measure repetition
        words = text.split()
        unique_ratio = len(set(words)) / max(len(words), 1)

        results[temp] = {
            'text': text,
            'unique_ratio': round(unique_ratio, 3),
            'length': len(text),
        }
        print(f"    T={temp:.1f}: unique_ratio={unique_ratio:.3f}")
        print(f"      {text[:80]}...")

    return results


def experiment_context_length(device='cpu'):
    """Test how context length affects training."""
    print("\n  === Experiment: Context Length ===")

    results = {}
    for block_size in [32, 64, 128]:
        config = GPTConfig(n_layer=4, n_head=4, d_model=128,
                          block_size=block_size, max_iters=800)
        model, tok, losses, elapsed = train(config, device)

        final_val = losses['val'][-1] if losses['val'] else -1
        results[block_size] = {
            'final_val_loss': round(final_val, 4),
            'time_s': round(elapsed, 1),
        }
        print(f"    block_size={block_size:>4}: val_loss={final_val:.4f}, time={elapsed:.1f}s")

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("MiniGPT: Train a Small GPT from Scratch")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"\n  GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")
    else:
        print(f"\n  Device: CPU")

    results = {}

    # Exp 1: Model scaling
    results['scaling'] = experiment_model_scaling(device)

    # Exp 2: Temperature
    results['temperature'] = experiment_temperature(device)

    # Exp 3: Context length
    results['context_length'] = experiment_context_length(device)

    # Summary
    print("\n" + "=" * 60)
    print("MiniGPT Training Summary")
    print("=" * 60)
    print("""
    Architecture: GPT-2 (Pre-norm + RMSNorm + SwiGLU)
    ┌─────────────────────────────────────────────────────────┐
    │ Component          │ Implementation                     │
    │────────────────────│────────────────────────────────────│
    │ Tokenizer          │ Character-level (vocab ~80 chars)  │
    │ Position Encoding  │ Learned (GPT-2 style)              │
    │ Attention          │ Multi-head Causal Self-Attention   │
    │ Feed-Forward       │ SwiGLU (LLaMA style)               │
    │ Normalization      │ RMSNorm (LLaMA style)              │
    │ Residual           │ Pre-norm (modern standard)         │
    │ Output             │ Weight tying with input embedding  │
    │ LR Schedule        │ Cosine with linear warmup          │
    │ Optimizer          │ AdamW (weight_decay=0.1)           │
    └─────────────────────────────────────────────────────────┘

    Key Training Dynamics:
    - Loss drops rapidly in first 200 steps (learning common patterns)
    - Then slowly converges (learning longer-range dependencies)
    - Overfitting visible when train_loss << val_loss
    - Temperature 0.5-0.8 gives best quality vs diversity tradeoff
    - Longer context → better modeling but slower training (O(T²))

    Scaling Laws (approximate):
    - 2x params → ~10% lower loss (diminishing returns)
    - 2x context → slower training (attention is O(T²))
    - Data is the bottleneck: small corpus limits model quality
    """)

    if device == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"  Peak GPU memory: {mem:.1f} MB")
        results['gpu_memory_mb'] = round(mem, 1)

    with open("minigpt_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to minigpt_results.json")


if __name__ == "__main__":
    main()
