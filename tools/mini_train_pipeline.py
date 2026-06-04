#!/usr/bin/env python3
"""Mini LLM Training Pipeline — Pretraining + SFT
===================================================
Demonstrates the complete LLM training lifecycle:
1. Pretraining: Next-token prediction on raw text
2. SFT (Supervised Fine-Tuning): Instruction following
3. Evaluation: Loss curves, generation quality

Runs on GPU (RTX 4090) with a ~3M parameter GPT-like model.
Uses synthetic data — no model download needed.

Educational purpose: understand training infra from first principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import os


# ============================================================
# Model: GPT-like Transformer
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Flash Attention (if available) or manual
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size=256, block_size=256, n_embd=128,
                 n_head=4, n_layer=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying: share embedding and output weights
        self.tok_emb.weight = self.head.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Data: Synthetic Pretraining + SFT Data
# ============================================================

def create_pretrain_data(vocab_size=256, seq_len=128, num_sequences=2000, device='cpu'):
    """Create synthetic pretraining data (character-level)."""
    # Simple patterns: repeating sequences, arithmetic, etc.
    data = []
    for _ in range(num_sequences):
        # Mix of patterns
        pattern_type = torch.randint(0, 4, (1,)).item()
        if pattern_type == 0:
            # Repeating pattern
            pattern_len = torch.randint(4, 16, (1,)).item()
            pattern = torch.randint(10, vocab_size // 4, (pattern_len,))
            repeats = seq_len // pattern_len + 1
            seq = pattern.repeat(repeats)[:seq_len]
        elif pattern_type == 1:
            # Counting sequence
            start = torch.randint(0, vocab_size // 4, (1,)).item()
            step = torch.randint(1, 5, (1,)).item()
            seq = torch.tensor([(start + i * step) % (vocab_size // 2) for i in range(seq_len)])
        elif pattern_type == 2:
            # Fibonacci-like
            a, b = torch.randint(1, 20, (2,)).tolist()
            fib = [a % (vocab_size // 2), b % (vocab_size // 2)]
            for i in range(seq_len - 2):
                fib.append((fib[-1] + fib[-2]) % (vocab_size // 2))
            seq = torch.tensor(fib)
        else:
            # Random with bias towards some tokens
            bias = torch.zeros(vocab_size)
            center = torch.randint(10, vocab_size // 2, (1,)).item()
            bias[max(0, center - 20):center + 20] = 1.0
            bias = bias / bias.sum()
            seq = torch.multinomial(bias, seq_len, replacement=True)
        data.append(seq)
    return torch.stack(data).to(device)


def create_sft_data(vocab_size=256, seq_len=64, num_pairs=500, device='cpu'):
    """Create synthetic SFT data (instruction → response)."""
    # Special tokens
    INST_START = vocab_size - 4
    INST_END = vocab_size - 3
    RESP_START = vocab_size - 2
    RESP_END = vocab_size - 1

    pairs = []
    for _ in range(num_pairs):
        # Instruction: random tokens
        inst_len = torch.randint(8, 24, (1,)).item()
        instruction = torch.randint(10, vocab_size // 2, (inst_len,))

        # Response: patterned (reversal or doubling)
        resp_type = torch.randint(0, 2, (1,)).item()
        if resp_type == 0:
            response = instruction.flip(0) % (vocab_size // 2)  # Reverse
        else:
            response = (instruction * 2) % (vocab_size // 2)  # Double

        # Format: [INST_START] instruction [INST_END] [RESP_START] response [RESP_END]
        full = torch.cat([
            torch.tensor([INST_START]),
            instruction,
            torch.tensor([INST_END, RESP_START]),
            response,
            torch.tensor([RESP_END]),
        ])
        # Pad or truncate to seq_len
        if len(full) > seq_len:
            full = full[:seq_len]
        else:
            full = torch.cat([full, torch.zeros(seq_len - len(full), dtype=torch.long)])
        pairs.append(full)

    return torch.stack(pairs).to(device), INST_START, INST_END, RESP_START, RESP_END


# ============================================================
# Training Loop
# ============================================================

def train(model, data, epochs, batch_size, lr, max_grad_norm=1.0, label="train"):
    """Standard training loop with gradient clipping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * (len(data) // batch_size))
    model.train()

    losses = []
    tokens_per_sec = []
    n_params = model.count_parameters()

    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(len(data))
        data_shuffled = data[perm]
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(data) - batch_size, batch_size):
            batch = data_shuffled[i:i + batch_size]
            # Input: all tokens except last, Target: all tokens except first
            x = batch[:, :-1]
            y = batch[:, 1:]

            start = time.time()
            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            elapsed = time.time() - start
            n_tokens = x.numel()
            tps = n_tokens / elapsed

            epoch_loss += loss.item()
            n_batches += 1
            tokens_per_sec.append(tps)

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            avg_tps = sum(tokens_per_sec[-n_batches:]) / max(n_batches, 1)
            lr_now = scheduler.get_last_lr()[0]
            print(f"  [{label}] Epoch {epoch:>3}/{epochs}: loss={avg_loss:.4f}, "
                  f"lr={lr_now:.2e}, throughput={avg_tps:,.0f} tok/s")

    return losses


# ============================================================
# Main: Full Pipeline
# ============================================================

def main():
    print("=" * 60)
    print("Mini LLM Training Pipeline — Pretraining + SFT")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"\n  GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")
    else:
        print(f"\n  Device: CPU (will be slow)")

    results = {}

    # Model config
    vocab_size = 256
    block_size = 128
    n_embd = 128
    n_head = 4
    n_layer = 4

    # ===== Phase 1: Pretraining =====
    print("\n" + "=" * 60)
    print("Phase 1: Pretraining (Next-Token Prediction)")
    print("=" * 60)

    model = MiniGPT(vocab_size, block_size, n_embd, n_head, n_layer, dropout=0.1).to(device)
    n_params = model.count_parameters()
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"\n  Model: {n_params:,} parameters ({model_size_mb:.1f} MB)")
    print(f"  Architecture: {n_layer} layers, {n_head} heads, d_model={n_embd}")

    # Create data
    pt_data = create_pretrain_data(vocab_size, seq_len=block_size, num_sequences=2000, device=device)
    print(f"  Pretrain data: {pt_data.shape[0]} sequences, {pt_data.shape[1]} tokens each")

    # Train
    print("\n  --- Pretraining ---")
    pt_losses = train(model, pt_data, epochs=30, batch_size=32, lr=3e-4, label="pretrain")

    # Test generation after pretraining
    print("\n  Generation sample (after pretraining):")
    model.eval()
    ctx = torch.randint(10, vocab_size // 2, (1, 8), device=device)
    sample = model.generate(ctx, max_new_tokens=16, temperature=0.8, top_k=40)
    print(f"    Input:  {ctx[0, :8].tolist()}")
    print(f"    Output: {sample[0].tolist()}")

    results["pretrain"] = {
        "params": n_params,
        "model_size_mb": round(model_size_mb, 1),
        "final_loss": round(pt_losses[-1], 4),
        "initial_loss": round(pt_losses[0], 4),
        "loss_reduction": round(pt_losses[0] - pt_losses[-1], 4),
    }

    # ===== Phase 2: SFT =====
    print("\n" + "=" * 60)
    print("Phase 2: SFT (Supervised Fine-Tuning)")
    print("=" * 60)

    sft_data, INST_START, INST_END, RESP_START, RESP_END = \
        create_sft_data(vocab_size, seq_len=block_size, num_pairs=500, device=device)
    print(f"  SFT data: {sft_data.shape[0]} instruction-response pairs")
    print(f"  Special tokens: INST_START={INST_START}, INST_END={INST_END}, RESP_START={RESP_START}, RESP_END={RESP_END}")

    # Fine-tune with lower LR
    print("\n  --- SFT Training ---")
    sft_losses = train(model, sft_data, epochs=20, batch_size=16, lr=1e-4, label="sft")

    # Test instruction following
    print("\n  Instruction following test:")
    model.eval()
    test_inst = torch.randint(10, vocab_size // 2, (1, 6), device=device)
    prompt = torch.cat([
        torch.tensor([[INST_START]], device=device),
        test_inst,
        torch.tensor([[INST_END, RESP_START]], device=device),
    ], dim=1)
    response = model.generate(prompt, max_new_tokens=12, temperature=0.5, top_k=20)
    print(f"    Instruction: {test_inst[0].tolist()}")
    print(f"    Response:    {response[0, prompt.shape[1]:].tolist()}")
    # Expected: reverse of instruction
    expected = test_inst[0].flip(0).tolist()
    actual = response[0, prompt.shape[1]:prompt.shape[1] + len(expected)].tolist()
    match = sum(a == e for a, e in zip(actual, expected))
    print(f"    Expected:    {expected}")
    print(f"    Match: {match}/{len(expected)} tokens correct")

    results["sft"] = {
        "final_loss": round(sft_losses[-1], 4),
        "initial_loss": round(sft_losses[0], 4),
        "instruction_match": f"{match}/{len(expected)}",
    }

    # ===== Phase 3: Analysis =====
    print("\n" + "=" * 60)
    print("Phase 3: Training Analysis")
    print("=" * 60)

    # Memory analysis
    if device == 'cuda':
        alloc_mem = torch.cuda.max_memory_allocated() / 1e6
        reserved_mem = torch.cuda.max_memory_reserved() / 1e6
        print(f"\n  Peak GPU memory: {alloc_mem:.1f} MB allocated, {reserved_mem:.1f} MB reserved")
        results["gpu_memory_mb"] = round(alloc_mem, 1)

    # Parameter breakdown
    total = n_params
    emb_params = model.tok_emb.weight.numel() + model.pos_emb.weight.numel()
    attn_params = sum(p.numel() for name, p in model.named_parameters() if 'attn' in name or 'qkv' in name)
    mlp_params = sum(p.numel() for name, p in model.named_parameters() if 'mlp' in name)
    ln_params = sum(p.numel() for name, p in model.named_parameters() if 'ln' in name)

    print(f"\n  Parameter breakdown:")
    print(f"    Embedding:     {emb_params:>8,} ({emb_params/total*100:.1f}%)")
    print(f"    Attention:     {attn_params:>8,} ({attn_params/total*100:.1f}%)")
    print(f"    MLP:           {mlp_params:>8,} ({mlp_params/total*100:.1f}%)")
    print(f"    LayerNorm:     {ln_params:>8,} ({ln_params/total*100:.1f}%)")
    print(f"    Total:         {total:>8,}")

    # Training compute estimate
    # FLOPS per token ≈ 6 * n_params (forward + backward)
    total_tokens = pt_data.numel() * 30 + sft_data.numel() * 20  # epochs * data
    total_flops = 6 * n_params * total_tokens
    print(f"\n  Training compute:")
    print(f"    Total tokens processed: {total_tokens:,}")
    print(f"    Total FLOPS: {total_flops:.2e}")
    print(f"    ≈ {total_flops / 1e15:.2f} PFLOPS")

    # Scaling projections
    print(f"\n  Scaling projections (same architecture ratio):")
    for name, scale_params in [("125M", 125e6), ("1.3B", 1.3e9), ("7B", 7e9)]:
        scale = scale_params / n_params
        scale_flops = total_flops * (scale ** 2)  # FLOPS scale as params²
        print(f"    {name}: {scale_flops:.2e} FLOPS ({scale_flops/1e18:.2f} EFLOPS)")

    with open("mini_train_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to mini_train_results.json")


if __name__ == "__main__":
    main()
