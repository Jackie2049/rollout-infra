#!/usr/bin/env python3
"""MiniGPT 训练实战 — 模式学习 + 性能分析"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch, torch.nn as nn, torch.nn.functional as F, time, json, math
from torch.cuda.amp import autocast, GradScaler

print(f"GPU: {torch.cuda.get_device_name(0)}")

class CharLM(nn.Module):
    def __init__(self, vocab=128, hidden=256, n_layers=4, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.pos = nn.Embedding(256, hidden)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(hidden),
                'attn': nn.MultiheadAttention(hidden, n_heads, batch_first=True),
                'ln2': nn.LayerNorm(hidden),
                'fc1': nn.Linear(hidden, 4*hidden),
                'fc2': nn.Linear(4*hidden, hidden),
            }))
        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x):
        B, S = x.shape
        h = self.emb(x) + self.pos(torch.arange(S, device=x.device))
        for block in self.blocks:
            residual = h
            h = block['ln1'](h)
            mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
            h2, _ = block['attn'](h, h, h, need_weights=False, attn_mask=mask)
            h = residual + h2
            residual = h
            h = block['ln2'](h)
            h = residual + block['fc2'](F.gelu(block['fc1'](h)))
        return self.head(self.ln_f(h))


def make_data(batch, seq, pattern_len=8):
    data = []
    for b in range(batch):
        pattern = torch.randint(10, 90, (pattern_len,))
        seq_data = pattern.repeat(seq // pattern_len + 1)[:seq]
        noise = torch.randint(0, 3, (seq,))
        data.append((seq_data + noise).clamp(0, 127))
    return torch.stack(data)


model = CharLM().cuda()
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.1f}M params")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
scaler = GradScaler()

results = []
print(f"\n{'Step':<6} {'Loss':<10} {'ms':<8} {'tok/s':<10} {'Peak MB':<10} {'LR':<12}")
print("-" * 56)

t0 = time.time()
for step in range(500):
    x = make_data(32, 128).cuda()
    optimizer.zero_grad()

    with autocast():
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 128), x[:, 1:].reshape(-1))

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    if step % 50 == 0 or step == 499:
        now = time.time()
        peak = torch.cuda.max_memory_allocated() / 1e6
        lr = scheduler.get_last_lr()[0]
        elapsed = (now - t0) * 1000
        throughput = 32 * 128 * 50 / (now - t0) if step > 0 else 0
        print(f"{step:<6} {loss.item():<10.4f} {elapsed if step == 0 else elapsed/50:<8.1f} {throughput:<10.0f} {peak:<10.0f} {lr:<12.6f}")
        results.append({
            "step": step, "loss": round(loss.item(), 4),
            "peak_mb": round(peak, 1), "lr": round(lr, 6),
        })
        t0 = time.time()

# Evaluate
print("\n=== Evaluation ===")
model.eval()
x = make_data(4, 128).cuda()
with torch.no_grad(), autocast():
    logits = model(x)
    preds = logits.argmax(-1)
    correct = (preds[:, :-1] == x[:, 1:]).float().mean().item()
    print(f"Next-token accuracy: {correct*100:.1f}%")

for pl in [4, 8, 16, 32]:
    x = make_data(4, 128, pattern_len=pl).cuda()
    with torch.no_grad(), autocast():
        logits = model(x)
        preds = logits.argmax(-1)
        acc = (preds[:, :-1] == x[:, 1:]).float().mean().item()
        print(f"  pattern_len={pl}: accuracy={acc*100:.1f}%")

with open("/root/mini_train_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nDone!")
