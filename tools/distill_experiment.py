#!/usr/bin/env python3
"""Knowledge Distillation Experiment
======================================
Demonstrates knowledge distillation from teacher to student model:
1. Teacher: Larger model (e.g., 4 layers, 4 heads, d=256)
2. Student: Smaller model (e.g., 2 layers, 2 heads, d=128)
3. Distillation: Student learns from teacher's soft labels (KL divergence)
4. Comparison: Student-from-scratch vs Distilled-student

Key concepts:
- Soft labels: Teacher's output probability distribution (richer than hard labels)
- Temperature: Softmax temperature to soften probability distribution
- KL Loss: KL divergence between teacher and student distributions
- Combined Loss: α * KL_loss + (1-α) * CE_loss (distillation + hard label)

Runs on GPU (RTX 4090) with synthetic data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import copy


# ============================================================
# Model: Reuse MiniGPT architecture (variable size)
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
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
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
        self.n_embd = n_embd
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
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

    def get_logits(self, idx):
        """Get raw logits (for distillation)."""
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

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
# Distillation Loss
# ============================================================

def distillation_loss(student_logits, teacher_logits, targets,
                      temperature=4.0, alpha=0.7):
    """Combined distillation loss.

    L = α * T² * KL(softmax(teacher/T) || softmax(student/T))
      + (1-α) * CE(student, hard_labels)

    Args:
        student_logits: [B, T, V] student model logits
        teacher_logits: [B, T, V] teacher model logits
        targets: [B, T] ground truth token IDs
        temperature: softmax temperature (higher = softer distribution)
        alpha: weight for KL loss vs CE loss

    Returns:
        total_loss, kl_loss, ce_loss
    """
    # Soft target loss: KL divergence between softened distributions
    # Use the last T positions (student might have shorter context)
    T = student_logits.shape[1]
    student_logits_flat = student_logits.reshape(-1, student_logits.size(-1))
    teacher_logits_flat = teacher_logits[:, -T:, :].reshape(-1, teacher_logits.size(-1))
    targets_flat = targets.reshape(-1)

    # Softened probabilities
    student_soft = F.log_softmax(student_logits_flat / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits_flat / temperature, dim=-1)

    # KL divergence: sum over vocab, mean over batch
    kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

    # Hard target loss: standard cross-entropy
    ce_loss = F.cross_entropy(student_logits_flat, targets_flat)

    # Combined loss (multiply KL by T² as in Hinton et al. 2015)
    total_loss = alpha * (temperature ** 2) * kl_loss + (1 - alpha) * ce_loss

    return total_loss, kl_loss.item(), ce_loss.item()


# ============================================================
# Training Functions
# ============================================================

def train_teacher(model, data, epochs=30, batch_size=32, lr=3e-4, device='cpu'):
    """Train teacher model with standard cross-entropy."""
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
            x = batch[:, :-1]
            y = batch[:, 1:]
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"  [teacher] Epoch {epoch:>3}/{epochs}: loss={avg_loss:.4f}")

    return losses


def train_student_baseline(model, data, epochs=30, batch_size=32, lr=3e-4, device='cpu'):
    """Train student model from scratch (no teacher)."""
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
            x = batch[:, :-1]
            y = batch[:, 1:]
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"  [student-baseline] Epoch {epoch:>3}/{epochs}: loss={avg_loss:.4f}")

    return losses


def train_student_distilled(student, teacher, data, epochs=30, batch_size=32,
                            lr=3e-4, temperature=4.0, alpha=0.7, device='cpu'):
    """Train student model using knowledge distillation."""
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * (len(data) // batch_size))
    teacher.eval()
    student.train()

    losses = {'total': [], 'kl': [], 'ce': []}

    for epoch in range(epochs):
        perm = torch.randperm(len(data))
        data_shuffled = data[perm]
        epoch_total = 0
        epoch_kl = 0
        epoch_ce = 0
        n_batches = 0

        start = time.time()

        for i in range(0, len(data) - batch_size, batch_size):
            batch = data_shuffled[i:i + batch_size]
            x = batch[:, :-1]
            y = batch[:, 1:]

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = teacher.get_logits(x)

            # Student forward
            student_logits = student.get_logits(x)

            # Distillation loss
            total_loss, kl_loss, ce_loss = distillation_loss(
                student_logits, teacher_logits, y,
                temperature=temperature, alpha=alpha
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_total += total_loss.item()
            epoch_kl += kl_loss
            epoch_ce += ce_loss
            n_batches += 1

        elapsed = time.time() - start
        avg_total = epoch_total / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)
        avg_ce = epoch_ce / max(n_batches, 1)

        losses['total'].append(avg_total)
        losses['kl'].append(avg_kl)
        losses['ce'].append(avg_ce)

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            tps = (len(data) * data.shape[1]) / elapsed
            print(f"  [distilled] Epoch {epoch:>3}/{epochs}: total={avg_total:.4f}, "
                  f"kl={avg_kl:.4f}, ce={avg_ce:.4f}, tps={tps:,.0f}")

    return losses


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, data, n_samples=200, device='cpu'):
    """Evaluate model perplexity on test data."""
    model.eval()
    total_loss = 0
    n = 0

    with torch.no_grad():
        for i in range(0, min(n_samples, len(data)), 32):
            batch = data[i:i + 32]
            x = batch[:, :-1]
            y = batch[:, 1:]
            _, loss = model(x, y)
            total_loss += loss.item() * len(batch)
            n += len(batch)

    avg_loss = total_loss / max(n, 1)
    perplexity = math.exp(min(avg_loss, 20))  # Clamp to prevent overflow
    return avg_loss, perplexity


def evaluate_topk_agreement(student, teacher, data, k=5, n_samples=200, device='cpu'):
    """Evaluate how often student's top-k matches teacher's top-k."""
    student.eval()
    teacher.eval()
    agreements = 0
    total = 0

    with torch.no_grad():
        for i in range(0, min(n_samples, len(data)), 32):
            batch = data[i:i + 32]
            x = batch[:, :-1]

            teacher_logits = teacher.get_logits(x)[:, -1, :]
            student_logits = student.get_logits(x)[:, -1, :]

            teacher_topk = teacher_logits.topk(k, dim=-1).indices
            student_topk = student_logits.topk(k, dim=-1).indices

            # Check overlap
            for b in range(teacher_topk.shape[0]):
                overlap = len(set(teacher_topk[b].tolist()) & set(student_topk[b].tolist()))
                agreements += overlap
                total += k

    return agreements / max(total, 1)


# ============================================================
# Data
# ============================================================

def create_training_data(vocab_size=256, seq_len=128, num_sequences=2000, device='cpu'):
    """Create synthetic training data with learnable patterns."""
    data = []
    for _ in range(num_sequences):
        pattern_type = torch.randint(0, 4, (1,)).item()
        if pattern_type == 0:
            pattern_len = torch.randint(4, 16, (1,)).item()
            pattern = torch.randint(10, vocab_size // 4, (pattern_len,))
            repeats = seq_len // pattern_len + 1
            seq = pattern.repeat(repeats)[:seq_len]
        elif pattern_type == 1:
            start = torch.randint(0, vocab_size // 4, (1,)).item()
            step = torch.randint(1, 5, (1,)).item()
            seq = torch.tensor([(start + i * step) % (vocab_size // 2) for i in range(seq_len)])
        elif pattern_type == 2:
            a, b = torch.randint(1, 20, (2,)).tolist()
            fib = [a % (vocab_size // 2), b % (vocab_size // 2)]
            for i in range(seq_len - 2):
                fib.append((fib[-1] + fib[-2]) % (vocab_size // 2))
            seq = torch.tensor(fib)
        else:
            bias = torch.zeros(vocab_size)
            center = torch.randint(10, vocab_size // 2, (1,)).item()
            bias[max(0, center - 20):center + 20] = 1.0
            bias = bias / bias.sum()
            seq = torch.multinomial(bias, seq_len, replacement=True)
        data.append(seq)
    return torch.stack(data).to(device)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Knowledge Distillation Experiment")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"\n  GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")
    else:
        print(f"\n  Device: CPU")

    results = {}
    vocab_size = 256
    block_size = 128

    # Create data
    data = create_training_data(vocab_size, seq_len=block_size, num_sequences=2000, device=device)
    # Split train/test
    train_data = data[:1800]
    test_data = data[1800:]
    print(f"  Data: {len(train_data)} train, {len(test_data)} test sequences")

    # ===== Teacher Model (larger) =====
    print("\n" + "=" * 60)
    print("Step 1: Train Teacher Model")
    print("=" * 60)

    teacher = MiniGPT(vocab_size, block_size, n_embd=256, n_head=8,
                      n_layer=6, dropout=0.1).to(device)
    teacher_params = teacher.count_parameters()
    print(f"\n  Teacher: {teacher_params:,} parameters (6 layers, 8 heads, d=256)")

    teacher_losses = train_teacher(teacher, train_data, epochs=30, batch_size=32, lr=3e-4, device=device)

    teacher_loss, teacher_ppl = evaluate_model(teacher, test_data, device=device)
    print(f"\n  Teacher evaluation: loss={teacher_loss:.4f}, ppl={teacher_ppl:.2f}")

    results["teacher"] = {
        "params": teacher_params,
        "train_loss": round(teacher_losses[-1], 4),
        "test_loss": round(teacher_loss, 4),
        "test_ppl": round(teacher_ppl, 2),
    }

    # ===== Student Baseline (from scratch) =====
    print("\n" + "=" * 60)
    print("Step 2: Train Student from Scratch (Baseline)")
    print("=" * 60)

    student_baseline = MiniGPT(vocab_size, block_size, n_embd=128, n_head=4,
                               n_layer=2, dropout=0.1).to(device)
    student_params = student_baseline.count_parameters()
    compression_ratio = teacher_params / student_params
    print(f"\n  Student: {student_params:,} parameters (2 layers, 4 heads, d=128)")
    print(f"  Compression: {compression_ratio:.1f}x")

    baseline_losses = train_student_baseline(
        student_baseline, train_data, epochs=30, batch_size=32, lr=3e-4, device=device
    )

    baseline_loss, baseline_ppl = evaluate_model(student_baseline, test_data, device=device)
    baseline_agreement = evaluate_topk_agreement(student_baseline, teacher, test_data, device=device)
    print(f"\n  Baseline evaluation: loss={baseline_loss:.4f}, ppl={baseline_ppl:.2f}, "
          f"top-5 agreement={baseline_agreement:.3f}")

    results["student_baseline"] = {
        "params": student_params,
        "compression_ratio": round(compression_ratio, 1),
        "train_loss": round(baseline_losses[-1], 4),
        "test_loss": round(baseline_loss, 4),
        "test_ppl": round(baseline_ppl, 2),
        "top5_agreement": round(baseline_agreement, 3),
    }

    # ===== Distilled Student =====
    print("\n" + "=" * 60)
    print("Step 3: Train Student with Knowledge Distillation")
    print("=" * 60)

    student_distilled = MiniGPT(vocab_size, block_size, n_embd=128, n_head=4,
                                n_layer=2, dropout=0.1).to(device)
    print(f"\n  Student: {student_distilled.count_parameters():,} parameters")
    print(f"  Temperature: 4.0, α=0.7 (KL weight)")

    distill_losses = train_student_distilled(
        student_distilled, teacher, train_data,
        epochs=30, batch_size=32, lr=3e-4,
        temperature=4.0, alpha=0.7, device=device
    )

    distill_loss, distill_ppl = evaluate_model(student_distilled, test_data, device=device)
    distill_agreement = evaluate_topk_agreement(student_distilled, teacher, test_data, device=device)
    print(f"\n  Distilled evaluation: loss={distill_loss:.4f}, ppl={distill_ppl:.2f}, "
          f"top-5 agreement={distill_agreement:.3f}")

    results["student_distilled"] = {
        "train_loss": round(distill_losses['total'][-1], 4),
        "test_loss": round(distill_loss, 4),
        "test_ppl": round(distill_ppl, 2),
        "top5_agreement": round(distill_agreement, 3),
    }

    # ===== Experiment: Temperature Sweep =====
    print("\n" + "=" * 60)
    print("Experiment: Temperature Sweep")
    print("=" * 60)

    temp_results = {}
    for temp in [1.0, 2.0, 4.0, 8.0, 16.0]:
        student_t = MiniGPT(vocab_size, block_size, n_embd=128, n_head=4,
                            n_layer=2, dropout=0.1).to(device)
        losses_t = train_student_distilled(
            student_t, teacher, train_data,
            epochs=20, batch_size=32, lr=3e-4,
            temperature=temp, alpha=0.7, device=device
        )
        loss_t, ppl_t = evaluate_model(student_t, test_data, device=device)
        agree_t = evaluate_topk_agreement(student_t, teacher, test_data, device=device)
        temp_results[temp] = {
            "loss": round(loss_t, 4),
            "ppl": round(ppl_t, 2),
            "agreement": round(agree_t, 3),
        }
        print(f"  T={temp:>5.1f}: loss={loss_t:.4f}, ppl={ppl_t:.2f}, agreement={agree_t:.3f}")

    results["temperature_sweep"] = temp_results

    # ===== Experiment: Alpha Sweep =====
    print("\n" + "=" * 60)
    print("Experiment: Alpha Sweep (KL vs CE weight)")
    print("=" * 60)

    alpha_results = {}
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        student_a = MiniGPT(vocab_size, block_size, n_embd=128, n_head=4,
                            n_layer=2, dropout=0.1).to(device)
        losses_a = train_student_distilled(
            student_a, teacher, train_data,
            epochs=20, batch_size=32, lr=3e-4,
            temperature=4.0, alpha=alpha, device=device
        )
        loss_a, ppl_a = evaluate_model(student_a, test_data, device=device)
        alpha_results[str(alpha)] = {
            "loss": round(loss_a, 4),
            "ppl": round(ppl_a, 2),
        }
        print(f"  α={alpha:.1f}: loss={loss_a:.4f}, ppl={ppl_a:.2f}")

    results["alpha_sweep"] = alpha_results

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("Summary: Distillation vs Baseline vs Teacher")
    print("=" * 60)

    print(f"\n  {'Model':<25} {'Params':>10} {'Loss':>8} {'PPL':>8} {'Top-5 Agr':>12}")
    print("  " + "-" * 65)
    print(f"  {'Teacher':<25} {teacher_params:>10,} {teacher_loss:>8.4f} {teacher_ppl:>8.2f} {'—':>12}")
    print(f"  {'Student (baseline)':<25} {student_params:>10,} {baseline_loss:>8.4f} {baseline_ppl:>8.2f} {baseline_agreement:>12.3f}")
    print(f"  {'Student (distilled)':<25} {student_params:>10,} {distill_loss:>8.4f} {distill_ppl:>8.2f} {distill_agreement:>12.3f}")

    improvement = (baseline_loss - distill_loss) / baseline_loss * 100
    ppl_improvement = (baseline_ppl - distill_ppl) / baseline_ppl * 100
    print(f"\n  Distillation improvement:")
    print(f"    Loss: {improvement:+.1f}% (lower is better)")
    print(f"    PPL: {ppl_improvement:+.1f}% (lower is better)")
    print(f"    Top-5 agreement: {distill_agreement - baseline_agreement:+.3f}")

    # GPU memory
    if device == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"\n  Peak GPU memory: {mem:.1f} MB")
        results["gpu_memory_mb"] = round(mem, 1)

    with open("distill_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to distill_results.json")


if __name__ == "__main__":
    main()
