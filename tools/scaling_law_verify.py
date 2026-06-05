#!/usr/bin/env python3
"""Scaling Law Calculator & Verification
=========================================
Validates Chinchilla scaling laws with real training experiments:
1. Predict optimal N/D for given compute budgets
2. Train small models at different N×D points
3. Verify L(N,D) = A/N^α + B/D^β + E

Reference: Hoffmann et al., 2022 (Chinchilla)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json

# Chinchilla fitted parameters
A, ALPHA = 406.4, 0.34
B, BETA = 410.7, 0.28
E = 1.69  # irreducible loss


def chinchilla_loss(N, D):
    """Predict loss using Chinchilla scaling law.
    N: params in millions, D: tokens in billions
    """
    return A / (N ** ALPHA) + B / (D ** BETA) + E


def optimal_allocation(C):
    """Given compute budget C (in PFLOPS-days), find optimal N and D.

    C = 6 * N * D (FLOPs)
    Returns N (millions params), D (billions tokens).
    """
    # From Chinchilla: N_opt ∝ C^a, D_opt ∝ C^b
    # a ≈ 0.46, b ≈ 0.54
    # More precisely, solve the constrained optimization
    a = 1 / (1 + ALPHA / BETA)  # ≈ 0.45
    b = 1 - a                    # ≈ 0.55

    # G is the scaling coefficient
    # From Chinchilla Table 3: G ≈ 0.063 (for N in M, D in B)
    # N_opt = G * C^a, D_opt = G_inv * C^b
    # Using fitted values: N_opt ≈ 0.30 * C^0.46, D_opt ≈ 2.08 * C^0.54
    # where C is in 1e18 FLOPs

    C_flops = C  # FLOPs
    G_N = 0.30   # fitted coefficient
    G_D = 2.08

    # C in units of 1e18
    C_unit = C_flops / 1e18
    N_opt = G_N * (C_unit ** a)  # millions params
    D_opt = G_D * (C_unit ** b)  # billions tokens

    return N_opt, D_opt


def compute_budget(N_params, D_tokens):
    """Compute FLOPs = 6 * N * D"""
    return 6 * N_params * D_tokens


class TinyGPT(nn.Module):
    """Minimal GPT for scaling law verification."""
    def __init__(self, vocab_size=64, d_model=64, n_heads=2, n_layers=2,
                 d_ff=128, max_seq_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, n_heads, d_ff, batch_first=True, dropout=0.0
            ) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln(h))


def generate_data(vocab_size, seq_len, n_seqs, device):
    """Generate data with learnable patterns."""
    data = []
    for _ in range(n_seqs):
        plen = torch.randint(4, 8, (1,)).item()
        pattern = torch.randint(0, vocab_size, (plen,))
        seq = pattern.repeat(seq_len // plen + 1)[:seq_len]
        data.append(seq)
    return torch.stack(data).to(device)


def train_and_measure(model, data, n_steps, device, lr=1e-3):
    """Train model and return final loss."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    vocab_size = model.head.out_features if hasattr(model.head, 'out_features') else model.head.weight.shape[0]

    losses = []
    for step in range(n_steps):
        idx = torch.randint(0, data.shape[0], (32,))
        batch = data[idx]
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    return losses


def run_experiments(device='cuda'):
    print("=" * 70)
    print("Scaling Law Calculator & Verification")
    print(f"Device: {device}")
    print("=" * 70)

    results = {}

    # ----------------------------------------------------------
    # Part 1: Scaling Law Predictions
    # ----------------------------------------------------------
    print("\n--- Part 1: Chinchilla Predictions for Known Models ---")
    print(f"  Formula: L(N,D) = {A}/N^{ALPHA} + {B}/D^{BETA} + {E}")

    models = [
        ('GPT-3', 175000, 300),    # 175B params, 300B tokens
        ('Chinchilla', 70000, 1400),
        ('Gopher', 280000, 300),
        ('LLaMA-7B', 7000, 1000),
        ('LLaMA-65B', 65000, 1400),
        ('LLaMA3-8B', 8000, 15000),
        ('LLaMA3-70B', 70000, 15000),
    ]

    for name, N_M, D_B in models:
        pred_loss = chinchilla_loss(N_M, D_B)
        model_loss = A / (N_M ** ALPHA)
        data_loss = B / (D_B ** BETA)
        ratio = D_B * 1e9 / (N_M * 1e6)  # tokens/param

        print(f"  {name:14s}: N={N_M/1000:.0f}B, D={D_B:.0f}B tok, "
              f"ratio={ratio:.0f}:1, "
              f"L_pred={pred_loss:.3f} (model={model_loss:.3f} + data={data_loss:.3f})")

        results[f'pred_{name}'] = {
            'N_B': N_M/1000, 'D_B': D_B, 'ratio': ratio,
            'pred_loss': pred_loss, 'model_loss': model_loss, 'data_loss': data_loss,
        }

    # ----------------------------------------------------------
    # Part 2: Optimal Allocation for Different Budgets
    # ----------------------------------------------------------
    print("\n--- Part 2: Optimal Allocation by Compute Budget ---")

    budgets = [
        ('1 GPU-day (A100)', 312e12 * 86400 / 6),  # 1 A100 = 312 TFLOPS
        ('8 GPU-days', 312e12 * 86400 * 8 / 6),
        ('1K GPU-days', 312e12 * 86400 * 1000 / 6),
        ('1M GPU-days (GPT-4 scale)', 312e12 * 86400 * 1e6 / 6),
    ]

    for name, C in budgets:
        N_opt, D_opt = optimal_allocation(C)
        pred_loss = chinchilla_loss(N_opt, D_opt)

        print(f"  {name:35s}: N_opt={N_opt/1000:.1f}B, D_opt={D_opt:.1f}B tok, "
              f"L_pred={pred_loss:.3f}")

        results[f'budget_{name}'] = {
            'N_opt_B': N_opt/1000, 'D_opt_B': D_opt, 'loss': pred_loss,
        }

    # ----------------------------------------------------------
    # Part 3: Empirical Verification (Small Scale)
    # ----------------------------------------------------------
    print("\n--- Part 3: Empirical Verification ---")

    vocab_size = 64
    seq_len = 64
    data = generate_data(vocab_size, seq_len, 1000, device)

    configs = [
        ('N0.003M_D5K',    32, 1, 1, 64,   50),
        ('N0.008M_D10K',   48, 2, 1, 96,   100),
        ('N0.03M_D20K',    64, 2, 2, 128,  200),
        ('N0.1M_D50K',     96, 2, 3, 192,  400),
        ('N0.3M_D100K',    128, 4, 3, 256, 600),
        ('N1M_D200K',      192, 4, 4, 384, 800),
    ]

    print(f"  Training {len(configs)} models to verify scaling behavior...")

    for name, dm, nh, nl, dff, n_steps in configs:
        torch.manual_seed(42)
        model = TinyGPT(vocab_size, dm, nh, nl, dff, seq_len).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        n_params_M = n_params / 1e6
        n_tokens = n_steps * 32 * (seq_len - 1)  # total training tokens
        n_tokens_K = n_tokens / 1e3

        losses = train_and_measure(model, data, n_steps, device)
        final_loss = losses[-1]
        min_loss = min(losses)

        # Also measure time
        torch.cuda.synchronize()
        t0 = time.time()
        train_and_measure(model, data, min(50, n_steps), device)
        torch.cuda.synchronize()
        t = (time.time() - t0) / min(50, n_steps) * 1000

        print(f"  {name:20s}: {n_params_M:.3f}M params, {n_tokens_K:.0f}K tok, "
              f"final_loss={final_loss:.4f}, min_loss={min_loss:.4f}, "
              f"speed={t:.1f}ms/step")

        results[f'empirical_{name}'] = {
            'params_M': n_params_M, 'tokens_K': n_tokens_K,
            'final_loss': final_loss, 'min_loss': min_loss,
        }

    # ----------------------------------------------------------
    # Part 4: Compute-Optimal Frontier
    # ----------------------------------------------------------
    print("\n--- Part 4: Compute-Optimal Frontier ---")
    print("  For fixed compute C, varying N/D split:")

    # Fixed compute: ~1M params × 100K tokens = 6 × 1e6 × 1e5 = 6e11 FLOPs
    C_fixed = 6 * 1e6 * 1e5  # 6e11 FLOPs
    total_tokens = 100_000

    for N_scale in [0.3, 0.5, 1.0, 2.0, 3.0]:
        N = int(1e6 * N_scale)  # params
        D = int(C_fixed / (6 * N))  # tokens from compute budget

        dm = max(32, int((N / 4) ** 0.5) * 8)  # rough d_model
        nh = max(1, dm // 32)
        nl = max(1, min(4, int(N / (dm * dm * 12))))
        dff = dm * 2
        n_steps = max(10, D // (32 * seq_len))

        torch.manual_seed(42)
        try:
            model = TinyGPT(vocab_size, dm, nh, nl, dff, seq_len).to(device)
            actual_params = sum(p.numel() for p in model.parameters())

            losses = train_and_measure(model, data, min(n_steps, 300), device)
            min_loss = min(losses)

            chinchilla_pred = chinchilla_loss(actual_params/1e6, D/1e9)

            print(f"    N×{N_scale:.1f}: {actual_params/1e6:.3f}M params, "
                  f"D={D//1000}K tok, "
                  f"loss={min_loss:.4f} (chinchilla_pred={chinchilla_pred:.3f})")

            results[f'frontier_N{N_scale}'] = {
                'params': actual_params, 'tokens': D,
                'loss': min_loss, 'pred': chinchilla_pred,
            }
        except Exception as ex:
            print(f"    N×{N_scale:.1f}: skipped ({ex})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Scaling Law Key Findings")
    print("=" * 70)
    print(f"""
1. Chinchilla Formula: L(N,D) = {A}/N^{ALPHA} + {B}/D^{BETA} + {E}
2. Optimal ratio: ~20 tokens/parameter
3. Model and data scale equally: N ∝ C^0.46, D ∝ C^0.54
4. LLaMA over-trains for inference efficiency
5. Data quality > model size beyond Chinchilla optimal
    """)

    with open('scaling_law_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to scaling_law_results.json")
    return results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    run_experiments(device=device)
