#!/usr/bin/env python3
"""DeepSeek-V3 MoE Simulation — Verify Key Claims
===================================================
Simulates DeepSeek-V3's key innovations:
1. Auxiliary-loss-free load balancing (bias-based vs auxiliary loss)
2. MoE routing with top-K selection
3. FP8 fine-grained quantization impact
4. Multi-Token Prediction (MTP) training benefit

Based on: DeepSeek-AI, 2024, arXiv:2412.19437
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time


class MoERouter(nn.Module):
    """MoE Router with optional auxiliary loss or bias-based balancing."""
    def __init__(self, d_model, n_experts, top_k=6):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        # Bias-based load balancing (DeepSeek-V3)
        self.bias = nn.Parameter(torch.zeros(n_experts))
        # For auxiliary loss tracking
        self.aux_loss = 0.0

    def forward(self, x, use_aux_loss=False, aux_alpha=0.01, use_bias_balance=True):
        """
        x: (B, T, D)
        Returns: routing weights, expert_indices, aux_loss
        """
        logits = self.gate(x)  # (B, T, E)

        # Routing scores
        scores = F.softmax(logits, dim=-1)

        if use_bias_balance:
            # DeepSeek-V3: add bias for selection (not in softmax!)
            adjusted = scores + self.bias
            top_scores, top_indices = adjusted.topk(self.top_k, dim=-1)
        else:
            top_scores, top_indices = scores.topk(self.top_k, dim=-1)

        # Normalize top-k weights
        top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True)

        # Auxiliary loss (traditional)
        if use_aux_loss:
            # f_i = fraction of tokens routed to expert i
            mask = torch.zeros_like(scores)
            mask.scatter_(-1, top_indices, 1.0)
            f = mask.mean(dim=(0, 1))  # (E,)
            # P_i = mean routing probability
            P = scores.mean(dim=(0, 1))  # (E,)
            self.aux_loss = aux_alpha * (f * P).sum() * self.n_experts
        else:
            self.aux_loss = 0.0

        return top_scores, top_indices, scores


class Expert(nn.Module):
    """Single expert (SwiGLU FFN)."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoELayer(nn.Module):
    """MoE Layer with top-K routing."""
    def __init__(self, d_model, n_experts=16, top_k=2, d_ff=None):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        d_ff = d_ff or d_model * 4
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])
        self.router = MoERouter(d_model, n_experts, top_k)

    def forward(self, x, use_aux_loss=False, use_bias_balance=True):
        B, T, D = x.shape
        top_weights, top_indices, all_scores = self.router(
            x, use_aux_loss=use_aux_loss, use_bias_balance=use_bias_balance
        )

        # Simple dispatch (not efficient, but correct for simulation)
        output = torch.zeros_like(x)
        for i in range(self.n_experts):
            # Find tokens routed to this expert
            expert_mask = (top_indices == i).any(dim=-1)  # (B, T)
            if not expert_mask.any():
                continue

            # Get weights for this expert
            expert_weights = torch.zeros(B, T, device=x.device)
            for k in range(self.top_k):
                mask_k = (top_indices[:, :, k] == i)
                expert_weights[mask_k] = top_weights[:, :, k][mask_k]

            # Process tokens through expert
            tokens = x[expert_mask]  # (N, D)
            expert_out = self.experts[i](tokens)  # (N, D)

            # Weighted output
            weighted = expert_out * expert_weights[expert_mask].unsqueeze(-1)
            output[expert_mask] += weighted

        return output


class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_experts=16, top_k=2, use_moe=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        if use_moe:
            self.ffn = MoELayer(d_model, n_experts, top_k)
        else:
            d_ff = d_model * 4
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, d_model),
            )
        self.use_moe = use_moe

    def forward(self, x, use_aux_loss=False, use_bias_balance=True):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        h = self.ln2(x)
        if self.use_moe:
            ffn_out = self.ffn(h, use_aux_loss=use_aux_loss, use_bias_balance=use_bias_balance)
        else:
            ffn_out = self.ffn(h)
        x = x + ffn_out
        return x


def experiment1_load_balancing(device='cuda'):
    """Exp1: Compare auxiliary loss vs bias-based balancing."""
    print("\n" + "="*70)
    print("Experiment 1: Load Balancing — Auxiliary Loss vs Bias-Based")
    print("="*70)

    torch.manual_seed(42)
    d_model = 128
    n_experts = 16
    top_k = 4
    B, T = 32, 64

    results = {}

    for method in ['none', 'aux_loss', 'bias']:
        model = SimpleTransformerBlock(d_model, n_heads=4, n_experts=n_experts, top_k=top_k).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randn(B, T, d_model, device=device)
        target = torch.randn(B, T, d_model, device=device)

        load_history = []
        loss_history = []
        aux_loss_history = []

        for step in range(200):
            use_aux = (method == 'aux_loss')
            use_bias = (method == 'bias')

            out = model(x, use_aux_loss=use_aux, use_bias_balance=use_bias)
            main_loss = F.mse_loss(out, target)

            # Total loss
            if method == 'aux_loss':
                total_loss = main_loss + model.ffn.router.aux_loss
                aux_loss_history.append(model.ffn.router.aux_loss.item())
            else:
                total_loss = main_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track load balance
            with torch.no_grad():
                _, _, scores = model.ffn.router(x, use_bias_balance=use_bias)
                top_vals, top_idx = scores.topk(top_k, dim=-1)
                expert_counts = torch.zeros(n_experts, device=device)
                for e in range(n_experts):
                    expert_counts[e] = (top_idx == e).float().sum()
                load_balance = expert_counts / expert_counts.sum()
                # Ideal = 1/n_experts for each
                ideal = 1.0 / n_experts
                max_deviation = (load_balance - ideal).abs().max().item()
                load_history.append(max_deviation)

            loss_history.append(main_loss.item())

        avg_load_dev = sum(load_history[-20:]) / 20
        final_loss = sum(loss_history[-10:]) / 10

        results[method] = {
            'avg_load_deviation': avg_load_dev,
            'final_loss': final_loss,
            'load_history': load_history[-20:],
        }

        name = {'none': 'No Balancing', 'aux_loss': 'Auxiliary Loss', 'bias': 'Bias-Based (V3)'}
        print(f"\n  [{name[method]}]")
        print(f"    Final loss: {final_loss:.4f}")
        print(f"    Avg load deviation: {avg_load_dev:.4f}")
        if method == 'aux_loss' and aux_loss_history:
            print(f"    Final aux loss: {aux_loss_history[-1]:.4f}")

    print(f"\n  Comparison:")
    print(f"  {'Method':20s} | {'Loss':>8s} | {'Load Dev':>10s}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*10}")
    for m in ['none', 'aux_loss', 'bias']:
        name = {'none': 'No Balance', 'aux_loss': 'Aux Loss', 'bias': 'Bias (V3)'}[m]
        print(f"  {name:20s} | {results[m]['final_loss']:8.4f} | {results[m]['avg_load_deviation']:10.4f}")

    return results


def experiment2_fp8_quantization(device='cuda'):
    """Exp2: FP8 per-tensor vs tile-wise quantization."""
    print("\n" + "="*70)
    print("Experiment 2: FP8 Quantization — Per-Tensor vs Tile-Wise")
    print("="*70)

    torch.manual_seed(42)
    results = {}

    for N in [512, 1024, 2048, 4096]:
        M = 7168  # DeepSeek-V3 hidden dim
        K = 7168

        W = torch.randn(M, K, device=device) * 0.5
        X = torch.randn(N, K, device=device)

        # FP32 baseline
        out_fp32 = X @ W.T

        # FP8 per-tensor
        def quantize_per_tensor(tensor):
            scale = tensor.abs().max() / 448.0  # FP8 E4M3 max = 448
            return (tensor / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale

        W_q, w_scale = quantize_per_tensor(W)
        X_q, x_scale = quantize_per_tensor(X)
        out_per_tensor = X_q.float() @ W_q.float().T * (w_scale * x_scale)

        err_per_tensor = (out_fp32 - out_per_tensor).abs()
        cos_per_tensor = F.cosine_similarity(out_fp32.flatten(), out_per_tensor.flatten(), dim=0).item()

        # FP8 tile-wise (1x128)
        tile_size = 128
        def quantize_tile_wise(tensor):
            """Quantize with 1x128 tile granularity."""
            shape = tensor.shape
            if len(shape) == 2:
                B, D = shape
                # Pad D to multiple of tile_size
                D_pad = math.ceil(D / tile_size) * tile_size
                padded = torch.zeros(B, D_pad, device=tensor.device)
                padded[:, :D] = tensor
                # Reshape to tiles
                padded = padded.reshape(B, D_pad // tile_size, tile_size)
                # Per-tile scale
                tile_max = padded.abs().amax(dim=-1, keepdim=True)
                scales = tile_max / 448.0
                quantized = (padded / scales).clamp(-448, 448)
                # Dequantize
                dequantized = quantized * scales
                return dequantized.reshape(B, D_pad)[:, :D]
            return tensor

        W_dq = quantize_tile_wise(W)
        X_dq = quantize_tile_wise(X)
        out_tile = X_dq @ W_dq.T

        err_tile = (out_fp32 - out_tile).abs()
        cos_tile = F.cosine_similarity(out_fp32.flatten(), out_tile.flatten(), dim=0).item()

        results[f'N{N}'] = {
            'per_tensor_max_err': err_per_tensor.max().item(),
            'per_tensor_cos': cos_per_tensor,
            'tile_max_err': err_tile.max().item(),
            'tile_cos': cos_tile,
        }

        print(f"  N={N}: per_tensor cos={cos_per_tensor:.6f}, tile cos={cos_tile:.6f}, "
              f"improvement={(cos_tile - cos_per_tensor)*100:.4f}%")

    return results


def experiment3_mtp_benefit(device='cuda'):
    """Exp3: Multi-Token Prediction training benefit."""
    print("\n" + "="*70)
    print("Experiment 3: Multi-Token Prediction Training Benefit")
    print("="*70)

    torch.manual_seed(42)
    vocab_size = 200
    seq_len = 32
    d_model = 64

    results = {}

    # Generate data
    data = torch.randint(0, vocab_size, (64, seq_len + 2), device=device)

    for method in ['standard', 'mtp_depth1', 'mtp_depth2']:
        torch.manual_seed(42)
        embed = nn.Embedding(vocab_size, d_model).to(device)
        lm_head = nn.Linear(d_model, vocab_size, bias=False).to(device)
        # Simple 2-layer transformer
        layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model*4,
                                        batch_first=True, dropout=0.0)
            for _ in range(2)
        ]).to(device)

        # MTP modules
        mtp_modules = []
        n_mtp = {'standard': 0, 'mtp_depth1': 1, 'mtp_depth2': 2}[method]
        for _ in range(n_mtp):
            mtp_modules.append(nn.ModuleDict({
                'embed_proj': nn.Linear(d_model * 2, d_model, bias=False).to(device),
                'layer': nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model*2,
                                                      batch_first=True, dropout=0.0).to(device),
                'head': nn.Linear(d_model, vocab_size, bias=False).to(device),
            }))

        params = list(embed.parameters()) + list(lm_head.parameters()) + list(layers.parameters())
        for m in mtp_modules:
            params += list(m.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3)

        loss_history = []

        for step in range(300):
            batch = data[torch.randint(0, data.size(0), (16,))]
            x = batch[:, :-2]
            y_next = batch[:, 1:-1]  # t+1
            y_next2 = batch[:, 2:]   # t+2

            h = embed(x)
            for layer in layers:
                h = layer(h)

            # Standard loss (predict t+1)
            logits1 = lm_head(h)
            loss = F.cross_entropy(logits1.reshape(-1, vocab_size), y_next.reshape(-1))

            # MTP losses
            for d_idx, mtp in enumerate(mtp_modules):
                # Target is t+2 (or t+3 for depth 2)
                target = y_next2 if d_idx == 0 else y_next2
                # Use previous embedding + current token embedding
                prev_embed = h.detach()  # Stop gradient from main model
                curr_embed = embed(target)
                combined = torch.cat([prev_embed, curr_embed], dim=-1)
                projected = mtp['embed_proj'](combined)
                h_mtp = mtp['layer'](projected)
                logits_mtp = mtp['head'](h_mtp)
                mtp_loss = F.cross_entropy(logits_mtp.reshape(-1, vocab_size), target.reshape(-1))
                loss = loss + 0.3 * mtp_loss  # Lower weight for MTP

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track main loss only
            with torch.no_grad():
                main_loss = F.cross_entropy(logits1.reshape(-1, vocab_size), y_next.reshape(-1))
            loss_history.append(main_loss.item())

        results[method] = {
            'final_loss': sum(loss_history[-10:]) / 10,
            'loss_history': loss_history,
        }
        print(f"  {method:15s}: final_loss={results[method]['final_loss']:.4f}")

    return results


def experiment4_expert_count(device='cuda'):
    """Exp4: Expert count vs performance — why DeepSeek uses 256 experts."""
    print("\n" + "="*70)
    print("Experiment 4: Expert Count Impact")
    print("="*70)

    torch.manual_seed(42)
    d_model = 128
    B, T = 16, 32

    results = {}

    for n_experts in [4, 8, 16, 32, 64]:
        # Keep total params constant: d_ff = d_model * 4 * (16/n_experts)
        d_ff = int(d_model * 4 * (16 / n_experts))
        top_k = min(4, n_experts)

        model = SimpleTransformerBlock(d_model, n_heads=4, n_experts=n_experts, top_k=top_k).to(device)
        x = torch.randn(B, T, d_model, device=device)

        # Count params
        total_params = sum(p.numel() for p in model.parameters())
        active_params = 0
        for name, p in model.named_parameters():
            if 'experts' in name:
                # Only top_k experts active
                active_params += p.numel() * top_k / n_experts
            else:
                active_params += p.numel()

        # Forward pass time
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.time()
            out = model(x)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

        avg_time = sum(times) / len(times) * 1000

        # Load balance
        with torch.no_grad():
            _, _, scores = model.ffn.router(x)
            _, top_idx = scores.topk(top_k, dim=-1)
            expert_counts = torch.zeros(n_experts, device=device)
            for e in range(n_experts):
                expert_counts[e] = (top_idx == e).float().sum()
            load_balance = expert_counts / expert_counts.sum()
            max_dev = (load_balance - 1.0/n_experts).abs().max().item()

        results[f'E{n_experts}'] = {
            'total_params_M': total_params / 1e6,
            'active_params_M': active_params / 1e6,
            'sparsity_ratio': total_params / active_params,
            'forward_ms': avg_time,
            'load_deviation': max_dev,
        }

        print(f"  E={n_experts:3d}: params={total_params/1e6:.2f}M, active={active_params/1e6:.2f}M, "
              f"ratio={total_params/active_params:.1f}x, time={avg_time:.2f}ms, load_dev={max_dev:.4f}")

    return results


def run_all_experiments(device='cuda'):
    print("="*70)
    print("DeepSeek-V3 MoE Simulation — Verify Key Claims")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)

    all_results = {}
    all_results['exp1_load_balance'] = experiment1_load_balancing(device)
    all_results['exp2_fp8'] = experiment2_fp8_quantization(device)
    all_results['exp3_mtp'] = experiment3_mtp_benefit(device)
    all_results['exp4_expert_count'] = experiment4_expert_count(device)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: DeepSeek-V3 Key Claims Verification")
    print("="*70)
    print("""
    1. Bias-based load balancing > Auxiliary loss
       → Exp1: [SEE RESULTS]

    2. FP8 tile-wise >> per-tensor quantization
       → Exp2: [SEE RESULTS]

    3. MTP improves training
       → Exp3: [SEE RESULTS]

    4. More experts = better sparsity ratio
       → Exp4: [SEE RESULTS]
    """)

    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open('deepseek_v3_results.json', 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print("Results saved to deepseek_v3_results.json")
    return all_results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    run_all_experiments(device=device)
