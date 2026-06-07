#!/usr/bin/env python3
"""Prefix Sharing Gradient Flow Validation — RTX 4090

Validates that gradient backpropagation works correctly through PS:
  Provider forward → KV store → Reuser suffix-only forward with KV injection
  → loss → backward → verify gradients propagate through KV path

Key question: Does PS produce equivalent gradients to normal forward+backward?

Experiment design:
1. Normal forward+backward: full sequence → loss → backward → record gradients
2. PS forward+backward: Provider prefix → store KV → Reuser suffix → loss → backward → record gradients
3. Compare: gradient cos_sim between normal and PS for each parameter

This validates PS is safe for TRAINING (not just inference).
"""

import json
import time
import torch
import torch.nn as nn
import math

DEVICE = "cuda:0"
DTYPE = torch.float32  # Use FP32 for gradient accuracy

class SimpleTransformerLayer(nn.Module):
    """Minimal transformer layer for gradient flow testing."""
    def __init__(self, hidden_dim=256, num_heads=4, num_kv_heads=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.g = num_heads // num_kv_heads

        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

        # MLP (SwiGLU-style)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.down_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

        # LayerNorm
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward_attention(self, hidden_states, prefix_kv=None):
        """Attention with optional KV injection for PS."""
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)

        B, S, H = hidden_states.shape
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        # Reshape to [B, S, heads, head_dim]
        Q = Q.view(B, S, self.num_heads, self.head_dim)
        K_kv = K.view(B, S, self.num_kv_heads, self.head_dim)  # kv_heads format (stored)
        V_kv = V.view(B, S, self.num_kv_heads, self.head_dim)  # kv_heads format (stored)

        if prefix_kv is not None:
            # PS mode: inject prefix KV
            prefix_K_kv, prefix_V_kv = prefix_kv  # Both in kv_heads format
            # Expand GQA for all KV before SDPA
            if self.g > 1:
                prefix_K = prefix_K_kv.repeat_interleave(self.g, dim=2)
                prefix_V = prefix_V_kv.repeat_interleave(self.g, dim=2)
                K_exp = K_kv.repeat_interleave(self.g, dim=2)
                V_exp = V_kv.repeat_interleave(self.g, dim=2)
            else:
                prefix_K = prefix_K_kv
                prefix_V = prefix_V_kv
                K_exp = K_kv
                V_exp = V_kv
            # Concat: [prefix; suffix] for KV
            K_full = torch.cat([prefix_K, K_exp], dim=1)
            V_full = torch.cat([prefix_V, V_exp], dim=1)

            # Block-causal mask
            prefix_len = prefix_kv[0].shape[1]
            total_kv_len = prefix_len + S
            mask = torch.zeros(S, total_kv_len, device=hidden_states.device, dtype=DTYPE)
            q_pos = torch.arange(S, device=hidden_states.device)
            kv_pos = torch.arange(total_kv_len, device=hidden_states.device)
            future_mask = kv_pos.unsqueeze(0) > (q_pos.unsqueeze(1) + prefix_len)
            mask[future_mask] = float('-inf')
            attn_mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)

            Q_t = Q.transpose(1, 2)
            K_full_t = K_full.transpose(1, 2)
            V_full_t = V_full.transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(
                Q_t, K_full_t, V_full_t, attn_mask=attn_mask, is_causal=False
            )
        else:
            # Normal mode: GQA expand + causal attention
            if self.g > 1:
                K_exp = K_kv.repeat_interleave(self.g, dim=2)
                V_exp = V_kv.repeat_interleave(self.g, dim=2)
            else:
                K_exp = K_kv
                V_exp = V_kv

            Q_t = Q.transpose(1, 2)
            K_t = K_exp.transpose(1, 2)
            V_t = V_exp.transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(
                Q_t, K_t, V_t, is_causal=True
            )

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.o_proj(out)
        hidden_states = residual + out

        return hidden_states, (K_kv, V_kv)  # Store in kv_heads format

    def forward_mlp(self, hidden_states):
        """MLP with SwiGLU."""
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(gate * up)
        return residual + hidden_states

    def forward(self, hidden_states, prefix_kv=None, store_kv=False):
        """Full layer forward."""
        hidden_states, kv = self.forward_attention(hidden_states, prefix_kv=prefix_kv)
        hidden_states = self.forward_mlp(hidden_states)
        if store_kv:
            return hidden_states, kv
        return hidden_states


class SimpleTransformer(nn.Module):
    """Multi-layer transformer for gradient flow testing."""
    def __init__(self, num_layers=4, hidden_dim=256, num_heads=4, num_kv_heads=2):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(hidden_dim, num_heads, num_kv_heads)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.vocab_proj = nn.Linear(hidden_dim, 1000)  # Small vocab for testing

    def forward_normal(self, input_embeds):
        """Normal forward: full sequence through all layers."""
        hidden = input_embeds
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_ln(hidden)
        logits = self.vocab_proj(hidden)
        return logits

    def forward_ps(self, prefix_embeds, suffix_embeds):
        """PS forward: Provider processes prefix, Reuser processes suffix with KV injection."""
        # Provider pass: full forward for prefix tokens, storing KV at each layer
        kv_stores = []
        hidden = prefix_embeds
        for layer in self.layers:
            hidden, kv = layer(hidden, store_kv=True)
            kv_stores.append(kv)

        # Reuser pass: suffix-only forward with KV injection
        hidden = suffix_embeds
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, prefix_kv=kv_stores[i])

        hidden = self.final_ln(hidden)
        logits = self.vocab_proj(hidden)
        return logits


def compute_loss(logits, targets, mask):
    """Cross-entropy loss with mask."""
    # logits: [B, S, vocab], targets: [B, S]
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction='none'
    )
    loss = (loss * mask.view(-1)).sum() / mask.sum()
    return loss


def main():
    print("=" * 60)
    print("Prefix Sharing Gradient Flow Validation — RTX 4090")
    print("Verifying PS produces equivalent gradients to normal training")
    print("=" * 60)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}, Memory: {gpu_mem:.1f} GB\n")

    # Model config
    num_layers = 4
    hidden_dim = 256
    num_heads = 4
    num_kv_heads = 2  # GQA
    vocab_size = 1000

    model = SimpleTransformer(num_layers, hidden_dim, num_heads, num_kv_heads).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_layers} layers, hidden={hidden_dim}, heads={num_heads}, kv_heads={num_kv_heads}")
    print(f"Parameters: {param_count:,} ({param_count * 4 / 1e6:.1f}M bytes FP32)\n")

    results = []

    for prefix_len in [32, 64, 128, 256]:
        suffix_len = 64
        total_len = prefix_len + suffix_len
        batch_size = 4

        print(f"--- prefix={prefix_len}, suffix={suffix_len}, total={total_len}, batch={batch_size} ---")

        # Create input embeddings (random)
        prefix_embeds = torch.randn(batch_size, prefix_len, hidden_dim, device=DEVICE, dtype=DTYPE)
        suffix_embeds = torch.randn(batch_size, suffix_len, hidden_dim, device=DEVICE, dtype=DTYPE)
        full_embeds = torch.cat([prefix_embeds, suffix_embeds], dim=1)

        # Target tokens (random)
        prefix_targets = torch.randint(0, vocab_size, (batch_size, prefix_len), device=DEVICE)
        suffix_targets = torch.randint(0, vocab_size, (batch_size, suffix_len), device=DEVICE)
        full_targets = torch.cat([prefix_targets, suffix_targets], dim=1)

        # Masks
        prefix_mask = torch.ones(batch_size, prefix_len, device=DEVICE)
        suffix_mask = torch.ones(batch_size, suffix_len, device=DEVICE)
        full_mask = torch.ones(batch_size, total_len, device=DEVICE)

        # === Method 1: Normal forward+backward ===
        model.zero_grad()
        logits_normal = model.forward_normal(full_embeds)
        # Loss on ALL tokens (prefix + suffix) — standard training
        loss_normal = compute_loss(logits_normal, full_targets, full_mask)
        loss_normal.backward()
        grad_normal = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        loss_normal_val = loss_normal.item()

        # === Method 2: PS forward+backward ===
        # Provider: prefix forward → store KV
        # Reuser: suffix forward with KV injection → loss on suffix only
        # Also need prefix-last restore: logits at prefix positions from provider forward
        model.zero_grad()

        # Provider forward: compute logits for prefix positions
        provider_logits = model.forward_normal(prefix_embeds)  # Full forward on prefix
        # PS suffix forward: KV injection + suffix-only forward
        ps_suffix_logits = model.forward_ps(prefix_embeds, suffix_embeds)

        # Combined logits: prefix from provider, suffix from PS
        # This ensures gradient flows through both paths
        combined_logits = torch.cat([provider_logits, ps_suffix_logits], dim=1)

        # Loss on all tokens (same as normal)
        loss_ps = compute_loss(combined_logits, full_targets, full_mask)
        loss_ps.backward()
        grad_ps = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        loss_ps_val = loss_ps.item()

        # === Compare gradients ===
        print(f"  Normal loss: {loss_normal_val:.6f}, PS loss: {loss_ps_val:.6f}")
        print(f"  Loss diff: {abs(loss_normal_val - loss_ps_val):.6f}")

        grad_cos_sims = {}
        grad_max_diffs = {}
        total_cos = 0
        total_count = 0

        for name in grad_normal:
            if name not in grad_ps:
                continue
            gn = grad_normal[name].flatten().float()
            gp = grad_ps[name].flatten().float()

            cos_sim = torch.nn.functional.cosine_similarity(gn.unsqueeze(0), gp.unsqueeze(0)).item()
            max_diff = (gn - gp).abs().max().item()
            mean_diff = (gn - gp).abs().mean().item()
            rel_diff = max_diff / (gn.abs().mean().item() + 1e-8)

            grad_cos_sims[name] = cos_sim
            grad_max_diffs[name] = max_diff

            total_cos += cos_sim
            total_count += 1

        avg_cos = total_cos / total_count if total_count > 0 else 0

        print(f"  Gradient cos_sim (avg): {avg_cos:.6f}")
        print(f"  Gradient cos_sim (min): {min(grad_cos_sims.values()):.6f}")
        print(f"  Gradient max_diff (max): {max(grad_max_diffs.values()):.6f}")

        # Detail per parameter
        for name in sorted(grad_cos_sims.keys()):
            cs = grad_cos_sims[name]
            md = grad_max_diffs[name]
            pass_fail = "PASS" if cs > 0.99 else "FAIL"
            print(f"    {name}: cos_sim={cs:.6f}, max_diff={md:.6f} [{pass_fail}]")

        results.append({
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "total_len": total_len,
            "batch_size": batch_size,
            "loss_normal": loss_normal_val,
            "loss_ps": loss_ps_val,
            "loss_diff": abs(loss_normal_val - loss_ps_val),
            "grad_cos_sim_avg": avg_cos,
            "grad_cos_sim_min": min(grad_cos_sims.values()),
            "grad_max_diff_max": max(grad_max_diffs.values()),
            "per_param_cos_sim": grad_cos_sims,
        })

        del prefix_embeds, suffix_embeds, full_embeds, logits_normal, ps_suffix_logits
        del provider_logits, combined_logits
        torch.cuda.empty_cache()

    # Save results
    all_results = {
        "gpu": gpu_name,
        "model": f"{num_layers}L-{hidden_dim}H-{num_heads}QH-{num_kv_heads}KVH",
        "param_count": param_count,
        "results": results,
    }

    with open("ps_gradient_flow_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("GRADIENT FLOW VALIDATION SUMMARY")
    print("=" * 60)
    print("\nPrefix → Normal vs PS gradient equivalence:")
    for r in results:
        avg_cos = r["grad_cos_sim_avg"]
        min_cos = r["grad_cos_sim_min"]
        pass_fail = "PASS" if min_cos > 0.99 else "FAIL"
        print(f"  prefix={r['prefix_len']}: avg_cos={avg_cos:.6f}, min_cos={min_cos:.6f} [{pass_fail}]")

    print("\nConclusion:")
    all_pass = all(r["grad_cos_sim_min"] > 0.99 for r in results)
    if all_pass:
        print("  ALL PASS — PS gradient flow is equivalent to normal training!")
        print("  PS can be safely used in GRPO/PPO training without gradient distortion.")
    else:
        print("  SOME FAIL — PS gradient flow differs from normal training!")
        print("  Need to investigate which parameters have gradient mismatch.")

    print(f"\nResults saved to ps_gradient_flow_results.json")


if __name__ == "__main__":
    main()