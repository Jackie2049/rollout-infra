"""Block-Causal Mask Backend Benchmark — RTX 4090

Compares two approaches for implementing block-causal attention mask:
1. Single SDPA call with float attn_mask (math backend fallback)
2. Two FlashAttention calls + LSE merge (prefix full-attn + suffix causal)

The LSE merge approach:
  Call 1: Q_suffix × K_prefix, V_prefix (is_causal=False) → out_prefix, LSE_prefix
  Call 2: Q_suffix × K_suffix, V_suffix (is_causal=True) → out_suffix, LSE_suffix
  Merge: out = (e^LSE_p × out_p + e^LSE_s × out_s) / (e^LSE_p + e^LSE_s)

This is mathematically exact (same as vLLM's cascade attention).

3 experiments:
  Exp1: Precision validation — LSE merge vs single SDPA block-causal vs baseline
  Exp2: Performance comparison — SDPA math vs FlashAttention LSE merge timing
  Exp3: KV injection with LSE merge — full pipeline precision + speedup
"""

import json
import time
import torch
import torch.nn as nn
import math

DEVICE = "cuda:0"
DTYPE = torch.float16


def warmup(device, n=100):
    x = torch.randn(256, 256, device=device, dtype=DTYPE)
    for _ in range(n):
        y = x @ x
    torch.cuda.synchronize()


def lse_merge_attn(q, prefix_k, prefix_v, suffix_k, suffix_v, num_heads, num_kv_heads, head_dim):
    """Block-causal attention via two FlashAttention calls + LSE merge.

    Mathematically equivalent to single SDPA call with block-causal mask,
    but uses FlashAttention's fast backend for both calls.

    Returns: merged attention output (same shape as single SDPA call)
    """
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        # Fallback: compute LSE manually from softmax weights
        return lse_merge_attn_fallback(q, prefix_k, prefix_v, suffix_k, suffix_v,
                                       num_heads, num_kv_heads, head_dim)

    g = num_heads // num_kv_heads
    suffix_len = q.shape[2]
    prefix_len = prefix_k.shape[2]

    # Expand GQA KV if needed
    if g > 1:
        prefix_k = prefix_k.repeat_interleave(g, dim=1)
        prefix_v = prefix_v.repeat_interleave(g, dim=1)
        suffix_k = suffix_k.repeat_interleave(g, dim=1)
        suffix_v = suffix_v.repeat_interleave(g, dim=1)

    # FlashAttention requires [batch, seqlen, num_heads, head_dim] format
    # q is [1, num_heads, suffix_len, head_dim] → permute to [1, suffix_len, num_heads, head_dim]
    q_fa = q.permute(0, 2, 1, 3)  # [1, suffix_len, num_heads, head_dim]
    p_k_fa = prefix_k.permute(0, 2, 1, 3)  # [1, prefix_len, num_heads, head_dim]
    p_v_fa = prefix_v.permute(0, 2, 1, 3)
    s_k_fa = suffix_k.permute(0, 2, 1, 3)  # [1, suffix_len, num_heads, head_dim]
    s_v_fa = suffix_v.permute(0, 2, 1, 3)

    # Call 1: Q_suffix × K_prefix, V_prefix (full attention — suffix sees all prefix)
    out_prefix, lse_prefix, _ = flash_attn_func(
        q_fa, p_k_fa, p_v_fa, causal=False, return_attn_probs=True
    )
    # lse_prefix shape: [batch, num_heads, seqlen_q] = [1, num_heads, suffix_len]

    # Call 2: Q_suffix × K_suffix, V_suffix (causal attention)
    out_suffix, lse_suffix, _ = flash_attn_func(
        q_fa, s_k_fa, s_v_fa, causal=True, return_attn_probs=True
    )
    # lse_suffix shape: [1, num_heads, suffix_len]

    # LSE merge: out = (e^LSE_p × out_p + e^LSE_s × out_s) / (e^LSE_p + e^LSE_s)
    # Stable merge: subtract max LSE before exp
    lse_prefix = lse_prefix.squeeze(0).transpose(0, 1)  # [suffix_len, num_heads]
    lse_suffix = lse_suffix.squeeze(0).transpose(0, 1)  # [suffix_len, num_heads]

    max_lse = torch.maximum(lse_prefix, lse_suffix)
    exp_p = torch.exp(lse_prefix - max_lse)  # [suffix_len, num_heads]
    exp_s = torch.exp(lse_suffix - max_lse)  # [suffix_len, num_heads]
    total = exp_p + exp_s  # [suffix_len, num_heads]

    # Weighted combination
    # out_prefix: [1, suffix_len, num_heads, head_dim] → [suffix_len, num_heads, head_dim]
    out_p = out_prefix.squeeze(0)  # [suffix_len, num_heads, head_dim]
    out_s = out_suffix.squeeze(0)  # [suffix_len, num_heads, head_dim]

    alpha_p = (exp_p / total).unsqueeze(-1)  # [suffix_len, num_heads, 1]
    alpha_s = (exp_s / total).unsqueeze(-1)  # [suffix_len, num_heads, 1]

    merged = alpha_p * out_p + alpha_s * out_s  # [suffix_len, num_heads, head_dim]

    # Convert back to [1, suffix_len, num_heads * head_dim]
    merged = merged.reshape(1, suffix_len, num_heads * head_dim)

    return merged


def lse_merge_attn_fallback(q, prefix_k, prefix_v, suffix_k, suffix_v, num_heads, num_kv_heads, head_dim):
    """Fallback LSE merge when flash_attn is not available.

    Uses SDPA for attention output + manual LSE computation.
    """
    g = num_heads // num_kv_heads
    suffix_len = q.shape[2]
    prefix_len = prefix_k.shape[2]

    if g > 1:
        prefix_k = prefix_k.repeat_interleave(g, dim=1)
        prefix_v = prefix_v.repeat_interleave(g, dim=1)
        suffix_k = suffix_k.repeat_interleave(g, dim=1)
        suffix_v = suffix_v.repeat_interleave(g, dim=1)

    scale = 1.0 / math.sqrt(head_dim)

    # Compute LSE manually: LSE = log(sum(exp(QK^T * scale)))
    # Prefix: Q @ K_prefix^T
    attn_weights_p = torch.matmul(q * scale, prefix_k.transpose(-2, -1))  # [1, num_heads, suffix_len, prefix_len]
    lse_prefix = torch.logsumexp(attn_weights_p, dim=-1)  # [1, num_heads, suffix_len]
    out_prefix = torch.nn.functional.scaled_dot_product_attention(
        q, prefix_k, prefix_v, is_causal=False
    )

    # Suffix: Q @ K_suffix^T (causal)
    attn_weights_s = torch.matmul(q * scale, suffix_k.transpose(-2, -1))  # [1, num_heads, suffix_len, suffix_len]
    # Apply causal mask to softmax computation for LSE
    causal_mask = torch.triu(torch.ones(suffix_len, suffix_len, device=q.device, dtype=torch.bool), diagonal=1)
    attn_weights_s_masked = attn_weights_s.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    lse_suffix = torch.logsumexp(attn_weights_s_masked, dim=-1)  # [1, num_heads, suffix_len]
    out_suffix = torch.nn.functional.scaled_dot_product_attention(
        q, suffix_k, suffix_v, is_causal=True
    )

    # Stable LSE merge
    lse_p = lse_prefix.squeeze(0).transpose(0, 1)  # [suffix_len, num_heads]
    lse_s = lse_suffix.squeeze(0).transpose(0, 1)  # [suffix_len, num_heads]

    max_lse = torch.maximum(lse_p, lse_s)
    exp_p = torch.exp(lse_p - max_lse)
    exp_s = torch.exp(lse_s - max_lse)
    total = exp_p + exp_s

    out_p = out_prefix.permute(2, 1, 3)  # [suffix_len, num_heads, head_dim]
    out_s = out_suffix.permute(2, 1, 3)  # [suffix_len, num_heads, head_dim]

    alpha_p = (exp_p / total).unsqueeze(-1)
    alpha_s = (exp_s / total).unsqueeze(-1)
    merged = alpha_p * out_p + alpha_s * out_s

    merged = merged.permute(1, 0, 2).reshape(1, suffix_len, num_heads * head_dim)
    return merged


def single_sdpa_block_causal(q, prefix_k, prefix_v, suffix_k, suffix_v, num_heads, num_kv_heads, head_dim, prefix_len):
    """Single SDPA call with block-causal float mask (math backend fallback).

    This is our validated approach from the KV injection prototype.
    """
    g = num_heads // num_kv_heads
    suffix_len = q.shape[2]

    # Expand GQA KV
    if g > 1:
        prefix_k = prefix_k.repeat_interleave(g, dim=1)
        prefix_v = prefix_v.repeat_interleave(g, dim=1)
        suffix_k = suffix_k.repeat_interleave(g, dim=1)
        suffix_v = suffix_v.repeat_interleave(g, dim=1)

    # Concat prefix KV + suffix KV
    full_k = torch.cat([prefix_k, suffix_k], dim=2)
    full_v = torch.cat([prefix_v, suffix_v], dim=2)

    # Block-causal mask (vectorized)
    total_kv_len = prefix_len + suffix_len
    model_dtype = q.dtype
    mask_2d = torch.zeros(suffix_len, total_kv_len, dtype=model_dtype, device=q.device)
    suffix_idx = torch.arange(suffix_len, device=q.device)
    kv_idx = torch.arange(total_kv_len, device=q.device)
    future_positions = kv_idx.unsqueeze(0) > (suffix_idx.unsqueeze(1) + prefix_len)
    mask_2d[future_positions] = float('-inf')
    attn_mask = mask_2d.unsqueeze(0).unsqueeze(0).expand(1, num_heads, -1, -1).contiguous()

    out = torch.nn.functional.scaled_dot_product_attention(
        q, full_k, full_v, attn_mask=attn_mask, is_causal=False
    )
    out = out.permute(0, 2, 1, 3).reshape(1, suffix_len, num_heads * head_dim).to(model_dtype)
    return out


class MiniTransformerLayer(nn.Module):
    def __init__(self, hidden_size=2560, num_heads=20, num_kv_heads=4, head_dim=128, mlp_ratio=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        mlp_intermediate = int(mlp_ratio * hidden_size)
        self.gate_proj = nn.Linear(hidden_size, mlp_intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, mlp_intermediate, bias=False)
        self.down_proj = nn.Linear(mlp_intermediate, hidden_size, bias=False)

        self.norm_weight = nn.Parameter(torch.ones(hidden_size))

    def _rms_norm(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + 1e-6) * self.norm_weight).to(x.dtype)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        B, S, _ = q.shape
        q = q.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        g = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, -1)
        attn_out = self.o_proj(attn_out)

        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states)
        gate = torch.nn.functional.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(gate * up)
        hidden_states = residual + hidden_states

        return hidden_states

    def forward_with_kv_injection_sdpa(self, hidden_states, prefix_kv, prefix_len):
        """KV injection using single SDPA block-causal (math backend)."""
        residual = hidden_states
        normed = self._rms_norm(hidden_states)
        suffix_len = hidden_states.shape[1]
        model_dtype = hidden_states.dtype

        q = self.q_proj(normed).view(1, suffix_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        s_k = self.k_proj(normed).view(1, suffix_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        s_v = self.v_proj(normed).view(1, suffix_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_out = single_sdpa_block_causal(q, prefix_kv[0], prefix_kv[1], s_k, s_v,
                                             self.num_heads, self.num_kv_heads, self.head_dim, prefix_len)
        attn_out = self.o_proj(attn_out)

        hidden_states = (residual + attn_out).to(model_dtype)

        residual = hidden_states
        normed = self._rms_norm(hidden_states)
        gate = torch.nn.functional.silu(self.gate_proj(normed))
        up = self.up_proj(normed)
        hidden_states = self.down_proj(gate * up)
        hidden_states = (residual + hidden_states).to(model_dtype)

        return hidden_states

    def forward_with_kv_injection_lse(self, hidden_states, prefix_kv, prefix_len):
        """KV injection using LSE merge (two FlashAttention calls)."""
        residual = hidden_states
        normed = self._rms_norm(hidden_states)
        suffix_len = hidden_states.shape[1]
        model_dtype = hidden_states.dtype

        q = self.q_proj(normed).view(1, suffix_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        s_k = self.k_proj(normed).view(1, suffix_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        s_v = self.v_proj(normed).view(1, suffix_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_out = lse_merge_attn(q, prefix_kv[0], prefix_kv[1], s_k, s_v,
                                  self.num_heads, self.num_kv_heads, self.head_dim)
        attn_out = self.o_proj(attn_out.to(model_dtype))

        hidden_states = (residual + attn_out).to(model_dtype)

        residual = hidden_states
        normed = self._rms_norm(hidden_states)
        gate = torch.nn.functional.silu(self.gate_proj(normed))
        up = self.up_proj(normed)
        hidden_states = self.down_proj(gate * up)
        hidden_states = (residual + hidden_states).to(model_dtype)

        return hidden_states


class MiniTransformer(nn.Module):
    def __init__(self, num_layers=24, hidden_size=2560, num_heads=20,
                 num_kv_heads=4, head_dim=128, vocab_size=8000):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            MiniTransformerLayer(hidden_size, num_heads, num_kv_heads, head_dim)
            for _ in range(num_layers)
        ])
        self.final_norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _rms_norm(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + 1e-6) * self.final_norm_weight).to(x.dtype)

    def forward(self, input_ids):
        hidden_states = self.embed(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self._rms_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, hidden_states

    def forward_with_prefix_sharing(self, provider_ids, reuser_suffix_ids, method="sdpa"):
        """Full-model PS with KV injection.

        method: "sdpa" (single call block-causal) or "lse" (two FA calls + merge)
        """
        prefix_len = provider_ids.shape[1] - reuser_suffix_ids.shape[1]

        # Provider forward, extracting prefix KV
        provider_hidden = self.embed(provider_ids)
        prefix_kv_per_layer = []

        for layer in self.layers:
            residual = provider_hidden
            normed = layer._rms_norm(provider_hidden)

            k_full = layer.k_proj(normed).view(1, -1, layer.num_kv_heads, layer.head_dim).permute(0, 2, 1, 3)
            v_full = layer.v_proj(normed).view(1, -1, layer.num_kv_heads, layer.head_dim).permute(0, 2, 1, 3)
            g = layer.num_heads // layer.num_kv_heads
            k_expanded = k_full.repeat_interleave(g, dim=1)
            v_expanded = v_full.repeat_interleave(g, dim=1)

            # Store prefix KV in unexpanded format (num_kv_heads, before GQA expand)
            prefix_kv_per_layer.append((k_full[:, :, :prefix_len, :].clone(),
                                        v_full[:, :, :prefix_len, :].clone()))

            q = layer.q_proj(normed).view(1, -1, layer.num_heads, layer.head_dim).permute(0, 2, 1, 3)
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k_expanded, v_expanded, is_causal=True)
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(1, -1, layer.num_heads * layer.head_dim)
            attn_out = layer.o_proj(attn_out)
            provider_hidden = residual + attn_out

            residual = provider_hidden
            normed = layer._rms_norm(provider_hidden)
            gate = torch.nn.functional.silu(layer.gate_proj(normed))
            up = layer.up_proj(normed)
            provider_hidden = residual + layer.down_proj(gate * up)

        provider_hidden = self._rms_norm(provider_hidden)
        provider_logits = self.lm_head(provider_hidden)

        # Reuser suffix-only forward with KV injection
        reuser_hidden = self.embed(reuser_suffix_ids)

        kv_injection_fn = layer.forward_with_kv_injection_sdpa if method == "sdpa" else layer.forward_with_kv_injection_lse
        for i, layer in enumerate(self.layers):
            prefix_kv = prefix_kv_per_layer[i]
            if method == "sdpa":
                reuser_hidden = layer.forward_with_kv_injection_sdpa(reuser_hidden, prefix_kv, prefix_len)
            else:
                reuser_hidden = layer.forward_with_kv_injection_lse(reuser_hidden, prefix_kv, prefix_len)

        reuser_hidden = self._rms_norm(reuser_hidden)
        reuser_logits = self.lm_head(reuser_hidden)

        return provider_logits, reuser_logits


def exp1_precision_validation():
    """Validate LSE merge precision against baseline and single SDPA block-causal."""
    model = MiniTransformer(num_layers=24, hidden_size=2560, num_heads=20,
                            num_kv_heads=4, head_dim=128, vocab_size=8000).to(DEVICE).to(DTYPE)
    results = []

    # Warmup
    warmup_ids = torch.randint(0, 8000, (1, 512), device=DEVICE)
    with torch.no_grad():
        logits, _ = model(warmup_ids)

    for prefix_len in [384, 1024]:
        suffix_len = 128 if prefix_len == 384 else 256
        total_len = prefix_len + suffix_len

        # Baseline: full forward
        test_ids = torch.randint(0, 8000, (1, total_len), device=DEVICE)
        with torch.no_grad():
            baseline_logits, _ = model(test_ids)

        # SDPA block-causal
        with torch.no_grad():
            _, sdpa_reuser_logits = model.forward_with_prefix_sharing(test_ids, test_ids[:, prefix_len:], method="sdpa")

        # LSE merge
        with torch.no_grad():
            _, lse_reuser_logits = model.forward_with_prefix_sharing(test_ids, test_ids[:, prefix_len:], method="lse")

        baseline_suffix = baseline_logits[:, prefix_len:, :]

        # SDPA vs baseline
        cos_sim_sdpa = torch.nn.functional.cosine_similarity(
            baseline_suffix.flatten().unsqueeze(0).float(),
            sdpa_reuser_logits.flatten().unsqueeze(0).float()
        ).item()
        max_diff_sdpa = (baseline_suffix.float() - sdpa_reuser_logits.float()).abs().max().item()

        # LSE merge vs baseline
        cos_sim_lse = torch.nn.functional.cosine_similarity(
            baseline_suffix.flatten().unsqueeze(0).float(),
            lse_reuser_logits.flatten().unsqueeze(0).float()
        ).item()
        max_diff_lse = (baseline_suffix.float() - lse_reuser_logits.float()).abs().max().item()
        mean_diff_lse = (baseline_suffix.float() - lse_reuser_logits.float()).abs().mean().item()

        # LSE merge vs SDPA (should be very close)
        cos_sim_lse_vs_sdpa = torch.nn.functional.cosine_similarity(
            sdpa_reuser_logits.flatten().unsqueeze(0).float(),
            lse_reuser_logits.flatten().unsqueeze(0).float()
        ).item()
        max_diff_lse_vs_sdpa = (sdpa_reuser_logits.float() - lse_reuser_logits.float()).abs().max().item()

        results.append({
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "total_len": total_len,
            "prefix_ratio": round(prefix_len / total_len, 2),
            "cos_sim_sdpa_vs_baseline": round(cos_sim_sdpa, 6),
            "max_diff_sdpa_vs_baseline": round(max_diff_sdpa, 4),
            "cos_sim_lse_vs_baseline": round(cos_sim_lse, 6),
            "max_diff_lse_vs_baseline": round(max_diff_lse, 4),
            "mean_diff_lse_vs_baseline": round(mean_diff_lse, 6),
            "cos_sim_lse_vs_sdpa": round(cos_sim_lse_vs_sdpa, 6),
            "max_diff_lse_vs_sdpa": round(max_diff_lse_vs_sdpa, 4),
        })

    del model
    torch.cuda.empty_cache()
    return results


def exp2_performance_comparison():
    """Compare SDPA math backend vs LSE merge timing."""
    model = MiniTransformer(num_layers=24, hidden_size=2560, num_heads=20,
                            num_kv_heads=4, head_dim=128, vocab_size=8000).to(DEVICE).to(DTYPE)
    results = []

    warmup_ids = torch.randint(0, 8000, (1, 512), device=DEVICE)
    with torch.no_grad():
        logits, _ = model(warmup_ids)

    for prefix_len in [384, 1024, 2048]:
        suffix_len = 128 if prefix_len == 384 else 256
        total_len = prefix_len + suffix_len

        # Baseline timing: single full forward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = torch.randint(0, 8000, (1, total_len), device=DEVICE)
            logits, _ = model(ids)
        torch.cuda.synchronize()
        baseline_ms = (time.perf_counter() - t0) * 1000

        # SDPA block-causal timing (1 provider + 1 reuser)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            provider_ids = torch.randint(0, 8000, (1, total_len), device=DEVICE)
            reuser_ids = torch.randint(0, 8000, (1, suffix_len), device=DEVICE)
            p_logits, r_logits = model.forward_with_prefix_sharing(provider_ids, reuser_ids, method="sdpa")
        torch.cuda.synchronize()
        sdpa_ms = (time.perf_counter() - t0) * 1000

        # LSE merge timing (1 provider + 1 reuser)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            provider_ids = torch.randint(0, 8000, (1, total_len), device=DEVICE)
            reuser_ids = torch.randint(0, 8000, (1, suffix_len), device=DEVICE)
            p_logits, r_logits = model.forward_with_prefix_sharing(provider_ids, reuser_ids, method="lse")
        torch.cuda.synchronize()
        lse_ms = (time.perf_counter() - t0) * 1000

        # Pure attention timing comparison (single layer)
        # Create test tensors for a single attention layer
        hidden_test = torch.randn(1, suffix_len, 2560, device=DEVICE, dtype=DTYPE)
        prefix_kv_test = (
            torch.randn(1, 4, prefix_len, 128, device=DEVICE, dtype=DTYPE),  # prefix K (unexpanded, num_kv_heads)
            torch.randn(1, 4, prefix_len, 128, device=DEVICE, dtype=DTYPE),  # prefix V (unexpanded, num_kv_heads)
        )
        normed_test = model.layers[0]._rms_norm(hidden_test)
        q_test = model.layers[0].q_proj(normed_test).view(1, suffix_len, 20, 128).permute(0, 2, 1, 3)
        s_k_test = model.layers[0].k_proj(normed_test).view(1, suffix_len, 4, 128).permute(0, 2, 1, 3)
        s_v_test = model.layers[0].v_proj(normed_test).view(1, suffix_len, 4, 128).permute(0, 2, 1, 3)

        # Single SDPA block-causal attention timing
        reps = 20
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(reps):
                single_sdpa_block_causal(q_test, prefix_kv_test[0], prefix_kv_test[1],
                                         s_k_test, s_v_test, 20, 4, 128, prefix_len)
        torch.cuda.synchronize()
        sdpa_attn_ms = (time.perf_counter() - t0) * 1000 / reps

        # LSE merge attention timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(reps):
                lse_merge_attn(q_test, prefix_kv_test[0], prefix_kv_test[1],
                               s_k_test, s_v_test, 20, 4, 128)
        torch.cuda.synchronize()
        lse_attn_ms = (time.perf_counter() - t0) * 1000 / reps

        results.append({
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "total_len": total_len,
            "baseline_ms": round(baseline_ms, 3),
            "sdpa_full_ms": round(sdpa_ms, 3),
            "lse_full_ms": round(lse_ms, 3),
            "sdpa_attn_ms": round(sdpa_attn_ms, 3),
            "lse_attn_ms": round(lse_attn_ms, 3),
            "attn_speedup_lse_vs_sdpa": round(sdpa_attn_ms / lse_attn_ms, 2) if lse_attn_ms > 0 else 0,
            "full_speedup_lse_vs_sdpa": round(sdpa_ms / lse_ms, 2) if lse_ms > 0 else 0,
        })

    del model
    torch.cuda.empty_cache()
    return results


def main():
    print("=" * 60)
    print("Block-Causal Mask Backend Benchmark — RTX 4090")
    print("SDPA math vs FlashAttention LSE merge")
    print("=" * 60)

    gpu_name = torch.cuda.get_device_name(DEVICE)
    gpu_mem = torch.cuda.get_device_properties(DEVICE).total_memory / 1e9
    print(f"GPU: {gpu_name}, Memory: {gpu_mem:.1f} GB")

    warmup(DEVICE)

    all_results = {"gpu": gpu_name, "gpu_mem_gb": round(gpu_mem, 1)}

    # Exp1: Precision
    print("\n--- Exp1: Precision Validation (SDPA vs LSE merge vs baseline) ---")
    r1 = exp1_precision_validation()
    for r in r1:
        print(f"  prefix={r['prefix_len']} ({r['prefix_ratio']}): "
              f"SDPA→baseline cos_sim={r['cos_sim_sdpa_vs_baseline']}, "
              f"LSE→baseline cos_sim={r['cos_sim_lse_vs_baseline']}, "
              f"LSE→SDPA cos_sim={r['cos_sim_lse_vs_sdpa']}, "
              f"LSE→baseline max_diff={r['max_diff_lse_vs_baseline']}")
    all_results["exp1_precision"] = r1

    # Exp2: Performance
    print("\n--- Exp2: Performance Comparison ---")
    r2 = exp2_performance_comparison()
    for r in r2:
        print(f"  prefix={r['prefix_len']}: "
              f"baseline={r['baseline_ms']}ms, "
              f"SDPA_full={r['sdpa_full_ms']}ms, LSE_full={r['lse_full_ms']}ms, "
              f"attn: SDPA={r['sdpa_attn_ms']}ms, LSE={r['lse_attn_ms']}ms, "
              f"attn_speedup={r['attn_speedup_lse_vs_sdpa']}x, "
              f"full_speedup={r['full_speedup_lse_vs_sdpa']}x")
    all_results["exp2_performance"] = r2

    # Save
    with open("block_causal_mask_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to block_causal_mask_benchmark_results.json")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nPrecision:")
    for r in r1:
        print(f"  prefix={r['prefix_len']}: LSE→baseline cos_sim={r['cos_sim_lse_vs_baseline']}, "
              f"max_diff={r['max_diff_lse_vs_baseline']}")

    print("\nPerformance (attention-only):")
    for r in r2:
        print(f"  prefix={r['prefix_len']}: SDPA={r['sdpa_attn_ms']}ms, "
              f"LSE merge={r['lse_attn_ms']}ms, speedup={r['attn_speedup_lse_vs_sdpa']}x")

    print("\nKEY INSIGHT:")
    print("  LSE merge uses two FlashAttention calls (fast backend) → should be faster than")
    print("  single SDPA call with float mask (math backend fallback). If validated,")
    print("  this is the production path for KV injection in verl.")


if __name__ == "__main__":
    main()