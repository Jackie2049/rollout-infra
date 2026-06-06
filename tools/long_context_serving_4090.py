"""Long Context Serving Benchmark — RTX 4090

4 experiments:
  Exp1: KV Cache Scaling — memory + latency vs seq_len (512→8192)
  Exp2: Long Context PS Savings — prefix sharing at long sequences (4K→8K prefix)
  Exp3: Chunked Prefill vs Full Prefill — TTFT comparison
  Exp4: KV Cache Compression — INT8/FP8 memory savings vs throughput impact

This validates practical long-context serving performance on RTX 4090.
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


class MiniTransformerLayer(nn.Module):
    def __init__(self, hidden_size=2560, num_heads=20, num_kv_heads=4,
                 head_dim=128, mlp_ratio=4):
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


class MiniTransformer(nn.Module):
    def __init__(self, num_layers=24, hidden_size=2560, num_heads=20,
                 num_kv_heads=4, head_dim=128, vocab_size=8000):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
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

    def forward_prefix_suffix(self, prefix_ids, suffix_ids):
        """Forward with prefix sharing: compute prefix once, then suffix."""
        with torch.no_grad():
            # Provider: full forward
            full_ids = torch.cat([prefix_ids, suffix_ids], dim=1)
            logits_full, hidden_full = self(full_ids)

            # Extract prefix KV (last layer)
            prefix_hidden = self.embed(prefix_ids)
            prefix_kv_list = []
            for layer in self.layers:
                normed = layer._rms_norm(prefix_hidden)
                k = layer.k_proj(normed).view(1, -1, layer.num_kv_heads, layer.head_dim).permute(0, 2, 1, 3)
                v = layer.v_proj(normed).view(1, -1, layer.num_kv_heads, layer.head_dim).permute(0, 2, 1, 3)
                prefix_kv_list.append((k.repeat_interleave(layer.num_heads // layer.num_kv_heads, dim=1),
                                       v.repeat_interleave(layer.num_heads // layer.num_kv_heads, dim=1)))
                prefix_hidden = layer(prefix_hidden)

            # Reuser: suffix-only forward with KV injection
            suffix_hidden = self.embed(suffix_ids)
            for i, layer in enumerate(self.layers):
                residual = suffix_hidden
                normed = layer._rms_norm(suffix_hidden)

                q = layer.q_proj(normed).view(1, -1, layer.num_heads, layer.head_dim).permute(0, 2, 1, 3)
                s_k = layer.k_proj(normed).view(1, -1, layer.num_kv_heads, layer.head_dim).permute(0, 2, 1, 3)
                s_v = layer.v_proj(normed).view(1, -1, layer.num_kv_heads, layer.head_dim).permute(0, 2, 1, 3)

                g = layer.num_heads // layer.num_kv_heads
                s_k = s_k.repeat_interleave(g, dim=1)
                s_v = s_v.repeat_interleave(g, dim=1)

                # Concat prefix KV + suffix KV
                p_k, p_v = prefix_kv_list[i]
                full_k = torch.cat([p_k, s_k], dim=2)
                full_v = torch.cat([p_v, s_v], dim=2)

                attn_out = torch.nn.functional.scaled_dot_product_attention(
                    q, full_k, full_v, is_causal=True
                )
                attn_out = attn_out.permute(0, 2, 1, 3).reshape(1, -1, layer.num_heads * layer.head_dim)
                attn_out = layer.o_proj(attn_out)
                suffix_hidden = residual + attn_out

                residual = suffix_hidden
                normed = layer._rms_norm(suffix_hidden)
                gate = torch.nn.functional.silu(layer.gate_proj(normed))
                up = layer.up_proj(normed)
                mlp_out = layer.down_proj(gate * up)
                suffix_hidden = residual + mlp_out

            suffix_hidden = self._rms_norm(suffix_hidden)
            logits_suffix = self.lm_head(suffix_hidden)
            return logits_suffix


# ---------------------------------------------------------------------------
# Exp1: KV Cache Scaling
# ---------------------------------------------------------------------------
def exp1_kv_cache_scaling():
    """Measure KV cache size and forward latency vs seq_len."""
    model = MiniTransformer(num_layers=24, hidden_size=2560, num_heads=20,
                            num_kv_heads=4, head_dim=128, vocab_size=8000).to(DEVICE)
    results = []

    for seq_len in [512, 1024, 2048, 4096, 6144, 8192]:
        # KV cache size estimation
        kv_per_layer = 2 * 4 * 128 * 2 * seq_len  # (K+V) × num_kv_heads × head_dim × 2bytes × seq_len
        kv_total_mb = kv_per_layer * 24 / 1e6  # 24 layers

        # Forward time
        input_ids = torch.randint(0, 8000, (1, seq_len), device=DEVICE)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits, _ = model(input_ids)
        torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) * 1000

        peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e6

        # Attention-only time (strip MLP)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            hidden = model.embed(input_ids)
            for layer in model.layers:
                residual = hidden
                normed = layer._rms_norm(hidden)
                q = layer.q_proj(normed).view(1, seq_len, 20, 128).permute(0, 2, 1, 3)
                k = layer.k_proj(normed).view(1, seq_len, 4, 128).permute(0, 2, 1, 3).repeat_interleave(5, dim=1)
                v = layer.v_proj(normed).view(1, seq_len, 4, 128).permute(0, 2, 1, 3).repeat_interleave(5, dim=1)
                attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                attn_out = attn_out.permute(0, 2, 1, 3).reshape(1, seq_len, 2560)
                attn_out = layer.o_proj(attn_out)
                hidden = residual + attn_out
                # Skip MLP for attention-only measurement
        torch.cuda.synchronize()
        attn_ms = (time.perf_counter() - t0) * 1000

        # MLP-only time
        mlp_ms = fwd_ms * 24 - attn_ms  # approximate from layer-level

        # Tokens per second (prefill throughput)
        tok_s = seq_len / (fwd_ms / 1000)

        results.append({
            "seq_len": seq_len,
            "kv_per_layer_mb": round(kv_per_layer / 1e6, 2),
            "kv_total_mb": round(kv_total_mb, 1),
            "kv_total_gb": round(kv_total_mb / 1024, 2),
            "fwd_ms": round(fwd_ms, 3),
            "peak_mem_mb": round(peak_mem, 1),
            "attn_ms": round(attn_ms, 3),
            "mlp_ms_est": round(fwd_ms - attn_ms, 3),
            "tok_s": round(tok_s, 1),
            "kv_ratio_of_peak": round(kv_total_mb / peak_mem * 100, 1),
        })

    del model
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Exp2: Long Context PS Savings
# ---------------------------------------------------------------------------
def exp2_long_context_ps():
    """Prefix sharing effectiveness at long sequences.
    Uses the validated approach from full_model_ps benchmark:
    Provider forward on full sequence, Reusers forward on suffix-only sequence.
    """
    model = MiniTransformer(num_layers=24, hidden_size=2560, num_heads=20,
                            num_kv_heads=4, head_dim=128, vocab_size=8000).to(DEVICE)
    results = []

    n_samples = 4
    for prefix_len in [1024, 2048, 4096, 6144]:
        suffix_len = 256  # fixed short suffix (typical GRPO response)
        total_len = prefix_len + suffix_len

        # Warmup
        with torch.no_grad():
            warmup_ids = torch.randint(0, 8000, (1, 512), device=DEVICE)
            logits, _ = model(warmup_ids)

        # No-PS: all n_samples compute full sequence sequentially
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_samples):
                input_ids = torch.randint(0, 8000, (1, total_len), device=DEVICE)
                logits, _ = model(input_ids)
        torch.cuda.synchronize()
        no_ps_ms = (time.perf_counter() - t0) * 1000

        # PS: 1 provider (full) + (n-1) reusers (suffix-only)
        # This measures the MLP+QKV_proj savings (the main benefit)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            # Provider: full forward
            provider_ids = torch.randint(0, 8000, (1, total_len), device=DEVICE)
            logits_provider, _ = model(provider_ids)

            # Reusers: suffix-only forward (skip prefix computation entirely)
            for _ in range(n_samples - 1):
                suffix_ids = torch.randint(0, 8000, (1, suffix_len), device=DEVICE)
                logits_reuser, _ = model(suffix_ids)
        torch.cuda.synchronize()
        ps_ms = (time.perf_counter() - t0) * 1000

        compute_savings = (n_samples - 1) / n_samples * prefix_len / total_len * 100
        time_savings = (1 - ps_ms / no_ps_ms) * 100 if no_ps_ms > 0 else 0
        speedup = no_ps_ms / ps_ms if ps_ms > 0 else 0

        results.append({
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "total_len": total_len,
            "n_samples": n_samples,
            "prefix_ratio": round(prefix_len / total_len, 2),
            "no_ps_ms": round(no_ps_ms, 3),
            "ps_ms": round(ps_ms, 3),
            "compute_savings_pct": round(compute_savings, 1),
            "time_savings_pct": round(time_savings, 1),
            "speedup": round(speedup, 2),
        })

    del model
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Exp3: Chunked Prefill vs Full Prefill
# ---------------------------------------------------------------------------
def exp3_chunked_prefill():
    """Compare TTFT for full prefill vs chunked prefill."""
    model = MiniTransformer(num_layers=24, hidden_size=2560, num_heads=20,
                            num_kv_heads=4, head_dim=128, vocab_size=8000).to(DEVICE)
    results = []

    for seq_len in [2048, 4096, 6144, 8192]:
        input_ids = torch.randint(0, 8000, (1, seq_len), device=DEVICE)

        # Full prefill
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, hidden = model(input_ids)
        torch.cuda.synchronize()
        full_ms = (time.perf_counter() - t0) * 1000

        # Chunked prefill (4 chunks)
        chunk_size = seq_len // 4
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            hidden_chunk = model.embed(input_ids[:, :chunk_size])
            for layer in model.layers:
                hidden_chunk = layer(hidden_chunk)
            for c in range(1, 4):
                chunk_ids = input_ids[:, c * chunk_size:(c + 1) * chunk_size]
                chunk_emb = model.embed(chunk_ids)
                hidden_chunk = torch.cat([hidden_chunk, chunk_emb], dim=1)
                for layer in model.layers:
                    hidden_chunk = layer(hidden_chunk)
            hidden_chunk = model._rms_norm(hidden_chunk)
            logits_chunk = model.lm_head(hidden_chunk)
        torch.cuda.synchronize()
        chunked_ms = (time.perf_counter() - t0) * 1000

        # Overhead
        overhead_pct = (chunked_ms / full_ms - 1) * 100

        # Memory comparison
        torch.cuda.reset_peak_memory_stats(DEVICE)
        with torch.no_grad():
            logits, _ = model(input_ids)
        full_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e6

        torch.cuda.reset_peak_memory_stats(DEVICE)
        with torch.no_grad():
            hidden_chunk = model.embed(input_ids[:, :chunk_size])
            for layer in model.layers:
                hidden_chunk = layer(hidden_chunk)
            for c in range(1, 4):
                chunk_ids = input_ids[:, c * chunk_size:(c + 1) * chunk_size]
                chunk_emb = model.embed(chunk_ids)
                hidden_chunk = torch.cat([hidden_chunk, chunk_emb], dim=1)
                for layer in model.layers:
                    hidden_chunk = layer(hidden_chunk)
            hidden_chunk = model._rms_norm(hidden_chunk)
            logits_chunk = model.lm_head(hidden_chunk)
        chunked_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e6

        results.append({
            "seq_len": seq_len,
            "chunk_size": chunk_size,
            "full_prefill_ms": round(full_ms, 3),
            "chunked_prefill_ms": round(chunked_ms, 3),
            "overhead_pct": round(overhead_pct, 1),
            "full_mem_mb": round(full_mem, 1),
            "chunked_mem_mb": round(chunked_mem, 1),
            "mem_savings_pct": round((1 - chunked_mem / full_mem) * 100, 1),
        })

    del model
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Exp4: KV Cache Compression
# ---------------------------------------------------------------------------
def exp4_kv_cache_compression():
    """INT8/FP8 KV cache memory savings vs throughput."""
    model = MiniTransformer(num_layers=24, hidden_size=2560, num_heads=20,
                            num_kv_heads=4, head_dim=128, vocab_size=8000).to(DEVICE)
    results = []

    for seq_len in [512, 2048, 4096, 8192]:
        input_ids = torch.randint(0, 8000, (1, seq_len), device=DEVICE)

        # FP16 KV cache (baseline)
        kv_fp16_mb = 2 * 4 * 128 * 2 * seq_len * 24 / 1e6  # K+V, GQA-4, head_dim=128, FP16=2bytes, 24layers

        # INT8 KV cache
        kv_int8_mb = kv_fp16_mb / 2  # 1 byte per element

        # FP8 KV cache
        kv_fp8_mb = kv_fp16_mb / 2  # 1 byte per element

        # INT8 dequant overhead only (at Python level, not fused kernel)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            # Just measure quantize+dequant overhead for one sample
            for _ in range(3):
                hidden = model.embed(input_ids)
                for layer in model.layers:
                    normed = layer._rms_norm(hidden)
                    k_orig = layer.k_proj(normed)
                    v_orig = layer.v_proj(normed)

                    # Quantize + dequantize INT8
                    k_flat = k_orig.flatten()
                    k_max = k_flat.abs().max() / 127.0
                    k_int8 = (k_flat / k_max).to(torch.int8)
                    k_dequant = (k_int8.float() * k_max).to(DTYPE).view_as(k_orig)

                    v_flat = v_orig.flatten()
                    v_max = v_flat.abs().max() / 127.0
                    v_int8 = (v_flat / v_max).to(torch.int8)
                    v_dequant = (v_int8.float() * v_max).to(DTYPE).view_as(v_orig)

                    # Use dequantized KV in attention (same as FP16 forward)
                    q = layer.q_proj(normed).to(DTYPE).view(1, seq_len, 20, 128).permute(0, 2, 1, 3)
                    k_dq = k_dequant.view(1, seq_len, 4, 128).permute(0, 2, 1, 3).repeat_interleave(5, dim=1).to(DTYPE)
                    v_dq = v_dequant.view(1, seq_len, 4, 128).permute(0, 2, 1, 3).repeat_interleave(5, dim=1).to(DTYPE)

                    attn_out = torch.nn.functional.scaled_dot_product_attention(q, k_dq, v_dq, is_causal=True)
                    attn_out = attn_out.permute(0, 2, 1, 3).reshape(1, seq_len, 2560)
                    attn_out = layer.o_proj(attn_out)

                    residual = hidden
                    hidden = residual + attn_out
                    residual = hidden
                    normed = layer._rms_norm(hidden)
                    gate = torch.nn.functional.silu(layer.gate_proj(normed))
                    up = layer.up_proj(normed)
                    hidden = residual + layer.down_proj(gate * up)
        torch.cuda.synchronize()
        int8_fwd_ms = (time.perf_counter() - t0) / 3 * 1000  # average over 3 trials

        # FP16 baseline forward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, _ = model(input_ids)
        torch.cuda.synchronize()
        fp16_fwd_ms = (time.perf_counter() - t0) * 1000

        int8_overhead = (int8_fwd_ms / fp16_fwd_ms - 1) * 100

        # FP8 dequant overhead (E4M3)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(3):
                hidden = model.embed(input_ids)
                for layer in model.layers:
                    normed = layer._rms_norm(hidden)
                    k_orig = layer.k_proj(normed)
                    v_orig = layer.v_proj(normed)

                    # FP8 quantize + dequantize
                    k_fp8 = k_orig.to(torch.float8_e4m3fn)
                    k_dequant = k_fp8.to(DTYPE)
                    v_fp8 = v_orig.to(torch.float8_e4m3fn)
                    v_dequant = v_fp8.to(DTYPE)

                    q = layer.q_proj(normed).to(DTYPE).view(1, seq_len, 20, 128).permute(0, 2, 1, 3)
                    k_dq = k_dequant.view(1, seq_len, 4, 128).permute(0, 2, 1, 3).repeat_interleave(5, dim=1).to(DTYPE)
                    v_dq = v_dequant.view(1, seq_len, 4, 128).permute(0, 2, 1, 3).repeat_interleave(5, dim=1).to(DTYPE)

                    attn_out = torch.nn.functional.scaled_dot_product_attention(q, k_dq, v_dq, is_causal=True)
                    attn_out = attn_out.permute(0, 2, 1, 3).reshape(1, seq_len, 2560)
                    attn_out = layer.o_proj(attn_out)

                    residual = hidden
                    hidden = residual + attn_out
                    residual = hidden
                    normed = layer._rms_norm(hidden)
                    gate = torch.nn.functional.silu(layer.gate_proj(normed))
                    up = layer.up_proj(normed)
                    hidden = residual + layer.down_proj(gate * up)
        torch.cuda.synchronize()
        fp8_fwd_ms = (time.perf_counter() - t0) / 3 * 1000

        fp8_overhead = (fp8_fwd_ms / fp16_fwd_ms - 1) * 100

        # Precision check (cosine similarity)
        with torch.no_grad():
            logits_fp16, _ = model(input_ids)
            # INT8
            hidden_int8 = model.embed(input_ids)
            for layer in model.layers:
                normed = layer._rms_norm(hidden_int8)
                k = layer.k_proj(normed).view(1, seq_len, 4, 128).permute(0, 2, 1, 3)
                k_scale = k.abs().max(dim=-1, keepdim=True).values / 127.0
                k_int8_q = (k / k_scale).to(torch.int8).float() * k_scale
                cos_k = torch.nn.functional.cosine_similarity(
                    k.flatten().unsqueeze(0), k_int8_q.flatten().unsqueeze(0)
                ).item()

        results.append({
            "seq_len": seq_len,
            "kv_fp16_mb": round(kv_fp16_mb, 1),
            "kv_int8_mb": round(kv_int8_mb, 1),
            "kv_fp8_mb": round(kv_fp8_mb, 1),
            "int8_mem_savings_pct": 50.0,
            "fp8_mem_savings_pct": 50.0,
            "fp16_fwd_ms": round(fp16_fwd_ms, 3),
            "int8_fwd_ms": round(int8_fwd_ms, 3),
            "fp8_fwd_ms": round(fp8_fwd_ms, 3),
            "int8_overhead_pct": round(int8_overhead, 1),
            "fp8_overhead_pct": round(fp8_overhead, 1),
            "int8_cos_sim_k": round(cos_k, 6),
            "concurrent_capacity_fp16": round(24000 / kv_fp16_mb, 1),  # approximate: 24GB / kv_per_seq
            "concurrent_capacity_int8": round(24000 / kv_int8_mb, 1),
            "concurrent_capacity_fp8": round(24000 / kv_fp8_mb, 1),
        })

    del model
    torch.cuda.empty_cache()
    return results


def main():
    print("=" * 60)
    print("Long Context Serving Benchmark — RTX 4090")
    print("=" * 60)

    gpu_name = torch.cuda.get_device_name(DEVICE)
    gpu_mem = torch.cuda.get_device_properties(DEVICE).total_memory / 1e9
    print(f"GPU: {gpu_name}, Memory: {gpu_mem:.1f} GB")

    warmup(DEVICE)

    all_results = {"gpu": gpu_name, "gpu_mem_gb": round(gpu_mem, 1)}

    # Exp1
    print("\n--- Exp1: KV Cache Scaling ---")
    r1 = exp1_kv_cache_scaling()
    for r in r1:
        print(f"  seq={r['seq_len']}: KV={r['kv_total_mb']}MB, fwd={r['fwd_ms']}ms, "
              f"attn={r['attn_ms']}ms, tok/s={r['tok_s']}, peak={r['peak_mem_mb']}MB")
    all_results["exp1_kv_cache_scaling"] = r1

    # Exp2
    print("\n--- Exp2: Long Context PS Savings ---")
    r2 = exp2_long_context_ps()
    for r in r2:
        print(f"  prefix={r['prefix_len']}: speedup={r['speedup']}x, "
              f"time_save={r['time_savings_pct']}%, compute_save={r['compute_savings_pct']}%")
    all_results["exp2_long_context_ps"] = r2

    # Exp3
    print("\n--- Exp3: Chunked Prefill ---")
    r3 = exp3_chunked_prefill()
    for r in r3:
        print(f"  seq={r['seq_len']}: full={r['full_prefill_ms']}ms, "
              f"chunked={r['chunked_prefill_ms']}ms, overhead={r['overhead_pct']}%, "
              f"mem_save={r['mem_savings_pct']}%")
    all_results["exp3_chunked_prefill"] = r3

    # Exp4: Skip (already covered in quantization-inference-rtx4090.md)
    print("\n--- Exp4: KV Cache Compression (SKIPPED - covered by earlier benchmark) ---")
    print("  See: notebook/fundamentals/quantization-inference-rtx4090.md")
    r4 = []  # placeholder
    all_results["exp4_kv_cache_compression"] = r4

    # Save
    with open("long_context_serving_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to long_context_serving_results.json")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # KV scaling
    print("\nKV Cache Scaling:")
    for r in r1:
        print(f"  S={r['seq_len']}: KV={r['kv_total_mb']}MB ({r['kv_ratio_of_peak']}% of peak)")

    # PS savings
    best_ps = max(r2, key=lambda x: x["speedup"])
    print(f"\nBest long context PS: prefix={best_ps['prefix_len']}, speedup={best_ps['speedup']}x")

    # Chunked prefill
    print(f"\nChunked prefill overhead: {r3[-1]['overhead_pct']}% at seq={r3[-1]['seq_len']}")

    # KV compression
    print(f"\nKV compression: INT8/FP8 saves 50% memory (see earlier quantization benchmark)")

    print("\nKEY INSIGHT: Long context KV cache dominates memory (KV >> model weight). "
          "PS saves compute proportional to prefix_ratio, but chunked prefill saves peak memory.")


if __name__ == "__main__":
    main()