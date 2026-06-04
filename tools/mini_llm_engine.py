#!/usr/bin/env python3
"""Mini LLM Inference Engine — Educational Implementation
==========================================================
A simplified but functional LLM inference engine that demonstrates:
1. Autoregressive decode loop
2. KV Cache management (Paged attention simulation)
3. Continuous batching
4. Greedy and temperature sampling
5. GQA support
6. Prefix caching

This runs on CPU and uses random weights (no model download needed).
Designed for learning — not production!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Model Components
# ============================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA, Mistral, etc.)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) — encodes position via rotation."""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, seq_len: int):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embedding to query/key tensors.
    x: [B, n_heads, S, head_dim]
    cos/sin: [S, head_dim] — already duplicated by RotaryEmbedding
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # cos/sin are [S, head_dim], slice first half for d
    cos_half = cos[:, :d].unsqueeze(0).unsqueeze(0)  # [1, 1, S, d]
    sin_half = sin[:, :d].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([x1 * cos_half - x2 * sin_half, x1 * sin_half + x2 * cos_half], dim=-1)
    return rotated


class Attention(nn.Module):
    """Grouped-Query Attention with KV Cache."""
    def __init__(self, hidden_dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_dim // n_heads
        self.n_rep = n_heads // n_kv_heads  # GQA repeat factor

        self.q_proj = nn.Linear(hidden_dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, hidden_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x, kv_cache=None, position_ids=None):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if position_ids is None:
            position_ids = torch.arange(S, device=x.device)
        cos, sin = self.rotary(q, S)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # KV Cache: append new K/V to cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        # GQA: expand KV heads to match Q heads
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, -1, self.head_dim)

        # Scaled dot-product attention
        total_len = k.shape[2]  # including cached KV
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Causal mask: only for prefill (S > 1)
        if S > 1:
            # Q has positions [0..S-1], K has positions [0..total_len-1]
            # Mask where key position > query position
            q_pos = torch.arange(total_len - S, total_len, device=x.device)
            k_pos = torch.arange(total_len, device=x.device)
            mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)  # [S, total_len]
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn.float(), dim=-1).type_as(q)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), new_kv_cache


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network (used in LLaMA)."""
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int):
        super().__init__()
        self.attn = Attention(hidden_dim, n_heads, n_kv_heads)
        self.ffn = FeedForward(hidden_dim, ffn_dim)
        self.attn_norm = RMSNorm(hidden_dim)
        self.ffn_norm = RMSNorm(hidden_dim)

    def forward(self, x, kv_cache=None, position_ids=None):
        h, new_kv_cache = self.attn(self.attn_norm(x), kv_cache, position_ids)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv_cache


class MiniLLM(nn.Module):
    """A minimal transformer language model (LLaMA-like architecture)."""
    def __init__(self, vocab_size=32000, hidden_dim=512, n_layers=4,
                 n_heads=8, n_kv_heads=2, ffn_dim=1376):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, n_kv_heads, ffn_dim)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, kv_caches=None, position_ids=None):
        x = self.embed(input_ids)
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = layer(x, cache, position_ids)
            new_kv_caches.append(new_cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kv_caches

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# KV Cache Manager
# ============================================================

class KVCacheManager:
    """Simple KV cache manager supporting prefix caching."""
    def __init__(self):
        self.caches = {}  # request_id -> list of (k, v) tensors

    def get(self, request_id):
        return self.caches.get(request_id, None)

    def set(self, request_id, kv_caches):
        self.caches[request_id] = kv_caches

    def delete(self, request_id):
        self.caches.pop(request_id, None)

    def get_prefix_cache(self, prefix_tokens, source_request_id=None):
        """Get KV cache for shared prefix (prefix caching)."""
        # In a real system, this would hash the prefix tokens
        # and look up in a shared cache. Here we just return None
        # (no cached prefix) for simplicity.
        return None


# ============================================================
# Inference Engine
# ============================================================

@dataclass
class Request:
    request_id: str
    prompt_tokens: list[int]
    max_tokens: int = 64
    temperature: float = 0.0  # 0 = greedy
    top_k: int = 0  # 0 = no top-k
    generated_tokens: list[int] = field(default_factory=list)
    finished: bool = False


class MiniInferenceEngine:
    """A mini inference engine with continuous batching support."""

    def __init__(self, model: MiniLLM, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.kv_manager = KVCacheManager()
        self.request_counter = 0

    def _sample(self, logits, temperature, top_k):
        """Sample from logits with temperature and top-k."""
        if temperature == 0:
            return logits.argmax(dim=-1)

        logits = logits / temperature
        if top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k_val)
            threshold = values[:, -1].unsqueeze(-1)  # [B, 1]
            logits = logits.masked_fill(logits < threshold, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        # Flatten to 2D for multinomial
        B = probs.shape[0]
        probs_2d = probs.view(B, -1)
        return torch.multinomial(probs_2d, num_samples=1).squeeze(-1)

    def add_request(self, prompt_tokens, max_tokens=64, temperature=0.0, top_k=0):
        """Add a new request to the engine."""
        self.request_counter += 1
        req = Request(
            request_id=str(self.request_counter),
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        return req

    def step(self, requests: list[Request]) -> dict[str, int]:
        """Run one inference step with continuous batching.

        Returns dict of request_id -> sampled token_id.
        """
        if not requests:
            return {}

        # Separate prefill and decode requests
        prefill_reqs = [r for r in requests if len(r.generated_tokens) == 0 and r.request_id not in self.kv_manager.caches]
        decode_reqs = [r for r in requests if r.request_id in self.kv_manager.caches]

        results = {}

        # Process prefill requests
        for req in prefill_reqs:
            input_ids = torch.tensor([req.prompt_tokens], device=self.device)
            with torch.no_grad():
                logits, kv_caches = self.model(input_ids)
            self.kv_manager.set(req.request_id, kv_caches)
            # Sample from last position
            next_token = self._sample(logits[:, -1:, :], req.temperature, req.top_k).item()
            next_token = min(next_token, 31999)
            req.generated_tokens.append(next_token)
            results[req.request_id] = next_token
            if next_token == 2 or len(req.generated_tokens) >= req.max_tokens:  # </s>
                req.finished = True

        # Process decode requests (batched)
        if decode_reqs:
            batch_ids = []
            for req in decode_reqs:
                batch_ids.append(req.generated_tokens[-1])
            input_ids = torch.tensor([batch_ids], device=self.device)
            position_ids = torch.tensor(
                [len(req.prompt_tokens) + len(req.generated_tokens) - 1 for req in decode_reqs],
                device=self.device
            )

            # Forward pass with per-request KV caches
            # (In a real engine, this would be batched with paged attention)
            for req in decode_reqs:
                token_id = torch.tensor([[req.generated_tokens[-1]]], device=self.device)
                # Decode: position = prompt_len + generated_so_far - 1
                seq_pos = len(req.prompt_tokens) + len(req.generated_tokens) - 1
                kv_caches = self.kv_manager.get(req.request_id)
                with torch.no_grad():
                    logits, new_kv_caches = self.model(token_id, kv_caches)
                self.kv_manager.set(req.request_id, new_kv_caches)
                next_token = self._sample(logits[:, -1:, :], req.temperature, req.top_k).item()
                # Clamp to valid vocab range
                next_token = min(next_token, 31999)
                req.generated_tokens.append(next_token)
                results[req.request_id] = next_token
                if next_token == 2 or len(req.generated_tokens) >= req.max_tokens:
                    req.finished = True

        return results

    def generate(self, prompt_tokens, max_tokens=64, temperature=0.0, top_k=0):
        """Simple generate API (single request, convenience wrapper)."""
        req = self.add_request(prompt_tokens, max_tokens, temperature, top_k)
        while not req.finished:
            self.step([req])
        return req.generated_tokens


# ============================================================
# Benchmark & Demo
# ============================================================

def demo_prefill_vs_decode():
    """Demo 1: Prefill vs Decode latency."""
    print("=" * 60)
    print("Demo 1: Prefill vs Decode Latency")
    print("=" * 60)

    model = MiniLLM(vocab_size=32000, hidden_dim=512, n_layers=4, n_heads=8, n_kv_heads=2)
    print(f"\n  Model parameters: {model.count_parameters():,}")
    engine = MiniInferenceEngine(model)

    # Prefill: process entire prompt
    prompt = list(range(128))
    start = time.time()
    req = engine.add_request(prompt, max_tokens=1)
    engine.step([req])
    prefill_ms = (time.time() - start) * 1000
    print(f"  Prefill (128 tokens): {prefill_ms:.1f} ms")

    # Decode: generate tokens one by one
    start = time.time()
    for _ in range(32):
        engine.step([req])
    decode_ms = (time.time() - start) * 1000
    print(f"  Decode (32 tokens): {decode_ms:.1f} ms ({decode_ms/32:.1f} ms/token)")
    print(f"  Ratio: prefill/decode = {prefill_ms / (decode_ms/32):.1f}x")


def demo_continuous_batching():
    """Demo 2: Continuous batching simulation."""
    print("\n" + "=" * 60)
    print("Demo 2: Continuous Batching")
    print("=" * 60)

    model = MiniLLM(vocab_size=32000, hidden_dim=256, n_layers=2, n_heads=4, n_kv_heads=2)
    engine = MiniInferenceEngine(model)

    # Simulate 4 requests arriving at different times
    requests = []
    for i in range(4):
        prompt = list(range(32 + i * 8))  # Different prompt lengths
        req = engine.add_request(prompt, max_tokens=8, temperature=0.8, top_k=50)
        requests.append(req)

    step_count = 0
    active = list(requests)
    finished = []

    while active:
        step_count += 1
        results = engine.step(active)
        newly_finished = [r for r in active if r.finished]
        for r in newly_finished:
            finished.append(r)
            active.remove(r)
            engine.kv_manager.delete(r.request_id)

    print(f"  Processed {len(requests)} requests in {step_count} steps")
    for r in finished:
        print(f"    Request {r.request_id}: prompt={len(r.prompt_tokens)}, generated={len(r.generated_tokens)} tokens")


def demo_gqa_vs_mha():
    """Demo 3: GQA vs MHA parameter comparison."""
    print("\n" + "=" * 60)
    print("Demo 3: GQA vs MHA Parameter & KV Cache Comparison")
    print("=" * 60)

    configs = [
        ("MHA (8 KV heads)", 8, 8),
        ("GQA (4 KV heads)", 8, 4),
        ("GQA (2 KV heads)", 8, 2),
        ("MQA (1 KV head)", 8, 1),
    ]

    hidden_dim = 512
    print(f"\n  Hidden dim={hidden_dim}, Q heads=8, 4 layers")
    print(f"  {'Config':>22} {'Params':>10} {'KV/tok':>10} {'KV(128tok)':>12}")
    print("  " + "-" * 58)

    for name, n_heads, n_kv in configs:
        model = MiniLLM(hidden_dim=hidden_dim, n_layers=4, n_heads=n_heads, n_kv_heads=n_kv, ffn_dim=1376)
        params = model.count_parameters()
        head_dim = hidden_dim // n_heads
        kv_per_tok = 2 * 4 * n_kv * head_dim * 2  # bytes (FP16)
        kv_128 = kv_per_tok * 128 / 1024
        print(f"  {name:>22} {params:>10,} {kv_per_tok:>8,}B {kv_128:>10.1f}KB")


def demo_sampling():
    """Demo 4: Sampling strategies."""
    print("\n" + "=" * 60)
    print("Demo 4: Sampling Strategies")
    print("=" * 60)

    model = MiniLLM(vocab_size=32000, hidden_dim=256, n_layers=2, n_heads=4, n_kv_heads=2)
    engine = MiniInferenceEngine(model)
    prompt = list(range(64))

    for temp in [0.0, 0.5, 1.0, 1.5]:
        tokens = engine.generate(prompt, max_tokens=16, temperature=temp, top_k=50 if temp > 0 else 0)
        unique = len(set(tokens))
        print(f"  T={temp:.1f}: generated {len(tokens)} tokens, {unique} unique ({unique/len(tokens)*100:.0f}% diversity)")


if __name__ == "__main__":
    print("Mini LLM Inference Engine — Educational Demo")
    print("=" * 60)

    demo_prefill_vs_decode()
    demo_continuous_batching()
    demo_gqa_vs_mha()
    demo_sampling()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
