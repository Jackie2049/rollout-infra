#!/usr/bin/env python3
"""Debug Triton decode attention kernel correctness"""
import torch
import torch.nn.functional as F
import math
import sys
sys.path.insert(0, "/home/zxw/rollout-infra/tools")
from triton_decode_attn_benchmark_4090 import triton_decode_attn

# Simple test
B, n_heads, S, d_head = 1, 4, 8, 16
scale = 1.0 / math.sqrt(d_head)

Q = torch.randn(B, n_heads, 1, d_head, device="cuda", dtype=torch.float16)
K = torch.randn(B, n_heads, S, d_head, device="cuda", dtype=torch.float16)
V = torch.randn(B, n_heads, S, d_head, device="cuda", dtype=torch.float16)

# Reference: naive implementation (no causal mask needed for Q=1)
# For decode: Q has 1 token at position S, so it can see ALL previous K tokens
scores = Q @ K.transpose(-2, -1) * scale  # (B, n_heads, 1, S)
weights = torch.softmax(scores, dim=-1)  # (B, n_heads, 1, S)
naive_out = weights @ V  # (B, n_heads, 1, d_head)

# SDPA (should match naive for Q=1)
sdpa_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# Triton
triton_out = triton_decode_attn(Q, K, V, n_kv_heads=n_heads, scale=scale)

print("=== Debug Results ===")
print(f"SDPA[0,0,0,:8]:  {sdpa_out[0,0,0,:8].float()}")
print(f"Triton[0,0,0,:8]: {triton_out[0,0,0,:8].float()}")
print(f"Naive[0,0,0,:8]:  {naive_out[0,0,0,:8].float()}")
print(f"Max diff (SDPA vs Triton): {(sdpa_out.float() - triton_out.float()).abs().max().item():.6f}")
print(f"Max diff (Naive vs Triton): {(naive_out.float() - triton_out.float()).abs().max().item():.6f}")
print(f"Cos sim (SDPA vs Triton): {F.cosine_similarity(sdpa_out.flatten().float(), triton_out.flatten().float(), dim=0).item():.6f}")
print(f"Cos sim (Naive vs Triton): {F.cosine_similarity(naive_out.flatten().float(), triton_out.flatten().float(), dim=0).item():.6f}")