#!/usr/bin/env python3
"""GQA/MQA Decode KV Cache Bandwidth Analysis"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch, torch.nn.functional as F, time

print("=== GQA/MQA Decode KV Cache Bandwidth ===")
print(f"GPU: {torch.cuda.get_device_name(0)}")

B, kv_len, D = 16, 2048, 64

print(f"\nDecode: batch={B}, kv_len={kv_len}, D={D}")
print(f"{'Config':<25} {'KV_Heads':<10} {'KV_MB':<10} {'ms':<10} {'BW_GB/s':<10} {'vs_MHA'}")
print("-" * 75)

configs = [
    ("MHA (16 KV)", 16),
    ("GQA (8 KV)", 8),
    ("GQA (4 KV)", 4),
    ("GQA (2 KV)", 2),
    ("MQA (1 KV)", 1),
]

mha_bw = None
for name, n_kv in configs:
    n_q = 16
    q = torch.randn(B, n_q, 1, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, n_kv, kv_len, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, n_kv, kv_len, D, device="cuda", dtype=torch.float16)

    n_rep = n_q // n_kv
    if n_rep > 1:
        k = k.unsqueeze(2).expand(B, n_kv, n_rep, kv_len, D).reshape(B, n_q, kv_len, D)
        v = v.unsqueeze(2).expand(B, n_kv, n_rep, kv_len, D).reshape(B, n_q, kv_len, D)

    for _ in range(10):
        F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(100):
        F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 100 * 1000

    kv_bytes = 2 * B * n_kv * kv_len * D * 2
    kv_mb = kv_bytes / 1e6
    bw = kv_bytes / ms / 1e6

    if mha_bw is None:
        mha_bw = bw

    vs_mha = f"{bw/mha_bw:.2f}x" if n_kv < 16 else "1.00x"
    print(f"  {name:<23} {n_kv:<10} {kv_mb:<10.2f} {ms:<10.4f} {bw:<10.1f} {vs_mha}")

    del q, k, v
    torch.cuda.empty_cache()

# KV Cache memory comparison
print(f"\n=== KV Cache Memory per Token (FP16) ===")
print(f"  {'Model':<25} {'KV_Heads':<10} {'D':<6} {'bytes/tok':<12} {'S=2K_MB':<10} {'S=32K_MB':<10}")
print("  " + "-" * 83)

models = [
    ("LLaMA-7B MHA", 32, 128),
    ("LLaMA-7B GQA-4", 8, 128),
    ("LLaMA-7B MQA", 1, 128),
    ("Mixtral-8x7B GQA", 8, 128),
    ("DeepSeek-V2 MLA", 1, 512),
]

for name, n_kv, d in models:
    bpt = 2 * n_kv * d * 2
    s2k = bpt * 2048 / 1e6
    s32k = bpt * 32768 / 1e6
    print(f"  {name:<25} {n_kv:<10} {d:<6} {bpt:<12} {s2k:<10.1f} {s32k:<10.1f}")

print("\nDone!")
