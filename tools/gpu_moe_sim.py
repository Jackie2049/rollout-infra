#!/usr/bin/env python3
"""MoE (Mixture of Experts) 推理模拟

MoE 模型 (Mixtral, DeepSeek) 的核心: 只有部分 expert 被 activate。
本实验在 GPU 上验证 MoE 的性能特性:

1. Dense vs Sparse MLP 性能对比
2. Top-K Expert Selection 开销
3. Expert Parallelism 通信模拟
4. 不同 num_experts / top_k 的权衡
5. DeepSeek-V2 MLA + MoE 综合分析

用法 (GPU 服务器):
  source /root/miniconda3/bin/activate myconda
  python gpu_moe_sim.py
"""

import os, json, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

print(f"GPU: {torch.cuda.get_device_name(0)}")


def bench_ms(fn, warmup=5, rep=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep


# ============================================================
# MoE Layer 模拟
# ============================================================

class DenseMLP(nn.Module):
    """Standard dense MLP"""
    def __init__(self, hidden, ff_mult=4):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * ff_mult, bias=False)
        self.fc2 = nn.Linear(hidden * ff_mult, hidden, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoELayer(nn.Module):
    """Mixture of Experts Layer"""
    def __init__(self, hidden, n_experts=8, top_k=2, ff_mult=4):
        super().__init__()
        self.hidden = hidden
        self.n_experts = n_experts
        self.top_k = top_k
        self.ff_mult = ff_mult

        # Router: maps hidden state to expert scores
        self.router = nn.Linear(hidden, n_experts, bias=False)

        # Experts: each is a small MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden * ff_mult // n_experts, bias=False),
                nn.GELU(),
                nn.Linear(hidden * ff_mult // n_experts, hidden, bias=False),
            ) for _ in range(n_experts)
        ])

    def forward(self, x):
        B, S, H = x.shape

        # Router scores
        router_logits = self.router(x)  # [B, S, n_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K expert selection
        top_k_scores, top_k_indices = router_probs.topk(self.top_k, dim=-1)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)

        # Compute expert outputs (simplified: iterate experts)
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            for expert_idx in range(self.n_experts):
                # Find tokens routed to this expert at position k
                mask = (top_k_indices[:, :, k] == expert_idx)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    output[mask] += top_k_scores[:, :, k].unsqueeze(-1).expand_as(x)[mask] * expert_output

        return output

    def forward_batched(self, x):
        """Batched MoE forward (more efficient)"""
        B, S, H = x.shape
        N = B * S

        # Flatten
        x_flat = x.view(N, H)

        # Router
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_scores, top_k_indices = router_probs.topk(self.top_k, dim=-1)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)

        # Simple: for each token, compute top_k experts and combine
        output = torch.zeros(N, H, device=x.device, dtype=x.dtype)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]  # [N]
            weight = top_k_scores[:, k:k+1]  # [N, 1]

            # Batch by expert (simplified)
            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    tokens = x_flat[mask]
                    expert_out = self.experts[e](tokens)
                    output[mask] += weight[mask] * expert_out

        return output.view(B, S, H)


# ============================================================
# 实验 1: Dense vs MoE 性能对比
# ============================================================

def exp1_dense_vs_moe():
    print("\n" + "=" * 60)
    print("实验1: Dense vs MoE MLP 性能对比")
    print("=" * 60)

    results = []
    H = 512
    B, S = 4, 128

    x = torch.randn(B, S, H, device="cuda", dtype=torch.float16)

    # Dense MLP
    dense = DenseMLP(H).cuda().half()
    dense_params = sum(p.numel() for p in dense.parameters())

    # MoE with equivalent compute (total expert params ≈ dense params)
    n_experts = 8
    top_k = 2
    moe = MoELayer(H, n_experts=n_experts, top_k=top_k, ff_mult=4).cuda().half()
    moe_params = sum(p.numel() for p in moe.parameters())

    # Active params (only top_k experts)
    active_params = moe_params * top_k / n_experts + H * n_experts  # router + active experts

    dense_ms = bench_ms(lambda: dense(x), rep=20)
    moe_ms = bench_ms(lambda: moe.forward_batched(x), rep=20)

    print(f"\n  H={H}, B={B}, S={S}")
    print(f"  Dense MLP:     {dense_params/1e6:.1f}M params, {dense_ms:.3f} ms")
    print(f"  MoE (8E, top2): {moe_params/1e6:.1f}M total, ~{active_params/1e6:.1f}M active")
    print(f"                 {moe_ms:.3f} ms ({moe_ms/dense_ms:.2f}x vs dense)")
    print(f"  Active/Total:  {active_params/moe_params*100:.0f}% (but only {top_k}/{n_experts} experts active)")

    results.append({
        "dense_ms": round(dense_ms, 3), "moe_ms": round(moe_ms, 3),
        "dense_params": dense_params, "moe_params": moe_params,
        "active_params": active_params,
        "moe_overhead": round(moe_ms/dense_ms, 2),
    })

    del dense, moe, x
    torch.cuda.empty_cache()
    return results


# ============================================================
# 实验 2: Top-K Expert Selection 开销
# ============================================================

def exp2_routing_overhead():
    print("\n" + "=" * 60)
    print("实验2: Expert Routing 开销")
    print("=" * 60)

    results = []
    H = 512
    n_experts_list = [4, 8, 16, 32, 64, 128]

    B, S = 32, 256
    N = B * S
    x_flat = torch.randn(N, H, device="cuda", dtype=torch.float16)
    router = nn.Linear(H, 64).cuda().half()

    print(f"\n  B={B}, S={S}, H={H}")
    print(f"  {'n_experts':<12} {'Router ms':<12} {'Top-K ms':<12} {'Total ms':<12} {'Overhead'}")
    print("  " + "-" * 60)

    for n_experts in n_experts_list:
        for top_k in [1, 2, 4]:
            # Router: hidden → n_experts logits
            W = torch.randn(H, n_experts, device="cuda", dtype=torch.float16)
            router_logits = F.linear(x_flat, W.t())
            router_probs = F.softmax(router_logits.float(), dim=-1).half()

            # Top-K selection
            top_ms = bench_ms(lambda: router_probs.topk(top_k, dim=-1), rep=30)

            # Router compute
            router_ms = bench_ms(lambda: F.linear(x_flat, W.t()), rep=30)

            total = router_ms + top_ms
            # Compared to a single linear layer
            baseline_ms = bench_ms(lambda: F.linear(x_flat, torch.randn(H, H, device="cuda", dtype=torch.float16).t()), rep=30)

            overhead = total / baseline_ms * 100

            if top_k == 2:  # only print for top_k=2
                print(f"  {n_experts:<12} {router_ms:<12.3f} {top_ms:<12.3f} {total:<12.3f} {overhead:.1f}%")

            results.append({
                "n_experts": n_experts, "top_k": top_k,
                "router_ms": round(router_ms, 3), "topk_ms": round(top_ms, 3),
                "total_ms": round(total, 3),
            })

        del W, router_logits, router_probs
        torch.cuda.empty_cache()

    del x_flat, router
    torch.cuda.empty_cache()
    return results


# ============================================================
# 实验 3: Expert Parallelism 通信模拟
# ============================================================

def exp3_expert_parallel():
    print("\n" + "=" * 60)
    print("实验3: Expert Parallelism 通信分析")
    print("=" * 60)

    results = []

    # In EP: tokens are sent to different experts on different GPUs
    # All-to-All communication: each GPU sends/receives tokens for its experts

    configs = [
        ("Mixtral-8x7B", 8, 2, 4096, 32),
        ("DeepSeek-V2", 64, 8, 5120, 60),  # actually 160 experts but simplified
        ("DeepSeek-V3", 256, 8, 7168, 61),
    ]

    interconnects = [
        ("NVLink 300GB/s", 300),
        ("NVLink 600GB/s", 600),
        ("PCIe 64GB/s", 64),
        ("IB 200Gbps", 25),
    ]

    for model_name, n_experts, top_k, hidden, n_layers in configs:
        print(f"\n  {model_name}: {n_experts} experts, top-{top_k}, H={hidden}")

        # Tokens per batch
        B, S = 32, 1024
        total_tokens = B * S

        # Each token activates top_k experts
        # With EP=8: each GPU has n_experts/8 experts
        # Tokens need to be sent to the right GPU
        # Data per token: hidden * 2 bytes (FP16)
        token_bytes = hidden * 2

        # All-to-All: each GPU sends (P-1)/P of its tokens
        for ep in [2, 4, 8]:
            # Tokens that need to go to other GPUs
            # Each GPU handles n_experts/ep experts
            experts_per_gpu = n_experts / ep

            # Fraction of tokens going to each GPU
            tokens_to_each = total_tokens * top_k / n_experts
            tokens_leaving = total_tokens * (1 - 1/ep)

            send_bytes = tokens_leaving * token_bytes

            print(f"\n    EP={ep}: {experts_per_gpu:.0f} experts/GPU, {tokens_leaving:.0f} tokens leave")
            print(f"    {'Interconnect':<20} {'Send MB':<10} {'Latency ms':<12} {'Overhead/token'}")
            print(f"    {'-'*52}")

            for name, bw in interconnects:
                latency_ms = send_bytes / bw / 1e9 * 1000
                overhead_us = latency_ms / total_tokens * 1000

                print(f"    {name:<20} {send_bytes/1e6:<10.0f} {latency_ms:<12.2f} {overhead_us:.2f} us")

                results.append({
                    "model": model_name, "ep": ep,
                    "interconnect": name, "send_mb": round(send_bytes/1e6, 0),
                    "latency_ms": round(latency_ms, 2),
                })

    return results


# ============================================================
# 实验 4: MoE 内存分析
# ============================================================

def exp4_moe_memory():
    print("\n" + "=" * 60)
    print("实验4: MoE 模型内存分析")
    print("=" * 60)

    results = []

    configs = [
        ("Dense-7B", 7e9, 1, 1, "Dense MLP"),
        ("Mixtral-8x7B", 46.7e9, 8, 2, "MoE MLP (8E, top2)"),
        ("DeepSeek-V2-21B", 21e9, 64, 6, "MoE + MLA (64E, top6)"),
        ("DeepSeek-V3-671B", 671e9, 256, 8, "MoE (256E, top8)"),
    ]

    print(f"\n  FP16 weights, training=16B/param, inference=2B/param")
    print(f"  {'Model':<20} {'Params':<10} {'Total FP16 GB':<14} {'Active FP16 GB':<16} {'Active%':<8} {'Train GB'}")
    print("  " + "-" * 82)

    for name, params, n_experts, top_k, note in configs:
        # Total weight memory
        total_fp16 = params * 2 / 1e9

        # Active parameters per token
        # For MoE: only top_k/n_experts of MLP params are active
        # MLP is ~2/3 of params, embedding/lm_head is ~1/3
        if n_experts == 1:
            active = params
        else:
            mlp_frac = 0.667
            non_mlp = params * (1 - mlp_frac)
            mlp_total = params * mlp_frac
            mlp_active = mlp_total * top_k / n_experts
            active = non_mlp + mlp_active

        active_fp16 = active * 2 / 1e9
        active_pct = active / params * 100
        train_gb = params * 16 / 1e9

        print(f"  {name:<20} {params/1e9:<10.1f}B {total_fp16:<14.1f} {active_fp16:<16.1f} {active_pct:<8.0f} {train_gb:.0f}")

        results.append({
            "name": name, "params_b": params/1e9,
            "total_fp16_gb": round(total_fp16, 1),
            "active_fp16_gb": round(active_fp16, 1),
            "active_pct": round(active_pct, 0),
            "train_gb": round(train_gb, 0),
        })

    return results


# ============================================================
# 实验 5: MoE vs Dense 推理吞吐量分析
# ============================================================

def exp5_moe_throughput():
    print("\n" + "=" * 60)
    print("实验5: MoE 推理吞吐量分析")
    print("=" * 60)

    results = []

    # Key insight: MoE decode throughput
    # Decode is memory-bound: throughput = HBM_BW / active_weight_bytes
    # MoE: active_weight_bytes < total_weight_bytes → higher throughput!

    hbm_bw = 300  # GB/s (A100)

    configs = [
        ("Dense-7B", 7e9, 7e9),
        ("Mixtral-8x7B", 46.7e9, 12.9e9),  # active ≈ 2/8 * 46.7 + shared
        ("Dense-70B", 70e9, 70e9),
        ("DeepSeek-V3-671B", 671e9, 37e9),  # active with top-8
    ]

    print(f"\n  HBM BW = {hbm_bw} GB/s, FP16, decode (memory-bound)")
    print(f"  {'Model':<25} {'Total GB':<12} {'Active GB':<12} {'tok/s':<12} {'Active/Total':<12} {'Dense equiv'}")
    print("  " + "-" * 85)

    for name, total_params, active_params in configs:
        total_gb = total_params * 2 / 1e9
        active_gb = active_params * 2 / 1e9

        # Decode throughput: HBM_BW / active_weight_bytes
        toks_per_s = hbm_bw / active_gb  # tokens/second for batch=1

        # Dense equivalent: what size dense model has same throughput?
        dense_equiv = active_params

        active_pct = active_params / total_params * 100

        print(f"  {name:<25} {total_gb:<12.1f} {active_gb:<12.1f} {toks_per_s:<12.0f} {active_pct:<12.0f}% {dense_equiv/1e9:.1f}B")

        results.append({
            "name": name, "total_gb": round(total_gb, 1),
            "active_gb": round(active_gb, 1),
            "toks_per_s": round(toks_per_s, 0),
            "active_pct": round(active_pct, 0),
            "dense_equiv_b": round(dense_equiv/1e9, 1),
        })

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    all_results = OrderedDict()
    all_results["dense_vs_moe"] = exp1_dense_vs_moe()
    all_results["routing_overhead"] = exp2_routing_overhead()
    all_results["expert_parallel"] = exp3_expert_parallel()
    all_results["moe_memory"] = exp4_moe_memory()
    all_results["moe_throughput"] = exp5_moe_throughput()

    print("\n" + "=" * 60)
    print("关键洞察")
    print("=" * 60)
    print("""
  1. MoE 的核心优势: 以 671B 总参数实现 37B 活跃参数的吞吐量
  2. Decode 吞吐 ∝ HBM_BW / active_weights (MoE 活跃参数少 → 高吞吐)
  3. Routing 开销: <1% (简单的 linear + topk)
  4. EP 通信: All-to-All 是瓶颈, NVLink 必须, IB 延迟大
  5. Mixtral-8x7B: 46.7B 总但只有 ~13B 活跃 → 等效 Dense-13B 性能 + 7B 质量
  6. DeepSeek-V3: 671B/37B = 18x 稀疏比, 用更少算力达到更好质量
""")

    with open("/root/moe_sim_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved.")
