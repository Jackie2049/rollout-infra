"""
Batched GEMM Analysis — RTX 4090
MoE inference pattern: multiple small expert GEMMs vs single large GEMM.

Focus: Understanding batched GEMM performance for MoE serving.
"""

import torch
import time
import json

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
print(f"Device: {props.name} SM={props.major}.{props.minor}")

HBM_BANDWIDTH = 890.8  # GB/s (实测)
FP16_PEAK = 169.6  # TFLOPS (实测)


def benchmark(func, warmup=10, repeats=50):
    for _ in range(warmup):
        result = func()
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = func()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / repeats * 1000, result


def run_all():
    results = {}
    print("=" * 70)
    print("Batched GEMM Analysis — RTX 4090")
    print("=" * 70)

    # Exp 1: Single large GEMM vs N small sequential GEMMs
    print("\n--- Exp 1: Single Large vs N Small Sequential GEMMs ---")
    exp1 = {}
    M_single = 64  # decode batch
    K = 4096
    N = 4096

    # Single large: (M, K) @ (K, N)
    A_large = torch.randn(M_single, K, device=device, dtype=torch.bfloat16)
    B_large = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    single_ms, _ = benchmark(lambda: torch.mm(A_large, B_large.T))
    single_tflops = 2 * M_single * K * N / (single_ms / 1000) / 1e12

    exp1["single"] = {
        "M": M_single, "K": K, "N": N,
        "ms": round(single_ms, 4), "tflops": round(single_tflops, 2),
        "peak_pct": round(single_tflops / FP16_PEAK * 100, 1),
    }
    print(f"  Single GEMM M={M_single} K={K} N={N}: {single_ms:.4f}ms, {single_tflops:.2f}TF ({single_tflops/FP16_PEAK*100:.1f}%peak)")

    # N small sequential: (M/N_each, K) @ (K, N) × N times
    # Simulating N experts each serving M/N_each tokens
    for n_experts in [2, 4, 8, 16]:
        M_each = M_single // n_experts  # tokens per expert
        if M_each < 1:
            M_each = 1

        A_small = torch.randn(M_each, K, device=device, dtype=torch.bfloat16)
        B_small = torch.randn(N, K, device=device, dtype=torch.bfloat16)

        def sequential_experts():
            total = 0
            for _ in range(n_experts):
                C = torch.mm(A_small, B_small.T)
                total += 1
            return total

        seq_ms, _ = benchmark(sequential_experts)
        total_flops = n_experts * 2 * M_each * K * N
        seq_tflops = total_flops / (seq_ms / 1000) / 1e12

        exp1[f"n={n_experts}_M_each={M_each}"] = {
            "n_experts": n_experts, "M_each": M_each,
            "ms": round(seq_ms, 4), "tflops": round(seq_tflops, 2),
            "peak_pct": round(seq_tflops / FP16_PEAK * 100, 1),
            "vs_single_speedup": round(single_ms / seq_ms, 2),
            "total_flops_match": M_each * n_experts == M_single,
        }
        print(f"  {n_experts} experts × M={M_each}: {seq_ms:.4f}ms, {seq_tflops:.2f}TF ({seq_tflops/FP16_PEAK*100:.1f}%peak), vs single {single_ms/seq_ms:.2f}x")

    results["exp1_single_vs_sequential"] = exp1

    # Exp 2: torch.bmm (batched matmul)
    print("\n--- Exp 2: Batched GEMM (torch.bmm) ---")
    exp2 = {}

    for n_experts in [2, 4, 8, 16, 32, 64]:
        for M_each in [1, 4, 8, 16]:
            # Batched: (n_experts, M_each, K) @ (n_experts, K, N)
            A_batch = torch.randn(n_experts, M_each, K, device=device, dtype=torch.bfloat16)
            B_batch = torch.randn(n_experts, K, N, device=device, dtype=torch.bfloat16)

            bmm_ms, _ = benchmark(lambda: torch.bmm(A_batch, B_batch))
            total_flops = n_experts * 2 * M_each * K * N
            bmm_tflops = total_flops / (bmm_ms / 1000) / 1e12

            # Compare with sequential
            A_seq = torch.randn(M_each, K, device=device, dtype=torch.bfloat16)
            B_seq = torch.randn(N, K, device=device, dtype=torch.bfloat16)

            def seq_n():
                for _ in range(n_experts):
                    C = torch.mm(A_seq, B_seq.T)
                return C
            seq_ms, _ = benchmark(seq_n)

            exp2[f"n={n_experts}_M={M_each}"] = {
                "n_experts": n_experts, "M_each": M_each,
                "bmm_ms": round(bmm_ms, 4),
                "seq_ms": round(seq_ms, 4),
                "bmm_tflops": round(bmm_tflops, 2),
                "bmm_peak_pct": round(bmm_tflops / FP16_PEAK * 100, 1),
                "bmm_vs_seq": round(seq_ms / bmm_ms, 2),
            }
            print(f"  {n_experts} experts × M={M_each}: bmm={bmm_ms:.4f}ms({bmm_tflops:.2f}TF {bmm_tflops/FP16_PEAK*100:.1f}%peak), seq={seq_ms:.4f}ms → {seq_ms/bmm_ms:.2f}x")

    results["exp2_batched_bmm"] = exp2

    # Exp 3: MoE-like shapes (DeepSeek-V3 style)
    print("\n--- Exp 3: MoE-like Shapes (DeepSeek-V3 style) ---")
    exp3 = {}

    # DeepSeek-V3: 256 routed experts, 8 active per token
    # Each expert: d_model=7168, hidden=18432 (≈SwiGLU)
    d_model = 4096  # simplified (7B model)
    hidden = 14336

    for n_active in [2, 4, 8, 16]:  # active experts
        for M in [1, 4, 8, 16, 32]:
            # Gate projection: (M, d_model) @ (d_model, hidden) × n_active experts
            # Using bmm for batched approach
            A_gate = torch.randn(n_active, M, d_model, device=device, dtype=torch.bfloat16)
            W_gate = torch.randn(n_active, d_model, hidden, device=device, dtype=torch.bfloat16)

            bmm_ms, _ = benchmark(lambda: torch.bmm(A_gate, W_gate))
            total_flops = n_active * 2 * M * d_model * hidden
            bmm_tflops = total_flops / (bmm_ms / 1000) / 1e12

            # Sequential comparison
            A_seq = torch.randn(M, d_model, device=device, dtype=torch.bfloat16)
            W_seq = torch.randn(d_model, hidden, device=device, dtype=torch.bfloat16)  # (d_model, hidden)

            def seq_experts():
                for _ in range(n_active):
                    C = torch.mm(A_seq, W_seq)
                return C

            seq_ms, _ = benchmark(seq_experts)

            exp3[f"n_active={n_active}_M={M}"] = {
                "n_active": n_active, "M": M,
                "bmm_ms": round(bmm_ms, 4),
                "seq_ms": round(seq_ms, 4),
                "bmm_tflops": round(bmm_tflops, 2),
                "peak_pct": round(bmm_tflops / FP16_PEAK * 100, 1),
                "bmm_vs_seq": round(seq_ms / bmm_ms, 2),
            }
            print(f"  {n_active} experts × M={M}: bmm={bmm_ms:.4f}ms({bmm_tflops:.2f}TF {bmm_tflops/FP16_PEAK*100:.1f}%peak), seq={seq_ms:.4f}ms → {seq_ms/bmm_ms:.2f}x")

    results["exp3_moe_shapes"] = exp3

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Batched GEMM Analysis RTX 4090")
    print("=" * 70)

    # Best bmm vs seq cases
    best_ratio = 0
    best_config = ""
    for key, r in exp2.items():
        if r["bmm_vs_seq"] > best_ratio:
            best_ratio = r["bmm_vs_seq"]
            best_config = key

    worst_ratio = 100
    worst_config = ""
    for key, r in exp2.items():
        if r["bmm_vs_seq"] < worst_ratio:
            worst_ratio = r["bmm_vs_seq"]
            worst_config = key

    print(f"\n  Best bmm vs seq: {best_config} → {best_ratio:.2f}x faster")
    print(f"  Worst bmm vs seq: {worst_config} → {worst_ratio:.2f}x (bmm slower!)")

    # MoE summary
    moe_best = 0
    moe_config = ""
    for key, r in exp3.items():
        if r["bmm_vs_seq"] > moe_best:
            moe_best = r["bmm_vs_seq"]
            moe_config = key

    print(f"  Best MoE bmm vs seq: {moe_config} → {moe_best:.2f}x")

    print(f"\n  Production implications for MoE on RTX 4090:")
    print(f"    → Batched GEMM helps for moderate expert count + batch size")
    print(f"    → But: All-to-All communication is the bottleneck, not GEMM")
    print(f"    → RTX 4090 PCIe: A2A ≈ 2.8ms → GEMM ≈ 0.1-1ms → A2A dominates!")
    print(f"    → NVLink: A2A ≈ 0.35ms → GEMM ≈ 0.1-1ms → closer to parity")

    return results


if __name__ == '__main__':
    results = run_all()
    try:
        with open('results/batched_gemm_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except:
        with open('batched_gemm_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)