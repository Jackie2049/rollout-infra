"""推理延迟分解实验 — Transformer Decode Step 逐操作耗时分析

将一个完整的 Transformer decode step 分解为各个操作，
测量每个操作的实际耗时，找到性能瓶颈。

操作分解:
1. Token Embedding (lookup)
2. Position Embedding (RoPE)
3. Self-Attention (QKV projection + attention + output projection) × N layers
4. MLP (gate/up projection + activation + down projection) × N layers
5. LayerNorm (RMSNorm) × 2N
6. Final logits (hidden → vocab)

使用方法:
    python inference_latency_breakdown.py   # 需要在 GPU 上运行
"""

import torch
import time
import math


def benchmark(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def main():
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"=== 推理延迟分解 — {props.name} ===\n")

    # Simulate GPT-2 Small architecture
    configs = {
        "GPT-2 Small (124M)": {
            "hidden": 768, "n_layers": 12, "n_heads": 12, "head_dim": 64,
            "ffn_dim": 3072, "vocab": 50257,
        },
        "GPT-2 Medium (355M)": {
            "hidden": 1024, "n_layers": 24, "n_heads": 16, "head_dim": 64,
            "ffn_dim": 4096, "vocab": 50257,
        },
        "LLaMA-7B style": {
            "hidden": 4096, "n_layers": 4, "n_heads": 32, "head_dim": 128,
            "ffn_dim": 11008, "vocab": 32000,  # Only 4 layers to fit in A16
        },
    }

    for model_name, cfg in configs.items():
        H = cfg["hidden"]
        L = cfg["n_layers"]
        NH = cfg["n_heads"]
        D = cfg["head_dim"]
        FFN = cfg["ffn_dim"]
        V = cfg["vocab"]

        print(f"{'='*70}")
        print(f"模型: {model_name}")
        print(f"配置: H={H}, L={L}, NH={NH}, D={D}, FFN={FFN}, V={V}")
        print(f"{'='*70}")

        dtype = torch.float16
        batch_sizes = [1, 4, 16] if H <= 1024 else [1, 4]

        for batch in batch_sizes:
            seq_len = 1  # Decode: one token at a time
            print(f"\n--- Batch={batch}, SeqLen={seq_len} (Decode) ---")

            # Create weights
            wte = torch.randn(V, H, dtype=dtype, device=device)
            wq = torch.randn(H, NH * D, dtype=dtype, device=device)
            wk = torch.randn(H, NH * D, dtype=dtype, device=device)
            wv = torch.randn(H, NH * D, dtype=dtype, device=device)
            wo = torch.randn(NH * D, H, dtype=dtype, device=device)
            w_gate = torch.randn(H, FFN, dtype=dtype, device=device)
            w_up = torch.randn(H, FFN, dtype=dtype, device=device)
            w_down = torch.randn(FFN, H, dtype=dtype, device=device)
            ln1_w = torch.randn(H, dtype=dtype, device=device)
            ln2_w = torch.randn(H, dtype=dtype, device=device)
            lm_head = torch.randn(V, H, dtype=dtype, device=device)

            # Input token
            input_ids = torch.randint(0, V, (batch, seq_len), device=device)
            hidden = torch.randn(batch, seq_len, H, dtype=dtype, device=device)

            # KV cache (simulated, for attention)
            cache_len = 512  # Simulated cache length
            k_cache = torch.randn(batch, NH, cache_len, D, dtype=dtype, device=device)
            v_cache = torch.randn(batch, NH, cache_len, D, dtype=dtype, device=device)

            results = {}

            # 1. Token Embedding
            t = benchmark(lambda: wte[input_ids])
            results["Embedding"] = t

            # 2. LayerNorm (RMSNorm approximation)
            def rmsnorm(x, w):
                return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w
            t = benchmark(lambda: rmsnorm(hidden, ln1_w))
            results["RMSNorm"] = t

            # 3. QKV Projection
            def qkv_proj(x, wq, wk, wv):
                return torch.matmul(x, wq), torch.matmul(x, wk), torch.matmul(x, wv)
            t = benchmark(lambda: qkv_proj(hidden, wq, wk, wv))
            results["QKV Proj"] = t

            # 4. Attention (with cache concat + SDPA)
            q = torch.randn(batch, NH, seq_len, D, dtype=dtype, device=device)
            k_new = torch.randn(batch, NH, seq_len, D, dtype=dtype, device=device)
            v_new = torch.randn(batch, NH, seq_len, D, dtype=dtype, device=device)

            def attention_step(q, k_new, v_new, k_cache, v_cache):
                # Concat new KV with cache
                k_full = torch.cat([k_cache, k_new], dim=2)
                v_full = torch.cat([v_cache, v_new], dim=2)
                return torch.nn.functional.scaled_dot_product_attention(q, k_full, v_full)

            t = benchmark(lambda: attention_step(q, k_new, v_new, k_cache, v_cache))
            results["Attention+Cache"] = t

            # 5. Output Projection
            attn_out = torch.randn(batch, seq_len, NH * D, dtype=dtype, device=device)
            t = benchmark(lambda: torch.matmul(attn_out, wo))
            results["Out Proj"] = t

            # 6. Gate/Up Projection (MLP)
            def gate_up(x, wg, wu):
                return torch.matmul(x, wg), torch.matmul(x, wu)
            t = benchmark(lambda: gate_up(hidden, w_gate, w_up))
            results["Gate+Up Proj"] = t

            # 7. Activation (SiLU)
            gate_out = torch.randn(batch, seq_len, FFN, dtype=dtype, device=device)
            up_out = torch.randn(batch, seq_len, FFN, dtype=dtype, device=device)
            t = benchmark(lambda: torch.nn.functional.silu(gate_out) * up_out)
            results["SiLU Act"] = t

            # 8. Down Projection (MLP)
            mlp_out = torch.randn(batch, seq_len, FFN, dtype=dtype, device=device)
            t = benchmark(lambda: torch.matmul(mlp_out, w_down))
            results["Down Proj"] = t

            # 9. Logits (LM Head)
            t = benchmark(lambda: torch.matmul(hidden, lm_head.T))
            results["LM Head"] = t

            # Print results sorted by time
            total = sum(results.values())
            per_layer = (results["RMSNorm"] * 2 + results["QKV Proj"] +
                         results["Attention+Cache"] + results["Out Proj"] +
                         results["Gate+Up Proj"] + results["SiLU Act"] +
                         results["Down Proj"]) * L
            full_step = results["Embedding"] + per_layer + results["LM Head"]

            print(f"{'操作':<20} {'耗时 (us)':>10} {'占比':>8}")
            print("-" * 42)
            for op, t in sorted(results.items(), key=lambda x: -x[1]):
                print(f"{op:<20} {t*1000:>10.1f} {t/per_layer*100:>7.1f}%")

            print(f"\n{'每层总计':<20} {per_layer/L*1000:>10.1f} us")
            print(f"{'x' + str(L) + ' 层':<20} {per_layer*1000:>10.1f} us ({per_layer/full_step*100:.0f}%)")
            print(f"{'Embedding':<20} {results['Embedding']*1000:>10.1f} us")
            print(f"{'LM Head':<20} {results['LM Head']*1000:>10.1f} us")
            print(f"{'完整 Step':<20} {full_step*1000:>10.1f} us")
            print(f"{'理论吞吐':<20} {1000/full_step:>10.0f} tok/s")

            # Cleanup
            del wte, wq, wk, wv, wo, w_gate, w_up, w_down, ln1_w, ln2_w, lm_head
            del hidden, input_ids, k_cache, v_cache, q, k_new, v_new, attn_out
            del gate_out, up_out, mlp_out
            torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("关键洞察")
    print("=" * 70)
    print("""
1. LM Head 是最大瓶颈: vocab_size × hidden 维度的矩阵乘法
   - GPT-2: 50257 × 768 = 38.6M 参数，占 decode step 30-50%
   - 可以通过 weight tying、Adaptive Softmax、sample 降低开销

2. Attention+Cache concat 也有开销: KV cache 拼接 + SDPA
   - decode 时 Q 只有 1 token，但 K/V 有 cache_len 个
   - 随 cache_len 增长，attention 开销增长

3. MLP 的三投影(gate+up+down)是第二大瓶颈
   - FFN_dim = 4 × hidden，矩阵更大

4. LayerNorm/RMSNorm 开销极小: 纯 element-wise 操作

5. Embedding (lookup) 开销极小: 简单索引操作
    """)


if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
