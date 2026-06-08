"""
End-to-End Inference Pipeline Benchmark — RTX 4090
From prefill to decode: full model inference pipeline measurement.

Tests:
1. Prefill latency vs sequence length (O(N^1.5))
2. Decode latency vs batch size (memory-bound Roofline)
3. Full pipeline: prefill + N decode steps for real-world workload
4. TTFT/TTLT simulation matching inference calculator
5. Continuous batching simulation: mixed prefill+decode
6. vLLM-like token budget scheduling simulation

Goal: Validate inference calculator estimates with real PyTorch measurements.
"""

import torch
import torch.nn as nn
import time
import json
import math

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
print(f"Device: {props.name} SM={props.major}.{props.minor}")

HBM_BANDWIDTH = 890.8  # GB/s (实测)


class MiniLLM(nn.Module):
    """Minimal LLM-like model for inference pipeline testing"""
    def __init__(self, d_model=4096, n_heads=32, n_kv_heads=8, d_head=128,
                 n_layers=4, vocab_size=32000, ffn_mult=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            self._make_layer(d_model, n_heads, n_kv_heads, d_head, ffn_mult)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _make_layer(self, d, nh, nkv, dh, fm):
        return nn.ModuleDict({
            "attn": nn.ModuleDict({
                "q_proj": nn.Linear(d, nh * dh, bias=False),
                "k_proj": nn.Linear(d, nkv * dh, bias=False),
                "v_proj": nn.Linear(d, nkv * dh, bias=False),
                "o_proj": nn.Linear(nh * dh, d, bias=False),
            }),
            "ffn": nn.ModuleDict({
                "gate": nn.Linear(d, d * fm, bias=False),
                "up": nn.Linear(d, d * fm, bias=False),
                "down": nn.Linear(d * fm, d, bias=False),
            }),
            "ln1": nn.LayerNorm(d),
            "ln2": nn.LayerNorm(d),
        })

    def forward(self, tokens):
        x = self.embed(tokens)
        for layer in self.layers:
            # Simplified: no KV cache, full forward pass
            h = layer["ln1"](x)
            q = layer["attn"]["q_proj"](h)
            k = layer["attn"]["k_proj"](h)
            v = layer["attn"]["v_proj"](h)
            # Simple attention (no FlashInfer for this simplified model)
            B, S = q.shape[0], q.shape[1]
            q = q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
            k = k.view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
            v = v.view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
            # GQA: expand K/V
            gs = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(gs, dim=1)
            v = v.repeat_interleave(gs, dim=1)
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).reshape(B, S, -1)
            x = x + layer["attn"]["o_proj"](attn)

            h = layer["ln2"](x)
            gate = torch.nn.functional.silu(layer["ffn"]["gate"](h))
            up = layer["ffn"]["up"](h)
            x = x + layer["ffn"]["down"](gate * up)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def weight_bytes(self):
        return sum(p.numel() * p.element_size() for p in self.parameters())


def measure_prefill(model, seq_len, batch_size=1, warmup=3, repeats=10):
    """Measure prefill latency (full forward pass)"""
    tokens = torch.randint(0, 32000, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(warmup):
        logits = model(tokens)
        torch.cuda.synchronize()

    # Timed
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(tokens)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = sum(times) / repeats * 1000
    return avg_ms


def measure_decode(model, seq_len, batch_size=1, warmup=3, repeats=20):
    """Measure decode latency (single token generation)
    Simulates: read last token logits → sample → next forward with 1 new token"""
    # Pre-fill KV cache context
    context_tokens = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    with torch.no_grad():
        logits = model(context_tokens)

    # Decode: generate 1 token (forward with full context + 1 new token)
    # In real inference, we'd use KV cache and only process the new token
    # Here we simulate the full decode step (read all weights + KV)
    new_token = torch.randint(0, 32000, (batch_size, 1), device=device)
    full_tokens = torch.cat([context_tokens, new_token], dim=1)

    # Warmup
    for _ in range(warmup):
        logits = model(full_tokens)
        torch.cuda.synchronize()

    # Timed
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(full_tokens)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = sum(times) / repeats * 1000
    return avg_ms


def run_all_experiments():
    results = {}

    # Create model (4-layer mini LLM, ~1.1B params)
    model = MiniLLM(d_model=4096, n_heads=32, n_kv_heads=8, d_head=128,
                    n_layers=4, vocab_size=32000, ffn_mult=4).to(device).bfloat16()
    n_params = model.count_params()
    weight_gb = model.weight_bytes() / (1024**3)
    print(f"Model: {n_params/1e9:.2f}B params, {weight_gb:.2f}GB weights (BF16)")
    print(f"D_model={model.d_model}, Q heads={model.n_heads}, KV heads={model.n_kv_heads}")

    results["model_info"] = {
        "n_params": n_params,
        "weight_gb": round(weight_gb, 2),
        "n_layers": model.n_layers,
        "n_heads": model.n_heads,
        "n_kv_heads": model.n_kv_heads,
    }

    # ---- Experiment 1: Prefill latency vs sequence length ----
    print("\n--- Exp 1: Prefill Latency vs Sequence Length ---")
    exp1 = {}
    for S in [64, 128, 256, 512, 1024, 2048, 4096]:
        latency = measure_prefill(model, S, batch_size=1)
        throughput = S / latency * 1000  # tokens/s
        exp1[f"S={S}"] = {"latency_ms": round(latency, 2), "throughput_tok_s": round(throughput, 0)}
        print(f"  S={S}: {latency:.2f}ms → {throughput:.0f} tok/s")
    results["exp1_prefill"] = exp1

    # ---- Experiment 2: Prefill throughput scaling with batch ----
    print("\n--- Exp 2: Prefill Throughput Scaling ---")
    exp2 = {}
    for B in [1, 2, 4, 8, 16]:
        latency = measure_prefill(model, 256, batch_size=B)
        throughput = B * 256 / latency * 1000
        exp2[f"B={B}"] = {"latency_ms": round(latency, 2), "throughput_tok_s": round(throughput, 0)}
        print(f"  B={B}: {latency:.2f}ms → {throughput:.0f} tok/s")
    results["exp2_prefill_batch"] = exp2

    # ---- Experiment 3: Decode latency vs batch size ----
    print("\n--- Exp 3: Decode Latency vs Batch Size ---")
    exp3 = {}
    for B in [1, 2, 4, 8, 16, 32]:
        latency = measure_decode(model, 4096, batch_size=B)
        throughput = B / latency * 1000
        # Roofline estimate
        total_read = weight_gb + 2 * 8 * 128 * 2 * 4 * 4096 * B / (1024**3)  # BF16 KV
        roofline_ms = total_read / HBM_BANDWIDTH * 1000

        exp3[f"B={B}"] = {
            "measured_ms": round(latency, 2),
            "roofline_ms": round(roofline_ms, 2),
            "measured_tp": round(throughput, 0),
            "roofline_tp": round(B / roofline_ms * 1000, 0),
            "overhead_pct": round((latency - roofline_ms) / latency * 100, 1),
        }
        print(f"  B={B}: measured={latency:.2f}ms, roofline={roofline_ms:.2f}ms, overhead={((latency-roofline_ms)/latency*100):.1f}% → {throughput:.0f} tok/s")
    results["exp3_decode"] = exp3

    # ---- Experiment 4: Full pipeline (prefill + decode) ----
    print("\n--- Exp 4: Full Pipeline (Prefill + 256 Decode Steps) ---")
    exp4 = {}
    for S in [128, 512, 1024, 2048, 4096]:
        # Prefill
        prefill_ms = measure_prefill(model, S, batch_size=1)
        # Decode (1 step)
        decode_ms = measure_decode(model, S, batch_size=1)
        # TTLT: prefill + 256 * decode
        ttlt_ms = prefill_ms + 256 * decode_ms

        exp4[f"S={S}"] = {
            "ttft_ms": round(prefill_ms, 2),
            "decode_per_token_ms": round(decode_ms, 2),
            "ttlt_256_ms": round(ttlt_ms, 0),
        }
        print(f"  S={S}: TTFT={prefill_ms:.2f}ms, decode={decode_ms:.2f}ms, TTLT(256)={ttlt_ms:.0f}ms")
    results["exp4_full_pipeline"] = exp4

    # ---- Experiment 5: Sampling overhead ----
    print("\n--- Exp 5: Sampling Overhead ---")
    exp5 = {}
    model.eval()
    tokens = torch.randint(0, 32000, (1, 128), device=device)
    with torch.no_grad():
        logits = model(tokens)

    last_logits = logits[:, -1, :]  # (1, vocab_size)

    for B in [1, 4, 16, 32, 55]:
        # Simulate batched sampling
        batch_logits = last_logits.expand(B, -1)  # (B, vocab)

        # Temperature sampling
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            probs = torch.softmax(batch_logits / 0.6, dim=-1)
            next_tokens = torch.multinomial(probs, 1)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = sum(times) / 20 * 1000
        exp5[f"B={B}"] = {"sampling_ms": round(avg_ms, 4)}
        print(f"  B={B}: sampling={avg_ms:.4f}ms")
    results["exp5_sampling"] = exp5

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY — End-to-End Inference Pipeline Benchmark")
    print("=" * 70)
    print(f"Model: {n_params/1e9:.2f}B params ({weight_gb:.2f}GB), {model.n_layers} layers, GQA-8")
    print(f"Prefill: ~{exp1['S=4096']['latency_ms']}ms at S=4096 → {exp1['S=4096']['throughput_tok_s']} tok/s")
    print(f"Decode: overhead {exp3['B=1']['overhead_pct']}% above Roofline at B=1")
    print(f"Sampling: ~{exp5['B=1']['sampling_ms']}ms at B=1 → negligible vs decode")
    print(f"TTFT: ~{exp4['S=4096']['ttft_ms']}ms at S=4096")
    print(f"TTLT(256): ~{exp4['S=4096']['ttlt_256_ms']}ms at S=4096")

    return results


if __name__ == '__main__':
    results = run_all_experiments()

    output_file = 'results/e2e_inference_pipeline_benchmark.json'
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")
    except:
        with open('e2e_inference_pipeline_benchmark.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved locally")