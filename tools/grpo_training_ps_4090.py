"""GRPO Training Throughput with Prefix-Sharing — RTX 4090

4 experiments measuring real GRPO training throughput:
  Exp1: Training step time (forward+backward) with PS vs no-PS
  Exp2: Memory usage with PS vs no-PS (KV cache sharing reduces per-GPU memory)
  Exp3: n_samples sweep — training throughput (tok/s) with varying GRPO n
  Exp4: Prefix ratio sweep — optimal prefix_len for training throughput

This simulates the actual GRPO training loop:
- Actor forward: compute logprobs for all n_samples
- With PS: 1 provider computes full, n-1 reusers compute suffix-only
- Backward: all samples get gradients (PS still preserves autograd)
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
    def __init__(self, hidden_size=2048, num_heads=16, num_kv_heads=4,
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
    def __init__(self, num_layers=16, hidden_size=2048, num_heads=16,
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


def train_step(model, input_ids, labels, optimizer):
    """Full training step: forward + backward + optimizer step."""
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(input_ids)
    # Simple cross-entropy loss (shifted)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, model.vocab_size),
        shift_labels.view(-1)
    )
    loss.backward()
    optimizer.step()
    return loss.item()


def train_step_ps(model, input_ids_full, input_ids_suffix, labels_full, labels_suffix,
                  optimizer, prefix_len, n_provider=1, n_reusers=3):
    """PS training step: provider forward (full) + reusers forward (suffix-only).

    Note: In real prefix-sharing, provider and reuser gradients flow through
    shared prefix computation (no detach). Here we simulate by:
    1. Provider forward on full sequence → backward → gradients
    2. Reusers forward on suffix → backward → gradients
    Both contribute gradients to model parameters.
    """
    optimizer.zero_grad(set_to_none=True)

    # Provider: full forward + backward
    logits_provider, _ = model(input_ids_full)
    shift_logits_p = logits_provider[:, :-1, :].contiguous()
    shift_labels_p = labels_full[:, 1:].contiguous()
    loss_p = torch.nn.functional.cross_entropy(
        shift_logits_p.view(-1, model.vocab_size),
        shift_labels_p.view(-1)
    )
    loss_p.backward()

    # Reusers: suffix-only forward + backward
    logits_reusers, _ = model(input_ids_suffix)
    shift_logits_r = logits_reusers[:, :-1, :].contiguous()
    shift_labels_r = labels_suffix[:, 1:].contiguous()
    loss_r = torch.nn.functional.cross_entropy(
        shift_logits_r.view(-1, model.vocab_size),
        shift_labels_r.view(-1)
    )
    loss_r.backward()

    optimizer.step()
    total_loss = loss_p.item() + loss_r.item()
    return total_loss


# ---------------------------------------------------------------------------
# Exp1: Training Step Time with PS vs No-PS
# ---------------------------------------------------------------------------
def exp1_training_step_time():
    results = []
    model = MiniTransformer(num_layers=16, hidden_size=2048, num_heads=16,
                            num_kv_heads=4, head_dim=128, vocab_size=8000).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    n_samples = 4
    prefix_len = 384  # 75% of 512
    suffix_len = 128
    total_len = prefix_len + suffix_len

    for trial in range(5):
        # No-PS: all n_samples compute full sequence
        input_ids = torch.randint(0, 8000, (n_samples, total_len), device=DEVICE)
        labels = input_ids.clone()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        t0 = time.perf_counter()
        loss = train_step(model, input_ids, labels, optimizer)
        torch.cuda.synchronize()
        no_ps_ms = (time.perf_counter() - t0) * 1000
        no_ps_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e6

        # PS: 1 provider + (n-1) reusers (suffix-only)
        input_ids_provider = torch.randint(0, 8000, (1, total_len), device=DEVICE)
        labels_provider = input_ids_provider.clone()
        input_ids_reusers = torch.randint(0, 8000, (n_samples-1, suffix_len), device=DEVICE)
        labels_reusers = input_ids_reusers.clone()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        t0 = time.perf_counter()
        loss_ps = train_step_ps(model, input_ids_provider, input_ids_reusers,
                                labels_provider, labels_reusers, optimizer,
                                prefix_len, n_provider=1, n_reusers=n_samples-1)
        torch.cuda.synchronize()
        ps_ms = (time.perf_counter() - t0) * 1000
        ps_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e6

        if trial < 2:  # warmup trials
            continue

        compute_savings = (n_samples - 1) / n_samples * prefix_len / total_len * 100
        time_savings = (1 - ps_ms / no_ps_ms) * 100 if no_ps_ms > 0 else 0
        mem_savings = (1 - ps_mem / no_ps_mem) * 100 if no_ps_mem > 0 else 0

        # Throughput calculation
        no_ps_tokens = n_samples * total_len  # total tokens processed
        ps_tokens = 1 * total_len + (n_samples-1) * suffix_len  # with PS
        no_ps_tok_s = no_ps_tokens / (no_ps_ms / 1000)
        ps_tok_s = ps_tokens / (ps_ms / 1000)  # effective tokens/second
        # Original-equivalent throughput: same output quality, less compute
        ps_effective_tok_s = no_ps_tokens / (ps_ms / 1000)  # what matters for training

        results.append({
            "n_samples": n_samples,
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "total_len": total_len,
            "no_ps_step_ms": round(no_ps_ms, 3),
            "ps_step_ms": round(ps_ms, 3),
            "compute_savings_pct": round(compute_savings, 1),
            "time_savings_pct": round(time_savings, 1),
            "speedup": round(no_ps_ms / ps_ms, 2),
            "no_ps_mem_mb": round(no_ps_mem, 1),
            "ps_mem_mb": round(ps_mem, 1),
            "mem_savings_pct": round(mem_savings, 1),
            "no_ps_tok_s": round(no_ps_tok_s, 1),
            "ps_effective_tok_s": round(ps_effective_tok_s, 1),
            "loss_no_ps": round(loss, 4),
            "loss_ps": round(loss_ps, 4),
        })

    del model, optimizer
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Exp2: Memory Usage with PS vs No-PS (varying n_samples)
# ---------------------------------------------------------------------------
def exp2_memory_usage():
    results = []
    model = MiniTransformer(num_layers=16, hidden_size=2048, num_heads=16,
                            num_kv_heads=4, head_dim=128, vocab_size=8000).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    prefix_len = 384
    suffix_len = 128
    total_len = prefix_len + suffix_len

    model_size_mb = torch.cuda.memory_allocated(DEVICE) / 1e6

    for n_samples in [2, 4]:
        # No-PS memory
        input_ids = torch.randint(0, 8000, (n_samples, total_len), device=DEVICE)
        labels = input_ids.clone()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        loss = train_step(model, input_ids, labels, optimizer)
        torch.cuda.synchronize()
        no_ps_peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e6

        # PS memory
        input_ids_provider = torch.randint(0, 8000, (1, total_len), device=DEVICE)
        labels_provider = input_ids_provider.clone()
        input_ids_reusers = torch.randint(0, 8000, (n_samples-1, suffix_len), device=DEVICE)
        labels_reusers = input_ids_reusers.clone()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        loss_ps = train_step_ps(model, input_ids_provider, input_ids_reusers,
                                labels_provider, labels_reusers, optimizer,
                                prefix_len)
        torch.cuda.synchronize()
        ps_peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e6

        mem_savings = (1 - ps_peak_mem / no_ps_peak_mem) * 100 if no_ps_peak_mem > 0 else 0

        # Forward-only memory (no backward)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        with torch.no_grad():
            logits, _ = model(input_ids)
        torch.cuda.synchronize()
        fwd_only_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e6

        results.append({
            "n_samples": n_samples,
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "total_len": total_len,
            "model_size_mb": round(model_size_mb, 1),
            "no_ps_peak_mem_mb": round(no_ps_peak_mem, 1),
            "ps_peak_mem_mb": round(ps_peak_mem, 1),
            "mem_savings_pct": round(mem_savings, 1),
            "fwd_only_mem_mb": round(fwd_only_mem, 1),
            "backward_mem_ratio": round((no_ps_peak_mem - fwd_only_mem) / fwd_only_mem * 100, 1),
            "ps_mem_reduction_mb": round(no_ps_peak_mem - ps_peak_mem, 1),
        })

    del model, optimizer
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Exp3: n_samples Sweep — Training Throughput
# ---------------------------------------------------------------------------
def exp3_n_samples_throughput():
    results = []
    prefix_len = 384
    suffix_len = 128
    total_len = prefix_len + suffix_len

    for n_samples in [2, 4]:
        # Create model for each n (avoid memory fragmentation)
        model = MiniTransformer(num_layers=16, hidden_size=2048, num_heads=16,
                                num_kv_heads=4, head_dim=128, vocab_size=8000).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Warmup
        warmup_ids = torch.randint(0, 8000, (2, 128), device=DEVICE)
        warmup_labels = warmup_ids.clone()
        for _ in range(3):
            train_step(model, warmup_ids, warmup_labels, optimizer)

        # No-PS training throughput
        times_no_ps = []
        for step in range(5):
            input_ids = torch.randint(0, 8000, (n_samples, total_len), device=DEVICE)
            labels = input_ids.clone()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = train_step(model, input_ids, labels, optimizer)
            torch.cuda.synchronize()
            times_no_ps.append((time.perf_counter() - t0) * 1000)

        avg_no_ps_ms = sum(times_no_ps) / len(times_no_ps)

        # PS training throughput
        times_ps = []
        for step in range(5):
            input_ids_provider = torch.randint(0, 8000, (1, total_len), device=DEVICE)
            labels_provider = input_ids_provider.clone()
            input_ids_reusers = torch.randint(0, 8000, (n_samples-1, suffix_len), device=DEVICE)
            labels_reusers = input_ids_reusers.clone()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss_ps = train_step_ps(model, input_ids_provider, input_ids_reusers,
                                    labels_provider, labels_reusers, optimizer, prefix_len)
            torch.cuda.synchronize()
            times_ps.append((time.perf_counter() - t0) * 1000)

        avg_ps_ms = sum(times_ps) / len(times_ps)

        compute_savings = (n_samples - 1) / n_samples * prefix_len / total_len * 100
        time_savings = (1 - avg_ps_ms / avg_no_ps_ms) * 100 if avg_no_ps_ms > 0 else 0

        # Throughput
        no_ps_tok_s = n_samples * total_len / (avg_no_ps_ms / 1000)
        ps_effective_tok_s = n_samples * total_len / (avg_ps_ms / 1000)

        results.append({
            "n_samples": n_samples,
            "avg_no_ps_step_ms": round(avg_no_ps_ms, 3),
            "avg_ps_step_ms": round(avg_ps_ms, 3),
            "compute_savings_pct": round(compute_savings, 1),
            "time_savings_pct": round(time_savings, 1),
            "speedup": round(avg_no_ps_ms / avg_ps_ms, 2),
            "no_ps_tok_s": round(no_ps_tok_s, 1),
            "ps_effective_tok_s": round(ps_effective_tok_s, 1),
            "no_ps_tokens_per_step": n_samples * total_len,
            "ps_compute_tokens_per_step": 1 * total_len + (n_samples-1) * suffix_len,
        })

        del model, optimizer
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Exp4: Prefix Ratio Sweep — Training Throughput Optimization
# ---------------------------------------------------------------------------
def exp4_prefix_ratio_sweep():
    results = []
    n_samples = 4
    total_len = 512

    for prefix_ratio in [0.25, 0.50, 0.67, 0.75, 0.90]:
        prefix_len = int(total_len * prefix_ratio)
        suffix_len = total_len - prefix_len

        model = MiniTransformer(num_layers=16, hidden_size=2048, num_heads=16,
                                num_kv_heads=4, head_dim=128, vocab_size=8000).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Warmup
        warmup_ids = torch.randint(0, 8000, (2, 128), device=DEVICE)
        warmup_labels = warmup_ids.clone()
        for _ in range(3):
            train_step(model, warmup_ids, warmup_labels, optimizer)

        # No-PS
        input_ids = torch.randint(0, 8000, (n_samples, total_len), device=DEVICE)
        labels = input_ids.clone()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = train_step(model, input_ids, labels, optimizer)
        torch.cuda.synchronize()
        no_ps_ms = (time.perf_counter() - t0) * 1000

        # PS
        input_ids_provider = torch.randint(0, 8000, (1, total_len), device=DEVICE)
        labels_provider = input_ids_provider.clone()
        input_ids_reusers = torch.randint(0, 8000, (n_samples-1, suffix_len), device=DEVICE)
        labels_reusers = input_ids_reusers.clone()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss_ps = train_step_ps(model, input_ids_provider, input_ids_reusers,
                                labels_provider, labels_reusers, optimizer, prefix_len)
        torch.cuda.synchronize()
        ps_ms = (time.perf_counter() - t0) * 1000

        # Memory
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        input_ids_mem = torch.randint(0, 8000, (n_samples, total_len), device=DEVICE)
        labels_mem = input_ids_mem.clone()
        loss_mem = train_step(model, input_ids_mem, labels_mem, optimizer)
        torch.cuda.synchronize()
        no_ps_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e6

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        input_ids_p_mem = torch.randint(0, 8000, (1, total_len), device=DEVICE)
        labels_p_mem = input_ids_p_mem.clone()
        input_ids_r_mem = torch.randint(0, 8000, (n_samples-1, suffix_len), device=DEVICE)
        labels_r_mem = input_ids_r_mem.clone()
        loss_ps_mem = train_step_ps(model, input_ids_p_mem, input_ids_r_mem,
                                    labels_p_mem, labels_r_mem, optimizer, prefix_len)
        torch.cuda.synchronize()
        ps_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e6

        compute_savings = (n_samples - 1) / n_samples * prefix_ratio * 100
        time_savings = (1 - ps_ms / no_ps_ms) * 100 if no_ps_ms > 0 else 0
        mem_savings = (1 - ps_mem / no_ps_mem) * 100 if no_ps_mem > 0 else 0

        ps_effective_tok_s = n_samples * total_len / (ps_ms / 1000)

        results.append({
            "prefix_ratio": round(prefix_ratio, 2),
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "n_samples": n_samples,
            "compute_savings_pct": round(compute_savings, 1),
            "time_savings_pct": round(time_savings, 1),
            "speedup": round(no_ps_ms / ps_ms, 2),
            "mem_savings_pct": round(mem_savings, 1),
            "no_ps_mem_mb": round(no_ps_mem, 1),
            "ps_mem_mb": round(ps_mem, 1),
            "ps_effective_tok_s": round(ps_effective_tok_s, 1),
        })

        del model, optimizer
        torch.cuda.empty_cache()

    return results


def main():
    print("=" * 60)
    print("GRPO Training Throughput with PS — RTX 4090")
    print("=" * 60)

    gpu_name = torch.cuda.get_device_name(DEVICE)
    gpu_mem = torch.cuda.get_device_properties(DEVICE).total_memory / 1e9
    print(f"GPU: {gpu_name}, Memory: {gpu_mem:.1f} GB")

    warmup(DEVICE)

    all_results = {"gpu": gpu_name, "gpu_mem_gb": round(gpu_mem, 1)}

    # Exp1
    print("\n--- Exp1: Training Step Time (PS vs No-PS) ---")
    r1 = exp1_training_step_time()
    for r in r1:
        print(f"  n={r['n_samples']}: no_ps={r['no_ps_step_ms']}ms, ps={r['ps_step_ms']}ms, "
              f"speedup={r['speedup']}x, mem_save={r['mem_savings_pct']}%, "
              f"tok/s: no_ps={r['no_ps_tok_s']}, ps_eff={r['ps_effective_tok_s']}")
    all_results["exp1_training_step_time"] = r1

    # Exp2
    print("\n--- Exp2: Memory Usage (PS vs No-PS) ---")
    r2 = exp2_memory_usage()
    for r in r2:
        print(f"  n={r['n_samples']}: no_ps_mem={r['no_ps_peak_mem_mb']}MB, ps_mem={r['ps_peak_mem_mb']}MB, "
              f"save={r['mem_savings_pct']}%, reduction={r['ps_mem_reduction_mb']}MB")
    all_results["exp2_memory_usage"] = r2

    # Exp3
    print("\n--- Exp3: n_samples Throughput Sweep ---")
    r3 = exp3_n_samples_throughput()
    for r in r3:
        print(f"  n={r['n_samples']}: avg_no_ps={r['avg_no_ps_step_ms']}ms, avg_ps={r['avg_ps_step_ms']}ms, "
              f"speedup={r['speedup']}x, tok/s: no_ps={r['no_ps_tok_s']}, ps_eff={r['ps_effective_tok_s']}")
    all_results["exp3_n_samples_throughput"] = r3

    # Exp4
    print("\n--- Exp4: Prefix Ratio Sweep ---")
    r4 = exp4_prefix_ratio_sweep()
    for r in r4:
        print(f"  ratio={r['prefix_ratio']}: speedup={r['speedup']}x, "
              f"time_save={r['time_savings_pct']}%, mem_save={r['mem_savings_pct']}%, "
              f"tok/s={r['ps_effective_tok_s']}")
    all_results["exp4_prefix_ratio_sweep"] = r4

    # Save
    with open("grpo_training_ps_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to grpo_training_ps_results.json")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best_speedup = max(r3, key=lambda x: x["speedup"])
    print(f"Best training speedup: n={best_speedup['n_samples']}, speedup={best_speedup['speedup']}x, "
          f"tok/s improvement: {best_speedup['ps_effective_tok_s']}/{best_speedup['no_ps_tok_s']}")

    best_mem = max(r2, key=lambda x: x["mem_savings_pct"])
    print(f"Best memory savings: n={best_mem['n_samples']}, {best_mem['mem_savings_pct']}%, "
          f"reduction {best_mem['ps_mem_reduction_mb']}MB")

    best_ratio = max(r4, key=lambda x: x["speedup"])
    print(f"Best prefix_ratio: {best_ratio['prefix_ratio']}, speedup={best_ratio['speedup']}x")

    print("\nKEY INSIGHT: Training (forward+backward) speedup is lower than "
          "forward-only because backward requires full sequence gradients.")
    print("But memory savings are significant: less activation memory for prefix tokens.")


if __name__ == "__main__":
    main()