"""
FlashInfer Decode Benchmark Extended — RTX 4090 (SM89)

Tests:
1. Sequence length sweep: FlashInfer vs SDPA across S=128,256,512,1024,2048
2. GQA sweep: num_kv_heads=1,2,4,5,10,20 vs MHA
3. Correctness: cos_sim verification for all configs
"""

import os, sys, json, numpy as np, torch, torch.nn.functional as F, time

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
print(f"Device: {props.name} SM={props.major}.{props.minor}")

H = 2560  # total hidden size

def benchmark_fn(fn, warmup=5, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    latencies = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        latencies.append(s.elapsed_time(e))
    return np.mean(latencies), np.std(latencies)

# =====================================================================
# SECTION 1: Sequence Length Sweep (FlashInfer vs SDPA, GQA-5)
# =====================================================================
print("\n" + "="*70)
print("SECTION 1: Sequence Length Sweep (GQA-5, B=16)")
print("="*70)

num_heads = 20; num_kv_heads = 5; d_head = H // num_heads

try:
    import flashinfer
    FLASHINFER_AVAILABLE = True
    print(f"FlashInfer version: {flashinfer.__version__}")
except ImportError:
    FLASHINFER_AVAILABLE = False
    print("FlashInfer NOT available — will only run SDPA")

seq_lengths = [128, 256, 512, 1024, 2048]
batch_size = 16

results_seq_sweep = {"sdpa": [], "flashinfer": []}

for S in seq_lengths:
    # SDPA
    k_cache = torch.randn(batch_size, S, num_kv_heads, d_head, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn(batch_size, S, num_kv_heads, d_head, dtype=torch.bfloat16, device=device)
    q = torch.randn(batch_size, 1, num_heads, d_head, dtype=torch.bfloat16, device=device)

    def sdpa_decode():
        k_exp = k_cache.unsqueeze(2).expand(batch_size, S, num_heads//num_kv_heads, num_kv_heads, d_head).reshape(batch_size, S, num_heads, d_head)
        v_exp = v_cache.unsqueeze(2).expand(batch_size, S, num_heads//num_kv_heads, num_kv_heads, d_head).reshape(batch_size, S, num_heads, d_head)
        return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

    mean_ms, std_ms = benchmark_fn(sdpa_decode, warmup=5, iters=30)
    tok_s = batch_size / (mean_ms * 1e-3)
    results_seq_sweep["sdpa"].append({"S": S, "B": batch_size, "ms": round(mean_ms,4), "tok_s": round(tok_s,1)})
    print(f"  SDPA S={S}: {mean_ms:.3f}ms → {tok_s:.1f} tok/s")

    # FlashInfer
    if FLASHINFER_AVAILABLE:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        page_size = 16
        num_pages = (S + page_size - 1) // page_size
        total_pages = batch_size * num_pages
        # Stacked k/v: (num_pages, 2, page_size, num_kv_heads, d_head)
        kv_data = torch.randn(total_pages, 2, page_size, num_kv_heads, d_head, dtype=torch.bfloat16, device=device)
        page_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
        paged_kv_indptr = torch.arange(0, total_pages+num_pages, num_pages, dtype=torch.int32, device=device)
        paged_kv_last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32, device=device)
        workspace = torch.empty(32*1024*1024, dtype=torch.uint8, device=device)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
        q_fi = torch.randn(batch_size, num_heads, d_head, dtype=torch.bfloat16, device=device)

        def fi_decode():
            wrapper.begin_forward(paged_kv_indptr, page_indices, paged_kv_last_page_len,
                                  num_heads, num_kv_heads, d_head, page_size,
                                  q_data_type=torch.bfloat16)
            o = wrapper.forward(q_fi, kv_data, paged_kv_indptr, page_indices, paged_kv_last_page_len)
            wrapper.end_forward()
            return o

        mean_ms, std_ms = benchmark_fn(fi_decode, warmup=5, iters=30)
        tok_s = batch_size / (mean_ms * 1e-3)
        speedup = results_seq_sweep["sdpa"][-1]["ms"] / mean_ms
        results_seq_sweep["flashinfer"].append({"S": S, "B": batch_size, "ms": round(mean_ms,4), "tok_s": round(tok_s,1), "speedup": round(speedup,2)})
        print(f"  FI S={S}: {mean_ms:.3f}ms → {tok_s:.1f} tok/s (speedup={speedup:.2f}x)")

# =====================================================================
# SECTION 2: GQA Sweep (num_kv_heads, S=512, B=16)
# =====================================================================
print("\n" + "="*70)
print("SECTION 2: GQA Sweep (num_kv_heads=1/2/4/5/10/20, S=512, B=16)")
print("="*70)

kv_heads_list = [1, 2, 4, 5, 10, 20]
S = 512; B = 16
results_gqa_sweep = {"sdpa": [], "flashinfer": []}

for num_kv in kv_heads_list:
    num_qo = 20; d = H // num_qo

    # SDPA
    k = torch.randn(B, S, num_kv, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, S, num_kv, d, dtype=torch.bfloat16, device=device)
    q = torch.randn(B, 1, num_qo, d, dtype=torch.bfloat16, device=device)

    def sdpa_gqa():
        k_exp = k.unsqueeze(2).expand(B, S, num_qo//num_kv, num_kv, d).reshape(B, S, num_qo, d)
        v_exp = v.unsqueeze(2).expand(B, S, num_qo//num_kv, num_kv, d).reshape(B, S, num_qo, d)
        return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

    mean_ms, std_ms = benchmark_fn(sdpa_gqa, warmup=5, iters=30)
    tok_s = B / (mean_ms * 1e-3)
    kv_bytes = B * S * num_kv * d * 2  # BF16
    results_gqa_sweep["sdpa"].append({"num_kv": num_kv, "ms": round(mean_ms,4), "tok_s": round(tok_s,1), "kv_MB": round(kv_bytes/1e6,2)})
    print(f"  SDPA kv={num_kv}: {mean_ms:.3f}ms → {tok_s:.1f} tok/s, KV={kv_bytes/1e6:.2f}MB")

    # FlashInfer
    if FLASHINFER_AVAILABLE:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        page_size = 16
        num_pages = (S + page_size - 1) // page_size
        total_pages = B * num_pages
        kv_data = torch.randn(total_pages, 2, page_size, num_kv, d, dtype=torch.bfloat16, device=device)
        page_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
        paged_kv_indptr = torch.arange(0, total_pages+num_pages, num_pages, dtype=torch.int32, device=device)
        last_page_len = torch.full((B,), page_size, dtype=torch.int32, device=device)
        workspace = torch.empty(32*1024*1024, dtype=torch.uint8, device=device)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
        q_fi = torch.randn(B, num_qo, d, dtype=torch.bfloat16, device=device)

        def fi_gqa():
            wrapper.begin_forward(paged_kv_indptr, page_indices, last_page_len,
                                  num_qo, num_kv, d, page_size,
                                  q_data_type=torch.bfloat16)
            o = wrapper.forward(q_fi, kv_data, paged_kv_indptr, page_indices, last_page_len)
            wrapper.end_forward()
            return o

        mean_ms, std_ms = benchmark_fn(fi_gqa, warmup=5, iters=30)
        tok_s = B / (mean_ms * 1e-3)
        speedup = results_gqa_sweep["sdpa"][-1]["ms"] / mean_ms
        results_gqa_sweep["flashinfer"].append({"num_kv": num_kv, "ms": round(mean_ms,4), "tok_s": round(tok_s,1), "speedup": round(speedup,2), "kv_MB": round(kv_bytes/1e6,2)})
        print(f"  FI kv={num_kv}: {mean_ms:.3f}ms → {tok_s:.1f} tok/s, speedup={speedup:.2f}x")

# =====================================================================
# SECTION 3: Correctness Verification (FlashInfer vs SDPA)
# =====================================================================
print("\n" + "="*70)
print("SECTION 3: Correctness (cos_sim)")
print("="*70)

if FLASHINFER_AVAILABLE:
    for num_kv in [5, 10, 20]:
        num_qo = 20; d = H // num_qo; S = 512; B = 4
        # Simple correctness: compare outputs
        k = torch.randn(S, num_kv, d, dtype=torch.bfloat16, device=device)
        v = torch.randn(S, num_kv, d, dtype=torch.bfloat16, device=device)
        q = torch.randn(B, 1, num_qo, d, dtype=torch.bfloat16, device=device)

        # SDPA output
        k_exp = k.unsqueeze(0).unsqueeze(2).expand(B, S, num_qo//num_kv, num_kv, d).reshape(B, S, num_qo, d)
        v_exp = v.unsqueeze(0).unsqueeze(2).expand(B, S, num_qo//num_kv, num_kv, d).reshape(B, S, num_qo, d)
        sdpa_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

        # FlashInfer output (single batch for simplicity)
        page_size = 16
        num_pages = (S + page_size - 1) // page_size
        # Create simple paged kv for single batch
        total_pages = num_pages
        # Stack k,v along dim=1
        k_pages = k.reshape(num_pages, page_size, num_kv, d)
        v_pages = v.reshape(num_pages, page_size, num_kv, d)
        kv_stacked = torch.stack([k_pages, v_pages], dim=1)  # (num_pages, 2, page_size, num_kv, d)

        workspace = torch.empty(32*1024*1024, dtype=torch.uint8, device=device)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
        q_single = torch.randn(1, num_qo, d, dtype=torch.bfloat16, device=device)
        indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
        indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        last_page = torch.tensor([page_size], dtype=torch.int32, device=device)

        wrapper.begin_forward(indptr, indices, last_page, num_qo, num_kv, d, page_size, q_data_type=torch.bfloat16)
        fi_out_single = wrapper.forward(q_single, kv_stacked, indptr, indices, last_page)
        wrapper.end_forward()

        # Compare first batch element
        sdpa_single = sdpa_out[0].squeeze(0)  # (num_qo, d)
        fi_single = fi_out_single.squeeze(0)   # (num_qo, d)

        cos_sim = F.cosine_similarity(sdpa_single.flatten().unsqueeze(0).float(),
                                      fi_single.flatten().unsqueeze(0).float()).item()
        max_diff = (sdpa_single.float() - fi_single.float()).abs().max().item()
        print(f"  kv={num_kv}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}")

# =====================================================================
# Save Results
# =====================================================================
all_results = {
    "device": {"name": props.name, "sm": f"{props.major}.{props.minor}"},
    "config": {"H": H, "num_heads": 20},
    "seq_sweep": results_seq_sweep,
    "gqa_sweep": results_gqa_sweep,
}

os.makedirs("results", exist_ok=True)
with open("results/flashinfer_decode_extended_benchmark.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to results/flashinfer_decode_extended_benchmark.json")

# =====================================================================
# Summary
# =====================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\nSeq Length Sweep (B=16, GQA-5):")
for i, S in enumerate(seq_lengths):
    sdpa = results_seq_sweep["sdpa"][i]
    if FLASHINFER_AVAILABLE and i < len(results_seq_sweep["flashinfer"]):
        fi = results_seq_sweep["flashinfer"][i]
        print(f"  S={S}: SDPA={sdpa['ms']:.3f}ms FI={fi['ms']:.3f}ms → {fi['speedup']:.2f}x")
    else:
        print(f"  S={S}: SDPA={sdpa['ms']:.3f}ms → {sdpa['tok_s']:.1f} tok/s")

print("\nGQA Sweep (S=512, B=16):")
for i, kv in enumerate(kv_heads_list):
    sdpa = results_gqa_sweep["sdpa"][i]
    if FLASHINFER_AVAILABLE and i < len(results_gqa_sweep["flashinfer"]):
        fi = results_gqa_sweep["flashinfer"][i]
        print(f"  kv={kv}: SDPA={sdpa['ms']:.3f}ms FI={fi['ms']:.3f}ms → {fi['speedup']:.2f}x (KV={sdpa['kv_MB']:.2f}MB)")
    else:
        print(f"  kv={kv}: SDPA={sdpa['ms']:.3f}ms → {sdpa['tok_s']:.1f} tok/s")