#!/usr/bin/env python3
"""Top-n-sigma Vectorized Correctness & Performance Verification
================================================================

Verifies that the vectorized apply() produces identical results to the
original row-by-row implementation, reproduces previous sampling behavior
experiments, and measures performance improvement.

Runs on RTX 4090 (university server) with PyTorch 2.9+cu128.

Experiments:
1. Bit-for-bit correctness: vectorized vs row-by-row output comparison
2. Edge case coverage: NaN/Inf, std=0, empty req_info, mixed n_sigma
3. Argmax invariance: both versions preserve argmax token
4. Sampling behavior reproduction: token survival, temperature stability
5. Performance benchmark: vectorized vs row-by-row speedup
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
import json
import time
import math

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")


# ============================================================
# Implementations
# ============================================================

def apply_rowbyrow(logits, req_info):
    """Original row-by-row TopNSigmaLogitsProcessor.apply() logic."""
    result = logits.clone()
    for row_idx, n_sigma in req_info.items():
        row_logits = result[row_idx]
        if not torch.isfinite(row_logits).all():
            continue
        max_logit = row_logits.max()
        std_logit = row_logits.std()
        if std_logit == 0:
            continue
        threshold = max_logit - n_sigma * std_logit
        mask = row_logits < threshold
        row_logits[mask] = float("-inf")
    return result


def apply_vectorized(logits, req_info):
    """New vectorized TopNSigmaLogitsProcessor.apply() logic."""
    if not req_info:
        return logits.clone()

    rows = torch.tensor(
        list(req_info.keys()), dtype=torch.long, device=logits.device
    )
    n_sigmas = torch.tensor(
        list(req_info.values()), dtype=logits.dtype, device=logits.device
    )

    selected_logits = logits[rows].clone()

    finite_mask = torch.isfinite(selected_logits).all(dim=-1)
    std_logits = selected_logits.std(dim=-1)
    nonzero_std_mask = std_logits != 0

    process_mask = finite_mask & nonzero_std_mask

    rows_to_process = rows[process_mask]
    n_sigmas_to_process = n_sigmas[process_mask]
    logits_to_process = selected_logits[process_mask]

    if rows_to_process.numel() == 0:
        return logits.clone()

    max_logits = logits_to_process.max(dim=-1, keepdim=True).values
    std_logits_valid = logits_to_process.std(dim=-1, keepdim=True)

    thresholds = max_logits - n_sigmas_to_process.unsqueeze(-1) * std_logits_valid
    masks = logits_to_process < thresholds
    logits_to_process[masks] = float("-inf")

    result = logits.clone()
    result[rows_to_process] = logits_to_process

    return result


def bench_ms(fn, warmup=10, rep=100):
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


def tensors_equal(a, b):
    """Compare tensors, handling NaN/Inf correctly (IEEE 754: NaN != NaN)."""
    # NaN positions must match
    nan_match = (torch.isnan(a) == torch.isnan(b)).all().item()
    # Inf positions and signs must match
    inf_match = (torch.isinf(a) == torch.isinf(b)).all().item()
    # For finite values, use exact ==
    finite_mask = torch.isfinite(a) & torch.isfinite(b)
    finite_vals_match = (a[finite_mask] == b[finite_mask]).all().item() if finite_mask.any() else True
    return nan_match and inf_match and finite_vals_match


# ============================================================
# Experiment 1: Bit-for-bit Correctness
# ============================================================

def exp1_correctness():
    print("\n" + "=" * 70)
    print("Exp1: Bit-for-bit Correctness — Vectorized vs Row-by-Row")
    print("=" * 70)

    torch.manual_seed(42)
    results = []
    all_pass = True

    configs = [
        ("uniform_n", {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0}),
        ("mixed_n", {0: 1.0, 1: 2.0, 2: 3.0, 3: 5.0}),
        ("sparse", {0: 2.0, 3: 2.0}),  # rows 1,2 have no filter
        ("single_row", {2: 2.0}),
        ("full_batch_32", {i: 2.0 for i in range(32)}),
    ]

    for name, req_info in configs:
        B = max(req_info.keys()) + 1
        for V in [1024, 32000]:
            for dtype_name, dtype in [("FP32", torch.float32), ("FP16", torch.float16)]:
                logits = torch.randn(B, V, device="cuda", dtype=dtype)

                out_row = apply_rowbyrow(logits, req_info)
                out_vec = apply_vectorized(logits, req_info)

                # Compare
                diff = (out_row.float() - out_vec.float()).abs()
                max_diff = diff.max().item()
                identical = (out_row == out_vec).all().item()

                # Compare survival counts
                survive_row = (out_row > float('-inf')).sum(dim=-1)
                survive_vec = (out_vec > float('-inf')).sum(dim=-1)
                survive_match = (survive_row == survive_vec).all().item()

                # Argmax preserved in both?
                argmax_orig = logits.argmax(dim=-1)
                argmax_row = out_row.argmax(dim=-1)
                argmax_vec = out_vec.argmax(dim=-1)
                argmax_row_ok = (argmax_row == argmax_orig).all().item()
                argmax_vec_ok = (argmax_vec == argmax_orig).all().item()

                pass_ = identical and survive_match and argmax_row_ok and argmax_vec_ok
                all_pass = all_pass and pass_

                print(f"  {name:20s} B={B} V={V:5d} {dtype_name}: "
                      f"identical={identical} max_diff={max_diff:.2e} "
                      f"survive_match={survive_match} argmax_row={argmax_row_ok} "
                      f"argmax_vec={argmax_vec_ok} {'PASS' if pass_ else 'FAIL'}")

                results.append({
                    "config": name, "B": B, "V": V, "dtype": dtype_name,
                    "identical": identical, "max_diff": max_diff,
                    "survive_match": survive_match,
                    "argmax_row_ok": argmax_row_ok,
                    "argmax_vec_ok": argmax_vec_ok,
                    "pass": pass_,
                })

                del logits, out_row, out_vec
                torch.cuda.empty_cache()

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    return results, all_pass


# ============================================================
# Experiment 2: Edge Cases
# ============================================================

def exp2_edge_cases():
    print("\n" + "=" * 70)
    print("Exp2: Edge Cases")
    print("=" * 70)

    torch.manual_seed(42)
    results = []
    all_pass = True

    # Case 1: NaN/Inf in logits
    logits_nan = torch.randn(4, 1000, device="cuda")
    logits_nan[1, :5] = float("inf")
    logits_nan[3, :3] = float("nan")
    req_info = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0}

    out_row = apply_rowbyrow(logits_nan, req_info)
    out_vec = apply_vectorized(logits_nan, req_info)
    nan_match = tensors_equal(out_row, out_vec)
    # Rows 1,3 should be unchanged (NaN/Inf) — use NaN-aware comparison
    row1_unchanged_row = tensors_equal(out_row[1], logits_nan[1])
    row3_unchanged_row = tensors_equal(out_row[3], logits_nan[3])
    row1_unchanged_vec = tensors_equal(out_vec[1], logits_nan[1])
    row3_unchanged_vec = tensors_equal(out_vec[3], logits_nan[3])
    pass1 = nan_match
    all_pass = all_pass and pass1
    print(f"  NaN/Inf: identical={nan_match}, row1_skip_row={row1_unchanged_row}, "
          f"row3_skip_row={row3_unchanged_row}, row1_skip_vec={row1_unchanged_vec}, "
          f"row3_skip_vec={row3_unchanged_vec} {'PASS' if pass1 else 'FAIL'}")
    results.append({"case": "nan_inf", "identical": nan_match, "pass": pass1})

    # Case 2: All equal logits (std=0)
    logits_const = torch.ones(4, 1000, device="cuda")
    req_info = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0}

    out_row = apply_rowbyrow(logits_const, req_info)
    out_vec = apply_vectorized(logits_const, req_info)
    const_match = (out_row == out_vec).all().item()
    # All rows should be unchanged (std=0 → skip)
    unchanged_row = (out_row == logits_const).all().item()
    unchanged_vec = (out_vec == logits_const).all().item()
    pass2 = const_match and unchanged_row and unchanged_vec
    all_pass = all_pass and pass2
    print(f"  Constant: identical={const_match}, unchanged_row={unchanged_row}, "
          f"unchanged_vec={unchanged_vec} {'PASS' if pass2 else 'FAIL'}")
    results.append({"case": "constant", "identical": const_match, "pass": pass2})

    # Case 3: Empty req_info
    logits = torch.randn(4, 1000, device="cuda")
    req_info_empty = {}

    out_row = apply_rowbyrow(logits, req_info_empty)
    out_vec = apply_vectorized(logits, req_info_empty)
    empty_match = (out_row == out_vec).all().item()
    unchanged = (out_row == logits).all().item()
    pass3 = empty_match and unchanged
    all_pass = all_pass and pass3
    print(f"  Empty: identical={empty_match}, unchanged={unchanged} "
          f"{'PASS' if pass3 else 'FAIL'}")
    results.append({"case": "empty", "identical": empty_match, "pass": pass3})

    # Case 4: Mixed — some rows normal, some constant, some NaN
    logits_mix = torch.randn(8, 1000, device="cuda")
    logits_mix[2] = 1.0  # constant (std=0)
    logits_mix[5, :3] = float("nan")  # NaN
    req_info = {0: 2.0, 1: 1.5, 2: 2.0, 3: 3.0, 5: 2.0, 7: 2.5}

    out_row = apply_rowbyrow(logits_mix, req_info)
    out_vec = apply_vectorized(logits_mix, req_info)
    mix_match = tensors_equal(out_row, out_vec)
    pass4 = mix_match
    all_pass = all_pass and pass4
    print(f"  Mixed: identical={mix_match} {'PASS' if pass4 else 'FAIL'}")
    results.append({"case": "mixed", "identical": mix_match, "pass": pass4})

    # Case 5: Different n_sigma per row — verify each row's threshold
    logits = torch.randn(6, 1000, device="cuda")
    req_info = {0: 1.0, 1: 2.0, 2: 3.0, 3: 5.0, 4: 0.5, 5: 10.0}

    out_row = apply_rowbyrow(logits, req_info)
    out_vec = apply_vectorized(logits, req_info)
    diff_n_match = (out_row == out_vec).all().item()
    pass5 = diff_n_match
    all_pass = all_pass and pass5
    print(f"  Diff n_sigma: identical={diff_n_match} {'PASS' if pass5 else 'FAIL'}")
    results.append({"case": "diff_n_sigma", "identical": diff_n_match, "pass": pass5})

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    return results, all_pass


# ============================================================
# Experiment 3: Argmax Invariance (Reproducing Previous Results)
# ============================================================

def exp3_argmax_invariance():
    print("\n" + "=" * 70)
    print("Exp3: Argmax Invariance Verification")
    print("=" * 70)

    torch.manual_seed(42)
    results = []

    for V in [1000, 32000]:
        for n_sigma in [1.0, 2.0, 3.0, 5.0]:
            for sharpness in [0.3, 1.0, 2.0, 5.0]:
                logits = torch.randn(4, V, device="cuda") * sharpness
                req_info = {0: n_sigma, 1: n_sigma, 2: n_sigma, 3: n_sigma}

                orig_argmax = logits.argmax(dim=-1)

                out_row = apply_rowbyrow(logits, req_info)
                out_vec = apply_vectorized(logits, req_info)

                row_argmax = out_row.argmax(dim=-1)
                vec_argmax = out_vec.argmax(dim=-1)

                row_ok = (row_argmax == orig_argmax).all().item()
                vec_ok = (vec_argmax == orig_argmax).all().item()

                results.append({
                    "V": V, "n": n_sigma, "sharpness": sharpness,
                    "row_argmax_ok": row_ok, "vec_argmax_ok": vec_ok,
                })

    all_row_ok = all(r["row_argmax_ok"] for r in results)
    all_vec_ok = all(r["vec_argmax_ok"] for r in results)

    print(f"  Tested {len(results)} configurations")
    print(f"  Row-by-row argmax preserved: {'ALL YES' if all_row_ok else 'SOME NO'}")
    print(f"  Vectorized argmax preserved: {'ALL YES' if all_vec_ok else 'SOME NO'}")

    return results, all_row_ok and all_vec_ok


# ============================================================
# Experiment 4: Token Survival & Temperature Stability
# ============================================================

def exp4_sampling_behavior():
    print("\n" + "=" * 70)
    print("Exp4: Sampling Behavior Reproduction (Token Survival)")
    print("=" * 70)

    torch.manual_seed(42)
    V = 32000
    results = []

    # 4a: Token survival at different sharpness levels
    print("\n  4a: Token survival by distribution sharpness")
    for sharpness_name, sharpness in [("sharp", 0.3), ("medium", 1.0), ("flat", 3.0)]:
        logits = torch.randn(4, V, device="cuda") * sharpness
        for n in [1.0, 2.0, 3.0]:
            req_info = {i: n for i in range(4)}
            out_row = apply_rowbyrow(logits, req_info)
            out_vec = apply_vectorized(logits, req_info)

            survive_row = (out_row > float('-inf')).sum(dim=1).float().mean().item()
            survive_vec = (out_vec > float('-inf')).sum(dim=1).float().mean().item()

            match = survive_row == survive_vec
            print(f"    {sharpness_name:8s} n={n}: row={survive_row:.0f} vec={survive_vec:.0f} "
                  f"match={match}")
            results.append({
                "test": "survival", "sharpness": sharpness_name, "n": n,
                "row_survive": survive_row, "vec_survive": survive_vec,
                "match": match,
            })

    # 4b: Temperature stability
    print("\n  4b: Temperature stability (surviving token count)")
    logits_base = torch.randn(1, V, device="cuda") * 2.0
    # Add dominant tokens
    logits_base[0, :5] += 5.0

    for temp in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        scaled = logits_base / temp
        req_info = {0: 2.0}

        out_row = apply_rowbyrow(scaled, req_info)
        out_vec = apply_vectorized(scaled, req_info)

        survive_row = (out_row > float('-inf')).sum().item()
        survive_vec = (out_vec > float('-inf')).sum().item()

        match = survive_row == survive_vec
        print(f"    T={temp:5.1f}: row={survive_row} vec={survive_vec} match={match}")
        results.append({
            "test": "temp_stability", "temp": temp,
            "row_survive": survive_row, "vec_survive": survive_vec,
            "match": match,
        })

    all_match = all(r["match"] for r in results)
    print(f"\n  Overall: {'ALL MATCH' if all_match else 'SOME MISMATCH'}")
    return results, all_match


# ============================================================
# Experiment 5: Performance Benchmark
# ============================================================

def exp5_performance():
    print("\n" + "=" * 70)
    print("Exp5: Performance Benchmark — Vectorized vs Row-by-Row")
    print("=" * 70)

    results = []

    print(f"\n  {'Vocab':<8} {'Batch':<8} {'Row-by-row ms':<16} {'Vectorized ms':<16} {'Speedup'}")
    print("  " + "-" * 64)

    for V, B_list in [(32000, [1, 4, 16, 32, 128, 256]),
                       (128000, [1, 4, 16, 32])]:
        for B in B_list:
            logits = torch.randn(B, V, device="cuda", dtype=torch.float32)
            req_info = {i: 2.0 for i in range(B)}

            row_ms = bench_ms(lambda: apply_rowbyrow(logits, req_info), rep=50)
            vec_ms = bench_ms(lambda: apply_vectorized(logits, req_info), rep=50)
            speedup = row_ms / vec_ms

            print(f"  {V:<8} {B:<8} {row_ms:<16.4f} {vec_ms:<16.4f} {speedup:.1f}x")
            results.append({
                "V": V, "B": B,
                "row_ms": round(row_ms, 4),
                "vec_ms": round(vec_ms, 4),
                "speedup": round(speedup, 1),
            })

            del logits
            torch.cuda.empty_cache()

    # FP16 benchmark
    print(f"\n  FP16 Performance:")
    V = 32000
    for B in [32, 128]:
        logits = torch.randn(B, V, device="cuda", dtype=torch.float16)
        req_info = {i: 2.0 for i in range(B)}

        row_ms = bench_ms(lambda: apply_rowbyrow(logits, req_info), rep=50)
        vec_ms = bench_ms(lambda: apply_vectorized(logits, req_info), rep=50)
        speedup = row_ms / vec_ms

        print(f"  FP16 V={V} B={B}: row={row_ms:.4f}ms vec={vec_ms:.4f}ms {speedup:.1f}x")
        results.append({
            "V": V, "B": B, "dtype": "FP16",
            "row_ms": round(row_ms, 4),
            "vec_ms": round(vec_ms, 4),
            "speedup": round(speedup, 1),
        })

        del logits
        torch.cuda.empty_cache()

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    all_results = {}

    r1, p1 = exp1_correctness()
    all_results["exp1_correctness"] = {"results": r1, "all_pass": p1}

    r2, p2 = exp2_edge_cases()
    all_results["exp2_edge_cases"] = {"results": r2, "all_pass": p2}

    r3, p3 = exp3_argmax_invariance()
    all_results["exp3_argmax_invariance"] = {"results": r3, "all_pass": p3}

    r4, p4 = exp4_sampling_behavior()
    all_results["exp4_sampling_behavior"] = {"results": r4, "all_pass": p4}

    r5 = exp5_performance()
    all_results["exp5_performance"] = r5

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_tests_pass = p1 and p2 and p3 and p4
    print(f"  Correctness (bit-for-bit): {'PASS' if p1 else 'FAIL'}")
    print(f"  Edge cases: {'PASS' if p2 else 'FAIL'}")
    print(f"  Argmax invariance: {'PASS' if p3 else 'FAIL'}")
    print(f"  Sampling behavior: {'PASS' if p4 else 'FAIL'}")

    if r5:
        key_speedups = [r for r in r5 if r.get("V") == 32000 and r.get("B") in [32, 128]]
        for r in key_speedups:
            print(f"  Performance V={r['V']} B={r['B']}: {r['speedup']}x speedup")

    print(f"\n  Overall: {'ALL TESTS PASS — vectorized is ready for PR' if all_tests_pass else 'SOME TESTS FAIL — need investigation'}")

    # Save
    def convert(obj):
        if isinstance(obj, torch.Tensor): return obj.item()
        elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert(v) for v in obj]
        elif isinstance(obj, bool): return obj
        return obj

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'top_n_sigma_vectorized_test_results.json')
    with open(out_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"  Results saved to {out_path}")