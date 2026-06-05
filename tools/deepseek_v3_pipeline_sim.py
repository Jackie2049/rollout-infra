#!/usr/bin/env python3
"""
DeepSeek-V3 Inference Pipeline Simulator
=========================================
Models the full decode inference pipeline incorporating real performance data
from FlashMLA, DeepGEMM, and DeepEP.

Pipeline per decode step:
  1. MLA Attention (FlashMLA) — Q projection + MLA decode kernel
  2. MoE Routing — Gate network + Top-K selection
  3. Expert Parallel Dispatch (DeepEP) — All-to-All send
  4. Expert Computation (DeepGEMM) — Grouped GEMM (SwiGLU)
  5. Expert Parallel Combine (DeepEP) — All-to-All recv
  6. Output projection + residual

Reference data sources:
  - FlashMLA: 3000 GB/s (memory-bound), 660 TFLOPS (compute-bound) on H800
  - DeepGEMM: 1550 TFLOPS FP8 GEMM on H800
  - DeepEP V2: 726-740 GB/s NVLink EP8, SM=24
  - DeepSeek-V3: 256 experts, top-8, 7168 hidden, 18432 intermediate
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Optional


# ─── DeepSeek-V3 Model Config ───────────────────────────────────────────────

@dataclass
class DeepSeekV3Config:
    """DeepSeek-V3 model parameters."""
    hidden: int = 7168                # d_model
    num_layers: int = 61
    num_attention_heads: int = 128    # h_q
    num_kv_heads: int = 1             # MQA (MLA compressed)
    head_dim_qk: int = 576            # d_qk (includes RoPE)
    head_dim_v: int = 512             # d_v
    kv_compress_dim: int = 512        # MLA compression dimension
    num_experts: int = 256
    num_topk: int = 8
    intermediate_size: int = 18432    # SwiGLU intermediate
    vocab_size: int = 129280
    max_seq_len: int = 131072         # 128K


# ─── Hardware Config ─────────────────────────────────────────────────────────

@dataclass
class HardwareConfig:
    """GPU hardware parameters."""
    name: str = "H800"
    fp8_tflops: float = 1979          # FP8 tensor core peak
    bf16_tflops: float = 989          # BF16 tensor core peak
    fp32_tflops: float = 67           # FP32 peak
    memory_bw_gbs: float = 3352       # GB/s HBM3
    nvlink_bw_gbs: float = 900        # GB/s bidirectional per direction
    num_gpus: int = 8
    sm_count: int = 132
    sm_for_ep: int = 24               # SMs allocated for EP communication
    sm_for_compute: int = 108         # SMs for compute


# ─── Pipeline Stage Models ───────────────────────────────────────────────────

class MLAAttentionStage:
    """FlashMLA decode kernel — memory-bound or compute-bound depending on batch."""

    def __init__(self, config: DeepSeekV3Config, hw: HardwareConfig):
        self.config = config
        self.hw = hw

    def estimate_latency_us(self, batch_size: int, seq_len: int, is_fp8_kv: bool = False):
        """Estimate MLA decode attention latency per layer.

        MLA decode has two regimes:
        - Memory-bound: dominated by KV cache read (small batch)
        - Compute-bound: dominated by Q×KV matmul (large batch, h_q=128)
        """
        c = self.config
        h = self.hw

        # Compute-memory ratio for MLA: ≈ 2 * h_q * s_q = 256 for s_q=1
        compute_mem_ratio = 2 * c.num_attention_heads  # s_q=1 for decode

        # Arithmetic intensity threshold
        ai_threshold = h.fp8_tflops * 1e3 / h.memory_bw_gbs  # TFLOPS→GFLOPS / GB/s

        # Bytes to read from KV cache per query
        kv_bytes_per_token = c.kv_compress_dim * 2  # BF16 (or FP8 compressed)
        if is_fp8_kv:
            kv_bytes_per_token = 656 // 1  # 656 bytes per token (FP8 format)
        kv_read_bytes = seq_len * kv_bytes_per_token * c.num_kv_heads

        # Total memory bytes (Q + KV read + O write)
        q_bytes = batch_size * c.num_attention_heads * c.head_dim_qk * 2  # BF16
        o_bytes = batch_size * c.num_attention_heads * c.head_dim_v * 2   # BF16
        total_bytes = kv_read_bytes * batch_size + q_bytes + o_bytes

        # Total FLOPs: Q×K^T + softmax×V
        # Per query: 2 * seq_len * (d_qk + d_v) * h_q
        flops_per_query = 2 * seq_len * (c.head_dim_qk + c.head_dim_v) * c.num_attention_heads
        total_flops_gflops = flops_per_query * batch_size / 1e9  # convert to GFLOPs

        # Memory-bound latency
        mem_latency = total_bytes / (h.memory_bw_gbs * 1e9) * 1e6  # us

        # Compute-bound latency (using FlashMLA measured: 660 TFLOPS compute-bound)
        flashmla_compute_tflops = 660  # measured on H800
        compute_latency = total_flops_gflops / (flashmla_compute_tflops * 1e3) * 1e6  # us

        # FlashMLA measured: 3000 GB/s memory-bound throughput
        flashmla_mem_bw = 3000  # GB/s measured
        flashmla_mem_latency = total_bytes / (flashmla_mem_bw * 1e9) * 1e6  # us

        # Take the bottleneck
        actual_latency = max(flashmla_mem_latency, compute_latency)

        return {
            "stage": "MLA Attention (FlashMLA)",
            "total_bytes_mb": total_bytes / 1e6,
            "total_flops_gflops": total_flops_gflops,
            "mem_latency_us": flashmla_mem_latency,
            "compute_latency_us": compute_latency,
            "actual_latency_us": actual_latency,
            "bottleneck": "memory" if flashmla_mem_latency > compute_latency else "compute",
            "kv_bytes_per_token": kv_bytes_per_token,
        }


class MoERouterStage:
    """Gate network + Top-K expert selection."""

    def __init__(self, config: DeepSeekV3Config, hw: HardwareConfig):
        self.config = config
        self.hw = hw

    def estimate_latency_us(self, batch_size: int):
        c = self.config
        h = self.hw

        # Gate: linear(hidden, num_experts) + top-k selection
        gate_flops = 2 * batch_size * c.hidden * c.num_experts  # matmul
        gate_flops_gflops = gate_flops / 1e9

        # Gate weight read: num_experts * hidden * 2 bytes (BF16)
        gate_weight_bytes = c.num_experts * c.hidden * 2
        total_bytes = gate_weight_bytes + batch_size * c.hidden * 2  # weight + input

        mem_latency = total_bytes / (h.memory_bw_gbs * 1e9) * 1e6
        compute_latency = gate_flops_gflops / (h.bf16_tflops * 1e3) * 1e6

        actual_latency = max(mem_latency, compute_latency) * 1.1  # 10% overhead for topk

        return {
            "stage": "MoE Router (Gate + Top-K)",
            "total_bytes_mb": total_bytes / 1e6,
            "total_flops_gflops": gate_flops_gflops,
            "actual_latency_us": actual_latency,
        }


class EPDispatchStage:
    """Expert Parallel All-to-All dispatch (DeepEP)."""

    def __init__(self, config: DeepSeekV3Config, hw: HardwareConfig):
        self.config = config
        self.hw = hw

    def estimate_latency_us(self, batch_size: int, is_fp8: bool = False):
        c = self.config
        h = self.hw

        # Each token goes to top-k experts, distributed across num_gpus
        # Average tokens per GPU: batch_size * topk / num_gpus
        tokens_per_gpu = batch_size * c.num_topk / h.num_gpus

        # Data per token: hidden dimension
        bytes_per_token = c.hidden * (1 if is_fp8 else 2)  # FP8 or BF16
        total_bytes_per_link = tokens_per_gpu * bytes_per_token

        # DeepEP measured: NVLink EP8 ≈ 726 GB/s (dispatch), SM=24
        # Or theoretical: NVLink BW
        effective_bw = 726  # GB/s measured (DeepEP V2)
        comm_latency = total_bytes_per_link / (effective_bw * 1e9) * 1e6  # us

        # Add overhead: metadata, layout computation
        overhead_us = 5 + batch_size * 0.01  # base + per-token overhead

        return {
            "stage": "EP Dispatch (DeepEP)",
            "tokens_per_gpu": tokens_per_gpu,
            "bytes_per_link_mb": total_bytes_per_link / 1e6,
            "comm_latency_us": comm_latency,
            "overhead_us": overhead_us,
            "actual_latency_us": comm_latency + overhead_us,
            "effective_bw_gbs": effective_bw,
        }


class ExpertComputeStage:
    """Expert computation: SwiGLU with Grouped GEMM (DeepGEMM)."""

    def __init__(self, config: DeepSeekV3Config, hw: HardwareConfig):
        self.config = config
        self.hw = hw

    def estimate_latency_us(self, batch_size: int, is_mega_moe: bool = False):
        """Estimate expert computation time.

        Each MoE layer has: Linear1 (hidden→intermediate) + SwiGLU + Linear2 (intermediate→hidden)
        Using DeepGEMM Grouped GEMM (masked layout for decode).
        """
        c = self.config
        h = self.hw

        # Tokens per expert (after dispatch)
        tokens_per_expert = batch_size * c.num_topk / c.num_experts
        total_tokens = batch_size * c.num_topk

        # Two GEMMs per expert layer:
        # GEMM1: [tokens, hidden] × [intermediate, hidden] → [tokens, intermediate]
        # GEMM2: [tokens, intermediate] × [hidden, intermediate] → [tokens, hidden]
        # With EP: each GPU processes total_tokens/num_gpus tokens
        tokens_per_gpu = total_tokens // h.num_gpus
        flops_gemm1 = 2 * tokens_per_gpu * c.hidden * c.intermediate_size
        flops_gemm2 = 2 * tokens_per_gpu * c.intermediate_size * c.hidden
        total_flops = flops_gemm1 + flops_gemm2
        total_flops_gflops = total_flops / 1e9

        # Weight read bytes — with EP, each GPU has num_experts/num_gpus experts
        experts_per_gpu = c.num_experts // h.num_gpus
        weight_bytes_per_gpu = (c.hidden * c.intermediate_size * 2) * experts_per_gpu * 1  # FP8=1 byte (W1+W2)
        # Activation bytes (tokens arriving at this GPU after dispatch)
        tokens_per_gpu = total_tokens // h.num_gpus  # tokens assigned to this GPU's experts
        total_bytes = weight_bytes_per_gpu + tokens_per_gpu * c.hidden * 1  # weight + activation FP8

        # DeepGEMM measured: 1550 TFLOPS FP8 GEMM on H800
        deepgemm_tflops = 1550
        compute_latency = total_flops_gflops / (deepgemm_tflops * 1e3) * 1e6  # us
        mem_latency = total_bytes / (h.memory_bw_gbs * 1e9) * 1e6

        # Mega MoE fusion: overlap EP comm + compute, ~30% latency reduction
        if is_mega_moe:
            compute_latency *= 0.7  # communication overlapped

        actual_latency = max(compute_latency, mem_latency)

        return {
            "stage": "Expert Compute (DeepGEMM)" + (" [Mega MoE]" if is_mega_moe else ""),
            "total_tokens": total_tokens,
            "tokens_per_expert": tokens_per_expert,
            "total_flops_gflops": total_flops_gflops,
            "compute_latency_us": compute_latency,
            "mem_latency_us": mem_latency,
            "actual_latency_us": actual_latency,
        }


class EPCombineStage:
    """Expert Parallel All-to-All combine (DeepEP)."""

    def __init__(self, config: DeepSeekV3Config, hw: HardwareConfig):
        self.config = config
        self.hw = hw

    def estimate_latency_us(self, batch_size: int):
        c = self.config
        h = self.hw

        # Combine always uses BF16 (precision requirement)
        tokens_per_gpu = batch_size * c.num_topk / h.num_gpus
        bytes_per_token = c.hidden * 2  # BF16
        total_bytes_per_link = tokens_per_gpu * bytes_per_token

        # DeepEP measured: NVLink EP8 ≈ 740 GB/s (combine)
        effective_bw = 740
        comm_latency = total_bytes_per_link / (effective_bw * 1e9) * 1e6

        overhead_us = 5 + batch_size * 0.01

        return {
            "stage": "EP Combine (DeepEP)",
            "bytes_per_link_mb": total_bytes_per_link / 1e6,
            "comm_latency_us": comm_latency,
            "overhead_us": overhead_us,
            "actual_latency_us": comm_latency + overhead_us,
        }


# ─── Full Pipeline Simulator ─────────────────────────────────────────────────

class DeepSeekV3Pipeline:
    """Full DeepSeek-V3 decode pipeline simulator."""

    def __init__(self, config: Optional[DeepSeekV3Config] = None, hw: Optional[HardwareConfig] = None):
        self.config = config or DeepSeekV3Config()
        self.hw = hw or HardwareConfig()

        self.stages = {
            "attention": MLAAttentionStage(self.config, self.hw),
            "router": MoERouterStage(self.config, self.hw),
            "dispatch": EPDispatchStage(self.config, self.hw),
            "expert": ExpertComputeStage(self.config, self.hw),
            "combine": EPCombineStage(self.config, self.hw),
        }

    def simulate_decode_step(
        self,
        batch_size: int,
        seq_len: int,
        is_fp8_kv: bool = False,
        is_fp8_dispatch: bool = False,
        is_mega_moe: bool = False,
        is_sparse_attn: bool = False,
        sparse_topk: int = 2048,
    ):
        """Simulate one decode step latency breakdown."""
        c = self.config

        # 1. MLA Attention (per layer)
        attn_result = self.stages["attention"].estimate_latency_us(
            batch_size, seq_len, is_fp8_kv
        )

        # Sparse attention: only attend to topk tokens
        if is_sparse_attn:
            attn_result_sparse = self.stages["attention"].estimate_latency_us(
                batch_size, sparse_topk, is_fp8_kv
            )
            # Sparse MLA measured: 410 TFLOPS
            attn_result_sparse["stage"] = f"MLA Attention (FlashMLA Sparse, topk={sparse_topk})"
            attn_result = attn_result_sparse

        # 2. MoE Router (once per MoE layer)
        router_result = self.stages["router"].estimate_latency_us(batch_size)

        # 3. EP Dispatch
        dispatch_result = self.stages["dispatch"].estimate_latency_us(
            batch_size, is_fp8_dispatch
        )

        # 4. Expert Compute
        expert_result = self.stages["expert"].estimate_latency_us(
            batch_size, is_mega_moe
        )

        # 5. EP Combine
        combine_result = self.stages["combine"].estimate_latency_us(batch_size)

        # Per-layer latency (attention layers)
        attn_layer_us = attn_result["actual_latency_us"]

        # Per-MoE-layer latency
        moe_layer_us = (
            router_result["actual_latency_us"]
            + dispatch_result["actual_latency_us"]
            + expert_result["actual_latency_us"]
            + combine_result["actual_latency_us"]
        )

        # For Mega MoE: dispatch + compute + combine are fused
        if is_mega_moe:
            moe_layer_us = (
                router_result["actual_latency_us"]
                + max(
                    dispatch_result["actual_latency_us"]
                    + combine_result["actual_latency_us"],
                    expert_result["actual_latency_us"],
                )
            )

        # Total per-token latency (simplified: alternate dense-attn and MoE layers)
        # DeepSeek-V3: first layer is dense, rest alternate
        # Rough: 30 dense attn layers + 30 MoE layers
        num_dense_layers = 1  # first layer
        num_moe_layers = c.num_layers - num_dense_layers

        total_latency_us = (
            num_dense_layers * attn_layer_us
            + num_moe_layers * (attn_layer_us + moe_layer_us)
        )

        # Throughput
        tokens_per_second = batch_size / (total_latency_us / 1e6)

        # KV Cache per request
        kv_bytes_per_token = attn_result.get("kv_bytes_per_token", 512 * 2)
        kv_cache_per_request_mb = seq_len * kv_bytes_per_token / 1e6

        return {
            "config": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "is_fp8_kv": is_fp8_kv,
                "is_fp8_dispatch": is_fp8_dispatch,
                "is_mega_moe": is_mega_moe,
                "is_sparse_attn": is_sparse_attn,
                "sparse_topk": sparse_topk if is_sparse_attn else None,
            },
            "stages": {
                "attention": attn_result,
                "router": router_result,
                "dispatch": dispatch_result,
                "expert": expert_result,
                "combine": combine_result,
            },
            "per_layer_us": {
                "dense_attn": attn_layer_us,
                "moe_total": moe_layer_us,
                "moe_router": router_result["actual_latency_us"],
                "moe_dispatch": dispatch_result["actual_latency_us"],
                "moe_expert": expert_result["actual_latency_us"],
                "moe_combine": combine_result["actual_latency_us"],
            },
            "total_latency_ms": total_latency_us / 1000,
            "tokens_per_second": tokens_per_second,
            "kv_cache_per_request_mb": kv_cache_per_request_mb,
            "bottleneck_layer": "attention" if attn_layer_us > moe_layer_us else "moe",
            "bottleneck_stage": self._find_bottleneck(attn_result, router_result,
                                                       dispatch_result, expert_result,
                                                       combine_result),
        }

    def _find_bottleneck(self, attn, router, dispatch, expert, combine):
        stages = [
            ("attention", attn["actual_latency_us"]),
            ("router", router["actual_latency_us"]),
            ("dispatch", dispatch["actual_latency_us"]),
            ("expert", expert["actual_latency_us"]),
            ("combine", combine["actual_latency_us"]),
        ]
        return max(stages, key=lambda x: x[1])[0]


# ─── Sweep & Comparison ─────────────────────────────────────────────────────

def run_sweep(pipeline: DeepSeekV3Pipeline, seq_lens=None, batch_sizes=None):
    """Run parameter sweep and return results."""
    seq_lens = seq_lens or [1024, 4096, 8192, 32768, 131072]
    batch_sizes = batch_sizes or [1, 8, 32, 128, 512]

    results = []
    for bs in batch_sizes:
        for sl in seq_lens:
            # Baseline: BF16 KV, BF16 dispatch, no fusion
            r = pipeline.simulate_decode_step(bs, sl)
            results.append(("baseline", bs, sl, r))

            # FP8 KV cache
            r = pipeline.simulate_decode_step(bs, sl, is_fp8_kv=True)
            results.append(("fp8_kv", bs, sl, r))

            # FP8 dispatch + FP8 KV
            r = pipeline.simulate_decode_step(bs, sl, is_fp8_kv=True, is_fp8_dispatch=True)
            results.append(("fp8_all", bs, sl, r))

            # Mega MoE fusion
            r = pipeline.simulate_decode_step(bs, sl, is_fp8_kv=True, is_mega_moe=True)
            results.append(("mega_moe", bs, sl, r))

            # Sparse attention
            r = pipeline.simulate_decode_step(bs, sl, is_fp8_kv=True,
                                               is_sparse_attn=True, sparse_topk=2048)
            results.append(("sparse", bs, sl, r))

    return results


def print_comparison_table(results):
    """Print a comparison table for different optimization configs."""
    print("\n" + "=" * 100)
    print("DeepSeek-V3 Decode Pipeline — Optimization Comparison (H800 × 8, NVLink EP)")
    print("=" * 100)

    # Group by (batch_size, seq_len)
    from collections import defaultdict
    groups = defaultdict(list)
    for config_name, bs, sl, r in results:
        groups[(bs, sl)].append((config_name, r))

    # Print header
    configs = ["baseline", "fp8_kv", "fp8_all", "mega_moe", "sparse"]
    header = f"{'Batch':>5} {'SeqLen':>7}"
    for c in configs:
        header += f" | {c:>12}"
    header += f" | {'Best Config':>12}"
    print(header)
    print("-" * len(header))

    for (bs, sl) in sorted(groups.keys()):
        row_data = {cn: r for cn, r in groups[(bs, sl)]}
        row = f"{bs:>5} {sl:>7}"
        best_tps = 0
        best_config = ""
        for c in configs:
            r = row_data.get(c)
            if r:
                tps = r["tokens_per_second"]
                row += f" | {tps:>12.0f}"
                if tps > best_tps:
                    best_tps = tps
                    best_config = c
            else:
                row += f" | {'N/A':>12}"
        row += f" | {best_config:>12}"
        print(row)


def print_detailed_breakdown(result):
    """Print detailed latency breakdown for a single configuration."""
    cfg = result["config"]
    print(f"\n{'─' * 70}")
    print(f"  Batch={cfg['batch_size']}, SeqLen={cfg['seq_len']}, "
          f"FP8_KV={cfg['is_fp8_kv']}, FP8_dispatch={cfg['is_fp8_dispatch']}, "
          f"MegaMoE={cfg['is_mega_moe']}, Sparse={cfg['is_sparse_attn']}")
    print(f"{'─' * 70}")

    print(f"\n  Per-Layer Latency Breakdown (μs):")
    pl = result["per_layer_us"]
    print(f"    Dense Attention:    {pl['dense_attn']:>10.1f} μs")
    print(f"    MoE Total:          {pl['moe_total']:>10.1f} μs")
    print(f"      ├─ Router:        {pl['moe_router']:>10.1f} μs")
    print(f"      ├─ EP Dispatch:   {pl['moe_dispatch']:>10.1f} μs")
    print(f"      ├─ Expert GEMM:   {pl['moe_expert']:>10.1f} μs")
    print(f"      └─ EP Combine:    {pl['moe_combine']:>10.1f} μs")

    print(f"\n  Pipeline Summary:")
    print(f"    Total Latency:      {result['total_latency_ms']:>10.1f} ms")
    print(f"    Throughput:         {result['tokens_per_second']:>10.0f} tok/s")
    print(f"    KV Cache/Request:   {result['kv_cache_per_request_mb']:>10.1f} MB")
    print(f"    Bottleneck Layer:   {result['bottleneck_layer']:>10s}")
    print(f"    Bottleneck Stage:   {result['bottleneck_stage']:>10s}")


def print_deepseek_stack_summary():
    """Print summary of DeepSeek infra stack performance data."""
    print("\n" + "=" * 70)
    print("  DeepSeek Infrastructure Stack — Measured Performance (H800)")
    print("=" * 70)

    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │                    DeepSeek-V3 Inference Stack                  │
  │                                                                 │
  │  Layer 1: Communication (DeepEP V2)                            │
  │    NVLink EP8: 726/740 GB/s (dispatch/combine), SM=24→4-6     │
  │    IB EP8×2:  90/81 GB/s, SM=12                               │
  │    Features: ElasticBuffer, Handle Caching, 0-SM RDMA          │
  │                                                                 │
  │  Layer 2: Computation (DeepGEMM)                               │
  │    FP8 Dense GEMM: 1550 TFLOPS                                │
  │    Mega MoE: fused EP+GEMM+SwiGLU+EP (FP8×FP4)               │
  │    Grouped GEMM: contiguous (train) + masked (decode)          │
  │                                                                 │
  │  Layer 3: Attention (FlashMLA)                                 │
  │    Dense MLA Decode: 3000 GB/s (mem) / 660 TFLOPS (compute)  │
  │    Sparse MLA Decode: 410 TFLOPS (FP8 KV, topk=2048)         │
  │    Sparse MLA Prefill: 640 TFLOPS (H800)                      │
  │    Key innovations: Seesaw Scheduling, Crossover dequant      │
  │                                                                 │
  │  Data flow (per decode step):                                  │
  │    Q → FlashMLA (attn) → Router → DeepEP (dispatch) →        │
  │    DeepGEMM (expert GEMM) → DeepEP (combine) → Output         │
  └─────────────────────────────────────────────────────────────────┘
    """)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V3 Inference Pipeline Simulator")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--detailed", action="store_true", help="Print detailed breakdown")
    parser.add_argument("--all-optimizations", action="store_true", help="Compare all optimization configs")
    parser.add_argument("--json-output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    pipeline = DeepSeekV3Pipeline()

    print_deepseek_stack_summary()

    if args.sweep:
        results = run_sweep(pipeline)
        print_comparison_table(results)

        if args.json_output:
            serializable = []
            for cn, bs, sl, r in results:
                entry = {"config_name": cn, "batch_size": bs, "seq_len": sl}
                entry["latency_ms"] = r["total_latency_ms"]
                entry["throughput"] = r["tokens_per_second"]
                entry["bottleneck"] = r["bottleneck_stage"]
                serializable.append(entry)
            with open(args.json_output, "w") as f:
                json.dump(serializable, f, indent=2)
            print(f"\nResults saved to {args.json_output}")
        return

    if args.all_optimizations:
        configs = [
            ("Baseline (BF16)", dict()),
            ("FP8 KV Cache", dict(is_fp8_kv=True)),
            ("FP8 KV + FP8 Dispatch", dict(is_fp8_kv=True, is_fp8_dispatch=True)),
            ("FP8 KV + Mega MoE", dict(is_fp8_kv=True, is_mega_moe=True)),
            ("FP8 KV + Sparse (topk=2048)", dict(is_fp8_kv=True, is_sparse_attn=True, sparse_topk=2048)),
            ("All Optimized", dict(is_fp8_kv=True, is_fp8_dispatch=True, is_mega_moe=True,
                                    is_sparse_attn=True, sparse_topk=2048)),
        ]

        print(f"\nOptimization Comparison — Batch={args.batch_size}, SeqLen={args.seq_len}")
        print("=" * 80)
        print(f"{'Config':<30} {'Latency(ms)':>12} {'Throughput':>12} {'Bottleneck':>12}")
        print("-" * 80)

        all_results = []
        for name, kwargs in configs:
            r = pipeline.simulate_decode_step(args.batch_size, args.seq_len, **kwargs)
            print(f"{name:<30} {r['total_latency_ms']:>12.1f} "
                  f"{r['tokens_per_second']:>12.0f} {r['bottleneck_stage']:>12}")
            if args.detailed:
                print_detailed_breakdown(r)
            all_results.append((name, r))

        if args.json_output:
            with open(args.json_output, "w") as f:
                json.dump({name: r for name, r in all_results}, f, indent=2, default=str)
            print(f"\nResults saved to {args.json_output}")
        return

    # Default: single detailed run
    r = pipeline.simulate_decode_step(args.batch_size, args.seq_len)
    print_detailed_breakdown(r)

    # Also show optimized version
    print(f"\n{'─' * 70}")
    print("  With all optimizations (FP8 KV + Mega MoE + Sparse Attention):")
    r_opt = pipeline.simulate_decode_step(
        args.batch_size, args.seq_len,
        is_fp8_kv=True, is_mega_moe=True, is_sparse_attn=True, sparse_topk=2048
    )
    print_detailed_breakdown(r_opt)

    speedup = r["total_latency_ms"] / r_opt["total_latency_ms"]
    print(f"\n  Overall Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
