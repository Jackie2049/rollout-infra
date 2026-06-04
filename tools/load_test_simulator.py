#!/usr/bin/env python3
"""
LLM 推理服务负载测试模拟器
==========================
模拟 LLM serving 在不同负载下的延迟-吞吐行为。
CPU 可运行，基于排队论 + Roofline 模型。

实验:
1. 延迟-吞吐权衡曲线: 并发数 vs 延迟 vs 吞吐
2. SLO 达成率: TTFT/TPOT/E2E 合规分析
3. 泊松到达 vs 均匀到达: 尾延迟差异
4. 成本效率优化: GPU 选型 + 配置推荐
"""

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Data Models
# ============================================================

@dataclass
class GPUConfig:
    name: str
    tflops_fp16: float      # FP16 Tensor Core TFLOPS
    hbm_bw_gb: float        # HBM bandwidth GB/s
    vram_gb: float           # VRAM GB
    cost_per_hour: float     # USD/GPU-hour

GPUS = {
    "A100-80": GPUConfig("A100-80GB", 312, 2039, 80, 1.21),
    "H100-SXM": GPUConfig("H100-SXM", 990, 3350, 80, 2.50),
    "H200": GPUConfig("H200", 990, 4800, 141, 3.50),
    "L40S": GPUConfig("L40S", 362, 864, 48, 0.65),
    "A16": GPUConfig("A16", 15, 200, 15, 0.08),
}


@dataclass
class ModelConfig:
    name: str
    num_params_b: float     # billions
    hidden_dim: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int

MODELS = {
    "7B": ModelConfig("LLaMA-7B", 7, 4096, 32, 32, 128, 32000),
    "8B": ModelConfig("LLaMA3-8B", 8, 4096, 32, 8, 128, 128256),
    "70B": ModelConfig("LLaMA-70B", 70, 8192, 80, 8, 128, 32000),
}


@dataclass
class Request:
    id: int
    arrive_time: float      # seconds
    prompt_tokens: int
    max_output_tokens: int
    priority: int = 0       # 0 = normal

    # Filled during simulation
    start_time: float = 0.0
    first_token_time: float = 0.0
    end_time: float = 0.0
    output_tokens: int = 0


@dataclass
class SLOConfig:
    ttft_ms: float = 2000   # Time To First Token < 2s
    tpot_ms: float = 100    # Time Per Output Token < 100ms
    e2e_ms: float = 30000   # End-to-end < 30s


# ============================================================
# Performance Model
# ============================================================

class InferencePerfModel:
    """基于 Roofline 的推理性能模型"""

    def __init__(self, model: ModelConfig, gpu: GPUConfig):
        self.model = model
        self.gpu = gpu

    def prefill_time_ms(self, seq_len: int, batch_size: int = 1) -> float:
        """Prefill 时间 (compute-bound)"""
        # FLOPs ≈ 2 × P × seq_len (P = params)
        flops = 2 * self.model.num_params_b * 1e9 * seq_len
        # Throughput ≈ TFLOPS × util
        tflops = self.gpu.tflops_fp16 * 0.45  # ~45% MFU for prefill
        time_s = flops / (tflops * 1e12)
        # Batch amortizes fixed overhead
        return time_s * 1000 / math.sqrt(batch_size)  # sqrt scaling for prefill

    def decode_time_ms(self, seq_len: int, batch_size: int = 1) -> float:
        """Decode 单步时间 (memory-bound)"""
        # Weight read: 2 × P × dtype_bytes
        weight_bytes = 2 * self.model.num_params_b * 1e9 * 2  # FP16
        # KV cache read: 2 × layers × kv_heads × head_dim × 2 × seq_len × 2bytes
        kv_bytes = (2 * self.model.num_layers * self.model.num_kv_heads
                    * self.model.head_dim * 2 * seq_len * 2)
        total_bytes = weight_bytes + kv_bytes

        # Effective BW with batching (weight reuse)
        effective_bw = self.gpu.hbm_bw_gb * 1e9
        time_s = total_bytes / effective_bw

        # Batch scaling: near-linear for memory-bound
        return time_s * 1000 * (1 + 0.02 * math.log2(max(1, batch_size)))

    def max_batch_size(self, max_seq_len: int) -> int:
        """最大 batch size (受 VRAM 限制)"""
        # Model weights
        weight_gb = self.model.num_params_b * 2 / 1e9  # FP16
        # KV cache per token per layer: 2 * kv_heads * head_dim * 2 bytes
        kv_per_token_bytes = 2 * self.model.num_layers * self.model.num_kv_heads * self.model.head_dim * 2
        # Available for KV
        kv_budget_gb = self.gpu.vram_gb - weight_gb - 2  # 2GB overhead
        if kv_budget_gb <= 0:
            return 0
        kv_per_request_bytes = kv_per_token_bytes * max_seq_len
        max_bs = int(kv_budget_gb * 1e9 / kv_per_request_bytes)
        return max(1, min(max_bs, 512))


# ============================================================
# Simulator
# ============================================================

class LoadTestSimulator:
    """负载测试模拟器"""

    def __init__(self, model: ModelConfig, gpu: GPUConfig, slo: SLOConfig = None):
        self.model = model
        self.gpu = gpu
        self.slo = slo or SLOConfig()
        self.perf = InferencePerfModel(model, gpu)

    def generate_requests(
        self,
        num_requests: int,
        avg_prompt_len: int = 256,
        avg_output_len: int = 128,
        arrival_rate: float = 10.0,  # requests/second
        distribution: str = "poisson",
    ) -> List[Request]:
        """生成请求流"""
        requests = []
        rng = np.random.default_rng(42)

        for i in range(num_requests):
            if distribution == "poisson":
                interval = rng.exponential(1.0 / arrival_rate)
            else:  # uniform
                interval = 1.0 / arrival_rate

            prompt_len = max(16, int(rng.lognormal(math.log(avg_prompt_len), 0.5)))
            output_len = max(8, int(rng.lognormal(math.log(avg_output_len), 0.5)))

            arrive = (requests[-1].arrive_time + interval) if requests else 0.0
            requests.append(Request(
                id=i,
                arrive_time=arrive,
                prompt_tokens=prompt_len,
                max_output_tokens=output_len,
            ))

        return requests

    def simulate(
        self,
        requests: List[Request],
        max_batch_size: int = None,
        max_concurrent: int = 32,
    ) -> Dict:
        """运行模拟"""
        if max_batch_size is None:
            max_batch_size = self.perf.max_batch_size(2048)

        # Simple event-driven simulation
        running: List[Request] = []
        completed: List[Request] = []
        waiting = list(requests)
        current_time = 0.0

        stats = {
            "total_requests": len(requests),
            "ttft": [],
            "tpot": [],
            "e2e": [],
            "batch_sizes": [],
            "throughput_per_step": [],
        }

        step = 0
        while waiting or running:
            # 1. Admit new requests
            while waiting and len(running) < max_concurrent:
                req = waiting[0]
                if req.arrive_time <= current_time:
                    waiting.pop(0)
                    req.start_time = max(current_time, req.arrive_time)
                    running.append(req)
                else:
                    break

            if not running:
                if waiting:
                    # Jump to next arrival
                    current_time = waiting[0].arrive_time
                    continue
                break

            # 2. Separate prefill and decode requests
            prefill_reqs = [r for r in running if r.output_tokens == 0]
            decode_reqs = [r for r in running if r.output_tokens > 0]

            # 3. Process prefill (one at a time, simplified)
            for req in prefill_reqs:
                t = self.perf.prefill_time_ms(req.prompt_tokens) / 1000
                req.first_token_time = current_time + t
                req.output_tokens = 1
                current_time = max(current_time, req.first_token_time)

            # 4. Process decode step (batched)
            batch_size = len(decode_reqs)
            if batch_size > 0:
                avg_seq_len = np.mean([r.prompt_tokens + r.output_tokens for r in decode_reqs])
                step_time = self.perf.decode_time_ms(int(avg_seq_len), batch_size) / 1000

                stats["batch_sizes"].append(batch_size)
                stats["throughput_per_step"].append(batch_size / step_time)

                for req in decode_reqs:
                    req.output_tokens += 1

                current_time += step_time

            # 5. Check completed
            finished = [r for r in running if r.output_tokens >= r.max_output_tokens]
            for req in finished:
                req.end_time = current_time
                running.remove(req)
                completed.append(req)

                ttft = req.first_token_time - req.start_time
                tpot = (req.end_time - req.first_token_time) / req.output_tokens
                e2e = req.end_time - req.start_time

                stats["ttft"].append(ttft * 1000)
                stats["tpot"].append(tpot * 1000)
                stats["e2e"].append(e2e * 1000)

            step += 1
            if step > 100000:  # Safety limit
                break

        # Compute summary
        summary = self._compute_summary(stats, completed)
        summary["max_batch"] = max_batch_size
        summary["completed"] = len(completed)
        summary["sim_steps"] = step
        summary["sim_time_s"] = current_time

        return summary

    def _compute_summary(self, stats: Dict, completed: List[Request]) -> Dict:
        summary = {}
        for metric in ["ttft", "tpot", "e2e"]:
            vals = stats[metric]
            if vals:
                summary[f"{metric}_p50"] = np.percentile(vals, 50)
                summary[f"{metric}_p90"] = np.percentile(vals, 90)
                summary[f"{metric}_p99"] = np.percentile(vals, 99)
                summary[f"{metric}_mean"] = np.mean(vals)
            else:
                summary[f"{metric}_p50"] = 0

        # SLO compliance
        slo = self.slo
        if stats["ttft"]:
            summary["slo_ttft_ok"] = sum(1 for v in stats["ttft"] if v < slo.ttft_ms) / len(stats["ttft"]) * 100
        if stats["tpot"]:
            summary["slo_tpot_ok"] = sum(1 for v in stats["tpot"] if v < slo.tpot_ms) / len(stats["tpot"]) * 100
        if stats["e2e"]:
            summary["slo_e2e_ok"] = sum(1 for v in stats["e2e"] if v < slo.e2e_ms) / len(stats["e2e"]) * 100

        # Throughput
        if stats["throughput_per_step"]:
            summary["avg_throughput_tok_s"] = np.mean(stats["throughput_per_step"])
            summary["peak_throughput_tok_s"] = np.max(stats["throughput_per_step"])
        if stats["batch_sizes"]:
            summary["avg_batch"] = np.mean(stats["batch_sizes"])

        return summary


# ============================================================
# Experiments
# ============================================================

def exp1_latency_throughput_tradeoff():
    """实验 1: 延迟-吞吐权衡曲线"""
    print("=" * 70)
    print("实验 1: 延迟-吞吐权衡曲线")
    print("=" * 70)

    model = MODELS["8B"]
    gpu = GPUS["A100-80"]
    sim = LoadTestSimulator(model, gpu)

    print(f"\n  模型: {model.name}, GPU: {gpu.name}")
    print(f"  最大 batch size (seq=2048): {sim.perf.max_batch_size(2048)}")
    print()

    print(f"  {'并发':>6} | {'TTFT P50':>10} | {'TPOT P50':>10} | {'E2E P50':>10} | {'吞吐 tok/s':>12} | {'SLO TTFT':>8} | {'SLO TPOT':>8}")
    print(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*12} | {'-'*8} | {'-'*8}")

    for concurrency in [1, 4, 8, 16, 32, 64, 128]:
        reqs = sim.generate_requests(200, avg_prompt_len=256, avg_output_len=128,
                                     arrival_rate=concurrency * 0.5, distribution="poisson")
        result = sim.simulate(reqs, max_concurrent=concurrency)

        ttft = result.get("ttft_p50", 0)
        tpot = result.get("tpot_p50", 0)
        e2e = result.get("e2e_p50", 0)
        throughput = result.get("avg_throughput_tok_s", 0)
        slo_ttft = result.get("slo_ttft_ok", 0)
        slo_tpot = result.get("slo_tpot_ok", 0)

        print(f"  {concurrency:>6} | {ttft:>9.1f}ms | {tpot:>9.1f}ms | {e2e:>9.1f}ms | {throughput:>10.0f} | {slo_ttft:>6.1f}% | {slo_tpot:>6.1f}%")

    print()
    print("  关键洞察:")
    print("  - 吞吐随并发线性增长 → 直到 TPOT 违规")
    print("  - 最优操作点 = SLO 刚好满足的最大并发")
    print("  - 8B on A100: 最优 ~32-64 并发")
    print()

    return {"experiment": 1, "model": model.name, "gpu": gpu.name}


def exp2_slo_compliance():
    """实验 2: SLO 达成率分析"""
    print("=" * 70)
    print("实验 2: SLO 达成率分析")
    print("=" * 70)

    model = MODELS["8B"]

    configs = [
        ("A100-80", 32, "最优"),
        ("H100-SXM", 64, "高配"),
        ("L40S", 16, "经济"),
    ]

    print(f"\n  模型: {model.name}")
    print(f"  SLO: TTFT < 2000ms, TPOT < 100ms, E2E < 30000ms")
    print()

    for gpu_name, concurrency, tier in configs:
        gpu = GPUS[gpu_name]
        slo = SLOConfig(ttft_ms=2000, tpot_ms=100, e2e_ms=30000)
        sim = LoadTestSimulator(model, gpu, slo)

        reqs = sim.generate_requests(300, avg_prompt_len=512, avg_output_len=256,
                                     arrival_rate=concurrency * 0.3)
        result = sim.simulate(reqs, max_concurrent=concurrency)

        print(f"  {gpu.name} ({tier}场景):")
        print(f"    TTFT: P50={result.get('ttft_p50',0):.1f}ms, P99={result.get('ttft_p99',0):.1f}ms, SLO={result.get('slo_ttft_ok',0):.1f}%")
        print(f"    TPOT: P50={result.get('tpot_p50',0):.1f}ms, P99={result.get('tpot_p99',0):.1f}ms, SLO={result.get('slo_tpot_ok',0):.1f}%")
        print(f"    E2E:  P50={result.get('e2e_p50',0):.1f}ms, P99={result.get('e2e_p99',0):.1f}ms, SLO={result.get('slo_e2e_ok',0):.1f}%")
        print(f"    吞吐: {result.get('avg_throughput_tok_s',0):.0f} tok/s")
        print(f"    成本: ${gpu.cost_per_hour:.2f}/GPU-hr")
        cost_per_mtok = gpu.cost_per_hour / (result.get('avg_throughput_tok_s', 1) * 3600 / 1e6)
        print(f"    单位成本: ${cost_per_mtok:.2f}/M tok")
        print()

    return {"experiment": 2}


def exp3_arrival_patterns():
    """实验 3: 泊松到达 vs 均匀到达"""
    print("=" * 70)
    print("实验 3: 泊松到达 vs 均匀到达")
    print("=" * 70)

    model = MODELS["8B"]
    gpu = GPUS["A100-80"]
    sim = LoadTestSimulator(model, gpu)

    print(f"\n  模型: {model.name}, GPU: {gpu.name}")
    print()

    for rate in [5, 15, 30]:
        print(f"  到达率: {rate} req/s")

        for dist in ["poisson", "uniform"]:
            reqs = sim.generate_requests(500, avg_prompt_len=256, avg_output_len=128,
                                         arrival_rate=rate, distribution=dist)
            result = sim.simulate(reqs, max_concurrent=64)

            p50 = result.get("e2e_p50", 0)
            p99 = result.get("e2e_p99", 0)
            slo_ok = result.get("slo_e2e_ok", 0)
            throughput = result.get("avg_throughput_tok_s", 0)

            dist_name = "泊松" if dist == "poisson" else "均匀"
            print(f"    {dist_name}: P50={p50:.0f}ms, P99={p99:.0f}ms, SLO={slo_ok:.1f}%, 吞吐={throughput:.0f}tok/s")

        print()

    print("  关键洞察:")
    print("  - 泊松到达 P99 延迟显著高于均匀到达 (burst 效应)")
    print("  - 高负载时差异更明显 (30 req/s)")
    print("  - 生产环境应按泊松建模, 均匀过于乐观")
    print()

    return {"experiment": 3}


def exp4_cost_optimization():
    """实验 4: 成本效率优化"""
    print("=" * 70)
    print("实验 4: 成本效率优化")
    print("=" * 70)

    print("\n  场景: 8B 模型在线服务, SLO: TTFT<2s, TPOT<100ms")
    print()

    model = MODELS["8B"]
    slo = SLOConfig()

    print(f"  {'GPU':>12} | {'最大并发':>8} | {'吞吐 tok/s':>12} | {'$/Mtok':>8} | {'SLO TTFT':>8} | {'SLO TPOT':>8} | {'推荐':>6}")
    print(f"  {'-'*12} | {'-'*8} | {'-'*12} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*6}")

    best_cost = float('inf')
    best_gpu = ""

    for gpu_name, gpu in GPUS.items():
        sim = LoadTestSimulator(model, gpu, slo)
        max_bs = sim.perf.max_batch_size(2048)
        concurrency = min(max_bs, 64)

        reqs = sim.generate_requests(300, avg_prompt_len=256, avg_output_len=128,
                                     arrival_rate=concurrency * 0.3)
        result = sim.simulate(reqs, max_concurrent=concurrency)

        throughput = result.get("avg_throughput_tok_s", 0)
        if throughput > 0:
            cost_per_mtok = gpu.cost_per_hour / (throughput * 3600 / 1e6)
        else:
            cost_per_mtok = float('inf')

        slo_ttft = result.get("slo_ttft_ok", 0)
        slo_tpot = result.get("slo_tpot_ok", 0)

        # SLO 合格且成本最低
        recommend = ""
        if slo_ttft > 95 and slo_tpot > 95:
            if cost_per_mtok < best_cost:
                best_cost = cost_per_mtok
                best_gpu = gpu.name
            recommend = "✓"

        print(f"  {gpu.name:>12} | {concurrency:>8} | {throughput:>10.0f} | {cost_per_mtok:>7.3f} | {slo_ttft:>6.1f}% | {slo_tpot:>6.1f}% | {recommend:>6}")

    print()
    print(f"  最佳性价比: {best_gpu} @ ${best_cost:.3f}/M tok")
    print()

    # 70B 场景
    print("  场景: 70B 模型 (需要 TP=2-8)")
    model_70b = MODELS["70B"]
    for gpu_name in ["A100-80", "H100-SXM", "H200"]:
        gpu = GPUS[gpu_name]
        tp = max(1, math.ceil(2 * model_70b.num_params_b * 2 / gpu.vram_gb))
        sim = LoadTestSimulator(model_70b, gpu, slo)
        # With TP, multiply effective compute and BW
        effective_gpu = GPUConfig(
            f"{gpu.name}×{tp}", gpu.tflops_fp16 * tp, gpu.hbm_bw_gb * tp,
            gpu.vram_gb * tp, gpu.cost_per_hour * tp
        )
        sim_tp = LoadTestSimulator(model_70b, effective_gpu, slo)
        concurrency = min(sim_tp.perf.max_batch_size(2048), 32)

        reqs = sim_tp.generate_requests(200, avg_prompt_len=256, avg_output_len=128,
                                         arrival_rate=concurrency * 0.2)
        result = sim_tp.simulate(reqs, max_concurrent=concurrency)

        throughput = result.get("avg_throughput_tok_s", 0)
        cost = effective_gpu.cost_per_hour / max(1, throughput * 3600 / 1e6)
        slo_ttft = result.get("slo_ttft_ok", 0)
        slo_tpot = result.get("slo_tpot_ok", 0)

        print(f"    {effective_gpu.name}: TP={tp}, 吞吐={throughput:.0f}tok/s, "
              f"${cost:.2f}/Mtok, SLO TTFT={slo_ttft:.0f}%, TPOT={slo_tpot:.0f}%")

    print()

    return {"experiment": 4, "best_cost_gpu": best_gpu}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LLM 推理服务负载测试模拟器")
    parser.add_argument("--experiment", "-e", type=int, default=0,
                        help="实验编号 (1-4, 0=全部)")
    parser.add_argument("--model", type=str, default="8B",
                        help="模型名称 (7B/8B/70B)")
    parser.add_argument("--gpu", type=str, default="A100-80",
                        help="GPU 名称 (A100-80/H100-SXM/H200/L40S/A16)")
    args = parser.parse_args()

    experiments = {
        1: exp1_latency_throughput_tradeoff,
        2: exp2_slo_compliance,
        3: exp3_arrival_patterns,
        4: exp4_cost_optimization,
    }

    results = {}

    if args.experiment > 0:
        if args.experiment in experiments:
            results[args.experiment] = experiments[args.experiment]()
        else:
            print(f"Unknown experiment {args.experiment}")
            return
    else:
        for i, func in experiments.items():
            results[i] = func()

    with open("load_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"结果已保存到 load_test_results.json")


if __name__ == "__main__":
    main()
