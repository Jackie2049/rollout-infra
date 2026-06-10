#!/usr/bin/env python3
"""
AI Serving Observability Simulator
Simulates metrics, SLO compliance, autoscaling, and incident response.

Usage:
    python3 tools/ai_serving_observability_simulator.py [class_name]

Classes:
    - MetricsSimulator: vLLM Prometheus metrics simulation
    - SLOSimulator: SLO/SLA compliance and error budget tracking
    - AutoscalingSimulator: HPA/VPA scaling simulation for AI serving
    - IncidentResponseSimulator: Common AI serving incident simulation and runbooks
"""

import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestMetrics:
    ttft_ms: float  # Time to first token
    itl_ms: float  # Inter-token latency
    total_tokens: int
    num_generation_tokens: int
    success: bool
    preempted: bool = False


class MetricsSimulator:
    """Simulates vLLM V1 Prometheus metrics for a production serving scenario."""

    def __init__(self, model_size: str = "7b_int4", gpu_type: str = "rtx4090"):
        self.model_size = model_size
        self.gpu_type = gpu_type
        # Baseline numbers from RTX 4090 benchmarks
        self.baseline_throughput = {
            "7b_int4": 4791,  # tokens/s
            "7b_bf16": 79,   # tokens/s B=1
            "0.5b_int4": 13598,
        }
        self.baseline_kv_usage_pct = {
            "7b_int4_int8kv": 50,  # moderate usage with INT8 KV
            "7b_bf16": 95,  # near OOM with BF16
        }

    def simulate_metrics(self, n_requests: int = 100, batch_size: int = 32) -> dict[str, Any]:
        """Simulate a batch of requests and compute aggregate metrics."""
        requests = []
        base_throughput = self.baseline_throughput.get(self.model_size, 1000)
        itl_base_ms = 1000 / base_throughput * batch_size  # ms per token at given batch size

        for i in range(n_requests):
            # Simulate TTFT (prefill time depends on prompt length)
            prompt_length = np.random.randint(64, 2048)
            ttft_ms = prompt_length * 0.05 + np.random.exponential(20)  # ~0.05ms per prompt token

            # Simulate ITL (decode time)
            itl_ms = itl_base_ms + np.random.exponential(0.5)

            # Some requests may be preempted
            preempted = np.random.random() < 0.02  # 2% preemption rate
            if preempted:
                ttft_ms *= 3  # 3x TTFT on re-computation
                itl_ms *= 1.5

            # Some requests may fail
            success = np.random.random() > 0.001  # 99.9% success

            total_tokens = prompt_length + np.random.randint(32, 512)
            num_gen_tokens = total_tokens - prompt_length

            requests.append(RequestMetrics(
                ttft_ms=ttft_ms,
                itl_ms=itl_ms,
                total_tokens=total_tokens,
                num_generation_tokens=num_gen_tokens,
                success=success,
                preempted=preempted,
            ))

        # Compute aggregate metrics
        ttfts = [r.ttft_ms for r in requests if r.success]
        itls = [r.itl_ms for r in requests if r.success]
        total_gen_tokens = sum(r.num_generation_tokens for r in requests if r.success)
        total_prompt_tokens = sum(r.total_tokens - r.num_generation_tokens for r in requests if r.success)
        n_success = sum(1 for r in requests if r.success)
        n_preempted = sum(1 for r in requests if r.preempted)

        # KV cache usage (varies with concurrent requests)
        kv_usage_pct = min(100, np.random.uniform(30, 80) + n_requests / batch_size * 5)

        result = {
            "model_size": self.model_size,
            "gpu_type": self.gpu_type,
            "n_requests": n_requests,
            "batch_size": batch_size,
            "n_success": n_success,
            "n_failed": n_requests - n_success,
            "n_preempted": n_preempted,
            "availability_pct": n_success / n_requests * 100,

            # Throughput metrics
            "prompt_throughput_tok_s": total_prompt_tokens / (sum(ttfts) / 1000) if ttfts else 0,
            "generation_throughput_tok_s": total_gen_tokens / (sum(itls) * len(itls) / 1000) if itls else 0,

            # Latency metrics
            "ttft_p50_ms": np.percentile(ttfts, 50) if ttfts else None,
            "ttft_p90_ms": np.percentile(ttfts, 90) if ttfts else None,
            "ttft_p99_ms": np.percentile(ttfts, 99) if ttfts else None,
            "itl_p50_ms": np.percentile(itls, 50) if itls else None,
            "itl_p99_ms": np.percentile(itls, 99) if itls else None,

            # Cache metrics
            "gpu_cache_usage_pct": kv_usage_pct,
            "prefix_cache_hit_rate_pct": np.random.uniform(20, 70),  # varies with workload

            # Prometheus metric names
            "prometheus_metrics": {
                "vllm:num_requests_running": batch_size,
                "vllm:gpu_cache_usage_perc": kv_usage_pct,
                "vllm:e2e_request_latency_seconds": np.mean(ttfts) / 1000,
                "vllm:prompt_tokens_total": total_prompt_tokens,
                "vllm:generation_tokens_total": total_gen_tokens,
            },
        }
        return result


class SLOSimulator:
    """Simulates SLO compliance and error budget tracking."""

    def __init__(self, slo_config: dict[str, float] | None = None):
        self.slo_config = slo_config or {
            "ttft_p99_ms": 500,
            "itl_p99_ms": 100,
            "throughput_tok_s": 1000,
            "availability_pct": 99.9,
        }

    def check_slo_compliance(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Check if metrics meet SLO targets."""
        results = {}
        for slo_name, slo_target in self.slo_config.items():
            metric_name = slo_name
            actual_value = metrics.get(metric_name)

            if actual_value is not None:
                compliant = actual_value <= slo_target if "p99" in slo_name or "p50" in slo_name else actual_value >= slo_target
                results[slo_name] = {
                    "target": slo_target,
                    "actual": actual_value,
                    "compliant": compliant,
                    "gap_pct": ((actual_value - slo_target) / slo_target * 100) if slo_target > 0 else 0,
                }

        # Compute error budget
        availability_actual = metrics.get("availability_pct", 100)
        availability_slo = self.slo_config["availability_pct"]
        error_budget_pct = availability_actual - availability_slo

        result = {
            "slo_checks": results,
            "all_compliant": all(r["compliant"] for r in results.values()),
            "error_budget_pct": error_budget_pct,
            "error_budget_remaining_pct": max(0, error_budget_pct),
            "error_budget_consumed_pct": max(0, -error_budget_pct),
            "recommendation": "All SLOs met → continue normal operation" if all(r["compliant"] for r in results.values()) else "SLO violation → investigate and remediate",
        }
        return result

    def simulate_error_budget_burn(self, n_days: int = 30) -> dict[str, Any]:
        """Simulate error budget burn rate over time."""
        availability_slo = self.slo_config["availability_pct"]
        total_budget_pct = 100 - availability_slo  # e.g. 0.1% for 99.9%
        # In 30 days, total budget = 0.1% * 30 * 24 * 60 = 432 minutes

        daily_budget_pct = total_budget_pct / n_days

        # Simulate daily incidents
        daily_downtime_min = np.random.exponential(5, n_days)  # average 5 min downtime per day
        cumulative_downtime_min = np.cumsum(daily_downtime_min)
        total_budget_min = total_budget_pct / 100 * n_days * 24 * 60

        budget_remaining_pct = (total_budget_min - cumulative_downtime_min) / total_budget_min * 100

        budget_exhausted_day = None
        for i in range(n_days):
            if budget_remaining_pct[i] <= 0:
                budget_exhausted_day = i + 1
                break

        result = {
            "availability_slo_pct": availability_slo,
            "total_error_budget_pct": total_budget_pct,
            "total_error_budget_min": total_budget_min,
            "daily_budget_min": daily_budget_pct / 100 * 24 * 60,
            "avg_daily_downtime_min": np.mean(daily_downtime_min),
            "budget_exhausted_day": budget_exhausted_day,
            "budget_remaining_at_30d_pct": max(0, budget_remaining_pct[-1]),
            "burn_rate_pct_per_day": np.mean(daily_downtime_min) / (total_budget_min / n_days) * 100,
            "recommendation": f"Budget exhausted at day {budget_exhausted_day}" if budget_exhausted_day else f"Budget healthy at 30d: {budget_remaining_pct[-1]:.1f}% remaining",
        }
        return result


class AutoscalingSimulator:
    """Simulates autoscaling strategies for AI serving."""

    def __init__(self, gpu_type: str = "rtx4090", model: str = "7b_int4"):
        self.gpu_type = gpu_type
        self.model = model
        self.max_throughput_per_pod = 4791  # tokens/s for 7B INT4 on RTX 4090

    def simulate_hpa(self, n_hours: int = 24, min_pods: int = 1, max_pods: int = 10,
                     target_utilization_pct: float = 70) -> dict[str, Any]:
        """Simulate HPA behavior over time."""
        # Simulate request rate over 24 hours (business hours pattern)
        hours = np.arange(n_hours)
        # Peak at 10-14, low at 0-6
        request_rate = np.array([
            max(0.2, 0.5 + 0.5 * np.sin((h - 10) * np.pi / 12)) for h in hours
        ])
        request_rate = request_rate * 5000  # scale to tokens/s demand

        pods = []
        utilization = []
        cold_starts = 0
        total_tokens_served = 0

        for h in range(n_hours):
            # Calculate required pods
            throughput_needed = request_rate[h]
            n_pods_needed = max(min_pods, min(max_pods,
                int(np.ceil(throughput_needed / (self.max_throughput_per_pod * target_utilization_pct / 100)))))

            if n_pods_needed > len(pods) if pods else min_pods:
                # Scale up → cold start
                new_pods = n_pods_needed - (len(pods) if pods else 0)
                cold_starts += max(0, new_pods)
                n_pods = n_pods_needed
            else:
                # Scale down → delayed by cooldown
                n_pods = len(pods) if pods else min_pods

            pods.append(n_pods)

            # Utilization
            actual_throughput = min(throughput_needed, n_pods * self.max_throughput_per_pod)
            util = actual_throughput / (n_pods * self.max_throughput_per_pod) * 100
            utilization.append(util)
            total_tokens_served += int(actual_throughput * 3600)  # tokens in 1 hour

        result = {
            "strategy": "HPA",
            "model": self.model,
            "gpu_type": self.gpu_type,
            "max_throughput_per_pod_tok_s": self.max_throughput_per_pod,
            "target_utilization_pct": target_utilization_pct,
            "min_pods": min_pods,
            "max_pods": max_pods,
            "n_hours": n_hours,
            "total_cold_starts": cold_starts,
            "cold_start_time_s": 30 * cold_starts,  # 30s per cold start
            "avg_pods": np.mean(pods),
            "peak_pods": max(pods),
            "avg_utilization_pct": np.mean(utilization),
            "total_tokens_served": total_tokens_served,
            "pod_hours": sum(pods),
            "cost_per_pod_hour": 1.0,  # normalized cost
            "total_cost": sum(pods) * 1.0,
            "cold_start_overhead_pct": cold_starts * 30 / (n_hours * 3600) * 100,
        }
        return result

    def simulate_sleep_wake(self, n_hours: int = 24) -> dict[str, Any]:
        """Simulate sleep/wake mode instead of traditional HPA."""
        # Same request pattern
        hours = np.arange(n_hours)
        request_rate = np.array([
            max(0.2, 0.5 + 0.5 * np.sin((h - 10) * np.pi / 12)) for h in hours
        ]) * 5000

        # Single pod always present, sleep when idle
        active_hours = sum(1 for h in range(n_hours) if request_rate[h] > 500)
        sleep_hours = n_hours - active_hours

        # Wake time ~1s vs cold start ~30s
        wake_time_s = active_hours * 1  # 1s per wake
        total_tokens = sum(int(request_rate[h] * 3600) for h in range(n_hours))

        result = {
            "strategy": "Sleep/Wake",
            "pod_count": 1,  # always 1 pod
            "active_hours": active_hours,
            "sleep_hours": sleep_hours,
            "wake_time_per_wake_s": 1,
            "total_wake_time_s": wake_time_s,
            "total_tokens_served": total_tokens,
            "cost": n_hours * 1.0,  # 1 pod always present but sleep saves compute
            "vs_hpa_comparison": {
                "cold_start_s": 30,
                "wake_time_s": 1,
                "speedup": 30,  # sleep/wake 30x faster than cold start
                "trade_off": "1 pod always running (cost) vs HPA pods start/stop (cold start)",
            },
            "recommendation": "Sleep/wake > HPA for RTX 4090 (single pod, no cold start, fast wake)",
        }
        return result


class IncidentResponseSimulator:
    """Simulates common AI serving incidents and generates runbooks."""

    INCIDENT_TYPES = {
        "oom": {
            "name": "GPU OOM",
            "symptoms": "gpu_cache_usage > 95%, request failures, CUDA OOM errors",
            "severity": "high",
            "mttr_min": 10,
        },
        "high_latency": {
            "name": "High P99 Latency",
            "symptoms": "P99 TTFT > 2s or P99 ITL > 200ms",
            "severity": "medium",
            "mttr_min": 30,
        },
        "gpu_failure": {
            "name": "GPU Hardware Failure",
            "symptoms": "ECC errors, XID errors, GPU unreachable",
            "severity": "critical",
            "mttr_min": 120,
        },
        "low_throughput": {
            "name": "Low Throughput",
            "symptoms": "generation_throughput < 500 tok/s despite sufficient requests",
            "severity": "medium",
            "mttr_min": 15,
        },
    }

    def simulate_incident(self, incident_type: str = "oom") -> dict[str, Any]:
        """Simulate an incident and generate a runbook."""
        incident = self.INCIDENT_TYPES[incident_type]
        mttr = incident["mttr_min"] + np.random.exponential(5)

        # Impact estimation
        requests_per_min = 100  # baseline
        downtime_min = mttr
        affected_requests = requests_per_min * downtime_min

        result = {
            "incident_type": incident_type,
            "incident_name": incident["name"],
            "severity": incident["severity"],
            "symptoms": incident["symptoms"],
            "mttr_min": mttr,
            "affected_requests": int(affected_requests),
            "availability_impact_pct": downtime_min / (24 * 60) * 100,

            # Runbook
            "runbook": {
                "oom": [
                    "1. Check gpu_cache_usage_perc → if > 95% → reduce max_num_seqs",
                    "2. Check concurrent requests → if > threshold → implement queuing",
                    "3. Enable INT8 KV cache → 50% memory saving → may resolve",
                    "4. Enable INT4 model weights → 75% memory saving → if still OOM",
                    "5. Add GPU → scale horizontally → if all above fail",
                ],
                "high_latency": [
                    "1. Check TTFT vs ITL → which is high → different root causes",
                    "2. TTFT high → check prefill time → chunked prefill → reduce chunk size",
                    "3. ITL high → check batch size → increase → amortize weight reads",
                    "4. Check preemptions → if high → reduce concurrent or increase KV cache",
                    "5. Profile with Nsight → find slowest kernel → optimize",
                ],
                "gpu_failure": [
                    "1. Check DCGM → ECC errors → XID → confirm hardware failure",
                    "2. Drain requests from failed GPU → route to healthy GPUs",
                    "3. Restart vLLM on healthy GPU → checkpoint recovery if training",
                    "4. Replace GPU → if ECC persistent → hardware replacement",
                    "5. Update incident log → post-mortem → preventive measures",
                ],
                "low_throughput": [
                    "1. Check num_requests_running → if low → increase max_num_seqs",
                    "2. Check gpu_utilization → if low → GPU idle → increase batch",
                    "3. Check prefix_cache_hit_rate → if low → optimize prefix sharing",
                    "4. Check speculative decoding → if draft quality low → disable or switch",
                    "5. Verify model config → INT4 vs BF16 → quantization matters!",
                ],
            }[incident_type],

            "post_mortem_template": {
                "title": f"Post-Mortem: {incident['name']}",
                "date": "YYYY-MM-DD",
                "severity": incident["severity"],
                "impact": f"{affected_requests} requests affected, {downtime_min:.0f} min downtime",
                "root_cause": "TBD - investigate during post-mortem",
                "resolution": "TBD - document steps taken",
                "action_items": "TBD - preventive measures for future",
                "lessons_learned": "TBD - share with team",
            },
        }
        return result

    def simulate_all_incidents(self) -> dict[str, Any]:
        """Simulate all incident types."""
        results = {}
        for incident_type in self.INCIDENT_TYPES:
            results[incident_type] = self.simulate_incident(incident_type)

        return {
            "incidents": results,
            "annual_incident_budget": {
                "target_availability_pct": 99.9,
                "allowed_downtime_min_per_year": 525.6,  # 0.1% of 365*24*60
                "current_downtime_min_per_year": sum(r["mttr_min"] * 4 for r in results.values()),  # 4 incidents/year per type
                "budget_status": "healthy" if sum(r["mttr_min"] * 4 for r in results.values()) < 525.6 else "exhausted",
            },
        }


def run_all():
    """Run all simulators and print results."""
    print("=" * 80)
    print("AI SERVING OBSERVABILITY SIMULATOR")
    print("=" * 80)

    # 1. Metrics simulation
    print("\n1. vLLM Metrics Simulation (7B INT4, RTX 4090)")
    metrics_sim = MetricsSimulator(model_size="7b_int4")
    metrics = metrics_sim.simulate_metrics(n_requests=100, batch_size=32)
    print(f"  TTFT P50={metrics['ttft_p50_ms']:.1f}ms, P99={metrics['ttft_p99_ms']:.1f}ms")
    print(f"  ITL P50={metrics['itl_p50_ms']:.2f}ms, P99={metrics['itl_p99_ms']:.2f}ms")
    print(f"  Availability={metrics['availability_pct']:.1f}%")
    print(f"  KV cache usage={metrics['gpu_cache_usage_pct']:.1f}%")
    print(f"  Prefix cache hit rate={metrics['prefix_cache_hit_rate_pct']:.1f}%")
    print(f"  Preempted={metrics['n_preempted']} requests")

    # 2. SLO compliance
    print("\n2. SLO Compliance Check")
    slo_sim = SLOSimulator()
    slo_result = slo_sim.check_slo_compliance(metrics)
    for name, check in slo_result["slo_checks"].items():
        status = "PASS" if check["compliant"] else "FAIL"
        print(f"  {name}: target={check['target']}, actual={check['actual']:.1f} → {status}")
    print(f"  Error budget: {slo_result['error_budget_pct']:.2f}%")
    print(f"  Recommendation: {slo_result['recommendation']}")

    # 3. Error budget burn
    print("\n3. Error Budget Burn Rate (30 days)")
    budget = slo_sim.simulate_error_budget_burn(30)
    print(f"  Total budget: {budget['total_error_budget_pct']:.3f}% ({budget['total_error_budget_min']:.0f} min)")
    print(f"  Avg daily downtime: {budget['avg_daily_downtime_min']:.1f} min")
    print(f"  Budget remaining at 30d: {budget['budget_remaining_at_30d_pct']:.1f}%")
    print(f"  Recommendation: {budget['recommendation']}")

    # 4. Autoscaling
    print("\n4. Autoscaling: HPA vs Sleep/Wake")
    auto_sim = AutoscalingSimulator()
    hpa = auto_sim.simulate_hpa()
    sleep_wake = auto_sim.simulate_sleep_wake()
    print(f"  HPA: avg_pods={hpa['avg_pods']:.1f}, cold_starts={hpa['total_cold_starts']}, overhead={hpa['cold_start_overhead_pct']:.2f}%")
    print(f"  Sleep/Wake: active_hours={sleep_wake['active_hours']}, wake_time={sleep_wake['total_wake_time_s']}s")
    print(f"  Speedup: sleep/wake {sleep_wake['vs_hpa_comparison']['speedup']}x faster than HPA cold start")

    # 5. Incident response
    print("\n5. Incident Response Runbooks")
    incident_sim = IncidentResponseSimulator()
    incidents = incident_sim.simulate_all_incidents()
    for name, incident in incidents["incidents"].items():
        print(f"  {incident['incident_name']} ({incident['severity']}): MTTR≈{incident['mttr_min']:.0f}min, affected={incident['affected_requests']} reqs")
    budget_status = incidents["annual_incident_budget"]["budget_status"]
    print(f"  Annual incident budget: {budget_status}")

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("class_name", nargs="?", default="all",
                        choices=["MetricsSimulator", "SLOSimulator",
                                 "AutoscalingSimulator", "IncidentResponseSimulator", "all"])
    args = parser.parse_args()

    if args.class_name == "all":
        run_all()
    elif args.class_name == "MetricsSimulator":
        sim = MetricsSimulator()
        print(sim.simulate_metrics())
    elif args.class_name == "SLOSimulator":
        sim = SLOSimulator()
        metrics = MetricsSimulator().simulate_metrics()
        print(sim.check_slo_compliance(metrics))
    elif args.class_name == "AutoscalingSimulator":
        sim = AutoscalingSimulator()
        print({"hpa": sim.simulate_hpa(), "sleep_wake": sim.simulate_sleep_wake()})
    elif args.class_name == "IncidentResponseSimulator":
        sim = IncidentResponseSimulator()
        print(sim.simulate_all_incidents())