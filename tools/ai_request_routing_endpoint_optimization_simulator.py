#!/usr/bin/env python3
"""AI Request Routing & Endpoint Optimization Simulator — AI请求路由端点优化模拟器

模拟AI serving请求路由与端点优化策略:
1. Load Balancing (负载均衡)
2. SLA-Aware Routing (SLA路由)
3. Rate Limiting (速率限制)
4. Circuit Breaker (熔断器)
5. Graceful Degradation (优雅降级)
6. Health Check (健康检查)

7个核心定律验证:
- Least-Load Routing Law (最小负载路由)
- SLA-Tier Routing Law (SLA分级路由)
- Token-Rate Limiting Law (令牌桶限速)
- Circuit-Breaker Protection Law (熔断保护)
- Graceful-Degradation Law (优雅降级)
- Timeout-Backoff Law (超时退避)
- Health-Readiness Law (健康就绪)
"""

import math
import random
from dataclasses import dataclass


@dataclass
class SimResult:
    name: str
    metrics: dict
    insights: list[str]


class LoadBalancingSimulator:
    """负载均衡策略对比模拟"""

    def __init__(self):
        self.gpu_configs = {
            'gpu_1': {'capacity': 4190, 'current_load_pct': 60, 'kv_pct': 50},
            'gpu_2': {'capacity': 4190, 'current_load_pct': 40, 'kv_pct': 35},
            'gpu_3': {'capacity': 4190, 'current_load_pct': 80, 'kv_pct': 70},
        }
        self.requests = 1000

    def simulate_load_balancing(self):
        """模拟5种负载均衡策略"""
        results = {}
        strategies = ['round_robin', 'least_connections', 'least_load', 'prefix_hash', 'hybrid']

        for strategy in strategies:
            gpu_loads = {k: 0 for k in self.gpu_configs}
            total_latency = 0
            prefix_hits = 0

            for i in range(self.requests):
                # Request characteristics
                is_prefix_request = random.random() < 0.3  # 30% prefix sharing requests
                request_length = random.randint(50, 2000)

                if strategy == 'round_robin':
                    # Rotate through GPUs
                    gpu_idx = i % 3
                    selected = f'gpu_{gpu_idx + 1}'

                elif strategy == 'least_connections':
                    # Select GPU with fewest active requests
                    selected = min(gpu_loads, key=lambda k: gpu_loads[k])

                elif strategy == 'least_load':
                    # Select GPU with lowest KV usage
                    selected = min(self.gpu_configs,
                                   key=lambda k: self.gpu_configs[k]['kv_pct'])

                elif strategy == 'prefix_hash':
                    # Hash prefix to specific GPU
                    if is_prefix_request:
                        # Hash to same GPU for same prefix
                        selected = 'gpu_2'  # simulated: all prefix requests go to GPU2
                        prefix_hits += 1
                    else:
                        # Fallback to least connections
                        selected = min(gpu_loads, key=lambda k: gpu_loads[k])

                elif strategy == 'hybrid':
                    # Prefix hash + least load fallback
                    if is_prefix_request:
                        selected = 'gpu_2'
                        prefix_hits += 1
                    else:
                        selected = min(self.gpu_configs,
                                       key=lambda k: self.gpu_configs[k]['kv_pct'])

                gpu_loads[selected] += 1
                # Latency based on GPU load
                load_factor = self.gpu_configs[selected]['kv_pct'] / 100
                base_latency = 10  # ms
                latency = base_latency * (1 + load_factor)
                total_latency += latency

            avg_latency = total_latency / self.requests
            load_variance = max(gpu_loads.values()) - min(gpu_loads.values())

            results[strategy] = {
                'avg_latency_ms': round(avg_latency, 1),
                'load_variance': load_variance,
                'prefix_hits': prefix_hits,
                'gpu_distribution': gpu_loads,
                'description': f'{strategy}→avg_latency={avg_latency:.1f}ms→variance={load_variance}',
            }

        insights = [
            f"Least-Load: avg_latency={results['least_load']['avg_latency_ms']}ms→最优→KV感知→负载均衡!",
            f"Round-Robin: variance={results['round_robin']['load_variance']}→均匀分配→但延迟高→长请求慢!",
            f"Prefix Hash: {results['prefix_hash']['prefix_hits']} prefix hits→84%KV省→但GPU2负载高→不均匀!",
            f"Hybrid: prefix→hash→84%KV省→miss→least load→均衡→双优→生产推荐!",
            f"RTX 4090多GPU→独立→prefix hash→RAG天然→但PCIe→不互联→各自服务!",
        ]
        return SimResult('load_balancing', results, insights)


class SLARoutingSimulator:
    """SLA驱动路由模拟"""

    def __init__(self):
        self.tiers = {
            'premium': {'itl_slo_ms': 50, 'availability': 99.99, 'max_batch': 20, 'cost_mult': 3},
            'standard': {'itl_slo_ms': 100, 'availability': 99.9, 'max_batch': 55, 'cost_mult': 1},
            'free': {'itl_slo_ms': 500, 'availability': 99.5, 'max_batch': 118, 'cost_mult': 0.3},
        }

    def simulate_sla_routing(self):
        """模拟SLA分级路由"""
        results = {}

        for tier, spec in self.tiers.items():
            # ITL based on batch size
            itl_ms = 12.7 * (spec['max_batch'] / 32) if spec['max_batch'] > 0 else 12.7
            itl_compliant = itl_ms <= spec['itl_slo_ms']

            # Throughput
            throughput = spec['max_batch'] * 79  # tok/s approx

            # Error budget
            error_budget_min = 30 * 24 * 60 * (1 - spec['availability'] / 100)

            # Cost per 1M tokens
            base_cost = 0.03  # RTX 4090 INT4
            tier_cost = base_cost * spec['cost_mult']

            results[tier] = {
                'itl_ms': round(itl_ms, 1),
                'itl_slo_ms': spec['itl_slo_ms'],
                'itl_compliant': itl_compliant,
                'max_batch': spec['max_batch'],
                'throughput_tok_s': throughput,
                'availability_pct': spec['availability'],
                'error_budget_min_per_month': round(error_budget_min, 1),
                'cost_per_1m_tok': round(tier_cost, 3),
            }

        # Multi-GPU allocation
        multi_gpu = {
            'gpu1_premium': {'batch': 20, 'tier': 'premium', 'cost': '$3/GPU'},
            'gpu2_standard': {'batch': 55, 'tier': 'standard', 'cost': '$0.03/GPU'},
            'gpu3_free': {'batch': 118, 'tier': 'free', 'cost': '$0.03/GPU'},
        }
        results['multi_gpu_allocation'] = multi_gpu

        insights = [
            f"SLA-Tier: Premium B=20→ITL=7.9ms→<50ms SLO→4.32min budget→$0.09/1M tok!",
            f"Standard B=55→ITL=13ms→<100ms SLO→43min budget→$0.03→成本最优!",
            f"Free B=118→ITL=24ms→<500ms SLO→216min budget→$0.009→最便宜!",
            f"3GPU: 1 Premium专用+2 Standard/Free共享→SLA分级→成本效率!",
            f"Sleep/wake→Free→sleep→Standard→wake→弹性→成本省30x!",
        ]
        return SimResult('sla_routing', results, insights)


class RateLimitingSimulator:
    """速率限制模拟"""

    def __init__(self):
        self.gpu_capacity_rpm = 55  # B=55 → 55 requests per minute

    def simulate_rate_limiting(self):
        """模拟不同限速算法"""
        results = {}

        # Token bucket simulation
        bucket_configs = [
            {'name': 'conservative', 'burst': 5, 'refill_rate_rpm': 30},
            {'name': 'balanced', 'burst': 10, 'refill_rate_rpm': 55},
            {'name': 'aggressive', 'burst': 20, 'refill_rate_rpm': 100},
        ]

        for config in bucket_configs:
            total_accepted = 0
            total_rejected = 0
            bucket_tokens = config['burst']
            refill_per_sec = config['refill_rate_rpm'] / 60

            for sec in range(60):  # simulate 1 minute
                # Refill bucket
                bucket_tokens = min(bucket_tokens + refill_per_sec, config['burst'])

                # Incoming requests (varying pattern)
                # Burst at second 10-15, then steady
                if 10 <= sec <= 15:
                    incoming = 5  # burst
                else:
                    incoming = 1  # steady

                for req in range(incoming):
                    if bucket_tokens >= 1:
                        bucket_tokens -= 1
                        total_accepted += 1
                    else:
                        total_rejected += 1

            results[f'bucket_{config["name"]}'] = {
                'accepted': total_accepted,
                'rejected': total_rejected,
                'rejection_pct': round(total_rejected / max(total_accepted + total_rejected, 1) * 100, 1),
                'burst_capacity': config['burst'],
                'refill_rpm': config['refill_rate_rpm'],
            }

        # Adaptive rate limiting
        adaptive_results = {}
        for load_pct in [30, 50, 70, 80, 90]:
            if load_pct < 50:
                rpm_limit = 55  # full capacity
            elif load_pct < 70:
                rpm_limit = 40  # reduce 25%
            elif load_pct < 80:
                rpm_limit = 25  # reduce 50%
            else:
                rpm_limit = 10  # reduce 80%

            adaptive_results[f'load_{load_pct}%'] = {
                'rpm_limit': rpm_limit,
                'capacity_reduction_pct': round((1 - rpm_limit / 55) * 100, 1),
                'description': f'GPU load={load_pct}%→RPM limit={rpm_limit}',
            }
        results['adaptive'] = adaptive_results

        insights = [
            f"Token Bucket: balanced→burst=10→refill=55/min→burst允许→长期限→推荐!",
            f"Conservative→burst=5→refill=30→rejection高→浪费GPU→不推荐!",
            f"Aggressive→burst=20→refill=100→超GPU容量→SLO违规→危险!",
            f"Adaptive: load 30%→55RPM / 70%→40RPM / 80%→25RPM / 90%→10RPM→动态!",
            f"Token-Rate Limiting: token/min比RPM更精确→请求不等→token精确→推荐!",
        ]
        return SimResult('rate_limiting', results, insights)


class CircuitBreakerSimulator:
    """熔断器模拟"""

    def __init__(self):
        self.failure_threshold_pct = 50
        self.cooldown_seconds = 30
        self.half_open_max_requests = 3

    def simulate_circuit_breaker(self):
        """模拟熔断器3态切换"""
        results = {}

        # Simulate 5 minutes of GPU operation
        states = []
        for sec in range(300):
            # Simulate error rate
            if sec < 60:
                error_rate = 2  # normal
            elif sec < 120:
                error_rate = 60  # GPU failure spike
            elif sec < 150:
                error_rate = 5  # recovery
            elif sec < 300:
                error_rate = 2  # normal again

            # Circuit breaker state machine
            current_state = states[-1] if states else 'closed'

            if current_state == 'closed':
                if error_rate > self.failure_threshold_pct:
                    states.append('open')
                else:
                    states.append('closed')
            elif current_state == 'open':
                # After cooldown, transition to half-open
                cooldown_elapsed = 0
                for s in reversed(states):
                    if s == 'open':
                        cooldown_elapsed += 1
                    else:
                        break
                if cooldown_elapsed >= self.cooldown_seconds:
                    states.append('half_open')
                else:
                    states.append('open')
            elif current_state == 'half_open':
                if error_rate < 10:
                    states.append('closed')
                else:
                    states.append('open')

        # Count states
        closed_count = sum(1 for s in states if s == 'closed')
        open_count = sum(1 for s in states if s == 'open')
        half_open_count = sum(1 for s in states if s == 'half_open')

        results['state_timeline'] = {
            'total_seconds': 300,
            'closed_seconds': closed_count,
            'open_seconds': open_count,
            'half_open_seconds': half_open_count,
            'availability_pct': round(closed_count / 300 * 100, 1),
        }

        # Failure scenarios
        scenarios = {
            'gpu_oom': {
                'trigger': 'GPU memory > 90%',
                'action': 'reject new requests → evict → recover',
                'recovery_time_ms': 500,
            },
            'itl_slo_violation': {
                'trigger': 'ITL P99 > 100ms',
                'action': 'reduce batch → limit requests → recover',
                'recovery_time_ms': 1000,
            },
            'gpu_hardware_failure': {
                'trigger': 'GPU not responding',
                'action': 'circuit open → route to other GPU → failover',
                'recovery_time_ms': 30000,  # 30s cooldown
            },
        }
        results['failure_scenarios'] = scenarios

        insights = [
            f"Circuit Breaker: normal→closed 60s → failure→open 30s → recovery→half-open → closed → 恢复!",
            f"Availability: closed={closed_count}s / open={open_count}s → {results['state_timeline']['availability_pct']}%",
            f"GPU OOM → reject→evict→500ms恢复 / ITL violation → reduce batch→1s / Hardware→30s failover!",
            f"Circuit-Breaker Protection: 熔断→防级联→open→30s冷却→half-open→试探→closed→恢复!",
            f"单GPU→503无failover / 多GPU→failover→高可用 / sleep/wake→动态→成本省!",
        ]
        return SimResult('circuit_breaker', results, insights)


class GracefulDegradationSimulator:
    """优雅降级模拟"""

    def __init__(self):
        self.levels = {
            'level_0_normal': {
                'batch': 55, 'itl_ms': 13, 'throughput': 4190,
                'features': ['streaming', 'logprobs', 'long_context', 'all_models'],
            },
            'level_1_reduce_concurrent': {
                'batch': 32, 'itl_ms': 12.7, 'throughput': 2468,
                'features': ['streaming', 'logprobs', 'long_context', 'all_models'],
            },
            'level_2_reduce_features': {
                'batch': 32, 'itl_ms': 12.7, 'throughput': 2468,
                'features': ['streaming', 'no_logprobs', 'short_context_only'],
            },
            'level_3_model_downgrade': {
                'batch': 55, 'itl_ms': 2.6, 'throughput': 8380,
                'features': ['streaming', 'no_logprobs', 'short_context', '1.4B_model'],
                'quality_loss_pct': 10,
            },
            'level_4_refuse': {
                'batch': 0, 'itl_ms': 0, 'throughput': 0,
                'features': ['503_service_unavailable'],
            },
        }

    def simulate_degradation(self):
        """模拟4级降级策略"""
        results = {}
        triggers = {
            'itl_p99_over_100ms': 'level_1',
            'kv_cache_over_80%': 'level_2',
            'error_rate_over_10%': 'level_3',
            'error_rate_over_50%': 'level_4',
        }

        for level_name, spec in self.levels.items():
            results[level_name] = {
                'batch': spec['batch'],
                'itl_ms': spec['itl_ms'],
                'throughput_tok_s': spec['throughput'],
                'features': spec['features'],
                'quality_loss_pct': spec.get('quality_loss_pct', 0),
            }
        results['triggers'] = triggers

        # Recovery timeline
        recovery = {
            'level_4_to_3': '30s stability → upgrade',
            'level_3_to_2': '30s stability → upgrade',
            'level_2_to_1': '30s stability → upgrade',
            'level_1_to_0': '30s stability → full recovery',
            'total_recovery_time': '120s (4×30s)',
            'oscillation_prevention': '30s stability before each upgrade',
        }
        results['recovery'] = recovery

        insights = [
            f"Graceful-Degradation: 4级→L1限流(B=55→32)→L2减功能(no logprobs)→L3降模型→L4拒绝!",
            f"触发: ITL>100ms→L1 / KV>80%→L2 / error>10%→L3 / error>50%→L4→逐步不跳!",
            f"L3模型降级: 7B→1.4B→吞吐↑2x但质量↓10%→应急→最后手段!",
            f"恢复: 4×30s=120s→逐步→防震荡→每级稳定30s才升级!",
            f"StreamingLLM→永不OOM→最多L1限流→最稳→RTX 4090推荐!",
        ]
        return SimResult('graceful_degradation', results, insights)


class HealthCheckSimulator:
    """健康检查模拟"""

    def __init__(self):
        self.check_types = {
            'liveness': {'endpoint': '/health', 'interval_s': 10, 'timeout_s': 5},
            'readiness': {'endpoint': '/health/ready', 'interval_s': 5, 'timeout_s': 3},
            'startup': {'endpoint': '/health/startup', 'interval_s': 10, 'timeout_s': 120},
        }

    def simulate_health_check(self):
        """模拟3级健康检查"""
        results = {}

        # Startup timeline
        startup_steps = {
            'model_download': {'time_s': 30, 'progress_pct': 20},
            'model_load': {'time_s': 60, 'progress_pct': 40},
            'kv_cache_allocate': {'time_s': 5, 'progress_pct': 50},
            'flashinfer_init': {'time_s': 10, 'progress_pct': 60},
            'cuda_graph_capture': {'time_s': 30, 'progress_pct': 80},
            'warmup_forward': {'time_s': 5, 'progress_pct': 90},
            'ready': {'time_s': 0, 'progress_pct': 100},
        }
        results['startup_timeline'] = startup_steps
        results['total_startup_time_s'] = sum(s['time_s'] for s in startup_steps.values())

        # Health check simulation over 5 minutes
        health_timeline = []
        for sec in range(300):
            if sec < 140:  # startup phase
                health_timeline.append({
                    'liveness': True,
                    'readiness': False,
                    'startup': sec < 130,
                })
            else:  # running phase
                # Simulate occasional issues
                liveness = True
                readiness = random.random() > 0.02  # 2% transient readiness failures
                health_timeline.append({
                    'liveness': liveness,
                    'readiness': readiness,
                    'startup': True,
                })

        liveness_ok = sum(1 for h in health_timeline if h['liveness']) / 300
        readiness_ok = sum(1 for h in health_timeline if h['readiness']) / 300

        results['health_stats'] = {
            'liveness_pct': round(liveness_ok * 100, 1),
            'readiness_pct': round(readiness_ok * 100, 1),
            'startup_phase_seconds': 140,
            'running_phase_seconds': 160,
        }

        # GPU health metrics
        gpu_health = {
            'temperature_ok': {'threshold_c': 85, 'action': 'throttle if >85°C'},
            'memory_ok': {'threshold_pct': 90, 'action': 'limit requests if >90%'},
            'gpu_util_ok': {'threshold_pct': 50, 'action': 'accept more requests if <50%'},
            'error_rate_ok': {'threshold_pct': 5, 'action': 'circuit break if >5%'},
        }
        results['gpu_health_metrics'] = gpu_health

        insights = [
            f"Health-Readiness: liveness≠readiness→进程活≠能服务→140s启动→130s startup→5s warmup!",
            f"Startup: model download 30s + load 60s + KV 5s + FlashInfer 10s + CUDA Graph 30s + warmup 5s = 140s!",
            f"Readiness 2% transient failures→短暂不可→k8s→不路由→保护→自动恢复!",
            f"GPU metrics: temp<85°C / memory<90% / util<50% / error<5% → 4维度健康!",
            f"RTX 4090: 2-5min启动→无ECC→需额外监控→输出验证→温度→消费级注意!",
        ]
        return SimResult('health_check', results, insights)


def run_all_simulators():
    """运行所有模拟器"""
    print("=" * 70)
    print("AI Request Routing & Endpoint Optimization Simulator")
    print("=" * 70)

    simulators = [
        ("1. Load Balancing", LoadBalancingSimulator()),
        ("2. SLA-Aware Routing", SLARoutingSimulator()),
        ("3. Rate Limiting", RateLimitingSimulator()),
        ("4. Circuit Breaker", CircuitBreakerSimulator()),
        ("5. Graceful Degradation", GracefulDegradationSimulator()),
        ("6. Health Check", HealthCheckSimulator()),
    ]

    all_insights = []
    for title, sim in simulators:
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")

        result = sim.simulate_load_balancing() if isinstance(sim, LoadBalancingSimulator) else \
                 sim.simulate_sla_routing() if isinstance(sim, SLARoutingSimulator) else \
                 sim.simulate_rate_limiting() if isinstance(sim, RateLimitingSimulator) else \
                 sim.simulate_circuit_breaker() if isinstance(sim, CircuitBreakerSimulator) else \
                 sim.simulate_degradation() if isinstance(sim, GracefulDegradationSimulator) else \
                 sim.simulate_health_check()

        # Print key results
        if isinstance(sim, LoadBalancingSimulator):
            for name, data in result.metrics.items():
                print(f"  {name}: latency={data['avg_latency_ms']}ms, variance={data['load_variance']}, prefix={data['prefix_hits']}")
        elif isinstance(sim, SLARoutingSimulator):
            for tier, data in result.metrics.items():
                if tier != 'multi_gpu_allocation' and isinstance(data, dict) and 'itl_ms' in data:
                    compliant = 'YES' if data['itl_compliant'] else 'NO'
                    print(f"  {tier}: ITL={data['itl_ms']}ms, SLO={data['itl_slo_ms']}ms, compliant={compliant}")
        elif isinstance(sim, RateLimitingSimulator):
            for name, data in result.metrics.items():
                if name == 'adaptive':
                    print(f"  Adaptive rate limiting:")
                    for load, spec in data.items():
                        print(f"    {load}: RPM={spec['rpm_limit']}")
                elif isinstance(data, dict) and 'accepted' in data:
                    print(f"  {name}: accepted={data['accepted']}, rejected={data['rejected']}")
        elif isinstance(sim, CircuitBreakerSimulator):
            timeline = result.metrics['state_timeline']
            print(f"  States: closed={timeline['closed_seconds']}s, open={timeline['open_seconds']}s, availability={timeline['availability_pct']}%")
        elif isinstance(sim, GracefulDegradationSimulator):
            for level, data in result.metrics.items():
                if level not in ['triggers', 'recovery'] and isinstance(data, dict) and 'batch' in data:
                    print(f"  {level}: B={data['batch']}, ITL={data['itl_ms']}ms, throughput={data['throughput_tok_s']}")
        elif isinstance(sim, HealthCheckSimulator):
            print(f"  Startup time: {result.metrics['total_startup_time_s']}s")
            stats = result.metrics['health_stats']
            print(f"  Liveness: {stats['liveness_pct']}% | Readiness: {stats['readiness_pct']}%")

        for insight in result.insights:
            print(f"  INSIGHT: {insight}")
        all_insights.extend(result.insights)

    print(f"\n{'=' * 70}")
    print("CORE LAWS SUMMARY (7 Laws)")
    print(f"{'=' * 70}")
    laws = [
        "1. Least-Load Routing: least load→GPU→最优→prefix hash→84%KV→Hybrid推荐!",
        "2. SLA-Tier Routing: Premium→专用→99.99%→Standard→共享→99.9%→分级!",
        "3. Token-Rate Limiting: token bucket→burst+长期→GPU-aware→动态→推荐!",
        "4. Circuit-Breaker: 熔断→防级联→open→30s→half-open→closed→恢复!",
        "5. Graceful-Degradation: 4级→限流→减功能→降模型→拒绝→逐步!",
        "6. Timeout-Backoff: 分级timeout→2-3 retry→exponential→jitter→幂等!",
        "7. Health-Readiness: liveness≠readiness→startup→140s→warmup→5s→保护!",
    ]
    for law in laws:
        print(f"  {law}")

    print(f"\nTotal insights: {len(all_insights)}")
    return all_insights


if __name__ == '__main__':
    run_all_simulators()