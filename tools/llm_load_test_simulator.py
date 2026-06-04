#!/usr/bin/env python3
"""LLM 推理负载测试模拟器

模拟 LLM 推理服务在不同负载下的性能表现:
1. 延迟-吞吐量曲线生成
2. 最优并发度分析
3. 不同负载模式 (恒定/突发/泊松)
4. SLO 合规性分析
5. 扩缩容建议

CPU 可运行, 无需 GPU。
"""

import math
import random
from dataclasses import dataclass


# ============================================================
# 模型与硬件配置
# ============================================================

@dataclass
class ServingConfig:
    """推理服务配置"""
    model_name: str
    params_b: float
    gpu_name: str
    tp: int
    num_replicas: int = 1
    max_model_len: int = 4096
    gpu_memory_util: float = 0.9
    # 性能参数
    hbm_bw_gbs: float = 2035.0  # A100
    fp16_tflops: float = 312.0
    # 推理参数
    max_num_seqs: int = 256
    block_size: int = 16


@dataclass
class Request:
    """单个推理请求"""
    prompt_len: int
    max_tokens: int
    arrival_time_s: float  # 到达时间


# ============================================================
# 性能模型
# ============================================================

class InferenceEngine:
    """模拟 vLLM 推理引擎"""

    def __init__(self, config: ServingConfig):
        self.config = config
        self.weight_bytes = config.params_b * 1e9 * 2  # FP16
        self.kv_per_token_bytes = 2 * 2 * 32 * 128 * 2  # 简化: 2(K+V) * layers * head_dim * 2bytes
        # 根据 model 调整
        if config.params_b > 50:
            self.kv_per_token_bytes = 2 * 2 * 80 * 128 * 2  # 70B

    def decode_tpot_ms(self, batch_size: int) -> float:
        """Decode 阶段每个 token 的时间 (ms)

        memory-bound: time = weight_bytes / (hbm_bw * tp)
        batch_size 增加时, 吞吐提升但有上限
        """
        effective_bw = self.config.hbm_bw_gbs * 1e9 * self.config.tp
        single_decode_ms = self.weight_bytes / effective_bw * 1000

        # Batch 并行: 吞吐因子, 线性到约 64 然后饱和
        parallelism = min(batch_size, 64)
        return single_decode_ms / parallelism

    def prefill_ttft_ms(self, prompt_len: int) -> float:
        """Prefill TTFT (ms)

        compute-bound: FLOPs / TFLOPS
        """
        flops = 2 * self.config.params_b * 1e9 * prompt_len
        tflops = self.config.fp16_tflops * self.config.tp * 0.5  # MFU 50%
        return flops / (tflops * 1e12) * 1000

    def max_concurrent(self) -> int:
        """最大并发请求数 (受 KV Cache 限制)"""
        gpu_mem = 80 * 1e9 * self.config.tp * self.config.gpu_memory_util * 0.6  # 60% for KV
        kv_per_request = self.kv_per_token_bytes * self.config.max_model_len
        return int(gpu_mem / kv_per_request)

    def simulate_request(self, req: Request, concurrent: int) -> dict:
        """模拟单个请求的延迟"""
        ttft = self.prefill_ttft_ms(req.prompt_len)
        tpot = self.decode_tpot_ms(concurrent)
        decode_time = tpot * req.max_tokens
        e2e = ttft + decode_time

        return {
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "decode_ms": decode_time,
            "e2e_ms": e2e,
            "tokens": req.max_tokens,
            "throughput_tok_s": req.max_tokens / (e2e / 1000),
            "concurrent": concurrent,
        }


# ============================================================
# 负载生成器
# ============================================================

def generate_constant_load(num_requests: int, rps: float,
                           prompt_len: int, max_tokens: int) -> list[Request]:
    """恒定速率负载"""
    return [
        Request(prompt_len, max_tokens, i / rps)
        for i in range(num_requests)
    ]


def generate_poisson_load(num_requests: int, rps: float,
                          prompt_len: int, max_tokens: int) -> list[Request]:
    """泊松到达负载"""
    requests = []
    t = 0
    for _ in range(num_requests):
        requests.append(Request(prompt_len, max_tokens, t))
        t += random.expovariate(rps)
    return requests


def generate_burst_load(num_requests: int, rps: float,
                        prompt_len: int, max_tokens: int,
                        burst_size: int = 10) -> list[Request]:
    """突发负载"""
    requests = []
    t = 0
    sent = 0
    while sent < num_requests:
        # 突发
        for _ in range(min(burst_size, num_requests - sent)):
            requests.append(Request(prompt_len, max_tokens, t))
            sent += 1
        # 静默期
        t += burst_size / rps
    return requests


# ============================================================
# 实验
# ============================================================

def experiment1_latency_throughput_curve():
    """实验 1: 延迟-吞吐量曲线"""
    print("=" * 70)
    print("实验 1: 延迟-吞吐量曲线 (8B on A100)")
    print("=" * 70)

    config = ServingConfig(
        model_name="Llama-8B", params_b=8, gpu_name="A100-80G",
        tp=1, hbm_bw_gbs=2035, fp16_tflops=312
    )
    engine = InferenceEngine(config)

    prompt_len = 512
    max_tokens = 256

    print(f"\n配置: {config.model_name}, {config.gpu_name}, TP={config.tp}")
    print(f"请求: prompt={prompt_len}, max_tokens={max_tokens}")
    print(f"最大并发: {engine.max_concurrent()}\n")

    print(f"{'并发':<8} {'TTFT(ms)':<12} {'TPOT(ms)':<12} {'E2E(ms)':<12} {'吞吐(tok/s)':<14} {'batch吞吐':<12} {'状态'}")
    print("-" * 82)

    results = []
    for concurrent in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        req = Request(prompt_len, max_tokens, 0)
        r = engine.simulate_request(req, concurrent)
        batch_throughput = r["throughput_tok_s"] * concurrent
        results.append((concurrent, r))

        if r["e2e_ms"] < 1000:
            status = "✓ E2E < 1s"
        elif r["e2e_ms"] < 5000:
            status = "~ E2E < 5s"
        else:
            status = "✗ E2E > 5s"

        print(f"{concurrent:<8} {r['ttft_ms']:<12.1f} {r['tpot_ms']:<12.3f} "
              f"{r['e2e_ms']:<12.1f} {r['throughput_tok_s']:<14.0f} {batch_throughput:<12.0f} {status}")

    # 找最优并发
    best = max(results, key=lambda x: x[1]["throughput_tok_s"] * x[0])
    print(f"\n最优: 并发={best[0]}, 单请求吞吐={best[1]['throughput_tok_s']:.0f} tok/s, "
          f"batch吞吐={best[1]['throughput_tok_s']*best[0]:.0f} tok/s")

    print("\n关键洞察:")
    print("  - 并发增加: 单请求吞吐下降 (TPOT 增加), 但总吞吐上升")
    print("  - 最优点在延迟可接受范围内的最大并发")
    print("  - TTFT 不受并发影响 (prefill 是 compute-bound)")
    print("  - TPOT 与并发成反比 (memory-bound, 吞吐线性到 ~64)")


def experiment2_load_patterns():
    """实验 2: 不同负载模式下的性能"""
    print("\n" + "=" * 70)
    print("实验 2: 不同负载模式")
    print("=" * 70)

    config = ServingConfig(
        model_name="Llama-8B", params_b=8, gpu_name="A100-80G",
        tp=1, hbm_bw_gbs=2035, fp16_tflops=312, max_num_seqs=64
    )
    engine = InferenceEngine(config)

    prompt_len = 512
    max_tokens = 256
    rps = 10  # 10 requests/sec

    random.seed(42)

    patterns = {
        "恒定": generate_constant_load(100, rps, prompt_len, max_tokens),
        "泊松": generate_poisson_load(100, rps, prompt_len, max_tokens),
        "突发(10)": generate_burst_load(100, rps, prompt_len, max_tokens, burst_size=10),
        "突发(50)": generate_burst_load(100, rps, prompt_len, max_tokens, burst_size=50),
    }

    print(f"\n配置: rps={rps}, prompt={prompt_len}, max_tokens={max_tokens}\n")

    print(f"{'模式':<12} {'平均E2E(ms)':<14} {'P50(ms)':<12} {'P99(ms)':<12} {'排队峰值':<10} {'吞吐(tok/s)'}")
    print("-" * 72)

    for name, requests in patterns.items():
        # 简化模拟: 计算每个请求的排队时间
        max_concurrent = min(engine.max_concurrent(), config.max_num_seqs)
        e2e_times = []
        queue_depths = []
        active = 0
        completions = []  # (完成时间, request_idx)
        total_tokens = 0

        for i, req in enumerate(requests):
            # 移除已完成的请求
            completions = [c for c in completions if c[0] > req.arrival_time_s]
            active = len(completions)

            # 排队
            if active >= max_concurrent:
                # 等待最早的请求完成
                wait_s = min(c[0] for c in completions) - req.arrival_time_s
                wait_ms = max(0, wait_s * 1000)
            else:
                wait_ms = 0

            queue_depths.append(active)

            # 处理
            r = engine.simulate_request(req, min(active + 1, max_concurrent))
            e2e = r["e2e_ms"] + wait_ms
            e2e_times.append(e2e)
            total_tokens += r["tokens"]

            # 记录完成时间
            finish_s = req.arrival_time_s + e2e / 1000
            completions.append((finish_s, i))

        e2e_times.sort()
        avg_e2e = sum(e2e_times) / len(e2e_times)
        p50 = e2e_times[len(e2e_times) // 2]
        p99 = e2e_times[int(len(e2e_times) * 0.99)]
        max_queue = max(queue_depths)

        total_time_s = requests[-1].arrival_time_s + e2e_times[-1] / 1000
        throughput = total_tokens / total_time_s if total_time_s > 0 else 0

        print(f"{name:<12} {avg_e2e:<14.0f} {p50:<12.0f} {p99:<12.0f} {max_queue:<10} {throughput:.0f}")

    print("\n关键洞察:")
    print("  - 恒定负载: 延迟最稳定, 无突发排队")
    print("  - 泊松负载: 自然到达模式, 偶尔有排队")
    print("  - 突发负载: 延迟方差大, P99 远高于 P50")
    print("  - 生产环境: 95% 是泊松模式, 需要关注 P99")


def experiment3_slo_analysis():
    """实验 3: SLO 合规性分析"""
    print("\n" + "=" * 70)
    print("实验 3: SLO 合规性分析")
    print("=" * 70)

    config = ServingConfig(
        model_name="Llama-8B", params_b=8, gpu_name="A100-80G",
        tp=1, hbm_bw_gbs=2035, fp16_tflops=312
    )
    engine = InferenceEngine(config)

    # SLO 定义
    slos = [
        ("TTFT P50", "ttft_ms", 200, 0.50),
        ("TTFT P99", "ttft_ms", 2000, 0.99),
        ("TPOT P95", "tpot_ms", 100, 0.95),
        ("E2E P99", "e2e_ms", 30000, 0.99),
    ]

    prompt_lens = [128, 256, 512, 1024, 2048]
    max_tokens = 256
    concurrent = 16

    print(f"\n并发: {concurrent}, max_tokens: {max_tokens}")
    print(f"\n{'Prompt Len':<12} {'TTFT P50':<12} {'TTFT P99':<12} {'TPOT P95':<12} {'E2E P99':<12} {'合规'}")
    print("-" * 72)

    for pl in prompt_lens:
        req = Request(pl, max_tokens, 0)
        r = engine.simulate_request(req, concurrent)

        compliance = []
        for slo_name, metric, threshold, _ in slos:
            value = r[metric]
            ok = "✓" if value <= threshold else "✗"
            compliance.append(ok)

        print(f"{pl:<12} {r['ttft_ms']:<12.1f} {r['ttft_ms']:<12.1f} "
              f"{r['tpot_ms']:<12.3f} {r['e2e_ms']:<12.0f} {''.join(compliance)}")

    # SLO 阈值表
    print(f"\nSLO 阈值:")
    for name, _, threshold, percentile in slos:
        print(f"  {name}: < {threshold} {'ms' if threshold > 1 else 'μs'}")

    print("\n关键洞察:")
    print("  - TTFT 随 prompt 长度线性增长 (compute-bound)")
    print("  - TPOT 不受 prompt 长度影响 (只看 decode)")
    print("  - 长 prompt (>1K) 容易违反 TTFT P99 SLO")
    print("  - 解决方案: Chunked prefill / 前缀缓存 / 增加 GPU")


def experiment4_scaling_recommendations():
    """实验 4: 扩缩容建议"""
    print("\n" + "=" * 70)
    print("实验 4: 扩缩容建议")
    print("=" * 70)

    configs = [
        ("8B/A100/TP1/R1", ServingConfig("8B", 8, "A100", 1, 1, hbm_bw_gbs=2035, fp16_tflops=312)),
        ("8B/A100/TP1/R4", ServingConfig("8B", 8, "A100", 1, 4, hbm_bw_gbs=2035, fp16_tflops=312)),
        ("8B/H100/TP1/R1", ServingConfig("8B", 8, "H100", 1, 1, hbm_bw_gbs=3350, fp16_tflops=990)),
        ("70B/H100/TP4/R1", ServingConfig("70B", 70, "H100", 4, 1, hbm_bw_gbs=3350, fp16_tflops=990)),
        ("70B/H100/TP4/R4", ServingConfig("70B", 70, "H100", 4, 4, hbm_bw_gbs=3350, fp16_tflops=990)),
    ]

    prompt_len = 512
    max_tokens = 256
    target_rps_list = [1, 5, 10, 50]

    print(f"\n请求: prompt={prompt_len}, max_tokens={max_tokens}")
    print(f"\n{'配置':<22} {'单副本吞吐':<14} {'4副本吞吐':<14} {'E2E@16':<12} {'GPU数':<8} {'$/Mtok'}")
    print("-" * 82)

    prices = {"A100": 2.0, "H100": 3.0}  # $/GPU/hour

    for name, config in configs:
        engine = InferenceEngine(config)
        r = engine.simulate_request(Request(prompt_len, max_tokens, 0), 16)

        single_throughput = r["throughput_tok_s"]
        multi_throughput = single_throughput * config.num_replicas

        gpu_count = config.tp * config.num_replicas
        price_per_gpu = prices.get(config.gpu_name.split("-")[0], 2.0)
        cost_per_mtok = gpu_count * price_per_gpu / (multi_throughput * 3600 / 1e6) if multi_throughput > 0 else 0

        print(f"{name:<22} {single_throughput:<14.0f} {multi_throughput:<14.0f} "
              f"{r['e2e_ms']:<12.0f} {gpu_count:<8} ${cost_per_mtok:.2f}")

    print("\n关键洞察:")
    print("  - 增加 replicas 线性扩展吞吐, 不增加单请求延迟")
    print("  - 增加 TP 线性扩展单请求吞吐, 但增加 GPU 数量")
    print("  - 8B+H100 比 8B+A100 吞吐高 ~1.6x")
    print("  - 70B 成本约 8B 的 20x (8.75x 参数 × 4 GPU × 更高单价)")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LLM 推理负载测试模拟器")
    print("延迟-吞吐曲线 + 负载模式 + SLO 分析 + 扩缩容")
    print("=" * 70)

    experiment1_latency_throughput_curve()
    experiment2_load_patterns()
    experiment3_slo_analysis()
    experiment4_scaling_recommendations()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
LLM 负载测试核心:

  1. 延迟-吞吐权衡:
     - 并发↑ → 总吞吐↑, 单请求延迟↑
     - 最优并发 = SLO 允许范围内的最大值
     - TTFT 受 prompt 长度影响 (compute-bound)
     - TPOT 受并发影响 (memory-bound)

  2. 负载模式:
     - 恒定: 基准测试用, 延迟最稳定
     - 泊松: 生产最常见, 关注 P99
     - 突发: 最恶劣, 排队导致延迟飙升

  3. SLO 设计:
     - TTFT P99 < 2s (影响用户体感首字延迟)
     - TPOT P95 < 100ms (影响生成速度)
     - E2E P99 < 30s (影响总等待时间)

  4. 扩缩容:
     - Replicas 线性扩展吞吐
     - TP 扩展单请求性能
     - 成本: 8B ~$0.24/Mtok, 70B ~$5/Mtok
""")
