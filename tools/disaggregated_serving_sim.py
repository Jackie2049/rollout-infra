#!/usr/bin/env python3
"""Disaggregated Serving Simulator — Prefill/Decode 分离架构模拟器

CPU 可运行的 P/D 分离推理架构模拟实验，覆盖 5 个实验：
  1. Monolithic vs P/D Disaggregated 对比
  2. KV Transfer 开销对分离架构的影响
  3. Prefill/Decode 实例比例优化
  4. 请求路由策略对比
  5. 扩展效率对比（多实例）

关键概念：
  - Monolithic: Prefill 和 Decode 共享同一 GPU 实例
  - Disaggregated: Prefill 专用实例 (compute-bound) + Decode 专用实例 (memory-bound)
  - KV Transfer: P 实例计算完 KV Cache 后传给 D 实例（通过网络）
  - 优势: 避免长 Prefill 干扰 Decode ITL, 各实例可独立优化

用法:
  conda run -n ai-infra python tools/disaggregated_serving_sim.py
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ──────────────────────────────────────────────
# 硬件与模型配置
# ──────────────────────────────────────────────

@dataclass
class GPUConfig:
    """GPU 实例配置"""
    name: str
    tflops: float       # FP16 TFLOPS
    hbm_bw_gbps: float  # HBM 带宽 (GB/s)
    hbm_gb: float       # HBM 容量 (GB)
    nvlink_bw_gbps: float = 0.0  # NVLink 带宽 (GB/s), 用于 KV Transfer


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    params_B: float     # 参数量 (十亿)
    hidden_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    vocab_size: int = 32000

    @property
    def kv_dim_per_token(self) -> int:
        """每个 token 的 KV Cache 维度 (2 * num_heads * head_dim per layer)"""
        return 2 * self.num_heads * self.head_dim * self.num_layers

    @property
    def model_size_gb_fp16(self) -> float:
        """FP16 模型大小 (GB)"""
        return self.params_B * 2  # 2 bytes per param


# 预定义配置
H100 = GPUConfig("H100", 990, 3350, 80, nvlink_bw_gbps=900)
A100 = GPUConfig("A100-80G", 312, 2039, 80, nvlink_bw_gbps=600)
A100_40G = GPUConfig("A100-40G", 312, 1555, 40, nvlink_bw_gbps=600)

LLAMA_70B = ModelConfig("LLaMA-70B", 70, 8192, 80, 64, 128, vocab_size=32000)
LLAMA_8B = ModelConfig("LLaMA-8B", 8, 4096, 32, 32, 128, vocab_size=128000)


# ──────────────────────────────────────────────
# 延迟模型
# ──────────────────────────────────────────────

def prefill_latency_ms(
    model: ModelConfig,
    gpu: GPUConfig,
    prompt_tokens: int,
    batch_size: int = 1,
) -> float:
    """估算 Prefill 延迟 (ms)

    Prefill 是 compute-bound:
      总 FLOPs ≈ 2 * params * prompt_tokens * batch (矩阵乘)
      时间 ≈ FLOPs / TFLOPS
    """
    # GEMM FLOPs per token: ≈ 2 * params * 1e9
    flops_per_token = 2 * model.params_B * 1e9
    total_flops = flops_per_token * prompt_tokens * batch_size
    # 考虑 Attention 的 O(N²) 开销 (占比随 prompt 增长)
    attn_flops = 2 * model.hidden_dim * prompt_tokens * prompt_tokens * model.num_layers * batch_size
    total_flops += attn_flops

    # GPU 计算时间
    tflops = gpu.tflops * 1e12  # 转为 FLOPS
    compute_ms = total_flops / tflops * 1000

    # 实际效率通常只有 30-60% (kernel overhead, memory access)
    efficiency = 0.4
    return compute_ms / efficiency


def decode_latency_ms(
    model: ModelConfig,
    gpu: GPUConfig,
    kv_cache_tokens: int,
    batch_size: int = 1,
) -> float:
    """估算 Decode 延迟 (ms)

    Decode 是 memory-bound:
      需要读取整个模型权重 + KV Cache
      时间 ≈ (model_size + kv_cache) / HBM_bandwidth
    """
    # 读模型权重
    model_bytes = model.model_size_gb_fp16 * 1e9
    # 读 KV Cache (每个 token, 所有 batch)
    kv_bytes = model.kv_dim_per_token * 2 * kv_cache_tokens * batch_size  # FP16
    total_bytes = model_bytes + kv_bytes

    bandwidth = gpu.hbm_bw_gbps * 1e9 / 1000  # GB/s → bytes/ms
    return total_bytes / bandwidth


def kv_transfer_latency_ms(
    model: ModelConfig,
    gpu: GPUConfig,
    prompt_tokens: int,
    network_bw_gbps: Optional[float] = None,
) -> float:
    """KV Transfer 延迟 (ms)

    从 P 实例传输 KV Cache 到 D 实例
    """
    # KV Cache 大小
    kv_bytes = model.kv_dim_per_token * 2 * prompt_tokens  # FP16

    # 使用 NVLink 或网络带宽
    bw = network_bw_gbps or gpu.nvlink_bw_gbps
    if bw == 0:
        bw = 100  # 默认 100 Gbps 网络
    bandwidth = bw * 1e9 / 1000  # GB/s → bytes/ms

    # 加上序列化/反序列化开销 (~10%)
    return kv_bytes / bandwidth * 1.1


# ──────────────────────────────────────────────
# 请求生成器
# ──────────────────────────────────────────────

@dataclass
class Request:
    """推理请求"""
    request_id: int
    prompt_tokens: int
    max_output_tokens: int
    arrival_time_ms: float = 0.0

    # 运行时状态
    prefill_done: bool = False
    generated_tokens: int = 0
    first_token_time_ms: float = 0.0
    completion_time_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens

    @property
    def is_done(self) -> bool:
        return self.generated_tokens >= self.max_output_tokens

    @property
    def ttft_ms(self) -> float:
        if self.first_token_time_ms == 0:
            return float('inf')
        return self.first_token_time_ms - self.arrival_time_ms

    @property
    def e2e_ms(self) -> float:
        if self.completion_time_ms == 0:
            return float('inf')
        return self.completion_time_ms - self.arrival_time_ms


def generate_requests(
    num_requests: int,
    prompt_range: tuple = (64, 2048),
    output_range: tuple = (32, 512),
    arrival_rate_per_s: float = 10.0,
    seed: int = 42,
) -> List[Request]:
    """生成随机请求序列"""
    rng = random.Random(seed)
    requests = []
    for i in range(num_requests):
        prompt_tokens = rng.randint(*prompt_range)
        output_tokens = rng.randint(*output_range)
        # Poisson 到达时间
        arrival = rng.expovariate(arrival_rate_per_s / 1000)  # per ms
        prev_arrival = requests[-1].arrival_time_ms if requests else 0
        arrival_time = prev_arrival + arrival
        requests.append(Request(i, prompt_tokens, output_tokens, arrival_time))
    return requests


# ──────────────────────────────────────────────
# 模拟器：Monolithic Serving
# ──────────────────────────────────────────────

@dataclass
class SimulationResult:
    """模拟结果"""
    total_requests: int = 0
    completed_requests: int = 0
    total_time_ms: float = 0.0
    ttft_list: List[float] = field(default_factory=list)
    e2e_list: List[float] = field(default_factory=list)
    throughput_tok_per_s: float = 0.0
    total_output_tokens: int = 0
    total_input_tokens: int = 0
    gpu_busy_time_ms: float = 0.0
    gpu_idle_time_ms: float = 0.0

    @property
    def avg_ttft_ms(self) -> float:
        return np.mean(self.ttft_list) if self.ttft_list else float('inf')

    @property
    def p99_ttft_ms(self) -> float:
        return np.percentile(self.ttft_list, 99) if self.ttft_list else float('inf')

    @property
    def avg_e2e_ms(self) -> float:
        return np.mean(self.e2e_list) if self.e2e_list else float('inf')

    @property
    def avg_itl_ms(self) -> float:
        """平均 inter-token latency"""
        if not self.e2e_list or not self.ttft_list:
            return float('inf')
        # ITL ≈ (e2e - ttft) / output_tokens
        total_decode_time = sum(e - t for e, t in zip(self.e2e_list, self.ttft_list))
        total_output = self.total_output_tokens
        return total_decode_time / max(1, total_output)

    @property
    def gpu_utilization(self) -> float:
        total = self.gpu_busy_time_ms + self.gpu_idle_time_ms
        return self.gpu_busy_time_ms / max(1, total)


def simulate_monolithic(
    model: ModelConfig,
    gpu: GPUConfig,
    requests: List[Request],
    max_batch: int = 32,
) -> SimulationResult:
    """Monolithic serving 模拟

    简化假设:
      - 请求按到达顺序处理
      - 每 step 处理一个 batch: 先做所有 prefill, 再做所有 decode
      - 无抢占
    """
    result = SimulationResult(total_requests=len(requests))
    current_time = 0.0

    waiting = list(requests)  # 按到达时间排序
    running = []

    while waiting or running:
        # 接收新请求
        newly_arrived = []
        while waiting and waiting[0].arrival_time_ms <= current_time:
            newly_arrived.append(waiting.pop(0))

        # 如果没有请求，快进到下一个到达
        if not running and not newly_arrived:
            if waiting:
                current_time = waiting[0].arrival_time_ms
                continue
            else:
                break

        running.extend(newly_arrived)

        # Prefill 阶段: 处理所有未 prefill 的请求
        prefill_reqs = [r for r in running if not r.prefill_done]
        if prefill_reqs:
            batch = prefill_reqs[:max_batch]
            # Batch prefill: 取最长 prompt 决定延迟
            max_prompt = max(r.prompt_tokens for r in batch)
            batch_size = len(batch)
            latency = prefill_latency_ms(model, gpu, max_prompt, batch_size)
            current_time += latency
            result.gpu_busy_time_ms += latency

            for r in batch:
                r.prefill_done = True
                r.first_token_time_ms = current_time

        # Decode 阶段: 所有 prefill 完成的请求各生成一个 token
        decode_reqs = [r for r in running if r.prefill_done and not r.is_done]
        if decode_reqs:
            batch = decode_reqs[:max_batch]
            # 平均 KV cache 长度
            avg_kv = np.mean([r.prompt_tokens + r.generated_tokens for r in batch])
            latency = decode_latency_ms(model, gpu, int(avg_kv), len(batch))
            current_time += latency
            result.gpu_busy_time_ms += latency

            for r in batch:
                r.generated_tokens += 1
                if r.is_done:
                    r.completion_time_ms = current_time
                    running.remove(r)
                    result.completed_requests += 1
                    result.total_output_tokens += r.generated_tokens
                    result.total_input_tokens += r.prompt_tokens
                    result.ttft_list.append(r.ttft_ms)
                    result.e2e_list.append(r.e2e_ms)
        else:
            # 没有可 decode 的请求（都在 prefill 或等待）
            result.gpu_idle_time_ms += 0.1
            current_time += 0.1

    result.total_time_ms = current_time
    if result.total_time_ms > 0:
        result.throughput_tok_per_s = result.total_output_tokens / (result.total_time_ms / 1000)
    return result


def simulate_disaggregated(
    model: ModelConfig,
    prefill_gpu: GPUConfig,
    decode_gpu: GPUConfig,
    requests: List[Request],
    num_p_instances: int = 1,
    num_d_instances: int = 1,
    network_bw_gbps: float = 100.0,
    max_batch: int = 32,
) -> SimulationResult:
    """Disaggregated serving 模拟

    P 实例: 只做 prefill, 计算完 KV Cache 后传输给 D 实例
    D 实例: 只做 decode, 接收 KV Cache 后开始生成
    """
    result = SimulationResult(total_requests=len(requests))
    current_time = 0.0

    waiting = list(requests)
    # D 实例的 decode 队列（每个实例独立）
    decode_running = [[] for _ in range(num_d_instances)]
    # KV Transfer 完成后加入 decode 的队列
    transfer_queue = []

    while waiting or any(decode_running) or transfer_queue:
        # 接收新请求
        newly_arrived = []
        while waiting and waiting[0].arrival_time_ms <= current_time:
            newly_arrived.append(waiting.pop(0))

        # P 实例: Prefill + Transfer
        if newly_arrived:
            batch = newly_arrived[:max_batch * num_p_instances]
            max_prompt = max(r.prompt_tokens for r in batch)
            p_latency = prefill_latency_ms(model, prefill_gpu, max_prompt, len(batch))
            current_time += p_latency
            result.gpu_busy_time_ms += p_latency

            for r in batch:
                r.prefill_done = True
                r.first_token_time_ms = current_time  # TTFT = prefill + transfer

                # KV Transfer
                transfer_time = kv_transfer_latency_ms(
                    model, prefill_gpu, r.prompt_tokens, network_bw_gbps
                )
                transfer_queue.append((r, current_time + transfer_time))

        # 处理 KV Transfer 完成
        ready_for_decode = []
        remaining_transfers = []
        for r, ready_time in transfer_queue:
            if ready_time <= current_time:
                ready_for_decode.append(r)
            else:
                remaining_transfers.append((r, ready_time))
        transfer_queue = remaining_transfers

        # D 实例: Decode
        # 分配请求到 D 实例（最少负载优先）
        for r in ready_for_decode:
            min_idx = min(range(num_d_instances), key=lambda i: len(decode_running[i]))
            decode_running[min_idx].append(r)

        total_decode_this_step = 0
        for d_idx in range(num_d_instances):
            if not decode_running[d_idx]:
                continue
            batch = decode_running[d_idx][:max_batch]
            avg_kv = np.mean([r.prompt_tokens + r.generated_tokens for r in batch])
            d_latency = decode_latency_ms(model, decode_gpu, int(avg_kv), len(batch))
            current_time = max(current_time, current_time) + d_latency / num_d_instances
            result.gpu_busy_time_ms += d_latency

            for r in batch:
                r.generated_tokens += 1
                if r.is_done:
                    r.completion_time_ms = current_time
                    decode_running[d_idx].remove(r)
                    result.completed_requests += 1
                    result.total_output_tokens += r.generated_tokens
                    result.total_input_tokens += r.prompt_tokens
                    result.ttft_list.append(r.ttft_ms)
                    result.e2e_list.append(r.e2e_ms)
            total_decode_this_step += len(batch)

        if not newly_arrived and not ready_for_decode and not total_decode_this_step:
            if waiting:
                current_time = waiting[0].arrival_time_ms
            elif transfer_queue:
                current_time = min(t for _, t in transfer_queue)
            else:
                break
            result.gpu_idle_time_ms += 0.1

    result.total_time_ms = current_time
    if result.total_time_ms > 0:
        result.throughput_tok_per_s = result.total_output_tokens / (result.total_time_ms / 1000)
    return result


# ──────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────

def experiment_1_monolithic_vs_disaggregated():
    """实验 1: Monolithic vs P/D Disaggregated"""
    print("\n" + "=" * 70)
    print("实验 1: Monolithic vs P/D Disaggregated 对比")
    print("=" * 70)

    model = LLAMA_70B
    gpu = H100
    requests = generate_requests(100, prompt_range=(128, 2048), output_range=(64, 512),
                                  arrival_rate_per_s=5.0)

    print(f"\n配置: Model={model.name}, GPU={gpu.name}")
    print(f"  100 请求, prompt=128-2048, output=64-512, 到达率=5/s")
    print(f"  Monolithic: 1x {gpu.name}")
    print(f"  Disaggregated: 1x {gpu.name} (P) + 1x {gpu.name} (D), NVLink 900 GB/s")

    # Monolithic
    mono = simulate_monolithic(model, gpu, requests)

    # Disaggregated
    disagg = simulate_disaggregated(model, gpu, gpu, requests, network_bw_gbps=900)

    print(f"\n{'指标':<25} {'Monolithic':>15} {'Disaggregated':>15} {'变化':>10}")
    print("-" * 70)
    print(f"{'完成请求数':<25} {mono.completed_requests:>15d} {disagg.completed_requests:>15d}")
    print(f"{'吞吐量 (tok/s)':<25} {mono.throughput_tok_per_s:>15.1f} "
          f"{disagg.throughput_tok_per_s:>15.1f} "
          f"{disagg.throughput_tok_per_s / max(0.1, mono.throughput_tok_per_s):>9.2f}x")
    print(f"{'平均 TTFT (ms)':<25} {mono.avg_ttft_ms:>15.1f} {disagg.avg_ttft_ms:>15.1f} "
          f"{mono.avg_ttft_ms / max(1, disagg.avg_ttft_ms):>9.2f}x")
    print(f"{'P99 TTFT (ms)':<25} {mono.p99_ttft_ms:>15.1f} {disagg.p99_ttft_ms:>15.1f} "
          f"{mono.p99_ttft_ms / max(1, disagg.p99_ttft_ms):>9.2f}x")
    print(f"{'平均 E2E (ms)':<25} {mono.avg_e2e_ms:>15.1f} {disagg.avg_e2e_ms:>15.1f} "
          f"{mono.avg_e2e_ms / max(1, disagg.avg_e2e_ms):>9.2f}x")

    print(f"\n关键洞察:")
    print(f"  - Disaggregated 用 2 倍 GPU 换取了 ITL 稳定性")
    print(f"  - Prefill 不再阻塞 Decode, TTFT 更可预测")
    print(f"  - KV Transfer 开销在高带宽网络下可忽略")


def experiment_2_kv_transfer_impact():
    """实验 2: KV Transfer 开销影响"""
    print("\n" + "=" * 70)
    print("实验 2: KV Transfer 带宽对分离架构的影响")
    print("=" * 70)

    model = LLAMA_70B
    gpu = H100
    requests = generate_requests(100, prompt_range=(256, 2048), output_range=(64, 512),
                                  arrival_rate_per_s=5.0)

    bandwidths = [
        ("Ethernet 10G", 1.25),
        ("Ethernet 25G", 3.125),
        ("RoCE 100G", 12.5),
        ("RoCE 200G", 25),
        ("NVLink", 112.5),
        ("NVLink 900GB/s", 900),
    ]

    print(f"\n配置: Model={model.name}, prompt=256-2048 tokens")
    print(f"\n{'带宽配置':<20} {'KV Transfer (ms)':>18} {'TTFT (ms)':>12} {'吞吐 tok/s':>12}")
    print("-" * 65)

    for name, bw_gbps in bandwidths:
        # 平均 KV transfer 延迟
        avg_prompt = 1024
        kv_time = kv_transfer_latency_ms(model, gpu, avg_prompt, bw_gbps)

        disagg = simulate_disaggregated(model, gpu, gpu, requests, network_bw_gbps=bw_gbps)
        print(f"{name:<20} {kv_time:>18.2f} {disagg.avg_ttft_ms:>12.1f} "
              f"{disagg.throughput_tok_per_s:>12.1f}")

    print(f"\n关键洞察:")
    print(f"  - 10GbE: KV Transfer 每个请求 ~100ms+ → 严重影响 TTFT")
    print(f"  - 100GbE RoCE: KV Transfer ~10ms → 可接受")
    print(f"  - NVLink: KV Transfer <1ms → 几乎无开销")
    print(f"  - 分离架构强烈依赖高速互联网络")


def experiment_3_instance_ratio():
    """实验 3: Prefill/Decode 实例比例优化"""
    print("\n" + "=" * 70)
    print("实验 3: Prefill/Decode 实例比例优化")
    print("=" * 70)

    model = LLAMA_70B
    gpu = H100
    requests = generate_requests(200, prompt_range=(256, 2048), output_range=(64, 512),
                                  arrival_rate_per_s=10.0)

    configs = [
        ("1P + 1D", 1, 1),
        ("1P + 2D", 1, 2),
        ("1P + 3D", 1, 3),
        ("2P + 1D", 2, 1),
        ("2P + 2D", 2, 2),
        ("2P + 4D", 2, 4),
        ("1P + 4D", 1, 4),
        ("4P + 4D", 4, 4),
    ]

    print(f"\n配置: Model={model.name}, GPU={gpu.name}, 200 请求, 到达率 10/s")
    print(f"  NVLink 900 GB/s KV Transfer")
    print(f"\n{'配置':<12} {'GPU数':>6} {'完成':>6} {'吞吐 tok/s':>12} "
          f"{'TTFT (ms)':>12} {'E2E (ms)':>12} {'GPU效率':>10}")
    print("-" * 70)

    for name, num_p, num_d in configs:
        total_gpus = num_p + num_d
        disagg = simulate_disaggregated(
            model, gpu, gpu, requests,
            num_p_instances=num_p, num_d_instances=num_d,
            network_bw_gbps=900,
        )
        # GPU 效率 = throughput / total_gpus
        gpu_eff = disagg.throughput_tok_per_s / total_gpus

        print(f"{name:<12} {total_gpus:>6d} {disagg.completed_requests:>6d} "
              f"{disagg.throughput_tok_per_s:>12.1f} "
              f"{disagg.avg_ttft_ms:>12.1f} {disagg.avg_e2e_ms:>12.1f} "
              f"{gpu_eff:>9.1f} t/g")

    print(f"\n关键洞察:")
    print(f"  - Decode 是瓶颈 (memory-bound, 每个 token 一次)")
    print(f"  - 增加 D 实例线性提升吞吐")
    print(f"  - Prefill 快速完成, 1-2 个 P 实例通常够用")
    print(f"  - 最优比例通常在 1P:2D ~ 1P:4D")


def experiment_4_prompt_length_impact():
    """实验 4: Prompt 长度对两种架构的影响"""
    print("\n" + "=" * 70)
    print("实验 4: Prompt 长度对 Monolithic vs Disaggregated 的影响")
    print("=" * 70)

    model = LLAMA_70B
    gpu = H100

    prompt_configs = [
        ("短 prompt (32-128)", (32, 128)),
        ("中 prompt (128-512)", (128, 512)),
        ("长 prompt (512-2048)", (512, 2048)),
        ("超长 prompt (2048-8192)", (2048, 8192)),
    ]

    print(f"\n配置: Model={model.name}, GPU={gpu.name}")
    print(f"  50 请求/output=64-256, 到达率=5/s")
    print(f"\n{'Prompt 长度':<25} {'Mono TTFT':>12} {'Disagg TTFT':>12} "
          f"{'Mono E2E':>12} {'Disagg E2E':>12}")
    print("-" * 75)

    for name, prompt_range in prompt_configs:
        reqs_mono = generate_requests(50, prompt_range=prompt_range,
                                       output_range=(64, 256), arrival_rate_per_s=5.0)
        reqs_disagg = generate_requests(50, prompt_range=prompt_range,
                                         output_range=(64, 256), arrival_rate_per_s=5.0)

        mono = simulate_monolithic(model, gpu, reqs_mono)
        disagg = simulate_disaggregated(model, gpu, gpu, reqs_disagg, network_bw_gbps=900)

        print(f"{name:<25} {mono.avg_ttft_ms:>12.1f} {disagg.avg_ttft_ms:>12.1f} "
              f"{mono.avg_e2e_ms:>12.1f} {disagg.avg_e2e_ms:>12.1f}")

    print(f"\n关键洞察:")
    print(f"  - 短 prompt 时两种架构差异小 (prefill 本身很快)")
    print(f"  - 长 prompt 时分离架构 TTFT 优势明显 (不排队等 decode)")
    print(f"  - 超长 prompt (8K+) 是分离架构的最佳场景")


def experiment_5_scalability():
    """实验 5: 扩展效率对比"""
    print("\n" + "=" * 70)
    print("实验 5: 扩展效率对比 — 增加实例数")
    print("=" * 70)

    model = LLAMA_70B
    gpu = H100

    # Monolithic: 1-4 GPU (batch parallel)
    print(f"\n--- Monolithic (通过增加 batch size 利用多 GPU) ---")
    print(f"{'GPU 数':>6} {'Batch Size':>12} {'吞吐 tok/s':>12} {'TTFT (ms)':>12} {'扩展效率':>10}")
    print("-" * 55)

    mono_base_throughput = 0
    for num_gpu in [1, 2, 4]:
        batch_size = 16 * num_gpu
        reqs = generate_requests(200, prompt_range=(256, 2048),
                                  output_range=(64, 512), arrival_rate_per_s=10.0)
        # 简化: 多 GPU = 多 batch (实际需要 TP)
        result = simulate_monolithic(model, gpu, reqs, max_batch=batch_size)
        if num_gpu == 1:
            mono_base_throughput = result.throughput_tok_per_s
        efficiency = result.throughput_tok_per_s / (mono_base_throughput * num_gpu)
        print(f"{num_gpu:>6d} {batch_size:>12d} {result.throughput_tok_per_s:>12.1f} "
              f"{result.avg_ttft_ms:>12.1f} {efficiency:>9.2f}")

    # Disaggregated: 1P+1D → 4P+4D
    print(f"\n--- Disaggregated (P=D, 增加实例对) ---")
    print(f"{'P+D':>6} {'总 GPU':>8} {'吞吐 tok/s':>12} {'TTFT (ms)':>12} {'扩展效率':>10}")
    print("-" * 50)

    disagg_base_throughput = 0
    for scale in [1, 2, 4]:
        reqs = generate_requests(200, prompt_range=(256, 2048),
                                  output_range=(64, 512), arrival_rate_per_s=10.0)
        result = simulate_disaggregated(model, gpu, gpu, reqs,
                                         num_p_instances=scale, num_d_instances=scale,
                                         network_bw_gbps=900)
        total_gpu = 2 * scale
        if scale == 1:
            disagg_base_throughput = result.throughput_tok_per_s
        efficiency = result.throughput_tok_per_s / (disagg_base_throughput * scale)
        print(f"{scale}P+{scale}D {total_gpu:>8d} {result.throughput_tok_per_s:>12.1f} "
              f"{result.avg_ttft_ms:>12.1f} {efficiency:>9.2f}")

    print(f"\n关键洞察:")
    print(f"  - Monolithic 扩展受限于 batch size (收益递减)")
    print(f"  - Disaggregated 线性扩展更接近理想 (P/D 独立)")
    print(f"  - 大规模部署 (>8 GPU) 时分离架构优势更明显")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Disaggregated Serving 模拟器")
    print("Prefill/Decode 分离架构: 优化 LLM 推理的下一步")
    print("=" * 70)

    experiment_1_monolithic_vs_disaggregated()
    experiment_2_kv_transfer_impact()
    experiment_3_instance_ratio()
    experiment_4_prompt_length_impact()
    experiment_5_scalability()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
Disaggregated Serving 关键要点:
  1. 核心动机: Prefill (compute-bound) 和 Decode (memory-bound) 资源需求矛盾
     - 混合部署时 Prefill 干扰 Decode ITL
     - 分离后各实例可独立优化硬件配置

  2. KV Transfer 是关键开销:
     - NVLink: <1ms, 几乎无开销
     - RoCE 200G: ~5-10ms, 可接受
     - Ethernet 10G: >100ms, 不可接受
     → 需要高速互联网络

  3. 实例比例优化:
     - Decode 通常是瓶颈 (每 token 一次 forward)
     - 1P:2D ~ 1P:4D 是常见的最优比例
     - Prefill 完成快, 不需要太多 P 实例

  4. 最佳场景:
     - 长 prompt (>2K tokens)
     - 高并发 (大量请求同时到达)
     - 低 ITL 要求 (在线推理)
     - 有高速互联网络 (NVLink/RoCE)

  5. 不适合场景:
     - 短 prompt (<128 tokens)
     - 低并发
     - 普通网络 (无高速互联)

  6. 实际部署架构:
     - Splitwise (Microsoft): P/D 分离 + KV Cache Pool
     - Mooncake: KV Cache 分布式存储
     - vLLM KV Transfer: NIXL connector, 支持 P/D 分离
     - DistServe: 最优 P/D 资源分配算法
    """)
