"""Continuous Batching 模拟器 — 静态 vs 连续批处理调度对比

模拟 LLM 推理服务中的批处理调度策略:
1. Static Batching: 等整个 batch 完成才加入新请求
2. Continuous Batching: 每个迭代可以加入/移除请求 (iteration-level scheduling)
3. Chunked Prefill: 长 prompt 分块处理，避免 decode 饥饿

使用方法:
    python continuous_batching_sim.py   # CPU 可运行
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque


@dataclass
class Request:
    """推理请求。"""
    req_id: str
    prompt_len: int           # prompt token 数
    output_len: int           # 需要生成的 token 数
    arrival_time: float       # 到达时间 (ms)
    priority: int = 0         # 优先级 (越小越高)

    # 运行时状态
    prompt_processed: int = 0  # 已处理的 prompt token 数
    tokens_generated: int = 0  # 已生成的 output token 数
    is_prefill_complete: bool = False
    first_token_time: Optional[float] = None  # 第一个 output token 时间
    start_time: Optional[float] = None        # 开始处理时间
    end_time: Optional[float] = None
    preempted_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def is_complete(self) -> bool:
        return self.tokens_generated >= self.output_len

    @property
    def kv_cache_tokens(self) -> int:
        """当前占用的 KV cache tokens。"""
        return self.prompt_processed + self.tokens_generated

    def reset(self):
        """重置运行时状态。"""
        self.prompt_processed = 0
        self.tokens_generated = 0
        self.is_prefill_complete = False
        self.first_token_time = None
        self.start_time = None
        self.end_time = None
        self.preempted_count = 0


@dataclass
class Metrics:
    """调度指标。"""
    total_requests: int = 0
    completed_requests: int = 0
    total_iterations: int = 0
    total_decode_tokens: int = 0
    total_prefill_tokens: int = 0
    gpu_busy_iterations: int = 0
    total_e2e_latency_ms: float = 0.0
    total_ttft_ms: float = 0.0        # Time to First Token
    total_preemptions: int = 0
    max_concurrent: int = 0
    schedule_time_ms: float = 0.0

    @property
    def throughput_tok_per_s(self) -> float:
        if self.schedule_time_ms == 0:
            return 0
        return self.total_decode_tokens / (self.schedule_time_ms / 1000)

    @property
    def avg_e2e_latency_ms(self) -> float:
        if self.completed_requests == 0:
            return 0
        return self.total_e2e_latency_ms / self.completed_requests

    @property
    def avg_ttft_ms(self) -> float:
        if self.completed_requests == 0:
            return 0
        return self.total_ttft_ms / self.completed_requests

    @property
    def gpu_utilization(self) -> float:
        if self.total_iterations == 0:
            return 0
        return self.gpu_busy_iterations / self.total_iterations * 100


# ============================================================
# Static Batching
# ============================================================

class StaticBatchScheduler:
    """静态批处理: 等整个 batch 完成才加入新请求。"""

    def __init__(self, max_batch_size: int, decode_time_ms: float,
                 prefill_time_ms: float):
        self.max_batch_size = max_batch_size
        self.decode_time_ms = decode_time_ms
        self.prefill_time_ms = prefill_time_ms
        self.metrics = Metrics()

    def simulate(self, requests: List[Request]) -> Metrics:
        sorted_reqs = sorted(requests, key=lambda r: r.arrival_time)
        current_time = 0.0
        batch_idx = 0

        while batch_idx * self.max_batch_size < len(sorted_reqs):
            start = batch_idx * self.max_batch_size
            end = min(start + self.max_batch_size, len(sorted_reqs))
            batch = sorted_reqs[start:end]

            batch_arrival = max(r.arrival_time for r in batch)
            current_time = max(current_time, batch_arrival)

            for r in batch:
                r.start_time = current_time

            # Prefill: 串行 (batch 内每个请求依次 prefill)
            for r in batch:
                current_time += r.prompt_len * self.prefill_time_ms
                r.prompt_processed = r.prompt_len
                r.is_prefill_complete = True
                r.first_token_time = current_time
                self.metrics.total_prefill_tokens += r.prompt_len

            # Decode: 并行 (所有活跃请求同步步进)
            max_decode_steps = max(r.output_len for r in batch)
            for step in range(max_decode_steps):
                active = 0
                for r in batch:
                    if not r.is_complete:
                        r.tokens_generated += 1
                        active += 1
                        self.metrics.total_decode_tokens += 1
                current_time += self.decode_time_ms
                self.metrics.total_iterations += 1
                if active > 0:
                    self.metrics.gpu_busy_iterations += 1
                    self.metrics.max_concurrent = max(self.metrics.max_concurrent, active)

            for r in batch:
                r.end_time = current_time
                self.metrics.completed_requests += 1
                self.metrics.total_e2e_latency_ms += r.end_time - r.arrival_time
                self.metrics.total_ttft_ms += r.first_token_time - r.arrival_time

            batch_idx += 1

        self.metrics.total_requests = len(sorted_reqs)
        self.metrics.schedule_time_ms = current_time
        return self.metrics


# ============================================================
# Continuous Batching
# ============================================================

class ContinuousBatchScheduler:
    """连续批处理: 每个迭代可以加入/移除请求。

    简化模型:
    - 每步固定时间 = decode_time_ms (假设 decode 是瓶颈)
    - 每步每个 decode 请求生成 1 token
    - Prefill 在同一步内处理，消耗 token budget
    - KV cache 用 token 总数限制 (简化 block 计算)
    """

    def __init__(self, max_batch_size: int, max_tokens_per_step: int,
                 decode_time_ms: float, prefill_time_ms: float,
                 max_kv_tokens: int = 32000,
                 policy: str = "fcfs",
                 long_prefill_threshold: int = 4096):
        self.max_batch_size = max_batch_size
        self.max_tokens_per_step = max_tokens_per_step
        self.decode_time_ms = decode_time_ms
        self.prefill_time_ms = prefill_time_ms
        self.policy = policy
        self.long_prefill_threshold = long_prefill_threshold
        self.max_kv_tokens = max_kv_tokens
        self.metrics = Metrics()

    def _total_kv_usage(self, running: List[Request]) -> int:
        return sum(r.kv_cache_tokens for r in running)

    def _sort_waiting(self, waiting: deque) -> List[Request]:
        items = list(waiting)
        if self.policy == "sjf":
            items.sort(key=lambda r: r.output_len)
        elif self.policy == "priority":
            items.sort(key=lambda r: (r.priority, r.arrival_time))
        else:
            items.sort(key=lambda r: r.arrival_time)
        return items

    def simulate(self, requests: List[Request]) -> Metrics:
        sorted_reqs = sorted(requests, key=lambda r: r.arrival_time)
        req_idx = 0
        current_time = 0.0

        waiting: deque = deque()
        running: List[Request] = []
        completed: List[Request] = []
        max_iters = len(sorted_reqs) * 500  # 安全上限

        for iteration in range(max_iters):
            # 1. 到达: 将已到达请求加入等待队列
            while req_idx < len(sorted_reqs) and sorted_reqs[req_idx].arrival_time <= current_time:
                waiting.append(sorted_reqs[req_idx])
                req_idx += 1

            # 2. 终止条件
            if not running and not waiting:
                if req_idx >= len(sorted_reqs):
                    break
                current_time = sorted_reqs[req_idx].arrival_time
                continue

            self.metrics.total_iterations += 1
            step_token_budget = self.max_tokens_per_step
            step_active = 0
            step_prefill_tokens = 0
            step_decode_count = 0

            # 3. 处理 RUNNING 请求
            still_running = []
            for req in running:
                if not req.is_prefill_complete:
                    # Prefill 请求: 处理一个 chunk
                    remaining = req.prompt_len - req.prompt_processed
                    chunk = min(remaining, self.long_prefill_threshold,
                                step_token_budget)
                    if chunk > 0:
                        req.prompt_processed += chunk
                        step_token_budget -= chunk
                        step_prefill_tokens += chunk
                        self.metrics.total_prefill_tokens += chunk
                        if req.prompt_processed >= req.prompt_len:
                            req.is_prefill_complete = True
                        step_active += 1
                    still_running.append(req)
                else:
                    # Decode 请求: 生成 1 token
                    if step_token_budget > 0:
                        req.tokens_generated += 1
                        step_token_budget -= 1
                        step_decode_count += 1
                        self.metrics.total_decode_tokens += 1
                        if req.first_token_time is None:
                            req.first_token_time = current_time
                        step_active += 1
                    still_running.append(req)

            running = still_running

            # 4. 从 WAITING 提升新请求 (仅当有 prefill budget)
            if step_token_budget > 0 and len(running) < self.max_batch_size:
                sorted_waiting = self._sort_waiting(waiting)
                admitted = []
                for req in sorted_waiting:
                    if len(running) >= self.max_batch_size:
                        break
                    if step_token_budget <= 0:
                        break

                    # 检查 KV cache 是否够
                    kv_needed = req.prompt_len + 1  # 至少 prompt + 1 decode token
                    kv_after = self._total_kv_usage(running) + kv_needed
                    if kv_after > self.max_kv_tokens:
                        continue  # 跳过，不够放

                    # Prefill chunk
                    chunk = min(req.prompt_len, self.long_prefill_threshold,
                                step_token_budget)
                    req.prompt_processed = chunk
                    step_token_budget -= chunk
                    step_prefill_tokens += chunk
                    self.metrics.total_prefill_tokens += chunk
                    if chunk >= req.prompt_len:
                        req.is_prefill_complete = True
                    req.start_time = current_time
                    if req.is_prefill_complete:
                        req.first_token_time = current_time
                    running.append(req)
                    admitted.append(req)
                    step_active += 1

                for req in admitted:
                    waiting.remove(req)

            # 5. 完成请求出队
            newly_completed = []
            still_running = []
            for req in running:
                if req.is_complete:
                    req.end_time = current_time
                    newly_completed.append(req)
                else:
                    still_running.append(req)

            running = still_running
            for req in newly_completed:
                completed.append(req)
                self.metrics.completed_requests += 1
                self.metrics.total_e2e_latency_ms += req.end_time - req.arrival_time
                if req.first_token_time is not None:
                    self.metrics.total_ttft_ms += req.first_token_time - req.arrival_time

            # 6. KV cache 压力检查: 抢占 (仅在确实超额时)
            kv_usage = self._total_kv_usage(running)
            if kv_usage > self.max_kv_tokens:
                # 抢占最新的 decode 请求 (最早到达的最后抢占)
                decode_reqs = [r for r in running if r.is_prefill_complete]
                while kv_usage > self.max_kv_tokens and decode_reqs:
                    # 抢占最晚到达的 decode 请求
                    victim = max(decode_reqs, key=lambda r: r.arrival_time)
                    kv_freed = victim.kv_cache_tokens
                    running.remove(victim)
                    decode_reqs.remove(victim)
                    victim.reset()
                    victim.preempted_count += 1
                    self.metrics.total_preemptions += 1
                    waiting.appendleft(victim)
                    kv_usage -= kv_freed

            # 7. 更新指标
            if step_active > 0:
                self.metrics.gpu_busy_iterations += 1
            self.metrics.max_concurrent = max(self.metrics.max_concurrent, step_active)

            # 8. 推进时间
            # 每步时间由 decode 和 prefill 共同决定
            step_time = self.decode_time_ms  # 基础 decode step
            if step_prefill_tokens > 0:
                prefill_time = step_prefill_tokens * self.prefill_time_ms
                step_time = max(step_time, prefill_time)
            current_time += step_time

        self.metrics.total_requests = len(sorted_reqs)
        self.metrics.schedule_time_ms = current_time
        self.metrics.total_preemptions = sum(r.preempted_count for r in completed)
        return self.metrics


# ============================================================
# 实验函数
# ============================================================

def generate_requests(n_requests: int, prompt_range: Tuple[int, int] = (64, 512),
                      output_range: Tuple[int, int] = (32, 256),
                      arrival_rate: float = 10.0, seed: int = 42) -> List[Request]:
    """生成模拟请求流 (Poisson arrival)。"""
    rng = np.random.default_rng(seed)
    intervals = rng.exponential(1000.0 / arrival_rate, size=n_requests)
    arrival_times = np.cumsum(intervals)
    requests = []
    for i in range(n_requests):
        prompt_len = int(rng.integers(prompt_range[0], prompt_range[1]))
        output_len = int(rng.integers(output_range[0], output_range[1]))
        requests.append(Request(
            req_id=f"req_{i}",
            prompt_len=prompt_len,
            output_len=output_len,
            arrival_time=float(arrival_times[i]),
        ))
    return requests


def reset_requests(requests: List[Request]):
    for r in requests:
        r.reset()


def experiment_static_vs_continuous():
    """实验 1: Static vs Continuous Batching 对比。"""
    print("=" * 60)
    print("实验 1: Static vs Continuous Batching 对比")
    print("=" * 60)

    prefill_time = 0.1    # ms/token
    decode_time = 0.5     # ms/token
    max_batch = 32
    n_requests = 200

    print(f"\n  配置: {n_requests} 请求, max_batch={max_batch}")
    print(f"  Prefill: {prefill_time} ms/token, Decode: {decode_time} ms/token")
    print(f"  Prompt: 64-512 tokens, Output: 32-256 tokens")

    print(f"\n  {'到达率(req/s)':>14} {'策略':<25} {'吞吐(tok/s)':>12} "
          f"{'E2E延迟(ms)':>12} {'TTFT(ms)':>10} {'GPU利用率':>10} {'总时间(ms)':>12}")
    print(f"  {'-'*14} {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*12}")

    for arrival_rate in [5, 10, 20, 50]:
        requests = generate_requests(n_requests, arrival_rate=arrival_rate, seed=42)

        # Static
        static = StaticBatchScheduler(max_batch, decode_time, prefill_time)
        static_m = static.simulate(requests)

        # Continuous
        reset_requests(requests)
        cont = ContinuousBatchScheduler(
            max_batch_size=max_batch,
            max_tokens_per_step=max_batch * 2,
            decode_time_ms=decode_time,
            prefill_time_ms=prefill_time,
            max_kv_tokens=50000,
        )
        cont_m = cont.simulate(requests)

        for label, m in [("Static Batch", static_m), ("Continuous Batch", cont_m)]:
            prefix = f"{arrival_rate:>14}" if "Static" in label else f"{'':>14}"
            print(f"  {prefix} {label:<25} {m.throughput_tok_per_s:>12.0f} "
                  f"{m.avg_e2e_latency_ms:>12.0f} {m.avg_ttft_ms:>10.0f} "
                  f"{m.gpu_utilization:>9.1f}% {m.schedule_time_ms:>12.0f}")


def experiment_scheduling_policies():
    """实验 2: 调度策略对比 (FCFS / SJF / Priority)。"""
    print("\n" + "=" * 60)
    print("实验 2: 调度策略对比 (FCFS vs SJF vs Priority)")
    print("=" * 60)

    prefill_time = 0.1
    decode_time = 0.5
    max_batch = 16
    n_requests = 100
    arrival_rate = 20

    print(f"\n  配置: {n_requests} 请求, 到达率={arrival_rate} req/s, max_batch={max_batch}")
    print(f"\n  {'策略':<25} {'吞吐(tok/s)':>12} {'E2E延迟(ms)':>12} "
          f"{'TTFT(ms)':>10} {'GPU利用率':>10} {'抢占次数':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    for policy in ["fcfs", "sjf", "priority"]:
        requests = generate_requests(n_requests, arrival_rate=arrival_rate, seed=42)
        if policy == "priority":
            rng = np.random.default_rng(42)
            for r in requests:
                r.priority = int(rng.integers(0, 5))

        scheduler = ContinuousBatchScheduler(
            max_batch_size=max_batch,
            max_tokens_per_step=max_batch * 2,
            decode_time_ms=decode_time,
            prefill_time_ms=prefill_time,
            max_kv_tokens=30000,
            policy=policy,
        )
        m = scheduler.simulate(requests)
        print(f"  {policy.upper() + ' (Continuous)':<25} {m.throughput_tok_per_s:>12.0f} "
              f"{m.avg_e2e_latency_ms:>12.0f} {m.avg_ttft_ms:>10.0f} "
              f"{m.gpu_utilization:>9.1f}% {m.total_preemptions:>10}")


def experiment_kv_cache_pressure():
    """实验 3: KV Cache 大小对性能的影响。"""
    print("\n" + "=" * 60)
    print("实验 3: KV Cache 大小对 Continuous Batching 的影响")
    print("=" * 60)

    prefill_time = 0.1
    decode_time = 0.5
    max_batch = 32
    n_requests = 150
    arrival_rate = 20

    print(f"\n  配置: {n_requests} 请求, 到达率={arrival_rate} req/s, max_batch={max_batch}")
    print(f"  Prompt: 128-1024 tokens (加重负载)")
    print(f"\n  {'KV Cache Tokens':>16} {'吞吐(tok/s)':>12} {'E2E延迟(ms)':>12} "
          f"{'TTFT(ms)':>10} {'抢占次数':>10} {'最大并发':>10}")
    print(f"  {'-'*16} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    for kv_tokens in [8000, 16000, 32000, 64000, 128000]:
        requests = generate_requests(
            n_requests, prompt_range=(128, 1024), output_range=(64, 512),
            arrival_rate=arrival_rate, seed=42)

        scheduler = ContinuousBatchScheduler(
            max_batch_size=max_batch,
            max_tokens_per_step=max_batch * 2,
            decode_time_ms=decode_time,
            prefill_time_ms=prefill_time,
            max_kv_tokens=kv_tokens,
        )
        m = scheduler.simulate(requests)
        print(f"  {kv_tokens:>16} {m.throughput_tok_per_s:>12.0f} "
              f"{m.avg_e2e_latency_ms:>12.0f} {m.avg_ttft_ms:>10.0f} "
              f"{m.total_preemptions:>10} {m.max_concurrent:>10}")


def experiment_batch_size_scaling():
    """实验 4: Batch Size 对吞吐和延迟的影响。"""
    print("\n" + "=" * 60)
    print("实验 4: Batch Size 扩展效率")
    print("=" * 60)

    prefill_time = 0.1
    decode_time = 0.5
    n_requests = 100
    arrival_rate = 30

    print(f"\n  配置: {n_requests} 请求, 到达率={arrival_rate} req/s")
    print(f"\n  {'Batch Size':>12} {'策略':<20} {'吞吐(tok/s)':>12} "
          f"{'E2E延迟(ms)':>12} {'TTFT(ms)':>10} {'GPU利用率':>10}")
    print(f"  {'-'*12} {'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    for batch_size in [4, 8, 16, 32, 64]:
        requests = generate_requests(n_requests, arrival_rate=arrival_rate, seed=42)

        static = StaticBatchScheduler(batch_size, decode_time, prefill_time)
        sm = static.simulate(requests)

        reset_requests(requests)
        cont = ContinuousBatchScheduler(
            max_batch_size=batch_size,
            max_tokens_per_step=batch_size * 2,
            decode_time_ms=decode_time,
            prefill_time_ms=prefill_time,
            max_kv_tokens=100000,
        )
        cm = cont.simulate(requests)

        for label, m in [("Static", sm), ("Continuous", cm)]:
            print(f"  {batch_size if 'Static' in label else '':>12} {label:<20} "
                  f"{m.throughput_tok_per_s:>12.0f} {m.avg_e2e_latency_ms:>12.0f} "
                  f"{m.avg_ttft_ms:>10.0f} {m.gpu_utilization:>9.1f}%")


def experiment_chunked_prefill_impact():
    """实验 5: Chunked Prefill 对延迟的影响。"""
    print("\n" + "=" * 60)
    print("实验 5: Chunked Prefill 对短请求延迟的影响")
    print("=" * 60)

    prefill_time = 0.1
    decode_time = 0.5
    max_batch = 32

    # 混合请求: 20% 长 prompt + 80% 短 prompt
    rng = np.random.default_rng(42)
    requests = []
    arrival_time = 0.0
    for i in range(100):
        if rng.random() < 0.2:
            prompt_len = int(rng.integers(1024, 2048))
            output_len = int(rng.integers(128, 512))
        else:
            prompt_len = int(rng.integers(32, 128))
            output_len = int(rng.integers(16, 64))
        arrival_time += float(rng.exponential(50))
        requests.append(Request(
            req_id=f"req_{i}", prompt_len=prompt_len, output_len=output_len,
            arrival_time=arrival_time,
        ))

    print(f"\n  配置: 100 请求 (20% 长 prompt 1024-2048t, 80% 短 prompt 32-128t)")
    print(f"\n  {'Chunk阈值':>12} {'吞吐(tok/s)':>12} {'E2E延迟(ms)':>12} "
          f"{'TTFT(ms)':>10} {'GPU利用率':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    for threshold in [256, 512, 1024, 2048, 8192]:
        reset_requests(requests)
        scheduler = ContinuousBatchScheduler(
            max_batch_size=max_batch,
            max_tokens_per_step=max_batch * 2,
            decode_time_ms=decode_time,
            prefill_time_ms=prefill_time,
            max_kv_tokens=100000,
            long_prefill_threshold=threshold,
        )
        m = scheduler.simulate(requests)
        print(f"  {threshold:>12} {m.throughput_tok_per_s:>12.0f} "
              f"{m.avg_e2e_latency_ms:>12.0f} {m.avg_ttft_ms:>10.0f} "
              f"{m.gpu_utilization:>9.1f}%")


def main():
    print("=" * 60)
    print("Continuous Batching 模拟器 — LLM 推理调度策略对比")
    print("=" * 60)

    experiment_static_vs_continuous()
    experiment_scheduling_policies()
    experiment_kv_cache_pressure()
    experiment_batch_size_scaling()
    experiment_chunked_prefill_impact()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
Continuous Batching 核心知识:

1. 核心思想:
   Static Batching 等整个 batch 完成才加入新请求 → GPU 利用率低 (30-50%)
   Continuous Batching 每个迭代可以加入/移除请求 → GPU 利用率高 (80-95%)

2. 关键机制:
   - Iteration-level scheduling: 每步重新决定哪些请求参与
   - Mixed prefill/decode: 同一步中可以混合处理
   - Request preemption: 内存不够时抢占正在运行的请求
   - Chunked prefill: 长 prompt 分块处理，避免 decode 饥饿

3. 调度策略:
   - FCFS: 先到先服务，公平，vLLM 默认
   - SJF: 最短作业优先，吞吐更优
   - Priority: 按优先级调度，适合多租户场景

4. KV Cache 管理:
   - 内存不够 → 抢占 (preemption)，释放 KV cache
   - vLLM V1 用重计算 (recomputation)，不用 swap
   - Prefix caching 可以加速被抢占请求的恢复

5. Chunked Prefill:
   - 将长 prompt 分成小块，穿插 decode 请求
   - 防止长 prefill 阻塞短请求的 decode
   - vLLM 参数: long_prefill_token_threshold
""")


if __name__ == "__main__":
    main()
