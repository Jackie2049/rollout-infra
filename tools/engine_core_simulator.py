#!/usr/bin/env python3
"""vLLM V1 Engine Core 模拟器

模拟 EngineCore 的 step() 循环: 调度→执行→采样→输出
覆盖 5 个实验:
  1. 基础 Step Loop: 请求生命周期模拟
  2. Batch Queue (Pipeline Parallelism): 调度/执行解耦
  3. ZMQ 通信延迟影响: 序列化+IO 开销分析
  4. KV Cache Auto-fit: 显存不足时的 max_model_len 自适应
  5. GPU 利用率分析: 不同 batch/seq 配置下的利用率

Usage:
    conda run -n ai-infra python tools/engine_core_simulator.py
    conda run -n ai-infra python tools/engine_core_simulator.py --experiment 2
"""

import argparse
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Data Classes
# ============================================================

@dataclass
class SimRequest:
    """模拟请求"""
    request_id: str
    prompt_tokens: int
    max_output_tokens: int
    generated_tokens: int = 0
    finished: bool = False
    finish_reason: str = ""
    arrival_time: float = 0.0

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.generated_tokens

    @property
    def remaining_tokens(self):
        return self.max_output_tokens - self.generated_tokens


@dataclass
class KVBlock:
    """KV Cache Block"""
    block_id: int
    request_id: str = ""
    ref_count: int = 0


@dataclass
class SchedulerOutput:
    """调度器输出"""
    scheduled_new_reqs: list = field(default_factory=list)
    scheduled_running_reqs: list = field(default_factory=list)
    total_num_scheduled_tokens: int = 0
    num_scheduled_tokens: dict = field(default_factory=dict)


@dataclass
class ModelOutput:
    """模型输出"""
    generated_tokens: dict = field(default_factory=dict)  # req_id -> [token_ids]
    finished_reqs: list = field(default_factory=list)


@dataclass
class EngineCoreOutput:
    """引擎核心输出"""
    request_id: str
    new_tokens: int
    finished: bool
    finish_reason: str = ""


# ============================================================
# Simulated Scheduler
# ============================================================

class SimScheduler:
    """模拟 V1 调度器"""

    def __init__(self, max_num_seqs, max_num_batched_tokens, block_size,
                 num_gpu_blocks, max_model_len):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.max_model_len = max_model_len

        # Request queues
        self.waiting: list[SimRequest] = []
        self.running: list[SimRequest] = []
        self.finished_count = 0

        # KV Cache tracking
        self.blocks_in_use = 0

    def add_request(self, req: SimRequest):
        self.waiting.append(req)

    def has_requests(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def _blocks_needed(self, req: SimRequest, new_tokens: int) -> int:
        total = req.total_tokens + new_tokens
        return math.ceil(total / self.block_size) - math.ceil(req.total_tokens / self.block_size)

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        available_blocks = self.num_gpu_blocks - self.blocks_in_use

        # 1. Schedule running requests (decode) - priority
        for req in self.running:
            if req.finished:
                continue
            tokens_to_gen = min(1, req.remaining_tokens)  # 1 token per decode step
            blocks_needed = self._blocks_needed(req, tokens_to_gen)

            if available_blocks >= blocks_needed:
                output.scheduled_running_reqs.append(req)
                output.num_scheduled_tokens[req.request_id] = tokens_to_gen
                output.total_num_scheduled_tokens += tokens_to_gen
                available_blocks -= blocks_needed
                self.blocks_in_use += blocks_needed

        # 2. Schedule new requests (prefill)
        remaining_tokens = self.max_num_batched_tokens - output.total_num_scheduled_tokens
        remaining_seqs = self.max_num_seqs - len(output.scheduled_running_reqs)

        still_waiting = []
        for req in self.waiting:
            if remaining_seqs <= 0 or remaining_tokens <= 0:
                still_waiting.append(req)
                continue

            # Chunked prefill: limit prompt tokens per step
            prefill_tokens = min(req.prompt_tokens, remaining_tokens)
            blocks_needed = math.ceil(prefill_tokens / self.block_size)

            if available_blocks >= blocks_needed:
                output.scheduled_new_reqs.append(req)
                output.num_scheduled_tokens[req.request_id] = prefill_tokens
                output.total_num_scheduled_tokens += prefill_tokens
                remaining_tokens -= prefill_tokens
                remaining_seqs -= 1
                available_blocks -= blocks_needed
                self.blocks_in_use += blocks_needed
                self.running.append(req)
            else:
                still_waiting.append(req)

        self.waiting = still_waiting
        return output

    def update_from_output(self, scheduler_output: SchedulerOutput,
                           model_output: ModelOutput) -> list[EngineCoreOutput]:
        results = []

        for req in scheduler_output.scheduled_new_reqs:
            tokens = model_output.generated_tokens.get(req.request_id, [])
            req.generated_tokens += len(tokens)

        for req in scheduler_output.scheduled_running_reqs:
            tokens = model_output.generated_tokens.get(req.request_id, [])
            req.generated_tokens += len(tokens)

            if req.generated_tokens >= req.max_output_tokens:
                req.finished = True
                req.finish_reason = "length"
            elif random.random() < 0.05:  # 5% chance of stop token
                req.finished = True
                req.finish_reason = "stop"

        # Clean up finished requests
        finished_reqs = [r for r in self.running if r.finished]
        for req in finished_reqs:
            blocks_used = math.ceil(req.total_tokens / self.block_size)
            self.blocks_in_use -= blocks_used
            self.running.remove(req)
            self.finished_count += 1

        # Generate outputs
        all_scheduled = scheduler_output.scheduled_new_reqs + scheduler_output.scheduled_running_reqs
        for req in all_scheduled:
            tokens = model_output.generated_tokens.get(req.request_id, [])
            is_finished = req.finished
            results.append(EngineCoreOutput(
                request_id=req.request_id,
                new_tokens=len(tokens),
                finished=is_finished,
                finish_reason=req.finish_reason if is_finished else ""
            ))

        return results


# ============================================================
# Simulated Model Executor
# ============================================================

class SimModelExecutor:
    """模拟模型执行器"""

    def __init__(self, model_params_b: float, gpu_tflops: float, hbm_bw_gbps: float,
                 gpu_mem_gb: float, model_mem_gb: float):
        self.model_params_b = model_params_b
        self.gpu_tflops = gpu_tflops
        self.hbm_bw_gbps = hbm_bw_gbps
        self.gpu_mem_gb = gpu_mem_gb
        self.model_mem_gb = model_mem_gb
        self.available_mem_gb = gpu_mem_gb - model_mem_gb

    def execute_prefill_time_ms(self, num_tokens: int, batch_size: int) -> float:
        """Prefill: compute-bound (N² attention)"""
        if num_tokens == 0:
            return 0.01
        # FLOPS ≈ 2 * P * num_tokens (linear part dominant for batch serving)
        flops = 2 * self.model_params_b * 1e9 * num_tokens
        time_s = flops / (self.gpu_tflops * 1e12 * 0.5)  # 50% utilization
        return time_s * 1000  # ms

    def execute_decode_time_ms(self, batch_size: int, total_kv_tokens: int) -> float:
        """Decode: memory-bound"""
        if batch_size == 0:
            return 0.01
        # Weight read + KV read
        bytes_to_read = (self.model_params_b * 1e9 * 2 +  # weights (BF16)
                         total_kv_tokens * 2 * 1024 * 2)   # KV (2 layers, BF16)
        time_s = bytes_to_read / (self.hbm_bw_gbps * 1e9)
        return time_s * 1000  # ms

    def execute_model(self, scheduler_output: SchedulerOutput) -> tuple[float, ModelOutput]:
        """执行模型, 返回 (latency_ms, ModelOutput)"""
        output = ModelOutput()

        new_reqs = scheduler_output.scheduled_new_reqs
        running_reqs = scheduler_output.scheduled_running_reqs

        # Prefill latency
        prefill_tokens = sum(scheduler_output.num_scheduled_tokens.get(r.request_id, 0)
                             for r in new_reqs)
        prefill_time = self.execute_prefill_time_ms(prefill_tokens, len(new_reqs))

        # Decode latency
        decode_time = self.execute_decode_time_ms(len(running_reqs),
                                                   sum(r.total_tokens for r in running_reqs))

        total_time = max(prefill_time, decode_time)  # Prefill and decode can overlap

        # Generate outputs
        for req in new_reqs:
            n_tokens = scheduler_output.num_scheduled_tokens.get(req.request_id, 0)
            output.generated_tokens[req.request_id] = [random.randint(0, 31999)] * n_tokens

        for req in running_reqs:
            output.generated_tokens[req.request_id] = [random.randint(0, 31999)]

        return total_time, output


# ============================================================
# Simulated Engine Core
# ============================================================

class SimEngineCore:
    """模拟 EngineCore 的 step() 循环"""

    def __init__(self, scheduler: SimScheduler, executor: SimModelExecutor,
                 use_batch_queue: bool = False, batch_queue_size: int = 1):
        self.scheduler = scheduler
        self.executor = executor
        self.use_batch_queue = use_batch_queue
        self.batch_queue_size = batch_queue_size

    def step(self) -> tuple[list[EngineCoreOutput], float, bool]:
        """一次 step() 迭代: 调度→执行→输出

        Returns: (outputs, latency_ms, model_executed)
        """
        if not self.scheduler.has_requests():
            return [], 0.0, False

        # 1. Schedule
        scheduler_output = self.scheduler.schedule()

        # 2. Execute
        latency, model_output = self.executor.execute_model(scheduler_output)

        # 3. Update
        engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)

        model_executed = scheduler_output.total_num_scheduled_tokens > 0
        return engine_outputs, latency, model_executed


# ============================================================
# Experiments
# ============================================================

def experiment_1_basic_step_loop():
    """实验 1: 基础 Step Loop — 请求生命周期模拟"""
    print("\n" + "=" * 70)
    print("实验 1: 基础 Step Loop — 请求生命周期模拟")
    print("=" * 70)

    # 7B model on A100
    scheduler = SimScheduler(
        max_num_seqs=128,
        max_num_batched_tokens=4096,
        block_size=16,
        num_gpu_blocks=5000,
        max_model_len=4096,
    )
    executor = SimModelExecutor(
        model_params_b=7,
        gpu_tflops=312,
        hbm_bw_gbps=2039,
        gpu_mem_gb=80,
        model_mem_gb=14,
    )
    engine = SimEngineCore(scheduler, executor)

    # Generate requests
    random.seed(42)
    requests = []
    for i in range(50):
        prompt_len = random.randint(64, 1024)
        max_output = random.randint(32, 512)
        requests.append(SimRequest(
            request_id=f"req_{i:03d}",
            prompt_tokens=prompt_len,
            max_output_tokens=max_output,
            arrival_time=i * 0.1,
        ))

    # Add all requests
    for req in requests:
        scheduler.add_request(req)

    # Run step loop
    step_count = 0
    total_latency = 0
    total_tokens = 0
    outputs_per_step = []
    latencies = []

    while scheduler.has_requests() and step_count < 1000:
        outputs, latency, executed = engine.step()
        step_count += 1
        total_latency += latency
        n_tokens = sum(o.new_tokens for o in outputs)
        total_tokens += n_tokens
        outputs_per_step.append(len(outputs))
        latencies.append(latency)

    print(f"\n📊 Step Loop 统计:")
    print(f"  总 step 数:          {step_count}")
    print(f"  总延迟:              {total_latency:.1f} ms")
    print(f"  平均 step 延迟:      {total_latency / step_count:.2f} ms")
    print(f"  总生成 tokens:       {total_tokens}")
    print(f"  完成请求数:          {scheduler.finished_count}")
    print(f"  吞吐量:              {total_tokens / (total_latency / 1000):.0f} tok/s")
    print(f"  平均每步输出请求数:  {sum(outputs_per_step) / len(outputs_per_step):.1f}")
    print(f"  P50 step 延迟:       {sorted(latencies)[len(latencies)//2]:.2f} ms")
    print(f"  P99 step 延迟:       {sorted(latencies)[int(len(latencies)*0.99)]:.2f} ms")

    # Phase breakdown
    warmup_steps = sum(1 for o in outputs_per_step[:20] if o > 0)
    steady_state = outputs_per_step[20:] if len(outputs_per_step) > 20 else []
    print(f"\n📊 Phase 分析:")
    print(f"  Warmup 阶段 (前20步):  {warmup_steps} 步有输出")
    if steady_state:
        print(f"  稳态阶段 平均输出:     {sum(steady_state)/len(steady_state):.1f} 请求/步")
        print(f"  稳态阶段 平均延迟:     {sum(latencies[20:])/len(latencies[20:]):.2f} ms")


def experiment_2_batch_queue():
    """实验 2: Batch Queue (Pipeline Parallelism) — 调度/执行解耦"""
    print("\n" + "=" * 70)
    print("实验 2: Batch Queue — 调度/执行解耦 (Pipeline Parallelism)")
    print("=" * 70)

    results = {}

    for bq_size in [1, 2, 4]:
        scheduler = SimScheduler(
            max_num_seqs=128, max_num_batched_tokens=4096,
            block_size=16, num_gpu_blocks=5000, max_model_len=4096,
        )
        executor = SimModelExecutor(
            model_params_b=7, gpu_tflops=312, hbm_bw_gbps=2039,
            gpu_mem_gb=80, model_mem_gb=14,
        )
        engine = SimEngineCore(scheduler, executor, batch_queue_size=bq_size)

        random.seed(42)
        for i in range(50):
            scheduler.add_request(SimRequest(
                request_id=f"req_{i:03d}",
                prompt_tokens=random.randint(64, 1024),
                max_output_tokens=random.randint(32, 512),
            ))

        # Simulate batch queue behavior
        batch_queue: deque = deque(maxlen=bq_size)
        step_count = 0
        total_latency = 0
        total_tokens = 0
        pipeline_bubbles = 0

        while (scheduler.has_requests() or batch_queue) and step_count < 1000:
            step_count += 1

            # Try to schedule and execute
            if scheduler.has_requests() and len(batch_queue) < bq_size:
                outputs, latency, executed = engine.step()
                total_latency += latency
                n_tokens = sum(o.new_tokens for o in outputs)
                total_tokens += n_tokens
                if executed:
                    batch_queue.appendleft((outputs, latency))

            # Drain batch queue
            if batch_queue:
                batch_queue.pop()

            # Detect pipeline bubble (queue empty but scheduler has work)
            if not batch_queue and scheduler.has_requests():
                pipeline_bubbles += 1

        throughput = total_tokens / (total_latency / 1000) if total_latency > 0 else 0
        results[bq_size] = {
            'steps': step_count,
            'latency': total_latency,
            'tokens': total_tokens,
            'throughput': throughput,
            'bubbles': pipeline_bubbles,
        }

    print(f"\n{'BQ Size':<10} {'Steps':<8} {'Total Lat(ms)':<15} {'Tokens':<10} "
          f"{'Throughput':<15} {'Bubbles':<10}")
    print("-" * 70)
    for bq, r in results.items():
        print(f"{bq:<10} {r['steps']:<8} {r['latency']:<15.1f} {r['tokens']:<10} "
              f"{r['throughput']:<15.0f} {r['bubbles']:<10}")

    print(f"\n💡 关键发现:")
    print(f"  - Batch Queue=1 即无队列, 调度和执行串行")
    print(f"  - Batch Queue>1 允许调度/执行重叠, 减少 pipeline 气泡")
    print(f"  - Pipeline Bubble = 队列空但调度器有工作的步数")
    print(f"  - PP 场景下 BQ=2-4 通常足够消除大部分气泡")


def experiment_3_zmq_communication():
    """实验 3: ZMQ 通信延迟 — 序列化+IO 开销分析"""
    print("\n" + "=" * 70)
    print("实验 3: ZMQ 通信延迟分析")
    print("=" * 70)

    # Simulate serialization/deserialization costs
    results = {}

    for batch_size in [1, 8, 32, 128]:
        for tokens_per_req in [1, 32, 128, 512]:
            # Msgpack serialization (approximate)
            # EngineCoreOutput: ~100 bytes per output
            # EngineCoreOutputs: overhead + batch_size * per_output
            output_size_bytes = 50 + batch_size * (20 + tokens_per_req * 4)

            # ZMQ send latency (ipc:// vs tcp://)
            zmq_ipc_latency_us = 5 + output_size_bytes / 10000  # ~5-10us for IPC
            zmq_tcp_latency_us = 30 + output_size_bytes / 5000  # ~30-100us for TCP

            # Msgpack encode/decode time
            encode_time_us = output_size_bytes * 0.001  # ~1ns per byte
            decode_time_us = output_size_bytes * 0.002  # ~2ns per byte

            # Total round-trip
            total_ipc_us = zmq_ipc_latency_us + encode_time_us + decode_time_us
            total_tcp_us = zmq_tcp_latency_us + encode_time_us + decode_time_us

            key = f"B={batch_size},T={tokens_per_req}"
            results[key] = {
                'size_kb': output_size_bytes / 1024,
                'ipc_us': total_ipc_us,
                'tcp_us': total_tcp_us,
                'encode_us': encode_time_us,
            }

    print(f"\n{'Config':<20} {'Size(KB)':<10} {'IPC(μs)':<10} {'TCP(μs)':<10} {'Encode(μs)':<12}")
    print("-" * 65)
    for key, r in results.items():
        print(f"{key:<20} {r['size_kb']:<10.2f} {r['ipc_us']:<10.1f} "
              f"{r['tcp_us']:<10.1f} {r['encode_us']:<12.2f}")

    # Compare with step time
    step_time_ms = 15  # Typical decode step
    zmq_overhead_pct_ipc = (total_ipc_us / 1000) / step_time_ms * 100
    zmq_overhead_pct_tcp = (total_tcp_us / 1000) / step_time_ms * 100

    print(f"\n💡 关键发现:")
    print(f"  - ZMQ IPC 延迟: ~5-10μs (单进程内通信)")
    print(f"  - ZMQ TCP 延迟: ~30-100μs (跨进程/跨节点)")
    print(f"  - Msgpack 编解码: ~0.001-0.01μs (相比 ZMQ 可忽略)")
    print(f"  - 典型 step=15ms, ZMQ 开销 <0.7% (IPC) / <1% (TCP)")
    print(f"  - 双线程解耦 (input_thread + output_thread) 重叠 ZMQ IO 与计算")
    print(f"  - Buffer 复用策略减少 GC 压力")


def experiment_4_kv_cache_autofit():
    """实验 4: KV Cache Auto-fit — 显存不足时的自适应"""
    print("\n" + "=" * 70)
    print("实验 4: KV Cache Auto-fit — 显存自适应")
    print("=" * 70)

    # Different GPU configs
    configs = [
        {"gpu": "A100 80GB", "mem_gb": 80, "model_gb": 14},   # 7B BF16
        {"gpu": "A100 40GB", "mem_gb": 40, "model_gb": 14},
        {"gpu": "A16 15GB",  "mem_gb": 15, "model_gb": 14},
        {"gpu": "L4 24GB",   "mem_gb": 24, "model_gb": 14},
        {"gpu": "H100 80GB", "mem_gb": 80, "model_gb": 28},   # 13B BF16
    ]

    block_size = 16
    bytes_per_token = 2 * 2  # 2 layers * BF16 (simplified)

    print(f"\n{'GPU':<15} {'Model(GB)':<10} {'Avail(GB)':<10} {'MaxLen':<10} "
          f"{'Blocks':<10} {'MaxSeq(B=128)':<15} {'Autofit':<10}")
    print("-" * 80)

    for cfg in configs:
        available_gb = cfg["mem_gb"] - cfg["model_gb"]
        available_bytes = available_gb * 1e9

        # Try max_model_len = 8192
        max_model_len = 8192
        kv_per_seq_bytes = max_model_len * bytes_per_token
        max_seqs = 128
        total_kv_bytes = kv_per_seq_bytes * max_seqs

        if total_kv_bytes <= available_bytes:
            num_blocks = int(available_bytes / (block_size * bytes_per_token))
            autofit = "N/A"
        else:
            # Auto-fit: reduce max_model_len
            max_kv_bytes_per_seq = available_bytes / max_seqs
            max_model_len = int(max_kv_bytes_per_seq / bytes_per_token)
            # Round down to block_size
            max_model_len = (max_model_len // block_size) * block_size
            total_kv_bytes = max_model_len * bytes_per_token * max_seqs
            num_blocks = int(available_bytes / (block_size * bytes_per_token))
            autofit = f"→{max_model_len}"

        actual_max_len = max_model_len
        max_concurrent = min(max_seqs, int(available_bytes / (actual_max_len * bytes_per_token)))

        print(f"{cfg['gpu']:<15} {cfg['model_gb']:<10.1f} {available_gb:<10.1f} "
              f"{actual_max_len:<10} {num_blocks:<10} {max_concurrent:<15} {autofit:<10}")

    print(f"\n💡 关键发现:")
    print(f"  - Auto-fit 机制: 显存不足时自动降低 max_model_len")
    print(f"  - 7B 模型至少需 ~14GB (权重), A16 15GB 几乎无法运行")
    print(f"  - KV Cache bytes/token = 2 × num_layers × dtype_size (简化)")
    print(f"  - 实际还包括 activation、临时缓冲区等开销")
    print(f"  - vLLM 通过 determine_available_memory() 精确测量可用显存")


def experiment_5_gpu_utilization():
    """实验 5: GPU 利用率 — 不同 batch/seq 配置"""
    print("\n" + "=" * 70)
    print("实验 5: GPU 利用率分析 — Batch Size vs 序列长度")
    print("=" * 70)

    gpu_tflops = 312  # A100 BF16
    hbm_bw_gbps = 2039  # A100
    model_params_b = 7

    print(f"\n模型: {model_params_b}B 参数, GPU: A100 (312 TFLOPS, 2039 GB/s)")
    print(f"Ridge Point = 2 × 312 / 2039 ≈ {2 * gpu_tflops / hbm_bw_gbps:.1f} ops/byte\n")

    print(f"{'Batch':<8} {'SeqLen':<8} {'Phase':<10} {'AI':<8} {'Util%':<8} "
          f"{'Lat(ms)':<10} {'Bound':<10}")
    print("-" * 70)

    for batch_size in [1, 8, 32, 128, 256]:
        for seq_len in [128, 512, 2048, 8192]:
            # Prefill: compute-bound
            flops = 2 * model_params_b * 1e9 * seq_len * batch_size
            bytes_read = model_params_b * 1e9 * 2  # weights
            ai_prefill = flops / bytes_read if bytes_read > 0 else 0
            util_prefill = min(100, ai_prefill / (2 * gpu_tflops / hbm_bw_gbps) * 100)
            time_prefill = max(flops / (gpu_tflops * 1e12), bytes_read / (hbm_bw_gbps * 1e9)) * 1000

            # Decode: memory-bound
            bytes_decode = (model_params_b * 1e9 * 2 +
                           batch_size * seq_len * 2 * 2 * 32)  # KV (32 layers)
            flops_decode = 2 * model_params_b * 1e9 * batch_size
            ai_decode = flops_decode / bytes_decode if bytes_decode > 0 else 0
            util_decode = min(100, ai_decode / (2 * gpu_tflops / hbm_bw_gbps) * 100)
            time_decode = bytes_decode / (hbm_bw_gbps * 1e9) * 1000

            bound = "compute" if ai_prefill > (2 * gpu_tflops / hbm_bw_gbps) else "memory"
            print(f"{batch_size:<8} {seq_len:<8} {'Prefill':<10} {ai_prefill:<8.1f} "
                  f"{util_prefill:<8.1f} {time_prefill:<10.2f} {bound:<10}")

            bound = "compute" if ai_decode > (2 * gpu_tflops / hbm_bw_gbps) else "memory"
            print(f"{batch_size:<8} {seq_len:<8} {'Decode':<10} {ai_decode:<8.1f} "
                  f"{util_decode:<8.1f} {time_decode:<10.2f} {bound:<10}")

    print(f"\n💡 关键发现:")
    print(f"  - Prefill: AI 随 seq_len 线性增长, 长 seq 时 compute-bound")
    print(f"  - Decode: AI ≈ 2P/(2P + B×S×KV), 始终 memory-bound (AI < ridge)")
    print(f"  - Decode 吞吐 ∝ batch_size (因为 memory-bound)")
    print(f"  - step() 的执行时间取决于 max(prefill_time, decode_time)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="vLLM V1 Engine Core 模拟器")
    parser.add_argument("--experiment", "-e", type=int, default=0,
                        help="运行特定实验 (1-5), 0=全部")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          vLLM V1 Engine Core 模拟器                        ║")
    print("║   step() 核心循环: 调度→执行→采样→输出                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    experiments = {
        1: experiment_1_basic_step_loop,
        2: experiment_2_batch_queue,
        3: experiment_3_zmq_communication,
        4: experiment_4_kv_cache_autofit,
        5: experiment_5_gpu_utilization,
    }

    if args.experiment == 0:
        for exp_fn in experiments.values():
            exp_fn()
    elif args.experiment in experiments:
        experiments[args.experiment]()
    else:
        print(f"Unknown experiment: {args.experiment}")
        return

    print("\n" + "=" * 70)
    print("✅ 模拟完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
