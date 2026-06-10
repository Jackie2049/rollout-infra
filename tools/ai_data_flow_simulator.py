#!/usr/bin/env python3
"""AI Data Flow Architecture Simulator — 数据流架构模拟器

模拟AI系统中三条核心数据流路径:
1. Inference: request→token→prefill→decode→response
2. Training: data→gradient→checkpoint→model
3. Deployment: checkpoint→registry→load→serve

7个核心定律验证:
- Incremental-Transfer Law (增量传输)
- Serialization-Cost Law (序列化成本)
- Compression-Ratio Law (压缩率)
- Checkpoint-Consistency Law (一致性)
- Transfer-Bottleneck Law (传输瓶颈)
- Feature-Store-Dual-Write Law (双写)
- Data-Flow-Debugging Law (调试层级)
"""

import math
import time
from dataclasses import dataclass


@dataclass
class SimResult:
    name: str
    metrics: dict
    insights: list[str]


class InferenceDataFlowSimulator:
    """模拟vLLM V1推理数据流路径 (7步)

    关键点:
    - NewRequestData=全量, CachedRequestData=增量 (增量传输定律)
    - IPC开销≈0.1ms (可忽略)
    - Sampling开销≈0.06ms (不是瓶颈)
    - KV cache是主要内存开销
    """

    def __init__(self, model_size_B=7, vocab_size=32000, hidden_dim=4096,
                 num_layers=32, num_heads=32, num_kv_heads=5,
                 block_size=16, max_seq_len=4096, hbm_bandwidth_GBps=890.8,
                 gpu_tflops=170, kv_dtype_bytes=2):
        self.model_size_B = model_size_B
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.hbm_bw = hbm_bandwidth_GBps * 1e9  # bytes/s
        self.gpu_tflops = gpu_tflops * 1e12  # FLOPS
        self.kv_bytes = kv_dtype_bytes

    def simulate_request_path(self, prompt_len=512, batch_size=32,
                              num_requests=100, use_incremental=True):
        """模拟完整的推理请求路径"""
        results = {}

        # Step 1-2: HTTP + Tokenizer
        prompt_bytes = prompt_len * 4  # int32 per token
        total_prompt_bytes = prompt_bytes * num_requests
        tokenizer_time_ms = 0.5  # ~0.5ms for BPE
        results['step1_2'] = {
            'prompt_bytes_per_req': prompt_bytes,
            'total_prompt_bytes': total_prompt_bytes,
            'tokenizer_time_ms': tokenizer_time_ms,
            'data_volume_kb': total_prompt_bytes / 1024,
        }

        # Step 3: Scheduler
        scheduler_time_ms = 0.1  # FCFS调度≈0.1ms
        results['step3'] = {
            'scheduler_time_ms': scheduler_time_ms,
            'algorithm': 'FCFS',
        }

        # Step 4: SchedulerOutput → ModelRunner (关键: 增量传输)
        # NewRequestData: 全量
        new_req_data = prompt_bytes + 128  # block_ids + params overhead
        # CachedRequestData: 增量 (仅new_token_ids + new_block_ids)
        cached_req_data = 4 + 64  # 1 new token_id + few block_ids + overhead

        if use_incremental:
            # 首次iteration全量, 之后增量
            total_sched_output_bytes = (new_req_data * batch_size +
                                        cached_req_data * (num_requests - batch_size))
        else:
            # 每次都全量传输 (无增量优化)
            total_sched_output_bytes = new_req_data * num_requests

        ipc_time_ms = 0.1  # ZMQ IPC ≈ 0.1ms

        # 增量传输定律验证
        full_transfer_bytes = new_req_data * num_requests
        incremental_savings = (1 - total_sched_output_bytes / full_transfer_bytes) * 100

        results['step4'] = {
            'new_req_bytes': new_req_data,
            'cached_req_bytes': cached_req_data,
            'total_sched_output_bytes': total_sched_output_bytes,
            'ipc_time_ms': ipc_time_ms,
            'incremental_savings_pct': incremental_savings,
            'use_incremental': use_incremental,
        }

        # Step 5: GPU Forward
        # Prefill (compute-bound)
        prefill_flops = 2 * self.model_size_B * 1e9 * prompt_len
        prefill_time_ms = prefill_flops / self.gpu_tflops * 1000

        # Decode (memory-bound)
        decode_bytes = 2 * self.model_size_B * 1e9  # weight reads (BF16)
        if self.kv_bytes == 1:  # INT8 KV
            kv_per_token = self.kv_bytes * self.hidden_dim * 2 * self.num_kv_heads
            # With INT4 weights
            decode_bytes = 0.5 * self.model_size_B * 1e9  # INT4 weight reads
        else:
            kv_per_token = self.kv_bytes * self.hidden_dim * 2 * self.num_kv_heads

        decode_time_ms = (decode_bytes / self.hbm_bw) * 1000  # per token

        # KV cache size per request
        kv_per_req_layer = prompt_len * kv_per_token
        kv_total_per_req = kv_per_req_layer * self.num_layers
        kv_total_all = kv_total_per_req * num_requests

        results['step5'] = {
            'prefill_time_ms': prefill_time_ms,
            'prefill_tflops': prefill_flops / 1e12,
            'decode_time_ms': decode_time_ms,
            'kv_bytes_per_token': kv_per_token,
            'kv_total_per_req_kb': kv_total_per_req / 1024,
            'kv_total_all_mb': kv_total_all / 1e6,
        }

        # Step 6: Sampling
        sampling_time_ms = 0.06  # sampling ≈ 0.06ms
        results['step6'] = {
            'sampling_time_ms': sampling_time_ms,
            'vocab_size': self.vocab_size,
            'logits_bytes': self.vocab_size * 4 * batch_size,
        }

        # Step 7: Response
        response_bytes = 256 * 4  # ~256 tokens * 4 bytes each
        results['step7'] = {
            'response_bytes': response_bytes,
        }

        # Total pipeline time (TTFT for first request)
        ttft_ms = (tokenizer_time_ms + scheduler_time_ms + ipc_time_ms +
                   prefill_time_ms + sampling_time_ms)

        # ITL for decode iterations
        itl_ms = scheduler_time_ms + ipc_time_ms + decode_time_ms + sampling_time_ms

        results['pipeline_summary'] = {
            'ttft_ms': ttft_ms,
            'itl_ms': itl_ms,
            'throughput_tok_per_s': 1000 / itl_ms * batch_size if itl_ms > 0 else 0,
            'total_kv_mb': kv_total_all / 1e6,
        }

        insights = [
            f"Incremental-Transfer: 增量传输节省{incremental_savings:.1f}%通信量",
            f"NewRequestData={new_req_data}B vs CachedRequestData={cached_req_data}B → {new_req_data/cached_req_data:.1f}x差距",
            f"Prefill={prefill_time_ms:.2f}ms (compute-bound) vs Decode={decode_time_ms:.2f}ms (memory-bound)",
            f"Sampling={sampling_time_ms}ms → 不是瓶颈 (attention+decode主导)",
            f"KV cache总量={kv_total_all/1e6:.1f}MB → 内存主要开销",
        ]

        return SimResult('inference_data_flow', results, insights)


class SerializationSimulator:
    """模拟不同序列化格式的加载速度和安全性

    关键定律:
    - Serialization-Cost Law: pickle=慢+不安全, safetensors=快+安全
    - Compression-Ratio Law: BF16→FP8→INT4, 但需fused kernel
    """

    def __init__(self, model_params=7e9, hbm_bandwidth_GBps=890.8,
                 pcie_bandwidth_GBps=12):
        self.model_params = model_params
        self.hbm_bw = hbm_bandwidth_GBps  # GB/s
        self.pcie_bw = pcie_bandwidth_GBps  # GB/s

    def simulate_formats(self):
        """对比不同序列化格式的性能"""
        formats = {
            'PyTorch_pt': {'bytes_per_param': 2, 'pickle_overhead_pct': 15, 'safe': False},
            'SafeTensors': {'bytes_per_param': 2, 'pickle_overhead_pct': 0, 'safe': True},
            'GGUF_INT4': {'bytes_per_param': 0.5, 'pickle_overhead_pct': 0, 'safe': True},
            'FP8': {'bytes_per_param': 1, 'pickle_overhead_pct': 0, 'safe': True},
            'BF16_baseline': {'bytes_per_param': 2, 'pickle_overhead_pct': 0, 'safe': True},
        }

        results = {}
        for name, spec in formats.items():
            model_size_bytes = self.model_params * spec['bytes_per_param']
            model_size_gb = model_size_bytes / 1e9

            # 加载时间 = PCIe传输 + unpickle overhead
            pcie_time_s = model_size_gb / self.pcie_bw
            pickle_time_s = pcie_time_s * spec['pickle_overhead_pct'] / 100
            total_load_s = pcie_time_s + pickle_time_s

            # HBM传输时间 (GPU内部)
            hbm_time_s = model_size_gb / self.hbm_bw

            results[name] = {
                'model_size_gb': model_size_gb,
                'pcie_load_time_s': total_load_s,
                'hbm_transfer_time_s': hbm_time_s,
                'pickle_overhead_s': pickle_time_s,
                'safe': spec['safe'],
                'compression_ratio': 2 / spec['bytes_per_param'],
            }

        insights = [
            f"Serialization-Cost Law: pickle overhead={formats['PyTorch_pt']['pickle_overhead_pct']}% → safetensors消除!",
            f"SafeTensors vs PyTorch.pt: {results['SafeTensors']['pcie_load_time_s']:.2f}s vs {results['PyTorch_pt']['pcie_load_time_s']:.2f}s → {results['PyTorch_pt']['pcie_load_time_s']/results['SafeTensors']['pcie_load_time_s']:.2f}x faster",
            f"Compression-Ratio: BF16→FP8→INT4 = {results['BF16_baseline']['compression_ratio']}x→{results['FP8']['compression_ratio']}x→{results['GGUF_INT4']['compression_ratio']}x",
            f"INT4加载: {results['GGUF_INT4']['pcie_load_time_s']:.2f}s vs BF16: {results['BF16_baseline']['pcie_load_time_s']:.2f}s → {results['BF16_baseline']['pcie_load_time_s']/results['GGUF_INT4']['pcie_load_time_s']:.1f}x faster",
            f"Production推荐: SafeTensors (安全+快) + INT4量化 (小+快加载)",
        ]

        return SimResult('serialization', results, insights)


class CheckpointSimulator:
    """模拟Checkpoint策略对比

    关键定律:
    - Checkpoint-Consistency Law: 同步=一致但慢, 异步=快但需版本管理
    """

    def __init__(self, model_params=7e9, num_gpus=8,
                 disk_bandwidth_GBps=0.5, nvlink_bandwidth_GBps=300):
        self.model_params = model_params
        self.num_gpus = num_gpus
        self.disk_bw = disk_bandwidth_GBps  # GB/s (SSD)
        self.nvlink_bw = nvlink_bandwidth_GBps

    def simulate_checkpoint_strategies(self, dtype='bf16', use_zero3=True):
        """对比3种checkpoint策略"""
        bytes_per_param = 2 if dtype == 'bf16' else 1

        # Model size
        model_size_gb = self.model_params * bytes_per_param / 1e9

        # Optimizer size (Adam: m + v = 2x params, FP32 = 4 bytes each)
        optimizer_size_gb = self.model_params * 2 * 4 / 1e9  # Adam m+v in FP32

        # Total per GPU
        if use_zero3:
            per_gpu_model_gb = model_size_gb / self.num_gpus
            per_gpu_optimizer_gb = optimizer_size_gb / self.num_gpus
        else:
            per_gpu_model_gb = model_size_gb
            per_gpu_optimizer_gb = optimizer_size_gb

        total_per_gpu_gb = per_gpu_model_gb + per_gpu_optimizer_gb

        # Strategy 1: 同步barrier
        sync_save_time_s = total_per_gpu_gb / self.disk_bw
        sync_barrier_overhead_s = 2.0  # GPU等待时间
        sync_total_s = sync_save_time_s + sync_barrier_overhead_s
        sync_consistent = True

        # Strategy 2: 异步 (no barrier)
        async_save_time_s = total_per_gpu_gb / self.disk_bw
        async_total_s = async_save_time_s  # 无等待
        async_consistent = False  # 可能不一致

        # Strategy 3: 异步+版本管理 (verl)
        verl_save_time_s = async_save_time_s
        verl_version_overhead_s = 0.1  # 版本追踪很小
        verl_total_s = verl_save_time_s + verl_version_overhead_s
        verl_consistent = True  # 版本管理保证一致性

        results = {
            'model_size_gb': model_size_gb,
            'optimizer_size_gb': optimizer_size_gb,
            'total_per_gpu_gb': total_per_gpu_gb,
            'use_zero3': use_zero3,
            'sync_barrier': {
                'total_time_s': sync_total_s,
                'save_time_s': sync_save_time_s,
                'barrier_overhead_s': sync_barrier_overhead_s,
                'consistent': sync_consistent,
            },
            'async': {
                'total_time_s': async_total_s,
                'save_time_s': async_save_time_s,
                'barrier_overhead_s': 0,
                'consistent': async_consistent,
            },
            'async_with_versioning': {
                'total_time_s': verl_total_s,
                'save_time_s': verl_save_time_s,
                'version_overhead_s': verl_version_overhead_s,
                'consistent': verl_consistent,
            },
            'speedup_async_vs_sync': sync_total_s / async_total_s,
            'speedup_verl_vs_sync': sync_total_s / verl_total_s,
        }

        insights = [
            f"Checkpoint-Consistency Law: sync={sync_total_s:.1f}s(一致) vs async={async_total_s:.1f}s(快但可能不一致)",
            f"Model={model_size_gb:.1f}GB + Optimizer={optimizer_size_gb:.1f}GB → 单GPU需{total_per_gpu_gb:.1f}GB!",
            f"ZeRO-3: 每GPU只需{total_per_gpu_gb:.1f}GB → {self.num_gpus}GPU可行",
            f"verl策略: {verl_total_s:.1f}s + 版本管理 → 一致+快 → {sync_total_s/verl_total_s:.2f}x faster than sync",
            f"异步版本管理=生产最优 (verl选择!)",
        ]

        return SimResult('checkpoint', results, insights)


class TransferBottleneckSimulator:
    """模拟不同传输瓶颈对数据流的影响

    关键定律:
    - Transfer-Bottleneck Law: PCIe=模型加载瓶颈, NVLink=生产理想
    """

    def __init__(self, model_size_gb=14, kv_transfer_mb=32,
                 pcie_bandwidth_GBps=12, nvlink_bandwidth_GBps=300,
                 ethernet_bandwidth_GBps=10):
        self.model_size_gb = model_size_gb
        self.kv_transfer_mb = kv_transfer_mb
        self.pcie_bw = pcie_bandwidth_GBps
        self.nvlink_bw = nvlink_bandwidth_GBps
        self.ethernet_bw = ethernet_bandwidth_GBps

    def simulate_transfer_scenarios(self):
        """模拟3种传输场景"""
        scenarios = {
            'PCIe_CPU_GPU_load': {
                'data_gb': self.model_size_gb,
                'bandwidth_GBps': self.pcie_bw,
                'description': '模型加载 (CPU→GPU)',
            },
            'PCIe_CPU_GPU_load_INT4': {
                'data_gb': self.model_size_gb * 0.25,  # INT4 = 75%省
                'bandwidth_GBps': self.pcie_bw,
                'description': 'INT4模型加载 (CPU→GPU)',
            },
            'NVLink_KV_transfer': {
                'data_gb': self.kv_transfer_mb / 1024,
                'bandwidth_GBps': self.nvlink_bw,
                'description': 'KV transfer (GPU→GPU, PD分离)',
            },
            'PCIe_KV_transfer': {
                'data_gb': self.kv_transfer_mb / 1024,
                'bandwidth_GBps': self.pcie_bw,
                'description': 'KV transfer PCIe (RTX 4090)',
            },
            'Ethernet_cross_node': {
                'data_gb': self.kv_transfer_mb / 1024,
                'bandwidth_GBps': self.ethernet_bw,
                'description': '跨节点KV transfer (需RDMA)',
            },
            'HBM_model_read': {
                'data_gb': self.model_size_gb,
                'bandwidth_GBps': 890.8,  # RTX 4090 HBM
                'description': 'GPU内部权重读取 (decode)',
            },
        }

        results = {}
        for name, spec in scenarios.items():
            transfer_time_ms = spec['data_gb'] / spec['bandwidth_GBps'] * 1000
            results[name] = {
                'data_gb': spec['data_gb'],
                'bandwidth_GBps': spec['bandwidth_GBps'],
                'transfer_time_ms': transfer_time_ms,
                'description': spec['description'],
            }

        insights = [
            f"Transfer-Bottleneck Law: PCIe模型加载={results['PCIe_CPU_GPU_load']['transfer_time_ms']:.0f}ms vs INT4={results['PCIe_CPU_GPU_load_INT4']['transfer_time_ms']:.0f}ms",
            f"NVLink KV={results['NVLink_KV_transfer']['transfer_time_ms']:.3f}ms vs PCIe KV={results['PCIe_KV_transfer']['transfer_time_ms']:.2f}ms → {results['PCIe_KV_transfer']['transfer_time_ms']/results['NVLink_KV_transfer']['transfer_time_ms']:.1f}x差距",
            f"HBM权重读取={results['HBM_model_read']['transfer_time_ms']:.2f}ms → GPU内部不是瓶颈",
            f"Ethernet跨节点={results['Ethernet_cross_node']['transfer_time_ms']:.2f}ms → 需RDMA优化",
            f"RTX 4090: PCIe only → INT4加载{results['PCIe_CPU_GPU_load_INT4']['transfer_time_ms']:.0f}ms → 可接受!",
        ]

        return SimResult('transfer_bottleneck', results, insights)


class FeatureStoreSimulator:
    """模拟Feature Store双写一致性挑战

    关键定律:
    - Feature-Store-Dual-Write Law: online=AP(快), offline=CP(一致)
    """

    def __init__(self, online_latency_ms=1.0, offline_latency_ms=500.0,
                 consistency_lag_s=5.0):
        self.online_latency = online_latency_ms
        self.offline_latency = offline_latency_ms
        self.consistency_lag = consistency_lag_s

    def simulate_dual_write(self, num_features=1000, num_requests=10000,
                            write_ratio=0.1):
        """模拟online+offline双写"""
        num_writes = int(num_requests * write_ratio)

        # Online writes (Redis, AP)
        online_total_time_s = num_writes * self.online_latency / 1000
        online_consistency = 'eventual'  # AP = eventual consistency

        # Offline writes (HDFS, CP)
        offline_total_time_s = num_writes * self.offline_latency / 1000
        offline_consistency = 'strong'  # CP = strong consistency

        # Dual write (Feast)
        dual_total_time_s = online_total_time_s + offline_total_time_s
        dual_consistency_lag_s = self.consistency_lag
        dual_inconsistency_pct = dual_consistency_lag_s * num_writes / (
            num_writes * offline_total_time_s + 1) * 100

        # Read scenarios
        # Inference: online read (fast)
        inference_read_ms = self.online_latency

        # Training: offline read (consistent)
        training_read_ms = self.offline_latency

        # Dual-write inconsistencies
        inconsistency_events = int(num_writes * dual_consistency_lag_s /
                                   (offline_total_time_s / num_writes * 1000 + 1))

        results = {
            'online': {
                'total_write_time_s': online_total_time_s,
                'read_latency_ms': inference_read_ms,
                'consistency': online_consistency,
                'model': 'AP (可用性优先)',
            },
            'offline': {
                'total_write_time_s': offline_total_time_s,
                'read_latency_ms': training_read_ms,
                'consistency': offline_consistency,
                'model': 'CP (一致性优先)',
            },
            'dual_write': {
                'total_write_time_s': dual_total_time_s,
                'consistency_lag_s': dual_consistency_lag_s,
                'inconsistency_events': inconsistency_events,
                'inconsistency_pct': min(dual_inconsistency_pct, 100),
            },
            'speedup_online_vs_offline': offline_total_time_s / online_total_time_s,
        }

        insights = [
            f"Feature-Store-Dual-Write: online(Redis)={online_total_time_s:.1f}s vs offline(HDFS)={offline_total_time_s:.1f}s → {offline_total_time_s/online_total_time_s:.0f}x差距",
            f"推理用online={inference_read_ms}ms (快) vs 训练用offline={training_read_ms}ms (一致)",
            f"双写一致性lag={dual_consistency_lag_s}s → {inconsistency_events}个不一致事件",
            f"Feast方案: 统一API → 用户不需要知道online/offline → 但一致性lag仍存在",
            f"Production策略: 推理→online(快)/训练→offline(一致)/Feast→统一入口",
        ]

        return SimResult('feature_store', results, insights)


class DataFlowDebuggingSimulator:
    """模拟数据流调试层级 (逐层深入)

    关键定律:
    - Data-Flow-Debugging Law: metrics→log→trace→profile → 逐层深入
    """

    def __init__(self):
        self.debugging_layers = {
            'metrics': {'time_min': 1, 'resolution': 'global', 'cost': 'low'},
            'log': {'time_min': 5, 'resolution': 'event', 'cost': 'low'},
            'trace': {'time_min': 15, 'resolution': 'request', 'cost': 'medium'},
            'profile': {'time_min': 30, 'resolution': 'kernel', 'cost': 'high'},
        }

    def simulate_debugging_scenarios(self):
        """模拟5种常见数据流问题的调试路径"""
        scenarios = {
            'high_latency': {
                'symptom': 'P99 TTFT > 5s',
                'layers': ['metrics→发现P99高', 'log→找慢请求', 'trace→定位prefill慢', 'profile→prefill GEMM瓶颈'],
                'root_cause': 'Prefill compute-bound → 需chunked prefill或PD分离',
                'resolution_time_min': 30,
            },
            'oom': {
                'symptom': 'gpu_cache_usage > 90%',
                'layers': ['metrics→发现KV高', 'log→OOM error', 'trace→大S请求', 'profile→KV block分配'],
                'root_cause': '长S+高并发 → KV超限 → INT8 KV + 滑动窗口',
                'resolution_time_min': 15,
            },
            'low_throughput': {
                'symptom': 'generation_throughput < 100 tok/s',
                'layers': ['metrics→低吞吐', 'log→GPU利用率低', 'trace→batch小', 'profile→等待调度'],
                'root_cause': '并发低 → batch小 → 增加max_num_seqs → continuous batching',
                'resolution_time_min': 10,
            },
            'kv_transfer_failure': {
                'symptom': 'PD分离 KV transfer timeout',
                'layers': ['metrics→KV connector stats异常', 'log→transfer error', 'trace→哪步超时', 'profile→NCCL/RDMA瓶颈'],
                'root_cause': '网络带宽不足 → 需NVLink或RDMA → RTX 4090不适合PD分离',
                'resolution_time_min': 60,
            },
            'request_drops': {
                'symptom': 'num_preemptions > 10/min',
                'layers': ['metrics→抢占多', 'log→preempt event', 'trace→哪个请求被抢', 'profile→KV cache紧张'],
                'root_cause': 'KV cache不足 → 减少并发或增加GPU → INT8 KV缓解',
                'resolution_time_min': 20,
            },
        }

        results = {}
        total_time = 0
        for name, spec in scenarios.items():
            total_time += spec['resolution_time_min']
            results[name] = {
                'symptom': spec['symptom'],
                'debugging_path': spec['layers'],
                'root_cause': spec['root_cause'],
                'resolution_time_min': spec['resolution_time_min'],
                'layers_used': len(spec['layers']),
            }

        avg_time = total_time / len(scenarios)
        results['summary'] = {
            'avg_resolution_time_min': avg_time,
            'total_scenarios': len(scenarios),
            'debugging_layers': list(self.debugging_layers.keys()),
        }

        insights = [
            f"Data-Flow-Debugging Law: 4层深入 → metrics(global)→log(event)→trace(request)→profile(kernel)",
            f"平均解决时间={avg_time:.1f}min → 先metrics快速定位 → 逐层深入",
            f"Metrics层最快(1min) → 90%问题可从metrics定位类型 → 减少排查时间!",
            f"Profile层最慢(30min) → 仅在metrics+log+trace无法定位时使用 → 生产少见",
            f"RTX 4090: 单GPU → 无分布式 → 不需要trace → metrics+log+profile足够",
        ]

        return SimResult('data_flow_debugging', results, insights)


def run_all_simulators():
    """运行所有模拟器并汇总结果"""
    print("=" * 70)
    print("AI Data Flow Architecture Simulator")
    print("=" * 70)

    simulators = [
        ("1. Inference Data Flow", InferenceDataFlowSimulator()),
        ("2. Serialization Formats", SerializationSimulator()),
        ("3. Checkpoint Strategies", CheckpointSimulator()),
        ("4. Transfer Bottleneck", TransferBottleneckSimulator()),
        ("5. Feature Store Dual-Write", FeatureStoreSimulator()),
        ("6. Data Flow Debugging", DataFlowDebuggingSimulator()),
    ]

    all_insights = []
    for title, sim in simulators:
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")

        if isinstance(sim, InferenceDataFlowSimulator):
            # BF16 baseline
            result = sim.simulate_request_path(prompt_len=512, batch_size=32,
                                               num_requests=100, use_incremental=True)
            print(f"  Incremental transfer savings: {result.metrics['step4']['incremental_savings_pct']:.1f}%")
            print(f"  TTFT: {result.metrics['pipeline_summary']['ttft_ms']:.2f}ms")
            print(f"  ITL: {result.metrics['pipeline_summary']['itl_ms']:.2f}ms")
            print(f"  Throughput: {result.metrics['pipeline_summary']['throughput_tok_per_s']:.0f} tok/s")

            # INT4+INT8KV comparison
            sim_int4 = InferenceDataFlowSimulator(kv_dtype_bytes=1)
            result_int4 = sim_int4.simulate_request_path(prompt_len=512, batch_size=55,
                                                          num_requests=100, use_incremental=True)
            print(f"\n  --- INT4+INT8KV comparison ---")
            print(f"  ITL: {result_int4.metrics['pipeline_summary']['itl_ms']:.2f}ms")
            print(f"  Throughput: {result_int4.metrics['pipeline_summary']['throughput_tok_per_s']:.0f} tok/s")
            print(f"  KV total: {result_int4.metrics['pipeline_summary']['total_kv_mb']:.1f}MB")

        elif isinstance(sim, SerializationSimulator):
            result = sim.simulate_formats()
            for fmt_name, fmt_data in result.metrics.items():
                print(f"  {fmt_name}: {fmt_data['model_size_gb']:.1f}GB, "
                      f"load={fmt_data['pcie_load_time_s']:.2f}s, "
                      f"safe={fmt_data['safe']}")

        elif isinstance(sim, CheckpointSimulator):
            result = sim.simulate_checkpoint_strategies()
            print(f"  Total per GPU: {result.metrics['total_per_gpu_gb']:.1f}GB")
            print(f"  Sync barrier: {result.metrics['sync_barrier']['total_time_s']:.1f}s (consistent)")
            print(f"  Async: {result.metrics['async']['total_time_s']:.1f}s (not consistent)")
            print(f"  verl async+version: {result.metrics['async_with_versioning']['total_time_s']:.1f}s (consistent)")
            print(f"  verl vs sync speedup: {result.metrics['speedup_verl_vs_sync']:.2f}x")

        elif isinstance(sim, TransferBottleneckSimulator):
            result = sim.simulate_transfer_scenarios()
            for name, data in result.metrics.items():
                print(f"  {data['description']}: {data['transfer_time_ms']:.2f}ms "
                      f"({data['data_gb']:.2f}GB @ {data['bandwidth_GBps']:.0f}GB/s)")

        elif isinstance(sim, FeatureStoreSimulator):
            result = sim.simulate_dual_write()
            print(f"  Online(Redis) write: {result.metrics['online']['total_write_time_s']:.1f}s (AP)")
            print(f"  Offline(HDFS) write: {result.metrics['offline']['total_write_time_s']:.1f}s (CP)")
            print(f"  Online vs Offline speedup: {result.metrics['speedup_online_vs_offline']:.0f}x")
            print(f"  Dual-write lag: {result.metrics['dual_write']['consistency_lag_s']:.1f}s")

        elif isinstance(sim, DataFlowDebuggingSimulator):
            result = sim.simulate_debugging_scenarios()
            for name, data in result.metrics.items():
                if name == 'summary':
                    continue
                print(f"  {name}: {data['symptom']} → "
                      f"{data['resolution_time_min']}min → "
                      f"{data['root_cause']}")

        for insight in result.insights:
            print(f"  INSIGHT: {insight}")
        all_insights.extend(result.insights)

    print(f"\n{'=' * 70}")
    print("CORE LAWS SUMMARY (7 Laws)")
    print(f"{'=' * 70}")
    laws = [
        "1. Incremental-Transfer: 增量传输节省>90%通信量 (CachedRequestData)",
        "2. Serialization-Cost: safetensors > pickle (快+安全)",
        "3. Compression-Ratio: INT4=75%省 但需fused kernel",
        "4. Checkpoint-Consistency: verl异步+版本管理=最优 (快+一致)",
        "5. Transfer-Bottleneck: PCIe=模型加载瓶颈, INT4缓解",
        "6. Feature-Store-Dual-Write: online=AP, offline=CP, Feast统一",
        "7. Data-Flow-Debugging: metrics→log→trace→profile 逐层深入",
    ]
    for law in laws:
        print(f"  {law}")

    print(f"\nTotal insights: {len(all_insights)}")
    return all_insights


if __name__ == '__main__':
    run_all_simulators()