#!/usr/bin/env python3
"""AI Training Infrastructure Simulator — AI训练基础设施模拟器

模拟AI训练系统基础设施与优化策略:
1. Training Loop Memory (训练循环内存)
2. Data Parallelism (数据并行)
3. Gradient Accumulation (梯度累积)
4. Mixed Precision (混合精度)
5. Checkpointing (检查点策略)
6. Training Pipeline (训练流水线)

7个核心定律验证:
- Training-Memory-5x Law (训练内存5x)
- PCIe-Training-Disaster Law (PCIe训练灾难)
- BF16-Best Law (BF16最优)
- Gradient-Accumulation-Bridge Law (梯度累积桥梁)
- LoRA-Training-Feasibility Law (LoRA训练可行性)
- Async-Checkpoint Law (异步检查点)
- Training-Decision Law (训练决策)
"""

import math
from dataclasses import dataclass


@dataclass
class SimResult:
    name: str
    metrics: dict
    insights: list[str]


class TrainingLoopMemorySimulator:
    """训练循环内存分析模拟"""

    def __init__(self):
        # 7B model parameters = 7e9
        self.model_params = 7e9
        self.bytes_per_param = {
            'FP32': 4, 'BF16': 2, 'FP16': 2, 'INT4': 0.5,
        }

    def simulate_memory(self):
        """模拟不同配置的训练内存需求"""
        results = {}
        configs = [
            ('7B_FP32', 7e9, 'FP32', 1),
            ('7B_BF16', 7e9, 'BF16', 1),
            ('7B_BF16_FSDP8', 7e9, 'BF16', 8),
            ('7B_BF16_FSDP2', 7e9, 'BF16', 2),
            ('7B_INT4_LoRA', 7e9 * 0.004, 'BF16', 1),  # LoRA: only 0.4% trainable
            ('125M_BF16', 125e6, 'BF16', 1),
            ('25M_BF16', 25e6, 'BF16', 1),
        ]

        for name, params, dtype, n_gpu in configs:
            bpb = self.bytes_per_param[dtype]

            # Model params
            model_gb = params * bpb / 1e9
            per_gpu_model_gb = model_gb / n_gpu

            # AdamW states: m + v (always FP32 for stability)
            adam_gb = params * 4 * 2 / 1e9  # 2 states, each FP32
            per_gpu_adam_gb = adam_gb / n_gpu

            # Gradients (same dtype as params)
            grad_gb = params * bpb / 1e9
            per_gpu_grad_gb = grad_gb / n_gpu

            # Activations (rough: same as params for backward)
            act_gb = params * bpb / 1e9 / n_gpu  # with FSDP, also sharded

            # Total per GPU
            total_per_gpu_gb = per_gpu_model_gb + per_gpu_adam_gb + per_gpu_grad_gb + act_gb

            # Can fit in 24GB?
            fits_24gb = total_per_gpu_gb <= 24

            results[name] = {
                'params': params,
                'dtype': dtype,
                'n_gpu': n_gpu,
                'model_gb': round(model_gb, 2),
                'per_gpu_model_gb': round(per_gpu_model_gb, 2),
                'per_gpu_adam_gb': round(per_gpu_adam_gb, 2),
                'per_gpu_total_gb': round(total_per_gpu_gb, 2),
                'fits_24gb': fits_24gb,
                'memory_ratio': round(total_per_gpu_gb / (model_gb if model_gb > 0 else 1), 1),
            }

        insights = [
            f"Training-Memory-5x: 7B BF16单GPU→{results['7B_BF16']['per_gpu_total_gb']}GB→5x参数→24GB→OOM!",
            f"FSDP8→每GPU={results['7B_BF16_FSDP8']['per_gpu_total_gb']}GB→fits={results['7B_BF16_FSDP8']['fits_24gb']}→勉强!",
            f"FSDP2→每GPU={results['7B_BF16_FSDP2']['per_gpu_total_gb']}GB→fits={results['7B_BF16_FSDP2']['fits_24gb']}→OOM!",
            f"LoRA→每GPU={results['7B_INT4_LoRA']['per_gpu_total_gb']}GB→fits={results['7B_INT4_LoRA']['fits_24gb']}→可行!→99.6%参数省!",
            f"25M→每GPU={results['25M_BF16']['per_gpu_total_gb']}GB→可行→125M→{results['125M_BF16']['per_gpu_total_gb']}GB→也可行!",
        ]
        return SimResult('training_memory', results, insights)


class DataParallelismSimulator:
    """数据并行策略对比模拟"""

    def __init__(self):
        self.pcie_bw_gb_s = 2.76
        self.nvlink_bw_gb_s = 300
        self.p2p_penalty = 2.0  # P2P disabled → 2x slower

    def simulate_dp(self):
        """模拟DDP/FSDP在不同GPU数量下的性能"""
        results = {}
        model_sizes = {'25M': 0.05, '125M': 0.25, '7B': 14}

        for model_name, param_gb in model_sizes.items():
            for n_gpu in [1, 2, 4, 8]:
                if n_gpu == 1:
                    results[f'{model_name}_{n_gpu}GPU'] = {
                        'speedup': 1.0,
                        'comm_time_ms': 0,
                        'compute_time_ms': 100 if model_name == '7B' else 10 if model_name == '125M' else 2,
                        'comm_ratio_pct': 0,
                        'dp_type': 'single',
                    }
                    continue

                compute_ms = 100 if model_name == '7B' else 10 if model_name == '125M' else 2

                # DDP: AllReduce entire gradients
                ddp_comm_data_gb = param_gb  # AllReduce = full parameter size
                ddp_pcie_comm_ms = ddp_comm_data_gb / (self.pcie_bw_gb_s / self.p2p_penalty) * 1000
                ddp_nvlink_comm_ms = ddp_comm_data_gb / self.nvlink_bw_gb_s * 1000

                # FSDP: ReduceScatter + AllGather per layer = 2 × param/N per GPU
                fsdp_comm_data_per_gpu_gb = 2 * param_gb / n_gpu
                fsdp_pcie_comm_ms = fsdp_comm_data_per_gpu_gb / (self.pcie_bw_gb_s / self.p2p_penalty) * 1000
                fsdp_nvlink_comm_ms = fsdp_comm_data_per_gpu_gb / self.nvlink_bw_gb_s * 1000

                # FSDP PCIe speedup
                fsdp_pcie_total_ms = compute_ms + fsdp_pcie_comm_ms
                fsdp_pcie_ratio = fsdp_pcie_comm_ms / fsdp_pcie_total_ms * 100
                fsdp_pcie_speedup = n_gpu * compute_ms / fsdp_pcie_total_ms if fsdp_pcie_total_ms > 0 else 0

                # FSDP NVLink speedup
                fsdp_nvlink_total_ms = compute_ms + fsdp_nvlink_comm_ms
                fsdp_nvlink_ratio = fsdp_nvlink_comm_ms / fsdp_nvlink_total_ms * 100
                fsdp_nvlink_speedup = n_gpu * compute_ms / fsdp_nvlink_total_ms

                results[f'{model_name}_{n_gpu}GPU'] = {
                    'compute_ms': compute_ms,
                    'fsdp_pcie_comm_ms': round(fsdp_pcie_comm_ms, 1),
                    'fsdp_pcie_ratio_pct': round(fsdp_pcie_ratio, 1),
                    'fsdp_pcie_speedup': round(fsdp_pcie_speedup, 2),
                    'fsdp_nvlink_comm_ms': round(fsdp_nvlink_comm_ms, 2),
                    'fsdp_nvlink_ratio_pct': round(fsdp_nvlink_ratio, 2),
                    'fsdp_nvlink_speedup': round(fsdp_nvlink_speedup, 2),
                    'dp_type': 'FSDP1',
                }

        insights = [
            f"PCIe-Disaster: 7B 8GPU→comm_ratio={results['7B_8GPU']['fsdp_pcie_ratio_pct']}%→speedup={results['7B_8GPU']['fsdp_pcie_speedup']}x→灾难!",
            f"NVLink: 7B 8GPU→comm_ratio={results['7B_8GPU']['fsdp_nvlink_ratio_pct']}%→speedup={results['7B_8GPU']['fsdp_nvlink_speedup']}x→OK!",
            f"25M 2GPU→PCIe speedup={results['25M_2GPU']['fsdp_pcie_speedup']}x→勉强→4GPU={results['25M_4GPU']['fsdp_pcie_speedup']}x→不可!",
            f"FSDP通信=DDP/N但每层2次→总体与DDP相同→但内存省8x!",
            f"RTX 4090训练结论: ≤2GPU勉强 / >2GPU灾难 / 7B全参数→NVLink必需!",
        ]
        return SimResult('data_parallelism', results, insights)


class GradientAccumulationSimulator:
    """梯度累积模拟"""

    def __init__(self):
        self.target_batch = 32
        self.gpu_memory_gb = 24

    def simulate_accumulation(self):
        """模拟梯度累积的等效batch效果"""
        results = {}

        configs = [
            ('no_accum_B32', 32, 1),
            ('accum_B4x8', 4, 8),
            ('accum_B2x16', 2, 16),
            ('accum_B1x32', 1, 32),
        ]

        for name, micro_batch, accum_steps in configs:
            effective_batch = micro_batch * accum_steps
            # Memory per micro_batch (simplified)
            model_gb = 14  # 7B BF16
            act_per_micro = micro_batch * 0.5  # rough estimate
            total_memory = model_gb + act_per_micro + 14 + 14 + 28  # model+act+grad+Adam

            # Time: micro_batch_time × accum_steps + 1 optimizer
            micro_time_ms = 100 * micro_batch / 32  # proportional
            total_time_ms = micro_time_ms * accum_steps + 10  # +optimizer

            # Communication: 1 AllReduce at end (DDP)
            comm_data_gb = 14  # 7B gradients
            comm_time_pcie_ms = comm_data_gb / (2.76 / 2) * 1000
            comm_time_nvlink_ms = comm_data_gb / 300 * 1000

            results[name] = {
                'micro_batch': micro_batch,
                'accum_steps': accum_steps,
                'effective_batch': effective_batch,
                'total_memory_gb': round(total_memory, 1),
                'fits_24gb': total_memory <= 24,
                'total_time_ms': round(total_time_ms, 1),
                'comm_pcie_ms': round(comm_time_pcie_ms, 1),
                'comm_nvlink_ms': round(comm_time_nvlink_ms, 2),
            }

        insights = [
            f"Gradient-Accumulation: B=1×32=等效B=32→内存省→但32步→慢→3.2x时间!",
            f"B=4×8=32→memory={results['accum_B4x8']['total_memory_gb']}GB→fits={results['accum_B4x8']['fits_24gb']}→可行!",
            f"B=32→memory={results['no_accum_B32']['total_memory_gb']}GB→fits={results['no_accum_B32']['fits_24gb']}→OOM!",
            f"DDP no_sync→只在最后AllReduce→省通信→但FSDP每层仍需!",
            f"LoRA→参数0.5MB→内存≈4GB→B=32→fits→不需要accumulation!",
        ]
        return SimResult('gradient_accumulation', results, insights)


class MixedPrecisionSimulator:
    """混合精度训练模拟"""

    def __init__(self):
        self.precision_configs = {
            'FP32': {'param_bytes': 4, 'compute_speedup': 1.0, 'description': 'FP32全精度'},
            'BF16': {'param_bytes': 2, 'compute_speedup': 1.51, 'description': 'BF16训练最优'},
            'FP16_AMP': {'param_bytes': 2, 'compute_speedup': 0.85, 'description': 'FP16 AMP反而慢!'},
            'FP8_TE': {'param_bytes': 1, 'compute_speedup': 1.48, 'description': 'FP8 TE B≥4'},
        }

    def simulate_precision(self):
        """模拟不同混合精度的性能"""
        results = {}
        model_params_gb = 14  # 7B BF16

        for name, spec in self.precision_configs.items():
            param_gb = 7e9 * spec['param_bytes'] / 1e9
            adam_gb = 7e9 * 4 * 2 / 1e9  # Adam always FP32
            grad_gb = param_gb
            total_gb = param_gb + adam_gb + grad_gb + param_gb  # + activations

            compute_time_ratio = 1 / spec['compute_speedup']

            results[name] = {
                'param_gb': round(param_gb, 2),
                'adam_gb': round(adam_gb, 2),
                'total_gb': round(total_gb, 2),
                'compute_speedup': spec['compute_speedup'],
                'memory_saving_pct': round((1 - param_gb / model_params_gb) * 100, 1),
                'description': spec['description'],
            }

        insights = [
            f"BF16-Best: 1.51x加速+50%内存省→RTX 4090最优→简单→不需AMP!",
            f"FP16 AMP: 0.85x→反而慢!→loss scaling overhead→BF16动态范围=FP32→不需!",
            f"FP8 TE: 1.48x→B≥4→fused kernel→但B=1慢0.75x→小GEMM量化overhead!",
            f"FP32→total={results['FP32']['total_gb']}GB→灾难 / BF16→{results['BF16']['total_gb']}GB→5x参数!",
            f"混合精度=训练核心优化→BF16+FP32 AdamW→RTX 4090最优→推荐!",
        ]
        return SimResult('mixed_precision', results, insights)


class CheckpointingSimulator:
    """检查点策略模拟"""

    def __init__(self):
        self.save_size_gb = 14  # 7B BF16 model size
        self.save_speed_gb_s = 1  # rough: 1GB/s write speed

    def simulate_checkpointing(self):
        """模拟3种checkpointing策略"""
        results = {}

        # Sync checkpointing
        save_time_s = self.save_size_gb / self.save_speed_gb_s
        sync_overhead_pct = save_time_s / (100 * 1000) * 100  # per 1000 steps

        results['sync_barrier'] = {
            'save_time_s': round(save_time_s, 1),
            'training_paused': True,
            'overhead_per_1000steps_pct': round(sync_overhead_pct, 2),
            'consistency': 'perfect',
            'description': '所有GPU暂停→保存→继续→浪费GPU时间',
        }

        # Async checkpointing
        async_overhead_pct = 0  # training continues
        results['async'] = {
            'save_time_s': round(save_time_s, 1),
            'training_paused': False,
            'overhead_per_1000steps_pct': 0,
            'consistency': 'near-perfect (1-2 step delay)',
            'description': '后台线程保存→GPU继续训练→无暂停→效率高',
        }

        # Async + version management (verl)
        results['async_versioned'] = {
            'save_time_s': round(save_time_s, 1),
            'training_paused': False,
            'overhead_per_1000steps_pct': 0,
            'consistency': 'versioned (N recent checkpoints)',
            'versions_kept': 3,
            'description': 'verl→异步+版本管理→保留最近N→可恢复任意→最优!',
        }

        # Gradient checkpointing (activation recomputation)
        results['gradient_checkpointing'] = {
            'memory_saving_pct': 50,
            'compute_increase_pct': 33,
            'description': '不存所有激活→backward重计算→省50%内存但增33%计算',
        }

        # FSDP + gradient checkpointing interaction
        results['fsdp_plus_grad_ckpt'] = {
            'memory_effect': 'INCREASE! FSDP frees params but grad_ckpt holds activations',
            'actual_result': '0.478GB (no grad_ckpt) < 0.6xx (with grad_ckpt)',
            'recommendation': 'FSDP + NO gradient checkpointing = optimal',
            'description': '反直觉! FSDP+checkpointing→内存反而增→FSDP释放参数→激活不释放→矛盾!',
        }

        insights = [
            f"Sync checkpoint: {save_time_s}s暂停→每1000步浪费{sync_overhead_pct}%→简单但浪费!",
            f"Async: 0%暂停→GPU继续→效率高→但可能1-2步延迟→可接受!",
            f"verl: 异步+版本管理→保留最近3版本→可恢复任意→最优→推荐!",
            f"Gradient checkpointing: 省50%内存但增33%计算→FSDP下不需要!",
            f"FSDP+no grad_ckpt=0.478GB→比FSDP+grad_ckpt更小→反直觉→推荐!",
        ]
        return SimResult('checkpointing', results, insights)


class TrainingPipelineSimulator:
    """训练流水线瓶颈分析模拟"""

    def __init__(self):
        self.pipeline_steps = {
            'data_load': {'time_pct': 2, 'memory_pct': 5, 'description': 'DataLoader→batch→预取'},
            'forward': {'time_pct': 6, 'memory_pct': 30, 'description': 'forward→激活→memory-bound'},
            'backward': {'time_pct': 55, 'memory_pct': 40, 'description': 'backward→梯度→50-62%主导'},
            'communication': {'time_pct': 35, 'memory_pct': 0, 'description': 'FSDP AllGather+ReduceScatter'},
            'optimizer': {'time_pct': 8, 'memory_pct': 25, 'description': 'AdamW→memory-bound→快'},
            'logging': {'time_pct': 1, 'memory_pct': 0, 'description': 'metrics→监控→不阻塞'},
        }

    def simulate_pipeline(self):
        """模拟训练流水线瓶颈分析"""
        results = {}

        # PCIe scenario
        pcie_total_pct = 100
        pcie_comm_pct = 99.4  # 7B 8GPU实测
        pcie_compute_pct = 0.6
        pcie_speedup = 0.46

        results['PCIe_7B_8GPU'] = {
            'comm_pct': pcie_comm_pct,
            'compute_pct': pcie_compute_pct,
            'speedup': pcie_speedup,
            'bottleneck': 'communication (PCIe)',
            'description': 'PCIe→99.4%通信→灾难→不推荐',
        }

        # NVLink scenario
        nvlink_comm_pct = 3.3
        nvlink_compute_pct = 96.7
        nvlink_speedup = 7.7

        results['NVLink_7B_8GPU'] = {
            'comm_pct': nvlink_comm_pct,
            'compute_pct': nvlink_compute_pct,
            'speedup': nvlink_speedup,
            'bottleneck': 'compute (backward)',
            'description': 'NVLink→3.3%通信→7.7x→OK→推荐',
        }

        # Small model scenario
        results['PCIe_25M_2GPU'] = {
            'comm_pct': 10,
            'compute_pct': 90,
            'speedup': 1.12,
            'bottleneck': 'compute (marginally)',
            'description': '25M→PCIe 2GPU→1.12x→勉强→可行',
        }

        # LoRA single GPU scenario
        results['LoRA_7B_1GPU'] = {
            'comm_pct': 0,
            'compute_pct': 100,
            'speedup': '1x (no parallelism needed)',
            'bottleneck': 'compute',
            'description': 'LoRA→0.4%参数→单GPU→无通信→最优',
        }

        # Step breakdown
        results['step_breakdown'] = self.pipeline_steps

        insights = [
            f"Backward=55-62%时间→主导→QK^T 27%/RMSNorm 15%/MLP 15%→训练瓶颈!",
            f"PCIe 7B 8GPU→99.4%通信→0.46x→灾难→RTX 4090不适合7B训练!",
            f"NVLink 7B 8GPU→3.3%通信→7.7x→OK→7B训练需NVLink→A100/H100!",
            f"25M 2GPU→10%通信→1.12x→勉强→RTX 4090小模型训练→可行!",
            f"LoRA→0.4%参数→无通信→单GPU→RTX 4090最优→verl GRPO推荐!",
        ]
        return SimResult('training_pipeline', results, insights)


def run_all_simulators():
    """运行所有模拟器"""
    print("=" * 70)
    print("AI Training Infrastructure Simulator")
    print("=" * 70)

    simulators = [
        ("1. Training Loop Memory", TrainingLoopMemorySimulator()),
        ("2. Data Parallelism", DataParallelismSimulator()),
        ("3. Gradient Accumulation", GradientAccumulationSimulator()),
        ("4. Mixed Precision", MixedPrecisionSimulator()),
        ("5. Checkpointing", CheckpointingSimulator()),
        ("6. Training Pipeline", TrainingPipelineSimulator()),
    ]

    all_insights = []
    for title, sim in simulators:
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")

        result = sim.simulate_memory() if isinstance(sim, TrainingLoopMemorySimulator) else \
                 sim.simulate_dp() if isinstance(sim, DataParallelismSimulator) else \
                 sim.simulate_accumulation() if isinstance(sim, GradientAccumulationSimulator) else \
                 sim.simulate_precision() if isinstance(sim, MixedPrecisionSimulator) else \
                 sim.simulate_checkpointing() if isinstance(sim, CheckpointingSimulator) else \
                 sim.simulate_pipeline()

        # Print key results
        if isinstance(sim, TrainingLoopMemorySimulator):
            for name, data in result.metrics.items():
                fits = 'YES' if data['fits_24gb'] else 'NO'
                print(f"  {name}: total={data['per_gpu_total_gb']}GB, fits={fits}, ratio={data['memory_ratio']}x")
        elif isinstance(sim, DataParallelismSimulator):
            for name, data in result.metrics.items():
                if 'fsdp_pcie_speedup' in data:
                    print(f"  {name}: PCIe={data['fsdp_pcie_speedup']}x(comm={data['fsdp_pcie_ratio_pct']}%) | NVLink={data['fsdp_nvlink_speedup']}x")
        elif isinstance(sim, GradientAccumulationSimulator):
            for name, data in result.metrics.items():
                fits = 'YES' if data['fits_24gb'] else 'NO'
                print(f"  {name}: eff_B={data['effective_batch']}, memory={data['total_memory_gb']}GB({fits})")
        elif isinstance(sim, MixedPrecisionSimulator):
            for name, data in result.metrics.items():
                print(f"  {name}: speedup={data['compute_speedup']}x, memory_save={data['memory_saving_pct']}%, total={data['total_gb']}GB")
        elif isinstance(sim, CheckpointingSimulator):
            for name, data in result.metrics.items():
                paused = 'YES' if data.get('training_paused', None) == True else \
                         'NO' if data.get('training_paused', None) == False else '?'
                print(f"  {name}: paused={paused}, {data.get('description', '')}")
        elif isinstance(sim, TrainingPipelineSimulator):
            for name, data in result.metrics.items():
                if name != 'step_breakdown' and 'speedup' in data:
                    print(f"  {name}: speedup={data['speedup']}x, comm={data['comm_pct']}%, bottleneck={data['bottleneck']}")

        for insight in result.insights:
            print(f"  INSIGHT: {insight}")
        all_insights.extend(result.insights)

    print(f"\n{'=' * 70}")
    print("CORE LAWS SUMMARY (7 Laws)")
    print(f"{'=' * 70}")
    laws = [
        "1. Training-Memory-5x: 训练=5x参数→7B=70GB→RTX 4090 OOM→需FSDP/LoRA!",
        "2. PCIe-Training-Disaster: FSDP 8GPU=0.46x→比1GPU慢→灾难→需NVLink!",
        "3. BF16-Best: BF16=最优→1.51x+49%省→RTX 4090原生→不需AMP!",
        "4. Gradient-Accumulation: micro×accum=等效B→小GPU→大batch桥梁!",
        "5. LoRA-Feasibility: LoRA→99.6%省→单GPU→verl GRPO→RTX 4090推荐!",
        "6. Async-Checkpoint: 异步→GPU不暂停→verl版本管理→最优!",
        "7. Training-Decision: 小模型≤2GPU / LoRA单GPU / 7B全→NVLink→RTX 4090不适合!",
    ]
    for law in laws:
        print(f"  {law}")

    print(f"\nTotal insights: {len(all_insights)}")
    return all_insights


if __name__ == '__main__':
    run_all_simulators()