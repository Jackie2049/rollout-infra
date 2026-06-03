"""FSDP (Fully Sharded Data Parallel) 内存模拟器

模拟 PyTorch FSDP 的内存行为:
1. DDP vs FSDP 显存对比: 逐项分解参数/梯度/优化器/激活值
2. Sharding 策略对比: FULL_SHARD / SHARD_GRAD_OP / NO_SHARD
3. Prefetching 通信分析: AllGather/ReduceScatter 开销
4. Wrapping 策略影响: 不同粒度下的通信量
5. 多 GPU 扩展: 显存随 GPU 数量的变化

使用方法:
    python fsdp_memory_sim.py   # CPU 可运行
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ModelConfig:
    """模型配置。"""
    name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    head_dim: int
    ffn_dim: int
    vocab_size: int
    seq_len: int
    batch_size: int
    dtype_bytes: int = 2  # FP16/BF16


@dataclass
class GPUConfig:
    """GPU 配置。"""
    name: str
    memory_gb: float
    memory_bw_gb_s: float  # GB/s
    nvlink_bw_gb_s: float  # GB/s per link (bidirectional)
    num_nvlinks: int = 0
    network_bw_gb_s: float = 0  # Inter-GPU network BW (for non-NVLink)


def calc_model_params(model: ModelConfig) -> dict:
    """计算模型各部分参数量。"""
    h = model.hidden_size
    n = model.num_heads
    d = model.head_dim
    ff = model.ffn_dim
    v = model.vocab_size
    L = model.num_layers

    # Per layer
    # Attention: QKV projections + output projection
    qkv_params = h * (3 * d * n)  # Q, K, V projections
    out_params = h * h  # Output projection
    attn_params = qkv_params + out_params

    # FFN: gate + up + down (SwiGLU style)
    ffn_params = h * ff * 3  # gate, up, down projections

    # LayerNorm: 2 per layer (pre-attn + pre-ffn)
    ln_params = h * 2

    # Layer total
    layer_params = attn_params + ffn_params + ln_params

    # Non-layer params
    embedding_params = v * h  # Token embedding
    lm_head_params = v * h  # LM Head (may share with embedding)

    total = L * layer_params + embedding_params + lm_head_params

    return {
        'total': total,
        'per_layer': layer_params,
        'attn_per_layer': attn_params,
        'ffn_per_layer': ffn_params,
        'ln_per_layer': ln_params,
        'embedding': embedding_params,
        'lm_head': lm_head_params,
        'layer_count': L,
    }


def calc_memory_ddp(model: ModelConfig, num_gpus: int) -> dict:
    """DDP 内存分析: 每张 GPU 都保存完整副本。"""
    params = calc_model_params(model)
    B = model.dtype_bytes
    total_params = params['total']

    # FP32 master weights (mixed precision)
    fp32_params_bytes = total_params * 4

    # FP16 model parameters
    fp16_params_bytes = total_params * B

    # Gradients (FP16)
    grads_bytes = total_params * B

    # Optimizer states (AdamW: momentum + variance, FP32)
    optimizer_bytes = total_params * 4 * 2  # m + v

    # Activations (approximate)
    # Per layer: need to store activations for backward
    # Input to each layer: batch * seq * hidden
    activation_per_layer = model.batch_size * model.seq_len * model.hidden_size * B
    # Rough: store ~2x for attention (Q, K, V, attn_weights, etc.)
    activation_per_layer_total = activation_per_layer * 3
    total_activations = activation_per_layer_total * model.num_layers

    total_bytes = fp16_params_bytes + grads_bytes + optimizer_bytes + total_activations

    # DDP: each GPU has full copy, plus gradient AllReduce buffer
    per_gpu = total_bytes + fp32_params_bytes

    return {
        'strategy': 'DDP',
        'fp32_master': fp32_params_bytes,
        'fp16_params': fp16_params_bytes,
        'gradients': grads_bytes,
        'optimizer': optimizer_bytes,
        'activations': total_activations,
        'total_per_gpu': per_gpu,
        'total_per_gpu_gb': per_gpu / 1e9,
        'num_gpus': num_gpus,
        'gradient_allreduce_mb': fp16_params_bytes / 1e6,  # AllReduce gradient
    }


def calc_memory_fsdp(model: ModelConfig, num_gpus: int, strategy: str = "full_shard") -> dict:
    """FSDP 内存分析: 参数分片到各 GPU。"""
    params = calc_model_params(model)
    B = model.dtype_bytes
    total_params = params['total']

    # Sharding factors
    if strategy == "full_shard":
        # ZeRO-3: shard params, grads, optimizer states
        shard_factor = num_gpus
        param_shard = total_params // num_gpus
        grad_shard = total_params // num_gpus
        opt_shard = total_params // num_gpus
    elif strategy == "shard_grad_op":
        # ZeRO-2: replicate params, shard grads and optimizer
        param_shard = total_params  # Full params replicated
        grad_shard = total_params // num_gpus
        opt_shard = total_params // num_gpus
    else:  # no_shard
        # ZeRO-1/0: replicate everything
        param_shard = total_params
        grad_shard = total_params
        opt_shard = total_params

    # Per-GPU memory
    fp32_master_bytes = opt_shard * 4  # Only sharded portion
    fp16_params_bytes = param_shard * B  # Sharded or replicated
    grads_bytes = grad_shard * B  # Sharded or replicated
    optimizer_bytes = opt_shard * 4 * 2  # m + v (sharded)

    # Activations (not sharded, each GPU processes its own micro-batch)
    activation_per_layer = model.batch_size * model.seq_len * model.hidden_size * B
    total_activations = activation_per_layer * 3 * model.num_layers

    # Unshard buffer: during forward/backward, need to AllGather full params
    if strategy == "full_shard":
        unshard_buffer = params['per_layer'] * B  # One layer at a time
    elif strategy == "shard_grad_op":
        unshard_buffer = 0  # Params always full, no need to unshard
    else:
        unshard_buffer = 0

    per_gpu = fp16_params_bytes + grads_bytes + optimizer_bytes + total_activations + unshard_buffer

    # Communication: AllGather + ReduceScatter per layer
    if strategy == "full_shard":
        # Forward: AllGather params per layer (2x: unshard before, reshard after)
        # Backward: AllGather params + ReduceScatter grads
        # Per layer: 2 AllGather (fwd+bwd) + 1 ReduceScatter (bwd)
        allgather_per_layer = params['per_layer'] * B / 1e6  # MB
        reducescatter_per_layer = params['per_layer'] * B / 1e6  # MB
        total_allgather = allgather_per_layer * 2 * model.num_layers
        total_reducescatter = reducescatter_per_layer * model.num_layers
    elif strategy == "shard_grad_op":
        allgather_per_layer = 0
        reducescatter_per_layer = params['per_layer'] * B / 1e6
        total_allgather = 0
        total_reducescatter = reducescatter_per_layer * model.num_layers
    else:
        total_allgather = 0
        total_reducescatter = 0
        allgather_per_layer = 0
        reducescatter_per_layer = 0

    return {
        'strategy': strategy,
        'fp16_params_per_gpu': fp16_params_bytes,
        'gradients_per_gpu': grads_bytes,
        'optimizer_per_gpu': optimizer_bytes,
        'activations': total_activations,
        'unshard_buffer': unshard_buffer,
        'total_per_gpu': per_gpu,
        'total_per_gpu_gb': per_gpu / 1e9,
        'num_gpus': num_gpus,
        'shard_factor': num_gpus if strategy == "full_shard" else 1,
        'allgather_total_mb': total_allgather,
        'reducescatter_total_mb': total_reducescatter,
        'allgather_per_layer_mb': allgather_per_layer,
        'reducescatter_per_layer_mb': reducescatter_per_layer,
    }


def analyze_wrapping_impact(model: ModelConfig, num_gpus: int) -> dict:
    """分析不同 wrapping 策略对通信的影响。"""
    params = calc_model_params(model)
    B = model.dtype_bytes

    # Wrapping granularity determines unshard/reshard frequency
    wrap_strategies = {
        'per_layer': {
            'description': '每个 Transformer 层一个 FSDP unit',
            'units': model.num_layers,
            'params_per_unit': params['per_layer'],
        },
        'per_attn_ffn': {
            'description': 'Attention 和 FFN 分别包装',
            'units': model.num_layers * 2,
            'params_per_unit': max(params['attn_per_layer'], params['ffn_per_layer']),
        },
        'per_sublayer': {
            'description': '每个 QKV/Out/Gate/Up/Down 分别包装',
            'units': model.num_layers * 6,
            'params_per_unit': max(
                model.hidden_size * model.hidden_size,  # QKV approx
                model.hidden_size * model.ffn_dim,  # FFN approx
            ),
        },
        'entire_model': {
            'description': '整个模型一个 FSDP unit',
            'units': 1,
            'params_per_unit': params['total'],
        },
    }

    results = {}
    for name, cfg in wrap_strategies.items():
        # Each unit requires AllGather before compute and ReduceScatter after
        # Total AllGather: 2 per unit (fwd + bwd), ReduceScatter: 1 per unit (bwd)
        allgather_bytes = cfg['params_per_unit'] * B
        reducescatter_bytes = cfg['params_per_unit'] * B

        total_allgather_mb = allgather_bytes * 2 * cfg['units'] / 1e6
        total_reducescatter_mb = reducescatter_bytes * cfg['units'] / 1e6

        # Peak memory: need to hold one full unit's params
        peak_unshard_mb = allgather_bytes / 1e6

        # Number of communication events
        comm_events = cfg['units'] * 3  # 2 AllGather + 1 ReduceScatter per unit

        results[name] = {
            'description': cfg['description'],
            'units': cfg['units'],
            'allgather_total_mb': total_allgather_mb,
            'reducescatter_total_mb': total_reducescatter_mb,
            'total_comm_mb': total_allgather_mb + total_reducescatter_mb,
            'peak_unshard_mb': peak_unshard_mb,
            'comm_events': comm_events,
        }

    return results


def simulate_prefetch_overlap(model: ModelConfig, num_gpus: int, gpu: GPUConfig):
    """模拟 FSDP prefetching 的通信计算重叠效果。"""
    params = calc_model_params(model)
    B = model.dtype_bytes

    # Per-layer communication and computation
    layer_params_bytes = params['per_layer'] * B

    # AllGather: ring algorithm, effective BW = raw BW * (N-1)/N per GPU
    # In practice, AllGather BW is much less than raw link BW
    if gpu.num_nvlinks > 0:
        raw_bw = gpu.nvlink_bw_gb_s * gpu.num_nvlinks
    elif gpu.network_bw_gb_s > 0:
        raw_bw = gpu.network_bw_gb_s
    else:
        raw_bw = gpu.memory_bw_gb_s * 0.1  # Fallback: PCIe
    effective_bw = raw_bw * 0.5  # Protocol overhead, ring tax, contention
    allgather_time_ms = (layer_params_bytes / 1e6) / (effective_bw * 1000)  # ms

    # Computation per layer (rough: attn 4 matmuls + FFN 3 matmuls)
    h = model.hidden_size
    s = model.seq_len
    b = model.batch_size

    # Attention: 4 matmuls of (b*s, h) x (h, h) → 4 * 2 * b * s * h^2
    attn_flops = 4 * 2 * b * s * h * h
    # FFN (SwiGLU): gate + up + down → 3 * 2 * b * s * h * ffn_dim
    ffn_flops = 3 * 2 * b * s * h * model.ffn_dim
    compute_flops = attn_flops + ffn_flops

    # Effective GPU throughput (conservative 50% of peak)
    gpu_tflops = {
        'a100': 312 * 0.5,   # 156 TFLOPS effective
        'h100': 990 * 0.5,   # 495 TFLOPS effective
        'a16':  22 * 0.5,    # 11 TFLOPS effective
    }
    tflops = gpu_tflops.get(gpu.name, 100)
    compute_time_ms = compute_flops / (tflops * 1e12) * 1000  # ms

    # Without prefetch: sequential
    time_no_prefetch = (allgather_time_ms + compute_time_ms) * model.num_layers

    # With prefetch: overlap communication of layer N+1 with compute of layer N
    # Only the first layer's AllGather is exposed
    time_with_prefetch = allgather_time_ms + compute_time_ms * model.num_layers

    overlap_ratio = time_no_prefetch / time_with_prefetch if time_with_prefetch > 0 else 1.0

    return {
        'allgather_per_layer_ms': allgather_time_ms,
        'compute_per_layer_ms': compute_time_ms,
        'comm_compute_ratio': allgather_time_ms / compute_time_ms if compute_time_ms > 0 else 0,
        'time_no_prefetch_ms': time_no_prefetch,
        'time_with_prefetch_ms': time_with_prefetch,
        'speedup': overlap_ratio,
        'effective_bw_gbps': effective_bw,
    }


# ============================================================
# Predefined configs
# ============================================================

MODELS = {
    'gpt2-small': ModelConfig('GPT-2 Small', 12, 768, 12, 64, 3072, 50257, 1024, 8),
    'llama-7b': ModelConfig('LLaMA-7B', 32, 4096, 32, 128, 11008, 32000, 2048, 4),
    'llama-13b': ModelConfig('LLaMA-13B', 40, 5120, 40, 128, 13824, 32000, 2048, 4),
    'llama-70b': ModelConfig('LLaMA-70B', 80, 8192, 64, 128, 28672, 32000, 2048, 2),
}

GPUS = {
    'a100': GPUConfig('A100 80GB', 80, 2039, 50, 12),
    'h100': GPUConfig('H100 80GB', 80, 3350, 100, 18),
    'a100_eth': GPUConfig('A100 (Ethernet)', 80, 2039, 0, 0, 1.25),  # 10Gbps Ethernet
    'a100_roce': GPUConfig('A100 (RoCE/IB)', 80, 2039, 0, 0, 25.0),  # 200Gbps InfiniBand
    'a16': GPUConfig('A16 15GB', 15, 157, 0, 0, 2.0),  # PCIe ~16GB/s
}

PREDEFINED_MODELS = [
    ('gpt2-small', 'a100'),
    ('llama-7b', 'a100'),
    ('llama-13b', 'a100'),
    ('llama-70b', 'a100'),
]


def main():
    print("=" * 60)
    print("FSDP (Fully Sharded Data Parallel) 内存模拟器")
    print("=" * 60)

    # ============================================================
    # 实验 1: DDP vs FSDP 显存对比
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 1: DDP vs FSDP 显存对比 (LLaMA-7B, 4x A100)")
    print("=" * 60)

    model = MODELS['llama-7b']
    gpu = GPUS['a100']
    num_gpus = 4

    params = calc_model_params(model)
    print(f"\n  模型: {model.name}")
    print(f"    Layers: {model.num_layers}, Hidden: {model.hidden_size}, Heads: {model.num_heads}")
    print(f"    FFN dim: {model.ffn_dim}, Vocab: {model.vocab_size}")
    print(f"    总参数: {params['total']/1e9:.2f}B ({params['total']/1e6:.0f}M)")
    print(f"    每层参数: {params['per_layer']/1e6:.1f}M")
    print(f"    Batch: {model.batch_size}, SeqLen: {model.seq_len}")

    ddp = calc_memory_ddp(model, num_gpus)
    fsdp_full = calc_memory_fsdp(model, num_gpus, "full_shard")
    fsdp_grad = calc_memory_fsdp(model, num_gpus, "shard_grad_op")
    fsdp_none = calc_memory_fsdp(model, num_gpus, "no_shard")

    print(f"\n  {'项目':<20} {'DDP':>10} {'FSDP-full':>10} {'FSDP-grad':>10} {'FSDP-none':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    def fmt_gb(bytes_val):
        return f"{bytes_val/1e9:.2f}GB"

    print(f"  {'FP16 参数':<20} {fmt_gb(ddp['fp16_params']):>10} "
          f"{fmt_gb(fsdp_full['fp16_params_per_gpu']):>10} "
          f"{fmt_gb(fsdp_grad['fp16_params_per_gpu']):>10} "
          f"{fmt_gb(fsdp_none['fp16_params_per_gpu']):>10}")

    print(f"  {'梯度':<20} {fmt_gb(ddp['gradients']):>10} "
          f"{fmt_gb(fsdp_full['gradients_per_gpu']):>10} "
          f"{fmt_gb(fsdp_grad['gradients_per_gpu']):>10} "
          f"{fmt_gb(fsdp_none['gradients_per_gpu']):>10}")

    print(f"  {'优化器 (AdamW)':<20} {fmt_gb(ddp['optimizer']):>10} "
          f"{fmt_gb(fsdp_full['optimizer_per_gpu']):>10} "
          f"{fmt_gb(fsdp_grad['optimizer_per_gpu']):>10} "
          f"{fmt_gb(fsdp_none['optimizer_per_gpu']):>10}")

    print(f"  {'激活值':<20} {fmt_gb(ddp['activations']):>10} "
          f"{fmt_gb(fsdp_full['activations']):>10} "
          f"{fmt_gb(fsdp_grad['activations']):>10} "
          f"{fmt_gb(fsdp_none['activations']):>10}")

    print(f"  {'Unshard 缓冲':<20} {'N/A':>10} "
          f"{fmt_gb(fsdp_full['unshard_buffer']):>10} "
          f"{fmt_gb(fsdp_grad['unshard_buffer']):>10} "
          f"{fmt_gb(fsdp_none['unshard_buffer']):>10}")

    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'总计/GPU':<20} {ddp['total_per_gpu_gb']:>9.2f}GB "
          f"{fsdp_full['total_per_gpu_gb']:>9.2f}GB "
          f"{fsdp_grad['total_per_gpu_gb']:>9.2f}GB "
          f"{fsdp_none['total_per_gpu_gb']:>9.2f}GB")

    print(f"\n  显存节省: FSDP-full {(1-fsdp_full['total_per_gpu_gb']/ddp['total_per_gpu_gb'])*100:.1f}% "
          f"| FSDP-grad {(1-fsdp_grad['total_per_gpu_gb']/ddp['total_per_gpu_gb'])*100:.1f}%")

    # ============================================================
    # 实验 2: 不同模型的 DDP vs FSDP 对比
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 2: 不同模型规模的显存需求 (4x A100)")
    print("=" * 60)

    print(f"\n  {'模型':<15} {'参数量':>10} {'DDP':>10} {'FSDP-full':>10} {'节省':>8} {'可训练':>8}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    for model_name, gpu_name in PREDEFINED_MODELS:
        m = MODELS[model_name]
        g = GPUS[gpu_name]
        p = calc_model_params(m)

        d = calc_memory_ddp(m, num_gpus)
        f = calc_memory_fsdp(m, num_gpus, "full_shard")

        ddp_fit = "Y" if d['total_per_gpu_gb'] <= g.memory_gb else "N"
        fsdp_fit = "Y" if f['total_per_gpu_gb'] <= g.memory_gb else "N"

        saving = (1 - f['total_per_gpu_gb'] / d['total_per_gpu_gb']) * 100
        print(f"  {m.name:<15} {p['total']/1e9:>9.2f}B "
              f"{d['total_per_gpu_gb']:>9.2f}GB "
              f"{f['total_per_gpu_gb']:>9.2f}GB "
              f"{saving:>7.1f}% "
              f"DDP:{ddp_fit} FSDP:{fsdp_fit}")

    # ============================================================
    # 实验 3: Sharding 策略与 GPU 数量关系
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 3: GPU 数量对 FSDP 显存的影响 (LLaMA-70B)")
    print("=" * 60)

    model_70b = MODELS['llama-70b']
    print(f"\n  模型: {model_70b.name}, {calc_model_params(model_70b)['total']/1e9:.1f}B 参数")

    print(f"\n  {'GPU数':>6} {'DDP(GPU)':>10} {'Full(GPU)':>10} {'Grad(GPU)':>10} {'Full节省':>8} {'AllGather':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    for ng in [1, 2, 4, 8, 16, 32, 64]:
        d = calc_memory_ddp(model_70b, ng)
        f = calc_memory_fsdp(model_70b, ng, "full_shard")
        g = calc_memory_fsdp(model_70b, ng, "shard_grad_op")

        saving = (1 - f['total_per_gpu_gb'] / d['total_per_gpu_gb']) * 100
        print(f"  {ng:>6} {d['total_per_gpu_gb']:>9.2f}GB "
              f"{f['total_per_gpu_gb']:>9.2f}GB "
              f"{g['total_per_gpu_gb']:>9.2f}GB "
              f"{saving:>7.1f}% "
              f"{f['allgather_total_mb']:>9.0f}MB")

    # ============================================================
    # 实验 4: Wrapping 策略分析
    # ============================================================
    print("\n" + "=" * 60)
    print("实验 4: Wrapping 策略对通信的影响 (LLaMA-7B, 4 GPUs)")
    print("=" * 60)

    model_7b = MODELS['llama-7b']
    wrapping = analyze_wrapping_impact(model_7b, 4)

    print(f"\n  {'策略':<15} {'Units':>6} {'AllGather':>12} {'ReduceScat':>12} {'总计':>10} {'峰值缓冲':>10} {'事件数':>8}")
    print(f"  {'-'*15} {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")

    for name, info in wrapping.items():
        print(f"  {name:<15} {info['units']:>6} "
              f"{info['allgather_total_mb']:>10.0f}MB "
              f"{info['reducescatter_total_mb']:>10.0f}MB "
              f"{info['total_comm_mb']:>8.0f}MB "
              f"{info['peak_unshard_mb']:>8.0f}MB "
              f"{info['comm_events']:>8}")

    print(f"""
关键洞察:
  - per_layer: 最常用策略，通信量与层数成正比，峰值缓冲适中
  - per_attn_ffn: 更细粒度，通信量增加但每次通信更小，利于重叠
  - per_sublayer: 最细粒度，通信次数最多，但每次通信最小
  - entire_model: 通信量最少但峰值缓冲最大（需一次性 AllGather 全部参数）
  → 推荐: transformer_auto_wrap_policy (per Transformer layer)
    """)

    # ============================================================
    # 实验 5: Prefetching 通信计算重叠
    # ============================================================
    print("=" * 60)
    print("实验 5: Prefetching 通信计算重叠分析")
    print("=" * 60)

    # LLaMA-7B on different interconnects
    print(f"\n  模型: {model_7b.name}, 4x GPU")
    prefetch = simulate_prefetch_overlap(model_7b, 4, GPUS['a100'])

    print(f"\n  每层通信时间 (AllGather): {prefetch['allgather_per_layer_ms']:.3f} ms")
    print(f"  每层计算时间:             {prefetch['compute_per_layer_ms']:.3f} ms")
    print(f"  通信/计算比:              {prefetch['comm_compute_ratio']:.4f}")
    print(f"\n  无 Prefetch 总时间:  {prefetch['time_no_prefetch_ms']:.1f} ms")
    print(f"  有 Prefetch 总时间:  {prefetch['time_with_prefetch_ms']:.1f} ms")
    print(f"  Prefetch 加速:      {prefetch['speedup']:.2f}x")

    # 不同互联 + 模型对比
    print(f"\n  不同互联环境下的 Prefetch 效果 (LLaMA-7B / LLaMA-70B):")
    print(f"  {'互联':<20} {'有效带宽':>10} {'7B通信(ms)':>12} {'7B计算(ms)':>12} {'7B加速':>8} {'70B加速':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12} {'-'*8} {'-'*8}")

    model_70b = MODELS['llama-70b']
    for gpu_key in ['a100', 'h100', 'a100_roce', 'a100_eth', 'a16']:
        g = GPUS[gpu_key]
        pf_7b = simulate_prefetch_overlap(model_7b, 4, g)
        pf_70b = simulate_prefetch_overlap(model_70b, 4, g)
        print(f"  {g.name:<20} {pf_7b['effective_bw_gbps']:>8.0f}GB/s "
              f"{pf_7b['allgather_per_layer_ms']:>10.3f} "
              f"{pf_7b['compute_per_layer_ms']:>10.3f} "
              f"{pf_7b['speedup']:>7.2f}x "
              f"{pf_70b['speedup']:>7.2f}x")

    print(f"""
关键洞察:
  - NVLink (A100/H100): 通信时间 << 计算时间，Prefetch 收益小
  - Ethernet (10Gbps): 通信时间 > 计算时间，Prefetch 至关重要
  - 大模型 (70B): 每层参数更多，AllGather 数据量更大，Prefetch 收益更明显
  - 无 NVLink (A16 PCIe): AllGather 带宽受限，通信开销更大
    """)

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
FSDP (Fully Sharded Data Parallel) 核心知识:

1. 与 ZeRO 的关系:
   - FULL_SHARD ≈ ZeRO-3: 分片参数+梯度+优化器
   - SHARD_GRAD_OP ≈ ZeRO-2: 分片梯度+优化器 (参数完整复制)
   - NO_SHARD ≈ ZeRO-0: 不分片 (等同 DDP)

2. 显存节省:
   - FULL_SHARD: 每项都按 GPU 数量分片，显存节省最大
   - 代价: 每次 forward/backward 需要 AllGather 参数
   - 大模型 (70B+): FSDP 是在有限 GPU 上训练的唯一选择

3. 通信模式:
   - Forward: AllGather 参数 → 计算 → 丢弃 (reshard)
   - Backward: AllGather 参数 → 计算梯度 → ReduceScatter 梯度
   - 通信量 ∝ 模型参数量 × 层数

4. Prefetching:
   - 在计算第 N 层时，预先 AllGather 第 N+1 层参数
   - 当计算时间 >> 通信时间时，通信开销可被完全隐藏
   - NVLink (600GB/s bidirectional) 是关键

5. Wrapping 策略:
   - 粒度越细: 通信次数越多，但每次通信量越小
   - 粒度越粗: 通信次数越少，但峰值缓冲越大
   - 推荐: transformer_auto_wrap_policy (按 Transformer 层包装)
    """)


if __name__ == "__main__":
    main()
