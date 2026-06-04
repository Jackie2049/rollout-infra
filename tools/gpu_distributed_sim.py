#!/usr/bin/env python3
"""分布式训练模拟实验 — 单 GPU 模拟 Megatron-LM 风格的 TP/PP

在单 GPU 上模拟多卡分布式训练的关键操作:
1. Tensor Parallelism: ColumnParallel + RowParallel 的正确性和性能
2. Pipeline Parallelism: 1F1B 调度模拟
3. Gradient Accumulation: 微批次累积
4. ZeRO 优化器: 内存节省验证
5. AllReduce 模拟: 通信开销建模

用法 (GPU 服务器):
  source /root/miniconda3/bin/activate myconda
  export HF_HUB_OFFLINE=1
  python gpu_distributed_sim.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

GPU_NAME = torch.cuda.get_device_name(0)
TOTAL_MEM = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU_NAME}, VRAM: {TOTAL_MEM:.1f} GB")
print(f"CUDA: {torch.version.cuda}, Compute: {torch.cuda.get_device_capability()}")


def bench_ms(fn, warmup=5, rep=50):
    """CUDA Events timing"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


# ============================================================
# 实验 1: Tensor Parallelism 正确性 + 性能
# ============================================================

def exp1_tensor_parallel():
    print("\n" + "=" * 60)
    print("实验1: Tensor Parallelism 模拟")
    print("=" * 60)

    results = []
    B, S, H = 4, 512, 1024

    for tp_size in [1, 2, 4, 8]:
        x = torch.randn(B, S, H, device="cuda", dtype=torch.float16)

        # === ColumnParallelLinear 模拟 ===
        W_col_chunks = [torch.randn(H, H // tp_size, device="cuda", dtype=torch.float16)
                        for _ in range(tp_size)]

        # TP=1 baseline
        if tp_size == 1:
            W_full = torch.cat(W_col_chunks, dim=1)

        def column_parallel():
            outputs = [x @ chunk for chunk in W_col_chunks]
            return torch.cat(outputs, dim=-1)

        def column_baseline():
            return x @ W_full

        ms_tp = bench_ms(column_parallel)
        ms_base = bench_ms(column_baseline)

        out_tp = column_parallel()
        out_base = column_baseline()
        max_err = (out_tp - out_base).abs().max().item()

        # === RowParallelLinear 模拟 ===
        x_chunks = list(x.chunk(tp_size, dim=-1))
        W_row_chunks = [torch.randn(H // tp_size, H, device="cuda", dtype=torch.float16)
                        for _ in range(tp_size)]

        def row_parallel():
            partial = [xc @ wc for xc, wc in zip(x_chunks, W_row_chunks)]
            return sum(partial)  # AllReduce = sum

        ms_row = bench_ms(row_parallel)

        # Memory: TP rank 只需 1/tp_size 的权重
        weight_mem_mb = H * (H // tp_size) * 2 / 1e6  # fp16

        result = {
            "tp_size": tp_size,
            "column_ms": round(ms_tp, 4),
            "baseline_ms": round(ms_base, 4),
            "row_ms": round(ms_row, 4),
            "max_error": round(max_err, 6),
            "weight_mem_per_rank_mb": round(weight_mem_mb, 2),
            "memory_saving_pct": round((1 - 1/tp_size) * 100, 1),
        }
        results.append(result)

        print(f"\n  TP={tp_size}:")
        print(f"    ColumnParallel: {ms_tp:.4f}ms (vs baseline {ms_base:.4f}ms)")
        print(f"    RowParallel:    {ms_row:.4f}ms")
        print(f"    Max error:      {max_err:.6f}")
        print(f"    Weight/rank:    {weight_mem_mb:.1f} MB (saving {(1-1/tp_size)*100:.0f}%)")

    return results


# ============================================================
# 实验 2: MLP + Attention 完整 Transformer Block TP
# ============================================================

class SimpleTransformerBlock(nn.Module):
    """模拟 Megatron-LM 的 Transformer Block (无 TP)"""
    def __init__(self, hidden, n_heads, tp_size=1):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.tp_size = tp_size
        assert hidden % tp_size == 0

        h_per_tp = hidden // tp_size
        self.n_heads_per_tp = n_heads // tp_size

        # ColumnParallel: QKV projection
        self.qkv_proj = nn.Linear(hidden, 3 * h_per_tp, bias=False)
        # RowParallel: Output projection
        self.out_proj = nn.Linear(h_per_tp, hidden, bias=False)

        # MLP: ColumnParallel + RowParallel
        self.fc1 = nn.Linear(hidden, 4 * h_per_tp, bias=False)  # ColumnParallel
        self.fc2 = nn.Linear(4 * h_per_tp, hidden, bias=False)   # RowParallel

        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x):
        B, S, H = x.shape
        residual = x
        x = self.ln1(x)

        # Attention
        qkv = self.qkv_proj(x)  # [B, S, 3*H/TP]
        qkv = qkv.reshape(B, S, 3, self.n_heads_per_tp, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Reshape for attention: [B, n_heads_per_tp, S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, S, -1)

        # Output projection (模拟 AllReduce)
        x = self.out_proj(attn)
        if self.tp_size > 1:
            x = x  # 在真实 TP 中这里会 AllReduce

        x = residual + x
        residual = x
        x = self.ln2(x)

        # MLP
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)

        x = residual + x
        return x


def exp2_transformer_block():
    print("\n" + "=" * 60)
    print("实验2: Transformer Block TP 模拟")
    print("=" * 60)

    results = []
    B, S, H, n_heads = 4, 512, 512, 8

    for tp_size in [1, 2, 4]:
        model = SimpleTransformerBlock(H, n_heads, tp_size=tp_size).cuda().half()

        x = torch.randn(B, S, H, device="cuda", dtype=torch.float16)

        # Forward
        ms_fwd = bench_ms(lambda: model(x))

        # Forward + Backward
        def fwd_bwd():
            y = model(x)
            y.sum().backward()

        ms_fwdbwd = bench_ms(fwd_bwd, warmup=3, rep=20)
        model.zero_grad()

        # Memory
        param_mem = sum(p.numel() * 2 for p in model.parameters()) / 1e6  # MB fp16
        torch.cuda.reset_peak_memory_stats()
        _ = model(x)
        y = model(x)
        y.sum().backward()
        peak_mem = torch.cuda.max_memory_allocated() / 1e6  # MB

        result = {
            "tp_size": tp_size,
            "fwd_ms": round(ms_fwd, 4),
            "fwd_bwd_ms": round(ms_fwdbwd, 4),
            "params_mb": round(param_mem, 2),
            "peak_mem_mb": round(peak_mem, 2),
        }
        results.append(result)

        print(f"\n  TP={tp_size}:")
        print(f"    Forward:       {ms_fwd:.4f}ms")
        print(f"    Forward+Bwd:   {ms_fwdbwd:.4f}ms")
        print(f"    Params:        {param_mem:.1f} MB")
        print(f"    Peak Memory:   {peak_mem:.0f} MB")

        del model
        torch.cuda.empty_cache()

    return results


# ============================================================
# 实验 3: Pipeline Parallelism 1F1B 调度模拟
# ============================================================

def exp3_pipeline_parallel():
    print("\n" + "=" * 60)
    print("实验3: Pipeline Parallelism 1F1B 调度模拟")
    print("=" * 60)

    results = []

    for n_stages in [2, 4, 8]:
        n_microbatches = n_stages * 4  # 4x microbatches per stage
        H = 512

        # 模拟一个 stage 的计算时间
        stage_model = nn.Sequential(
            nn.Linear(H, 4 * H),
            nn.GELU(),
            nn.Linear(4 * H, H),
            nn.LayerNorm(H),
        ).cuda().half()

        x = torch.randn(1, 128, H, device="cuda", dtype=torch.float16)

        # 测量单个 stage 的 fwd+bwd 时间
        ms_fwd = bench_ms(lambda: stage_model(x))
        def fwd_bwd():
            y = stage_model(x)
            y.sum().backward()
        ms_fwdbwd = bench_ms(fwd_bwd, warmup=3, rep=20)

        # GPipe bubble ratio: (P-1) / (P-1+M)
        gpipe_bubble = (n_stages - 1) / (n_stages - 1 + n_microbatches)

        # 1F1B bubble ratio: same formula but memory is lower
        # Bubble = (P-1) / M
        of1b_bubble = (n_stages - 1) / n_microbatches

        # Interleaved 1F1B with V virtual stages
        V = 2  # common choice
        interleaved_bubble = of1b_bubble / V

        # Pipeline latency
        gpipe_latency = (n_microbatches + n_stages - 1) * ms_fwd * 2  # fwd+bwd per step
        of1b_latency = gpipe_latency  # same total compute

        # Peak activation memory (in microbatches)
        gpipe_peak_act = n_microbatches  # store all activations
        of1b_peak_act = n_stages  # store P activations

        result = {
            "n_stages": n_stages,
            "n_microbatches": n_microbatches,
            "stage_fwd_ms": round(ms_fwd, 4),
            "stage_fwd_bwd_ms": round(ms_fwdbwd, 4),
            "gpipe_bubble_pct": round(gpipe_bubble * 100, 1),
            "1f1b_bubble_pct": round(of1b_bubble * 100, 1),
            "interleaved_bubble_pct": round(interleaved_bubble * 100, 1),
            "gpipe_peak_act": gpipe_peak_act,
            "1f1b_peak_act": of1b_peak_act,
        }
        results.append(result)

        print(f"\n  PP={n_stages}, M={n_microbatches} microbatches:")
        print(f"    Stage fwd:     {ms_fwd:.4f}ms")
        print(f"    Stage fwd+bwd: {ms_fwdbwd:.4f}ms")
        print(f"    GPipe bubble:  {gpipe_bubble*100:.1f}%")
        print(f"    1F1B bubble:   {of1b_bubble*100:.1f}%")
        print(f"    Interleaved:   {interleaved_bubble*100:.1f}% (V={V})")
        print(f"    Peak activations: GPipe={gpipe_peak_act}, 1F1B={of1b_peak_act}")

        del stage_model
        torch.cuda.empty_cache()

    return results


# ============================================================
# 实验 4: Gradient Accumulation + 混合精度训练
# ============================================================

def exp4_gradient_accumulation():
    print("\n" + "=" * 60)
    print("实验4: Gradient Accumulation 模拟")
    print("=" * 60)

    results = []
    H = 256
    model = nn.Sequential(
        nn.Linear(H, 4 * H),
        nn.GELU(),
        nn.Linear(4 * H, H),
    ).cuda().half()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    total_steps = 20
    micro_batch = 8

    for accum_steps in [1, 2, 4, 8]:
        torch.cuda.reset_peak_memory_stats()
        model.zero_grad()

        t0 = time.time()
        for step in range(total_steps):
            for micro in range(accum_steps):
                x = torch.randn(micro_batch, H, device="cuda", dtype=torch.float16)
                y = model(x)
                loss = y.sum() / accum_steps
                loss.backward()

            optimizer.step()
            model.zero_grad()

        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 1e6

        # Effective batch size
        eff_batch = micro_batch * accum_steps

        result = {
            "accum_steps": accum_steps,
            "effective_batch": eff_batch,
            "total_time_s": round(elapsed, 3),
            "ms_per_step": round(elapsed / total_steps * 1000, 2),
            "peak_mem_mb": round(peak_mem, 1),
        }
        results.append(result)

        print(f"\n  Accum={accum_steps}, Effective batch={eff_batch}:")
        print(f"    Total time:  {elapsed:.3f}s")
        print(f"    Per step:    {elapsed/total_steps*1000:.2f}ms")
        print(f"    Peak memory: {peak_mem:.0f} MB")

        model.zero_grad()
        torch.cuda.empty_cache()

    del model, optimizer
    torch.cuda.empty_cache()
    return results


# ============================================================
# 实验 5: 模拟 AllReduce 通信开销
# ============================================================

def exp5_allreduce_simulation():
    print("\n" + "=" * 60)
    print("实验5: AllReduce 通信开销模拟")
    print("=" * 60)

    results = []

    # 模拟不同互连带宽下的 AllReduce 时间
    # AllReduce = 2 * (P-1)/P * data_size / bandwidth (ring algorithm)

    configs = [
        ("NVLink (300 GB/s)", 300),
        ("NVLink (600 GB/s)", 600),
        ("PCIe Gen4 (64 GB/s)", 64),
        ("Ethernet (25 Gbps)", 3.125),
        ("Ethernet (100 Gbps)", 12.5),
    ]

    model_sizes = {
        "125M": 250e6,    # bytes in fp16
        "1.3B": 2.6e9,
        "7B": 14e9,
        "13B": 26e9,
        "70B": 140e9,
    }

    print(f"\n{'Config':<25} {'125M':<10} {'1.3B':<10} {'7B':<10} {'13B':<10} {'70B':<10}")
    print("-" * 75)

    for name, bw in configs:
        row = {"interconnect": name, "bandwidth_gbs": bw, "latencies_ms": {}}
        line = f"{name:<25}"
        for mname, msize in model_sizes.items():
            # Ring AllReduce: 2*(P-1)/P * size / bw
            P = 8  # TP=8
            allreduce_bytes = 2 * (P - 1) / P * msize
            latency_ms = allreduce_bytes / bw / 1e6 * 1000
            row["latencies_ms"][mname] = round(latency_ms, 2)
            line += f" {latency_ms:<10.2f}"
        results.append(row)
        print(line)

    # 实测: 单 GPU 上的 "AllReduce" (sum) 作为参照
    print("\n--- 实测 (A16 single GPU) ---")
    for mname, msize in model_sizes.items():
        if msize > 500e6:  # skip if too large for A16
            continue
        n = int(msize / 2)  # fp16 = 2 bytes
        if n > 7e6:  # limit to fit in GPU
            n = 7e6
        x = torch.randn(int(n), device="cuda", dtype=torch.float16)
        ms = bench_ms(lambda: x.sum())
        bw_actual = n * 2 / ms / 1e6
        print(f"  {mname}: {ms:.4f}ms ({bw_actual:.0f} GB/s)")

    return results


# ============================================================
# 实验 6: ZeRO 优化器内存分析
# ============================================================

def exp6_zero_memory():
    print("\n" + "=" * 60)
    print("实验6: ZeRO 优化器内存分析 (实测)")
    print("=" * 60)

    results = []

    # 训练一个小模型，测量不同 ZeRO stage 的理论内存
    H = 256
    model = nn.Sequential(
        nn.Linear(H, 4 * H),
        nn.GELU(),
        nn.Linear(4 * H, H),
    ).cuda()

    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    print(f"\n  Model: {n_params:,} params, {param_bytes/1e6:.2f} MB (fp32)")

    # FP32 training memory breakdown
    model_states = param_bytes  # fp32 params
    gradients = param_bytes     # fp32 gradients
    # Adam: m (fp32) + v (fp32) = 2x params
    optimizer_states = 2 * param_bytes
    total = model_states + gradients + optimizer_states

    print(f"\n  {'Component':<25} {'Size (MB)':<15} {'%':<10}")
    print(f"  {'-'*50}")
    print(f"  {'Parameters (fp32)':<25} {model_states/1e6:<15.2f} {model_states/total*100:<10.1f}")
    print(f"  {'Gradients (fp32)':<25} {gradients/1e6:<15.2f} {gradients/total*100:<10.1f}")
    print(f"  {'Optimizer (Adam m+v)':<25} {optimizer_states/1e6:<15.2f} {optimizer_states/total*100:<10.1f}")
    print(f"  {'Total':<25} {total/1e6:<15.2f} {'100.0'}")

    # ZeRO stages (DP=8)
    DP = 8
    for stage in [0, 1, 2, 3]:
        if stage == 0:
            mem = total  # no sharding
        elif stage == 1:
            mem = model_states + gradients + optimizer_states / DP
        elif stage == 2:
            mem = model_states + (gradients + optimizer_states) / DP
        else:  # stage 3
            mem = total / DP

        saving = (1 - mem / total) * 100
        results.append({
            "zero_stage": stage,
            "mem_per_rank_mb": round(mem / 1e6, 2),
            "saving_pct": round(saving, 1),
        })
        print(f"  ZeRO-{stage}: {mem/1e6:.2f} MB/rank (saving {saving:.1f}%)")

    # 实测: 训练时实际内存
    model_hp = model.half().cuda()
    optimizer = torch.optim.Adam(model_hp.parameters(), lr=1e-4)

    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(16, H, device="cuda", dtype=torch.float16)
    y = model_hp(x)
    y.sum().backward()
    optimizer.step()

    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    print(f"\n  实测训练峰值: {peak_mem:.1f} MB")

    del model, model_hp, optimizer
    torch.cuda.empty_cache()

    return results


# ============================================================
# 实验 7: Sequence Parallelism 对比
# ============================================================

def exp7_sequence_parallel():
    print("\n" + "=" * 60)
    print("实验7: Sequence Parallelism 激活内存对比")
    print("=" * 60)

    results = []

    for seq_len in [512, 1024, 2048, 4096, 8192]:
        B, H = 4, 512
        tp_size = 4

        # 标准 TP: 每个 rank 存完整 activation
        # [B, S, H] per rank
        activation_tp = B * seq_len * H * 2 / 1e6  # fp16 MB

        # SP: 每个 rank 只存 S/TP 的 activation
        # [B, S/TP, H] per rank
        activation_sp = B * (seq_len // tp_size) * H * 2 / 1e6

        saving = (1 - activation_sp / activation_tp) * 100

        result = {
            "seq_len": seq_len,
            "tp_activation_mb": round(activation_tp, 2),
            "sp_activation_mb": round(activation_sp, 2),
            "saving_pct": round(saving, 1),
        }
        results.append(result)

        print(f"  S={seq_len}: TP={activation_tp:.1f}MB, SP={activation_sp:.1f}MB (save {saving:.0f}%)")

    # 实测: LayerNorm + Dropout 的 SP 通信
    print("\n--- 实测 SP AllGather/ReduceScatter 开销 ---")
    x = torch.randn(4, 512, 512, device="cuda", dtype=torch.float16)

    # AllGather (模拟 SP forward)
    chunks = list(x.chunk(4, dim=1))
    ms_ag = bench_ms(lambda: torch.cat(chunks, dim=1))

    # ReduceScatter (模拟 SP backward)
    ms_rs = bench_ms(lambda: x.chunk(4, dim=1))

    # AllReduce (标准 TP 的替代)
    ms_ar = bench_ms(lambda: x.sum(dim=1, keepdim=True).expand_as(x))

    print(f"  AllGather:      {ms_ag:.4f}ms")
    print(f"  ReduceScatter:  {ms_rs:.4f}ms")
    print(f"  AllReduce(ref): {ms_ar:.4f}ms")
    print(f"  SP vs TP 通信: AllGather+ReduceScatter ≈ AllReduce")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("分布式训练模拟实验 (单 GPU)")
    print("=" * 60)

    all_results = OrderedDict()

    all_results["tensor_parallel"] = exp1_tensor_parallel()
    all_results["transformer_block"] = exp2_transformer_block()
    all_results["pipeline_parallel"] = exp3_pipeline_parallel()
    all_results["gradient_accumulation"] = exp4_gradient_accumulation()
    all_results["allreduce_simulation"] = exp5_allreduce_simulation()
    all_results["zero_memory"] = exp6_zero_memory()
    all_results["sequence_parallel"] = exp7_sequence_parallel()

    print("\n" + "=" * 60)
    print("关键洞察")
    print("=" * 60)
    print("""
  1. Tensor Parallel: Column+Row 组合每层只需 1 次 AllReduce
  2. Pipeline Parallel: 1F1B 比 GPipe 内存更少，Interleaved 减少气泡
  3. Gradient Accumulation: 线性增加有效 batch，内存几乎不变
  4. ZeRO: Stage-3 可节省 N× 内存 (N=DP size)
  5. Sequence Parallel: 激活内存减少 TP 倍，通信量 ≈ AllReduce
  6. AllReduce: NVLink 下通信开销 < 5%，Ethernet 下可能 > 30%
""")

    with open("/root/distributed_sim_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Results saved to distributed_sim_results.json")
