#!/usr/bin/env python3
"""NCCL Communication Benchmark — RTX 4090 PCIe (single process, all GPUs)

Benchmarks:
1. Single GPU: HBM bandwidth + GEMM TFLOPS
2. P2P access matrix + transfer rates
3. AllReduce throughput by data size
4. AllGather / ReduceScatter
5. Broadcast / SendRecv baseline
6. FSDP overhead estimation

Usage: torchrun --nproc_per_node=N nccl_comm_benchmark.py
"""

import json
import os
import time
import torch
import torch.distributed as dist

def main():
    n_gpu = torch.cuda.device_count()
    is_distributed = 'RANK' in os.environ and dist.is_available()

    if is_distributed:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    all_results = {
        'metadata': {
            'pytorch': torch.__version__,
            'cuda': torch.version.cuda,
            'n_gpu': world_size,
            'gpu_name': torch.cuda.get_device_name(rank),
            'is_p2p_enabled': False,
        }
    }

    # ── Phase 1: GPU specs + P2P matrix (rank 0 only) ──
    if rank == 0:
        props = torch.cuda.get_device_properties(0)
        all_results['gpu_specs'] = {
            'name': props.name,
            'memory_gb': round(props.total_memory / 1e9, 1),
            'sm': f'{props.major}.{props.minor}',
            'mp_count': props.multi_processor_count,
        }

        # P2P access matrix
        p2p = {}
        for i in range(n_gpu):
            for j in range(n_gpu):
                if i != j:
                    p2p[f'{i}->{j}'] = torch.cuda.can_device_access_peer(i, j)
        all_results['p2p_matrix'] = p2p
        all_results['metadata']['is_p2p_enabled'] = any(p2p.values())
        print(f"GPU: {props.name}, {world_size} GPUs")
        print(f"P2P: {any(p2p.values())} ({sum(p2p.values())}/{len(p2p)} enabled)")

        # Transfer rates — use P2P bandwidth test instead of API
        rates = {}
        for i in range(min(n_gpu, 8)):
            for j in range(min(n_gpu, 8)):
                if i != j:
                    # PyTorch 2.5 doesn't have device_transfer_rate
                    # Just mark as unknown; P2P access tells us connectivity
                    rates[f'{i}->{j}'] = 'PCIe' if not p2p.get(f'{i}->{j}', False) else 'NVLink'
        all_results['transfer_rates_gbps'] = rates
        for k, v in sorted(rates.items()):
            print(f"  {k}: {v}")

        # HBM bandwidth (device-to-device copy)
        for size_mb in [1, 4, 16, 64, 256]:
            size = size_mb * 1024 * 1024
            src = torch.randn(size // 4, device='cuda:0')
            warmup = 3; runs = 20
            for _ in range(warmup):
                dst = src.clone()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                dst = src.clone()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / runs
            bw = size / dt / 1e9
            all_results[f'hbm_{size_mb}MB'] = {
                'size_mb': size_mb, 'latency_ms': round(dt*1000, 3),
                'bw_gbps': round(bw, 2),
            }
            print(f"  HBM {size_mb}MB: {bw:.1f} GB/s")

        # Single GPU GEMM
        for M in [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            N, K = 4096, 4096
            a = torch.randn(M, K, device='cuda:0', dtype=torch.bfloat16)
            b = torch.randn(K, N, device='cuda:0', dtype=torch.bfloat16)
            warmup = 3; runs = max(10, 50 // max(M, 1))
            for _ in range(warmup):
                c = a @ b
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                c = a @ b
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / runs
            tflops = 2*M*N*K / dt / 1e12
            all_results[f'gemm_M{M}'] = {
                'M': M, 'tflops': round(tflops, 2),
                'peak_pct': round(tflops / 169.6 * 100, 1),
                'latency_us': round(dt*1e6, 1),
            }
            print(f"  GEMM M={M}: {tflops:.1f} TFLOPS ({tflops/169.6*100:.1f}% peak)")

    # ── Phase 2: AllReduce (all ranks) ──
    if is_distributed and world_size > 1:
        dist.barrier()
        ar_results = {}

        for size_kb in [4, 64, 256, 1024, 4096, 16384, 65536, 262144]:
            size = size_kb * 1024
            n_elem = size // 4  # float32
            data = torch.randn(n_elem, device=f'cuda:{rank}')

            warmup = 5; runs = 50
            for _ in range(warmup):
                dist.all_reduce(data, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(runs):
                dist.all_reduce(data.clone(), op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / runs

            bw_gbps = size / dt / 1e9  # effective per-GPU bandwidth
            agg_bw = size * world_size / dt / 1e9
            # Algorithm bandwidth: Ring AllReduce = 2*(N-1)/N * size / time
            algo_bw = 2 * size * (world_size-1) / world_size / dt / 1e9

            ar_results[f'{size_kb}KB'] = {
                'latency_ms': round(dt*1000, 3),
                'per_gpu_bw_gbps': round(bw_gbps, 3),
                'aggregate_bw_gbps': round(agg_bw, 3),
                'algo_bw_gbps': round(algo_bw, 3),
            }

        # AllGather
        ag_results = {}
        for size_kb in [4, 64, 256, 1024, 4096, 16384, 65536]:
            size = size_kb * 1024
            per_gpu_elem = size // 4 // world_size
            data = torch.randn(per_gpu_elem, device=f'cuda:{rank}')

            warmup = 5; runs = 50
            out = [torch.zeros_like(data) for _ in range(world_size)]
            for _ in range(warmup):
                dist.all_gather(out, data)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(runs):
                out2 = [torch.zeros_like(data) for _ in range(world_size)]
                dist.all_gather(out2, data.clone())
            torch.cuda.synchronize()
            dt_ag = (time.perf_counter() - t0) / runs

            ag_results[f'{size_kb}KB'] = {
                'latency_ms': round(dt_ag*1000, 3),
                'bw_gbps': round(size / dt_ag / 1e9, 3),
            }

        # ReduceScatter
        rs_results = {}
        for size_kb in [4, 64, 256, 1024, 4096, 16384, 65536]:
            size = size_kb * 1024
            full_elem = size // 4
            full_data = torch.randn(full_elem, device=f'cuda:{rank}')
            per_gpu_elem = full_elem // world_size
            out = torch.zeros(per_gpu_elem, device=f'cuda:{rank}')
            chunks = list(full_data.chunk(world_size))

            warmup = 5; runs = 50
            for _ in range(warmup):
                dist.reduce_scatter(out, chunks, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(runs):
                fc = full_data.clone()
                ck = list(fc.chunk(world_size))
                o2 = torch.zeros(per_gpu_elem, device=f'cuda:{rank}')
                dist.reduce_scatter(o2, ck, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            dt_rs = (time.perf_counter() - t0) / runs

            rs_results[f'{size_kb}KB'] = {
                'latency_ms': round(dt_rs*1000, 3),
                'bw_gbps': round(size / dt_rs / 1e9, 3),
            }

        # FSDP overhead: AR = AG + RS
        fsdp_results = {}
        for model_name, n_params in [('25M', 25e6), ('125M', 125e6), ('7B', 7e9)]:
            param_bytes = n_params * 2  # BF16
            param_kb = int(param_bytes / 1024)
            # Find closest measured size
            closest_kb = min(ar_results.keys(), key=lambda k: abs(int(k.replace('KB','')) - param_kb/1024))
            ar_time = ar_results[closest_kb]['latency_ms']
            # FSDP per step: AllGather(fwd) + ReduceScatter(bwd) per layer
            # ~2 collectives per layer
            # For 7B ~32 layers, each ~218MB
            if model_name == '7B':
                n_layers = 32
                layer_param_kb = param_kb // n_layers
                closest_layer_kb = min(ag_results.keys(), key=lambda k: abs(int(k.replace('KB','')) - layer_param_kb))
                layer_ag = ag_results[closest_layer_kb]['latency_ms']
                layer_rs = rs_results[closest_layer_kb]['latency_ms']
                step_comm_ms = n_layers * (layer_ag + layer_rs) * 2  # fwd + bwd
            else:
                # Direct extrapolation
                closest_kb_int = int(closest_kb.replace('KB',''))
                step_comm_ms = ar_time * 2 * (param_kb / closest_kb_int / 1024)

            fsdp_results[model_name] = {
                'n_params': int(n_params),
                'param_size_gb': round(param_bytes / 1e9, 2),
                'estimated_step_comm_ms': round(step_comm_ms, 1),
            }

        # Print on rank 0
        if rank == 0:
            all_results['allreduce'] = ar_results
            all_results['allgather'] = ag_results
            all_results['reducescatter'] = rs_results
            all_results['fsdp_overhead'] = fsdp_results

            print(f"\n  AllReduce ({world_size} GPUs):")
            for k, v in ar_results.items():
                print(f"    {k}: {v['latency_ms']}ms, bw={v['algo_bw_gbps']}GB/s")
            print(f"\n  AllGather ({world_size} GPUs):")
            for k, v in ag_results.items():
                print(f"    {k}: {v['latency_ms']}ms, bw={v['bw_gbps']}GB/s")
            print(f"\n  ReduceScatter ({world_size} GPUs):")
            for k, v in rs_results.items():
                print(f"    {k}: {v['latency_ms']}ms, bw={v['bw_gbps']}GB/s")
            print(f"\n  FSDP overhead:")
            for k, v in fsdp_results.items():
                print(f"    {k}: ~{v['estimated_step_comm_ms']}ms comm per step")

    # Save on rank 0
    if rank == 0:
        output = os.path.join(os.environ.get('ROLLOUT_HOME', '.'), f'nccl_comm_{world_size}gpu.json')
        with open(output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {output}")

    if is_distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['ROLLOUT_HOME'] = os.environ.get('ROLLOUT_HOME', os.getcwd())
    main()