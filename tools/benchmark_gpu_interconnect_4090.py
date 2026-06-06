#!/usr/bin/env python3
"""GPU Interconnect Bandwidth on 8xRTX 4090 (Simplified)
=========================================================

Single-process test measuring:
1. P2P access check
2. Cross-GPU memcpy bandwidth (PCIe)
3. Same-GPU HBM bandwidth
4. PCIe topology
5. CUDA IPC bandwidth

NCCL tests require torchrun (separate script).
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import time
import json
import subprocess

N_GPUS = torch.cuda.device_count()
print(f"Available GPUs: {N_GPUS}")
for i in range(N_GPUS):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print("=" * 60)

results = {}

# ============================================================
# 1. P2P Access Check
# ============================================================
print("\n1. P2P Access Check")

p2p_enabled_count = 0
p2p_matrix = []
for i in range(N_GPUS):
    row = []
    for j in range(N_GPUS):
        if i == j:
            row.append("SELF")
        else:
            can_access = torch.cuda.can_device_access_peer(i, j)
            row.append("P2P" if can_access else "PROXY")
            if can_access:
                p2p_enabled_count += 1
    p2p_matrix.append(row)

print(f"  P2P pairs: {p2p_enabled_count} out of {N_GPUS*(N_GPUS-1)}")
if p2p_enabled_count == 0:
    print("  All cross-GPU communication is PCIe-proxied (no NVLink)")
results["p2p_matrix"] = p2p_matrix


# ============================================================
# 2. Cross-GPU Memcpy Bandwidth
# ============================================================
print("\n2. Cross-GPU Memcpy Bandwidth (PCIe-proxied)")

MEMCPY_SIZE_MB = 256
n_elements = MEMCPY_SIZE_MB * 1024 * 1024 // 4

memcpy_results = []
pairs_to_test = [(0,1), (0,2), (0,3), (0,4), (0,7), (1,2), (2,3), (3,4)]

for i, j in pairs_to_test:
    if i >= N_GPUS or j >= N_GPUS:
        continue

    src = torch.randn(n_elements, device=f'cuda:{i}', dtype=torch.float32)
    dst = torch.empty(n_elements, device=f'cuda:{j}', dtype=torch.float32)

    # Warmup
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()

    n = 20
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for _ in range(n):
        dst.copy_(src)
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    per_copy_ms = (t_end - t_start) * 1000 / n
    bw_gb_s = MEMCPY_SIZE_MB / per_copy_ms  # single-direction (data reads from src GPU)
    bidir_bw = 2 * MEMCPY_SIZE_MB / per_copy_ms  # bidirectional effective

    print(f"  GPU {i}→{j}: {per_copy_ms:.3f}ms, "
          f"unidir={bw_gb_s:.2f} GB/s, bidir={bidir_bw:.2f} GB/s")

    memcpy_results.append({
        "src": i, "dst": j,
        "copy_ms": round(per_copy_ms, 3),
        "unidir_bw": round(bw_gb_s, 2),
        "bidir_bw": round(bidir_bw, 2),
    })

    del src, dst

results["memcpy_bandwidth"] = memcpy_results


# ============================================================
# 3. Same-GPU HBM Bandwidth
# ============================================================
print("\n3. Same-GPU HBM Bandwidth (reference)")

hbm_results = []
for gpu_id in range(min(N_GPUS, 4)):
    size_mb = 100
    n_el = size_mb * 1024 * 1024 // 4

    src = torch.randn(n_el, device=f'cuda:{gpu_id}')
    dst = torch.empty(n_el, device=f'cuda:{gpu_id}')

    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize(f'cuda:{gpu_id}')

    n = 100
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(gpu_id):
        s.record()
        for _ in range(n):
            dst.copy_(src)
        e.record()
        torch.cuda.synchronize()

    ms = s.elapsed_time(e) / n
    bw = 2 * size_mb / ms  # read+write

    print(f"  GPU {gpu_id}: {bw:.1f} GB/s (RTX 4090 peak: 960 GB/s)")

    hbm_results.append({"gpu_id": gpu_id, "hbm_bw": round(bw, 1)})
    del src, dst

results["hbm_bandwidth"] = hbm_results


# ============================================================
# 4. PCIe Topology
# ============================================================
print("\n4. PCIe Topology (nvidia-smi topo -m)")

try:
    topo = subprocess.check_output(['nvidia-smi', 'topo', '-m'],
                                    stderr=subprocess.STDOUT, timeout=10).decode()
    print(topo)
    results["pcie_topology"] = topo
except Exception as e:
    print(f"  Error: {e}")
    results["pcie_topology"] = "N/A"


# ============================================================
# 5. CUDA IPC + Peer Memory Access Speed
# ============================================================
print("\n5. Cross-GPU Access via .to() (alternative path)")

ipc_results = []
for i, j in [(0,1), (0,4), (0,7)]:
    if i >= N_GPUS or j >= N_GPUS:
        continue

    src = torch.randn(n_elements, device=f'cuda:{i}', dtype=torch.float32)

    # Warmup
    for _ in range(5):
        dst = src.to(f'cuda:{j}')
    torch.cuda.synchronize()

    n = 20
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for _ in range(n):
        dst = src.to(f'cuda:{j}')
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    per_ms = (t_end - t_start) * 1000 / n
    bw = MEMCPY_SIZE_MB / per_ms

    print(f"  GPU {i}.to({j}): {per_ms:.3f}ms, {bw:.2f} GB/s")

    ipc_results.append({"src": i, "dst": j, "time_ms": round(per_ms, 3), "bw_gb_s": round(bw, 2)})
    del src

results["ipc_bandwidth"] = ipc_results


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY: 8xRTX 4090 PCIe Interconnect")
print("=" * 60)

if memcpy_results:
    avg_bw = sum(r["unidir_bw"] for r in memcpy_results) / len(memcpy_results)
    print(f"Average PCIe bandwidth: {avg_bw:.2f} GB/s (unidirectional)")
    print(f"PCIe 4.0 x16 theoretical: ~25 GB/s unidirectional")

if hbm_results:
    avg_hbm = sum(r["hbm_bw"] for r in hbm_results) / len(hbm_results)
    print(f"Average HBM bandwidth: {avg_hbm:.1f} GB/s")
    print(f"PCIe/HBM ratio: {avg_bw/avg_hbm:.1%}")
    print(f"  → GPU-to-GPU is {avg_hbm/avg_bw:.0f}x slower than same-GPU!")
    print(f"  → This limits TP/DDP scalability on consumer GPUs")

# Save
out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, 'gpu_interconnect_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {out_path}")