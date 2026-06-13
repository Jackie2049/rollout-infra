---
name: rdma-network-nav
description: Navigate RDMA, InfiniBand, RoCE v2, AI cluster networking topology, and NCCL transport internals. Quick reference for Verbs API, QP/CQ/SRQ/MR, fat-tree/rail-aligned topology, DCQCN congestion control, GPUDirect RDMA, and cluster configuration.
triggers:
  - RDMA
  - InfiniBand
  - RoCE
  - IB NIC
  - QP CQ SRQ
  - doorbell
  - memory registration
  - GPUDirect RDMA
  - DMA-BUF
  - fat-tree topology
  - rail-aligned
  - DCQCN
  - PFC ECN
  - lossless Ethernet
  - NCCL IB plugin
  - NCCL transport
  - cluster networking
  - NVLink NVSwitch
  - SHARP
  - MNNVL
  - PXN
---

# RDMA + AI Cluster Networking Navigator

## Architecture: 3-Layer Stack

```
Application → NCCL → RDMA (IB/RoCE) → Network (Fat-Tree/Rail-Aligned)
```

## Subsystem Index

### 1. RDMA Verbs API (libibverbs)
- **Files**: rdma-core → libibverbs/ (userspace library)
- **Notes**: `notebook/fundamentals/rdma-ai-networking-deep-dive.md` (deep dive), `notebook/fundamentals/rdma-networking.md` (reference)
- **Key**: Kernel bypass via doorbell (MMIO write), zero-copy DMA, QP state machine (RESET→INIT→RTR→RTS), CQE 64B polling, SRQ shared receive, ibv_reg_mr (lkey/rkey), ibv_post_send/recv/poll_cq

### 2. QP State Machine
- **5-step lifecycle**: RESET→INIT(recv only)→RTR(recv remote)→RTS(send+recv)→ERROR
- **Connection**: Exchange QP info via TCPStore (NCCL), then both sides RTR→RTS
- **Types**: RC(Reliable Connected), UC, UD, DCT(Dynamic Connected Transport)
- **NCCL**: RC QP per rank pair, NCCL_IB_QPS_PER_CONNECTION=1-128

### 3. Doorbell + CQE Mechanism
- **Doorbell**: ibv_post_send → serialize WR→WQE → write NIC MMIO (PCI BAR) → user-space, zero syscall!
- **CQE**: NIC DMA writes 64B completion entries → user polls ibv_poll_cq → ~0.5μs latency
- **SRQ**: N QPs share 1 receive queue → M buffers vs N² → memory savings

### 4. RoCE v2 Packet Format
- **Encapsulation**: IB transport headers (BTH+RETH) + UDP(port 4791) + IP(DSCP+ECN) + Ethernet
- **vs InfiniBand**: IB uses LRH(LID routing)+credit flow control → native lossless; RoCE needs PFC+ECN+DCQCN
- **NCCL_IB_TC=106**: DSCP=26 → priority=3 → RoCE QoS mapping
- **Notes**: `notebook/fundamentals/rdma-ai-networking-deep-dive.md` Section 2

### 5. Lossless Ethernet (3 Requirements)
- **PFC** (802.1Qbb): Priority-based PAUSE → xoff_threshold → stop specific priority
- **ECN** (RFC 3168): IP header 2-bit marking → switch marks CE → CNP feedback → DCQCN rate reduction
- **DCQCN**: Rate=Rate×(1-α/2), α=ECN ratio, CNP→sender throttles → Fast/Active/Hyper 3-phase recovery
- **PFC deadlock risk**: Ring topology → use fat-tree unidirectional flow → no deadlock

### 6. Fat-Tree / Rail-Aligned Topology
- **2-layer fat-tree**: leaf+spine → ≤64 nodes → k/2 leaf × k/2 spine
- **3-layer fat-tree**: leaf+spine+core → >64 nodes → Meta 16K GPU cluster
- **Rail-aligned**: GPU_i↔NIC_i → 8 rails → 8× bandwidth aggregation → NVLink RS + IB AllReduce + NVLink AG
- **Meta**: 24-rail aligned, 3-layer fat-tree, Broadcom Tomahawk4
- **Notes**: `notebook/fundamentals/rdma-ai-networking-deep-dive.md` Section 3

### 7. DCQCN Congestion Control
- **Algorithm**: Zhu et al. SIGCOMM 2015 → DCTCP + QCN for RDMA
- **Sender**: Rate=Rate×(1-α/2) on CNP → FastRecovery(50μs)→ActiveIncrease(5ms)→HyperIncrease(150ms)
- **Receiver**: ECN→CNP feedback → 1ms rate limit per flow
- **Switch**: ECN mark threshold → early signal → DCQCN handles 99% congestion → PFC for 1% extreme
- **Notes**: `notebook/fundamentals/rdma-ai-networking-deep-dive.md` Section 4

### 8. NCCL Transport Plugins
- **IB Plugin**: ncclNetIB → ibv_get_device_list → QP creation → RDMA Write (GPUDirect) → DMA-BUF registration
- **Socket Plugin**: ncclNetSocket → TCP fallback → RTX 4090 only option → 2.76GB/s AllReduce → disaster!
- **PXN**: Path Crossing NVLink → GPU borrows optimal NIC via NVLink → NCCL_PXN_DISABLE=0
- **Notes**: `notebook/fundamentals/rdma-ai-networking-deep-dive.md` Section 5

### 9. GPUDirect RDMA
- **nvidia-peermem**: IB peer memory client → registers GPU memory → NIC DMA GPU directly
- **DMA-BUF**: Linux 6.x+ → replaces peermem → NCCL_DMABUF_ENABLE=1 (NCCL 2.20+)
- **ACS**: Access Control Services → may block GPUDirect → disable with setpci or GRUB parameter

### 10. SHARP + NVLink Switch
- **IB SHARP**: Switch-internal Reduce → O(1) latency → CollNetChain/CollnetDirect algorithm
- **NVLS**: NVSwitch SHARP engine → NVLSTree algorithm → NCCL_NVLS_ENABLE=2
- **PAT**: Parallel AllReduce Tree → H100+ → multi-tree parallel bandwidth

## Key Data Flow

```
RDMA Write (GPUDirect):
  GPU buffer → NIC DMA read → IB/RoCE packet → switch → remote NIC DMA write → remote GPU buffer
  → Zero CPU copy, zero syscall, ~1μs latency

NCCL AllReduce (Rail-aligned, inter-node):
  NVLink ReduceScatter (intra) → IB AllReduce per rail (inter) → NVLink AllGather (intra)
  → 8 rails × 400Gb/s = 3200Gb/s aggregate → ~100GB/s effective

RTX 4090 (No RDMA):
  GPU → CPU copy → TCP Socket → Ethernet → TCP Socket → CPU → GPU
  → 4 copies, syscall overhead, ~50μs → 2.76GB/s → disaster → single GPU optimal
```

## RTX 4090 Specifics

- **No RDMA**: No IB NIC, no NVLink, no GPUDirect → PCIe + TCP Socket only
- **AllReduce**: 2.76GB/s → DDP 8GPU=0.46x → single GPU optimal
- **NCCL config**: NCCL_P2P_DISABLE=1, Socket plugin only, no IB configuration possible
- **Strategy**: LoRA + gradient checkpointing on single GPU; ZeRO-2 + CPU offload for full finetune

## NCCL RDMA Environment Variables Quick Reference

| Variable | NVLink GPU | RoCE v2 | Purpose |
|----------|------------|---------|---------|
| NCCL_IB_DISABLE | 0 | 0 | Enable IB |
| NCCL_IB_HCA | mlx5_0:1,mlx5_1:1 | mlx5_0:1 | Select IB device |
| NCCL_IB_GID_INDEX | -1 (auto) | 3 (RoCE) | GID for connection |
| NCCL_IB_TC | 0 | 106 | Traffic Class (RoCE QoS) |
| NCCL_NET_GDR_LEVEL | 5 | 5 | GPU Direct RDMA level |
| NCCL_DMABUF_ENABLE | 1 | 1 | DMA-BUF (Linux 6.x+) |
| NCCL_IB_QPS_PER_CONNECTION | 4 | 4 | Multi-QP per connection |
| NCCL_PXN_DISABLE | 0 | 0 | Path Crossing NVLink |
| NCCL_IB_ADAPTIVE_ROUTING | 1 | 0 | Adaptive routing (IB=yes, RoCE=no) |

## Common Issues & Debugging

1. **NCCL using TCP instead of IB**: Check ibv_devinfo → ACTIVE, libibverbs installed, NCCL_IB_DISABLE=0
2. **GPUDirect not working**: lsmod | grep nvidia_peermem, check ACS, check NUMA alignment
3. **RoCE GID error**: NCCL 2.21+ auto-select (NCCL_IB_GID_INDEX=-1), or show_gids for manual
4. **memlock insufficient**: /etc/security/limits.conf → memlock unlimited → re-login
5. **PFC deadlock**: Use fat-tree unidirectional flow → no ring topology → no deadlock
6. **ECN/PFC misconfigured**: DCQCN handles 99% → PFC for 1% extreme → set ecn_mark < xoff threshold

## Bandwidth Reference

| Interconnect | Bidirectional BW | Latency | Scope |
|--------------|-----------------|---------|-------|
| NVLink 4 (H100) | 900 GB/s | ~1μs | Intra-node |
| NVLink 5 (B200) | 1,800 GB/s | ~0.5μs | Intra-node |
| IB NDR 400G | 50 GB/s | ~1μs | Inter-node |
| IB HDR 200G | 25 GB/s | ~2μs | Inter-node |
| RoCE v2 200G | 25 GB/s | ~3-5μs | Inter-node |
| PCIe Gen4 x16 | 64 GB/s | ~8μs | GPU↔CPU |
| TCP Ethernet | 12.5 GB/s | ~50μs | Inter-node |
