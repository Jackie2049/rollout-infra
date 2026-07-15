# RDMA & InfiniBand Network Architecture for AI Clusters

**Date**: 2026-07-15 (Session 10 continued)
**Purpose**: Understand RDMA, InfiniBand, RoCE v2 networking for multi-node GRPO training deployment
**Sources**: NCCL transport layer reading, RDMA networking skill, distributed-nav skill, NVIDIA documentation

---

## 1. RDMA Fundamentals

```
RDMA (Remote Direct Memory Access):
  → One machine reads/writes another machine's memory WITHOUT involving either CPU
  → Kernel bypass: data flows directly from app memory → NIC → network → NIC → app memory
  → Zero-copy: no intermediate buffers, no TCP/IP stack overhead
  → Latency: ~1-2μs (RDMA) vs ~50-100μs (TCP/IP)
  → CPU utilization: ~0% (RDMA) vs ~20-30% (TCP/IP)

Key abstraction: "I can read/write YOUR memory as if it were MY memory"
  → But only for pre-registered memory regions (MR)
  → And only through pre-established queue pairs (QP)
```

---

## 2. InfiniBand vs RoCE v2 vs iWARP

| Property | InfiniBand | RoCE v2 | iWARP |
|----------|-----------|---------|-------|
| **Network** | Dedicated IB fabric | Ethernet + UDP/IP | Ethernet + TCP/IP |
| **Latency** | ~1μs | ~2-5μs | ~10-20μs |
| **Bandwidth** | HDR 200 Gb/s, NDR 400 Gb/s | 100-400 Gb/s Ethernet | 25-100 Gb/s |
| **Lossless** | Yes (credit-based) | Yes (PFC + ECN) | No (TCP retransmit) |
| **Switch** | IB switch (special) | Ethernet switch (PFC) | Standard Ethernet |
| **Cost** | $$$$ (dedicated fabric) | $$ (reuse Ethernet) | $ (standard Ethernet) |
| **GPUDirect** | Yes (native) | Yes (RDMA over Ethernet) | Limited |
| **NCCL transport** | Net (IB) + SHM (NVLink) | Net (RoCE) + SHM | Rarely used |

```
★★★★★★★★ For AI clusters:
  InfiniBand = gold standard (lowest latency, highest bandwidth, native GPUDirect RDMA)
  RoCE v2 = pragmatic choice (reuse Ethernet fabric, good enough for most training)
  iWARP = NOT recommended for AI (TCP overhead, lossy network)

RTX 4090 single GPU: NO RDMA needed (PCIe only, single-node)
Multi-node training: InfiniBand or RoCE v2 ESSENTIAL
```

---

## 3. Verbs API: RDMA Programming Interface

```
RDMA operations via Verbs API (libibverbs):

Key objects:
  Context (ibv_context): RDMA device handle
  Protection Domain (ibv_pd): memory isolation domain
  Memory Region (ibv_mr): registered memory buffer (pin + enable RDMA access)
  Queue Pair (ibv_qp): communication channel (send + receive queues)
  Completion Queue (ibv_cq): operation completion notifications
  Shared Receive Queue (ibv_srq): shared receive buffers for multiple QPs

QP States: RESET → INIT → RTR (Ready to Receive) → RTS (Ready to Send) → ERROR
  → Each state transition requires specific attributes
  → QP must reach RTS before data transfer possible

Memory Registration:
  ibv_reg_mr(pd, buf, length, access_flags)
    → Pins physical pages (no swap-out during transfer)
    → Returns lkey (local key) + rkey (remote key)
    → rkey shared with remote → remote can access this MR
    → access_flags: LOCAL_WRITE, REMOTE_WRITE, REMOTE_READ, REMOTE_ATOMIC

★★★★★★★★ NCCL uses Verbs API internally for IB/RoCE transport
  → QPs pre-established during connection setup
  → MRs registered for send/receive buffers
  → Completion polling (not interrupts) for lowest latency
```

---

## 4. RDMA Operations: Read, Write, Send/Recv

```
RDMA Read:
  → Remote machine reads from YOUR memory
  → YOU register MR with rkey → REMOTE sends RDMA Read request with rkey + remote_addr
  → Data flows: remote NIC → network → your NIC → your memory (CPU NOT involved)
  → Use: loading remote model weights, fetching remote KV cache

RDMA Write:
  → Remote machine writes to YOUR memory
  → YOU register MR with rkey → REMOTE sends RDMA Write request with rkey + remote_addr + data
  → Data flows: remote memory → remote NIC → network → your NIC → your memory
  → ★★★★★★★★ With IMMEDIATE data: RDMA Write + 1 byte immediate → triggers receive completion
  → This is how NCCL sends gradient shards: RDMA Write with immediate = notification

Send/Recv:
  → Traditional message passing (but still kernel-bypass)
  → Sender posts send WR → Receiver must have posted receive WR
  → If no receive WR posted → message dropped (IB) or buffered (iWARP/TCP)
  → NCCL uses Send/Recv for small control messages
  → RDMA Write+Immediate preferred for data transfer (receiver doesn't need to pre-post buffers)

★★★★★★★★ For GRPO multi-node:
  Gradient AllReduce: Ring AllReduce → each node RDMA Writes gradient shard to next node
  KV cache transfer (P/D disaggregation): RDMA Read to fetch remote KV cache blocks
  Model weight sync: RDMA Write to push updated weights to rollout workers
```

---

## 5. InfiniBand Architecture: Switches, Subnets, Paths

```
InfiniBand Network Architecture:

Subnet Manager (SM): manages the IB fabric
  → Discover topology (all switches, HCAs, links)
  → Assign LIDs (Local IDs) to each port
  → Compute routing tables (least-cost paths)
  → Monitor link health, handle failures

IB Switch: cut-through switching, credit-based flow control
  → No packet loss (credit-based: sender waits for credit before transmitting)
  → Latency: ~100ns per hop (vs Ethernet ~1-5μs per hop)
  → Bandwidth: HDR 200 Gb/s per port, NDR 400 Gb/s per port

HCA (Host Channel Adapter): NIC for InfiniBand
  → Connects CPU/GPU memory to IB fabric
  → Supports GPUDirect RDMA: GPU memory → HCA → IB fabric → HCA → GPU memory
  → No CPU involvement in data path!

Path selection:
  → OpenSM: min-hop routing (default)
  → Up/down routing: prevents loops in fat-tree topology
  → Adaptive routing: NVIDIA's AR mechanism (congestion-aware path selection)

★★★★★★★★ For GRPO cluster:
  Typical: 8-64 nodes, dual-rail IB (2 HCA ports per node)
  → Rail-aligned topology: rail 0 connects port 0, rail 1 connects port 1
  → Each rail = independent IB subnet → 2× bandwidth + fault isolation
  → NCCL uses both rails simultaneously for 2× throughput
```

---

## 6. Fat-Tree Topology: Standard AI Cluster Layout

```
2-level fat-tree (standard for 8-32 nodes):

Level 0: Leaf switches (connect to nodes)
  → Each leaf switch: 36-64 ports, connects to 18-32 nodes
  → Multiple leaf switches for larger clusters

Level 1: Core switches (connect leaf switches)
  → Each core switch: connects to all leaf switches
  → Provides full bisection bandwidth

Example: 32-node cluster with HDR200
  → 8 leaf switches × 32 ports = 256 node ports
  → Each node: 2 HCA ports (dual-rail)
  → 32 nodes × 2 rails = 64 connections to leaf switches
  → Core switches connect all leaf switches

★★★★★★★★ Key insight for RTX 4090:
  Single GPU = no network needed (PCIe only)
  But if we scale to multi-node:
  → PCIe bottleneck: RTX 4090 AllReduce 2.76 GB/s (vs NVLink 300+ GB/s)
  → IB HDR200: ~12.5 GB/s effective per rail, 25 GB/s dual-rail
  → 8-node RTX 4090 + IB: AllReduce ~8× slower than 8-node H100 NVLink
```

---

## 7. RoCE v2: RDMA over Converged Ethernet

```
RoCE v2 = RDMA encapsulated in UDP/IP packets:

Packet format:
  [Ethernet header] [IP header] [UDP header (port 4791)] [IB transport header] [IB payload] [IB ICRC]

Key differences from InfiniBand:
  1. Lossless Ethernet REQUIRED:
     → PFC (Priority Flow Control): pause frames per priority class
     → ECN (Explicit Congestion Notification): mark congested packets
     → DCQCN (Data Center QCN): congestion control for RoCE (combines PFC + ECN)

  2. Routing:
     → IP routing (not LID routing like IB)
     → Can traverse L3 routers → wider deployment flexibility
     → But: L3 routing may add latency and congestion

  3. Congestion management:
     → PFC: may cause head-of-line blocking → priority-based pausing
     → PFC deadlock: circular pause dependencies → must break with separate priority classes
     → ECN: sender reduces rate on ECN-marked packets → avoids PFC pause

★★★★★★★★ DCQCN Congestion Control (critical for RoCE):
  Rate-based congestion control:
  → CNP (Congestion Notification Packet): switch sends CNP on ECN marking
  → Sender receives CNP → reduces rate (similar to TCP congestion control)
  → Rate recovery: timer-based gradual increase + byte-count-based fast recovery
  → Parameters: initial_rate, min_rate, rate_increase, timeout

  For AI training:
  → Gradient AllReduce: large messages, sustained bandwidth → DCQCN must maintain high rate
  → KV cache transfer: bursty, medium messages → DCQCN adapts rate quickly
  → Critical: RoCE config must enable PFC + ECN + DCQCN → otherwise throughput drops 50-80%
```

---

## 8. GPUDirect RDMA: GPU-to-Network Direct Path

```
GPUDirect RDMA: NIC reads/writes GPU memory directly

Without GPUDirect:
  GPU memory → cudaMemcpy → CPU pinned memory → NIC → network → NIC → CPU pinned → cudaMemcpy → GPU memory
  → 2 copies, CPU involved, latency ~50-100μs per transfer

With GPUDirect RDMA:
  GPU memory → NIC → network → NIC → GPU memory
  → 0 copies, CPU NOT involved, latency ~5-10μs per transfer
  → 10-20× faster for GPU-to-GPU transfers!

How it works:
  1. cudaMalloc on GPU → get GPU virtual address
  2. nvidia_p2p_get_pages() → pin GPU pages → get physical pages
  3. ibv_reg_mr() on GPU pages → get lkey/rkey for RDMA access
  4. Remote node sends RDMA Read/Write with rkey → direct GPU memory access

★★★★★★★★★ NCCL GPUDirect RDMA:
  → NCCL registers GPU memory as IB MR → enables direct GPU→network transfer
  → This is why NCCL can achieve near-peak IB bandwidth for AllReduce
  → Without GPUDirect: AllReduce throughput drops 50%+ (CPU bounce buffer overhead)

For GRPO multi-node:
  → Gradient AllReduce: GPUDirect → 12.5 GB/s per rail (vs 6 GB/s without)
  → KV cache transfer: GPUDirect → direct GPU→GPU, no CPU bounce
  → Model weight sync: GPUDirect → push weights directly to remote GPU
```

---

## 9. NCCL Transport Layer: How NCCL Uses RDMA

```
NCCL Transport Architecture (from distributed-nav skill):

5-step lifecycle: setup → connect → send → recv → flush

Transport types:
  1. SHM (Shared Memory): intra-node, PCIe/NVLink
     → For RTX 4090: SHM is the ONLY transport (no NVLink P2P)
     → cudaMemcpy between GPU and pinned CPU memory
     → Bandwidth: ~12 GB/s (PCIe 4.0 x16)

  2. Net (Network): inter-node, IB/RoCE
     → Uses Verbs API (ibv_*) for RDMA operations
     → QP per channel for parallel pipelining
     → RDMA Write+Immediate for data + notification
     → Bandwidth: ~12.5 GB/s per rail (HDR200), 25 GB/s dual-rail

  3. CollNet (NVSwitch Hardware): intra-node, hardware-accelerated
     → NVIDIA NVSwitch: hardware AllReduce/ReduceScatter
     → 2-step reduction instead of ring (N steps)
     → Only available with NVSwitch (H100 DXG, DGX systems)
     → NOT available for RTX 4090!

NCCL Channel Architecture for Net transport:
  → Each channel = independent QP + proxy thread + GPU kernel thread block
  → N channels for large messages (bandwidth-focused pipelining)
  → Fewer channels for small messages (latency-focused)
  → Protocol auto-selected: LL (<4KB), LL128 (4-256KB), Simple (>256KB)

★★★★★★★★ For RTX 4090 single-node:
  SHM only → NCCL_P2P_DISABLE=1 (no NVLink P2P)
  AllReduce via SHM → 2.76 GB/s measured (PCIe bottleneck)
  → DDP 8-GPU = 0.46× slower than single GPU!
  → Use single GPU + ZeRO-2 CPU optimizer instead

For multi-node (when scaling):
  SHM (intra-node) + Net (inter-node, IB/RoCE)
  → NCCL_P2P_DISABLE=1 (RTX 4090 has no NVLink)
  → NCCL_SHM_DISABLE=0 (SHM required for intra-node)
  → NCCL_MAX_NRINGS=4 (limited by PCIe bandwidth)
  → NCCL_ALGO=RING (tree only beneficial with NVLink)
  → NCCL_PROTO=Simple (large gradient messages)
```

---

## 10. RDMA for GRPO Training Data Paths

```
GRPO training involves 3 distinct data paths that need RDMA:

1. Gradient AllReduce (training phase):
   Ring AllReduce: each node sends gradient shard to next node
   → RDMA Write: gradient data from local GPU → remote GPU
   → GPUDirect: direct GPU→NIC→network→NIC→GPU transfer
   → Bandwidth need: model_size / num_nodes per step
   → Example: 7B model, 8 nodes → 7B × 2bytes × 2 (fwd+bwd) / 8 ≈ 3.5 GB per node per step

2. KV Cache Transfer (P/D disaggregation, rollout phase):
   Prefill worker → KV cache → Decode worker
   → RDMA Write: KV blocks from prefill GPU → decode GPU
   → Or RDMA Read: decode worker pulls KV from prefill worker
   → Bandwidth need: KV_cache_size / prefill_time
   → Example: 7B model, 2048 tokens KV → ~2 GB per request

3. Model Weight Sync (training→rollout transition):
   Trainer → updated weights → Rollout worker
   → RDMA Write: weight deltas from trainer GPU → rollout GPU
   → With LoRA: only ~50 MiB delta → very fast
   → With full params: ~14 GiB → needs sustained bandwidth

★★★★★★★★★ SGLang NIXL connector:
  → Uses RDMA for KV cache transfer in P/D disaggregation
  → Zero-copy: KV blocks registered as MR → remote GPU directly reads/writes
  → Async DMA + ZMQ for coordination
  → This is the state-of-art approach for GRPO serving

verl TransferQueue:
  → Uses Ray for weight coordination (not RDMA directly)
  → But underlying NCCL uses RDMA for gradient sync
  → For single GPU (RTX 4090): no RDMA needed → all local
```

---

## 11. RDMA Performance Analysis for AI Workloads

```
Benchmark data (typical AI cluster):

AllReduce (7B model, 8 nodes):
  HDR200 dual-rail: ~25 GB/s → 7B × 2B = 14 GB → ~0.56s
  RoCE v2 (100GbE): ~6.25 GB/s → 14 GB → ~2.24s (4× slower)
  TCP/IP (100GbE): ~3 GB/s → 14 GB → ~4.67s (8× slower)
  PCIe SHM (8×4090): ~2.76 GB/s → 14 GB → ~5.07s (9× slower!)

KV Cache Transfer (P/D disaggregation):
  HDR200: ~200 Gb/s = 25 GB/s → 2 GB KV → ~80ms
  RoCE v2: ~100 Gb/s = 12.5 GB/s → 2 GB KV → ~160ms
  TCP/IP: ~50 Gb/s = 6.25 GB/s → 2 GB KV → ~320ms

★★★★★★★★ Key insight for RTX 4090 cluster:
  RTX 4090 + PCIe SHM: AllReduce slower than IB/RoCE by 3-9×
  → Even with RDMA, RTX 4090 cluster is bandwidth-limited by PCIe!
  → NVLink (300+ GB/s) is what makes A100/H100 clusters fast for distributed training
  → RTX 4090 is best for SINGLE GPU training (ZeRO-2 + CPU optimizer)
  → For multi-node: RTX 4090 + IB helps but still limited by intra-node PCIe

Recommendation hierarchy:
  1. Single RTX 4090 → ZeRO-2 + CPU optimizer (BEST for 7B)
  2. Multi-node RTX 4090 + IB → Ring AllReduce (slower but works)
  3. Multi-node RTX 4090 + Ethernet → TCP/IP (AVOID if possible)
  4. Single H100 → NVLink + GPUDirect (ideal for production)
```

---

## 12. RDMA Configuration for AI Clusters

```
InfiniBand configuration checklist:

1. Firmware:
   → mlxup -d (update Mellanox firmware)
   → mst start (MST service for firmware management)

2. Driver:
   → MLNX_OFED driver (not stock Linux kernel IB driver)
   → ibv_devinfo (verify HCA is visible)
   → ofed_info -s (check driver version)

3. Network:
   → OpenSM running (subnet manager)
   → ibnetdiscover (verify topology)
   → ibcheckerrors (check for errors)

4. PFC (for RoCE v2):
   → mlnx_qos --pfc 0,0,0,0,0,0,1,1 (enable PFC on priorities 6,7)
   → TC classifier: map RDMA traffic to priority 6/7

5. DCQCN (for RoCE v2):
   → ECN marking on switches: enable on RDMA priority
   → CNP generation on switches: enable on RDMA priority
   → Rate limiter on NIC: enable DCQCN parameters

6. GPUDirect RDMA:
   → nvidia-peermem module loaded (for GPUDirect)
   → Verify: cat /proc/driver/nvidia/gpus/*/peermem
   → NCCL GPUDirect: auto-detected, no special config needed

★★★★★★★★ RoCE v2 CRITICAL config (without this, throughput drops 50-80%):
  1. Enable PFC on RDMA priorities (6,7)
  2. Enable ECN marking on switches
  3. Enable CNP generation on switches
  4. Set TC classifier for RDMA traffic
  5. Disable IP forwarding on RDMA interfaces (prevent routing loops)

Common failure: PFC deadlock
  → Circular pause dependencies → traffic stops completely
  → Fix: separate priority classes, break circular dependencies
  → Or: use only ECN + DCQCN (no PFC) → but needs ECN-capable switches
```

---

## 13. RDMA vs TCP/IP for GRPO Training Decision Guide

```
Decision tree:

Single GPU (RTX 4090):
  → NO RDMA needed → all communication via PCIe SHM
  → NCCL_P2P_DISABLE=1 (no NVLink P2P on 4090)
  → ZeRO-2 + CPU optimizer → single-GPU optimal

Multi-GPU single node (8× RTX 4090):
  → SHM (PCIe) only → AllReduce 2.76 GB/s
  → DDP 0.46× → WORSE than single GPU!
  → Consider: ZeRO-2 with gradient partitioning → better than DDP
  → RDMA: NOT needed (all GPUs on same node)

Multi-node (2+ nodes with IB/RoCE):
  → RDMA essential for inter-node communication
  → NCCL Net transport: IB or RoCE
  → GPUDirect RDMA: direct GPU→network transfer
  → But: RTX 4090 PCIe still bottleneck within each node

★★★★★★★★ FINAL recommendation for RTX 4090 GRPO:
  STICK WITH SINGLE GPU unless you have:
  → H100/A100 nodes with NVLink (300+ GB/s intra-node)
  → InfiniBand HDR200+ inter-node (25+ GB/s)
  → Otherwise: multi-node RTX 4090 is SLOWER than single GPU!

Production path:
  1. Develop on single RTX 4090 (ZeRO-2 + CPU optimizer)
  2. Scale to H100 cluster when ready for production
  3. Use RDMA (IB/RoCE) + GPUDirect on H100 cluster
  4. SGLang NIXL connector for KV cache transfer (P/D disaggregation)
```

---

## 14. SGLang NIXL Connector: State-of-Art RDMA for KV Transfer

```
SGLang NIXL connector (from sglang-nav skill):

Architecture:
  PrefillWorker → NIXL RDMA → DecodeWorker
  → Zero-copy: KV blocks registered as MR
  → Async DMA: cudaMemcpyAsync + ibv_post_send
  → ZMQ coordination: metadata exchange (which blocks, offsets)
  → Bootstrap: establish QPs + exchange rkeys at startup

Data flow:
  1. PrefillWorker computes KV cache for prompt
  2. NIXL registers KV GPU memory as MR → get rkeys
  3. PrefillWorker sends rkeys + offsets to DecodeWorker via ZMQ
  4. DecodeWorker issues RDMA Read: remote GPU KV → local GPU memory
  5. DecodeWorker begins decode using received KV cache

★★★★★★★★★ NIXL = production-grade RDMA KV transfer
  → Used in SGLang P/D disaggregation
  → Also in vLLM KV transfer connector
  → Best approach for multi-node GRPO serving

Other connectors:
  Mooncake: RDMA + global KV cache store (like Redis for KV)
  MORI: experimental, for research
  Fake: testing only
```

---

## 15. Cross-Reference: RDMA Across 7 Frameworks

| Framework | RDMA Usage | Transport | GPUDirect | Key Feature |
|-----------|-----------|-----------|-----------|-------------|
| **NCCL** | Gradient AllReduce | IB/RoCE Verbs | Yes | Channel pipelining + proxy thread |
| **DeepSpeed** | Via NCCL | IB/RoCE | Yes | ZeRO gradient partitioning |
| **Megatron** | Via NCCL | IB/RoCE | Yes | TP AllReduce + EP All-to-All |
| **vLLM** | KV transfer (NIXL) | IB/RoCE | Yes | Zero-copy KV block transfer |
| **SGLang** | KV transfer (NIXL) + DP | IB/RoCE | Yes | Async DMA + ZMQ coordination |
| **verl** | Via NCCL (training) + Ray | IB/RoCE | Yes | TransferQueue + Ray orchestration |
| **rLLM** | Via NCCL | IB/RoCE | Partial | Single-node focused |

★★★★★★★★★ Universal pattern:
  ALL frameworks use NCCL for gradient sync → NCCL uses RDMA underneath
  KV transfer frameworks (vLLM, SGLang) use NIXL/RDMA directly
  For RTX 4090 single GPU: NO RDMA needed at all
  For multi-node: RDMA is critical for ALL frameworks

---

## Session Stats
- **RDMA fundamentals**: Verbs API, QP/CQ/SRQ/MR lifecycle, RDMA Read/Write/Send operations
- **InfiniBand vs RoCE v2**: comparison table, lossless vs lossy, DCQCN congestion control
- **GPUDirect RDMA**: GPU→network direct path, 10-20× faster than CPU bounce buffer
- **Fat-tree topology**: standard AI cluster layout, rail-aligned dual-rail
- **NCCL transport**: SHM (intra-node) + Net (inter-node IB/RoCE) + CollNet (NVSwitch)
- **GRPO data paths**: 3 distinct RDMA needs (gradient, KV, weight sync)
- **SGLang NIXL**: state-of-art RDMA KV transfer connector
- **RTX 4090 recommendation**: single GPU optimal, multi-node limited by PCIe
