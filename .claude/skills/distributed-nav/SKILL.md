---
name: distributed-nav
description: Navigate PyTorch torch.distributed, NCCL, and distributed training internals. Quick reference for ProcessGroupNCCL, DDP, FSDP, NCCL channels/proxy/protocol, and training configuration.
triggers:
  - distributed training
  - torch.distributed
  - ProcessGroupNCCL
  - NCCL internals
  - DDP bucketing
  - FSDP lifecycle
  - NCCL config
  - distributed debugging
  - NCCL timeout
  - gradient sync
  - allreduce
  - collective operation
  - NCCL channel
  - NCCL protocol
  - NCCL transport
---

# Distributed Training Internals Navigator

## Architecture: 4-Layer Stack

```
Application → torch.distributed → ProcessGroupNCCL → NCCL
```

## Subsystem Index

### 1. ProcessGroupNCCL (~3500+ lines C++)
- **File**: `torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp` (~1400 lines) + `.cpp` (~3500 lines)
- **Notes**: `notebook/fundamentals/pytorch-torch-distributed-internals-deep-dive.md`
- **Key**: ncclComm lazy init + cached reuse (devNCCLCommMap_), WorkNCCL async model, Watchdog+HeartbeatMonitor dual monitoring, stream-per-device double-barrier event model

### 2. WorkNCCL (async Work object)
- **File**: `ProcessGroupNCCL.hpp` (class WorkNCCL)
- **Key**: CUDA event tracking (ncclStartEvent/ncclEndEvent), stashed_for_allocator_safety_ (replaces recordStream), opTimeout_ (default 10min), future_ (Python async bridge)

### 3. Watchdog Thread
- **File**: `ProcessGroupNCCL.hpp` (class Watchdog)
- **Key**: Monitors WorkNCCL timeouts → ncclCommAbort on timeout, 4 ErrorHandlingModes (NoHandling/TearDown/CleanUpOnly/SkipCleanUp), heartbeat_ atomic counter, propagatePgError_ → TCPStore broadcast

### 4. HeartbeatMonitor
- **File**: `ProcessGroupNCCL.hpp` (class HeartbeatMonitor)
- **Key**: Monitors Watchdog thread heartbeat → dumps debug info + std::abort if Watchdog hangs (e.g., CUDA global lock deadlock)

### 5. DDP Reducer (~reducer.cpp)
- **File**: `torch/csrc/distributed/c10d/reducer.h` + `reducer.cpp`
- **Notes**: `notebook/fundamentals/pytorch-torch-distributed-internals-deep-dive.md`
- **Key**: Gradient bucketing (bucket_cap_mb=25, reverse order), autograd_hook→mark_variable_ready→mark_bucket_ready→allreduce, flat buffer (contiguous per-bucket tensor), communication-compute overlap

### 6. FSDP2 Lifecycle
- **File**: `torch/distributed/fsdp2/_fsdp_param_group.py`
- **Notes**: `notebook/fundamentals/pytorch-torch-distributed-internals-deep-dive.md`
- **Key**: Per-parameter sharding (not FlatParameter!), shard→unshard(all_gather)→compute→reshard→backward→reduce_scatter, param.data swap (zero-copy), 3 reshard strategies (NO_SHARD/SHARD_GRAD_OP/FULL_SHARD)

### 7. NCCL Channel Architecture
- **File**: `src/channel.h` + `src/channel.cc` (NCCL repo)
- **Notes**: `notebook/fundamentals/nccl-internals-architecture-deep-dive.md`
- **Key**: Independent parallel pipeline per channel, N channels for pipelining large messages, fewer channels for small messages (latency priority)

### 8. NCCL Proxy Thread
- **File**: `src/proxy.cc` (NCCL repo)
- **Key**: CPU-side async DMA/network handler, ncclProxyStep handshake mechanism, bidirectional flags (ready→done), decoupled from GPU kernel execution

### 9. NCCL Transport Layer
- **File**: `src/transport/` (NCCL repo)
- **Key**: Net (IB/RoCE/TCP) for inter-node, SHM (PCIe/NVLink) for intra-node, CollNet (NVSwitch hardware), 5-step lifecycle (setup→connect→send→recv→flush)

### 10. NCCL Protocol (LL/LL128/Simple)
- **File**: `src/collectives.cc` (NCCL repo)
- **Key**: LL (<4KB, interleaved packed, min latency), LL128 (4-256KB, 128B aligned, balance), Simple (>256KB, contiguous flat, max throughput), auto-selected by message size

### 11. NCCL Topo + Algorithm Selection
- **File**: `src/graph/topo.cc` + `src/graph/search.cc` (NCCL repo)
- **Key**: topo discovery (NVML+PCI+NIC) → graph construction → simulated annealing search → Ring (large msgs, bandwidth) vs Tree (small msgs, latency) vs CollNet (NVSwitch, 2 steps)

### 12. NCCL GPU Kernel
- **File**: `src/kernel.cc` (NCCL repo)
- **Key**: Warp-level reduce, spin-wait on handshake flags, per-channel thread block, data copy/reduce via shared buffers

## Key Data Flow

```
DDP Backward:
  autograd_hook → mark_variable_ready → flat buffer copy → mark_bucket_ready → allreduce (async) → copy back

FSDP Forward:
  pre_forward_unshard → all_gather → param.data=unsharded → forward → post_forward_reshard → param.data=sharded

NCCL AllReduce:
  ncclGroupStart → ncclAllReduce(input,output,ncclComm,stream) → ncclGroupEnd
    → algorithm selection (Ring/Tree/CollNet)
    → channel allocation (N parallel pipelines)
    → protocol selection (LL/LL128/Simple)
    → GPU kernel (data copy/reduce) ↔ Proxy thread (DMA/network)
    → Transport (NVLink/PCIe/IB)

ProcessGroupNCCL dispatch:
  user stream → recordEvent → NCCL stream waitEvent → collective → recordEvent → user stream waitEvent
```

## RTX 4090 Specifics

- **PCIe bottleneck**: AllReduce 2.76GB/s实测 (理论15.5%) → DDP 8GPU=0.46x → 单GPU最优
- **NCCL config**: NCCL_P2P_DISABLE=1, NCCL_IGNORE_DISABLED_P2P=1, NCCL_SHM_DISABLE=0, NCCL_MAX_NRINGS=4, NCCL_ALGO=RING, NCCL_PROTO=Simple
- **No NVLink P2P**: SHM(PCIe) transport only, Ring algorithm, protocol irrelevant (bandwidth bottleneck)
- **Training**: LoRA + gradient checkpointing on single GPU (14.6GB fits 24GB)
- **DDP useless**: 8GPU DDP 0.46x → use single GPU + CPU optimizer offload instead
- **FSDP not applicable**: PCIe all_gather 1.5s → worse than single GPU
- **ZeRO-2 + offload**: Best for full finetune if single GPU (CPU stores optimizer states)

## NCCL Environment Variables Quick Reference

| Variable | RTX 4090 | NVLink GPU | Purpose |
|----------|----------|------------|---------|
| NCCL_P2P_DISABLE | 1 | 0 | Disable P2P (4090 has none) |
| NCCL_IGNORE_DISABLED_P2P | 1 | 0 | Skip P2P check |
| NCCL_MAX_NRINGS | 4 | 8 | Channel count |
| NCCL_ALGO | RING | auto | Algorithm |
| NCCL_PROTO | Simple | auto | Protocol |
| NCCL_DEBUG | WARN | WARN | Log level |
| NCCL_ASYNC_ERROR_HANDLING | 1 | 1 | Enable async error handling |
| NCCL_ENABLE_MONITORING | 1 | 1 | Enable HeartbeatMonitor |

## Common Issues & Debugging

1. **NCCL timeout**: Watchdog detects → ncclCommAbort → check TORCH_NCCL_ASYNC_ERROR_HANDLING + TORCH_NCCL_ENABLE_MONITORING
2. **PCIe scaling disaster**: RTX 4090 8GPU AllReduce slow → use NCCL_P2P_DISABLE=1 + single GPU if possible
3. **DDP gradient sync stuck**: Check bucket_cap_mb, find_unused_parameters, static graph mode
4. **FSDP OOM**: Check reshard_after_forward policy, add CPU optimizer offload
5. **NCCL SHM transport**: RTX 4090 only has SHM → NCCL_SHM_DISABLE=0 (must keep enabled)
6. **Watchdog hang**: HeartbeatMonitor detects → dumps trace → aborts process → check logs

## Weight Format Pipeline

```
Training: pickle (.pt) — optimizer+scheduler+rng — security risk!
  → Distribution: safetensors (.safetensors) — mmap+zero-copy — HF standard
  → Inference: safetensors BF16 → runtime FP8 quantize, or AWQ/GPTQ INT4 safetensors → vLLM
```

Safetensors spec: 8B header length (uint64 LE) + N bytes JSON metadata + contiguous data buffer
mmap: 0.1ms vs pickle: 30s → 300,000x faster, multi-process shared physical memory (14GB shared vs 8×14=112GB)