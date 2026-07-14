# PyTorch Distributed Internals: GRPO Relevance

## Overview
4-layer stack: `Application → torch.distributed → ProcessGroupNCCL → NCCL`. Understanding this stack is critical for diagnosing GRPO training issues.

## RTX 4090 Critical Findings

| Finding | Detail | GRPO Impact |
|---------|--------|-------------|
| **DDP useless on 8×RTX 4090** | PCIe bottleneck → 8GPU = 0.46x single | Single GPU optimal |
| **FSDP worse** | PCIe all_gather = 1.5s | Worse than single GPU |
| **ZeRO-2 + offload** | CPU stores optimizer states | Best for full finetune on single GPU |
| **NCCL P2P disabled** | No NVLink → NCCL_P2P_DISABLE=1 | Only SHM(PCIe) transport |
| **LoRA + grad ckpt** | 14.6GB fits 24GB | Viable on single RTX 4090 |

## FSDP2 Lifecycle (for H20-3e partner server)
```
pre_forward_unshard → all_gather → param.data=unsharded → forward
→ post_forward_reshard → param.data=sharded → backward → reduce_scatter
```
3 reshard strategies: NO_SHARD / SHARD_GRAD_OP / FULL_SHARD.

**For H20-3e (NVLink)**: FSDP2 works well with NVLink interconnect. Our verl #7005 PR (FSDP2 staging skip → 50x export speedup) directly optimizes this path.

## DDP Gradient Bucketing
- `bucket_cap_mb=25` (default), reverse parameter order
- `autograd_hook → mark_variable_ready → mark_bucket_ready → allreduce`
- Communication-compute overlap via async allreduce + copy back

## ProcessGroupNCCL Architecture
- ncclComm lazy init + cached reuse
- WorkNCCL async model with CUDA event tracking
- Watchdog + HeartbeatMonitor dual monitoring (10min op timeout)
- Stream-per-device double-barrier event model

## NCCL for RTX 4090

```bash
NCCL_P2P_DISABLE=1   # No P2P between 4090s
NCCL_ALGO=RING       # Ring algorithm (no NVSwitch)
NCCL_PROTO=Simple    # Simple protocol for bandwidth
NCCL_MAX_NRINGS=4    # 4 channels
NCCL_SHM_DISABLE=0   # SHM must stay enabled
```

## Weight Format Pipeline
```
Training: pickle (.pt) → Distribution: safetensors → Inference: FP8 Quant
```
safetensors mmap: 0.1ms vs pickle: 30s → 300,000x faster. Critical for weight sync in verl GRPO.

## GRPO Debugging with These Internals

| Symptom | Likely Root | Check |
|---------|------------|-------|
| NaN gradients | FSDP unshard race | NCCL_DEBUG=WARN, CUDA_LAUNCH_BLOCKING=1 |
| OOM on DP>1 | FSDP reshard policy | Set reshard_after_forward=True |
| Slow multi-GPU | PCIe bottleneck | Use single GPU or NVLink |
| Hang on allreduce | NCCL timeout | TORCH_NCCL_ASYNC_ERROR_HANDLING=1 |
| Weight sync slow | pickle format | Convert to safetensors |

## Key Files
- `torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp` (~1400 lines) — NCCL process group
- `torch/csrc/distributed/c10d/reducer.{h,cpp}` — DDP gradient bucketing
- `torch/distributed/fsdp2/_fsdp_param_group.py` — FSDP2 param lifecycle
- NCCL: `src/channel.h`, `src/proxy.cc`, `src/collectives.cc`, `src/graph/topo.cc`
