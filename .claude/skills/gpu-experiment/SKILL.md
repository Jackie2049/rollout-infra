---
name: gpu-experiment
description: Run GPU experiments on matpool.com A16 server. Pushes script to GPU, runs it, pulls results, creates experiment notes.
triggers:
  - run gpu experiment
  - gpu benchmark
  - gpu test
  - 跑GPU实验
  - GPU benchmark
  - A16 benchmark
---

# GPU Experiment Runner

## GPU Server Connection

```
Host: hz-t3.matpool.com
Port: 28959
User: root
SSH: SSHPASS='TUR]Nr3fyxM%7)iD' sshpass -e ssh -p 28959 -o StrictHostKeyChecking=no root@hz-t3.matpool.com
SCP: SSHPASS='TUR]Nr3fyxM%7)iD' sshpass -e scp -P 28959 <local> root@hz-t3.matpool.com:<remote>
Conda: source /root/miniconda3/bin/activate myconda
GPU: A16 15GB, CUDA 11.7, Driver 510.54, 10 SM
```

**IMPORTANT**: Always check GPU availability first with `nvidia-smi`. If unreachable, GPU is offline.

## Available Experiments

### Core Benchmarks
| Script | Description | Key Findings |
|--------|-------------|--------------|
| `gpu_benchmark.py` | GEMM + HBM bandwidth | 15.1 TFLOPS peak, ~170GB/s HBM |
| `gpu_micro_benchmark.py` | Kernel launch, memory copy | 6.8us launch overhead |
| `gpu_kernel_tuning.py` | LayerNorm, Softmax, GEMM fusion | LayerNorm 8.2x fusion speedup |
| `gpu_cuda_streams.py` | Multi-stream overlap | A16: 0.75x (worse!) dual-stream |
| `gpu_cuda_graph_deep.py` | CUDA Graph deep analysis | 8.13x speedup for 100 ops |
| `gpu_profiling_a16.py` (renamed from gpu_profile_experiment) | A16 capabilities | A16 ≈ 1/10 A100 performance |

### Inference
| Script | Description | Key Findings |
|--------|-------------|--------------|
| `gpu_inference_sim.py` | LLM inference simulation | 125M: 291 tok/s→13K@B=512 |
| `gpu_llm_inference.py` | End-to-end LLM latency | Decode always memory-bound |
| `gpu_gqa_decode.py` | GQA decode benchmark | Grouped KV 87.5% load reduction |
| `gpu_paged_attention.py` | Paged vs naive attention | Python overhead 145-332% |
| `gpu_speculative_decode.py` | Speculative decoding | Optimal K=3 at p=0.7 |
| `gpu_spec_decode_efficiency.py` | Spec decode detailed | T=sharp→5.1x, draft cost must be <0.15x |
| `gpu_quant_inference.py` | Quantized inference | FP16 4-5x over FP32 |
| `gpu_quant_accuracy.py` | Quantization accuracy | KV INT8 cos_sim>0.9999 |
| `gpu_kv_offload.py` | KV cache offloading | CPU offload +345-1384% latency |
| `gpu_lora_serving.py` | LoRA serving benchmark | Merged 1.71x faster than on-the-fly |
| `gpu_scheduler_sim.py` | Scheduler simulation | A16/2500 blocks: max 78 concurrent |
| `gpu_prefix_caching.py` | Prefix caching benchmark | GRPO n=8 saves 68% compute |
| `gpu_prefix_cache_workload.py` | Workload simulation | Multi-turn 16→80.4% hit rate |
| `gpu_model_distillation.py` | Model distillation | 3.3x compress→4.0x latency reduce |

### Training
| Script | Description | Key Findings |
|--------|-------------|--------------|
| `gpu_megatron_tp_bench.py` | Megatron TP benchmark | TP=4: 95% efficiency |
| `gpu_moe_routing.py` | MoE routing analysis | Compute/A2A≥678x on A16 |
| `gpu_moe_sim.py` | MoE serving simulation | Active params = throughput bottleneck |
| `gpu_grad_reduce_step.py` | Grad AllReduce + optimizer | Adam=75% memory, GA=16 saves 39% |
| `gpu_gradient_checkpoint.py` | Gradient checkpointing | 40% memory save, 14% compute overhead |
| `gpu_mini_train.py` | Mini training loop | AMP BF16 safest, FP16 w/o AMP diverges |
| `gpu_training_latency_model.py` | Training latency model | Ridge point 86.5 ops/byte |
| `gpu_training_practice.py` | Training practice | 16B/param for Adam training |
| `gpu_training_practice_v2.py` | Training practice v2 | Full training loop |

### Communication
| Script | Description | Key Findings |
|--------|-------------|--------------|
| `gpu_comm_compute_overlap.py` | Comm-compute overlap | A16 dual-stream SLOWER (0.75x) |
| `gpu_distributed_sim.py` | Distributed training sim | AllReduce cost scales with model |
| `gpu_multigpu_sim.py` | Multi-GPU simulation | NVLink comm <2% |

## Standard Workflow

1. **Check GPU**: `nvidia-smi` via SSH
2. **Push script**: `scp` the script to `/root/`
3. **Run**: `conda activate myconda && python script.py`
4. **Capture output**: Save to results file
5. **Create note**: `notebook/experiments/gpu-<name>.md`
6. **Update MEMORY.md**: Add key findings

## Constraints

- **CUDA 11.7**: No Triton 3.x, no FlashInfer, use PyTorch C++ Extensions
- **10 SM**: Dual-stream overlap is counterproductive (0.75x)
- **15GB VRAM**: Max ~878M params without ZeRO for training
- **PCIe only**: No NVLink, ~170GB/s HBM bandwidth
- **HF offline**: `export HF_HUB_OFFLINE=1` needed
