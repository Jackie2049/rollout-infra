# GPU 实验汇总: LoRA Serving / CUDA Graph / Scheduler

> 日期: 2026-06-04
> GPU: NVIDIA A16 (15GB), CUDA 11.7, PyTorch 2.5.1

## 1. LoRA Serving (`tools/gpu_lora_serving.py`)

### LoRA Rank 对推理性能的影响
- Baseline (无 LoRA): 0.104ms
- Rank 1-128: 0.178-0.199ms, 开销 71-92%
- **Rank=8 性价比最优**: 0.178ms, 71% 开销, 仅 24KB

### 合并策略
| 策略 | 耗时 | 说明 |
|------|------|------|
| On-the-fly | 0.178ms | base + lora, 3次 matmul |
| **Merged** | **0.104ms** | 1次 matmul, 1.71x 加速 |
| Two-step | 0.181ms | base + x@A + xA@B |

**结论**: LoRA 合并到 base weight 最快，但切换 adapter 有 merge 开销。

### Multi-LoRA Batched
- Sequential vs Batched: batch_size>4 时 Batched 更快
- vLLM Punica kernel 用 segmented matmul 实现

### LoRA Cache (LRU)
| Cache Size | Hit Rate | GPU Mem |
|-----------|----------|---------|
| 4 | 11% | 0.1 MB |
| 16 | 38% | 0.4 MB |
| 32 | 55% | 0.8 MB |
| **64** | **82%** | 1.6 MB |

Zipf 分布下缓存 64 个 adapter 即达 82% 命中率。

### 精度对比
- FP16 LoRA: max_err=0.06 (可接受)
- **BF16 LoRA: max_err=0.47** (不可接受! A16 BF16 有精度问题)

## 2. CUDA Graph Deep (`tools/gpu_cuda_graph_deep.py`)

### Launch 开销消除
| Ops | Eager (ms) | Graph (ms) | Speedup |
|-----|-----------|-----------|---------|
| 5 | 0.062 | 0.013 | 4.86x |
| 10 | 0.142 | 0.021 | 6.85x |
| 50 | 0.757 | 0.093 | 8.13x |
| **100** | **1.491** | **0.183** | **8.13x** |

- Launch 开销: ~13us/op
- 1 op 时 Graph 反而更慢 (capture 开销 > launch 开销)

### Graph Pool (vLLM 策略)
- 预录制不同 batch size 的 graph
- 所有 batch size 均获 3.5-4.4x 加速
- 需要额外内存存储中间 activation

### Graph Break 分析
| 场景 | 耗时 | 相对开销 |
|------|------|---------|
| No break (graph) | 0.013ms | 1.0x |
| One break (eager) | 0.040ms | 3.1x |
| Two breaks (eager) | 0.100ms | 7.8x |
| 5 ops (graph) | 0.026ms | — |
| 5 ops (eager) | 0.164ms | 6.3x |

- **CPU sync (.item()) 开销 ≈ 44us/次**
- 5 个小 op 用 Graph 可消除 84% 开销

### vLLM 应用
- FULL_AND_PIECEWISE 模式: 大部分 op 录制为一个 graph, 部分不可录制的单独执行
- Decode step ~10 个 kernel → Graph 节省 ~68us/step
- 避免 graph break 是关键优化

## 3. vLLM V1 Scheduler (`tools/gpu_scheduler_sim.py`)

### FCFS Throughput
| Prompt | Output | Throughput | Max Queue |
|--------|--------|-----------|-----------|
| 128 | 64 | 23.7 tok/step | 69 |
| 256 | 128 | 14.2 tok/step | 82 |
| 512 | 256 | 7.7 tok/step | 89 |

Prefill 越长 → 吞吐越低 (block 分配压力大)

### Block 容量 vs 并发
| GPU | Seq=512 | Seq=4096 |
|-----|---------|----------|
| A16-15GB | 78 | 9 |
| A100-80GB | 128+ | 128+ |

### Decode Batch Scaling (GPU 实测)
- B=1: 9.7K tok/s
- B=64: 282K tok/s
- **B=128: 301K tok/s** (memory-bound, 完美线性)

### Preemption: Recompute vs Swap
| SeqLen | Recompute | Swap | Winner |
|--------|-----------|------|--------|
| 128 | 0.048ms | 0.080ms | Recompute |
| 512 | 0.039ms | 0.243ms | Recompute |
| 1024 | 0.046ms | 0.469ms | Recompute |

GPU 实测: Recompute 始终优于 Swap (PCIe 传输是瓶颈)

### SLO Tracking
| 负载 | TTFT p99 | TPOT p99 | TPOT 违规 |
|------|---------|---------|-----------|
| Low (2/s) | 0.0s | 0.01s | 0% |
| Medium (5/s) | 0.0s | 0.02s | 0% |
| **High (15/s)** | 0.0s | **0.83s** | **43.5%** |

高负载时 TPOT 违规严重 → 需要 prefill/decode 分离调度
