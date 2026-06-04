# MoE 路由模拟 GPU 实验

> 文件: `tools/gpu_moe_routing.py`
> 日期: 2026-06-04
> GPU: A16 15GB

## 实验 1: Router Top-K 选择 + 负载均衡

| Strategy | Load Std | Max/Min Ratio | Entropy |
|----------|----------|:---:|---------|
| random_init | 0.67 | 1.18 | 1.95 |
| balanced | 1.13 | 1.29 | 1.94 |
| **imbalanced** | **6.38** | **4.10** | 1.82 |

- Aux Loss (Z-loss): 0.34 — logit magnitude regularization
- Load Balance Aux Loss: 1.0003 (1.0 = perfect balance)

---

## 实验 2: Dispatch/Combine 性能

| Batch | Serial ms | Batched ms | Speedup |
|:---:|---:|---:|:---:|
| 256 | 102.7 | 102.4 | 1.00x |
| 512 | 158.1 | 157.5 | 1.00x |
| 1024 | 331.5 | 333.1 | 1.00x |
| 2048 | 480.8 | 482.9 | 1.00x |
| 4096 | 1006.6 | 1008.4 | 1.00x |

**结论**: 朴素实现下 serial 和 batched 几乎相同（都在同一个 for-e 循环中）。真正的 dispatch 开销来自 indexing/scatter + combine 的 reduce-add。

---

## 实验 3: 负载不均衡对吞吐的影响

| Distribution | Max Tok/Exp | Min Tok/Exp | Time |
|--------------|:---:|:---:|---:|
| Uniform (1/8 each) | 256 | 256 | 306.8ms |
| Mixtral-style | 517 | 81 | 298.2ms |
| **Skewed** (80%→0,1) | **1029** | **40** | **271.6ms** |

**意外发现**: 倾斜分布反而更快！原因: GPU 上大 batch 处理单个 expert 效率更高（大 GEMM 利用率 > 多个小 GEMM）。

实际问题: straggler expert 在分布式场景成为延迟瓶颈，但在单 GPU 上反而是效率优势。

---

## 实验 4: All-to-All 通信 vs 计算

| Hidden Dim | Compute ms | All2All ms | Comm% |
|:---:|---:|---:|:---:|
| 1024 | 0.9 | 0.11 | 10.7% |
| 2048 | 3.3 | 0.12 | 3.6% |
| 4096 | 14.3 | 0.23 | 1.6% |
| 8192 | 85.5 | 0.43 | **0.5%** |

**结论**: Compute 增长 O(D²)，Comm 增长 O(D)。D≥4096 时通信占比 <2%，推理几乎纯 compute-bound。

注意: A16 上 A2A 是模拟（clone），实际 PCIe 通信会更慢。

---

## 实验 5: EP 配置对比

| E | EP | Tok/Rank | Exp/Rank | Comp ms | A2A ms | Ratio |
|:---:|:---:|:---:|:---:|---:|---:|:---:|
| 8 | 1 | 2048 | 8 | 2163 | 0.73 | 2950x |
| 8 | 2 | 1024 | 4 | 540 | 0.37 | 1447x |
| 8 | 4 | 512 | 2 | 136 | 0.20 | 678x |
| 16 | 4 | 2048 | 4 | 1086 | 0.74 | 1465x |
| 16 | 8 | 2048 | 8 | 2173 | 0.75 | 2897x |

**Compute/A2A = 678x-2950x** — 推理时 compute 绝对主导。

---

## 关键洞察

1. **Router Top-K**: DeepSeek/Mixtral 都用 Top-2，Aux Loss 防止路由坍塌
2. **负载倾斜的悖论**: 单 GPU 上倾斜反而快（大 batch 效率高），分布式下才是问题（straggler）
3. **All-to-All 被放大**: O(D²) compute vs O(D) comm → D>4096 时通信可忽略
4. **EP 决策**: Compute/A2A >5x 时 EP 才有效，推理时这个比例通常 >>100x
5. **容量因子**: CF=1.25 是实际部署的缓冲区标准（防止 token dropping）
