# LLM 推理性能优化技术全景图

> 整合 vLLM/SGLang/TensorRT-LLM 源码阅读 + RTX 4090 实测数据
> 最后更新: 2026-06-05

## 1. 优化层级

```
┌───────────────────────────────────────────┐
│  L7: 系统级 (Disaggregated Serving)       │  P/D分离, 多实例调度
│  L6: 服务级 (Continuous Batching)          │  调度策略, 优先级, 抢占
│  L5: 请求级 (Prefix Caching, Spec Decode) │  KV复用, 推测解码
│  L4: 模型级 (量化, 蒸馏, MoE)             │  FP8/INT4, 知识蒸馏
│  L3: 算子级 (FlashAttention, Triton)       │  自定义kernel, 融合
│  L2: 内存级 (Paged KV, KV Offload)         │  分页管理, SSD卸载
│  L1: 硬件级 (CUDA Graph, Tensor Core)      │  Graph捕获, FP16加速
└───────────────────────────────────────────┘
```

## 2. 各层级详细优化

### L1: 硬件级
| 技术 | 原理 | 实测效果 |
|------|------|----------|
| Tensor Core (FP16) | 矩阵专用加速器 | 2.76x over FP32 (RTX 4090) |
| CUDA Graph | 捕获kernel序列, 消除launch开销 | 1.02x@4090(快), 5.4x@A16(慢) |
| Multi-Stream | 多CUDA流并行执行 | 0.93x@4090(GPU饱和), 10x@小kernel |

### L2: 内存级
| 技术 | 原理 | 效果 |
|------|------|------|
| Paged KV Cache | block_size=16分页, O(1)分配/释放 | 消除碎片, 支持 preemption |
| KV Offload | CPU/SSD存储冷KV | seq>32K时有效, 6-50x容量扩展 |
| Hybrid KV Cache | Full+SW分区, 短序列用SW | 42% VRAM节省 |

### L3: 算子级
| 技术 | 原理 | 实测效果 |
|------|------|----------|
| FlashAttention | tiling + online softmax, O(N)内存 | 4.3-8.5x加速 |
| Triton GEMM | 自定义矩阵乘kernel | 98.8% cuBLAS性能 |
| Fused Softmax | Triton融合kernel | 3x加速 (4096²) |
| Fused LayerNorm | Triton融合kernel | ≈PyTorch (已优化) |
| Gumbel-Max采样 | probs/exp().argmax() | 消除CPU-GPU同步 |
| Rejection Sampling | Triton GPU端spec decode | 无CPU介入 |

### L4: 模型级
| 技术 | 原理 | 效果 |
|------|------|------|
| FP8量化 | E4M3格式, 专用Tensor Core | <0.5%精度损失, 2x内存节省 |
| INT4 AWQ | 激活感知权重量化 | ~4%损失, 4x内存节省 |
| GQA | 减少KV head数量 | 4x KV节省, 质量接近MHA |
| MLA | DeepSeek-V2 latent压缩 | 16-32x KV压缩 |
| MoE | 稀疏专家激活 | 18x稀疏比, 吞吐∝active_params |
| 知识蒸馏 | 70B→8B teacher-student | 8.75x压缩, ~80%质量 |

### L5: 请求级
| 技术 | 原理 | 实测效果 |
|------|------|----------|
| Prefix Caching | 共享前缀只计算一次 | 70-91%计算节省(GRPO/多轮) |
| Speculative Decoding | 小模型猜, 大模型验证 | 2.4x@B=1, 无效@B=128 |
| RadixAttention | 树形前缀共享(SGLang) | 无block对齐限制, 5x多轮加速 |
| Structured Output | Bitmask FSM过滤无效token | XGrammar(C++)最快 |
| Chunked Prefill | 长序列分块, 与decode混合 | TTFT 13s→94ms |

### L6: 服务级
| 技术 | 原理 | 效果 |
|------|------|------|
| Continuous Batching | 请求级调度, 不等batch填满 | 10-100x TTFT改善 |
| FCFS + Priority | 先来先服务+优先级覆盖 | 低延迟优先 |
| Preemption | KV block抢占, swap/recompute | 防OOM, recompute更快(A16) |
| Auto-fit | binary search调整block数 | ~20步收敛 |

### L7: 系统级
| 技术 | 原理 | 效果 |
|------|------|------|
| P/D Separation | Prefill/Decode独立实例 | 各自优化, KV transfer |
| KV Transfer | NIXL/RDMA跨实例传KV | <4ms (NVLink) |
| DP Serving | 多实例负载均衡 | MoE EP/独立实例 |
| UBatch/DBO | Micro-batch + 通信计算重叠 | 50%通信隐藏 |

## 3. 优化选择决策树

```
目标: 降低延迟 / 提高吞吐
│
├─ Decode太慢 (memory-bound)?
│   ├─ 量化: FP8 > INT4 > FP16
│   ├─ 减KV: GQA > MLA > MQA
│   ├─ Speculative Decoding (低batch)
│   └─ KV Offload (长序列)
│
├─ Prefill太慢 (compute-bound)?
│   ├─ Chunked Prefill
│   ├─ Prefix Caching (重复前缀)
│   ├─ FlashAttention
│   └─ Tensor Core利用
│
├─ 并发不够 (容量限制)?
│   ├─ Paged KV (消除碎片)
│   ├─ GQA/MLA (减KV大小)
│   ├─ Continuous Batching
│   └─ 更多GPU (P/D分离)
│
└─ 多卡效率低?
    ├─ TP (NVLink必需)
    ├─ PP (batch_queue填充pipeline)
    ├─ EP (MoE专家并行)
    └─ DP (独立实例负载均衡)
```

## 4. RTX 4090 vs A100 vs H100 推理成本分析

| GPU | FP16 TFLOPS | HBM BW | VRAM | 推理$| 适用场景 |
|-----|-------------|--------|------|----------|----------|
| RTX 4090 | 170 | 891 GB/s | 24GB | - | 开发/测试, 小模型 |
| A100 | 312 | 2030 GB/s | 80GB | $0.16-0.30 | 生产部署 |
| H100 | 990 | 3352 GB/s | 80GB | $0.09-0.30 | 大规模服务 |

**关键发现**:
- RTX 4090 无NVLink → 多卡TP不适合推理 (PCIe仅5GB/s)
- RTX 4090 单卡 decode 吞吐高 (529K tok/s@B=512)
- H100 FP8 Tensor Core → $0.09/M tok (MoE最优)
