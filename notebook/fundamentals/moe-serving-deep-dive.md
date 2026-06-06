# MoE (Mixture-of-Experts) Serving Deep Dive

> 2026-06-07 | 基于 DeepSeek-V3 (671B/37B), Switch Transformer, vLLM MoE serving
> 已有模拟器结果 (Load Std=6.38, A2A<2% for D≥4096, 实测 Python MoE慢5.8-26.3x)

## 1. MoE 架构: 为什么稀疏

**Dense Transformer**: 每个 token 经过所有参数 → 计算 ∝ 总参数
**MoE Transformer**: 每个 token 只经过 K 个 expert → 计算 ∝ K/N × 总参数 (N experts)

**DeepSeek-V3 关键数据**:
```
总参数: 671B (all experts combined)
激活参数: 37B (top-K=6 selected per token)
稀疏比: 671/37 ≈ 18x → 只用 5.5% 的参数处理每个 token!
KV Cache: 同 dense 37B (共享 attention + shared expert)
```

### Router 设计:

```
# Top-K routing with auxiliary loss for load balancing
router_logits = hidden @ W_router  # [B, seq, N_experts]
weights, selected_experts = topk(router_logits, K=6)
weights = softmax(weights)

# Auxiliary loss (encourage uniform load)
aux_loss = α × N × Σ_f (f_f × P_f)
where f_f = fraction of tokens routed to expert f
      P_f = fraction of router probability allocated to expert f
α = 0.01 (small, don't dominate main loss)
```

## 2. MoE Serving: 三个核心挑战

### Challenge 1: Expert Weight Loading (Memory)

```
671B × FP16 = 1342 GB → 远超单 GPU!

解决方案:
- Tensor Parallel (TP=8): 每GPU 168GB → 仍超 80GB (A100)
- Expert Parallel (EP=8): 每GPU存 671B/8=84B → 168GB → 仍超
- TP=8 + EP=8 (混合): 671B / 64 = 10.5B per GPU → 21GB → OK!

实际 DeepSeek-V3 探荐: TP=8 + EP ≥ 4 → 最少 32 GPU
```

### Challenge 2: All-to-All Communication (EP)

```
EP 的通信模式:
1. Dispatch: 将 token 发送到对应的 expert GPU
   All-to-All: 每个GPU 发送到所有其他EP GPU
2. Expert Compute: 各GPU处理本地expert
3. Gather: 将结果发送回原始GPU
   All-to-All: 又一次 All-to-All

通信量:
Dispatch: B × seq × d × (EP-1)/EP bytes
Gather:   同上

对 DeepSeek-V3, EP=8, d=7168:
一次 All-to-All = 2 × B × seq × 7168 × 7/8 × 4 bytes
                 ≈ 50KB/token × (EP-1)

NVLink (300 GB/s): 50KB/0.3GB ≈ 167us → 快
Ethernet (25 Gbps): 50KB/3.1GB ≈ 16ms → 慢!
```

### Challenge 3: Load Imbalance

```
理想: 每个 expert 处理 B/N_experts tokens → 均匀分布
现实: "hot experts" → 某些 expert 处理远多于平均

实测: Load Std = 6.38 → 意味着某些 expert 处理 6x 平均负载

影响:
- 计算浪费: GPU 等待最慢的 expert → All-to-All 需要 all-to-all 同步
- 内存浪费: hot expert 的 KV/buffer 需要更多空间
- 解决: auxiliary loss + capacity factor + expert duplication
```

## 3. vLLM MoE Serving Pipeline

```
vLLM V1 MoE serving flow (per step):

1. Scheduler: FCFS scheduling, no special MoE handling needed
   (MoE 的调度同 Dense — 区别在 ModelRunner 内部)

2. ModelRunner.execute_model():
   a. Attention (shared): 所有 token 经过共享 attention → 无 All-to-All
   b. Router: 计算 top-K expert assignment → 本地
   c. Dispatch (All-to-All): token hidden → 对应 expert GPU
   d. Expert Compute: 本地 expert 处理 token → GEMM
   e. Gather (All-to-All): expert output → 原始 GPU
   f. Combine: weighted sum of expert outputs → 本地

3. All-to-All Backend:
   - DeepEP: NVIDIA GPU 专用, NVLink + Ethernet 混合
   - DeepEP-HT: high-throughput mode (大batch)
   - DeepEP-LL: low-latency mode (小batch, 单token decode)
   - Triton: 通用实现 (慢但兼容性好)
```

## 4. EP + TP 混合: 为什么需要

```
TP=8 (全 NVLink): 通信延迟低, 但每GPU内存压力大
EP=8 (跨节点): 通信延迟高 (Ethernet), 但每GPU内存轻

最优: TP (NVLink内) + EP (跨NVLink组)

例子: 8×NVLink组 × 4×EP组 = 32 GPU
- 每NVLink组: TP=8 → attention + shared expert 本地处理
- 跨组: EP=4 → 8 experts/组, 每GPU 2 experts → 168B/4=42B → 84GB (仍超)
→ 需要更多 GPU 或 ZeRO for expert weights

DeepSeek 实际部署: 8×TP × 8×EP = 64 GPU (或更多)
```

## 5. MoE Decode vs Prefill 性能差异

```
Prefill (长序列, compute-bound):
- Expert GEMM: [B×seq/K, d, d_mid] → 大矩阵 → GPU 高利用率
- All-to-All: 大数据块 → NVLink 带宽充分利用
- 总体: compute-bound → MoE prefill ≈ Dense prefill (吞吐 ∝ 激活参数)

Decode (单token, memory-bound):
- Expert GEMM: [B/K, d, d_mid] → 极小矩阵 → GPU 低利用率
- All-to-All: 极小数据块 → 通信延迟主导
- 总体: communication-bound → MoE decode 比Dense慢 (All-to-All overhead)

实测: 模拟器显示 Compute/A2A ≥ 678x for D≥4096 → compute dominates for large hidden
但 decode batch=1: A2A ≈ 1ms (NVLink) → significant overhead relative to compute (~0.5ms)
```

## 6. MoE 的优化策略

### Strategy 1: Expert Duplication
```
将 "hot expert" 复制到多个 GPU → 分散负载
代价: 多用内存 (duplication factor × expert_size)
收益: 负载均衡更好 → 减少等待时间

DeepSeek-V3 实际: 无 expert duplication (辅助损失足够)
```

### Strategy 2: Capacity Factor
```
每个 expert 最多处理 capacity × (B/N) tokens
超出的 token 被丢弃或路由到次优 expert

capacity = 1.0: 严格均衡 → 可能丢弃 token (quality loss)
capacity = 1.5: 允许 50% 过载 → 更少丢弃, 但更多等待
```

### Strategy 3: Micro-batching (UBatch)
```
将 token batch 切分为 microbatch → 通信和计算重叠
vLLM 的 DBO (Disaggregated Batch Optimization):
  compute_stream: micro-batch 计算
  comm_stream: All-to-All 通信
  两个stream 通过 CUDA Event 同步 → 无 CPU 介入

实测: 大GPU (A100/H100 NVLink) 通信被计算完全隐藏
小GPU (A16 PCIe) 重叠无收益 (stream切换开销346%)
```

### Strategy 4: Expert Weight Offloading
```
将冷 expert weights 存在 CPU → 只在需要时加载到 GPU
代价: 加载延迟 (PCIe ~6GB/s for 7B expert → ~2.3s)
收益: GPU 内存减少 (只存热 expert)

vLLM 目前不支持 expert offloading → 是潜在贡献方向!
```

## 7. 关键 Takeaways

1. **MoE 推理 = 稀疏激活**: 671B 模型只用 37B 参数/step → 但所有 expert weights 都需存储
2. **EP All-to-All 是关键瓶颈**: decode 每步两次 All-to-All → NVLink 快 (<1ms), Ethernet 慢 (>10ms)
3. **TP+EP 混合是最优**: NVLink 内 TP (attention + shared expert), 跨组 EP (routed experts)
4. **Load imbalance 是核心挑战**: auxiliary loss + capacity factor + expert duplication
5. **Decode 比 Dense 慢**: 通信开销显著 (特别是小batch)
6. **Prefill ≈ Dense**: compute-bound, 大GEMM 充分利用 GPU
7. **Micro-batching 隐藏通信**: 大GPU上 compute/comm 重叠有效, 小GPU无效
8. **Expert offloading 是潜在贡献**: vLLM 不支持 → 可以实现类似 KV Cache offload 的机制