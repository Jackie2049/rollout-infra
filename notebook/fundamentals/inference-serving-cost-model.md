# Inference Serving Cost Model — GPU选型+量化+部署成本分析

> 2026-06-07 | 基于Roofline模型的推理成本计算器, 5类GPU×5类模型×量化策略

## 核心公式

### Decode吞吐 (memory-bound)

```
throughput = batch_size / (weight_bytes/tp + kv_bytes) / HBM_BW
kv_bytes = 2 × layers × kv_heads × head_dim × (bits/8) × seq_len × batch
```

关键洞察:
- Decode始终memory-bound → 吞吐∝HBM带宽
- KV cache随batch×seq线性增长 → 大batch/长序列KV可能超过权重
- GQA/MQA大幅减少KV → 更多并发容量

### Prefill吞吐 (compute-bound)

```
throughput = TFLOPS × utilization / FLOPS_per_token
FLOPS_per_token ≈ 6 × params (forward only)
```

## GPU选型决策树

| 模型规模 | 最佳GPU | 配置 | $/M tok |
|----------|---------|------|---------|
| 7B FP16 | A100-80GB | TP=1, B=25 | $0.55 |
| 7B FP8 | A100-80GB | TP=1, B=57 | $0.24 |
| 70B FP16 | H200-141GB | TP=2, B=81 | $0.99 |
| 70B FP8 | H200-141GB | TP=1, B=81 | $0.34 |
| Mixtral-8x7B FP8 | A100-80GB | TP=1, MoE | $0.11 |
| DS-V3-671B FP8 | H200-141GB | TP=6+ | $13.30 |

### 关键发现

1. **7B: A100性价比最高** — $0.55/M tok FP16, $0.24 FP8
2. **70B FP8: 单H100/H200可部署** — 从TP=2→TP=1, 成本降50%
3. **MoE: FP8权重性价比极高** — Mixtral-8x7B $0.11/M tok (active params仅37B)
4. **量化是最大杠杆** — INT4→成本降97%, FP8→降40%

## 量化策略对比 (LLaMA-70B on H100)

| 方案 | TP | Weight GB | Max Batch | tok/s | $/M tok | 节省 |
|------|-----|-----------|-----------|-------|---------|------|
| FP16 | 2 | 130 | 4 | 190 | $8.75 | baseline |
| FP8 权重+KV | 1 | 65 | 2 | 101 | $8.25 | -6% |
| INT4 权重 | 1 | 32.5 | 52 | 3488 | $0.24 | -97% |

**核心**: INT4权重从TP=2→TP=1 → batch从4→52 → 吞吐18x → 成本降97%
- FP8因batch太小(仅2), 吞吐提升有限 → 需要大HBM(H200)才真正发挥FP8优势

## 典型部署场景

| 场景 | 配置 | 吞吐/卡 | 卡数 | $/小时 | $/M tok |
|------|------|---------|------|--------|---------|
| 对话API (7B,4K) | A100 TP=1 B=32 | 797 | 2 | $3.00 | $0.52 |
| 代码助手 (70B,8K) | H200 TP=2 B=16 | 1423 | 2 | $8.00 | $1.56 |
| RAG (7B,32K) | A100 TP=1 B=8 | 216 | 1 | $1.50 | $1.93 |
| RL GRPO (7B,2K) | H100 TP=1 B=64 | 2624 | 1 | $3.00 | $0.32 |
| 大模型API (70B FP8) | H100 TP=1 B=32 | 1240 | 1 | $3.00 | $0.67 |

### 场景优化建议

- **RAG/文档**: FP8量化+Prefix Caching → 90%+计算节省
- **RL训练**: H100高吞吐+GRPO无Critic → 2模型代替4模型
- **多轮对话**: SGLang RadixAttention → 前缀复用3-6x加速
- **长序列**: KV cache线性增长 → GQA/MQA必需 → MLA 32x压缩

## 能效分析 (LLaMA-7B, tok/kWh)

| GPU | TDP (W) | tok/s | tok/kWh | $/M tok | 排名 |
|-----|---------|-------|---------|---------|------|
| H200-141GB | 700 | 1880 | 9667 | $0.59 | 1 |
| A100-80GB | 300 | 797 | 9562 | $0.52 | 2 |
| A100-40GB | 250 | 609 | 8768 | $0.50 | 3 |
| H100-80GB | 700 | 1312 | 6746 | $0.64 | 4 |
| L40S-48GB | 350 | 338 | 3480 | $0.66 | 5 |
| A16-16GB | 250 | 61 | 885 | $1.36 | 6 |

**意外**: H200能效与A100持平! HBM BW 4.8TB/s × 2.4x吞吐 → 弥补700W TDP
**结论**: A100能效最优(吞吐适中+低TDP), H200吞吐最高(大batch场景首选)

## API vs Self-hosted 成本对比

| 方案 | 7B | 70B | 备注 |
|------|-----|------|------|
| OpenAI API | $0.15/M tok | $2.00/M tok | 包含运维+基础设施 |
| 自托管 A100 | $0.55/M tok | $1.00/M tok | 需运维+电力 |
| 自托管 H100 | $0.64/M tok | — | 高吞吐但成本稍高 |

**关键**: API对小模型有规模优势(共享基础设施), 自托管对大模型+高吞吐更划算

## 与PS/GRPO的联系

1. **GRPO n=8 rollout**: prefix=512→58%计算节省 → 1.87x training speedup
2. **长prompt GRPO**: prefix=4096→3.35x forward speedup → RL训练成本降低3x
3. **KV Injection**: block-causal mask验证通过(cos_sim=0.999999) → PS安全用于训练
4. **MoE PS**: DeepSeek-V3 671B/37B=18x稀疏 → PS加速主要来自skip inactive experts

## 成本优化清单

1. **量化**: FP8首选(精度损失<0.5%), INT4次选(4%损失,97%成本降)
2. **GPU**: 7B→A100, 70B→H200, MoE→A100 FP8
3. **KV**: GQA必需(长序列), MLA最优(32x压缩), Prefix Caching(RAG)
4. **并行**: 单卡优先, NVLink TP=2/4, 超大→TP+PP+EP
5. **框架**: 通用→vLLM, 前缀复用→SGLang, 延迟→TRT-LLM

Sources:
- 计算器: tools/inference_cost_calculator.py (318行)
- GPU数据: NVIDIA官方specs + 云厂商定价 (2026-06)
- 量化数据: FP8精度实测(cos_sim>0.999), INT4 AWQ/Marlin
- 之前实测: RTX 4090 7B decode 580 tok/s, KV占86%(B=32)