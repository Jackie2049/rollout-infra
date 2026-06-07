# RTX 4090 PCIe 统一决策指南 — 训练 + 推理

> 2026-06-07 | 基于8×RTX 4090 PCIe集群10+项实测的综合决策树

## 一句话总结

```
RTX 4090 PCIe = 消费级GPU, 无NVLink, 所有多GPU通信瓶颈!
→ 训练: 小模型DDP/FSDP1, 大模型单GPU+ZeRO-3, 绝不Ring Attention
→ 推理: 单GPU最优, 多GPUTP不可行(通信96%), 连续批处理是关键
```

## 训练决策树

```
┌─────────────────────────────────────────────────────────────┐
│ RTX 4090 PCIe 训练配置决策树                                │
│                                                             │
│ 模型大小?                                                   │
│                                                             │
│ <10M (3M/76K/2.28M)                                       │
│   → 单GPU训练最优!                                         │
│   → 如需多GPU: DDP (效率89-94%)                           │
│   → 不用FSDP(开销>收益)                                    │
│   → 不用checkpointing(activation<0.1%)                     │
│   → 不用Ring Attention(PCIe慢7-67x!)                      │
│   → BF16 native最佳(1.23x+39%eval+37%内存省)             │
│   → AdamW(lr=0.001, wd=0.1) + constant lr                │
│                                                             │
│ 10-100M (25M/46M)                                         │
│   → FSDP1最优! (2GPU=125%,比单GPU快!)                     │
│   → FSDP1+no checkpointing (0.478GB最低内存)              │
│   → 不用FSDP2(比FSDP1慢2x)                                │
│   → 不用DDP(AllReduce瓶颈→65%效率)                        │
│   → 不用Ring Attention(PCIe慢!)                           │
│   → Scaling限制: 4/8GPU效率退到70%/61%                    │
│   → 最佳: FSDP1 + 2GPU (125%效率!)                        │
│                                                             │
│ >100M (7B+)                                               │
│   → 单GPU+ZeRO-3(FSDP1) + BF16                           │
│   → FSDP1 49%内存省(0.329GB vs 0.639)                    │
│   → 不用checkpointing(反增内存11-32%!)                    │
│   → 不用FSDP2(composable API开销太大)                     │
│   → FSDP2+compile可尝试(14%改善但仍有gap)                │
│   → 更多GPU → FSDP1 ZeRO-3分片(DP=8→8x内存降)           │
│                                                             │
│ RL训练 (GRPO/PPO)                                         │
│   → 3.3M: DDP 89.5%效率(2GPU)                            │
│   → 46M: DDP 0.87x(更慢! 通信瓶颈56-79%)                 │
│   → → RL大模型: 单GPU+梯度累积+ZeRO-3                    │
│   → GRPO: 2模型省50%内存, 规则reward再省50%             │
│   → 7B GRPO fits单卡RTX 4090(14GB<24GB)                  │
│   → SFT暖启动是决定性因素(2x差距!)                        │
│                                                             │
│ 绝对不要:                                                  │
│   ✗ Ring Attention/序列并行 (7-67x慢)                    │
│   ✗ TP(通信96%, 无NVLink)                                │
│   ✗ PP(气泡(P-1)/(M+P-1), 少GPU意义不大)                │
│   ✗ FSDP+checkpointing (反增内存!)                       │
│   ✗ FP16+AMP(BF16 native更安全+更快)                    │
│   ✗ SGD(收敛差/loss爆炸)                                 │
│   ✗ L2正则+Adam(wd被√v缩放→灾难!)                       │
└─────────────────────────────────────────────────────────────┘
```

## 推理决策树

```
┌─────────────────────────────────────────────────────────────┐
│ RTX 4090 PCIe 推理配置决策树                                │
│                                                             │
│ 推理场景?                                                   │
│                                                             │
│ 单模型推理                                                  │
│   → 单GPU最优!                                             │
│   → TP不可行(PCIe通信96%)                                 │
│   → 7B FP16=14GB fits 24GB → 单卡可推理                  │
│   → FlashAttention省85-97%内存(防OOM)                     │
│   → Decode严重memory-bound → HBM带宽是瓶颈                │
│   → OPT-125M peak 30K tok/s@B=512                        │
│   → Continuous Batching必需(单GPU吞吐↑23x)               │
│                                                             │
│ 多模型服务                                                  │
│   → 每GPU一个模型实例(DP)                                 │
│   → 不用TP/PP(PCIe不可行)                                 │
│   → Prefix Caching(短prompt省81-96%)                      │
│   → Speculative Decoding(K=2-3最佳)                       │
│   → LoRA Multi-serve(LRU Cache 82%命中率)                 │
│                                                             │
│ 长上下文推理                                                │
│   → 单GPU+FlashAttention(内存省85-97%)                    │
│   → 不用Ring Attention/CP(PCIe慢!)                        │
│   → KV Cache是瓶颈(7B/128K=32GB→超model weight)          │
│   → 解决方案: GQA-8(KV降4x)/MLA(降56.9x)/FP8 KV(降50%) │
│   → Chunked Prefill(TTFT 13s→94ms)                       │
│                                                             │
│ 绝对不要:                                                  │
│   ✗ TP推理(PCIe通信96%, 比单GPU更慢)                    │
│   ✗ Ring Attention/CP推理(7-67x慢)                       │
│   ✗ Python-level dequant(全部比FP16慢!)                  │
│   ✗ Paged Attention Python(138%开销) → 用Triton/FlashInfer│
│   ✗ FP8训练(SM89不支持addmm_cuda)                        │
│   ✓ FP8推理需fused kernel(Marlin/compressed-tensors)     │
└─────────────────────────────────────────────────────────────┘
```

## 关键数值参考

```
RTX 4090 PCIe 关键性能数据 (实测):

计算:
  FP16 peak: 167.14 TFLOPS (101%标称!)
  FP32 peak: 53.95 TFLOPS (65%标称)
  Decode M=1: 0.75 TFLOPS (0.45%peak → 严重memory-bound)
  Ridge point: AI≈182 (FP16)

通信 (PCIe瓶颈!):
  AllReduce: P=2 7.59 GB/s, P=4 3.31, P=8 3.01
  all_gather: P=2 5.94, P=4 4.30, P=8 4.67 GB/s
  P2P: 全禁用! 无NVLink → 不支持直接GPU间通信
  NVLink对比: A100 300 GB/s → RTX 4090差50-60x!

内存:
  HBM: 24GB, 实测BW 920 GB/s
  训练: 7B = 14GB(params) + 28GB(optimizer) → 需ZeRO-3!
  推理: 7B FP16 = 14GB → fits 24GB → 单卡OK
  ZeRO-3 DP=8 → 训练内存降8x → 7B仅需14.5GB

训练效率:
  DDP <10M: 89-94%
  FSDP1 25M 2GPU: 125%!
  FSDP1 25M 4/8GPU: 70%/61%
  FSDP2: 比FSDP1慢2x(composable API开销)
  GRPO DDP 3.3M 8GPU: 58.8%
  GRPO DDP 46M 2GPU: 0.87x (更慢!)

推理吞吐:
  OPT-125M: 30K tok/s@B=512 (vLLM V1)
  Decode memory-bound → throughput∝batch_size
  Continuous Batching必需 → B=1仅2%峰值吞吐
```

## 实测支持的决策依据

```
每个决策背后的实测数据:

训练决策:
1. "<10M→DDP" → DDP 89-94%效率实测
2. "10-100M→FSDP1" → FSDP1 125%效率实测(25M 2GPU)
3. ">100M→FSDP1 ZeRO-3" → FSDP1 内存省36%实测
4. "不用checkpointing+FSDP" → checkpointing反增11-32%内存实测
5. "不用FSDP2" → FSDP2慢2x实测(composable API)
6. "不用Ring Attention" → Ring慢7-67x实测(PCIe通信39-83%)
7. "BF16 native" → BF16 1.23x加速+39%eval实测
8. "不用SGD" → SGD loss爆炸实测
9. "不用Adam+L2" → L2在Adam灾难实测(loss 173→1696)

推理决策:
10. "单GPU推理" → TP通信96%实测, 比单GPU更慢
11. "FlashAttention省内存" → 内存省85-97%实测
12. "Continuous Batching" → B=1→512吞吐↑23x实测
13. "GQA/MLA省KV" → GQA-8 KV占38%/MHA 44.5%实测
14. "不用Python dequant" → 全部0.17-0.56x实测
15. "fused kernel FP8" → Python FP8慢0.31-0.56x实测

通信决策:
16. "PCIe瓶颈5-6 GB/s" → all_gather实测5.94 GB/s(P=2)
17. "NVLink 300 GB/s" → A100 NVLink 21-52x faster实测
18. "8GPU效率退到61%" → FSDP1 scaling实测
```

## GPU选择建议

```
什么时候用RTX 4090 vs A100/H100?

RTX 4090 (24GB, PCIe, 无NVLink) 适用:
  → <10M模型训练(DDP效率高)
  → 10-100M模型训练(FSDP1 2GPU)
  → 7B单卡推理(14GB fits)
  → 小团队/个人研究(性价比高)
  → 实验验证/原型开发

A100/H100 (80GB, NVLink) 适用:
  → >100M模型训练(需NVLink+ZeRO-3)
  → Ring Attention/CP长序列训练(128K+)
  → TP推理(多GPU推理需要)
  → 生产级部署(延迟要求严格)
  → MoE EP训练(需高速All-to-All)

性价比对比:
  RTX 4090 @ $1.5K: FP16 167 TFLOPS → $1/TFLOPS
  A100 @ $15K: FP16 312 TFLOPS → $48/TFLOPS (但+NVLink+80GB!)
  → 4090计算性价比48x高! 但缺少NVLink和80GB内存
  → 选择取决于是否需要多GPU通信和内存
```

## 工具索引

```
所有相关benchmark工具:

训练相关:
  tools/fsdp2_benchmark_4090.py        — FSDP1/FSDP2/DDP训练benchmark
  tools/checkpointing_fsdp_benchmark_4090.py — Checkpointing+FSDP交互
  tools/benchmark_ddp_scaling_4090.py  — DDP scaling
  tools/benchmark_nccl_allreduce_4090.py — NCCL AllReduce带宽
  tools/benchmark_grpo_training_4090.py — GRPO训练benchmark
  tools/ring_attention_benchmark_4090.py — Ring Attention多GPU
  tools/batch_size_scaling.py           — Batch size scaling
  tools/mini_grpo_training.py           — Mini GRPO训练(7模式)
  tools/unified_rl_comparison.py        — 7方法统一RL对比

推理相关:
  tools/benchmark_flash_attention_4090.py — FlashAttention
  tools/benchmark_kv_cache_bandwidth_4090.py — KV cache BW
  tools/benchmark_decode_gemm_4090.py   — Decode GEMM
  tools/benchmark_quantization_4090.py   — 量化
  tools/continuous_batching_sim.py       — Continuous Batching模拟
  tools/inference_sim_results_a16.json  — 推理模拟结果

计算相关:
  tools/benchmark_gemm_roofline_4090.py  — GEMM Roofline
  tools/benchmark_gpu_interconnect_4090.py — GPU互连
  tools/benchmark_sampling_pipeline_4090.py — Sampling pipeline
  tools/benchmark_cuda_graph_4090.py     — CUDA Graph