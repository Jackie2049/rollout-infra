# AI Infra Engineer's Quick Reference Card

> 7框架核心参数速查 | RTX 4090最优配置 | 关键决策树

## 1. 训练策略决策树 (3秒决策)

```
GPU数量=1 且 GPU<24GB?
  → LoRA + CPU_Adam (DeepSpeed ZeRO-2) 或 GRPO + LoRA (verl/rLLM)
  → ★ RTX 4090最优: rLLM TinkerBackend + GRPO + LoRA-32 + bypass_mode

GPU数量>=2 且 NVLink?
  → FSDP2 + compile(reduce-overhead) + BF16
  → ★ H100最优: FSDP2 + compile + Float8 → +48-50% MFU

GPU数量>=2 且 PCIe only?
  → DDP + LoRA (单GPU足够) 或 ZeRO-2 + CPU offload (2GPU分参数)
  → ★ PCIe: TP/PP/ZeRO-3全部灾难级慢 → 避免分布式通信!

需要critic (PPO)?
  → 需要2×模型内存 → RTX 4090 7B PPO=48GB ✗ → 不可行
  → ★ 用GRPO代替 → 无critic → 17GB ✓ → 单GPU可行

MoE训练?
  → DeepSpeed AutoEP (PR#7938, 零代码) → ZeRO-0/1/2 + EP
  → Megatron DeepEP/HybridEP → 需SM90 → RTX 4090 ✗
  → ★ RTX 4090: EP=1 → LoRA → DeepSpeed ZeRO-2 + AutoEP
```

## 2. 推理策略决策树 (3秒决策)

```
推理吞吐最优先?
  → vLLM V1 + INT4 + INT8KV + GQA-8 + prefix caching → 4,791 tok/s (7B)
  → SGLang + INT4 + RadixAttention → 多轮对话5x加速

推理+speculative decoding?
  → EAGLE + INT4 → 9,088 tok/s → RTX 4090最优!
  → MTP (shared layer) → 最轻量 → ~0.5-1GB额外
  → Draft model → 7GB额外 → RTX 4090不可行

GRPO rollout推理?
  → rollout_n=8 → SGLang system prompt KV只计算1次 → 7×省prefill
  → ★ SGLang更适合GRPO rollout → prefix复用更强

昇腾NPU推理?
  → vLLM-Ascend + 6种attention + BudgetRefiner SLO + block_size=128
  → MindIE核心(ATB) → 不开源 → vLLM-Ascend替代
  → ★ RTX 4090不适用 → NVIDIA GPU → 用vLLM/SGLang
```

## 3. torch.compile 决策树 (3秒决策)

```
训练?
  → torch.compile(mode='reduce-overhead', fullgraph=False)
  → ★ FSDP2+compile = 最佳组合; ZeRO-3+compile = 不兼容!
  → ★ 首次编译慢 → 但缓存后快 → 2.8默认启用缓存

推理?
  → torch.compile(mode='max-autotune', fullgraph=True)
  → ★ 首次编译5-10min → 缓存后零开销
  → INT4模型 → Triton收益有限(~5-10%)

graph breaks太多?
  → 1. 数据依赖控制流 → 用torch.where/torch.cond(2.12+)
  → 2. FSDP2 hooks → intentional breaks → fullgraph=False
  → 3. ZeRO-3 → 不兼容 → 用FSDP2替代
  → 4. 不支持op → torch._dynamo.explain()诊断

recompilation风暴?
  → Symbolic Shapes(2.7+) → 1次编译覆盖所有seq_len
  → cache_size_limit=64 → 超过fallback → 调大或固定shape
```

## 4. RTX 4090最优配置 (按场景)

### 训练 (7B模型)
```
框架: rLLM TinkerBackend / verl HYBRID
算法: GRPO (不是PPO!)
LoRA: rank=32, target=all(attn+mlp+unembed)
精度: BF16 (不是FP16!)
优化: CPU_Adam (DeepSpeed ZeRO-2) 或 LoRA_Adam (Tinker)
compile: reduce-overhead (可选)
batch: train_batch=32, rollout_n=8
内存: ~17GB ✓ (24GB内)
```

### 推理 (7B模型)
```
框架: vLLM V1 或 SGLang
量化: INT4(W4A16) + INT8 KV cache
attention: FlashInfer + GQA-8
block_size: 16 (vLLM) / prefix caching ON
speculative: EAGLE (不需要draft model!)
KV blocks: ~5K for 7B (INT8KV+GQA)
吞吐: 4,791 tok/s (INT4) / 9,088 tok/s (EAGLE+INT4)
内存: ~5GB weights + ~8GB KV + ~2GB activation = 15GB ✓
```

### 研究 (多框架对比)
```
FSDP2研究: 单GPU EP=1 → 无分片意义 → 需多GPU
MoE研究: EP=1 → 所有expert本地 → 可运行但不能测试EP通信
ZeRO研究: ZeRO-2+LoRA → 可行; ZeRO-3 → PCIe灾难
Megatron研究: TP>1/PP>1 → PCIe瓶颈 → 单GPU+recompute
compile研究: reduce-overhead → 本地CPU编译慢 → 需GPU
```

## 5. 7框架关键参数速查

| 参数 | DeepSpeed | Megatron | vLLM | verl | MindIE | rLLM | PyTorch |
|------|-----------|----------|------|------|--------|------|---------|
| 分布式 | ZeRO(allreduce/RS) | TP/PP/EP/SP/CP | TP(推理) | Ray NCCL | HCCL | 无(单GPU) | FSDP2(DTensor) |
| 内存管理 | ds_tensor状态机 | 无显式管理 | Paged KV(block) | sleep/wake | CaMem | SamplingClient | DTensor shard |
| 通信量/step | 3Ψ(ZeRO-3)/2Ψ(ZeRO-2) | Ψ(TP)/Ψ/PP(P2P) | 无训练通信 | naive=0/Hybrid=0 | HCCL | 0 | 2Ψ(FSDP2) |
| compile兼容 | ZeRO-3✗/ZeRO-2? | 部分✓ | 不需要(推理) | FSDP2✓ | ✗ | ✗(Tinker) | FSDP2✓/ZeRO-3✗ |
| checkpoint | universal→HF | dist_ckpt/TRT-LLM | 内部HF | FSDP shard→HF | ATB格式 | save_pretrained→HF | DTensor state→HF |
| RTX 4090可行 | ZeRO-2+LoRA✓ | 单GPU✓ | INT4推理✓ | GRPO+LoRA✓ | ✗(NPU) | GRPO✓ | FSDP2单GPU✗/LoRA✓ |
| LoRA支持 | ZeRO兼容,PEFT不兼容 | 无原生 | Punica SGMV | 原生+ref_in_actor | 无 | Tinker auto-init | FSDP2+LoRA |

## 6. 通信量速查 (7B BF16, 8GPU)

| 策略 | AllGather | ReduceScatter | AllReduce | P2P | 总/step | PCIe时间 |
|------|-----------|--------------|-----------|-----|---------|----------|
| DDP | 0 | 0 | 14GB | 0 | 14GB | 0.58s |
| ZeRO-2 | 0 | 14GB | 0 | 0 | 14GB | 0.58s |
| ZeRO-3 fwd | 14GB | 0 | 0 | 0 | 14GB | 0.58s |
| ZeRO-3 total | 42GB(3AG) | 14GB | 0 | 0 | 56GB | 2.33s |
| FSDP2 total | 28GB(2AG) | 14GB | 0 | 0 | 42GB | 1.75s |
| TP | 14GB | 0 | 14GB | 0 | 28GB | 1.17s |
| PP | 0 | 0 | 0 | 14GB | 14GB | 11s!(PCIe) |
| LoRA+DDP | 0 | 0 | 0.1GB | 0 | 0.1GB | 4.56ms✓ |
| Tinker | 0 | 0 | 0 | 0 | 0 | 0✓ |

## 7. Checkpoint → 推理 最简路径

```
★ 最简: verl/rLLM GRPO+LoRA → merge → HF → INT4 vLLM → 4,791 tok/s
4步: ZeRO-3 → FP32 → HF → INT4 vLLM
3步: FSDP2 → FSDPModelMerger → HF → INT4 vLLM
2步: Megatron → TRT-LLM export → deploy
1步: rLLM Tinker → save_pretrained → HF → deploy
0步: SGLang/vLLM → 直接加载HF → 零转换
```

## 8. 关键数字速查

```
7B BF16 weights = 14GB
7B INT4 weights = 3.5GB
7B LoRA r=32 = ~500MB adapters
7B Adam(LoRA) FP32 = ~1-2GB
7B KV cache per token = 512KB (BF16) / 256KB (INT8)
PPO memory per GPU = ~270GB (3×model) → ✗ RTX 4090
GRPO memory per GPU = ~17GB (1×model) → ✓ RTX 4090
FSDP2 peak = 2Ψ/module (reshard=True) vs ZeRO-3 = 3Ψ
CUDA graph overhead = fixed buffers → 额外~10-20% peak
compile warmup = 10-100x首次 → 后续<1s (缓存)
Guard recompilation max = 64次 → fallback eager
```
