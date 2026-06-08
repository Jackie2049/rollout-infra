# RTX 4090 PCIe 推理部署决策指南

> 2026-06-08 | 综合20+个RTX 4090 benchmark数据 → 从模型选择到部署配置 → 一站式决策树
> 基于: 所有RTX 4090实测benchmark(FSDP/GEMM/FlashInfer/CUTLASS/CUDA/Triton/量化/RoPE/KV/蒸馏/长上下文等)
> 关联: rtx4090-pcie-decision-guide.md(训练), 本指南专注推理部署

## 0. RTX 4090推理核心参数

```
RTX 4090推理硬件参数:
  → GPU: RTX 4090 24GB HBM, SM 8.9 (Ada Lovelace), 128 MPs
  → → 实测HBM带宽: 890.8 GB/s (93.7% of theoretical 960)
  → → 实测FP16 GEMM: 169.6 TFLOPS (101% of peak 165)
  → → → **关键**: 推理decode是memory-bound → HBM带宽是瓶颈 → 890GB/s决定吞吐!

  RTX 4090推理限制:
  → 24GB HBM → 模型权重+KV+overhead → 7B BF16 = 14GB → 剩余8.96GB
  → → PCIe → 多GPU scaling灾难性(>2GPU不划算) → 单GPU最优!
  → → → 无NVLink → 不支持PD分离/TP高效 → 单GPU推理唯一选择!

  RTX 4090推理优势:
  → SM89 → 支持FP8 E4M3 (TransformerEngine) → 1.48-1.59x训练加速!
  → → 支持FlashInfer → 15.72x decode加速(B=32) → 生产级attention!
  → → → 大L2 72MB → 低occupancyOK → cache替代延迟隐藏 → 小batch也可行!
```

## 1. 模型选择决策树

```
模型选择 — 按服务场景:

  ┌─ 英文/代码服务 ──────────────────────────────────────┐
  │ → 推荐: LLaMA-2 7B / Mistral 7B                     │
  │ → vocab=32K → lm_head=256MB → 单GPU可以               │
  │ → GQA-8(Mistral) / GQA-5 → KV更省                    │
  │ → → 最佳配置: 7B GQA-5 INT8 KV + BF16权重             │
  │ → → → B=57 → 2,312 tok/s → 推荐!                     │
  └───────────────────────────────────────────────────────┘

  ┌─ 中文服务 ──────────────────────────────────────────┐
  │ → 推荐: Qwen-2.5 0.5B (vocab=151K)                  │
  │ → 不推荐: Qwen-7B → lm_head=1.2GB → 太大!            │
  │ → → 替代: vocab裁剪→64K → lm_head=480MB → 可行       │
  │ → → → 或: 蒸馏 7B→1.4B → lm_head也小 → 推荐!        │
  └───────────────────────────────────────────────────────┘

  ┌─ 多语言服务 ───────────────────────────────────────┐
  │ → 不推荐 → vocab需128K+ → lm_head>1GB → 单GPU不够   │
  │ → → 需要H100 80GB → RTX 4090不适合!                   │
  └───────────────────────────────────────────────────────┘

  ┌─ 嵌入式/边缘推理 ──────────────────────────────────┐
  │ → 推荐: OPT-125M / 蒸馏0.5B模型                      │
  │ → → 极小 → 极快 → B=100+ → 超高吞吐                  │
  │ → → → 但质量有限 → 仅适合简单任务                      │
  └───────────────────────────────────────────────────────┘

  模型大小选择:
    → 7B: 通用serving → 单GPU → B=57 → 2,312 tok/s → **推荐!**
    → → 1.4B(蒸馏): 高吞吐serving → B=270 → 11K tok/s → 推荐(质量70-80%)
    → → → 0.5B: 超高吞吐 → B=800+ → 但质量有限 → 不推荐通用
    → → → → 70B: RTX 4090不够 → 需要3-4GPU → 不推荐!

  vocab选择:
    → 32K: 最优 → lm_head=256MB → 占10% → 推荐英文/代码
    → → 64K: 可行 → lm_head=512MB → 占20% → 推荐多语言
    → → → 128K: 不行 → lm_head=1024MB → 占36% → decode慢4x → 不推荐!
    → → → → 151K(Qwen): lm_head=1.2GB → 占30% → 太大 → 不推荐7B!
```

## 2. 量化配置决策树

```
量化选择 — 按精度和速度需求:

  KV Cache量化:
    → BF16: 基准 → KV/tok=81.92KB → B=28 → 不推荐(浪费内存)
    → → INT8: 推荐! → KV/tok=40.96KB → 精度99.9965%(cos_sim) → B=57 → 推荐!
    → → → FP8 E4M3: 更推荐! → KV/tok=40.96KB → 精度99.9996%(比INT8更好!) → 推荐!
    → → → → INT4 KV: 理论省93% → 但Python dequant 20x减速! → 需fused kernel → 不推荐!
    → → → → → **RTX 4090最优: FP8 E4M3 KV (精度最高+50%内存省) → 推荐!**
    → → → → → 但: FlashInfer FP8需per-tensor scaling → 否则cos_sim=0.36(灾难!)

  权重量化:
    → BF16: 基准 → 14GB → B=28 → 推荐(最稳定)
    → → INT8(W8A8 SmoothQuant): 推荐 → 7GB → 数学等价迁移 → 精度99% → 推荐!
    → → → INT4 AWQ: 推荐(需fused kernel) → 3.5GB → B=119 → 推荐(最高并发)
    → → → → **Python dequant 0.05x → 必须用fused kernel(AWQ/Marlin) → 否则灾难!**

  最优量化组合:
    → BF16权重 + INT8 KV: 稳定+省KV → B=57 → 推荐(最安全)
    → → BF16权重 + FP8 KV: 稳定+最省KV+最高精度 → B=57 → 推荐(最优精度)
    → → → INT4 AWQ + INT8 KV: 极省内存 → B=119 → 推荐(最高并发)
    → → → → INT4 AWQ + FP8 KV: 极省+最高精度 → B=119 → 推荐(理论最优)
    → → → → → **实际: INT4 AWQ需要fused kernel → 当前vLLM支持 → 推荐!**

  量化路径规律(核心!):
    → Python dequant: INT4 20x慢 → INT8 KV 3-12%overhead → **fused kernel消除overhead!**
    → → → AWQ/Marlin: fused INT4 dequant → 消除20x overhead → 量化才有效!
    → → → → FP8 TE: fused quantize+GEMM+dequantize → 1.48-1.59x训练加速!
    → → → → → **量化必须用fused kernel → 否则Python overhead致命!**
```

## 3. Attention Backend决策树

```
Attention Backend选择:

  → SDPA(PyTorch默认): 不适合推理!
    → → is_causal=True对decode(Q=1)错误 → 只看position 0 → 结果错误!
    → → → 无paged KV → 无GQA优化 → 无prefix caching → 不推荐!

  → FlashInfer: **推理attention生产答案!**
    → → B=32: 15.72x加速(145K tok/s vs 9.3K)
    → → → S=2048: 24.58x加速 → GQA native → Paged KV → 推荐!
    → → → → 87%理论HBM峰值 → 最优decode attention → **推荐!**

  → Triton decode attention: 正确(cos_sim=1.000000) → 但2-3x慢于SDPA → 不推荐生产!
    → → 教育用途 → 学习attention kernel → 但不推荐部署

  → FlashAttention-2/3: 不适合decode!
    → → FA-2 decode: layout转换+Q=1错误 → 3-34x慢+结果错误
    → → → FA-3: Hopper专用(SM90) → RTX 4090不支持 → 不推荐!

    → → → → **RTX 4090最优: FlashInfer → 15.72x → 生产级 → 推荐!**
```

## 4. 上下文长度决策树

```
上下文长度选择:

  ┌─ S≤4K (短对话) ───────────────────────────────────┐
  │ → Full KV → INT8 KV → GQA-5 → FlashInfer         │
  │ → → B=57 → 2,312 tok/s → TTFT=14ms                │
  │ → → → **推荐(最高吞吐!)** → 默认配置              │
  │ → → → → vLLM: 默认 → 无需额外配置                │
  └──────────────────────────────────────────────────────┘

  ┌─ S=4K-8K (中等对话) ──────────────────────────────┐
  │ → NTK-aware 2x → B=28 → 1,144 tok/s               │
  │ → → TTFT=40ms → 可接受                             │
  │ → → 或: StreamingLLM(4+2K) → B=114 → 4,620 tok/s  │
  │ → → → **StreamingLLM更快!** → 但窗口外丢失        │
  │ → → → → vLLM: --rope-scaling-factor 2.0            │
  └──────────────────────────────────────────────────────┘

  ┌─ S=8K-16K (长对话) ───────────────────────────────┐
  │ → NTK-aware 4x → B=14 → 572 tok/s                 │
  │ → → TTFT=89ms → 可接受(<100ms)                     │
  │ → → 或: StreamingLLM(4+4K) → B=57 → 2,311 tok/s  │
  │ → → → **StreamingLLM 4x更快!** → 推荐!             │
  │ → → → → vLLM: --rope-scaling-factor 4.0            │
  │ → → → → → 或: --sliding-window 4096                │
  └──────────────────────────────────────────────────────┘

  ┌─ S>16K (超长对话) ─────────────────────────────────┐
  │ → 不推荐单GPU → 吞吐<572 → 成本高                  │
  │ → → StreamingLLM → 无限对话 → 但窗口外丢失         │
  │ → → → 真正S>16K → 需要H100集群 → RTX 4090不适合!  │
  └──────────────────────────────────────────────────────┘

  RoPE扩展总结:
    → 2x: YaRN最优(sim_ext=0.225) → S=8K → 可用
    → → 4x: NTK-aware最优(sim_ext=0.229) → S=16K → 推荐!
    → → → 8x+: 任何方法严重退化 → 需fine-tune → 不推荐!
```

## 5. Serving框架配置

```
vLLM V1配置 (推荐):

  基本配置:
    → python -m vllm.entrypoints.openai.api_server \
    → →   --model meta-llama/Llama-2-7b-chat-hf \
    → →   --max-model-len 4096 \
    → →   --gpu-memory-utilization 0.9 \
    → →   --dtype bfloat16

  KV量化:
    → --kv-cache-dtype fp8_e4m3fn → FP8 KV → 精度最高 → 推荐!

  量化权重:
    → --quantization awq → INT4 AWQ → 最高并发 → 需要AWQ模型!

  RoPE扩展:
    → --rope-scaling-factor 4.0 --rope-scaling-rope-type ntk_aware → 4x扩展

  Sliding window:
    → --sliding-window 4096 → StreamingLLM模式 → 无限对话

  Chunked prefill:
    → --max-num-batched-tokens 2048 → chunk=2048 → 改善公平性

  Attention backend:
    → V1自动使用FlashInfer → 不需要配置 → 推荐!

  Scheduler:
    → V1 unified token budget + FCFS → 默认最优 → 不需要配置

  不推荐配置:
    → --enable-prefix-caching → 1:N hash → 前缀共享 → 但开销 → 按需开启
    → → --swap-space 4 → CPU swap → RTX 4090 PCIe慢 → 用recomputation → 不推荐swap!
    → → → TP>1 → PCIe灾难 → 不推荐!

SGLang配置 (替代):
    → python -m sglang.launch_server \
    → →   --model-path meta-llama/Llama-2-7b-chat-hf \
    → →   --context-length 4096 \
    → →   --mem-fraction-static 0.9

    → RadixAttention → 更细粒度prefix共享 → ITL更稳定 → 推荐(需要稳定ITL)
    → → → vLLM吞吐更高 → SGLang ITL更稳定 → 按场景选择!
```

## 6. 推理吞吐量预期

```
推理吞吐量总结 (7B模型, RTX 4090):

    | 配置 | B | tok/s | 延迟(ms/tok) | 场景 |
    |------|---|-------|-------------|------|
    | BF16 MHA SDPA S=4K | 4 | 68 | 15 | baseline(最差) |
    | BF16 GQA-5 FlashInfer S=4K | 57 | 2,312 | 24.7 | **推荐!** |
    | INT8 KV GQA-5 FlashInfer S=4K | 57 | 2,312 | 24.7 | KV精度99.9% |
    | FP8 KV GQA-5 FlashInfer S=4K | 57 | 2,312 | 24.7 | KV精度99.999% → **最优!** |
    | INT4 AWQ INT8 KV GQA-5 S=4K | 119 | ~4,500 | ~24.7 | 最高并发 → 推荐 |
    | INT4 AWQ INT8 KV GQA-5 S=16K | 14 | ~572 | ~24.5 | 长上下文(NTK 4x) |
    | StreamingLLM(4+4K) INT8 KV | 57 | 2,311 | 24.7 | 无限对话 |
    | 蒸馏1.4B INT8 KV S=4K | 270 | ~11,000 | ~9 | 高吞吐 |
    | OPT-125M INT8 KV S=4K | 800+ | ~30,000 | ~3 | 超高吞吐 |

    → **最优配置: 7B GQA-5 FP8 KV FlashInfer S=4K → B=57 → 2,312 tok/s → 推荐!**
    → → → 精度: KV=99.999% → 权重=BF16 → 完全无损 → 推荐!
    → → → → 加INT4 AWQ → B=119 → 吞吐更高 → 推荐(需要fused kernel)!

  推理延迟总结:
    → TTFT(S=4K): 14ms → 极快 → 用户即时!
    → → ITL(B=57): 24.7ms → 可接受 → 40 tok/s per request
    → → → TTLT(256tok): 6.3s → 可接受 → 推荐!
```

## 7. 训练决策(简要)

```
RTX 4090训练决策(简要):

  单GPU训练:
    → ≤10M参数: DDP → 最简单 → 推荐
    → → 10-100M: FSDP1 → BF16+FSDP1最佳 → 1.51x+49%内存省 → 推荐
    → → → >100M: FSDP2 → 但PCIe scaling差 → 单GPU最优
    → → → → **RTX 4090最优: 单GPU + FSDP1 + BF16 → 推荐!**

  多GPU训练:
    → >2GPU: PCIe灾难 → FSDP 4GPU=0.48x → 不推荐!
    → → → 唯一可行: ≤2GPU FSDP → 25M模型 → 1.12x →勉强
    → → → → **>2GPU: RTX 4090不适合! → 需NVLink→H100!**

  FP8训练:
    → B≥4: FP8 1.48-1.59x加速 → TE fused kernel → 推荐!
    → → → B=1: FP8 0.75x(慢!) → 小batch量化开销占优 → 不推荐!
    → → → → **RTX 4090 FP8训练: B≥4 → 推荐!**

  详细训练决策 → notebook/fundamentals/rtx4090-pcie-decision-guide.md
```

## 8. 核心决策矩阵

```
RTX 4090推理部署核心决策矩阵:

    | 决策维度 | 推荐选择 | 原因 |
    |---------|---------|------|
    | 模型 | 7B GQA-5 | 单GPU最优, B=57 |
    | vocab | 32K | lm_head=256MB, ≤10% |
    | KV dtype | FP8 E4M3 | 精度99.999%, 50%省 |
    | 权重dtype | BF16 | 稳定, 或INT4 AWQ(需fused) |
    | Attention | FlashInfer | 15.72x, 生产级 |
    | 上下文 | S=4K默认 | 最高吞吐2,312 |
    | 长对话 | StreamingLLM(4+4K) | 无限对话, 2,311 |
    | 长上下文 | NTK-aware 4x | S=16K, 572 tok/s |
    | Scheduler | vLLM V1 FCFS | 吞吐最高 |
    | 蒸馏 | 7B→1.4B SFT | 5x推理加速 |
    | Chunked PF | chunk=2048 | 改善公平性 |
    | 量化路径 | fused kernel | Python dequant致命! |

    → **RTX 4090推理最优配置**:
    → → 7B GQA-5 + FP8 KV + FlashInfer + vLLM V1 + S=4K
    → → → B=57 → 2,312 tok/s → 无限对话(StreamingLLM) → 推荐!
```

---

**综合参考**: 所有RTX 4090 benchmark数据和笔记见 notebook/fundamentals/ 和 results/

**Related notes**: rtx4090-pcie-decision-guide.md(训练), flashinfer-attention-deep-dive.md, kv-cache-management-deep-dive.md, quantization-pruning-theory.md, long-context-serving-deep-dive.md, rope-scaling-deep-dive.md, distillation-deep-dive.md