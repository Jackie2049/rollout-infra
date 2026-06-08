# LLM Architecture Evolution: MHA→GQA→MLA→MoE→MTP

> 2026-06-08 | 模型架构演进=推理效率演进, 每一代优化KV带宽+计算效率
> 基于: LLaMA(MHA), LLaMA-2(GQA), DeepSeek-V2/V3(MLA+MoE+MTP), Mistral(SlidingWindow+GQA)
> 关联: mla-architecture-deep-dive.md, flashinfer-attention-deep-dive.md, lora-peft-deep-dive.md

## 0. 核心定律: 架构演进 = 降低KV/计算瓶颈

```
LLM架构演进趋势:

  第一代(MHA): KV带宽和内存是瓶颈 → 多头注意力→每头独立KV→KV大
  第二代(GQA): KV分组共享→省75%KV→FlashInfer native→15.72x加速
  第三代(MLA): KV低秩压缩→省56.9x→但需fused kernel(FlashMLA)→Hopper专用
  第四代(MoE): 计算稀疏→只激活8/256 expert→18x计算省→但EP通信瓶颈
  第五代(MTP): 多token预测→1次预测2token→spec decoding零成本→推理加速

  统一视角:
    → MHA→GQA: 省KV带宽(75%)
    → GQA→MLA: 省KV内存(56.9x vs MHA)
    → Dense→MoE: 省计算(18x稀疏)
    → Single→MTP: 省推理步骤(2x decode)
    → → 每一代都在解决不同瓶颈! → 组合使用效果最大!

  RTX 4090最优架构:
    → 7B + GQA-5 + INT8KV + FlashInfer → 解决KV带宽和内存
    → → 不需要MLA(无FlashMLA支持) → 不需要MoE(PCIe EP不可行)
    → → MTP(Multi-Token Prediction)可以考虑 → 推理加速!
```

## 1. MHA: 标准多头注意力 (baseline)

```
MHA (Multi-Head Attention) — Transformer原始设计:

  数学:
    → Q = W_q × x → [B, S, num_heads, d_head]
    → K = W_k × x → [B, S, num_heads, d_head]
    → V = W_v × x → [B, S, num_heads, d_head]
    → Attn = softmax(QK^T/√d) × V → 每头独立
    → Output = W_o × concat(attn_heads) → 恢复维度

  KV Cache per token per layer:
    → 2 × num_heads × d_head × dtype_size
    → → LLaMA-7B MHA: 2×32×128×2 = 16,384B ≈ 16KB/tok/layer
    → → LLaMA-70B MHA: 2×64×128×2 = 32,768B ≈ 32KB/tok/layer
    → → 7B 32层: 16KB × 32 = 512KB/tok → 24GB → 46并发(S=4K) → 很少!

  推理瓶颈:
    → decode: KV读取=2×num_heads×d_head×S×2 → memory-bound
    → → LLaMA-7B S=4K: 2×32×128×4096×2 = 4MB KV读取 → 0.9ms @ 890GB/s
    → → → 纯MHA推理太慢 → 需要GQA/MLA/量化优化!

  代表模型:
    → GPT-2/GPT-3: MHA (原始设计)
    → LLaMA-1: MHA (32 heads, 70B也有64 heads)
    → OPT系列: MHA
```

## 2. GQA: 分组查询注意力 (推理革命)

```
GQA (Grouped-Query Attention) — LLaMA-2核心改进:

  数学:
    → num_kv_heads < num_heads → KV heads分组共享!
    → → Q: [B, S, num_heads, d_head] → 32 query heads
    → → K,V: [B, S, num_kv_heads, d_head] → 5 KV heads
    → → → 每6.4个Q头共享1个KV头 → KV读取省 num_heads/num_kv_heads = 6.4x!

  KV Cache per token per layer:
    → 2 × num_kv_heads × d_head × dtype_size
    → → GQA-5 7B: 2×5×128×2 = 2,560B ≈ 2.5KB/tok/layer → 6.4x省!
    → → GQA-8 70B: 2×8×128×2 = 4,096B ≈ 4KB/tok/layer → 8x省!

  FlashInfer GQA native加速:
    → FlashInfer: bdy = num_qo_heads / num_kv_heads → 无需expand!
    → → SDPA: 需要 K = K.expand(-1, num_heads/num_kv_heads, ...) → 6.4x内存浪费!
    → → FlashInfer: 直接在压缩KV上做attention → 省读取+省内存
    → → 实测: B=32 FlashInfer GQA-5 = 15.72x vs SDPA!

  KV带宽对比:
    → MHA decode KV读取: 2×32×128×S×2 = 4MB (S=4K)
    → GQA-5 decode KV读取: 2×5×128×S×2 = 0.625MB (S=4K) → 6.4x省!
    → → GQA-5 + INT8 KV: 0.3125MB → 12.8x省!

  代表模型:
    → LLaMA-2: GQA (7B=32kv, 70B=8kv → 后者KV省8x)
    → Mistral-7B: GQA-8 + SlidingWindow → KV省8x + KV不随S增长!
    → Qwen系列: GQA → 大部分新模型用GQA
```

## 3. MLA: 多头潜在注意力 (容量革命)

```
MLA (Multi-head Latent Attention) — DeepSeek-V2/V3:

  数学 (已在mla-architecture-deep-dive.md详细分析):
    → KV压缩: c_KV = W^DKV × h → 512维 (从7168→512 = 14x压缩!)
    → → KV Cache: (512+64)×2×60 = 69KB/tok → vs MHA 1966KB/tok → 56.9x省!

  推理关键: 矩阵吸收
    → 训练时: 需要显式恢复K/V → k_C = W^UK × c_KV → 恢复到16384维
    → 推理时: W^Q × W^UK → 吸收 → 直接在latent空间做attention → 不恢复!
    → → 但实际vLLM/SGLang默认不吸收 → 仍然做up-projection → 慢!

  FlashMLA (Hopper专用):
    → fused up-projection + attention → 省global memory读写 → 3000GB/s!
    → → RTX 4090不支持(SM89) → MLA在RTX 4090上慢2-8x → 不推荐!

  vs GQA:
    → MLA容量更好: 56.9x vs MHA (GQA: 6.4-8x vs MHA)
    → MLA速度更差(无FlashMLA): 比GQA慢2-8x → RTX 4090不可行!
    → → 选择取决于GPU: Hopper→MLA, RTX 4090→GQA!

  代表模型:
    → DeepSeek-V2: MLA (首次引入)
    → DeepSeek-V3: MLA + MoE + MTP (2024.12)
```

## 4. MoE: 混合专家 (计算革命)

```
MoE (Mixture of Experts) — 计算稀疏化:

  数学:
    → MLP层: 每token选择top-K experts → 只计算K/N
    → → DeepSeek-V3: 256 experts, top-8 → 8/256 = 3.1% → 97%计算省!
    → → 但: active params = 37B (8 expert × 每个4.6B) → vs total 671B

  推理影响:
    → 计算省: active 37B vs dense 671B → 18x计算省!
    → → 但: 每个token不同expert → batch内token不能共享MLP计算
    → → → MoE推理: per-expert batch → 小batch → memory-bound!

  EP瓶颈 (已在deepep-all-to-all-deep-dive.md详细分析):
    → Expert Parallel: 每个GPU放不同expert → All-to-All通信
    → → NVLink: ~0.35ms → 可行! (H800集群)
    → → PCIe: ~2.8ms → 不可行! (RTX 4090)
    → → → MoE EP只在NVLink集群可行 → RTX 4090只能TP

  DeepSeek-V3 MoE特殊设计:
    → 256 routed experts + 1 shared expert → shared expert不需A2A
    → → shared expert = dense计算 → 所有token共享 → 可TP
    → → routed experts = 稀疏计算 → 需EP → NVLink必需

  RTX 4090 MoE:
    → Mixtral-8x7B: 87GB FP16 → RTX 4090 24GB不够 → INT4才行
    → → INT4 Mixtral: ~22GB →勉强 → 但EP不可行 → 只能TP → 需4GPU
    → → → RTX 4090不适合MoE serving → 只适合dense小模型!

  代表模型:
    → Mixtral-8x7B: 8 experts, top-2 → 8x稀疏
    → DeepSeek-V3: 256 experts, top-8 → 32x稀疏
    → Qwen-MoE系列: 多种MoE配置
```

## 5. MTP: 多Token预测 (推理加速革命)

```
MTP (Multi-Token Prediction) — DeepSeek-V3新特性:

  数学:
    → 标准推理: 每步预测1 token → 需要N步才能生成N tokens
    → MTP: 每步预测2 tokens → 只需N/2步 → 理论2x加速!
    → → 实现: 1个main head + 1个 auxiliary head → 2次输出
    → → → main head预测t+1, auxiliary head预测t+2

  为什么MTP > Speculative Decoding:
    → Spec Dec: 需要训练draft模型 → 接受率<100% → 实际加速<理论
    → MTP: draft模型=模型自身 → 接受率=100% → 实际加速=理论!
    → → → DeepSeek-V3: MTP训练→推理时speculative decoding→无额外模型!

  推理流程:
    → Step 1: 模型预测token_1(token t+1) + token_2(token t+2)
    → → 两个token同时验证 → 如果都对 → 接受2 tokens → 2x加速!
    → → 如果token_2错 → 接受token_1 → 1x(不更慢!)

  RTX 4090 MTP:
    → MTP不需要特殊硬件 → 任何GPU都可以 → RTX 4090可以用!
    → → 但需要模型支持 → 目前只有DeepSeek-V3有MTP head
    → → → 未来: 更多模型可能加入MTP → 推理加速的重要方向!

  代表模型:
    → DeepSeek-V3: MTP + MLA + MoE → 三重优化!
    → → 未来趋势: MTP将成为标准配置 → 推理2x加速!
```

## 6. 其他架构创新

```
Sliding Window Attention (Mistral):
  → 限制attention到最近W tokens → KV不随S增长!
  → → Mistral-7B: W=4096 → KV最大S=4096 → 内存固定!
  → → → 但: 长context(S>W)需要滚动attention → 信息丢失
  → → → 适合: 短context推理(S<4096) → 不适合长context!

RoPE (Rotary Position Embedding):
  → 相对位置编码 → 通过旋转矩阵注入位置信息
  → → 优势: 可以外推到更长context(NTK-aware scaling)
  → → → LLaMA/Mistral/Qwen都用RoPE → 标准选择!
  → → → 但: RoPE阻止MLA完全矩阵吸收 → 需要解耦设计!

SwiGLU (Swish-Gated Linear Unit):
  → MLP激活: SwiGLU(x) = x × Swish(gate_proj(x))
  → → → 比GeLU更好 → LLaMA-2/V3都用SwiGLU → 标准选择!

RMSNorm (Root Mean Square Normalization):
  → LayerNorm简化: 不需要mean shift → 更快
  → → → 我们的CUDA C++ RMSNorm: 9x over PyTorch → 实测验证!
  → → → LLaMA/Mistral/DeepSeek都用RMSNorm → 标准选择!
```

## 7. 架构组合决策树 (RTX 4090)

```
RTX 4090架构组合决策:

  小模型推理(7B):
    → ✅ GQA-5 + INT8KV + FlashInfer → 15.72x decode加速 → 最优!
    → ✅ SwiGLU + RMSNorm + RoPE → 标准LLaMA架构
    → ✅ SlidingWindow(S≤4096) → KV不随S增长 → 适合短context
    → ❌ MLA → FlashMLA不支持 → 比GQA慢2-8x → 不推荐
    → ❌ MoE → EP不可行(PCIe) → INT4勉强但4GPU TP → 不划算
    → ⚠️ MTP → 需模型支持 → 未来可用

  中等模型推理(70B):
    → ✅ GQA-8 + FP8 + TP=2 → H100/H200最优
    → → RTX 4090: INT4勉强 → 但24GB不够 → 不推荐
    → ✅ INT4 AWQ + INT8KV + FlashInfer → 单H100部署

  大模型推理(671B DeepSeek-V3):
    → ✅ MLA + MoE EP + MTP → H800集群最优 → FlashMLA加速
    → → RTX 4090: 完全不可行 → 需TP=8 → PCIe scaling灾难性

  训练(7B):
    → ✅ LoRA r=8 → 单卡可训练 → 95%全量性能 → 推荐!
    → ✅ FSDP2 B≤4 → 2GPU勉强 → 不如LoRA单卡
    → → RTX 4090训练最优=LoRA → 不需要多卡!

  总结:
    → RTX 4090最优组合: 7B + GQA-5 + INT8KV + FlashInfer + LoRA微调
    → → 推理: 145K tok/s(B=32) → $0.01/Mtok
    → → 训练: 单卡LoRA → $0.35/hr
    → → 不需要MLA/MoE → GQA-5足够 → 简单有效!
```

---

**Sources**:
- LLaMA (Touvron et al. 2023), LLaMA-2 (2023)
- DeepSeek-V2 (2024), DeepSeek-V3 (2024.12)
- Mistral (2023), Mixtral (2024)
- FlashMLA (deepseek-ai/FlashMLA, 2025.2)

**Related notes**: mla-architecture-deep-dive.md, flashinfer-attention-deep-dive.md, lora-peft-deep-dive.md, inference-cost-analysis.md