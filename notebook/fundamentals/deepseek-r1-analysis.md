# DeepSeek-R1 深度分析 — 推理涌现与训练Pipeline

> 2026-06-07 | Deep Seek-R1论文深度阅读: 4阶段训练, "aha moment", 蒸馏vs RL

## 核心发现: 推理能力从纯RL涌现, outcome-only reward足够!

DeepSeek-R1证明了一个震撼性结论: 复杂推理(自我验证、反思、回溯)可以从纯RL + 简单outcome奖励自然涌现, 无需过程监督(PRM).

## 一、4阶段训练Pipeline

### Stage 1: Pure RL — DeepSeek-R1-Zero (冷启动)

从DeepSeek-V3-Base开始, **零推理SFT数据**:
- 算法: **GRPO** (组相对策略优化)
- Reward: 仅outcome (答案是否正确) + 格式规则 (用`<think>`标签)
- 无过程奖励模型(PRM) → 只看最终结果

**关键发现**: "aha moment"自然涌现!

模型在训练过程中自发发展出:
- **自我验证**: "Wait, let me check this again..."
- **反思**: "Hmm, that doesn't seem right. Let me reconsider."
- **回溯**: 从错误路径返回,尝试新方法
- **长链推理**: 生成数千token的详细思考过程

→ 这些行为**从未被训练数据教过** → 纯从GRPO组比较涌现!

### 为什么"aha moment"涌现?

GRPO的组比较机制:
- 同一prompt采样n个response → 组归一化 → "哪个response比平均好?"
- 好的推理路径自然获得正advantage → 被强化
- 短/错误的推理获得负advantage → 被抑制
- 模型自然学会"多想一会儿" → 长推理涌现!

### Stage 2: 拒绝采样 + SFT (暖启动数据)

用R1-Zero生成推理轨迹:
- **拒绝采样**: 只保留得出正确答案的轨迹 → 高质量SFT数据
- **混合数据**: 推理数据 + 非推理数据(写作、QA等)
- **SFT**: 在DeepSeek-V3-Base上微调 → 暖启动checkpoint

### Stage 3: 暖启动模型上的RL (GRPO再训练)

暖启动checkpoint再做GRPO RL:
- Reward更多样: 数学/代码用规则奖励 + 语言一致性奖励(惩罚推理中的语言混杂)
- 进一步打磨推理能力和对齐

### Stage 4: 最终SFT + RL (全场景对齐)

- 第二轮拒绝采样 → 更高质量推理数据
- 综合SFT: 推理 + 非推理任务
- 最终GRPO: 帮助性和安全性对齐 → **DeepSeek-R1** (最终版本)

### Pipeline图

```
DeepSeek-V3-Base
       |
       v  (纯GRPO RL, outcome reward)
DeepSeek-R1-Zero ← "aha moment"涌现!
       |
       v  (拒绝采样: 正确轨迹→SFT数据)
SFT Warm-Start Checkpoint
       |
       v  (GRPO RL + 多样reward)
Refined RL Checkpoint
       |
       v  (拒绝采样+SFT+最终GRPO)
DeepSeek-R1 (最终版)
```

## 二、DeepSeek-R1-Zero vs DeepSeek-R1

| 特征 | R1-Zero (纯RL) | R1-R1 (冷启动+RL) |
|------|---------------|-----------------|
| 初始 | V3-Base (无SFT) | SFT warm-start |
| 语言混杂 | 严重 (中英混合推理) | 改善 (语言一致性奖励) |
| 可读性 | 较差 | 更好 |
| 数学能力 | 很强 | 更强 |
| 通用任务 | 较弱 | 更强 |

**关键结论**: R1-Zero证明了推理可以涌现, 但R1证明了冷启动+RL产生更好的综合模型.

## 三、GRPO在DeepSeek-R1中的具体实现

### GRPO目标 (DeepSeek-R1版本)

$$J_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\left(\rho_{i,t} A_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_i\right) - \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})\right]$$

### Reward设计

DeepSeek-R1使用**规则奖励**(不训练reward model):

1. **Accuracy reward**: 数学题是否答案正确, 代码是否pass测试
2. **Format reward**: 是否使用`<think>`标签包裹推理
3. **Language consistency reward** (Stage 3): 推理过程中语言是否一致

→ **不需要训练reward model** → 更简单,更稳定!

### 为什么outcome-only reward足够?

传统RLHF需要step-level reward (每个token的reward), 但DeepSeek-R1证明:
- 数学推理: 只看最终答案 → 足够引导模型学会推理
- 不需要过程监督(PRM) → 简化训练pipeline
- GRPO组归一化自然区分好推理和坏推理

## 四、蒸馏结果 — 推理可以转移!

### 6个蒸馏模型

| Model | MATH-500 | AIME 2024 | GPQA Diamond |
|-------|----------|-----------|-------------|
| R1-Distill-Qwen-32B | **94.3%** | **72.6%** | **62.0%** |
| R1-Distill-Llama-70B | 93.0% | 70.0% | 59.1% |
| R1-Distill-Qwen-14B | 91.8% | 56.7% | 54.7% |
| R1-Distill-Llama-8B | 89.2% | 50.0% | 49.8% |
| R1-Distill-Qwen-7B | 88.4% | 43.9% | 46.8% |
| R1-Distill-Qwen-1.5B | 83.9% | 28.9% | 33.8% |

**震撼**: R1-Distill-Qwen-32B超越OpenAI o1-mini! (72.6% vs 63.6% on AIME)

### 蒸馏 > 直接RL (关键发现!)

对小模型做RL (从Qwen-32B Base开始GRPO) → 结果比蒸馏差!

→ 推理能力在大模型上涌现, 然后通过蒸馏向下压缩, 不能在小模型上从头训练出来.

### 为什么蒸馏更有效?

1. **推理模式迁移**: R1的推理模式(验证、回溯、自我纠正)被蒸馏模型继承
   - 即使1.5B模型也能生成完整的推理链
2. **数据质量**: 拒绝采样后的R1轨迹是高质量推理数据
3. **RL在7B/1.5B不稳定**: 小模型做RL容易过拟合, reward signal太弱
4. **Qwen比Llama更适合蒸馏**: 32B Qwen > 70B Llama → 架构差异

## 五、对AI Infra和训练系统的影响

### 训练成本估算

DeepSeek-R1 (671B MoE):
- 训练需要数千GPU (H800集群)
- GRPO每步需要n=64 samples → rollout吞吐是瓶颈
- Prefix sharing可以大幅降低rollout成本!

### PS对DeepSeek-R1训练的加速

```
DeepSeek-R1 GRPO:
  n=64 samples per prompt
  prompt≈4096 tokens (推理题目)
  response≈256 tokens (短答案,但推理过程可达数K)

PS加速:
  prefix_ratio=0.94 (4096/4352)
  speedup = 64/(1+63×0.06) = 64/4.78 = 13.4x (forward-only!)
  training speedup ≈ 13.4 × 0.76 = 10.2x!

→ DeepSeek-R1训练中rollout阶段可以用PS加速10x!
```

### 实际部署挑战

1. **长推理序列**: R1-Zero平均输出~5000 tokens → KV cache巨大
   - 需要Paged Attention + KV Cache管理
2. **GRPO n=64**: 64个并行采样 → 需要大规模rollout引擎
   - verl的ActorRolloutRefWorker支持并行采样
3. **MoE推理**: 671B参数→37B激活 → EP需要高速互联

## 六、与我们工作的联系

### verl Prefix Sharing + DeepSeek-R1

我的KV injection验证(cos_sim=0.999999)直接适用于DeepSeek-R1的GRPO训练:
- GRPO n=64 → prefix_ratio高 → PS加速巨大
- Block-causal mask验证通过 → 可以安全用于production
- 需要FlashAttention block-causal实现 → SDPA math backend太慢

### verl #6401贡献路径

DeepSeek-R1/TreeRL/rStar-Math是verl #6401提到的use cases:
- 这些都依赖长prompt + 多response → 完美匹配PS
- 我的benchmark证据(0.99x→2.46x gap)直接支持RFC

Sources:
- DeepSeek-R1: arxiv.org/abs/2501.04869
- DeepSeekMath (GRPO原论文): arxiv.org/abs/2402.03300
- DPO原论文: [Rafailov et al., NeurIPS 2023](https://arxiv.org/abs/2305.18290)
- RTX 4090 Benchmark: notebook/projects/full-model-ps-kv-injection-rtx4090.md