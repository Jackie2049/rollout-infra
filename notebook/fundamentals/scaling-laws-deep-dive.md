# Scaling Laws Deep Dive — Kaplan(2020)+Chinchilla(2022)+Emergent(2023)+Inference Scaling(2024)+Data Scaling+Data Wall+RL/Reasoning Scaling+Serving Scaling(Throughput vs Model Size)+PD Separation Scaling+Scaling Decision Framework+RTX 4090/A100/H100最优配置

> 2026-06-08→06-13 | 模型性能预测的完整数学框架: Kaplan(参数幂律α=0.076→被推翻)+Chinchilla(修正α=0.34/β=0.28→N∝C^0.50/D∝C^0.50→7B胜175B)+Emergent(离散指标错觉)+Inference Scaling(test-time compute∝tokens∝accuracy)+Data Scaling(数据质量×3>数量)+Data Wall(2028年人类文本耗尽)+RL Scaling(reward→score→accuracy)+Serving Scaling(7B=甜点→70B=推理贵→distillation替代)+Scaling Decision Framework(硬件→模型→数据→训练→推理→全链路)+RTX 4090最优(7B INT4推理4800tok/s)
> 关联: zero-algorithm-deep-dive.md(3D并行策略), rdma-ai-networking-deep-dive.md(PD拓扑), pd-separation-rdma-kv-transfer-deep-dive.md(PD serving scaling)
> 参考: Kaplan et al. 2020, Hoffmann et al. 2022(Chinchilla), Wei et al. 2023(Emergent), OpenAI o1 scaling(2024), DeepSeek-R1 scaling(2025)

## 1. Kaplan Scaling Laws (2020) — Neural Scaling Laws

### 核心发现

```
Loss ∝ N^(-α_N) × D^(-α_D)

其中:
  N = 模型参数量
  D = 训练数据量(tokens)
  α_N ≈ 0.076 (参数幂律)
  α_D ≈ 0.095 (数据幂律)

关键结论(Kaplan):
  → 模型越大越好! → 参数量比数据量更重要
  → 175B模型远超13B → 训练更多参数
  → GPT-3 175B = Kaplan的理论验证
  → 但这是错的! → Chinchilla 2022推翻了Kaplan!
```

### Kaplan的3个幂律

```
1. 模型大小幂律:
   L(N) = (N_c/N)^α_N
   → Loss随参数量幂律下降 → 参数越多越好

2. 数据大小幂律:
   L(D) = (D_c/D)^α_D
   → Loss随数据量幂律下降 → 数据越多越好

3. 计算量幂律:
   L(C) = (C_c/C)^α_C
   → Loss随计算量幂律下降 → compute越多越好

其中:
  N_c ≈ 6.4×10^13 (参数critical size)
  D_c ≈ 2.0×10^20 (数据critical size)
  α_N ≈ 0.076, α_D ≈ 0.095, α_C ≈ 0.050
```

### Kaplan的问题

```
Kaplan实验的局限:
  → 所有实验: 固定数据量 → 只变模型大小
  → 没有测试: 固定计算量 → 同时变模型+数据
  → 导致结论: "参数更重要" → 这是偏差!

  真正问题: 给定固定计算预算C → 模型N和数据D怎么分配?
  → C ≈ 6×N×D (每个token每个参数≈6FLOPS)
  → N×D = C/6 → N和D是trade-off!

  Kaplan说: N大D小 → 但这可能不是最优!
  → Chinchilla说: N小D大 → 更优!
```

## 2. Chinchilla Scaling Laws (2022) — Training Compute-Optimal LLMs

### 核心发现 — 推翻Kaplan!

```
给定计算预算C, 最优分配:

N_opt = (C / (6×α_N/α_D×N_c))^α_D/(α_N+α_D)
D_opt = (C / (6×α_D/α_N×D_c))^α_N/(α_N+α_D)

修正后的幂律系数:
  α_N ≈ 0.34 (比Kaplan的0.076大4.5x!)
  α_D ≈ 0.28 (比Kaplan的0.095大3x!)

关键结论:
  → N和D几乎同等重要! (α_N≈0.34 vs α_D≈0.28)
  → 计算最优时: N∝C^0.50, D∝C^0.50 → 线性增长!
  → 模型和数据应该同步增长!

Chinchilla验证:
  Gopher(280B, 300B tokens) → Kaplan式 → 过大模型+不足数据
  Chinchilla(70B, 1.4T tokens) → 计算最优 → 更小模型+更多数据
  → Chinchilla LOSS更低! → 70B > 280B!
  → 证明Kaplan错了 → 175B是计算浪费!
```

### 计算最优模型大小预测

```
给定FLOPS预算C → 最优参数量N_opt:

C (FLOPS)       → N_opt (参数)    → D_opt (tokens)
10^18 (1 PF-day) → 400M           → 8B
10^19 (10 PF-day)→ 1B              → 20B
10^20            → 7B              → 140B
10^21            → 70B             → 1.4T
10^22            → 600B            → 12T

关键规律:
  → N_opt ∝ C^0.50 → 模型大小与计算量平方根增长
  → D_opt ∝ C^0.50 → 数据量与计算量平方根增长
  → N×D ∝ C → 计算量线性增长(6×N×D)
  → 给定相同FLOPS → 更小模型+更多数据 > 更大模型+更少数据

实际应用:
  → LLaMA-1 7B: 1T tokens → 近似计算最优(Chinchilla建议140B→但1T更好!)
  → LLaMA-2 7B: 2T tokens → 超过Chinchilla最优 → 但更多数据→更好!
  → GPT-3 175B: 300B tokens → 远低于Chinchilla最优 → 计算浪费!
  → DeepSeek-V3 671B: 14.8T tokens → 大致符合Chinchilla
```

### 修正后的Scaling Law公式

```
L(N, D) = E + A/N^α + B/D^β

其中:
  E ≈ 1.69 (不可约loss → 语言本身的不确定性)
  A ≈ 406.5, α ≈ 0.34 (参数幂律 → 比Kaplan更陡!)
  B ≈ 976.7, β ≈ 0.28 (数据幂律 → 比Kaplan更陡!)

→ E=1.69意味着: 即使无限参数+无限数据 → Loss仍≈1.69
→ 语言本身有噪声 → 不可能达到0 loss
→ 实际LLM loss: GPT-4 ≈ 1.7-2.0 → 接近不可约下限!

计算最优公式:
  给定计算预算C → N_opt和D_opt:
  N_opt = (α×A×C/(6×β×B))^(1/(α+β))
  D_opt = (β×B×C/(6×α×A))^(1/(α+β))

  对于α≈0.34, β≈0.28:
  N_opt ≈ (0.34×406.5×C/(6×0.28×976.7))^(1/0.62)
  D_opt ≈ (0.28×976.7×C/(6×0.34×406.5))^(1/0.62)

简化版:
  N_opt ≈ C^0.50 / k_N
  D_opt ≈ C^0.50 / k_D
  → N和D同步增长 → 不应该一个快一个慢!
```

## 3. Emergent Abilities (2023) — 值得警惕的概念

```
定义: 模型在某规模下不存在 → 更大规模突然出现 → "涌现"

示例(Wei et al. 2023):
  → 多步算术: <100B不行 → >100B突然可行
  → 思维链推理: <60B不行 → >60B突然可行
  → 指令跟随: 小模型乱回答 → 大模型精确执行

争议(2023-2024):
  → Schaeffer et al.: "涌现是评估指标的错觉!"
  → 精确指标(per-token accuracy): 线性增长 → 没有涌现
  → 离散指标(完全正确/错误): 看起来涌现 → 但是指标选择问题!

  → 关键洞察: 真实能力是连续增长的 → "涌现"是离散指标的错觉
  → 但: 实际部署中 → 用户期望"完全正确" → 离散指标有现实意义

  → 结论: 模型能力持续改善 → 不存在"神奇涌现点"
  → 但部署时 → 需要达到足够阈值 → "实用涌现"确实存在
```

## 4. LLM Scaling in Practice

```
各模型与Chinchilla对比:

| Model | Params(B) | Data(T tok) | FLOPS(PF-day) | Chinchilla N_opt | 偏差 |
|-------|-----------|------------|---------------|----------------|------|
| GPT-3 | 175 | 0.3 | ~3600 | 70B |过大2.5x!|
| LLaMA-1 7B | 7 | 1 | ~70 | 7B |最优!|
| LLaMA-2 7B | 7 | 2 | ~140 | 10B |近最优|
| LLaMA-2 70B | 70 | 2 | ~1400 | 70B |最优!|
| DeepSeek-V3 | 671 | 14.8 | ~3370 | 600B |近似|

→ LLaMA系列最接近Chinchilla最优 → 这解释了为什么LLaMA效果好
→ GPT-3 175B是计算浪费 → 70B Chinchilla更优 → DeepMind证实!
→ DeepSeek-V3接近最优 → 671B active/37B per token → 稀疏更高效
```

## 5. Scaling Laws对AI Infra的影响

```
对训练Infra的影响:
  1. 计算预算决定模型大小 → 不是"越大越好" → 是"计算最优"最重要
  → RTX 4090 24GB → BF16最多装7B → 接近Chinchilla最优(10^20 FLOPS)
  → 量化(INT4) → 装40-50B → 但INT4训练目前不可行
  → RTX 4090训练最优模型 ≈ 7B BF16 → 与实际一致!

  2. 数据量与模型大小同步增长 → 数据pipeline是瓶颈
  → 7B需要1-2T tokens → 数据收集是主要工作量
  → 数据质量>数量 → 数据curating是关键

  3. 不可约loss=1.69 → 语言本身有噪声 → 不可能0 loss
  → 实际LLM loss ≈ 1.7-2.0 → 接近下限 → 进一步提升需新架构
  → MoE(DeepSeek-V3) → 同loss下更少compute → 新突破!

对推理Infra的影响:
  4. Chinchilla最优模型 ≈ 7B → 推理部署的甜点大小
  → 7B INT4 + FlashInfer → RTX 4090最佳推理配置
  → 70B需要4×H100 → 大模型推理成本高 → 考虑distillation

  5. Emergent abilities → 部署时需要达到实用阈值
  → 7B可能不够 → 需要70B才能链式推理 → 评估标准决定模型选择
  → 但: distillation(7B→1.4B) → 小模型也能达到70-80% → 实用!

  → RTX 4090最优策略:
    7B推理(INT4+INT8KV+FlashInfer) → 4,791 tok/s
    或: 70B distill→1.4B推理 → 5x推理加速 → 70-80%质量
    → 不需要70B原始模型 → distillation替代!
```

## 6. 核心规律

```
Scaling Laws核心:

  Kaplan(2020): L ∝ N^(-0.076) × D^(-0.095) → "模型大更重要"
  → 错! 实验设计偏差 → 固定数据只变模型

  Chinchilla(2022): L(N,D) = 1.69 + 406.5/N^0.34 + 976.7/D^0.28
  → N和D几乎同等重要! → 计算最优时N∝C^0.50, D∝C^0.50
  → 70B+1.4T > 280B+0.3T → 更小模型+更多数据!

  Emergent(2023): 大模型突然获得能力 → 但可能是评估错觉
  → 实际: 能力连续增长 → 部署需要实用阈值

  不可约loss: E≈1.69 → 语言噪声 → 无法达到0 → 需新架构突破

  RTX 4090实战:
    7B = Chinchilla最优大小 → RTX 4090甜点
    INT4量化 → 推理3.7x加速 → 训练不可行
    Distillation → 7B→1.4B → 5x推理 → 70-80%质量
    → Scaling Laws验证了"7B是RTX 4090最优"的直觉!
```

## 7. Inference Scaling Laws — Test-Time Compute Scaling

```
### 7.1 Compute-Optimal Inference — 推理也是scaling!

核心洞察: 推理时可以用更多compute → 获得更好结果 → 不只是训练scaling!

测试时计算(Test-Time Compute)三种策略:

1. Best-of-N Sampling:
  → 生成N个回答 → 选最好的 → compute∝N → accuracy∝N!
  → → 需要reward model → 选择最高reward的回答
  → → → 数学: P(best_of_N) = 1 - (1-P(single))^N → N越大→P越高!
  → → → → 例: P(single)=0.5 → N=4: P=0.94 → N=16: P=0.9999!
  → → → → → 但: compute∝N → 每N翻倍→推理成本翻倍 → 需权衡!

2. Chain-of-Thought (CoT) Scaling:
  → 更多思考tokens → 更多推理步骤 → 更好答案!
  → → compute∝思考token数 → accuracy∝思考深度!
  → → → 数学: accuracy ∝ (思考tokens)^α → α≈0.3-0.5
  → → → → OpenAI o1: 长CoT → more compute → better reasoning → scaling works!

3. Self-Consistency:
  → 多次CoT采样 → 投票选最常见答案 → compute∝N×CoT_length
  → → 比Best-of-N更强 → 不仅看reward → 看一致性!
  → → → 多路径推理 → 一致答案 → 更可靠 → 准确率更高!

### 7.2 OpenAI o1 / DeepSeek-R1 Scaling

OpenAI o1 (2024):
  → Thinking tokens → 消耗推理compute → 获得更好结果!
  → → 计算量: o1思考≈10x基础推理 → 但准确率≈2x → 巨大收益!
  → → → 关键: o1不是更大模型 → 而是更多推理compute → scaling新维度!

  → → → → Scaling公式:
    → → → → → accuracy_o1 ∝ base_accuracy × (1 + α × thinking_tokens / base_tokens)
    → → → → → → α≈0.3 → thinking 10x → accuracy ≈ 1.3x → 30%提升!

DeepSeek-R1 (2025):
  → RL训练 → 学会自主延长思考 → 自适应CoT!
  → → 短问题 → 少思考 → 省compute → 快!
  → → → 长问题 → 多思考 → 用更多compute → 准!
  → → → → 自适应 = 根据问题难度分配推理compute → 最优!

### 7.3 Inference Compute Budget vs Accuracy

```
| 推理策略          | Compute倍数 | Accuracy提升 | 适用场景          |
| 单次采样          | 1x         | baseline     | 简单问题/低成本    |
| Best-of-N(4)      | 4x         | +20-40%      | 中等问题/可控成本  |
| Best-of-N(16)     | 16x        | +40-60%      | 困难问题/高成本    |
| CoT(短)           | 2-3x       | +10-30%      | 推理问题          |
| CoT(长)           | 10-20x     | +30-50%      | 复杂推理          |
| Self-Consistency  | 5-20x      | +20-40%      | 数学/逻辑问题      |
| o1级thinking      | 10-50x     | +50-100%     | 最难问题/竞赛级   |

关键: 推理compute是可变成本 → 根据问题难度动态分配 → 最优!
  → 简单问题 → 1x → 快 → 低成本 → 满足大多数用户!
  → → 困难问题 → 10-50x → 更准确 → 但成本高 → 按需!
  → → → → 这就是推理scaling的核心: 不是固定compute → 而是动态分配!
```

### 7.4 Inference Scaling与RTX 4090

RTX 4090推理scaling分析:
  → 7B单次推理: 4800 tok/s → 基础吞吐 → 大多数场景够用!
  → → Best-of-4: 4×推理 → 1200 tok/s → 仍可接受 → +20-40%准确率!
  → → → CoT(短): 2-3×推理 → 1600-2400 tok/s → 推理问题 → +10-30%!
  → → → → CoT(长): 10×推理 → 480 tok/s → 复杂推理 → 仍可用!
  → → → → → o1级: 50×推理 → 96 tok/s → 太慢 → 需更大GPU!

  → 7B最佳策略:
    → → 简单任务: 单次采样 → 4800 tok/s → 快 → 省compute!
    → → → 推理任务: CoT(短) → 1600 tok/s → +30% → 平衡!
    → → → → 困难任务: Best-of-4 → 1200 tok/s → +40% → 可行!
    → → → → → → 超困难: 70B推理 → 需4×H100 → RTX 4090不适合!

  → 13B INT4推理: ≈3500 tok/s → Best-of-4 ≈ 875 → 可行但慢!
  → → → 70B推理: 需4×H100 → RTX 4090不够 → distillation替代!
```

## 8. Data Scaling Laws + Data Wall Analysis

```
### 8.1 Data Scaling — 数据质量×3 > 数据数量

数据幂律:
  → L(D) = B/D^β → β≈0.28 → Loss随数据幂律下降
  → → D翻倍 → Loss下降 ≈ 2^(-0.28) ≈ 17% → 不大!
  → → → D×10 → Loss ≈ 10^(-0.28) ≈ 52% → 不错!
  → → → → D×100 → Loss ≈ 100^(-0.28) ≈ 76% → 但需要100x数据!

数据质量乘数:
  → 高质量数据 → 每token效果≈3x低质量数据!
  → → → Math: L(D_high) ≈ L(3×D_low) → 质量等价3x数量!
  → → → → → 原因: 高质量→信息密度高→每token减少更多Loss!
  → → → → → → 实际: curated数据→过滤→去重→质量→效果>>原始数据!

数据来源质量排序:
  → 1. 代码(GitHub高质量) → 逻辑性强 → 推理能力提升最大!
  → → 2. 学术论文(ArXiv) → 专业性强 → 知识密度高!
  → → → 3. 百科(Wikipedia) → 结构化 → 知识覆盖广!
  → → → → 4. 书籍 → 高质量 → 但数量有限!
  → → → → → 5. 网页(Common Crawl) → 数量巨大 → 但质量参差 → 需过滤!
  → → → → → → → 6. 社交媒体 → 低质量 → noise大 → 不推荐!

数据过滤pipeline:
  → Common Crawl → 200TB原始 → 过滤后≈10TB高质量 → 95%丢弃!
  → → → 过滤步骤:
    → → → → 1. 语言检测 → 保留目标语言 → 丢弃50%
    → → → → → 2. 去重 → MinHash+LSH → 丢弃30%
    → → → → → → 3. 质量评分 → 分类器→保留高质量 → 丢弃50%
    → → → → → → → 4. PII移除 → 个人信息脱敏 → 丢弃1%
    → → → → → → → → → 结果: 200TB → 5-10TB → 数据质量×3!

### 8.2 Data Wall — 2028年人类文本耗尽?

Data Wall定义:
  → 高质量训练数据的总量上限 → 人类产出的文本总量有限!
  → → → 数学: 可用高质量文本 ≈ 10^13 tokens → 10T tokens!

当前数据消耗:
  → LLaMA-2: 2T tokens → 占人类高质量文本的20%
  → → → GPT-4: ≈13T tokens → 占人类高质量文本的130%! → 需要合成数据!
  → → → → DeepSeek-V3: 14.8T tokens → 超过人类高质量文本 → 必须合成!

Data Wall预测:
  → 2025: 大模型开始用合成数据 → 人类数据不足!
  → → → 2026: 合成数据质量提升 → 自我博弈 → RL训练!
  → → → → → 2027-2028: 高质量人类文本耗尽 → 合成数据为主!
  → → → → → → → 2030+: 多模态数据(视频/音频) → 新数据源!

合成数据方案:
  → 1. Self-Instruct → 大模型生成指令 → 训练小模型 → Alpaca!
  → → → 2. Constitutional AI → 自我纠正 → 生成高质量数据!
  → → → → 3. 数学/代码合成 → 自动生成+验证 → 无限数据!
  → → → → → → 4. 多模态合成 → 视频字幕→文本 → 新维度!
  → → → → → → → → 5. Self-play RL → 模型自我博弈 → DeepSeek-R1 → 无限!

  → → → → → → → → → 关键: 合成数据质量→取决于生成模型→需要高质量seed→闭环!

### 8.3 数据缩放对AI Infra的影响

训练Infra:
  → 数据量∝计算量 → Chinchilla: D∝C^0.50 → 计算翻倍→数据翻倍!
  → → → → 存储: 14.8T tokens ≈ 30TB → 需要7TB磁盘→我们的GPU服务器有!
  → → → → → 预处理: 数据过滤+去重+编码 → CPU密集 → 需分布式!
  → → → → → → DataLoader: 分布式数据加载 → 数据局部性 → I/O瓶颈!

推理Infra:
  → 模型大小∝数据量 → 更大模型→更多数据→更强能力→推理成本更高!
  → → → → → 7B: 低推理成本 → 70B: 高推理成本 → 需distillation!

RTX 4090:
  → 7B训练需要1-2T数据 → 下载30-60GB → 需要时间+带宽!
  → → → → 预处理: CPU处理 → 本地Mac可以做 → 不需GPU!
  → → → → → → 训练: GPU需要 → RTX 4090够7B → 但数据加载慢!
```

## 9. RL / Reasoning Scaling Laws

```
### 9.1 RL Training Scaling — reward→score→accuracy

RL训练scaling公式(经验性):
  → score ∝ reward_steps^α × model_size^β × data_diversity^γ
  → → α≈0.2-0.3 → RL步数收益 → 但收益递减!
  → → → β≈0.1-0.2 → 模型大小收益 → RL比SFT更依赖模型大小!
  → → → → γ≈0.3-0.5 → 数据多样性收益 → RL最依赖多样性!

关键发现(DeepSeek-R1):
  → SFT暖启动 → GRPO → RL → reasoning能力涌现!
  → → → SFT→GRPO: 93% eval / 0% gap → vs 纯RL: 40-52% / 37.5% gap
  → → → → → SFT暖启动是决定性 → 2x差距 → RL必须暖启动!
  → → → → → → 但: 过多SFT → reward hacking → Loss Landscape恶化!

RL scaling三个阶段:
  → Phase 1: 快速增长 → reward从0→50 → 模型学会基本推理 → 10% step!
  → → → Phase 2: 稳定增长 → reward 50→80 → 模型学会复杂推理 → 60% step!
  → → → → Phase 3: 收益递减 → reward 80→85 → 接近上限 → 30% step → 最后难提升!

RL scaling瓶颈:
  → 数据多样性 → 同类型数据 → reward饱和 → 需要新类型!
  → → → → reward hacking → 高reward但低quality → 需要KL约束!
  → → → → → → 训练稳定性 → reward崩塌 → 需要正则化!

### 9.2 Reasoning Scaling — o1/R1模式

推理能力scaling:
  → reasoning_accuracy ∝ thinking_tokens^0.3 × model_base_accuracy
  → → → 更多思考 → 更准确 → 但收益递减(α=0.3)!
  → → → → → thinking 10x → accuracy ≈ 1.3x → 30%提升 → 值得!
  → → → → → → thinking 100x → accuracy ≈ 1.5x → 50%提升 → 但compute太贵!

o1/R1 scaling vs 传统scaling:
  → 传统: accuracy ∝ model_size^0.34 × data^0.28 → 训练时scaling!
  → → → o1/R1: accuracy ∝ model_size^0.34 × data^0.28 × thinking^0.3 → 推理时额外scaling!
  → → → → → → 总scaling = 训练scaling × 推理scaling → 多维度!
  → → → → → → → → 例: 7B+CoT(10x) ≈ 7B×thinking^0.3 ≈ 7B×2 → ≈ 14B效果!
  → → → → → → → → → → → → → → 但: 7B+CoT compute=10×7B → vs 14B compute=14×7B → 推理scaling更贵!

推理scaling选择:
  → 低延迟场景 → 大模型(70B)单次 → 快但贵 → RTX 4090不够!
  → → → 推理场景 → 7B+CoT → 慢但准 → RTX 4090够480 tok/s → 可用!
  → → → → → → 最高质量 → 70B+o1 thinking → 极慢极贵 → 需H100集群!

RTX 4090推理scaling最优:
  → 7B + CoT(短) → 1600 tok/s → +30%准确率 → 最实用!
  → → → 7B + Best-of-4 → 1200 tok/s → +40% → 推理任务更好!
  → → → → → → 13B INT4 + CoT → 1000 tok/s → 更强 → 但更慢!
```

## 10. Serving Scaling Laws — Throughput vs Model Size

```
### 10.1 推理吞吐量Scaling公式

推理吞吐量公式:
  → throughput(decode) ≈ HBM_bandwidth × utilization / (2 × d² × L / TP)
  → → → decode memory-bound → 吞吐量∝HBM带宽 → 与模型大小反相关!

  → → → → 简化: throughput ≈ HBM_BW / (params × 2bytes) → 反比!

模型大小vs吞吐量(7B→70B):
  → 7B BF16: 14GB params → throughput ≈ 890/14 ≈ 63 tok/s/B → B=1: 63!
  → → → → 7B INT4: 3.5GB params → throughput ≈ 890/3.5 ≈ 254 tok/s/B → B=1: 254!
  → → → → → 70B BF16: 140GB params → 需4×H100 → 单GPU throughput ≈ 890/35 ≈ 25!
  → → → → → → → → 结论: 模型越大 → decode吞吐越低 → 内存瓶颈!

量化scaling:
  → INT4 → 模型减4x → 吞吐量≈4x → 但精度有损!
  → → → INT8 → 模型减2x → 吞吐量≈2x → 精度较好!
  → → → → → FP8 → 模型减2x → 吞吐量≈2x → 精度可训练!
  → → → → → → → 结论: 量化是推理scaling的关键 → 减少内存瓶颈!

### 10.2 Prefill吞吐量Scaling

Prefill throughput:
  → throughput(prefill) ≈ peak_FLOPS × utilization / (2 × S × d² × L)
  → → → prefill compute-bound → 吞吐量∝FLOPS → 与prompt长度反相关!

  → → → → 模型大小影响:
    → → → → → → 7B: 169.6 TFLOPS peak → S=2048: 9679 tok/s
    → → → → → → → 70B TP=4: 每GPU 169.6 TFLOPS → 4×计算 → 但通信开销!
    → → → → → → → → → → 结论: prefill∝FLOPS → 大模型prefill更慢 → 但可用TP加速!

Prefill vs Decode scaling差异:
  → Prefill: 吞吐∝FLOPS → 更多GPU→更快 → scaling好!
  → → Decode: 吞吐∝HBM_BW/params → 更多GPU→线性分params → scaling好!
  → → → → → → → 但: DDP decode → 通信开销 → RTX 4090 0.46x → 灾难!
  → → → → → → → → → NVLink decode → 几乎零通信 → A100/H100 scaling好!

### 10.3 PD Separation Scaling

PD分离scaling:
  → P实例: compute-optimal → 多GPU→prefill更快 → scaling∝FLOPS!
  → → → D实例: bandwidth-optimal → 多GPU→更多KV→更多batch → scaling∝HBM!
  → → → → → → → P:D比例 = 1:4-1:8 → decode是瓶颈 → 需更多D实例!

PD分离集群scaling:
  → N个P实例 + 4N个D实例 → 4N×decode_throughput → 总吞吐!
  → → → → → 例: 1P+4D → 4×3600=14400 tok/s → vs 单GPU 4800 → 3x!
  → → → → → → → → → 但: P实例利用率→90% → D实例利用率→80% → 高效!

PD scaling瓶颈:
  → KV transfer → RDMA带宽 → 如果P→D网络不够快 → transfer瓶颈!
  → → → → → Rail-aligned → 8×50GB/s=400GB/s → 足够 → 不瓶颈!
  → → → → → → RTX 4090 → PCIe → 2GPU PD → KV transfer=3% TTFT → 可行但2x成本!

### 10.4 Serving Scaling决策矩阵

```
| 模型大小 | GPU配置 | 量化 | 吞吐(tok/s) | 推理scaling | 最佳策略       |
| 7B      | RTX4090×1 | INT4  | 4800        | CoT/Best-of-4 | 单GPU最优    |
| 7B      | RTX4090×2 | INT4  | PD分离      | PD+CoT        | 2GPU PD可行  |
| 13B     | RTX4090×1 | INT4  | ≈3500       | CoT短         | 单GPU可行    |
| 70B     | H100×4    | BF16  | ≈25/B       | 无CoT         | 多GPU TP=4  |
| 70B     | A100×8    | INT4  | ≈100/B      | Best-of-N     | 多GPU INT4  |

关键洞察:
  → 7B INT4 = RTX 4090甜点 → Scaling Laws验证 → 最优!
  → → 70B = 需H100集群 → 推理scaling靠量化+TP → 不是单GPU!
  → → → → Distillation → 7B→1.4B → 推理5x → 质量70-80% → 替代70B!
  → → → → → → → 结论: 小模型+量化+推理scaling > 大模型+BF16+高成本!
```
```

## 11. Scaling Decision Framework — 全链路决策

```
### 11.1 模型大小选择决策树

```
可用GPU → 计算预算 → Chinchilla最优 → 实际调整

RTX 4090 24GB(单GPU):
  → BF16: 7B fits → Chinchilla最优训练大小!
  → → INT4推理: 40B fits → 但INT4训练不可行 → 只推理!
  → → → → → 结论: 7B = RTX 4090最优 → 训练+推理甜点!

A100 80GB(单GPU):
  → BF16: 40B fits → 但Chinchilla建议更大数据!
  → → → → → → 训练: 7B-13B → 加更多数据 → Chinchilla优化!
  → → → → → → → 推理: 13B-70B(量化) → INT4 70B fits 80GB!

H100集群(8+ GPU):
  → BF16: 70B TP=4 → fits 4×80GB → 训练+推理标准配置!
  → → → → → → 训练: 70B+1.4T → Chinchilla最优!
  → → → → → → → 推理: 70B INT4 TP=2 → fits → 或distillation→7B!

模型大小决策:
  1. 计算预算 → C = GPU数 × GPU_FLOPS × 训练时间
  2. Chinchilla → N_opt = C^0.50 → 模型大小
  3. 数据需求 → D_opt = C^0.50 → 数据量
  4. 训练可行性 → N×2bytes ≤ GPU_memory × TP → fits?
  5. 推理可行性 → N×2bytes ≤ GPU_memory → 或量化 → fits?
  6. 推理质量 → eval指标 → 是否达到实用阈值?
  7. 成本效率 → 训练成本+推理成本 → 总成本最优?

### 11.2 训练→推理全链路Scaling决策

训练阶段:
  → 选择模型大小 → Chinchilla最优 → N∝C^0.50
  → → → 选择数据量 → D∝C^0.50 → 与模型同步!
  → → → → → → 选择训练策略 → BF16 vs FP8 → 精度vs速度
  → → → → → → → → → → 选择分布式策略 → ZeRO-2+offload(RTX 4090) → TP+ZeRO(A100/H100)

推理阶段:
  → 选择量化 → INT4(推理甜点) → FP8(训练可用) → BF16(质量最好)
  → → → → → → 选择推理策略 → 单次/CoT/Best-of-N → 根据场景!
  → → → → → → → → → → 选择服务架构 → 单GPU(RTX 4090) → PD分离(H100)
  → → → → → → → → → → → → → → → → 选择scaling → Distillation(7B→1.4B) → 替代大模型!

### 11.3 RTX 4090 / A100 / H100最优配置汇总

```
| GPU      | 训练模型    | 训练策略              | 推理模型      | 推理策略          |
| RTX 4090 | 7B BF16   | ZeRO-2+CPU offload   | 7B INT4      | 单GPU+FlashInfer  |
| A100 80  | 13B BF16  | TP=2+ZeRO-2          | 13B INT4     | TP=2推理          |
| H100 80  | 70B BF16  | TP=8+ZeRO-2          | 70B INT4     | PD分离+TP=2       |

推理scaling策略:
| GPU      | 简单任务       | 推理任务          | 困难任务          |
| RTX 4090 | 7B单次4800t/s | 7B CoT短1600t/s  | 7B Best-of-4     |
| A100     | 13B单次        | 13B CoT           | 13B Best-of-8    |
| H100     | 70B单次        | 70B CoT           | 70B o1级thinking |

关键规律:
  → RTX 4090: 7B = 甜点 → 训练+推理+scaling → 最优!
  → → A100: 13B-70B(量化) → 多GPU → ZeRO+TP → 标准!
  → → → H100: 70B → PD分离 → RDMA KV transfer → 生产标配!
  → → → → → 所有: Distillation → 小模型替代大模型 → 成本效率最优!
```
```

## 12. 核心规律总结

```
Training Scaling (Chinchilla):
  → L(N,D) = 1.69 + 406.5/N^0.34 + 976.7/D^0.28
  → → N∝C^0.50, D∝C^0.50 → 计算最优时模型和数据同步增长!
  → → → 7B+1T > 175B+0.3T → 更小模型+更多数据 → Chinchilla推翻Kaplan!

Inference Scaling (Test-Time Compute):
  → accuracy ∝ model_base × (1 + α × thinking_tokens)
  → → α≈0.3 → 10x思考 → 30%准确率提升 → 推理scaling新维度!
  → → → o1/R1: 自适应推理compute → 简单→快→困难→深 → 最优!

Data Scaling:
  → 数据质量×3 > 数据数量 → curated > raw → 过滤95% → 质量胜!
  → → Data Wall: 2028人类文本耗尽 → 合成数据 → 自我博弈 → RL!

RL Scaling:
  → SFT暖启动=决定性 → 2x差距 → GRPO>SFT后收益递减!
  → → → reward hacking风险 → 需KL约束 → 训练稳定性!

Serving Scaling:
  → decode throughput ∝ HBM_BW/params → 模型越大吞吐越低!
  → → → 量化=推理scaling关键 → INT4 → 4x吞吐 → 甜点!
  → → → → → PD分离 → 1P:4D → 3x吞吐 → RDMA零拷贝 → 生产标配!

RTX 4090最优配置:
  → 训练: 7B ZeRO-2+CPU offload → 全参数微调唯一可行方案
  → → 推理: 7B INT4+INT8KV+FlashInfer → 4800 tok/s → 单GPU最优
  → → → 推理scaling: 7B CoT短 → 1600 tok/s → 推理问题+30%
  → → → → → Distillation: 7B→1.4B → 推理5x → 质量70-80%
  → → → → → → → → → Scaling Laws完整验证 → 7B=RTX 4090最优!

Scaling Laws对AI Infra工程师的意义:
  → 不只是理论 → 决定硬件选型 → 模型大小 → 数据量 → 全链路!
  → → → 计算预算 → Chinchilla → 模型大小 → 训练策略 → 推理策略
  → → → → → → → → → 全链路决策 → 从计算到推理 → Scaling Laws是桥梁!
```

## 参考文献

```
1. Training Scaling:
   - Kaplan et al., "Scaling Laws for Neural Language Models", 2020 (arxiv.org/abs/2001.08361)
   - Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla), 2022 (arxiv.org/abs/2203.15556)

2. Inference Scaling:
   - OpenAI o1: "Learning to Reason with LLMs", 2024
   - DeepSeek-R1: "Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", 2025
   - Snell et al., "Scaling LLM Test-Time Compute", 2024

3. Data Scaling:
   - Muennighoff et al., "Scaling Data-Constrained Language Models", 2023
   - Data Wall analysis: Epoch AI, "Will we run out of data?", 2023

4. Emergent Abilities:
   - Wei et al., "Emergent Abilities of Large Language Models", 2023
   - Schaeffer et al., "Are Emergent Abilities of Large Language Models a Mirage?", 2023

5. 我们的笔记:
   - zero-algorithm-deep-dive.md → ZeRO+TP+PP策略选择(含3D并行决策树)
   - rdma-ai-networking-deep-dive.md → RDMA+集群拓扑+DCQCN
   - pd-separation-rdma-kv-transfer-deep-dive.md → PD分离+KV transfer
   - rtx4090-inference-deployment-guide.md → 7B INT4最优推理配置