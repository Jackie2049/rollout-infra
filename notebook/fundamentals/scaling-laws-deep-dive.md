# Scaling Laws — Chinchilla & Kaplan Deep Dive

> 2026-06-08 | 模型性能预测的精确数学, 从Kaplan到Chinchilla到Emergent
> 关键: Chinchilla推翻Kaplan → 计算最优模型更小+数据更多 → 7B胜175B!

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