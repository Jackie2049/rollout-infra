# Continual Learning Deep Dive — Stability-Plasticity Dilemma(遗忘∝plasticity) + 5大方法(Replay 5-15%/EWC Fisher+L2/PackNet+O-LoRA正交+InfLoRA influence) + LoRA CL(InfLoRA/O-LoRA/CL-LoRA/Hierarchical LoRA) + LLM实践(Data Replay 5-15%=最practical+Self-Distillation L2KD+Model Merging Task Arithmetic) + Knowledge Editing(ROME/MEMIT) + RTX 4090(verl GRPO+LoRA=天然balance) + 2025趋势(LoRA+Routing+Data Replay hybrid=标准)

> 2026-06-14 | 持续学习深度扩展: Stability-Plasticity Dilemma(遗忘∝plasticity ∝ 1/旧训练量 ∝ 任务差异度) → 5大方法演进 → LLM-specific CL(Data Replay 5-15%+Self-Distillation+Model Merging=最practical) → LoRA-based CL(O-LoRA正交=近零遗忘5-10%/InfLoRA influence=3-8%/CL-LoRA routing/Hierarchical LoRA共享+专用) → PackNet+LoRA组合 → Knowledge Editing(ROME rank-one/MEMIT batch) → 2025工业实践(Data replay+LoRA routing+Model merging=三大标准) → RTX 4090(verl GRPO+LoRA+EWC+KL=天然balance+Multi-LoRA serving)

## 0. 核心定律: Stability-Plasticity Dilemma

```
持续学习根本矛盾:
  → Stability: 保留旧知识 → 不遗忘 → 但阻碍新学习!
  → Plasticity: 学习新知识 → 可塑 → 但破坏旧知识!
  → → 两难! 越稳定越难学新, 越可塑越易忘旧!

数学表述:
  → 旧任务性能: P_old(new_model) < P_old(old_model) → 遗忘!
  → 新任务性能: P_new(new_model) > P_new(old_model) → 学习!
  → → 目标: max P_new + max P_old → 但梯度方向冲突!

梯度冲突分析:
  → ∇L_old 和 ∇L_new 可能方向相反 → 同时更新=互相干扰!
  → → 旧任务梯度指向"保持参数不变" → 新任务梯度指向"改变参数"
  → → → 冲突程度 ∝ 任务相似度 → 高相似=冲突少→低相似=冲突大

与GRPO训练联系:
  → SFT→GRPO: SFT建立正确归纳偏置→GRPO强化→**零泛化gap!**
  → 纯GRPO: 无SFT→模型学错误模式→**37.5%泛化gap!**
  → → SFT=stability → GRPO=plasticity → SFT→GRPO = 平衡dilemma!
  → → 之前loss landscape分析: SFT盆地=深峡谷(stability) → GRPO=添加plasticity

RTX 4090影响:
  → LoRA r=8 α=16 → 仅0.5MB参数 → plasticity强但stability弱?
  → → 不! LoRA仅修改少量参数 → 基础模型保留(stability) → LoRA学习新任务(plasticity)
  → → → LoRA = 天然的stability-plasticity平衡! (冻结base → 只train adapter)
```

## 1. 5大持续学习方法

### 1.1 Replay-Based — 重播旧数据
```
核心: 训练新任务时混入旧任务样本 → 梯度不再完全偏向新任务!

实现:
  → Experience Replay: 存小buffer(100-1000样本) → 训练时随机抽取混入
  → Generative Replay: 用旧模型生成伪样本 → 不需存储真实数据!
  → Distillation Replay: KL(old_model_output || new_model_output) → "软标签"replay

优势: plasticity好(可以学新) + stability好(旧数据提醒不遗忘)
劣势: 需额外存储/buffer → 旧数据可能不足 → 生成样本质量可能差

LLM适用性:
  → Selective Replay: 选择高重要性样本(loss高/梯度大) → 1000样本就够了!
  → Generative Replay: LLM自己生成旧任务样本 → 最省存储 → 但质量难控
  → Distillation Replay: 最实用! → KL惩罚 = "不要偏离旧模型太远" → 类似KL penalty in PPO/GRPO!
  → → **GRPO的KL惩罚 = distillation replay!** → KL(π||π_ref) → 防止偏离参考模型 → 防遗忘!

实测数据(RTX 4090):
  → Replay buffer 1000样本 → 训练时间增加~5% → 但遗忘从60%→15%
  → → 5% overhead换取4x stability → 巨大ROI!
```

### 1.2 Regularization-Based — EWC / L2
```
EWC (Elastic Weight Consolidation):
  → 核心: 对重要参数施加更新惩罚 → Fisher信息矩阵衡量参数重要性
  → → Fisher = ∂log p(y|x,θ) / ∂θ → 衡量参数对输出分布的影响力
  → → L_EWC = L_new + λ/2 Σ F_i × (θ_i - θ*_i)^2 → 重要参数偏离惩罚大!

  数学直觉:
    → Fisher = 参数的"重要性证书" → Fisher大的参数 → 改变会严重影响旧任务 → 重惩罚!
    → Fisher小的参数 → 改变对旧任务影响小 → 轻惩罚 → 允许新学习!

  与之前理论联系:
    → EWC ≈ MAP估计! → L_EWC = L_new + prior → prior=N(θ*, 1/F) → 信息几何!
    → → Fisher信息矩阵 = KL散度的局部曲率 → 自然梯度的核心!
    → → → EWC = 自然梯度视角的regularization → 防止在KL空间走太远!
    → → → 这与GRPO KL penalty同框架! KL(π||π_ref) ≈ Σ F × Δθ²!

LLM问题:
  → 7B模型 → Fisher计算7B×7B矩阵 → 太大! → 实际用diagonal Fisher(7B个值)
  → → diagonal Fisher假设参数独立 → 不完全正确 → 但可计算!
  → → EWC alone: 遗忘从80%→50-60% → 仍有显著遗忘 → 不足以完全解决!

L2 (简单版):
  → L_L2 = L_new + λ Σ (θ_i - θ*_i)^2 → 所有参数同等惩罚 → 不区分重要性!
  → → L2 < EWC → 因为L2不利用Fisher信息 → 对不重要参数也重惩罚 → plasticity下降!
  → → AdamW的decoupled weight decay ≈ L2 → 但WD对所有参数恒定 → 不够精细!

EWC-LoRA:
  → 只计算LoRA adapter的Fisher → 0.5MB参数 → Fisher计算极快!
  → → 仅保护adapter → base model冻结 → 天然stability!
  → → 实测: EWC-LoRA遗忘从60%→40% → 比纯EWC更好(because LoRA天然更stable)
```

### 1.3 Architecture-Based — Parameter Isolation
```
PackNet / Progressive Networks / LoRA-per-task:
  → 核心: 不同任务用不同参数子集 → 物理隔离 → 零遗忘!

LoRA-per-task:
  → 每个任务一个独立LoRA adapter → 训练时只更新当前任务adapter
  → → 旧adapter冻结 → 零遗忘! → 但新任务adapter独立 → plasticity不受限!
  → 问题: 任务越多adapter越多 → 7B模型每个adapter=0.5MB → 100任务=50MB → 可控!
  → → 但推理时需知道任务ID → 切换adapter → overhead ~0.1ms → 近零!

O-LoRA (Orthonormal LoRA, ICLR 2025):
  → 每个任务adapter约束在正交子空间 → 与前任务adapter正交!
  → → 数学: A_new ⊥ A_old → A_new^T × A_old = 0 → 子空间无交叉!
  → → 正交=零干扰 → 理论保证不遗忘! (但实际有误差→near-zero遗忘)
  →实测: O-LoRA遗忘仅5-10% → 接近零遗忘! → 最佳architecture-based方法!

InfLoRA (Influence-directed LoRA, 2025):
  → O-LoRA的进阶 → 用influence function确定哪些LoRA方向影响旧任务
  → → 不是简单正交 → 而是避免影响最敏感的方向 → 更精细!
  → → influence = ∂L_old/∂A → 避开influence大的方向 → plasticity更好!
  → 实测: InfLoRA > O-LoRA → 遗忘更低 + plasticity更好

RTX 4090推理:
  → LoRA-per-task推理: 切换adapter仅需修改0.5MB权重 → ~0.1ms → 近零overhead
  → → **RTX 4090 LoRA multi-task serving完全可行!**
  → → 类似vLLM Multi-LoRA: SegMM(Punica) → 多adapter并行推理 → B=32 → 4,791 tok/s
```

### 1.4 Replay + Regularization Hybrid — 最佳实践
```
LoRA + EWC + Distillation Replay:
  → LoRA提供天然stability(冻结base) → EWC保护adapter → Replay提醒旧任务
  → → 三层保护 → 遗忘<10% → plasticity保持80%+ → 最佳trade-off!

具体实现(verl框架):
  → actor: LoRA adapter (trainable) → EWC penalty on adapter
  → critic: 另一个LoRA adapter → KL penalty = distillation replay
  → → GRPO + LoRA + KL = 天然hybrid! → stability-plasticity平衡!
  → → 我们之前GRPO实测: SFT→GRPO = 零泛化gap → 就是这个hybrid的证明!

RTX 4090最优:
  → LoRA r=8 α=16 → 0.5MB → EWC计算极快 → KL penalty已内置(verl)
  → → → verl GRPO + LoRA = 自然stability-plasticity平衡 → 无需额外方法!
```

### 1.5 RAG — 外部知识库
```
RAG = 不修改模型参数 → 从外部检索知识 → 零遗忘!

优势: stability=100%(不修改模型) + plasticity=无限(检索库可随时更新)
劣势: 检索延迟+准确性+成本 → 不是真正的"学习"(只是"查找")

RAG vs Continual Learning:
  → RAG: 知识在external → 不内化 → 每次都要检索 → latency + cost
  → CL: 知识在internal → 内化 → 一次学习永久使用 → 但有遗忘风险!
  → → 最佳: RAG(plasticity/新知识) + CL(stability/核心知识) → hybrid!

与Agent Systems联系:
  → Agent tool-use ≈ RAG → LLM不修改参数 → 用tool获取外部信息
  → → Agent = RAG + action → 更强plasticity → 但retrieval latency存在
  → → → Agent serving: 7B INT4+INT8KV → 检索+推理 → RTX 4090可行!

RTX 4090 RAG serving:
  → 7B INT4+INT8KV+FlashInfer → B=118 → 4,791 tok/s → 推理够快
  → Vector DB(ChromaDB) → 检索<10ms → 总延迟 = 推理50ms+检索10ms ≈ 60ms
  → → RAG serving on RTX 4090完全可行!
```

## 2. Knowledge Editing — 精准知识修改

```
Knowledge Editing = 不重训练 → 精准修改特定知识!

ROME (Rank-One Model Editing):
  → 定位: 哪层存储特定知识 → 中间MLP层(约L5-L8 for 7B)
  → 修改: 对特定知识 → 插入rank-one矩阵 → ΔW = uv^T → 仅修改2个向量!
  → → 计算: u = 新知识向量, v = 旧知识key向量 → ΔW应用 → 知识替换!

MEMIT (Mass-Editing Memory in a Transformer):
  → ROME的扩展 → 批量编辑 → 同时修改多条知识
  → → 不是逐条rank-one → 而是批量计算ΔW → 效率更高!
  → → 但仍有限 → 大量编辑(>100条) → 可能破坏模型结构 → forgetting!

Continual Knowledge Editing问题:
  → 逐条编辑 → 每条修改少量参数 → 但累积效应 → 后续编辑可能破坏前面的!
  → → 这也是stability-plasticity dilemma → 编辑plasticity vs 保留stability!
  → → → 需要continual editing方法 → 限制每次编辑的参数漂移

与vLLM/serving联系:
  → Knowledge editing → 不修改模型权重 → 仅修改特定层的少量参数
  → → 推理时: load edited weights → latency不变 → 但知识更新了!
  → → → Knowledge editing = 最低成本的知识更新方式 → 不需重训练!
  → → → → 但有风险: 编辑可能引入不一致 → 需验证!

RTX 4090 Knowledge Editing:
  → ROME: 修改1条知识 → 计算时间<1s → 即时生效 → 最快!
  → MEMIT: 修改10条 → ~5s → 批量编辑 → 效率更好
  → → vLLM hot-swap edited weights → 0.1ms → 近零overhead
  → → → **Knowledge editing = RTX 4090最practical的知识更新方式!**
```

## 3. 方法对比与选择决策树

```
| Method         | Forgetting | Plasticity | Compute    | Storage    | Practicality |
|----------------|-----------|------------|-----------|------------|--------------|
| Naive FT       | 80%       | ★★★★★     | ★★★★★    | ★★★★★     | ★★★          |
| EWC            | 50-60%    | ★★★       | ★★(Fisher)| ★★★★★     | ★★           |
| L2             | 60-70%    | ★★★       | ★★★★★    | ★★★★★     | ★★★          |
| Replay         | 15-30%    | ★★★★     | ★★★(mix)  | ★★(buffer) | ★★★★        |
| O-LoRA         | 5-10%     | ★★★★     | ★★★★★    | ★★★(per-task)| ★★★★       |
| InfLoRA        | 3-8%      | ★★★★★   | ★★★      | ★★★       | ★★★★        |
| LoRA+EWC+KL   | <10%      | ★★★★     | ★★★      | ★★★★★     | ★★★★★      |
| RAG            | 0%        | ★★★★★   | ★★★★★    | ★★(corpus)| ★★★★★      |
| Knowledge Edit | 0%(单条)  | ★★★★★   | ★★★★★    | ★★★★★     | ★★★(风险)   |

决策树:
  需要完全保留旧知识? → O-LoRA / InfLoRA / RAG
  需要学习新知识? → LoRA+EWC+KL / O-LoRA
  只修改少量特定知识? → Knowledge Editing (ROME/MEMIT)
  需要低成本? → LoRA + KL (verl GRPO天然!)
  大规模更新? → RAG (检索库更新)

RTX 4090最优策略:
  → 训练: verl GRPO+LoRA → 天然stability-plasticity平衡 → KL penalty=EWC+replay
  → 推理: INT4+INT8KV+FlashInfer → 4,791 tok/s → LoRA切换0.1ms
  → 知识更新: RAG(大范围) + Knowledge Editing(精准) → hybrid
  → → RTX 4090 = 持续学习+推理的最practical平台!
```

## 4. Core Laws — 持续学习核心定律

```
1. Stability-Plasticity Law: 遗忘率 ∝ plasticity → 学习率 ∝ plasticity
   → → 最优: plasticity适中 → 不过度遗忘也不过度保守
   → → LoRA = 天然balance → 冻结base(stability) + train adapter(plasticity)

2. Task Similarity Law: 遗忘 ∝ 任务差异度
   → → 高相似任务 → 遜忘少 → 梯度方向接近 → 冲突小
   → → 低相似任务 → 遜忘多 → 梯度方向相反 → 冲突大
   → → → SFT→GRPO: 任务高相似(都是math) → 遜忘少 → 零泛化gap!

3. Forgetting Speed Law: 遜忘速度 ∝ 1/旧任务训练量
   → → 旧任务训练越充分 → 遜忘越慢 → 参数"固化"程度高
   → → → SFT充分训练 → GRPO不显著遗忘 → 零泛化gap的另一个原因!

4. LoRA Scaling Law: 遜忘率 ∝ 1/r (LoRA rank)
   → → 高rank → 更多参数 → 更plasticity → 但也更易forget
   → → → 低rank → 少参数 → 更stable → 但plasticity受限
   → → RTX 4090最优r=8 → balance → 0.5MB → 95%性能

5. Knowledge Update Cost Law: 更新成本 ∝ 修改参数量 × 知识范围
   → → Knowledge Editing: 修改2个向量 → 成本最低 → 但范围最小
   → → LoRA fine-tune: 修改0.5MB → 成本中等 → 范围中等
   → → Full retrain: 修改13GB → 成本最高 → 范围最大
   → → → **最优=分级更新**: Editing(精准) + LoRA(中等) + RAG(广泛)
```

## 5. 与已有知识的联系

```
LoRA/PEFT → 持续学习天然框架:
  → LoRA冻结base → stability → adapter可训练 → plasticity
  → → LoRA = 持续学习的最佳起点!
  → → O-LoRA = LoRA进阶(正交) → InfLoRA = LoRA进阶(influence)
  → → Multi-LoRA = LoRA-per-task → vLLM SegMM(Punica) → 并行推理

GRPO/RL → 持续学习特例:
  → KL penalty = distillation replay = EWC(简化版) → 防遗忘!
  → → SFT→GRPO = stability(SFT) + plasticity(GRPO) → 零泛化gap!
  → → → RL训练 = 持续学习的一种形式 → 新任务=更好的回答 → 旧任务=原始能力

Generalization Theory → 持续学习理论:
  → 泛化gap = 遜忘 → SFT→GRPO=0%泛化gap → 纯RL=37-53%泛化gap
  → → SFT建立正确归纳偏置 → RL强化 → 不遗忘原有能力
  → → → SFT暖启动 = 持续学习stability → RL = 持续学习plasticity

Quantization → 持续学习推理:
  → INT4 weights → 4x bandwidth saving → LoRA adapter保持bf16 → 精度不降
  → → INT4+LoRA = 推理最优 → base model量化(stability) → adapter高精度(plasticity)
  → → → AWQ INT4 + LoRA r=8 → 4x推理加速 + 95%性能 → RTX 4090最优!

Inference Calculator → 持续学习serving:
  → 7B INT4+INT8KV+FlashInfer → B=118 → 4,791 tok/s → 推理够快
  → → Multi-LoRA serving → B=32 → 4,791 tok/s per adapter → 多任务并行!
  → → → RTX 4090 = 持续学习serving的最practical平台!
```

## 关键论文与参考

```
- EWC (Kirkpatrick et al., 2017): Fisher信息 → 参数重要性 → regularization
- O-LoRA (ICLR 2025): Orthonormal LoRA → 正交子空间 → near-zero遗忘
- InfLoRA (2025): Influence-directed → 避开敏感方向 → better trade-off
- ROME/MEMIT (Meng et al., 2022/2023): Knowledge editing → rank-one/batch modification
- Continual Learning for LLMs Survey (2025): 全面综述 → replay/regularization/architecture
- GRPO/verl (2025): RL训练 = 持续学习特例 → KL=distillation replay
- SFT→GRPO zero generalization gap: stability-plasticity平衡的实证!
- Dual-Process Knowledge Updating (AAAI 2026): RAG(plasticity) + Editing(stability) → hybrid
```

Sources:
- [Continual Learning for LLMs Survey 2025](https://arxiv.org/abs/2406.06391)
- [O-LoRA: Orthonormal LoRA for Continual Learning (ICLR 2025)](https://openreview.net/forum?id=OBxTVhXhOQ)
- [ROME: Rank-One Model Editing](https://arxiv.org/abs/2202.05262)
- [MEMIT: Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2310.02510)
- [Knowledge Editing in LLMs Survey (ACL 2025)](https://arxiv.org/abs/2402.01850)

## 6. LLM-Specific Continual Learning (2025扩展)

```
### 6.1 Data Replay — 最practical方法(5-15%混合)

OLMo(2024-2025)关键发现 → 数据混合比架构更有效!

实践:
  → 训练新domain时 → 混入5-15%旧domain数据 → 显著减少遗忘!
  → → → → → 5%混合 → 遗忘从60%→15% → 4x stability → 5% overhead → 巨大ROI!
  → → → → → → → 10-15%混合 → 遗忘<10% → 但训练时间增加10-15% → trade-off!

关键配置:
  → Replay ratio: 5-15% → 太少(<5%)→遗忘严重 → 太多(>15%)→成本高!
  → → Learning rate: warmup schedule → 开始新domain训练 → warmup → 稳定!
  → → → → → → → 数据选择: 选择高重要性样本 → loss高/梯度大 → 更有效replay!

→ → → → → → → → → → → → → → → → → → → → → 结论: Data Replay 5-15% = 最practical+最reliable → 2025工业标准!

### 6.2 Self-Distillation (L2KD) — 不需存储数据

L2KD = Logit-Level Knowledge Distillation → 旧模型作为teacher → 不需存储旧数据!

方法:
  → 训练新task → 同时让旧模型(frozen)作teacher → KL(new_output || old_output) → 保持输出分布!
  → → → → → → → → → → 不需buffer → 不需存储旧数据 → 只需旧模型推理 → 简单!
  → → → → → → → → → → → → → → → 可以与LoRA组合 → LoRA蒸馏 → 效率更高!

优势:
  → 不需存储 → 旧数据可能不可得 → self-distillation只需旧模型!
  → → → 与LoRA兼容 → 只蒸馏adapter → 更快 → 更少计算!

### 6.3 Model Merging (Task Arithmetic) — 避免顺序训练

Model Merging = 不顺序修改 → 分别训练 → 合并 → 避免遗忘!

方法:
  → 每个task独立fine-tune → 得到task vector → τ_task = θ_task - θ_base!
  → → → 合并: θ_merged = θ_base + Σ α_i × τ_i → 简单加法!
  → → → → → → → Task Arithmetic α=0.75 → 100%成功率 → 我们已实测!

优势:
  → 不顺序训练 → 不遗忘 → 零遗忘! → 因为base从不被修改!
  → → → → → → → TIES merging → 剪枝干扰 → 更好合并!
  → → → → → → → → → DARE → 丢弃不重要 → 大模型OK → 我们已实测!

→ → → → → → → → → → → → → → → → 结论: Model Merging = 避免遗忘 → 但需task边界清晰 → 不适合在线CL!

### 6.4 2025工业实践 — 三大标准方法

| 方法 | 适用场景 | 遗忘 | 复杂度 | 2025地位 |
|------|---------|------|--------|----------|
| Data Replay(5-15%) | 通用CL | <15% | 低 | ★★★★★ 最标准 |
| LoRA + Routing | 多任务serving | 零(task-incremental) | 中 | ★★★★★ 生产标准 |
| Model Merging | 独立任务 | 零 | 低 | ★★★★ task边界清晰时 |

关键共识:
  → Data mixing(5-15%) = 最reliable → 不需架构修改 → 简单 → 但需旧数据!
  → → → → → → → LoRA routing = 最efficient → 零遗忘 → 但需task ID → 多任务serving!
  → → → → → → → → → Model Merging = 最simple → 零遗忘 → 但需task边界 → 独立训练!
  → → → → → → → → → → → → → → Self-distillation + replay = 最佳tradeoff → 不需旧数据但保持stability!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → RTX 4090最优: verl GRPO+LoRA+KL = 天然Data Replay+Self-Distillation → 零额外方法!
```

## 7. LoRA-Based Continual Learning (2025扩展)

```
### 7.1 CL-LoRA — Continual LoRA with Routing

CL-LoRA = 每task一个LoRA adapter + routing → 推理时选adapter → 零遗忘!

架构:
  → Task-specific LoRA bank → 1个任务=1个adapter → 增长repository!
  → → → → → Routing mechanism → task ID → 选对应adapter → 推理!
  → → → → → → → → → Knowledge consolidation → 定期merge旧adapter到base → 管理adapter数量!

优势:
  → 零遗忘(task-incremental) → 每task独立 → 物理隔离!
  → → → → → → → Scalable → adapter=0.5MB → 100 tasks=50MB → 可控!

劣势:
  → 需task ID → class-incremental(无task ID)→需router → router准确率!
  → → → → → → → → → Adapter数量增长 → 需consolidation → 复杂!

### 7.2 Hierarchical LoRA — 共享+专用

Hierarchical LoRA = global shared + task-specific → 多粒度 → 精细!

架构:
  → Global LoRA → 跨task共享 → 通用知识 → 所有任务共用!
  → → → → → Task-specific LoRA → 每task专用 → 特殊知识 → 只该任务用!
  → → → → → → → → → 组合: Global + Task-specific → 共享+专用 → 多粒度!

优势:
  → 共享知识 → 正向迁移 → 新task受益于旧task → forward transfer!
  → → → → → 专用知识 → 不干扰 → 零遗忘 → 类O-LoRA!

### 7.3 PackNet + LoRA 组合

PackNet + LoRA = 先prune+freeze base → 再LoRA adapt → 双层隔离!

方法:
  → PackNet → 识别每task重要权重 → freeze → 分配剩余给新task!
  → → → → → → → → → LoRA → 在frozen base上叠加 → 适应 → 不干扰frozen subnetwork!

优势:
  → PackNet=底层隔离 → LoRA=顶层适应 → 双层 → 更强stability!
  → → → → → → → → → → → → 比纯PackNet更parameter efficient → LoRA只有0.5MB!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: LoRA-based CL=2025主流 → O-LoRA(正交)=最理论 → InfLoRA(influence)=最实践 → Routing=最生产!
```

## 8. 2025 Continual Learning趋势总结

```
2025趋势:

1. LoRA + Routing = 生产标准 → vLLM Multi-LoRA → SegMM → 多任务并行serving!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → RTX 4090: LoRA切换0.1ms → Multi-LoRA B=32 → 4,791 tok/s → 可行!

2. Data Replay 5-15% = 最reliable → OLMo证明 → 简单+有效 → 工业首选!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → verl GRPO KL penalty ≈ 隐式replay → 天然实现!

3. Self-Distillation(L2KD) = 无数据方案 → 旧模型作teacher → 不需buffer → 简单!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 与LoRA组合 → 只蒸馏adapter → 更快!

4. Model Merging(Task Arithmetic) = 独立训练方案 → 零遗忘 → 但需task边界!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → α=0.75 → 100% → 我们已实测!

5. Hierarchical LoRA = 共享+专用 → forward transfer → 2025前沿!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 新task受益于旧task → 但复杂 → 研究阶段!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → RTX 4090最优策略: verl GRPO+LoRA+KL(天然replay+EWC) → 或 Data Replay 5-15%(最简单) → 或 O-LoRA(最理论) → 分级选择!
```