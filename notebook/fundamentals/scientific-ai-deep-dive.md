# Scientific AI Deep Dive — 3大领域突破(AlphaFold蛋白质→GNoME材料→AlphaProof数学) + 范式转变(AI=发现新知识, 不只是解决已知问题) + AlphaFold 2(2020 CASP14 GDT-TS远超竞争→200M+结构→50年难题) + AlphaFold 3(2024 蛋白质+DNA+RNA+小分子→但缺动力学+亲和力→限制) + Drug Discovery(30+AI药物临床→Insilico Phase II→hit rate 1%→10-15%→但临床仍是瓶颈) + GNoME(2.2M晶体≈800年传统→380K稳定→电池+催化剂) + MatterGen(2025 Microsoft扩散生成→按需属性) + AlphaProof(RL+Lean形式验证→IMO银牌4/6→机器验证→不幻觉!) + AlphaGeometry 2(19秒解IMO几何→神经+符号) + FunSearch(LLM发现新数学→cap set→超越已知最优) + Self-driving Labs(Berkeley A-Lab→机器人+AI→自动验证) + RTX 4090(小分子推理可行+VLM科学助手+Agent+LLM-as-judge) + 2026趋势(AI-as-collaborator+基础模型+跨领域+形式验证+伦理)

> 2026-06-14 | Scientific AI深度分析: 3大领域突破+范式转变 → AlphaFold 2(2020 CASP14→200M+蛋白质→50年难题解决!) → AlphaFold 3(2024扩展→蛋白质+DNA+RNA+小分子→但缺动力学→限制) → Drug Discovery(30+AI药物临床→Insilico INS018-055 Phase II→生成式分子→hit rate 10-15%→但临床瓶颈→10-12年) → Materials Science(GNoME 2.2M≈800年→380K稳定→电池+催化剂→A-Lab验证→MatterGen 2025按需生成) → Math(AI)(AlphaProof RL+Lean→IMO银牌→机器验证→不幻觉!+AlphaGeometry 2 19秒+FunSearch cap set→新知识!) → Self-driving Labs(A-Lab→机器人+AI→41验证→扩展) → 范式: AI从解决已知→发现新知识 → RTX 4090(小分子+VLM助手+Agent) → 2026(AI-collaborator+基础模型+形式验证+伦理)
> 关联: ai-expert-knowledge-map-gap-analysis.md(Scientific AI ★→★★★★), agent-system-deep-dive.md(Agent科学助手), multimodal-vlm-deep-dive.md(VLM科学理解), evaluation-benchmarking-deep-dive.md(AIME/ARC-AGI)
> 参考: AlphaFold 2(Jumper et al. 2021 Nature), AlphaFold 3(Abramson et al. 2024 Nature), GNoME(Merchant et al. 2023 Nature), AlphaProof/AlphaGeometry 2(DeepMind 2024), FunSearch(Romera-Paredes et al. 2024 Nature), MatterGen(Microsoft 2025), Insilico Medicine, Recursion, A-Lab(Berkeley 2023)

## 0. 核心定律: AI从解决已知→发现新知识 → 范式转变 → Scientific AI=AI专家跨领域创新!

```
Scientific AI = 范式转变 → 不只是解决已知问题 → 而是发现新知识!

  传统科学 → 人类提出假设 → 实验验证 → 发现新知识 → 慢(数年-数十年)!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 例: 药物发现 → 10-12年 → $2.6B → 极慢+极贵!

  AI加速科学 → AI提出假设 → AI/实验验证 → 发现新知识 → 快(数天-数月)!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → 例: AlphaFold → 50年难题 → AI秒解 → 范式转变!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 例: GNoME → 800年传统发现 → AI数月 → 范式转变!

  3大突破:
    → AlphaFold(生物学): 蛋白质折叠 → 50年难题 → 200M+结构 → 生物革命!
    → → → GNoME(材料学): 2.2M新晶体 → 800年传统 → 材料+电池+催化剂 → 材料革命!
    → → → → → AlphaProof(数学): IMO银牌 → 机器验证 → 不幻觉 → 数学革命!

  关键模式 → AI+形式验证 = 可信发现:
    → AlphaProof → RL+Lean → 形式验证 → 机器检查 → 不幻觉 → 可信!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → GNoME → GNN+DFT验证 → 物理计算 → 可信!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → AlphaFold → 模型预测+实验验证 → 双验证 → 可信!

  与我们已有知识的联系:
    → AI Infra → Scientific AI需要推理 → GPU+serving → 我们的核心!
    → → → → → → → → → → → → → → → → → → → Agent → Scientific AI = 科学Agent → 工具调用 → 我们已学!
    → → → → → → → → → → → → → → → → → → → → → → VLM → Scientific AI需要视觉理解 → 图表+分子 → 我们已学!
    → → → → → → → → → → → → → → → → → → → → → → → → Evaluation → Scientific AI需评估 → AIME/ARC-AGI → 我们已学!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: Scientific AI=AI专家的跨领域创新 → Infra+Agent+VLM+Eval → 全链路!
```

## 1. AlphaFold — 蛋白质折叠革命

```
### 1.1 AlphaFold 2 (2020) — 解决50年难题

AlphaFold 2 (Jumper et al. 2021) → CASP14 → GDT-TS远超竞争 → 50年蛋白质折叠难题 → 解决!

  蛋白质折叠问题:
    → 给定氨基酸序列 → 预测3D结构 → 50年未解 → 生物学核心难题!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 传统: X射线晶体学/冷冻电镜 → 数月-数年 → 贵+慢!

  AlphaFold 2突破:
    → CASP14 (2020) → GDT-TS=92.4 → 远超第二名(~75) → 精度接近实验!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 200M+蛋白质结构 → 几乎所有已知蛋白质 → 免费开放!

  架构创新:
    → Evoformer → 48层 → 注意力+三角更新 → 捕获残基间几何关系!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Structure Module → IPA(Invariant Point Attention) → 等变 → 3D旋转不变!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → MSA(Multiple Sequence Alignment) → 进化信息 → 同源序列 → 结构约束!

  影响:
    → 蛋白质结构 → 从数月→数分钟 → 加速1000x+ → 生物革命!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 药物设计 → 结构已知 → docking加速 → hit identification加速!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Nobel Prize consideration → 被认为Nobel级突破 → 生物学里程碑!

### 1.2 AlphaFold 3 (2024) — 扩展到相互作用

AlphaFold 3 (Abramson et al. 2024) → 蛋白质+DNA+RNA+小分子 → 相互作用 → 但限制仍存在!

  扩展能力:
    → 蛋白质-小分子(ligand) → 药物设计直接相关! → 最重要扩展!
    → → → → → 蛋白质-DNA → 基因调控理解 → 转录因子 → 重要!
    → → → → → → → 蛋白质-RNA → RNA结合蛋白 → 重要!
    → → → → → → → → → 蛋白质-蛋白质 → 复合物 → 之前Multimer → 继续!

  架构变化:
    → Diffusion Module → 替代Structure Module → 生成式 → 更灵活!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Pairwise attention → 捕获所有分子间相互作用 → 更全面!

  限制:
    → 不能预测动力学 → 单静态构象 → 但蛋白质是动态的! → 重要限制!
    → → → → → → 不能预测结合亲和力 → KD值 → 药物关键 → 不能!
    → → → → → → → → → Ligand pose accuracy ~76% → vs 90%+专用docking → 不够精确!
    → → → → → → → → → → → → 不能预测多构象 → 蛋白质有多种状态 → 不能!

  关键对比:
    | Feature | AlphaFold 2 | AlphaFold 3 |
    |---------|-------------|-------------|
    | 蛋白质单体 | ✅ 极好 | ✅ 极好 |
    | 蛋白质-蛋白质 | ✅ Multimer | ✅ 更好 |
    | 蛋白质-ligand | ❌ | ✅ 但76%精度 |
    | 蛋白质-DNA/RNA | ❌ | ✅ 支持 |
    | 动力学 | ❌ | ❌ 仍不能 |
    | 结合亲和力 | ❌ | ❌ 仍不能 |
    | 多构象 | ❌ | ❌ 仍不能 |

→ → → → → → → 结论: AlphaFold=生物革命 → AF2解决50年难题 → AF3扩展相互作用 → 但动力学/亲和力仍是限制 → 需MD+docking补充!
```

## 2. Drug Discovery — AI加速药物发现

```
### 2.1 AI Drug Discovery Pipeline

传统药物发现 → 10-12年 → $2.6B → 极慢+极贵!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → AI加速 → 缩短preclinical → 但临床仍是瓶颈 → 不能缩短Phase I-III!

  AI加速阶段:
    → Target Identification → AlphaFold → 结构已知 → 加速2-4月!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Hit Identification → 生成式分子 → hit rate从1%→10-15% → 加速!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Lead Optimization → AI辅助优化 → ADMET+binding → 加速!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Clinical Trial → AI辅助设计 → 但仍需Phase I-III → 不能加速!

### 2.2 生成式分子设计

2025生成式分子设计 → Diffusion+Transformer → 多目标 → 更好!

  方法:
    → Diffusion-based(DiffSBDD/TargetDiff) → 3D分子生成 → 条件于蛋白质口袋 → 最前沿!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Transformer-based → SMILES序列生成 → 快 → 但3D信息少!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → VAE → 变分自编码器 → 潜在空间 → 可控生成 → 但质量较低!

  多目标优化:
    → 结合亲和力 → 药效 → 最重要!
    → → → → → 可合成性 → 能否制造 → 实际必须!
    → → → → → → → ADMET → 吸收+分布+代谢+排泄+毒性 → 安全必须!
    → → → → → → → → → → → 同时优化 → 多目标 → Pareto最优 → 2025方法!

  关键数据:
    → 传统HTS hit rate: ~1% → 10000分子→100hit → 低!
    → → → → → → → → → → AI生成+筛选 hit rate: 10-15% → 100分子→10-15hit → 10-15x提升!
    → → → → → → → → → → → → → → → → Insilico INS018-055 → AI设计 → Phase II → 首个AI设计药物进临床!

### 2.3 临床瓶颈

AI加速发现 → 但临床仍是瓶颈 → Phase I-III不能加速!

  → Phase I(安全性): 1-2年 → 人体首次 → 安全测试 → 不能加速!
  → → → → → → → Phase II(有效性): 2-3年 → 患者测试 → 效果验证 → 不能加速!
  → → → → → → → → → → → → Phase III(大规模): 3-4年 → 大规模验证 → 确认 → 不能加速!

  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: AI压缩preclinical → 但10-12年 → 临床仍6-8年 → 总时间缩短30-40% → 不是100%!

### 2.4 AI Drug Discovery市场

2025 → $4.1B市场 → 28% CAGR → 快增长!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 大药厂+AI公司 → Roche-Recursion/Merck-Exscientia/Sanofi-Insilico → 合作!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 30+AI设计药物 → 临床 → 2025里程碑!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: Drug Discovery = AI压缩preclinical+临床仍是瓶颈+hit rate 10-15x提升+30+药物临床!
```

## 3. Materials Science — AI发现新材料

```
### 3.1 GNoME (2023) — 2.2M新晶体

GNoME (Merchant et al. 2023 Nature) → Graph Networks for Materials Exploration → 2.2M新晶体!

  传统材料发现 → 慢 → 一个材料数月-数年 → 800年才累积2.2M!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → GNoME → 数月 → 2.2M → 380K稳定 → 加速800年!

  方法:
    → GNN(Graph Neural Network) → 晶体→图 → 原子→节点 → 化学键→边 → 学习!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 预测: 形成能(formation energy) → 稳定性 → 是否可以合成!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Active learning → 高不确定性 → 优先DFT验证 → 逐步改进 → 效率!

  应用:
    → 电池材料 → 锂离子+钠离子 → 新电极+新电解质 → 2025实验验证进行中!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 催化剂 → 绿氢+CO₂还原+燃料电池 → 绿色化学 → 关键!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 超导材料 → 热电材料 → 专项筛选 → 前沿!

  验证:
    → Berkeley A-Lab → Autonomous Lab → 机器人合成+表征 → 自动验证!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 41个GNoME预测 → A-Lab验证 → 全部成功 → 可信!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → A-Lab 2.0 → 2025-2026 → 每年数百材料 → 扩展!

### 3.2 MatterGen (2025) — 按需生成

MatterGen (Microsoft 2025) → Diffusion生成 → 按属性生成 → 目标导向!

  → 不是预测稳定性 → 而是生成指定属性的材料 → 更强!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 例: "生成高导电+低形成能材料" → 模型生成 → 满足属性 → 目标导向!

  方法:
    → Diffusion模型 → 类似图像生成 → 但在3D晶体空间 → 条件生成!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 条件: 属性(导电性/形成能/磁性等) → 引导生成 → 目标属性!

  与GNoME对比:
    → GNoME → 预测已有+候选 → 被动 → 从已知中筛选!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → MatterGen → 生成新+指定属性 → 主动 → 从需求出发!

  应用:
    → 电池正极 → 按需 → 高容量+稳定 → 生成候选 → DFT验证!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 催化剂 → 按需 → 高活性+低成本 → 生成候选 → 实验验证!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: GNoME=被动筛选→380K稳定→材料革命 → MatterGen=主动生成→按需属性→2025前沿!
```

## 4. AI for Mathematics — AlphaProof + AlphaGeometry + FunSearch

```
### 4.1 AlphaProof — RL+Lean形式验证

AlphaProof (DeepMind 2024) → RL+Lean → IMO银牌(4/6, 28/42) → 机器验证 → 不幻觉!

  架构:
    → Step 1: 自然语言数学 → formalization → Lean语句 → 翻译!
    → → → → → → → Step 2: Lean问题 → RL proof search → AlphaZero-style → 搜索证明!
    → → → → → → → → → → → → Step 3: 发现证明 → Lean验证 → 机器检查 → 100%可信!

  关键创新:
    → AlphaZero-style RL → 搜索树 → 逐步构建证明 → 策略网络+价值网络!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Lean形式验证 → 所有证明机器验证 → 不幻觉 → 不错误 → 100%可信!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 这是与LLM最大的区别 → LLM生成→可能错误 → AlphaProof生成→机器验证→确保正确!

  IMO 2024成绩:
    → AlphaProof → 4/6(algebra+number theory) → 银牌级!
    → → → → → → → AlphaGeometry 2 → 1/6(geometry) → 19秒解决 → 银牌级!
    → → → → → → → → → → → → → 总: 5/6 → 28/42 → 银牌(差金牌4分) → 极强!

  与我们RL知识的联系:
    → AlphaZero-style RL → 与GRPO同框架 → 搜索+策略 → RL!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Self-correction → RL loop → reward=证明通过 → 类似GRPO reward=答案正确!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Formal verification = ground truth reward → 比人类评分更可靠 → 不会reward hacking!

### 4.2 AlphaGeometry 2 — 神经+符号

AlphaGeometry 2 → 几何推理 → 神经模型+符号引擎 → 19秒解IMO几何!

  架构:
    → Neural model → 辅助构造(auxiliary point) → 提出"这里加一个点" → 创意!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Symbolic engine → 逻辑推导 → 从构造出发 → 推理 → 证明 → 精确!

  关键发现:
    → 纯神经 → 不能保证正确 → 可能幻觉 → 但有创意!
    → → → → → → → 纯符号 → 保证正确 → 但没创意 → 不能发现新构造!
    → → → → → → → → → → → → → 神经+符号 → 创意+精确 → 互补 → 最强!

### 4.3 FunSearch — LLM发现新数学

FunSearch (Romera-Paredes et al. 2024 Nature) → LLM+进化 → 发现新数学 → cap set!

  方法:
    → LLM生成程序 → 评估 → 保留好的 → 进化 → 迭代 → 发现新解!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 类似Genetic Programming → 但用LLM代替变异 → 更有创意!

  成果:
    → Cap set problem → 发现新解 → 超越之前最好的构造解 → 新知识!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 这是AI发现新数学知识 → 不只是解决已知问题 → 范式转变!

  与我们知识联系:
    → LLM生成+评估 → 类似Self-Instruct → 生成+筛选 → 我们已学!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 进化搜索 → 类似RL → reward=评估分数 → 策略改进 → RL框架!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: AI for Math → AlphaProof(形式验证→不幻觉)+AlphaGeometry(神经+符号)+FunSearch(发现新知识) → 范式转变!
```

## 5. Self-driving Labs — AI+机器人=自动实验

```
Self-driving Labs = AI提出假设 → 机器人执行实验 → AI评估结果 → 自动循环!

  Berkeley A-Lab(2023):
    → GNoME预测 → A-Lab机器人合成 → XRD+其他表征 → 自动验证 → 成功!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 41个预测 → 全部验证 → 100%成功 → 可信!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → A-Lab 2.0(2025-2026) → 每年数百 → 扩展 → 加速!

  流程:
    → AI预测 → 哪些材料稳定 → 值得合成 → 假设!
    → → → → → → 机器人 → 自动合成 → 自动表征 → 实验!
    → → → → → → → → → → AI分析 → 是否稳定 → 是否有目标属性 → 验证!
    → → → → → → → → → → → → → → → → AI更新 → 新数据 → 改进模型 → 下一步 → 循环!

  与Agent的联系:
    → Self-driving Lab = 科学Agent → 工具=机器人+仪器 → Agent架构!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → ReAct → Think(AI分析)→Act(机器人执行)→Observe(结果) → Agent模式!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 我们已学: Agent=Planning+Tool Use+Memory+Execution → Self-driving Lab=科学Agent!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: Self-driving Labs=科学Agent → AI+机器人 → 自动验证 → 闭环 → 加速科学发现!
```

## 6. RTX 4090 Scientific AI策略

```
### 6.1 RTX 4090能力限制

RTX 4090 (24GB) → 小规模推理 → Scientific AI应用有限但可行!

  可行:
    → 小分子推理 → SMILES生成 → Transformer → 7B → 可行!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → VLM科学助手 → 图表理解+OCR+分子图 → Phi-3.5-Vision-4B → 可行!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Agent → 科学文献搜索+工具调用 → 7B+MCP → 可行!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → LLM-as-judge → 科学答案评估 → 7B+评判 → 可行!

  不可行:
    → AlphaFold推理 → 模型太大 → 需专业GPU → RTX 4090不够!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 大规模DFT计算 → 需集群 → RTX 4090不够!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 大分子生成 → Diffusion模型太大 → RTX 4090不够!

### 6.2 RTX 4090 Scientific AI最优策略

RTX 4090 → 科学Agent+VLM助手 → 最practical!

  策略:
    → 7B INT4+SGLang → 科学文献Agent → 搜索+摘要+分析 → MCP → 可行!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Phi-3.5-Vision-4B INT4 → VLM科学助手 → 图表+分子图+论文理解 → 可行!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Cascade Serving → Phi-3 routing + 7B reasoning → 分级 → 可行!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: RTX 4090 → 科学Agent+VLM助手 → 7B INT4+SGLang+MCP → 最practical路径!
```

## 7. 核心规律

```
Scientific AI核心规律:

  1. 范式转变: AI从解决已知→发现新知识 → AlphaFold/GNoME/FunSearch → 3大突破!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 不只是效率提升 → 而是发现方式改变 → 从人类假设→AI假设!

  2. AI+形式验证=可信发现: AlphaProof+Lean → 机器验证 → 不幻觉 → 100%可信!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → GNoME+DFT → 物理验证 → 可信 → 但AlphaProof更精确(形式验证=数学100%可信)!

  3. 神经+符号=最强: AlphaGeometry → 神经创意+符号精确 → 互补 → 最强!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 类似我们GRPO → RL(创意)+ground truth(验证) → 同模式!

  4. Self-driving Labs=科学Agent: AI+机器人 → 自动实验 → 闭环 → 加速发现!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → ReAct模式 → Think→Act→Observe → Agent架构 → 我们已学!

  5. 临床仍是瓶颈: Drug Discovery → AI压缩preclinical → 但Phase I-III不能加速!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 10-12年 → AI缩短30-40% → 不是100% → 临床必须人工!

  6. RTX 4090 → 科学Agent+VLM助手 → 小规模但practical → Infra×Science交叉!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 7B INT4+SGLang+MCP → 科学文献Agent → 最practical路径!

  知识Gap修复:
    → Scientific AI从★(1/5) → ★★★★(4/5) → AlphaFold 2/3+GNoME+MatterGen+AlphaProof+AlphaGeometry+FunSearch+Drug Discovery+Self-driving labs → 全面!
    → → → → → 但仍需实践 → GPU可用时 → VLM科学助手 → Phi-3.5-Vision → 图表+分子 → 实测!
```

## 参考文献

```
1. 蛋白质:
   - AlphaFold 2: Jumper et al. 2021, Nature, doi.org/10.1038/s41586-021-03819-2
   - AlphaFold 3: Abramson et al. 2024, Nature, doi.org/10.1038/s41586-024-07487-w
   - ColabFold: Mirdita et al. 2022, Nature Methods
   - ESMFold: Lin et al. 2023, Meta AI

2. 药物:
   - Insilico Medicine: INS018-055, insilico.com
   - Recursion Pharmaceuticals: recursion.com
   - Exscientia: exscientia.ai
   - DiffSBDD/TargetDiff: 3D molecular generation

3. 材料:
   - GNoME: Merchant et al. 2023, Nature, doi.org/10.1038/s41586-023-06735-9
   - MatterGen: Microsoft Research, 2025
   - Materials Project: materialsproject.org
   - A-Lab: Berkeley Lab, 2023

4. 数学:
   - AlphaProof: DeepMind, 2024, deepmind.google/blog
   - AlphaGeometry 2: DeepMind, 2024
   - FunSearch: Romera-Paredes et al. 2024, Nature
   - Lean: leanprover-community.github.io

5. 我们的笔记:
   - ai-expert-knowledge-map-gap-analysis.md → Scientific AI gap评估
   - agent-system-deep-dive.md → Agent架构(科学Agent)
   - multimodal-vlm-deep-dive.md → VLM(图表+分子理解)
   - evaluation-benchmarking-deep-dive.md → AIME/ARC-AGI评估
   - continual-learning-deep-dive.md → RL+self-correction

Sources:
- [AlphaFold 3 Blog](https://blog.google/technology/ai/google-deepmind-isomorphic-labs-alphafold-3-ai-model/)
- [AlphaProof Blog](https://deepmind.google/blog/ai-solves-international-mathematical-olympiad-problems-at-a-silver-medal-level/)
- [GNoME Paper](https://www.nature.com/articles/s41586-023-06735-9)
- [FunSearch Paper](https://www.nature.com/articles/s41586-023-06899-7)
- [A-Lab](https://newscenter.lbl.gov/2023/autonomous-lab/)
