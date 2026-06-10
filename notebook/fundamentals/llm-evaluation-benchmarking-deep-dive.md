# LLM Evaluation & Benchmarking Deep Dive

> 2026-06-08 | 评估=衡量LLM能力的核心工具! 5代演进: static→dynamic→human→holistic→agent, 关键问题=数据污染+饱和+偏差
> 关联: scaling-laws-deep-dive.md, ai-safety-alignment-deep-dive.md, generalization-theory.md

## 0. 核心定律: Evaluation = 模型能力的度量衡

```
评估根本矛盾:
  → 需要全面: 模型有很多能力 → 语言/推理/代码/数学/安全...
  → → 但每个benchmark只测一部分 → 不全面!
  → → → 需要多benchmark组合 → 但组合结果如何interpret?

  → 需要客观: benchmark分数应该反映真实能力
  → → 但数据污染(contamination) → 训练集包含测试集 → 分数虚高!
  → → → 需要防污染 → 但新benchmark制作成本高 → 月度更新(LiveBench)才有!

  → 需要可比较: 不同模型分数应该可横向对比
  → → 但benchmark不同版本/不同prompt → 分数不可直接对比!
  → → → Chatbot Arena(Elo) = 最公平 → 但有噪声 → 人类偏好 ≠ 真实能力!

与泛化理论联系:
  → training-eval gap = 泛化gap → GRPO 37.5% / SFT→GRPO 0%
  → → → 评估分数 ≠ 训练分数 → 评估测试的是"模型是否真正理解"而非"是否记住训练数据"
  → → → → SFT暖启动 → 正确归纳偏置 → 泛化好 → 评估分数接近训练分数!
  → → → → → 纯RL → 学高分而非正确 → reward hacking → 评估分数远低于训练分数!

RTX 4090影响:
  → 本地评估 → 不需云API → 7B INT4推理 → 4,791 tok/s → 1小时跑完MMLU!
  → → → 本地benchmark = 最省成本 + 最可控(无数据泄露) + 最快迭代!
  → → → → RTX 4090 = LLM评估最优平台!
```

## 1. 五代评估演进

### 1.1 Static Benchmarks — MMLU/HumanEval
```
MMLU (Massive Multitask Language Understanding):
  → 57个学科 → 14,000+题 → 4选择 → STEM+人文+社科+其他
  → → 测量"知识广度" → 从小学到研究生 → 跨领域理解力
  → 问题: 4选择 → 简单 → 模型可通过模式匹配答对 → 不需要真正理解!
  → → MMLU-Pro: 10选择 → 更难 → 更不容易shortcut → 但仍然static!

HumanEval (Code Generation):
  → 164道Python编程题 → 函数签名+docstring → 生成完整函数 → pass@k指标
  → → pass@1 = 一次通过率 → pass@100 = 100次中至少一次通过 → 评估codegen能力
  → 问题: 简单题 → 大模型pass@1>90% → 饱和! → 不再区分模型!
  → → HumanEval+ → 更多test case → 80x测试 → 捕获更多edge case → 更公平!
  → → SWE-bench → 真实GitHub issue → 最practical → 但最难!

数据污染问题:
  → 训练数据可能包含benchmark题目 → 模型"记住"答案 → 分数虚高!
  → → 检测方法: n-gram重叠检测 → PPL检测(模型对测试题异常低loss → 可能见过!)
  → → → LiveBench: 每月更新 → 新题 → 不可能记住 → 防污染!

RTX 4090评估MMLU:
  → 7B INT4推理 → 4,791 tok/s → 14,000题 → 每题1-2 tokens → ~3秒跑完全部!
  → → → 本地MMLU评估 = 极快 → 立即反馈 → 训练迭代加速!
  → → → → 但要注意: 本地跑可能有prompt偏差 → 需标准化评估脚本!
```

### 1.2 Dynamic/Live Benchmarks — LiveBench/MMLU-Pro
```
LiveBench (2025):
  → 每月更新 → 新题来自最新数学竞赛/编程比赛/研究论文
  → → 不可能contamination → 因为题目发布后才创建!
  → → → 最公平 → 但覆盖面窄 → 每月只有几十题 → 不如MMLU全面!

MMLU-Pro (2025):
  → 10选择 → 更难 → 更robust → 但仍然static → 仍然有contamination风险!
  → → 但10选择 >> 4选择 → 模式匹配更难 → 需要真正推理!
  → → → MMLU-Pro = static benchmark的极限 → 但仍需动态更新防污染!

ARC-AGI (Chollet's Abstraction Reasoning Corpus):
  → 核心思想: 测novel problem-solving → 不是pattern matching!
  → → 格栅推理 → 新pattern → 模型必须从3-5例子推断规则 → 不可记忆!
  → → → ARC = 真正测reasoning → 但模型表现很差 → o1也只解决部分!
  → → → → ARC = frontier benchmark → 推理能力的真正测试!

RTX 4090跑ARC-AGI:
  → 7B推理 → 每题需要多步推理 → 每步0.5ms → 5步 → 2.5ms → 极快!
  → → → 但7B模型ARC分数很低 → 需要更强的模型或更好的推理方法!
```

### 1.3 Human Preference — Chatbot Arena
```
Chatbot Arena (LMSYS):
  → 人类盲评 → 两个模型回答同一问题 → 用户投票哪个更好 → Elo排名!
  → → 最动态 → 人类偏好最真实 → 但有噪声 → 不同用户标准不同!
  → → → Elo = Bradley-Terry模型 → P(A beats B) = σ(rating_A - rating_B)
  → → → → 类似国际象棋排名 → 可比较不同模型!

优势:
  → 动态 → 不可能contamination → 因为问题来自真实用户!
  → 全面 → 涵盖所有话题 → 不限于特定领域!
  → 真实 → 测的是用户实际体验 → 不是静态分数!

劣势:
  → 噪声 → 人类偏好不稳定 → 同一用户不同时间可能选不同模型!
  → 偏差 → 长回答可能更受欢迎(但不一定更好!) → 需控制长度!
  → 成本 → 需大量人类标注 → 每次投票=1分钟 → 10万投票=1667小时!

Arena分类:
  → Chat Arena: 对话能力 → 最通用 → 主要排名
  → Coding Arena: 代码生成 → 专业排名 → 更technical
  → Vision Arena: 多模态 → 视觉+语言 → 新兴!

与GRPO联系:
  → GRPO reward = 自动评分 → 数学题correct=1, wrong=0 → deterministic!
  → → → Chatbot Arena = human reward → 人类偏好 → subjective!
  → → → → GRPO reward更客观 → Arena reward更真实 → 两者互补!
  → → → → → DPO = 从Arena数据训练 → 人类偏好 → RLHF → 与GRPO同框架!
```

### 1.4 Holistic — HELM
```
HELM (Holistic Evaluation of Language Models, Stanford):
  → 不只测accuracy → 还测calibration/robustness/fairness/bias/toxicity/efficiency!
  → → 7维度 → 全面 → 但复杂 → 需大量计算!
  → → → HELM = 最全面的LLM评估 → 但不是最practical → 太慢!

关键维度:
  → Accuracy: 正确率 → MMLU/推理/代码 → 基础能力
  → Calibration: P(correct|confidence) → 模型是否"知道自己不知道什么"?
  → → → 好模型: 高confidence=高accuracy → 低confidence=低accuracy → 知道自己边界!
  → → → 差模型: confidence和accuracy无关 → 不知道自己是否正确!
  → Robustness: 对抗性输入 → 格式变化/措辞变化 → 分数不变?
  → → → 好模型: 同题不同表述 → 同答案 → 稳定!
  → → → 差模型: 稍微改表述 → 答错 → 不stable!
  → Fairness: 不同群体 → 性别/种族/文化 → 分数是否公平?
  → → → 好模型: 所有群体准确率相近 → fair!
  → → → 差模型: 某群体准确率低 → 有偏!
  → Efficiency: 推理速度/内存/成本 → RTX 4090指标 → Infra角度!
  → → → HELM也测推理效率 → 对Infra工程师很有价值!

RTX 4090 HELM评估:
  → 7B INT4推理 → 完整HELM → 需跑16个场景 → ~2小时 → 可接受!
  → → → HELM efficiency维度 → 7B INT4+INT8KV → 4,791 tok/s → 推理效率最高!
  → → → → RTX 4090 = HELM效率评估的最practical平台!
```

### 1.5 Agent/Tool-Use Benchmarks — GAIA/WebArena
```
GAIA (General AI Assistants):
  → 真实世界任务 → 需推理+浏览+工具 → 最接近真实使用场景!
  → → Level 1: 简单 → Level 2: 中等 → Level 3: 困难 → 模型普遍困难!
  → → → GAIA = Agent能力的前沿测试 → 但需要tool execution → 复杂!

WebArena:
  → 网页操作 → 需要Agent操控浏览器 → 搜索/导航/表单 → 真实web任务!
  → → 测量: Agent能否完成真实web操作 → 不是生成文本 → 是执行行动!
  → → → WebArena = Agent系统的practical benchmark → 但需要环境设置!

OSWorld:
  → 操控操作系统 → 文件管理/应用操作/终端命令 → 最comprehensive!
  → → 需要GUI理解+命令执行+多步推理 → 最难 → 模型几乎0分!

与Agent Systems联系:
  → Agent benchmark = 测tool-use+multi-step reasoning+error recovery
  → → 不像static benchmark → Agent需要execute → 需要sandbox → 复杂!
  → → → 但这是未来方向 → Agent能力越来越重要!

RTX 4090 Agent评估:
  → 7B INT4 Agent → 工具调用 → 每步50ms → 10步 → 500ms → 快!
  → → → 本地Agent评估 → 不需云API → 最省成本!
  → → → → 但7B模型Agent能力有限 → 需更大模型(70B)或蒸馏!
```

## 2. 评估方法论

```
评估设计原则:
  → 防contamination: LiveBench(月度更新) > MMLU(static)
  → 防shortcut: MMLU-Pro(10选择) > MMLU(4选择) > ARC-AGI(推理)
  → 防bias: HELM(多维度) > 单指标(Accuracy only)
  → 防noise: Chatbot Arena(Elo+B-T模型) > 单次投票

评估指标:
  → Accuracy: 简单 → 直观 → 但不全面
  → pass@k: code generation → k次尝试通过率 → k越大越宽松!
  → F1/BLEU/ROUGE: 生成质量 → 与参考答案比较 → 但reference可能不唯一!
  → Elo: Chatbot Arena → 相对排名 → 最公平但最慢!
  → Calibration: P(correct|confidence) → 模型自信度 → 是否知道边界!
  → cos_sim: 数值精确度 → 我们benchmark中用 → SFT→GRPO=0.9999 → 接近完美!

评估pipeline(RTX 4090本地):
  → Step 1: 下载benchmark数据(HF mirror) → 本地存储
  → Step 2: 7B INT4推理 → 4,791 tok/s → 快速生成答案
  → Step 3: 自动评分 → deterministic(数学/代码) → 或LLM-as-judge(开放题)
  → Step 4: 记录分数 → 与其他模型对比 → 发现改进方向
  → → → 本地评估 = 最快最省 → 1小时完成全benchmark!
```

## 3. Core Laws — 评估核心定律

```
1. Contamination Law: 分数虚高 ∝ 污染程度
   → static benchmark → 污染风险 → MMLU分数可能5-10%虚高!
   → → → LiveBench/Chatbot Arena → 动态 → 防污染 → 更真实!

2. Saturation Law: benchmark区分度 ∝ 1/(模型能力 - benchmark难度)
   → → 简单benchmark → 大模型接近满分 → 区分度低 → 不再有用!
   → → → 需要更难benchmark → MMLU-Pro/ARC-AGI/SWE-bench → frontier!

3. Evaluation-Generalization Law: 评估分数 ∝ 泛化能力
   → → SFT→GRPO: 评估93% → 泛化0% gap → 真正理解!
   → → → 纯GRPO: 评估52% → 泛化37.5% gap → reward hacking → 假高分!
   → → → → → 高训练分数 + 低评估分数 = reward hacking = 评估generalization gap暴露!

4. Calibration Law: 模型准确性 ∝ 自信度 → 好模型calibrated!
   → → 理想: P(correct|confidence=0.9) ≈ 0.9 → 知道自己90%正确!
   → → → 实际: LLM通常overconfident → confidence=0.9但accuracy=0.7 → 需校准!
   → → → → Temperature scaling: 降低temperature → 使confidence更match accuracy!

5. Efficiency-Evaluation Law: 评估成本 ∝ benchmark复杂度 × 模型推理成本
   → → MMLU: 14K题 × 1 token → ~14K tokens → 7B INT4 → 3秒 → 极快!
   → → → Chatbot Arena: 1题 × 100 tokens → +人类时间 → 1分钟 → 极慢!
   → → → → 本地RTX 4090评估 = 最fast + 最cheap → 但需标准化!
```

## 关键论文与参考

```
- MMLU (Hendrycks et al., 2021): 57学科 → 知识广度 → static但广泛应用
- MMLU-Pro (2024): 10选择 → 更难 → 防shortcut
- HumanEval (Chen et al., 2021): Python代码 → pass@k → 饱和问题
- SWE-bench (2024): 真实GitHub issue → frontier code benchmark
- Chatbot Arena (LMSYS, 2023): Elo → 人类偏好 → 最trusted排名
- LiveBench (2025): 月度更新 → 防污染 → 动态评估
- ARC-AGI (Chollet, 2019): 抽象推理 → novel problem-solving → 最难!
- HELM (Stanford, 2022): 7维度 → 全面 → 但复杂
- GAIA (2023): Agent benchmark → reasoning+tool-use → frontier
- WebArena (2023): 网页操作 → real-world tasks → practical
- Scaling Laws (Kaplan/Chinchilla): 评估需要与scaling结合 → 分数∝N^0.34,D^0.28
```

Sources:
- [MMLU-Pro](https://arxiv.org/abs/2406.01564)
- [Chatbot Arena](https://chat.lmsys.org/)
- [HELM](https://crfm.stanford.edu/helm/)
- [SWE-bench](https://www.swebench.com/)
- [LiveBench](https://livebench.github.io/)
- [ARC-AGI](https://arcprize.org/)
- [GAIA](https://arxiv.org/abs/2311.12983)