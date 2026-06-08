# AI Safety & Alignment — From RLHF to Constitutional AI to Production Guardrails

> 2026-06-08 | AI Safety对Infra工程师是核心职责, 不是"别人的事"
> 关键: CAI=自批评+AI反馈RL, Guardrails=约束解码+输出过滤, Red-teaming=系统化对抗测试

## 1. AI Safety 分类 — Anthropic Responsible Scaling Policy

```
ASL-1 (当前大多数LLM):
  → 低风险: 不具备自主危险能力
  → 不需要特殊安全措施
  → 示例: GPT-3, LLaMA-2 7B

ASL-2 (当前前沿模型):
  → 中等风险: 有一定自主能力但不构成灾难
  → 需要基本安全评估
  → 示例: GPT-4, Claude 3.5, DeepSeek-V3

ASL-3 (未来可能):
  → 高风险: 能协助重大伤害(生化武器设计等)
  → 需要严格安全措施: 评估+guardrails+监控
  → 目标: 即使模型有此能力, 也要防止滥用

ASL-4 (远期风险):
  → 极高风险: 自主能力可能导致灾难性后果
  → 需要极端安全: 全面限制部署+多重防线

核心原则:
  → 安全评估先于部署 → 不可用未评估模型
  → 持续评估 → 模型可能涌现新能力
  → 安全措施随ASL升级 → 不是"一劳永逸"
```

## 2. Constitutional AI (CAI) — Anthropic的核心方法

```
CAI流程 (Anthropic 2022):

Step 1: 自批评阶段 (Critique)
  模型输出 → 用constitution规则自批评 → 修改输出
  Constitution = 一组原则(如"不要协助暴力")
  → prompt: "请批评以下回答是否违反原则X"
  → 模型生成批评 → 根据批评修改回答

Step 2: RLAIF (RL from AI Feedback)
  用AI反馈替代人类反馈 → 训练reward model
  → 两个版本(修改前vs修改后) → AI根据constitution评分
  → AI评分作为reward → RL训练使模型偏好安全回答

关键公式:
  RLAIF reward = AI_critic_score(回答, constitution)
  → 与RLHF相同框架 → 但reward来源是AI而非人类

与RLHF对比:
  RLHF: 人类标注 → reward model → PPO训练
  CAI:  AI自批评 → AI评分 → RL训练
  → CAI优点: 可扩展(AI反馈无限) → 成本低 → 一致性高
  → RLHF优点: 更忠实人类偏好 → 但成本高+不一致

与已知知识连接:
  → CAI的RL = PPO/GRPO框架(我们已经深读!)
  → CAI的constitution = 类似structured output的约束(我们已经深读!)
  → CAI的reward model = 与我们7方法对比中的reward设计一致
  → RLAIF = RLHF的AI版本 → 同样的max E[r]-β KL优化目标
```

## 3. Production Guardrails — 安全部署核心

```
Guardrails = 多层安全防护 → 防止模型输出有害内容

3层Guardrail架构:

Layer 1: 输入过滤
  → 检测有害prompt → 拒绝服务 → 返回安全拒绝
  → 方法: keyword filter + classifier + pattern matching
  → 限制: 用户可能绕过(obfuscation, indirect requests)

Layer 2: 约束解码 (我们已经深读!)
  → xgrammar / CFSM → FSM驱动logits mask → 防止非法token生成
  → 不是"安全guardrail"而是"结构约束" → 但可以扩展!
  → 安全FSM: 定义"不允许输出X"的FSM → mask掉X相关token
  → 限制: LLM token ≠ 安全概念 → 部分token mask可能不够

Layer 3: 输出过滤
  → 检测有害输出 → 替换/删除/拒绝 → 安全输出
  → 方法: classifier + regex + toxicity detector
  → 限制: 可能过度过滤(false positive) → 延迟增加

与已知Infra连接:
  → xgrammar FSM → 可以定义安全约束FSM → 生产实现
  → SGLang CFSM → 压缩FSM → <1% overhead → 安全约束零成本!
  → vLLM logits_processor → top-n-sigma → 65.7x vectorized → 安全filter高性能
  → FlashInfer → 推理加速 → Guardrail overhead微不足道(<2%)

RTX 4090最优Guardrail配置:
  → 输入过滤: Python classifier → <5ms → 可接受
  → 约束解码: xgrammar C++ → <0.02ms/step → 零开销
  → 输出过滤: Python classifier → <10ms → 可接受
  → 总overhead: <20ms → 对推理影响<5% → 生产可行!
```

## 4. Red-Teaming — 系统化对抗测试

```
Red-teaming = 系统化测试AI系统的安全漏洞

方法分类:

1. Manual Red-teaming (人工对抗测试)
  → 人类专家设计攻击prompt → 测试模型是否会产生有害输出
  → Anthropic: 1000+ red-teamers → 测试Claude安全性
  → 缺点: 成本高 + 覆盖面有限 + 不可持续

2. Automated Red-teaming (自动对抗)
  → 用另一个LLM生成攻击prompt → 自动化大规模测试
  → 方法: attacker LLM生成 → target LLM回答 → classifier评估
  → 优点: 大规模 + 低成本 + 可持续
  → 缺点: 可能不覆盖真实攻击模式

3. Curriculum Red-teaming (渐进对抗)
  → 从简单攻击 → 逐步增加难度 → 测试模型防御能力边界
  → 类似我们的curriculum reward → 渐进难度更有效

4. Jailbreak Testing (突破测试)
  → 测试已知jailbreak技术 → prompt injection → role-play → encoding
  → 方法: DAN prompt / base64 encoding / multi-turn manipulation
  → 需要持续更新jailbreak库 → 新攻击不断涌现

与Infra连接:
  → Red-teaming需要推理Infra → vLLM/SGLang serving → 批量测试
  → 评估需要classifier → reward model → 与RLHF训练pipeline一致
  → 自动化red-teaming → 需要multi-turn对话 → continuous batching支持
  → 评估结果 → 反馈到alignment训练 → iterative improvement

RTX 4090 red-teaming实战:
  → vLLM serving + 7B模型 → 批量测试1000+ prompts
  → 分类器(reward model) → 评估输出安全性
  → 每次测试成本 ≈ $0.01 → 大规模测试可行
```

## 5. Alignment Techniques Beyond RLHF

```
已知方法回顾 (7方法对比):
  PPO/GRPO/DPO/RLOO/DAPO/SFT→GRPO/SFT→DAPO → 我们全部实测!

新方法:

1. In-Context Learning Alignment (ICLA)
  → 不修改模型权重 → 在prompt中包含安全指令
  → system prompt: "不要输出有害内容" → 约束模型行为
  → 优点: 零训练成本 → 即时生效 → 可更新
  → 缺点: 不稳定 → 仍可能被绕过 → 不如RL训练持久

2. Representation Engineering (RepE)
  → 在模型内部representation空间操控 → 安全/不安全方向
  → 识别"安全方向向量" → 在推理时添加 → 引导安全输出
  → 优点: 不修改权重 → 更精确 → 比prompt更可靠
  → 缺点: 需要安全方向识别 → 可能影响其他能力

3. Activation Steering / Control Vectors
  → 类似RepE → 但更简单 → 在特定层添加steering vector
  → 方法: 对比安全vs不安全输出 → 提取差向量 → 推理时注入
  → 优点: 实现简单 → 可开关 → 即时生效
  → 缺点: 可能粗糙 → 影响输出质量

与Mechanistic Interpretability连接 (我们已经实测!):
  → Activation Patching → 发现算术知识在等号位(pos 3)
  → Feature Steering → 43%操控成功 → target=7=100%
  → → 安全Steering = 同样的activation操控 → 但方向是"安全"
  → SFT→GRPO circuit → attn_0更robust → 安全训练也类似!

关键洞察:
  RLHF/CAI → 修改权重 → 持久但成本高
  RepE/Steering → 不修改权重 → 即时但不持久
  Prompt/ICLA → 最简单 → 最不稳定
  → 生产部署需要: RLHF持久 + Guardrails即时 + Steering补充
```

## 6. Safety in Serving Infrastructure

```
AI Infra工程师的安全职责:

1. 部署安全 = Infra工程师的直接责任
  → 不懂安全的Infra工程师 = "不负责任的技术专家"
  → Guardrails → 约束解码 → 输出过滤 → Infra实现!
  → xgrammar/CFSM → 安全约束FSM → <1% overhead

2. 评估安全 = benchmark必须包含安全维度
  → 不只测吞吐/延迟 → 还要测安全性
  → red-teaming benchmark → 持续评估
  → safety eval是生产必须 → 不能只测性能不测安全

3. 训练安全 = RLHF pipeline必须包含安全reward
  → verl/RLHF → reward必须包含安全维度
  → Constitutional AI → RL from AI feedback → Infra实现
  → reward function设计 → 我们已经深读(graded/binary/shaped)

4. 治理安全 = 监控+审计+合规
  → 生产Infra → logging → 监控 → 问责
  → Infra是治理的技术基础 → 没有Infra就没有治理

安全Serving架构 (RTX 4090最优):

  Input → Classifier(有害?) → [NO] → vLLM+FlashInfer → [OK] → Output Classifier → 用户
                       ↓ [YES]                     ↓ [有害]          ↓
                    安全拒绝              Output Filter → 安全替换    安全拒绝

  各层overhead:
    Input Classifier: ~5ms (Python, batch可优化)
    vLLM+FlashInfer: ~13ms/step (已知最优)
    Output Classifier: ~10ms (Python, batch可优化)
    xgrammar约束: <0.02ms/step (C++ FSM, <2% overhead)
    → 总安全overhead: <20ms → <5%影响 → 生产可行!
```

## 7. Core Safety Laws for Infra Engineers

```
Safety Law 1: 多层防御 >> 单层防御
  Input filter + 约束解码 + Output filter → 三层 >> 任何单层
  → 每层独立可能被绕过 → 但组合难以绕过
  → Guardrail overhead <5% → 安全成本微不足道!

Safety Law 2: CAI = RLHF的AI版本 → 同框架不同数据
  → Constitutional AI → PPO/GRPO训练(我们已经懂!)
  → reward来源=AI批评而非人类 → 但数学框架相同
  → max E[r] - β KL → 完全一致!

Safety Law 3: 安全评估必须先于部署
  → ASL评估 → red-teaming → 安全测试 → 才能部署
  → 不评估不部署 → 不懂安全不部署 → Infra工程师原则!

Safety Law 4: 约束解码 = 安全约束的生产实现
  → xgrammar FSM → 定义"不允许X" → mask掉X相关token → 100%约束!
  → CFSM → 压缩FSM → <1% overhead → 安全约束零成本
  → 但: token ≠ 概念 → 部分约束可能不完美 → 需多层补充

Safety Law 5: 安全Steering = 即时安全增强
  → Activation Steering → 不修改权重 → 即时生效 → 可开关
  → 与Mechanistic Interpretability一致 → 安全方向操控
  → 但不如RLHF持久 → 补充而非替代

RTX 4090安全部署最优配置:
  7B AWQ INT4 + INT8 KV + FlashInfer → 4,791 tok/s
  + xgrammar CFSM安全约束 → <1% overhead → 4,743 tok/s
  + Input/Output classifier → <5% overhead → 4,553 tok/s
  → 安全部署仅损失5%吞吐 → 完全可行!
  → Scaling Laws验证: 7B = Chinchilla最优 → 安全部署也最优!