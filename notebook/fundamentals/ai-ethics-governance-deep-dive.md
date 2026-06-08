# AI Ethics & Governance — 负责任AI的制度设计

> 2026-06-08 | AI Ethics是Infra工程师最大知识盲区(★→★★!) → 不可忽视!
> 关键: 治理需要Infra支持 → Infra工程师是制度的技术执行者

## 1. AI Ethics核心原则

```
5大AI Ethics原则(全球共识):

1. 透明性(Transparency)
  → 模型决策可解释 → 用户可以理解为什么得到某输出
  → Infra实现: logging → 记录每个决策 → 可审计
  → 但: LLM内部不可解释(mechanistic interpretability还在研究!) → 部分透明

2. 公平性(Fairness)
  → 模型不歧视 → 所有群体平等对待
  → Infra实现: 去偏pipeline → 数据清洗 → 评估公平性
  → 问题: 定义"公平"本身有歧义 → 统计公平 vs 个体公平 → 不统一!

3. 安全性(Safety) → 我们已经深读!
  → 模型不产生有害输出 → 多层防护
  → Infra实现: guardrails + red-teaming + evaluation
  → → 详见ai-safety-alignment-deep-dive.md

4. 隐私性(Privacy)
  → 用户数据保护 → 不泄露个人信息
  → Infra实现: 数据加密 → 访问控制 → 差分隐私训练
  → DP-SGD(differential privacy SGD): 梯度裁剪+噪声 → ε-privacy保证
    → → 每步梯度裁剪到C → 加噪声σ=C×z → 保证隐私预算ε
    → → 但: DP训练loss增加 → 模型质量下降 → trade-off!

5. 问责性(Accountability)
  → 模型决策可以追溯 → 出错可以问责
  → Infra实现: logging + monitoring + audit trail
  → → Infra是问责的技术基础 → 没有logging=没有问责!
```

## 2. 偏见与公平性 (Bias & Fairness)

```
AI偏见来源:

1. 数据偏见
  → 训练数据不代表真实分布 → 模型学到偏见的映射
  → 示例: 面部识别 → 主要用白人数据 → 对黑人识别率低!
  → → 数据pipeline需要多样性检查 → 我们已经深读!

2. 模型偏见
  → 模型架构本身可能偏好 → 如语言模型偏好英语 → 中文输出差
  → → tokenizer偏见: 大vocab中文更公平(我们深读tokenization!)
  → → 解决: 多语言数据平衡 → vocab设计考虑公平性

3. 评估偏见
  → benchmark本身有偏见 → 某些群体被忽略 → 评估不公平
  → → 需要inclusive benchmark → 不同文化/语言/群体

公平性定义(4种主流):

1. Demographic Parity: 输出独立于群体 → P(Y|A=a)=P(Y|A=b)
   → 简单 → 但可能牺牲精度 → 某些场景不适用

2. Equalized Odds: 各群体TPR/FPR相等 → 正确/错误率一致
   → 更严格 → 但可能不可能实现(不同群体不同分布!)

3. Individual Fairness: 相似个体得相似输出 → d(Y_i,Y_j)≤L×d(X_i,X_j)
   → 最理想 → 但需要定义"相似"(metric问题!)

4. Calibration: 各群体预测概率=实际概率 → P(Y=1|Ŷ=p,A=a)=p
   → 预测可信 → 但可能隐藏偏见(各群体不同基率!)

→ 关键: 不存在"万能公平定义" → 需要场景选择 → trade-off!

与Infra连接:
  → 去偏pipeline → 数据多样性检查 → 模型评估 → Infra实现!
  → 评估公平性 → 需要按群体分析 → logging+analytics → Infra!
  → 差分隐私 → DP-SGD → 训练Infra实现(verl+DP optimizer?)
  → → Infra工程师不只是搭GPU → 还要搭公平pipeline!
```

## 3. AI治理框架

```
全球AI治理框架(2024-2026):

1. EU AI Act (2024年生效)
  → 4级风险分类 → 与ASL类似!
    → Unacceptable risk: 禁止(如社会评分系统)
    → High risk: 严格要求(如医疗AI → 必须认证)
    → Limited risk: 透明义务(如chatbot → 必须声明是AI)
    → Minimal risk: 无特殊要求(如垃圾邮件过滤)
  → 关键: 高风险AI → 必须通过合规评估 → 才能部署!
  → → Infra工程师: 需要构建合规评估pipeline → logging+审计

2. US Executive Order on AI (2023)
  → 安全评估 → 公平性 → 透明性 → 联邦机构AI使用规范
  → 关键: 前沿模型 → 必须报告安全评估结果
  → → Anthropic/Meta/OpenAI → 都报告了 → 合规!

3. 中国AI治理
  → 《生成式AI服务管理暂行办法》(2023)
  → 算法备案 → 内容安全 → 数据来源透明 → 反偏见
  → → 中国更注重内容安全 → 与EU注重风险分级不同!
  → → vLLM/SGLang中国部署 → 需要内容过滤 → guardrails!

4. ISO/IEC 42001 (2023)
  → AI管理系统国际标准 → 通用框架
  → → 组织可以认证 → 证明AI治理合规
  → → 类似ISO 9001(质量管理) → 但专门针对AI

治理对Infra的影响:
  → EU高风险AI → 必须logging+审计 → Infra必须支持!
  → 中国内容安全 → guardrails → Infra实现!
  → 安全评估 → benchmark+red-teaming → Infra执行!
  → → 治理=制度 → Infra=技术执行 → 不可分离!
```

## 4. AI经济影响

```
AI对经济的影响(2025-2030):

生产力提升:
  → McKinsey预测: AI→$13万亿经济增量(到2030)
  → → 1.2%额外GDP增长/年 → 巨大!
  → → 但: 分布不均 → 高技能受益 → 低技能受冲击!

就业影响:
  → 创造新岗位: AI工程师/数据科学家/AI安全专家 → 我们的目标!
  → 替代旧岗位: 文案/客服/翻译 → AI替代60-70%任务
  → → 但: 完全替代<5% → 大部分是任务增强而非替代!
  → → 关键: "AI增强人类"而非"AI替代人类" → 正确观念!

Infra工程师的经济价值:
  → AI infra = AI的基础设施 → 类似互联网infra → 不可替代!
  → → 全球AI infra市场: $100B+ (到2025) → 巨大!
  → → GPU需求: 2023 50万H100 → 2024 200万 → 2025 400万+!
  → → → Infra工程师需求巨大 → 供不应求!

成本优化:
  → RTX 4090推理: $0.01/Mtok → vs H100 $0.06 → 6x性价比!
  → → 消费级GPU = AI普惠化 → 降低门槛 → 更多人可用!
  → → 但: 训练仍需H100 → 推理可以RTX 4090 → 分层!
```

## 5. Human-AI Interaction

```
人-AI交互设计原则:

1. 信任管理(Trust Management)
  → 用户不应该过度信任AI → AI有错误 → 需要怀疑
  → → "AI增强但不替代" → 用户保持最终决策权
  → → Infra: confidence score → 输出confidence → 用户知道可信度

2. 错误管理(Error Management)
  → AI会犯错 → 需要优雅的错误处理 → 不崩溃
  → → Infra: fallback机制 → AI不确定→人类介入
  → → 评估: calibration → confidence=accuracy → 可信!

3. 交互设计(Interaction Design)
  → 对话式 → 用户提问 → AI回答 → 自然
  → → 但: 需要约束 → 不能回答所有问题 → guardrails
  → → Infra: 对话管理 → context window → 有限交互

4. 可解释性(Explainability)
  → 用户问"为什么?" → AI应该能解释 → 透明
  → → 但: LLM不可解释 → 只能说"基于训练数据"
  → → 未来: mechanistic interpretability → 我们已经研究!
  → → → Activation Patching → 发现知识在特定位置 → 可解释!

与Infra连接:
  → 对话Infra → vLLM/SGLang serving → 人-AI交互的技术基础
  → 信任 → confidence score → 需要评估pipeline → Infra
  → 约束 → guardrails → xgrammar/CFSM → Infra实现!
  → 可解释性 → interpretability工具 → 未来Infra扩展
```

## 6. 核心规律

```
AI Ethics & Governance核心:

1. 5原则不可缺: 透明+公平+安全+隐私+问责 → 全面治理
   → 安全我们已经深读 → 其他需要继续深化 → 特别是公平和隐私

2. 治理需要Infra支持 → Infra是制度的技术执行者
   → EU AI Act → logging+审计 → Infra实现
   → 中国内容安全 → guardrails → Infra实现
   → 安全评估 → benchmark → Infra执行
   → → 没有Infra=没有治理 → Infra工程师是治理的关键角色!

3. 公平性没有万能定义 → 需要场景选择
   → Demographic Parity → 简单但可能牺牲精度
   → Equalized Odds → 严格但可能不可能
   → Individual Fairness → 最理想但需要metric
   → → 选择取决于场景 → 没有银弹!

4. 差分隐私 → DP-SGD → 训练隐私保证
   → ε-privacy → 但loss增加 → trade-off
   → → Infra: DP optimizer → 未来方向!

5. AI经济影响巨大 → 但分布不均 → 需要公平分配
   → 高技能受益 → 低技能受冲击 → 需要政策调节
   → → AI普惠化 → RTX 4090推理 → 降低门槛!

6. Human-AI交互 → 信任+错误+解释 → 需要Infra支持
   → confidence score → fallback → interpretability → Infra工具!

RTX 4090 Ethics实践:
  → 部署7B + guardrails → 安全+合规 → EU/中国法规
  → 多语言支持 → 中文公平 → tokenizer设计考虑
  → Logging+审计 → 每次推理记录 → 可追溯 → 可问责
  → Privacy → 差分隐私训练 → 未来DP optimizer
  → → AI Ethics不是"软知识" → 是"硬实践" → Infra必须支持!