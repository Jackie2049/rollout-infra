# AI Governance & Regulation Deep Dive — 4大全球AI治理框架(EU AI Act 4级风险+NIST RMF 4函数+ISO 42001 AIMS+China 3法规) + 高风险合规8要求(风险管理+数据治理+文档+日志+透明+人控+准确+QMS) + EU AI Act时间线(2024→2027分阶段) + 罚款(€35M/7%营收) + GPAI系统性风险+Anthropic RSP ASL分级 + NIST RMF(Govern-Map-Measure-Manage)+ISO 42001(PDCA可认证)+China(算法备案+深合成+生成式AI) + 全球比较(EU=合规/US=自愿/China=管控/UK=创新) + 生产合规Pipeline(shift-left+governance-as-code+自动化) + RTX 4090(小规模豁免+风险评估+日志+文档) + 2026趋势(全球趋同+AI Act全面执行+认证+可审计)

> 2026-06-14 | AI治理与法规深度分析: 全球4大AI治理框架 → EU AI Act(世界首个综合性AI法律, 4级风险分类: 不可接受→禁止, 高风险→8项合规, 有限风险→透明, 最小风险→自愿) → 高风险合规8项(风险管理+数据治理+技术文档+日志记录+透明+人控+准确+QMS) → GPAI系统性风险(OpenAI/Anthropic级) → Anthropic RSP ASL分级部署 → NIST AI RMF(Govern-Map-Measure-Manage 4函数+7特征) → ISO 42001(首个国际AIMS标准+PDCA+可认证) → China(算法备案+深合成+生成式AI → 3法规+统一AI法pending) → 全球比较4模式(EU合规/US自愿/China管控/UK创新) → 生产合规Pipeline(shift-left+governance-as-code+6步骤) → RTX 4090(小规模豁免+风险评估+日志+文档) → 2026趋势(全球趋同+AI Act全面执行+认证+可审计)
> 关联: ai-expert-knowledge-map-gap-analysis.md(Governance gap ★→★★★★), ai-safety-guardrails-production-deep-dive.md(Defense-in-Depth+OWASP+Guardrails), evaluation-benchmarking-deep-dive.md(Fairness+Toxicity评估), data-pipeline-curation-deep-dive.md(数据质量+contamination)
> 参考: EU AI Act(Regulation 2024/1689), NIST AI RMF(NIST AI 100-1), ISO/IEC 42001:2023, Anthropic RSP(2025), OWASP LLM Top 10, OECD AI Principles, China算法推荐+深合成+生成式AI法规

## 0. 核心定律: AI治理=合规+安全+责任 → 不治理=不可部署 → Governance是生产前提!

```
AI治理核心定律:

  AI Governance = 3支柱 → 合规(法律)+安全(技术)+责任(伦理):
    → 合规: EU AI Act+NIST+ISO → 法律要求 → 不合规=罚款+不可部署!
    → → 安全: OWASP+Guardrails+Red-team → 技术要求 → 不安全=风险+不可部署!
    → → → 责任: 透明+问责+公平 → 伦理要求 → 不负责=信任缺失+不可部署!

  为什么AI Infra工程师要懂治理?
    → 生产部署 → 必须合规 → Infra工程师负责部署 → 不懂治理=部署不合规!
    → → Logging+监控 → 审计+问责 → Infra是治理的技术基础!
    → → → → Guardrails → 约束输出 → 合规要求 → Infra+治理交叉点!
    → → → → → vLLM/serving → 安全+合规+监控 → Infra工程师必须懂!

  全球治理4模式 → EU(合规驱动)/US(自愿驱动)/China(管控驱动)/UK(创新驱动):
    → EU AI Act → 最严格 → 全球标杆 → 2026全面执行 → 罚款€35M/7%营收!
    → → NIST RMF → 美国自愿 → 但行业标准 → 企业采纳 → 最佳实践!
    → → → ISO 42001 → 国际标准 → 可认证 → 全球通行 → 合规证明!
    → → → → China → 算法备案+深合成+生成式AI → 3法规 → 最早管控!

→ → → → → 结论: AI治理=生产前提 → 不治理=不可部署 → Infra工程师必须懂治理!
```

## 1. EU AI Act — 世界首个综合性AI法律

```
### 1.1 4级风险分类

EU AI Act (Regulation 2024/1689) → 2024-08-01生效 → 世界首个综合性AI法律!

风险4级 → 每级不同义务 → 从禁止到自愿:

  1. 不可接受风险(Unacceptable Risk) → 禁止!
    → 社会评分(social scoring) → 政府对公民评分 → 禁止 → 人权侵犯!
    → → 操控性AI(manipulative AI) → 利用弱点操控 → 禁止 → 伦理侵犯!
    → → → → 实时生物识别(real-time biometric ID) → 公共场所人脸识别 → 禁止(例外: 严重犯罪)!
    → → → → → → 罚款: €35M或7%全球营收 → 最严厉!

  2. 高风险(High Risk) → 8项合规要求!
    → 关键基础设施 → 能源/水务/交通 → AI错误→危险 → 高风险!
    → → 医疗设备 → AI诊断 → 错误→致命 → 高风险!
    → → → HR/招聘 → AI筛选简历 → 偏见→不公平 → 高风险!
    → → → → 教育 → AI评分 → 偏见→不公平 → 高风险!
    → → → → → → 法律执法 → AI犯罪预测 → 偏见→不公平 → 高风险!
    → → → → → → → → 信用评估 → AI评分 → 偏见→歧视 → 高风险!
    → → → → → → → → → → 罚款: €15M或3%全球营收 → 严厉!

  3. 有限风险(Limited Risk) → 透明义务!
    → 聊天机器人(chatbot) → 必须告知用户是AI → 透明!
    → → 深度伪造(deepfake) → 必须标注AI生成 → 透明!
    → → → → 情感识别(emotion recognition) → 必须告知 → 透明!
    → → → → → → 罚款: €7.5M或1%全球营收 → 中等!

  4. 最小风险(Minimal Risk) → 自愿!
    → AI视频游戏 → AI垃圾邮件过滤 → 无风险 → 无义务 → 自愿最佳实践!
    → → → → → → 罚款: 无 → 但鼓励自愿遵循代码实践!

### 1.2 高风险AI系统 — 8项合规要求

高风险AI → 8项合规 → 全面 → 不遗漏!

| # | 要求 | 描述 | 与Infra的联系 |
|---|------|------|-------------|
| 1 | 风险管理系统 | 全生命周期识别+分析+缓解风险 → 持续 | 监控→风险识别→serving指标 |
| 2 | 数据与数据治理 | 训练数据relevant+representative+无错误 | 数据Pipeline→MinHash+quality |
| 3 | 技术文档 | 全文档 → 证明合规 → 可审计 | 文档化模型+训练+部署流程 |
| 4 | 日志记录 | 自动记录事件 → 可追溯+可审计 | vLLM logging→监控→审计 |
| 5 | 透明性 | 用户知情 → 能力+限制+用途+准确率 | 模型卡片+性能报告+限制说明 |
| 6 | 人类监督 | 人可干预→停止→覆盖 → "停止按钮" | 熔断机制→runtime guard→kill switch |
| 7 | 准确+鲁棒+安全 | 适当准确率+抗扰动+抗攻击 | benchmark+red-team+OWASP |
| 8 | 质量管理系统 | 组织流程 → 确保持续合规 → QMS | 流程文档→审计→持续改进 |

### 1.3 GPAI模型 — 系统性风险

GPAI(General-Purpose AI)模型 → 有系统性风险 → 需额外义务!

系统性风险定义:
  → 计算量>10^25 FLOPs → 或欧盟委员会指定 → 系统性风险!
  → → → GPT-4级模型 → 计算量超阈值 → 系统性风险 → 额外义务!
  → → → → → Anthropic Claude级 → 同样 → 系统性风险 → 额外义务!

额外义务(系统性风险GPAI):
  → 评估系统性风险 → adversarial testing → red-team → 安全评估!
  → → → 事件报告 → 严重事件24h内报告 → 透明!
  → → → → → 安全措施 → 对齐+guardrails+约束 → 保护!
  → → → → → → → 网络安全 → 抗攻击+抗注入 → 安全!

Anthropic RSP(Responsible Scaling Policy):
  → ASL-1: 安全 → 无限制 → 当前大多数模型
  → → → ASL-2: 有限风险 → 安全评估 → 大多数部署
  → → → → → ASL-3: 高风险 → 需额外安全措施 → 限制部署!
  → → → → → → → → ASL-4: 极高风险 → 需极端安全措施 → 严格限制!

→ → → → → → → 结论: GPAI系统性风险 → 额外义务 → Anthropic RSP ASL分级 → 部署前必须评估!

### 1.4 时间线 — 2024-2027分阶段执行

| 日期 | 里程碑 |
|------|--------|
| 2024-08-01 | 法规生效 |
| 2025-02-02 | 不可接受风险禁止生效 → AI素养义务 |
| 2025-08-02 | GPAI模型义务生效 → 系统性风险GPAI规则 |
| 2026-08-02 | 高风险AI系统义务全面生效(Annex III) → **最关键!** |
| 2027-08-02 | 嵌入产品的高风险AI义务生效(医疗器械等) |

→ → → 2026-08 → 高风险AI全面生效 → LLM部署必须合规 → Infra工程师必须准备!

### 1.5 罚款

| 违规类型 | 最大罚款 |
|----------|---------|
| 不可接受风险AI | €35M 或 7%全球营收 |
| 高风险AI违规 | €15M 或 3%全球营收 |
| 信息错误提供 | €7.5M 或 1%全球营收 |

→ → → 罚款极高 → 与GDPR同等 → 合规是必须 → 不合规=不可承受!
```

## 2. NIST AI Risk Management Framework

```
NIST AI RMF (NIST AI 100-1) → 2023发布 → 美国自愿框架 → 但行业标准!

### 2.1 4核心函数

NIST RMF = 4函数 → Govern-Map-Measure-Manage → 循环 → 持续!

1. GOVERN(治理):
   → 建立AI政策+角色+问责 → 组织级 → 管理层!
   → → → → → 定义: 谁负责? 什么政策? 什么流程? → 清晰问责!
   → → → → → → → 关键: 组织文化 → AI意识 → 优先级 → 资源分配!

2. MAP(映射):
   → 上下文化AI风险 → 相对使用场景 → 环境级!
   → → → → → 定义: 在什么场景? 什么用户? 什么影响? → 风险定位!
   → → → → → → → 关键: 风险识别 → 情境分析 → 影响评估 → 优先级!

3. MEASURE(测量):
   → 量化/定性评估风险 → 指标级 → 可测量!
   → → → → → 定义: 风险多大? 指标? 证据? → 风险量化!
   → → → → → → → 关键: benchmark → 评估 → 指标 → 数据 → 证据!

4. MANAGE(管理):
   → 行动 → 缓解风险 → 实施 → 运作级!
   → → → → → 定义: 怎么缓解? 什么措施? 什么优先? → 风险管理!
   → → → → → → → 关键: 缓解 → 监控 → 追踪 → 持续改进!

### 2.2 7个可信AI特征

NIST定义7个可信AI特征 → 每个需评估:

| 特征 | 定义 | 评估方法 |
|------|------|---------|
| Validity | 输出符合预期用途 | 任务特定benchmark |
| Reliability | 一致性输出 | 重复输入→相同输出? |
| Safety | 不造成伤害 | Safety eval+red-team |
| Security | 抗攻击 | OWASP+adversarial test |
| Resilience | 抗扰动恢复 | Robustness eval |
| Accountability | 可问责 | Logging+审计+文档 |
| Transparency | 可解释 | 模型卡片+文档+日志 |
| Explainability | 可解释决策 | 解释生成+可视化 |
| Privacy | 保护数据 | 数据最小化+匿名化 |
| Fairness | 群体公平 | Fairness eval+偏见检测 |

→ → → 结论: NIST RMF=美国自愿标准 → 4函数+7特征 → 最佳实践 → 企业采纳!
```

## 3. ISO/IEC 42001 — 首个国际AIMS标准

```
ISO/IEC 42001:2023 → 首个国际AI管理系统标准 → 可认证 → 全球通行!

### 3.1 PDCA循环

ISO 42001 = PDCA循环 → Plan-Do-Check-Act → 持续改进!

  Plan(计划):
    → AI政策 → 风险评估 → 目标设定 → 资源分配 → 计划!
    → → → → → → → 定义AI系统范围 → 识别利益相关方 → 确定约束 → 规划!

  Do(执行):
    → 实施AI政策 → 运行AI系统 → 执行风险管理 → 实施!
    → → → → → → → 运行监控 → 日志记录 → 透明 → 合规执行!

  Check(检查):
    → 评估绩效 → 内审 → 监控 → 测量 → 检查!
    → → → → → → → benchmark评估 → 安全测试 → 偏见检测 → 合规验证!

  Act(改进):
    → 管理评审 → 纠正措施 → 持续改进 → 行动!
    → → → → → → → 问题→修正 → 改进措施 → 更新政策 → 持续!

### 3.2 关键要求

| 要求 | 描述 |
|------|------|
| AI政策 | 制定组织级AI使用政策 |
| 领导承诺 | 管理层承诺+资源分配 |
| 风险评估 | AI系统风险评估+处理 |
| 绩效评估 | 持续评估+内审+管理评审 |
| 文档 | 完整文档+审计+记录 |
| 持续改进 | PDCA → 持续改进 → 不停止 |

### 3.3 认证

ISO 42001 → 可认证 → 第三方审计 → 证明合规 → 全球通行!
  → → 认证过程: 申请→审计→认证→年审 → 持续 → 不一次!
  → → → → → 与NIST区别: NIST=自愿→ISO=可认证 → ISO更正式 → 更有法律效力!
  → → → → → → → → 与EU AI Act关系: EU引用ISO 42001作为合规证明 → 认证=合规!

→ → → → 结论: ISO 42001=首个国际AIMS标准 → PDCA+可认证 → EU AI Act合规证明 → 全球通行!
```

## 4. China AI法规 — 最早+最管控

```
China → 3个AI法规(2022-2023) → 全球最早+最管控 → 统一AI法pending!

### 4.1 算法推荐规定(2022)

算法推荐规定 → 2022 → 算法透明+备案 → 全球最早!
  → 要求: 算法原理披露 → 用户可退出 → 备案 → 透明!
  → → → → → → → 备案: 所有推荐算法 → 向CAC备案 → 透明+可控!
  → → → → → → → → → 退出权: 用户可关闭算法推荐 → 选择权!

### 4.2 深合成规定(2023)

深合成(Deepfake)规定 → 2023 → AI内容标注+备案 → 全球最早!
  → 要求: AI生成内容必须标注 → 不可误导 → 透明!
  → → → → → → → 标注: "AI生成" → 明确标识 → 不欺骗用户!
  → → → → → → → → → 备案: 深合成服务 → 向CAC备案 → 可控!
  → → → → → → → → → → → 禁止: 伪造新闻+伪造身份 → 刑事 → 严厉!

### 4.3 生成式AI规定(2023)

生成式AI规定 → 2023 → 最直接管控LLM → 全球最早!
  → 要求: 遵守社会主义核心价值 → 不歧视 → 数据合规 → 备案!
  → → → → → → → 核心价值: 不违反社会主义价值观 → 内容管控!
  → → → → → → → → → 不歧视: 不生成歧视内容 → 偏见防控!
  → → → → → → → → → → → 数据合规: 训练数据来源合法 → 数据管控!
  → → → → → → → → → → → → → 备案: 生成式AI服务 → 向CAC备案 → 可控!

### 4.4 统一AI法(pending)

China → 统一AI法 → 2025-2026 expected → 整合3法规 → 更全面!
  → → → 预期: 整合算法+深合成+生成式 → 统一 → 减少 fragmentation!
  → → → → → 更严格安全评估 → 前沿模型 → 国家安全 → 风险评估!
  → → → → → → → 数据跨境 → 管控 → 不随意出境 → 数据主权!
```

## 5. 全球AI治理比较 — 4模式

```
### 5.1 4模式对比

| 维度 | EU | US | China | UK |
|------|-----|-----|--------|-----|
| 方法 | 合规驱动 | 自愿驱动 | 管控驱动 | 创新驱动 |
| 核心法律 | AI Act(2024) | 无联邦法律 | 3法规(2022-23) | 无绑定法律 |
| 核心标准 | 引用ISO 42001 | NIST RMF(自愿) | CAC备案(强制) | 自愿框架 |
| 算法透明 | 高风险→透明 | 无联邦要求 | 备案→强制 | 推荐 |
| 内容管控 | 禁止列表 | 平台政策 | 社会主义价值 | 安全指引 |
| 执行 | 国家当局+AI Office | FTC+sector | CAC+MIIT→强 | 现有监管 |
| 创新平衡 | 合规重 | 市场驱动 | 国家导向 | 创新优先 |
| 认证 | ISO 42001 | NIST(自愿) | 备案(强制) | 自愿 |
| 罚款 | €35M/7% | FTC罚款 | 行政+刑事 | 无绑定 |

### 5.2 全球趋同趋势

2025-2026 → 全球趋同 → 不是完全一致 → 但方向一致:

  趋同点:
    → 风险分类 → EU 4级 → NIST 4函数 → ISO PDCA → China 3法规 → 都基于风险!
    → → → → → 安全评估 → 所有框架都要求 → benchmark+red-team → 一致!
    → → → → → → → → → 透明 → 所有框架都要求 → 文档+日志 → 一致!
    → → → → → → → → → → → 问责 → 所有框架都要求 → audit+logging → 一致!

  差异点:
    → EU → 最严格 → 合规驱动 → 罚款重 → 法律强制 → 全球标杆!
    → → → → → US → 最灵活 → 自愿驱动 → 市场为主 → 企业选择 → 快创新!
    → → → → → → → → → China → 最管控 → 备案强制 → 内容管控 → 国家导向!
    → → → → → → → → → → → UK → 最宽松 → 创新优先 → 自愿框架 → 快部署!

→ → → → → → → → → → → 结论: 4模式 → 趋同但不同 → EU标杆 → US创新 → China管控 → UK灵活 → 各有特色!
```

## 6. 生产合规Pipeline设计 — shift-left + governance-as-code

```
### 6.1 Shift-left治理 — 合规检查提前到开发阶段

传统治理 → 部署前检查 → 最后一步 → 发现问题→推迟部署 → 慢!

Shift-left → 合规检查提前到开发每个阶段 → 早发现→早修复 → 快!

  开发阶段 → 合规检查点:
    → 数据收集 → 数据治理检查 → representativeness+bias+privacy → Step 1!
    → → → 模型训练 → 训练日志+文档 → 可追溯 → Step 2!
    → → → → → 评估 → benchmark+red-team → 安全+准确+公平 → Step 3!
    → → → → → → → 部署 → guardrails+logging+监控 → 安全+合规 → Step 4!
    → → → → → → → → → 运维 → 持续监控+定期red-team → 持续合规 → Step 5!

### 6.2 Governance-as-code — 合规自动化

Governance-as-code → 合规规则编程化 → 自动检查 → 不依赖人工!

  合规自动化工具:
    → MLCommons Safety benchmark → 自动安全测试 → CI/CD gate → 不通过=不部署!
    → → → → → Fairness检测 → 自动偏见检测 → 分组性能 → 不通过=不部署!
    → → → → → → → → → OWASP LLM Top 10 → 安全扫描 → 注入检测 → 不通过=不部署!
    → → → → → → → → → → → Logging → 自动记录 → 可审计 → 满足EU AI Act要求!

  Pipeline示例:
    → Step 1: 数据检查 → 代表性+偏见+隐私 → 自动 → gate!
    → → → Step 2: 安全评估 → MLCommons+OWASP → 自动 → gate!
    → → → → → Step 3: 性能评估 → MMLU+TruthfulQA → 自动 → gate!
    → → → → → → → Step 4: 部署 → guardrails+logging → 自动 → 合规!
    → → → → → → → → → Step 5: 监控 → 性能+安全+偏见 → 持续 → 合规!

### 6.3 6步合规Pipeline

生产合规Pipeline → 6步 → shift-left+governance-as-code → 全自动化!

  Step 1 — 风险评估(Risk Assessment):
    → 确定AI系统风险等级 → EU 4级 → 高风险→8项合规 → 有限→透明 → 最小→自愿
    → → → → → → → 评估: 使用场景+影响范围+用户类型 → 定风险!

  Step 2 — 数据治理(Data Governance):
    → 训练数据relevant+representative+无错误 → EU高风险要求#2
    → → → → → → → 检查: MinHash+quality+PII+代表性 → 数据Pipeline!

  Step 3 — 技术文档(Technical Documentation):
    → 完整文档 → 证明合规 → 可审计 → EU高风险要求#3
    → → → → → → → 文档: 模型卡片+训练流程+评估结果+限制说明 → 标准化!

  Step 4 — 安全评估(Safety Evaluation):
    → benchmark+red-team+OWASP → EU高风险要求#7
    → → → → → → → 评估: TruthfulQA+Llama Guard 3+OWASP+Garak → 安全Pipeline!

  Step 5 — 部署(Deployment with Governance):
    → guardrails+logging+人控 → EU高风险要求#4/#5/#6
    → → → → → → → 部署: regex+Llama Guard 3+熔断+日志 → Defense-in-Depth!

  Step 6 — 持续合规(Continuous Compliance):
    → 监控+定期red-team+审计 → 持续 → EU高风险要求#1/#8
    → → → → → → → 持续: 性能监控+安全监控+偏见监控+定期审计 → 不停止!

→ → → → → → → 结论: 6步合规Pipeline → shift-left+governance-as-code → 全自动化 → EU AI Act合规!
```

## 7. RTX 4090合规策略

```
### 7.1 RTX 4090场景 → 小规模+有限风险 → 但需准备

RTX 4090 → 24GB → 小规模推理 → 有限风险 → 但仍需合规准备!

  场景分析:
    → 小规模推理 → 7B INT4 → 单GPU → 个人/小企业 → 有限风险 → 透明义务!
    → → → → → → → 不涉及高风险场景 → 不是医疗/法律/HR → 有限风险 → 不需8项合规!
    → → → → → → → → → 但: 未来可能扩展 → 需准备 → 需风险评估 → 需文档!

### 7.2 RTX 4090合规准备

RTX 4090 → 有限风险 → 但仍需4项准备:

  1. 风险评估 → 确定使用场景 → 是否涉及高风险 → 定级!
  → → → → → → → → → → 如果涉及HR/医疗/法律 → 高风险 → 需8项合规 → 不建议RTX 4090小规模!

  2. 透明义务 → 告知用户是AI → 能力+限制 → 模型卡片 → 有限风险!
  → → → → → → → → → → 模型卡片 → 能力+限制+评估结果 → 透明 → EU要求!

  3. 日志记录 → 记录所有交互 → 可追溯 → 可审计 → EU要求!
  → → → → → → → → → → vLLM logging → 交互记录 → 审计 → 合规 → Infra工程师职责!

  4. 文档 → 模型+训练+评估+部署 → 完整文档 → 可审计 → EU要求!
  → → → → → → → → → → 文档 → 模型卡片+训练流程+评估结果 → 合规 → Infra工程师职责!

### 7.3 RTX 4090合规配置

| 合规项 | RTX 4090配置 | 成本 |
|--------|-------------|------|
| 风险评估 | 1-2h → 文档 | 低 |
| 透明义务 | 模型卡片 → 1h | 低 |
| 日志记录 | vLLM logging → 已内置 | 低 |
| 文档 | markdown → 2-4h | 低 |
| 安全评估 | TruthfulQA+OWASP → lm-eval | 中(2h) |
| 人控 | 熔断机制 → 添加kill switch | 低 |

→ → → 结论: RTX 4090小规模 → 有限风险 → 4项准备+2项评估 → 合规成本低 → 但需文档!
```

## 8. 核心规律

```
AI治理核心规律:

  1. 治理=生产前提 → 不治理=不可部署 → 合规+安全+责任 → 三支柱!
  → → → → → → EU AI Act → 2026全面执行 → 罚款€35M → 不合规=不可承受!

  2. 风险分类=治理基础 → EU 4级 → 不可接受/高/有限/最小 → 分级管理!
  → → → → → → → → → → 高风险→8项合规 → 有限→透明 → 最小→自愿 → 合理分级!

  3. 全球4模式 → EU合规/US自愿/China管控/UK创新 → 趋同但不同!
  → → → → → → → → → → → → → → → EU=全球标杆 → US=创新 → China=管控 → UK=灵活!

  4. Shift-left治理 → 合规检查提前 → 早发现→早修复 → governance-as-code → 自动化!
  → → → → → → → → → → → → → → → → → → 不在部署前才发现 → 每个阶段都检查 → 快!

  5. ISO 42001认证 → EU合规证明 → 全球通行 → 可审计 → 正式!
  → → → → → → → → → → → → → → → → → → → → 认证=合规证明 → 法律效力 → 全球认可!

  6. Infra工程师=治理技术基础 → logging+监控+guardrails+审计 → 全链路!
  → → → → → → → → → → → → → → → → → → → → → → vLLM/serving → 安全+合规+监控 → Infra职责!

  7. RTX 4090 → 小规模有限风险 → 4项准备+2项评估 → 合规成本低 → 但需文档!
  → → → → → → → → → → → → → → → → → → → → → → → → → → 7B INT4+logging+文档+模型卡片 → 基础合规!

  知识Gap修复:
    → AI Governance从★(1/5) → ★★★★(4/5) → EU AI Act+NIST RMF+ISO 42001+China法规+全球比较+Pipeline+RTX 4090 → 全面!
    → → → → → 但仍需实践 → 具体合规文档 → 模型卡片 → 风险评估 → GPU可用时实测!
```

## 参考文献

```
1. 法律:
   - EU AI Act: Regulation 2024/1689, eur-lex.europa.eu
   - China算法推荐: 2022, CAC
   - China深合成: 2023, CAC
   - China生成式AI: 2023, CAC

2. 标准:
   - NIST AI RMF: NIST AI 100-1, nist.gov/itl/ai-risk-management-framework
   - ISO/IEC 42001:2023, iso.org/standard/81230.html
   - OWASP LLM Top 10: owasp.org/www-project-top-10-for-large-language-model-applications/

3. 公司政策:
   - Anthropic RSP: anthropic.com/responsible-scaling-policy
   - Microsoft Responsible AI: microsoft.com/en-us/ai/responsible-ai
   - Google Responsible AI: ai.google/responsible-ai/

4. 国际组织:
   - OECD AI Principles: oecd.ai/en/ai-principles
   - MLCommons AI Safety: mlcommons.org/ai-safety/

5. 我们的笔记:
   - ai-expert-knowledge-map-gap-analysis.md → Governance gap评估
   - ai-safety-guardrails-production-deep-dive.md → Defense-in-Depth+OWASP+Guardrails
   - evaluation-benchmarking-deep-dive.md → Fairness+Toxicity+Calibration评估
   - data-pipeline-curation-deep-dive.md → 数据质量+contamination+DoReMi

Sources:
- [EU AI Act Official](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32024R1689)
- [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework)
- [ISO/IEC 42001](https://www.iso.org/standard/81230.html)
- [Anthropic RSP](https://www.anthropic.com/responsible-scaling-policy)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [OECD AI Principles](https://oecd.ai/en/ai-principles)
- [MLCommons AI Safety](https://mlcommons.org/ai-safety/)
