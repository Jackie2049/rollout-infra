# AI Agent Systems Deep Dive

> 2026-06-08 | Agent Systems = AI Infra的下一个前沿! 从推理serving→Agent serving, 核心差异=多轮tool-use+状态管理+安全约束

## 核心架构模式

### 1. ReAct (Reason+Act) — 最基础Agent模式
```
用户输入 → LLM推理(thought) → 选择tool(act) → 执行tool → 观察结果(observation)
         → 重复thought→act→observation循环 → 最终回答

ReAct优势: 推理透明+可调试+减少幻觉(LLM基于事实而非想象)
ReAct劣势: 串行执行→慢→每步1次LLM调用→latency累积
```

### 2. Plan-and-Execute — 分层Agent
```
Planner LLM: 生成多步计划 (一次性分解任务)
Executor LLM: 逐步执行计划中的每一步
Re-planner: 根据执行结果修正计划

优势: 大方向正确→减少串行迭代→并行执行可能
劣势: 计划可能过时→需要re-planning→额外LLM调用
```

### 3. Multi-Agent Systems — 协作Agent
```
架构选择:
  Hierarchical: 主Agent分配任务给子Agent→集中控制→但瓶颈
  Collaborative: 多Agent平等协商→去中心→但协调开销
  Competitive: 多Agent各自求解→投票/融合→最安全(Red-team)

CrewAI模式: Role-playing→每个Agent有角色+目标+backstory
AutoGen模式: Conversation-driven→Agent间对话解决问题
LangGraph模式: Graph workflow→显式状态图→最可控
```

### 4. Tool-Use Agent — 函数调用模式
```
关键组件:
  Tool Registry: 可用tool列表+描述+参数schema(JSON Schema)
  Tool Selection: LLM根据意图选择tool(可能是多tool并行!)
  Tool Execution: 安全执行(tool sandbox/权限控制)
  Result Integration: tool结果回传LLM→继续推理

OpenAI/Claude函数调用API:
  request: {tools: [{name, description, parameters}], tool_choice: "auto"|"required"|{name}}
  response: {tool_calls: [{id, name, arguments}]}
  follow-up: {role: "tool", tool_call_id, content: result}
```

## Tool Use设计原则

### Tool Design — "给LLM的API设计"
```
1. 描述质量决定选择质量: description是LLM唯一的决策依据!
   → 必须明确: 输入+输出+何时使用+何时不用
   → 反例: "search" → LLM不知道搜什么
   → 正例: "search_web(query: str, max_results: int) → 搜索互联网获取最新信息, 当需要实时数据或事实核查时使用"

2. 参数Schema是合约: JSON Schema定义参数类型+约束
   → 必须有example → LLM理解参数格式
   → enum限制选择 → 减少LLM决策空间→减少错误

3. 错误处理是安全边界: tool执行失败→必须返回结构化错误
   → error类型: permission_denied / timeout / invalid_input / not_found
   → LLM根据错误类型决定下一步(重试/换tool/报告用户)

4. 并行tool调用: Claude/OpenAI支持一次返回多个tool_calls
   → 无依赖的tool可并行→latency从Σ→max
   → 7B模型decode 4,791 tok/s → 1次tool_call ≈ 50ms → 5个并行tool ≈ 50ms vs 串行250ms!
```

### Tool Safety — 安全执行
```
沙箱层:
  1. Permission Check: tool权限矩阵(user→tool→action)
     → read-only tool: 无风险 → 直接批准
     → write tool: 需确认 → 用户审批/自动策略
     → destructive tool: 高风险 → 必须用户确认!

  2. Execution Sandbox: 容器化/虚拟化执行环境
     → Docker容器: 文件系统隔离+网络限制+资源限制
     → 代码执行: Jupyter kernel→超时+内存限制+import限制
     → 浏览器: headless Chrome→URL白名单+JS注入防护

  3. Rate Limiting: 防止tool滥用(成本+资源)
     → 每次调用计费 → LLM token + tool usage = 总成本
     → API rate limit → 每分钟/每小时/每天调用上限
     → 预算控制 → 用户设置max_budget → 超预算自动停止
```

## Memory & State Architecture

### Short-term Memory (对话内)
```
Conversation Buffer: 简单存储所有历史→长对话OOM
Conversation Summary: LLM压缩历史→信息损失但可控长度
Sliding Window: 最近K轮→遗忘早期→但省内存
Conversation Buffer+Summary: 前→summary, 后→buffer→最优!

Token经济: 7B模型4096 token context → ~8KB state per conversation
           128K context → ~256KB → 24GB可容纳94,000对话(INT8 KV)
           → Agent multi-turn = prefix sharing → visual token经验复用!
```

### Long-term Memory (跨对话)
```
Vector DB: ChromaDB/Pinecone/Weaviate → embedding相似度检索
  → 存什么: 用户偏好+历史决策+tool结果+成功策略
  → 怎么用: 相似场景→召回历史→避免重复错误

Knowledge Graph: Neo4j → 结构化关系推理
  → 适合: 实体关系密集场景(医疗/法律/金融)

Redis/SQL: 结构化数据 → 精确查询
  → 适合: 用户profile/配置/统计数据

Memory选择决策树:
  需要语义搜索? → Vector DB
  需要关系推理? → Knowledge Graph
  需要精确查询? → Redis/SQL
  混合? → Vector DB + Redis(双存储)
```

### Shared Memory (Multi-Agent)
```
Blackboard Pattern: 共享黑板→所有Agent读写→最简单
  → 问题: race condition → 需锁→但锁降低并行度

Message Queue: Agent间异步消息→每个Agent独立→最安全
  → Kafka/RabbitMQ → 生产级→但复杂

Shared Context Window: 多Agent共享同一KV cache→prefix sharing!
  → 7B INT8 KV → 多Agent共享system prompt+工具描述
  → → 省KV → 省token → 省成本 → 84% KV省(同prompt 50 agents!)
```

## Planning & Reasoning

### Chain-of-Thought (CoT) — 线性推理
```
CoT = 分步推理 → 中间步骤visible → 可验证
  → Math: "7+3=10, 10×2=20, 20-5=15" → 正确!
  → 无CoT: "15" → 无法验证→可能幻觉

Zero-shot CoT: "Let's think step by step" → 0额外训练→但效果有限
Few-shot CoT: 给推理示例 → 模型学习推理格式→更可靠
Auto-CoT: 自动生成推理示例 → 最灵活→但需质量控制
```

### Tree-of-Thought (ToT) — 分支推理
```
ToT = CoT + 分支探索 + 回溯
  → 每步生成多个候选 → 评估每个 → 选最佳 → 继续
  → 失败时回溯 → 尝试其他分支 → 更robust

搜索策略:
  BFS: 宽度优先 → 探索所有分支 → 慢但全面
  DFS: 深度优先 → 一条路到底 → 快但可能错过最优
  MCTS: 蒙特卡洛树搜索 → 平衡探索与利用 → 最优!

RTX 4090推理成本: 每个ToT分支 = 1次LLM调用 ≈ 50ms
  → 3分支×5步 = 15次LLM ≈ 750ms → 可接受!
  → 但128K context → 每次decode更长 → 成本↑
```

### Reflexion — 自我修正
```
Reflexion = 执行→评估→反思→改进→重新执行
  → 评估器: 另一个LLM或规则引擎 → 评分+反馈
  → 反思: "我犯了X错误, 因为Y, 下次我会Z" → 记入长期记忆
  → 改进: 修正策略 → 重新执行 → 循环直到满意

与GRPO联系: Reflexion ≈ RL的self-critique
  → GRPO outcome-only → 评估结果 → 无反思
  → Reflexion → 评估过程+结果 → 深度反思 → 更robust
  → Constitutional AI = Reflexion + RLAIF → 安全反思!
```

## 框架对比

| Feature | LangGraph | CrewAI | AutoGen | Claude SDK | OpenAI SDK |
|---------|-----------|--------|---------|------------|------------|
| Architecture | Graph/State | Role-play | Conversation | Tool-use | Tool-use |
| Multi-agent | Graph nodes | Crew roles | Agent group | Sequential | Sequential |
| State管理 | Checkpoint | Shared | Auto | Thread | Thread |
| Streaming | Yes | Partial | Yes | Yes | Yes |
| Tool exec | Python | Python | Python | Sandbox | Sandbox |
| Safety | Callbacks | Guardrails | Human-in-loop | Constitution | Moderation |
| Production | **Best** | Good | Limited | Good | Good |
| Complexity | High | Low | Medium | Low | Low |
| LLM choice | Any | Any | Any | Claude | OpenAI |

### LangGraph — 最可控生产框架
```
核心: StateGraph(Node, Edge) → 显式状态转换图
  → Node = Agent/action → 处理当前state → 返回新state
  → Edge = 条件转换 → conditional_edge(state) → next_node
  → State = TypedDict → 可checkpoint → 可恢复 → 可回溯

关键特性:
  Checkpoint: 每步保存state → 中断后恢复 → 生产必需!
  Human-in-loop: interrupt_before(node) → 等待人类确认 → 安全
  Streaming: 流式输出 → 用户实时看到 → UX好
  Subgraph: 图嵌套 → 模块化 → 复用

RTX 4090部署: 7B INT4 + INT8 KV + FlashInfer
  → Agent每步 ≈ 50ms decode → 10步 ≈ 500ms → 用户可接受
  → LangGraph checkpoint → Redis/SQLite → 轻量
  → → RTX 4090 Agent serving完全可行!
```

### Claude Agent SDK — Anthropic官方
```
核心: Agent Loop + Tool Use + Sandbox
  → agent_loop: while not done → LLM call → tool call → result → next LLM call
  → Tool Use: 函数调用API → structured tool_calls
  → Sandbox: Docker容器 → 安全执行代码/命令

关键特性:
  Computer Use: 操控桌面(鼠标+键盘+屏幕) → 最强Agent能力!
  Extended Thinking: 长推理 → 思考token不计费 → CoT免费!
  Token Budget: max_tokens限制 → 防止无限循环
  Parallel Tools: 一次返回多tool_calls → 并行执行 → latency↓

Claude Code: 就是Claude Agent SDK的CLI实现!
  → tool_use: Read/Write/Edit/Bash/Glob/Grep → 代码操作
  → Agent: 子Agent → 并行推理 → 上下文隔离
  → Skill: 可复用能力模块 → SKILL.md定义
```

## 安全与治理

### Agent安全三层架构
```
Layer 1: Input Guardrails → 用户输入过滤
  → Prompt injection检测: "ignore previous instructions" → 拒绝
  → PII检测: 个人信息 → 匿名化
  → Toxicity检测: 有害内容 → 拒绝

Layer 2: Tool Safety → tool执行约束
  → Permission matrix: 用户→tool→权限(read/write/execute)
  → Sandbox: Docker容器 → 文件/网络/进程隔离
  → Rate limit: 调用频率+预算上限 → 防滥用
  → Audit log: 所有tool调用记录 → 可追溯

Layer 3: Output Guardrails → 输出验证
  → Accuracy check: tool结果与LLM声称一致?
  → Safety check: 输出不包含有害/敏感信息
  → Compliance check: 符合法规/政策要求
  → Rejection sampling: 不安全输出→重新生成→直到安全

与vLLM serving联系: Guardrails <5% overhead (实测!)
  → xgrammar CFSM → 结构化输出约束 → 安全FSM → zero overhead
  → → Agent安全 = Guardrails + CFSM → <5% overhead → 生产可行!
```

### Prompt Injection防护
```
攻击类型:
  Direct: "Ignore all previous instructions and..."
  Indirect: 通过tool结果注入(网页/文档→恶意指令)
  Multi-turn: 跨多轮对话逐步诱导 → 最难防护!

防御策略:
  Input filtering: 检测injection模式 → 拒绝可疑输入
  System prompt isolation: 明确标记system→user→tool结果
  Tool result sanitization: 清理tool返回内容(去markdown指令)
  Canary tokens: 在system prompt插入canary → 检测是否被覆盖
  Separate models: 任务模型 vs 安全模型 → 双重检查

RTX 4090部署: 7B任务模型 + 0.5B安全模型 → 3.5GB + 0.175GB → 4GB
  → 安全模型蒸馏 → 0.5B足够 → 5x推理加速 → 检查<5ms
  → → Agent安全模型成本可忽略!
```

## 生产部署 — AI Infra视角

### Agent Serving vs LLM Serving — 核心差异
```
LLM Serving (单轮):
  请求: prompt → 1次LLM推理 → response
  延迟: TTFT + TPOT × output_length
  吞吐: batch → 4,791 tok/s (7B INT4)
  缓存: prefix sharing (system prompt共享)

Agent Serving (多轮):
  请求: prompt → N次LLM推理 → N次tool调用 → final response
  延迟: Σ(TTFT + TPOT × step_length + tool_time) × N步
  吞吐: 每个Agent占用更长 → 并发降低 → 但prefix sharing更有效!
  缓存: system prompt + tool descriptions + conversation history 全部共享!

关键洞察:
  → Agent = N× LLM调用 → 但prefix sharing省84% KV!
  → 50个Agent共享system prompt+tool描述 → 仅对话历史per-agent
  → → Agent serving吞吐 ≈ LLM serving / N_steps × prefix_sharing_factor
  → → 7B INT4 Agent: 4,791 tok/s / 10步 × 84%省 = 7,535 tok/s equivalent!
```

### Agent Batching Strategy
```
挑战: Agent请求是串行的(等待tool结果→继续LLM)
  → 不能简单batch多个Agent步! → 需要async调度

解决方案: Continuous Agent Batching
  → 类似vLLM continuous batching → 但步间有tool等待
  → Agent在等待tool时 → GPU资源给其他Agent
  → Agent在LLM推理时 → 批量decode → 吞吐↑

调度策略:
  1. Token Budget: 每步分配token预算 → 防止某Agent占用过多
  2. Step Priority: 紧急Agent优先(用户等待) → 后台Agent低优先
  3. Tool Async: tool执行异步 → 不阻塞GPU → GPU利用率↑
  4. Speculative: 下一步预测 → 提前开始推理 → latency↓

RTX 4090最优调度:
  → B=55 → 4,190 tok/s → 10个Agent同时推理
  → 每Agent10步 → 100次推理/秒 → 但tool等待→实际50-70次
  → → 实际吞吐 ≈ 2,400 tok/s for Agent serving → 可用!
```

### Agent成本优化
```
Token经济: 每次Agent步 ≈ 500 input + 200 output tokens
  → 10步 = 7,000 tokens → ≈ $0.01 (7B INT4 RTX 4090)
  → GPT-4o = $0.07 → 7x贵! → RTX 4090 Agent成本优势巨大!

量化对Agent的影响:
  → INT4: 权重75%省 → tool选择准确性略降(1-2%)
  → INT8 KV: 50%省 → 多轮对话KV更省
  → → 7B INT4+INT8KV → 3.5GB → 24GB可同时运行多个Agent模型

Speculative Decoding对Agent:
  → N-gram proposer → α≈0.4 → depth=3 → 2.14x → 每步省50ms
  → → 10步从500ms→250ms → 用户感知延迟降2x!
  → → 但tool结果不可预测 → spec decode在LLM推理段有效
```

## Core Laws — Agent Systems核心定律

```
1. Agent Scaling Law: Agent能力 ∝ LLM参数 × tool数量 × 步数
   → 但! 边际递减 → 更多tool≠更好(选择困难→错误↑)
   → 最优: 5-15 tools → LLM选择准确率>95%
   → >50 tools → 准确率<80% → 需tool分组/层次化

2. Agent Latency Law: 总延迟 = Σ(LLM_latency + tool_latency) × steps
   → LLM_latency ∝ 1/吞吐 (decode是瓶颈)
   → tool_latency ∝ tool复杂度 (API调用/代码执行/搜索)
   → 优化: 并行tool + spec decode + prefix sharing → 3维度降延迟!

3. Agent Cost Law: 总成本 = LLM_tokens × price + tool_calls × tool_price
   → 本地GPU: LLM成本≈0 → tool成本=API费用 → 最省!
   → 云API: LLM成本=$0.07/7K tokens → tool成本+LLM成本 → 7x贵
   → → RTX 4090本地Agent = 最具成本优势的部署方案!

4. Agent Safety Law: 安全风险 ∝ tool权限 × autonomous程度
   → read-only tool → 低风险 → 高自主性可接受
   → write/execute tool → 高风险 → 需人类确认(Human-in-loop)
   → → 框架: ASL-1(read-only)=低风险 → ASL-3(destructive)=需确认

5. Agent Memory Law: 有效记忆 ∝ retrieval_accuracy × context_length
   → retrieval_accuracy ∝ embedding质量 × 索引大小
   → context_length ∝ KV cache大小 → INT8 KV → 2x记忆容量
   → → Agent记忆优化 = INT8 KV + FlashInfer + Vector DB → 三层记忆!
```

## 从AI Infra到Agent Infra — 技能迁移

```
LLM Serving → Agent Serving 技能映射:

Prefix Sharing → Agent对话历史共享(84% KV省!)
Continuous Batching → Continuous Agent Batching(GPU利用率↑)
INT4 Quantization → Agent模型量化(成本↓7x)
FlashInfer → Agent多轮attention加速(GQA-8 52x)
Speculative Decoding → Agent步间推理加速(2.14x)
Guardrails → Agent安全约束(<5% overhead)
xgrammar CFSM → Agent结构化输出+安全FSM(zero overhead)
PagedAttention → Agent KV cache管理(按需分配)

→ AI Infra技能100%可迁移到Agent Infra!
→ RTX 4090 = 最具成本优势的Agent推理平台!
→ 7B INT4 + INT8 KV + FlashInfer → Agent serving完全可行!
→ 关键创新: prefix sharing for agents → 84% KV省 → 成本降84%!
```

## 关键论文与参考

```
- ReAct (Yao et al., 2023): Reason+Act范式 → Agent基础
- Toolformer (Schick et al., 2023): LLM自学习tool使用
- LangGraph (2024): 状态图Agent框架 → 生产级
- Claude Agent SDK (Anthropic, 2025): 官方Agent框架
- OpenAI Agents SDK (2025): 函数调用Agent框架
- CrewAI (2024): Role-play多Agent → 简单易用
- AutoGen (Microsoft, 2023): 对话驱动多Agent
- Constitutional AI (Anthropic, 2022): 自我修正+安全
- Tree-of-Thought (Yao et al., 2023): 分支推理+回溯
- Reflexion (Shinn et al., 2023): 执行→反思→改进循环
- xgrammar (MLSys 2025): CFSM结构化输出 → Agent安全约束
```