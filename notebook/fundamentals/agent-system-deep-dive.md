# Agent System Architecture Deep Dive — LLM Agent=Planning+Tool Use+Memory+Execution(ReAct→Plan-and-Execute→Multi-Agent) + Toolformer(Function Calling+MCP Protocol+Structured JSON) + Frameworks(LangChain/LangGraph/AutoGen/CrewAI/Swarm/MetaGPT) + Serving Overhead(推理40%+API30%+Prompt20%+Parse10%→Agent延迟2-4x单次) + SGLang KV Cache Reuse(20-40%减) + RTX 4090(7B 3-5ms/tok+3-step agent 3.2-5.5s+Phi-3-mini 150 tok/s=实时agent可行) + 2026趋势(MCP标准化+Self-correction+World Model+Cascade Serving)

> 2026-06-14 | LLM Agent系统架构深度分析: 从ReAct(2022 Reasoning+Acting交织)到Plan-and-Execute(规划→执行分离)到Multi-Agent(AutoGen/CrewAI/Swarm) → Toolformer(2023 自监督工具学习)到Function Calling(2025并行+strict+MCP标准化) → 4层Agent架构(Planning+Memory+Tool Use+Execution) → Agent serving overhead(推理仅40%! → API30%+Prompt20%+Parse10% → 总延迟2-4x单次推理) → SGLang KV cache reuse+RadixAttention → RTX 4090 7B agent 3.2-5.5s → Phi-3-mini 150 tok/s → 2026 MCP标准化+World Model+Cascade Serving
> 关联: ai-expert-knowledge-map-gap-analysis.md(Agent gap★→★★★★), inference-perf skill(推理性能), multimodal-vlm-deep-dive.md(VLM Agent)
> 参考: ReAct(Yao et al. 2022), Toolformer(Schick et al. 2023), Lilian Weng Agent Survey(2023), Anthropic Building Effective Agents(2025), MCP Protocol(Anthropic 2024-25), SGLang(Liang et al. 2024-25), AutoGen(Microsoft), CrewAI, LangGraph

## 0. 核心定律: Agent=Planning+Tool Use+Memory+Execution → 推理仅40%延迟 → SGLang KV reuse减20-40%

```
Agent架构核心:

  LLM Agent = 4个核心组件:
    → Planning(规划): 任务分解 → 子任务 → 步骤 → ReAct/Plan-and-Solve/MCTS
    → → → Memory(记忆): 短期(context)+长期(vector DB) → 保持上下文!
    → → → → → Tool Use(工具): Function Calling → API → MCP → 外部能力!
    → → → → → → → Execution(执行): 调用工具 → 获取结果 → 继续推理!

  Agent延迟 = 推理40% + API 30% + Prompt重建20% + 解析10%:
    → → GPU推理不是瓶颈 → API调用+prompt重建才是!
    → → → → → 每步agent → 50-200ms overhead → 3-5步 → 总延迟2-4x单次推理!
    → → → → → → → SGLang KV cache reuse → 20-40%减 → 关键优化!

  RTX 4090 Agent推理:
    → 7B INT4 → 3-5ms/tok → 单次推理快 → 但agent多步 → 总延迟5.5s(standard)→3.2s(SGLang)
    → → → → → Phi-3-mini 3.8B → 150 tok/s → 实时agent可行 → cascade serving!
    → → → → → → → → → 关键: 小模型做routing → 大模型做reasoning → 分级!
```

## 1. ReAct — Reasoning+Acting交织(Foundation Pattern)

```
### 1.1 ReAct原理

ReAct(2022) = Reasoning + Acting → 交织 → 不是分开 → 同时推理+行动!

流程:
  → Think: "我需要查找巴黎的人口数据" → reasoning trace → 生成思考!
  → → → Act: search("Paris population") → 调用工具 → 执行行动!
  → → → → → → → → → → Observe: "巴黎人口2.16M" → 观察结果 → 更新上下文!
  → → → → → → → → → → → → → → → → Think: "现在我需要比较巴黎和伦敦..." → 继续推理!

vs 纯推理(CoT):
  → CoT: 只推理 → 不行动 → 无法获取外部信息 → 局限!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → ReAct: 推理+行动 → 获取外部信息 → 更准确!

vs 纯行动(Act-only):
  → Act-only: 只行动 → 不推理 → 行动无方向 → 效率低!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → ReAct: 先推理再行动 → 有方向 → 更高效!

### 1.2 ReAct变体

ReAct→基础 → 2025扩展:

  Plan-and-Execute(2023):
    → 不是交织 → 而是分离 → 先规划全部步骤 → 再逐步执行!
    → → → → → → → → → → → → → → → → → → → → → → → 优势: 规划更全局 → 执行更可控 → 减少走弯路!
    → → → → → → → → → → → → → → → → → → → → → → → → → → 劣势: 规划可能不适应执行反馈 → 需动态调整!

  LLM Compiler(2024-2025):
    → Plan-and-Execute进阶 → 并行执行 → 多步骤同时进行 → 加速!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 优势: 并行 → 更快 → 减少总延迟!

  Tree of Thoughts(ToT):
    → 搜索树 → 多条推理路径 → 选择最优 → 更精细规划!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → MCTS结合 → 2025前沿!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: ReAct=基础 → Plan-and-Execute=进阶 → ToT=前沿 → 从简到复杂!
```

## 2. Toolformer & Function Calling — 工具使用能力

```
### 2.1 Toolformer — 自监督学习工具调用

Toolformer(2023) = LLM自学习 → 何时调用+调用哪个+传什么参数+如何整合!

方法:
  → 给LLM一组API工具 → LLM自己尝试调用 → 评估有用性 → 保留有用调用!
  → → → → → → → → → → → → → → → → → → → → → → → → → → 自监督 → 不需人工标注 → 模型自己学会!

关键能力:
  → 何时调用: 模型判断"需要外部信息" → 决定调用 → 不每次都调用!
  → → → → → 调用哪个: 选择最相关API → search vs calculator vs translator → 正确选择!
  → → → → → → → 传什么参数: 构造正确参数 → search("Paris population") → 准确!
  → → → → → → → → → 如何整合: 把结果融入后续推理 → 自然衔接 → 无缝!

### 2.2 Function Calling(2025) — 结构化工具调用

2025 Function Calling = OpenAI/Anthropic结构化 → JSON schema → 严格 → 生产!

演进:
  → 2023(OpenAI): 基础function calling → JSON → 简单 → 但不稳定!
  → → → 2025: 并行function calling → 多工具同时调用 → 加速!
  → → → → → → → → → Strict mode → schema强制 → 100%符合 → 生产可靠!
  → → → → → → → → → → → → → → → → → → Multi-turn → 多轮工具调用 → agent循环!

### 2.3 MCP(Model Context Protocol) — Anthropic标准化

MCP(2024-2025) = Anthropic开源协议 → 标准化 → LLM连接外部工具 → 统一!

核心:
  → Open protocol → 任何LLM → 任何工具 → 任何数据源 → 标准化!
  → → → → → → → → → → → → → → → → → → → → → → → → → → 替代ad-hoc function calling → 统一协议!

架构:
  → MCP Host(LLM应用) → MCP Client(连接) → MCP Server(工具/数据) → 标准!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Tools/Resources/Prompts → 三类暴露 → 统一!

优势:
  → 标准化 → 所有工具统一接入 → 不需要每工具单独适配!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 安全 → 权限控制 → 审计 → 生产必需!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: MCP=2026标准化方向 → Claude Code本身就是MCP client → 我们直接体验!
```

## 3. Agent Memory — 记忆系统

```
Agent Memory = 短期+长期 → 保持上下文 → 多步推理必要!

短期记忆(Working Memory):
  → Context window → 当前对话 → 最近的工具调用结果 → 临时!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 限制: 4K-128K tokens → 长agent对话 → 可能溢出 → 需管理!

长期记忆(Long-term Memory):
  → Vector DB → ChromaDB/Pinecone → 历史经验 → 持久!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 语义检索 → 相似经验 → 参考过去 → 避免重复!

Episodic Memory(经验记忆):
  → 过去任务 → 成功路径 → 失败原因 → 学习!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Self-reflection → 评估 → 修正 → 改进 → 持续学习!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → RTX 4090: 7B 4K context → agent对话 → 需管理 → KV cache + prefix sharing → 有效!
```

## 4. Multi-Agent架构 — 多智能体协作

```
### 4.1 5大Multi-Agent框架

| 框架 | 架构 | 特色 | 适用场景 |
|------|------|------|---------|
| AutoGen(Microsoft) | 对话式 | 多agent拓扑+灵活+工具集成 | 研究+对话协作 |
| LangGraph | 图式 | 循环workflow+stateful+human-in-loop | 生产+复杂流程 |
| CrewAI | 角色式 | sequential/hierarchical/consensus | 企业+角色分工 |
| Swarm(OpenAI) | handoff式 | agent交接+routine → 简洁 | 教育+轻量 |
| MetaGPT | SOP式 | 标准操作流程+角色 → 结构化 | 代码生成+开发 |

### 4.2 Multi-Agent协调模式

Anthropic(2025)3种模式:

1. Hierarchical(层级):
   → 主agent→规划 → 子agent→执行 → 层级分工!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 例: Planner → Researcher → Writer → 3层 → 规划→研究→写作!

2. Peer-to-Peer(平等):
   → 多agent平等协作 → 对话→协商 → 无层级!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 例: 3个agent讨论 → 各抒己见 → 共同决策 → 协商!

3. Broadcast(广播):
   → 一个agent发出任务 → 所有agent响应 → 并行!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 例: 分解任务 → 并行执行 → 合并结果 → 加速!

### 4.3 Agent设计哲学(Anthropic 2025)

Anthropic建议 → "从最简开始 → 只在需要时增加复杂度":

  Level 1 — Augmented LLM + Tools:
    → 单LLM + 工具 → 最简单 → 大多数任务够用!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 例: Claude Code = Level 1 → 单LLM + tools(read/write/bash) → 够用!

  Level 2 — ReAct Loop:
    → Think→Act→Observe循环 → 多步推理 → 需要外部信息时用!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 例: search → read → analyze → 多步!

  Level 3 — Multi-Agent:
    → 多agent协作 → 复杂任务 → 只有真正需要才用!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 例: 规划+研究+写作 → 分工 → 多agent!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 关键: Level 1够用 → 不要过度设计 → 简单=好 → Claude Code证明!
```

## 5. Agent Serving Overhead — 延迟分析

```
### 5.1 Agent延迟组成

Agent总延迟 ≠ 推理延迟 → 推理只40% → 其他开销占60%!

延迟分解(Modal Blog 2025):
  → 推理(GPU): 40% → LLM推理 → 这部分我们优化最多!
  → → → → → → API调用(工具执行): 30% → 搜索/数据库/计算 → 外部服务!
  → → → → → → → → → → Prompt重建: 20% → 构造新prompt → 加入工具结果 → 重写!
  → → → → → → → → → → → → → → → → 解析/路由: 10% → JSON解析 → tool选择 → routing!

3步agent总延迟:
  → Standard vLLM: ~5.5s → 3步 × (推理+API+重建+解析) → 累积!
  → → → → → → → → → SGLang optimized: ~3.2s → KV cache reuse → 减重复推理 → 20-40%减!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → A100: ~3.5s(standard) / ~2.1s(SGLang) → GPU更强 → 但非GPU开销不变!

### 5.2 Tool Call Overhead详解

每步tool call overhead → 50-200ms → 不是GPU → 而是系统开销!

组成:
  → JSON parsing: ~5-15ms → 解析function call → 提取参数 → 结构化!
  → → → → → API execution: ~30-100ms → 调用外部API → 网络延迟 → 服务处理!
  → → → → → → → → → → → → → → → Prompt reconstruction: ~10-50ms → 加入结果 → 重写 → tokenizer!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → GPU re-inference: ~20-40ms → context switch → 重新推理 → 每步!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 关键: API+prompt重建占80% → 不是GPU → 减少外部调用 → 最有效优化!

### 5.3 SGLang优化 — KV Cache Reuse

SGLang关键创新 → RadixAttention → KV cache跨步骤共享 → 减重复计算!

原理:
  → Agent步骤1 → 推理 → KV cache → 存储!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 步骤2 → 共享前缀KV → 只计算新增部分 → 不重算整个prompt!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 步骤3 → 继续共享 → 前缀不变 → 只加新 → 更快!

效果:
  → 20-40%延迟减 → RTX 4090 3-step agent → 5.5s→3.2s → 显著!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结构化输出 → JSON mode → 减解析50% → 配合KV reuse → 最优!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: SGLang+structured output+KV reuse = agent serving最优 → 20-40%减!
```

## 6. RTX 4090 Agent推理策略

```
### 6.1 RTX 4090 Agent推理配置

| 模型 | INT4推理 | 3-step agent(standard) | 3-step agent(SGLang) | 可行? |
|------|---------|------------------------|---------------------|-------|
| Phi-3-mini 3.8B | ~150 tok/s | ~2.5s | ~1.5s | ✅ 实时! |
| Llama-3-8B | ~200 tok/s | ~5.5s | ~3.2s | ✅ 可行 |
| Llama-3-8B + INT4 | ~4800 tok/s decode | ~3s | ~2s | ✅ 最优! |
| 7B(INT4+INT8KV) | ~4800 tok/s | ~3s | ~2s | ✅ 最优! |

关键配置:
  → 7B INT4+INT8KV+FlashInfer+GQA-8 → 4,791 tok/s → 推理快 → agent可行!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → SGLang + RadixAttention → KV reuse → 20-40%减 → 总延迟~2s → 实时!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Structured output(JSON mode) → 减解析50% → 配合 → 最优!

### 6.2 Cascade Serving — 分级推理

Cascade Serving = 小模型routing + 大模型reasoning → 分级 → 最practical!

架构:
  → Level 1 — 小模型(Phi-3-mini 3.8B): routing → tool选择 → 分类 → 快(~1ms)!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Level 2 — 大模型(7B/13B): reasoning → 复杂推理 → 只在需要时调用!

优势:
  → 小模型快 → routing/分类 → 150 tok/s → 实时 → 不等大模型!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 大模型精确 → reasoning → 只复杂任务 → 节省GPU时间 → 成本优化!

RTX 4090 cascade:
  → Phi-3-mini INT4 → ~2GB → routing → 快!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 7B INT4 → ~4GB → reasoning → 24GB有剩余 → KV cache!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 同时在24GB → routing+reasoning → 都可行 → 单GPUcascade!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: Cascade=RTX 4090最优 → 小模型routing+大模型reasoning → 单GPU可行!
```

## 7. 2025-2026 Agent趋势

```
Agent架构趋势:

1. MCP标准化(Anthropic 2024-2025):
   → 开源协议 → 任何LLM→任何工具 → 统一 → 2026标配!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Claude Code = MCP client → 我们直接使用 → 体验标准化!

2. Self-correction Loops(自我修正):
   → Agent执行 → 评估结果 → 发现错误 → 重新规划 → 修正 → 闭环!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 关联: GRPO RL = self-correction → reward→修正 → RL loop → 同框架!

3. World Model Planning(世界模型):
   → Agent内部模型 → 预测环境 → 模拟行动 → 选择最优 → 更强规划!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 2025前沿 → ToT+MCTS → 搜索+模拟 → 2026深化!

4. Cascade Serving(分级推理):
   → 小模型routing → 大模型reasoning → 分级 → 成本优化 → 生产标准!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → a16z报告: cascade=2025生产趋势 → 小模型+大模型 → 分级!

5. VLM Agent(视觉行动):
   → VLM+行动 → GUI导航 → 屏幕理解 → 行动 → 2025大方向!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Gemini 2.0 → agentic → 视觉行动 → 2026主流!

6. Persistent Agent(持久化):
   → Agent不再是一次性 → 持久 → 长期记忆 → 学习 → 进化!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 关联: continual-learning → Agent持续学习 → 不遗忘 → evolution!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: 2026→MCP标准化+Self-correction+Cascade Serving+VLM Agent → 从简到复杂 → Anthropic方向!
```

## 8. 核心规律

```
Agent核心规律:

  Agent=4组件 → Planning+Memory+Tool Use+Execution → ReAct=基础pattern!
  → → Planning: ReAct(交织)→Plan-and-Execute(分离)→ToT(搜索) → 从简到复杂!
  → → → Memory: 短期(context)+长期(vector DB)+经验(episodic) → 三层记忆!
  → → → → → Tool Use: Toolformer→Function Calling→MCP → 标准化方向!
  → → → → → → → Execution: 调用工具→获取结果→继续推理 → agent循环!

  Agent延迟 ≠ 推理延迟 → 推理仅40% → API30%+Prompt20%+Parse10%!
  → → → → → 每步50-200ms overhead → 3-5步 → 总延迟2-4x → 需优化非GPU部分!
  → → → → → → → SGLang KV reuse → 20-40%减 → structured output → 减解析50% → 最优!

  Anthropic建议 → "从最简开始" → Level 1(LLM+tools)够用 → 不过度设计!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Claude Code = Level 1 → 单LLM+tools → 成功 → 证明简单=好!

  RTX 4090 Agent最优:
    → 7B INT4+INT8KV+SGLang+structured output → 3-step agent ~2s → 实时!
    → → Cascade Serving → Phi-3-mini routing + 7B reasoning → 单24GB → 分级!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Guardrails → Agent输出也要过滤 → xgrammar+MCP权限 → 安全!

  知识Gap修复:
    → Agent系统从★★(2/5) → ★★★★(4/5) → ReAct+Toolformer+Function Calling+MCP+Multi-Agent+Serving+RTX 4090 → 全面!
    → → → → → 但仍需实践 → GPU可用时 → vLLM agent serving+MCP部署 → 实测!
```

## 参考文献

```
1. Agent基础:
   - ReAct: Yao et al. 2022, arxiv.org/abs/2210.1356
   - Toolformer: Schick et al. 2023, arxiv.org/abs/2302.04761
   - Lilian Weng Agent Survey: 2023, lilianweng.github.io/posts/2023-06-23-agent

2. 2025实践:
   - Anthropic Building Effective Agents: anthropic.com/research/building-effective-agents
   - MCP Protocol: modelcontextprotocol.io
   - Anthropic Multi-Agent Patterns: anthropic.com/research/multi-agent-patterns

3. Frameworks:
   - AutoGen: microsoft.github.io/autogen
   - LangGraph: langchain-ai.github.io/langgraph
   - CrewAI: docs.crewai.com
   - Swarm: github.com/openai/swarm
   - MetaGPT: github.com/geekan/MetaGPT

4. Serving:
   - SGLang: github.com/sgl-project/sglang
   - vLLM function calling: vllm-project/vllm
   - Modal Agent Latency: modal.com/blog
   - Hamel Husain Agent Overhead: hamel.dev/blog

5. 我们的笔记:
   - ai-expert-knowledge-map-gap-analysis.md → Agent gap评估
   - inference-perf skill → 推理性能+agent延迟
   - multimodal-vlm-deep-dive.md → VLM Agent(视觉行动)
   - continual-learning-deep-dive.md → Agent持续学习
