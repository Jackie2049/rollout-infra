# verl Multi-Turn Agent Loop 源码阅读

> 2026-06-15 | 源码: verl/experimental/agent_loop/ + verl/workers/config/rollout.py
> 核心: AgentLoop=pluggable coroutine agent框架 → response_mask=[1,0,1,...] → 只LLM tokens参与loss → ToolAgentLoop状态机(GENERATING→PROCESSING_TOOLS) → ToolParser 4种格式

## 1. AgentLoop架构概览

```
verl multi-turn rollout 架构:

AgentLoopManager (coordinator)
  ├── AgentLoopWorker[] (Ray remote actors)
  │     ├── asyncio.gather → 每个sample 1个coroutine
  │     ├── AgentLoopBase.run() → 执行agent loop
  │     └── postprocess → DataProto batch
  │
  ├── split → 分配给多个Worker
  └── merge → 合合Worker结果

AgentLoopBase (ABC) → 只需实现 async def run()
  ├── SingleTurnAgentLoop (@register("single_turn_agent"))
  └── ToolAgentLoop (@register("tool_agent"))
```

## 2. AgentLoopOutput 数据结构

```python
# agent_loop.py:88
class AgentLoopOutput(BaseModel):
    prompt_ids: list[int]           # prompt token ids
    response_ids: list[int]         # response(LLM生成+tool response混合)
    response_mask: list[int]        # ★ 1=LLM tokens(训练), 0=tool/padding(不训练)
    response_logprobs: Optional[list[float]] = None
    num_turns: int = 0              # 总交互轮次(user+assistant+tool)
    reward_score: Optional[float] = None
    metrics: AgentLoopMetrics
    extra_fields: dict[str, Any] = {}

# ★ response_mask是核心! → 多步RL标准设计:
#   LLM生成的action tokens → mask=1 → 参与PPO/GRPO loss
#   Tool/environment response tokens → mask=0 → 不参与loss
#   → 策略只学"做什么"不学"看什么"!
```

## 3. ToolAgentLoop 状态机

```python
# tool_agent_loop.py
class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"        # LLM生成tokens → action
    PROCESSING_TOOLS = "processing_tools"  # 执行tool → observation
    TERMINATED = "terminated"

# 状态转换:
PENDING → GENERATING (LLM produces tokens)
GENERATING → if tool_calls found → PROCESSING_TOOLS
PROCESSING_TOOLS → append tool response → GENERATING again
GENERATING → if no tool_calls → TERMINATED

# 每轮:
  1. LLM生成 → output_ids → mask=[1]*len → action tokens
  2. 如果有tool_call → 解析→执行→response_ids → mask=[0]*len → observation tokens
  3. response_ids+mask append到AgentData → 继续下一轮
```

### AgentData 状态对象

```python
# tool_agent_loop.py:56
class AgentData:
    messages: list[dict[str, Any]]   # 对话历史
    prompt_ids: list[int] = []
    response_ids: list[int] = []
    response_mask: list[int] = []    # ★ 增量构建: LLM=[1], tool=[0]
    response_logprobs: list[float] = []
    turn_scores: list[float] = []
    tool_rewards: list[float] = []
    user_turns: int = 0
    assistant_turns: int = 0
    tool_calls: list[FunctionCall] = []
    extra_fields: dict[str, Any] = {}
```

## 4. ToolParser 注册式架构

```python
# tool_parser.py
class ToolParser(ABC):
    """从LLM输出token IDs提取tool calls"""
    @abstractmethod
    def parse_tool_calls(self, token_ids) -> list[FunctionCall]: ...

# 4种注册parser:
@register("hermes")       → hermes格式(通用)
@register("gpt_oss")      → OpenAI function calling格式
@register("qwen3_coder")  → Qwen3 Coder专用格式
@register("gemini4")      → Gemini4格式(实验性)

# 功能:
# 1. 从LLM输出token IDs提取tool calls
# 2. 解析function_name + arguments
# 3. 返回FunctionCall对象列表 → AgentData.tool_calls

# 关键: ToolParser是pluggable → 不同模型格式 → 不同parser → 一行注册!
```

## 5. MultiTurnConfig

```python
# rollout.py:66
@dataclass
class MultiTurnConfig(BaseConfig):
    enable: bool = False
    max_assistant_turns: Optional[int]
    max_user_turns: Optional[int]
    max_parallel_calls: int = 1             # 并行tool calls数
    max_tool_response_length: int = 256     # tool response最大长度
    tool_response_truncate_side: str = "middle"
    format: str = "hermes"                  # tool parser格式
```

## 6. SingleTurn vs ToolAgentLoop对比

```
SingleTurnAgentLoop:
  → num_turns=2(user+assistant) → 1次LLM生成
  → response_mask=[1]*len → 所有response参与loss
  → 等价于标准单步RL → 与传统PPO/GRPO一致

ToolAgentLoop:
  → 多轮GENERATING→PROCESSING_TOOLS循环
  → response_mask=[1,0,1,0,...] → 只LLM tokens参与loss
  → 适用: ReAct agent / tool-use agent / multi-step reasoning
  → 环境→action→观察→再action → RL只优化action选择!
```

## 7. 与rLLM多步RL对比

| 维度 | verl AgentLoop | rLLM prefix-merge |
|------|---------------|------------------|
| **数据格式** | AgentLoopOutput(prompt+response+mask) | Datum(prompt+A0+obs1+A1+obs2+mask) |
| **mask构建** | 增量: LLM=[1], tool=[0] | 增量: action=[1], observation=[0] |
| **loss参与** | mask=1 → PPO/GRPO loss | mask=1 → PPO/IS/GRPO loss |
| **环境交互** | ToolParser→FunctionCall→execute | env.step()→observation→append |
| **跨Worker** | Ray AgentLoopWorker → DataProto | in-process Tinker → 直接 |
| **工具格式** | 4种ToolParser注册式 | 自定义env.step() |

```
等价性: mask设计完全一致! → [1,0,1,...] → 多步RL标准设计
差异: verl ToolParser→标准化tool execution; rLLM→in-process最快
```

## 8. GRPO multi-turn训练流程

```
1. AgentLoopManager.generate_sequences()
   → ToolAgentLoop.run() → 多轮GENERATING+PROCESSING_TOOLS
   → AgentLoopOutput(prompt_ids + response_ids + mask)

2. postprocess → DataProto(batch={
     input_ids, response_mask=[1,0,1,0,...], log_probs(only mask=1)
   })

3. compute_grpo_advantage
   → outcome reward × mask → 只action tokens
   → mask=0 tokens advantage=0 → 不影响loss!

4. ActorWorker.update_policy
   → GRPO loss × response_mask → 只action tokens参与
   → observation tokens loss=0 → 策略不学"看什么"
```

## 9. RTX 4090实战

```
RTX 4090 24GB (7B, GRPO+multi-turn):

可行: HYBRID+naive+GRPO+LoRA+ToolAgentLoop → 17GB peak ✓
  → multi-turn response ~1500 tokens → INT4推理 0.31s → 可接受
  → 但batch推理受限 → 8×multi-turn → 2.5s

vs rLLM Tinker: in-process更快 → 但verl ToolParser更标准化
选择: 需要tool-use → verl ToolAgentLoop; 纯RL → rLLm Tinker
```

## 10. 关键设计洞察

```
1. response_mask=[1,0,1,...] → 多步RL标准设计 → 与rLLm等价
2. 状态机 → GENERATING→PROCESSING_TOOLS → 清晰LLM/tool交替
3. ToolParser注册式 → 4种格式 → 新模型一行注册
4. AgentLoopOutput → 统一多步数据 → GRPO/PPO×mask → 通用!
5. verl vs rLLm: 设计一致, 实现不同 → 单GPU=rLLm, 多GPU=verl
6. Multi-turn是2025-2026核心方向 → Agent RL + 推理scaling
```

---

Sources:
- verl/experimental/agent_loop/agent_loop.py — AgentLoopBase + AgentLoopOutput
- verl/experimental/agent_loop/tool_agent_loop.py — ToolAgentLoop状态机 + AgentData
- verl/experimental/agent_loop/single_turn_agent_loop.py — SingleTurnAgentLoop
- verl/experimental/agent_loop/tool_parser.py — ToolParser注册式
- verl/workers/config/rollout.py — MultiTurnConfig
- notebook/projects/rllm-gateway-backend-trainer-source-reading.md
