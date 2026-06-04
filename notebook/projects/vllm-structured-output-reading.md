# vLLM V1 Structured Output 源码阅读

> Structured Output / Guided Decoding: JSON Schema、Grammar、Bitmask Token 过滤

## 1. 架构概览

```
API 请求 (response_format=json_schema)
    │
    ├── 1. 请求验证 (Processor._validate_structured_output)
    │
    ├── 2. Grammar 编译 (StructuredOutputManager.grammar_init)
    │     └── JSON Schema → Grammar → FSM
    │
    ├── 3. Bitmask 生成 (grammar_bitmask)
    │     └── 每步生成 token 接受位掩码 [vocab_size]
    │
    ├── 4. Logit 修改 (apply_grammar_bitmask)
    │     └── Triton kernel: logits[~bitmask] = -inf
    │
    └── 5. 采样 (从合法 token 中采样)
```

## 2. 支持的结构化输出类型

| 类型 | 说明 | 示例 |
|------|------|------|
| `JSON` | JSON Schema 验证 | `{"type": "object", "properties": {"name": {"type": "string"}}}` |
| `JSON_OBJECT` | 强制 JSON 对象 | 任何合法 JSON object |
| `REGEX` | 正则表达式 | `[A-Z]{3}-\d{4}` |
| `GRAMMAR` | 上下文无关文法 (EBNF) | 自定义语法规则 |
| `CHOICE` | 预定义选项 | `["positive", "negative", "neutral"]` |
| `STRUCTURAL_TAG` | 文本中结构化标签 | reasoning 输出中的结构化部分 |

## 3. 后端架构

### 3.1 统一接口

```python
class StructuredOutputBackend(ABC):
    def compile_grammar(...)       # 编译语法规范
    def allocate_token_bitmask(...) # 创建 token bitmask
    def destroy(...)               # 清理资源

class StructuredOutputGrammar(ABC):
    def accept_tokens(...)    # 推进有限状态机
    def validate_tokens(...)  # 验证但不推进
    def rollback(...)         # 回退状态 (Spec Decode)
    def fill_bitmask(...)     # 创建 token 接受 bitmask
    def is_terminated(...)    # 检查生成是否完成
```

### 3.2 后端选项

| 后端 | 文件 | 特点 |
|------|------|------|
| **XGrammar** | `backend_xgrammar.py` | 推荐, 最快, C++ 实现 |
| **Guidance** | `backend_guidance.py` | LLGuidance, 灵活 |
| **Outlines** | `backend_outlines.py` | JSON→regex→index |
| **LM Format Enforcer** | `backend_lm_format_enforcer.py` | 兼容性后端 |

## 4. 核心文件

| 文件 | 作用 |
|------|------|
| `vllm/v1/structured_output/__init__.py` | StructuredOutputManager 主类 |
| `vllm/v1/structured_output/backend_types.py` | 抽象基类定义 |
| `vllm/v1/structured_output/backend_xgrammar.py` | XGrammar 后端 |
| `vllm/v1/structured_output/backend_guidance.py` | Guidance 后端 |
| `vllm/v1/structured_output/backend_outlines.py` | Outlines 后端 |

## 5. Bitmask 机制

### 5.1 工作原理

```
合法 token 集合 (来自 FSM):
    {token_42, token_103, token_289, token_456, ...}

Bitmask: [0, 0, ..., 1, ..., 0, ..., 1, ..., 0, ..., 1, ...]
         (vocab_size 位, 1=合法, 0=非法)

应用:
    logits[bitmask == 0] = -inf  → 非法 token 概率为 0
    softmax(logits) → 只在合法 token 上分布
```

### 5.2 Bitmask 生成流程

```python
# StructuredOutputManager.grammar_bitmask()
def grammar_bitmask(self, requests, request_ids, spec_tokens):
    for req_id in request_ids:
        grammar = self.requests[req_id]
        # 1. 推进 FSM (接受之前的 token)
        grammar.accept_tokens(prev_tokens)
        # 2. 填充 bitmask (哪些 token 合法)
        grammar.fill_bitmask(bitmask, index)
    return bitmask
```

### 5.3 Triton Kernel 应用

```python
# apply_grammar_bitmask() 使用 Triton kernel
# 高效地将 bitmask 应用到 logits
# logits[~bitmask] = -inf
```

## 6. 调度器集成

### 6.1 Scheduler 中的 Grammar 处理

```python
def get_grammar_bitmask(self, scheduler_output):
    # 收集需要结构化输出的请求
    structured_req_ids = [
        req_id for req_id in scheduler_output.scheduled_new_reqs
        if req_id in self.structured_output_manager.requests
    ]

    # 生成 bitmask
    bitmask = self.structured_output_manager.grammar_bitmask(
        self.requests,
        structured_req_ids,
        scheduler_output.scheduled_spec_decode_tokens,
    )
    return GrammarOutput(structured_req_ids, bitmask)
```

### 6.2 Model Runner 集成

```python
# gpu_model_runner.py execute_model():
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

## 7. JSON Schema → Grammar 转换

### 7.1 XGrammar 后端

```python
# 使用 xgrammar.GrammarCompiler 编译 JSON Schema
compiler = xgrammar.GrammarCompiler(tokenizer)
compiled_grammar = compiler.compile_json_schema(json_schema)
```

限制:
- 不支持 `multipleOf` (数值范围)
- 不支持 `uniqueItems` (数组唯一性)
- 不支持 `patternProperties` (对象模式)
- 部分 string format 不支持

### 7.2 Guidance 后端

```python
# 使用 llguidance.LLMatcher
# 自动添加 additionalProperties: false
# 支持空白灵活处理
```

### 7.3 Outlines 后端

```python
# JSON Schema → Regex → Compiled Index
regex = outlines_core.json_schema.build_regex_from_schema(json_schema)
index = regex_guide(indexer, regex)  # LRU 缓存
```

## 8. Speculative Decoding 兼容

结构化输出完全支持 Speculative Decoding:

```python
class StructuredOutputGrammar:
    def rollback(self, num_tokens):
        """回退 FSM 状态 (spec decode 拒绝时)"""

    max_rollback_tokens: int  # 最大可回退 token 数
```

流程:
1. Draft model 生成候选 token
2. Target model 验证时检查 grammar 约束
3. 如果拒绝 → rollback FSM 状态
4. 从修正分布中重新采样 (grammar 兼容)

## 9. 性能优化

| 优化 | 方法 |
|------|------|
| 异步编译 | Grammar 编译不阻塞请求 |
| Bitmask 批处理 | 并行生成多个请求的 bitmask |
| Triton Kernel | GPU 加速 bitmask 应用 |
| LRU 缓存 | Outlines 缓存编译后的 index |
| 预分配 | Bitmask 张量预分配，避免 realloc |
| 并行填充 | 超过阈值时并行处理 bitmask |

## 10. 配置与使用

### 10.1 API 使用

```python
# JSON Schema
response = client.chat.completions.create(
    model="my-model",
    messages=[...],
    extra_body={
        "guided_json": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
    }
)

# Regex
response = client.chat.completions.create(
    model="my-model",
    messages=[...],
    extra_body={"guided_regex": "[A-Z]{3}-\\d{4}"}
)

# Choice
response = client.chat.completions.create(
    model="my-model",
    messages=[...],
    extra_body={"guided_choice": ["positive", "negative", "neutral"]}
)
```

### 10.2 后端选择

```bash
# 默认使用 XGrammar
vllm serve my-model

# 手动选择后端
--guided-decode-backend xgrammar
--guided-decode-backend guidance
--guided-decode-backend outlines
```

## 11. 关键洞察

1. **Bitmask 机制**: O(1) token 过滤，不需要修改模型权重
2. **FSM 驱动**: 有限状态机跟踪生成进度，每步更新合法 token 集合
3. **后端可插拔**: 统一接口支持多种 Grammar 引擎
4. **异步编译**: Grammar 编译不阻塞推理
5. **Spec Decode 兼容**: rollback 机制支持投机解码的 token 拒绝
6. **XGrammar 推荐**: C++ 实现，最快
7. **Structural Tag**: 支持推理模型的结构化输出 (thinking + answer)
8. **Triton Kernel**: GPU 加速 bitmask 应用，避免 CPU 瓶颈
9. **LRU 缓存**: 相同 schema 不重复编译
10. **与 LoRA 正交**: 结构化输出与 LoRA 独立工作

## 参考资料

- `vllm/v1/structured_output/__init__.py` — StructuredOutputManager
- `vllm/v1/structured_output/backend_types.py` — 后端抽象基类
- `vllm/v1/structured_output/backend_xgrammar.py` — XGrammar 后端
- `vllm/v1/structured_output/backend_guidance.py` — Guidance 后端
- `vllm/v1/structured_output/backend_outlines.py` — Outlines 后端
- `vllm/v1/core/sched/scheduler.py` — Grammar bitmask 集成
- `vllm/v1/worker/gpu_model_runner.py` — Bitmask 应用
- 相关: [Sampling Pipeline](vllm-sampling-pipeline-reading.md)
