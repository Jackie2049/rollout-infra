# vLLM V1 LogitsProcessor Pipeline Deep Dive

> 2026-06-07 | 源码版本: vLLM main (commit b4b4aaa70)
> 关联 PR: Jackie2049/vllm#7 (Top-nσ example)

## 概览

vLLM V1 的 logits processor 是一个 **batch-level** 处理管线，在 softmax 之前对 logits 张量进行预处理。核心设计思想：

- **批量处理**: 不是 per-request callback，而是 batch-level processor 直接操作 `[B, V]` logits 张量
- **argmax 分类**: processor 分为两类 — argmax_invariant (不影响贪心) 和 non_argmax_invariant (可能改变贪心结果)
- **持久状态**: 通过 `BatchUpdate` 维护跨 step 的 batch slot 映射

## 1. 数据流: SamplingParams → GPU apply()

```
用户代码                     vLLM内部
─────────                    ─────────
SamplingParams               Scheduler._add_request()
  .extra_args={"top_n_sigma":2.0}     → params 存入 Request对象
                                       → schedule 时 params 传给 InputBatch

InputBatch                   GPUModelRunner._prepare_inputs()
  .logitsprocs                         → build SamplingMetadata
  .logitsprocs_need_output_token_ids   → 传给 Sampler

Sampler                      GPU 上执行
  .apply_logits_processors()           → logits.to(float32)
                                       → allowed_token_ids_mask
                                       → bad_words
                                       → non_argmax_invariant processors ← 在 penalty 之前
                                       → apply_penalties()
                                       → thinking_budget
                                       → (回到 sample())
                                       → temperature
                                       → argmax_invariant processors ← 在 top_k/top_p 之前
                                       → topk_topp_sampler
                                       → greedy vs random 选择
```

**关键**: logits processors 在两个位置执行：
1. `apply_logits_processors()` 中: **non_argmax_invariant** processors (在 penalty 之前)
2. `sample()` 中: **argmax_invariant** processors (在 temperature 之后, top_k/top_p 之前)

## 2. build_logitsprocs(): 处理器链构建

```
gpu_model_runner.py:668-674

logitsprocs = build_logitsprocs(
    self.vllm_config,      # VllmConfig (含 scheduler_config)
    self.device,            # torch.device
    self.pin_memory,        # bool
    self.is_pooling_model,  # bool → pooling 不支持 custom logitproc
    custom_logitsprocs,     # Sequence[str | type[LogitsProcessor]]
)
```

**构建顺序** (`__init__.py:212-217`):
```python
LogitsProcessors(
    ctor(vllm_config, device, is_pin_memory)
    for ctor in itertools.chain(
        BUILTIN_LOGITS_PROCESSORS,    # [MinTokens, LogitBias, MinP]
        custom_logitsprocs_classes,   # 用户自定义
    )
)
```

_builtin 先初始化，custom 后初始化。初始化顺序即 apply 顺序。_

**custom 加载三种方式** (`__init__.py:158-181`):
1. **Entrypoint**: `importlib.metadata.entry_points(group="vllm.logits_processors")` → 插件机制
2. **FQCN**: `"module.path:ClassName"` → `importlib.import_module()` + `getattr()` 动态加载
3. **Class object**: 直接传 `TopNSigmaLogitsProcessor` 类型 → 无需导入，最简单

**Top-nσ 使用方式 3**:
```python
llm = LLM(model="...", logits_processors=[TopNSigmaLogitsProcessor])
```

## 3. LogitsProcessors: argmax 分类

```
state.py:148-166

class LogitsProcessors:
    argmax_invariant: list[LogitsProcessor]      # 不影响贪心
    non_argmax_invariant: list[LogitsProcessor]   # 可能改变贪心
```

**分类结果** (当前 builtin + Top-nσ):
| Processor | argmax_invariant | 执行位置 |
|-----------|-----------------|----------|
| MinTokens | **False** | apply_logits_processors() (penalty 前) |
| LogitBias | **False** | apply_logits_processors() (penalty 前) |
| MinP | **True** | sample() (temperature 后, top_k/top_p 前) |
| Top-nσ | **True** | sample() (temperature 后, top_k/top_p 前) |

**为什么分两类？**
- non_argmax_invariant processors 在 `apply_logits_processors()` 中执行，影响贪心采样 → 需在 penalty 前运行，因为 penalty 也可能改变 argmax
- argmax_invariant processors 在 `sample()` 中执行，不影响贪心 → 可以在 temperature + top_k/top_p 之前运行，让它们先筛选再进 random sampling

## 4. BatchUpdate: 持久状态管理

```
interface.py:36-58

@dataclass(frozen=True)
class BatchUpdate:
    batch_size: int               # 当前 batch 大小
    removed: Sequence[int]        # 被移除的 batch slot indices
    added: Sequence[AddedRequest] # (slot_index, SamplingParams, prompt_ids, output_ids)
    moved: Sequence[MovedRequest] # (from_idx, to_idx, directionality)
```

**处理顺序**: removed → added → moved (接口文档明确要求)

**AddedRequest 的关键设计**:
```python
AddedRequest = (index, SamplingParams, prompt_tok_ids, output_tok_ids)
```
- `output_tok_ids` 是 **引用** (不是拷贝!) → processor 总能看到最新的生成 token 列表
- 这对 MinTokensLogitsProcessor 很关键: 它需要知道当前已生成多少 token 来决定是否屏蔽 EOS

**BatchUpdateBuilder** (`state.py:18-146`):
- 维护 removed/added/moved 列表
- removed 列表始终 **降序排序** → 从高 index 开始移除，避免 index 偏移
- `pop_removed()` 返回最低 index → 用于 "condensing" (压缩 batch 填充空位)

## 5. process_dict_updates(): sparse dict 通用更新

```
builtin.py:294-332

def process_dict_updates(
    req_entries: dict[int, T],          # 如 {slot: n_sigma_value}
    batch_update: BatchUpdate | None,
    new_state: Callable[[SamplingParams, ...], T | None],
) -> bool:
```

**操作逻辑**:
1. **added**: 对每个新请求调用 `new_state(params, prompt_ids, output_ids)` → 如果返回值非 None，存入 `req_entries[slot]`
2. **removed**: 对每个移除的 slot，`req_entries.pop(slot)` → 如果有值，返回 True 表示需要更新
3. **moved**:
   - UNIDIRECTIONAL (a→b): `pop(a)` → 存入 `req_entries[b]`，清除 a
   - SWAP (a↔b): `pop(a)` → 存入 b，`pop(b)` → 存入 a

**Top-nσ 使用方式**:
```python
# top_n_sigma.py:86-90
process_dict_updates(
    self.req_info,                    # {slot: n_sigma_float}
    batch_update,
    lambda params, _, __: extract_n_sigma(params),
)
```

**返回 bool 的用途**: LogitBiasLogitsProcessor 用它判断是否需要重建 scatter tensors。

## 6. 完整执行顺序 (一个 step)

```
Scheduler._schedule()
  → 构建 BatchUpdateBuilder (removed/added/moved)
  → 构建 BatchUpdate
  → 传给 InputBatch.sync_batch()
    → logitsprocs.update_state(batch_update)  ← 每个 processor 更新自己的 dict
    → 每个 processor 内部调用 process_dict_updates()

ModelRunner.execute_model()
  → model.forward() → logits [B, V] GPU tensor
  → Sampler.forward()
    → logits.to(float32)
    → compute raw_logprobs (可选)
    → apply_logits_processors(logits, metadata)
      → allowed_token_ids_mask
      → bad_words
      → non_argmax_invariant processors.apply()  ← LogitBias, MinTokens
      → apply_penalties()                        ← freq/pres/repetition
      → thinking_budget.apply_to_logits()        ← (如有)
    → sample(logits, metadata)
      → greedy_sample(logits)                    ← argmax (不受 invariant processors 影响)
      → apply_temperature(logits)
      → argmax_invariant processors.apply()      ← MinP, Top-nσ
      → topk_topp_sampler(logits)                ← top_k + top_p
      → torch.where(T < ε, greedy, random)       ← 混合选择
    → gather logprobs (可选)
    → return sampled tokens
```

## 7. Top-nσ 在管线中的精确位置

Top-nσ 是 argmax_invariant=True，所以在 `sample()` 中执行:
```
temperature → Top-nσ → top_k/top_p → random sampling
```

这意味着:
- Top-nσ 在 temperature 缩放后执行 → temperature invariance 自然保证
- Top-nσ 在 top_k/top_p 之前执行 → 先用统计阈值筛选，再用 top_k/top_p 进一步过滤
- Top-nσ 不影响 greedy decoding → argmax 路径完全跳过 Top-nσ

**为什么 Top-nσ 在 temperature 后执行是关键?**
因为 `threshold = max(logits/T) - n * std(logits/T)` = `(max(logits) - n * std(logits)) / T`
temperature 缩放后 threshold 和 logits 都被 T 缩放，但 **相对比较不变** → 哪些 token 被过滤不受 T 影响。

## 8. 自定义 Processor 编写模式

### Pattern 1: Engine-level (直接继承 LogitsProcessor)
```python
class TopNSigmaLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.req_info: dict[int, float] = {}

    def is_argmax_invariant(self) -> bool:
        return True  # Top-nσ 不影响 greedy

    def update_state(self, batch_update):
        process_dict_updates(self.req_info, batch_update,
            lambda params, _, __: params.extra_args.get("top_n_sigma"))

    def apply(self, logits) -> torch.Tensor:
        # vectorized batch 操作
        ...
```

### Pattern 2: Request-level adapter (继承 AdapterLogitsProcessor)
```python
class WrappedPerReqLogitsProcessor(AdapterLogitsProcessor):
    def new_req_logits_processor(self, params):
        # 返回 per-request Callable 或 None
        ...
```

### Pattern 3: AdapterLogitsProcessor + __init__ override
- Pattern 2 的扩展，在 __init__ 中设置更多参数

**Top-nσ 使用 Pattern 1**，因为算法本身是 batch-vectorized 的，不需要 per-request wrapper。

## 9. 关键文件映射

| 文件 | 行数 | 内容 |
|------|------|------|
| `interface.py` | 107 | ABC 定义 + BatchUpdate 数据类 |
| `state.py` | 166 | LogitsProcessors 容器 + BatchUpdateBuilder |
| `builtin.py` | 333 | MinP + LogitBias + MinTokens + process_dict_updates |
| `__init__.py` | 357 | build_logitsprocs + 加载机制 + AdapterLogitsProcessor |
| `sampler.py` | ~450 | apply_logits_processors + sample 管线 |
| `gpu_model_runner.py` | ~7000 | InputBatch 构建 + logitsprocs 传给 Sampler |

## 10. 设计启示

1. **batch-level > request-level**: 现代 LLM serving 是 batch 化的，per-request callback 是 Python 瓶颈。vLLM V1 把 processor 设计为 batch-level + vectorized，避免 Python 循环。
2. **argmax 分类**: 把 "不影响贪心" 的 processor 推迟到 temperature 后执行，是性能优化 — greedy 请求完全跳过这些 processor。
3. **dict + process_dict_updates**: sparse dict 模式让只处理需要处理的 batch slots，避免全 batch 扫描。Top-nσ 只有部分请求启用时，只处理 `req_info` 中有值的行。
4. **output_tok_ids 引用**: MinTokens 需要知道当前输出长度，通过引用 (而非拷贝) 传 output_tok_ids，processor 总能看到最新状态。
5. **三种加载方式**: entrypoint (插件) / FQCN (动态) / class (直接) — 从生产到实验的完整覆盖。