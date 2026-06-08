# LLM Serving Scheduler Architecture: vLLM V1 vs SGLang Comparison
> 2026-06-08 | 调度是serving的心脏 — 从token budget到prefix caching的5层对比
> 基于: vllm/v1/core/sched/scheduler.py (~2400行), SGLang scheduler + RadixAttention
> 关联: vllm-v1-scheduler-deep-read.md, sglang-radix-attention.md, inference-serving.md

## 0. 核心定律: 调度 = 在有限资源下最大化吞吐

```
Serving调度问题:
  → GPU资源有限: KV cache内存 + compute capacity
  → 请求源源不断: prefill(长输入) + decode(逐token输出)
  → 目标: 最大化throughput (tok/s) + 最小化latency (TTFT + ITL)

  等式:
    throughput = total_tokens_per_step / step_time
    total_tokens = decode_tokens + prefill_tokens
    step_time ≈ max(prefill_time, decode_time) → 需要平衡!

  → decode优先(已有KV → memory-bound → constant time)
  → prefill填充剩余budget(compute-bound → 需要足够compute)
  → 两者混合 → 一个step内同时做prefill和decode → continuous batching!
```

## 1. Token Budget管理

### 1.1 vLLM V1: Unified Token Budget

```
vLLM V1 scheduler:
  → 每步有固定token budget = max_num_batched_tokens
  → RUNNING请求(decode + chunked prefill)优先分配
  → WAITING请求(prefill)填充剩余budget
  → 无传统"phase"分离! 统一token计数 → num_computed_tokens vs total_tokens

  调度流程:
    1. RUNNING: decode优先 → 分配1 token per decode request
    2. RUNNING: chunked prefill → 分配chunk_tokens (≤long_prefill_threshold)
    3. WAITING: 新prefill → 从剩余budget分配
    4. 如果KV空间不足 → preempt(优先级或FCFS)

  Budget示例(B=32, S=512, 7B GQA-5):
    → max_batched_tokens=512 → 32 decode×1=32 + 480 prefill tokens
    → 但prefill需要完整compute → chunk到480 tokens per step
    → decode ~0.22ms + prefill ~480/2560→~ms → 一步完成!
```

### 1.2 SGLang: Token-level Budget

```
SGLang scheduler:
  → 类似token budget → rem_total_tokens + rem_input_tokens
  → PrefillAdder控制准入 → 计算剩余token容量
  → 更细粒度: 区分total_tokens(所有)和input_tokens(新prefill部分)

  关键差异:
    → vLLM: scheduler直接分配tokens → step-level调度
    → SGLang: PrefillAdder逐步填充 → 更渐进的budget消耗
    → → SGLang在prefill密集时更保守 → 避免decode延迟波动!
```

## 2. Scheduling Policy

### 2.1 优先级 vs FCFS

```
vLLM V1:
  → 默认FCFS: 先到先服务 → 简单公平
  → 支持priority: 用户可设优先级 → 高优先级请求优先调度
  → Preempt策略:
    → FCFS: preempt最新请求(pop) → 最老请求最安全
    → Priority: preempt最低优先级 → 保护VIP请求
  → 问题: preempt = recomputation → 重新计算prefill → 延迟增加

SGLang:
  → 默认FCFS → 与vLLM相同
  → 无显式priority支持 → 但可以通过RadixAttention间接实现
  → prefix cache hit高的请求 → "更快" → 间接优先级!

RTX 4090实测:
  → recomputation preemption: 短请求快(~ms), 长请求慢(~100ms)
  → swap preemption: PCIe带宽12GB/s → swap慢 → RTX 4090用recomputation
  → → vLLM/SGLang都默认recomputation on RTX 4090
```

## 3. Prefix Caching: BlockHash vs RadixTree

### 3.1 vLLM: BlockHashToBlockMap (1:N Hash)

```
vLLM prefix caching:
  → block_size=16 → 每个block=16 tokens
  → hash = hash(token_ids[:16]) → BlockHash
  → BlockHashToBlockMap: 1:N mapping → 一个hash可映射多个block
  → hash chain: 不同请求可能产生相同hash → chain of blocks
  → FreeKVCacheBlockQueue: O(1) LRU驱逐 → 释放最久未使用block

  优势:
    → 简单 → hash查找O(1) → 快速匹配
    → block-level → 固定大小 → 管理简单
    → 1:N → 允许不同请求共享相同prefix block

  局限:
    → block_size=16 → 最细粒度16 tokens → 不能共享更短prefix
    → 1:N mapping → hash冲突可能 → 需要遍历chain
    → 无树结构 → 不能做"分裂"操作
```

### 3.2 SGLang: RadixAttention (基数树)

```
SGLang prefix caching:
  → RadixTree → 边标记为可变长度token序列 → 不固定block_size!
  → TreeNode → value=KV池索引 → children=子节点
  → match_prefix() → 查找最长匹配 → 精确到任意token数
  → _split_node() → 节点分裂 → 共享prefix→分裂→保留各自unique部分
  → 7种驱逐策略: LRU/LFU/FIFO/Age/AccessCount/Mixed/Gradient

  优势:
    → 更细粒度 → 可以共享任意长度的prefix (不只16 tokens)
    → 树结构 → 自然分裂 → 请求共享system prompt时精确匹配
    → 多驱逐策略 → 更灵活 → 可以根据负载选择

  局限:
    → 树查找 → O(L) where L=path length → 理论上比hash慢
    → 但实际L很短(system prompt+few shots) → 实际不慢!
    → 树分裂 → 需要clone indices → 有微小overhead

  实际性能:
    → system prompt共享: 256 tokens prefix
    → vLLM: 16 blocks=256/16 → 16 hash查找 → O(blocks)
    → SGLang: 1 tree match → O(1) → 更快!
    → → SGLang在system prompt场景更高效
```

### 3.3 实际prefix sharing效果

```
RTX 4090实测(7B GQA-5, B=16):
  → Prefix sharing throughput: 1.59x (training)
  → prefix=512 tokens → ~1.5x decode加速(减少KV读取)
  → → prefix caching对吞吐有实际价值

  关键: prefix sharing = 减少 KV读取 → decode memory-bound → 省带宽 = 省延迟!
  → 共享256 tokens → 每个decode请求少读256×5×128×2=327KB → 显著!
```

## 4. Continuous Batching实现

### 4.1 vLLM: Step-level Batching

```
vLLM V1 batching:
  → EngineCore.step() → 一次调度 → 同时prefill+decode
  → SchedulerOutput → 传递给Executor → 执行一个step
  → 一步内:
    → decode requests: 1 token each → B_decode tokens
    → prefill requests: chunk_tokens → B_prefill×chunk tokens
    → 总token ≤ max_batched_tokens → budget限制

  GPU执行:
    → 所有token合并为一个大batch → 一次forward pass!
    → → decode部分: attention KV lookup → memory-bound
    → → prefill部分: full attention → compute-bound
    → → 两者share GPU → compute+memory同时利用!

  我们之前simulator实测: 76K-171K tok/s → continuous batching有效!
```

### 4.2 SGLang: Merge-based Batching

```
SGLang batching:
  → 更保守的准入 → PrefillAdder逐步加入
  → 每步: running requests(decode) + new prefill(如果budget允许)
  → 区别: SGLang更强调"不让prefill干扰decode"
  → → prefill添加到step时 → 确保decode请求不受影响
  → → 如果budget不足 → 不添加新prefill → decode优先完成

  对比:
    → vLLM: 更激进 → 填满budget → throughput最大化
    → SGLang: 更保守 → 保护decode延迟 → ITL更稳定
    → → vLLM throughput更高但ITL波动更大
    → → SGLang throughput略低但ITL更稳定
```

## 5. 内存管理

### 5.1 KV Cache分配

```
vLLM V1:
  → Paged KV cache → block_size=16 → 按需分配
  → BlockAllocator → free blocks池 → 请求时分配 → 完成时回收
  → 最大KV blocks = (HBM - weights) / block_size_bytes
  → → 7B GQA-5 BF16: (24GB-14GB)/81.92KB ≈ 122K blocks → 2M tokens capacity

SGLang:
  → Token-to-KV pool → 更细粒度 → 每个token一个slot
  → token_to_kv_pool_allocator → 管理GPU KV slots
  → req_to_token_pool → 请求到slot的映射 → 间接层

  关键差异:
    → vLLM: block-level → 16 tokens一个block → 最小分配16 tokens
    → SGLang: token-level → 更精确 → 但管理更复杂
    → → vLLM碎片: 最多15 tokens/block浪费 → (15/16)=6.25%碎片
    → → SGLang: 0碎片 → 但需要更复杂的索引管理
```

### 5.2 Preemption: Recomputation vs Swap

```
当KV cache满了 → 需要preempt请求:

  Recomputation (两者默认):
    → 释放被preempt请求的KV cache
    → 被preempt请求重新进入WAITING → 重新prefill
    → → short requests: 快(~ms) → long requests: 慢(~100ms)
    → → RTX 4090推荐recomputation (PCIe swap太慢)

  Swap (vLLM支持, SGLang不常用):
    → 将KV cache复制到CPU pinned memory
    → 需要时从CPU复制回GPU → PCIe瓶颈!
    → RTX 4090 PCIe 12GB/s → swap 100MB KV ≈ 8ms → 可以接受
    → → 但recomputation通常更快(短请求) → RTX 4090默认recomputation

  RTX 4090决策:
    → 短请求(S<512): recomputation → ~1-5ms → 快!
    → 长请求(S>2048): recomputation → ~10-100ms → 可能慢!
    → → 最佳: 限制running requests → 避免preempt → 控制并发数
```

## 6. Speculative Decoding调度

```
vLLM V1 Speculative Decoding调度:
  → num_tokens_with_spec = num_computed_tokens + spec_tokens
  → → spec_tokens = draft模型提议的token数(如5)
  → → 调度器不区分spec tokens和real tokens → 统一处理!
  → → 目标模型验证 → 接受部分 → 拒绝部分 → 修正num_computed_tokens

SGLang Speculative Decoding:
  → EAGLE推测解码 → tree-based speculation
  → RadixAttention + is_bigram → 支持draft tree的prefix sharing
  → → draft tree的共享prefix可以cached → 更高效!

RTX 4090实测:
  → 未训练draft: 接受率18% → 反而更慢 → 不推荐!
  → ngram/Eagle: 接受率30-80% → 1.5-5x理论可行
  → → RTX 4090推荐ngram(零额外模型)或Eagle(训练draft head)
```

## 7. 决策树: vLLM vs SGLang选择

```
选择依据:

| 场景 | vLLM V1 | SGLang | 原因 |
|------|---------|--------|------|
| 最高throughput | ✅ | ❌ | vLLM更激进batch → 更多tok/s |
| 最稳定ITL | ❌ | ✅ | SGLang保守调度 → ITL波动小 |
| system prompt共享 | ❌ | ✅ | RadixTree更细粒度 |
| 简单部署 | ✅ | ❌ | vLLM更成熟+更多社区支持 |
| 多模态 | ✅ | ❌ | vLLM V1支持vision/audio |
| LoRA serving | ✅ | ❌ | vLLM V1支持multi-LoRA |
| MoE serving | ✅(V1 DP) | ✅ | vLLM DPEngineCoreProc |
| 生产级 | ✅ | ✅ | 两者都可用于生产 |

RTX 4090推荐:
  → 单GPU推理: vLLM V1 (最成熟+FlashInfer backend)
  → prefix-heavy场景: SGLang (RadixAttention更高效)
  → 稳定延迟需求: SGLang (ITL更稳定)
  → 通用推理: vLLM V1 → 更简单+更成熟+更多优化

关键洞察: 调度器选择影响throughput和latency → 但不是唯一因素!
  → attention backend(FlashInfer)更重要 → 15.72x加速!
  → 量化(INT4+INT8KV)更重要 → 75%+50%内存省!
  → → 调度器优化是锦上添花 → 硬件优化是根本!
```

---

**Sources**:
- vLLM V1 scheduler: `vllm/v1/core/sched/scheduler.py`
- SGLang scheduler: `sglang/srt/layers/radix_attention.py`
- 详细见: vllm-v1-scheduler-deep-read.md, sglang-radix-attention.md

**Related notes**: flashinfer-attention-deep-dive.md, inference-cost-analysis.md, kv-cache-management-deep-dive.md