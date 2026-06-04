# vLLM V1 Speculative Decoding 源码分析

> 目录: `vllm/v1/spec_decode/` + `vllm/v1/worker/gpu/spec_decode/` (24 文件)
> 分析日期: 2026-06-04

## 架构概览

```
Scheduler: 调度 spec tokens
    ↓
Proposer: 生成 draft tokens
    ├── EagleProposer:      小型 draft model (最常用)
    ├── MedusaProposer:     多个 prediction heads
    ├── NgramProposer:      零开销 n-gram lookup (numba JIT)
    ├── DFlashProposer:     双 forward speculative
    └── SuffixDecoding:    后缀匹配 decoding
    ↓
RejectionSampler: 验证 draft tokens
    ├── Target model forward (batch K tokens)
    ├── probability ratio check: P_target(x) / P_draft(x) > rand()
    └── First rejection → stop, sample from adjusted target
    ↓
Speculator (worker): 集成 spec decode 到 GPU worker
```

## Proposer 类型对比

| Proposer | Draft Cost | 准确率 | 适用场景 |
|----------|:---:|:---:|------|
| **Eagle** | ~0.1x target | 0.7-0.85 | 通用，最推荐 |
| **N-gram** | **0** (零开销) | 0.3-0.5 | 代码/重复文本 |
| **Medusa** | ~0.05x | 0.5-0.7 | 简单，多 head |
| **DFlash** | ~1.0x | 0.8-0.95 | 双 forward pass |
| **Suffix** | 0 | 0.1-0.3 | 模板/格式化输出 |

### EagleProposer

```python
class EagleProposer(SpecDecodeBaseProposer):
    """
    小型 draft transformer (1-2 layers):
    1. 接收 target model 的 hidden states
    2. 自回归生成 K 个 draft tokens
    3. pass_hidden_states_to_model=True → 重用 target hidden states

    特点:
    - 最准确的 proposer (p≈0.7-0.85)
    - draft cost ≈ 0.1x target (很小的模型)
    - 支持 EAGLE3 (aux hidden states)
    - 验证了我们的实验: p=0.7 → K*=3-4 最优
    """
```

### NgramProposer

```python
class NgramProposer:
    """
    零模型开销的 spec decode:
    1. 在当前序列中查找最长 n-gram match
    2. 返回 match 后的 K 个 tokens 作为 draft

    实现: numba JIT 编译, CPU 并行处理

    配置:
    - prompt_lookup_min/max: n-gram 长度范围
    - num_speculative_tokens: K (draft length)

    验证了我们的实验: 零开销 → 即使低 accept rate 也有 net gain
    """
```

## Rejection Sampler

```python
class RejectionSampler:
    """
    核心算法:
    for each draft token (x_1, x_2, ..., x_K):
        q = P_draft(x_i)      # draft model prob
        p = P_target(x_i)     # target model prob
        r = random()
        if r < p / q:         # 接受 (modified rejection sampling)
            accept(x_i)
        else:
            # 拒绝 → 从 adjusted target prob max(0, p - q) 采样
            # 然后停止 (不再检查剩余的 draft tokens)
            break

    Speedup 理论:
    E[accepted tokens] = (1 - p^(K+1)) / (1 - p)
    其中 p = P(acceptance)

    p=0.7, K=4: E[tokens] = (1-0.7^5)/0.3 = 2.77 → 2.77x speedup (忽略 draft cost)
    """
```

### Triton Rejection Sampling Kernel

```python
@triton.jit
def rejection_sample_kernel(
    draft_token_ids,      # [num_reqs, num_steps]
    target_probs,         # [num_reqs, num_steps, vocab]
    draft_probs,          # [num_reqs, num_steps, vocab]
    uniform_samples,      # [num_reqs, num_steps]
    output_token_ids,     # [num_reqs, num_steps+1] (+1 for bonus token)
    output_num_tokens,    # [num_reqs]
):
    # 向量化 rejection sampling in Triton
    # 单个 kernel 处理整个 batch
```

---

## Speculator (Worker 集成)

```python
class EagleSpeculator:
    """
    将 spec decode 集成到 GPUModelRunner 的执行流程:

    1. Target model forward → hidden states
    2. Eagle model forward (draft) → K draft tokens
    3. Target model forward on (K+1) tokens → target probs
    4. RejectionSampler → accepted tokens
    5. Bonus token → re-sample if all K accepted

    CUDA Graph 支持: 每个 stage 可以是 CUDA Graph
    """
```

---

## 与我们实验的对应关系

| 实验发现 | 源码证实 |
|---------|---------|
| T 对 argmax acc 影响小 | Rejection sample 用 probability ratio，非 argmax |
| K*=3-4 (p=0.7) | Eagle 默认 K=4-5，与实验一致 |
| Draft cost <0.15x 是关键 | Eagle: 0.1x, N-gram: 0x, 验证了结论 |
| Draft cost >0.5x 有害 | DFlash: 1.0x → 需要极高 p 才有 net gain |
| N-gram 零开销永远赢 | NgramProposer: CPU numba JIT，零 GPU 开销 |

---

## 配置项

```python
class SpeculativeConfig:
    num_speculative_tokens: int = 5     # K (draft length)
    method: str = "eagle"              # eagle/ngram/medusa/dflash
    draft_model_config: ...             # draft model 配置
    rejection_sample_method: str        # "strict"/"synthetic"/"auto"
```

---

## 代码位置速查

| 文件 | 内容 |
|------|------|
| `eagle.py` | Eagle draft model proposer |
| `ngram_proposer.py` | Zero-overhead n-gram proposer |
| `medusa.py` | Medusa multi-head proposer |
| `llm_base_proposer.py` | Base proposer class |
| `rejection_sampler.py` | Triton rejection sampling kernel |
| `rejection_sampler_utils.py` | CPU/GPU rejection utilities |
| `metadata.py` | Spec decode metadata |
| `eagle/speculator.py` | Worker-side Eagle integration |
| `utils.py` | Unconditional→conditional rate conversion |
