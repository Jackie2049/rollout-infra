# verl PrefixGrouper 源码深度阅读

> 训练侧前缀复用的核心实现 — 减少共享 prompt 的重复计算

## 1. 核心问题：GRPO 中的前缀冗余

### 1.1 问题背景

在 GRPO/PPO 训练中，同一个 prompt（问题/指令）会生成多个 response（response）：

```
Prompt: "请解释量子计算"
  → Response 1: "量子计算是..."
  → Response 2: "量子计算利用..."
  → Response 3: "量子计算的核心..."
  → Response 4: "量子计算与传统..."
```

标准训练流程中，这 4 个 (prompt, response) 对会被独立处理：
```
GPU 计算:
  [prompt + response1] 的完整前向/反向 ← prompt 部分重复计算 4 次！
  [prompt + response2] 的完整前向/反向
  [prompt + response3] 的完整前向/反向
  [prompt + response4] 的完整前向/反向
```

**问题**：prompt 部分被重复计算了 G 次（G = 每个prompt的response数），造成大量浪费。

### 1.2 PrefixGrouper 的解决方案

将冗余的自注意力分解为：
1. **Prefix self-attention**：只计算一次 prompt 的 attention
2. **Suffix concat-attention**：所有 response 与已计算的 prefix KV 一起做 attention

```
优化后:
  1. 计算 prefix 的 KV Cache (一次)
  2. 所有 suffix 共享 prefix KV Cache
  3. 各 suffix 独立计算自己的 attention

计算量从 O(G × L_prefix²) 降到 O(L_prefix² + G × L_suffix²)
```

## 2. 源码分析

### 2.1 文件位置

```
verl/verl/trainer/ppo/prefix_grouper_utils.py  ← verl 集成层
external: pip install prefix_grouper             ← 核心库
```

核心库来自 https://github.com/johncaged/PrefixGrouper

### 2.2 build_pg_from_micro_batch()

```python
def build_pg_from_micro_batch(micro_batch, pad_token_id, padding_mode):
    """从 micro-batch 构建 PrefixGrouper"""
    prompts = micro_batch["prompts"]         # [bs, prompt_len]
    responses = micro_batch["responses"]     # [bs, response_len]
    response_mask = micro_batch["response_mask"]
    uids = micro_batch["uid"]               # 每个 sample 的 prompt ID

    # 1. 按 uid 分组 — 相同 uid = 共享相同 prompt
    group_sizes = []
    cur = 1
    for i in range(1, bs):
        if uids[i] == uids[i - 1]:
            cur += 1       # 同一组，累积
        else:
            group_sizes.append(cur)
            cur = 1        # 新组
    group_sizes.append(cur)

    # 2. 提取每组的 prefix（只取第一行，因为同组 prompt 相同）
    prefix_indices = torch.tensor([0, group_sizes[0], group_sizes[0]+group_sizes[1], ...])
    prefix_ids = prompts.index_select(0, prefix_indices)  # 每组一行
    prefix_mask = prefix_ids.ne(pad_token_id)

    # 3. 构建 PrefixGrouper 对象
    prefix_grouper = PrefixGrouper.from_ungrouped_masks(
        prefix_mask=prefix_mask,       # [num_groups, prefix_len]
        suffix_mask=response_mask,     # [bs, response_len]
        group_sizes=group_sizes,       # [num_groups]
        padding_mode=padding_mode,
    )

    # 4. 拼接输入: [prefix + suffix1 + suffix2 + ...]
    concat_input_ids = prefix_grouper.concat_input(
        prefix_ids, prefix_mask, responses, response_mask
    )

    # 5. 构建 position_ids
    position_ids = build_position_ids_for_prefix_grouper(prefix_grouper)

    return prefix_grouper, concat_input_ids, attention_mask, position_ids, ...
```

### 2.3 build_position_ids_for_prefix_grouper()

```python
def build_position_ids_for_prefix_grouper(prefix_grouper):
    """为分组后的输入构建 position IDs"""
    position_ids = torch.zeros(num_samples, max_len, ...)

    for i, group in enumerate(prefix_grouper.group_info):
        prefix_len = group.prefix_len

        # Prefix 部分: 正常位置 [0, 1, 2, ..., prefix_len-1]
        position_ids[i, :prefix_len] = torch.arange(prefix_len)

        # 每个 suffix: 从 prefix_len 开始重新编号
        cur_pos = prefix_len
        for suffix_len in group.suffix_lens:
            position_ids[i, cur_pos:cur_pos+suffix_len] = torch.arange(
                prefix_len, prefix_len + suffix_len
            )
            cur_pos += suffix_len

    return position_ids
```

**关键**：每个 suffix 的 position_ids 从 `prefix_len` 开始，确保因果性。

### 2.4 pg_forward()

```python
def pg_forward(model, prefix_grouper, concat_input_ids, ...):
    """使用 PrefixGrouper 的前向传播"""
    # 模型前向 — 传入 prefix_grouper 参数
    logits = model(
        input_ids=concat_input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        prefix_grouper=prefix_grouper,  ← 关键参数
    ).logits

    # 分割输出: prefix 部分 + suffix 部分
    prefix_out, prefix_mask, suffix_out_raw, suffix_mask_raw = \
        prefix_grouper.split_output(logits, include_prefix_last=1)

    # 计算 log probs
    suffix_out = suffix_out_raw[:, :-1].float()
    log_probs = logprobs_from_logits(suffix_out, completion_ids_right)

    return log_probs, entropy, suffix_mask
```

### 2.5 Attention Monkey Patch

```python
# verl/verl/models/transformers/monkey_patch.py
# 自动 patch attention 函数支持 prefix_grouper 参数

# 当 use_prefix_grouper=True 时:
# 1. Patch transformers 的 ALL_ATTENTION_FUNCTIONS
# 2. 每个 attention 函数检查是否有 prefix_grouper 参数
# 3. 有 → 使用 PrefixGrouper 的优化 attention
# 4. 无 → 调用原始 attention（兼容模式）
```

## 3. 数据流可视化

```
原始 GRPO (G=4, prompt="量子计算", response 各不同):
  Input: [prompt + resp1]
         [prompt + resp2]    ← prompt 重复 4 次
         [prompt + resp3]
         [prompt + resp4]

  Attention: 4 次独立计算 prompt 部分的 attention

PrefixGrouper 优化后:
  Input (concat): [prefix | resp1 | resp2 | resp3 | resp4]

  Layer 1 Attention:
    1. Prefix self-attention: Q_p, K_p, V_p → KV_cache_prefix
    2. Suffix attention:
       resp1: Q_1 × concat(K_p, K_1) → 复用 KV_cache_prefix
       resp2: Q_2 × concat(K_p, K_2) → 复用 KV_cache_prefix
       resp3: Q_3 × concat(K_p, K_3) → 复用 KV_cache_prefix
       resp4: Q_4 × concat(K_p, K_4) → 复用 KV_cache_prefix

  结果: prefix attention 只计算 1 次 (而非 G 次)
```

## 4. 性能数据

| Context Length | Metric | PG | No PG | Speedup |
|----------------|--------|-----|-------|---------|
| 4K | old_log_prob | 1.31s | 1.70s | **1.30x** |
| 4K | update_actor | 4.80s | 6.07s | **1.26x** |
| 8K | old_log_prob | 1.69s | 2.63s | **1.56x** |
| 8K | update_actor | 5.98s | 10.18s | **1.70x** |

**结论**：prompt 越长，加速越明显（prefix 冗余计算占比更大）。

## 5. 使用限制

```
当前限制:
  - 仅支持 FSDP worker (不支持 Megatron)
  - 不兼容 use_dynamic_bsz=True
  - 不兼容 use_remove_padding=True (Flash Attention V2 variable length)
  - 不兼容 use_fused_kernels=True
  - 不兼容 Ulysses SP 和 Ring Attention
  - balance_batch=True 需要 batch_size % (world_size * rollout.n) == 0
```

## 6. 与 prefix-sharing 项目的关系

PrefixGrouper 是训练侧的 prefix 复用，与用户的前缀共享项目直接相关：

| 层级 | 技术 | 作用 |
|------|------|------|
| 推理层 | vLLM prefix caching | 复用 KV Cache，加速 rollout 生成 |
| 路由层 | sticky session | 确保相同 prefix 路由到同一 GPU |
| 训练层 | PrefixGrouper | 复用 prefix attention，减少训练重复计算 |

**可能的研究方向**：
- PrefixGrouper + Megatron TP/PP 的兼容
- PrefixGrouper + longer prefix（如 system prompt + multi-turn context）
- 三级缓存的联合优化策略

## 7. 学习要点

1. **核心思想**：将 prefix attention 计算一次，suffix 共享 prefix KV Cache
2. **分组关键**：通过 uid 标识相同 prompt，按 uid 分组
3. **position_ids 特殊处理**：每个 suffix 从 prefix_len 开始编号
4. **monkey patch 方式**：无需修改模型代码，自动 patch attention 函数
5. **长 prompt 收益更大**：prefix 冗余占比更高
6. **外部库**：`prefix_grouper` 是独立包，可独立使用

## 参考

- [PrefixGrouper GitHub](https://github.com/johncaged/PrefixGrouper)
- [verl prefix_grouper example](https://github.com/volcengine/verl/tree/main/examples/prefix_grouper)
