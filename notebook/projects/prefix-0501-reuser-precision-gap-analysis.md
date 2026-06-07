# prefix-0501 Reuser Precision Gap Root Cause Analysis

> 2026-06-07 | cos_sim 0.985 vs 0.999999 target → 根因: 单次前向与两遍PS集成不兼容

## 问题

E2E PS精度测试 (`run_ps_e2e.py`) 显示:
- Provider: cos_sim=0.9999, max_diff=0.066 (OK)
- Reuser: cos_sim≈0.985-0.988, min=0.729, max_diff=4.18 (FAIL)
- 配置: n=4, prefix_len=16, suffix_len=32, tp=4, bf16

## 根因: 单次前向 + 两遍PS集成 = 不兼容

### 两遍PS架构 (`VerlQwen3_6Integration`)

`VerlQwen3_6Integration` 为 verl 的 `ParallelQwen3_6AttentionRmPad` 设计了**两遍(prefix/suffix) PS**:

- **Prefix pass** (store无KV → `ctx.store.contains(slot_id)`=False):
  1. 计算QKV + QK norm + partial RoPE (所有token一起处理)
  2. 存储KV到 `prefix_store` (4D格式, 便于后续expand)
  3. 运行 `flash_attn_varlen_func` 正常因果注意力
  4. 返回注意力输出 (所有token的attention结果)

- **Suffix pass** (store有KV → `ctx.store.contains(slot_id)`=True):
  1. 计算suffix QKV + QK norm
  2. 生成prefix_len+suffix_len范围的cos/sin, 切片suffix部分
  3. 对suffix tokens应用partial RoPE
  4. 加载存储的prefix KV, expand到N序列, concat with suffix KV
  5. 构建cu_seqlens: Q=suffix_len, KV=prefix_len+suffix_len
  6. 运行 `flash_attn_varlen_func(causal=True)` → **block-causal mask** (Q短于KV时, causal=True自动产生prefix全可见+suffix因果的mask)
  7. 返回suffix-only attention输出

### `run_ps_e2e.py` 的问题

该脚本只做**一次模型前向**:

```python
output_ps = model(input_ids=input_ids_ps, attention_mask=attention_mask_ps, position_ids=position_ids_ps)
```

在单次前向中, 每层attention只被调用一次. `slot_id` 包含 `layer_id`, 所以:
- Layer 0: store.contains(slot_id_0)=False → **prefix pass** → 存储KV → 返回正常attention输出
- Layer 1: store.contains(slot_id_1)=False → **prefix pass** → 存储KV → 返回正常attention输出
- ...每层都是prefix pass, suffix pass永远不执行!

结果: Reuser的suffix tokens在每层attention中都没有prefix KV注入 → 等于无prefix的因果attention → 与正常forward(有prefix的因果attention)不等价 → cos_sim 0.985.

### `flash_attn_varlen_func(causal=True)` 的block-causal语义

当 Q_len < KV_len 且 `causal=True`:
- Q[0] 可见 KV[0..prefix_len] (所有prefix + 第1个suffix) ✓
- Q[i] 可见 KV[0..prefix_len+i+1] (所有prefix + 前i+1个suffix) ✓
- 这就是block-causal mask!

所以suffix pass的 `causal=True` 是正确的. 但问题不是mask实现, 而是**suffix pass从未执行**.

## 正确的两遍方法: `run_ps_e2e_twopass_v2.py`

该脚本实现了正确的两遍:

1. **Pass 1 (prefix)**: 只处理provider的prefix tokens → 所有层存储KV + DeltaNet states
2. **Pass 2 (suffix)**: 处理所有序列的suffix tokens → 所有层加载并注入KV + states

关键: 两遍是**两次独立的前向传播**, 不是一次前向内的两个调用. Pass 1和Pass 2分别遍历所有层.

DeltaNet层还需要:
- recurrent_state injection (initial_state for chunked forward)
- conv1d overlap context (最后3个prefix hidden_states)
- chunk boundary alignment (prefix_len >= chunk_size=64)

## 替代方案: 单遍PS (`MegatronAttentionIntegration` + `TorchReferenceBackend`)

`MegatronAttentionIntegration` 支持单遍PS:
- Patch `megatron.core.transformer.attention.SelfAttention.forward`
- 在同一次前向中, 逐序列处理:
  - Provider: 正常attention, 存储KV
  - Reuser: 加载provider KV, expand, block-causal attention (`_attention_row()` + SDPA math backend)
- `_causal_q_kv_mask()` 正确实现block-causal mask

**局限**: 只patch Megatron core的SelfAttention, 不支持verl特有的模型类(如`ParallelQwen3_6AttentionRmPad`). 对Qwen3.6-27B需要用Megatron core的model class.

## 解决方案

| 方案 | 优点 | 缺点 |
|------|------|------|
| **两遍PS** (twopass_v2) | DeltaNet+全attention完整支持, 已验证架构 | 两次前向开销, prefix_len>=64(chunk boundary), 手动逐层遍历 |
| **单遍PS** (MegatronIntegration) | 一次前向, 更简单 | 不支持DeltaNet state injection, 不支持verl model classes, SDPA math backend慢于flash_attn |
| **单遍+flash_attn** (改进TorchReferenceBackend) | 一次前向+flash_attn加速 | 需要实现packed flash_attn block-causal, DeltaNet state injection仍需两遍 |

**推荐**: 两遍PS (twopass_v2) 是当前唯一完整验证的方案. 单遍方案需要额外开发:
1. `TorchReferenceBackend.attention()` 改用 `flash_attn_varlen_func` (prefix+suffix separate calls)
2. DeltaNet state injection 在单遍中不可行 (需要前序层的结果来注入)

## 关键代码路径

| 文件 | 角色 |
|------|------|
| `prefix-sharing/integrations/verl_attention.py` | 两遍PS: patch ParallelQwen3_6AttentionRmPad + ParallelQwen3_6GatedDeltaNetRmPad |
| `prefix-sharing/integrations/megatron_attention.py` | 单遍PS: patch Megatron SelfAttention.forward |
| `prefix-sharing/backends/torch_ref.py` | 单遍backend: `_attention_row()` + SDPA math |
| `prefix-sharing/backends/packed_layout.py` | PackedBatchLayout: THD格式, position_ids |
| `prefix-sharing/integrations/megatron_runtime.py` | Runtime hook: maybe_run_prefix_sharing_attention |
| `prefix-0501/scripts/run_ps_e2e.py` | **有bug**: 单次前向 + 两遍集成 = 不兼容 |
| `prefix-0501/scripts/run_ps_e2e_twopass_v2.py` | **正确**: 两遍前向, 逐层遍历 |

## 下一步

1. 运行 `run_ps_e2e_twopass_v2.py` 验证两遍PS精度 (prefix_len=64, suffix_len=64)
2. 如果精度OK, 将两遍PS作为production path
3. 考虑单遍PS优化: 在 `TorchReferenceBackend.attention()` 中改用 `flash_attn_varlen_func` 实现block-causal
4. DeltaNet state injection 需要两遍 — 这是根本限制 (第N层的state依赖第N-1层的结果)

Sources:
- E2E results: ~/rollout-prefix/ps_e2e_results.json (n=4, prefix=16, suffix=32, tp=4)
- Two-pass script: prefix-0501/scripts/run_ps_e2e_twopass_v2.py (687 lines)
- VerlQwen3_6Integration: prefix-sharing/integrations/verl_attention.py (two-pass design)
- MegatronAttentionIntegration: prefix-sharing/integrations/megatron_attention.py (single-pass design)