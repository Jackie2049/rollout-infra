# Candidate: process_validation_metrics 崩在 sparse reward_extra_info 的 None

> Issue: [volcengine/verl#6830](https://github.com/volcengine/verl/issues/6830)
> 标签：**bug + help wanted**（官方邀请社区修复）
> 状态：`verified-static`（根因三重确认；**待本地复现 + patch**）
> 首次发现：2026-08-04
> **竞争风险：极低** — 0 评论、0 assignee、`help wanted`（与 #7213 作者自认领不同）

## 0. 为什么这是本轮最佳候选

| 维度 | 评估 |
|---|---|
| 认领情况 | 0 评论、0 assignee、作者 giladfrid009 **没有说"我来修"** → 无人认领 |
| 官方态度 | 带 `help wanted` 标签 → maintainer 明确希望社区修 |
| 根因清晰度 | 报告者给了**完整根因 + 最小触发 + 修复指引 + workaround 为何不行** |
| 范围 | 自包含（metric_utils.py 一处 guard），不碰训练主路径 |
| 契合度 | 命中 RL 训练 + 指标聚合，我们能本地复现 |

## 1. Bug 现象

reward 函数对**部分** validation 样本才发某个 `reward_extra_info` key（sparse key）时：
- `_validate` 为了让所有样本共享同一 schema，给缺失样本的该 key **填 None**
- `process_validation_metrics` 聚合时对该 key 的 per-uid 组做 `np.mean([..., None, ...])`
- → `TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'`

**任何**条件性发 extra-info key 的 reward 函数，只要有一个样本没发该 key，validation 就崩。

## 2. 根因（代码追踪确认）

**两半各自合理，但合起来不兼容：**

**① Null-fill（填 None 统一 schema）** — `_validate` / dump 路径
```python
# 缺失样本的 sparse key 填 None
for key in reward_extra_infos_dict:
    if key != "reward" and key not in reward_extra_info:
        reward_extra_infos_dict[key].append(None)
```

**② 聚合不能容忍 None** — `verl/trainer/ppo/metric_utils.py:982-986`
```python
for var_name, var_vals in var2vals.items():
    # skip empty or string values
    if not var_vals or isinstance(var_vals[0], str):   # ← 不跳过 [None]
        continue
    n_resps = len(var_vals)
    metric = {f"mean@{n_resps}": float(np_mean(var_vals))}   # np.mean([None]) → TypeError
```

一个 uid 若所有 rollout 都没发该 key，`var_vals == [None]` → 崩。

**None 来源**：`verl/trainer/ppo/reward.py:163` `extract_reward` 透传 `reward_extra_keys` 的值；reward manager 对 sparse key 填 None。

## 3. 期望行为（issue 正文）

报告者明确排除了两个错误修法：
- ❌ **数值 sentinel**（如 -1）填缺失：会**偏斜 mean**
- ❌ **`nan`** 填缺失：verl 用 `np.mean`（非 `np.nanmean`），nan 会传染整个聚合

→ 正确修法是**在聚合前过滤掉 None**，只对有值的样本求统计。

## 4. Patch 设计（draft）

核心：在 `process_validation_metrics` 聚合前，对每个 per-uid 组**剔除 None**；若整组都是 None 则跳过该 var（无可统计值）。

```python
# metric_utils.py 聚合循环内
for var_name, var_vals in var2vals.items():
    # NEW: 剔除 None（sparse extra-info 的 null-fill），只统计有值样本
    var_vals = [v for v in var_vals if v is not None]
    # skip empty or string values
    if not var_vals or isinstance(var_vals[0], str):
        continue
    ...
```

注意点：
- 过滤后 `n_resps = len(var_vals)` 会反映**实际有值的样本数**（`mean@k`，k 可能 < 组大小）——这正是期望语义（对 sparse 指标，只统计 emit 了它的样本）。
- bootstrap / maj 路径同样基于过滤后的 `var_vals`，None 不再进入 `np.max/np.min/zip`。
- `pred` 相关（majority vote）的 `zip(var_vals, pred_vals, strict=True)` 需确保 None 过滤对 val/pred 对齐——**这里要谨慎**：若只过滤 val 的 None，需同步过滤对应 pred，否则 `strict=True` 会因长度不等报错。实现时应同时剔除两者中 val 为 None 的位置。

## 5. 待办（Phase ④ 下一步）

- [ ] **本地复现**：写最小脚本构造 `infos_dict` 含 None 的 sparse key → 调 `process_validation_metrics` → 确认 TypeError
- [ ] **应用 patch** → 确认不再崩、且有值样本统计正确（mean 不被 None 偏斜）
- [ ] **边界**：整组全 None（该 var 跳过）、val/pred 对齐过滤、`n_resps` 语义
- [ ] **回归**：非 sparse（所有样本都有值）时行为不变
- [ ] **补单测**：`tests/` 下加 sparse-key 的 process_validation_metrics 用例
- [ ] **（对外，需你同意）** 提 PR

## 6. 评估

- I=4（validation 直接崩，阻断任何 sparse 指标的 reward 函数；正确性相关）
- C=2（自包含 guard + 单测；pred 对齐需小心）
- A=5（`help wanted` + 无人认领 + 报告者给了修复指引 → maintainer 想要）
- F=4（RL 指标聚合，本地可复现）
- **性价比 = (4×5×4)/2 = 40** —— 本轮最高，建议优先动工（本地 patch + 复现，不对外）
