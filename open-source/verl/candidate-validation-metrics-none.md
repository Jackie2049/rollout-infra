# Candidate: process_validation_metrics 崩在 sparse reward_extra_info 的 None

> Issue: [volcengine/verl#6830](https://github.com/volcengine/verl/issues/6830)
> 标签：**bug + help wanted**（官方邀请社区修复）
> 状态：`verified-works`（根因确认 + 本地复现 TypeError + patch 5 项测试全过）
> 首次发现：2026-08-04
> **竞争风险：极低** — 0 评论、0 assignee、`help wanted`（与 #7213 作者自认领不同）

## ⚠️ verl 贡献政策（CLAUDE.md，提交前必读）

- **不允许纯 code-agent PR**：必须人类提交者端到端理解并辩护每一行 → 正好契合"你审阅后一起提交"
- **AI-assisted PR 描述必须包含**：为何不重复现有 PR / 跑了哪些测试及结果 / 明确声明用了 AI 协助
- **不允许低价值 busywork PR**（单行 typo/风格）：机械清理只有捆绑实质工作才可
- **提交前先查重**：`gh issue view <n> --comments` + `gh pr list --search <n> in:body` 确认没有别人已开 PR 修同一个
- commit message 用 trailer：`Co-authored-by: Claude` + `Signed-off-by: <你的名字>`

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

## 4. Patch（已实现 + 本地验证通过）

改动位置：`verl/trainer/ppo/metric_utils.py`，`process_validation_metrics` 的聚合循环开头。核心：先过滤 None（剔除 sparse key 的 null-fill），再让 pred 按位置对齐过滤，最后才进 skip-filter 和统计。

```python
for var_name, var_vals in var2vals.items():
    # Drop null-filled values from sparse reward_extra_info keys.
    orig_var_vals = var_vals
    var_vals = [v for v in var_vals if v is not None]
    if var_name != "pred" and var2vals.get("pred") is not None:
        # Keep pred aligned with the filtered values for the maj@N vote.
        var2vals["pred"] = [p for v, p in zip(orig_var_vals, var2vals["pred"], strict=True) if v is not None]
    # skip empty or string values
    if not var_vals or isinstance(var_vals[0], str):
        continue
    pred_vals = var2vals.get("pred")
    has_pred = pred_vals is not None
    n_resps = len(var_vals)
    metric = {f"mean@{n_resps}": float(np_mean(var_vals))}
    ...
```

关键设计决策：
- **过滤 None 而非 sentinel/nan**：符合 issue 期望（sentinel 偏斜 mean，nan 用 np.mean 会传染）
- **pred 按位置对齐**（`zip(orig_var_vals, pred, strict=True)`）：避免 val 过滤后 maj-vote 的 val/pred 错位；`strict=True` 让真实长度不齐尽早暴露而非静默截断
- **删除旧的外层 `pred_vals/has_pred`**：移到过滤后重算，否则旧 has_pred 与过滤后 pred 不一致
- **n_resps 语义**：过滤后 = 实际有值样本数（`mean@k`，k ≤ 组大小），正是 sparse 指标的正确语义

## 5. 验证结果（4090 服务器 verl-pr env，torch 2.6.0，PYTHONPATH 指向最新 verl_main）

| 用例 | 结果 |
|---|---|
| 1. sparse None (n=1) — 原崩溃 | ✅ 不崩，`cp: mean@1=0.85=(0.9+0.8)/2`（None 剔除） |
| 2. sparse + pred（maj 路径）— 原崩溃 | ✅ 不崩，maj@2 正常，val/pred 对齐 |
| 3. 全 None 组 | ✅ `cp` 被跳过（不进结果），无 crash |
| 4. dense 回归（无 None） | ✅ `cp: mean@2=0.75`（4 值全算），行为不变 |
| 5. 混合 dense+sparse | ✅ dense 用全样本、sparse 只用有值样本，互不干扰 |

**复现确认（patch 前）**：
- n=1 组：`TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'`（与 issue 一致）
- 含 pred 组：`TypeError: ... for +: 'float' and 'NoneType'`（maj 路径也崩）

## 6. 待办（PR 前）

- [x] **补 pytest 单测**：在 `tests/trainer/ppo/test_metric_utils_on_cpu.py` 的 `TestProcessValidationMetrics` 追加 3 个用例（sparse None / 全 None 跳过 / sparse+pred 对齐）
- [x] **跑全套测试**：`pytest test_metric_utils_on_cpu.py` → **39 passed**（3 新 + 36 现有无回归），4.28s
- [ ] **pre-commit**：`uv pip install pre-commit hydra-core && pre-commit install` 后跑过 lint（待本地或服务器跑）
- [ ] **查重**（提交前）：`gh pr list --search "6830 in:body"` + `--search "process_validation_metrics"` 确认没人已开 PR
- [ ] **（对外，需你同意）** 把 patch 放到你的 fork（`Jackie2049/verl`）开 PR，PR 描述按 CLAUDE.md 要求写（为何不重复/测试结果/声明 AI 协助）
- [x] diff 目前在本机 `verl-pr/verl_main/`，未提交到任何远端（符合"只在 fork 建 PR"规则）

## 7. 评估

- I=4（validation 直接崩，阻断任何 sparse 指标的 reward 函数；正确性相关）
- C=2（自包含 guard + 3 单测；pred 对齐是唯一小心点，已用 zip strict=True + 测试覆盖）
- A=5（`help wanted` + 无人认领 + 报告者给了修复指引 → maintainer 想要）
- F=4（RL 指标聚合，本地可复现）
- **性价比 = (4×5×4)/2 = 40** —— 本轮最高，patch+测试已就绪，等你审阅后一起在 fork 开 PR
