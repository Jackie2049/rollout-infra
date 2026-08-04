# Candidate: filter_overlong_prompts 多进程 SIGTERM 挂死 (#7163)

> Issue: [volcengine/verl#7163](https://github.com/volcengine/verl/issues/7163)
> 标签：bug
> 状态：`verified-works`（signal 继承机制在 Linux fork 复现 + patch 验证）
> 首次发现：2026-08-04
> **竞争风险：低** — 0 评论、0 assignee、reporter `mikequan0425` 未认领修复

## 1. Bug

`data.filter_overlong_prompts=True` + `filter_overlong_prompts_workers>1` 时，`datasets.filter` 多进程清理阶段**挂死不退出**。reporter 在 NPU 环境（verl 0.9.0.dev）遇到，但根因是平台无关的 signal 继承。

## 2. 根因（reporter 已诊断 + 本地验证）

`verl/utils/dataset/rl_dataset.py:267` 的 `dataframe.filter(..., num_proc=self.num_workers)`：
- datasets.filter 用 multiprocessing fork worker
- Ray worker（verl 数据加载在 Ray actor 内）注册了 SIGTERM handler（raise SystemExit(1)）
- **fork 的子进程继承父进程的 signal handler 表** → filter worker 继承了 Ray 的 SystemExit handler
- 清理阶段 datasets.filter 给 worker 发 SIGTERM → worker 跑 SystemExit handler → 清理挂死

## 3. Patch（已实现 + Linux 验证）

filter 前临时把 SIGTERM 重置为 `SIG_DFL`，fork 后子进程继承默认行为（正常终止），filter 后恢复：

```python
import signal
_prior_sigterm = signal.getsignal(signal.SIGTERM)
signal.signal(signal.SIGTERM, signal.SIG_DFL)
try:
    dataframe = dataframe.filter(
        lambda doc: doc2len(doc) <= self.max_prompt_length,
        num_proc=self.num_workers,
        desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
    )
finally:
    signal.signal(signal.SIGTERM, _prior_sigterm)
```

## 4. 验证结果（Linux fork 复现 `/tmp/repro_7163.py`，4090 服务器）

| 场景 | 子进程继承的 SIGTERM handler |
|---|---|
| BUG（不修复） | `ray_like_handler`（raise SystemExit）→ 清理挂死 ❌ |
| FIX（fork 前设 SIG_DFL） | `SIG_DFL`（默认终止）→ 清理正常 ✅ |
| 恢复 | filter 后父进程 handler 恢复为原 Ray handler ✅ |

机制完全确认：fork 子进程继承 fork 时刻父进程的 handler 表；设 SIG_DFL 后子进程继承 SIG_DFL。

## 5. 待办

- [ ] 补单测：验证 filter 前后父进程 SIGTERM handler 恢复（fork 继承在 pytest 里可能需 mock，或测 `num_proc=1` 路径不挂）
- [ ] pre-commit + 查重（`gh pr list --search "7163 in:body"` / `"filter_overlong_prompts SIGTERM"`）
- [ ] **（对外，需你同意）** 在 fork 开 PR
- [x] patch 在本地 `verl-pr/verl_main/verl/utils/dataset/rl_dataset.py`，未碰远端

## 6. 评估

- I=3（多进程数据过滤挂死，需特定配置 `filter_overlong_prompts=True + workers>1`；非默认但常见） 
- C=2（signal 重置 wrap，~10 行 + try/finally）
- A=4（无认领；reporter 做了详细 root-cause；修复方向标准）
- F=4（信号处理 + 多进程，平台无关逻辑，Linux 本地可验证）
- **性价比 = (3×4×4)/2 = 24** —— 强候选，patch 已就绪
