# verl 贡献追踪

> 跨扫描持久维护。每次 SOP 启动时先读本文件，避免重复挖。
> 上游：https://github.com/volcengine/verl

## 当前活跃

| # | 候选 | 类别 | 性价比 | PR | 状态 | 首次发现 |
|---|---|---|---|---|---|---|
| 1 | del_local_ckpt_after_load 不删 checkpoint (#7213) | P0 | 30→12 | — | BLOCKED(作者 yuyz-cyber 已认领意图，需先协调) | 2026-08-03 |
| 2 | veomni transformer del_local_after_load 类型标注错误 | P1 | 12 | — | NEW | 2026-08-03 |
| 3 | tool_parser.py 静默吞异常（17 处） | P1 | 12 | — | NEW | 2026-08-03 |

## 已拒 / 已弃

| # | 候选 | 状态 | 原因 |
|---|---|---|---|
| — | — | — | — |

## 已过期（verl 上游已修）

| # | 候选 | verl commit | 备注 |
|---|---|---|---|
| — | 蒸馏损失 micro-batch 归一化 (#7200) | 594c51b (#7225) | 扫描当天就被修了，verl 团队响应很快 |

## 经验教训

- **2026-08-03**：**必须读完 issue 正文再评 Acceptance**。首次扫 #7213 只看到"无 assignee/0 评论"就给了 A=5；读正文才发现作者 `yuyz-cyber` 明确写了 "I can submit a focused fix"（已认领意图）。A 应从 5 降到 2，且需先协调归属再动手。
- **2026-08-03**：**checkpoint manager 的删除语义要细看 issue 期望**。初稿以为删单文件（`os.remove`），issue 正文明确要删整个角色目录（`shutil.rmtree`）+ barrier 后 + 保留 `data.pt`。Megatron 侧 `os.remove(目录)` 本来就会失败——容易漏。
- **2026-08-03**：verl 团队响应很快（#7200 当天报告当天修）。扫到的活跃 issue 要尽快动手，否则容易被抢。
- **2026-08-03**：verl 不使用 `good first issue` / `call for contribution` 标签（均为 0 个）。issue 信号靠 reaction/comment 排序 + 人工读标题。
- **2026-08-03**：`experimental/` 目录（45 个 .py）是技术债集中区（121 个 TODO 中的大量在此，79 个 bare except 集中在 agent_loop/tool_parser.py）。但 experimental/ 模块变动快，改动容易被冲掉，A 分要扣。
