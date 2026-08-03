# verl 贡献追踪

> 跨扫描持久维护。每次 SOP 启动时先读本文件，避免重复挖。
> 上游：https://github.com/volcengine/verl

## 当前活跃

| # | 候选 | 类别 | 性价比 | PR | 状态 | 首次发现 |
|---|---|---|---|---|---|---|
| 1 | del_local_ckpt_after_load 不删 checkpoint (#7213) | P0 | 30 | — | NEW | 2026-08-03 |
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

- **2026-08-03**：verl 团队响应速度很快（#7200 当天报告当天修）。扫到的活跃 issue 要尽快动手，否则容易被抢。#7213 / #7232 均无 assignee，窗口期可能很短。
- **2026-08-03**：verl 不使用 `good first issue` / `call for contribution` 标签（均为 0 个）。issue 信号靠 reaction/comment 排序 + 人工读标题。
- **2026-08-03**：`experimental/` 目录（45 个 .py）是技术债集中区（121 个 TODO 中的大量在此，79 个 bare except 集中在 agent_loop/tool_parser.py）。但 experimental/ 模块变动快，改动容易被冲掉，A 分要扣。
