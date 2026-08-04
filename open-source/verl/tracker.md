# verl 贡献追踪

> 跨扫描持久维护。每次 SOP 启动时先读本文件，避免重复挖。
> 上游：https://github.com/volcengine/verl

## 当前活跃

| # | 候选 | 类别 | 性价比 | PR | 状态 | 首次发现 |
|---|---|---|---|---|---|---|
| 0 | process_validation_metrics 崩在 sparse reward_extra_info None (#6830) | P0 | 40 | — | **VERIFIED-WORKS**(根因+复现+patch 5测试全过；待补单测+pre-commit，PR 待你同意) | 2026-08-04 |
| 1 | del_local_ckpt_after_load 不删 checkpoint (#7213) | P0 | 30→12 | — | BLOCKED(作者 yuyz-cyber 已认领意图，需先协调；回头提醒 review) | 2026-08-03 |
| 2 | veomni transformer del_local_after_load 类型标注错误 | P1 | 12 | — | HOLD(单行类型标注，攒着等同文件实质改动顺带修，不单独提) | 2026-08-03 |

## 已拒 / 已弃

| # | 候选 | 状态 | 原因 |
|---|---|---|---|
| 3 | tool_parser.py 静默吞异常 | REJECTED | Phase ④ 复核：仅 3 处真静默（L404/469/527），14 处有 logger。3 行 debug 日志 + experimental/ 变动快 → 性价比 <10 不达标 |

## 已过期（verl 上游已修）

| # | 候选 | verl commit | 备注 |
|---|---|---|---|
| — | 蒸馏损失 micro-batch 归一化 (#7200) | 594c51b (#7225) | 扫描当天就被修了，verl 团队响应很快 |

## 经验教训

- **2026-08-03**：**`grep except | wc -l` ≠ 静默吞异常**。首轮扫到 tool_parser.py "79 个 bare except"就报了"17 处静默吞异常"；Phase ④ 严格复核（看每个 except 后第一非空语句）发现 **14 处都有 `logger.warning`，只有 3 处真静默**。这类信号必须逐个看 except 后有没有日志，否则会高估价值、浪费 Phase ④。
- **2026-08-03**：**必须读完 issue 正文再评 Acceptance**。首次扫 #7213 只看到"无 assignee/0 评论"就给了 A=5；读正文才发现作者 `yuyz-cyber` 明确写了 "I can submit a focused fix"（已认领意图）。A 应从 5 降到 2，且需先协调归属再动手。
- **2026-08-03**：**单行类型标注 / 3 行 debug 日志不值得单独 PR**。这类 P1 太琐碎，maintainer 大概率觉得刷 commit。策略：攒着，等同文件有实质改动时顺带修（候选 #2 已按此降级为 HOLD）。
- **2026-08-03**：**checkpoint manager 的删除语义要细看 issue 期望**。初稿以为删单文件（`os.remove`），issue 正文明确要删整个角色目录（`shutil.rmtree`）+ barrier 后 + 保留 `data.pt`。Megatron 侧 `os.remove(目录)` 本来就会失败——容易漏。
- **2026-08-03**：verl 团队响应很快（#7200 当天报告当天修）。扫到的活跃 issue 要尽快动手，否则容易被抢。
- **2026-08-03**：verl 不使用 `good first issue` / `call for contribution` 标签（均为 0 个）。issue 信号靠 reaction/comment 排序 + 人工读标题。
- **2026-08-03**：`experimental/` 目录（45 个 .py）是技术债集中区（121 个 TODO 中的大量在此）。但 experimental/ 模块变动快，改动容易被冲掉，A 分要扣。
