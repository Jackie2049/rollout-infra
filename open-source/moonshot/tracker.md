# MoonshotAI 贡献追踪

> 跨扫描持久维护。每次 SOP 启动时先读本文件。
> 上游：https://github.com/MoonshotAI（MoonEP / FlashKDA / minitriton 等）
> 你的 fork：`Jackie2049/MoonEP`（已有 1 commit：`792f1f6` 限制无界 lru_cache）
> 本地 clone：`/Volumes/Agent/workspace/projects/MoonEP/`（origin=upstream, jackie2049=push）

## 仓库概况（2026-08-04 侦察）

| 仓库 | 方向 | 状态 | 备注 |
|---|---|---|---|
| **MoonEP** | EP/MoE 动态冗余专家 | 2026-07-24 开源，MIT，1011★ | Kimi K3 基础设施；maintainer `asp0ex`（COLLABORATOR）只回 QST issue，bug issue 0 回应；7 个 PR 堆着未 review（review 慢） |
| FlashKDA | Kimi Delta Attention CUDA kernel | 2026-04-20，1148★ | 活跃，体量小易切入（待深挖） |
| minitriton | 最小 Triton 项目 | 2026-07-23，68★ | 极早期，待观察 |

## 当前活跃

| # | 候选 | 仓库 | 类别 | 性价比 | PR | 状态 | 首次发现 |
|---|---|---|---|---|---|---|---|
| M1 | Buffer 错误路径资源泄漏（mc_fd / mc_handle）(#17) | MoonEP | P1 | TBD | — | VERIFIED-STATIC(mc_fd trivial 修复确认；mc_handle 需查 release 机制，报告者分析有 gap) | 2026-08-04 |
| M2 | plan-reuse dispatch R≤96 cap (#18) | MoonEP | P1 | TBD | — | NEW(无认领，1 行 thread count / 注释，需 maintainer 定意图) | 2026-08-04 |

## 已拒 / 已弃（MoonEP）

| # | 候选 | 状态 | 原因 |
|---|---|---|---|
| #9 | CUDA ext 失败 exit 进程 | SKIP | reporter `morluto` 已开 PR #13 |
| #8 | plan ownership 跨 Buffer | SKIP | reporter `morluto` 已开 PR #12 |
| #7 | num_sms 校验 | SKIP | reporter `morluto` 已开 PR #10 |
| #1 | assert→exception（部分） | SKIP | PR #11 部分进行中 |
| #16 | Buffer() 拒绝奇数 EP 含 R=1 | DEFER | 需 maintainer 设计输入（EOFF_SUB 是否他路径重发） |

## 经验教训

- **2026-08-04**：MoonEP maintainer `asp0ex` 只回 QST issue（#6/#15/#21），bug issue（#1/#7-9/#16-19）0 回应，7 个 PR 未 review。**贡献前预期 review 周期长**，但 bug issue 无竞争（除了 reporter 自己开 PR 的）。
- **2026-08-04**：reporter `morluto` 给 #7/#8/#9 都**自己开了 PR**——和 verl #7213 同模式（reporter 自认领）。筛候选必须查 `gh pr list --search "<issue> in:body"`，不能只看 issue 评论。
- **2026-08-04**：`yurekami` 的 issue 质量极高（行号 + 先例 + 修复模式），但 #17 的"复用现有 binding"判断有 gap——`mc_handle` 无 Python release binding（只有 VMM 的 `nvl_release_mem_handle`）。**报告者的"现有 binding"判断要独立核实**。
- **2026-08-04**：MoonEP 是 NVSwitch SHARP 硬件相关，4090（SM89）无法跑测试，验证只能止于静态代码分析。
