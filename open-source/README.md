# 开源贡献挖掘

> 系统性地从主流开源仓库（verl / ROLL / pytorch / vllm / ...）发现高性价比的贡献机会，按 I×A×F/C 排序产出候选清单。

## 目录结构（浅层，方便手机 GitHub app 浏览）

```
open-source/
├── README.md            # 本文件
├── SOP.md               # 通用挖掘方法论（5 阶段流水线 + 评分公式）
├── verl/                # 每个目标仓库一个目录
│   ├── tracker.md       # 跨扫描持久追踪（候选 → PR → merged/rejected 生命周期）
│   ├── scan-MMDD-HHMM.md # 每次扫描的完整报告（信号 + 评分 + 候选清单）
│   └── candidate-*.md   # 进入"准备提 PR"阶段的候选详细方案
├── vllm/                # 未来扩展
└── roll/                # 未来扩展
```

## 与 `notebook/` 的区别

| 位置 | 用途 | 产物类型 |
|---|---|---|
| `notebook/verl-XXXX-deep-reading.md` | **理解**某个 issue/PR（学习） | 深度阅读笔记 |
| `open-source/verl/tracker.md` | **追踪**要贡献什么（行动） | 候选生命周期表 |
| `open-source/verl/scan-MMDD.md` | **发现**（一次性扫描快照） | 信号收集 + 评分排序 |
| `open-source/verl/candidate-*.md` | **行动**（具体 PR 计划） | 方案 + patch + 复现 |

一个 candidate 可以 **链接** 到 `notebook/` 里的深度阅读笔记（如果有的话），但两者不互相替代。

## 如何启动一次扫描

调用 Claude Code skill `jackie-verl-pr`（verl 目标），说：
- "扫一遍 verl"
- "verl 有什么能做的"
- "verl 优化点挖掘"

Skill 会按 `SOP.md` 的 5 阶段流水线执行，产出新的 `scan-MMDD-HHMM.md` 并更新 `tracker.md`。

## 优先级口径

| 级别 | 类别 | 说明 |
|---|---|---|
| **P0** | bugfix | 正确性 bug、静默错结果、崩溃 — 接受率最高 |
| **P1** | 健壮性 | 报错信息、参数校验、防御性 guard、边界条件 |
| **P2** | 显存优化 | 内存占用 / OOM 缓解 |
| **P3** | 可量化性能优化 | ⚠️ 高失败率，必须有 profiling 证据 |
| **P4** | 其他 | 重构、文档、补测试、示例 |
