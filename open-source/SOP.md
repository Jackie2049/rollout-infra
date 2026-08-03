# 开源贡献挖掘 SOP

> 本文件是 `jackie-verl-pr` skill 的文档化版本。Skill 是执行入口（在 `~/.claude/skills/jackie-verl-pr/SKILL.md`），本文件是方法论沉淀，供人阅读和跨目标复用。

## 五阶段流水线

```
① 同步快照  →  ② 收集信号  →  ③ 评分排序  →  ④ 可行性验证(按需)  →  ⑤ 产出+沉淀
   (本地)        (本地)         (本地)            (4090 服务器)           (本地)
```

### ① 同步快照

本地拉取目标仓库最新 main，记录扫描 SHA。增量优先（对照上次 SHA 看 diff）。

### ② 收集信号（四个信号源并行）

#### 2.1 Git 挖矿
- 近 N commit 改最多的文件（热点 = 不稳定 = 可能还有 bug）
- 被 revert 的 commit（方向错误 / 引入 regression）
- 近期 fix commit 模式（同类 bug 反复出现 = 根因未修）

#### 2.2 静态扫描
- `TODO/FIXME/HACK/XXX`（技术债地图）
- bare `except:` / `except Exception:`（静默吞异常 = 调试地狱）
- lib 代码里的 `print(`（应为 logger）
- runtime `assert`（不该用 assert 做校验的地方）

#### 2.3 Issue 挖掘
- Open issues 无指派、按 reaction/comment 排序（用户痛点）
- `good first issue` / `call for contribution` 标签（verl 目前没用这两个标签，靠 reaction 排序）
- 最近 closed 没合的 issue（maintainer 不想要的方向，避开）

#### 2.4 PR 模式学习
- 近期 merged PR（学风格 / 大小 / 接受的改动类型）
- Closed-unmerged PR（学什么会被拒）

### ③ 评分排序

#### 优先级类别

| 级别 | 类别 | 备注 |
|---|---|---|
| **P0** | bugfix | 接受率最高、价值最硬、通常 diff 小 |
| **P1** | 健壮性 | 小而安全，攒 merge 数 |
| **P2** | 显存优化 | 有数字说话 |
| **P3** | 可量化性能优化 | ⚠️ **高失败率**：经常"看起来明显"但实测鸟用没有 |
| **P4** | 其他 | 刷 merged count + 熟悉代码 |

> **P3 教训**：之前经常以为找到明显 perf 点，写完测完发现没用。P3 候选必须有 profiling 证据 + baseline vs patch 实测数据，否则只列不推。

#### 性价比四维度打分（1-5 分）

| 维度 | 含义 | 5 分（最优） | 1 分（最差） |
|---|---|---|---|
| **I** Impact | 对用户多重要 | 正确性 bug / 静默错结果 | 纯文档/装饰 |
| **C** Cost | 改动量+风险+验证成本（**越低越好**） | <30 行、单文件、单卡可测 | >300 行、多文件、需 8 卡 |
| **A** Acceptance | maintainer 会合的概率 | 明确 bugfix、无竞争 | 大重构、方向相悖、已有人在做 |
| **F** Fit | 我们的独特优势 | 命中 FSDP/GRPO/prefix/4090/RL 知识 | 需 H100、完全陌生领域 |

**性价比分 = (I × A × F) / C**，越高越优先。< 10 不列入最终清单。

### ④ 可行性验证（按需，仅 top 候选）

只对"性价比高 + 必须 GPU 实测"的候选上 4090 服务器验证。三档状态：
- `verified-reproduces`：bug 复现成功
- `verified-works`：patch 跑通且有数据支撑
- `verified-rejected`：实测没效果（记录下来避免重复挖）

### ⑤ 产出 + 沉淀

- **scan-MMDD-HHMM.md**：本次扫描的完整报告（信号 + 评分 + 候选清单）
- **tracker.md**：更新跨扫描追踪表
- **candidate-*.md**：仅对"决定投入"的候选写详细方案
- **diary/YYYY-MM-DD.md**：追加一条工作日志

## 常见陷阱

1. **"明显性能优化"高失败率** — P3 必须附 profiling 证据
2. **和 maintainer 方向冲突** — 扫之前看近 30 天 merged PR 方向；冲突的 A 直接给 1
3. **重复挖已死的点** — 每次扫描先查 tracker，上游已修的进"已过期"区
4. **过度追求数量** — 一次 3-5 个高质量候选足够
5. **co-author 规则** — `Co-authored-by` 只写 commit trailer，绝不进 issue/PR 正文

## 环境约定

| 阶段 | 在哪跑 | 原因 |
|---|---|---|
| git sync / 信号收集 / 评分 | 本地 Mac | 只读静态扫描，不需 GPU |
| 可行性验证 | 4090 服务器 | 仅 top 候选且需 GPU 时 |

服务器信息见 skill `jackie-debug-4090`。verl 专用 conda env = `verl-pr`（克隆自 `verl080_fsdp`），工作目录 `/home/zxw/verl-pr/`。
