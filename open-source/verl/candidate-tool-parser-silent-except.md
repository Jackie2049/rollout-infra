# Candidate: tool_parser.py 静默吞异常

> 状态：`re-assessed`（首轮判断被修正，价值大幅缩水）
> 首次发现：2026-08-03
> 来源：静态扫描 → Phase ④ 严格复核

## 1. 首轮误判 → 修正

首轮扫描报告"tool_parser.py 有 17 处静默吞异常"。Phase ④ 严格复核（看每个 `except` 后第一非空语句）后：

| 类型 | 数量 | 结论 |
|---|---|---|
| except 后带 `logger.warning(...)` | **14** | **不是静默**，有日志，只是 catch-all 粒度粗 |
| except 后直接 `return value`（无日志） | **3**（L404, L469, L527） | 真静默 |

**SOP 教训**：`grep "except Exception" | wc -l` ≠ 静默吞异常。必须逐个看 except 后有没有 logger。

## 2. 真正的 3 处静默

`verl/experimental/agent_loop/tool_parser.py` L404 / L469 / L527，均为：

```python
        try:
            ...
        except Exception:
            return value        # ← 吞掉解析失败，静默降级，无任何日志
```

这三处是 `tool call` 解析失败时的降级路径，吞掉异常后静默返回原始 value。用户调试 agent loop 的 tool 解析时，这三条路径失败完全无迹可查。

## 3. 修复（若做）

给 3 处加 debug 级日志：

```python
        except Exception:
            logger.debug(f"Failed to parse tool call, degenerating to raw value: {e}")  # e 需 as e
            return value
```

## 4. 评估（下调）

- I=2（只 3 处，且是 debug 级观测性，非用户可见 bug）
- C=1（3 行）
- A=2（**experimental/ 变动快**，且这是"加 debug 日志"级别的改动，maintainer 大概率觉得不值得收）
- F=4 → 性价比 12 → **实际约 6**（A 下调）

## 5. 结论：暂不做

3 行 debug 日志 + experimental/ 变动快 + maintainer 不太会专门收 → **性价比 < 10，不达到开工门槛**。

**建议**：**丢弃本候选**。如果未来在 `agent_loop/tool_parser.py` 有实质改动（bug fix / 功能），顺带给这 3 处补 `logger.debug`。不单独提 PR。

## 6. 待办

- [x] Phase ④ 严格复核 → 发现首轮误判
- [ ] 丢弃 / 挂起，不单独提 PR
